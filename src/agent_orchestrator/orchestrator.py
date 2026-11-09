"""Wires config, providers, MCP servers and agents together, and runs requests."""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentResult, BudgetExceeded, Event, EventSink, RunContext, current_run, delegation_tool
from .config import AgentConfig, AppConfig
from .llm.base import LLMProvider, Message
from .mcp_tools import MCPToolset
from .tools import BUILTIN_TOOLS, Tool
from .tracing import Tracer


@dataclass
class RunRecord:
    run_id: str
    agent: str
    input: str
    output: str
    status: str
    input_tokens: int
    output_tokens: int
    tool_calls: int
    duration_ms: float
    trace_id: str | None
    events: list[dict[str, Any]] = field(default_factory=list)


def build_providers(config: AppConfig) -> dict[str, LLMProvider]:
    from .llm.anthropic import AnthropicProvider
    from .llm.openai_compat import OpenAICompatibleProvider

    providers: dict[str, LLMProvider] = {}
    for name, p in config.providers.items():
        if p.type == "anthropic":
            providers[name] = AnthropicProvider(api_key=p.api_key or None)
        else:
            providers[name] = OpenAICompatibleProvider(
                base_url=p.base_url or "https://api.openai.com/v1", api_key=p.api_key or "", name=name
            )
    return providers


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        providers: dict[str, LLMProvider],
        *,
        tracer: Tracer | None = None,
        extra_tools: list[Tool] | None = None,
        max_runs_kept: int = 500,
        max_sessions: int = 1000,
    ) -> None:
        self.config = config
        self.providers = providers
        self.tracer = tracer or Tracer(
            endpoint=os.getenv("GENAI_LENS_ENDPOINT"), api_key=os.getenv("GENAI_LENS_API_KEY", "")
        )
        self.mcp = MCPToolset()
        self.agents: dict[str, Agent] = {}
        self._extra = {t.name: t for t in extra_tools or []}
        self.runs: OrderedDict[str, RunRecord] = OrderedDict()
        self.sessions: OrderedDict[str, list[Message]] = OrderedDict()
        self._max_runs, self._max_sessions = max_runs_kept, max_sessions
        self._background: set[asyncio.Task[None]] = set()

    async def start(self, mcp_targets: dict[str, Any] | None = None) -> None:
        """Connect MCP servers and build agents. ``mcp_targets`` overrides how a
        server is reached (e.g. an in-process server in tests)."""
        for server in self.config.mcp_servers:
            if mcp_targets and server.name in mcp_targets:
                await self.mcp.connect(server.name, mcp_targets[server.name])
            else:
                await self.mcp.connect_config(server)

        # Workers first, so supervisors can wrap them as delegation tools.
        pending = list(self.config.agents)
        while pending:
            ready = [a for a in pending if all(d in self.agents for d in a.delegates)]
            if not ready:
                raise ValueError(f"circular delegation between: {[a.name for a in pending]}")
            for cfg in ready:
                tools = self._resolve_tools(cfg) + [delegation_tool(self.agents[d]) for d in cfg.delegates]
                self.agents[cfg.name] = Agent(cfg, self.providers[cfg.provider], tools)
                pending.remove(cfg)

    async def aclose(self) -> None:
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        await self.mcp.aclose()

    def _resolve_tools(self, cfg: AgentConfig) -> list[Tool]:
        tools: list[Tool] = []
        for ref in cfg.tools:
            if ref in self._extra:
                tools.append(self._extra[ref])
            elif ref in BUILTIN_TOOLS:
                tools.append(BUILTIN_TOOLS[ref]())
            elif ref.endswith("__*"):
                prefix = ref[:-1]
                matched = [t for name, t in self.mcp.tools.items() if name.startswith(prefix)]
                if not matched:
                    raise ValueError(f"agent {cfg.name!r}: no MCP tools match {ref!r}")
                tools += matched
            elif ref in self.mcp.tools:
                tools.append(self.mcp.tools[ref])
            else:
                raise ValueError(f"agent {cfg.name!r}: unknown tool {ref!r}")
        return tools

    async def run(
        self,
        task: str,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        sink: EventSink | None = None,
    ) -> RunRecord:
        name = agent or self.config.default_agent
        if name not in self.agents:
            raise KeyError(name)
        run_id = "run_" + secrets.token_hex(8)
        events: list[dict[str, Any]] = []

        async def record(event: Event) -> None:
            events.append({"type": event.type, "agent": event.agent, "depth": event.depth, **event.data})
            if sink:
                await sink(event)

        ctx = RunContext(limits=self.config.limits, tracer=self.tracer, sink=record)
        history = list(self.sessions.get(session_id, [])) if session_id else []
        started = time.perf_counter()
        token = current_run.set(ctx)
        trace_id: str | None = None
        try:
            with self.tracer.span(f"run.{name}", "chain", input=task[:4000]) as root:
                trace_id = root.trace_id
                root.attributes["run_id"] = run_id
                try:
                    result = await asyncio.wait_for(
                        self.agents[name].run(task, history), timeout=self.config.limits.run_timeout_s
                    )
                except TimeoutError:
                    result = AgentResult(
                        f"error: run timed out after {self.config.limits.run_timeout_s}s", "error", [], 0
                    )
                except BudgetExceeded as exc:
                    result = AgentResult(f"error: {exc}", "error", [], 0)
                if result.status != "completed":
                    root.status, root.error = "error", result.output[:500]
                root.output = result.output[:4000]
        finally:
            current_run.reset(token)
        if trace_id:
            # keep a reference: asyncio only holds weak refs to tasks
            export = asyncio.create_task(self.tracer.export(trace_id))
            self._background.add(export)
            export.add_done_callback(self._background.discard)

        if session_id and result.status == "completed":
            self.sessions[session_id] = [
                m for m in result.messages if m.role in ("user", "assistant") and not m.tool_calls
            ]
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > self._max_sessions:
                self.sessions.popitem(last=False)

        rec = RunRecord(
            run_id=run_id,
            agent=name,
            input=task,
            output=result.output,
            status=result.status,
            input_tokens=ctx.usage.input_tokens,
            output_tokens=ctx.usage.output_tokens,
            tool_calls=ctx.tool_calls,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            trace_id=trace_id,
            events=events,
        )
        self.runs[run_id] = rec
        while len(self.runs) > self._max_runs:
            self.runs.popitem(last=False)
        return rec
