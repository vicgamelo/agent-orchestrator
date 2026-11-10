"""Wires config, providers, MCP servers and agents together, and runs requests."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentResult, BudgetExceeded, Event, EventSink, RunContext, current_run, delegation_tool
from .config import AgentConfig, AppConfig
from .llm.base import LLMProvider, Message
from .llm.fallback import FallbackLLM, Route
from .mcp_tools import MCPToolset
from .tools import BUILTIN_TOOLS, Tool
from .tracing import Tracer

logger = logging.getLogger(__name__)

# Placeholder in an agent's instructions, replaced by the delegates available in the run.
TEAM = "{team}"


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
                base_url=p.base_url or "https://api.openai.com/v1",
                api_key=p.api_key or "",
                name=name,
                timeout=p.timeout_s,
                reasoning_effort=p.reasoning_effort,
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
        self.agents: dict[str, Agent] = {}  # the team with every connected MCP server on
        self.servers_up: set[str] = set()
        self._own_tools: dict[str, list[Tool]] = {}  # resolved tools of each agent, without delegation
        self._needs_mcp: dict[str, bool] = {}  # agents that only make sense with their MCP tools
        self._views: dict[frozenset[str], dict[str, Agent]] = {}
        self._extra = {t.name: t for t in extra_tools or []}
        self.runs: OrderedDict[str, RunRecord] = OrderedDict()
        self.sessions: OrderedDict[str, list[Message]] = OrderedDict()
        self._max_runs, self._max_sessions = max_runs_kept, max_sessions
        self._background: set[asyncio.Task[None]] = set()

    async def start(self, mcp_targets: dict[str, Any] | None = None) -> None:
        """Connect MCP servers and build agents. ``mcp_targets`` overrides how a
        server is reached (e.g. an in-process server in tests). A server that can't be reached is
        logged and left out: its tools, and the agents that depend only on them, are unavailable."""
        for server in self.config.mcp_servers:
            try:
                if mcp_targets and server.name in mcp_targets:
                    await self.mcp.connect(server.name, mcp_targets[server.name])
                else:
                    await self.mcp.connect_config(server)
            except Exception as exc:
                logger.warning("mcp server %s unavailable: %s", server.name, exc)
                continue
            self.servers_up.add(server.name)

        for cfg in self.config.agents:
            self._own_tools[cfg.name] = self._resolve_tools(cfg)
            self._needs_mcp[cfg.name] = any("__" in ref for ref in cfg.tools)
        self.agents = self.team(None)
        if self.config.default_agent not in self.agents:
            raise ValueError(f"default agent {self.config.default_agent!r} has no tools available")

    def team(self, servers: list[str] | None) -> dict[str, Agent]:
        """The agents of a run with only ``servers`` on (None: every connected server). A worker
        whose MCP tools are all off leaves the team, and its supervisors stop seeing it."""
        if servers is not None:
            unknown = set(servers) - {s.name for s in self.config.mcp_servers}
            if unknown:
                raise ValueError(f"unknown MCP servers: {sorted(unknown)}")
        enabled = frozenset(self.servers_up if servers is None else set(servers) & self.servers_up)
        if enabled not in self._views:
            self._views[enabled] = self._build_team(enabled)
        return self._views[enabled]

    def _build_team(self, enabled: frozenset[str]) -> dict[str, Agent]:
        team: dict[str, Agent] = {}
        # Workers first, so supervisors can wrap them as delegation tools.
        pending = list(self.config.agents)
        while pending:
            ready = [a for a in pending if all(d not in {p.name for p in pending} for d in a.delegates)]
            if not ready:
                raise ValueError(f"circular delegation between: {[a.name for a in pending]}")
            for cfg in ready:
                pending.remove(cfg)
                tools = [
                    t for t in self._own_tools[cfg.name] if not t.source.startswith("mcp:") or t.source[4:] in enabled
                ]
                if self._needs_mcp[cfg.name] and not any(t.source.startswith("mcp:") for t in tools):
                    continue
                workers = [team[d] for d in cfg.delegates if d in team]
                if TEAM in cfg.instructions:
                    roster = "\n".join(f"- {w.name}: {w.config.description}" for w in workers) or "- (nobody)"
                    cfg = cfg.model_copy(update={"instructions": cfg.instructions.replace(TEAM, roster)})
                team[cfg.name] = Agent(cfg, self._llm_for(cfg), tools + [delegation_tool(w) for w in workers])
        return team

    def servers(self) -> list[dict[str, Any]]:
        """MCP servers, whether they are up, their tools and the agents that use them."""
        out = []
        for server in self.config.mcp_servers:
            prefix = f"{server.name}__"
            tools = sorted(name.removeprefix(prefix) for name in self.mcp.tools if name.startswith(prefix))
            source = f"mcp:{server.name}"
            agents = [a.name for a in self.config.agents if any(t.source == source for t in self._own_tools[a.name])]
            out.append({"name": server.name, "up": server.name in self.servers_up, "tools": tools, "agents": agents})
        return out

    def _llm_for(self, cfg: AgentConfig) -> LLMProvider:
        # A provider whose api_key expanded to "" (OPENROUTER_API_KEY unset) would only answer 401:
        # drop it from the chain. With no key anywhere, keep the chain so the error says which is missing.
        chain = [p for p in cfg.provider_chain if self.config.providers[p].api_key != ""] or cfg.provider_chain
        if chain == [cfg.provider_chain[0]]:
            return self.providers[chain[0]]  # the agent already asks for this provider's model
        return FallbackLLM([Route(self.providers[p], cfg.model_for(p)) for p in chain])

    async def aclose(self) -> None:
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        await self.mcp.aclose()

    def _resolve_tools(self, cfg: AgentConfig) -> list[Tool]:
        tools: list[Tool] = []
        configured = {s.name for s in self.config.mcp_servers}
        for ref in cfg.tools:
            server = ref.split("__", 1)[0] if "__" in ref else None
            if server in configured and server not in self.servers_up:
                continue  # that server is down; the agent runs without it (or leaves the team)
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
        servers: list[str] | None = None,
    ) -> RunRecord:
        """``servers``: MCP servers on for this run (None: all). Raises ValueError for an unknown
        server and KeyError for an agent that isn't in the team (unknown, or all its tools off)."""
        name = agent or self.config.default_agent
        agents = self.team(servers)
        if name not in agents:
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
                if servers is not None:
                    root.attributes["mcp_servers"] = ",".join(sorted(servers))
                try:
                    result = await asyncio.wait_for(
                        agents[name].run(task, history), timeout=self.config.limits.run_timeout_s
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
