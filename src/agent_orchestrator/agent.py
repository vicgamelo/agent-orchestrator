"""The agent loop: call the model, run the requested tools (in parallel), feed the
results back, repeat until the model answers or a guardrail trips."""

from __future__ import annotations

import asyncio
import contextvars
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import AgentConfig, Limits
from .llm.base import LLMError, LLMProvider, Message, ToolCall, Usage
from .tools import Tool
from .tracing import Tracer

EventSink = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    type: Literal["agent_start", "llm", "tool_call", "tool_result", "agent_end", "error"]
    agent: str
    depth: int
    data: dict[str, Any] = field(default_factory=dict)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunContext:
    """State shared by every agent taking part in one run (including delegates)."""

    limits: Limits
    tracer: Tracer
    sink: EventSink | None = None
    usage: Usage = field(default_factory=Usage)
    depth: int = 0
    tool_calls: int = 0

    async def emit(self, event: Event) -> None:
        if self.sink:
            await self.sink(event)

    def check_budget(self) -> None:
        total = self.usage.input_tokens + self.usage.output_tokens
        if total > self.limits.max_total_tokens:
            raise BudgetExceeded(f"token budget exceeded ({total} > {self.limits.max_total_tokens})")


current_run: contextvars.ContextVar[RunContext] = contextvars.ContextVar("current_run")


@dataclass
class AgentResult:
    output: str
    status: Literal["completed", "max_steps", "error"]
    messages: list[Message]
    steps: int


class Agent:
    def __init__(self, config: AgentConfig, provider: LLMProvider, tools: list[Tool]) -> None:
        self.config = config
        self.provider = provider
        self.tools = {t.name: t for t in tools}

    @property
    def name(self) -> str:
        return self.config.name

    async def run(self, task: str, history: list[Message] | None = None) -> AgentResult:
        ctx = current_run.get()
        cfg = self.config
        messages = [*(history or []), Message(role="user", content=task)]
        await ctx.emit(Event("agent_start", self.name, ctx.depth, {"task": task}))

        with ctx.tracer.span(f"agent.{self.name}", "agent", input=task[:4000]) as agent_span:
            for step in range(1, cfg.max_steps + 1):
                ctx.check_budget()
                with ctx.tracer.span(f"{self.provider.name}.complete", "llm", provider=self.provider.name) as s:
                    try:
                        resp = await self.provider.complete(
                            model=cfg.model,
                            system=cfg.instructions,
                            messages=messages,
                            tools=[t.spec for t in self.tools.values()],
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                        )
                    except LLMError as exc:
                        await ctx.emit(Event("error", self.name, ctx.depth, {"message": str(exc)}))
                        agent_span.status, agent_span.error = "error", str(exc)
                        return AgentResult(f"error: {exc}", "error", messages, step)
                    s.model, s.input_tokens, s.output_tokens = (
                        resp.model,
                        resp.usage.input_tokens,
                        resp.usage.output_tokens,
                    )
                    s.output = resp.text[:4000] or None
                ctx.usage += resp.usage
                await ctx.emit(
                    Event(
                        "llm",
                        self.name,
                        ctx.depth,
                        {"step": step, "text": resp.text, "tool_calls": [c.name for c in resp.tool_calls]},
                    )
                )

                if not resp.tool_calls:
                    messages.append(Message(role="assistant", content=resp.text))
                    agent_span.output = resp.text[:4000]
                    await ctx.emit(Event("agent_end", self.name, ctx.depth, {"output": resp.text, "steps": step}))
                    return AgentResult(resp.text, "completed", messages, step)

                messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls))
                # Independent tool calls from the same turn run concurrently.
                results = await asyncio.gather(*(self._call_tool(c, ctx) for c in resp.tool_calls))
                messages += [
                    Message(role="tool", content=r, tool_call_id=c.id, name=c.name)
                    for c, r in zip(resp.tool_calls, results, strict=True)
                ]

            note = f"stopped after {cfg.max_steps} steps without a final answer"
            agent_span.status, agent_span.error = "error", note
            await ctx.emit(Event("agent_end", self.name, ctx.depth, {"output": note, "steps": cfg.max_steps}))
            return AgentResult(note, "max_steps", messages, cfg.max_steps)

    async def _call_tool(self, call: ToolCall, ctx: RunContext) -> str:
        ctx.tool_calls += 1
        await ctx.emit(Event("tool_call", self.name, ctx.depth, {"tool": call.name, "arguments": call.arguments}))
        tool = self.tools.get(call.name)
        kind = "agent" if tool and tool.source == "agent" else "tool"
        with ctx.tracer.span(
            f"tool.{call.name}", kind, input=json.dumps(call.arguments, ensure_ascii=False)[:4000]
        ) as s:
            if tool is None:
                result = f"error: unknown tool {call.name!r}. Available: {', '.join(self.tools)}"
            else:
                # Delegation has its own run timeout; plain tools get the per-tool one.
                timeout = None if tool.source == "agent" else ctx.limits.tool_timeout_s
                try:
                    result = await asyncio.wait_for(tool(call.arguments), timeout)
                except TimeoutError:
                    result = f"error: tool {call.name} timed out after {timeout}s"
                except BudgetExceeded:
                    raise
                except Exception as exc:  # the model gets the error and can recover
                    result = f"error: {type(exc).__name__}: {exc}"
            if result.startswith("error:"):
                s.status, s.error = "error", result[:500]
            s.output = result[:4000]
        await ctx.emit(Event("tool_result", self.name, ctx.depth, {"tool": call.name, "result": result[:2000]}))
        return result


def delegation_tool(worker: Agent) -> Tool:
    """Wrap an agent as a tool so a supervisor can hand it a sub-task. The worker
    runs with a fresh conversation and shares the run's budget and event stream."""

    async def delegate(arguments: dict[str, Any]) -> str:
        parent = current_run.get()
        if parent.depth >= parent.limits.max_delegation_depth:
            return "error: maximum delegation depth reached, solve it yourself"
        child = RunContext(
            limits=parent.limits, tracer=parent.tracer, sink=parent.sink, usage=parent.usage, depth=parent.depth + 1
        )
        token = current_run.set(child)
        try:
            result = await worker.run(str(arguments.get("task", "")))
        finally:
            current_run.reset(token)
            parent.tool_calls += child.tool_calls
        return result.output

    return Tool(
        name=f"delegate_to_{worker.name}",
        description=f"Hand a self-contained sub-task to the `{worker.name}` agent. {worker.config.description} "
        "Write the task with all the context it needs; it cannot see this conversation.",
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string", "description": "complete instructions for the sub-task"}},
            "required": ["task"],
        },
        fn=delegate,
        source="agent",
    )
