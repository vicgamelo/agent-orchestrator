import asyncio
import time

import pytest

from agent_orchestrator.agent import Agent, Event, RunContext, current_run
from agent_orchestrator.config import AgentConfig, Limits
from agent_orchestrator.llm.scripted import ScriptedProvider, Step
from agent_orchestrator.tools import Tool, function_tool
from agent_orchestrator.tracing import Tracer

pytestmark = pytest.mark.anyio


def agent(provider: ScriptedProvider, tools: list[Tool], **cfg: object) -> Agent:
    config = AgentConfig.model_validate(
        {"name": "a", "description": "d", "provider": "p", "model": "m", "instructions": "i", **cfg}
    )
    return Agent(config, provider, tools)


def ctx(**limits: float) -> tuple[RunContext, list[Event]]:
    events: list[Event] = []

    async def sink(e: Event) -> None:
        events.append(e)

    c = RunContext(limits=Limits(**limits), tracer=Tracer(), sink=sink)
    current_run.set(c)
    return c, events


async def test_tool_loop_feeds_results_back() -> None:
    provider = ScriptedProvider({"m": [Step(calls=[("calculator", {"expression": "2+2"})]), Step(text="it is 4")]})
    c, events = ctx()
    from agent_orchestrator.tools import BUILTIN_TOOLS

    result = await agent(provider, [BUILTIN_TOOLS["calculator"]()]).run("2+2?")
    assert (result.output, result.status, result.steps) == ("it is 4", "completed", 2)
    second_call = provider.requests[1]["messages"]
    assert second_call[-1].role == "tool" and second_call[-1].content == "4"
    assert [e.type for e in events] == ["agent_start", "llm", "tool_call", "tool_result", "llm", "agent_end"]
    assert c.usage.input_tokens == 200 and c.tool_calls == 1


async def test_unknown_tool_and_tool_exception_go_back_to_model() -> None:
    def boom(x: int) -> str:
        raise RuntimeError("database down")

    provider = ScriptedProvider({"m": [Step(calls=[("nope", {}), ("boom", {"x": 1})]), Step(text="sorry")]})
    ctx()
    result = await agent(provider, [function_tool(boom)]).run("go")
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert tool_msgs[0].content.startswith("error: unknown tool 'nope'")
    assert tool_msgs[1].content == "error: RuntimeError: database down"
    assert result.status == "completed"


async def test_parallel_tool_calls_run_concurrently() -> None:
    async def slow(n: int) -> str:
        await asyncio.sleep(0.2)
        return str(n)

    calls = [("slow", {"n": i}) for i in range(5)]
    provider = ScriptedProvider({"m": [Step(calls=calls), Step(text="done")]})
    ctx()
    started = time.perf_counter()
    await agent(provider, [function_tool(slow)]).run("go")
    assert time.perf_counter() - started < 0.6  # 5 x 0.2s sequentially would be 1s


async def test_tool_timeout() -> None:
    async def hang() -> str:
        await asyncio.sleep(5)
        return "never"

    provider = ScriptedProvider({"m": [Step(calls=[("hang", {})]), Step(text="ok")]})
    ctx(tool_timeout_s=0.05)
    result = await agent(provider, [function_tool(hang)]).run("go")
    assert "timed out" in next(m.content for m in result.messages if m.role == "tool")


async def test_max_steps_guardrail() -> None:
    provider = ScriptedProvider({"m": [Step(calls=[("calculator", {"expression": "1"})])]})  # loops forever
    ctx()
    from agent_orchestrator.tools import BUILTIN_TOOLS

    result = await agent(provider, [BUILTIN_TOOLS["calculator"]()], max_steps=3).run("go")
    assert result.status == "max_steps" and len(provider.requests) == 3


async def test_spans_are_nested() -> None:
    provider = ScriptedProvider({"m": [Step(calls=[("calculator", {"expression": "1+1"})]), Step(text="2")]})
    c, _ = ctx()
    from agent_orchestrator.tools import BUILTIN_TOOLS

    with c.tracer.span("root", "chain") as root:
        await agent(provider, [BUILTIN_TOOLS["calculator"]()]).run("go")
    spans = {s.name: s for s in c.tracer.take(root.trace_id)}
    assert spans["agent.a"].parent_id == root.span_id
    assert spans["tool.calculator"].parent_id == spans["agent.a"].span_id
    assert spans["scripted.complete"].input_tokens == 100
