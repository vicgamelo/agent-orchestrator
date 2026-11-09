import time
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer

from agent_orchestrator.llm.scripted import ScriptedProvider, Step
from agent_orchestrator.orchestrator import Orchestrator

from .conftest import make_config

pytestmark = pytest.mark.anyio


async def build(
    scripts: dict[str, list[Step]], delay: float = 0.0, **cfg: Any
) -> tuple[Orchestrator, ScriptedProvider]:
    provider = ScriptedProvider(scripts, delay=delay)
    orch = Orchestrator(make_config(**cfg), {"fake": provider})
    await orch.start()
    return orch, provider


async def test_supervisor_delegates_in_parallel_and_aggregates_usage() -> None:
    orch, provider = await build(
        {
            "coordinator": [
                Step(
                    calls=[("delegate_to_economist", {"task": "selic?"}), ("delegate_to_researcher", {"task": "cnpj?"})]
                ),
                Step(text="Selic 13.75%, empresa ativa"),
            ],
            "economist": [Step(calls=[("calculator", {"expression": "13.75/12"})]), Step(text="selic 13.75")],
            "researcher": [Step(text="ativa")],
        },
        delay=0.15,
    )
    started = time.perf_counter()
    rec = await orch.run("resuma")
    elapsed = time.perf_counter() - started

    assert rec.status == "completed" and rec.output == "Selic 13.75%, empresa ativa"
    # 5 LLM calls in total, each costing 100 in / 20 out on the scripted provider
    assert (rec.input_tokens, rec.output_tokens) == (500, 100)
    assert rec.tool_calls == 3  # two delegations + the economist's calculator
    # economist (2 calls) and researcher (1 call) run concurrently: 2 + 2 sequential LLM calls
    assert elapsed < 0.15 * 5
    depths = {(e["agent"], e["depth"]) for e in rec.events if e["type"] == "agent_start"}
    assert depths == {("coordinator", 0), ("economist", 1), ("researcher", 1)}
    # the worker starts with a fresh conversation containing only the delegated task
    econ_first = next(r for r in provider.requests if r["model"] == "economist")
    assert [m.content for m in econ_first["messages"]] == ["selic?"]


async def test_trace_tree_spans_delegation() -> None:
    orch, _ = await build(
        {
            "coordinator": [Step(calls=[("delegate_to_researcher", {"task": "x"})]), Step(text="ok")],
            "researcher": [Step(text="r")],
            "economist": [Step(text="e")],
        }
    )
    exported: list[Any] = []

    async def capture(trace_id: str) -> None:
        exported.extend(orch.tracer.take(trace_id))

    orch.tracer.export = capture  # type: ignore[method-assign]
    rec = await orch.run("go")
    import asyncio

    await asyncio.sleep(0)
    by_name = {s.name: s for s in exported}
    assert by_name["run.coordinator"].parent_id is None
    assert by_name["agent.researcher"].parent_id == by_name["tool.delegate_to_researcher"].span_id
    assert by_name["tool.delegate_to_researcher"].kind == "agent"
    assert all(s.trace_id == rec.trace_id for s in exported)


async def test_delegation_depth_limit() -> None:
    orch, _ = await build(
        {
            "coordinator": [Step(calls=[("delegate_to_researcher", {"task": "x"})]), Step(text="ok")],
            "researcher": [Step(text="r")],
            "economist": [Step(text="e")],
        },
        limits={"max_delegation_depth": 0},
    )
    rec = await orch.run("go")
    result = next(e for e in rec.events if e["type"] == "tool_result")
    assert result["result"].startswith("error: maximum delegation depth")


async def test_token_budget_stops_the_run() -> None:
    orch, _ = await build(
        {
            "coordinator": [Step(calls=[("delegate_to_researcher", {"task": "x"})])],
            "researcher": [Step(calls=[("nope", {})])],
            "economist": [Step(text="e")],
        },
        limits={"max_total_tokens": 350},
    )
    rec = await orch.run("go")
    assert rec.status == "error" and "token budget exceeded" in rec.output


async def test_session_memory_carries_previous_turns() -> None:
    orch, provider = await build(
        {
            "coordinator": [Step(text="Oi Victor"), Step(text="Você se chama Victor")],
            "researcher": [Step(text="")],
            "economist": [Step(text="")],
        }
    )
    await orch.run("meu nome é Victor", session_id="s1")
    await orch.run("qual meu nome?", session_id="s1")
    second = provider.requests[-1]["messages"]
    assert [m.content for m in second] == ["meu nome é Victor", "Oi Victor", "qual meu nome?"]


async def test_mcp_tools_are_exposed_namespaced() -> None:
    server = MCPServer(name="test")

    @server.tool()
    def selic() -> dict[str, float]:
        """Current Selic."""
        return {"selic": 13.75}

    config = make_config(
        mcp_servers=[{"name": "brasil", "url": "http://unused"}],
        agents=[
            {
                "name": "economist",
                "description": "d",
                "provider": "fake",
                "model": "economist",
                "instructions": "i",
                "tools": ["brasil__*"],
            }
        ],
        default_agent="economist",
    )
    provider = ScriptedProvider({"economist": [Step(calls=[("brasil__selic", {})]), Step(text="13.75")]})
    orch = Orchestrator(config, {"fake": provider})
    await orch.start(mcp_targets={"brasil": server})
    try:
        assert provider.requests == []
        rec = await orch.run("selic?")
        assert rec.output == "13.75"
        tool_result = next(e for e in rec.events if e["type"] == "tool_result")
        assert tool_result["result"] == '{"selic": 13.75}'
        assert orch.agents["economist"].tools["brasil__selic"].source == "mcp:brasil"
    finally:
        await orch.aclose()


async def test_unknown_tool_reference_fails_at_startup() -> None:
    config = make_config(
        agents=[
            {"name": "a", "description": "d", "provider": "fake", "model": "m", "instructions": "i", "tools": ["nope"]}
        ],
        default_agent="a",
    )
    orch = Orchestrator(config, {"fake": ScriptedProvider({})})
    with pytest.raises(ValueError, match="unknown tool 'nope'"):
        await orch.start()
