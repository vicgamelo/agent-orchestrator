"""MCP servers on and off per run: the team shrinks to the agents that still have tools."""

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver import MCPServer

from agent_orchestrator.api import create_app
from agent_orchestrator.llm.scripted import ScriptedProvider, Step
from agent_orchestrator.orchestrator import Orchestrator

from .conftest import make_config

pytestmark = pytest.mark.anyio


def mcp(name: str, tool: str) -> MCPServer:
    server = MCPServer(name=name)

    def fn() -> dict[str, str]:
        return {"from": name}

    server.add_tool(fn, name=tool, description=f"{tool} from {name}")
    return server


def config(**extra: Any) -> Any:
    return make_config(
        mcp_servers=[
            {"name": "brasil", "url": "http://unused"},
            {"name": "noticias", "url": "http://unused"},
            {"name": "agenda", "url": "http://unused"},
        ],
        agents=[
            {
                "name": "coordinator",
                "description": "boss",
                "provider": "fake",
                "model": "coordinator",
                "instructions": "Your team:\n{team}\nOnly answer with them.",
                "tools": ["current_datetime"],
                "delegates": ["economist", "reporter", "scheduler"],
            },
            {
                "name": "economist",
                "description": "Banco Central numbers",
                "provider": "fake",
                "model": "economist",
                "instructions": "e",
                "tools": ["brasil__selic", "calculator"],
            },
            {
                "name": "reporter",
                "description": "Brazilian news",
                "provider": "fake",
                "model": "reporter",
                "instructions": "r",
                "tools": ["noticias__*"],
            },
            {
                "name": "scheduler",
                "description": "Victor's free slots",
                "provider": "fake",
                "model": "scheduler",
                "instructions": "s",
                "tools": ["agenda__slots"],
            },
        ],
        **extra,
    )


SCRIPTS = {
    "coordinator": [Step(text="ok")],
    "economist": [Step(text="e")],
    "reporter": [Step(text="r")],
    "scheduler": [Step(text="s")],
}


async def started(targets: dict[str, Any] | None = None) -> tuple[Orchestrator, ScriptedProvider]:
    provider = ScriptedProvider(SCRIPTS)
    orch = Orchestrator(config(), {"fake": provider})
    await orch.start(
        mcp_targets=targets
        if targets is not None
        else {"brasil": mcp("brasil", "selic"), "noticias": mcp("noticias", "buscar"), "agenda": mcp("agenda", "slots")}
    )
    return orch, provider


async def test_all_servers_on_by_default() -> None:
    orch, provider = await started()
    try:
        await orch.run("x")
        req = provider.requests[0]
        assert set(req["tools"]) == {
            "current_datetime",
            "delegate_to_economist",
            "delegate_to_reporter",
            "delegate_to_scheduler",
        }
        assert "- economist: Banco Central numbers\n- reporter: Brazilian news" in req["system"]
        assert "{team}" not in req["system"]
    finally:
        await orch.aclose()


async def test_turning_a_server_off_removes_its_agent_from_the_team() -> None:
    orch, provider = await started()
    try:
        rec = await orch.run("x", servers=["brasil", "agenda"])
        req = provider.requests[0]
        assert set(req["tools"]) == {"current_datetime", "delegate_to_economist", "delegate_to_scheduler"}
        assert "reporter" not in req["system"]
        assert rec.status == "completed"
        # the default team is untouched by a run with fewer servers
        assert "delegate_to_reporter" in orch.agents["coordinator"].tools
    finally:
        await orch.aclose()


async def test_worker_keeps_builtin_tools_only_while_its_server_is_on() -> None:
    orch, _ = await started()
    try:
        assert "economist" not in orch.team(["noticias"])  # calculator alone isn't the economist's job
        assert set(orch.team(["brasil"])["economist"].tools) == {"brasil__selic", "calculator"}
    finally:
        await orch.aclose()


async def test_all_servers_off_leaves_the_coordinator_alone() -> None:
    orch, provider = await started()
    try:
        await orch.run("x", servers=[])
        req = provider.requests[0]
        assert req["tools"] == ["current_datetime"]
        assert "- (nobody)" in req["system"]
    finally:
        await orch.aclose()


async def test_unknown_server_and_unavailable_agent() -> None:
    orch, _ = await started()
    try:
        with pytest.raises(ValueError, match="unknown MCP servers"):
            await orch.run("x", servers=["brasil", "internet"])
        with pytest.raises(KeyError):
            await orch.run("x", agent="reporter", servers=["brasil"])
    finally:
        await orch.aclose()


async def test_server_down_at_startup_is_not_fatal() -> None:
    class Broken:
        pass

    orch, provider = await started(
        {"brasil": mcp("brasil", "selic"), "noticias": Broken(), "agenda": mcp("agenda", "slots")}
    )
    try:
        assert orch.servers_up == {"brasil", "agenda"}
        assert "reporter" not in orch.agents
        listing = {s["name"]: s for s in orch.servers()}
        assert listing["noticias"] == {"name": "noticias", "up": False, "tools": [], "agents": []}
        assert listing["brasil"] == {"name": "brasil", "up": True, "tools": ["selic"], "agents": ["economist"]}
        # asking for the down server is allowed: it just stays off
        await orch.run("x", servers=["noticias"])
        assert provider.requests[0]["tools"] == ["current_datetime"]
    finally:
        await orch.aclose()


def api_client() -> TestClient:
    orch, _ = asyncio.run(started())
    return TestClient(create_app(orch, api_keys=[]))


def test_api_lists_servers_and_takes_them_per_run() -> None:
    with api_client() as c:
        servers = {s["name"]: s for s in c.get("/v1/servers").json()}
        assert servers["noticias"]["agents"] == ["reporter"]
        rec = c.post("/v1/runs", json={"input": "x", "servers": ["agenda"]}).json()
        assert rec["status"] == "completed"
        assert c.post("/v1/runs", json={"input": "x", "servers": ["internet"]}).status_code == 422
        assert c.post("/v1/runs", json={"input": "x", "servers": ["Bad Name"]}).status_code == 422
        assert c.post("/v1/runs", json={"input": "x", "servers": ["a"] * 11}).status_code == 422
        r = c.post("/v1/runs", json={"input": "x", "agent": "reporter", "servers": ["brasil"]})
        assert r.status_code == 404
        with c.stream("POST", "/v1/runs/stream", json={"input": "x", "servers": ["internet"]}) as s:
            assert s.status_code == 422
