"""Agent studio: a caller writes one agent and runs it, within the models, tools and limits allowed."""

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError

from agent_orchestrator.api import create_app
from agent_orchestrator.config import StudioAgent
from agent_orchestrator.llm.scripted import ScriptedProvider, Step
from agent_orchestrator.orchestrator import Orchestrator

from .conftest import make_config

pytestmark = pytest.mark.anyio


def mcp(name: str, *tools: str) -> MCPServer:
    server = MCPServer(name=name)

    def make(tool: str) -> Any:
        def fn(q: str = "") -> dict[str, str]:
            return {"from": f"{name}.{tool}", "q": q}

        return fn

    for tool in tools:
        server.add_tool(make(tool), name=tool, description=f"{tool} from {name}")
    return server


STUDIO = {
    "models": [
        {"id": "big", "label": "Big", "provider": "fake", "model": "big-model"},
        {"id": "small", "label": "Small", "provider": "fake", "model": "small-model"},
        {"id": "nokey", "label": "No key", "provider": "paid", "model": "x:free"},
        {"id": "local", "label": "Local", "provider": "fake", "model": "local-model", "enabled": False},
    ],
    "tools": ["calculator", "brasil__*", "noticias__*"],
    "templates": ["economist", "reporter", "coordinator"],
    "max_steps": 4,
    "max_tokens": 300,
    "instructions_max_chars": 200,
}


def config(studio: dict[str, Any] | None = STUDIO) -> Any:
    extra: dict[str, Any] = {} if studio is None else {"studio": studio}
    return make_config(
        providers={
            "fake": {"type": "openai", "base_url": "http://unused"},
            # api_key "" is what ${OPENROUTER_API_KEY:-} expands to when the key isn't set
            "paid": {"type": "openai", "base_url": "http://unused", "api_key": "", "free_models_only": True},
        },
        mcp_servers=[
            {"name": "brasil", "url": "http://unused"},
            {"name": "noticias", "url": "http://unused"},
        ],
        agents=[
            {
                "name": "coordinator",
                "description": "boss",
                "provider": "fake",
                "model": "coordinator",
                "instructions": "{team}",
                "delegates": ["economist", "reporter"],
            },
            {
                "name": "economist",
                "description": "Banco Central numbers",
                "provider": "fake",
                "model": "small-model",
                "instructions": "You are an economist.",
                "tools": ["brasil__selic", "calculator", "current_datetime"],
                "temperature": 0.1,
                "max_steps": 8,
            },
            {
                "name": "reporter",
                "description": "news",
                "provider": "fake",
                "model": "other-model",
                "instructions": "You read the news.",
                "tools": ["noticias__*"],
            },
        ],
        **extra,
    )


SCRIPTS = {
    "big-model": [Step(calls=[("brasil__selic", {"q": "hoje"})]), Step(text="Selic é 15%")],
    "small-model": [Step(text="pequeno")],
    "coordinator": [Step(text="ok")],
}


async def started(*, noticias_up: bool = True, studio: dict[str, Any] | None = STUDIO) -> tuple[Orchestrator, Any]:
    provider = ScriptedProvider(SCRIPTS)
    orch = Orchestrator(config(studio), {"fake": provider, "paid": ScriptedProvider({})})

    class Broken:
        pass

    await orch.start(
        mcp_targets={
            "brasil": mcp("brasil", "selic", "cep"),
            "noticias": mcp("noticias", "buscar") if noticias_up else Broken(),
        }
    )
    return orch, provider


def spec(**kw: Any) -> StudioAgent:
    base: dict[str, Any] = {"instructions": "Responda em português.", "model": "big", "tools": ["brasil__selic"]}
    return StudioAgent.model_validate(base | kw)


async def test_catalog_lists_enabled_models_allowed_tools_and_templates() -> None:
    orch, _ = await started()
    try:
        cat = orch.studio_catalog()
        assert cat["models"] == [
            {"id": "big", "label": "Big", "provider": "fake", "available": True},
            {"id": "small", "label": "Small", "provider": "fake", "available": True},
            {"id": "nokey", "label": "No key", "provider": "paid", "available": False},
        ]
        tools = {t["name"]: t for t in cat["tools"]}
        # current_datetime is a builtin but not on the studio list
        assert set(tools) == {"calculator", "brasil__selic", "brasil__cep", "noticias__buscar"}
        assert tools["brasil__cep"]["server"] == "brasil"
        assert tools["calculator"]["server"] is None
        assert tools["brasil__cep"]["description"] == "cep from brasil"
        templates = {t["name"]: t for t in cat["templates"]}
        econ = templates["economist"]
        # tools outside the studio list are left out; its own model maps to the "small" entry
        assert econ["tools"] == ["brasil__selic", "calculator"]
        assert econ["model"] == "small"
        assert econ["instructions"] == "You are an economist."
        assert econ["temperature"] == 0.1
        assert econ["max_steps"] == 4  # capped at the studio limit
        # a model that isn't on the list: the first available one
        assert templates["reporter"]["model"] == "big"
        assert cat["limits"] == {"max_steps": 4, "max_tokens": 300, "instructions_max_chars": 200}
    finally:
        await orch.aclose()


async def test_template_and_tools_of_a_down_server_are_left_out() -> None:
    orch, _ = await started(noticias_up=False)
    try:
        cat = orch.studio_catalog()
        assert "reporter" not in {t["name"] for t in cat["templates"]}
        assert "noticias__buscar" not in {t["name"] for t in cat["tools"]}
        with pytest.raises(ValueError, match="tools not available"):
            orch.studio_agent(spec(tools=["noticias__buscar"]))
    finally:
        await orch.aclose()


async def test_studio_run_uses_only_what_the_caller_chose() -> None:
    orch, provider = await started()
    try:
        rec = await orch.run_studio(spec(temperature=0.7, max_steps=3), "Qual a Selic?")
        assert rec.status == "completed"
        assert rec.output == "Selic é 15%"
        assert rec.agent == "studio"
        assert rec.tool_calls == 1
        first = provider.requests[0]
        assert first["model"] == "big-model"
        assert first["system"] == "Responda em português."
        assert first["tools"] == ["brasil__selic"]  # not brasil__cep, not delegation
        tool_result = next(e for e in rec.events if e["type"] == "tool_result")
        assert json.loads(tool_result["result"]) == {"from": "brasil.selic", "q": "hoje"}
        # the configured team is untouched
        assert "delegate_to_economist" in orch.agents["coordinator"].tools
    finally:
        await orch.aclose()


async def test_studio_agent_limits() -> None:
    orch, _ = await started()
    try:
        agent = orch.studio_agent(spec(tools=["calculator", "calculator", "brasil__cep"], temperature=1.2))
        assert list(agent.tools) == ["calculator", "brasil__cep"]  # repeats dropped
        assert agent.config.max_tokens == 300  # the studio cap, not the caller's
        assert agent.config.temperature == 1.2
        assert orch.studio_agent(spec(tools=[])).tools == {}
        cases = [
            (spec(model="nope"), "unknown model"),
            (spec(model="local"), "unknown model"),  # disabled models don't exist for callers
            (spec(model="nokey"), "not available"),
            (spec(tools=["current_datetime"]), "tools not available"),
            (spec(tools=["delegate_to_economist"]), "tools not available"),
            (spec(tools=["noticias__*"]), "tools not available"),  # names, not wildcards
            (spec(instructions="x" * 201), "longer than 200"),
            (spec(max_steps=5), "max_steps above 4"),
        ]
        for bad, message in cases:
            with pytest.raises(ValueError, match=message):
                orch.studio_agent(bad)
    finally:
        await orch.aclose()


async def test_max_steps_stops_a_looping_studio_agent() -> None:
    orch, _ = await started()
    try:
        orch.providers["fake"] = ScriptedProvider({"big-model": [Step(calls=[("brasil__selic", {})])]})
        rec = await orch.run_studio(spec(max_steps=2), "loop")
        assert rec.status == "max_steps"
        assert rec.tool_calls == 2
    finally:
        await orch.aclose()


async def test_studio_not_configured() -> None:
    orch, _ = await started(studio=None)
    try:
        with pytest.raises(LookupError):
            orch.studio_catalog()
        with pytest.raises(LookupError):
            orch.studio_agent(spec())
    finally:
        await orch.aclose()


def test_studio_config_is_checked() -> None:
    def bad(**changes: Any) -> None:
        with pytest.raises(ValidationError):
            config(STUDIO | changes)

    bad(models=[{"id": "a", "label": "A", "provider": "ghost", "model": "m"}])
    bad(models=[{"id": "a", "label": "A", "provider": "paid", "model": "paid-model"}])  # not :free
    bad(models=[{"id": "a", "label": "A", "provider": "fake", "model": "m"}] * 2)
    bad(models=[{"id": "Has Space", "label": "A", "provider": "fake", "model": "m"}])
    bad(templates=["ghost"])
    bad(models=[])


def client(studio: dict[str, Any] | None = STUDIO) -> TestClient:
    orch, _ = asyncio.run(started(studio=studio))
    return TestClient(create_app(orch, api_keys=[]))


def body(**agent: Any) -> dict[str, Any]:
    base = {"instructions": "Responda em português.", "model": "big", "tools": ["brasil__selic"]}
    return {"input": "Qual a Selic?", "agent": base | agent}


def test_api_catalog_and_runs() -> None:
    with client() as c:
        cat = c.get("/v1/studio").json()
        assert [m["id"] for m in cat["models"]] == ["big", "small", "nokey"]

        rec = c.post("/v1/studio/runs", json=body()).json()
        assert rec["status"] == "completed"
        assert rec["output"] == "Selic é 15%"

        for bad in (body(model="nope"), body(tools=["current_datetime"]), body(max_steps=5)):
            r = c.post("/v1/studio/runs", json=bad)
            assert r.status_code == 422, bad
        # shape errors (pydantic) are 422 too
        assert c.post("/v1/studio/runs", json=body(name="x")).status_code == 422
        assert c.post("/v1/studio/runs", json=body(temperature=3)).status_code == 422
        assert c.post("/v1/studio/runs", json=body(tools=["t"] * 41)).status_code == 422
        assert c.post("/v1/studio/runs", json={"input": "", "agent": body()["agent"]}).status_code == 422


def test_api_stream_sends_events_then_done() -> None:
    with client() as c:
        with c.stream("POST", "/v1/studio/runs/stream", json=body()) as s:
            assert s.status_code == 200
            text = "".join(s.iter_text())
        events = [line.removeprefix("event: ").strip() for line in text.splitlines() if line.startswith("event:")]
        assert events[0] == "agent_start"
        assert "tool_call" in events and "tool_result" in events
        assert events[-1] == "done"
        with c.stream("POST", "/v1/studio/runs/stream", json=body(model="nokey")) as s:
            assert s.status_code == 422


def test_api_without_studio_is_404() -> None:
    with client(studio=None) as c:
        assert c.get("/v1/studio").status_code == 404
        assert c.post("/v1/studio/runs", json=body()).status_code == 404


def test_api_key_required() -> None:
    orch, _ = asyncio.run(started())
    with TestClient(create_app(orch, api_keys=["k"])) as c:
        assert c.get("/v1/studio").status_code == 401
        assert c.post("/v1/studio/runs", json=body()).status_code == 401
        assert c.get("/v1/studio", headers={"X-API-Key": "k"}).status_code == 200
