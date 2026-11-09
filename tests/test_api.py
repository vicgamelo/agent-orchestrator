import asyncio
import json

from fastapi.testclient import TestClient

from agent_orchestrator.api import create_app
from agent_orchestrator.llm.scripted import ScriptedProvider, Step
from agent_orchestrator.orchestrator import Orchestrator

from .conftest import make_config


def client(keys: list[str] | None = None) -> TestClient:
    provider = ScriptedProvider(
        {
            "coordinator": [Step(calls=[("delegate_to_economist", {"task": "t"})]), Step(text="final")],
            "economist": [Step(text="13.75")],
            "researcher": [Step(text="r")],
        }
    )
    orch = Orchestrator(make_config(), {"fake": provider})
    asyncio.run(orch.start())
    return TestClient(create_app(orch, api_keys=keys or []))


def test_agents_listing() -> None:
    with client() as c:
        agents = {a["name"]: a for a in c.get("/v1/agents").json()}
        assert agents["coordinator"]["default"] is True
        assert {t["name"] for t in agents["coordinator"]["tools"]} == {
            "delegate_to_economist",
            "delegate_to_researcher",
        }


def test_run_and_fetch() -> None:
    with client() as c:
        rec = c.post("/v1/runs", json={"input": "selic?"}).json()
        assert rec["output"] == "final" and rec["status"] == "completed" and "events" not in rec
        again = c.get(f"/v1/runs/{rec['run_id']}").json()
        assert again["events"][0]["type"] == "agent_start"
        assert c.get("/v1/runs/run_missing").status_code == 404


def test_unknown_agent_and_validation() -> None:
    with client() as c:
        assert c.post("/v1/runs", json={"input": "x", "agent": "ghost"}).status_code == 404
        assert c.post("/v1/runs", json={"input": ""}).status_code == 422


def test_auth() -> None:
    with client(keys=["secret"]) as c:
        assert c.get("/v1/agents").status_code == 401
        assert c.get("/v1/agents", headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_stream_emits_events_then_done() -> None:
    with client() as c, c.stream("POST", "/v1/runs/stream", json={"input": "selic?"}) as r:
        body = "".join(r.iter_text())
    events = [line.removeprefix("event: ").strip() for line in body.splitlines() if line.startswith("event:")]
    assert events[0] == "agent_start" and events[-1] == "done"
    assert "tool_call" in events and events.count("agent_start") == 2
    done = [line for line in body.splitlines() if line.startswith("data:")][-1]
    assert json.loads(done.removeprefix("data:").strip())["output"] == "final"
