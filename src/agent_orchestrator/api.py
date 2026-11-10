"""HTTP API."""

import asyncio
import dataclasses
import hmac
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .agent import Event
from .config import StudioAgent, load_config
from .orchestrator import Orchestrator, RunRecord, build_providers


class RunRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    agent: str | None = None
    session_id: str | None = Field(default=None, max_length=100)
    include_events: bool = False
    # MCP servers on for this run; omitted means all. Agents left without tools leave the team.
    servers: list[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=40)]] | None = Field(
        default=None, max_length=10
    )


class StudioRunRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    agent: StudioAgent
    include_events: bool = False


def create_app(orchestrator: Orchestrator | None = None, api_keys: list[str] | None = None) -> FastAPI:
    keys = api_keys if api_keys is not None else [k for k in os.getenv("ORCH_API_KEYS", "").split(",") if k]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if orchestrator is not None:
            app.state.orch = orchestrator
            yield
            return
        config = load_config(os.getenv("ORCH_CONFIG", "agents.yaml"))
        orch = Orchestrator(config, build_providers(config))
        await orch.start()
        app.state.orch = orch
        try:
            yield
        finally:
            await orch.aclose()

    app = FastAPI(title="agent-orchestrator", version=__version__, lifespan=lifespan)

    def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if keys and not any(hmac.compare_digest(x_api_key or "", k) for k in keys):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def orch() -> Orchestrator:
        return app.state.orch  # type: ignore[no-any-return]

    Orch = Annotated[Orchestrator, Depends(orch)]

    def check_run(o: Orchestrator, req: RunRequest) -> None:
        try:
            team = o.team(req.servers)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        name = req.agent or o.config.default_agent
        if name not in team:
            raise HTTPException(404, f"agent {name!r} not available; available: {sorted(team)}")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/agents", dependencies=[Depends(auth)])
    def list_agents(o: Orch) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "description": a.config.description,
                "model": a.config.model_for(a.config.provider_chain[0]),
                "provider": a.config.provider,
                "tools": [{"name": t.name, "source": t.source} for t in a.tools.values()],
                "default": a.name == o.config.default_agent,
            }
            for a in o.agents.values()
        ]

    @app.get("/v1/servers", dependencies=[Depends(auth)])
    def list_servers(o: Orch) -> list[dict[str, Any]]:
        return o.servers()

    @app.post("/v1/runs", dependencies=[Depends(auth)])
    async def create_run(req: RunRequest, o: Orch) -> dict[str, Any]:
        check_run(o, req)
        rec = await o.run(req.input, agent=req.agent, session_id=req.session_id, servers=req.servers)
        return _public(rec, req.include_events)

    @app.post("/v1/runs/stream", dependencies=[Depends(auth)])
    async def stream_run(req: RunRequest, o: Orch) -> EventSourceResponse:
        check_run(o, req)
        return _stream(
            lambda sink: o.run(req.input, agent=req.agent, session_id=req.session_id, sink=sink, servers=req.servers)
        )

    # ---------------------------------------------------------------- studio: agents written by the caller

    def studio_check(o: Orchestrator, req: StudioRunRequest) -> None:
        try:
            o.studio_agent(req.agent)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/studio", dependencies=[Depends(auth)])
    def studio_catalog(o: Orch) -> dict[str, Any]:
        try:
            return o.studio_catalog()
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/v1/studio/runs", dependencies=[Depends(auth)])
    async def studio_run(req: StudioRunRequest, o: Orch) -> dict[str, Any]:
        studio_check(o, req)
        return _public(await o.run_studio(req.agent, req.input), req.include_events)

    @app.post("/v1/studio/runs/stream", dependencies=[Depends(auth)])
    async def studio_stream(req: StudioRunRequest, o: Orch) -> EventSourceResponse:
        studio_check(o, req)
        return _stream(lambda sink: o.run_studio(req.agent, req.input, sink=sink))

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str, o: Orch) -> dict[str, Any]:
        rec = o.runs.get(run_id)
        if rec is None:
            raise HTTPException(404, "run not found")
        return _public(rec, True)

    return app


def _stream(start: Callable[[Callable[[Event], Awaitable[None]]], Awaitable[RunRecord]]) -> EventSourceResponse:
    """Run with every event sent as SSE, then ``done`` with the run record."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def sink(event: Event) -> None:
        await queue.put({"event": event.type, "data": json.dumps(dataclasses.asdict(event), ensure_ascii=False)})

    async def produce() -> None:
        try:
            rec = await start(sink)
            await queue.put({"event": "done", "data": json.dumps(_public(rec, False), ensure_ascii=False)})
        finally:
            await queue.put(None)

    async def events() -> AsyncIterator[dict[str, Any]]:
        task = asyncio.create_task(produce())
        try:
            while (item := await queue.get()) is not None:
                yield item
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(events())


def _public(rec: RunRecord, include_events: bool) -> dict[str, Any]:
    data = dataclasses.asdict(rec)
    if not include_events:
        data.pop("events")
    return data


def main() -> None:
    import uvicorn

    uvicorn.run(
        "agent_orchestrator.api:create_app",
        factory=True,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
    )
