"""HTTP API."""

import asyncio
import dataclasses
import hmac
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .agent import Event
from .config import load_config
from .orchestrator import Orchestrator, RunRecord, build_providers


class RunRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    agent: str | None = None
    session_id: str | None = Field(default=None, max_length=100)
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

    def check_agent(o: Orchestrator, name: str | None) -> None:
        if name and name not in o.agents:
            raise HTTPException(404, f"unknown agent {name!r}; available: {sorted(o.agents)}")

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

    @app.post("/v1/runs", dependencies=[Depends(auth)])
    async def create_run(req: RunRequest, o: Orch) -> dict[str, Any]:
        check_agent(o, req.agent)
        rec = await o.run(req.input, agent=req.agent, session_id=req.session_id)
        return _public(rec, req.include_events)

    @app.post("/v1/runs/stream", dependencies=[Depends(auth)])
    async def stream_run(req: RunRequest, o: Orch) -> EventSourceResponse:
        check_agent(o, req.agent)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def sink(event: Event) -> None:
            await queue.put({"event": event.type, "data": json.dumps(dataclasses.asdict(event), ensure_ascii=False)})

        async def produce() -> None:
            try:
                rec = await o.run(req.input, agent=req.agent, session_id=req.session_id, sink=sink)
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

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str, o: Orch) -> dict[str, Any]:
        rec = o.runs.get(run_id)
        if rec is None:
            raise HTTPException(404, "run not found")
        return _public(rec, True)

    return app


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
