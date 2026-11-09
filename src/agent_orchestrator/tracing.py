"""Run tracing, exported to a genai-lens collector (https://github.com/vicgamelo/genai-lens).

Spans nest through ``contextvars`` (so parallel tool calls and delegated agents
land in the right place) and are shipped in one batch at the end of each run.
Export failures are logged, never raised.
"""

from __future__ import annotations

import contextvars
import logging
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar[TraceSpan | None] = contextvars.ContextVar("orch_span", default=None)


@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start_time_ns: int
    end_time_ns: int | None = None
    status: str = "ok"
    error: str | None = None
    model: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input: str | None = None
    output: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, endpoint: str | None = None, api_key: str = "", project: str = "agent-orchestrator") -> None:
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.api_key = api_key
        self.project = project
        self._buffers: dict[str, list[TraceSpan]] = {}

    @contextmanager
    def span(self, name: str, kind: str, **fields: Any) -> Iterator[TraceSpan]:
        parent = _current.get()
        span = TraceSpan(
            trace_id=parent.trace_id if parent else secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_time_ns=time.time_ns(),
            **fields,
        )
        token = _current.set(span)
        try:
            yield span
        except BaseException as exc:
            span.status, span.error = "error", f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            _current.reset(token)
            span.end_time_ns = time.time_ns()
            self._buffers.setdefault(span.trace_id, []).append(span)

    def take(self, trace_id: str) -> list[TraceSpan]:
        return self._buffers.pop(trace_id, [])

    async def export(self, trace_id: str) -> None:
        spans = self.take(trace_id)
        if not self.endpoint or not spans:
            return
        payload = {"project": self.project, "spans": [s.__dict__ for s in spans]}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(f"{self.endpoint}/v1/spans", json=payload, headers={"X-Lens-Key": self.api_key})
                r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("trace export failed: %s", exc)
