"""Deterministic provider for tests and offline demos.

Each agent gets a script: a list of steps, where a step is either a final text
answer or a list of tool calls. Scripts are keyed by the agent's ``model`` name
so several agents can share one provider instance.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .base import LLMResponse, Message, ToolCall, ToolSpec, Usage


@dataclass
class Step:
    text: str = ""
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stop_reason: str | None = None  # default: tool_use / end_turn


class ScriptedProvider:
    name = "scripted"

    def __init__(self, scripts: dict[str, list[Step]], delay: float = 0.0) -> None:
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._pos: dict[str, int] = defaultdict(int)
        self._ids = itertools.count(1)
        self.delay = delay
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        self.requests.append(
            {"model": model, "system": system, "messages": list(messages), "tools": [t.name for t in tools]}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        script = self._scripts[model]
        step = script[min(self._pos[model], len(script) - 1)]
        self._pos[model] += 1
        calls = [ToolCall(id=f"call_{next(self._ids)}", name=n, arguments=a) for n, a in step.calls]
        return LLMResponse(
            text=step.text,
            tool_calls=calls,
            usage=Usage(input_tokens=100, output_tokens=20),
            model=model,
            stop_reason=step.stop_reason or ("tool_use" if calls else "end_turn"),
        )
