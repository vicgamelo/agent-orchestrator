"""Ordered provider chain: try the next provider when one is rate limited or down.

Built for free tiers (Groq first, OpenRouter ``:free`` as backup): a 429, 5xx or
timeout moves on to the next provider; a bad request (400, 401...) does not,
since another provider would reject it too or it is a config error that should
surface. A short ``Retry-After`` is waited out and the same provider retried
once; a long one puts that provider on cooldown so later calls skip it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ..tracing import TraceSpan, current_span
from .base import LLMError, LLMProvider, LLMResponse, Message, ProviderError, ToolSpec


@dataclass(frozen=True)
class Route:
    provider: LLMProvider
    model: str | None = None  # None: use the model passed to ``complete``


@dataclass(frozen=True)
class Attempt:
    provider: str
    model: str
    error: str


class AllProvidersFailed(LLMError):
    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        tried = "; ".join(f"{a.provider} ({a.model}): {a.error}" for a in attempts)
        super().__init__(f"all providers failed [{', '.join(a.provider for a in attempts)}]: {tried}")


class FallbackLLM:
    name = "fallback"

    def __init__(
        self,
        routes: Sequence[Route],
        *,
        max_retry_after: float = 5.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not routes:
            raise ValueError("FallbackLLM needs at least one provider")
        self.routes = list(routes)
        self.max_retry_after = max_retry_after
        self._sleep = sleep
        self._clock = clock
        self._cooldown_until: dict[int, float] = {}

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
        attempts: list[Attempt] = []
        span = current_span()
        for i, route in enumerate(self.routes):
            name, route_model = route.provider.name, route.model or model
            wait_left = self._cooldown_until.get(i, 0.0) - self._clock()
            if wait_left > 0:
                attempts.append(Attempt(name, route_model, f"skipped, rate limited for {wait_left:.0f}s more"))
                continue
            retried = False
            while True:
                try:
                    resp = await route.provider.complete(
                        model=route_model,
                        system=system,
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except ProviderError as exc:
                    if not exc.retryable:
                        # Bad request, auth error, or the answer broke off after it started:
                        # don't replay it somewhere else.
                        self._annotate(span, name, attempts)
                        raise
                    wait = exc.retry_after
                    if wait is not None and wait <= self.max_retry_after and not retried:
                        retried = True
                        await self._sleep(wait)
                        continue
                    if wait is not None:
                        self._cooldown_until[i] = self._clock() + wait
                    attempts.append(Attempt(name, route_model, str(exc)))
                    break
                self._cooldown_until.pop(i, None)
                self._annotate(span, name, attempts)
                return resp
        self._annotate(span, None, attempts)
        raise AllProvidersFailed(attempts)

    @staticmethod
    def _annotate(span: TraceSpan | None, provider: str | None, attempts: list[Attempt]) -> None:
        if span is None:
            return
        if provider:
            span.provider = provider
        if attempts:
            span.attributes["fallback.failed"] = [a.__dict__ for a in attempts]
