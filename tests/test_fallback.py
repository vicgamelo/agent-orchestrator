"""FallbackLLM over real OpenAICompatibleProvider instances, with a fake HTTP transport."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from agent_orchestrator.config import AppConfig
from agent_orchestrator.llm.base import ProviderError
from agent_orchestrator.llm.fallback import AllProvidersFailed, FallbackLLM, Route
from agent_orchestrator.llm.openai_compat import OpenAICompatibleProvider, parse_retry_after
from agent_orchestrator.orchestrator import Orchestrator
from agent_orchestrator.tracing import Tracer

pytestmark = pytest.mark.anyio

Reply = httpx.Response | Exception


def ok(text: str, model: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )


class FakeServer:
    """Answers each request with the next scripted reply (the last one repeats)."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.models: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.models.append(json.loads(request.content)["model"])
        reply = self.replies[min(len(self.models), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def provider(self, name: str) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            f"https://{name}.test/v1", name=name, http=httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def chain(groq: FakeServer, openrouter: FakeServer, clock: Clock | None = None) -> FallbackLLM:
    clock = clock or Clock()
    return FallbackLLM(
        [Route(groq.provider("groq"), "gpt-oss"), Route(openrouter.provider("openrouter"), "gemma:free")],
        sleep=clock.sleep,
        clock=clock,
    )


async def ask(llm: FallbackLLM) -> str:
    resp = await llm.complete(model="unused", system="s", messages=[], tools=[], max_tokens=10, temperature=0)
    return resp.text


async def test_429_moves_to_next_provider() -> None:
    groq, openrouter = FakeServer(httpx.Response(429, text="rate limited")), FakeServer(ok("hi", "gemma:free"))
    assert await ask(chain(groq, openrouter)) == "hi"
    assert (groq.models, openrouter.models) == (["gpt-oss"], ["gemma:free"])


async def test_500_moves_to_next_provider() -> None:
    groq, openrouter = FakeServer(httpx.Response(500, text="boom")), FakeServer(ok("hi", "gemma:free"))
    assert await ask(chain(groq, openrouter)) == "hi"
    assert openrouter.models == ["gemma:free"]


async def test_timeout_moves_to_next_provider() -> None:
    groq, openrouter = FakeServer(httpx.ReadTimeout("slow")), FakeServer(ok("hi", "gemma:free"))
    assert await ask(chain(groq, openrouter)) == "hi"


async def test_groq_413_tokens_per_minute_is_a_rate_limit() -> None:
    tpm = {"error": {"message": "Request too large ... tokens per minute (TPM)", "code": "rate_limit_exceeded"}}
    groq, openrouter = FakeServer(httpx.Response(413, json=tpm)), FakeServer(ok("hi", "gemma:free"))
    assert await ask(chain(groq, openrouter)) == "hi"


async def test_all_failing_lists_every_provider_tried() -> None:
    groq, openrouter = FakeServer(httpx.Response(429, text="slow down")), FakeServer(httpx.Response(503, text="down"))
    with pytest.raises(AllProvidersFailed) as info:
        await ask(chain(groq, openrouter))
    assert [a.provider for a in info.value.attempts] == ["groq", "openrouter"]
    msg = str(info.value)
    assert msg.startswith("all providers failed [groq, openrouter]")
    assert "429" in msg and "503" in msg and "gemma:free" in msg


async def test_short_retry_after_is_waited_and_same_provider_retried() -> None:
    clock = Clock()
    groq = FakeServer(httpx.Response(429, headers={"retry-after": "2"}), ok("groq ok", "gpt-oss"))
    openrouter = FakeServer(ok("or", "gemma:free"))
    assert await ask(chain(groq, openrouter, clock)) == "groq ok"
    assert clock.sleeps == [2.0]
    assert groq.models == ["gpt-oss", "gpt-oss"] and openrouter.models == []


async def test_short_retry_after_is_retried_only_once() -> None:
    clock = Clock()
    groq = FakeServer(httpx.Response(429, headers={"retry-after": "1"}))
    openrouter = FakeServer(ok("or", "gemma:free"))
    assert await ask(chain(groq, openrouter, clock)) == "or"
    assert clock.sleeps == [1.0] and len(groq.models) == 2


async def test_long_retry_after_skips_provider_until_it_expires() -> None:
    clock = Clock()
    groq = FakeServer(httpx.Response(429, headers={"retry-after": "60"}), ok("groq back", "gpt-oss"))
    openrouter = FakeServer(ok("or", "gemma:free"))
    llm = chain(groq, openrouter, clock)

    assert await ask(llm) == "or"
    assert clock.sleeps == []  # never blocks a request for longer than max_retry_after
    assert await ask(llm) == "or"
    assert len(groq.models) == 1  # second call skipped groq: still cooling down

    clock.now += 61
    assert await ask(llm) == "groq back"


async def test_bad_request_does_not_switch_provider() -> None:
    groq = FakeServer(httpx.Response(400, json={"error": {"message": "tool schema invalid"}}))
    openrouter = FakeServer(ok("or", "gemma:free"))
    with pytest.raises(ProviderError, match="400") as info:
        await ask(chain(groq, openrouter))
    assert info.value.provider == "groq" and not info.value.retryable
    assert openrouter.models == []


async def test_answer_that_broke_off_after_starting_is_not_replayed() -> None:
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"choices": [{"message": {"content": "par'
            raise httpx.ReadError("connection reset")

    groq = FakeServer(httpx.Response(200, stream=BrokenBody()))
    openrouter = FakeServer(ok("or", "gemma:free"))
    with pytest.raises(ProviderError, match="broke off"):
        await ask(chain(groq, openrouter))
    assert openrouter.models == []


def test_parse_retry_after() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("0.5") == 0.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


async def test_span_records_the_provider_that_answered() -> None:
    groq, openrouter = FakeServer(httpx.Response(429, text="slow down")), FakeServer(ok("final", "gemma:free"))
    config = AppConfig.model_validate(
        {
            "providers": {
                "groq": {"type": "openai", "base_url": "https://groq.test/v1"},
                "openrouter": {"type": "openai", "base_url": "https://openrouter.test/v1"},
            },
            "agents": [
                {
                    "name": "coordinator",
                    "description": "d",
                    "provider": ["groq", "openrouter"],
                    "models": {"groq": "gpt-oss", "openrouter": "gemma:free"},
                    "instructions": "i",
                }
            ],
            "default_agent": "coordinator",
        }
    )
    orch = Orchestrator(
        config,
        {"groq": groq.provider("groq"), "openrouter": openrouter.provider("openrouter")},
        tracer=Tracer(),
    )
    await orch.start()
    exported: list[Any] = []

    async def capture(trace_id: str) -> None:
        exported.extend(orch.tracer.take(trace_id))

    orch.tracer.export = capture  # type: ignore[method-assign]
    rec = await orch.run("hello")
    import asyncio

    await asyncio.sleep(0)
    assert rec.output == "final" and rec.status == "completed"
    llm_span = next(s for s in exported if s.kind == "llm")
    assert llm_span.provider == "openrouter"
    assert llm_span.model == "gemma:free"
    assert [a["provider"] for a in llm_span.attributes["fallback.failed"]] == ["groq"]
