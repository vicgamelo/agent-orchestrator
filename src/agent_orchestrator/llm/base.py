"""Provider-neutral message and response types.

Internally we use an OpenAI-like shape (role + content + tool_calls) because it
maps 1:1 to most providers; each provider adapter converts at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for role="tool"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    stop_reason: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (object)


class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


class ProviderError(LLMError):
    """A failed call to one provider, classified so a fallback chain knows whether
    trying the next provider can help (rate limit, 5xx, timeout) or not (bad request)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


def is_retryable_status(status: int, body: str = "") -> bool:
    """429, 408 and 5xx are worth another provider. Groq also answers 413 when a
    request goes over the free tier's tokens-per-minute limit, which is a rate
    limit too; a 413 for any other reason is not."""
    if status in (408, 429) or status >= 500:
        return True
    return status == 413 and "rate_limit" in body
