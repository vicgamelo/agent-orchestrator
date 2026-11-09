"""Anthropic Messages API provider (``pip install agent-orchestrator[anthropic]``)."""

from __future__ import annotations

from typing import Any

from .base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("install the extra: pip install 'agent-orchestrator[anthropic]'") from exc
            client = anthropic.AsyncAnthropic(api_key=api_key)
        self._client = client

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
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": to_anthropic(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"anthropic: {exc}") from exc
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            model=resp.model,
            stop_reason=resp.stop_reason,
        )


def to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert to Anthropic format: tool results are user turns made of
    ``tool_result`` blocks, and consecutive results must share one turn."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant" and m.tool_calls:
            content: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
            content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
            out.append({"role": "assistant", "content": content})
        elif m.role in ("user", "assistant"):
            out.append({"role": m.role, "content": m.content})
    return out
