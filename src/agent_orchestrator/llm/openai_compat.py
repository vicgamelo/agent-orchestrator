"""OpenAI Chat Completions compatible provider.

Covers OpenAI itself and every server that speaks the same API: Ollama
(``http://localhost:11434/v1``), vLLM, LM Studio, Groq, Together, OpenRouter...
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        *,
        name: str = "openai",
        timeout: float = 120.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http or httpx.AsyncClient(timeout=timeout, headers=headers)

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
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *map(_to_openai, messages)],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        try:
            response = await self._http.post(self._url, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"{self.name} returned {exc.response.status_code}: {exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc

        data = response.json()
        choice = data["choices"][0]
        msg = choice["message"]
        calls: list[ToolCall] = []
        for raw in msg.get("tool_calls") or []:
            fn = raw["function"]
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                parsed = {"_raw": args}
            calls.append(ToolCall(id=raw.get("id") or f"call_{len(calls)}", name=fn["name"], arguments=parsed))
        usage = data.get("usage") or {}
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=calls,
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            model=data.get("model", model),
            stop_reason=choice.get("finish_reason"),
        )


def _to_openai(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in m.tool_calls
        ]
    return out
