"""OpenAI Chat Completions compatible provider.

Covers OpenAI itself and every server that speaks the same API: Ollama
(``http://localhost:11434/v1``), vLLM, LM Studio, Groq, Together, OpenRouter...
"""

from __future__ import annotations

import json
import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .base import LLMResponse, Message, ProviderError, ToolCall, ToolSpec, Usage, is_retryable_status


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
        data = await self._post(body)
        if not data.get("choices"):
            # Some gateways (OpenRouter) report upstream failures as 200 + {"error": ...}.
            err = data.get("error")
            detail = err.get("message", err) if isinstance(err, dict) else err or str(data)[:300]
            raise ProviderError(f"{self.name} returned no choices: {detail}", provider=self.name, status=200)
        choice = data["choices"][0]
        msg = choice.get("message") or {}
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

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Errors before the server starts answering (status >= 400, connect error,
        timeout waiting for headers) are marked retryable when another provider could
        help. Once a 200 has started streaming in, failures are not: the request was
        accepted and may have been billed/counted, so the caller should not replay it."""
        try:
            async with self._http.stream("POST", self._url, json=body) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode(errors="replace")
                    raise ProviderError(
                        f"{self.name} returned {response.status_code}: {text[:300]}",
                        provider=self.name,
                        status=response.status_code,
                        retryable=is_retryable_status(response.status_code, text),
                        retry_after=parse_retry_after(response.headers.get("retry-after")),
                    )
                try:
                    await response.aread()
                    data = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise ProviderError(
                        f"{self.name} response broke off after it started: {exc!r}",
                        provider=self.name,
                        status=response.status_code,
                    ) from exc
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name} timed out: {exc!r}", provider=self.name, retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} request failed: {exc!r}", provider=self.name, retryable=True) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{self.name} returned unexpected JSON: {str(data)[:300]}", provider=self.name)
        return data


def parse_retry_after(value: str | None) -> float | None:
    """``Retry-After`` is either delta-seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


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
