import json

import httpx
import pytest

from agent_orchestrator.llm.anthropic import to_anthropic
from agent_orchestrator.llm.base import LLMError, Message, ToolCall, ToolSpec
from agent_orchestrator.llm.openai_compat import OpenAICompatibleProvider

pytestmark = pytest.mark.anyio


async def test_openai_compatible_round_trip() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen3.5:9b",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"x": 1}'}}
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    provider = OpenAICompatibleProvider(
        "http://ollama/v1", http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall("c0", "calc", {"x": 0})]),
        Message(role="tool", content="0", tool_call_id="c0"),
    ]
    resp = await provider.complete(
        model="qwen3.5:9b",
        system="sys",
        messages=history,
        tools=[ToolSpec("calc", "d", {"type": "object"})],
        max_tokens=10,
        temperature=0,
    )
    assert resp.tool_calls[0].arguments == {"x": 1}
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (11, 7)
    msgs = seen["messages"]
    assert isinstance(msgs, list)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"x": 0}'
    assert msgs[3] == {"role": "tool", "tool_call_id": "c0", "content": "0"}


async def test_openai_compatible_http_error() -> None:
    provider = OpenAICompatibleProvider(
        "http://x/v1",
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429, text="slow down"))),
    )
    with pytest.raises(LLMError, match="429"):
        await provider.complete(model="m", system="", messages=[], tools=[], max_tokens=1, temperature=0)


def test_anthropic_merges_parallel_tool_results_into_one_user_turn() -> None:
    out = to_anthropic(
        [
            Message(role="user", content="q"),
            Message(
                role="assistant", content="let me check", tool_calls=[ToolCall("a", "t1", {}), ToolCall("b", "t2", {})]
            ),
            Message(role="tool", content="r1", tool_call_id="a"),
            Message(role="tool", content="r2", tool_call_id="b"),
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]
