import asyncio
import importlib
import sys
import types
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from avalan.entities import (
    Message,
    MessageRole,
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallResult,
    ToolCallToken,
    TransformerEngineSettings,
)


class AsyncIter:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:  # pragma: no cover - helper
            raise StopAsyncIteration from exc


@pytest.fixture(scope="module")
def anthropic_mod():
    class DeltaEvent:
        def __init__(self, delta, index=0):
            self.delta = delta
            self.index = index

    class StopEvent:
        pass

    stub = types.ModuleType("anthropic")
    stub.AsyncAnthropic = MagicMock()
    types_mod = types.ModuleType("anthropic.types")
    types_mod.RawContentBlockDeltaEvent = DeltaEvent
    types_mod.RawMessageStopEvent = StopEvent
    stub.types = types_mod
    patcher = patch.dict(
        sys.modules, {"anthropic": stub, "anthropic.types": types_mod}
    )
    patcher.start()
    mod = importlib.import_module("avalan.model.nlp.text.vendor.anthropic")
    yield mod, stub
    patcher.stop()


def test_stream_variants(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="tid", name="tname"
            ),
            index=0,
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace())
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(thinking="think"))
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"a":1}')
        )
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json="frag"), index=1
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(text="txt"))
        yield SimpleNamespace(
            type="content_block_stop",
            content_block=SimpleNamespace(
                type="tool_use", id="tid", name="tname", input={"x": 1}
            ),
            index=0,
        )
        yield SimpleNamespace(type="message_stop")

    async def collect():
        with patch.object(
            mod.TextGenerationVendor,
            "build_tool_call_token",
            return_value="call",
        ) as btt:
            stream = mod.AnthropicStream(agen())
            out = []
            async for token in stream:
                out.append(token)
        return out, btt

    out, btt = asyncio.run(collect())

    assert len(out) == 5
    assert isinstance(out[0], ReasoningToken)
    assert out[0].token == "think"
    assert isinstance(out[1], ToolCallToken)
    assert out[1].token == '{"a":1}'
    assert isinstance(out[2], ToolCallToken)
    assert out[2].token == "frag"
    assert isinstance(out[3], Token)
    assert out[3].token == "txt"
    assert out[4] == "call"
    btt.assert_called_once_with("tid", "tname", {"x": 1})


def test_client_call_and_model(anthropic_mod):
    mod, stub = anthropic_mod
    ctx_instance = SimpleNamespace(
        __aenter__=AsyncMock(return_value=AsyncIter([])),
        __aexit__=AsyncMock(return_value=False),
    )
    stub.AsyncAnthropic.return_value.messages.stream = MagicMock(
        return_value=ctx_instance
    )
    exit_stack = AsyncMock(spec=AsyncExitStack)
    client = mod.AnthropicClient("tok", "url", exit_stack=exit_stack)
    client._system_prompt = MagicMock(return_value="sys")
    client._template_messages = MagicMock(return_value=[{"content": "c"}])

    async def invoke():
        with patch.object(
            mod.AnthropicClient, "_tool_schemas", return_value=[{"n": 1}]
        ) as ts:
            result = await client("m", [], tool=MagicMock())
        return result, ts

    result, ts = asyncio.run(invoke())
    stub.AsyncAnthropic.assert_called_once_with(api_key="tok", base_url="url")
    exit_stack.enter_async_context.assert_awaited_once_with(ctx_instance)
    client._client.messages.stream.assert_called_once()
    ts.assert_called_once()
    assert isinstance(result, mod.AnthropicStream)

    with patch.object(mod, "AnthropicClient") as ClientMock:
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="t",
            base_url="b",
        )
        model = mod.AnthropicModel("m", settings)
        loaded = model._load_model()
    ClientMock.assert_called_once_with(
        api_key="t", base_url="b", exit_stack=model._exit_stack
    )
    assert loaded is ClientMock.return_value


def test_template_messages_and_exclude_roles(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("k", exit_stack=exit_stack)

    @dataclass
    class Res:
        x: int

    call = ToolCall(id="id1", name="pkg.tool", arguments={"a": 1})
    result = ToolCallResult(
        id="id1", name="pkg.tool", call=call, result=Res(x=2)
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content="s"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="ok"),
        Message(role=MessageRole.TOOL, tool_call_result=result),
    ]
    templated = client._template_messages(messages, ["system"])
    assert templated[1]["content"][1]["name"] == "pkg__tool"
    assert templated[2]["content"][0]["tool_use_id"] == "id1"

    dup = client._template_messages(
        [
            Message(role=MessageRole.USER, content="x"),
            Message(role=MessageRole.USER, content="x"),
        ]
    )
    assert len(dup) == 1


def test_tool_schemas_variants(anthropic_mod):
    mod, _ = anthropic_mod

    class DummyTool:
        def __init__(self, schemas):
            self._schemas = schemas

        def json_schemas(self):
            return self._schemas

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "pkg.tool",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {"type": "noop"},
    ]
    out = mod.AnthropicClient._tool_schemas(DummyTool(schemas))
    assert out == [
        {
            "name": "pkg__tool",
            "description": "d",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]
    assert mod.AnthropicClient._tool_schemas(DummyTool([])) is None
    assert mod.AnthropicClient._tool_schemas(DummyTool(None)) is None
