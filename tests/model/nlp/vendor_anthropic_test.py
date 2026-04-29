import asyncio
import importlib
import sys
import types
from base64 import b64encode
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageRole,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallError,
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
    class APIStatusError(Exception):
        def __init__(self, message, *, response=None, body=None):
            super().__init__(message)
            self.status_code = getattr(response, "status_code", None)
            self.body = body

    class NotFoundError(APIStatusError):
        pass

    class DeltaEvent:
        def __init__(self, delta, index=0):
            self.delta = delta
            self.index = index

    class StopEvent:
        pass

    stub = types.ModuleType("anthropic")
    stub.APIStatusError = APIStatusError
    stub.AsyncAnthropic = MagicMock()
    stub.NotFoundError = NotFoundError
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


def test_client_non_stream_tool_messages(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    call = ToolCall(id="call1", name="pkg.tool", arguments={"a": 1})
    result = ToolCallResult(
        id="call1", name="pkg.tool", call=call, result={"ok": True}
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="ack"),
        Message(role=MessageRole.TOOL, tool_call_result=result),
    ]

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(
                type="tool_use",
                id="call1",
                name="pkg__tool",
                input={"a": 1},
            ),
        ]
    )

    stub.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=response
    )

    with patch.object(
        mod.TextGenerationVendor,
        "build_tool_call_token",
        return_value=ToolCallToken(token="<tool_call />"),
    ) as build_token:
        stream = asyncio.run(
            client(
                "model",
                messages,
                use_async_generator=False,
            )
        )

    from avalan.model.stream import TextGenerationSingleStream

    assert isinstance(stream, TextGenerationSingleStream)
    assert stream.content == "hello<tool_call />"
    build_token.assert_called_once_with("call1", "pkg__tool", {"a": 1})

    create_mock = stub.AsyncAnthropic.return_value.messages.create
    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    assert all(msg["role"] != "tool" for msg in kwargs["messages"])


def test_client_forwards_reasoning_effort(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")]
    )
    stub.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=response
    )

    asyncio.run(
        client(
            "model",
            [Message(role=MessageRole.USER, content="hi")],
            settings=GenerationSettings(
                reasoning=ReasoningSettings(effort=ReasoningEffort.XHIGH)
            ),
            use_async_generator=False,
        )
    )

    kwargs = stub.AsyncAnthropic.return_value.messages.create.await_args.kwargs
    assert kwargs["output_config"] == {"effort": "max"}


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

    no_assistant = client._template_messages(
        [
            Message(role=MessageRole.SYSTEM, content="s"),
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.TOOL, tool_call_result=result),
        ],
        ["system"],
    )
    assert no_assistant[-1]["content"][0]["tool_use_id"] == "id1"
    assert no_assistant[-1]["role"] == "user"

    dup = client._template_messages(
        [
            Message(role=MessageRole.USER, content="x"),
            Message(role=MessageRole.USER, content="x"),
        ]
    )
    assert len(dup) == 1


def test_template_messages_tool_error_details(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("k", exit_stack=exit_stack)

    call = ToolCall(id="call1", name="pkg.tool", arguments={"a": 1})
    error = ToolCallError(
        id="err1",
        name="pkg.tool",
        call=call,
        error=Exception("boom"),
        message="boom",
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content="s"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="ok"),
        Message(role=MessageRole.TOOL, tool_call_error=error),
    ]
    templated = client._template_messages(messages, ["system"])
    tool_result = templated[2]["content"][0]
    assert tool_result["tool_use_id"] == "call1"
    assert tool_result["is_error"] is True


def test_file_content_translation_and_beta_header(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContentFile(
                    type="file",
                    file={
                        "citations": True,
                        "context": "ctx",
                        "file_id": "file_1",
                        "title": "Doc",
                    },
                ),
                MessageContentImage(
                    type="image_url", image_url={"file_id": "img_1"}
                ),
            ],
        )
    ]

    stub.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")]
        )
    )

    asyncio.run(client("model", messages, use_async_generator=False))

    create_mock = stub.AsyncAnthropic.return_value.messages.create
    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    assert kwargs["extra_headers"] == {
        "anthropic-beta": "files-api-2025-04-14"
    }
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file_1"},
                    "title": "Doc",
                    "context": "ctx",
                    "citations": {"enabled": True},
                },
                {
                    "type": "image",
                    "source": {"type": "file", "file_id": "img_1"},
                },
            ],
        }
    ]


def test_document_source_variants(anthropic_mod):
    mod, _ = anthropic_mod

    assert mod.AnthropicClient._document_source(
        {"url": "https://example.com/doc.pdf"}
    ) == {"type": "url", "url": "https://example.com/doc.pdf"}
    assert mod.AnthropicClient._document_source(
        {
            "data": "hello",
            "mime_type": "text/plain",
        }
    ) == {
        "type": "text",
        "media_type": "text/plain",
        "data": "hello",
    }
    assert mod.AnthropicClient._document_source(
        {
            "file_data": b64encode(b"hello").decode("ascii"),
            "mime_type": "text/plain",
        }
    ) == {
        "type": "text",
        "media_type": "text/plain",
        "data": "hello",
    }
    assert mod.AnthropicClient._document_source({"file_data": "YWJj"}) == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "YWJj",
    }


def test_document_source_rejects_missing_source_and_invalid_utf8(
    anthropic_mod,
):
    mod, _ = anthropic_mod
    invalid_utf8 = b64encode(b"\xff\xfe").decode("ascii")

    assert mod.AnthropicClient._document_source(
        {
            "file_data": invalid_utf8,
            "mime_type": "text/plain",
        }
    ) == {
        "type": "text",
        "media_type": "text/plain",
        "data": invalid_utf8,
    }

    with pytest.raises(
        AssertionError,
        match="Anthropic file blocks require file_id, file_url, or file_data",
    ):
        mod.AnthropicClient._document_source({"title": "Doc"})


def test_output_config_and_content_block_variants(anthropic_mod):
    mod, _ = anthropic_mod

    assert mod.AnthropicClient._output_config(
        GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.NONE)
        )
    ) == {"effort": "low"}
    assert mod.AnthropicClient._output_config(
        GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH)
        )
    ) == {"effort": "high"}
    assert mod.AnthropicClient._content_block(
        {
            "type": "text",
            "text": "hello",
        }
    ) == {"type": "text", "text": "hello"}
    assert mod.AnthropicClient._content_block(
        {
            "type": "other",
            "value": 1,
        }
    ) == {"type": "other", "value": 1}


def test_image_source_and_files_api_variants(anthropic_mod):
    mod, _ = anthropic_mod

    assert mod.AnthropicClient._image_source(
        {"url": "https://example.com/image.png"}
    ) == {"type": "url", "url": "https://example.com/image.png"}
    assert mod.AnthropicClient._image_source(
        {
            "data": "YWJj",
            "mime_type": "image/jpeg",
        }
    ) == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "YWJj",
    }
    with pytest.raises(
        AssertionError,
        match="Anthropic image blocks require file_id, url, or data",
    ):
        mod.AnthropicClient._image_source({})

    assert mod.AnthropicClient._uses_files_api(
        [
            Message(
                role=MessageRole.USER,
                content=MessageContentImage(
                    type="image_url", image_url={"file_id": "img_1"}
                ),
            )
        ]
    )


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


def test_non_stream_response_content_from_dict(anthropic_mod):
    mod, _ = anthropic_mod
    response = {
        "content": [
            {"type": "text", "text": "alpha"},
            {
                "type": "tool_use",
                "id": "call-id",
                "name": "pkg.tool",
                "input": {"foo": "bar"},
            },
        ]
    }
    token = SimpleNamespace(token="<tool>")
    with patch.object(
        mod.TextGenerationVendor,
        "build_tool_call_token",
        return_value=token,
    ) as build:
        text = mod.AnthropicClient._non_stream_response_content(response)

    assert text == "alpha<tool>"
    build.assert_called_once_with("call-id", "pkg.tool", {"foo": "bar"})


def test_translate_api_error_for_retired_model(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    error = stub.NotFoundError(
        "missing model",
        response=SimpleNamespace(status_code=404),
        body={
            "error": {
                "type": "not_found_error",
                "message": "model: claude-3-5-sonnet-latest",
            }
        },
    )
    client._client.messages.stream = MagicMock(side_effect=error)

    with pytest.raises(ValueError, match="Use 'claude-sonnet-4-6' instead"):
        asyncio.run(client("claude-3-5-sonnet-latest", []))


def test_translate_api_error_for_unknown_model(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    error = stub.NotFoundError(
        "missing model",
        response=SimpleNamespace(status_code=404),
        body={
            "error": {
                "type": "not_found_error",
                "message": "model: claude-missing",
            }
        },
    )
    client._client.messages.create = AsyncMock(side_effect=error)

    with pytest.raises(
        ValueError, match="Verify the model identifier against Anthropic"
    ):
        asyncio.run(
            client(
                "claude-missing",
                [],
                use_async_generator=False,
            )
        )


def test_stream_skips_tool_events_with_missing_indexes(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="x", name="y"),
            index=None,
        )
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"x":1}'),
            index=None,
        )
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen())
        out = []
        async for token in stream:
            out.append(token)
        return out

    assert asyncio.run(collect()) == []


def test_call_reraises_when_translator_does_not_raise(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncMock(spec=AsyncExitStack)
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    err = RuntimeError("boom")
    client._client.messages.stream = MagicMock(side_effect=err)

    with patch.object(
        mod.AnthropicClient, "_translate_api_error"
    ) as translate:
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(client("model", []))

    translate.assert_called_once_with("model", err)


def test_error_helpers_cover_false_paths(anthropic_mod):
    mod, stub = anthropic_mod

    error = ValueError("plain")
    assert mod.AnthropicClient._error_message(error) == "plain"
    assert mod.AnthropicClient._is_missing_model_error(error) is False

    not_found_wrong_status = stub.NotFoundError(
        "bad",
        response=SimpleNamespace(status_code=500),
        body={"error": {"message": "model missing"}},
    )
    assert (
        mod.AnthropicClient._is_missing_model_error(not_found_wrong_status)
        is False
    )

    mod.AnthropicClient._translate_api_error("model", RuntimeError("x"))


def test_template_messages_skips_dynamic_none_results(anthropic_mod):
    mod, _ = anthropic_mod
    client = mod.AnthropicClient("k", exit_stack=AsyncExitStack())

    class DynamicToolMessage:
        role = MessageRole.TOOL
        tool_call_error = None

        def __init__(self) -> None:
            self._calls = 0

        @property
        def tool_call_result(self):
            self._calls += 1
            if self._calls == 1:
                return object()
            return None

    with patch.object(
        mod.TextGenerationVendor,
        "_template_messages",
        return_value=[{"role": "assistant", "content": "ok"}],
    ):
        output = client._template_messages([DynamicToolMessage()])

    assert output == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }
    ]


def test_non_stream_response_content_ignores_non_list_content(anthropic_mod):
    mod, _ = anthropic_mod
    response = {"content": "not-a-list"}

    assert mod.AnthropicClient._non_stream_response_content(response) == ""
