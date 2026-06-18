import asyncio
import importlib
import sys
import types
from contextlib import AsyncExitStack
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamValidationError,
)
from avalan.model.vendor import (
    TextGenerationVendor,
    TextGenerationVendorStream,
)


class AsyncIter:
    def __init__(self, items: list[object]) -> None:
        self._iter = iter(items)

    def __aiter__(self) -> "AsyncIter":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        return False


class DummyVendor(TextGenerationVendor):
    async def __call__(self, *args: object, **kwargs: object) -> object:
        return await super().__call__(*args, **kwargs)


def _mixed_messages(
    *files: MessageContentFile | MessageContentImage,
) -> list[Message]:
    return [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContentText(type="text", text="Summarize"),
                *files,
            ],
        )
    ]


def test_base_vendor_keeps_file_content_generic() -> None:
    templated = DummyVendor()._template_messages(
        _mixed_messages(
            MessageContentFile(
                type="file",
                file={
                    "file_id": "provider-file-1",
                    "mime_type": "application/pdf",
                },
            ),
            MessageContentFile(
                type="file",
                file={
                    "file_url": "https://files.example/report.pdf",
                    "mime_type": "application/pdf",
                },
            ),
            MessageContentFile(
                type="file",
                file={
                    "file_data": "YWJj",
                    "mime_type": "application/pdf",
                },
            ),
        )
    )

    assert templated == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "provider-file-1",
                        "mime_type": "application/pdf",
                    },
                },
                {
                    "type": "file",
                    "file": {
                        "file_url": "https://files.example/report.pdf",
                        "mime_type": "application/pdf",
                    },
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": "YWJj",
                        "mime_type": "application/pdf",
                    },
                },
            ],
        }
    ]
    rendered = repr(templated)
    for provider_key in (
        "input_file",
        "document",
        "file_data",
        "inline_data",
        "s3Location",
    ):
        if provider_key == "file_data":
            continue
        assert provider_key not in rendered


def test_base_vendor_boundary_branches() -> None:
    vendor = DummyVendor()

    with pytest.raises(NotImplementedError):
        asyncio.run(vendor("model", []))

    assert (
        vendor._system_prompt(
            [
                Message(
                    role=MessageRole.SYSTEM,
                    content=MessageContentText(type="text", text="sys"),
                )
            ]
        )
        == "sys"
    )
    assert (
        vendor._system_prompt(
            [
                Message(
                    role=MessageRole.SYSTEM,
                    content=MessageContentFile(
                        type="file", file={"file_id": "file-system"}
                    ),
                )
            ]
        )
        is None
    )
    assert vendor._template_messages(
        [
            Message(role=MessageRole.SYSTEM, content="skip"),
            Message(role=MessageRole.USER, content="keep"),
        ],
        ["system"],
    ) == [{"role": "user", "content": "keep"}]

    invalid_arguments = TextGenerationVendor.build_tool_call_token(
        object(), "tool", "{"
    )
    assert invalid_arguments.call.id.startswith("<object object at ")
    assert invalid_arguments.call.name == "tool"
    assert invalid_arguments.call.arguments == {}
    assert invalid_arguments.call.provider_arguments_malformed is True

    dict_arguments = TextGenerationVendor.build_tool_call_token(
        None, "pkg__tool", {"ok": True}
    )
    assert dict_arguments.call.id is None
    assert dict_arguments.call.name == "pkg__tool"
    assert dict_arguments.call.arguments == {"ok": True}
    assert '"id"' not in dict_arguments.token

    async def agen() -> Any:
        yield "token"

    async def first_public_item_kind() -> None:
        stream = TextGenerationVendorStream(cast(Any, agen()))
        iterator = stream()
        assert iterator is not stream
        await anext(iterator)

    with pytest.raises(StreamValidationError):
        asyncio.run(first_public_item_kind())


def test_openai_file_envelopes_cover_streaming_and_non_streaming() -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = SimpleNamespace(create=AsyncMock())

    openai_stub = types.ModuleType("openai")
    openai_stub.__spec__ = ModuleSpec("openai", loader=None)
    openai_stub.AsyncOpenAI = FakeAsyncOpenAI
    openai_stub.Omit = type("Omit", (), {})

    with patch.dict(sys.modules, {"openai": openai_stub}):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.openai")
        importlib.reload(mod)
        client = mod.OpenAIClient(api_key="key", base_url="url")

    messages = _mixed_messages(
        MessageContentFile(
            type="file",
            file={"file_id": "file-openai", "mime_type": "application/pdf"},
        ),
        MessageContentFile(
            type="file",
            file={
                "file_url": "https://files.example/report.pdf",
                "mime_type": "application/pdf",
            },
        ),
        MessageContentFile(
            type="file",
            file={"file_data": "YWJj", "mime_type": "application/pdf"},
        ),
        MessageContentFile(
            type="file",
            file={
                "file_url": "s3://bucket/private/report.pdf",
                "mime_type": "application/pdf",
            },
        ),
        MessageContentImage(
            type="image_url",
            image_url={
                "data": "aW1hZ2U=",
                "mime_type": "image/png",
            },
        ),
    )
    expected_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize"},
                {"type": "input_file", "file_id": "file-openai"},
                {
                    "type": "input_file",
                    "file_url": "https://files.example/report.pdf",
                },
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,YWJj",
                },
                {
                    "type": "input_file",
                    "file_url": "s3://bucket/private/report.pdf",
                },
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                },
            ],
        }
    ]

    client._client.responses.create.return_value = SimpleNamespace(output=[])
    asyncio.run(client("gpt", messages, use_async_generator=False))
    kwargs = client._client.responses.create.await_args.kwargs
    assert kwargs["stream"] is False
    assert kwargs["input"] == expected_input

    client._client.responses.create.reset_mock()
    client._client.responses.create.return_value = AsyncIter([])
    asyncio.run(client("gpt", messages, use_async_generator=True))
    kwargs = client._client.responses.create.await_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["input"] == expected_input


def test_openai_stream_skips_custom_tool_call_without_string_id() -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = SimpleNamespace(create=AsyncMock())

    openai_stub = types.ModuleType("openai")
    openai_stub.__spec__ = ModuleSpec("openai", loader=None)
    openai_stub.AsyncOpenAI = FakeAsyncOpenAI
    openai_stub.Omit = type("Omit", (), {})

    with patch.dict(sys.modules, {"openai": openai_stub}):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.openai")
        importlib.reload(mod)

    stream = mod.OpenAIStream(
        AsyncIter(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        id=object(),
                        custom_tool_call=SimpleNamespace(
                            id=object(), name="pkg__tool"
                        ),
                    ),
                )
            ]
        )
    )

    async def collect() -> list[CanonicalStreamItem]:
        return [item async for item in stream]

    items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert (
        items[1].data["message"]
        == "response tool call id must be a non-empty string"
    )


def test_anthropic_file_envelopes_cover_streaming_and_non_streaming() -> None:
    class APIStatusError(Exception):
        status_code = 500

    class FakeMessages:
        def __init__(self) -> None:
            self.create = AsyncMock()
            self.stream = MagicMock()

    anthropic_stub = types.ModuleType("anthropic")
    anthropic_stub.__spec__ = ModuleSpec("anthropic", loader=None)
    anthropic_stub.APIStatusError = APIStatusError
    anthropic_stub.AsyncAnthropic = MagicMock()
    anthropic_stub.AsyncAnthropic.return_value.messages = FakeMessages()
    types_mod = types.ModuleType("anthropic.types")
    types_mod.__spec__ = ModuleSpec("anthropic.types", loader=None)
    types_mod.RawContentBlockDeltaEvent = type(
        "RawContentBlockDeltaEvent", (), {}
    )
    types_mod.RawMessageStopEvent = type("RawMessageStopEvent", (), {})

    with patch.dict(
        sys.modules,
        {"anthropic": anthropic_stub, "anthropic.types": types_mod},
    ):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.anthropic")
        importlib.reload(mod)
        exit_stack = AsyncExitStack()
        client = mod.AnthropicClient("key", exit_stack=exit_stack)

    messages = _mixed_messages(
        MessageContentFile(
            type="file",
            file={"file_id": "file-anthropic", "mime_type": "application/pdf"},
        ),
        MessageContentFile(
            type="file",
            file={
                "file_url": "https://files.example/report.pdf",
                "mime_type": "application/pdf",
            },
        ),
        MessageContentFile(
            type="file",
            file={"file_data": "YWJj", "mime_type": "application/pdf"},
        ),
        MessageContentFile(
            type="file",
            file={"file_data": "aGVsbG8=", "mime_type": "text/plain"},
        ),
    )
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize"},
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file-anthropic"},
                },
                {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": "https://files.example/report.pdf",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "YWJj",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "hello",
                    },
                },
            ],
        }
    ]

    client._client.messages.create.return_value = SimpleNamespace(content=[])
    asyncio.run(client("claude", messages, use_async_generator=False))
    kwargs = client._client.messages.create.await_args.kwargs
    assert kwargs["messages"] == expected_messages
    assert kwargs["extra_headers"] == {
        "anthropic-beta": "files-api-2025-04-14"
    }

    client._client.messages.stream.return_value = AsyncContext(AsyncIter([]))
    asyncio.run(client("claude", messages, use_async_generator=True))
    stream_kwargs = client._client.messages.stream.call_args.kwargs
    assert stream_kwargs["messages"] == expected_messages
    assert stream_kwargs["extra_headers"] == {
        "anthropic-beta": "files-api-2025-04-14"
    }
    asyncio.run(exit_stack.aclose())


def _bedrock_module() -> Any:
    aioboto3_stub = types.ModuleType("aioboto3")
    aioboto3_stub.__spec__ = ModuleSpec("aioboto3", loader=None)
    aioboto3_stub.Session = MagicMock()
    with patch.dict(sys.modules, {"aioboto3": aioboto3_stub}):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.bedrock")
        importlib.reload(mod)
    return mod


def test_bedrock_helper_and_error_hint_branches() -> None:
    class FakeBedrockError(Exception):
        def __init__(self, code: str, message: str) -> None:
            super().__init__(message)
            self.response = {"Error": {"Code": code, "Message": message}}

    mod = _bedrock_module()

    assert mod._bedrock_error_code(Exception("plain")) is None
    assert (
        mod._bedrock_error_code(
            SimpleNamespace(response={"Error": "not-a-dict"})
        )
        is None
    )
    assert mod._bedrock_error_message(Exception("plain")) == "plain"
    assert mod._geo_inference_prefix(None) is None
    assert mod._geo_inference_prefix("eu-west-1") == "eu."
    assert mod._geo_inference_prefix("ap-southeast-2") is None

    client_without_region = mod.BedrockClient(exit_stack=AsyncExitStack())
    with pytest.raises(
        ValueError, match="geo-prefixed inference profile"
    ) as invalid_model:
        client_without_region._raise_invalid_model_identifier(
            "anthropic.claude-test",
            FakeBedrockError(
                "ValidationException",
                "The model identifier is invalid.",
            ),
        )
    assert "Try 'us.'" not in str(invalid_model.value)

    regional_client = mod.BedrockClient(
        exit_stack=AsyncExitStack(), region_name="us-west-2"
    )
    with pytest.raises(ValueError, match="active 'us.'-prefixed profile"):
        regional_client._raise_end_of_life_model_error(
            "anthropic.claude-test",
            FakeBedrockError(
                "ResourceNotFoundException",
                "This model version has reached the end of its life.",
            ),
        )


def test_google_file_envelopes_cover_streaming_and_non_streaming() -> None:
    class FakeGoogleClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(
                        return_value=SimpleNamespace(text="ok")
                    ),
                    generate_content_stream=AsyncMock(
                        return_value=AsyncIter([])
                    ),
                )
            )

    genai_stub = types.ModuleType("google.genai")
    genai_stub.__spec__ = ModuleSpec("google.genai", loader=None)
    genai_stub.Client = FakeGoogleClient
    types_mod = types.ModuleType("google.genai.types")
    types_mod.__spec__ = ModuleSpec("google.genai.types", loader=None)
    types_mod.GenerateContentResponse = SimpleNamespace
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []
    google_pkg.genai = genai_stub

    with patch.dict(
        sys.modules,
        {
            "google": google_pkg,
            "google.genai": genai_stub,
            "google.genai.types": types_mod,
        },
    ):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.google")
        importlib.reload(mod)
        client = mod.GoogleClient("key")

    messages = _mixed_messages(
        MessageContentFile(
            type="file",
            file={"file_id": "google-file-1", "mime_type": "application/pdf"},
        ),
        MessageContentFile(
            type="file",
            file={
                "file_url": "https://files.example/report.pdf",
                "mime_type": "application/pdf",
            },
        ),
        MessageContentFile(
            type="file",
            file={
                "file_url": "gs://bucket/report.pdf",
                "mime_type": "application/pdf",
            },
        ),
        MessageContentFile(
            type="file",
            file={"file_data": "YWJj", "mime_type": "application/pdf"},
        ),
    )
    expected_contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Summarize"},
                {
                    "file_data": {
                        "file_uri": "google-file-1",
                        "mime_type": "application/pdf",
                    }
                },
                {
                    "file_data": {
                        "file_uri": "https://files.example/report.pdf",
                        "mime_type": "application/pdf",
                    }
                },
                {
                    "file_data": {
                        "file_uri": "gs://bucket/report.pdf",
                        "mime_type": "application/pdf",
                    }
                },
                {
                    "inline_data": {
                        "data": "YWJj",
                        "mime_type": "application/pdf",
                    }
                },
            ],
        }
    ]

    asyncio.run(client("gemini", messages, use_async_generator=False))
    kwargs = client._client.aio.models.generate_content.await_args.kwargs
    assert kwargs["contents"] == expected_contents

    asyncio.run(client("gemini", messages, use_async_generator=True))
    stream_kwargs = (
        client._client.aio.models.generate_content_stream.await_args.kwargs
    )
    assert stream_kwargs["contents"] == expected_contents


def test_bedrock_file_envelopes_cover_streaming_and_non_streaming() -> None:
    fake_client = SimpleNamespace(
        converse=AsyncMock(), converse_stream=AsyncMock()
    )
    session = MagicMock()
    session.client.return_value = AsyncContext(fake_client)
    aioboto3_stub = types.ModuleType("aioboto3")
    aioboto3_stub.__spec__ = ModuleSpec("aioboto3", loader=None)
    aioboto3_stub.Session = MagicMock(return_value=session)

    with patch.dict(sys.modules, {"aioboto3": aioboto3_stub}):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.bedrock")
        importlib.reload(mod)
        exit_stack = AsyncExitStack()
        client = mod.BedrockClient(exit_stack=exit_stack)

    messages = _mixed_messages(
        MessageContentFile(
            type="file",
            file={
                "file_url": "s3://bucket/report.pdf",
                "bucket_owner": "123456789012",
                "mime_type": "application/pdf",
            },
        ),
        MessageContentFile(
            type="file",
            file={"file_data": "YWJj", "mime_type": "application/pdf"},
        ),
        MessageContentFile(
            type="file",
            file={"file_data": "hello", "mime_type": "text/plain"},
        ),
    )
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"text": "Summarize"},
                {
                    "document": {
                        "name": "Document",
                        "source": {
                            "s3Location": {
                                "uri": "s3://bucket/report.pdf",
                                "bucketOwner": "123456789012",
                            }
                        },
                        "format": "pdf",
                    }
                },
                {
                    "document": {
                        "name": "Document",
                        "source": {"bytes": b"abc"},
                        "format": "pdf",
                    }
                },
                {
                    "document": {
                        "name": "Document",
                        "source": {"text": "hello"},
                        "format": "txt",
                    }
                },
            ],
        }
    ]

    fake_client.converse.return_value = {
        "output": {"message": {"content": []}}
    }
    asyncio.run(client("bedrock-model", messages, use_async_generator=False))
    kwargs = fake_client.converse.await_args.kwargs
    assert kwargs["messages"] == expected_messages

    fake_client.converse_stream.return_value = {"stream": AsyncIter([])}
    asyncio.run(client("bedrock-model", messages, use_async_generator=True))
    stream_kwargs = fake_client.converse_stream.await_args.kwargs
    assert stream_kwargs["messages"] == expected_messages
    asyncio.run(exit_stack.aclose())


@pytest.mark.parametrize(
    "file_payload,error_match",
    [
        (
            {
                "file_url": "https://files.example/report.pdf",
                "mime_type": "application/pdf",
            },
            "Bedrock document URLs must use s3:// URIs",
        ),
        (
            {"file_id": "bedrock-file-1", "mime_type": "application/pdf"},
            "Bedrock documents require inline data or a file URL",
        ),
    ],
)
def test_bedrock_rejects_unsupported_file_references_safely(
    file_payload: dict[str, Any], error_match: str
) -> None:
    aioboto3_stub = types.ModuleType("aioboto3")
    aioboto3_stub.__spec__ = ModuleSpec("aioboto3", loader=None)
    aioboto3_stub.Session = MagicMock()

    with patch.dict(sys.modules, {"aioboto3": aioboto3_stub}):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.bedrock")
        importlib.reload(mod)
        client = mod.BedrockClient(exit_stack=AsyncExitStack())

    with pytest.raises(AssertionError, match=error_match) as exc_info:
        client._template_messages(
            _mixed_messages(MessageContentFile(type="file", file=file_payload))
        )

    error_text = str(exc_info.value)
    assert "https://files.example/report.pdf" not in error_text
    assert "bedrock-file-1" not in error_text
