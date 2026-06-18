import importlib
import sys
import types
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ReasoningEffort,
    ReasoningSettings,
    ToolCall,
    ToolCallResult,
    TransformerEngineSettings,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    accumulate_canonical_stream_items,
)
from avalan.task.usage import (
    usage_observation_from_response,
    usage_totals_from_response,
)


class AsyncIter:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def _canonical_items(
    stream: AsyncIterable[CanonicalStreamItem],
) -> list[CanonicalStreamItem]:
    return [item async for item in stream]


def _answer_text(items: list[CanonicalStreamItem]) -> str:
    return accumulate_canonical_stream_items(items).answer_text


class AnthropicTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
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
        self.patch = patch.dict(
            sys.modules,
            {"anthropic": stub, "anthropic.types": types_mod},
        )
        self.patch.start()
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.anthropic"
        )
        importlib.reload(self.mod)
        self.stub = stub

    def tearDown(self):
        self.patch.stop()

    async def test_stream_and_client_and_model(self):
        Delta = self.stub.types.RawContentBlockDeltaEvent
        Stop = self.stub.types.RawMessageStopEvent

        async def agen():
            yield SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(
                    type="tool_use", id="tid", name="tool"
                ),
                index=0,
            )
            yield Delta(types.SimpleNamespace(partial_json="val"))
            yield SimpleNamespace(
                type="content_block_stop",
                content_block=SimpleNamespace(
                    type="tool_use", id="tid", name="tool"
                ),
                index=0,
            )
            yield Stop()

        stream = self.mod.AnthropicStream(agen())
        items = await _canonical_items(stream)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.tool_call_arguments, {"tid": "val"})
        self.assertIs(
            accumulator.terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

        ctx_instance = SimpleNamespace(
            __aenter__=AsyncMock(return_value=AsyncIter([])),
            __aexit__=AsyncMock(return_value=False),
        )
        self.stub.AsyncAnthropic.return_value.messages.stream = MagicMock(
            return_value=ctx_instance
        )

        exit_stack = AsyncMock(spec=AsyncExitStack)

        with patch.object(self.mod, "AnthropicStream") as StreamMock:
            client = self.mod.AnthropicClient(
                "tok", "url", exit_stack=exit_stack
            )
            client._system_prompt = MagicMock(return_value="sys")
            client._template_messages = MagicMock(
                return_value=[{"content": "c"}]
            )
            result = await client("m", [])

        self.stub.AsyncAnthropic.assert_called_once_with(
            api_key="tok", base_url="url"
        )
        client._client.messages.stream.assert_called_once()
        exit_stack.enter_async_context.assert_awaited_once_with(ctx_instance)
        StreamMock.assert_called_once_with(
            events=exit_stack.enter_async_context.return_value
        )
        self.assertIs(result, StreamMock.return_value)

        with patch.object(self.mod, "AnthropicClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="b",
            )
            model = self.mod.AnthropicModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(
            api_key="t", base_url="b", exit_stack=model._exit_stack
        )
        self.assertIs(loaded, ClientMock.return_value)

    async def test_stream_variants(self):
        Delta = self.stub.types.RawContentBlockDeltaEvent

        async def agen():
            yield SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(
                    type="tool_use", id="tid", name="tname"
                ),
                index=0,
            )
            yield Delta(SimpleNamespace(thinking="think"))
            yield Delta(SimpleNamespace(partial_json='{"a":1}'))
            yield Delta(SimpleNamespace(text="txt"))
            yield SimpleNamespace(
                type="content_block_stop",
                content_block=SimpleNamespace(
                    type="tool_use",
                    id="tid",
                    name="tname",
                    input={"x": 1},
                ),
                index=0,
            )
            yield SimpleNamespace(type="message_stop")

        stream = self.mod.AnthropicStream(agen())
        items = await _canonical_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "think")
        self.assertEqual(items[2].text_delta, '{"a":1}')
        self.assertEqual(items[3].text_delta, "txt")
        self.assertEqual(items[4].data, {"name": "tname"})
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.reasoning_text, "think")
        self.assertEqual(accumulator.tool_call_arguments, {"tid": '{"a":1}'})
        self.assertEqual(accumulator.answer_text, "txt")

    def test_template_messages_and_tool_schemas(self):
        exit_stack = AsyncExitStack()
        client = self.mod.AnthropicClient("k", exit_stack=exit_stack)

        @dataclass
        class Res:
            x: int

        call = ToolCall(id="id1", name="pkg.tool", arguments={"a": 1})
        result = ToolCallResult(
            id="id1", name="pkg.tool", call=call, result=Res(x=2)
        )
        messages = [
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.ASSISTANT, content="ok"),
            Message(role=MessageRole.TOOL, tool_call_result=result),
        ]
        templated = client._template_messages(messages)
        self.assertEqual(templated[1]["content"][1]["name"], "avl_cGtnLnRvb2w")
        self.assertEqual(templated[2]["content"][0]["tool_use_id"], "id1")

        dup = client._template_messages(
            [
                Message(role=MessageRole.USER, content="x"),
                Message(role=MessageRole.USER, content="x"),
            ]
        )
        self.assertEqual(len(dup), 1)

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
            }
        ]
        schema_out = self.mod.AnthropicClient._tool_schemas(DummyTool(schemas))
        self.assertEqual(
            schema_out,
            [
                {
                    "name": "avl_cGtnLnRvb2w",
                    "description": "d",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ],
        )
        self.assertIsNone(
            self.mod.AnthropicClient._tool_schemas(DummyTool(None))
        )


class GoogleTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        class DummyClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.aio = SimpleNamespace(
                    models=SimpleNamespace(
                        generate_content_stream=AsyncMock(
                            return_value=AsyncIter([SimpleNamespace(text="s")])
                        ),
                        generate_content=AsyncMock(
                            return_value=SimpleNamespace(
                                text="r",
                                usage_metadata=SimpleNamespace(
                                    promptTokenCount=6,
                                    cachedContentTokenCount=2,
                                    candidatesTokenCount=4,
                                    thoughtsTokenCount=1,
                                    totalTokenCount=10,
                                ),
                            )
                        ),
                    )
                )

        stub = types.ModuleType("google.genai")
        stub.Client = DummyClient
        types_mod = types.ModuleType("google.genai.types")
        types_mod.GenerateContentResponse = SimpleNamespace
        stub.types = types_mod
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []
        google_pkg.genai = stub
        self.p1 = patch.dict(
            sys.modules,
            {
                "google.genai": stub,
                "google.genai.types": types_mod,
                "google": google_pkg,
            },
        )
        self.p1.start()
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.google"
        )
        importlib.reload(self.mod)
        self.stub = stub

    def tearDown(self):
        self.p1.stop()

    async def test_call_and_model(self):
        client = self.mod.GoogleClient("k")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        with patch.object(self.mod, "GoogleStream") as StreamMock:
            result = await client("m", msgs, use_async_generator=True)
        client._client.aio.models.generate_content_stream.assert_awaited_once()
        StreamMock.assert_called_once()
        self.assertIs(result, StreamMock.return_value)

        gen = await client("m", msgs, use_async_generator=False)
        items = [item async for item in gen]
        totals = usage_totals_from_response(gen)
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "r",
        )
        self.assertEqual(gen.provider_family, "google")
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 6)
        self.assertEqual(totals.cached_input_tokens, 2)
        self.assertEqual(totals.reasoning_tokens, 1)
        client._client.aio.models.generate_content.assert_awaited_once()

        stream = self.mod.GoogleStream(AsyncIter([SimpleNamespace(text="x")]))
        stream_items = [item async for item in stream]
        self.assertEqual(
            accumulate_canonical_stream_items(stream_items).answer_text,
            "x",
        )

        generator_stream = self.mod.GoogleStream(
            AsyncIter([SimpleNamespace(text="g")])
        )
        generator_items = await _canonical_items(
            cast(
                AsyncIterable[CanonicalStreamItem],
                generator_stream._generator,
            )
        )
        self.assertEqual(
            [item.kind for item in generator_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(_answer_text(generator_items), "g")
        self.assertEqual(
            {item.provider_family for item in generator_items}, {"google"}
        )

        with patch.object(self.mod, "GoogleClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="tok",
            )
            model = self.mod.GoogleModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(api_key="tok")
        self.assertIs(loaded, ClientMock.return_value)

    async def test_client_stream_direct_anext_yields_canonical_items(
        self,
    ) -> None:
        client = self.mod.GoogleClient("k")
        msgs = [Message(role=MessageRole.USER, content="hi")]

        stream = await client("m", msgs, use_async_generator=True)
        first = await stream.__anext__()
        second = await stream.__anext__()
        tail: list[CanonicalStreamItem] = []
        while True:
            try:
                tail.append(await stream.__anext__())
            except StopAsyncIteration:
                break
        items = [first, second, *tail]

        client._client.aio.models.generate_content_stream.assert_awaited_once()
        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(5)))
        self.assertEqual(_answer_text(items), "s")
        self.assertEqual({item.provider_family for item in items}, {"google"})

    async def test_stream_preserves_model_dump_provider_payload(self) -> None:
        modes: list[str] = []
        payload = {"text": "x", "native": True}

        class ModelDumpChunk:
            text = "x"

            def model_dump(self, *, mode: str) -> dict[str, object]:
                modes.append(mode)
                return payload

        stream = self.mod.GoogleStream(AsyncIter([ModelDumpChunk()]))

        items = [item async for item in stream]

        self.assertEqual(modes, ["json"])
        self.assertEqual(items[1].provider_payload, payload)
        self.assertEqual(_answer_text(items), "x")

    async def test_stream_records_usage_metadata_after_full_consumption(self):
        modes: list[str] = []
        usage = SimpleNamespace(
            promptTokenCount=4,
            cachedContentTokenCount=1,
            candidatesTokenCount=3,
            thoughtsTokenCount=2,
            totalTokenCount=9,
        )
        payload = {"usageMetadata": {"promptTokenCount": 4}}

        class UsageChunk:
            text = None
            usage_metadata = usage

            def model_dump(self, *, mode: str) -> dict[str, object]:
                modes.append(mode)
                return payload

        stream = self.mod.GoogleStream(
            AsyncIter(
                [
                    SimpleNamespace(text="x"),
                    UsageChunk(),
                ]
            )
        )

        items = [item async for item in stream]
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "x",
        )
        usage_item = next(
            item
            for item in items
            if item.kind is StreamItemKind.USAGE_COMPLETED
        )
        observation = usage_observation_from_response(stream)
        totals = usage_totals_from_response(stream)

        self.assertEqual(modes, ["json"])
        self.assertEqual(usage_item.provider_payload, payload)
        self.assertEqual(stream.provider_family, "google")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.metadata, {"provider_family": "google"})
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 4)
        self.assertEqual(totals.cached_input_tokens, 1)
        self.assertEqual(totals.output_tokens, 3)
        self.assertEqual(totals.reasoning_tokens, 2)
        self.assertEqual(totals.total_tokens, 9)

    async def test_stream_defers_usage_metadata_until_exhaustion(self):
        stream = self.mod.GoogleStream(
            AsyncIter(
                [
                    {
                        "text": None,
                        "usageMetadata": {
                            "promptTokenCount": 0,
                            "candidatesTokenCount": 0,
                            "thoughtsTokenCount": 0,
                            "totalTokenCount": 0,
                        },
                    },
                    SimpleNamespace(text="late"),
                ]
            )
        )

        iterator = stream.__aiter__()
        seen = []
        while True:
            item = await iterator.__anext__()
            seen.append(item)
            if item.kind is StreamItemKind.ANSWER_DELTA:
                break

        self.assertEqual(item.text_delta, "late")
        self.assertIsNone(stream.usage)
        seen.extend([item async for item in iterator])
        self.assertEqual(
            accumulate_canonical_stream_items(seen).answer_text,
            "late",
        )
        totals = usage_totals_from_response(stream)

        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 0)
        self.assertEqual(totals.output_tokens, 0)
        self.assertEqual(totals.reasoning_tokens, 0)
        self.assertEqual(totals.total_tokens, 0)

    async def test_stream_failure_after_usage_metadata_keeps_unavailable(
        self,
    ):
        class FailingIter:
            def __init__(self):
                self._count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._count += 1
                if self._count == 1:
                    return {
                        "text": None,
                        "usageMetadata": {"promptTokenCount": 1},
                    }
                raise RuntimeError("provider failure")

        stream = self.mod.GoogleStream(FailingIter())

        items = await _canonical_items(stream)
        self.assertEqual(items[-2].kind, StreamItemKind.STREAM_ERRORED)
        self.assertIs(
            items[-2].terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        error_data = items[-2].data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "RuntimeError")
        self.assertEqual(error_data["message"], "provider failure")
        self.assertIsNone(stream.usage)
        self.assertIsNone(usage_totals_from_response(stream))

    async def test_stream_supports_camel_usage_metadata_and_none(self):
        stream = self.mod.GoogleStream(
            AsyncIter(
                [
                    {
                        "text": None,
                        "usageMetadata": {
                            "promptTokenCount": 0,
                            "candidatesTokenCount": 0,
                            "totalTokenCount": 0,
                        },
                    }
                ]
            )
        )

        items = [item async for item in stream]
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "",
        )
        totals = usage_totals_from_response(stream)

        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 0)
        self.assertEqual(totals.output_tokens, 0)
        self.assertEqual(totals.total_tokens, 0)

        no_usage = self.mod.GoogleStream(
            AsyncIter([SimpleNamespace(text=None)])
        )
        no_usage_items = [item async for item in no_usage]
        self.assertEqual(
            accumulate_canonical_stream_items(no_usage_items).answer_text,
            "",
        )
        self.assertIsNone(no_usage.usage)
        self.assertIsNone(usage_totals_from_response(no_usage))

    async def test_provider_instructions_are_rejected_before_api_call(self):
        client = self.mod.GoogleClient("k")
        stream_mock = client._client.aio.models.generate_content_stream
        generate_mock = client._client.aio.models.generate_content

        with self.assertRaisesRegex(AssertionError, "provider instructions"):
            await client(
                "m",
                [Message(role=MessageRole.USER, content="hi")],
                instructions="private policy",
            )

        stream_mock.assert_not_awaited()
        generate_mock.assert_not_awaited()

    async def test_call_supports_file_and_image_parts(self):
        client = self.mod.GoogleClient("k")
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(type="text", text="Summarize this"),
                    MessageContentFile(
                        type="file",
                        file={
                            "file_url": "https://example.com/files/1",
                            "filename": "report.pdf",
                            "mime_type": "application/pdf",
                        },
                    ),
                    MessageContentFile(
                        type="file",
                        file={
                            "file_data": "YWJj",
                            "filename": "inline.pdf",
                            "mime_type": "application/pdf",
                        },
                    ),
                    MessageContentImage(
                        type="image_url",
                        image_url={
                            "data": "aW1hZ2U=",
                            "mime_type": "image/jpeg",
                        },
                    ),
                ],
            ),
            Message(role=MessageRole.ASSISTANT, content="Earlier reply"),
        ]
        settings = GenerationSettings(
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.8,
            top_k=4,
            stop_strings="STOP",
        )

        await client("gemini-2.5-flash", messages, settings)

        stream_mock = client._client.aio.models.generate_content_stream
        kwargs = stream_mock.await_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(
            kwargs["config"],
            {
                "system_instruction": "sys",
                "max_output_tokens": 64,
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 4,
                "stop_sequences": ["STOP"],
            },
        )
        self.assertEqual(
            kwargs["contents"],
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Summarize this"},
                        {
                            "file_data": {
                                "file_uri": "https://example.com/files/1",
                                "mime_type": "application/pdf",
                                "display_name": "report.pdf",
                            }
                        },
                        {
                            "inline_data": {
                                "data": "YWJj",
                                "mime_type": "application/pdf",
                                "display_name": "inline.pdf",
                            }
                        },
                        {
                            "inline_data": {
                                "data": "aW1hZ2U=",
                                "mime_type": "image/jpeg",
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [{"text": "Earlier reply"}],
                },
            ],
        )

    async def test_call_forwards_reasoning_effort_for_gemini_3(self):
        client = self.mod.GoogleClient("k")
        settings = GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.XHIGH)
        )

        await client(
            "gemini-3-flash-preview",
            [Message(role=MessageRole.USER, content="hi")],
            settings,
        )

        kwargs = (
            client._client.aio.models.generate_content_stream.await_args.kwargs
        )
        self.assertEqual(
            kwargs["config"]["thinking_config"],
            {"thinking_level": "high"},
        )

    def test_helper_methods_cover_reasoning_and_part_fallbacks(self):
        self.assertEqual(
            self.mod.GoogleClient._thinking_config(
                "gemini-3-flash",
                GenerationSettings(
                    reasoning=ReasoningSettings(effort=ReasoningEffort.NONE)
                ),
            ),
            {"thinking_level": "minimal"},
        )
        self.assertEqual(
            self.mod.GoogleClient._thinking_config(
                "gemini-3-flash",
                GenerationSettings(
                    reasoning=ReasoningSettings(effort=ReasoningEffort.MEDIUM)
                ),
            ),
            {"thinking_level": "medium"},
        )
        self.assertEqual(
            self.mod.GoogleClient._message_role(str(MessageRole.DEVELOPER)),
            str(MessageRole.USER),
        )
        self.assertEqual(
            self.mod.GoogleClient._parts({"type": "text", "text": "one"}),
            [{"text": "one"}],
        )
        self.assertEqual(self.mod.GoogleClient._parts(123), [{"text": "123"}])
        self.assertEqual(
            self.mod.GoogleClient._part({"type": "unknown", "value": 1}),
            {"text": "{'type': 'unknown', 'value': 1}"},
        )


class HuggingfaceTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        client = MagicMock()
        client.chat_completion = AsyncMock()
        stub = types.ModuleType("huggingface_hub")
        stub.AsyncInferenceClient = MagicMock(return_value=client)
        patcher = patch.dict(sys.modules, {"huggingface_hub": stub})
        patcher.start()
        self.patcher = patcher
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.huggingface"
        )
        importlib.reload(self.mod)
        self.client = client

    def tearDown(self):
        self.patcher.stop()

    async def test_call_and_model(self):
        hf_client = self.mod.HuggingfaceClient("k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        settings = GenerationSettings()
        stream_obj = AsyncIter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content="t"))
                    ]
                )
            ]
        )
        self.client.chat_completion = AsyncMock(return_value=stream_obj)
        with patch.object(self.mod, "HuggingfaceStream") as StreamMock:
            result = await hf_client(
                "m", msgs, settings, use_async_generator=True
            )
        StreamMock.assert_called_once_with(stream_obj)
        self.assertIs(result, StreamMock.return_value)

        usage = {
            "prompt_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 10,
        }
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="r"))],
            usage=usage,
        )
        self.client.chat_completion = AsyncMock(return_value=resp)
        gen = await hf_client("m", msgs, settings, use_async_generator=False)
        items = await _canonical_items(gen)
        observation = usage_observation_from_response(gen)
        totals = usage_totals_from_response(gen)
        self.assertEqual(_answer_text(items), "r")
        self.assertEqual(gen.provider_family, "hugging_face")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata, {"provider_family": "hugging_face"}
        )
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 6)
        self.assertEqual(totals.cached_input_tokens, 2)
        self.assertEqual(totals.output_tokens, 4)
        self.assertEqual(totals.reasoning_tokens, 1)
        self.assertEqual(totals.total_tokens, 10)

        stream = self.mod.HuggingfaceStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="x"))
                        ]
                    )
                ]
            )
        )
        stream_items = await _canonical_items(stream)
        self.assertEqual(_answer_text(stream_items), "x")

        direct_stream = self.mod.HuggingfaceStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="hi")
                            )
                        ]
                    )
                ]
            )
        )
        started = await direct_stream.__anext__()
        delta = await direct_stream.__anext__()
        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "hi")
        self.assertEqual(delta.provider_family, "hugging_face")

        generator_stream = self.mod.HuggingfaceStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="g"))
                        ]
                    )
                ]
            )
        )
        generator_items = await _canonical_items(
            cast(
                AsyncIterable[CanonicalStreamItem],
                generator_stream._generator,
            )
        )
        self.assertEqual(
            [item.kind for item in generator_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(_answer_text(generator_items), "g")
        self.assertEqual(
            {item.provider_family for item in generator_items},
            {"hugging_face"},
        )

        modes: list[str] = []
        payload = {"choices": [{"delta": {"content": "x"}}], "native": True}

        class ModelDumpChunk:
            choices = [
                SimpleNamespace(delta=SimpleNamespace(content="x")),
            ]

            def model_dump(self, *, mode: str) -> dict[str, object]:
                modes.append(mode)
                return payload

        payload_stream = self.mod.HuggingfaceStream(
            AsyncIter([ModelDumpChunk()])
        )
        payload_items = await _canonical_items(payload_stream)

        self.assertEqual(modes, ["json"])
        self.assertEqual(payload_items[1].provider_payload, payload)
        self.assertEqual(_answer_text(payload_items), "x")

        with patch.object(self.mod, "HuggingfaceClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="tok",
                base_url="url",
            )
            model = self.mod.HuggingfaceModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(api_key="tok", base_url="url")
        self.assertIs(loaded, ClientMock.return_value)

    async def test_stream_records_usage_after_full_consumption(self):
        usage = {
            "prompt_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 1},
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 8,
        }
        stream = self.mod.HuggingfaceStream(
            AsyncIter(
                [
                    {"choices": [{"delta": {"content": "x"}}]},
                    {"usage": usage},
                ]
            )
        )

        items = await _canonical_items(stream)
        self.assertEqual(_answer_text(items), "x")
        usage_item = next(
            item
            for item in items
            if item.kind is StreamItemKind.USAGE_COMPLETED
        )
        observation = usage_observation_from_response(stream)
        totals = usage_totals_from_response(stream)

        self.assertEqual(usage_item.provider_payload, {"usage": usage})
        self.assertEqual(stream.provider_family, "hugging_face")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata, {"provider_family": "hugging_face"}
        )
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 5)
        self.assertEqual(totals.cached_input_tokens, 1)
        self.assertEqual(totals.output_tokens, 3)
        self.assertEqual(totals.reasoning_tokens, 2)
        self.assertEqual(totals.total_tokens, 8)

    async def test_malformed_usage_is_unavailable(self):
        hf_client = self.mod.HuggingfaceClient("k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        resp = {
            "choices": [{"message": {"content": "bad"}}],
            "usage": {
                "prompt_tokens": "private prompt",
                "completion_tokens": -1,
                "total_tokens": True,
                "provider_family": "private-provider",
            },
        }
        self.client.chat_completion = AsyncMock(return_value=resp)

        gen = await hf_client("m", msgs, use_async_generator=False)
        items = await _canonical_items(gen)

        self.assertEqual(_answer_text(items), "bad")
        self.assertEqual(gen.provider_family, "hugging_face")
        self.assertIsNone(usage_observation_from_response(gen))
        self.assertIsNone(usage_totals_from_response(gen))

        self.client.chat_completion = AsyncMock(
            return_value={
                "usage": {"prompt_tokens": "private prompt"},
            }
        )
        empty_gen = await hf_client("m", msgs, use_async_generator=False)
        empty_items = await _canonical_items(empty_gen)

        self.assertEqual(_answer_text(empty_items), "")
        self.assertEqual(empty_gen.provider_family, "hugging_face")
        self.assertIsNone(usage_observation_from_response(empty_gen))
        self.assertIsNone(usage_totals_from_response(empty_gen))

    async def test_provider_instructions_are_rejected_before_api_call(self):
        hf_client = self.mod.HuggingfaceClient("k", base_url="b")

        with self.assertRaisesRegex(AssertionError, "provider instructions"):
            await hf_client(
                "m",
                [Message(role=MessageRole.USER, content="hi")],
                instructions="private policy",
            )

        self.client.chat_completion.assert_not_awaited()


class OllamaTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        class DummyAsyncClient:
            def __init__(self, host=None):
                self.host = host

            async def chat(self, *args, **kwargs):
                return kwargs

        stub = types.ModuleType("ollama")
        stub.AsyncClient = DummyAsyncClient
        self.patch = patch.dict(sys.modules, {"ollama": stub})
        self.patch.start()
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.ollama"
        )
        importlib.reload(self.mod)

    def tearDown(self):
        self.patch.stop()

    async def test_call_and_model(self):
        client = self.mod.OllamaClient(base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        client._client.chat = AsyncMock(
            return_value=AsyncIter([{"message": {"content": "s"}}])
        )
        with patch.object(self.mod, "OllamaStream") as StreamMock:
            result = await client("m", msgs, use_async_generator=True)
        client._client.chat.assert_awaited_once()
        StreamMock.assert_called_once()
        self.assertIs(result, StreamMock.return_value)

        client._client.chat = AsyncMock(
            return_value={
                "message": {"content": "x"},
                "prompt_eval_count": 5,
                "eval_count": 3,
            }
        )
        gen = await client("m", msgs, use_async_generator=False)
        items = await _canonical_items(gen)
        self.assertEqual(_answer_text(items), "x")
        self.assertEqual(gen.provider_family, "ollama")
        self.assertIsNone(usage_observation_from_response(gen))
        self.assertIsNone(usage_totals_from_response(gen))

        stream = self.mod.OllamaStream(
            AsyncIter(
                [
                    {"message": {"content": "a"}},
                    {"prompt_eval_count": 5, "eval_count": 3},
                ]
            )
        )
        stream_items = await _canonical_items(stream)
        self.assertEqual(_answer_text(stream_items), "a")
        usage_item = next(
            item
            for item in stream_items
            if item.kind is StreamItemKind.USAGE_COMPLETED
        )
        self.assertEqual(stream.provider_family, "ollama")
        self.assertEqual(
            stream.usage,
            {"prompt_eval_count": 5, "eval_count": 3},
        )
        self.assertEqual(
            usage_item.provider_payload,
            {"prompt_eval_count": 5, "eval_count": 3},
        )
        self.assertIsNone(usage_observation_from_response(stream))
        self.assertIsNone(usage_totals_from_response(stream))

        direct_stream = self.mod.OllamaStream(
            AsyncIter([{"message": {"content": "hi"}}])
        )
        started = await direct_stream.__anext__()
        delta = await direct_stream.__anext__()
        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "hi")
        self.assertEqual(delta.provider_family, "ollama")

        generator_stream = self.mod.OllamaStream(
            AsyncIter([{"message": {"content": "g"}}])
        )
        generator_items = await _canonical_items(
            cast(
                AsyncIterable[CanonicalStreamItem],
                generator_stream._generator,
            )
        )
        self.assertEqual(
            [item.kind for item in generator_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(_answer_text(generator_items), "g")
        self.assertEqual(
            {item.provider_family for item in generator_items}, {"ollama"}
        )

        with patch.object(self.mod, "OllamaClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                base_url="u",
            )
            model = self.mod.OllamaModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="u")
        self.assertIs(loaded, ClientMock.return_value)
        self.assertFalse(model._settings.enable_eval)

    async def test_provider_instructions_are_rejected_before_api_call(self):
        client = self.mod.OllamaClient(base_url="b")
        client._client.chat = AsyncMock()

        with self.assertRaisesRegex(AssertionError, "provider instructions"):
            await client(
                "m",
                [Message(role=MessageRole.USER, content="hi")],
                instructions="private policy",
            )

        client._client.chat.assert_not_awaited()


class OpenAIVendorsTestCase(TestCase):
    vendors = [
        (
            "avalan.model.nlp.text.vendor.anyscale",
            "AnyScaleClient",
            "AnyScaleModel",
            "https://api.endpoints.anyscale.com/v1",
        ),
        (
            "avalan.model.nlp.text.vendor.deepinfra",
            "DeepInfraClient",
            "DeepInfraModel",
            "https://api.deepinfra.com/v1/openai",
        ),
        (
            "avalan.model.nlp.text.vendor.deepseek",
            "DeepSeekClient",
            "DeepSeekModel",
            "https://api.deepseek.com",
        ),
        (
            "avalan.model.nlp.text.vendor.groq",
            "GroqClient",
            "GroqModel",
            "https://api.groq.com/openai/v1",
        ),
        (
            "avalan.model.nlp.text.vendor.hyperbolic",
            "HyperbolicClient",
            "HyperbolicModel",
            "https://api.hyperbolic.ai/v1",
        ),
    ]

    def setUp(self):
        class Omit:
            def __bool__(self):
                return False

        self.openai_stub = types.ModuleType("openai")
        self.openai_stub.AsyncOpenAI = MagicMock()
        self.openai_stub.AsyncStream = MagicMock()
        self.openai_stub.Omit = Omit
        self.patch = patch.dict(sys.modules, {"openai": self.openai_stub})
        self.patch.start()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )

    def tearDown(self):
        self.patch.stop()

    def test_clients_and_models(self):
        for module_path, client_name, model_name, default_url in self.vendors:
            with self.subTest(module=module_path):
                mod = importlib.import_module(module_path)
                importlib.reload(mod)
                self.openai_stub.AsyncOpenAI.reset_mock()
                getattr(mod, client_name)(api_key="k", base_url=None)
                self.openai_stub.AsyncOpenAI.assert_called_once_with(
                    base_url=default_url, api_key="k"
                )
                with patch.object(mod, client_name) as ClientMock:
                    settings = TransformerEngineSettings(
                        auto_load_model=False,
                        auto_load_tokenizer=False,
                        access_token="t",
                        base_url="b",
                    )
                    model = getattr(mod, model_name)("m", settings)
                    loaded = model._load_model()
                ClientMock.assert_called_once_with(base_url="b", api_key="t")
                self.assertIs(loaded, ClientMock.return_value)


class LiteLLMTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        stub = types.ModuleType("litellm")
        stub.acompletion = AsyncMock()
        self.patch = patch.dict(sys.modules, {"litellm": stub})
        self.patch.start()
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.litellm"
        )
        importlib.reload(self.mod)
        self.stub = stub

    def tearDown(self):
        self.patch.stop()

    async def test_call_and_model(self):
        client = self.mod.LiteLLMClient(api_key="k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        stream_obj = AsyncIter([{"choices": [{"delta": {"content": "s"}}]}])
        self.stub.acompletion = AsyncMock(return_value=stream_obj)
        with patch.object(self.mod, "LiteLLMStream") as StreamMock:
            result = await client("m", msgs, use_async_generator=True)
        self.stub.acompletion.assert_awaited_once()
        StreamMock.assert_called_once_with(stream_obj)
        self.assertIs(result, StreamMock.return_value)

        resp = {
            "choices": [{"message": {"content": "r"}}],
            "usage": {
                "prompt_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens": 4,
                "completion_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 11,
            },
        }
        self.stub.acompletion = AsyncMock(return_value=resp)
        gen = await client("m", msgs, use_async_generator=False)
        items = await _canonical_items(gen)
        observation = usage_observation_from_response(gen)
        totals = usage_totals_from_response(gen)
        self.assertEqual(_answer_text(items), "r")
        self.assertEqual(gen.provider_family, "openai_compatible")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata,
            {"provider_family": "openai_compatible"},
        )
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 7)
        self.assertEqual(totals.cached_input_tokens, 2)
        self.assertEqual(totals.output_tokens, 4)
        self.assertEqual(totals.reasoning_tokens, 1)
        self.assertEqual(totals.total_tokens, 11)

        stream = self.mod.LiteLLMStream(
            AsyncIter([{"choices": [{"delta": {"content": "x"}}]}])
        )
        stream_items = await _canonical_items(stream)
        self.assertEqual(_answer_text(stream_items), "x")

        direct_stream = self.mod.LiteLLMStream(
            AsyncIter([{"choices": [{"delta": {"content": "hi"}}]}])
        )
        started = await direct_stream.__anext__()
        delta = await direct_stream.__anext__()
        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "hi")
        self.assertEqual(delta.provider_family, "openai_compatible")

        self.assertEqual(
            self.mod.LiteLLMClient._delta_text(
                {"choices": [{"delta": {"content": "helper"}}]}
            ),
            "helper",
        )
        self.assertIsNone(self.mod.LiteLLMClient._delta_text({"choices": []}))
        self.assertIsNone(
            self.mod.LiteLLMClient._delta_text(
                {"choices": [{"delta": {"content": 3}}]}
            )
        )

        generator_usage = {"prompt_tokens": 1, "completion_tokens": 2}
        generator_stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    {"choices": [{"delta": {"content": "canonical"}}]},
                    {"usage": generator_usage},
                    {"choices": []},
                ]
            )
        )
        generator_items = await _canonical_items(
            cast(
                AsyncIterable[CanonicalStreamItem],
                generator_stream._generator,
            )
        )
        self.assertEqual(
            [item.kind for item in generator_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(generator_items)
        self.assertEqual(accumulator.answer_text, "canonical")
        self.assertEqual(accumulator.final_usage, generator_usage)
        self.assertEqual(generator_stream.usage, generator_usage)

        with patch.object(self.mod, "LiteLLMClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="u",
            )
            model = self.mod.LiteLLMModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(api_key="t", base_url="u")
        self.assertIs(loaded, ClientMock.return_value)

    async def test_stream_records_usage_after_full_consumption(self):
        usage = {
            "prompt_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 1},
            "completion_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 9,
        }
        stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    {"choices": [{"delta": {"content": "x"}}]},
                    {"usage": usage},
                ]
            )
        )

        items = await _canonical_items(stream)
        self.assertEqual(_answer_text(items), "x")
        observation = usage_observation_from_response(stream)
        totals = usage_totals_from_response(stream)

        self.assertEqual(stream.provider_family, "openai_compatible")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata,
            {"provider_family": "openai_compatible"},
        )
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 5)
        self.assertEqual(totals.cached_input_tokens, 1)
        self.assertEqual(totals.output_tokens, 4)
        self.assertEqual(totals.reasoning_tokens, 2)
        self.assertEqual(totals.total_tokens, 9)

    async def test_streaming_object_chunk(self):
        client = self.mod.LiteLLMClient(api_key="k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="s"))]
        )
        stream_obj = AsyncIter([chunk])
        self.stub.acompletion = AsyncMock(return_value=stream_obj)
        result = await client("m", msgs, use_async_generator=True)
        self.stub.acompletion.assert_awaited_once()
        result_items = await _canonical_items(result)
        self.assertEqual(_answer_text(result_items), "s")

    async def test_canonical_stream_maps_chat_chunks(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 3}
        stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": "think ",
                                    "content": "hi ",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-1",
                                            "function": {
                                                "name": "pkg.lookup",
                                                "arguments": '{"city"',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": ':"Paris"}'
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                    {"usage": usage},
                ]
            )
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(12)))
        self.assertEqual(
            {item.provider_family for item in items}, {"openai_compatible"}
        )
        self.assertEqual(
            items[0].metadata["capabilities"]["backend"], "hosted"
        )
        self.assertEqual(
            items[1].provider_event_type,
            "chat.completion.reasoning.delta",
        )
        self.assertIs(items[1].channel, StreamChannel.REASONING)
        self.assertEqual(
            {
                item.correlation.tool_call_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"call-1"},
        )
        self.assertEqual(items[5].data, {"name": "pkg.lookup"})
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "hi ")
        self.assertEqual(accumulator.reasoning_text, "think ")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-1": '{"city":"Paris"}'},
        )
        self.assertEqual(accumulator.final_usage, usage)
        self.assertIs(
            accumulator.terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_canonical_stream_preserves_chat_model_dump_payloads(self):
        first_payload = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "think ",
                        "content": "hi ",
                    }
                }
            ]
        }
        tool_payload = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "pkg.lookup",
                                    "arguments": '{"city"',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        usage_payload = {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}
        modes: list[tuple[str, str]] = []

        class ModelDumpChunk:
            def __init__(self, payload: dict[str, object], label: str):
                self.choices = payload.get("choices")
                self.usage = payload.get("usage")
                self._label = label
                self._payload = payload

            def model_dump(self, *, mode: str) -> dict[str, object]:
                modes.append((self._label, mode))
                return self._payload

        stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    ModelDumpChunk(first_payload, "first"),
                    ModelDumpChunk(tool_payload, "tool"),
                    ModelDumpChunk(usage_payload, "usage"),
                ]
            )
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(
            modes,
            [("first", "json"), ("tool", "json"), ("usage", "json")],
        )
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(items[1].provider_payload, first_payload)
        self.assertEqual(items[2].provider_payload, first_payload)
        self.assertEqual(items[3].provider_payload, tool_payload)
        self.assertEqual(items[4].provider_payload, tool_payload)
        self.assertEqual(items[5].provider_payload, tool_payload)
        self.assertEqual(items[8].provider_payload, usage_payload)

    async def test_canonical_stream_ignores_non_object_chat_provider_payload(
        self,
    ):
        class ModelDumpChunk:
            choices = [
                {
                    "delta": {
                        "content": "hi",
                    }
                }
            ]

            def model_dump(self, *, mode: str) -> object:
                return ["not", "an", "event", mode]

        stream = self.mod.LiteLLMStream(AsyncIter([ModelDumpChunk()]))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(items[1].text_delta, "hi")
        self.assertIsNone(items[1].provider_payload)

    async def test_canonical_stream_preserves_same_chunk_usage_order(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 1}
        stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    {
                        "choices": [{"delta": {"content": "done"}}],
                        "usage": usage,
                    },
                ]
            )
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "done")
        self.assertEqual(accumulator.final_usage, usage)
        self.assertIs(
            accumulator.terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_canonical_stream_rejects_content_after_usage_chunk(self):
        stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    {"usage": {"prompt_tokens": 2}},
                    {"choices": [{"delta": {"content": "late"}}]},
                ]
            )
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_data = items[2].data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "StreamValidationError")
        self.assertIn("final usage", error_data["message"])
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    async def test_canonical_stream_maps_chat_errors_to_terminal(self):
        stream = self.mod.LiteLLMStream(
            AsyncIter([{"error": {"message": "bad request"}}])
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].data, {"error": {"message": "bad request"}})
        self.assertIs(
            items[1].terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    async def test_provider_events_from_chunk_stops_after_error(self):
        stream = self.mod.LiteLLMStream(AsyncIter([]))

        events = [
            event
            async for event in stream._provider_events_from_chunk(
                {
                    "error": {"message": "bad request"},
                    "usage": {"prompt_tokens": 1},
                }
            )
        ]

        self.assertEqual(
            [event.kind for event in events],
            [StreamItemKind.STREAM_ERRORED],
        )

    async def test_canonical_stream_maps_malformed_chat_chunks_to_error(
        self,
    ):
        stream = self.mod.LiteLLMStream(
            AsyncIter([{"choices": [{"delta": {"content": 3}}]}])
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].data["error_type"], "ValueError")
        self.assertIn("content", items[1].data["message"])

    async def test_canonical_stream_ignores_empty_chat_control_chunks(self):
        stream = self.mod.LiteLLMStream(
            AsyncIter(
                [
                    SimpleNamespace(),
                    {"choices": [], "usage": {"prompt_tokens": 1}},
                    {"choices": [{}]},
                    {"choices": [{"delta": {}}]},
                ]
            )
        )

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="chat-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].usage, {"prompt_tokens": 1})
        self.assertIsNone(items[2].usage)

    async def test_canonical_stream_maps_malformed_tool_chunks_to_error(
        self,
    ):
        cases = (
            ({"choices": "bad"}, "choices"),
            ({"choices": [{"delta": {"reasoning": 3}}]}, "reasoning"),
            ({"choices": [{"delta": {"tool_calls": {}}}]}, "tool_calls"),
            (
                {"choices": [{"delta": {"tool_calls": [{"index": "0"}]}}]},
                "index",
            ),
            (
                {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]},
                "id is missing",
            ),
            (
                {"choices": [{"delta": {"tool_calls": [{"id": ""}]}}]},
                "id is invalid",
            ),
            (
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {"name": 3},
                                    }
                                ]
                            }
                        }
                    ]
                },
                "name",
            ),
            (
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {"arguments": {}},
                                    }
                                ]
                            }
                        }
                    ]
                },
                "arguments",
            ),
        )

        for chunk, message in cases:
            with self.subTest(message=message):
                stream = self.mod.LiteLLMStream(AsyncIter([chunk]))

                items = [
                    item
                    async for item in stream.canonical_stream(
                        stream_session_id="chat-stream",
                        run_id="run-1",
                        turn_id="turn-1",
                    )
                ]

                self.assertEqual(items[1].kind, StreamItemKind.STREAM_ERRORED)
                self.assertIs(
                    items[1].terminal_outcome,
                    StreamTerminalOutcome.ERRORED,
                )
                error_data = items[1].data
                assert isinstance(error_data, dict)
                self.assertEqual(error_data["error_type"], "ValueError")
                self.assertIn(message, error_data["message"])

    async def test_no_stream_object_response(self):
        client = self.mod.LiteLLMClient(api_key="k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="r"))]
        )
        self.stub.acompletion = AsyncMock(return_value=resp)
        gen = await client("m", msgs, use_async_generator=False)
        items = await _canonical_items(gen)
        self.assertEqual(_answer_text(items), "r")

    async def test_malformed_usage_is_unavailable(self):
        client = self.mod.LiteLLMClient(api_key="k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        resp = {
            "choices": [{"message": {"content": "bad"}}],
            "usage": {
                "prompt_tokens": "private prompt",
                "completion_tokens": -1,
                "total_tokens": True,
                "provider_family": "private-provider",
            },
        }
        self.stub.acompletion = AsyncMock(return_value=resp)

        gen = await client("m", msgs, use_async_generator=False)
        items = await _canonical_items(gen)

        self.assertEqual(_answer_text(items), "bad")
        self.assertEqual(gen.provider_family, "openai_compatible")
        self.assertIsNone(usage_observation_from_response(gen))
        self.assertIsNone(usage_totals_from_response(gen))

        self.stub.acompletion = AsyncMock(
            return_value={"usage": {"prompt_tokens": "private prompt"}}
        )
        empty_gen = await client("m", msgs, use_async_generator=False)
        empty_items = await _canonical_items(empty_gen)

        self.assertEqual(_answer_text(empty_items), "")
        self.assertEqual(empty_gen.provider_family, "openai_compatible")
        self.assertIsNone(usage_observation_from_response(empty_gen))
        self.assertIsNone(usage_totals_from_response(empty_gen))

    async def test_provider_instructions_are_rejected_before_api_call(self):
        client = self.mod.LiteLLMClient(api_key="k", base_url="b")

        with self.assertRaisesRegex(AssertionError, "provider instructions"):
            await client(
                "m",
                [Message(role=MessageRole.USER, content="hi")],
                instructions="private policy",
            )

        self.stub.acompletion.assert_not_awaited()


if __name__ == "__main__":
    from unittest import main

    main()
