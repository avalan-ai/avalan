import importlib
import sys
import types
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
    MessageRole,
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
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def patch_openai_imports():
    openai_stub = types.ModuleType("openai")
    openai_stub.__spec__ = ModuleSpec("openai", loader=None)
    openai_stub.AsyncOpenAI = MagicMock()
    openai_stub.AsyncStream = MagicMock()
    openai_stub.AsyncOpenAI.return_value.responses = MagicMock()

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.__spec__ = ModuleSpec("transformers", loader=None)
    transformers_stub.PreTrainedModel = MagicMock()
    transformers_stub.PreTrainedTokenizer = MagicMock()
    transformers_stub.PreTrainedTokenizerFast = MagicMock()
    transformers_stub.__getattr__ = lambda name: MagicMock()

    transformers_utils_stub = types.ModuleType("transformers.utils")
    transformers_utils_stub.get_json_schema = MagicMock()
    transformers_logging_stub = types.ModuleType("transformers.utils.logging")
    transformers_logging_stub.disable_progress_bar = MagicMock()
    transformers_logging_stub.enable_progress_bar = MagicMock()
    transformers_utils_stub.logging = transformers_logging_stub
    transformers_tokenization_stub = types.ModuleType(
        "transformers.tokenization_utils_base"
    )
    transformers_tokenization_stub.BatchEncoding = MagicMock()
    transformers_stub.tokenization_utils_base = transformers_tokenization_stub
    transformers_generation_stub = types.ModuleType("transformers.generation")
    transformers_generation_stub.StoppingCriteria = MagicMock()
    transformers_generation_stub.StoppingCriteriaList = MagicMock()
    transformers_stub.generation = transformers_generation_stub
    transformers_stub.utils = transformers_utils_stub

    diffusers_stub = types.ModuleType("diffusers")
    diffusers_stub.__spec__ = ModuleSpec("diffusers", loader=None)
    diffusers_stub.DiffusionPipeline = MagicMock()

    patcher = patch.dict(
        sys.modules,
        {
            "openai": openai_stub,
            "transformers": transformers_stub,
            "transformers.utils": transformers_utils_stub,
            "transformers.utils.logging": transformers_logging_stub,
            "transformers.tokenization_utils_base": (
                transformers_tokenization_stub
            ),
            "transformers.generation": transformers_generation_stub,
            "diffusers": diffusers_stub,
        },
    )
    patcher.start()
    return openai_stub, patcher


class OpenAITestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def test_stream_client_and_model(self):
        chunks = [
            SimpleNamespace(type="response.output_text.delta", delta="x"),
            SimpleNamespace(type="response.output_text.delta", delta="y"),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(chunks))
        t1 = await stream.__anext__()
        self.assertIsInstance(t1, Token)
        self.assertEqual(t1.token, "x")
        t2 = await stream.__anext__()
        self.assertIsInstance(t2, Token)
        self.assertEqual(t2.token, "y")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        with patch.object(self.mod, "OpenAIStream") as StreamMock:
            result = await client("m", [])
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="b", api_key="k"
        )
        client._client.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"c": 1}],
            stream=True,
            timeout=None,
        )
        StreamMock.assert_called_once_with(stream=stream_instance)
        self.assertIs(result, StreamMock.return_value)

    async def test_client_consumes_tokens(self):
        chunks = [
            SimpleNamespace(type="response.output_text.delta", delta="a"),
            SimpleNamespace(type="response.output_text.delta", delta="b"),
        ]
        stream_instance = AsyncIter(chunks)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        result = await client("m", [])
        t1 = await result.__anext__()
        self.assertIsInstance(t1, Token)
        self.assertEqual(t1.token, "a")
        t2 = await result.__anext__()
        self.assertIsInstance(t2, Token)
        self.assertEqual(t2.token, "b")
        with self.assertRaises(StopAsyncIteration):
            await result.__anext__()

        with patch.object(self.mod, "OpenAIClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="u",
            )
            model = self.mod.OpenAIModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="u", api_key="t")
        self.assertIs(loaded, ClientMock.return_value)

    async def test_stream_event_types(self):
        events = [
            SimpleNamespace(type="response.output_item.added"),
            SimpleNamespace(type="response.content_part.added"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="r1"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="r2"),
            SimpleNamespace(type="response.output_item.done"),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    id="c1",
                    custom_tool_call=SimpleNamespace(
                        id="c1", name="pkg__func"
                    ),
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c1",
                delta="{",
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c1",
                delta="}",
            ),
            SimpleNamespace(
                type="response.output_item.done", item=SimpleNamespace(id="c1")
            ),
            SimpleNamespace(type="response.output_item.added"),
            SimpleNamespace(type="response.content_part.added"),
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
            SimpleNamespace(type="response.output_item.done"),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))
        t1 = await stream.__anext__()
        self.assertIsInstance(t1, ReasoningToken)
        self.assertEqual(t1.token, "r1")
        t2 = await stream.__anext__()
        self.assertIsInstance(t2, ReasoningToken)
        self.assertEqual(t2.token, "r2")
        t3 = await stream.__anext__()
        self.assertIsInstance(t3, ToolCallToken)
        self.assertEqual(t3.token, "{")
        t4 = await stream.__anext__()
        self.assertIsInstance(t4, ToolCallToken)
        self.assertEqual(t4.token, "}")
        t5 = await stream.__anext__()
        self.assertIsInstance(t5, ToolCallToken)
        self.assertEqual(t5.call.id, "c1")
        self.assertEqual(t5.call.name, "pkg.func")
        self.assertEqual(t5.call.arguments, {})
        self.assertEqual(
            t5.token,
            '<tool_call>{"name": "pkg.func", "arguments": {}}</tool_call>',
        )
        t6 = await stream.__anext__()
        self.assertIsInstance(t6, Token)
        self.assertEqual(t6.token, "hi")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

    async def test_function_call_events(self):
        events = [
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c2",
                delta="{",
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c2",
                delta="}",
            ),
            SimpleNamespace(
                type="response.output_item.done", item=SimpleNamespace(id="c2")
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="c3",
                    name="pkg__f",
                    arguments='{"p": 1}',
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))
        await stream.__anext__()
        await stream.__anext__()
        t3 = await stream.__anext__()
        self.assertEqual(t3.call.id, "c2")
        self.assertEqual(t3.call.arguments, {})
        t4 = await stream.__anext__()
        self.assertEqual(t4.call.id, "c3")
        self.assertEqual(t4.call.name, "pkg.f")
        self.assertEqual(t4.call.arguments, {"p": 1})
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

    async def test_generation_settings_and_tools(self):
        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        tool = MagicMock()
        tool.json_schemas.return_value = [
            {"type": "function", "function": {"name": "pkg.func"}}
        ]
        settings = GenerationSettings(
            temperature=0.5,
            top_p=0.8,
            max_new_tokens=10,
            stop_strings=["stop"],
            response_format={"type": "json_schema"},
        )
        await client(
            "m",
            [],
            settings=settings,
            tool=tool,
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"c": 1}],
            stream=True,
            timeout=None,
            max_output_tokens=10,
            temperature=0.5,
            top_p=0.8,
            text={"stop": ["stop"]},
            response_format={"type": "json_schema"},
            tools=[{"type": "function", "name": "pkg__func"}],
        )


class VendorClientsTestCase(TestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )

    def tearDown(self):
        self.patch.stop()

    def test_openrouter_client_and_model(self):
        mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openrouter"
        )
        importlib.reload(mod)
        self.openai_stub.AsyncOpenAI.reset_mock()
        client = mod.OpenRouterClient(api_key="k", base_url=None)
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="k"
        )
        client._client.headers.update.assert_called_once_with(
            {
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
                "X-Title": "avalan",
            }
        )
        with patch.object(mod, "OpenRouterClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="b",
            )
            model = mod.OpenRouterModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="b", api_key="t")
        self.assertIs(loaded, ClientMock.return_value)

    def test_together_client_and_model(self):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.together")
        importlib.reload(mod)
        self.openai_stub.AsyncOpenAI.reset_mock()
        mod.TogetherClient(api_key="k", base_url=None)
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="https://api.together.xyz/v1", api_key="k"
        )
        with patch.object(mod, "TogetherClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="b",
            )
            model = mod.TogetherModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="b", api_key="t")
        self.assertIs(loaded, ClientMock.return_value)


class NonStreamingResponseTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def test_response_single_stream(self):
        resp = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="ok")])]
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=resp
        )
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="tok",
            base_url="url",
        )
        model = self.mod.OpenAIModel("m", settings)
        model._model = model._load_model()
        gen = GenerationSettings(use_async_generator=False)
        response = await model("hi", settings=gen)
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="url", api_key="tok"
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"role": "user", "content": "hi"}],
            stream=False,
            timeout=None,
            temperature=1.0,
            top_p=1.0,
        )
        from avalan.model.stream import TextGenerationSingleStream

        self.assertIsInstance(response._output_fn, TextGenerationSingleStream)
        self.assertFalse(response._use_async_generator)
        self.assertEqual(await response.to_str(), "ok")


class TemplateMessagesFormatTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def _assert_messages(self, content, expected_content):
        resp = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=resp
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        message = Message(role=MessageRole.USER, content=content)
        await client("model", [message], use_async_generator=False)
        create_mock = (
            self.openai_stub.AsyncOpenAI.return_value.responses.create
        )
        create_mock.assert_awaited_once()
        kwargs = create_mock.await_args.kwargs
        self.assertEqual(
            kwargs["input"], [{"role": "user", "content": expected_content}]
        )

    async def test_string_message(self):
        await self._assert_messages("hi", "hi")

    async def test_text_message_content(self):
        content = MessageContentText(type="text", text="hi")
        await self._assert_messages(content, "hi")

    async def test_image_message_content(self):
        content = MessageContentImage(type="image_url", image_url={"url": "u"})
        await self._assert_messages(
            content, [{"type": "image_url", "image_url": {"url": "u"}}]
        )

    async def test_mixed_message_content(self):
        content = [
            MessageContentText(type="text", text="hi"),
            MessageContentImage(type="image_url", image_url={"url": "u"}),
        ]
        await self._assert_messages(
            content,
            [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "u"}},
            ],
        )

    async def test_non_stream_tool_call_output(self):
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="output_text",
                    content=[SimpleNamespace(text="hello ")],
                ),
                SimpleNamespace(
                    type="tool_call",
                    call=SimpleNamespace(
                        id="call1",
                        function=SimpleNamespace(
                            name="pkg__tool", arguments='{"a":1}'
                        ),
                    ),
                ),
            ]
        )

        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )

        with patch.object(
            self.mod.TextGenerationVendor,
            "build_tool_call_token",
            return_value=ToolCallToken(token="<tool_call />"),
        ) as build_token:
            client = self.mod.OpenAIClient(api_key="key", base_url="url")
            message = Message(role=MessageRole.USER, content="hi")
            stream = await client(
                "model", [message], use_async_generator=False
            )

        from avalan.model.stream import TextGenerationSingleStream

        self.assertIsInstance(stream, TextGenerationSingleStream)
        self.assertEqual(stream.content, "hello <tool_call />")
        build_token.assert_called_once_with("call1", "pkg__tool", '{"a":1}')


class TemplateAndToolSchemaTestCase(TestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    def test_tool_schemas_none(self):
        tool = MagicMock()
        tool.json_schemas.return_value = None
        self.assertIsNone(self.mod.OpenAIClient._tool_schemas(tool))
        tool.json_schemas.return_value = [{"type": "x"}]
        self.assertEqual(self.mod.OpenAIClient._tool_schemas(tool), [])

    def test_template_messages_tool_results(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")

        @dataclass
        class R:
            v: int

        call1 = ToolCall(id="c1", name="pkg.func", arguments={"a": 1})
        result1 = ToolCallResult(
            id="c1",
            name="pkg.func",
            arguments={"a": 1},
            call=call1,
            result=R(v=2),
        )
        msg1 = Message(role=MessageRole.TOOL, tool_call_result=result1)

        call2 = ToolCall(id="c2", name="pkg.func2")
        result2 = ToolCallResult(
            id="c2", name="pkg.func2", call=call2, result={"x": 3}
        )
        msg2 = Message(role=MessageRole.TOOL, tool_call_result=result2)

        templated = client._template_messages([msg1, msg2])
        self.assertEqual(
            templated,
            [
                {
                    "type": "function_call",
                    "name": "pkg__func",
                    "call_id": "c1",
                    "arguments": '{"a": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"v": 2}',
                },
                {
                    "type": "function_call",
                    "name": "pkg__func2",
                    "call_id": "c2",
                    "arguments": "null",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": '{"x": 3}',
                },
            ],
        )

    def test_template_messages_tool_error(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        call = ToolCall(id="c1", name="pkg.func", arguments={"a": 1})
        error = ToolCallError(
            id="c1",
            name="pkg.func",
            call=call,
            error=ValueError("boom"),
            message="boom",
        )
        msg = Message(role=MessageRole.TOOL, tool_call_error=error)
        templated = client._template_messages([msg])
        self.assertEqual(
            templated,
            [
                {
                    "type": "function_call",
                    "name": "pkg__func",
                    "call_id": "c1",
                    "arguments": '{"a": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"error": "boom"}',
                },
            ],
        )


class OpenAIAdditionalCoverageTestCase(TestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    def test_non_stream_response_content_handles_dict(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"text": "hello"}],
                },
                {
                    "type": "tool_call",
                    "call": {
                        "id": "call-id",
                        "function": {
                            "name": "pkg.tool",
                            "arguments": '{"a":1}',
                        },
                    },
                },
            ]
        }
        token = SimpleNamespace(token="<tool>")
        with patch.object(
            self.mod.TextGenerationVendor,
            "build_tool_call_token",
            return_value=token,
        ) as build:
            text = self.mod.OpenAIClient._non_stream_response_content(response)

        self.assertEqual(text, "hello<tool>")
        build.assert_called_once_with("call-id", "pkg.tool", '{"a":1}')

    def test_non_streaming_response_str_variants(self):
        settings = GenerationSettings()
        response = self.mod.OpenAINonStreamingResponse(
            lambda **_: "value",
            logger=MagicMock(),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
            static_response_text="cached",
        )
        self.assertEqual(str(response), "cached")

        buffered = self.mod.OpenAINonStreamingResponse(
            lambda **_: "value",
            logger=MagicMock(),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        buffered._buffer.write("buffered")
        self.assertEqual(str(buffered), "buffered")

        fallback = self.mod.OpenAINonStreamingResponse(
            lambda **_: "value",
            logger=MagicMock(),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        fallback._buffer = SimpleNamespace(getvalue=lambda: None)
        self.assertIn("OpenAINonStreamingResponse", str(fallback))


class OpenAIModelStreamingFlagTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def test_call_returns_streaming_response(self):
        settings = TransformerEngineSettings(access_token="tok")
        model = self.mod.OpenAIModel("model-id", settings)
        model._model = AsyncMock(
            return_value=lambda *_args, **_kwargs: AsyncIter([])
        )

        response = await model(
            "prompt",
            system_prompt="sys",
            developer_prompt="dev",
            settings=GenerationSettings(),
        )

        self.assertIsInstance(response, self.mod.TextGenerationResponse)


if __name__ == "__main__":
    from unittest import main

    main()
