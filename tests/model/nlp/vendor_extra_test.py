from contextlib import AsyncExitStack
from dataclasses import dataclass
import importlib
import sys
import types
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    GenerationSettings,
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
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class AnthropicTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
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
            yield Delta(types.SimpleNamespace())
            yield Delta(types.SimpleNamespace(partial_json="val"))
            yield Stop()

        stream = self.mod.AnthropicStream(agen())
        token = await stream.__anext__()
        self.assertIsInstance(token, ToolCallToken)
        self.assertEqual(token.token, "val")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

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
            yield Delta(SimpleNamespace(partial_json="frag"), index=1)
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

        with patch.object(
            self.mod.TextGenerationVendor,
            "build_tool_call_token",
            return_value="call",
        ) as btt:
            stream = self.mod.AnthropicStream(agen())
            out = []
            while True:
                try:
                    out.append(await stream.__anext__())
                except StopAsyncIteration:
                    break

        self.assertEqual(len(out), 5)
        self.assertIsInstance(out[0], ReasoningToken)
        self.assertEqual(out[0].token, "think")
        self.assertIsInstance(out[1], ToolCallToken)
        self.assertEqual(out[1].token, '{"a":1}')
        self.assertIsInstance(out[2], ToolCallToken)
        self.assertEqual(out[2].token, "frag")
        self.assertIsInstance(out[3], Token)
        self.assertEqual(out[3].token, "txt")
        self.assertEqual(out[4], "call")
        btt.assert_called_once_with("tid", "tname", {"x": 1})

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
        self.assertEqual(templated[1]["content"][1]["name"], "pkg__tool")
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
                    "name": "pkg__tool",
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
                            return_value=SimpleNamespace(text="r")
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
        out = [t async for t in gen]
        self.assertEqual(out, ["r"])
        client._client.aio.models.generate_content.assert_awaited_once()

        stream = self.mod.GoogleStream(AsyncIter([SimpleNamespace(text="x")]))
        self.assertEqual(await stream.__anext__(), "x")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

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

        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="r"))]
        )
        self.client.chat_completion = AsyncMock(return_value=resp)
        gen = await hf_client("m", msgs, settings, use_async_generator=False)
        out = [t async for t in gen]
        self.assertEqual(out, ["r"])

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
        self.assertEqual(await stream.__anext__(), "x")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

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
            return_value={"message": {"content": "x"}}
        )
        gen = await client("m", msgs, use_async_generator=False)
        out = [t async for t in gen]
        self.assertEqual(out, ["x"])

        stream = self.mod.OllamaStream(
            AsyncIter([{"message": {"content": "a"}}])
        )
        self.assertEqual(await stream.__anext__(), "a")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

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
        self.openai_stub = types.ModuleType("openai")
        self.openai_stub.AsyncOpenAI = MagicMock()
        self.openai_stub.AsyncStream = MagicMock()
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

        resp = {"choices": [{"message": {"content": "r"}}]}
        self.stub.acompletion = AsyncMock(return_value=resp)
        gen = await client("m", msgs, use_async_generator=False)
        out = [t async for t in gen]
        self.assertEqual(out, ["r"])

        stream = self.mod.LiteLLMStream(
            AsyncIter([{"choices": [{"delta": {"content": "x"}}]}])
        )
        self.assertEqual(await stream.__anext__(), "x")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

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
        self.assertEqual(await result.__anext__(), "s")
        with self.assertRaises(StopAsyncIteration):
            await result.__anext__()

    async def test_no_stream_object_response(self):
        client = self.mod.LiteLLMClient(api_key="k", base_url="b")
        msgs = [Message(role=MessageRole.USER, content="hi")]
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="r"))]
        )
        self.stub.acompletion = AsyncMock(return_value=resp)
        gen = await client("m", msgs, use_async_generator=False)
        out = [t async for t in gen]
        self.assertEqual(out, ["r"])


if __name__ == "__main__":
    from unittest import main

    main()
