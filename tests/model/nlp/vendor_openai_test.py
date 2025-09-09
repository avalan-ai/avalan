import importlib
import sys
import types
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


class OpenAITestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub = types.ModuleType("openai")
        self.openai_stub.__spec__ = ModuleSpec("openai", loader=None)
        self.openai_stub.AsyncOpenAI = MagicMock()
        self.openai_stub.AsyncStream = MagicMock()
        self.openai_stub.AsyncOpenAI.return_value.responses = MagicMock()
        self.patch = patch.dict(sys.modules, {"openai": self.openai_stub})
        self.patch.start()
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
        self.assertEqual(await stream.__anext__(), "x")
        self.assertEqual(await stream.__anext__(), "y")
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
        self.assertEqual(await result.__anext__(), "a")
        self.assertEqual(await result.__anext__(), "b")
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
        self.openai_stub = types.ModuleType("openai")
        self.openai_stub.__spec__ = ModuleSpec("openai", loader=None)
        self.openai_stub.AsyncOpenAI = MagicMock()
        self.openai_stub.AsyncStream = MagicMock()
        self.openai_stub.AsyncOpenAI.return_value.responses = MagicMock()
        self.patch = patch.dict(sys.modules, {"openai": self.openai_stub})
        self.patch.start()
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
        self.openai_stub = types.ModuleType("openai")
        self.openai_stub.AsyncOpenAI = MagicMock()
        self.openai_stub.AsyncOpenAI.return_value.responses = MagicMock()
        self.openai_stub.AsyncStream = MagicMock()
        self.patch = patch.dict(sys.modules, {"openai": self.openai_stub})
        self.patch.start()
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
        self.openai_stub = types.ModuleType("openai")
        self.openai_stub.AsyncOpenAI = MagicMock()
        self.openai_stub.AsyncOpenAI.return_value.responses = MagicMock()
        self.openai_stub.AsyncStream = MagicMock()
        self.patch = patch.dict(sys.modules, {"openai": self.openai_stub})
        self.patch.start()
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


if __name__ == "__main__":
    from unittest import main

    main()
