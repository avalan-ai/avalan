import importlib
import sys
import types
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import TransformerEngineSettings


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
        self.openai_stub.AsyncOpenAI = MagicMock()
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

    async def test_stream_client_and_model(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="x"))]
        )
        stream = self.mod.OpenAIStream(AsyncIter([chunk]))
        self.assertEqual(await stream.__anext__(), "x")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.chat.completions.create = (
            AsyncMock(return_value=stream_instance)
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        with patch.object(self.mod, "OpenAIStream") as StreamMock:
            result = await client("m", [])
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="b", api_key="k"
        )
        client._client.chat.completions.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            messages=[{"c": 1}],
            stream=True,
            timeout=None,
            response_format=None,
        )
        StreamMock.assert_called_once_with(stream=stream_instance)
        self.assertIs(result, StreamMock.return_value)

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


class VendorClientsTestCase(TestCase):
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


if __name__ == "__main__":
    from unittest import main

    main()
