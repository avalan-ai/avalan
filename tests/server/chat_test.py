from pytest import importorskip
from unittest import IsolatedAsyncioTestCase
import sys
import importlib
from types import ModuleType
from pathlib import Path

from avalan.model.nlp.text import TextGenerationResponse


class ChatCompletionEndpointTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.FastAPI = FastAPI
        self.TestClient = TestClient

        # Create a dummy `avalan.server` package to avoid executing its __init__
        server_pkg = ModuleType("avalan.server")
        server_pkg.__path__ = [str(Path("src/avalan/server").resolve())]
        sys.modules["avalan.server"] = server_pkg

        # Alias `Agent` inside avalan.agent for router import
        import avalan.agent as agent_mod
        self.agent_mod = agent_mod

        class DummyAgent:
            async def __call__(self, messages, settings=None):
                return TextGenerationResponse(lambda: "ok", use_async_generator=False)

        agent_mod.Agent = DummyAgent
        self.DummyAgent = DummyAgent

        # Import router
        self.chat = importlib.import_module("avalan.server.routers.chat")

    def tearDown(self):
        sys.modules.pop("avalan.server.routers.chat", None)
        sys.modules.pop("avalan.server", None)
        if hasattr(self.agent_mod, "Agent"):
            delattr(self.agent_mod, "Agent")

    async def test_non_streaming_completion(self):
        app = self.FastAPI()
        app.state.agent = self.DummyAgent()
        app.include_router(self.chat.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        }
        resp = client.post("/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(
            data["choices"][0]["message"]["content"],
            "ok",
        )

    async def test_streaming_completion(self):
        async def output_gen():
            yield "a"
            yield "b"

        def output_fn():
            return output_gen()

        class StreamingAgent:
            async def __call__(self, messages, settings=None):
                return TextGenerationResponse(output_fn, use_async_generator=True)

        self.agent_mod.Agent = StreamingAgent

        app = self.FastAPI()
        app.state.agent = StreamingAgent()
        app.include_router(self.chat.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"}
            ],
            "stream": True,
        }
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            self.assertEqual(resp.status_code, 200)
            chunks = list(resp.iter_lines())
        self.assertTrue(chunks[-1].decode().endswith("[DONE]"))
        self.assertTrue(chunks[0].decode().startswith("data: {"))
