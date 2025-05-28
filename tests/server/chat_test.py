from avalan.agent.orchestrator import Orchestrator
from avalan.model import TextGenerationResponse
import importlib
from pathlib import Path
import sys
from types import ModuleType
from unittest import IsolatedAsyncioTestCase, skip
from unittest.mock import AsyncMock


class DummyOrchestrator(Orchestrator):
    async def __call__(self, messages, settings=None):
        return TextGenerationResponse(lambda: "ok", use_async_generator=False)


@skip(
    "FastAPI imports produce TypeError: Cannot create a consistent method resolution order (MRO) for bases object, WebSocketDisconnect"
)
class ChatCompletionEndpointTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.FastAPI = FastAPI
        self.TestClient = TestClient

        # Create a dummy `avalan.server` package to avoid executing its __init__
        server_pkg = ModuleType("avalan.server")
        server_pkg.__path__ = [str(Path("src/avalan/server").resolve())]
        sys.modules["avalan.server"] = server_pkg

        # Import router
        self.chat = importlib.import_module("avalan.server.routers.chat")

    def tearDown(self):
        sys.modules.pop("avalan.server.routers.chat", None)
        sys.modules.pop("avalan.server", None)

    async def test_non_streaming_completion(self):
        app = self.FastAPI()
        app.state.orchestrator = AsyncMock(spec=DummyOrchestrator)
        app.state.orchestrator.__call__.side_effect = TextGenerationResponse(
            lambda: "ok", use_async_generator=False
        )
        app.include_router(self.chat.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
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

        class StreamingOrchestrator(Orchestrator):
            async def __call__(self, messages, settings=None):
                return TextGenerationResponse(
                    output_fn, use_async_generator=True
                )

        app = self.FastAPI()
        app.state.orchestrator = AsyncMock(spec=StreamingOrchestrator)
        app.include_router(self.chat.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            self.assertEqual(resp.status_code, 200)
            chunks = list(resp.iter_lines())
        self.assertEqual(chunks[0], "data: [DONE]")
        self.assertEqual(chunks[1], "")
