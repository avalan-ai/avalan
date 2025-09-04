import importlib
import sys
from logging import getLogger
from pathlib import Path
from types import ModuleType
from unittest import IsolatedAsyncioTestCase

from avalan.agent.orchestrator import Orchestrator
from avalan.model import TextGenerationResponse


class StreamingOrchestrator(Orchestrator):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        pass

    async def __call__(self, messages, settings=None):
        async def gen():
            yield "a"
            yield "b"

        return TextGenerationResponse(
            gen, logger=getLogger(), use_async_generator=True
        )


class SimpleOrchestrator(Orchestrator):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        self.synced = False

    async def __call__(self, messages, settings=None):
        def output_fn(**_):
            return "c"

        return TextGenerationResponse(
            output_fn,
            logger=getLogger(),
            use_async_generator=False,
            inputs={"input_ids": [[1, 2, 3]]},
        )

    async def sync_messages(self):  # type: ignore[override]
        self.synced = True


class ResponsesEndpointTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.FastAPI = FastAPI
        self.TestClient = TestClient

        server_pkg = ModuleType("avalan.server")
        server_pkg.__path__ = [str(Path("src/avalan/server").resolve())]

        from fastapi import Request

        def _get_logger(request: Request):
            return getLogger()

        def _get_orchestrator(request: Request):
            return request.app.state.orchestrator

        server_pkg.di_get_logger = _get_logger
        server_pkg.di_get_orchestrator = _get_orchestrator
        sys.modules["avalan.server"] = server_pkg
        self.responses = importlib.import_module(
            "avalan.server.routers.responses"
        )

    def tearDown(self):
        sys.modules.pop("avalan.server.routers.responses", None)
        sys.modules.pop("avalan.server", None)

    async def test_streaming_responses(self):
        app = self.FastAPI()
        app.state.orchestrator = StreamingOrchestrator()
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "input": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        with client.stream("POST", "/responses", json=payload) as resp:
            self.assertEqual(resp.status_code, 200)
            lines = list(resp.iter_lines())

        events = [
            i
            for i, line in enumerate(lines)
            if line == "event: response.output_text.delta"
        ]
        self.assertIn(
            'data: {"type":"response.output_text.delta","delta":"a",'
            '"output_index":0,"content_index":0,"sequence_number":0}',
            lines[events[0] + 1],
        )
        self.assertIn(
            'data: {"type":"response.output_text.delta","delta":"b",'
            '"output_index":0,"content_index":0,"sequence_number":1}',
            lines[events[1] + 1],
        )

        self.assertIn("event: response.completed", lines)
        self.assertEqual(lines[-1], "")

    async def test_non_streaming_response(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "input": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        resp = client.post("/responses", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["output"][0]["content"][0]["text"], "c")
        self.assertTrue(orchestrator.synced)
        self.assertEqual(
            body["usage"],
            {
                "input_text_tokens": 3,
                "output_text_tokens": 1,
                "total_tokens": 4,
            },
        )
