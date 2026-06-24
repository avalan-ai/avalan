import importlib
import sys
from collections.abc import AsyncIterator
from json import loads
from logging import getLogger
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import MessageContentFile, MessageContentText
from avalan.event import Event, EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.model import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.container_policy import RemoteContainerRequestPolicy


async def _canonical_answer_gen(
    *deltas: str,
) -> AsyncIterator[CanonicalStreamItem]:
    sequence = 0
    yield CanonicalStreamItem(
        stream_session_id="responses-test-stream",
        run_id="responses-test-run",
        turn_id="responses-test-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )
    sequence += 1
    for delta in deltas:
        yield CanonicalStreamItem(
            stream_session_id="responses-test-stream",
            run_id="responses-test-run",
            turn_id="responses-test-turn",
            sequence=sequence,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=delta,
        )
        sequence += 1
    if deltas:
        yield CanonicalStreamItem(
            stream_session_id="responses-test-stream",
            run_id="responses-test-run",
            turn_id="responses-test-turn",
            sequence=sequence,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        sequence += 1
    yield CanonicalStreamItem(
        stream_session_id="responses-test-stream",
        run_id="responses-test-run",
        turn_id="responses-test-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_COMPLETED,
        channel=StreamChannel.CONTROL,
        usage=cast(Any, {"input_tokens": 0, "output_tokens": len(deltas)}),
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )
    sequence += 1
    yield CanonicalStreamItem(
        stream_session_id="responses-test-stream",
        run_id="responses-test-run",
        turn_id="responses-test-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_CLOSED,
        channel=StreamChannel.CONTROL,
    )


class StreamingOrchestrator(Orchestrator):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        pass

    async def __call__(self, messages, settings=None):
        return TextGenerationResponse(
            lambda: _canonical_answer_gen("a", "b"),
            logger=getLogger(),
            use_async_generator=True,
            provider_family="transformers",
        )


class SimpleOrchestrator(Orchestrator):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        self.synced = False
        self._model_ids = {"server-model"}
        self.last_messages = None
        self.last_settings = None

    async def __call__(self, messages, settings=None):
        self.last_messages = messages
        self.last_settings = settings

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


class EventfulServerOrchestrator(Orchestrator):
    def __init__(
        self, *, streaming: bool = False, collect_stats: bool = False
    ) -> None:
        self._event_manager = EventManager(
            mode=EventManagerMode.SERVER,
            collect_stats=collect_stats,
        )
        self._streaming = streaming

    async def __call__(self, messages, settings=None):  # type: ignore[no-untyped-def]
        _ = messages, settings
        for _ in range(10):
            await self.event_manager.trigger(Event(type=EventType.START))

        if self._streaming:
            return TextGenerationResponse(
                lambda: _canonical_answer_gen("bounded"),
                logger=getLogger(),
                use_async_generator=True,
                provider_family="transformers",
            )

        def output_fn(**_):
            return "bounded"

        return TextGenerationResponse(
            output_fn,
            logger=getLogger(),
            use_async_generator=False,
        )

    async def sync_messages(self):  # type: ignore[override]
        return None


class ResponsesEndpointTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        sys.modules.pop("avalan.server.routers.responses", None)
        sys.modules.pop("avalan.server.routers", None)
        sys.modules.pop("avalan.server", None)

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
        self.assertEqual(len(events), 1)
        first_data = loads(lines[events[0] + 1][6:])
        self.assertEqual(first_data["delta"], "ab")

        self.assertIn("event: response.completed", lines)
        self.assertEqual(lines[-1], "")

    async def test_non_streaming_response(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "input": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        resp = client.post("/responses", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["model"], "server-model")
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

    async def test_response_endpoint_rejects_remote_runtime_authority(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        resp = client.post(
            "/responses",
            json={
                "input": "hi",
                "container": {
                    "profiles": {
                        "unsafe": {
                            "image": "registry.example/untrusted:latest"
                        }
                    }
                },
            },
        )

        self.assertEqual(resp.status_code, 422)
        self.assertIn("runtime authority", resp.text)
        self.assertIsNone(orchestrator.last_messages)

    async def test_response_endpoint_rejects_unexposed_container_profile(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        resp = client.post(
            "/responses",
            json={
                "input": "hi",
                "container": {"profile": "workspace-readonly"},
            },
        )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("is not exposed", resp.text)
        self.assertIsNone(orchestrator.last_messages)

    async def test_response_endpoint_allows_exposed_container_profile(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.state.remote_container_policy = RemoteContainerRequestPolicy(
            exposed_profiles=("workspace-readonly",)
        )
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        resp = client.post(
            "/responses",
            json={
                "input": "hi",
                "container": {"profile": "workspace-readonly"},
            },
        )

        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(orchestrator.last_messages)

    async def test_repeated_requests_without_ui_listener_do_not_retain_events(
        self,
    ):
        app = self.FastAPI()
        orchestrator = EventfulServerOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "input": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        for _ in range(3):
            resp = client.post("/responses", json=payload)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(
                resp.json()["output"][0]["content"][0]["text"],
                "bounded",
            )

        self.assertEqual(orchestrator.event_manager.history, [])
        self.assertEqual(orchestrator.event_manager._history_bytes, 0)
        self.assertEqual(orchestrator.event_manager._delivery_queue.qsize(), 0)
        self.assertEqual(orchestrator.event_manager.stats.published, 0)
        self.assertEqual(orchestrator.event_manager.stats.queue_depth, 0)
        self.assertEqual(orchestrator.event_manager.stats.dropped, 0)

    async def test_server_event_stats_can_be_enabled_explicitly(self):
        app = self.FastAPI()
        orchestrator = EventfulServerOrchestrator(collect_stats=True)
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        resp = client.post(
            "/responses",
            json={
                "input": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(orchestrator.event_manager.history, [])
        self.assertEqual(orchestrator.event_manager.stats.published, 10)
        self.assertEqual(orchestrator.event_manager.stats.queue_depth, 0)
        self.assertEqual(orchestrator.event_manager.stats.dropped, 0)

    async def test_repeated_streaming_requests_do_not_retain_events(self):
        app = self.FastAPI()
        orchestrator = EventfulServerOrchestrator(streaming=True)
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "input": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        for _ in range(3):
            with client.stream("POST", "/responses", json=payload) as resp:
                self.assertEqual(resp.status_code, 200)
                lines = list(resp.iter_lines())
            self.assertIn("event: response.output_text.delta", lines)
            self.assertIn("event: response.completed", lines)

        self.assertEqual(orchestrator.event_manager.history, [])
        self.assertEqual(orchestrator.event_manager._history_bytes, 0)
        self.assertEqual(orchestrator.event_manager._delivery_queue.qsize(), 0)
        self.assertEqual(orchestrator.event_manager.stats.published, 0)
        self.assertEqual(orchestrator.event_manager.stats.queue_depth, 0)
        self.assertEqual(orchestrator.event_manager.stats.dropped, 0)

    async def test_non_streaming_response_accepts_string_input(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        resp = client.post(
            "/responses",
            json={"input": "Find the claim", "stream": False},
        )

        self.assertEqual(resp.status_code, 200)
        assert orchestrator.last_messages is not None
        self.assertEqual(len(orchestrator.last_messages), 1)
        message = orchestrator.last_messages[0]
        self.assertEqual(message.role.value, "user")
        self.assertEqual(
            message.content,
            MessageContentText(type="text", text="Find the claim"),
        )

    async def test_non_streaming_response_accepts_input_text_blocks(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Find the claim"},
                        {
                            "type": "input_file",
                            "file_id": "file-1",
                            "filename": "report.pdf",
                        },
                    ],
                }
            ],
            "stream": False,
        }
        resp = client.post("/responses", json=payload)

        self.assertEqual(resp.status_code, 200)
        assert orchestrator.last_messages is not None
        message = orchestrator.last_messages[0]
        assert isinstance(message.content, list)
        self.assertEqual(
            message.content,
            [
                MessageContentText(type="text", text="Find the claim"),
                MessageContentFile(
                    type="file",
                    file={"file_id": "file-1", "filename": "report.pdf"},
                ),
            ],
        )

    async def test_non_streaming_response_accepts_text_format(self):
        app = self.FastAPI()
        orchestrator = SimpleOrchestrator()
        app.state.orchestrator = orchestrator
        app.include_router(self.responses.router)

        client = self.TestClient(app)
        payload = {
            "input": [{"role": "user", "content": "hi"}],
            "text": {
                "format": {"type": "json_object"},
                "stop": "DONE",
            },
            "stream": False,
        }
        resp = client.post("/responses", json=payload)

        self.assertEqual(resp.status_code, 200)
        assert orchestrator.last_settings is not None
        self.assertEqual(
            orchestrator.last_settings.response_format,
            {"type": "json_object"},
        )
        self.assertEqual(orchestrator.last_settings.stop_strings, "DONE")
