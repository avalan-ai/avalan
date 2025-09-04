import importlib
import sys
from logging import getLogger
from pathlib import Path
from types import ModuleType
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import MessageRole, ReasoningToken, ToolCallToken
from avalan.model import TextGenerationResponse
from avalan.server.entities import ChatMessage, ResponsesRequest


class CreateResponseSSEEventsTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:  # type: ignore[override]
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

    def tearDown(self) -> None:  # type: ignore[override]
        sys.modules.pop("avalan.server.routers.responses", None)
        sys.modules.pop("avalan.server", None)

    async def test_streaming_emits_all_events(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        tokens = [ReasoningToken("r"), ToolCallToken("t"), "a"]

        async def gen():
            for token in tokens:
                yield token

        response = TextGenerationResponse(
            gen, logger=logger, use_async_generator=True
        )

        async def orchestrate_stub(request, logger, orch):
            return response, uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        text = "".join(chunks)
        blocks = [b for b in text.strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]

        expected = [
            "response.created",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.completed",
            "done",
        ]

        self.assertEqual(events, expected)
        self.assertIn('"delta":"r"', data_lines[1])
        self.assertIn('"delta":"t"', data_lines[3])
        self.assertIn('"delta":"a"', data_lines[5])
        orchestrator.sync_messages.assert_awaited_once()
