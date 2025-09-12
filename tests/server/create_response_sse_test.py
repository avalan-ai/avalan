import importlib
import sys
from json import loads
from logging import getLogger
from pathlib import Path
from types import ModuleType
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    MessageRole,
    ReasoningToken,
    ToolCall,
    ToolCallToken,
)
from avalan.event import Event, EventType
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

        tokens = [ReasoningToken("r"), ToolCallToken(token="t"), "a"]

        async def gen():
            for token in tokens:
                yield token

        class DummyResponse:
            def __init__(self, items) -> None:  # type: ignore[no-untyped-def]
                self._items = items
                self.input_token_count = 0
                self.output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in self._items:
                        yield item

                return gen()

        response = DummyResponse(tokens)

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
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
            "done",
        ]

        self.assertEqual(events, expected)
        self.assertIn(
            '"delta":"r"',
            data_lines[events.index("response.reasoning_text.delta")],
        )
        self.assertIn(
            '"delta":"t"',
            data_lines[events.index("response.custom_tool_call_input.delta")],
        )
        self.assertIn(
            '"delta":"a"',
            data_lines[events.index("response.output_text.delta")],
        )
        content_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.added"
        ]
        self.assertIn(
            '"part":{"type":"reasoning_text"}',
            data_lines[content_indices[0]],
        )
        self.assertIn(
            '"part":{"type":"input_text"}',
            data_lines[content_indices[1]],
        )
        self.assertIn(
            '"part":{"type":"output_text"}',
            data_lines[content_indices[2]],
        )
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_emits_done_events_for_multiple_groups(
        self,
    ) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        tokens = [
            ReasoningToken("r1"),
            ReasoningToken("r2"),
            ToolCallToken(token="t1"),
            ToolCallToken(token="t2"),
            "a1",
            "a2",
            ReasoningToken("r3"),
            ReasoningToken("r4"),
            ToolCallToken(token="t3"),
            ToolCallToken(token="t4"),
            "a3",
            "a4",
        ]

        class DummyResponse:
            def __init__(self, items) -> None:  # type: ignore[no-untyped-def]
                self._items = items
                self.input_token_count = 0
                self.output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in self._items:
                        yield item

                return gen()

        response = DummyResponse(tokens)

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
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
            "done",
        ]

        self.assertEqual(events, expected)
        reasoning_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.reasoning_text.delta"
        ]
        self.assertIn('"delta":"r1"', data_lines[reasoning_indices[0]])
        self.assertIn('"delta":"r2"', data_lines[reasoning_indices[1]])
        tool_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.custom_tool_call_input.delta"
        ]
        self.assertIn('"delta":"t1"', data_lines[tool_indices[0]])
        self.assertIn('"delta":"t2"', data_lines[tool_indices[1]])
        answer_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.output_text.delta"
        ]
        self.assertIn('"delta":"a1"', data_lines[answer_indices[0]])
        self.assertIn('"delta":"a2"', data_lines[answer_indices[1]])
        self.assertIn('"delta":"r3"', data_lines[reasoning_indices[2]])
        self.assertIn('"delta":"r4"', data_lines[reasoning_indices[3]])
        self.assertIn('"delta":"t3"', data_lines[tool_indices[2]])
        self.assertIn('"delta":"t4"', data_lines[tool_indices[3]])
        self.assertIn('"delta":"a3"', data_lines[answer_indices[2]])
        self.assertIn('"delta":"a4"', data_lines[answer_indices[3]])
        content_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.added"
        ]
        self.assertIn(
            '"part":{"type":"reasoning_text"}',
            data_lines[content_indices[0]],
        )
        self.assertIn(
            '"part":{"type":"input_text"}',
            data_lines[content_indices[1]],
        )
        self.assertIn(
            '"part":{"type":"output_text"}',
            data_lines[content_indices[2]],
        )
        self.assertIn(
            '"part":{"type":"reasoning_text"}',
            data_lines[content_indices[3]],
        )
        self.assertIn(
            '"part":{"type":"input_text"}',
            data_lines[content_indices[4]],
        )
        self.assertIn(
            '"part":{"type":"output_text"}',
            data_lines[content_indices[5]],
        )
        orchestrator.sync_messages.assert_awaited_once()

    async def test_custom_tool_call_call_wraps_items_with_id(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        call = ToolCall(id="c1", name="t", arguments={})
        tokens = [
            "a",
            Event(type=EventType.TOOL_PROCESS, payload=[call]),
            ToolCallToken(token="t"),
            "b",
        ]

        class DummyResponse:
            def __init__(self, items) -> None:  # type: ignore[no-untyped-def]
                self._items = items
                self.input_token_count = 0
                self.output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in self._items:
                        yield item

                return gen()

        response = DummyResponse(tokens)

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

        expected_with_call = [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.custom_tool_call_input.call",
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
            "done",
        ]
        expected_without_call = [
            e
            for e in expected_with_call
            if e != "response.custom_tool_call_input.call"
        ]

        self.assertIn(events, (expected_with_call, expected_without_call))

        func_delta_index = events.index(
            "response.function_call_arguments.delta"
        )
        data = loads(data_lines[func_delta_index][6:])
        self.assertEqual(data["id"], "c1")
        delta_obj = loads(data["delta"])
        self.assertEqual(delta_obj["name"], "t")

        output_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.output_item.added"
        ]
        self.assertIn('"id":"c1"', data_lines[output_indices[1]])

        content_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.added"
        ]
        self.assertIn('"id":"c1"', data_lines[content_indices[1]])

        content_done_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.done"
        ]
        self.assertIn('"id":"c1"', data_lines[content_done_indices[1]])

        output_done_indices = [
            i for i, e in enumerate(events) if e == "response.output_item.done"
        ]
        self.assertIn('"id":"c1"', data_lines[output_done_indices[1]])

        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_ignores_events(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        tokens = [Event(type=EventType.START), "a"]

        class DummyResponse:
            def __init__(self, items) -> None:  # type: ignore[no-untyped-def]
                self._items = items
                self.input_token_count = 0
                self.output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in self._items:
                        yield item

                return gen()

        response = DummyResponse(tokens)

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

        expected = [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
            "done",
        ]

        self.assertEqual(events, expected)
        delta_index = events.index("response.output_text.delta")
        self.assertIn('"delta":"a"', blocks[delta_index])
        orchestrator.sync_messages.assert_awaited_once()
