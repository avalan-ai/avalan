import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from json import loads
from logging import getLogger
from pathlib import Path
from time import perf_counter
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    MessageRole,
    ReasoningToken,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamTerminalOutcome,
    StreamValidationError,
    project_canonical_stream_item,
)
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

        reasoning_data = loads(
            data_lines[events.index("response.reasoning_text.delta")][6:]
        )
        self.assertEqual(reasoning_data["delta"], "r")

        tool_data = loads(
            data_lines[events.index("response.custom_tool_call_input.delta")][
                6:
            ]
        )
        self.assertEqual(tool_data["delta"], "t")

        answer_data = loads(
            data_lines[events.index("response.output_text.delta")][6:]
        )
        self.assertEqual(answer_data["delta"], "a")

        content_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.added"
        ]
        reasoning_part = loads(data_lines[content_indices[0]][6:])
        self.assertEqual(reasoning_part["part"], {"type": "reasoning_text"})
        tool_part = loads(data_lines[content_indices[1]][6:])
        self.assertEqual(tool_part["part"], {"type": "input_text"})
        answer_part = loads(data_lines[content_indices[2]][6:])
        self.assertEqual(answer_part["part"], {"type": "output_text"})

        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_uses_response_projections(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        class ProjectionResponse:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self.close_count = 0
                self.consumer_kwargs: dict[str, str] | None = None

            def __aiter__(self) -> AsyncIterator[str]:
                raise AssertionError("raw iterator used")

            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                self.consumer_kwargs = {
                    "stream_session_id": stream_session_id,
                    "run_id": run_id,
                    "turn_id": turn_id,
                }

                async def gen() -> AsyncIterator[object]:
                    for item in (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="projected",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    ):
                        yield project_canonical_stream_item(item)

                return gen()

            async def aclose(self) -> None:
                self.close_count += 1

        response = ProjectionResponse()
        response_id = str(uuid4())

        async def orchestrate_stub(request, logger, orch):
            return response, response_id, 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks = [
            chunk.decode() if isinstance(chunk, bytes) else chunk
            async for chunk in streaming_resp.body_iterator
        ]
        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]
        answer_data = loads(
            data_lines[events.index("response.output_text.delta")][6:]
        )

        self.assertEqual(answer_data["delta"], "projected")
        self.assertIn("response.completed", events)
        self.assertEqual(events[-1], "done")
        self.assertEqual(
            response.consumer_kwargs,
            {
                "stream_session_id": "responses-sse-stream",
                "run_id": response_id,
                "turn_id": "responses-sse-turn",
            },
        )
        self.assertEqual(response.close_count, 1)
        orchestrator.sync_messages.assert_awaited_once()

    def test_response_sse_serialization_per_item_overhead_within_budget(
        self,
    ) -> None:
        budget = StreamPerformanceBudget()
        events = [
            self.responses._ResponsesSSEEvent(
                event="response.output_text.delta",
                data={
                    "type": "response.output_text.delta",
                    "delta": "x",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": sequence,
                },
            )
            for sequence in range(1000)
        ]

        started = perf_counter()
        messages = [event.message() for event in events]
        elapsed_us = (perf_counter() - started) * 1_000_000

        self.assertEqual(len(messages), len(events))
        self.assertLessEqual(
            elapsed_us / len(events),
            budget.per_item_overhead_us,
        )

    def test_response_sse_event_rejects_incompatible_coalesce(self) -> None:
        left = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "a",
                "output_index": 0,
                "content_index": 0,
            },
        )
        right = self.responses._ResponsesSSEEvent(
            event="response.reasoning_text.delta",
            data={
                "type": "response.reasoning_text.delta",
                "delta": "b",
                "output_index": 0,
                "content_index": 0,
            },
        )

        self.assertFalse(left.can_coalesce(right))
        with self.assertRaises(AssertionError):
            left.coalesce(right)

    async def test_streaming_response_closes_source_on_completion(
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

        class Stream:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self.close_count = 0
                self.cancel_count = 0
                self._items = iter(["answer"])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        stream = Stream()

        async def orchestrate_stub(request, logger, orch):
            return stream, uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks = [
            chunk.decode() if isinstance(chunk, bytes) else chunk
            async for chunk in streaming_resp.body_iterator
        ]

        self.assertIn("response.completed", "".join(chunks))
        self.assertEqual(stream.close_count, 1)
        self.assertEqual(stream.cancel_count, 0)
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_response_disconnect_closes_source_before_pull(
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

        class Stream:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self.close_count = 0
                self.cancel_count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                return "late"

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        stream = Stream()

        async def orchestrate_stub(request, logger, orch):
            return stream, uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        iterator = streaming_resp.body_iterator
        first = await anext(iterator)
        await iterator.aclose()

        self.assertIn("response.created", first)
        self.assertEqual(stream.close_count, 1)
        self.assertEqual(stream.cancel_count, 0)
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_response_cancellation_cancels_source(
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

        class Stream:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self.close_count = 0
                self.cancel_count = 0
                self.started = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.started.set()
                await asyncio.Event().wait()
                raise StopAsyncIteration

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        stream = Stream()

        async def orchestrate_stub(request, logger, orch):
            return stream, uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        iterator = streaming_resp.body_iterator
        first = await anext(iterator)
        self.assertIn("response.created", first)
        task = asyncio.create_task(anext(iterator))
        await stream.started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(stream.cancel_count, 1)
        self.assertEqual(stream.close_count, 1)
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_response_long_stream_flushes_bounded_deltas(
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

        class Stream:
            input_token_count = 0
            output_token_count = 0

            def __init__(self, count: int) -> None:
                self._remaining = count
                self.read_count = 0
                self.close_count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._remaining <= 0:
                    raise StopAsyncIteration
                self._remaining -= 1
                self.read_count += 1
                return "x"

            async def aclose(self) -> None:
                self.close_count += 1

        max_delta_chars = self.responses._MAX_COALESCED_DELTA_CHARS
        stream = Stream(max_delta_chars + 1)

        async def orchestrate_stub(request, logger, orch):
            return stream, uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        iterator = streaming_resp.body_iterator
        chunks = [
            await anext(iterator),
            await anext(iterator),
            await anext(iterator),
            await anext(iterator),
        ]
        first_delta = chunks[-1]
        payload = loads(first_delta.split("\n")[1][6:])

        self.assertEqual(payload["type"], "response.output_text.delta")
        self.assertEqual(len(payload["delta"]), max_delta_chars)
        self.assertEqual(stream.read_count, max_delta_chars + 1)

        remaining = [chunk async for chunk in iterator]
        blocks = [
            block
            for block in "".join([*chunks, *remaining]).strip().split("\n\n")
            if block
        ]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]
        delta_indices = [
            i
            for i, event in enumerate(events)
            if event == first_delta.split("\n")[0].split(": ")[1]
        ]

        self.assertEqual(len(delta_indices), 2)
        self.assertEqual(
            [len(loads(data_lines[i][6:])["delta"]) for i in delta_indices],
            [max_delta_chars, 1],
        )
        self.assertEqual(stream.close_count, 1)
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_emits_canonical_items(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="plan",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta='{"x"',
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]

        self.assertIn("response.reasoning_text.delta", events)
        self.assertIn("response.custom_tool_call_input.delta", events)
        self.assertIn("response.output_text.delta", events)
        reasoning_data = loads(
            data_lines[events.index("response.reasoning_text.delta")][6:]
        )
        tool_data = loads(
            data_lines[events.index("response.custom_tool_call_input.delta")][
                6:
            ]
        )
        answer_data = loads(
            data_lines[events.index("response.output_text.delta")][6:]
        )
        self.assertEqual(reasoning_data["delta"], "plan")
        self.assertEqual(tool_data["delta"], '{"x"')
        self.assertEqual(tool_data["id"], "call-1")
        self.assertEqual(answer_data["delta"], "answer")
        self.assertNotIn("stream.started", "".join(chunks))
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_keeps_same_tool_call_group_open(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="a",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="b",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="done",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]
        data = [loads(line[6:]) for line in data_lines]

        tool_delta_indices = [
            i
            for i, event in enumerate(events)
            if event == "response.custom_tool_call_input.delta"
        ]
        tool_added = [
            item
            for item in data
            if item.get("type") == "response.output_item.added"
            and item["item"].get("id") == "call-1"
        ]
        tool_parts = [
            item
            for item in data
            if item.get("type") == "response.content_part.added"
            and item["part"].get("id") == "call-1"
        ]
        tool_done = [
            item
            for item in data
            if item.get("type") == "response.custom_tool_call_input.done"
        ]

        self.assertEqual(len(tool_delta_indices), 1)
        self.assertEqual(data[tool_delta_indices[0]]["delta"], "ab")
        self.assertEqual(data[tool_delta_indices[0]]["id"], "call-1")
        self.assertEqual(len(tool_added), 1)
        self.assertEqual(len(tool_parts), 1)
        self.assertEqual(
            tool_done,
            [
                {
                    "type": "response.custom_tool_call_input.done",
                    "output_index": 0,
                    "content_index": 0,
                    "id": "call-1",
                }
            ],
        )
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_rejects_canonical_content_after_terminal(
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

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        iterator = streaming_resp.body_iterator
        first = await anext(iterator)
        self.assertIn("response.created", first)
        with self.assertRaises(StreamValidationError):
            async for _chunk in iterator:
                pass
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_rejects_projected_content_after_terminal(
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

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield project_canonical_stream_item(item)

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        iterator = streaming_resp.body_iterator
        first = await anext(iterator)
        self.assertIn("response.created", first)
        with self.assertRaises(StreamValidationError):
            async for _chunk in iterator:
                pass
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_emits_usage_before_completion(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="ok",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={"total_tokens": 2},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]

        self.assertLess(
            events.index("response.usage.completed"),
            events.index("response.completed"),
        )
        usage_data = loads(
            data_lines[events.index("response.usage.completed")][6:]
        )
        self.assertEqual(usage_data["usage"], {"total_tokens": 2})
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_preserves_canonical_sequence_numbers(
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

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=10,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=20,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=30,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=40,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={"total_tokens": 3},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=50,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]

        output_data = loads(
            data_lines[events.index("response.output_text.delta")][6:]
        )
        usage_data = loads(
            data_lines[events.index("response.usage.completed")][6:]
        )
        completed_data = loads(
            data_lines[events.index("response.completed")][6:]
        )

        self.assertEqual(output_data["sequence_number"], 20)
        self.assertEqual(usage_data["sequence_number"], 40)
        self.assertEqual(completed_data["sequence_number"], 50)
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_preserves_cancelled_terminal(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

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

        self.assertIn("response.cancelled", events)
        self.assertNotIn("response.completed", events)
        self.assertEqual(events[-1], "done")
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_preserves_failed_terminal_error(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={
                    "error_type": "RuntimeError",
                    "message": "provider failed",
                },
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]
        failed_index = events.index("response.failed")
        failed_data = loads(data_lines[failed_index][6:])

        self.assertIn("response.output_text.delta", events)
        self.assertIn("response.output_text.done", events)
        self.assertNotIn("response.completed", events)
        self.assertEqual(events[-1], "done")
        self.assertEqual(failed_data["sequence_number"], 2)
        self.assertEqual(
            failed_data["error"],
            {"error_type": "RuntimeError", "message": "provider failed"},
        )
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_preserves_projected_failed_terminal_error(
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

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={
                    "error_type": "RuntimeError",
                    "message": "provider failed",
                },
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield project_canonical_stream_item(item)

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        chunks: list[str] = []
        async for chunk in streaming_resp.body_iterator:
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]
        data_lines = [block.split("\n")[1] for block in blocks]
        failed_index = events.index("response.failed")
        failed_data = loads(data_lines[failed_index][6:])

        self.assertIn("response.output_text.delta", events)
        self.assertIn("response.output_text.done", events)
        self.assertNotIn("response.completed", events)
        self.assertEqual(events[-1], "done")
        self.assertEqual(failed_data["sequence_number"], 2)
        self.assertEqual(
            failed_data["error"],
            {"error_type": "RuntimeError", "message": "provider failed"},
        )
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_rejects_canonical_stream_missing_terminal(
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

        items = [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
        ]

        class DummyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self):  # type: ignore[override]
                async def gen():
                    for item in items:
                        yield item

                return gen()

        async def orchestrate_stub(request, logger, orch):
            return DummyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        with self.assertRaises(StreamValidationError):
            async for _chunk in streaming_resp.body_iterator:
                pass
        orchestrator.sync_messages.assert_not_awaited()

    async def test_streaming_rejects_mixed_stream_surfaces(self) -> None:
        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        canonical_item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        projection = project_canonical_stream_item(canonical_item)
        cases = (
            (
                ("legacy", canonical_item),
                "canonical stream item after legacy stream item",
            ),
            (
                ("legacy", projection),
                "canonical stream item after legacy stream item",
            ),
            (
                (canonical_item, "legacy"),
                "legacy stream item after canonical stream item",
            ),
            (
                (projection, "legacy"),
                "legacy stream item after canonical stream item",
            ),
        )

        for items, message in cases:
            with self.subTest(message=message, first=type(items[0]).__name__):
                logger = getLogger()
                orchestrator = Orchestrator.__new__(Orchestrator)
                orchestrator.sync_messages = AsyncMock()

                class DummyResponse:
                    input_token_count = 0
                    output_token_count = 0

                    def __aiter__(self):  # type: ignore[override]
                        async def gen():
                            for item in items:
                                yield item

                        return gen()

                async def orchestrate_stub(request, logger, orch):
                    return DummyResponse(), uuid4(), 0

                self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

                streaming_resp = await self.responses.create_response(
                    request, logger, orchestrator
                )

                with self.assertRaisesRegex(StreamValidationError, message):
                    async for _chunk in streaming_resp.body_iterator:
                        pass
                orchestrator.sync_messages.assert_not_awaited()

    def test_canonical_control_items_do_not_emit_response_sse(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )

        projection = project_canonical_stream_item(item)

        self.assertEqual(self.responses._token_to_sse(projection, 0), [])
        self.assertIsNone(self.responses._new_state(projection))
        self.assertIsNone(self.responses._new_state(None))

    def test_projection_items_emit_response_sse(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        projection = project_canonical_stream_item(item)

        events = self.responses._token_to_sse(projection, 7)
        payload = loads(events[0].split("\n")[1][6:])

        self.assertIs(
            self.responses._new_state(projection),
            self.responses.ResponseState.ANSWERING,
        )
        self.assertEqual(payload["type"], "response.output_text.delta")
        self.assertEqual(payload["delta"], "answer")
        self.assertEqual(payload["sequence_number"], 7)

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
        reasoning_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.reasoning_text.delta"
        ]
        self.assertEqual(
            [loads(data_lines[i][6:])["delta"] for i in reasoning_indices],
            ["r1r2", "r3r4"],
        )

        tool_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.custom_tool_call_input.delta"
        ]
        self.assertEqual(
            [loads(data_lines[i][6:])["delta"] for i in tool_indices],
            ["t1t2", "t3t4"],
        )

        answer_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.output_text.delta"
        ]
        self.assertEqual(
            [loads(data_lines[i][6:])["delta"] for i in answer_indices],
            ["a1a2", "a3a4"],
        )

        content_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.added"
        ]
        expected_parts = [
            {"type": "reasoning_text"},
            {"type": "input_text"},
            {"type": "output_text"},
            {"type": "reasoning_text"},
            {"type": "input_text"},
            {"type": "output_text"},
        ]
        actual_parts = [
            loads(data_lines[i][6:])["part"] for i in content_indices
        ]
        self.assertEqual(actual_parts, expected_parts)
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_includes_tool_call_token_with_call(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        call = ToolCall(id="call-1", name="adder", arguments={"x": 1})
        error = ToolCallError(
            id="call-1",
            name="adder",
            arguments={"x": 1},
            call=call,
            error=RuntimeError("fail"),
            message="fail",
        )

        tokens = [
            Event(type=EventType.TOOL_PROCESS, payload=[call]),
            ToolCallToken(token="payload", call=call),
            Event(type=EventType.TOOL_RESULT, payload={"result": error}),
            "final",
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

        function_indices = [
            i
            for i, event in enumerate(events)
            if event == "response.function_call_arguments.delta"
        ]
        self.assertEqual(len(function_indices), 3)

        first_data = loads(data_lines[function_indices[0]][6:])
        self.assertEqual(first_data["id"], "call-1")
        first_delta = loads(first_data["delta"])
        self.assertEqual(first_delta["name"], "adder")

        second_data = loads(data_lines[function_indices[1]][6:])
        self.assertEqual(second_data["id"], "call-1")
        second_delta = loads(second_data["delta"])
        self.assertEqual(second_delta["arguments"], {"x": 1})

        third_data = loads(data_lines[function_indices[2]][6:])
        self.assertEqual(
            third_data["error"],
            {"type": "RuntimeError", "message": "Tool call failed."},
        )
        third_delta = loads(third_data["delta"])
        self.assertEqual(
            third_delta["error"],
            {"type": "RuntimeError", "message": "Tool call failed."},
        )
        self.assertNotIn('"message":"fail"', data_lines[function_indices[2]])
        self.assertIn("response.function_call_arguments.done", events)
        self.assertNotIn("response.custom_tool_call_input.done", events)

        output_index = events.index("response.output_text.delta")
        output_data = loads(data_lines[output_index][6:])
        self.assertEqual(output_data["delta"], "final")

        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_preserves_falsy_tool_result(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        call = ToolCall(id="call-1", name="counter", arguments={})
        result = ToolCallResult(
            id="call-1",
            call=call,
            name="counter",
            arguments={},
            result=0,
        )
        tokens = [
            Event(type=EventType.TOOL_PROCESS, payload=[call]),
            Event(type=EventType.TOOL_RESULT, payload={"result": result}),
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

        blocks = [b for b in "".join(chunks).strip().split("\n\n") if b]
        result_blocks = [
            block
            for block in blocks
            if block.startswith(
                "event: response.function_call_arguments.delta"
            )
        ]

        self.assertEqual(len(result_blocks), 2)
        data = loads(result_blocks[1].split("\n")[1][6:])
        self.assertEqual(data["id"], "call-1")
        self.assertEqual(data["result"], "0")
        self.assertEqual(loads(data["delta"])["result"], "0")
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_includes_tool_diagnostic_event(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        call = ToolCall(id="call-d", name="missing", arguments={})
        diagnostic = ToolCallDiagnostic(
            id="diag-d",
            call_id=call.id,
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        tokens = [
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={"diagnostics": ["bad"]},
            ),
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={"call": call, "diagnostic": diagnostic},
            ),
            "final",
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

        self.assertIn("response.tool_call_diagnostic.delta", events)
        diagnostic_index = events.index("response.tool_call_diagnostic.delta")
        diagnostic_data = loads(data_lines[diagnostic_index][6:])
        self.assertEqual(diagnostic_data["id"], "call-d")
        self.assertEqual(diagnostic_data["diagnostic"]["code"], "tool.unknown")
        output_index = events.index("response.output_text.delta")
        output_data = loads(data_lines[output_index][6:])
        self.assertEqual(output_data["delta"], "final")
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_serializes_tool_result_temporal_types(
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

        call = ToolCall(id="call-1", name="database.sample", arguments={})
        result = ToolCallResult(
            id="call-1",
            name="database.sample",
            arguments={},
            call=call,
            result=[
                {
                    "id": UUID("019b7589-672b-766d-81c6-1da5efd5f49a"),
                    "check_date": date(2025, 9, 19),
                    "gross_check_amount": Decimal("524.46"),
                }
            ],
        )

        tokens = [
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": result},
            )
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

        function_index = events.index("response.function_call_arguments.delta")
        payload = loads(data_lines[function_index][6:])
        result_payload = loads(payload["result"])

        self.assertEqual(
            result_payload,
            [
                {
                    "id": "019b7589-672b-766d-81c6-1da5efd5f49a",
                    "check_date": "2025-09-19",
                    "gross_check_amount": "524.46",
                }
            ],
        )

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

        expected = [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
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

        func_delta_index = events.index(
            "response.function_call_arguments.delta"
        )
        data = loads(data_lines[func_delta_index][6:])
        self.assertEqual(data["id"], "c1")
        delta_obj = loads(data["delta"])
        self.assertEqual(delta_obj["name"], "t")

        custom_delta_index = events.index(
            "response.custom_tool_call_input.delta"
        )
        custom_data = loads(data_lines[custom_delta_index][6:])
        self.assertEqual(custom_data["id"], "c1")
        self.assertEqual(custom_data["delta"], "t")

        output_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.output_item.added"
        ]
        output_data = loads(data_lines[output_indices[1]][6:])
        self.assertEqual(output_data["item"]["id"], "c1")
        self.assertEqual(output_data["item"]["type"], "function_call")

        custom_output_data = loads(data_lines[output_indices[2]][6:])
        self.assertEqual(
            custom_output_data["item"],
            {"type": "custom_tool_call_input", "id": "c1"},
        )

        content_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.content_part.added"
        ]
        content_data = loads(data_lines[content_indices[1]][6:])
        self.assertEqual(content_data["part"]["id"], "c1")

        output_done_indices = [
            i for i, e in enumerate(events) if e == "response.output_item.done"
        ]
        output_done_data = loads(data_lines[output_done_indices[1]][6:])
        self.assertEqual(output_done_data["item"]["id"], "c1")

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
        delta_data = loads(blocks[delta_index].split("\n")[1][6:])
        self.assertEqual(delta_data["delta"], "a")
        orchestrator.sync_messages.assert_awaited_once()
