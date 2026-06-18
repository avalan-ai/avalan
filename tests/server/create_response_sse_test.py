import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from gc import collect
from json import loads
from logging import getLogger
from pathlib import Path
from time import perf_counter
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from uuid import UUID, uuid4
from weakref import ReferenceType, ref

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    MessageRole,
    ToolCall,
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


def _canonical_answer_stream_items(
    *text_deltas: str,
) -> tuple[CanonicalStreamItem, ...]:
    assert text_deltas
    items = [
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
    ]
    sequence = 1
    for text_delta in text_deltas:
        items.append(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text_delta,
            )
        )
        sequence += 1
    items.extend(
        [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=sequence + 1,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=sequence + 2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
    )
    return tuple(items)


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

        correlation = StreamItemCorrelation(tool_call_id="tool-1")
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
                text_delta="r",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                text_delta="t",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="a",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=8,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=9,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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

        response = DummyResponse(items)

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
            "response.usage.completed",
            "response.completed",
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
        self.assertEqual(
            tool_part["part"], {"type": "input_text", "id": "tool-1"}
        )
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
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=3,
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
        self.assertEqual(events[-1], "response.completed")
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

    async def test_streaming_yields_adapter_close_events(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        class TerminalResponse:
            input_token_count = 0
            output_token_count = 0
            usage = None

            def __init__(self) -> None:
                self.close_count = 0
                self._items = iter(
                    (
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
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        ),
                    )
                )

            def __aiter__(self) -> AsyncIterator[object]:
                return self

            async def __anext__(self) -> object:
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def aclose(self) -> None:
                self.close_count += 1

        responses_module = self.responses

        class ClosingAdapter:
            state = None

            @property
            def active_tool_call_id(self) -> str | None:
                return None

            def switch(self, token: object) -> list[str]:
                return []

            def close(self) -> list[str]:
                return [
                    responses_module._ResponsesSSEEvent(
                        event="response.output_text.done",
                        data={"type": "response.output_text.done"},
                    ).message()
                ]

        response = TerminalResponse()

        async def orchestrate_stub(request, logger, orch):
            return response, uuid4(), 0

        original_adapter = self.responses._ResponsesSSEProjectionAdapter
        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]
        self.responses._ResponsesSSEProjectionAdapter = ClosingAdapter
        try:
            streaming_resp = await self.responses.create_response(
                request, logger, orchestrator
            )
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in streaming_resp.body_iterator
            ]
        finally:
            self.responses._ResponsesSSEProjectionAdapter = original_adapter

        text = "".join(chunks)
        blocks = [block for block in text.strip().split("\n\n") if block]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]

        self.assertIn("response.output_text.done", events)
        self.assertEqual(
            events[-2:],
            ["response.output_text.done", "response.completed"],
        )
        self.assertEqual(response.close_count, 1)
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_flushes_pending_event_at_iterator_end(
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

        class EmptyResponse:
            input_token_count = 0
            output_token_count = 0
            usage = None

            def __init__(self) -> None:
                self.close_count = 0

            def __aiter__(self) -> AsyncIterator[object]:
                return self

            async def __anext__(self) -> object:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.close_count += 1

        response = EmptyResponse()

        def fake_stream_consumer_iterator(*_args, **_kwargs):
            async def gen():
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
                        text_delta="pending",
                    ),
                ):
                    yield project_canonical_stream_item(item)

            return gen()

        async def orchestrate_stub(request, logger, orch):
            return response, uuid4(), 0

        original_orchestrate = self.responses.orchestrate
        original_iterator = self.responses.stream_consumer_iterator
        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]
        self.responses.stream_consumer_iterator = fake_stream_consumer_iterator
        try:
            streaming_resp = await self.responses.create_response(
                request, logger, orchestrator
            )
            iterator = streaming_resp.body_iterator
            chunks: list[str] = []
            for _ in range(4):
                chunk = await anext(iterator)
                chunks.append(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
            await iterator.aclose()
        finally:
            self.responses.orchestrate = original_orchestrate
            self.responses.stream_consumer_iterator = original_iterator

        text = "".join(chunks)
        self.assertIn("response.output_text.delta", text)
        self.assertIn("pending", text)
        self.assertEqual(response.close_count, 1)

    async def test_repeated_response_stream_requests_release_sources(
        self,
    ) -> None:
        logger = getLogger()
        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        closed: list[int] = []
        cancelled: list[int] = []

        class Source:
            input_token_count = 0
            output_token_count = 0

            def __init__(self, index: int) -> None:
                self._items = iter(_canonical_answer_stream_items("chunk"))
                self.index = index

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def cancel(self) -> None:
                cancelled.append(self.index)

            async def aclose(self) -> None:
                closed.append(self.index)

        source_refs: list[ReferenceType[Source]] = []

        class StreamingOrchestrator(Orchestrator):
            model_ids = {"m"}

            def __init__(self) -> None:
                self.sync_count = 0
                self.response_count = 0

            async def __call__(self, messages, settings=None):  # type: ignore[no-untyped-def]
                _ = messages, settings
                source = Source(self.response_count)
                self.response_count += 1
                source_refs.append(ref(source))
                return source

            async def sync_messages(self) -> None:  # type: ignore[override]
                self.sync_count += 1

        orchestrator = StreamingOrchestrator()
        previous_refs = 0
        for _ in range(6):
            streaming_response = await self.responses.create_response(
                request, logger, orchestrator
            )
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in streaming_response.body_iterator
            ]

            self.assertIn("response.output_text.delta", "".join(chunks))
            self.assertIn("event: response.completed", chunks[-1])
            self.assertEqual(len(source_refs), previous_refs + 1)
            previous_refs = len(source_refs)
            del chunks
            del streaming_response

        collect()

        self.assertEqual(closed, list(range(6)))
        self.assertEqual(cancelled, [])
        self.assertEqual(orchestrator.sync_count, 6)
        self.assertEqual(
            [source_ref() for source_ref in source_refs],
            [None for _ in source_refs],
        )

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

    def test_response_sse_event_coalesces_only_adjacent_matching_events(
        self,
    ) -> None:
        left = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "a",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        right = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "b",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 2,
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        gap = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "c",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 4,
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        wrong_channel = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "d",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 2,
            },
            canonical_channel=StreamChannel.REASONING,
        )
        wrong_index = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "e",
                "output_index": 1,
                "content_index": 0,
                "sequence_number": 2,
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        missing_index = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "f",
                "output_index": 0,
                "sequence_number": 2,
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        bool_index = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "g",
                "output_index": True,
                "content_index": 0,
                "sequence_number": 2,
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        left_tool = self.responses._ResponsesSSEEvent(
            event="response.custom_tool_call_input.delta",
            data={
                "type": "response.custom_tool_call_input.delta",
                "delta": "a",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
            },
            correlation_key="call-1",
            canonical_channel=StreamChannel.TOOL_CALL,
        )
        right_tool = self.responses._ResponsesSSEEvent(
            event="response.custom_tool_call_input.delta",
            data={
                "type": "response.custom_tool_call_input.delta",
                "delta": "b",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 2,
            },
            correlation_key="call-2",
            canonical_channel=StreamChannel.TOOL_CALL,
        )
        tool_output = self.responses._ResponsesSSEEvent(
            event="response.tool_execution.output",
            data={
                "type": "response.tool_execution.output",
                "delta": "log",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "data": {"stream": "stderr"},
            },
            correlation_key="call-1",
            canonical_channel=StreamChannel.TOOL_EXECUTION,
        )
        tool_output_next = self.responses._ResponsesSSEEvent(
            event="response.tool_execution.output",
            data={
                "type": "response.tool_execution.output",
                "delta": "line",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 2,
            },
            correlation_key="call-1",
            canonical_channel=StreamChannel.TOOL_EXECUTION,
        )

        self.assertTrue(left.can_coalesce(right))
        coalesced = left.coalesce(right)
        self.assertEqual(coalesced.data["delta"], "ab")
        self.assertEqual(coalesced.data["sequence_number"], 2)
        self.assertFalse(left.can_coalesce(gap))
        self.assertFalse(left.can_coalesce(wrong_channel))
        self.assertFalse(left.can_coalesce(wrong_index))
        self.assertFalse(left.can_coalesce(missing_index))
        self.assertFalse(left.can_coalesce(bool_index))
        self.assertFalse(left_tool.can_coalesce(right_tool))
        self.assertFalse(tool_output.can_coalesce(tool_output_next))

    def test_custom_tool_call_preserves_synthetic_legacy_id(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="legacy-tool-call"),
            text_delta="{}",
        )
        projection = project_canonical_stream_item(item)
        adapter = self.responses._ResponsesSSEProjectionAdapter()

        added_events = adapter.switch(projection)
        delta_events = self.responses._token_to_sse_events(
            projection,
            projection.sequence,
            adapter.active_tool_call_id,
        )

        added_data = loads(added_events[0].split("data: ", maxsplit=1)[1])
        self.assertEqual(added_data["item"]["id"], "legacy-tool-call")
        self.assertEqual(delta_events[0].data["id"], "legacy-tool-call")
        self.assertEqual(
            delta_events[0].correlation_key,
            "legacy-tool-call",
        )

    def test_response_sse_item_state_rejects_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            self.responses._ResponsesSSEItemState(
                output_item_type="invalid",
            )
        with self.assertRaises(AssertionError):
            self.responses._ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="invalid",
            )
        with self.assertRaises(AssertionError):
            self.responses._ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                tool_call_id="",
            )

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
                self._items = iter(_canonical_answer_stream_items("answer"))

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
                self._started = False
                self.read_count = 0
                self.close_count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._started:
                    self._started = True
                    return CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                if self._remaining <= 0:
                    if self._remaining == 0:
                        self._remaining -= 1
                        return CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=self.read_count + 1,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        )
                    if self._remaining == -1:
                        self._remaining -= 1
                        return CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=self.read_count + 2,
                            kind=StreamItemKind.USAGE_COMPLETED,
                            channel=StreamChannel.USAGE,
                            usage={},
                        )
                    if self._remaining == -2:
                        self._remaining -= 1
                        return CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=self.read_count + 3,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        )
                    raise StopAsyncIteration
                self._remaining -= 1
                self.read_count += 1
                return CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=self.read_count,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="x",
                )

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
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta='{"x"',
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=8,
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
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="done",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
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
        self.assertEqual(events[-1], "response.cancelled")
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
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
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
        self.assertEqual(events[-1], "response.failed")
        self.assertEqual(failed_data["sequence_number"], 3)
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
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
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
        self.assertEqual(events[-1], "response.failed")
        self.assertEqual(failed_data["sequence_number"], 3)
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

    async def test_streaming_rejects_empty_stream_missing_terminal(
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

        class EmptyResponse:
            input_token_count = 0
            output_token_count = 0

            def __aiter__(self) -> AsyncIterator[object]:
                return self

            async def __anext__(self) -> object:
                raise StopAsyncIteration

        async def orchestrate_stub(request, logger, orch):
            return EmptyResponse(), uuid4(), 0

        self.responses.orchestrate = orchestrate_stub  # type: ignore[attr-defined]

        streaming_resp = await self.responses.create_response(
            request, logger, orchestrator
        )
        iterator = streaming_resp.body_iterator
        created = await anext(iterator)

        self.assertIn("event: response.created", created)
        with self.assertRaisesRegex(
            StreamValidationError,
            "stream missing terminal outcome",
        ):
            await anext(iterator)
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
                "unsupported stream item for Responses SSE projection",
            ),
            (
                ("legacy", projection),
                "unsupported stream item for Responses SSE projection",
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
        self.assertIsNone(
            self.responses._response_projection_state(projection)
        )
        self.assertIsNone(self.responses._response_projection_state(None))

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
        state = self.responses._response_projection_state(projection)

        self.assertEqual(state.output_item_type, "output_text")
        self.assertEqual(state.content_part_type, "output_text")
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

        first_tool = StreamItemCorrelation(tool_call_id="tool-1")
        second_tool = StreamItemCorrelation(tool_call_id="tool-2")
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
                text_delta="r1r2",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=first_tool,
                text_delta="t1t2",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=first_tool,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=first_tool,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="a1a2",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=second_tool,
                text_delta="t3t4",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=8,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=second_tool,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=9,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=second_tool,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=10,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="a3a4",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=11,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=12,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=13,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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

        response = DummyResponse(items)

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
            "response.usage.completed",
            "response.completed",
        ]

        self.assertEqual(events, expected)
        reasoning_indices = [
            i
            for i, e in enumerate(events)
            if e == "response.reasoning_text.delta"
        ]
        self.assertEqual(
            [loads(data_lines[i][6:])["delta"] for i in reasoning_indices],
            ["r1r2"],
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
            {"type": "input_text", "id": "tool-1"},
            {"type": "output_text"},
            {"type": "input_text", "id": "tool-2"},
            {"type": "output_text"},
        ]
        actual_parts = [
            loads(data_lines[i][6:])["part"] for i in content_indices
        ]
        self.assertEqual(actual_parts, expected_parts)
        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_includes_canonical_tool_error(self) -> None:
        logger = getLogger()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        call = ToolCall(id="call-1", name="adder", arguments={"x": 1})
        tokens = [
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
                correlation=StreamItemCorrelation(tool_call_id=call.id),
                text_delta='{"x":1}',
                data={"name": call.name, "arguments": call.arguments},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
                data={"name": call.name, "arguments": call.arguments},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_EXECUTION_ERROR,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
                data={
                    "error": {
                        "type": "RuntimeError",
                        "message": "Tool call failed.",
                    }
                },
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="final",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=8,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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
        self.assertEqual(len(function_indices), 1)

        first_data = loads(data_lines[function_indices[0]][6:])
        self.assertEqual(first_data["id"], "call-1")
        first_delta = loads(first_data["delta"])
        self.assertEqual(first_delta["name"], "adder")
        self.assertEqual(first_delta["arguments"], {"x": 1})

        error_index = events.index("response.tool_execution.error")
        error_data = loads(data_lines[error_index][6:])
        self.assertEqual(error_data["id"], "call-1")
        self.assertEqual(
            error_data["data"]["error"],
            {"type": "RuntimeError", "message": "Tool call failed."},
        )
        self.assertIn("response.function_call_arguments.done", events)
        self.assertNotIn("response.custom_tool_call_input.done", events)

        output_index = events.index("response.output_text.delta")
        output_data = loads(data_lines[output_index][6:])
        self.assertEqual(output_data["delta"], "final")

        orchestrator.sync_messages.assert_awaited_once()

    async def test_streaming_preserves_consecutive_tool_output_metadata(
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
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="out",
                data={"category": "stdout", "chunk": 1},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="err",
                data={"category": "stderr", "chunk": 2},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="log",
                data={"category": "log", "chunk": 3},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                data={"category": "progress", "progress": 0.5},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                data={"result": "done"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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
        output_payloads = [
            loads(data_lines[index][6:])
            for index, event in enumerate(events)
            if event == "response.tool_execution.output"
        ]
        progress_payloads = [
            loads(data_lines[index][6:])
            for index, event in enumerate(events)
            if event == "response.tool_execution.progress"
        ]
        completed_index = events.index("response.tool_execution.completed")
        live_indexes = [
            index
            for index, event in enumerate(events)
            if event
            in {
                "response.tool_execution.output",
                "response.tool_execution.progress",
            }
        ]

        self.assertEqual(
            [payload["delta"] for payload in output_payloads],
            ["out", "err", "log"],
        )
        self.assertEqual(
            [payload["data"] for payload in output_payloads],
            [
                {"category": "stdout", "chunk": 1},
                {"category": "stderr", "chunk": 2},
                {"category": "log", "chunk": 3},
            ],
        )
        self.assertEqual(
            [payload["data"]["category"] for payload in progress_payloads],
            ["progress"],
        )
        self.assertEqual(
            [payload["sequence_number"] for payload in output_payloads],
            [2, 3, 4],
        )
        self.assertEqual(
            [payload["sequence_number"] for payload in progress_payloads],
            [5],
        )
        self.assertTrue(all(index < completed_index for index in live_indexes))
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
        tokens = [
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
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
                data={"result": 0},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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
            if block.startswith("event: response.tool_execution.completed")
        ]

        self.assertEqual(len(result_blocks), 1)
        data = loads(result_blocks[0].split("\n")[1][6:])
        self.assertEqual(data["id"], "call-1")
        self.assertEqual(data["data"]["result"], 0)
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

        tokens = [
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
                kind=StreamItemKind.STREAM_DIAGNOSTIC,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(tool_call_id="call-d"),
                text_delta="Unknown tool.",
                data={
                    "diagnostic": {
                        "id": "diag-d",
                        "call_id": "call-d",
                        "code": "tool.unknown",
                        "stage": "resolve",
                    }
                },
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="final",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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

        self.assertIn("response.diagnostic", events)
        diagnostic_index = events.index("response.diagnostic")
        diagnostic_data = loads(data_lines[diagnostic_index][6:])
        self.assertEqual(diagnostic_data["delta"], "Unknown tool.")
        self.assertEqual(
            diagnostic_data["data"]["diagnostic"]["code"], "tool.unknown"
        )
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
        tokens = [
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
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id=call.id),
                data={
                    "result": [
                        {
                            "id": UUID("019b7589-672b-766d-81c6-1da5efd5f49a"),
                            "check_date": date(2025, 9, 19),
                            "gross_check_amount": Decimal("524.46"),
                        }
                    ]
                },
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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

        tool_index = events.index("response.tool_execution.completed")
        payload = loads(data_lines[tool_index][6:])
        result_payload = payload["data"]["result"]

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

        tokens = [
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
                text_delta="a",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="c1"),
                text_delta="t",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="c1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="c1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="b",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
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
            "response.usage.completed",
            "response.completed",
        ]

        self.assertEqual(events, expected)

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
        custom_output_data = loads(data_lines[output_indices[1]][6:])
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

    async def test_streaming_rejects_legacy_events(self) -> None:
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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item for Responses SSE projection",
        ):
            async for chunk in streaming_resp.body_iterator:
                chunks.append(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )

        text = "".join(chunks)
        blocks = [b for b in text.strip().split("\n\n") if b]
        events = [block.split("\n")[0].split(": ")[1] for block in blocks]

        self.assertEqual(events, ["response.created"])
        orchestrator.sync_messages.assert_not_awaited()
