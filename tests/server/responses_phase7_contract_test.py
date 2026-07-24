import importlib
from collections.abc import AsyncIterator
from json import dumps, loads
from logging import Logger, getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from fastapi.responses import JSONResponse

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import MessageRole
from avalan.model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    project_canonical_stream_item,
)
from avalan.server.entities import (
    SKILL_CONTENT_REDACTION,
    ChatMessage,
    ModelVisibleServerProtocolTextRedactor,
    ResponsesRequest,
    ServerOutputRedactionSettings,
)


class _CanonicalResponse:
    input_token_count = 3
    output_token_count = 5

    def __init__(
        self,
        items: tuple[CanonicalStreamItem, ...],
        *,
        answer_text: str = "",
    ) -> None:
        self._items = items
        self._answer_text = answer_text
        self.close_count = 0

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        async def iterate() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._items:
                yield item

        return iterate()

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        **_kwargs: object,
    ) -> AsyncIterator[CanonicalStreamItem]:
        assert stream_session_id
        assert run_id
        assert turn_id
        return self.__aiter__()

    async def to_str(self) -> str:
        return self._answer_text

    async def aclose(self) -> None:
        self.close_count += 1


class _FailingCanonicalResponse:
    input_token_count = 0
    output_token_count = 0

    def __init__(
        self,
        items: tuple[CanonicalStreamItem, ...],
        *,
        source_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self._items = iter(items)
        self._source_error = source_error
        self._cleanup_error = cleanup_error
        self.close_count = 0

    def __aiter__(self) -> "_FailingCanonicalResponse":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        try:
            return next(self._items)
        except StopIteration as error:
            if self._source_error is not None:
                raise self._source_error from error
            raise StopAsyncIteration from error

    async def aclose(self) -> None:
        self.close_count += 1
        if self._cleanup_error is not None:
            raise self._cleanup_error


def _item(
    sequence: int,
    kind: StreamItemKind,
    channel: StreamChannel,
    *,
    correlation: StreamItemCorrelation | None = None,
    text_delta: str | None = None,
    data: Any = None,
    usage: Any = None,
    metadata: dict[str, Any] | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    representation: StreamReasoningRepresentation | None = None,
    segment_ordinal: int | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="phase7-stream",
        run_id="phase7-run",
        turn_id="phase7-turn",
        sequence=sequence,
        kind=kind,
        channel=channel,
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=data,
        usage=usage,
        terminal_outcome=terminal_outcome,
        visibility=(
            StreamVisibility.PRIVATE
            if kind is StreamItemKind.REASONING_DELTA
            else StreamVisibility.PUBLIC
        ),
        reasoning_representation=representation,
        segment_instance_ordinal=segment_ordinal,
        metadata=metadata or {},
    )


def _started(sequence: int = 10) -> CanonicalStreamItem:
    return _item(
        sequence,
        StreamItemKind.STREAM_STARTED,
        StreamChannel.CONTROL,
    )


def _reasoning_delta(
    sequence: int,
    text: str,
    *,
    representation: StreamReasoningRepresentation,
    segment_ordinal: int,
    provider_item_id: str | None = None,
    provider_output_index: int | None = None,
    provider_summary_index: int | None = None,
    continuation_id: str | None = None,
    follows_completion: bool = False,
) -> CanonicalStreamItem:
    return _item(
        sequence,
        StreamItemKind.REASONING_DELTA,
        StreamChannel.REASONING,
        correlation=StreamItemCorrelation(
            protocol_item_id=provider_item_id,
            provider_output_index=provider_output_index,
            provider_summary_index=provider_summary_index,
            model_continuation_id=continuation_id,
        ),
        text_delta=text,
        representation=representation,
        segment_ordinal=segment_ordinal,
        metadata=(
            {REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"}
            if follows_completion
            else None
        ),
    )


def _reasoning_done(sequence: int) -> CanonicalStreamItem:
    return _item(
        sequence,
        StreamItemKind.REASONING_DONE,
        StreamChannel.REASONING,
    )


def _terminal(
    sequence: int,
    outcome: StreamTerminalOutcome = StreamTerminalOutcome.COMPLETED,
    *,
    data: Any = None,
) -> CanonicalStreamItem:
    kind = {
        StreamTerminalOutcome.COMPLETED: StreamItemKind.STREAM_COMPLETED,
        StreamTerminalOutcome.ERRORED: StreamItemKind.STREAM_ERRORED,
        StreamTerminalOutcome.CANCELLED: StreamItemKind.STREAM_CANCELLED,
        StreamTerminalOutcome.INPUT_REQUIRED: (
            StreamItemKind.STREAM_INPUT_REQUIRED
        ),
    }[outcome]
    correlation = (
        StreamItemCorrelation(
            request_id="request-1",
            continuation_id="continuation-1",
            agent_id="agent-1",
            branch_id="branch-1",
        )
        if outcome is StreamTerminalOutcome.INPUT_REQUIRED
        else None
    )
    return _item(
        sequence,
        kind,
        StreamChannel.CONTROL,
        correlation=correlation,
        data=data,
        usage={} if outcome is StreamTerminalOutcome.COMPLETED else None,
        terminal_outcome=outcome,
    )


def _parse_sse(chunks: list[str]) -> list[dict[str, Any]]:
    blocks = [
        block for block in "".join(chunks).strip().split("\n\n") if block
    ]
    records: list[dict[str, Any]] = []
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) == 2
        event = lines[0].removeprefix("event: ")
        data = loads(lines[1].removeprefix("data: "))
        assert data["type"] == event
        records.append(data)
    return records


class ResponsesPhase7ContractTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:  # type: ignore[override]
        self.responses = importlib.import_module(
            "avalan.server.routers.responses"
        )
        self._original_orchestrate = self.responses.orchestrate

    def tearDown(self) -> None:  # type: ignore[override]
        self.responses.orchestrate = self._original_orchestrate

    async def _stream(
        self,
        items: tuple[CanonicalStreamItem, ...],
        *,
        response_id: str,
        output_redaction_settings: ServerOutputRedactionSettings | None = None,
    ) -> tuple[list[dict[str, Any]], AsyncMock]:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()
        response = _CanonicalResponse(items)

        async def orchestrate_stub(_request, _logger, _orchestrator):
            return response, response_id, 0

        self.responses.orchestrate = orchestrate_stub
        request = ResponsesRequest(
            model="phase7-model",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        streaming_response = await self.responses.create_response(
            request,
            getLogger(),
            orchestrator,
            output_redaction_settings=(
                output_redaction_settings
                if output_redaction_settings is not None
                else ServerOutputRedactionSettings()
            ),
        )
        chunks = [
            chunk.decode() if isinstance(chunk, bytes) else chunk
            async for chunk in streaming_response.body_iterator
        ]
        self.assertEqual(response.close_count, 1)
        return _parse_sse(chunks), orchestrator.sync_messages

    async def _open_stream_for_source(
        self,
        response: object,
        *,
        response_id: str,
        logger: Logger | None = None,
        output_redaction_settings: ServerOutputRedactionSettings | None = None,
    ) -> tuple[Any, AsyncMock]:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        async def orchestrate_stub(_request, _logger, _orchestrator):
            return response, response_id, 0

        self.responses.orchestrate = orchestrate_stub
        request = ResponsesRequest(
            model="phase7-model",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        streaming_response = await self.responses.create_response(
            request,
            logger or getLogger(),
            orchestrator,
            output_redaction_settings=(
                output_redaction_settings
                if output_redaction_settings is not None
                else ServerOutputRedactionSettings()
            ),
        )
        return streaming_response, orchestrator.sync_messages

    def _configure_non_stream_source(
        self,
        response: object,
        *,
        response_id: str,
    ) -> tuple[ResponsesRequest, Orchestrator]:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        async def orchestrate_stub(_request, _logger, _orchestrator):
            return response, response_id, 0

        self.responses.orchestrate = orchestrate_stub
        request = ResponsesRequest(
            model="phase7-model",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=False,
        )
        return request, orchestrator

    async def _non_stream(
        self,
        items: tuple[CanonicalStreamItem, ...],
        *,
        response_id: str,
        answer_text: str = "",
        output_redaction_settings: ServerOutputRedactionSettings | None = None,
    ) -> tuple[int, dict[str, Any], AsyncMock]:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()
        response = _CanonicalResponse(items, answer_text=answer_text)

        async def orchestrate_stub(_request, _logger, _orchestrator):
            return response, response_id, 0

        self.responses.orchestrate = orchestrate_stub
        request = ResponsesRequest(
            model="phase7-model",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=False,
        )
        result = await self.responses.create_response(
            request,
            getLogger(),
            orchestrator,
            output_redaction_settings=(
                output_redaction_settings
                if output_redaction_settings is not None
                else ServerOutputRedactionSettings()
            ),
        )
        if isinstance(result, JSONResponse):
            payload = loads(result.body)
            status_code = result.status_code
        else:
            assert isinstance(result, dict)
            payload = result
            status_code = 200
        return status_code, payload, orchestrator.sync_messages

    def _assert_response_local_sequences(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        self.assertEqual(
            [record["sequence_number"] for record in records],
            list(range(len(records))),
        )

    def _assert_retention_failure(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        event_types = [record["type"] for record in records]
        self.assertEqual(event_types[-1], "response.failed")
        self.assertNotIn("response.completed", event_types)
        self.assertEqual(
            records[-1]["error"]["code"],
            "reasoning_summary_retention_exceeded",
        )
        output_done = next(
            record
            for record in reversed(records)
            if record["type"] == "response.output_item.done"
        )
        self.assertEqual(output_done["item"]["status"], "incomplete")

    async def test_input_required_maps_to_native_incomplete_status(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(
            request_id="request-1",
            continuation_id="continuation-1",
            agent_id="agent-1",
            branch_id="branch-1",
        )
        items = (
            _started(10),
            _item(
                20,
                StreamItemKind.INTERACTION_PENDING,
                StreamChannel.INTERACTION,
                correlation=correlation,
            ),
            _terminal(30, StreamTerminalOutcome.INPUT_REQUIRED),
        )

        records, stream_sync = await self._stream(
            items,
            response_id="response-input-required-stream",
        )
        self.assertEqual(
            [record["type"] for record in records],
            ["response.created", "response.incomplete"],
        )
        self.assertNotIn(
            "response.failed", [record["type"] for record in records]
        )
        stream_sync.assert_not_awaited()

        status_code, body, non_stream_sync = await self._non_stream(
            items,
            response_id="response-input-required-non-stream",
        )
        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "incomplete")
        self.assertEqual(body["output"], [])
        self.assertNotIn("error", body)
        non_stream_sync.assert_not_awaited()

    async def test_summary_uses_exact_summary_lifecycle(self) -> None:
        response_id = "resp-summary"
        records, sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "Check records.",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-reasoning",
                    provider_output_index=7,
                    provider_summary_index=4,
                    continuation_id="continuation-a",
                ),
                _reasoning_done(30),
                _terminal(40),
            ),
            response_id=response_id,
        )

        self.assertEqual(
            [record["type"] for record in records],
            [
                "response.created",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
                "response.usage.completed",
                "response.completed",
            ],
        )
        self._assert_response_local_sequences(records)
        self.assertFalse(
            any("reasoning_text" in record["type"] for record in records)
        )

        item_id = f"rs_{response_id}_0"
        added = records[1]
        self.assertEqual(added["output_index"], 0)
        self.assertEqual(
            added["item"],
            {
                "id": item_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
        )
        for record in records[2:6]:
            self.assertEqual(record["item_id"], item_id)
            self.assertEqual(record["output_index"], 0)
            self.assertEqual(record["summary_index"], 0)
        self.assertEqual(
            records[2]["part"],
            {"type": "summary_text", "text": ""},
        )
        self.assertEqual(records[3]["delta"], "Check records.")
        self.assertEqual(records[4]["text"], "Check records.")
        self.assertEqual(
            records[5]["part"],
            {"type": "summary_text", "text": "Check records."},
        )
        self.assertEqual(
            records[6]["item"],
            {
                "id": item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [
                    {"type": "summary_text", "text": "Check records."}
                ],
            },
        )
        sync_messages.assert_awaited_once()

    async def test_native_reasoning_remains_reasoning_text(self) -> None:
        response_id = "resp-native"
        records, _sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    15,
                    "private native",
                    representation=(StreamReasoningRepresentation.NATIVE_TEXT),
                    segment_ordinal=0,
                ),
                _reasoning_done(20),
                _terminal(25),
            ),
            response_id=response_id,
        )

        self.assertEqual(
            [record["type"] for record in records],
            [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.usage.completed",
                "response.completed",
            ],
        )
        self._assert_response_local_sequences(records)
        self.assertFalse(
            any("reasoning_summary" in record["type"] for record in records)
        )
        self.assertEqual(records[1]["output_index"], 0)
        self.assertEqual(records[1]["item"]["id"], f"rs_{response_id}_0")
        self.assertEqual(records[3]["delta"], "private native")

    async def test_allocators_span_continuations_and_item_kinds(self) -> None:
        response_id = "resp-continuations"
        first_reasoning = StreamItemCorrelation(
            protocol_item_id="provider-reused-id",
            provider_output_index=0,
            provider_summary_index=8,
            model_continuation_id="continuation-0",
        )
        tool = StreamItemCorrelation(
            tool_call_id="provider-tool-id",
            model_continuation_id="continuation-0",
        )
        second_reasoning = StreamItemCorrelation(
            protocol_item_id="provider-reused-id",
            provider_output_index=0,
            provider_summary_index=3,
            model_continuation_id="continuation-1",
        )
        records, _sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "first summary",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id=first_reasoning.protocol_item_id,
                    provider_output_index=(
                        first_reasoning.provider_output_index
                    ),
                    provider_summary_index=(
                        first_reasoning.provider_summary_index
                    ),
                    continuation_id=(first_reasoning.model_continuation_id),
                ),
                _item(
                    30,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamChannel.TOOL_CALL,
                    correlation=tool,
                    text_delta="{}",
                ),
                _item(
                    40,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamChannel.TOOL_CALL,
                    correlation=tool,
                ),
                _item(
                    50,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamChannel.TOOL_CALL,
                    correlation=tool,
                ),
                _reasoning_delta(
                    60,
                    "second summary",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id=second_reasoning.protocol_item_id,
                    provider_output_index=(
                        second_reasoning.provider_output_index
                    ),
                    provider_summary_index=(
                        second_reasoning.provider_summary_index
                    ),
                    continuation_id=(second_reasoning.model_continuation_id),
                ),
                _reasoning_done(70),
                _item(
                    80,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta="final answer",
                ),
                _item(
                    90,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                ),
                _terminal(100),
            ),
            response_id=response_id,
        )

        self._assert_response_local_sequences(records)
        output_added = [
            record
            for record in records
            if record["type"] == "response.output_item.added"
        ]
        self.assertEqual(
            [record["output_index"] for record in output_added],
            [0, 1, 2, 3],
        )
        reasoning_added = [
            record
            for record in output_added
            if record["item"].get("type") == "reasoning"
        ]
        self.assertEqual(
            [record["item"]["id"] for record in reasoning_added],
            [f"rs_{response_id}_0", f"rs_{response_id}_2"],
        )
        self.assertNotIn(
            "provider-reused-id",
            {record["item"]["id"] for record in reasoning_added},
        )
        self.assertEqual(
            [
                record["output_index"]
                for record in records
                if record["type"] == "response.reasoning_summary_text.delta"
            ],
            [0, 2],
        )

    async def test_interleaved_tool_calls_keep_one_item_each_with_parity(
        self,
    ) -> None:
        tool_a = StreamItemCorrelation(
            tool_call_id="tool-a",
            model_continuation_id="continuation-a",
        )
        tool_b = StreamItemCorrelation(
            tool_call_id="tool-b",
            model_continuation_id="continuation-a",
        )
        items = (
            _started(0),
            _item(
                1,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=tool_a,
                text_delta='{"a":',
            ),
            _item(
                2,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=tool_b,
                text_delta='{"b":',
            ),
            _item(
                3,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=tool_a,
                text_delta="1}",
            ),
            _item(
                4,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=tool_b,
                text_delta="2}",
            ),
            _item(
                5,
                StreamItemKind.TOOL_CALL_READY,
                StreamChannel.TOOL_CALL,
                correlation=tool_b,
            ),
            _item(
                6,
                StreamItemKind.TOOL_CALL_DONE,
                StreamChannel.TOOL_CALL,
                correlation=tool_b,
            ),
            _item(
                7,
                StreamItemKind.TOOL_CALL_READY,
                StreamChannel.TOOL_CALL,
                correlation=tool_a,
            ),
            _item(
                8,
                StreamItemKind.TOOL_CALL_DONE,
                StreamChannel.TOOL_CALL,
                correlation=tool_a,
            ),
            _terminal(9),
        )

        stream_records, stream_sync = await self._stream(
            items,
            response_id="resp-interleaved-tools",
        )
        status_code, body, non_stream_sync = await self._non_stream(
            items,
            response_id="resp-interleaved-tools",
        )

        self.assertEqual(status_code, 200)
        self._assert_response_local_sequences(stream_records)
        added = [
            record
            for record in stream_records
            if record["type"] == "response.output_item.added"
        ]
        self.assertEqual(
            [
                (record["output_index"], record["item"]["id"])
                for record in added
            ],
            [(0, "tool-a"), (1, "tool-b")],
        )
        deltas = [
            record
            for record in stream_records
            if record["type"] == "response.custom_tool_call_input.delta"
        ]
        self.assertEqual(
            {
                tool_id: "".join(
                    record["delta"]
                    for record in deltas
                    if record["id"] == tool_id
                )
                for tool_id in ("tool-a", "tool-b")
            },
            {"tool-a": '{"a":1}', "tool-b": '{"b":2}'},
        )
        input_done = [
            record
            for record in stream_records
            if record["type"] == "response.custom_tool_call_input.done"
        ]
        output_done = [
            record
            for record in stream_records
            if record["type"] == "response.output_item.done"
        ]
        self.assertEqual(
            [(record["output_index"], record["id"]) for record in input_done],
            [(1, "tool-b"), (0, "tool-a")],
        )
        self.assertEqual(
            [
                (
                    record["output_index"],
                    record["item"]["id"],
                    record["item"]["input"],
                )
                for record in output_done
            ],
            [
                (1, "tool-b", '{"b":2}'),
                (0, "tool-a", '{"a":1}'),
            ],
        )
        stream_items_by_index = {
            record["output_index"]: record["item"] for record in output_done
        }
        self.assertEqual(
            body["output"],
            [stream_items_by_index[index] for index in range(2)],
        )
        self.assertEqual(
            [item["input"] for item in body["output"]],
            ['{"a":1}', '{"b":2}'],
        )
        stream_sync.assert_awaited_once()
        non_stream_sync.assert_awaited_once()

    async def test_summary_coalescing_stops_at_part_identity(self) -> None:
        response_id = "resp-coalesce"
        common = {
            "representation": StreamReasoningRepresentation.SUMMARY,
            "provider_item_id": "provider-item",
            "provider_output_index": 9,
            "continuation_id": "continuation-a",
        }
        records, _sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "A",
                    segment_ordinal=0,
                    provider_summary_index=5,
                    **common,
                ),
                _reasoning_delta(
                    30,
                    "B",
                    segment_ordinal=0,
                    provider_summary_index=5,
                    **common,
                ),
                _reasoning_delta(
                    40,
                    "C",
                    segment_ordinal=1,
                    provider_summary_index=11,
                    follows_completion=True,
                    **common,
                ),
                _reasoning_done(50),
                _terminal(60),
            ),
            response_id=response_id,
        )

        self._assert_response_local_sequences(records)
        deltas = [
            record
            for record in records
            if record["type"] == "response.reasoning_summary_text.delta"
        ]
        self.assertEqual(
            [(record["summary_index"], record["delta"]) for record in deltas],
            [(0, "AB"), (1, "C")],
        )
        part_added = [
            record
            for record in records
            if record["type"] == "response.reasoning_summary_part.added"
        ]
        self.assertEqual(
            [record["summary_index"] for record in part_added],
            [0, 1],
        )
        self.assertEqual(
            len(
                [
                    record
                    for record in records
                    if record["type"] == "response.output_item.added"
                ]
            ),
            1,
        )

    async def test_failure_and_cancellation_close_summary_incomplete(
        self,
    ) -> None:
        cases = (
            (
                StreamTerminalOutcome.ERRORED,
                "response.failed",
                {
                    "error": {
                        "type": "server_error",
                        "code": "response_incomplete",
                        "status": "incomplete",
                        "reason": "max_output_tokens",
                    }
                },
            ),
            (
                StreamTerminalOutcome.CANCELLED,
                "response.cancelled",
                {
                    "error": {
                        "type": "server_error",
                        "code": "provider_cancelled",
                        "status": "cancelled",
                    }
                },
            ),
        )
        for outcome, terminal_type, terminal_data in cases:
            with self.subTest(outcome=outcome):
                records, sync_messages = await self._stream(
                    (
                        _started(),
                        _reasoning_delta(
                            20,
                            "observed prefix",
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=0,
                            provider_item_id="provider-item",
                            provider_output_index=0,
                            provider_summary_index=0,
                        ),
                        _reasoning_done(30),
                        _terminal(
                            40,
                            outcome,
                            data=terminal_data,
                        ),
                    ),
                    response_id=f"resp-{outcome.value}",
                )

                self._assert_response_local_sequences(records)
                event_types = [record["type"] for record in records]
                self.assertEqual(event_types[-1], terminal_type)
                self.assertEqual(event_types.count(terminal_type), 1)
                self.assertNotIn("response.completed", event_types)
                self.assertEqual(
                    event_types[-4:],
                    [
                        "response.reasoning_summary_text.done",
                        "response.reasoning_summary_part.done",
                        "response.output_item.done",
                        terminal_type,
                    ],
                )
                output_done = records[-2]
                text_done = records[-4]
                part_done = records[-3]
                self.assertEqual(text_done["text"], "observed prefix")
                self.assertEqual(
                    part_done["part"]["text"],
                    "observed prefix",
                )
                self.assertEqual(output_done["item"]["status"], "incomplete")
                sync_messages.assert_not_awaited()
                if outcome is StreamTerminalOutcome.ERRORED:
                    self.assertEqual(
                        records[-1]["error"]["code"],
                        "response_incomplete",
                    )

    async def test_failure_and_cancellation_close_empty_observed_prefix(
        self,
    ) -> None:
        marker_source = (
            "# Demo Skill\n\nUse when handling private operator tasks.\n"
        )
        cases = (
            (
                StreamTerminalOutcome.ERRORED,
                "response.failed",
                {"error": {"code": "provider_failed"}},
            ),
            (
                StreamTerminalOutcome.CANCELLED,
                "response.cancelled",
                None,
            ),
        )
        for outcome, terminal_type, terminal_data in cases:
            with self.subTest(outcome=outcome):
                records, sync_messages = await self._stream(
                    (
                        _started(),
                        _reasoning_delta(
                            20,
                            marker_source,
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=0,
                            provider_item_id="provider-item",
                            provider_output_index=0,
                            provider_summary_index=0,
                            continuation_id="continuation-a",
                        ),
                        _reasoning_delta(
                            30,
                            "SUPPRESSED_EMPTY_PREFIX_SECRET",
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=1,
                            provider_item_id="provider-item",
                            provider_output_index=0,
                            provider_summary_index=1,
                            continuation_id="continuation-a",
                            follows_completion=True,
                        ),
                        _reasoning_done(40),
                        _terminal(
                            50,
                            outcome,
                            data=terminal_data,
                        ),
                    ),
                    response_id=f"resp-empty-prefix-{outcome.value}",
                    output_redaction_settings=(
                        ServerOutputRedactionSettings(enabled=True)
                    ),
                )

                self._assert_response_local_sequences(records)
                event_types = [record["type"] for record in records]
                self.assertEqual(event_types[-1], terminal_type)
                self.assertEqual(
                    sum(
                        event_type
                        in {
                            "response.failed",
                            "response.cancelled",
                            "response.completed",
                        }
                        for event_type in event_types
                    ),
                    1,
                )
                text_done = [
                    record
                    for record in records
                    if record["type"] == "response.reasoning_summary_text.done"
                ]
                part_done = [
                    record
                    for record in records
                    if record["type"] == "response.reasoning_summary_part.done"
                ]
                self.assertEqual(text_done[-1]["text"], "")
                self.assertEqual(part_done[-1]["part"]["text"], "")
                output_done = [
                    record
                    for record in records
                    if record["type"] == "response.output_item.done"
                ]
                self.assertEqual(
                    output_done[-1]["item"]["status"],
                    "incomplete",
                )
                projected = dumps(records)
                self.assertNotIn("SUPPRESSED_EMPTY_PREFIX_SECRET", projected)
                sync_messages.assert_not_awaited()

    async def test_response_incomplete_projects_exact_safe_error(self) -> None:
        safe_error = {
            "code": "response_incomplete",
            "message": "response incomplete: max_output_tokens",
            "reason": "max_output_tokens",
            "status": "incomplete",
            "response_id": "resp_provider_safe",
        }
        records, sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "partial summary",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(30),
                _terminal(
                    40,
                    StreamTerminalOutcome.ERRORED,
                    data={
                        "error": safe_error,
                        "provider_payload": "MUST_NOT_ESCAPE",
                    },
                ),
            ),
            response_id="resp-incomplete-safe-error",
        )

        self._assert_response_local_sequences(records)
        self.assertEqual(records[-1]["type"], "response.failed")
        self.assertEqual(records[-1]["error"], safe_error)
        self.assertNotIn("MUST_NOT_ESCAPE", dumps(records))
        self.assertEqual(
            [
                record["type"]
                for record in records
                if record["type"]
                in {
                    "response.failed",
                    "response.cancelled",
                    "response.completed",
                }
            ],
            ["response.failed"],
        )
        sync_messages.assert_not_awaited()

    async def test_response_cancelled_is_only_canonical_cancellation(
        self,
    ) -> None:
        cases = (
            (
                StreamTerminalOutcome.CANCELLED,
                None,
                "response.cancelled",
            ),
            (
                StreamTerminalOutcome.ERRORED,
                {
                    "error": {
                        "code": "provider_cancelled",
                        "status": "cancelled",
                    }
                },
                "response.failed",
            ),
        )
        for outcome, data, expected_terminal in cases:
            with self.subTest(outcome=outcome):
                records, sync_messages = await self._stream(
                    (
                        _started(),
                        _terminal(20, outcome, data=data),
                    ),
                    response_id=f"resp-terminal-{outcome.value}",
                )

                terminals = [
                    record["type"]
                    for record in records
                    if record["type"]
                    in {
                        "response.failed",
                        "response.cancelled",
                        "response.completed",
                    }
                ]
                self.assertEqual(terminals, [expected_terminal])
                self.assertEqual(
                    "response.cancelled" in terminals,
                    outcome is StreamTerminalOutcome.CANCELLED,
                )
                sync_messages.assert_not_awaited()

    async def test_source_terminal_then_reraises_content_free_once(
        self,
    ) -> None:
        secret = "SOURCE_SECRET_SHOULD_NOT_ESCAPE"
        source = _FailingCanonicalResponse(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "observed before source error",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(30),
                _terminal(
                    40,
                    StreamTerminalOutcome.ERRORED,
                    data={"error": {"code": "provider_failed"}},
                ),
            ),
            source_error=RuntimeError(secret),
        )
        logger = getLogger("responses-phase7-source-after-terminal")
        streaming_response, sync_messages = await self._open_stream_for_source(
            source,
            response_id="resp-terminal-reraise",
            logger=logger,
        )
        chunks: list[str] = []
        with self.assertLogs(logger, level="ERROR") as captured:
            with self.assertRaises(
                self.responses._ResponsesSourceAfterTerminalError
            ) as raised:
                async for chunk in streaming_response.body_iterator:
                    chunks.append(
                        chunk.decode() if isinstance(chunk, bytes) else chunk
                    )
        records = _parse_sse(chunks)

        self.assertIs(
            type(raised.exception),
            self.responses._ResponsesSourceAfterTerminalError,
        )
        self.assertEqual(
            str(raised.exception),
            self.responses._RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE,
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-source-after-terminal:"
                + self.responses._RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(secret, logged)
        self.assertNotIn("Traceback", logged)
        terminals = [
            record["type"]
            for record in records
            if record["type"]
            in {
                "response.failed",
                "response.cancelled",
                "response.completed",
            }
        ]
        self.assertEqual(terminals, ["response.failed"])
        output_done = next(
            record
            for record in records
            if record["type"] == "response.output_item.done"
        )
        self.assertEqual(output_done["item"]["status"], "incomplete")
        self.assertEqual(source.close_count, 1)
        sync_messages.assert_not_awaited()

    async def test_source_after_terminal_wins_over_cleanup_error(self) -> None:
        source_secret = "SOURCE_AFTER_TERMINAL_SECRET"
        cleanup_secret = "SOURCE_AFTER_TERMINAL_CLEANUP_SECRET"
        source = _FailingCanonicalResponse(
            (
                _started(),
                _terminal(
                    20,
                    StreamTerminalOutcome.ERRORED,
                    data={"error": {"code": "provider_failed"}},
                ),
            ),
            source_error=RuntimeError(source_secret),
            cleanup_error=RuntimeError(cleanup_secret),
        )
        logger = getLogger(
            "responses-phase7-source-and-cleanup-after-terminal"
        )
        streaming_response, sync_messages = await self._open_stream_for_source(
            source,
            response_id="resp-source-and-cleanup-after-terminal",
            logger=logger,
        )
        chunks: list[str] = []

        with self.assertLogs(logger, level="ERROR") as captured:
            with self.assertRaises(
                self.responses._ResponsesSourceAfterTerminalError
            ) as raised:
                async for chunk in streaming_response.body_iterator:
                    chunks.append(
                        chunk.decode() if isinstance(chunk, bytes) else chunk
                    )

        self.assertIs(
            type(raised.exception),
            self.responses._ResponsesSourceAfterTerminalError,
        )
        self.assertEqual(
            str(raised.exception),
            self.responses._RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE,
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-source-and-cleanup-after-terminal:"
                + self.responses._RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE,
                "ERROR:responses-phase7-source-and-cleanup-after-terminal:"
                + self.responses._RESPONSES_CLEANUP_ERROR_MESSAGE,
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(source_secret, logged)
        self.assertNotIn(cleanup_secret, logged)
        self.assertNotIn("Traceback", logged)
        terminals = [
            record["type"]
            for record in _parse_sse(chunks)
            if record["type"]
            in {
                "response.failed",
                "response.cancelled",
                "response.completed",
            }
        ]
        self.assertEqual(terminals, ["response.failed"])
        self.assertEqual(source.close_count, 1)
        sync_messages.assert_not_awaited()

    async def test_pre_terminal_source_error_wins_over_cleanup_error(
        self,
    ) -> None:
        source_secret = "PRE_TERMINAL_SOURCE_SECRET"
        cleanup_secret = "PRE_TERMINAL_CLEANUP_SECRET"
        source_error = RuntimeError(source_secret)
        source = _FailingCanonicalResponse(
            (_started(),),
            source_error=source_error,
            cleanup_error=RuntimeError(cleanup_secret),
        )
        logger = getLogger("responses-phase7-pre-terminal-source-and-cleanup")
        streaming_response, sync_messages = await self._open_stream_for_source(
            source,
            response_id="resp-pre-terminal-source-and-cleanup",
            logger=logger,
        )
        chunks: list[str] = []

        with self.assertLogs(logger, level="ERROR") as captured:
            with self.assertRaises(RuntimeError) as raised:
                async for chunk in streaming_response.body_iterator:
                    chunks.append(
                        chunk.decode() if isinstance(chunk, bytes) else chunk
                    )

        self.assertIs(raised.exception, source_error)
        self.assertEqual(str(raised.exception), source_secret)
        self.assertNotIn(cleanup_secret, str(raised.exception))
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-pre-terminal-source-and-cleanup:"
                + self.responses._RESPONSES_CLEANUP_ERROR_MESSAGE
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(source_secret, logged)
        self.assertNotIn(cleanup_secret, logged)
        self.assertNotIn("Traceback", logged)
        self.assertFalse(
            any(
                record["type"]
                in {
                    "response.failed",
                    "response.cancelled",
                    "response.completed",
                }
                for record in _parse_sse(chunks)
            )
        )
        self.assertEqual(source.close_count, 1)
        sync_messages.assert_not_awaited()

    async def test_non_stream_source_error_raises_and_never_syncs(
        self,
    ) -> None:
        source_error = RuntimeError("non-stream source failed")
        source = _FailingCanonicalResponse(
            (_started(),),
            source_error=source_error,
        )
        request, orchestrator = self._configure_non_stream_source(
            source,
            response_id="resp-non-stream-source-error",
        )

        with self.assertRaises(RuntimeError) as raised:
            await self.responses.create_response(
                request,
                getLogger(),
                orchestrator,
            )

        self.assertIs(raised.exception, source_error)
        self.assertEqual(source.close_count, 1)
        orchestrator.sync_messages.assert_not_awaited()

    async def test_non_stream_late_source_error_is_content_free_and_no_sync(
        self,
    ) -> None:
        secret = "NON_STREAM_LATE_SOURCE_SECRET"
        source = _FailingCanonicalResponse(
            (_started(), _terminal(20)),
            source_error=RuntimeError(secret),
        )
        request, orchestrator = self._configure_non_stream_source(
            source,
            response_id="resp-non-stream-late-source-error",
        )
        logger = getLogger("responses-phase7-non-stream-late-source")

        with self.assertLogs(logger, level="ERROR") as captured:
            with self.assertRaises(
                self.responses._ResponsesSourceAfterTerminalError
            ) as raised:
                await self.responses.create_response(
                    request,
                    logger,
                    orchestrator,
                )

        self.assertIs(
            type(raised.exception),
            self.responses._ResponsesSourceAfterTerminalError,
        )
        self.assertEqual(
            str(raised.exception),
            self.responses._RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE,
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-non-stream-late-source:"
                + self.responses._RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(secret, logged)
        self.assertNotIn("Traceback", logged)
        self.assertEqual(source.close_count, 1)
        orchestrator.sync_messages.assert_not_awaited()

    async def test_missing_terminal_and_validation_error_have_no_terminal(
        self,
    ) -> None:
        class InvalidSource:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self._items: list[object] = [_started(), "legacy chunk"]
                self.close_count = 0

            def __aiter__(self) -> "InvalidSource":
                return self

            async def __anext__(self) -> object:
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

            async def aclose(self) -> None:
                self.close_count += 1

        cases: tuple[tuple[str, object, str], ...] = (
            (
                "missing-terminal",
                _CanonicalResponse(
                    (
                        _started(),
                        _reasoning_delta(
                            20,
                            "partial",
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=0,
                            provider_item_id="provider-item",
                            provider_output_index=0,
                            provider_summary_index=0,
                        ),
                    )
                ),
                "stream missing terminal outcome",
            ),
            (
                "validation-error",
                InvalidSource(),
                "unsupported stream item for Responses SSE projection",
            ),
        )
        for label, source, message in cases:
            with self.subTest(case=label):
                (
                    streaming_response,
                    sync_messages,
                ) = await self._open_stream_for_source(
                    source,
                    response_id=f"resp-{label}",
                )
                chunks: list[str] = []
                with self.assertRaisesRegex(StreamValidationError, message):
                    async for chunk in streaming_response.body_iterator:
                        chunks.append(
                            chunk.decode()
                            if isinstance(chunk, bytes)
                            else chunk
                        )

                records = _parse_sse(chunks)
                self.assertFalse(
                    any(
                        record["type"]
                        in {
                            "response.failed",
                            "response.cancelled",
                            "response.completed",
                        }
                        for record in records
                    )
                )
                self.assertEqual(getattr(source, "close_count"), 1)
                sync_messages.assert_not_awaited()

    async def test_local_body_aclose_has_no_late_write_and_cleans_source(
        self,
    ) -> None:
        class CloseTrackedSource:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self._items = iter(
                    (
                        _started(),
                        _reasoning_delta(
                            20,
                            "partial before local close",
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=0,
                            provider_item_id="provider-item",
                            provider_output_index=0,
                            provider_summary_index=0,
                        ),
                        _item(
                            30,
                            StreamItemKind.STREAM_DIAGNOSTIC,
                            StreamChannel.CONTROL,
                            data={"code": "flush-boundary"},
                        ),
                    )
                )
                self.close_count = 0
                self.next_count = 0

            def __aiter__(self) -> "CloseTrackedSource":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                self.next_count += 1
                try:
                    return next(self._items)
                except StopIteration as error:
                    raise AssertionError(
                        "source read after local close"
                    ) from error

            async def aclose(self) -> None:
                self.close_count += 1

        source = CloseTrackedSource()
        streaming_response, sync_messages = await self._open_stream_for_source(
            source,
            response_id="resp-local-aclose",
        )
        iterator = streaming_response.body_iterator
        chunks: list[str] = []
        while not any(
            "response.reasoning_summary_text.delta" in chunk
            for chunk in chunks
        ):
            chunk = await anext(iterator)
            chunks.append(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )
        emitted_count = len(chunks)

        await iterator.aclose()

        self.assertEqual(len(chunks), emitted_count)
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)
        self.assertEqual(source.next_count, 3)
        self.assertEqual(source.close_count, 1)
        self.assertFalse(
            any(
                terminal in "".join(chunks)
                for terminal in (
                    "response.failed",
                    "response.cancelled",
                    "response.completed",
                )
            )
        )
        sync_messages.assert_not_awaited()

    async def test_cleanup_error_after_terminal_is_not_second_terminal(
        self,
    ) -> None:
        secret = "CLEANUP_SECRET_SHOULD_NOT_ESCAPE"

        class CleanupErrorSource:
            input_token_count = 0
            output_token_count = 0

            def __init__(self) -> None:
                self._items = iter(
                    (
                        _started(),
                        _item(
                            20,
                            StreamItemKind.ANSWER_DELTA,
                            StreamChannel.ANSWER,
                            text_delta="completed answer",
                        ),
                        _item(
                            30,
                            StreamItemKind.ANSWER_DONE,
                            StreamChannel.ANSWER,
                        ),
                        _terminal(40),
                    )
                )
                self.close_count = 0

            def __aiter__(self) -> "CleanupErrorSource":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self._items)
                except StopIteration as error:
                    raise StopAsyncIteration from error

            async def aclose(self) -> None:
                self.close_count += 1
                raise RuntimeError(secret)

        source = CleanupErrorSource()
        logger = getLogger("responses-phase7-cleanup-after-terminal")
        streaming_response, sync_messages = await self._open_stream_for_source(
            source,
            response_id="resp-cleanup-error",
            logger=logger,
        )
        chunks: list[str] = []
        with self.assertLogs(logger, level="ERROR") as captured:
            with self.assertRaises(
                self.responses._ResponsesCleanupError
            ) as raised:
                async for chunk in streaming_response.body_iterator:
                    chunks.append(
                        chunk.decode() if isinstance(chunk, bytes) else chunk
                    )
        records = _parse_sse(chunks)

        self.assertIs(
            type(raised.exception),
            self.responses._ResponsesCleanupError,
        )
        self.assertEqual(
            str(raised.exception),
            self.responses._RESPONSES_CLEANUP_ERROR_MESSAGE,
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-cleanup-after-terminal:"
                + self.responses._RESPONSES_CLEANUP_ERROR_MESSAGE
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(secret, logged)
        self.assertNotIn("Traceback", logged)
        terminals = [
            record["type"]
            for record in records
            if record["type"]
            in {
                "response.failed",
                "response.cancelled",
                "response.completed",
            }
        ]
        self.assertEqual(terminals, ["response.completed"])
        self.assertEqual(source.close_count, 1)
        sync_messages.assert_awaited_once()

    async def test_streaming_retention_failure_wins_over_cleanup_error(
        self,
    ) -> None:
        policy = StreamRetentionPolicy()
        reasoning_secret = "RETENTION_REASONING_SECRET"
        cleanup_secret = "RETENTION_STREAM_CLEANUP_SECRET"
        source = _FailingCanonicalResponse(
            (
                _started(),
                _reasoning_delta(
                    20,
                    reasoning_secret
                    * (
                        policy.responses_reasoning_item_character_limit
                        // len(reasoning_secret)
                        + 1
                    ),
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
            ),
            cleanup_error=RuntimeError(cleanup_secret),
        )
        logger = getLogger("responses-phase7-retention-and-cleanup")
        streaming_response, sync_messages = await self._open_stream_for_source(
            source,
            response_id="resp-retention-and-cleanup",
            logger=logger,
        )

        with self.assertLogs(logger, level="ERROR") as captured:
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in streaming_response.body_iterator
            ]

        records = _parse_sse(chunks)
        self._assert_retention_failure(records)
        terminals = [
            record["type"]
            for record in records
            if record["type"]
            in {
                "response.failed",
                "response.cancelled",
                "response.completed",
            }
        ]
        self.assertEqual(terminals, ["response.failed"])
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-retention-and-cleanup:"
                + self.responses._RESPONSES_CLEANUP_ERROR_MESSAGE
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(reasoning_secret, logged)
        self.assertNotIn(cleanup_secret, logged)
        self.assertNotIn("Traceback", logged)
        self.assertEqual(source.close_count, 1)
        sync_messages.assert_not_awaited()

    async def test_retention_rejects_1025th_outward_empty_part(self) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        items = [_started(0)]
        for index in range(1025):
            items.append(
                _reasoning_delta(
                    index + 1,
                    "# Demo Skill\n\n" if index == 0 else f"secret-{index}",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=index,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=index,
                    continuation_id="continuation-a",
                    follows_completion=index > 0,
                )
            )
        items.extend(
            [
                _reasoning_done(1026),
                _terminal(1027),
            ]
        )

        records, sync_messages = await self._stream(
            tuple(items),
            response_id="resp-part-limit",
            output_redaction_settings=settings,
        )

        self._assert_retention_failure(records)
        part_added = [
            record
            for record in records
            if record["type"] == "response.reasoning_summary_part.added"
        ]
        self.assertEqual(len(part_added), 1024)
        self.assertEqual(
            [record["summary_index"] for record in part_added],
            list(range(1024)),
        )
        deltas = [
            record["delta"]
            for record in records
            if record["type"] == "response.reasoning_summary_text.delta"
        ]
        self.assertEqual(deltas, [SKILL_CONTENT_REDACTION])
        sync_messages.assert_not_awaited()

    async def test_retention_rejects_oversized_first_delta_after_part_added(
        self,
    ) -> None:
        policy = StreamRetentionPolicy()
        oversized = "x" * (policy.responses_reasoning_item_character_limit + 1)

        records, sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    oversized,
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(30),
                _terminal(40),
            ),
            response_id="resp-oversized-first-delta",
        )

        self._assert_retention_failure(records)
        event_types = [record["type"] for record in records]
        self.assertEqual(
            event_types[:3],
            [
                "response.created",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
            ],
        )
        self.assertNotIn("response.reasoning_summary_text.delta", event_types)
        text_done = next(
            record
            for record in records
            if record["type"] == "response.reasoning_summary_text.done"
        )
        part_done = next(
            record
            for record in records
            if record["type"] == "response.reasoning_summary_part.done"
        )
        self.assertEqual(text_done["text"], "")
        self.assertEqual(part_done["part"]["text"], "")
        sync_messages.assert_not_awaited()

    async def test_retention_reserves_marker_before_redactor_mutation(
        self,
    ) -> None:
        policy = StreamRetentionPolicy()
        marker_candidate = "# Demo Skill\n\n"
        filler_length = (
            policy.responses_reasoning_item_character_limit
            - len(marker_candidate)
            - len(SKILL_CONTENT_REDACTION)
            + 1
        )
        filler = f"{'x' * (filler_length - 1)}\n"

        records, sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    filler,
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_delta(
                    30,
                    marker_candidate,
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(40),
                _terminal(50),
            ),
            response_id="resp-marker-reserve",
            output_redaction_settings=ServerOutputRedactionSettings(
                enabled=True
            ),
        )

        self._assert_retention_failure(records)
        projected_text = "".join(
            record["delta"]
            for record in records
            if record["type"] == "response.reasoning_summary_text.delta"
        )
        self.assertEqual(projected_text, filler)
        self.assertNotIn(SKILL_CONTENT_REDACTION, projected_text)
        self.assertNotIn(marker_candidate, projected_text)
        sync_messages.assert_not_awaited()

    async def test_retention_bounds_pending_host_path_tail(self) -> None:
        policy = StreamRetentionPolicy()
        half_limit = policy.responses_reasoning_item_character_limit // 2
        first_path_half = "a" * half_limit
        rejected_path_half = "b" * half_limit

        records, sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    f"See /tmp/{first_path_half}",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_delta(
                    30,
                    rejected_path_half,
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(40),
                _terminal(50),
            ),
            response_id="resp-host-path-bound",
            output_redaction_settings=ServerOutputRedactionSettings(
                enabled=True,
                rules=frozenset({"host_paths"}),
            ),
        )

        self._assert_retention_failure(records)
        projected_text = "".join(
            record.get("delta", "")
            for record in records
            if record["type"] == "response.reasoning_summary_text.delta"
        )
        self.assertIn("See ", projected_text)
        self.assertNotIn(rejected_path_half, projected_text)
        self.assertLessEqual(
            len(projected_text),
            policy.responses_reasoning_item_character_limit,
        )
        sync_messages.assert_not_awaited()

    async def test_marker_latch_suppresses_later_identities_and_resets(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        records, _sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "# Demo Skill\n\n",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-old",
                    provider_output_index=0,
                    provider_summary_index=0,
                    continuation_id="continuation-a",
                ),
                _reasoning_delta(
                    30,
                    "FIRST_LATER_IDENTITY_SECRET",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=1,
                    provider_item_id="provider-new",
                    provider_output_index=1,
                    provider_summary_index=0,
                    continuation_id="continuation-a",
                    follows_completion=True,
                ),
                _reasoning_delta(
                    40,
                    "SECOND_LATER_IDENTITY_SECRET",
                    representation=StreamReasoningRepresentation.NATIVE_TEXT,
                    segment_ordinal=2,
                    continuation_id="continuation-a",
                    follows_completion=True,
                ),
                _reasoning_done(50),
                _item(
                    60,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta="ANSWER_AFTER_LATCH_SECRET",
                ),
                _item(
                    70,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                ),
                _terminal(80),
            ),
            response_id="resp-marker-latch",
            output_redaction_settings=settings,
        )

        projected = dumps(records)
        marker_deltas = [
            record
            for record in records
            if record["type"] == "response.reasoning_summary_text.delta"
            and record["delta"] == SKILL_CONTENT_REDACTION
        ]
        self.assertEqual(len(marker_deltas), 1)
        self.assertNotIn("FIRST_LATER_IDENTITY_SECRET", projected)
        self.assertNotIn("SECOND_LATER_IDENTITY_SECRET", projected)
        self.assertNotIn("ANSWER_AFTER_LATCH_SECRET", projected)

        fresh_records, _fresh_sync = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "fresh stream text",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-fresh",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(30),
                _terminal(40),
            ),
            response_id="resp-fresh-owner",
            output_redaction_settings=settings,
        )
        fresh_projected = dumps(fresh_records)
        self.assertIn("fresh stream text", fresh_projected)
        self.assertNotIn(SKILL_CONTENT_REDACTION, fresh_projected)

    async def test_pending_marker_resolves_at_every_identity_boundary(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        boundary_cases = (
            (
                "representation",
                {
                    "representation": (
                        StreamReasoningRepresentation.NATIVE_TEXT
                    ),
                    "segment_ordinal": 1,
                    "continuation_id": "continuation-a",
                    "follows_completion": True,
                },
            ),
            (
                "ordinal",
                {
                    "representation": StreamReasoningRepresentation.SUMMARY,
                    "segment_ordinal": 1,
                    "provider_item_id": "provider-old",
                    "provider_output_index": 0,
                    "provider_summary_index": 0,
                    "continuation_id": "continuation-a",
                    "follows_completion": True,
                },
            ),
            (
                "provider_item_id",
                {
                    "representation": StreamReasoningRepresentation.SUMMARY,
                    "segment_ordinal": 1,
                    "provider_item_id": "provider-new",
                    "provider_output_index": 0,
                    "provider_summary_index": 0,
                    "continuation_id": "continuation-a",
                    "follows_completion": True,
                },
            ),
            (
                "provider_output_index",
                {
                    "representation": StreamReasoningRepresentation.SUMMARY,
                    "segment_ordinal": 1,
                    "provider_item_id": "provider-old",
                    "provider_output_index": 1,
                    "provider_summary_index": 0,
                    "continuation_id": "continuation-a",
                    "follows_completion": True,
                },
            ),
            (
                "provider_summary_index",
                {
                    "representation": StreamReasoningRepresentation.SUMMARY,
                    "segment_ordinal": 1,
                    "provider_item_id": "provider-old",
                    "provider_output_index": 0,
                    "provider_summary_index": 1,
                    "continuation_id": "continuation-a",
                    "follows_completion": True,
                },
            ),
            (
                "continuation",
                {
                    "representation": StreamReasoningRepresentation.SUMMARY,
                    "segment_ordinal": 0,
                    "provider_item_id": "provider-old",
                    "provider_output_index": 0,
                    "provider_summary_index": 0,
                    "continuation_id": "continuation-b",
                },
            ),
        )
        for label, next_identity in boundary_cases:
            with self.subTest(boundary=label):
                records, _sync_messages = await self._stream(
                    (
                        _started(),
                        _reasoning_delta(
                            20,
                            "# Demo Skill\n\n",
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=0,
                            provider_item_id="provider-old",
                            provider_output_index=0,
                            provider_summary_index=0,
                            continuation_id="continuation-a",
                        ),
                        _reasoning_delta(
                            30,
                            f"{label.upper()}_BOUNDARY_SECRET",
                            **next_identity,
                        ),
                        _reasoning_done(40),
                        _terminal(50),
                    ),
                    response_id=f"resp-boundary-{label}",
                    output_redaction_settings=settings,
                )

                projected = dumps(records)
                self.assertIn(SKILL_CONTENT_REDACTION, projected)
                self.assertNotIn(f"{label.upper()}_BOUNDARY_SECRET", projected)

        completion_records, _sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "# Demo Skill\n\n",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-old",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
                _reasoning_done(30),
                _item(
                    40,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta="COMPLETION_BOUNDARY_SECRET",
                ),
                _item(
                    50,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                ),
                _terminal(60),
            ),
            response_id="resp-boundary-completion",
            output_redaction_settings=settings,
        )
        completion_projected = dumps(completion_records)
        self.assertIn(SKILL_CONTENT_REDACTION, completion_projected)
        self.assertNotIn("COMPLETION_BOUNDARY_SECRET", completion_projected)

    async def test_no_marker_identity_loss_quarantines_once_then_recovers(
        self,
    ) -> None:
        records, _sync_messages = await self._stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "ordinary before identity loss",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-old",
                    provider_output_index=0,
                    provider_summary_index=0,
                    continuation_id="continuation-a",
                ),
                _reasoning_delta(
                    30,
                    "IDENTITY_LOSS_SECRET",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=1,
                    follows_completion=True,
                ),
                _reasoning_delta(
                    40,
                    "QUARANTINED_SECRET",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=2,
                    provider_item_id="provider-quarantined",
                    provider_output_index=1,
                    provider_summary_index=0,
                    continuation_id="continuation-a",
                    follows_completion=True,
                ),
                _reasoning_delta(
                    50,
                    "ordinary recovered",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=3,
                    provider_item_id="provider-recovered",
                    provider_output_index=2,
                    provider_summary_index=0,
                    continuation_id="continuation-a",
                    follows_completion=True,
                ),
                _reasoning_done(60),
                _terminal(70),
            ),
            response_id="resp-identity-loss",
            output_redaction_settings=ServerOutputRedactionSettings(
                enabled=True
            ),
        )

        projected = dumps(records)
        self.assertIn("ordinary before identity loss", projected)
        self.assertNotIn("IDENTITY_LOSS_SECRET", projected)
        self.assertNotIn("QUARANTINED_SECRET", projected)
        self.assertIn("ordinary recovered", projected)
        self.assertNotIn(SKILL_CONTENT_REDACTION, projected)

    async def test_non_stream_preserves_ordered_item_representations(
        self,
    ) -> None:
        response_id = "resp-non-stream-ordered"
        summary = "summary sentinel"
        native = "native sentinel"
        tool_input = '{"query":"safe"}'
        answer = '{"answer":{"ok":true}}'
        status_code, body, sync_messages = await self._non_stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    summary,
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-summary",
                    provider_output_index=0,
                    provider_summary_index=4,
                    continuation_id="continuation-a",
                ),
                _reasoning_delta(
                    30,
                    native,
                    representation=(StreamReasoningRepresentation.NATIVE_TEXT),
                    segment_ordinal=1,
                    continuation_id="continuation-a",
                    follows_completion=True,
                ),
                _reasoning_done(40),
                _item(
                    50,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamChannel.TOOL_CALL,
                    correlation=StreamItemCorrelation(
                        tool_call_id="tool-1",
                        model_continuation_id="continuation-a",
                    ),
                    text_delta=tool_input,
                ),
                _item(
                    60,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamChannel.TOOL_CALL,
                    correlation=StreamItemCorrelation(
                        tool_call_id="tool-1",
                        model_continuation_id="continuation-a",
                    ),
                ),
                _item(
                    70,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamChannel.TOOL_CALL,
                    correlation=StreamItemCorrelation(
                        tool_call_id="tool-1",
                        model_continuation_id="continuation-a",
                    ),
                ),
                _item(
                    80,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta=answer,
                ),
                _item(
                    90,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                ),
                _terminal(100),
            ),
            response_id=response_id,
            answer_text=answer,
        )

        self.assertEqual(status_code, 200)
        output = body["output"]
        self.assertEqual(
            [item["type"] for item in output],
            [
                "reasoning",
                "reasoning_text",
                "custom_tool_call_input",
                "message",
            ],
        )
        self.assertEqual(
            output[0],
            {
                "id": f"rs_{response_id}_0",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": summary}],
            },
        )
        self.assertEqual(output[1]["id"], f"rs_{response_id}_1")
        self.assertEqual(output[1]["status"], "completed")
        self.assertEqual(
            output[1]["content"],
            [{"type": "reasoning_text", "text": native}],
        )
        self.assertEqual(output[2]["id"], "tool-1")
        self.assertEqual(output[2]["status"], "completed")
        self.assertEqual(output[3]["role"], "assistant")
        self.assertEqual(
            output[3]["content"],
            [{"type": "output_text", "text": answer}],
        )
        self.assertEqual(
            loads(output[3]["content"][0]["text"]), {"answer": {"ok": True}}
        )
        self.assertNotIn(summary, dumps(output[1:]))
        self.assertNotIn(native, dumps((output[0], output[2], output[3])))
        self.assertNotIn(summary, dumps(output[3]))
        sync_messages.assert_awaited_once()

    async def test_non_stream_rebases_sparse_summary_emission_order(
        self,
    ) -> None:
        provider_order = (2, 7, 0, 1, 3, 4, 5, 6)
        items = [_started()]
        for outward_index, provider_index in enumerate(provider_order):
            items.append(
                _reasoning_delta(
                    20 + outward_index * 10,
                    f"provider-part-{provider_index}",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=outward_index,
                    provider_item_id="provider-sparse",
                    provider_output_index=4,
                    provider_summary_index=provider_index,
                    continuation_id="continuation-a",
                    follows_completion=outward_index > 0,
                )
            )
        items.extend([_reasoning_done(100), _terminal(110)])

        status_code, body, sync_messages = await self._non_stream(
            tuple(items),
            response_id="resp-sparse-summary",
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(len(body["output"]), 1)
        summary = body["output"][0]["summary"]
        self.assertEqual(
            [part["text"] for part in summary],
            [f"provider-part-{index}" for index in provider_order],
        )
        self.assertEqual(
            list(enumerate(part["text"] for part in summary)),
            [
                (index, f"provider-part-{provider_index}")
                for index, provider_index in enumerate(provider_order)
            ],
        )
        sync_messages.assert_awaited_once()

    async def test_non_stream_repeated_provider_ids_stay_response_unique(
        self,
    ) -> None:
        response_id = "resp-repeated-provider-id"
        tool_correlation = StreamItemCorrelation(
            tool_call_id="tool-1",
            model_continuation_id="continuation-a",
        )
        answer = "done"
        status_code, body, sync_messages = await self._non_stream(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "first",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-reused",
                    provider_output_index=0,
                    provider_summary_index=0,
                    continuation_id="continuation-a",
                ),
                _item(
                    30,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamChannel.TOOL_CALL,
                    correlation=tool_correlation,
                    text_delta="{}",
                ),
                _item(
                    40,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamChannel.TOOL_CALL,
                    correlation=tool_correlation,
                ),
                _item(
                    50,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamChannel.TOOL_CALL,
                    correlation=tool_correlation,
                ),
                _reasoning_delta(
                    60,
                    "second",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-reused",
                    provider_output_index=0,
                    provider_summary_index=0,
                    continuation_id="continuation-b",
                ),
                _reasoning_done(70),
                _item(
                    80,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta=answer,
                ),
                _item(
                    90,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                ),
                _terminal(100),
            ),
            response_id=response_id,
            answer_text=answer,
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(
            [item["type"] for item in body["output"]],
            ["reasoning", "custom_tool_call_input", "reasoning", "message"],
        )
        reasoning_ids = [
            item["id"]
            for item in body["output"]
            if item["type"] == "reasoning"
        ]
        self.assertEqual(
            reasoning_ids,
            [f"rs_{response_id}_0", f"rs_{response_id}_2"],
        )
        self.assertEqual(len(set(reasoning_ids)), 2)
        self.assertNotIn("provider-reused", reasoning_ids)
        sync_messages.assert_awaited_once()

    async def test_non_stream_abnormal_status_and_sync_ownership(self) -> None:
        cases = (
            (
                "failed",
                StreamTerminalOutcome.ERRORED,
                {"error": {"code": "provider_failed", "message": "failed"}},
                "failed",
                "provider_failed",
            ),
            (
                "cancelled",
                StreamTerminalOutcome.CANCELLED,
                None,
                "cancelled",
                None,
            ),
            (
                "incomplete",
                StreamTerminalOutcome.ERRORED,
                {
                    "error": {
                        "code": "response_incomplete",
                        "message": "response incomplete: max_output_tokens",
                        "reason": "max_output_tokens",
                        "status": "incomplete",
                    }
                },
                "failed",
                "response_incomplete",
            ),
        )
        for label, outcome, data, response_status, error_code in cases:
            with self.subTest(case=label):
                status_code, body, sync_messages = await self._non_stream(
                    (
                        _started(),
                        _reasoning_delta(
                            20,
                            "observed prefix",
                            representation=(
                                StreamReasoningRepresentation.SUMMARY
                            ),
                            segment_ordinal=0,
                            provider_item_id="provider-item",
                            provider_output_index=0,
                            provider_summary_index=0,
                        ),
                        _reasoning_done(30),
                        _terminal(40, outcome, data=data),
                    ),
                    response_id=f"resp-non-stream-{label}",
                )

                self.assertEqual(status_code, 200)
                self.assertEqual(body["status"], response_status)
                self.assertEqual(body["output"][0]["status"], "incomplete")
                if error_code is None:
                    self.assertIsNone(body.get("error"))
                else:
                    self.assertEqual(body["error"]["code"], error_code)
                    self.assertNotIn("provider_payload", dumps(body["error"]))
                sync_messages.assert_not_awaited()

    async def test_non_stream_retention_is_exact_all_or_nothing_http_500(
        self,
    ) -> None:
        policy = StreamRetentionPolicy()
        oversized_items = (
            _started(),
            _reasoning_delta(
                20,
                "x" * (policy.responses_reasoning_item_character_limit + 1),
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_ordinal=0,
                provider_item_id="provider-item",
                provider_output_index=0,
                provider_summary_index=0,
            ),
            _reasoning_done(30),
            _terminal(40),
        )
        part_limit_items = [_started(0)]
        for index in range(1025):
            part_limit_items.append(
                _reasoning_delta(
                    index + 1,
                    "x",
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=index,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=index,
                    continuation_id="continuation-a",
                    follows_completion=index > 0,
                )
            )
        part_limit_items.extend([_reasoning_done(1026), _terminal(1027)])
        marker_candidate = "# Demo Skill\n\n"
        filler_length = (
            policy.responses_reasoning_item_character_limit
            - len(marker_candidate)
            - len(SKILL_CONTENT_REDACTION)
            + 1
        )
        filler = f"{'x' * (filler_length - 1)}\n"
        marker_items = (
            _started(),
            _reasoning_delta(
                20,
                filler,
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_ordinal=0,
                provider_item_id="provider-item",
                provider_output_index=0,
                provider_summary_index=0,
            ),
            _reasoning_delta(
                30,
                marker_candidate,
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_ordinal=0,
                provider_item_id="provider-item",
                provider_output_index=0,
                provider_summary_index=0,
            ),
            _reasoning_done(40),
            _terminal(50),
        )
        cases = (
            ("oversized", oversized_items),
            ("part-limit", tuple(part_limit_items)),
            ("marker-reserve", marker_items),
        )
        expected = {
            "error": {
                "type": "server_error",
                "code": "reasoning_summary_retention_exceeded",
                "message": (
                    "Reasoning summary exceeded the configured retention "
                    "limit."
                ),
            }
        }
        for label, items in cases:
            with self.subTest(case=label):
                status_code, body, sync_messages = await self._non_stream(
                    items,
                    response_id=f"resp-retention-{label}",
                    output_redaction_settings=(
                        ServerOutputRedactionSettings(enabled=True)
                    ),
                )

                self.assertEqual(status_code, 500)
                self.assertEqual(body, expected)
                self.assertNotIn("output", body)
                sync_messages.assert_not_awaited()

    async def test_non_stream_retention_masks_cleanup_with_locked_error(
        self,
    ) -> None:
        secret = "RETENTION_CLEANUP_SECRET"
        policy = StreamRetentionPolicy()
        source = _FailingCanonicalResponse(
            (
                _started(),
                _reasoning_delta(
                    20,
                    "x"
                    * (policy.responses_reasoning_item_character_limit + 1),
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=0,
                    provider_summary_index=0,
                ),
            ),
            cleanup_error=RuntimeError(secret),
        )
        request, orchestrator = self._configure_non_stream_source(
            source,
            response_id="resp-retention-cleanup-error",
        )
        logger = getLogger("responses-phase7-retention-cleanup")

        with self.assertLogs(logger, level="ERROR") as captured:
            result = await self.responses.create_response(
                request,
                logger,
                orchestrator,
            )

        self.assertIs(type(result), JSONResponse)
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.media_type, "application/json")
        self.assertEqual(
            result.body,
            b'{"error":{"type":"server_error","code":'
            b'"reasoning_summary_retention_exceeded","message":'
            b'"Reasoning summary exceeded the configured retention '
            b'limit."}}',
        )
        self.assertEqual(
            captured.output,
            [
                "ERROR:responses-phase7-retention-cleanup:"
                + self.responses._RESPONSES_CLEANUP_ERROR_MESSAGE
            ],
        )
        logged = "\n".join(captured.output)
        self.assertNotIn(secret, logged)
        self.assertNotIn("Traceback", logged)
        self.assertEqual(source.close_count, 1)
        orchestrator.sync_messages.assert_not_awaited()

    async def test_fixed_id_streaming_and_non_stream_output_parity(
        self,
    ) -> None:
        response_id = "resp-fixed-parity"
        tool_correlation = StreamItemCorrelation(
            tool_call_id="tool-1",
            model_continuation_id="continuation-a",
        )
        answer = '{"ok":true}'
        items = (
            _started(),
            _reasoning_delta(
                20,
                "first-a",
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_ordinal=0,
                provider_item_id="provider-reused",
                provider_output_index=0,
                provider_summary_index=5,
                continuation_id="continuation-a",
            ),
            _reasoning_delta(
                30,
                "first-b",
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_ordinal=1,
                provider_item_id="provider-reused",
                provider_output_index=0,
                provider_summary_index=9,
                continuation_id="continuation-a",
                follows_completion=True,
            ),
            _item(
                40,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=tool_correlation,
                text_delta="{}",
            ),
            _item(
                50,
                StreamItemKind.TOOL_CALL_READY,
                StreamChannel.TOOL_CALL,
                correlation=tool_correlation,
            ),
            _item(
                60,
                StreamItemKind.TOOL_CALL_DONE,
                StreamChannel.TOOL_CALL,
                correlation=tool_correlation,
            ),
            _reasoning_delta(
                70,
                "second",
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_ordinal=0,
                provider_item_id="provider-reused",
                provider_output_index=0,
                provider_summary_index=7,
                continuation_id="continuation-b",
            ),
            _reasoning_done(80),
            _item(
                90,
                StreamItemKind.ANSWER_DELTA,
                StreamChannel.ANSWER,
                text_delta=answer,
            ),
            _item(
                100,
                StreamItemKind.ANSWER_DONE,
                StreamChannel.ANSWER,
            ),
            _terminal(110),
        )

        stream_records, _stream_sync = await self._stream(
            items,
            response_id=response_id,
        )
        status_code, body, non_stream_sync = await self._non_stream(
            items,
            response_id=response_id,
            answer_text=answer,
        )

        self.assertEqual(status_code, 200)
        stream_output = [
            record["item"]
            for record in stream_records
            if record["type"] == "response.output_item.done"
        ]
        self.assertEqual(stream_output, body["output"])
        self.assertEqual(
            [
                item["id"]
                for item in body["output"]
                if item["type"] == "reasoning"
            ],
            [f"rs_{response_id}_0", f"rs_{response_id}_2"],
        )
        self.assertEqual(
            [
                record["summary_index"]
                for record in stream_records
                if record["type"] == "response.reasoning_summary_part.added"
            ],
            [0, 1, 0],
        )
        self.assertEqual(
            [
                len(item["summary"])
                for item in body["output"]
                if item["type"] == "reasoning"
            ],
            [2, 1],
        )
        non_stream_sync.assert_awaited_once()

    async def test_non_stream_summary_isolated_from_json_answer(self) -> None:
        response_id = "resp-non-stream"
        summary = "private summary sentinel"
        answer = '{"ok":true}'
        response = _CanonicalResponse(
            (
                _started(),
                _reasoning_delta(
                    20,
                    summary,
                    representation=StreamReasoningRepresentation.SUMMARY,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                    provider_output_index=4,
                    provider_summary_index=6,
                ),
                _reasoning_done(30),
                _item(
                    40,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta=answer,
                ),
                _item(
                    50,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                ),
                _terminal(60),
            ),
            answer_text=answer,
        )
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.sync_messages = AsyncMock()

        async def orchestrate_stub(_request, _logger, _orchestrator):
            return response, response_id, 0

        self.responses.orchestrate = orchestrate_stub
        request = ResponsesRequest(
            model="phase7-model",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=False,
        )

        body = await self.responses.create_response(
            request,
            getLogger(),
            orchestrator,
        )

        self.assertIsInstance(body, dict)
        output = body["output"]
        self.assertEqual(
            output[0],
            {
                "id": f"rs_{response_id}_0",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": summary}],
            },
        )
        self.assertEqual(output[1]["type"], "message")
        self.assertEqual(output[1]["role"], "assistant")
        self.assertEqual(
            output[1]["content"],
            [{"type": "output_text", "text": answer}],
        )
        self.assertEqual(loads(output[1]["content"][0]["text"]), {"ok": True})
        self.assertNotIn(summary, dumps(output[1]))
        orchestrator.sync_messages.assert_awaited_once()

    def test_legacy_adapter_flushes_pending_text_at_state_boundaries(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        adapter = self.responses._ResponsesSSEProjectionAdapter(
            answer_redactor=ModelVisibleServerProtocolTextRedactor(
                settings,
                protocol="openai",
                channel="answer",
            ),
            reasoning_redactor=ModelVisibleServerProtocolTextRedactor(
                settings,
                protocol="openai",
                channel="reasoning",
            ),
        )
        reasoning = project_canonical_stream_item(
            _reasoning_delta(
                10,
                "seed",
                representation=StreamReasoningRepresentation.NATIVE_TEXT,
                segment_ordinal=0,
            )
        )
        answer = project_canonical_stream_item(
            _item(
                20,
                StreamItemKind.ANSWER_DELTA,
                StreamChannel.ANSWER,
                text_delta="",
            )
        )
        diagnostic = project_canonical_stream_item(
            _item(
                30,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamChannel.CONTROL,
            )
        )
        tool = project_canonical_stream_item(
            _item(
                40,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
                text_delta="{}",
            )
        )

        self.assertIs(
            adapter.model_text_redactor(answer), adapter.answer_redactor
        )
        self.assertIs(
            adapter.model_text_redactor(reasoning),
            adapter.reasoning_redactor,
        )
        self.assertIsNone(adapter.model_text_redactor(diagnostic))
        adapter.record_model_text_pending(reasoning, None)
        self.assertEqual(adapter.flush_model_text_before_switch(reasoning), [])

        adapter.switch(reasoning)
        self.assertEqual(adapter.reasoning_redactor.push("# Demo"), ())
        adapter.record_model_text_pending(
            reasoning, adapter.reasoning_redactor
        )
        self.assertEqual(adapter.flush_model_text_before_switch(reasoning), [])
        reasoning_events = adapter.flush_model_text_before_switch(answer)
        self.assertEqual(
            [(event.event, event.data["delta"]) for event in reasoning_events],
            [("response.reasoning_text.delta", "# Demo")],
        )
        adapter.record_model_text_pending(
            reasoning, adapter.reasoning_redactor
        )
        self.assertIsNone(adapter.reasoning_pending_sequence)

        adapter.switch(answer)
        self.assertEqual(adapter.answer_redactor.push("# Browser"), ())
        adapter.record_model_text_pending(answer, adapter.answer_redactor)
        answer_events = adapter.flush_model_text_before_switch(reasoning)
        self.assertEqual(
            [(event.event, event.data["delta"]) for event in answer_events],
            [("response.output_text.delta", "# Browser")],
        )
        adapter.record_model_text_pending(answer, adapter.answer_redactor)
        self.assertIsNone(adapter.answer_pending_sequence)

        adapter.switch(tool)
        self.assertEqual(
            adapter.flush_model_text_before_switch(diagnostic), []
        )
        self.assertTrue(adapter.close())
        self.assertIsNone(adapter.state)

    def test_projector_retention_and_idempotent_completion_guards(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        native = project_canonical_stream_item(
            _reasoning_delta(
                10,
                "native",
                representation=StreamReasoningRepresentation.NATIVE_TEXT,
                segment_ordinal=0,
                provider_item_id="provider-native",
            )
        )
        reasoning_done = project_canonical_stream_item(_reasoning_done(20))
        answer = project_canonical_stream_item(
            _item(
                30,
                StreamItemKind.ANSWER_DELTA,
                StreamChannel.ANSWER,
                text_delta="answer",
            )
        )
        answer_done = project_canonical_stream_item(
            _item(40, StreamItemKind.ANSWER_DONE, StreamChannel.ANSWER)
        )

        projector = self.responses._ResponsesSSEProjector("guard", settings)
        self.assertEqual(projector.events_for(reasoning_done), [])
        self.assertEqual(projector.events_for(answer_done), [])
        projector.events_for(native)
        self.assertTrue(projector.events_for(reasoning_done))
        self.assertEqual(projector.events_for(reasoning_done), [])
        projector.events_for(answer)
        self.assertTrue(projector.events_for(answer_done))
        self.assertEqual(projector.events_for(answer_done), [])

        segment_projector = self.responses._ResponsesSSEProjector(
            "segment-limit",
            settings,
            StreamRetentionPolicy(responses_reasoning_item_segment_limit=0),
        )
        segment_events = segment_projector.events_for(native)
        self.assertEqual(
            segment_events[-1].data["item"]["status"],
            "incomplete",
        )
        self.assertIsNotNone(segment_projector.failure)
        self.assertEqual(segment_projector.events_for(reasoning_done), [])

        byte_projector = self.responses._ResponsesSSEProjector(
            "byte-limit",
            settings,
            StreamRetentionPolicy(
                responses_reasoning_item_character_limit=10,
                responses_reasoning_item_text_byte_limit=3,
            ),
        )
        byte_events = byte_projector.events_for(
            project_canonical_stream_item(
                _reasoning_delta(
                    10,
                    "éé",
                    representation=StreamReasoningRepresentation.NATIVE_TEXT,
                    segment_ordinal=0,
                )
            )
        )
        self.assertEqual(
            byte_events[-1].data["item"]["status"],
            "incomplete",
        )
        self.assertIsNotNone(byte_projector.failure)

    def test_reasoning_flush_admission_enforces_character_and_byte_limits(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        identity = ("reasoning-part",)
        cases = (
            (0, 100, "character"),
            (100, 0, "byte"),
        )
        for character_limit, byte_limit, expected in cases:
            with self.subTest(limit=expected):
                redactor = ModelVisibleServerProtocolTextRedactor(
                    settings,
                    protocol="openai",
                    channel="reasoning",
                )
                self.assertEqual(redactor.push("/Users/private"), ())
                admission = self.responses._ResponsesReasoningAdmission(
                    segment_limit=1,
                    character_limit=character_limit,
                    utf8_byte_limit=byte_limit,
                )
                with self.assertRaisesRegex(
                    self.responses._ResponsesReasoningRetentionError,
                    expected,
                ):
                    admission.admit_flush(identity, redactor)

    def test_projector_quarantines_pending_reasoning_across_boundaries(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(enabled=True)
        diagnostic = project_canonical_stream_item(
            _item(
                30,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamChannel.CONTROL,
            )
        )
        for representation in (
            StreamReasoningRepresentation.SUMMARY,
            StreamReasoningRepresentation.NATIVE_TEXT,
        ):
            with self.subTest(representation=representation):
                projector = self.responses._ResponsesSSEProjector(
                    f"pending-{representation.value}",
                    settings,
                )
                projector.events_for(
                    project_canonical_stream_item(
                        _reasoning_delta(
                            10,
                            "/Users/private",
                            representation=representation,
                            segment_ordinal=0,
                            provider_item_id="provider-item",
                        )
                    )
                )
                projector.events_for(diagnostic)
                self.assertTrue(projector._quarantine_next_reasoning)

        identity_projector = self.responses._ResponsesSSEProjector(
            "native-identity-loss",
            settings,
        )
        identity_projector.events_for(
            project_canonical_stream_item(
                _reasoning_delta(
                    10,
                    "known",
                    representation=StreamReasoningRepresentation.NATIVE_TEXT,
                    segment_ordinal=0,
                    provider_item_id="provider-item",
                )
            )
        )
        identity_projector.events_for(
            project_canonical_stream_item(
                _reasoning_delta(
                    20,
                    "lost",
                    representation=StreamReasoningRepresentation.NATIVE_TEXT,
                    segment_ordinal=1,
                )
            )
        )
        self.assertTrue(identity_projector._quarantine_next_reasoning)

    def test_projector_closes_parallel_tools_and_rejects_type_change(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings()
        custom = project_canonical_stream_item(
            _item(
                10,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="custom"),
                text_delta="{}",
            )
        )
        function = project_canonical_stream_item(
            _item(
                20,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="function"),
                text_delta='{"x":1}',
                data={"name": "demo", "arguments": '{"x":1}'},
            )
        )
        projector = self.responses._ResponsesSSEProjector("tools", settings)
        projector.events_for(custom)
        projector.events_for(function)
        terminal_events = projector.events_for(
            project_canonical_stream_item(_terminal(30))
        )
        done_items = [
            event.data["item"]
            for event in terminal_events
            if event.event == "response.output_item.done"
        ]
        self.assertEqual(
            [(item["id"], item["type"]) for item in done_items],
            [
                ("custom", "custom_tool_call_input"),
                ("function", "function_call"),
            ],
        )

        changed = self.responses._ResponsesSSEProjector("changed", settings)
        changed.events_for(custom)
        changed_function = project_canonical_stream_item(
            _item(
                20,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="custom"),
                text_delta="{}",
                data={"name": "demo", "arguments": "{}"},
            )
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "changed protocol item type",
        ):
            changed.events_for(changed_function)

    async def test_non_stream_cleanup_missing_terminal_and_index_validation(
        self,
    ) -> None:
        cleanup_source = _FailingCanonicalResponse(
            (_started(), _terminal(20)),
            cleanup_error=RuntimeError("PRIVATE_CLEANUP_SENTINEL"),
        )
        cleanup_request, cleanup_orchestrator = (
            self._configure_non_stream_source(
                cleanup_source,
                response_id="non-stream-cleanup",
            )
        )
        cleanup_logger = getLogger("responses-phase7-non-stream-cleanup")
        with self.assertLogs(cleanup_logger, level="ERROR") as captured:
            with self.assertRaises(
                self.responses._ResponsesCleanupError
            ) as raised:
                await self.responses.create_response(
                    cleanup_request,
                    cleanup_logger,
                    cleanup_orchestrator,
                )
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertNotIn(
            "PRIVATE_CLEANUP_SENTINEL", "\n".join(captured.output)
        )
        cast(
            AsyncMock, cleanup_orchestrator.sync_messages
        ).assert_not_awaited()

        missing_source = _CanonicalResponse((_started(),))
        missing_request, missing_orchestrator = (
            self._configure_non_stream_source(
                missing_source,
                response_id="non-stream-missing-terminal",
            )
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "stream missing terminal outcome",
        ):
            await self.responses.create_response(
                missing_request,
                getLogger(),
                missing_orchestrator,
            )
        cast(
            AsyncMock, missing_orchestrator.sync_messages
        ).assert_not_awaited()

        class CorruptProjector:
            failure = None

            def __init__(self, indices: tuple[int, ...]) -> None:
                self._indices = iter(indices)

            def events_for(
                self,
                projection: StreamConsumerProjection,
            ) -> list[Any]:
                if projection.kind is not StreamItemKind.ANSWER_DELTA:
                    return []
                output_index = next(self._indices)
                return [
                    self_module._ResponsesSSEEvent(
                        event="response.output_item.done",
                        data={
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": {"id": f"item-{output_index}"},
                        },
                    )
                ]

        self_module = self.responses
        corruption_cases = (
            ("duplicate", (0, 0), "duplicate Responses outward output index"),
            ("gap", (1,), "non-contiguous Responses outward output indices"),
        )
        for label, indices, expected in corruption_cases:
            with self.subTest(corruption=label):
                items = [_started()]
                for sequence in range(len(indices)):
                    items.append(
                        _item(
                            20 + sequence,
                            StreamItemKind.ANSWER_DELTA,
                            StreamChannel.ANSWER,
                            text_delta="x",
                        )
                    )
                items.append(
                    _item(
                        40,
                        StreamItemKind.ANSWER_DONE,
                        StreamChannel.ANSWER,
                    )
                )
                items.append(_terminal(50))
                source = _CanonicalResponse(tuple(items))
                request, orchestrator = self._configure_non_stream_source(
                    source,
                    response_id=f"corrupt-{label}",
                )
                projector = CorruptProjector(indices)
                with patch.object(
                    self.responses,
                    "_ResponsesSSEProjector",
                    return_value=projector,
                ):
                    with self.assertRaisesRegex(
                        StreamValidationError, expected
                    ):
                        await self.responses.create_response(
                            request,
                            getLogger(),
                            orchestrator,
                        )
                cast(
                    AsyncMock, orchestrator.sync_messages
                ).assert_not_awaited()

    def test_legacy_projection_redactor_and_completed_usage_paths(
        self,
    ) -> None:
        redactor = ModelVisibleServerProtocolTextRedactor()
        self.assertEqual(
            self.responses._model_visible_stream_deltas("safe", redactor),
            ("safe",),
        )
        terminal = project_canonical_stream_item(_terminal(10))
        events = self.responses._token_to_sse_events(terminal, 10)
        self.assertEqual(
            [(event.event, event.data["usage"]) for event in events],
            [("response.usage.completed", {})],
        )

    def test_sse_coalescing_rejects_mixed_sequence_ownership(self) -> None:
        data = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "a",
            "sequence_number": 1,
        }
        sequenced = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data=data,
            canonical_channel=StreamChannel.ANSWER,
        )
        unsequenced = self.responses._ResponsesSSEEvent(
            event="response.output_text.delta",
            data={
                key: value
                for key, value in data.items()
                if key != "sequence_number"
            },
            canonical_channel=StreamChannel.ANSWER,
        )
        self.assertFalse(sequenced.can_coalesce(unsequenced))
