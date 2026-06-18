from json import loads
from unittest import TestCase

from avalan.entities import Token, ToolCall
from avalan.event import Event, EventType
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    project_canonical_stream_item,
    stream_channel_for_kind,
)
from avalan.server.routers.responses import (
    _RESPONSE_SSE_CONTENT_INDEX_FIELDS,
    _canonical_item_to_sse,
    _function_call_arguments_done,
    _response_projection_state,
    _response_sse_delta_data,
    _response_sse_indexed_data,
    _ResponsesSSEEvent,
    _ResponsesSSEItemState,
    _ResponsesSSEProjectionAdapter,
    _ResponsesSSEStreamEnvelope,
    _stream_tool_call_protocol_id,
    _switch_state,
    _terminal_response_events,
    _token_to_sse,
    _token_to_sse_events,
)
from avalan.server.sse import sse_message
from avalan.types import LooseJsonValue
from avalan.utils import to_json


def _canonical_stream_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    tool_call_id: str | None = None,
    data: LooseJsonValue | None = None,
    usage: LooseJsonValue | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    metadata: dict[str, LooseJsonValue] | None = None,
) -> CanonicalStreamItem:
    correlation = (
        StreamItemCorrelation(tool_call_id=tool_call_id)
        if tool_call_id is not None
        else StreamItemCorrelation()
    )
    return CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        correlation=correlation,
        text_delta=text_delta,
        data=data,
        usage=usage,
        terminal_outcome=terminal_outcome,
        metadata={} if metadata is None else metadata,
    )


def _canonical_projection(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    tool_call_id: str | None = None,
    data: LooseJsonValue | None = None,
    usage: LooseJsonValue | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    metadata: dict[str, LooseJsonValue] | None = None,
) -> StreamConsumerProjection:
    return project_canonical_stream_item(
        _canonical_stream_item(
            kind,
            sequence,
            text_delta=text_delta,
            tool_call_id=tool_call_id,
            data=data,
            usage=usage,
            terminal_outcome=terminal_outcome,
            metadata=metadata,
        )
    )


class ResponsesUtilsTestCase(TestCase):
    def test_token_to_sse_formats_canonical_projections(self) -> None:
        reasoning_projection = _canonical_projection(
            StreamItemKind.REASONING_DELTA,
            0,
            text_delta="r",
        )
        tool_call_projection = _canonical_projection(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            1,
            text_delta="t",
            tool_call_id="call-raw",
        )
        answer_projection = _canonical_projection(
            StreamItemKind.ANSWER_DELTA,
            2,
            text_delta="a",
        )
        detail_projection = _canonical_projection(
            StreamItemKind.ANSWER_DELTA,
            3,
            text_delta="b",
            metadata={
                "token_id": 9,
                "probability": 0.5,
                "step": 3,
            },
        )

        rt_event = _token_to_sse(reasoning_projection, 0)[0]
        self.assertIn("response.reasoning_text.delta", rt_event)
        self.assertEqual(loads(rt_event.split("data: ")[1])["delta"], "r")

        tc_event = _token_to_sse(tool_call_projection, 1)[0]
        self.assertIn("response.custom_tool_call_input.delta", tc_event)
        self.assertEqual(loads(tc_event.split("data: ")[1])["delta"], "t")

        tok_event = _token_to_sse(answer_projection, 2)[0]
        self.assertIn("response.output_text.delta", tok_event)
        self.assertEqual(loads(tok_event.split("data: ")[1])["delta"], "a")

        detail_event = _token_to_sse(detail_projection, 3)[0]
        self.assertEqual(loads(detail_event.split("data: ")[1])["delta"], "b")
        self.assertEqual(
            detail_projection.metadata,
            {"token_id": 9, "probability": 0.5, "step": 3},
        )

    def test_token_to_sse_legacy_rejection_unprojected_token(self) -> None:
        legacy_rejection_token = Token(token="raw")

        with self.assertRaises(AssertionError):
            _token_to_sse(legacy_rejection_token, 0)  # type: ignore[arg-type]

    def test_stream_tool_call_protocol_id_ignores_non_tool_projection(
        self,
    ) -> None:
        self.assertIsNone(
            _stream_tool_call_protocol_id(
                _canonical_projection(
                    StreamItemKind.ANSWER_DELTA,
                    0,
                    text_delta="a",
                )
            )
        )

    def test_token_to_sse_legacy_rejection_tool_events(self) -> None:
        legacy_rejection_event = Event(type=EventType.TOOL_RESULT, payload={})

        with self.assertRaises(AssertionError):
            _token_to_sse(legacy_rejection_event, 0)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            _token_to_sse_events(legacy_rejection_event, 0)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            _response_projection_state(legacy_rejection_event)  # type: ignore[arg-type]

    def test_sse_event_coalesces_only_compatible_deltas(self) -> None:
        first = _ResponsesSSEEvent(
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
        second = _ResponsesSSEEvent(
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
        different_event = _ResponsesSSEEvent(
            event="response.reasoning_text.delta",
            data={
                "type": "response.reasoning_text.delta",
                "delta": "b",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 2,
            },
            canonical_channel=StreamChannel.REASONING,
        )
        different_key = _ResponsesSSEEvent(
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

        self.assertTrue(first.can_coalesce(second))
        merged = first.coalesce(second)
        self.assertEqual(merged.data["delta"], "ab")
        self.assertEqual(merged.data["sequence_number"], 2)
        self.assertFalse(first.can_coalesce(different_event))
        self.assertFalse(first.can_coalesce(different_key))

    def test_sse_event_does_not_coalesce_structured_tool_output(self) -> None:
        first = _ResponsesSSEEvent(
            event="response.tool_execution.output",
            data={
                "type": "response.tool_execution.output",
                "id": "call-1",
                "delta": "out",
                "data": {"category": "stdout"},
                "sequence_number": 1,
            },
            correlation_key="call-1",
        )
        second = _ResponsesSSEEvent(
            event="response.tool_execution.output",
            data={
                "type": "response.tool_execution.output",
                "id": "call-1",
                "delta": "err",
                "data": {"category": "stderr"},
                "sequence_number": 2,
            },
            correlation_key="call-1",
        )

        self.assertFalse(first.can_coalesce(second))
        with self.assertRaises(AssertionError):
            first.coalesce(second)

    def test_response_stream_envelope_reuses_stable_provider_state(
        self,
    ) -> None:
        envelope = _ResponsesSSEStreamEnvelope(
            response_id="response-id",
            timestamp=7,
            model_id="model-id",
        )

        first = envelope.created_event()
        second = envelope.created_event()
        first.data["response"]["id"] = "changed"

        self.assertEqual(first.event, "response.created")
        self.assertEqual(second.data["response"]["id"], "response-id")
        self.assertEqual(second.data["response"]["created_at"], 7)
        self.assertEqual(second.data["response"]["model"], "model-id")

    def test_response_stream_envelope_rejects_invalid_state(self) -> None:
        with self.assertRaises(AssertionError):
            _ResponsesSSEStreamEnvelope(
                response_id=object(),
                timestamp=7,
                model_id="model-id",
            )
        with self.assertRaises(AssertionError):
            _ResponsesSSEStreamEnvelope(
                response_id="response-id",
                timestamp=True,
                model_id="model-id",
            )

    def test_response_sse_static_indexes_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            _RESPONSE_SSE_CONTENT_INDEX_FIELDS["output_index"] = 1  # type: ignore[index]

    def test_response_sse_delta_data_copies_static_indexes(self) -> None:
        first = _response_sse_delta_data(
            "response.output_text.delta",
            "a",
            1,
        )
        second = _response_sse_delta_data(
            "response.output_text.delta",
            "b",
            2,
        )

        first["output_index"] = 9
        first["delta"] = "changed"

        self.assertEqual(second["output_index"], 0)
        self.assertEqual(second["content_index"], 0)
        self.assertEqual(second["delta"], "b")
        self.assertEqual(
            list(second),
            [
                "type",
                "delta",
                "output_index",
                "content_index",
                "sequence_number",
            ],
        )

    def test_response_sse_delta_data_rejects_invalid_state(self) -> None:
        with self.assertRaises(AssertionError):
            _response_sse_delta_data("", "a", 0)
        with self.assertRaises(AssertionError):
            _response_sse_delta_data(
                "response.output_text.delta",
                object(),  # type: ignore[arg-type]
                0,
            )
        with self.assertRaises(AssertionError):
            _response_sse_delta_data(
                "response.output_text.delta",
                "a",
                True,
            )

    def test_response_sse_indexed_data_copies_static_indexes(self) -> None:
        output = _response_sse_indexed_data("response.output_item.done")
        content = _response_sse_indexed_data(
            "response.content_part.done",
            content_index=True,
        )

        output["output_index"] = 9

        self.assertEqual(content["output_index"], 0)
        self.assertEqual(content["content_index"], 0)
        self.assertEqual(
            list(content),
            ["type", "output_index", "content_index"],
        )

    def test_response_sse_indexed_data_rejects_invalid_state(self) -> None:
        with self.assertRaises(AssertionError):
            _response_sse_indexed_data("")

    def test_terminal_response_events_preserve_outcome(self) -> None:
        self.assertEqual(
            _terminal_response_events(StreamTerminalOutcome.COMPLETED)[
                0
            ].event,
            "response.completed",
        )
        self.assertEqual(
            _terminal_response_events(StreamTerminalOutcome.CANCELLED)[
                0
            ].event,
            "response.cancelled",
        )
        self.assertEqual(
            _terminal_response_events(StreamTerminalOutcome.ERRORED)[0].event,
            "response.failed",
        )
        self.assertEqual(
            _terminal_response_events(None)[0].event,
            "response.completed",
        )

    def test_terminal_response_events_preserve_error_data(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=5,
            kind=StreamItemKind.STREAM_ERRORED,
            channel=StreamChannel.CONTROL,
            data={"error_type": "RuntimeError", "message": "provider failed"},
            terminal_outcome=StreamTerminalOutcome.ERRORED,
        )

        event = _terminal_response_events(project_canonical_stream_item(item))[
            0
        ]

        self.assertEqual(event.event, "response.failed")
        self.assertEqual(event.data["sequence_number"], 5)
        self.assertEqual(
            event.data["error"],
            {"error_type": "RuntimeError", "message": "provider failed"},
        )

    def test_terminal_response_events_reject_non_terminal_projection(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )

        with self.assertRaises(AssertionError):
            _terminal_response_events(project_canonical_stream_item(item))

    def test_token_to_sse_maps_usage_items(self) -> None:
        update = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.USAGE_UPDATE,
            channel=StreamChannel.USAGE,
            usage={"input_tokens": 1},
        )
        completed = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.USAGE_COMPLETED,
            channel=StreamChannel.USAGE,
            usage={"total_tokens": 2},
        )

        update_data = loads(
            _token_to_sse(project_canonical_stream_item(update), 0)[0].split(
                "data: "
            )[1]
        )
        completed_data = loads(
            _token_to_sse(project_canonical_stream_item(completed), 1)[
                0
            ].split("data: ")[1]
        )

        self.assertEqual(update_data["type"], "response.usage.delta")
        self.assertEqual(update_data["usage"], {"input_tokens": 1})
        self.assertEqual(completed_data["type"], "response.usage.completed")
        self.assertEqual(completed_data["usage"], {"total_tokens": 2})

    def test_token_to_sse_maps_tool_execution_items(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-output"),
            text_delta="line",
            data={"category": "stdout"},
        )

        event = _token_to_sse(project_canonical_stream_item(item), 7)[0]
        data = loads(event.split("data: ")[1])

        self.assertIn("response.tool_execution.output", event)
        self.assertEqual(data["id"], "call-output")
        self.assertEqual(data["delta"], "line")
        self.assertEqual(data["data"], {"category": "stdout"})

    def test_token_to_sse_maps_stream_diagnostic_items(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_DIAGNOSTIC,
            channel=StreamChannel.CONTROL,
            text_delta="warning",
            data={"code": "stream.warning"},
        )

        event = _token_to_sse(project_canonical_stream_item(item), 10)[0]
        data = loads(event.split("data: ")[1])

        self.assertIn("response.diagnostic", event)
        self.assertEqual(data["delta"], "warning")
        self.assertEqual(data["data"], {"code": "stream.warning"})

    def test_canonical_item_to_sse_wraps_event_messages(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )

        event = _canonical_item_to_sse(
            project_canonical_stream_item(item),
            11,
        )[0]
        data = loads(event.split("data: ")[1])

        self.assertIn("response.output_text.delta", event)
        self.assertEqual(data["delta"], "answer")

    def test_token_to_sse_raises_on_unserializable_usage(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.USAGE_UPDATE,
            channel=StreamChannel.USAGE,
            usage=object(),  # type: ignore[arg-type]
        )

        with self.assertRaises(TypeError):
            _token_to_sse(project_canonical_stream_item(item), 0)

    def test_token_to_sse_handles_canonical_tool_call_with_call_data(
        self,
    ) -> None:
        call = ToolCall(id="c4", name="adder", arguments={"x": 1})
        projection = _canonical_projection(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            3,
            text_delta="ignored",
            tool_call_id=call.id,
            data={"name": call.name, "arguments": call.arguments},
        )

        events = _token_to_sse(projection, 3)
        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c4")
        delta = loads(data["delta"])
        self.assertEqual(delta["arguments"], {"x": 1})

    def test_token_to_sse_keeps_raw_canonical_tool_call_without_protocol_id(
        self,
    ) -> None:
        projection = _canonical_projection(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            3,
            text_delta="raw-input",
            tool_call_id="legacy-tool-call",
        )

        events = _token_to_sse(projection, 3)

        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["delta"], "raw-input")
        self.assertEqual(data["id"], "legacy-tool-call")

    def test_token_to_sse_uses_active_tool_call_id_for_raw_input(
        self,
    ) -> None:
        projection = _canonical_projection(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            3,
            text_delta="raw-input",
            tool_call_id="legacy-tool-call",
        )

        events = _token_to_sse_events(projection, 3, "c4")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].correlation_key, "legacy-tool-call")
        data = events[0].data
        self.assertEqual(data["delta"], "raw-input")
        self.assertEqual(data["id"], "legacy-tool-call")

    def test_token_to_sse_uses_canonical_projection_call_data(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-canonical"),
            text_delta='{"x":1}',
            data={"name": "math.add", "arguments": {"x": 1}},
        )

        projection = project_canonical_stream_item(item)
        events = _token_to_sse(projection, 8)

        self.assertEqual(len(events), 1)
        state = _response_projection_state(projection)
        self.assertEqual(
            state,
            _ResponsesSSEItemState(
                output_item_type="function_call",
                tool_call_id="call-canonical",
            ),
        )
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(
            data["type"], "response.function_call_arguments.delta"
        )
        self.assertEqual(data["id"], "call-canonical")
        self.assertEqual(data["sequence_number"], 8)
        self.assertEqual(
            loads(data["delta"]),
            {
                "id": "call-canonical",
                "name": "math.add",
                "arguments": {"x": 1},
            },
        )

    def test_token_to_sse_treats_malformed_projection_data_as_raw_input(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-raw"),
            text_delta="raw",
            data={"name": 1},
        )

        projection = project_canonical_stream_item(item)
        events = _token_to_sse(projection, 9)

        self.assertEqual(len(events), 1)
        state = _response_projection_state(projection)
        self.assertEqual(
            state,
            _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id="call-raw",
            ),
        )
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["type"], "response.custom_tool_call_input.delta")
        self.assertEqual(data["delta"], "raw")
        self.assertEqual(data["id"], "call-raw")

    def test_token_to_sse_ignores_canonical_non_delta_items(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )

        projection = project_canonical_stream_item(item)
        self.assertEqual(_token_to_sse(projection, 0), [])
        self.assertIsNone(_response_projection_state(projection))
        with self.assertRaises(AssertionError):
            _response_projection_state(object())  # type: ignore[arg-type]

    def test_switch_state_generates_events(self) -> None:
        state = _response_projection_state(
            _canonical_projection(
                StreamItemKind.REASONING_DELTA,
                0,
                text_delta="r",
            )
        )
        events = _switch_state(None, state)
        self.assertEqual(
            state,
            _ResponsesSSEItemState(
                output_item_type="reasoning_text",
                content_part_type="reasoning_text",
            ),
        )
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            ["response.output_item.added", "response.content_part.added"],
        )

        new_state = _response_projection_state(
            _canonical_projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta="t",
                tool_call_id="legacy-tool-call",
            )
        )
        events = _switch_state(state, new_state)
        self.assertEqual(
            new_state,
            _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id="legacy-tool-call",
            ),
        )
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            [
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )

        state = new_state
        new_state = _response_projection_state(
            _canonical_projection(
                StreamItemKind.ANSWER_DELTA,
                2,
                text_delta="answer",
            )
        )
        events = _switch_state(state, new_state)
        self.assertEqual(
            new_state,
            _ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="output_text",
            ),
        )
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            [
                "response.custom_tool_call_input.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )

        state = new_state
        new_state = _response_projection_state(None)
        events = _switch_state(state, new_state)
        self.assertIsNone(new_state)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            [
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
            ],
        )

    def test_switch_state_handles_new_tool_call_id(self) -> None:
        events = _switch_state(
            _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id="old",
            ),
            _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id="new",
            ),
        )

        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            [
                "response.custom_tool_call_input.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )

        data_items = [loads(e.split("data: ")[1]) for e in events]
        self.assertEqual(data_items[0]["id"], "old")
        self.assertEqual(data_items[1]["part"]["id"], "old")
        self.assertEqual(data_items[2]["item"]["id"], "old")
        self.assertEqual(data_items[3]["item"]["id"], "new")
        self.assertEqual(data_items[4]["part"]["id"], "new")

    def test_switch_state_keeps_same_tool_call_id_open(self) -> None:
        state = _ResponsesSSEItemState(
            output_item_type="custom_tool_call_input",
            content_part_type="input_text",
            tool_call_id="same",
        )
        events = _switch_state(
            state,
            state,
        )

        self.assertEqual(events, [])

    def test_projection_adapter_preserves_distinct_tool_call_id(
        self,
    ) -> None:
        projection = _canonical_projection(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            1,
            text_delta="a",
            tool_call_id="call-1",
        )
        adapter = _ResponsesSSEProjectionAdapter()

        open_events = adapter.switch(projection)
        carried_events = adapter.switch(
            _canonical_projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                2,
                text_delta="b",
                tool_call_id="legacy-tool-call",
            )
        )

        self.assertEqual(
            [event.split("\n")[0].split(": ")[1] for event in open_events],
            ["response.output_item.added", "response.content_part.added"],
        )
        self.assertEqual(
            [event.split("\n")[0].split(": ")[1] for event in carried_events],
            [
                "response.custom_tool_call_input.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )
        self.assertEqual(adapter.active_tool_call_id, "legacy-tool-call")
        self.assertEqual(
            adapter.state,
            _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id="legacy-tool-call",
            ),
        )

    def test_switch_state_uses_function_call_framing(self) -> None:
        events = _switch_state(
            _ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="output_text",
            ),
            _ResponsesSSEItemState(
                output_item_type="function_call",
                tool_call_id="call-1",
            ),
        )
        names = [event.split("\n")[0].split(": ")[1] for event in events]
        self.assertEqual(
            names,
            [
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
            ],
        )
        added_data = loads(events[-1].split("data: ")[1])
        self.assertEqual(
            added_data["item"], {"type": "function_call", "id": "call-1"}
        )

        done_events = _switch_state(
            _ResponsesSSEItemState(
                output_item_type="function_call",
                tool_call_id="call-1",
            ),
            None,
        )
        done_names = [
            event.split("\n")[0].split(": ")[1] for event in done_events
        ]
        self.assertEqual(
            done_names,
            [
                "response.function_call_arguments.done",
                "response.output_item.done",
            ],
        )
        self.assertNotIn("response.custom_tool_call_input.done", done_names)

    def test_function_call_arguments_done_formats_optional_id(self) -> None:
        with_id = loads(
            _function_call_arguments_done("call-1").split("data: ")[1]
        )
        without_id = loads(_function_call_arguments_done().split("data: ")[1])

        self.assertEqual(
            with_id,
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "id": "call-1",
            },
        )
        self.assertEqual(
            without_id,
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
            },
        )

    def test_response_sse_item_state_identifies_tool_items(self) -> None:
        self.assertTrue(
            _ResponsesSSEItemState(
                output_item_type="function_call"
            ).is_tool_call
        )
        self.assertTrue(
            _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input"
            ).is_tool_call
        )
        self.assertFalse(
            _ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="output_text",
            ).is_tool_call
        )

    def test_response_sse_item_state_rejects_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            _ResponsesSSEItemState(output_item_type="bad")
        with self.assertRaises(AssertionError):
            _ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="bad",
            )
        with self.assertRaises(AssertionError):
            _ResponsesSSEItemState(
                output_item_type="function_call",
                tool_call_id="",
            )

    def test_sse_formats_event_and_data(self) -> None:
        result = sse_message(to_json({"a": 1}), event="test.event")
        self.assertEqual(result, 'event: test.event\ndata: {"a": 1}\n\n')

    def test_sse_message_handles_empty_payload(self) -> None:
        result = sse_message("", event="empty")
        self.assertEqual(result, "event: empty\ndata: \n\n")
