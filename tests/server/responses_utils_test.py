from dataclasses import dataclass
from datetime import datetime
from json import loads
from unittest import TestCase

from avalan.entities import (
    ReasoningToken,
    Token,
    TokenDetail,
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
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.routers.responses import (
    ResponseState,
    _canonical_item_to_sse,
    _function_call_arguments_done,
    _is_tool_response_state,
    _new_state,
    _ResponsesSSEEvent,
    _stream_projection,
    _switch_state,
    _terminal_projection,
    _terminal_response_events,
    _token_to_sse,
    _token_to_sse_events,
    _tool_call_event_item,
)
from avalan.server.sse import sse_message
from avalan.utils import to_json


class ResponsesUtilsTestCase(TestCase):
    def test_token_to_sse_formats_tokens(self) -> None:
        rt = ReasoningToken(token="r")
        tc = ToolCallToken(token="t")
        tok = Token(token="a")
        detail = TokenDetail(token="b")

        rt_event = _token_to_sse(_stream_projection(rt, 0), 0)[0]
        self.assertIn("response.reasoning_text.delta", rt_event)
        self.assertEqual(loads(rt_event.split("data: ")[1])["delta"], "r")

        tc_event = _token_to_sse(_stream_projection(tc, 1), 1)[0]
        self.assertIn("response.custom_tool_call_input.delta", tc_event)
        self.assertEqual(loads(tc_event.split("data: ")[1])["delta"], "t")

        tok_event = _token_to_sse(_stream_projection(tok, 2), 2)[0]
        self.assertIn("response.output_text.delta", tok_event)
        self.assertEqual(loads(tok_event.split("data: ")[1])["delta"], "a")

        detail_event = _token_to_sse(_stream_projection(detail, 3), 3)[0]
        self.assertEqual(loads(detail_event.split("data: ")[1])["delta"], "b")

    def test_token_to_sse_rejects_unprojected_tokens(self) -> None:
        with self.assertRaises(AssertionError):
            _token_to_sse(Token(token="raw"), 0)  # type: ignore[arg-type]

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
        )

        self.assertTrue(first.can_coalesce(second))
        merged = first.coalesce(second)
        self.assertEqual(merged.data["delta"], "ab")
        self.assertEqual(merged.data["sequence_number"], 2)
        self.assertFalse(first.can_coalesce(different_event))
        self.assertFalse(first.can_coalesce(different_key))

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

        event = _terminal_response_events(_stream_projection(item, 5))[0]

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
            _terminal_response_events(_stream_projection(item, 0))

    def test_terminal_projection_returns_none_without_terminal(self) -> None:
        self.assertIsNone(_terminal_projection(CanonicalStreamAccumulator()))

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
            _token_to_sse(_stream_projection(update, 0), 0)[0].split("data: ")[
                1
            ]
        )
        completed_data = loads(
            _token_to_sse(_stream_projection(completed, 1), 1)[0].split(
                "data: "
            )[1]
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

        event = _token_to_sse(_stream_projection(item, 7), 7)[0]
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

        event = _token_to_sse(_stream_projection(item, 10), 10)[0]
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

        event = _canonical_item_to_sse(_stream_projection(item, 11), 11)[0]
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
            _token_to_sse(_stream_projection(item, 0), 0)

    def test_token_to_sse_handles_tool_result(self) -> None:
        call = ToolCall(id="c1", name="t", arguments={"p": 1})
        result = ToolCallResult(
            id="c1", call=call, name="t", arguments={"p": 1}, result={"v": 2}
        )
        event = Event(type=EventType.TOOL_RESULT, payload={"result": result})

        events = _token_to_sse(event, 0)
        self.assertEqual(len(events), 1)
        self.assertIn("response.function_call_arguments.delta", events[0])
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c1")
        self.assertEqual(data["result"], '{"v": 2}')

    def test_token_to_sse_preserves_falsy_tool_results(self) -> None:
        cases: tuple[tuple[object, str | None], ...] = (
            (0, "0"),
            (False, "false"),
            ("", '""'),
            (None, None),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                call = ToolCall(id="c1", name="t", arguments={})
                result = ToolCallResult(
                    id="c1",
                    call=call,
                    name="t",
                    arguments={},
                    result=value,
                )
                event = Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": result},
                )

                events = _token_to_sse(event, 0)
                data = loads(events[0].split("data: ")[1])

                self.assertEqual(data["result"], expected)
                self.assertEqual(loads(data["delta"])["result"], expected)

    def test_token_to_sse_handles_tool_result_error(self) -> None:
        call = ToolCall(id="c2", name="t", arguments={})
        error = ToolCallError(
            id="c2",
            name="t",
            arguments={},
            call=call,
            error=RuntimeError("boom"),
            message="boom",
        )
        event = Event(type=EventType.TOOL_RESULT, payload={"result": error})

        events = _token_to_sse(event, 1)
        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c2")
        self.assertEqual(
            data["error"],
            {"type": "RuntimeError", "message": "Tool call failed."},
        )
        delta = loads(data["delta"])
        self.assertEqual(
            delta["error"],
            {"type": "RuntimeError", "message": "Tool call failed."},
        )
        self.assertNotIn("boom", events[0])

    def test_token_to_sse_handles_tool_diagnostic(self) -> None:
        call = ToolCall(id="c-diagnostic", name="missing", arguments={})
        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            call_id=call.id,
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"call": call, "diagnostic": diagnostic},
        )

        events = _token_to_sse(event, 3)

        self.assertEqual(len(events), 1)
        self.assertIn("response.tool_call_diagnostic.delta", events[0])
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c-diagnostic")
        self.assertEqual(data["diagnostic"]["code"], "tool.unknown")
        delta = loads(data["delta"])
        self.assertEqual(delta["diagnostic"]["call_id"], "c-diagnostic")

    def test_tool_call_event_item_handles_tool_result_diagnostic(
        self,
    ) -> None:
        call = ToolCall(id="c-result-diagnostic", name="missing", arguments={})
        diagnostic = ToolCallDiagnostic(
            id="diag-result",
            call_id=call.id,
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        event = Event(
            type=EventType.TOOL_RESULT,
            payload={"call": call, "result": diagnostic},
        )

        item = _tool_call_event_item(event)

        self.assertEqual(item["id"], "c-result-diagnostic")
        self.assertEqual(item["diagnostic"]["code"], "tool.unknown")

    def test_tool_call_event_item_handles_diagnostic_timing_fields(
        self,
    ) -> None:
        diagnostic = ToolCallDiagnostic(
            id="diag-timing",
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
            started_at=datetime(2026, 1, 1, 12, 0, 0),
            finished_at=datetime(2026, 1, 1, 12, 0, 1),
            duration_ms=1000.0,
        )
        event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"diagnostics": [diagnostic]},
        )

        item = _tool_call_event_item(event)

        self.assertEqual(item["id"], "diag-timing")
        self.assertEqual(item["name"], "missing")
        self.assertEqual(
            item["diagnostic"]["started_at"], "2026-01-01T12:00:00"
        )
        self.assertEqual(
            item["diagnostic"]["finished_at"], "2026-01-01T12:00:01"
        )
        self.assertEqual(item["diagnostic"]["duration_ms"], 1000.0)

    def test_token_to_sse_ignores_malformed_tool_diagnostic(self) -> None:
        event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"diagnostics": ["bad"]},
        )

        self.assertEqual(_token_to_sse(event, 4), [])

    def test_token_to_sse_handles_tool_result_without_payload(self) -> None:
        call = ToolCall(id="c3", name="tool", arguments={})
        result = ToolCallResult(
            id="c3",
            name="tool",
            arguments={},
            call=call,
            result=None,
        )
        event = Event(type=EventType.TOOL_RESULT, payload={"result": result})

        events = _token_to_sse(event, 2)
        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertIsNone(data["result"])

    def test_token_to_sse_handles_current_none_tool_result(self) -> None:
        call = ToolCall(id="c-none", name="tool", arguments={"x": 1})
        event = Event(
            type=EventType.TOOL_RESULT,
            payload={0: call, "result": None},
        )

        events = _token_to_sse(event, 5)

        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c-none")
        self.assertEqual(data["name"], "tool")
        self.assertEqual(data["arguments"], {"x": 1})
        self.assertIsNone(data["result"])

    def test_tool_call_event_item_ignores_missing_tool_call(self) -> None:
        event = Event(
            type=EventType.TOOL_PROCESS,
            payload={"call": None},
        )

        self.assertIsNone(_tool_call_event_item(event))

    def test_tool_call_event_item_ignores_unexpected_payload(
        self,
    ) -> None:
        event = Event(type=EventType.TOOL_PROCESS, payload="bad")

        self.assertIsNone(_tool_call_event_item(event))

    def test_token_to_sse_handles_tool_call_token_with_call(self) -> None:
        call = ToolCall(id="c4", name="adder", arguments={"x": 1})
        token = ToolCallToken(token="ignored", call=call)

        events = _token_to_sse(_stream_projection(token, 3), 3)
        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c4")
        delta = loads(data["delta"])
        self.assertEqual(delta["arguments"], {"x": 1})

    def test_token_to_sse_keeps_raw_tool_call_token_without_protocol_id(
        self,
    ) -> None:
        token = ToolCallToken(token="raw-input")

        events = _token_to_sse(_stream_projection(token, 3), 3)

        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["delta"], "raw-input")
        self.assertNotIn("id", data)

    def test_token_to_sse_uses_active_tool_call_id_for_raw_input(
        self,
    ) -> None:
        token = ToolCallToken(token="raw-input")

        events = _token_to_sse_events(_stream_projection(token, 3), 3, "c4")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].correlation_key, "c4")
        data = events[0].data
        self.assertEqual(data["delta"], "raw-input")
        self.assertEqual(data["id"], "c4")

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

        events = _token_to_sse(_stream_projection(item, 8), 8)

        self.assertEqual(len(events), 1)
        self.assertEqual(
            _new_state(_stream_projection(item, 8)),
            ResponseState.FUNCTION_CALLING,
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

        events = _token_to_sse(_stream_projection(item, 9), 9)

        self.assertEqual(len(events), 1)
        self.assertEqual(
            _new_state(_stream_projection(item, 9)),
            ResponseState.CUSTOM_TOOL_CALLING,
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

        projection = _stream_projection(item, 0)
        self.assertEqual(_token_to_sse(projection, 0), [])
        self.assertIsNone(_new_state(projection))
        with self.assertRaises(AssertionError):
            _new_state(object())  # type: ignore[arg-type]

    def test_switch_state_generates_events(self) -> None:
        state = _new_state(_stream_projection(ReasoningToken(token="r"), 0))
        events = _switch_state(None, state, None, None)
        self.assertEqual(state, ResponseState.REASONING)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            ["response.output_item.added", "response.content_part.added"],
        )

        new_state = _new_state(_stream_projection(ToolCallToken(token="t"), 1))
        events = _switch_state(state, new_state, None, None)
        self.assertEqual(new_state, ResponseState.CUSTOM_TOOL_CALLING)
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
        new_state = _new_state(_stream_projection("answer", 2))
        events = _switch_state(state, new_state, None, None)
        self.assertEqual(new_state, ResponseState.ANSWERING)
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
        new_state = _new_state(None)
        events = _switch_state(state, new_state, None, None)
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
            ResponseState.CUSTOM_TOOL_CALLING,
            ResponseState.CUSTOM_TOOL_CALLING,
            "old",
            "new",
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
        events = _switch_state(
            ResponseState.CUSTOM_TOOL_CALLING,
            ResponseState.CUSTOM_TOOL_CALLING,
            "same",
            "same",
        )

        self.assertEqual(events, [])

    def test_switch_state_uses_function_call_framing(self) -> None:
        events = _switch_state(
            ResponseState.ANSWERING,
            ResponseState.FUNCTION_CALLING,
            None,
            "call-1",
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
            ResponseState.FUNCTION_CALLING,
            None,
            "call-1",
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

    def test_is_tool_response_state_classifies_tool_states(self) -> None:
        self.assertTrue(
            _is_tool_response_state(ResponseState.FUNCTION_CALLING)
        )
        self.assertTrue(
            _is_tool_response_state(ResponseState.CUSTOM_TOOL_CALLING)
        )
        self.assertFalse(_is_tool_response_state(ResponseState.ANSWERING))
        self.assertFalse(_is_tool_response_state(None))

    def test_tool_call_event_item_handles_custom_result(self) -> None:
        call = ToolCall(id="c5", name="calc", arguments={"a": 2})

        @dataclass(frozen=True, slots=True)
        class DummyResult:
            call: ToolCall
            payload: dict[str, str]

        result = DummyResult(call=call, payload={"status": "ok"})
        event = Event(type=EventType.TOOL_RESULT, payload={"result": result})

        item = _tool_call_event_item(event)
        self.assertEqual(item["id"], "c5")
        self.assertIs(item["result"], result)

        events = _token_to_sse(event, 4)
        data = loads(events[0].split("data: ")[1])
        delta = loads(data["delta"])
        result_payload = loads(delta["result"])
        self.assertEqual(result_payload["payload"], {"status": "ok"})

    def test_tool_call_event_item_supports_indexed_mapping_payload(
        self,
    ) -> None:
        call = ToolCall(id="c6", name="calc", arguments={"n": 1})

        class IndexedPayload(dict[int, ToolCall]):
            pass

        event = Event(
            type=EventType.TOOL_PROCESS,
            payload=IndexedPayload({0: call}),
        )

        item = _tool_call_event_item(event)
        self.assertEqual(item["id"], "c6")
        self.assertEqual(item["name"], "calc")

    def test_sse_formats_event_and_data(self) -> None:
        result = sse_message(to_json({"a": 1}), event="test.event")
        self.assertEqual(result, 'event: test.event\ndata: {"a": 1}\n\n')

    def test_sse_message_handles_empty_payload(self) -> None:
        result = sse_message("", event="empty")
        self.assertEqual(result, "event: empty\ndata: \n\n")
