from avalan.entities import ReasoningToken, ToolCallToken, Token, TokenDetail
from json import loads
from avalan.server.routers.responses import (
    ResponseState,
    _sse,
    _new_state,
    _switch_state,
    _token_to_sse,
)
from unittest import TestCase


class ResponsesUtilsTestCase(TestCase):
    def test_token_to_sse_formats_tokens(self) -> None:
        rt = ReasoningToken(token="r")
        tc = ToolCallToken(token="t")
        tok = Token(token="a")
        detail = TokenDetail(token="b")

        self.assertIn(
            "response.reasoning_text.delta",
            _token_to_sse(rt, 0)[0],
        )
        self.assertIn(
            '"delta":"r"',
            _token_to_sse(rt, 0)[0],
        )
        self.assertIn(
            "response.custom_tool_call_input.delta",
            _token_to_sse(tc, 1)[0],
        )
        self.assertIn(
            '"delta":"t"',
            _token_to_sse(tc, 1)[0],
        )
        self.assertIn(
            "response.output_text.delta",
            _token_to_sse(tok, 2)[0],
        )
        self.assertIn(
            '"delta":"a"',
            _token_to_sse(tok, 2)[0],
        )
        self.assertIn(
            '"delta":"b"',
            _token_to_sse(detail, 3)[0],
        )

    def test_token_to_sse_handles_tool_result(self) -> None:
        from avalan.entities import ToolCall, ToolCallResult
        from avalan.event import Event, EventType

        call = ToolCall(id="c1", name="t", arguments={"p": 1})
        result = ToolCallResult(
            id="c1", call=call, name="t", arguments={"p": 1}, result={"v": 2}
        )
        event = Event(type=EventType.TOOL_RESULT, payload={"result": result})

        events = _token_to_sse(event, 0)
        if len(events) == 2:
            self.assertIn("response.custom_tool_call_input.call", events[0])
            delta_event = events[1]
        else:
            self.assertEqual(len(events), 1)
            delta_event = events[0]
        self.assertIn("response.function_call_arguments.delta", delta_event)
        data = loads(delta_event.split("data: ")[1])
        self.assertEqual(data["delta"], '{"v": 2}')

    def test_switch_state_generates_events(self) -> None:
        state = _new_state(ReasoningToken(token="r"))
        events = _switch_state(None, state, None, None)
        self.assertEqual(state, ResponseState.REASONING)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            ["response.output_item.added", "response.content_part.added"],
        )

        new_state = _new_state(ToolCallToken(token="t"))
        events = _switch_state(state, new_state, None, None)
        self.assertEqual(new_state, ResponseState.TOOL_CALLING)
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
        new_state = _new_state("answer")
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

    def test_sse_formats_event_and_data(self) -> None:
        result = _sse("test.event", {"a": 1})
        self.assertEqual(result, 'event: test.event\ndata: {"a":1}\n\n')
