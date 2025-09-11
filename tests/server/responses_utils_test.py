import json

from avalan.entities import (
    ReasoningToken,
    ToolCall,
    ToolCallToken,
    Token,
    TokenDetail,
)
from avalan.server.routers.responses import (
    ResponseState,
    _sse,
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
            _token_to_sse(rt, 0),
        )
        self.assertIn(
            '"delta":"r"',
            _token_to_sse(rt, 0),
        )
        self.assertIn(
            "response.custom_tool_call_input.delta",
            _token_to_sse(tc, 1),
        )
        self.assertIn(
            '"delta":"t"',
            _token_to_sse(tc, 1),
        )
        self.assertIn(
            "response.output_text.delta",
            _token_to_sse(tok, 2),
        )
        self.assertIn(
            '"delta":"a"',
            _token_to_sse(tok, 2),
        )
        self.assertIn(
            '"delta":"b"',
            _token_to_sse(detail, 3),
        )

    def test_switch_state_generates_events(self) -> None:
        state, events = _switch_state(None, ReasoningToken(token="r"))
        self.assertEqual(state, ResponseState.REASONING)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            ["response.output_item.added", "response.content_part.added"],
        )

        state, events = _switch_state(state, ToolCallToken(token="t"))
        self.assertEqual(state, ResponseState.TOOL_CALLING)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            [
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
            ],
        )

        state, events = _switch_state(state, "answer")
        self.assertEqual(state, ResponseState.ANSWERING)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            [
                "response.custom_tool_call_input.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )

        state, events = _switch_state(state, None)
        self.assertIsNone(state)
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

    def test_tool_call_events_include_item_details(self) -> None:
        call = ToolCall(id="t1", name="pkg.func", arguments={"p": 1})
        token = ToolCallToken(token="{", call=call)

        state, events = _switch_state(None, token, call)
        self.assertEqual(state, ResponseState.TOOL_CALLING)
        names = [e.split("\n")[0].split(": ")[1] for e in events]
        self.assertEqual(
            names,
            ["response.output_item.added", "response.content_part.added"],
        )
        start_data = json.loads(events[0].split("\n")[1][len("data: ") :])
        self.assertEqual(start_data["item"]["custom_tool_call"], {"id": "t1"})
        part_data = json.loads(events[1].split("\n")[1][len("data: ") :])
        self.assertEqual(part_data["part"], {"type": "input_text"})

        state, events = _switch_state(state, "next", call)
        self.assertEqual(state, ResponseState.ANSWERING)
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
        done_data = json.loads(events[2].split("\n")[1][len("data: ") :])
        self.assertEqual(done_data["item"]["type"], "function_call")
        self.assertEqual(done_data["item"]["id"], "t1")
        self.assertEqual(done_data["item"]["name"], "pkg.func")
        self.assertEqual(done_data["item"]["arguments"], {"p": 1})
