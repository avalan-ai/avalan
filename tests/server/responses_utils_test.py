from avalan.entities import ReasoningToken, ToolCallToken, Token, TokenDetail
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
