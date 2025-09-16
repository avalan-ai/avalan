from dataclasses import dataclass
from json import loads
from unittest import TestCase

from avalan.entities import (
    ReasoningToken,
    ToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    Token,
    TokenDetail,
)
from avalan.event import Event, EventType
from avalan.server.routers.responses import (
    ResponseState,
    _new_state,
    _sse,
    _switch_state,
    _token_to_sse,
    _tool_call_event_item,
)


class ResponsesUtilsTestCase(TestCase):
    def test_token_to_sse_formats_tokens(self) -> None:
        rt = ReasoningToken(token="r")
        tc = ToolCallToken(token="t")
        tok = Token(token="a")
        detail = TokenDetail(token="b")

        rt_event = _token_to_sse(rt, 0)[0]
        self.assertIn("response.reasoning_text.delta", rt_event)
        self.assertEqual(loads(rt_event.split("data: ")[1])["delta"], "r")

        tc_event = _token_to_sse(tc, 1)[0]
        self.assertIn("response.custom_tool_call_input.delta", tc_event)
        self.assertEqual(loads(tc_event.split("data: ")[1])["delta"], "t")

        tok_event = _token_to_sse(tok, 2)[0]
        self.assertIn("response.output_text.delta", tok_event)
        self.assertEqual(loads(tok_event.split("data: ")[1])["delta"], "a")

        detail_event = _token_to_sse(detail, 3)[0]
        self.assertEqual(loads(detail_event.split("data: ")[1])["delta"], "b")

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
        self.assertEqual(data["error"], '"boom"')

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

    def test_token_to_sse_handles_tool_call_token_with_call(self) -> None:
        call = ToolCall(id="c4", name="adder", arguments={"x": 1})
        token = ToolCallToken(token="ignored", call=call)

        events = _token_to_sse(token, 3)
        self.assertEqual(len(events), 1)
        data = loads(events[0].split("data: ")[1])
        self.assertEqual(data["id"], "c4")
        delta = loads(data["delta"])
        self.assertEqual(delta["arguments"], {"x": 1})

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

    def test_switch_state_handles_new_tool_call_id(self) -> None:
        events = _switch_state(
            ResponseState.TOOL_CALLING,
            ResponseState.TOOL_CALLING,
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

    def test_sse_formats_event_and_data(self) -> None:
        result = _sse("test.event", {"a": 1})
        self.assertEqual(result, 'event: test.event\ndata: {"a": 1}\n\n')
