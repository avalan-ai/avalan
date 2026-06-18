from asyncio import CancelledError
from collections.abc import AsyncIterator
from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallToken,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamGoldenTrace,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    accumulate_canonical_stream_items,
    assemble_tool_observations,
    canonical_item_from_token,
)

_STREAM_SESSION_ID = "response-stream"
_RUN_ID = "response-run"
_TURN_ID = "response-turn"


def _response(
    output_fn: object,
) -> TextGenerationResponse:
    settings = GenerationSettings()
    return TextGenerationResponse(
        output_fn,  # type: ignore[arg-type]
        logger=getLogger("response-golden-trace"),
        use_async_generator=True,
        generation_settings=settings,
        settings=settings,
    )


def _control_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=kind,
        channel=StreamChannel.CONTROL,
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
    )


def _answer_done(sequence: int) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DONE,
        channel=StreamChannel.ANSWER,
    )


def _answer_delta(sequence: int, text_delta: str) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        text_delta=text_delta,
    )


def _reasoning_done(sequence: int) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=StreamItemKind.REASONING_DONE,
        channel=StreamChannel.REASONING,
        visibility=StreamVisibility.PRIVATE,
    )


def _reasoning_delta(sequence: int, text_delta: str) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=StreamItemKind.REASONING_DELTA,
        channel=StreamChannel.REASONING,
        visibility=StreamVisibility.PRIVATE,
        text_delta=text_delta,
    )


def _usage_item(sequence: int, usage: object) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=StreamItemKind.USAGE_COMPLETED,
        channel=StreamChannel.USAGE,
        usage=usage,  # type: ignore[arg-type]
    )


def _golden_trace(
    name: str,
    *items: CanonicalStreamItem,
) -> StreamGoldenTrace:
    return StreamGoldenTrace(name=name, items=tuple(items))


def _semantic_trace(
    items: tuple[CanonicalStreamItem, ...],
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for item in items:
        semantic: dict[str, object] = {
            "sequence": item.sequence,
            "kind": item.kind,
            "channel": item.channel,
            "visibility": item.visibility,
        }
        correlation = item.correlation.to_trace_dict()
        if correlation:
            semantic["correlation"] = correlation
        if item.text_delta is not None:
            semantic["text_delta"] = item.text_delta
        if item.data is not None:
            semantic["data"] = item.data
        if item.usage is not None:
            semantic["usage"] = item.usage
        if item.terminal_outcome is not None:
            semantic["terminal_outcome"] = item.terminal_outcome
        if item.metadata:
            semantic["metadata"] = item.metadata
        result.append(semantic)
    return tuple(result)


def _response_from_trace(trace: StreamGoldenTrace) -> TextGenerationResponse:
    async def gen() -> AsyncIterator[CanonicalStreamItem]:
        for item in trace.items:
            yield item

    return _response(lambda **_: gen())


def _legacy_rejection_response_from_raw_stream(
    *items: object,
) -> TextGenerationResponse:
    raw_items = items or ("legacy-only",)

    async def gen() -> AsyncIterator[object]:
        for item in raw_items:
            yield item

    return _response(lambda **_: gen())


def _tool_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    tool_call_id: str,
    text_delta: str | None = None,
    data: object | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=kind,
        channel=(
            StreamChannel.TOOL_EXECUTION
            if kind.name.startswith("TOOL_EXECUTION")
            else StreamChannel.TOOL_CALL
        ),
        correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
    )


def _protocol_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    protocol_item_id: str,
    channel: StreamChannel | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    metadata: dict[str, object] | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=kind,
        channel=channel
        or (
            StreamChannel.TOOL_EXECUTION
            if kind.name.startswith("TOOL_EXECUTION")
            else StreamChannel.CONTROL
        ),
        correlation=StreamItemCorrelation(
            protocol_item_id=protocol_item_id,
            tool_call_id=(
                "call-protocol" if kind.name.startswith("TOOL_") else None
            ),
        ),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
        visibility=visibility,
    )


def _legacy_fixture_trace_from_tokens(
    name: str,
    tokens: tuple[Token | str, ...],
    *,
    terminal_kind: StreamItemKind = StreamItemKind.STREAM_COMPLETED,
    usage: object | None = None,
) -> StreamGoldenTrace:
    items = [_control_item(StreamItemKind.STREAM_STARTED, 0)]
    reasoning_started = False
    tool_call_data: dict[str, dict[str, object] | None] = {}

    for sequence, token in enumerate(tokens, start=1):
        canonical_item = canonical_item_from_token(
            token,
            sequence,
            stream_session_id=_STREAM_SESSION_ID,
            run_id=_RUN_ID,
            turn_id=_TURN_ID,
        )
        items.append(canonical_item)
        if canonical_item.kind is StreamItemKind.REASONING_DELTA:
            reasoning_started = True
        if canonical_item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            tool_call_id = canonical_item.correlation.tool_call_id
            assert tool_call_id is not None
            data: dict[str, object] | None = None
            if isinstance(token, ToolCallToken) and token.call is not None:
                data = {
                    "name": token.call.name,
                    "arguments": token.call.arguments,
                }
            tool_call_data.setdefault(tool_call_id, data)

    terminal_outcome = StreamTerminalOutcome.COMPLETED
    if terminal_kind is StreamItemKind.STREAM_ERRORED:
        terminal_outcome = StreamTerminalOutcome.ERRORED
    elif terminal_kind is StreamItemKind.STREAM_CANCELLED:
        terminal_outcome = StreamTerminalOutcome.CANCELLED

    for tool_call_id, data in tool_call_data.items():
        if data is not None:
            items.append(
                _tool_item(
                    StreamItemKind.TOOL_CALL_READY,
                    len(items),
                    tool_call_id=tool_call_id,
                    data=data,
                )
            )
        items.append(
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                len(items),
                tool_call_id=tool_call_id,
                metadata=(
                    {}
                    if data is not None
                    else {"tool_call.close_reason": "error"}
                ),
            )
        )

    if reasoning_started:
        items.append(_reasoning_done(len(items)))
    items.append(_answer_done(len(items)))
    items.append(
        _control_item(
            terminal_kind,
            len(items),
            usage=(
                {}
                if usage is None
                and terminal_kind is StreamItemKind.STREAM_COMPLETED
                else usage
            ),
            terminal_outcome=terminal_outcome,
        )
    )
    return StreamGoldenTrace(name=name, items=tuple(items))


class TextGenerationResponseGoldenTraceTestCase(IsolatedAsyncioTestCase):
    async def test_canonical_stream_matches_current_golden_trace(
        self,
    ) -> None:
        trace = _golden_trace(
            "response-current-canonical-order",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "answer "),
            _reasoning_delta(2, "<thi"),
            _reasoning_delta(3, "nk>"),
            _reasoning_delta(4, " private "),
            _reasoning_delta(5, "</think>"),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                6,
                tool_call_id="call-1",
                text_delta='{"expression":"2+2"}',
                data={
                    "name": "math.calculator",
                    "arguments": {"expression": "2+2"},
                },
            ),
            _answer_delta(7, "done"),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                8,
                tool_call_id="call-1",
                data={
                    "name": "math.calculator",
                    "arguments": {"expression": "2+2"},
                },
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                9,
                tool_call_id="call-1",
            ),
            _reasoning_done(10),
            _answer_done(11),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                12,
                usage={"output_tokens": 7},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        response = _response_from_trace(trace)
        item_list: list[CanonicalStreamItem] = []
        async for item in response.canonical_stream(
            stream_session_id=_STREAM_SESSION_ID,
            run_id=_RUN_ID,
            turn_id=_TURN_ID,
        ):
            item_list.append(item)
        items = tuple(item_list)

        self.assertEqual(items, trace.items)
        self.assertEqual(
            _semantic_trace(items),
            (
                {
                    "sequence": 0,
                    "kind": StreamItemKind.STREAM_STARTED,
                    "channel": StreamChannel.CONTROL,
                    "visibility": StreamVisibility.PUBLIC,
                },
                {
                    "sequence": 1,
                    "kind": StreamItemKind.ANSWER_DELTA,
                    "channel": StreamChannel.ANSWER,
                    "visibility": StreamVisibility.PUBLIC,
                    "text_delta": "answer ",
                },
                {
                    "sequence": 2,
                    "kind": StreamItemKind.REASONING_DELTA,
                    "channel": StreamChannel.REASONING,
                    "visibility": StreamVisibility.PRIVATE,
                    "text_delta": "<thi",
                },
                {
                    "sequence": 3,
                    "kind": StreamItemKind.REASONING_DELTA,
                    "channel": StreamChannel.REASONING,
                    "visibility": StreamVisibility.PRIVATE,
                    "text_delta": "nk>",
                },
                {
                    "sequence": 4,
                    "kind": StreamItemKind.REASONING_DELTA,
                    "channel": StreamChannel.REASONING,
                    "visibility": StreamVisibility.PRIVATE,
                    "text_delta": " private ",
                },
                {
                    "sequence": 5,
                    "kind": StreamItemKind.REASONING_DELTA,
                    "channel": StreamChannel.REASONING,
                    "visibility": StreamVisibility.PRIVATE,
                    "text_delta": "</think>",
                },
                {
                    "sequence": 6,
                    "kind": StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    "channel": StreamChannel.TOOL_CALL,
                    "visibility": StreamVisibility.PUBLIC,
                    "correlation": {"tool_call_id": "call-1"},
                    "text_delta": '{"expression":"2+2"}',
                    "data": {
                        "name": "math.calculator",
                        "arguments": {"expression": "2+2"},
                    },
                },
                {
                    "sequence": 7,
                    "kind": StreamItemKind.ANSWER_DELTA,
                    "channel": StreamChannel.ANSWER,
                    "visibility": StreamVisibility.PUBLIC,
                    "text_delta": "done",
                },
                {
                    "sequence": 8,
                    "kind": StreamItemKind.TOOL_CALL_READY,
                    "channel": StreamChannel.TOOL_CALL,
                    "visibility": StreamVisibility.PUBLIC,
                    "correlation": {"tool_call_id": "call-1"},
                    "data": {
                        "name": "math.calculator",
                        "arguments": {"expression": "2+2"},
                    },
                },
                {
                    "sequence": 9,
                    "kind": StreamItemKind.TOOL_CALL_DONE,
                    "channel": StreamChannel.TOOL_CALL,
                    "visibility": StreamVisibility.PUBLIC,
                    "correlation": {"tool_call_id": "call-1"},
                },
                {
                    "sequence": 10,
                    "kind": StreamItemKind.REASONING_DONE,
                    "channel": StreamChannel.REASONING,
                    "visibility": StreamVisibility.PRIVATE,
                },
                {
                    "sequence": 11,
                    "kind": StreamItemKind.ANSWER_DONE,
                    "channel": StreamChannel.ANSWER,
                    "visibility": StreamVisibility.PUBLIC,
                },
                {
                    "sequence": 12,
                    "kind": StreamItemKind.STREAM_COMPLETED,
                    "channel": StreamChannel.CONTROL,
                    "visibility": StreamVisibility.PUBLIC,
                    "usage": {"output_tokens": 7},
                    "terminal_outcome": StreamTerminalOutcome.COMPLETED,
                },
            ),
        )
        self.assertEqual(
            accumulate_canonical_stream_items(trace.items).answer_text,
            "answer done",
        )

    async def test_canonical_stream_legacy_rejection_raw_stream(
        self,
    ) -> None:
        response = _legacy_rejection_response_from_raw_stream()
        items: list[CanonicalStreamItem] = []

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            async for item in response.canonical_stream(
                stream_session_id=_STREAM_SESSION_ID,
                run_id=_RUN_ID,
                turn_id=_TURN_ID,
            ):
                items.append(item)

        self.assertEqual(items, [])

    async def test_to_str_legacy_rejection_mixed_raw_stream(self) -> None:
        response = _legacy_rejection_response_from_raw_stream(
            "legacy-first",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "canonical stream item after legacy stream item",
        ):
            await response.to_str()

    async def test_to_str_matches_accumulated_golden_answer(self) -> None:
        trace = _golden_trace(
            "response-to-str-equivalence",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "answer "),
            _reasoning_delta(2, "private"),
            _reasoning_done(3),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                4,
                tool_call_id="call-1",
                text_delta='{"expression":"2+2"}',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                5,
                tool_call_id="call-1",
                data={"name": "math.calculator"},
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                6,
                tool_call_id="call-1",
            ),
            _answer_delta(7, "done"),
            _answer_done(8),
            _usage_item(9, {"output_tokens": 2}),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                10,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        response = _response_from_trace(trace)
        output = await response.to_str()

        self.assertEqual(output, "answer done")
        self.assertEqual(
            output,
            accumulate_canonical_stream_items(trace.items).answer_text,
        )
        self.assertEqual(await response.to_str(), output)

    async def test_canonical_stream_matches_to_str_for_local_response(
        self,
    ) -> None:
        trace = _golden_trace(
            "response-local-canonical-stream",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "Hel"),
            _answer_delta(2, "lo"),
            _answer_done(3),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                4,
                usage={"output_tokens": 2},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            _control_item(StreamItemKind.STREAM_CLOSED, 5),
        )

        stream_response = _response_from_trace(trace)
        item_list: list[CanonicalStreamItem] = []
        async for item in stream_response.canonical_stream(
            stream_session_id=_STREAM_SESSION_ID,
            run_id=_RUN_ID,
            turn_id=_TURN_ID,
            provider_family="transformers",
        ):
            item_list.append(item)
        items = tuple(item_list)

        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "Hello")
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items, trace.items)
        self.assertEqual(await _response_from_trace(trace).to_str(), "Hello")

    async def test_reasoning_trace_preserves_parsed_marker_whitespace(
        self,
    ) -> None:
        trace = _golden_trace(
            "response-reasoning-parsed-marker-whitespace",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "lead "),
            _answer_delta(2, " "),
            _reasoning_delta(3, "<think>"),
            _reasoning_delta(4, " "),
            _reasoning_delta(5, " private "),
            _reasoning_delta(6, " "),
            _reasoning_delta(7, "</think>"),
            _answer_delta(8, " "),
            _answer_delta(9, "tail"),
            _reasoning_done(10),
            _answer_done(11),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                12,
                usage={"output_tokens": 9},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        self.assertEqual(
            [
                item.text_delta
                for item in trace.items
                if item.kind is StreamItemKind.REASONING_DELTA
            ],
            ["<think>", " ", " private ", " ", "</think>"],
        )
        self.assertTrue(
            all(
                item.visibility is StreamVisibility.PRIVATE
                for item in trace.items
                if item.channel is StreamChannel.REASONING
            )
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "lead   tail")
        self.assertEqual(
            accumulator.reasoning_text,
            "<think>  private  </think>",
        )

    async def test_reasoning_trace_preserves_split_markers(self) -> None:
        trace = _golden_trace(
            "response-reasoning-split-markers",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "lead"),
            _reasoning_delta(2, "<"),
            _reasoning_delta(3, "think"),
            _reasoning_delta(4, ">"),
            _reasoning_delta(5, "inside"),
            _reasoning_delta(6, "<"),
            _reasoning_delta(7, "/think"),
            _reasoning_delta(8, ">"),
            _answer_delta(9, "tail"),
            _reasoning_done(10),
            _answer_done(11),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                12,
                usage={"output_tokens": 9},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        self.assertEqual(
            [
                item.text_delta
                for item in trace.items
                if item.kind is StreamItemKind.REASONING_DELTA
            ],
            ["<", "think", ">", "inside", "<", "/think", ">"],
        )
        self.assertEqual(
            [item.kind for item in trace.items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "leadtail")
        self.assertEqual(accumulator.reasoning_text, "<think>inside</think>")
        terminal = trace.items[-1]
        self.assertIs(terminal.kind, StreamItemKind.STREAM_COMPLETED)
        self.assertEqual(terminal.usage, {"output_tokens": 9})
        self.assertIs(
            terminal.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )

    async def test_reasoning_trace_keeps_unterminated_reasoning_private(
        self,
    ) -> None:
        trace = _golden_trace(
            "response-reasoning-unterminated",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "answer "),
            _reasoning_delta(2, "<think>"),
            _reasoning_delta(3, "private"),
            _reasoning_delta(4, "<"),
            _reasoning_delta(5, "/thi"),
            _reasoning_done(6),
            _answer_done(7),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                8,
                usage={"output_tokens": 5},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        self.assertEqual(
            [
                item.text_delta
                for item in trace.items
                if item.kind is StreamItemKind.REASONING_DELTA
            ],
            ["<think>", "private", "<", "/thi"],
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "answer ")
        self.assertEqual(accumulator.reasoning_text, "<think>private</thi")
        self.assertTrue(
            all(
                item.visibility is StreamVisibility.PRIVATE
                for item in trace.items
                if item.channel is StreamChannel.REASONING
            )
        )

    async def test_provider_error_characterizes_error_terminal_trace(
        self,
    ) -> None:
        trace = _golden_trace(
            "response-provider-error",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "before "),
            _answer_done(2),
            _control_item(
                StreamItemKind.STREAM_ERRORED,
                3,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        )

        terminal = trace.items[-1]
        self.assertIs(terminal.kind, StreamItemKind.STREAM_ERRORED)
        self.assertIs(terminal.terminal_outcome, StreamTerminalOutcome.ERRORED)
        self.assertEqual(
            accumulate_canonical_stream_items(trace.items).answer_text,
            "before ",
        )
        with self.assertRaisesRegex(RuntimeError, "stream errored"):
            await _response_from_trace(trace).to_str()

    async def test_tool_call_trace_preserves_deltas_ids_and_result(
        self,
    ) -> None:
        trace = _golden_trace(
            "response-tool-call-deltas-and-result",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "Calculating "),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                2,
                tool_call_id="call-stable",
                text_delta='{"expression"',
                data={"name": "math.calculator"},
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                3,
                tool_call_id="call-stable",
                text_delta=':"2+2"}',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                4,
                tool_call_id="call-stable",
                data={
                    "name": "math.calculator",
                    "arguments": {"expression": "2+2"},
                },
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                5,
                tool_call_id="call-stable",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                6,
                tool_call_id="call-stable",
                data={"name": "math.calculator"},
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                7,
                tool_call_id="call-stable",
                text_delta="4",
                data={"chunk_index": 0},
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                8,
                tool_call_id="call-stable",
                data={"completed_steps": 1, "total_steps": 1},
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                9,
                tool_call_id="call-stable",
                data={"result": "4", "status": "ok"},
            ),
            _answer_done(10),
            _usage_item(11, {"output_tokens": 3}),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                12,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )
        accumulator = accumulate_canonical_stream_items(trace.items)
        observations = assemble_tool_observations(trace.items)

        self.assertEqual(accumulator.answer_text, "Calculating ")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-stable": '{"expression":"2+2"}'},
        )
        self.assertEqual(
            accumulator.tool_execution_outputs, {"call-stable": "4"}
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].tool_call_id, "call-stable")
        self.assertEqual(observations[0].arguments, '{"expression":"2+2"}')
        self.assertEqual(observations[0].output, "4")
        self.assertIs(
            observations[0].terminal_kind,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
        )
        self.assertEqual(
            observations[0].terminal_data, {"result": "4", "status": "ok"}
        )
        self.assertEqual(
            [
                (item.kind, item.text_delta, item.data)
                for item in trace.items
                if item.channel is StreamChannel.TOOL_EXECUTION
            ],
            [
                (
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    None,
                    {"name": "math.calculator"},
                ),
                (
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    "4",
                    {"chunk_index": 0},
                ),
                (
                    StreamItemKind.TOOL_EXECUTION_PROGRESS,
                    None,
                    {"completed_steps": 1, "total_steps": 1},
                ),
                (
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    None,
                    {"result": "4", "status": "ok"},
                ),
            ],
        )
        self.assertEqual(
            [
                item.get("correlation", {}).get("tool_call_id")
                for item in trace.to_fixture()["items"]  # type: ignore[index]
                if item["channel"] in {"tool_call", "tool_execution"}
            ],
            ["call-stable"] * 8,
        )

    async def test_legacy_fixture_ds4_tool_call_trace_captures_current_shape(
        self,
    ) -> None:
        call = ToolCall(
            id="ds4_tool_abc",
            name="math.calculator",
            arguments={"expression": "2 + 2", "precision": 2},
        )
        tokens: tuple[Token | str, ...] = (
            "I will calculate.",
            ToolCallToken(token="2 + 2"),
            ToolCallToken(token="2"),
            ToolCallToken(token="", call=call),
        )
        trace = _legacy_fixture_trace_from_tokens(
            "legacy-fixture-ds4-tool-call-current-shape",
            tokens,
            usage={"output_tokens": len(tokens)},
        )
        accumulator = accumulate_canonical_stream_items(trace.items)

        self.assertEqual(accumulator.answer_text, "I will calculate.")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"legacy-tool-call": "2 + 22", "ds4_tool_abc": ""},
        )
        self.assertEqual(
            [
                item["correlation"]["tool_call_id"]
                for item in trace.to_fixture()["items"]  # type: ignore[index]
                if item["channel"] == "tool_call"
            ],
            [
                "legacy-tool-call",
                "legacy-tool-call",
                "ds4_tool_abc",
                "legacy-tool-call",
                "ds4_tool_abc",
                "ds4_tool_abc",
            ],
        )

    async def test_usage_callbacks_and_cancelled_terminal_trace(
        self,
    ) -> None:
        usage_payload = {"input_tokens": 3, "output_tokens": 2}
        usage_trace = _golden_trace(
            "response-usage-callbacks",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "one "),
            _answer_delta(2, "two"),
            _answer_done(3),
            _usage_item(4, usage_payload),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                5,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        class UsageFactory:
            usage = usage_payload
            provider_family = "openai"

            def __call__(
                self, **_: object
            ) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    for item in usage_trace.items:
                        yield item

                return gen()

        callbacks: list[str] = []

        async def async_callback() -> None:
            callbacks.append("async")

        def sync_callback() -> None:
            callbacks.append("sync")

        factory = UsageFactory()
        response = _response(factory)
        response.add_done_callback(sync_callback)
        response.add_done_callback(async_callback)

        self.assertEqual(response.usage, factory.usage)
        self.assertEqual(response.provider_family, "openai")
        self.assertEqual(await response.to_str(), "one two")
        self.assertEqual(await response.to_str(), "one two")
        self.assertEqual(callbacks, ["sync", "async"])
        self.assertEqual(response.usage, usage_payload)

        trace = _golden_trace(
            "response-provider-cancelled",
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            _answer_delta(1, "before "),
            _answer_done(2),
            _control_item(
                StreamItemKind.STREAM_CANCELLED,
                3,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
        )

        terminal = trace.items[-1]
        self.assertIs(terminal.kind, StreamItemKind.STREAM_CANCELLED)
        self.assertIs(
            terminal.terminal_outcome, StreamTerminalOutcome.CANCELLED
        )
        self.assertEqual(
            accumulate_canonical_stream_items(trace.items).terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )
        with self.assertRaises(CancelledError):
            await _response_from_trace(trace).to_str()

    def test_terminal_outcome_traces_cover_completed_errored_cancelled(
        self,
    ) -> None:
        cases = (
            (
                "terminal-completed",
                StreamItemKind.STREAM_COMPLETED,
                StreamTerminalOutcome.COMPLETED,
                {"output_tokens": 1},
            ),
            (
                "terminal-errored",
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                None,
            ),
            (
                "terminal-cancelled",
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
                None,
            ),
        )

        for name, kind, outcome, usage in cases:
            trace = _golden_trace(
                name,
                _control_item(StreamItemKind.STREAM_STARTED, 0),
                _control_item(
                    kind,
                    1,
                    usage=usage,
                    terminal_outcome=outcome,
                ),
            )

            accumulator = accumulate_canonical_stream_items(trace.items)
            terminal = trace.items[-1]
            self.assertIs(terminal.kind, kind)
            self.assertIs(terminal.terminal_outcome, outcome)
            self.assertIs(accumulator.terminal_outcome, outcome)
            self.assertEqual(terminal.usage, usage)

    def test_canonical_trace_rejects_content_after_terminal(self) -> None:
        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            _golden_trace(
                "content-after-terminal-rejection",
                _control_item(StreamItemKind.STREAM_STARTED, 0),
                _control_item(
                    StreamItemKind.STREAM_COMPLETED,
                    1,
                    usage={"output_tokens": 0},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
                _answer_delta(2, "late"),
            )

    async def test_protocol_output_traces_capture_current_projections(
        self,
    ) -> None:
        traces = (
            StreamGoldenTrace(
                name="protocol-sdk-iteration",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        1,
                        protocol_item_id="sdk-answer-1",
                        channel=StreamChannel.ANSWER,
                        text_delta="Hel",
                        data={"projection": "canonical.answer_delta"},
                        metadata={"protocol": "sdk.async_iteration"},
                    ),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        2,
                        protocol_item_id="sdk-answer-2",
                        channel=StreamChannel.ANSWER,
                        text_delta="lo",
                        data={"projection": "canonical.answer_delta"},
                        metadata={"protocol": "sdk.async_iteration"},
                    ),
                    _answer_done(3),
                    _usage_item(4, {"output_tokens": 2}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        5,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                ),
            ),
            StreamGoldenTrace(
                name="protocol-stdout",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        1,
                        protocol_item_id="stdout-write-1",
                        channel=StreamChannel.ANSWER,
                        text_delta="Hello",
                        data={"write": "Hello"},
                        metadata={"protocol": "stdout"},
                    ),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        2,
                        protocol_item_id="stdout-write-2",
                        channel=StreamChannel.ANSWER,
                        text_delta="\n",
                        data={"write": "\n"},
                        metadata={"protocol": "stdout"},
                    ),
                    _answer_done(3),
                    _usage_item(4, {"output_tokens": 2}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        5,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                ),
            ),
            StreamGoldenTrace(
                name="protocol-cli-rendering",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.REASONING_DELTA,
                        1,
                        protocol_item_id="cli-reasoning-frame",
                        channel=StreamChannel.REASONING,
                        text_delta="think",
                        data={"renderable": "reasoning"},
                        metadata={"protocol": "cli.theme"},
                        visibility=StreamVisibility.PRIVATE,
                    ),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        2,
                        protocol_item_id="cli-answer-frame",
                        channel=StreamChannel.ANSWER,
                        text_delta="answer",
                        data={"renderable": "answer"},
                        metadata={"protocol": "cli.theme"},
                    ),
                    _reasoning_done(3),
                    _answer_done(4),
                    _usage_item(5, {"output_tokens": 2}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        6,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                ),
            ),
            StreamGoldenTrace(
                name="protocol-chat-sse",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        1,
                        protocol_item_id="chat-sse-chunk-1",
                        channel=StreamChannel.ANSWER,
                        text_delta="a",
                        data={
                            "event": "chat.completion.chunk",
                            "delta": {"content": "a"},
                        },
                        metadata={"protocol": "openai.chat.sse"},
                    ),
                    _answer_done(2),
                    _usage_item(3, {"output_tokens": 1}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        4,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    _protocol_item(
                        StreamItemKind.STREAM_CLOSED,
                        5,
                        protocol_item_id="chat-sse-done",
                        data={"data": "[DONE]"},
                        metadata={"protocol": "openai.chat.sse"},
                    ),
                ),
            ),
            StreamGoldenTrace(
                name="protocol-responses-sse",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.REASONING_DELTA,
                        1,
                        protocol_item_id="responses-reasoning-delta",
                        channel=StreamChannel.REASONING,
                        text_delta="r",
                        data={
                            "event": "response.reasoning_text.delta",
                            "delta": "r",
                        },
                        metadata={"protocol": "openai.responses.sse"},
                        visibility=StreamVisibility.PRIVATE,
                    ),
                    _protocol_item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        2,
                        protocol_item_id="responses-tool-delta",
                        channel=StreamChannel.TOOL_CALL,
                        text_delta='{"x":1}',
                        data={
                            "event": "response.custom_tool_call_input.delta",
                            "delta": '{"x":1}',
                        },
                        metadata={"protocol": "openai.responses.sse"},
                    ),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        3,
                        protocol_item_id="responses-output-delta",
                        channel=StreamChannel.ANSWER,
                        text_delta="a",
                        data={
                            "event": "response.output_text.delta",
                            "delta": "a",
                        },
                        metadata={"protocol": "openai.responses.sse"},
                    ),
                    _protocol_item(
                        StreamItemKind.TOOL_CALL_READY,
                        4,
                        protocol_item_id="responses-tool-ready",
                        channel=StreamChannel.TOOL_CALL,
                        data={
                            "event": "response.custom_tool_call_input.done",
                        },
                        metadata={"protocol": "openai.responses.sse"},
                    ),
                    _protocol_item(
                        StreamItemKind.TOOL_CALL_DONE,
                        5,
                        protocol_item_id="responses-tool-done",
                        channel=StreamChannel.TOOL_CALL,
                        data={
                            "event": "response.custom_tool_call_input.done",
                        },
                        metadata={"protocol": "openai.responses.sse"},
                    ),
                    _reasoning_done(6),
                    _answer_done(7),
                    _usage_item(8, {"output_tokens": 3}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        9,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    _protocol_item(
                        StreamItemKind.STREAM_CLOSED,
                        10,
                        protocol_item_id="responses-done",
                        data={"event": "done", "data": "[DONE]"},
                        metadata={"protocol": "openai.responses.sse"},
                    ),
                ),
            ),
            StreamGoldenTrace(
                name="protocol-mcp",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.TOOL_EXECUTION_STARTED,
                        1,
                        protocol_item_id="mcp-tool-start",
                        metadata={"protocol": "mcp"},
                    ),
                    _protocol_item(
                        StreamItemKind.TOOL_EXECUTION_OUTPUT,
                        2,
                        protocol_item_id="mcp-resource-update",
                        text_delta="log",
                        data={
                            "jsonrpc": "2.0",
                            "method": "notifications/resources/updated",
                            "params": {"uri": "avalan://tools/call/stdout"},
                        },
                        metadata={"protocol": "mcp"},
                    ),
                    _protocol_item(
                        StreamItemKind.TOOL_EXECUTION_COMPLETED,
                        3,
                        protocol_item_id="mcp-tool-result",
                        data={
                            "result": {
                                "content": [{"type": "text", "text": "final"}]
                            }
                        },
                        metadata={"protocol": "mcp"},
                    ),
                    _usage_item(4, {"output_tokens": 1}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        5,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                ),
            ),
            StreamGoldenTrace(
                name="protocol-a2a",
                items=(
                    _control_item(StreamItemKind.STREAM_STARTED, 0),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        1,
                        protocol_item_id="a2a-artifact-delta",
                        channel=StreamChannel.ANSWER,
                        text_delta="partial",
                        data={
                            "event": "task.artifact-update",
                            "artifact": {"parts": [{"text": "partial"}]},
                        },
                        metadata={"protocol": "a2a"},
                    ),
                    _answer_done(2),
                    _usage_item(3, {"output_tokens": 1}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        4,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    _protocol_item(
                        StreamItemKind.STREAM_CLOSED,
                        5,
                        protocol_item_id="a2a-task-completed",
                        data={
                            "event": "task.status-update",
                            "status": {"state": "completed"},
                        },
                        metadata={"protocol": "a2a"},
                    ),
                ),
            ),
        )

        fixtures = {trace.name: trace.to_fixture() for trace in traces}

        self.assertEqual(
            {
                item["correlation"]["protocol_item_id"]
                for item in fixtures["protocol-sdk-iteration"]["items"]  # type: ignore[index]
                if "correlation" in item
            },
            {"sdk-answer-1", "sdk-answer-2"},
        )
        self.assertEqual(
            accumulate_canonical_stream_items(traces[1].items).answer_text,
            "Hello\n",
        )
        self.assertEqual(
            fixtures["protocol-cli-rendering"]["items"][1]["visibility"],  # type: ignore[index]
            "private",
        )
        self.assertEqual(
            fixtures["protocol-chat-sse"]["items"][-1]["data"],  # type: ignore[index]
            {"data": "[DONE]"},
        )
        self.assertEqual(
            fixtures["protocol-responses-sse"]["items"][2]["data"]["event"],  # type: ignore[index]
            "response.custom_tool_call_input.delta",
        )
        response_event_names = [
            item["data"]["event"]
            for item in fixtures["protocol-responses-sse"]["items"]  # type: ignore[index]
            if "data" in item and "event" in item["data"]
        ]
        self.assertNotIn(
            "response.custom_tool_call.done", response_event_names
        )
        self.assertEqual(
            response_event_names.count("response.custom_tool_call_input.done"),
            2,
        )
        self.assertEqual(
            fixtures["protocol-mcp"]["items"][2]["data"]["method"],  # type: ignore[index]
            "notifications/resources/updated",
        )
        self.assertEqual(
            fixtures["protocol-a2a"]["items"][-1]["data"]["status"],  # type: ignore[index]
            {"state": "completed"},
        )

        with self.assertRaises(AssertionError):
            StreamItemCorrelation(protocol_item_id="")

    async def test_legacy_fixture_reasoning_token_missing_source_id_is_stable(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[Token]:
            yield Token(token="<think>")

        response = _response(lambda **_: gen())
        tokens = [token async for token in response]

        self.assertEqual(len(tokens), 1)
        self.assertIsInstance(tokens[0], ReasoningToken)
        self.assertEqual(tokens[0].id, -1)
