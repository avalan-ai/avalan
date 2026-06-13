from asyncio import CancelledError
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


def _tool_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    tool_call_id: str,
    text_delta: str | None = None,
    data: object | None = None,
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


def _trace_from_tokens(
    name: str,
    tokens: tuple[Token | str, ...],
    *,
    terminal_kind: StreamItemKind = StreamItemKind.STREAM_COMPLETED,
    usage: object | None = None,
) -> StreamGoldenTrace:
    items = [
        _control_item(StreamItemKind.STREAM_STARTED, 0),
        *(
            canonical_item_from_token(
                token,
                sequence,
                stream_session_id=_STREAM_SESSION_ID,
                run_id=_RUN_ID,
                turn_id=_TURN_ID,
            )
            for sequence, token in enumerate(tokens, start=1)
        ),
    ]
    terminal_outcome = StreamTerminalOutcome.COMPLETED
    if terminal_kind is StreamItemKind.STREAM_ERRORED:
        terminal_outcome = StreamTerminalOutcome.ERRORED
    elif terminal_kind is StreamItemKind.STREAM_CANCELLED:
        terminal_outcome = StreamTerminalOutcome.CANCELLED
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
    async def test_async_iteration_matches_current_golden_trace(
        self,
    ) -> None:
        call = ToolCall(
            id="call-1",
            name="math.calculator",
            arguments={"expression": "2+2"},
        )

        async def gen():
            yield Token(token="answer ")
            yield "<thi"
            yield "nk>"
            yield " private "
            yield "</think>"
            yield ToolCallToken(
                token='{"expression":"2+2"}',
                call=call,
            )
            yield "done"

        response = _response(lambda **_: gen())
        tokens = tuple([token async for token in response])
        trace = _trace_from_tokens(
            "response-current-token-order",
            tokens,
            usage={"output_tokens": response.output_token_count},
        )

        self.assertEqual(
            trace.to_fixture(),
            {
                "format_version": 1,
                "name": "response-current-token-order",
                "items": [
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 0,
                        "kind": "stream.started",
                        "channel": "control",
                        "visibility": "public",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 1,
                        "kind": "answer.delta",
                        "channel": "answer",
                        "visibility": "public",
                        "text_delta": "answer ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 2,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "<thi",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 3,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "nk>",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 4,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": " private ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 5,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "</think>",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 6,
                        "kind": "tool_call.argument_delta",
                        "channel": "tool_call",
                        "visibility": "public",
                        "correlation": {"tool_call_id": "call-1"},
                        "text_delta": '{"expression":"2+2"}',
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 7,
                        "kind": "answer.delta",
                        "channel": "answer",
                        "visibility": "public",
                        "text_delta": "done",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 8,
                        "kind": "answer.done",
                        "channel": "answer",
                        "visibility": "public",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 9,
                        "kind": "stream.completed",
                        "channel": "control",
                        "visibility": "public",
                        "usage": {"output_tokens": 7},
                        "terminal_outcome": "completed",
                    },
                ],
            },
        )
        self.assertEqual(
            accumulate_canonical_stream_items(trace.items).answer_text,
            "answer done",
        )

    async def test_to_str_matches_accumulated_golden_answer(self) -> None:
        async def gen():
            yield "answer "
            yield "<think>"
            yield "private"
            yield "</think>"
            yield ToolCallToken(token='{"expression":"2+2"}')
            yield "done"

        tokens_response = _response(lambda **_: gen())
        tokens = tuple([token async for token in tokens_response])
        trace = _trace_from_tokens("response-to-str-equivalence", tokens)
        string_response = _response(lambda **_: gen())
        output = await string_response.to_str()

        self.assertEqual(output, "answer done")
        self.assertEqual(
            output,
            accumulate_canonical_stream_items(trace.items).answer_text,
        )
        self.assertEqual(await string_response.to_str(), output)

    async def test_canonical_stream_matches_to_str_for_local_response(
        self,
    ) -> None:
        async def gen():
            yield Token(token="Hel")
            yield "lo"

        stream_response = _response(lambda **_: gen())
        items = tuple(
            [
                item
                async for item in stream_response.canonical_stream(
                    stream_session_id=_STREAM_SESSION_ID,
                    run_id=_RUN_ID,
                    turn_id=_TURN_ID,
                    provider_family="transformers",
                )
            ]
        )

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
        self.assertEqual(
            {item.provider_family for item in items}, {"transformers"}
        )
        self.assertEqual(await _response(lambda **_: gen()).to_str(), "Hello")

    async def test_reasoning_trace_preserves_parsed_marker_whitespace(
        self,
    ) -> None:
        async def gen():
            yield "lead "
            yield " <think> "
            yield " private "
            yield " </think> "
            yield "tail"

        response = _response(lambda **_: gen())
        tokens = tuple([token async for token in response])
        trace = _trace_from_tokens(
            "response-reasoning-parsed-marker-whitespace",
            tokens,
            usage={"output_tokens": response.output_token_count},
        )

        self.assertIsInstance(tokens[1], ReasoningToken)
        self.assertIsInstance(tokens[2], ReasoningToken)
        self.assertIsInstance(tokens[3], ReasoningToken)
        self.assertEqual(
            [token.token for token in tokens[1:4]],
            [" <think> ", " private ", " </think> "],
        )
        self.assertEqual(
            trace.to_fixture(),
            {
                "format_version": 1,
                "name": "response-reasoning-parsed-marker-whitespace",
                "items": [
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 0,
                        "kind": "stream.started",
                        "channel": "control",
                        "visibility": "public",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 1,
                        "kind": "answer.delta",
                        "channel": "answer",
                        "visibility": "public",
                        "text_delta": "lead ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 2,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": " <think> ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 3,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": " private ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 4,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": " </think> ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 5,
                        "kind": "answer.delta",
                        "channel": "answer",
                        "visibility": "public",
                        "text_delta": "tail",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 6,
                        "kind": "answer.done",
                        "channel": "answer",
                        "visibility": "public",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 7,
                        "kind": "stream.completed",
                        "channel": "control",
                        "visibility": "public",
                        "usage": {"output_tokens": 5},
                        "terminal_outcome": "completed",
                    },
                ],
            },
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "lead tail")
        self.assertEqual(
            accumulator.reasoning_text,
            " <think>  private  </think> ",
        )

    async def test_reasoning_trace_preserves_split_markers(self) -> None:
        async def gen():
            yield "lead"
            yield "<"
            yield "think"
            yield ">"
            yield "inside"
            yield "<"
            yield "/think"
            yield ">"
            yield "tail"

        response = _response(lambda **_: gen())
        tokens = tuple([token async for token in response])
        trace = _trace_from_tokens(
            "response-reasoning-split-markers",
            tokens,
            usage={"output_tokens": response.output_token_count},
        )

        self.assertEqual(
            [token.token for token in tokens[1:8]],
            ["<", "think", ">", "inside", "<", "/think", ">"],
        )
        self.assertTrue(
            all(isinstance(token, ReasoningToken) for token in tokens[1:8])
        )
        self.assertEqual(
            [
                item["kind"]
                for item in trace.to_fixture()["items"]  # type: ignore[index]
            ],
            [
                "stream.started",
                "answer.delta",
                "reasoning.delta",
                "reasoning.delta",
                "reasoning.delta",
                "reasoning.delta",
                "reasoning.delta",
                "reasoning.delta",
                "reasoning.delta",
                "answer.delta",
                "answer.done",
                "stream.completed",
            ],
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "leadtail")
        self.assertEqual(accumulator.reasoning_text, "<think>inside</think>")
        self.assertEqual(
            trace.to_fixture()["items"][-1],  # type: ignore[index]
            {
                "stream_session_id": _STREAM_SESSION_ID,
                "run_id": _RUN_ID,
                "turn_id": _TURN_ID,
                "sequence": 11,
                "kind": "stream.completed",
                "channel": "control",
                "visibility": "public",
                "usage": {"output_tokens": 9},
                "terminal_outcome": "completed",
            },
        )

    async def test_reasoning_trace_keeps_unterminated_reasoning_private(
        self,
    ) -> None:
        async def gen():
            yield "answer "
            yield "<think>"
            yield "private"
            yield "<"
            yield "/thi"

        response = _response(lambda **_: gen())
        tokens = tuple([token async for token in response])
        trace = _trace_from_tokens(
            "response-reasoning-unterminated",
            tokens,
            usage={"output_tokens": response.output_token_count},
        )

        self.assertTrue(
            all(isinstance(token, ReasoningToken) for token in tokens[1:])
        )
        self.assertEqual(
            [token.token for token in tokens[1:]],
            ["<think>", "private", "<", "/thi"],
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "answer ")
        self.assertEqual(accumulator.reasoning_text, "<think>private</thi")
        self.assertEqual(
            trace.to_fixture(),
            {
                "format_version": 1,
                "name": "response-reasoning-unterminated",
                "items": [
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 0,
                        "kind": "stream.started",
                        "channel": "control",
                        "visibility": "public",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 1,
                        "kind": "answer.delta",
                        "channel": "answer",
                        "visibility": "public",
                        "text_delta": "answer ",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 2,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "<think>",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 3,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "private",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 4,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "<",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 5,
                        "kind": "reasoning.delta",
                        "channel": "reasoning",
                        "visibility": "private",
                        "text_delta": "/thi",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 6,
                        "kind": "answer.done",
                        "channel": "answer",
                        "visibility": "public",
                    },
                    {
                        "stream_session_id": _STREAM_SESSION_ID,
                        "run_id": _RUN_ID,
                        "turn_id": _TURN_ID,
                        "sequence": 7,
                        "kind": "stream.completed",
                        "channel": "control",
                        "visibility": "public",
                        "usage": {"output_tokens": 5},
                        "terminal_outcome": "completed",
                    },
                ],
            },
        )

    async def test_provider_error_characterizes_error_terminal_trace(
        self,
    ) -> None:
        async def gen():
            yield "before "
            raise RuntimeError("provider failed")

        response = _response(lambda **_: gen())
        iterator = response.__aiter__()
        first = await iterator.__anext__()

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await iterator.__anext__()

        trace = _trace_from_tokens(
            "response-provider-error",
            (first,),
            terminal_kind=StreamItemKind.STREAM_ERRORED,
        )

        self.assertEqual(
            trace.to_fixture()["items"][-1],
            {
                "stream_session_id": _STREAM_SESSION_ID,
                "run_id": _RUN_ID,
                "turn_id": _TURN_ID,
                "sequence": 3,
                "kind": "stream.errored",
                "channel": "control",
                "visibility": "public",
                "terminal_outcome": "errored",
            },
        )
        self.assertEqual(
            accumulate_canonical_stream_items(trace.items).answer_text,
            "before ",
        )

    async def test_tool_call_trace_preserves_deltas_ids_and_result(
        self,
    ) -> None:
        call = ToolCall(
            id="call-stable",
            name="math.calculator",
            arguments={"expression": "2+2"},
        )

        async def gen():
            yield "Calculating "
            yield ToolCallToken(token='{"expression"', call=call)
            yield ToolCallToken(token=':"2+2"}', call=call)

        response = _response(lambda **_: gen())
        tokens = tuple([token async for token in response])
        items = [
            _control_item(StreamItemKind.STREAM_STARTED, 0),
            *(
                canonical_item_from_token(
                    token,
                    sequence,
                    stream_session_id=_STREAM_SESSION_ID,
                    run_id=_RUN_ID,
                    turn_id=_TURN_ID,
                )
                for sequence, token in enumerate(tokens, start=1)
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                4,
                tool_call_id="call-stable",
                data={"name": "math.calculator"},
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
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                7,
                tool_call_id="call-stable",
                text_delta="4",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                8,
                tool_call_id="call-stable",
                data={"result": "4"},
            ),
            _answer_done(9),
            _usage_item(10, {"output_tokens": response.output_token_count}),
            _control_item(
                StreamItemKind.STREAM_COMPLETED,
                11,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
        trace = StreamGoldenTrace(
            name="response-tool-call-deltas-and-result",
            items=tuple(items),
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
            [
                item.get("correlation", {}).get("tool_call_id")
                for item in trace.to_fixture()["items"]  # type: ignore[index]
                if item["channel"] in {"tool_call", "tool_execution"}
            ],
            ["call-stable"] * 7,
        )

    async def test_ds4_tool_call_trace_captures_current_chunk_shape(
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
        trace = _trace_from_tokens(
            "response-ds4-tool-call-current-shape",
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
            ["legacy-tool-call", "legacy-tool-call", "ds4_tool_abc"],
        )

    async def test_usage_callbacks_and_cancelled_terminal_trace(
        self,
    ) -> None:
        class UsageFactory:
            usage = {"input_tokens": 3, "output_tokens": 2}
            provider_family = "openai"

            def __call__(self, **_: object):
                async def gen():
                    yield "one "
                    yield "two"

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

        async def cancelled_gen():
            yield "before "
            raise CancelledError()

        cancelled_response = _response(lambda **_: cancelled_gen())
        cancelled_iterator = cancelled_response.__aiter__()
        first = await cancelled_iterator.__anext__()
        with self.assertRaises(CancelledError):
            await cancelled_iterator.__anext__()
        trace = _trace_from_tokens(
            "response-provider-cancelled",
            (first,),
            terminal_kind=StreamItemKind.STREAM_CANCELLED,
        )

        self.assertEqual(
            trace.to_fixture()["items"][-1],
            {
                "stream_session_id": _STREAM_SESSION_ID,
                "run_id": _RUN_ID,
                "turn_id": _TURN_ID,
                "sequence": 3,
                "kind": "stream.cancelled",
                "channel": "control",
                "visibility": "public",
                "terminal_outcome": "cancelled",
            },
        )
        self.assertEqual(
            accumulate_canonical_stream_items(trace.items).terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
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
                        protocol_item_id="sdk-token-1",
                        channel=StreamChannel.ANSWER,
                        text_delta="Hel",
                        data={"item_type": "str"},
                        metadata={"protocol": "sdk.async_iteration"},
                    ),
                    _protocol_item(
                        StreamItemKind.ANSWER_DELTA,
                        2,
                        protocol_item_id="sdk-token-2",
                        channel=StreamChannel.ANSWER,
                        text_delta="lo",
                        data={"item_type": "Token"},
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
                        protocol_item_id="cli-token-frame",
                        channel=StreamChannel.ANSWER,
                        text_delta="answer",
                        data={"renderable": "tokens"},
                        metadata={"protocol": "cli.theme"},
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
                    _answer_done(4),
                    _usage_item(5, {"output_tokens": 3}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        6,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    _protocol_item(
                        StreamItemKind.STREAM_CLOSED,
                        7,
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
                        StreamItemKind.TOOL_EXECUTION_OUTPUT,
                        1,
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
                        2,
                        protocol_item_id="mcp-tool-result",
                        data={
                            "result": {
                                "content": [{"type": "text", "text": "final"}]
                            }
                        },
                        metadata={"protocol": "mcp"},
                    ),
                    _usage_item(3, {"output_tokens": 1}),
                    _control_item(
                        StreamItemKind.STREAM_COMPLETED,
                        4,
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
            {"sdk-token-1", "sdk-token-2"},
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
        self.assertEqual(
            fixtures["protocol-mcp"]["items"][1]["data"]["method"],  # type: ignore[index]
            "notifications/resources/updated",
        )
        self.assertEqual(
            fixtures["protocol-a2a"]["items"][-1]["data"]["status"],  # type: ignore[index]
            {"state": "completed"},
        )

        with self.assertRaises(AssertionError):
            StreamItemCorrelation(protocol_item_id="")

    async def test_reasoning_token_missing_source_id_is_stable(self) -> None:
        async def gen():
            yield Token(token="<think>")

        response = _response(lambda **_: gen())
        tokens = [token async for token in response]

        self.assertEqual(len(tokens), 1)
        self.assertIsInstance(tokens[0], ReasoningToken)
        self.assertEqual(tokens[0].id, -1)
