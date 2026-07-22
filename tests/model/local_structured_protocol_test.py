"""Verify the fail-closed local structured-output protocol boundary."""

from asyncio import run, wait_for
from collections.abc import AsyncIterator, Iterable
from math import nan
from unittest import TestCase

from avalan.model.nlp.text.local_protocol import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL,
)
from avalan.model.stream import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID,
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_METADATA_KEY,
    LocalTextStreamEventParser,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    TextGenerationNonStreamResult,
    local_tool_call_control_frame,
    normalize_provider_stream,
)

_TOOL_CALL_KINDS = frozenset(
    {
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
    }
)


def _parser_events(
    chunks: Iterable[str],
    *,
    completed: bool = True,
) -> tuple[StreamProviderEvent, ...]:
    parser = LOCAL_STRUCTURED_OUTPUT_PROTOCOL.parser()
    for chunk in chunks:
        assert parser.push(chunk) == ()
    return parser.flush(completed=completed)


def _answer_text(events: Iterable[StreamProviderEvent]) -> str:
    return "".join(
        event.text_delta or ""
        for event in events
        if event.kind is StreamItemKind.ANSWER_DELTA
    )


def _tool_events(
    events: Iterable[StreamProviderEvent],
) -> tuple[StreamProviderEvent, ...]:
    return tuple(event for event in events if event.kind in _TOOL_CALL_KINDS)


def _event_signature(event: StreamProviderEvent) -> tuple[object, ...]:
    return (
        event.kind,
        event.text_delta,
        event.data,
        event.correlation.tool_call_id,
    )


class LocalStructuredProtocolTestCase(TestCase):
    def test_empty_exact_flush_is_idempotently_empty(self) -> None:
        parser = LocalTextStreamEventParser(parse_tool_calls=True)

        self.assertEqual(parser.flush(), ())
        self.assertEqual(parser.flush(completed=False), ())

    def test_exact_and_plain_modes_preserve_distinct_public_contracts(
        self,
    ) -> None:
        frame = local_tool_call_control_frame(
            "call-1",
            "pkg.lookup",
            {"query": "safe"},
        )
        exact_events = _parser_events((frame,))
        plain = LocalTextStreamEventParser(parse_tool_calls=False)
        plain_events = (*plain.push(frame), *plain.flush())

        self.assertEqual(_answer_text(exact_events), "")
        self.assertEqual(
            [event.kind for event in _tool_events(exact_events)][-2:],
            [
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        self.assertEqual(_answer_text(plain_events), frame)
        self.assertEqual(_tool_events(plain_events), ())

    def test_adversarial_responses_are_only_complete_answer_text(self) -> None:
        frame = local_tool_call_control_frame(
            "call-1", "pkg.lookup", {"query": "safe"}
        )
        second = local_tool_call_control_frame(
            "call-2", "pkg.lookup", {"query": "other"}
        )
        cases = (
            "prefix " + frame,
            frame + " suffix",
            " " + frame,
            frame + "\n",
            frame + second,
            frame + " trailing bytes",
            frame.removesuffix(">") if frame.endswith(">") else frame[:-1],
            '<tool_call id="call-1" name="pkg.lookup">',
            '<tool_call id="call-1" name="pkg.lookup">[]</tool_call>',
            '<tool_call id="call-1" name="pkg.lookup">bad</tool_call>',
            (
                '<tool_call id="call-1" name="pkg.lookup">'
                '{"duplicate":1,"duplicate":2}</tool_call>'
            ),
            (
                '<tool_call id="call-1" name="pkg.lookup">'
                '{"value":NaN}</tool_call>'
            ),
            (
                '<tool_call>{"id":"call-1","name":"pkg.lookup",'
                '"arguments":{}}</tool_call>'
            ),
            '<tool_call id="bad\\x" name="pkg.lookup">{}</tool_call>',
            '<tool_call id="call-1" name="bad\\x">{}</tool_call>',
            '<tool_call id="bad\\u12ZZ" name="pkg.lookup">{}</tool_call>',
            '<tool_call id="call-1" name="bad\\u12ZZ">{}</tool_call>',
        )

        for text in cases:
            with self.subTest(text=text):
                stream_events = _parser_events((text,))
                non_stream = TextGenerationNonStreamResult.from_local_text(
                    text,
                    provider_family="transformers",
                    provider_event_type="transformers.generate",
                )
                non_stream_events = tuple(
                    event
                    for event in non_stream.events
                    if event.kind is not StreamItemKind.STREAM_COMPLETED
                )

                self.assertEqual(_answer_text(stream_events), text)
                self.assertEqual(_answer_text(non_stream_events), text)
                self.assertEqual(_tool_events(stream_events), ())
                self.assertEqual(_tool_events(non_stream_events), ())
                self.assertEqual(
                    tuple(map(_event_signature, stream_events)),
                    tuple(map(_event_signature, non_stream_events)),
                )
                self.assertTrue(
                    all(
                        event.metadata.get(
                            LOCAL_STRUCTURED_OUTPUT_PROTOCOL_METADATA_KEY
                        )
                        == LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID
                        for event in stream_events
                    )
                )

    def test_complete_frame_waits_for_eof_and_survives_every_boundary(
        self,
    ) -> None:
        call_id = 'call-"quoted"\\path-雪'
        name = 'pkg."quoted"\\工具'
        arguments = {"query": "café 雪", "nested": {"ok": True}}
        frame = local_tool_call_control_frame(call_id, name, arguments)

        for boundary in range(1, len(frame)):
            with self.subTest(boundary=boundary):
                events = _parser_events((frame[:boundary], frame[boundary:]))
                tool_events = _tool_events(events)
                self.assertEqual(
                    [event.kind for event in tool_events][-2:],
                    [
                        StreamItemKind.TOOL_CALL_READY,
                        StreamItemKind.TOOL_CALL_DONE,
                    ],
                )
                self.assertEqual(
                    {event.correlation.tool_call_id for event in tool_events},
                    {call_id},
                )
                ready = tool_events[-2]
                self.assertEqual(
                    ready.data,
                    {"name": name, "arguments": arguments},
                )
                self.assertEqual(_answer_text(events), "")

    def test_complete_looking_frame_is_revoked_by_later_content(self) -> None:
        frame = local_tool_call_control_frame("call-1", "pkg.lookup", {})
        trailing_values = (
            " trailing",
            local_tool_call_control_frame("call-2", "pkg.lookup", {}),
        )

        for trailing in trailing_values:
            with self.subTest(trailing=trailing):
                parser = LocalTextStreamEventParser(parse_tool_calls=True)
                self.assertEqual(parser.push(frame), ())
                self.assertEqual(parser.push(trailing), ())
                events = parser.flush()
                self.assertEqual(_answer_text(events), frame + trailing)
                self.assertEqual(_tool_events(events), ())

    def test_non_completed_stream_never_trusts_complete_frame(self) -> None:
        frame = local_tool_call_control_frame("call-1", "pkg.lookup", {})
        events = _parser_events((frame,), completed=False)

        self.assertEqual(_answer_text(events), frame)
        self.assertEqual(_tool_events(events), ())

    def test_stream_and_non_stream_preserve_escaped_identity_correlation(
        self,
    ) -> None:
        call_id = 'call-"quoted"\\path-雪'
        name = 'pkg."quoted"\\工具'
        frame = local_tool_call_control_frame(
            call_id,
            name,
            {"query": "value"},
        )
        stream_events = _parser_events((frame,))
        non_stream = TextGenerationNonStreamResult.from_local_text(
            frame,
            provider_family="mlx",
            provider_event_type="mlx.generate",
        )
        non_stream_events = tuple(
            event
            for event in non_stream.events
            if event.kind is not StreamItemKind.STREAM_COMPLETED
        )

        self.assertEqual(
            tuple(map(_event_signature, stream_events)),
            tuple(map(_event_signature, non_stream_events)),
        )
        self.assertEqual(
            {
                event.correlation.tool_call_id
                for event in stream_events
                if event.kind in _TOOL_CALL_KINDS
            },
            {call_id},
        )
        ready = next(
            event
            for event in stream_events
            if event.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data["name"], name)

    def test_fail_closed_answer_preserves_chunk_history_and_metadata(
        self,
    ) -> None:
        parser = LocalTextStreamEventParser(parse_tool_calls=True)
        chunks = (
            ("prefix ", {"token_id": 1}),
            (
                '<tool_call id="call-1" name="pkg.lookup">{}</tool_call>',
                {"token_id": 2},
            ),
        )
        for text, metadata in chunks:
            self.assertEqual(parser.push(text, metadata), ())
        events = parser.flush()

        self.assertEqual(
            [event.text_delta for event in events],
            [text for text, _ in chunks],
        )
        self.assertEqual(
            [event.metadata["token_id"] for event in events],
            [1, 2],
        )
        self.assertTrue(
            all(
                event.metadata[LOCAL_STRUCTURED_OUTPUT_PROTOCOL_METADATA_KEY]
                == LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID
                for event in events
            )
        )

    def test_finite_exact_stream_completes_under_timeout(self) -> None:
        frame = local_tool_call_control_frame("call-1", "pkg.lookup", {})

        async def provider_events() -> AsyncIterator[StreamProviderEvent]:
            parser = LocalTextStreamEventParser(parse_tool_calls=True)
            for character in frame:
                for event in parser.push(character):
                    yield event
            for event in parser.flush():
                yield event

        async def collect() -> tuple[object, ...]:
            stream = normalize_provider_stream(
                provider_events(),
                stream_session_id="local-protocol-stream",
                run_id="local-protocol-run",
                turn_id="local-protocol-turn",
                provider_family="transformers",
                capabilities=StreamProviderCapabilities(
                    backend=StreamProducerBackend.LOCAL,
                    provider_family="transformers",
                    supports_tool_calls=True,
                ),
            )
            return tuple([item async for item in stream])

        items = run(wait_for(collect(), timeout=1.0))
        self.assertTrue(
            any(item.kind is StreamItemKind.TOOL_CALL_DONE for item in items)
        )

    def test_control_frame_rejects_non_json_numbers(self) -> None:
        with self.assertRaises(ValueError):
            local_tool_call_control_frame(
                "call-1",
                "pkg.lookup",
                {"value": nan},
            )
