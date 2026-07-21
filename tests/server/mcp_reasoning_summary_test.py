from asyncio import Event as AsyncEvent
from json import loads
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
)
from avalan.server.entities import (
    SKILL_CONTENT_REDACTION,
    ChatCompletionRequest,
    ChatMessage,
    ServerOutputRedactionSettings,
)
from avalan.server.routers import mcp as mcp_router

_REQUIRED_SEGMENT_FIELDS = {
    "representation",
    "segment_instance_ordinal",
    "text",
    "completed",
    "status",
    "terminal_outcome",
}
_OPTIONAL_SEGMENT_FIELDS = {
    "provider_item_id",
    "output_index",
    "summary_index",
    "continuation_id",
}
_FORBIDDEN_SEGMENT_FIELDS = {
    "completion",
    "truncation",
    "truncated",
    "dropped_segments",
    "dropped_characters",
    "dropped_utf8_bytes",
    "leading_segment_partial",
}
_TRUNCATION_FIELDS = {
    "truncated",
    "dropped_segments",
    "dropped_characters",
    "dropped_utf8_bytes",
    "leading_segment_partial",
}


class _Response:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._index = 0
        self._response_iterator = None
        self.input_token_count = 3
        self.output_token_count = 2
        self.cancel_count = 0
        self.close_count = 0

    def __aiter__(self) -> "_Response":
        self._index = 0
        self._response_iterator = self
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def aclose(self) -> None:
        self.close_count += 1

    async def to_str(self) -> str:
        return ""


def _reasoning_item(
    text: str,
    *,
    sequence: int = 1,
    ordinal: int = 0,
    representation: StreamReasoningRepresentation = (
        StreamReasoningRepresentation.SUMMARY
    ),
    provider_item_id: str | None = "reasoning-1",
    output_index: int | None = 0,
    summary_index: int | None = 0,
    continuation_id: str | None = "continuation-1",
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=sequence,
        kind=StreamItemKind.REASONING_DELTA,
        channel=StreamChannel.REASONING,
        correlation=StreamItemCorrelation(
            protocol_item_id=provider_item_id,
            provider_output_index=output_index,
            provider_summary_index=summary_index,
            model_continuation_id=continuation_id,
        ),
        text_delta=text,
        visibility=StreamVisibility.PRIVATE,
        reasoning_representation=representation,
        segment_instance_ordinal=ordinal,
    )


def _owner(
    *,
    policy: StreamRetentionPolicy | None = None,
    settings: ServerOutputRedactionSettings | None = None,
) -> mcp_router._MCPReasoningOwner:
    return mcp_router._MCPReasoningOwner(
        settings or ServerOutputRedactionSettings(),
        retention_policy=policy,
    )


def _assert_exact_segment_shape(segment: dict[str, object]) -> None:
    assert _REQUIRED_SEGMENT_FIELDS.issubset(segment)
    assert set(segment).issubset(
        _REQUIRED_SEGMENT_FIELDS | _OPTIONAL_SEGMENT_FIELDS
    )
    assert _FORBIDDEN_SEGMENT_FIELDS.isdisjoint(segment)
    assert segment["representation"] in {"native_text", "summary"}
    assert type(segment["segment_instance_ordinal"]) is int
    assert cast(int, segment["segment_instance_ordinal"]) >= 0
    assert type(segment["text"]) is str
    assert type(segment["completed"]) is bool
    assert segment["status"] in {"in_progress", "completed", "incomplete"}
    assert segment["terminal_outcome"] in {
        None,
        "completed",
        "failed",
        "cancelled",
        "input_required",
    }
    for field_name in ("output_index", "summary_index"):
        if field_name in segment:
            assert type(segment[field_name]) is int
            assert cast(int, segment[field_name]) >= 0


def _assert_exact_truncation_shape(truncation: dict[str, object]) -> None:
    assert set(truncation) == _TRUNCATION_FIELDS
    assert type(truncation["truncated"]) is bool
    assert type(truncation["leading_segment_partial"]) is bool
    for field_name in (
        "dropped_segments",
        "dropped_characters",
        "dropped_utf8_bytes",
    ):
        assert type(truncation[field_name]) is int
        assert cast(int, truncation[field_name]) >= 0


def _start_item() -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=0,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )


def _done_item(sequence: int) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=sequence,
        kind=StreamItemKind.REASONING_DONE,
        channel=StreamChannel.REASONING,
    )


def _terminal_item(
    sequence: int,
    outcome: StreamTerminalOutcome = StreamTerminalOutcome.COMPLETED,
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
            continuation_id="input-continuation-1",
            agent_id="agent-1",
            branch_id="branch-1",
        )
        if outcome is StreamTerminalOutcome.INPUT_REQUIRED
        else StreamItemCorrelation()
    )
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=sequence,
        kind=kind,
        channel=StreamChannel.CONTROL,
        correlation=correlation,
        usage={} if outcome is StreamTerminalOutcome.COMPLETED else None,
        terminal_outcome=outcome,
    )


def _interaction_pending_item(sequence: int) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=sequence,
        kind=StreamItemKind.INTERACTION_PENDING,
        channel=StreamChannel.INTERACTION,
        correlation=StreamItemCorrelation(
            request_id="request-1",
            continuation_id="input-continuation-1",
            agent_id="agent-1",
            branch_id="branch-1",
        ),
    )


class MCPReasoningSummaryTestCase(IsolatedAsyncioTestCase):
    async def test_streaming_final_shape_and_answer_isolation(
        self,
    ) -> None:
        generic_accumulator = mcp_router.ProtocolStreamAccumulator()
        items: list[object] = [
            _start_item(),
            _reasoning_item("plan"),
            _done_item(2),
            CanonicalStreamItem(
                stream_session_id="stream-1",
                run_id="run-1",
                turn_id="turn-1",
                sequence=3,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="stream-1",
                run_id="run-1",
                turn_id="turn-1",
                sequence=4,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            _terminal_item(5),
        ]
        response = _Response(items)
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        chunks = []

        with patch.object(
            mcp_router,
            "ProtocolStreamAccumulator",
            return_value=generic_accumulator,
        ):
            async for chunk in mcp_router._stream_mcp_response(
                request_id="request-1",
                request_model=ChatCompletionRequest(
                    model="model-1",
                    messages=[ChatMessage(role="user", content="question")],
                    stream=True,
                ),
                response=cast(Any, response),
                response_id=uuid4(),
                timestamp=123,
                progress_token="progress",
                orchestrator=orchestrator,
                logger=getLogger("mcp-reasoning-shape"),
                resource_store=mcp_router.MCPResourceStore(),
                base_path="/mcp",
                cancel_event=AsyncEvent(),
            ):
                chunks.append(chunk.decode("utf-8"))

        generic_snapshot = generic_accumulator.snapshot()
        self.assertEqual(generic_snapshot.reasoning_text, "")
        self.assertEqual(generic_snapshot.reasoning_segments, ())
        self.assertNotIn("plan", repr(generic_snapshot))
        generic_items = generic_accumulator._accumulator.items
        self.assertFalse(
            any(
                item.kind
                in (
                    StreamItemKind.REASONING_DELTA,
                    StreamItemKind.REASONING_DONE,
                )
                for item in generic_items
            )
        )
        self.assertNotIn("plan", repr(generic_items))

        messages = [loads(line) for line in "".join(chunks).splitlines()]
        reasoning_notification = next(
            message
            for message in messages
            if message.get("method") == "notifications/message"
        )
        self.assertEqual(
            reasoning_notification["params"]["data"],
            {
                "type": "reasoning",
                "delta": "plan",
                "representation": "summary",
                "segment_instance_ordinal": 0,
                "completed": False,
                "status": "in_progress",
                "terminal_outcome": None,
                "provider_item_id": "reasoning-1",
                "output_index": 0,
                "summary_index": 0,
                "continuation_id": "continuation-1",
            },
        )
        result = next(
            message["result"] for message in messages if "result" in message
        )
        structured = result["structuredContent"]
        self.assertEqual(structured["reasoning"], "plan")
        self.assertEqual(
            structured["reasoningSegments"],
            [
                {
                    "representation": "summary",
                    "segment_instance_ordinal": 0,
                    "text": "plan",
                    "completed": True,
                    "status": "completed",
                    "terminal_outcome": "completed",
                    "provider_item_id": "reasoning-1",
                    "output_index": 0,
                    "summary_index": 0,
                    "continuation_id": "continuation-1",
                }
            ],
        )
        self.assertEqual(
            structured["reasoningTruncation"],
            {
                "truncated": False,
                "dropped_segments": 0,
                "dropped_characters": 0,
                "dropped_utf8_bytes": 0,
                "leading_segment_partial": False,
            },
        )
        self.assertEqual(
            result["content"], [{"type": "text", "text": "answer"}]
        )
        self.assertNotIn("plan", repr(result["content"]))

    async def test_input_required_uses_existing_mcp_error_path(self) -> None:
        state = mcp_router._MCPStreamProjectionState(
            accumulator=mcp_router.ProtocolStreamAccumulator(),
            tool_summaries={},
            resources={},
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
        )
        await mcp_router._mcp_canonical_stream_item_notifications(
            _start_item(), state, "progress"
        )
        await mcp_router._mcp_canonical_stream_item_notifications(
            _interaction_pending_item(1), state, "progress"
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "MCP input-required projection is unavailable",
        ):
            await mcp_router._mcp_canonical_stream_item_notifications(
                _terminal_item(2, StreamTerminalOutcome.INPUT_REQUIRED),
                state,
                "progress",
            )
        self.assertIs(
            state.accumulator.terminal_outcome,
            StreamTerminalOutcome.INPUT_REQUIRED,
        )

        response = _Response(
            [
                _start_item(),
                _interaction_pending_item(1),
                _terminal_item(2, StreamTerminalOutcome.INPUT_REQUIRED),
                CanonicalStreamItem(
                    stream_session_id="stream-1",
                    run_id="run-1",
                    turn_id="turn-1",
                    sequence=3,
                    kind=StreamItemKind.STREAM_CLOSED,
                    channel=StreamChannel.CONTROL,
                ),
            ]
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        chunks: list[str] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="request-input-required",
            request_model=ChatCompletionRequest(
                model="model-1",
                messages=[ChatMessage(role="user", content="question")],
                stream=True,
            ),
            response=cast(Any, response),
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [loads(line) for line in "".join(chunks).splitlines()]
        self.assertFalse(any("result" in message for message in messages))
        errors = [
            message["error"] for message in messages if "error" in message
        ]
        self.assertEqual(
            errors,
            [
                {
                    "code": -32603,
                    "message": "An internal server error occurred.",
                }
            ],
        )

    def test_exact_segment_fields_types_state_triples_and_bool_int_mutations(
        self,
    ) -> None:
        cases = (
            (None, (False, "in_progress", None)),
            (
                StreamTerminalOutcome.COMPLETED,
                (True, "completed", "completed"),
            ),
            (
                StreamTerminalOutcome.ERRORED,
                (False, "incomplete", "failed"),
            ),
            (
                StreamTerminalOutcome.CANCELLED,
                (False, "incomplete", "cancelled"),
            ),
            (
                StreamTerminalOutcome.INPUT_REQUIRED,
                (False, "incomplete", "input_required"),
            ),
        )
        self.assertEqual(
            {outcome for outcome, _expected in cases if outcome is not None},
            set(StreamTerminalOutcome),
        )
        for outcome, expected in cases:
            with self.subTest(outcome=outcome):
                owner = _owner()
                owner.push(_reasoning_item("plan"))
                if outcome is not None:
                    owner.finish(outcome)
                segment = mcp_router._MCPReasoningOwner._segment_payload(
                    owner.segments[0]
                )
                _assert_exact_segment_shape(cast(dict[str, object], segment))
                self.assertEqual(
                    (
                        segment["completed"],
                        segment["status"],
                        segment["terminal_outcome"],
                    ),
                    expected,
                )

        native = _owner()
        native.push(
            _reasoning_item(
                "native",
                representation=StreamReasoningRepresentation.NATIVE_TEXT,
                provider_item_id=None,
                output_index=None,
                summary_index=None,
                continuation_id=None,
            )
        )
        native.finish(StreamTerminalOutcome.COMPLETED)
        native_payload = cast(
            dict[str, object],
            mcp_router._MCPReasoningOwner._segment_payload(native.segments[0]),
        )
        self.assertEqual(set(native_payload), _REQUIRED_SEGMENT_FIELDS)

        completed = _owner()
        completed.push(_reasoning_item("plan"))
        completed.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(completed.complete(), ((), None))
        payload = completed.final_payload()
        truncation = cast(dict[str, object], payload["reasoningTruncation"])
        _assert_exact_truncation_shape(truncation)
        segment = cast(list[dict[str, object]], payload["reasoningSegments"])[
            0
        ]
        for field_name in ("completed", "segment_instance_ordinal"):
            mutated = dict(segment)
            mutated[field_name] = 1 if field_name == "completed" else True
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    _assert_exact_segment_shape(mutated)
        for field_name in ("truncated", "dropped_segments"):
            mutated_truncation = dict(truncation)
            mutated_truncation[field_name] = (
                1 if field_name == "truncated" else True
            )
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    _assert_exact_truncation_shape(mutated_truncation)

    def test_retention_table_covers_ascii_unicode_and_exact_limits(
        self,
    ) -> None:
        no_overflow = _owner()
        no_overflow.push(_reasoning_item("plan"))
        no_overflow.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(no_overflow.final_payload()["reasoning"], "plan")

        oldest = _owner(
            policy=StreamRetentionPolicy(mcp_reasoning_segment_limit=1)
        )
        oldest.push(_reasoning_item("old", ordinal=0))
        oldest.complete()
        oldest.push(
            _reasoning_item(
                "new",
                ordinal=1,
                provider_item_id="reasoning-2",
                summary_index=1,
            )
        )
        oldest.finish(StreamTerminalOutcome.COMPLETED)
        oldest_payload = oldest.final_payload()
        self.assertEqual(oldest_payload["reasoning"], "new")
        self.assertEqual(
            oldest_payload["reasoningTruncation"],
            {
                "truncated": True,
                "dropped_segments": 1,
                "dropped_characters": 5,
                "dropped_utf8_bytes": 5,
                "leading_segment_partial": False,
            },
        )

        character_pressure = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_segment_limit=4,
                mcp_reasoning_character_limit=8,
                mcp_reasoning_text_byte_limit=32,
            )
        )
        character_pressure.push(_reasoning_item("old", ordinal=0))
        character_pressure.complete()
        character_pressure.push(
            _reasoning_item(
                "longer",
                ordinal=1,
                provider_item_id="reasoning-2",
                summary_index=1,
            )
        )
        character_pressure.finish(StreamTerminalOutcome.COMPLETED)
        pressure_payload = character_pressure.final_payload()
        self.assertEqual(pressure_payload["reasoning"], "longer")
        self.assertEqual(
            cast(dict[str, object], pressure_payload["reasoningTruncation"])[
                "dropped_characters"
            ],
            5,
        )

        whitespace = _owner()
        whitespace.push(_reasoning_item("old\n ", ordinal=0))
        whitespace.complete()
        whitespace.push(
            _reasoning_item(
                "\nnew",
                ordinal=1,
                provider_item_id="reasoning-2",
                summary_index=1,
            )
        )
        whitespace.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(
            whitespace.final_payload()["reasoning"], "old\n \nnew"
        )

        separated = _owner()
        separated.push(_reasoning_item("old", ordinal=0))
        separated.complete()
        separated.push(
            _reasoning_item(
                "new",
                ordinal=1,
                provider_item_id="reasoning-2",
                summary_index=1,
            )
        )
        separated.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(separated.final_payload()["reasoning"], "old\n\nnew")

        leading = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_character_limit=4,
                mcp_reasoning_text_byte_limit=16,
            )
        )
        leading.push(_reasoning_item("ab"))
        leading.push(_reasoning_item("cdef", sequence=2))
        leading.finish(StreamTerminalOutcome.COMPLETED)
        leading_payload = leading.final_payload()
        self.assertEqual(leading_payload["reasoning"], "cdef")
        self.assertEqual(
            leading_payload["reasoningTruncation"],
            {
                "truncated": True,
                "dropped_segments": 0,
                "dropped_characters": 2,
                "dropped_utf8_bytes": 2,
                "leading_segment_partial": True,
            },
        )

        oversized = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_character_limit=8,
                mcp_reasoning_text_byte_limit=8,
            )
        )
        oversized.push(_reasoning_item("oversized"))
        oversized.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(
            oversized.final_payload(),
            {
                "reasoning": "",
                "reasoningSegments": [],
                "reasoningTruncation": {
                    "truncated": True,
                    "dropped_segments": 1,
                    "dropped_characters": 9,
                    "dropped_utf8_bytes": 9,
                    "leading_segment_partial": False,
                },
            },
        )

        unicode_owner = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_character_limit=8,
                mcp_reasoning_text_byte_limit=4,
            )
        )
        unicode_owner.push(_reasoning_item("éé"))
        unicode_owner.push(_reasoning_item("é", sequence=2))
        unicode_owner.finish(StreamTerminalOutcome.COMPLETED)
        unicode_payload = unicode_owner.final_payload()
        self.assertEqual(unicode_payload["reasoning"], "éé")
        self.assertEqual(
            cast(dict[str, object], unicode_payload["reasoningTruncation"])[
                "dropped_utf8_bytes"
            ],
            2,
        )

        for limit, accepted in ((0, False), (1, True), (2, True)):
            with self.subTest(segment_limit=limit):
                owner = _owner(
                    policy=StreamRetentionPolicy(
                        mcp_reasoning_segment_limit=limit
                    )
                )
                owner.push(_reasoning_item("x"))
                owner.finish(StreamTerminalOutcome.COMPLETED)
                self.assertEqual(bool(owner.segments), accepted)

    async def test_identity_redaction_latch_and_quarantine_table(
        self,
    ) -> None:
        settings = ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skill_body_echoes", "host_paths"}),
        )
        owner = _owner(settings=settings)
        old_item = _reasoning_item("# Demo Skill\n\n", ordinal=0)
        self.assertEqual(owner.push(old_item), ())
        marker_outputs = owner.push(
            _reasoning_item(
                "CROSS_PART_SECRET",
                sequence=2,
                ordinal=1,
                provider_item_id="reasoning-2",
                summary_index=1,
            )
        )
        self.assertEqual(marker_outputs[0].text, SKILL_CONTENT_REDACTION)
        self.assertTrue(owner.redaction.redaction_latched)
        self.assertEqual(
            marker_outputs[0].identity.segment_instance_ordinal, 0
        )
        for ordinal in (2, 3):
            self.assertEqual(
                owner.push(
                    _reasoning_item(
                        "LATER_SECRET",
                        sequence=ordinal + 1,
                        ordinal=ordinal,
                        provider_item_id=f"reasoning-{ordinal + 1}",
                        summary_index=ordinal,
                    )
                ),
                (),
            )

        fresh = _owner(settings=settings)
        self.assertTrue(fresh.push(_reasoning_item("fresh stream")))

        quarantine = _owner(settings=settings)
        self.assertTrue(quarantine.push(_reasoning_item("ordinary")))
        lost = _reasoning_item(
            "unidentified",
            sequence=2,
            ordinal=1,
            provider_item_id=None,
            output_index=None,
            summary_index=None,
            continuation_id=None,
        )
        self.assertEqual(quarantine.push(lost), ())
        quarantine.complete()
        self.assertTrue(
            quarantine.push(
                _reasoning_item(
                    "resumed",
                    sequence=3,
                    ordinal=2,
                    provider_item_id="reasoning-3",
                    summary_index=2,
                )
            )
        )

        marker_size = len(SKILL_CONTENT_REDACTION)
        candidate = "# Demo Skill\n\n"
        exact = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_character_limit=len(candidate) + marker_size,
                mcp_reasoning_text_byte_limit=(
                    len(candidate.encode("utf-8"))
                    + len(SKILL_CONTENT_REDACTION.encode("utf-8"))
                ),
            ),
            settings=settings,
        )
        self.assertEqual(exact.push(_reasoning_item(candidate)), ())
        outputs, _closed = exact.complete()
        self.assertEqual(outputs[0].text, SKILL_CONTENT_REDACTION)

        rejected = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_character_limit=(
                    len(candidate) + marker_size - 1
                ),
                mcp_reasoning_text_byte_limit=(
                    len(candidate.encode("utf-8"))
                    + len(SKILL_CONTENT_REDACTION.encode("utf-8"))
                    - 1
                ),
            ),
            settings=settings,
        )
        self.assertEqual(rejected.push(_reasoning_item(candidate)), ())
        rejected.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(rejected.segments, ())
        self.assertTrue(rejected.truncated)

        recovered = _owner(
            policy=StreamRetentionPolicy(
                mcp_reasoning_character_limit=3,
                mcp_reasoning_text_byte_limit=3,
            )
        )
        self.assertEqual(recovered.push(_reasoning_item("four")), ())
        self.assertEqual(
            recovered.push(_reasoning_item("more", sequence=2)), ()
        )
        self.assertEqual(
            recovered.push(
                _reasoning_item(
                    "new",
                    sequence=3,
                    ordinal=1,
                    provider_item_id="reasoning-2",
                    summary_index=1,
                )
            ),
            (),
        )
        self.assertTrue(
            recovered.push(
                _reasoning_item(
                    "ok",
                    sequence=4,
                    ordinal=2,
                    provider_item_id="reasoning-3",
                    summary_index=2,
                )
            )
        )
        recovered.finish(StreamTerminalOutcome.COMPLETED)
        self.assertEqual(recovered.final_payload()["reasoning"], "ok")

        generic_accumulator = mcp_router.ProtocolStreamAccumulator()
        state = mcp_router._MCPStreamProjectionState(
            accumulator=generic_accumulator,
            tool_summaries={},
            resources={},
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
            output_redaction_settings=settings,
            reasoning=_owner(settings=settings),
        )
        raw_pending = "# Demo Skill\n\n"
        raw_secret = "RAW_REASONING_SECRET"
        for item in (
            _start_item(),
            _reasoning_item(raw_pending),
            _reasoning_item(
                raw_secret,
                sequence=2,
                ordinal=1,
                provider_item_id="reasoning-2",
                summary_index=1,
            ),
            _done_item(3),
            _terminal_item(4),
        ):
            await mcp_router._mcp_canonical_stream_item_notifications(
                item,
                state,
                "progress",
            )

        generic_accumulator.validate_complete()
        generic_snapshot = generic_accumulator.snapshot()
        self.assertEqual(generic_snapshot.reasoning_text, "")
        self.assertEqual(generic_snapshot.reasoning_segments, ())
        self.assertNotIn(raw_pending, repr(generic_snapshot))
        self.assertNotIn(raw_secret, repr(generic_snapshot))
        self.assertFalse(
            any(
                item.kind
                in (
                    StreamItemKind.REASONING_DELTA,
                    StreamItemKind.REASONING_DONE,
                )
                for item in generic_accumulator._accumulator.items
            )
        )
        sanitized_payload = state.reasoning_owner.final_payload()
        self.assertEqual(
            sanitized_payload["reasoning"], SKILL_CONTENT_REDACTION
        )
        self.assertNotIn(raw_pending, repr(sanitized_payload))
        self.assertNotIn(raw_secret, repr(sanitized_payload))

    async def test_abnormal_terminals_and_local_aclose_emit_no_late_content(
        self,
    ) -> None:
        response = _Response(
            [
                _start_item(),
                _reasoning_item("observed-prefix"),
                RuntimeError("provider-private-error"),
            ]
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="request-1",
            request_model=ChatCompletionRequest(
                model="model-1",
                messages=[ChatMessage(role="user", content="question")],
                stream=True,
            ),
            response=cast(Any, response),
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))
        messages = [loads(line) for line in "".join(chunks).splitlines()]
        close_index = next(
            index
            for index, message in enumerate(messages)
            if message.get("method") == "notifications/message"
            and message["params"]["data"].get("status") == "incomplete"
        )
        error_indices = [
            index
            for index, message in enumerate(messages)
            if "error" in message
        ]
        self.assertEqual(len(error_indices), 1)
        self.assertLess(close_index, error_indices[0])
        self.assertEqual(
            messages[close_index]["params"]["data"]["terminal_outcome"],
            "failed",
        )
        self.assertNotIn("provider-private-error", repr(messages))
        self.assertFalse(any("result" in message for message in messages))

        terminal_response = _Response(
            [
                _start_item(),
                _reasoning_item("terminal-prefix"),
                _terminal_item(2, StreamTerminalOutcome.ERRORED),
            ]
        )
        terminal_orchestrator = MagicMock()
        terminal_orchestrator.sync_messages = AsyncMock()
        terminal_chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="request-terminal-error",
            request_model=ChatCompletionRequest(
                model="model-1",
                messages=[ChatMessage(role="user", content="question")],
                stream=True,
            ),
            response=cast(Any, terminal_response),
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=terminal_orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
            cancel_event=AsyncEvent(),
        ):
            terminal_chunks.append(chunk.decode("utf-8"))
        terminal_messages = [
            loads(line) for line in "".join(terminal_chunks).splitlines()
        ]
        terminal_closes = [
            message["params"]["data"]
            for message in terminal_messages
            if message.get("method") == "notifications/message"
            and message["params"]["data"].get("status") == "incomplete"
        ]
        self.assertEqual(len(terminal_closes), 1)
        self.assertEqual(terminal_closes[0]["terminal_outcome"], "failed")

        cancelled_state = mcp_router._MCPStreamProjectionState(
            accumulator=mcp_router.ProtocolStreamAccumulator(),
            tool_summaries={},
            resources={},
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
        )
        await mcp_router._mcp_canonical_stream_item_notifications(
            _start_item(), cancelled_state, "progress"
        )
        await mcp_router._mcp_canonical_stream_item_notifications(
            _reasoning_item("prefix"), cancelled_state, "progress"
        )
        cancelled = mcp_router._mcp_finish_reasoning_notifications(
            cancelled_state,
            StreamTerminalOutcome.CANCELLED,
        )
        self.assertEqual(
            cancelled[-1]["params"]["data"]["terminal_outcome"],
            "cancelled",
        )
        self.assertEqual(
            mcp_router._mcp_finish_reasoning_notifications(
                cancelled_state,
                StreamTerminalOutcome.CANCELLED,
            ),
            [],
        )

        close_response = _Response(
            [
                _start_item(),
                _reasoning_item("prefix"),
                _done_item(2),
                _terminal_item(3),
            ]
        )
        close_orchestrator = MagicMock()
        close_orchestrator.sync_messages = AsyncMock()
        stream = mcp_router._stream_mcp_response(
            request_id="request-close",
            request_model=ChatCompletionRequest(
                model="model-1",
                messages=[ChatMessage(role="user", content="question")],
                stream=True,
            ),
            response=cast(Any, close_response),
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=close_orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
            cancel_event=AsyncEvent(),
        )
        first = loads((await anext(stream)).decode("utf-8"))
        self.assertEqual(first["params"]["data"]["delta"], "prefix")
        await stream.aclose()
        self.assertGreaterEqual(close_response.close_count, 1)
        close_orchestrator.sync_messages.assert_awaited_once()
