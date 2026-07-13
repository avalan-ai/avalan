from typing import Any, cast
from unittest import TestCase

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamVisibility,
)
from avalan.server.entities import (
    SKILL_CONTENT_REDACTION,
    ServerOutputRedactionSettings,
)
from avalan.server.routers.streaming import (
    PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT,
    PROTOCOL_REASONING_REDACTION_MARKER_UTF8_BYTE_COUNT,
    ProtocolReasoningAdmission,
    ProtocolReasoningIdentity,
    ProtocolReasoningRedactedText,
    ProtocolReasoningRedactionState,
)

_SETTINGS = ServerOutputRedactionSettings(
    enabled=True,
    rules=frozenset({"skill_body_echoes"}),
)


def _identity(
    ordinal: int,
    *,
    provider_item_id: str | None = "reasoning-1",
    output_index: int | None = 0,
    summary_index: int | None = 0,
    continuation_id: str | None = "continuation-1",
    representation: StreamReasoningRepresentation = (
        StreamReasoningRepresentation.SUMMARY
    ),
) -> ProtocolReasoningIdentity:
    return ProtocolReasoningIdentity(
        representation=representation,
        segment_instance_ordinal=ordinal,
        provider_item_id=provider_item_id,
        output_index=output_index,
        summary_index=summary_index,
        continuation_id=continuation_id,
    )


class ProtocolReasoningRedactionTestCase(TestCase):
    def test_identity_from_item_preserves_required_and_optional_fields(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=0,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            correlation=StreamItemCorrelation(
                protocol_item_id="reasoning-1",
                provider_output_index=2,
                provider_summary_index=3,
                model_continuation_id="continuation-1",
            ),
            text_delta="plan",
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=StreamReasoningRepresentation.SUMMARY,
            segment_instance_ordinal=4,
        )

        identity = ProtocolReasoningIdentity.from_item(item)

        self.assertEqual(
            identity.representation, item.reasoning_representation
        )
        self.assertEqual(identity.segment_instance_ordinal, 4)
        self.assertEqual(identity.provider_item_id, "reasoning-1")
        self.assertEqual(identity.output_index, 2)
        self.assertEqual(identity.summary_index, 3)
        self.assertEqual(identity.continuation_id, "continuation-1")

        invalid_values = (
            lambda: ProtocolReasoningIdentity(
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_instance_ordinal=cast(Any, True),
            ),
            lambda: ProtocolReasoningIdentity(
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_instance_ordinal=0,
                output_index=cast(Any, True),
            ),
            lambda: ProtocolReasoningIdentity(
                representation=StreamReasoningRepresentation.SUMMARY,
                segment_instance_ordinal=0,
                provider_item_id="",
            ),
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(AssertionError):
                    invalid()

    def test_preview_accounts_raw_pending_and_fixed_marker_without_mutation(
        self,
    ) -> None:
        state = ProtocolReasoningRedactionState(_SETTINGS, protocol="mcp")
        identity = _identity(0)
        candidate = "# Demo Skill\n\n"

        admission = state.preview_push(identity, candidate)

        self.assertEqual(admission.candidate_character_count, len(candidate))
        self.assertEqual(
            admission.candidate_utf8_byte_count,
            len(candidate.encode("utf-8")),
        )
        self.assertEqual(
            admission.required_character_count,
            len(candidate)
            + PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT,
        )
        self.assertEqual(
            admission.required_utf8_byte_count,
            len(candidate.encode("utf-8"))
            + PROTOCOL_REASONING_REDACTION_MARKER_UTF8_BYTE_COUNT,
        )
        self.assertTrue(admission.marker_reserved)
        self.assertFalse(admission.suppressed)
        self.assertEqual(state.pending_character_count, 0)
        self.assertEqual(state.pending_utf8_byte_count, 0)
        self.assertFalse(state.marker_reserved)

        self.assertEqual(state.push(identity, candidate), ())
        self.assertIs(state.identity, identity)
        self.assertEqual(state.pending_character_count, len(candidate))
        self.assertEqual(
            state.pending_utf8_byte_count,
            len(candidate.encode("utf-8")),
        )
        self.assertTrue(state.marker_reserved)

        same_identity = state.preview_push(identity, "Use when private.")
        self.assertEqual(
            same_identity.required_character_count,
            len(candidate)
            + len("Use when private.")
            + PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT,
        )

        boundary = state.preview_push(_identity(1), "private")
        self.assertEqual(
            boundary.required_character_count,
            PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT,
        )
        self.assertEqual(
            boundary.required_utf8_byte_count,
            PROTOCOL_REASONING_REDACTION_MARKER_UTF8_BYTE_COUNT,
        )
        self.assertTrue(boundary.marker_reserved)
        self.assertTrue(boundary.suppressed)

    def test_identity_boundary_marker_latches_until_a_fresh_owner(
        self,
    ) -> None:
        old_identity = _identity(0)
        new_identity = _identity(1, summary_index=1)
        later_identity = _identity(
            2,
            provider_item_id="reasoning-2",
            output_index=1,
            summary_index=0,
            continuation_id="continuation-2",
        )
        state = ProtocolReasoningRedactionState(_SETTINGS, protocol="a2a")

        self.assertEqual(state.push(old_identity, "# Demo Skill\n\n"), ())
        emitted = state.push(new_identity, "CROSS_PART_SECRET")

        self.assertEqual(
            emitted,
            (
                ProtocolReasoningRedactedText(
                    identity=old_identity,
                    text=SKILL_CONTENT_REDACTION,
                ),
            ),
        )
        self.assertTrue(state.redaction_latched)
        self.assertEqual(state.pending_character_count, 0)
        self.assertEqual(state.push(new_identity, "later secret"), ())
        self.assertEqual(state.complete(new_identity), ())
        self.assertEqual(state.push(later_identity, "second later secret"), ())
        latched_admission = state.preview_push(later_identity, "suppressed")
        self.assertTrue(latched_admission.suppressed)
        self.assertEqual(latched_admission.required_character_count, 0)

        fresh = ProtocolReasoningRedactionState(_SETTINGS, protocol="a2a")
        self.assertEqual(
            fresh.push(later_identity, "fresh stream"),
            (
                ProtocolReasoningRedactedText(
                    identity=later_identity,
                    text="fresh stream",
                ),
            ),
        )

    def test_completion_resolves_pending_under_the_old_identity(self) -> None:
        identity = _identity(0)
        state = ProtocolReasoningRedactionState(_SETTINGS, protocol="mcp")

        self.assertEqual(state.push(identity, "# Demo Skill\n\n"), ())

        self.assertEqual(
            state.complete(identity),
            (
                ProtocolReasoningRedactedText(
                    identity=identity,
                    text=SKILL_CONTENT_REDACTION,
                ),
            ),
        )
        self.assertTrue(state.redaction_latched)
        self.assertEqual(state.pending_character_count, 0)

    def test_correlation_loss_quarantines_exactly_one_segment(self) -> None:
        old_identity = _identity(0)
        lost_identity = _identity(
            1,
            provider_item_id=None,
            output_index=None,
            summary_index=None,
            continuation_id=None,
        )
        resumed_identity = _identity(
            2,
            provider_item_id="reasoning-2",
            output_index=1,
            summary_index=0,
            continuation_id="continuation-2",
        )
        state = ProtocolReasoningRedactionState(_SETTINGS, protocol="mcp")

        self.assertEqual(
            state.push(old_identity, "ordinary"),
            (
                ProtocolReasoningRedactedText(
                    identity=old_identity,
                    text="ordinary",
                ),
            ),
        )
        admission = state.preview_push(lost_identity, "unidentified")
        self.assertTrue(admission.suppressed)
        self.assertEqual(admission.required_character_count, 0)
        self.assertEqual(state.push(lost_identity, "unidentified"), ())
        self.assertEqual(state.push(lost_identity, "still quarantined"), ())
        recovery_admission = state.preview_push(resumed_identity, "resumed")
        self.assertFalse(recovery_admission.suppressed)
        self.assertEqual(state.complete(lost_identity), ())
        self.assertEqual(
            state.push(resumed_identity, "resumed"),
            (
                ProtocolReasoningRedactedText(
                    identity=resumed_identity,
                    text="resumed",
                ),
            ),
        )

    def test_explicit_identity_loss_and_mismatched_completion_quarantine(
        self,
    ) -> None:
        old_identity = _identity(0)
        quarantined_identity = _identity(1, summary_index=1)
        resumed_identity = _identity(2, summary_index=2)
        state = ProtocolReasoningRedactionState(_SETTINGS, protocol="mcp")

        self.assertTrue(state.push(old_identity, "ordinary"))
        self.assertEqual(state.push(None, "unidentified"), ())
        self.assertTrue(
            state.preview_push(quarantined_identity, "quarantined").suppressed
        )
        self.assertEqual(state.complete(quarantined_identity), ())
        self.assertTrue(state.push(resumed_identity, "resumed"))

        mismatched = ProtocolReasoningRedactionState(
            _SETTINGS,
            protocol="a2a",
        )
        self.assertTrue(mismatched.push(old_identity, "ordinary"))
        self.assertEqual(mismatched.complete(quarantined_identity), ())
        self.assertEqual(
            mismatched.push(quarantined_identity, "quarantined"), ()
        )
        self.assertEqual(mismatched.complete(quarantined_identity), ())
        self.assertTrue(mismatched.push(resumed_identity, "resumed"))

    def test_safe_identity_change_and_direct_marker_are_deterministic(
        self,
    ) -> None:
        old_identity = _identity(0)
        new_identity = _identity(1, summary_index=1)
        safe = ProtocolReasoningRedactionState(_SETTINGS, protocol="mcp")

        self.assertTrue(safe.push(old_identity, "old"))
        self.assertEqual(
            safe.push(new_identity, "new"),
            (
                ProtocolReasoningRedactedText(
                    identity=new_identity,
                    text="new",
                ),
            ),
        )

        direct = ProtocolReasoningRedactionState(_SETTINGS, protocol="a2a")
        emitted = direct.push(
            old_identity,
            "# Demo Skill\n\nUse when private.\nDIRECT_SECRET",
        )
        self.assertEqual(
            emitted,
            (
                ProtocolReasoningRedactedText(
                    identity=old_identity,
                    text=SKILL_CONTENT_REDACTION,
                ),
            ),
        )
        self.assertTrue(direct.redaction_latched)

        lost_with_pending = ProtocolReasoningRedactionState(
            _SETTINGS,
            protocol="mcp",
        )
        self.assertEqual(
            lost_with_pending.push(old_identity, "# Demo Skill\n\n"), ()
        )
        lost_identity = _identity(
            1,
            provider_item_id=None,
            output_index=None,
            summary_index=None,
            continuation_id=None,
        )
        self.assertEqual(
            lost_with_pending.push(lost_identity, "unidentified"),
            (
                ProtocolReasoningRedactedText(
                    identity=old_identity,
                    text=SKILL_CONTENT_REDACTION,
                ),
            ),
        )
        self.assertTrue(lost_with_pending.redaction_latched)

    def test_admission_and_state_reject_invalid_protocol_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            ProtocolReasoningRedactionState(
                _SETTINGS,
                protocol=cast(Any, "openai"),
            )
        with self.assertRaises(AssertionError):
            ProtocolReasoningAdmission(
                candidate_character_count=cast(Any, True),
                candidate_utf8_byte_count=0,
                required_character_count=0,
                required_utf8_byte_count=0,
                marker_reserved=False,
                suppressed=False,
            )
