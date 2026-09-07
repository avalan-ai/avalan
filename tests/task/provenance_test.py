"""Check closed immutable provenance and explicit unsupported encodings."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

from pytest import raises

from avalan.task.provenance import (
    TaskProvenanceError,
    TriggerInvocationContext,
    trigger_provenance_from_payload,
    trigger_provenance_payload,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def invocation() -> TriggerInvocationContext:
    return TriggerInvocationContext(
        trigger_id="trigger",
        trigger_revision=2,
        occurrence_id="occurrence",
        scheduled_at=NOW,
        dispatched_at=NOW + timedelta(seconds=1),
    )


def test_provenance_round_trip_normalizes_utc_and_freezes_context() -> None:
    value = invocation()
    shifted = replace(
        value, scheduled_at=NOW.astimezone(timezone(timedelta(hours=3)))
    )
    assert shifted == value
    encoded = trigger_provenance_payload(value)
    assert encoded["format"] == "avalan.task.trigger_provenance"
    assert trigger_provenance_from_payload(encoded) == value
    with raises(FrozenInstanceError):
        setattr(value, "trigger_id", "changed")


def test_provenance_rejects_missing_extra_and_unsupported_envelopes() -> None:
    for invalid in (
        None,
        {},
        {"format": "old"},
        dict(trigger_provenance_payload(invocation()), extra=True),
    ):
        with raises(TaskProvenanceError, match="envelope"):
            trigger_provenance_from_payload(invalid)
    for field, value in (("format", "old"), ("version", 0), ("version", True)):
        encoded = trigger_provenance_payload(invocation())
        encoded[field] = value
        with raises(TaskProvenanceError, match="unsupported_version"):
            trigger_provenance_from_payload(encoded)
    for payload in (None, {}, {"private": "secret"}):
        encoded = trigger_provenance_payload(invocation())
        encoded["payload"] = payload
        with raises(TaskProvenanceError, match="payload"):
            trigger_provenance_from_payload(encoded)


def test_provenance_rejects_identity_precision_clock_and_calendar_errors() -> (
    None
):
    for field, value in (
        ("trigger_id", ""),
        ("occurrence_id", False),
        ("trigger_revision", True),
        ("trigger_revision", 0),
        ("scheduled_at", NOW),
        ("scheduled_at", "2026-10-01T00:00:00.1234567Z"),
        ("scheduled_at", "2026-13-01T00:00:00.000000Z"),
        ("dispatched_at", "2025-01-01T00:00:00.000000Z"),
    ):
        encoded = trigger_provenance_payload(invocation())
        payload = encoded["payload"]
        assert isinstance(payload, dict)
        payload[field] = value
        with raises(TaskProvenanceError):
            trigger_provenance_from_payload(encoded)
    with raises(AssertionError):
        replace(invocation(), scheduled_at=NOW.replace(tzinfo=None))
    with raises(AssertionError):
        replace(invocation(), dispatched_at=NOW - timedelta(seconds=1))
