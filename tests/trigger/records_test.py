from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Protocol, TypeVar, cast

from pytest import mark, raises

from avalan.trigger import (
    AtPolicy,
    AtTrigger,
    CronTrigger,
    IntervalTrigger,
    RecurringPolicy,
    TriggerError,
    TriggerErrorCode,
)
from avalan.trigger.records import (
    OccurrenceDisposition,
    OwnerScopeId,
    TriggerCoverageSpan,
    TriggerDefinition,
    TriggerEvent,
    TriggerEventKind,
    TriggerOccurrence,
    TriggerSealedInput,
    TriggerSnapshot,
    TriggerState,
    TriggerStatus,
    occurrence_id,
)
from avalan.trigger.schedule import ScheduleProvenance

NOW = datetime(2026, 9, 7, tzinfo=UTC)
OWNER = OwnerScopeId(value="owner")


_Record = TypeVar("_Record")


class _RecordReplacer(Protocol[_Record]):
    def __call__(self, value: _Record, **changes: object) -> _Record: ...


def invalid_replace(value: _Record, **changes: object) -> _Record:
    """Send intentionally malformed fixture fields through real validation."""
    return cast(_RecordReplacer[_Record], replace)(value, **changes)


def definition() -> TriggerDefinition:
    return TriggerDefinition(
        owner_scope_id=OWNER,
        trigger_id="trigger",
        revision=1,
        name="daily",
        task_definition_id="task",
        execution_deployment_id="deployment",
        semantic_hash="1" * 64,
        schedule=IntervalTrigger(every_seconds=60),
        input=TriggerSealedInput(
            ciphertext=b"encrypted-test-value", key_id="key", algorithm="test"
        ),
        policy=RecurringPolicy(),
        registered_at=NOW,
        resolved_anchor=NOW + timedelta(seconds=60),
        schedule_provenance=ScheduleProvenance(
            semantics_version=1, parser_version="builtin", timezone_data="UTC"
        ),
    )


def state() -> TriggerState:
    return TriggerState(
        owner_scope_id=OWNER,
        trigger_id="trigger",
        revision=1,
        generation=1,
        status=TriggerStatus.ACTIVE,
        next_at=NOW,
        last_processed_at=NOW,
    )


def occurrence() -> TriggerOccurrence:
    return TriggerOccurrence(
        owner_scope_id=OWNER,
        trigger_id="trigger",
        revision=1,
        occurrence_id=occurrence_id(OWNER, "trigger", 1, NOW),
        scheduled_at=NOW,
        decided_at=NOW,
        disposition=OccurrenceDisposition.ADMITTED,
        run_id="run",
    )


def span() -> TriggerCoverageSpan:
    return TriggerCoverageSpan(
        owner_scope_id=OWNER,
        trigger_id="trigger",
        revision=1,
        span_id="span",
        first_at=NOW,
        until_at=NOW + timedelta(seconds=60),
        decided_at=NOW,
        disposition=OccurrenceDisposition.SUPERSEDED,
    )


def event() -> TriggerEvent:
    return TriggerEvent(
        owner_scope_id=OWNER,
        trigger_id="trigger",
        revision=1,
        generation=1,
        event_id="event",
        kind=TriggerEventKind.DECIDED,
        recorded_at=NOW,
        occurrence_ids=(occurrence().occurrence_id,),
        span_ids=("span",),
    )


def test_occurrence_identity_is_utc_stable_and_scoped() -> None:
    same = NOW.astimezone(timezone(timedelta(hours=3)))
    identity = occurrence_id(OWNER, "trigger", 1, NOW)
    assert identity == occurrence_id(OWNER, "trigger", 1, same)
    assert identity != occurrence_id(
        OwnerScopeId(value="other"), "trigger", 1, NOW
    )
    assert identity != occurrence_id(OWNER, "trigger", 2, NOW)
    assert identity != occurrence_id(OWNER, "other", 1, NOW)
    assert identity != occurrence_id(
        OWNER, "trigger", 1, NOW + timedelta(microseconds=1)
    )
    assert "owner" not in repr(OWNER)
    with raises(TriggerError):
        OwnerScopeId(value=" ")
    with raises(TriggerError):
        occurrence_id(OWNER, "trigger", True, NOW)


def test_records_are_immutable_and_input_is_opaque() -> None:
    metadata = {"purpose": "test"}
    sealed = TriggerSealedInput(
        ciphertext=b"opaque", key_id="key", algorithm="test", metadata=metadata
    )
    metadata["purpose"] = "changed"
    assert sealed.metadata["purpose"] == "test"
    assert "opaque" not in repr(sealed)
    assert "encrypted-test-value" not in repr(definition())
    with raises(FrozenInstanceError):
        setattr(state(), "generation", 2)
    with raises(TypeError):
        cast(dict[str, str], sealed.metadata)["purpose"] = "changed"


@mark.parametrize(
    "changes",
    [
        {"ciphertext": b""},
        {"ciphertext": "plaintext"},
        {"metadata": {1: "invalid"}},
        {"metadata": {"key": 1}},
        {"metadata": None},
        {"metadata": {"": "invalid"}},
        {"key_id": ""},
        {"algorithm": ""},
    ],
)
def test_sealed_input_rejects_malformed_fields(
    changes: dict[str, object],
) -> None:
    with raises(TriggerError):
        invalid_replace(definition().input, **changes)


@mark.parametrize(
    "changes",
    [
        {"name": "Bad Name"},
        {"semantic_hash": "x" * 64},
        {"schema_version": 2},
        {"schedule_semantics_version": True},
        {"revision": 0},
        {"schedule": "interval"},
        {"policy": AtPolicy()},
        {"resolved_anchor": None},
        {"schedule": IntervalTrigger(every_seconds=60, start_at=NOW)},
        {"schedule": AtTrigger(at=NOW), "policy": AtPolicy()},
        {
            "schedule_provenance": ScheduleProvenance(
                semantics_version=True,
                parser_version="builtin",
                timezone_data="UTC",
            )
        },
        {
            "schedule_provenance": ScheduleProvenance(
                semantics_version=2,
                parser_version="builtin",
                timezone_data="UTC",
            )
        },
        {"task_definition_id": ""},
    ],
)
def test_definition_rejects_invalid_registered_identity(
    changes: dict[str, object],
) -> None:
    with raises(TriggerError):
        invalid_replace(definition(), **changes)


def test_definition_accepts_each_registered_schedule() -> None:
    base = definition()
    assert (
        replace(
            base,
            schedule=AtTrigger(at=NOW),
            policy=AtPolicy(),
            resolved_anchor=None,
        ).resolved_anchor
        is None
    )
    assert replace(
        base,
        schedule=CronTrigger(expression="* * * * *"),
        resolved_anchor=None,
    ).schedule
    assert (
        replace(
            base,
            schedule=IntervalTrigger(
                every_seconds=60, start_at=base.resolved_anchor
            ),
        ).resolved_anchor
        == base.resolved_anchor
    )


@mark.parametrize(
    "changes",
    [
        {"generation": 0},
        {"status": TriggerStatus.EXHAUSTED},
        {"next_at": None},
        {"status": TriggerStatus.ERROR},
        {"retry_after": NOW},
        {"last_processed_at": None},
        {
            "status": TriggerStatus.EXHAUSTED,
            "next_at": None,
            "retry_after": NOW,
            "failure_count": 1,
        },
    ],
)
def test_state_rejects_ambiguous_cursor_and_retry_values(
    changes: dict[str, object],
) -> None:
    with raises(TriggerError):
        invalid_replace(state(), **changes)


def test_state_preserves_retries_and_snapshot_identity() -> None:
    active = replace(
        state(),
        failure_count=1,
        retry_after=NOW,
        last_error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
    )
    assert active.retry_after == NOW
    error = replace(active, status=TriggerStatus.ERROR, retry_after=None)
    assert error.status is TriggerStatus.ERROR
    assert (
        TriggerSnapshot(definition=definition(), state=state()).state
        == state()
    )
    for field, value in (
        ("trigger_id", "other"),
        ("revision", 2),
        ("owner_scope_id", OwnerScopeId(value="other")),
    ):
        with raises(TriggerError):
            TriggerSnapshot(
                definition=definition(),
                state=invalid_replace(state(), **{field: value}),
            )


@mark.parametrize(
    "changes",
    [
        {"occurrence_id": "wrong"},
        {"run_id": None},
        {"run_id": ""},
        {"decided_at": NOW - timedelta(microseconds=1)},
        {"disposition": OccurrenceDisposition.EXPIRED},
    ],
)
def test_occurrence_requires_one_new_run_only_when_admitted(
    changes: dict[str, object],
) -> None:
    with raises(TriggerError):
        invalid_replace(occurrence(), **changes)


def test_skip_has_no_dispatch_time_or_run() -> None:
    assert occurrence().dispatched_at == NOW
    skipped = replace(
        occurrence(), disposition=OccurrenceDisposition.EXPIRED, run_id=None
    )
    assert skipped.dispatched_at is None


@mark.parametrize(
    "changes",
    [
        {"first_at": NOW + timedelta(seconds=60)},
        {"until_at": NOW},
        {"decided_at": NOW - timedelta(microseconds=1)},
        {"disposition": OccurrenceDisposition.ADMITTED},
        {"disposition": "expired"},
        {"exact_count": 0},
        {"exact_count": True},
    ],
)
def test_spans_reject_invalid_ranges_or_unknown_counts(
    changes: dict[str, object],
) -> None:
    with raises(TriggerError):
        invalid_replace(span(), **changes)


def test_span_exact_count_and_closed_event_fields() -> None:
    assert replace(span(), exact_count=1).exact_count == 1
    assert event().occurrence_ids == (occurrence().occurrence_id,)
    assert (
        replace(event(), error_code=TriggerErrorCode.CONFLICT).error_code
        is TriggerErrorCode.CONFLICT
    )
    with raises(TriggerError):
        replace(event(), occurrence_ids=("same", "same"))
    with raises(TriggerError):
        replace(event(), span_ids=("",))
