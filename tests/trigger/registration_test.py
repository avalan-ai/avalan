"""Exercise deterministic management decisions at transaction time."""

from .records_test import NOW, OWNER, definition, state

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from pytest import raises

from avalan.trigger import AtPolicy, AtTrigger, CronTrigger, IntervalTrigger
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.records import MAX_REVISION, TriggerSnapshot, TriggerStatus
from avalan.trigger.registration import (
    TriggerRegistration,
    apply_registration,
    record_failure,
    set_enabled,
)


def registration() -> TriggerRegistration:
    current = definition()
    return TriggerRegistration(
        name=current.name,
        task_definition_id=current.task_definition_id,
        execution_deployment_id=current.execution_deployment_id,
        semantic_hash=current.semantic_hash,
        schedule=current.schedule,
        input=current.input,
        policy=current.policy,
        schedule_provenance=current.schedule_provenance,
    )


def snapshot() -> TriggerSnapshot:
    return TriggerSnapshot(definition=definition(), state=state())


def test_create_resolves_anchor_and_accepts_initial_equal_at() -> None:
    result = apply_registration(
        OWNER,
        registration(),
        None,
        expected_generation=None,
        decision_time=NOW,
        trigger_id="trigger",
    )
    assert result.definition_created
    assert (
        result.snapshot.state.generation == result.snapshot.state.revision == 1
    )
    assert result.snapshot.state.next_at == NOW + timedelta(seconds=60)
    assert (
        result.snapshot.definition.resolved_anchor
        == result.snapshot.state.next_at
    )
    assert result.event is not None
    at = replace(
        registration(),
        schedule=AtTrigger(at=NOW),
        policy=AtPolicy(),
        desired_enabled=False,
    )
    paused = apply_registration(
        OWNER,
        at,
        None,
        expected_generation=None,
        decision_time=NOW,
        trigger_id="trigger",
    )
    assert paused.snapshot.state.next_at == NOW
    assert paused.snapshot.state.status is TriggerStatus.PAUSED
    cron = replace(
        registration(), schedule=CronTrigger(expression="* * * * *")
    )
    assert apply_registration(
        OWNER,
        cron,
        None,
        expected_generation=None,
        decision_time=NOW,
        trigger_id="trigger",
    ).snapshot.state.next_at == NOW + timedelta(minutes=1)


def test_apply_requires_existing_generation_even_for_noop() -> None:
    with raises(TriggerError):
        apply_registration(
            OWNER,
            registration(),
            None,
            expected_generation=1,
            decision_time=NOW,
            trigger_id="trigger",
        )
    for generation in (None, 0, 2):
        with raises(TriggerError):
            apply_registration(
                OWNER,
                registration(),
                snapshot(),
                expected_generation=generation,
                decision_time=NOW,
                trigger_id="trigger",
            )
    result = apply_registration(
        OWNER,
        registration(),
        snapshot(),
        expected_generation=1,
        decision_time=NOW,
        trigger_id="trigger",
    )
    assert result.snapshot == snapshot()
    assert result.event is None
    with raises(TriggerError):
        apply_registration(
            OWNER,
            registration(),
            snapshot(),
            expected_generation=1,
            decision_time=NOW,
            trigger_id="other",
        )


def test_replacement_closes_old_due_range_and_preserves_interval_anchor() -> (
    None
):
    now = NOW + timedelta(minutes=5)
    result = apply_registration(
        OWNER,
        replace(registration(), semantic_hash="2" * 64),
        snapshot(),
        expected_generation=1,
        decision_time=now,
        trigger_id="trigger",
    )
    assert result.closed_revision == 1
    assert (
        result.snapshot.state.revision == result.snapshot.state.generation == 2
    )
    assert (
        result.snapshot.definition.resolved_anchor
        == definition().resolved_anchor
    )
    assert result.snapshot.state.next_at == now + timedelta(minutes=1)
    assert result.superseded is not None
    assert result.superseded.first_at == NOW
    assert result.superseded.until_at == now + timedelta(microseconds=1)
    assert result.superseded.exact_count is None
    assert result.event is not None
    assert result.event.span_ids == (result.superseded.span_id,)


def test_replacement_uses_new_schedule_and_rejects_equal_one_shot() -> None:
    changed = replace(
        registration(),
        semantic_hash="2" * 64,
        schedule=IntervalTrigger(every_seconds=120),
    )
    current = replace(
        snapshot(), state=replace(state(), next_at=NOW + timedelta(hours=1))
    )
    result = apply_registration(
        OWNER,
        changed,
        current,
        expected_generation=1,
        decision_time=NOW,
        trigger_id="trigger",
    )
    assert result.superseded is None
    assert result.snapshot.state.next_at == NOW + timedelta(seconds=120)
    equal = replace(changed, schedule=AtTrigger(at=NOW), policy=AtPolicy())
    with raises(TriggerError) as error:
        apply_registration(
            OWNER,
            equal,
            snapshot(),
            expected_generation=1,
            decision_time=NOW,
            trigger_id="trigger",
        )
    assert error.value.code is TriggerErrorCode.PAST_SCHEDULE


def test_exhausted_unchanged_reapply_does_not_rearm() -> None:
    current = replace(
        snapshot(),
        state=replace(state(), status=TriggerStatus.EXHAUSTED, next_at=None),
    )
    result = apply_registration(
        OWNER,
        registration(),
        current,
        expected_generation=1,
        decision_time=NOW + timedelta(days=1),
        trigger_id="trigger",
    )
    assert result.snapshot == current
    assert result.event is None
    changed = replace(registration(), semantic_hash="2" * 64)
    assert (
        apply_registration(
            OWNER,
            changed,
            current,
            expected_generation=1,
            decision_time=NOW,
            trigger_id="trigger",
        ).superseded
        is None
    )


def test_pause_resume_and_declarative_resume_preserve_revision() -> None:
    paused = set_enabled(
        snapshot(), enabled=False, expected_generation=1, decision_time=NOW
    )
    assert paused.snapshot.state.status is TriggerStatus.PAUSED
    assert paused.snapshot.state.generation == 2
    assert (
        set_enabled(
            paused.snapshot,
            enabled=False,
            expected_generation=2,
            decision_time=NOW,
        ).event
        is None
    )
    resumed = apply_registration(
        OWNER,
        registration(),
        paused.snapshot,
        expected_generation=2,
        decision_time=NOW,
        trigger_id="trigger",
    )
    assert resumed.snapshot.state.status is TriggerStatus.ACTIVE
    assert resumed.snapshot.state.revision == 1
    assert resumed.snapshot.state.generation == 3
    with raises(TriggerError):
        set_enabled(
            resumed.snapshot,
            enabled=False,
            expected_generation=2,
            decision_time=NOW,
        )
    maximum = replace(
        snapshot(), state=replace(state(), generation=MAX_REVISION)
    )
    with raises(TriggerError):
        set_enabled(
            maximum,
            enabled=False,
            expected_generation=MAX_REVISION,
            decision_time=NOW,
        )


def test_failure_budget_persists_without_cursor_move() -> None:
    current = snapshot()
    for failure in range(1, 4):
        result = record_failure(
            current,
            expected_generation=current.state.generation,
            decision_time=NOW,
            error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
            attempts=3,
            base_seconds=2,
            max_seconds=3,
        )
        current = result.snapshot
        assert current.state.next_at == NOW
        assert current.state.failure_count == failure
        assert current.state.generation == failure + 1
        assert result.event is not None
        assert current.state.retry_after == (
            NOW + timedelta(seconds=min(3, 2**failure))
            if failure < 3
            else None
        )
    assert current.state.status is TriggerStatus.ERROR
    assert (
        current.state.last_error_code is TriggerErrorCode.ADMISSION_EXHAUSTED
    )
    with raises(TriggerError):
        record_failure(
            current,
            expected_generation=4,
            decision_time=NOW,
            error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
        )
    resumed = set_enabled(
        current, enabled=True, expected_generation=4, decision_time=NOW
    )
    assert resumed.snapshot.state.failure_count == 0
    assert resumed.snapshot.state.last_error_code is None
    permanent = record_failure(
        snapshot(),
        expected_generation=1,
        decision_time=NOW,
        error_code=TriggerErrorCode.INVALID_CONFIG,
        permanent=True,
    )
    assert permanent.snapshot.state.status is TriggerStatus.ERROR
    assert (
        permanent.snapshot.state.last_error_code
        is TriggerErrorCode.INVALID_CONFIG
    )


def test_retry_and_replacement_report_calendar_overflow() -> None:
    end = datetime.max.replace(tzinfo=UTC)
    with raises(TriggerError) as error:
        record_failure(
            snapshot(),
            expected_generation=1,
            decision_time=end,
            error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
        )
    assert error.value.code is TriggerErrorCode.DATETIME_OVERFLOW
    with raises(TriggerError) as replacement_error:
        apply_registration(
            OWNER,
            replace(registration(), semantic_hash="2" * 64),
            snapshot(),
            expected_generation=1,
            decision_time=end,
            trigger_id="trigger",
        )
    assert replacement_error.value.code is TriggerErrorCode.DATETIME_OVERFLOW
