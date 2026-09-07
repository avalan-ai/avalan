"""Check bounded proposal semantics before task preparation or locking."""

from .records_test import NOW, definition, state

from dataclasses import replace
from datetime import timedelta

from pytest import raises

from avalan.trigger import AtPolicy, AtTrigger, MisfirePolicy, RecurringPolicy
from avalan.trigger.error import TriggerError
from avalan.trigger.plan import (
    AdmissionLimits,
    PlannedCoverage,
    PlannedOccurrence,
    plan_admission,
)
from avalan.trigger.records import (
    OccurrenceDisposition,
    TriggerSnapshot,
    TriggerStatus,
)


def snapshot(policy: RecurringPolicy = RecurringPolicy()) -> TriggerSnapshot:
    return TriggerSnapshot(
        definition=replace(definition(), resolved_anchor=NOW, policy=policy),
        state=state(),
    )


def test_single_slot_uses_inclusive_grace_for_every_policy() -> None:
    for policy in MisfirePolicy:
        current = snapshot(
            RecurringPolicy(misfire=policy, misfire_grace_seconds=10)
        )
        within = plan_admission(current, NOW + timedelta(seconds=10))
        assert within.decisions == (
            PlannedOccurrence(
                scheduled_at=NOW, disposition=OccurrenceDisposition.ADMITTED
            ),
        )
        assert within.request_ids == within.admission_ids
        expired = plan_admission(
            current, NOW + timedelta(seconds=10, microseconds=1)
        )
        assert (
            expired.decisions[0].disposition is OccurrenceDisposition.EXPIRED
        )
        assert expired.admission_ids == ()


def test_latest_coalesces_and_skip_compresses_the_due_backlog() -> None:
    now = NOW + timedelta(minutes=3)
    latest = plan_admission(snapshot(), now)
    assert latest.decisions == (
        PlannedCoverage(
            first_at=NOW,
            until_at=now,
            disposition=OccurrenceDisposition.COALESCED,
        ),
        PlannedOccurrence(
            scheduled_at=now, disposition=OccurrenceDisposition.ADMITTED
        ),
    )
    limited = plan_admission(
        snapshot(), now, limits=AdmissionLimits(decisions=1)
    )
    assert limited.next_at == now
    assert len(limited.decisions) == 1
    skipped = plan_admission(
        snapshot(RecurringPolicy(misfire=MisfirePolicy.SKIP)), now
    )
    assert len(skipped.decisions) == 1
    assert (
        skipped.decisions[0].disposition
        is OccurrenceDisposition.SKIPPED_MISFIRE
    )
    assert skipped.next_at == now + timedelta(minutes=1)
    assert (
        skipped.request_ids
        == plan_admission(
            snapshot(RecurringPolicy(misfire=MisfirePolicy.SKIP)), now
        ).request_ids
    )


def test_all_compresses_expiry_and_stops_at_first_undecided_admission() -> (
    None
):
    current = snapshot(
        RecurringPolicy(misfire=MisfirePolicy.ALL, misfire_grace_seconds=60)
    )
    now = NOW + timedelta(minutes=5)
    plan = plan_admission(current, now, limits=AdmissionLimits(admissions=1))
    assert plan.decisions == (
        PlannedCoverage(
            first_at=NOW,
            until_at=NOW + timedelta(minutes=4),
            disposition=OccurrenceDisposition.EXPIRED,
        ),
        PlannedOccurrence(
            scheduled_at=NOW + timedelta(minutes=4),
            disposition=OccurrenceDisposition.ADMITTED,
        ),
    )
    assert plan.next_at == now
    one_expired = plan_admission(
        current, NOW + timedelta(minutes=1, microseconds=1)
    )
    assert isinstance(one_expired.decisions[0], PlannedOccurrence)
    assert (
        one_expired.decisions[0].disposition is OccurrenceDisposition.EXPIRED
    )


def test_no_due_work_and_exhausted_one_shot() -> None:
    assert (
        plan_admission(snapshot(), NOW - timedelta(seconds=1)).decisions == ()
    )
    current = TriggerSnapshot(
        definition=replace(
            definition(),
            schedule=AtTrigger(at=NOW),
            policy=AtPolicy(),
            resolved_anchor=None,
        ),
        state=state(),
    )
    plan = plan_admission(current, NOW)
    assert plan.next_at is None
    exhausted = replace(
        current,
        state=replace(state(), status=TriggerStatus.EXHAUSTED, next_at=None),
    )
    assert plan_admission(exhausted, NOW).decisions == ()


def test_drafts_reject_invalid_dispositions_and_duplicate_identity() -> None:
    with raises(TriggerError):
        PlannedOccurrence(
            scheduled_at=NOW, disposition=OccurrenceDisposition.SUPERSEDED
        )
    with raises(TriggerError):
        PlannedCoverage(
            first_at=NOW,
            until_at=NOW,
            disposition=OccurrenceDisposition.EXPIRED,
        )
    plan = plan_admission(snapshot(), NOW)
    with raises(TriggerError):
        replace(plan, decisions=plan.decisions * 2)
