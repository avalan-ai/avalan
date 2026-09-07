"""Recover requested ranges without loading unrelated retained decisions."""

from .records_test import NOW, OWNER
from .registration_test import registration

from dataclasses import replace
from datetime import timedelta
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.task.queues.memory_submission import (
    MemoryTaskSubmissionParticipant,
)
from avalan.task.stores.memory import InMemoryTaskStore
from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerCommitOutcome,
)
from avalan.trigger.definition import (
    IntervalTrigger,
    MisfirePolicy,
    RecurringPolicy,
)
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.plan import (
    PlannedCoverage,
    PlannedOccurrence,
    plan_admission,
)
from avalan.trigger.records import OccurrenceDisposition, TriggerCoverageSpan
from avalan.trigger.stores.memory import InMemoryTriggerStore
from avalan.trigger.stores.memory_admission import MemoryTriggerAdmissionStore


class MemoryHistoryTest(IsolatedAsyncioTestCase):
    async def test_retained_occurrences_do_not_exhaust_requested_recovery(
        self,
    ) -> None:
        now = NOW
        tasks = InMemoryTaskStore()
        store = InMemoryTriggerStore(OWNER, clock=lambda: now)
        admission = MemoryTriggerAdmissionStore(
            store,
            MemoryTaskSubmissionParticipant(tasks),
            tasks._artifact_ownership,
        )
        current = await store.apply(
            replace(
                registration(),
                schedule=IntervalTrigger(every_seconds=60, start_at=NOW),
                policy=RecurringPolicy(misfire_grace_seconds=0),
            ),
            expected_generation=None,
        )
        first = None
        for index in range(1002):
            assert current.state.next_at is not None
            now = current.state.next_at + timedelta(microseconds=1)
            plan = plan_admission(current, now)
            assert len(plan.decisions) == 1 and not plan.admission_ids
            prepared = PreparedTriggerAdmission(plan=plan, submissions=())
            result = await admission.admit(prepared)
            assert result.outcome == TriggerCommitOutcome.COMMITTED
            assert result.snapshot is not None
            if index == 0:
                first = prepared
            current = result.snapshot
        assert len(store._occurrences["daily"]) == 1002
        assert not tasks._runs
        assert first is not None
        recovered = await admission.recover(first.plan)
        repeated = await admission.admit(first)
        assert (
            recovered.outcome
            == repeated.outcome
            == TriggerCommitOutcome.COMMITTED
        )
        assert recovered.resolved == repeated.resolved
        assert recovered.resolved[0].decisions == (
            store._occurrences["daily"][0],
        )
        assert (await store.inspect("daily")) == current
        wide = replace(
            first.plan,
            decisions=(
                PlannedCoverage(
                    first_at=NOW,
                    until_at=now + timedelta(seconds=60),
                    disposition=OccurrenceDisposition.EXPIRED,
                ),
            ),
        )
        with raises(TriggerError) as error:
            await admission.recover(wide)
        assert error.value.code == TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED
        assert error.value.path == "recovery.history"

    async def test_span_intersection_keeps_half_open_boundaries(self) -> None:
        now = NOW
        tasks = InMemoryTaskStore()
        store = InMemoryTriggerStore(OWNER, clock=lambda: now)
        admission = MemoryTriggerAdmissionStore(
            store,
            MemoryTaskSubmissionParticipant(tasks),
            tasks._artifact_ownership,
        )
        current = await store.apply(
            replace(
                registration(),
                schedule=IntervalTrigger(every_seconds=60, start_at=NOW),
                policy=RecurringPolicy(misfire=MisfirePolicy.SKIP),
            ),
            expected_generation=None,
        )
        plans = []
        for _ in range(3):
            assert current.state.next_at is not None
            now = current.state.next_at + timedelta(minutes=3)
            plan = plan_admission(current, now)
            result = await admission.admit(
                PreparedTriggerAdmission(plan=plan, submissions=())
            )
            assert (
                result.outcome == TriggerCommitOutcome.COMMITTED
                and result.snapshot is not None
            )
            plans.append(plan)
            current = result.snapshot
        before, selected, after = store._spans["daily"]
        assert isinstance(selected, TriggerCoverageSpan)
        assert (
            before.until_at == selected.first_at
            and selected.until_at == after.first_at
        )
        assert admission._history(plans[1]) == (selected,)
        for instant, expected in (
            (selected.first_at, selected),
            (selected.until_at, after),
        ):
            query = replace(
                plans[1],
                decisions=(
                    PlannedOccurrence(
                        scheduled_at=instant,
                        disposition=OccurrenceDisposition.EXPIRED,
                    ),
                ),
            )
            assert admission._history(query) == (expected,)
        assert admission._history(replace(plans[1], decisions=())) == ()
        other_revision = replace(
            plans[1],
            snapshot=replace(
                plans[1].snapshot,
                definition=replace(plans[1].snapshot.definition, revision=2),
                state=replace(plans[1].snapshot.state, revision=2),
            ),
        )
        assert admission._history(other_revision) == ()
