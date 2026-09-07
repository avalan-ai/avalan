"""Check admission reduction with explicit persistence-only task fixtures."""

from .plan_test import snapshot
from .records_test import NOW, OWNER

from dataclasses import replace
from datetime import timedelta

from pytest import mark, raises
from task_submission_helpers import prepared_submission_fixture

from avalan.task.state import TaskRunState
from avalan.task.store import TaskExecutionRequest
from avalan.trigger import MisfirePolicy, OverlapPolicy, RecurringPolicy
from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    ResolvedTriggerDecision,
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerAdmissionWrite,
    TriggerCommitOutcome,
    evaluate_admission,
    recover_decisions,
)
from avalan.trigger.error import TriggerError
from avalan.trigger.plan import (
    PlannedCoverage,
    PlannedOccurrence,
    TriggerAdmissionPlan,
    plan_admission,
)
from avalan.trigger.records import (
    OccurrenceDisposition,
    TriggerCoverageSpan,
    TriggerOccurrence,
    TriggerStatus,
)


def prepared_plan(plan: TriggerAdmissionPlan) -> PreparedTriggerAdmission:
    definition = plan.snapshot.definition
    submissions = []
    for decision in plan.decisions:
        if (
            isinstance(decision, PlannedOccurrence)
            and decision.disposition is OccurrenceDisposition.ADMITTED
        ):
            item = prepared_submission_fixture(
                object(),
                TaskExecutionRequest(
                    definition_id=definition.task_definition_id
                ),
                queue_name="default",
                available_at=decision.scheduled_at,
            )
            submissions.append(
                replace(
                    item,
                    owner_scope=definition.owner_scope_id.value,
                    occurrence_id=plan.identity(decision),
                    execution_deployment_id=definition.execution_deployment_id,
                )
            )
    return PreparedTriggerAdmission(plan=plan, submissions=tuple(submissions))


def committed_decisions(
    plan: TriggerAdmissionPlan,
) -> tuple[TriggerOccurrence | TriggerCoverageSpan, ...]:
    result = evaluate_admission(
        OWNER,
        prepared_plan(plan),
        plan.snapshot,
        decision_time=plan.prepared_at,
    )
    assert isinstance(result, TriggerAdmissionWrite)
    return result.decisions


def test_prepared_values_must_match_every_planned_identity() -> None:
    prepared = prepared_plan(plan_admission(snapshot(), NOW))
    assert (
        prepared.submission(prepared.plan.admission_ids[0])
        == prepared.submissions[0]
    )
    with raises(TriggerError):
        prepared.submission("missing")
    with raises(TriggerError):
        replace(prepared, submissions=())
    for item in (
        replace(prepared.submissions[0], owner_scope="other"),
        replace(
            prepared.submissions[0], available_at=NOW + timedelta(seconds=1)
        ),
        replace(prepared.submissions[0], execution_deployment_id="other"),
        replace(
            prepared.submissions[0],
            execution=TaskExecutionRequest(
                definition_id="other", queue="default"
            ),
        ),
    ):
        with raises(TriggerError):
            replace(prepared, submissions=(item,))
    all_plan = plan_admission(
        snapshot(RecurringPolicy(misfire=MisfirePolicy.ALL)),
        NOW + timedelta(minutes=1),
    )
    pair = prepared_plan(all_plan)
    with raises(TriggerError):
        replace(
            pair,
            submissions=(
                pair.submissions[0],
                replace(
                    pair.submissions[1], run_id=pair.submissions[0].run_id
                ),
            ),
        )


def test_recovery_precedes_pause_revision_and_generation_checks() -> None:
    plan = plan_admission(snapshot(), NOW)
    decisions = committed_decisions(plan)
    paused = replace(
        plan.snapshot,
        state=replace(
            plan.snapshot.state, generation=2, status=TriggerStatus.PAUSED
        ),
    )
    result = evaluate_admission(
        OWNER,
        prepared_plan(plan),
        paused,
        decision_time=NOW,
        existing=decisions,
    )
    assert isinstance(result, TriggerAdmissionResult)
    assert result.outcome is TriggerCommitOutcome.COMMITTED
    assert result.unresolved_ids == ()
    assert result.resolved[0].decisions == decisions
    missing_current = evaluate_admission(
        OWNER,
        prepared_plan(plan),
        None,
        decision_time=NOW,
        existing=decisions,
    )
    assert isinstance(missing_current, TriggerAdmissionResult)
    assert missing_current.outcome is TriggerCommitOutcome.COMMITTED


def test_stale_or_future_plan_does_not_write_a_remainder() -> None:
    plan = plan_admission(snapshot(), NOW)
    for current in (
        None,
        replace(
            plan.snapshot, state=replace(plan.snapshot.state, generation=2)
        ),
        replace(
            plan.snapshot,
            state=replace(plan.snapshot.state, status=TriggerStatus.PAUSED),
        ),
    ):
        result = evaluate_admission(
            OWNER, prepared_plan(plan), current, decision_time=NOW
        )
        assert isinstance(result, TriggerAdmissionResult)
        assert result.outcome is TriggerCommitOutcome.NOT_COMMITTED
        assert result.unresolved_ids == plan.request_ids
    delayed = evaluate_admission(
        OWNER,
        prepared_plan(plan),
        plan.snapshot,
        decision_time=NOW + timedelta(minutes=2),
    )
    assert (
        isinstance(delayed, TriggerAdmissionResult) and delayed.replan_required
    )
    backward = evaluate_admission(
        OWNER,
        prepared_plan(plan),
        plan.snapshot,
        decision_time=NOW - timedelta(microseconds=1),
    )
    assert (
        isinstance(backward, TriggerAdmissionResult)
        and backward.replan_required
    )
    empty = plan_admission(snapshot(), NOW - timedelta(seconds=1))
    result = evaluate_admission(
        OWNER,
        prepared_plan(empty),
        empty.snapshot,
        decision_time=empty.prepared_at,
    )
    assert (
        isinstance(result, TriggerAdmissionResult)
        and result.outcome is TriggerCommitOutcome.NOT_COMMITTED
    )


@mark.parametrize("state", list(TaskRunState))
def test_every_canonical_nonterminal_state_blocks_overlap(
    state: TaskRunState,
) -> None:
    plan = plan_admission(snapshot(), NOW)
    result = evaluate_admission(
        OWNER,
        prepared_plan(plan),
        plan.snapshot,
        decision_time=NOW,
        outstanding_states=(state,),
    )
    assert isinstance(result, TriggerAdmissionWrite)
    terminal = state in {
        TaskRunState.SUCCEEDED,
        TaskRunState.FAILED,
        TaskRunState.CANCELLED,
        TaskRunState.EXPIRED,
    }
    assert result.decisions[0].disposition is (
        OccurrenceDisposition.ADMITTED
        if terminal
        else OccurrenceDisposition.SKIPPED_OVERLAP
    )
    assert len(result.submissions) == int(terminal)


def test_all_skip_admits_only_one_and_allow_admits_each() -> None:
    for overlap, count in ((OverlapPolicy.SKIP, 1), (OverlapPolicy.ALLOW, 3)):
        plan = plan_admission(
            snapshot(
                RecurringPolicy(misfire=MisfirePolicy.ALL, overlap=overlap)
            ),
            NOW + timedelta(minutes=2),
        )
        result = evaluate_admission(
            OWNER,
            prepared_plan(plan),
            plan.snapshot,
            decision_time=plan.prepared_at,
        )
        assert isinstance(result, TriggerAdmissionWrite)
        assert len(result.submissions) == count
        assert result.mutation.snapshot.state.next_at == NOW + timedelta(
            minutes=3
        )
        assert result.mutation.snapshot.state.generation == 2
        assert result.mutation.event is not None
        assert len(result.mutation.event.occurrence_ids) == 3


def test_range_recovery_matches_spans_and_complete_slot_sequences() -> None:
    plan = plan_admission(snapshot(), NOW + timedelta(minutes=2))
    decisions = committed_decisions(plan)
    recovered = recover_decisions(plan, decisions)
    assert len(recovered) == 2
    first = plan.decisions[0]
    assert isinstance(first, PlannedCoverage)
    # A later replacement can supersede the entire original requested range.
    assert isinstance(decisions[0], TriggerCoverageSpan)
    broad = replace(
        decisions[0],
        until_at=NOW + timedelta(minutes=3),
        decided_at=NOW + timedelta(minutes=3),
    )
    assert len(recover_decisions(plan, (broad,))) == 2
    all_plan = plan_admission(
        snapshot(RecurringPolicy(misfire=MisfirePolicy.ALL)),
        NOW + timedelta(minutes=2),
    )
    rows = committed_decisions(all_plan)
    assert len(recover_decisions(plan, rows)) == 2
    assert len(recover_decisions(plan, rows[1:])) == 1
    with raises(TriggerError):
        recover_decisions(plan, (broad,) * 1001)
    with raises(TriggerError):
        recover_decisions(plan, (replace(broad, revision=2),))


def test_result_retains_recovery_handle_and_rejects_forged_associations() -> (
    None
):
    plan = plan_admission(snapshot(), NOW)
    prepared = prepared_plan(plan)
    result = TriggerAdmissionResult(
        plan=plan, outcome=TriggerCommitOutcome.UNKNOWN, prepared=prepared
    )
    assert TriggerAdmissionCancelledError(result).result == result
    decision = committed_decisions(plan)[0]
    with raises(TriggerError):
        replace(
            result,
            resolved=(
                ResolvedTriggerDecision(
                    request_id="other", decisions=(decision,)
                ),
            ),
        )


def test_recovery_can_join_adjacent_spans_without_covering_a_gap() -> None:
    plan = plan_admission(snapshot(), NOW + timedelta(minutes=3))
    original = committed_decisions(plan)[0]
    assert isinstance(original, TriggerCoverageSpan)
    split = NOW + timedelta(minutes=1)
    left = replace(original, until_at=split)
    right = replace(original, span_id="other", first_at=split)
    assert len(recover_decisions(plan, (left, right))) == 1
    assert recover_decisions(plan, (left,)) == ()
    with raises(TriggerError):
        evaluate_admission(
            replace(OWNER, value="other"),
            prepared_plan(plan),
            plan.snapshot,
            decision_time=plan.prepared_at,
        )
