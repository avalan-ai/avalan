"""Validate prepared admission and reduce one authoritative decision batch."""

from ..task.state import TaskRunState, is_terminal_run_state
from ..task.submission import PreparedTaskSubmission
from .coverage import TriggerDecision, validate_decisions
from .definition import OverlapPolicy, RecurringPolicy, timestamp
from .error import TriggerError, TriggerErrorCode
from .plan import (
    PlannedCoverage,
    PlannedOccurrence,
    TriggerAdmissionPlan,
    plan_admission,
)
from .records import (
    OCCURRENCE_NAMESPACE,
    OccurrenceDisposition,
    OwnerScopeId,
    TriggerCoverageSpan,
    TriggerEvent,
    TriggerEventKind,
    TriggerOccurrence,
    TriggerSnapshot,
    TriggerStatus,
)
from .registration import TriggerMutation
from .search import ScheduleSearch

from asyncio import CancelledError
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum
from json import dumps
from uuid import uuid5


class TriggerCommitOutcome(StrEnum):
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedTriggerAdmission:
    """Retain the actual validated task preparation for every candidate."""

    plan: TriggerAdmissionPlan
    submissions: tuple[PreparedTaskSubmission, ...] = field(repr=False)

    def __post_init__(self) -> None:
        assert isinstance(self.plan, TriggerAdmissionPlan)
        assert isinstance(self.submissions, tuple)
        for submission in self.submissions:
            assert isinstance(submission, PreparedTaskSubmission)
        expected = self.plan.admission_ids
        if tuple(item.occurrence_id for item in self.submissions) != expected:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "prepared.occurrences"
            )
        definition = self.plan.snapshot.definition
        slots = {
            self.plan.identity(item): item.scheduled_at
            for item in self.plan.decisions
            if isinstance(item, PlannedOccurrence)
        }
        for prepared in self.submissions:
            assert prepared.occurrence_id is not None
            if (
                prepared.owner_scope != definition.owner_scope_id.value
                or prepared.execution.definition_id
                != definition.task_definition_id
                or prepared.execution_deployment_id
                != definition.execution_deployment_id
                or prepared.available_at != slots[prepared.occurrence_id]
            ):
                raise TriggerError(
                    TriggerErrorCode.INVALID_CONFIG, "prepared.identity"
                )
        for values in (
            tuple(item.run_id for item in self.submissions),
            tuple(item.submission_id for item in self.submissions),
        ):
            if len(values) != len(set(values)):
                raise TriggerError(
                    TriggerErrorCode.INVALID_CONFIG, "prepared.identities"
                )

    def submission(self, identity: str) -> PreparedTaskSubmission:
        for prepared in self.submissions:
            if prepared.occurrence_id == identity:
                return prepared
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "prepared.occurrence"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedTriggerDecision:
    """Associate a requested identity with immutable committed decisions."""

    request_id: str
    decisions: tuple[TriggerDecision, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.request_id, str) and self.request_id
        assert isinstance(self.decisions, tuple) and self.decisions
        for decision in self.decisions:
            assert isinstance(
                decision, TriggerOccurrence | TriggerCoverageSpan
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerAdmissionResult:
    """Separate this transaction's outcome from recovered associations.

    A failed new attempt can retain decisions committed by an earlier
    attempt. Unknown outcomes never authorize resource reclamation. Keep
    the plan and prepared handles until every requested identity settles.
    """

    plan: TriggerAdmissionPlan
    outcome: TriggerCommitOutcome
    resolved: tuple[ResolvedTriggerDecision, ...] = ()
    snapshot: TriggerSnapshot | None = None
    error_code: TriggerErrorCode | None = None
    replan_required: bool = False
    contended: bool = False
    prepared: PreparedTriggerAdmission | None = field(default=None, repr=False)
    cleanup_pending: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.plan, TriggerAdmissionPlan)
        assert isinstance(self.outcome, TriggerCommitOutcome)
        assert isinstance(self.resolved, tuple)
        identities = tuple(item.request_id for item in self.resolved)
        if len(identities) != len(set(identities)) or not set(
            identities
        ) <= set(self.plan.request_ids):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "result.identities"
            )
        if self.prepared is not None:
            assert self.prepared.plan == self.plan

    @property
    def unresolved_ids(self) -> tuple[str, ...]:
        resolved = {item.request_id for item in self.resolved}
        return tuple(
            item for item in self.plan.request_ids if item not in resolved
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerAdmissionWrite:
    """Carry provisional record writes without claiming a commit."""

    mutation: TriggerMutation
    decisions: tuple[TriggerDecision, ...]
    submissions: tuple[PreparedTaskSubmission, ...] = field(repr=False)


def recover_decisions(
    plan: TriggerAdmissionPlan,
    existing: tuple[TriggerDecision, ...],
) -> tuple[ResolvedTriggerDecision, ...]:
    """Resolve complete requested ranges before any current-state gate."""
    if len(existing) > 1000:
        raise TriggerError(
            TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED, "recovery.history"
        )
    state = plan.snapshot.state
    for item in existing:
        if (item.owner_scope_id, item.trigger_id, item.revision) != (
            state.owner_scope_id,
            state.trigger_id,
            state.revision,
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "recovery.identity"
            )
    ordered = sorted(
        existing,
        key=lambda item: (
            item.scheduled_at
            if isinstance(item, TriggerOccurrence)
            else item.first_at
        ),
    )
    resolved = []
    search = ScheduleSearch(plan.limits.search)
    for request in plan.decisions:
        first = (
            request.scheduled_at
            if isinstance(request, PlannedOccurrence)
            else request.first_at
        )
        through = (
            first
            if isinstance(request, PlannedOccurrence)
            else request.until_at - timedelta(microseconds=1)
        )
        cursor: datetime | None = first
        matching: list[TriggerDecision] = []
        for item in ordered:
            if cursor is None or cursor > through:
                break
            if isinstance(item, TriggerOccurrence):
                if item.scheduled_at != cursor:
                    continue
                matching.append(item)
                if isinstance(request, PlannedOccurrence):
                    cursor = None
                else:
                    cursor = search.next(
                        plan.snapshot.definition.schedule,
                        cursor,
                        anchor=plan.snapshot.definition.resolved_anchor,
                    )
            elif item.first_at <= cursor < item.until_at:
                matching.append(item)
                if item.until_at > through:
                    cursor = None
                else:
                    cursor = search.next(
                        plan.snapshot.definition.schedule,
                        item.until_at - timedelta(microseconds=1),
                        anchor=plan.snapshot.definition.resolved_anchor,
                    )
        if matching and (cursor is None or cursor > through):
            resolved.append(
                ResolvedTriggerDecision(
                    request_id=plan.identity(request),
                    decisions=tuple(matching),
                )
            )
    return tuple(resolved)


def evaluate_admission(
    owner: OwnerScopeId,
    prepared: PreparedTriggerAdmission,
    current: TriggerSnapshot | None,
    *,
    decision_time: datetime,
    existing: tuple[TriggerDecision, ...] = (),
    outstanding_states: tuple[TaskRunState, ...] = (),
) -> TriggerAdmissionWrite | TriggerAdmissionResult:
    """Recheck time, prefix and overlap against the locked canonical state."""
    plan = prepared.plan
    if plan.snapshot.state.owner_scope_id != owner:
        raise TriggerError(
            TriggerErrorCode.STORE_INCOMPATIBLE, "owner_scope_id"
        )
    resolved = recover_decisions(plan, existing)
    if plan.decisions and len(resolved) == len(plan.decisions):
        return TriggerAdmissionResult(
            plan=plan,
            outcome=TriggerCommitOutcome.COMMITTED,
            resolved=resolved,
            snapshot=current,
            prepared=prepared,
        )
    now = timestamp(decision_time)
    if (
        current is None
        or current != plan.snapshot
        or current.state.status is not TriggerStatus.ACTIVE
        or (
            current.state.retry_after is not None
            and current.state.retry_after > now
        )
    ):
        return TriggerAdmissionResult(
            plan=plan,
            outcome=TriggerCommitOutcome.NOT_COMMITTED,
            resolved=resolved,
            snapshot=current,
            error_code=TriggerErrorCode.CONFLICT,
            prepared=prepared,
        )
    expected = plan_admission(current, now, limits=plan.limits)
    if (
        expected.decisions != plan.decisions
        or expected.next_at != plan.next_at
    ):
        return TriggerAdmissionResult(
            plan=plan,
            outcome=TriggerCommitOutcome.NOT_COMMITTED,
            resolved=resolved,
            snapshot=current,
            replan_required=True,
            error_code=TriggerErrorCode.CONFLICT,
            prepared=prepared,
        )
    if not plan.decisions:
        return TriggerAdmissionResult(
            plan=plan,
            outcome=TriggerCommitOutcome.NOT_COMMITTED,
            snapshot=current,
            prepared=prepared,
        )
    blocking = any(
        not is_terminal_run_state(state) for state in outstanding_states
    )
    decisions: list[TriggerDecision] = []
    submissions: list[PreparedTaskSubmission] = []
    definition = current.definition
    for request in plan.decisions:
        identity = plan.identity(request)
        if isinstance(request, PlannedCoverage):
            decisions.append(
                TriggerCoverageSpan(
                    owner_scope_id=owner,
                    trigger_id=definition.trigger_id,
                    revision=definition.revision,
                    span_id=identity,
                    first_at=request.first_at,
                    until_at=request.until_at,
                    decided_at=now,
                    disposition=request.disposition,
                )
            )
            continue
        disposition = request.disposition
        submission = None
        if disposition is OccurrenceDisposition.ADMITTED:
            if (
                blocking
                and isinstance(definition.policy, RecurringPolicy)
                and definition.policy.overlap is OverlapPolicy.SKIP
            ):
                disposition = OccurrenceDisposition.SKIPPED_OVERLAP
            else:
                submission = prepared.submission(identity)
                submissions.append(submission)
                blocking = True
        decisions.append(
            TriggerOccurrence(
                owner_scope_id=owner,
                trigger_id=definition.trigger_id,
                revision=definition.revision,
                occurrence_id=identity,
                scheduled_at=request.scheduled_at,
                decided_at=now,
                disposition=disposition,
                run_id=submission.run_id if submission is not None else None,
            )
        )
    validate_decisions(
        definition,
        tuple(decisions),
        existing=existing,
        limits=plan.limits.search,
    )
    state = replace(
        current.state,
        generation=current.state.generation + 1,
        status=(
            TriggerStatus.EXHAUSTED
            if plan.next_at is None
            else TriggerStatus.ACTIVE
        ),
        next_at=plan.next_at,
        last_processed_at=now,
        retry_after=None,
        failure_count=0,
        last_error_code=None,
    )
    event_id = str(
        uuid5(
            OCCURRENCE_NAMESPACE,
            dumps(
                [
                    "event",
                    owner.value,
                    state.trigger_id,
                    state.generation,
                    plan.request_ids,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    )
    event = TriggerEvent(
        owner_scope_id=owner,
        trigger_id=state.trigger_id,
        revision=state.revision,
        generation=state.generation,
        event_id=event_id,
        kind=TriggerEventKind.DECIDED,
        recorded_at=now,
        occurrence_ids=tuple(
            item.occurrence_id
            for item in decisions
            if isinstance(item, TriggerOccurrence)
        ),
        span_ids=tuple(
            item.span_id
            for item in decisions
            if isinstance(item, TriggerCoverageSpan)
        ),
    )
    return TriggerAdmissionWrite(
        mutation=TriggerMutation(
            snapshot=TriggerSnapshot(definition=definition, state=state),
            event=event,
        ),
        decisions=tuple(decisions),
        submissions=tuple(submissions),
    )


class TriggerAdmissionCancelledError(CancelledError):
    """Preserve cancellation and its stable recovery handle."""

    def __init__(self, result: TriggerAdmissionResult) -> None:
        self.result = result
        super().__init__("Trigger admission cancelled; recovery is available.")
