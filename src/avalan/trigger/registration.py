"""Calculate bounded registration and control mutations under a store lock."""

from .definition import (
    AtPolicy,
    AtTrigger,
    IntervalTrigger,
    TriggerSpec,
    integer,
    timestamp,
)
from .error import TriggerError, TriggerErrorCode
from .records import (
    MAX_REVISION,
    OccurrenceDisposition,
    OwnerScopeId,
    TriggerCoverageSpan,
    TriggerDefinition,
    TriggerEvent,
    TriggerEventKind,
    TriggerSealedInput,
    TriggerSnapshot,
    TriggerState,
    TriggerStatus,
)
from .schedule import (
    ScheduleProvenance,
    SearchLimits,
    next_occurrence,
    resolve_anchor,
    validate_registration_time,
)

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import uuid4


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerRegistration:
    """Carry host-validated, encrypted revision values into persistence.

    This internal store request is not deployment verification or public
    configuration. The host must validate task capability, privacy and
    durable artifact references before creating it.
    """

    name: str
    task_definition_id: str
    execution_deployment_id: str
    semantic_hash: str
    schedule: TriggerSpec
    input: TriggerSealedInput
    policy: AtPolicy
    schedule_provenance: ScheduleProvenance
    desired_enabled: bool = True

    def __post_init__(self) -> None:
        assert type(self.desired_enabled) is bool


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerMutation:
    """Describe provisional state, revision closure and audit writes."""

    snapshot: TriggerSnapshot
    event: TriggerEvent | None
    definition_created: bool = False
    closed_revision: int | None = None
    superseded: TriggerCoverageSpan | None = None


def check_generation(state: TriggerState, expected_generation: int) -> None:
    """Require an exact bounded control generation."""
    integer(expected_generation, 1, MAX_REVISION, "expected_generation")
    if state.generation != expected_generation:
        raise TriggerError(TriggerErrorCode.CONFLICT, "generation")


def _event(
    state: TriggerState,
    kind: TriggerEventKind,
    now: datetime,
    *,
    span: TriggerCoverageSpan | None = None,
) -> TriggerEvent:
    return TriggerEvent(
        owner_scope_id=state.owner_scope_id,
        trigger_id=state.trigger_id,
        revision=state.revision,
        generation=state.generation,
        event_id=str(uuid4()),
        kind=kind,
        recorded_at=now,
        span_ids=(span.span_id,) if span is not None else (),
        error_code=state.last_error_code,
    )


def _first(
    schedule: TriggerSpec,
    anchor: datetime | None,
    now: datetime,
    *,
    initial: bool,
    limits: SearchLimits,
) -> datetime:
    if initial:
        if isinstance(schedule, AtTrigger):
            return schedule.at
        if isinstance(schedule, IntervalTrigger):
            assert anchor is not None
            return anchor
    first = next_occurrence(schedule, now, anchor=anchor, limits=limits)
    if first is None:
        raise TriggerError(TriggerErrorCode.PAST_SCHEDULE, "schedule")
    return first


def apply_registration(
    owner: OwnerScopeId,
    registration: TriggerRegistration,
    current: TriggerSnapshot | None,
    *,
    expected_generation: int | None,
    decision_time: datetime,
    trigger_id: str,
    limits: SearchLimits = SearchLimits(),
) -> TriggerMutation:
    """Apply a checked registration at the store's authoritative time."""
    now = timestamp(decision_time)
    if current is None:
        if expected_generation is not None:
            raise TriggerError(TriggerErrorCode.CONFLICT, "generation")
        validate_registration_time(registration.schedule, now)
        revision = generation = 1
        anchor = resolve_anchor(registration.schedule, now)
    else:
        if (
            expected_generation is None
            or current.state.owner_scope_id != owner
            or current.definition.name != registration.name
            or current.state.trigger_id != trigger_id
        ):
            raise TriggerError(TriggerErrorCode.CONFLICT, "registration")
        check_generation(current.state, expected_generation)
        if registration.semantic_hash == current.definition.semantic_hash:
            return set_enabled(
                current,
                enabled=registration.desired_enabled,
                expected_generation=expected_generation,
                decision_time=now,
            )
        revision = current.state.revision + 1
        generation = current.state.generation + 1
        if registration.schedule == current.definition.schedule:
            anchor = current.definition.resolved_anchor
        else:
            validate_registration_time(registration.schedule, now)
            anchor = resolve_anchor(registration.schedule, now)
    definition = TriggerDefinition(
        owner_scope_id=owner,
        trigger_id=trigger_id,
        revision=revision,
        name=registration.name,
        task_definition_id=registration.task_definition_id,
        execution_deployment_id=registration.execution_deployment_id,
        semantic_hash=registration.semantic_hash,
        schedule=registration.schedule,
        input=registration.input,
        policy=registration.policy,
        registered_at=now,
        schedule_provenance=registration.schedule_provenance,
        resolved_anchor=anchor,
    )
    span = None
    if (
        current is not None
        and current.state.next_at is not None
        and current.state.next_at <= now
    ):
        # A half-open boundary one microsecond after effective time includes
        # a slot equal to that instant without enumerating old backlog.
        try:
            until = now + timedelta(microseconds=1)
        except OverflowError:
            raise TriggerError(
                TriggerErrorCode.DATETIME_OVERFLOW, "replacement"
            ) from None
        span = TriggerCoverageSpan(
            owner_scope_id=owner,
            trigger_id=trigger_id,
            revision=current.state.revision,
            span_id=str(uuid4()),
            first_at=current.state.next_at,
            until_at=until,
            decided_at=now,
            disposition=OccurrenceDisposition.SUPERSEDED,
        )
    first = _first(
        definition.schedule,
        anchor,
        now,
        initial=current is None,
        limits=limits,
    )
    state = TriggerState(
        owner_scope_id=owner,
        trigger_id=trigger_id,
        revision=revision,
        generation=generation,
        status=(
            TriggerStatus.ACTIVE
            if registration.desired_enabled
            else TriggerStatus.PAUSED
        ),
        next_at=first,
        last_processed_at=now,
    )
    return TriggerMutation(
        snapshot=TriggerSnapshot(definition=definition, state=state),
        event=_event(
            state,
            (
                TriggerEventKind.CREATED
                if current is None
                else TriggerEventKind.REPLACED
            ),
            now,
            span=span,
        ),
        definition_created=True,
        closed_revision=(
            current.state.revision if current is not None else None
        ),
        superseded=span,
    )


def set_enabled(
    current: TriggerSnapshot,
    *,
    enabled: bool,
    expected_generation: int,
    decision_time: datetime,
) -> TriggerMutation:
    """Pause or resume without rearming an exhausted schedule."""
    assert type(enabled) is bool
    check_generation(current.state, expected_generation)
    now = timestamp(decision_time)
    state = current.state
    if (
        state.status is TriggerStatus.EXHAUSTED
        or (enabled and state.status is TriggerStatus.ACTIVE)
        or (not enabled and state.status is TriggerStatus.PAUSED)
    ):
        return TriggerMutation(snapshot=current, event=None)
    # Error resume is invoked only after the host revalidates the revision.
    state = replace(
        state,
        generation=state.generation + 1,
        status=TriggerStatus.ACTIVE if enabled else TriggerStatus.PAUSED,
        retry_after=None,
        failure_count=0,
        last_error_code=None,
        last_processed_at=now,
    )
    return TriggerMutation(
        snapshot=TriggerSnapshot(definition=current.definition, state=state),
        event=_event(
            state,
            TriggerEventKind.RESUMED if enabled else TriggerEventKind.PAUSED,
            now,
        ),
    )


def record_failure(
    current: TriggerSnapshot,
    *,
    expected_generation: int,
    decision_time: datetime,
    error_code: TriggerErrorCode,
    attempts: int = 5,
    base_seconds: int = 1,
    max_seconds: int = 60,
    permanent: bool = False,
) -> TriggerMutation:
    """Persist a retry budget without advancing the recurrence cursor."""
    integer(attempts, 1, 20, "retry.attempts")
    integer(base_seconds, 1, 60, "retry.base_seconds")
    integer(max_seconds, 1, 3600, "retry.max_seconds")
    assert isinstance(error_code, TriggerErrorCode)
    assert type(permanent) is bool
    check_generation(current.state, expected_generation)
    if current.state.status is not TriggerStatus.ACTIVE:
        raise TriggerError(TriggerErrorCode.CONFLICT, "status")
    now = timestamp(decision_time)
    failures = min(attempts, current.state.failure_count + 1)
    exhausted = permanent or failures >= attempts
    try:
        retry = (
            None
            if exhausted
            else now
            + timedelta(
                seconds=min(max_seconds, base_seconds * 2 ** (failures - 1))
            )
        )
    except OverflowError:
        raise TriggerError(
            TriggerErrorCode.DATETIME_OVERFLOW, "retry_after"
        ) from None
    state = replace(
        current.state,
        generation=current.state.generation + 1,
        status=TriggerStatus.ERROR if exhausted else TriggerStatus.ACTIVE,
        failure_count=failures,
        retry_after=retry,
        last_error_code=(
            error_code
            if permanent or not exhausted
            else TriggerErrorCode.ADMISSION_EXHAUSTED
        ),
        last_processed_at=now,
    )
    return TriggerMutation(
        snapshot=TriggerSnapshot(definition=current.definition, state=state),
        event=_event(state, TriggerEventKind.FAILED, now),
    )
