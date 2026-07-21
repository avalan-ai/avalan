"""Regress interaction-store authority and boundary precedence."""

from collections.abc import Callable
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from itertools import permutations
from typing import Literal, TypeAlias, cast

import pytest

from avalan.interaction import (
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AdvisoryWaitState,
    AdvisoryWaitStatus,
    AgentId,
    AnswerProvenance,
    BranchId,
    CancelInteractionCommand,
    ContinuationId,
    ControllerActivityApplied,
    ControllerId,
    ControllerLeaseExpiredApplied,
    DeadlineScheduleRevision,
    DeclinedResolution,
    DetachInteractionCommand,
    DueInteractionsApplied,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    InputErrorCode,
    InputRequestId,
    InputTransitionApplied,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionCorrelation,
    InteractionDeadline,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionOperation,
    InteractionPolicy,
    InteractionPresentationApplied,
    InteractionPresentationState,
    InteractionRecord,
    InteractionRequestAuthorizationTarget,
    InteractionStoreGeneration,
    InteractionStoreRevision,
    InteractionTerminalMetadata,
    InteractionTime,
    ModelCallId,
    PresentInteractionCommand,
    PrincipalScope,
    QuestionId,
    RecordControllerActivityCommand,
    RequestState,
    RequirementMode,
    ResolutionDecisionStage,
    ResolutionIdempotencyEntry,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    RunId,
    StateRevision,
    StreamSessionId,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionCommand,
    TerminalizeInteractionScopeCommand,
    TextQuestion,
    TimedOutResolution,
    TrustedDefaultResolutionRequest,
    TurnId,
    UserId,
    WaitForInteractionChangeCommand,
    apply_controller_activity,
    apply_due_interactions,
    apply_interaction_detachment,
    apply_interaction_presentation,
    apply_request_cancellation,
    apply_request_terminalization,
    apply_trusted_default_resolution,
    canonical_resolution_digest,
    create_input_request,
    mark_request_pending,
    project_authorized_interaction,
    resolve_request,
    select_next_interaction_deadline,
    semantic_request_fingerprint,
)
from avalan.interaction.store import (
    TrustedDefaultResolutionCommand,
    _apply_scope_cancellation,
    _begin_scope_transaction,
    _insert_interaction_store_backing_record,
    _new_interaction_store_backing,
    _new_trusted_default_resolution_command,
    _snapshot_interaction_store_backing,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
_POLICY = InteractionPolicy()

_BoundaryKind: TypeAlias = Literal["absolute", "advisory", "lease"]
_OperationKind: TypeAlias = Literal["cancel", "terminalize", "trusted_default"]


def _observation(seconds: float) -> InteractionTime:
    return InteractionTime.from_clock(
        wall_time=_NOW + timedelta(seconds=seconds),
        monotonic_seconds=seconds,
    )


def _principal(user_id: str = "owner-user") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(user_id))


def _trusted_default_command(
    *,
    correlation: InteractionCorrelation,
    expected_state_revision: StateRevision,
    actor: InteractionActor | None = None,
) -> TrustedDefaultResolutionCommand:
    """Mint one sealed command as the trusted broker boundary."""
    return _new_trusted_default_resolution_command(
        TrustedDefaultResolutionRequest(
            actor=actor or InteractionActor(principal=_principal()),
            correlation=correlation,
            expected_state_revision=expected_state_revision,
        )
    )


def _origin(
    *,
    principal: PrincipalScope | None = None,
    run_id: str = "run-1",
    branch_id: str = "branch-1",
    parent_branch_id: str | None = None,
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId("turn-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId(branch_id),
        parent_branch_id=(
            None if parent_branch_id is None else BranchId(parent_branch_id)
        ),
        model_call_id=ModelCallId(f"call-{run_id}-{branch_id}"),
        stream_session_id=StreamSessionId(f"stream-{run_id}-{branch_id}"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://authority-regression",
            agent_definition_revision="agent-r1",
            operation_id="operation-1",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capability-r1",
        ),
        principal=principal or _principal(),
    )


def _pending_record(
    request_id: str,
    *,
    principal: PrincipalScope | None = None,
    run_id: str = "run-1",
    branch_id: str = "branch-1",
    parent_branch_id: str | None = None,
    continuation_ttl_seconds: int = 86_400,
    advisory_wait_seconds: int | None = None,
) -> InteractionRecord:
    mode = (
        RequirementMode.REQUIRED
        if advisory_wait_seconds is None
        else RequirementMode.ADVISORY
    )
    created = create_input_request(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"continuation-{request_id}"),
        origin=_origin(
            principal=principal,
            run_id=run_id,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
        ),
        mode=mode,
        reason="Exercise authoritative interaction ordering.",
        questions=(
            TextQuestion(
                question_id=QuestionId("answer"),
                prompt="Provide the configured value.",
                required=True,
                default_value="trusted default",
            ),
        ),
        created_at=_NOW,
        continuation_ttl_seconds=continuation_ttl_seconds,
        advisory_wait_seconds=advisory_wait_seconds,
    )
    transition = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    request = transition.request
    advisory_wait = None
    if advisory_wait_seconds is not None:
        advisory_wait = AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=advisory_wait_seconds,
            remaining_seconds=advisory_wait_seconds,
        )
    return InteractionRecord(
        request=request,
        semantic_fingerprint=semantic_request_fingerprint(request),
        absolute_expires_at=(
            _NOW + timedelta(seconds=continuation_ttl_seconds)
        ),
        presentation=InteractionPresentationState.QUEUED,
        store_revision=InteractionStoreRevision(1),
        advisory_wait=advisory_wait,
    )


def _presented_advisory_record(
    request_id: str,
    *,
    advisory_wait_seconds: int = 60,
) -> InteractionRecord:
    queued = _pending_record(
        request_id,
        advisory_wait_seconds=advisory_wait_seconds,
    )
    command = PresentInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=queued.correlation,
        expected_store_revision=queued.store_revision,
    )
    result = apply_interaction_presentation(
        queued,
        command,
        _observation(5),
        _POLICY,
    )
    assert isinstance(result, InteractionPresentationApplied)
    return result.record


def _leased_record(request_id: str = "leased-request") -> InteractionRecord:
    presented = _presented_advisory_record(request_id)
    evidence = AcquireControllerActivity(
        request_id=presented.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    command = RecordControllerActivityCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=presented.correlation,
        evidence=evidence,
    )
    result = apply_controller_activity(
        presented,
        command,
        _observation(10),
        _POLICY,
        lease_nonce=ActiveControlLeaseNonce("lease-nonce-1"),
    )
    assert isinstance(result, ControllerActivityApplied)
    return result.record


def _terminal_record() -> InteractionRecord:
    previous = _pending_record("terminal-request")
    resolution = DeclinedResolution(
        request_id=previous.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )
    transition = resolve_request(
        previous.request,
        resolution,
        expected_state_revision=previous.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    digest = canonical_resolution_digest(resolution)
    return replace(
        previous,
        request=transition.request,
        semantic_fingerprint=semantic_request_fingerprint(transition.request),
        presentation=InteractionPresentationState.PRESENTED,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
        resolution_digest=digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("terminal-key"),
                resolution_digest=digest,
            ),
        ),
        resolved_by=_principal(),
    )


def _record_snapshot(record: InteractionRecord) -> tuple[object, ...]:
    return tuple(getattr(record, item.name) for item in fields(record))


def _stale_revision(record: InteractionRecord) -> StateRevision:
    assert record.request.state_revision > 0
    return StateRevision(record.request.state_revision - 1)


def _boundary_record(
    boundary_kind: _BoundaryKind,
) -> tuple[InteractionRecord, float]:
    match boundary_kind:
        case "absolute":
            return (
                _pending_record(
                    "absolute-request",
                    continuation_ttl_seconds=60,
                ),
                60,
            )
        case "advisory":
            record = _presented_advisory_record("advisory-request")
            wait = record.advisory_wait
            assert isinstance(wait, AdvisoryWaitState)
            assert wait.running_since_monotonic is not None
            return (
                record,
                wait.running_since_monotonic + wait.remaining_seconds,
            )
        case "lease":
            record = _leased_record()
            wait = record.advisory_wait
            assert isinstance(wait, AdvisoryWaitState)
            assert wait.lease_expires_at_monotonic is not None
            return record, wait.lease_expires_at_monotonic


def _stale_boundary_command(
    operation: _OperationKind,
    record: InteractionRecord,
) -> (
    CancelInteractionCommand
    | TerminalizeInteractionCommand
    | TrustedDefaultResolutionCommand
):
    stale_revision = _stale_revision(record)
    match operation:
        case "cancel":
            return CancelInteractionCommand(
                actor=InteractionActor(principal=_principal()),
                correlation=record.correlation,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=stale_revision,
            )
        case "terminalize":
            return TerminalizeInteractionCommand(
                actor=InteractionActor(principal=_principal()),
                correlation=record.correlation,
                status=ResolutionStatus.UNAVAILABLE,
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=stale_revision,
            )
        case "trusted_default":
            return _trusted_default_command(
                correlation=record.correlation,
                expected_state_revision=stale_revision,
            )


def _apply_boundary_command(
    operation: _OperationKind,
    record: InteractionRecord,
    command: (
        CancelInteractionCommand
        | TerminalizeInteractionCommand
        | TrustedDefaultResolutionCommand
    ),
    observed_at: InteractionTime,
) -> object:
    match operation, command:
        case "cancel", CancelInteractionCommand():
            return apply_request_cancellation(
                record,
                command,
                observed_at,
                _POLICY,
            )
        case "terminalize", TerminalizeInteractionCommand():
            return apply_request_terminalization(
                record,
                command,
                observed_at,
                _POLICY,
            )
        case "trusted_default", TrustedDefaultResolutionCommand():
            return apply_trusted_default_resolution(
                record,
                command,
                observed_at,
                _POLICY,
            )
        case _:
            raise AssertionError("operation and command must correspond")


def test_wait_projection_caps_alternate_to_terminal_metadata() -> None:
    """Cap alternate WAIT disclosure without weakening owner access."""
    record = _terminal_record()
    before = _record_snapshot(record)
    target = InteractionRequestAuthorizationTarget(
        request_id=record.request.request_id,
        origin=record.request.origin,
    )
    owner_wait = WaitForInteractionChangeCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=record.correlation,
        after_store_revision=InteractionStoreRevision(
            record.store_revision - 1
        ),
    )
    alternate_wait = WaitForInteractionChangeCommand(
        actor=InteractionActor(principal=_principal("alternate-user")),
        correlation=record.correlation,
        after_store_revision=InteractionStoreRevision(
            record.store_revision - 1
        ),
    )
    owner_decision = InteractionAuthorizationDecision(
        actor=owner_wait.actor,
        operation=InteractionOperation.WAIT,
        target=target,
        allowed=True,
        disclosure=InteractionDisclosure.FULL,
    )
    alternate_decision = InteractionAuthorizationDecision(
        actor=alternate_wait.actor,
        operation=InteractionOperation.WAIT,
        target=target,
        allowed=True,
        disclosure=InteractionDisclosure.TERMINAL_METADATA,
    )

    assert project_authorized_interaction(record, owner_decision) is record
    projection = project_authorized_interaction(record, alternate_decision)

    assert projection == InteractionTerminalMetadata(
        status=ResolutionStatus.DECLINED,
        resolved_at=_NOW + timedelta(seconds=1),
    )
    assert tuple(item.name for item in fields(projection)) == (
        "status",
        "resolved_at",
    )
    for private_name in (
        "request",
        "request_id",
        "origin",
        "resolution",
        "resolved_by",
        "resolution_digest",
        "idempotency_ledger",
    ):
        assert not hasattr(projection, private_name)
    assert _record_snapshot(record) == before


def test_scope_transaction_rejects_cross_backing_epoch_reuse() -> None:
    """Reject an otherwise identical plan presented to another backing."""
    record = _pending_record("scope-request")
    snapshot = (record,)
    before = tuple(_record_snapshot(item) for item in snapshot)
    generation = InteractionStoreGeneration(4)
    first_backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=generation,
    )
    second_backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=generation,
    )
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=InteractionExecutionScope(run_id=RunId("run-1")),
        provenance=AnswerProvenance.HUMAN,
    )
    transaction = _begin_scope_transaction(first_backing, command)

    with pytest.raises(InputValidationError) as raised:
        _apply_scope_cancellation(
            transaction,
            command,
            _observation(1),
            _POLICY,
            backing=second_backing,
        )

    assert raised.value.code is InputErrorCode.FORBIDDEN
    assert raised.value.path == "backing"
    assert (
        raised.value.safe_message
        == "scope transaction belongs to a different store backing"
    )
    assert tuple(_record_snapshot(item) for item in snapshot) == before


def test_scope_transaction_has_no_partial_snapshot_begin_or_commit_path() -> (
    None
):
    """Reject legacy subset inputs while preserving both in-scope records."""
    first = _pending_record("first-request")
    second = _pending_record("second-request")
    snapshot = (first, second)
    before = tuple(_record_snapshot(item) for item in snapshot)
    generation = InteractionStoreGeneration(8)
    backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=generation,
    )
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=InteractionExecutionScope(run_id=RunId("run-1")),
        provenance=AnswerProvenance.HUMAN,
    )
    legacy_begin = cast(Callable[..., object], _begin_scope_transaction)
    with pytest.raises(TypeError):
        legacy_begin(
            backing,
            (first,),
            (),
            command,
            generation,
        )

    transaction = _begin_scope_transaction(backing, command)
    assert transaction.selected_records == snapshot
    legacy_apply = cast(Callable[..., object], _apply_scope_cancellation)
    with pytest.raises(TypeError):
        legacy_apply(
            transaction,
            command,
            _observation(1),
            _POLICY,
            backing=backing,
            snapshot_records=(first,),
            branch_records=(),
            current_store_generation=generation,
        )

    authoritative = _snapshot_interaction_store_backing(backing)
    changed_second = replace(
        second,
        store_revision=InteractionStoreRevision(second.store_revision + 1),
    )
    for hostile_records in ((first,), (first, changed_second)):
        object.__setattr__(
            backing,
            "_snapshot",
            replace(authoritative, records=hostile_records),
        )
        with pytest.raises(InputValidationError) as raised:
            _apply_scope_cancellation(
                transaction,
                command,
                _observation(1),
                _POLICY,
                backing=backing,
            )
        assert raised.value.code is InputErrorCode.STALE_REVISION
        assert raised.value.path == "snapshot_records"
        object.__setattr__(backing, "_snapshot", authoritative)

    assert authoritative.records == snapshot
    assert (
        tuple(_record_snapshot(item) for item in authoritative.records)
        == before
    )


def test_scope_transaction_rejects_stale_store_generation() -> None:
    """Reject a plan when the backing generation advances before commit."""
    record = _pending_record("generation-request")
    snapshot = (record,)
    before = _record_snapshot(record)
    generation = InteractionStoreGeneration(11)
    backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=generation,
    )
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=InteractionExecutionScope(run_id=RunId("run-1")),
        provenance=AnswerProvenance.HUMAN,
    )
    transaction = _begin_scope_transaction(backing, command)
    inserted = _pending_record(
        "inserted-request",
        run_id="run-2",
        branch_id="branch-2",
    )
    _insert_interaction_store_backing_record(backing, inserted)

    with pytest.raises(InputValidationError) as raised:
        _apply_scope_cancellation(
            transaction,
            command,
            _observation(1),
            _POLICY,
            backing=backing,
        )

    assert raised.value.code is InputErrorCode.STALE_REVISION
    assert raised.value.path == "store_generation"
    assert (
        raised.value.safe_message
        == "scope transaction is stale for the commit store generation"
    )
    authoritative = _snapshot_interaction_store_backing(backing)
    assert _record_snapshot(authoritative.records[0]) == before
    assert authoritative.records[1] == inserted


def test_requestless_scope_has_no_mutation_or_disclosure() -> None:
    """Use branch ownership even when the target branch has no requests."""
    branch_record = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("requestless-branch"),
            parent_branch_id=BranchId("root-branch"),
            principal=_principal(),
        ),
        store_revision=InteractionStoreRevision(3),
    )
    branch_records = (branch_record,)
    before = branch_records
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal("alternate-user")),
        scope=InteractionExecutionScope(
            run_id=RunId("run-1"),
            branch_id=BranchId("requestless-branch"),
        ),
        provenance=AnswerProvenance.HUMAN,
    )

    with pytest.raises(InputValidationError) as raised:
        _begin_scope_transaction(
            _new_interaction_store_backing(
                branch_records=branch_records,
                store_generation=InteractionStoreGeneration(1),
            ),
            command,
        )

    assert raised.value.code is InputErrorCode.FORBIDDEN
    assert raised.value.path == "branch_records"
    assert (
        raised.value.safe_message
        == "scope branch ownership differs from the scope actor"
    )
    assert branch_records == before
    safe_error = str(raised.value)
    for private_value in (
        "requestless-branch",
        "root-branch",
        "owner-user",
        "alternate-user",
    ):
        assert private_value not in safe_error


@pytest.mark.parametrize(
    "operation",
    ("cancel", "terminalize", "trusted_default"),
)
@pytest.mark.parametrize(
    "boundary_kind",
    ("absolute", "advisory", "lease"),
)
@pytest.mark.parametrize(
    "offset",
    (-1, 0, 1),
    ids=("before", "equal", "after"),
)
def test_stale_cas_yields_to_due_boundary_winner(
    operation: _OperationKind,
    boundary_kind: _BoundaryKind,
    offset: int,
) -> None:
    """Reject stale CAS before a boundary and settle it at equality."""
    record, boundary = _boundary_record(boundary_kind)
    before = _record_snapshot(record)
    command = _stale_boundary_command(operation, record)
    observed_at = _observation(boundary + offset)

    if offset < 0:
        with pytest.raises(InputValidationError) as raised:
            _apply_boundary_command(
                operation,
                record,
                command,
                observed_at,
            )
        assert raised.value.code is InputErrorCode.STALE_REVISION
        assert raised.value.path == "command.expected_state_revision"
        assert raised.value.safe_message == "request revision is stale"
        assert _record_snapshot(record) == before
        return

    result = _apply_boundary_command(
        operation,
        record,
        command,
        observed_at,
    )
    if boundary_kind == "lease":
        assert isinstance(result, ControllerLeaseExpiredApplied)
        assert result.command is command
        assert result.previous is record
        assert result.record.request is record.request
        assert result.record.request.state is RequestState.PENDING
        assert result.record.request.resolution is None
        assert result.record.store_revision == record.store_revision + 1
        wait = result.record.advisory_wait
        assert isinstance(wait, AdvisoryWaitState)
        assert wait.status is AdvisoryWaitStatus.RUNNING
        assert wait.running_since_monotonic == boundary
        assert wait.controller_id is None
        assert wait.lease_nonce is None
        assert wait.lease_expires_at_monotonic is None
    else:
        assert isinstance(result, ResolveInteractionApplied)
        assert result.command is command
        assert result.previous is record
        assert result.decision_stage is ResolutionDecisionStage.DEADLINE
        resolution = result.record.request.resolution
        if boundary_kind == "absolute":
            assert isinstance(resolution, ExpiredResolution)
            assert result.record.request.state is RequestState.EXPIRED
        else:
            assert isinstance(resolution, TimedOutResolution)
            assert result.record.request.state is RequestState.TIMED_OUT
        assert resolution.resolved_at == observed_at.wall_time
        assert result.record.request.state_revision == (
            record.request.state_revision + 1
        )
        assert result.record.store_revision == record.store_revision + 1
    assert _record_snapshot(record) == before


@pytest.mark.parametrize(
    "offset",
    (-1, 0, 1),
    ids=("before", "equal", "after"),
)
def test_detachment_respects_lease_expiry_boundary(offset: int) -> None:
    """Detach before expiry but resume the budget at or after equality."""
    record = _leased_record("detachment-request")
    before = _record_snapshot(record)
    wait = record.advisory_wait
    assert isinstance(wait, AdvisoryWaitState)
    boundary = wait.lease_expires_at_monotonic
    assert boundary is not None
    command = DetachInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=record.correlation,
        expected_store_revision=record.store_revision,
    )

    result = apply_interaction_detachment(
        record,
        command,
        _observation(boundary + offset),
        _POLICY,
    )

    if offset < 0:
        assert isinstance(result, InteractionPresentationApplied)
        assert result.command is command
        assert result.previous is record
        assert (
            result.record.presentation is InteractionPresentationState.DETACHED
        )
        assert result.record.advisory_wait == record.advisory_wait
        assert result.record.store_revision == record.store_revision + 1
    else:
        assert isinstance(result, ControllerLeaseExpiredApplied)
        assert result.command is command
        assert result.previous is record
        assert result.record.presentation is record.presentation
        resumed = result.record.advisory_wait
        assert isinstance(resumed, AdvisoryWaitState)
        assert resumed.status is AdvisoryWaitStatus.RUNNING
        assert resumed.running_since_monotonic == boundary
        assert resumed.remaining_seconds == wait.remaining_seconds
        assert resumed.controller_id is None
        assert resumed.lease_nonce is None
        assert resumed.lease_expires_at_monotonic is None
        assert result.record.store_revision == record.store_revision + 1
    assert result.record.request.state is RequestState.PENDING
    assert result.record.request.resolution is None
    assert _record_snapshot(record) == before


def test_due_batch_is_bounded_and_canonical_across_all_permutations() -> None:
    """Order a bounded due batch by deadline before stable request ID."""
    records = (
        _presented_advisory_record(
            "z-earliest",
            advisory_wait_seconds=10,
        ),
        _presented_advisory_record("a-tie", advisory_wait_seconds=20),
        _presented_advisory_record("b-tie", advisory_wait_seconds=20),
        _presented_advisory_record("a-latest", advisory_wait_seconds=30),
    )
    all_orders = tuple(permutations(records))
    assert tuple(reversed(records)) in all_orders
    expected_ids = (
        InputRequestId("z-earliest"),
        InputRequestId("a-tie"),
        InputRequestId("b-tie"),
    )
    observed_at = _observation(40)
    command = TerminalizeDueInteractionsCommand(maximum_results=3)
    before = {
        record.request.request_id: _record_snapshot(record)
        for record in records
    }

    for order in all_orders:
        next_deadline = select_next_interaction_deadline(
            order,
            _observation(6),
            DeadlineScheduleRevision(7),
        )
        assert next_deadline.deadline == InteractionDeadline(
            request_id=InputRequestId("z-earliest"),
            monotonic_deadline=15,
        )
        batch = apply_due_interactions(
            order,
            command,
            observed_at,
            _POLICY,
        )

        assert isinstance(batch, DueInteractionsApplied)
        assert batch.previous == order
        assert len(batch.records) == command.maximum_results
        assert (
            tuple(record.request.request_id for record in batch.records)
            == expected_ids
        )
        for record in batch.records:
            resolution = record.request.resolution
            assert isinstance(resolution, TimedOutResolution)
            assert resolution.resolved_at == observed_at.wall_time
            assert record.request.state is RequestState.TIMED_OUT

    for record in records:
        assert _record_snapshot(record) == before[record.request.request_id]
