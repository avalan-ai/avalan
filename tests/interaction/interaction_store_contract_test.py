"""Exercise the authoritative async interaction store contract."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta
from inspect import signature
from typing import cast, get_type_hints

import pytest

import avalan.interaction as interaction_api
from avalan.interaction import (
    RESOLUTION_DECISION_PRECEDENCE,
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AdvisoryWaitState,
    AdvisoryWaitStatus,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelInteractionApplied,
    CancelInteractionCommand,
    CancelInteractionRejected,
    CancellationScope,
    CancelledResolution,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ControllerActivityAction,
    ControllerActivityApplied,
    ControllerActivityRejected,
    ControllerId,
    ControllerLeaseExpiredApplied,
    CreateInteractionApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    DeadlineScheduleRevision,
    DeclinedResolution,
    DetachInteractionCommand,
    DisconnectControllerActivity,
    DueInteractionsApplied,
    DueInteractionsRejected,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    InputCandidateResolution,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InputResolution,
    InputResumerRegistration,
    InputResumptionNotification,
    InputTransitionApplied,
    InputTransitionError,
    InputTransitionRejected,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionBranchAuthorizationTarget,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBranchRegistrationRejected,
    InteractionBranchRegistrationReplayed,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionCorrelation,
    InteractionDeadline,
    InteractionDeadlineSnapshot,
    InteractionDisclosure,
    InteractionDisclosureProjection,
    InteractionExecutionScope,
    InteractionNotFoundError,
    InteractionOperation,
    InteractionPolicy,
    InteractionPresentationApplied,
    InteractionPresentationRejected,
    InteractionPresentationState,
    InteractionRecord,
    InteractionReplayKind,
    InteractionRequestAuthorizationTarget,
    InteractionScopeAuthorizationTarget,
    InteractionStore,
    InteractionStoreClosedError,
    InteractionStoreFactory,
    InteractionStoreGeneration,
    InteractionStoreReplayed,
    InteractionStoreRevision,
    InteractionTerminalMetadata,
    InteractionTime,
    ListInteractionsCommand,
    ModelCallId,
    PresentInteractionCommand,
    PrincipalScope,
    PulseControllerActivity,
    QuestionId,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    ReleaseControllerActivity,
    RequestState,
    RequirementMode,
    ResolutionDecisionStage,
    ResolutionIdempotencyDisposition,
    ResolutionIdempotencyEntry,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopeCancellationReplayed,
    ScopeSupersessionApplied,
    ScopeSupersessionRejected,
    ScopeSupersessionReplayed,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionApplied,
    TerminalizeInteractionCommand,
    TerminalizeInteractionRejected,
    TerminalizeInteractionScopeCommand,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TrustedDefaultResolutionApplied,
    TrustedDefaultResolutionRequest,
    TurnId,
    UnavailableResolution,
    UserId,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    apply_candidate_resolution,
    apply_controller_activity,
    apply_create_interaction,
    apply_due_interaction,
    apply_due_interactions,
    apply_interaction_detachment,
    apply_interaction_presentation,
    apply_request_cancellation,
    apply_request_terminalization,
    apply_semantic_resolution_replay,
    apply_trusted_default_resolution,
    canonical_resolution_digest,
    create_input_request,
    evaluate_resolution_idempotency,
    mark_request_pending,
    project_authorized_interaction,
    resolve_request,
    select_next_interaction_deadline,
    semantic_request_fingerprint,
    validate_interaction_admission,
)
from avalan.interaction import store as interaction_store
from avalan.interaction.store import (
    _ADMISSION_CLEANUP_RESOLVER,
    _DEADLINE_RESOLVER,
    TrustedDefaultResolutionCommand,
    _apply_scope_cancellation,
    _apply_scope_supersession,
    _begin_scope_transaction,
    _bind_task_input_classifications,
    _BoundTaskInputClassifications,
    _InteractionAdmissionCleanupCommand,
    _InteractionAdmissionCleanupDisposition,
    _InteractionAdmissionCleanupResult,
    _InteractionAdmissionCreateCommand,
    _InteractionScopeTransaction,
    _new_interaction_admission_cleanup_result,
    _new_interaction_admission_commands,
    _new_interaction_store_backing,
    _new_task_input_classifier_binding,
    _new_trusted_default_resolution_command,
    _present_interaction_record,
    _resolve_interaction_branch_root,
    _task_input_classification_requests,
    _TaskInputClassifierBinding,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
_POLICY = InteractionPolicy()


def _classifier_binding(
    policy: InteractionPolicy = _POLICY,
) -> _TaskInputClassifierBinding:
    return _new_task_input_classifier_binding(
        classifier_id=policy.task_input_classifier_id,
        policy_revision=policy.task_input_policy_revision,
    )


def _bound_classifications(
    previous: InteractionRecord,
    command: ResolveInteractionCommand,
    *,
    decisions: tuple[TaskInputClassificationDecision, ...] | None = None,
    binding: _TaskInputClassifierBinding | None = None,
    policy: InteractionPolicy = _POLICY,
) -> tuple[_TaskInputClassifierBinding, _BoundTaskInputClassifications]:
    active_binding = binding or _classifier_binding(policy)
    requests = _task_input_classification_requests(
        previous,
        command,
        policy,
    )
    active_decisions = decisions or tuple(
        TaskInputClassificationDecision.ALLOW for _ in requests
    )
    assert len(active_decisions) == len(requests)
    outputs = tuple(
        TaskInputClassification(
            decision=decision,
            classifier_id=active_binding.classifier_id,
            classification_id=f"classification-{index}",
            policy_revision=active_binding.policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )
        for index, (request, decision) in enumerate(
            zip(requests, active_decisions, strict=True),
            start=1,
        )
    )
    proof = _bind_task_input_classifications(
        active_binding,
        previous,
        command,
        outputs,
        policy,
    )
    return active_binding, proof


class _RecordingResumer:
    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        del notification


def _observation(
    seconds: float,
    *,
    wall_seconds: float | None = None,
) -> InteractionTime:
    return InteractionTime.from_clock(
        wall_time=_NOW
        + timedelta(seconds=seconds if wall_seconds is None else wall_seconds),
        monotonic_seconds=seconds,
    )


def _principal() -> PrincipalScope:
    return PrincipalScope(user_id=UserId("user-1"))


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


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        model_call_id=ModelCallId("call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://support",
            agent_definition_revision="agent-r1",
            operation_id="operation-1",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capability-r1",
        ),
        principal=_principal(),
    )


def _created(
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=mode,
        reason="A response is required.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=_NOW,
        advisory_wait_seconds=(
            60 if mode is RequirementMode.ADVISORY else None
        ),
    )


def _pending() -> InputRequest:
    created = _created()
    transition = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    return transition.request


def _terminal_record() -> InteractionRecord:
    pending = _pending()
    resolution = DeclinedResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )
    transition = resolve_request(
        pending,
        resolution,
        expected_state_revision=pending.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    digest = canonical_resolution_digest(resolution)
    return InteractionRecord(
        request=transition.request,
        semantic_fingerprint=semantic_request_fingerprint(transition.request),
        absolute_expires_at=_NOW + timedelta(days=1),
        presentation=InteractionPresentationState.PRESENTED,
        store_revision=InteractionStoreRevision(2),
        resolution_digest=digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("key-1"),
                resolution_digest=digest,
            ),
        ),
        resolved_by=_principal(),
    )


def _pending_record() -> InteractionRecord:
    pending = _pending()
    return InteractionRecord(
        request=pending,
        semantic_fingerprint=semantic_request_fingerprint(pending),
        absolute_expires_at=_NOW + timedelta(days=1),
        presentation=InteractionPresentationState.PRESENTED,
        store_revision=InteractionStoreRevision(1),
    )


def _pending_record_on_branch(
    request_id: str,
    branch_id: str,
    *,
    parent_branch_id: str | None = None,
) -> InteractionRecord:
    previous = _pending_record()
    origin = replace(
        previous.request.origin,
        branch_id=BranchId(branch_id),
        parent_branch_id=(
            None if parent_branch_id is None else BranchId(parent_branch_id)
        ),
    )
    request = replace(
        previous.request,
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"{request_id}-continuation"),
        origin=origin,
    )
    return replace(
        previous,
        request=request,
        semantic_fingerprint=semantic_request_fingerprint(request),
    )


def _with_request_identity(
    record: InteractionRecord,
    request_id: str,
) -> InteractionRecord:
    request = replace(
        record.request,
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"{request_id}-continuation"),
    )
    return replace(
        record,
        request=request,
        semantic_fingerprint=semantic_request_fingerprint(request),
    )


def _queued_advisory_record() -> InteractionRecord:
    created = _created(RequirementMode.ADVISORY)
    transition = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    pending = transition.request
    return InteractionRecord(
        request=pending,
        semantic_fingerprint=semantic_request_fingerprint(pending),
        absolute_expires_at=_NOW + timedelta(days=1),
        presentation=InteractionPresentationState.QUEUED,
        store_revision=InteractionStoreRevision(1),
        advisory_wait=AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=60,
            remaining_seconds=60,
        ),
    )


def _presented_advisory_record() -> InteractionRecord:
    queued = _queued_advisory_record()
    result = _present_interaction_record(
        queued,
        _presentation_command(queued),
        _observation(5),
        _POLICY,
    )
    assert isinstance(result, InteractionPresentationApplied)
    return result.record


def _presentation_command(
    record: InteractionRecord,
) -> PresentInteractionCommand:
    return PresentInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=record.correlation,
        expected_store_revision=record.store_revision,
    )


def _detachment_command(
    record: InteractionRecord,
) -> DetachInteractionCommand:
    return DetachInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=record.correlation,
        expected_store_revision=record.store_revision,
    )


def _controller_command(
    record: InteractionRecord,
    evidence: (
        AcquireControllerActivity
        | PulseControllerActivity
        | ReleaseControllerActivity
        | DisconnectControllerActivity
    ),
) -> RecordControllerActivityCommand:
    return RecordControllerActivityCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=record.correlation,
        evidence=evidence,
    )


def _resolve_command(
    record: InteractionRecord,
    key: str,
    *,
    actor: InteractionActor | None = None,
    resolution: InputCandidateResolution | None = None,
) -> ResolveInteractionCommand:
    proposed = resolution
    if proposed is None:
        stored = record.request.resolution
        if isinstance(stored, DeclinedResolution):
            proposed = stored
        else:
            proposed = DeclinedResolution(
                request_id=record.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
            )
    return ResolveInteractionCommand(
        actor=actor or InteractionActor(principal=_principal()),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=proposed,
    )


def _deadline_record(
    previous: InteractionRecord,
    resolution: TimedOutResolution | ExpiredResolution,
) -> InteractionRecord:
    transition = resolve_request(
        previous.request,
        resolution,
        expected_state_revision=previous.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    advisory_wait = previous.advisory_wait
    if isinstance(resolution, TimedOutResolution):
        assert advisory_wait is not None
        advisory_wait = replace(
            advisory_wait,
            status=AdvisoryWaitStatus.EXHAUSTED,
            remaining_seconds=0,
            running_since_monotonic=None,
        )
    return InteractionRecord(
        request=transition.request,
        semantic_fingerprint=previous.semantic_fingerprint,
        absolute_expires_at=previous.absolute_expires_at,
        presentation=previous.presentation,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
        advisory_wait=advisory_wait,
        resolution_digest=canonical_resolution_digest(resolution),
        resolved_by=_DEADLINE_RESOLVER,
    )


def _acquired_controller_transition() -> tuple[
    InteractionRecord,
    InteractionRecord,
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
]:
    previous = _presented_advisory_record()
    evidence = AcquireControllerActivity(
        request_id=previous.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    nonce = ActiveControlLeaseNonce("server-nonce")
    acquired = apply_controller_activity(
        previous,
        _controller_command(previous, evidence),
        _observation(10),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(acquired, ControllerActivityApplied)
    return previous, acquired.record, evidence, nonce


def test_advisory_wait_starts_on_actual_presentation() -> None:
    """Keep queued requests pending until actual attached presentation."""
    queued = _queued_advisory_record()
    presented_at = _NOW + timedelta(seconds=5)
    applied = _present_interaction_record(
        queued,
        _presentation_command(queued),
        _observation(5),
        _POLICY,
    )
    assert isinstance(applied, InteractionPresentationApplied)
    presented = applied.record

    assert queued.request.advisory_deadline is None
    assert presented.request.advisory_deadline == presented_at + timedelta(
        seconds=60
    )
    assert presented.request.state_revision == queued.request.state_revision
    assert presented.store_revision == queued.store_revision + 1
    assert presented.advisory_wait is not None
    assert presented.advisory_wait.status is AdvisoryWaitStatus.RUNNING
    assert presented.advisory_wait.presented_at == presented_at
    assert presented.advisory_wait.running_since_monotonic == 5
    assert applied.presentation is InteractionPresentationState.PRESENTED
    with pytest.raises(InputValidationError):
        InteractionPresentationApplied(
            command=_presentation_command(queued),
            record=presented,
            previous=queued,
            observed_at=_observation(6),
            policy=_POLICY,
        )


def test_mutation_commands_exclude_caller_time_and_system_authority() -> None:
    """Keep trusted observations and deadline authority inside the store."""
    pending = _pending()
    proposed = DeclinedResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )
    command = ResolveInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=InteractionCorrelation.from_request(pending),
        expected_state_revision=pending.state_revision,
        idempotency_key=ResolutionIdempotencyKey("key-2"),
        proposed_resolution=proposed,
    )
    later = replace(
        proposed,
        resolved_at=proposed.resolved_at + timedelta(hours=1),
    )
    later_command = replace(command, proposed_resolution=later)

    assert command.resolution_digest == later_command.resolution_digest
    assert not hasattr(command, "trusted_resolution")
    for command_type in (
        CreateInteractionCommand,
        PresentInteractionCommand,
        ResolveInteractionCommand,
        TerminalizeInteractionCommand,
        TerminalizeInteractionScopeCommand,
        RecordControllerActivityCommand,
        TerminalizeDueInteractionsCommand,
    ):
        parameters = signature(command_type).parameters
        assert "observed_at" not in parameters
        assert "resolver" not in parameters


@pytest.mark.parametrize(
    "resolution",
    (
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
            scope=CancellationScope.REQUEST,
        ),
        TimedOutResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=60),
        ),
        UnavailableResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
        ExpiredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(days=1),
        ),
        SupersededResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
    ),
)
def test_external_resolve_command_rejects_non_candidate_outcomes(
    resolution: InputResolution,
) -> None:
    """Route system, cancellation, and terminalization outcomes elsewhere."""
    pending = _pending()

    with pytest.raises(InputValidationError) as error:
        ResolveInteractionCommand(
            actor=InteractionActor(principal=_principal()),
            correlation=InteractionCorrelation.from_request(pending),
            expected_state_revision=pending.state_revision,
            idempotency_key=ResolutionIdempotencyKey("key-1"),
            proposed_resolution=cast(InputCandidateResolution, resolution),
        )

    assert error.value.path == "resolve.proposed_resolution"


@pytest.mark.parametrize(
    "provenance",
    (AnswerProvenance.TRUSTED_DEFAULT, AnswerProvenance.POLICY),
)
def test_external_resolution_cannot_claim_trusted_provenance(
    provenance: AnswerProvenance,
) -> None:
    """Reserve trusted default and deadline provenance for sealed paths."""
    pending = _pending_record()
    with pytest.raises(InputValidationError) as error:
        ResolveInteractionCommand(
            actor=InteractionActor(principal=_principal()),
            correlation=pending.correlation,
            expected_state_revision=pending.request.state_revision,
            idempotency_key=ResolutionIdempotencyKey("forged-provenance"),
            proposed_resolution=DeclinedResolution(
                request_id=pending.request.request_id,
                provenance=provenance,
                resolved_at=_NOW,
            ),
        )
    assert error.value.code is InputErrorCode.FORBIDDEN


def test_trusted_default_resolution_derives_stored_default() -> None:
    """Derive trusted-default content without caller-supplied resolution."""
    created = replace(
        _created(),
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=False,
            ),
        ),
    )
    transition = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    previous = InteractionRecord(
        request=transition.request,
        semantic_fingerprint=semantic_request_fingerprint(transition.request),
        absolute_expires_at=_NOW + timedelta(days=1),
        presentation=InteractionPresentationState.PRESENTED,
        store_revision=InteractionStoreRevision(1),
    )
    command = _trusted_default_command(
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
    )

    result = apply_trusted_default_resolution(
        previous,
        command,
        _observation(1),
        _POLICY,
    )

    assert isinstance(result, TrustedDefaultResolutionApplied)
    assert result.record.request.state is RequestState.ANSWERED
    resolution = result.record.request.resolution
    assert isinstance(resolution, AnsweredResolution)
    assert resolution.provenance is AnswerProvenance.TRUSTED_DEFAULT
    assert len(resolution.answers) == 1
    answer = resolution.answers[0]
    assert isinstance(answer, ConfirmationAnswer)
    assert answer.value is False
    assert answer.provenance is AnswerProvenance.TRUSTED_DEFAULT
    assert result.record.idempotency_ledger == ()
    assert result.record.resolved_by is not None
    assert not isinstance(result.record.resolved_by, PrincipalScope)
    assert not hasattr(interaction_api, "TrustedDefaultResolutionCommand")
    assert not hasattr(
        interaction_api, "_new_trusted_default_resolution_command"
    )
    assert (
        "proposed_resolution"
        not in signature(TrustedDefaultResolutionRequest).parameters
    )


def test_classification_proof_is_bound_to_exact_candidate_commit() -> None:
    """Require sealed allow proof for every free-form candidate value."""
    created = replace(
        _created(),
        questions=(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Provide text.",
                required=True,
            ),
        ),
    )
    transition = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    pending = transition.request
    previous = InteractionRecord(
        request=pending,
        semantic_fingerprint=semantic_request_fingerprint(pending),
        absolute_expires_at=_NOW + timedelta(days=1),
        presentation=InteractionPresentationState.PRESENTED,
        store_revision=InteractionStoreRevision(1),
    )
    answer = TextAnswer(
        question_id=QuestionId("text"),
        provenance=AnswerProvenance.HUMAN,
        value="classified value",
    )
    proposed = AnsweredResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(answer,),
    )
    unclassified = ResolveInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("classified-key"),
        proposed_resolution=proposed,
    )
    with pytest.raises(InputValidationError) as missing:
        apply_candidate_resolution(
            previous,
            unclassified,
            _observation(1),
            _POLICY,
        )
    assert (
        missing.value.code is InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE
    )
    with pytest.raises(InputValidationError) as stale:
        apply_candidate_resolution(
            previous,
            replace(
                unclassified,
                expected_state_revision=StateRevision(0),
            ),
            _observation(1),
            _POLICY,
        )
    assert stale.value.code is InputErrorCode.STALE_REVISION
    assert stale.value.path == "expected_state_revision"

    binding, proof = _bound_classifications(previous, unclassified)
    committed = apply_candidate_resolution(
        previous,
        unclassified,
        _observation(1),
        _POLICY,
        classifier_binding=binding,
        classification_proof=proof,
    )
    assert isinstance(committed, ResolveInteractionApplied)
    assert committed.record.request.state is RequestState.ANSWERED

    with pytest.raises(InputValidationError) as prohibited:
        _bound_classifications(
            previous,
            unclassified,
            decisions=(TaskInputClassificationDecision.REJECT_SECRET,),
        )
    assert prohibited.value.code is InputErrorCode.PROHIBITED_INPUT
    changed = replace(
        proposed,
        answers=(replace(answer, value="different candidate"),),
    )
    changed_command = replace(unclassified, proposed_resolution=changed)
    with pytest.raises(InputValidationError) as stale_proof:
        apply_candidate_resolution(
            previous,
            changed_command,
            _observation(1),
            _POLICY,
            classifier_binding=binding,
            classification_proof=proof,
        )
    assert stale_proof.value.code is InputErrorCode.CORRELATION_MISMATCH


def test_resolution_replay_precedes_mutation_revision_cas() -> None:
    """Expose same-key and semantic replay before stale mutation-only CAS."""
    previous = _terminal_record()
    pending = _pending_record()
    resolution = previous.request.resolution
    assert isinstance(resolution, DeclinedResolution)
    command = ResolveInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("key-1"),
        proposed_resolution=resolution,
    )
    applied = ResolveInteractionApplied(
        record=previous,
        previous=pending,
        decision_stage=ResolutionDecisionStage.COMMIT,
        observed_at=_observation(1),
        policy=_POLICY,
        command=command,
        idempotency_key=ResolutionIdempotencyKey("key-1"),
    )
    same_key = InteractionStoreReplayed(
        command=command,
        record=previous,
        replay_kind=InteractionReplayKind.SAME_KEY,
    )
    digest = previous.resolution_digest
    assert digest is not None
    current = replace(
        previous,
        store_revision=InteractionStoreRevision(3),
        idempotency_ledger=previous.idempotency_ledger
        + (
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("key-2"),
                resolution_digest=digest,
            ),
        ),
    )
    semantic = InteractionStoreReplayed(
        command=replace(
            command,
            idempotency_key=ResolutionIdempotencyKey("key-2"),
        ),
        record=current,
        previous=previous,
        replay_kind=InteractionReplayKind.SEMANTIC_NEW_KEY,
    )
    conflict = ResolveInteractionRejected(
        command=command,
        error=InputTransitionError(
            code=InputErrorCode.IDEMPOTENCY_CONFLICT,
            path="resolve.idempotency_key",
            message="transport key is bound to different semantic content",
        ),
        decision_stage=ResolutionDecisionStage.IDEMPOTENCY_KEY,
    )

    assert RESOLUTION_DECISION_PRECEDENCE == (
        ResolutionDecisionStage.AUTHORIZATION,
        ResolutionDecisionStage.CORRELATION,
        ResolutionDecisionStage.DEADLINE,
        ResolutionDecisionStage.IDEMPOTENCY_KEY,
        ResolutionDecisionStage.SEMANTIC_REPLAY,
        ResolutionDecisionStage.STATE_REVISION,
        ResolutionDecisionStage.VALIDATION,
        ResolutionDecisionStage.COMMIT,
    )
    assert applied.decision_stage is ResolutionDecisionStage.COMMIT
    assert same_key.decision_stage is ResolutionDecisionStage.IDEMPOTENCY_KEY
    assert not same_key.store_mutation_applied
    assert semantic.decision_stage is ResolutionDecisionStage.SEMANTIC_REPLAY
    assert semantic.store_mutation_applied
    assert (
        semantic.record.request.state_revision
        == previous.request.state_revision
    )
    assert conflict.decision_stage is ResolutionDecisionStage.IDEMPOTENCY_KEY


def test_replay_results_bind_the_exact_resolve_command() -> None:
    """Reject replay commands with hostile identity or semantic content."""
    terminal = _terminal_record()
    stored = terminal.request.resolution
    assert isinstance(stored, DeclinedResolution)
    same_command = _resolve_command(
        terminal,
        "key-1",
        resolution=stored,
    )
    same = InteractionStoreReplayed(
        command=same_command,
        record=terminal,
        replay_kind=InteractionReplayKind.SAME_KEY,
    )
    assert same.command == same_command

    conflict_resolution = replace(
        stored,
        provenance=AnswerProvenance.POLICY,
    )
    with pytest.raises(InputValidationError) as forged_provenance:
        replace(
            same_command,
            proposed_resolution=conflict_resolution,
        )
    assert forged_provenance.value.code is InputErrorCode.FORBIDDEN
    missing_key = replace(
        same_command,
        idempotency_key=ResolutionIdempotencyKey("missing-key"),
    )
    wrong_actor = replace(
        same_command,
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("other-user"))
        ),
    )
    other = _pending_record_on_branch("other-replay", "branch-1")
    other_resolution = replace(
        stored,
        request_id=other.request.request_id,
    )
    wrong_correlation = replace(
        same_command,
        correlation=other.correlation,
        proposed_resolution=other_resolution,
    )
    for hostile in (
        missing_key,
        wrong_actor,
        wrong_correlation,
    ):
        with pytest.raises(InputValidationError):
            InteractionStoreReplayed(
                command=hostile,
                record=terminal,
                replay_kind=InteractionReplayKind.SAME_KEY,
            )

    semantic_command = replace(
        same_command,
        idempotency_key=ResolutionIdempotencyKey("key-2"),
    )
    digest = terminal.resolution_digest
    assert digest is not None
    semantic_record = replace(
        terminal,
        store_revision=InteractionStoreRevision(terminal.store_revision + 1),
        idempotency_ledger=terminal.idempotency_ledger
        + (
            ResolutionIdempotencyEntry(
                key=semantic_command.idempotency_key,
                resolution_digest=digest,
            ),
        ),
    )
    semantic = InteractionStoreReplayed(
        command=semantic_command,
        record=semantic_record,
        previous=terminal,
        replay_kind=InteractionReplayKind.SEMANTIC_NEW_KEY,
    )
    assert semantic.store_mutation_applied
    assert semantic.record.request == terminal.request

    already_bound = replace(
        semantic_command,
        idempotency_key=ResolutionIdempotencyKey("key-1"),
    )
    wrong_append = replace(
        semantic_record,
        idempotency_ledger=terminal.idempotency_ledger
        + (
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("wrong-key"),
                resolution_digest=digest,
            ),
        ),
    )
    changed_request = replace(
        semantic_record.request,
        reason="A replay cannot mutate lifecycle content.",
    )
    changed_lifecycle = replace(
        semantic_record,
        request=changed_request,
        semantic_fingerprint=semantic_request_fingerprint(changed_request),
    )
    hostile_semantic_cases = (
        (already_bound, semantic_record),
        (semantic_command, wrong_append),
        (semantic_command, changed_lifecycle),
        (wrong_actor, semantic_record),
        (wrong_correlation, semantic_record),
    )
    for hostile_command, hostile_record in hostile_semantic_cases:
        with pytest.raises(InputValidationError):
            InteractionStoreReplayed(
                command=hostile_command,
                record=hostile_record,
                previous=terminal,
                replay_kind=InteractionReplayKind.SEMANTIC_NEW_KEY,
            )


def test_semantic_replay_helper_appends_only_the_exact_new_key() -> None:
    """Give concrete stores one nonduplicative semantic-ledger reducer."""
    terminal = _terminal_record()
    stored = terminal.request.resolution
    assert isinstance(stored, DeclinedResolution)
    command = _resolve_command(
        terminal,
        "semantic-key",
        resolution=stored,
    )

    replay = apply_semantic_resolution_replay(terminal, command)

    assert replay.replay_kind is InteractionReplayKind.SEMANTIC_NEW_KEY
    assert replay.previous is terminal
    assert replay.record.request == terminal.request
    assert replay.record.store_revision == terminal.store_revision + 1
    assert replay.record.idempotency_ledger[-1].key == "semantic-key"
    with pytest.raises(InputValidationError) as already_bound:
        apply_semantic_resolution_replay(
            terminal,
            replace(
                command,
                idempotency_key=ResolutionIdempotencyKey("key-1"),
            ),
        )
    assert already_bound.value.code is InputErrorCode.IDEMPOTENCY_CONFLICT


def test_idempotency_ledger_overflow_is_explicit() -> None:
    """Replay a known key before rejecting a new thirty-third binding."""
    record = _terminal_record()
    digest = record.resolution_digest
    assert digest is not None
    full_ledger = tuple(
        ResolutionIdempotencyEntry(
            key=ResolutionIdempotencyKey(f"key-{index}"),
            resolution_digest=digest,
        )
        for index in range(32)
    )
    full = replace(record, idempotency_ledger=full_ledger)
    assert len(full.idempotency_ledger) == 32
    assert (
        evaluate_resolution_idempotency(
            full,
            key=ResolutionIdempotencyKey("key-1"),
            resolution_digest=digest,
        )
        is ResolutionIdempotencyDisposition.SAME_KEY
    )
    assert (
        evaluate_resolution_idempotency(
            full,
            key=ResolutionIdempotencyKey("key-1"),
            resolution_digest="0" * 64,
        )
        is ResolutionIdempotencyDisposition.SAME_KEY_CONFLICT
    )
    assert (
        evaluate_resolution_idempotency(
            full,
            key=ResolutionIdempotencyKey("overflow"),
            resolution_digest=digest,
        )
        is ResolutionIdempotencyDisposition.LEDGER_FULL
    )

    with pytest.raises(InputValidationError) as overflow:
        replace(
            full,
            idempotency_ledger=full.idempotency_ledger
            + (
                ResolutionIdempotencyEntry(
                    key=ResolutionIdempotencyKey("overflow"),
                    resolution_digest=digest,
                ),
            ),
        )
    assert overflow.value.code is InputErrorCode.IDEMPOTENCY_LEDGER_FULL


def test_metadata_and_deadline_revisions_cover_non_lifecycle_changes() -> None:
    """Give activity and earlier deadlines race-free change tokens."""
    pending = _pending()
    correlation = InteractionCorrelation.from_request(pending)
    actor = InteractionActor(principal=_principal())
    presentation = PresentInteractionCommand(
        actor=actor,
        correlation=correlation,
        expected_store_revision=InteractionStoreRevision(1),
    )
    activity = RecordControllerActivityCommand(
        actor=actor,
        correlation=correlation,
        evidence=AcquireControllerActivity(
            request_id=pending.request_id,
            controller_id=ControllerId("controller-1"),
        ),
    )
    wait = WaitForInteractionChangeCommand(
        actor=actor,
        correlation=correlation,
        after_store_revision=InteractionStoreRevision(4),
    )
    snapshot = InteractionDeadlineSnapshot(
        schedule_revision=DeadlineScheduleRevision(6),
        deadline=InteractionDeadline(
            request_id=pending.request_id,
            monotonic_deadline=10,
        ),
    )
    deadline_wait = WaitForDeadlineChangeCommand(
        after_schedule_revision=snapshot.schedule_revision
    )

    assert presentation.correlation == correlation
    assert activity.evidence.action is ControllerActivityAction.ACQUIRE
    assert wait.after_store_revision == 4
    assert deadline_wait.after_schedule_revision == 6
    assert snapshot.deadline is not None


def test_branch_registration_has_an_exact_authorization_target() -> None:
    """Bind branch registration authorization to the full proposed edge."""
    registration = InteractionBranchRegistration(
        run_id=RunId("run-1"),
        branch_id=BranchId("child"),
        parent_branch_id=BranchId("parent"),
        principal=_principal(),
    )
    branch_record = InteractionBranchRecord(
        registration=registration,
        store_revision=InteractionStoreRevision(1),
    )
    command = RegisterInteractionBranchCommand(
        actor=InteractionActor(principal=_principal()),
        registration=registration,
    )
    lookup = InteractionBranchRootLookup(
        actor=command.actor,
        run_id=registration.run_id,
        branch_id=registration.branch_id,
    )
    root = InteractionBranchRoot(
        run_id=registration.run_id,
        branch_id=registration.branch_id,
        root_branch_id=registration.parent_branch_id,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(root, "root_branch_id", BranchId("forged-root"))
    assert tuple(item.name for item in fields(root)) == (
        "run_id",
        "branch_id",
        "root_branch_id",
    )
    applied = InteractionBranchRegistrationApplied(
        command=command,
        record=branch_record,
    )
    replayed = InteractionBranchRegistrationReplayed(
        command=command,
        record=branch_record,
    )
    due = TerminalizeDueInteractionsCommand()

    assert applied.store_mutation_applied
    assert not replayed.store_mutation_applied
    assert command.authorization_target == (
        InteractionBranchAuthorizationTarget(
            run_id=registration.run_id,
            branch_id=registration.branch_id,
            parent_branch_id=registration.parent_branch_id,
            principal=registration.principal,
        )
    )
    assert lookup.authorization_target == InteractionScopeAuthorizationTarget(
        run_id=registration.run_id,
        branch_id=registration.branch_id,
        principal=registration.principal,
    )
    assert _resolve_interaction_branch_root((branch_record,), lookup) == root
    assert due.maximum_results == 256

    wrong_actor = replace(
        command,
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("other-user"))
        ),
    )
    wrong_registration = replace(
        registration,
        parent_branch_id=BranchId("other-parent"),
    )
    wrong_edge = replace(command, registration=wrong_registration)
    for hostile in (wrong_actor, wrong_edge):
        with pytest.raises(InputValidationError):
            InteractionBranchRegistrationApplied(
                command=hostile,
                record=branch_record,
            )
        with pytest.raises(InputValidationError):
            InteractionBranchRegistrationReplayed(
                command=hostile,
                record=branch_record,
            )


def test_deadline_resolution_cannot_bind_a_caller_transport_key() -> None:
    """Reserve deadline authority and its commit shape for store internals."""
    previous = _pending_record()
    resolution = ExpiredResolution(
        request_id=previous.request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW + timedelta(days=1),
    )
    transition = resolve_request(
        previous.request,
        resolution,
        expected_state_revision=previous.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    record = InteractionRecord(
        request=transition.request,
        semantic_fingerprint=previous.semantic_fingerprint,
        absolute_expires_at=previous.absolute_expires_at,
        presentation=previous.presentation,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
        resolution_digest=canonical_resolution_digest(resolution),
        resolved_by=_DEADLINE_RESOLVER,
    )
    applied = ResolveInteractionApplied(
        record=record,
        previous=previous,
        decision_stage=ResolutionDecisionStage.DEADLINE,
        observed_at=_observation(86_400),
        policy=_POLICY,
    )

    assert applied.idempotency_key is None
    assert applied.record.idempotency_ledger == previous.idempotency_ledger
    with pytest.raises(InputValidationError):
        ResolveInteractionApplied(
            record=record,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=_observation(86_400),
            policy=_POLICY,
            idempotency_key=ResolutionIdempotencyKey("caller-key"),
        )
    with pytest.raises(InputValidationError):
        replace(record, resolved_by=_principal())


@pytest.mark.parametrize("resolution_kind", ("timeout", "expiry"))
def test_deadline_resolution_requires_due_policy_time(
    resolution_kind: str,
) -> None:
    """Reject deadline outcomes before their authoritative deadline."""
    if resolution_kind == "timeout":
        previous = _presented_advisory_record()
        deadline = previous.request.advisory_deadline
        assert deadline is not None
        resolution: TimedOutResolution | ExpiredResolution = (
            TimedOutResolution(
                request_id=previous.request.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=deadline - timedelta(microseconds=1),
            )
        )
    else:
        previous = _pending_record()
        resolution = ExpiredResolution(
            request_id=previous.request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=previous.absolute_expires_at
            - timedelta(microseconds=1),
        )

    if isinstance(resolution, TimedOutResolution):
        transition = resolve_request(
            previous.request,
            resolution,
            expected_state_revision=previous.request.state_revision,
        )
        assert isinstance(transition, InputTransitionRejected)
    else:
        with pytest.raises(InputValidationError):
            _deadline_record(previous, resolution)


def test_presented_advisory_timeout_exhausts_only_its_wait_budget() -> None:
    """Commit a due timeout without changing unrelated request metadata."""
    previous = _presented_advisory_record()
    deadline = previous.request.advisory_deadline
    assert deadline is not None
    resolution = TimedOutResolution(
        request_id=previous.request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=deadline,
    )
    record = _deadline_record(previous, resolution)
    applied = ResolveInteractionApplied(
        record=record,
        previous=previous,
        decision_stage=ResolutionDecisionStage.DEADLINE,
        observed_at=_observation(65),
        policy=_POLICY,
    )

    assert applied.record.advisory_wait is not None
    assert applied.record.advisory_wait.status is AdvisoryWaitStatus.EXHAUSTED
    assert applied.record.advisory_wait.remaining_seconds == 0
    assert applied.record.absolute_expires_at == previous.absolute_expires_at
    assert applied.record.presentation is previous.presentation
    assert applied.record.idempotency_ledger == ()


@pytest.mark.parametrize(
    "resolution",
    (
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
            scope=CancellationScope.REQUEST,
        ),
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=1),
            scope=CancellationScope.CONTAINING_RUN,
        ),
        UnavailableResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
        SupersededResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
    ),
)
def test_non_candidate_terminal_outcomes_never_enter_caller_key_replay(
    resolution: InputResolution,
) -> None:
    """Reject caller-key replay for cancellation and terminalization."""
    previous = _pending_record()
    transition = resolve_request(
        previous.request,
        resolution,
        expected_state_revision=previous.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    record = replace(
        previous,
        request=transition.request,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
        resolution_digest=canonical_resolution_digest(resolution),
        resolved_by=_principal(),
    )
    digest = record.resolution_digest
    assert digest is not None

    assert (
        evaluate_resolution_idempotency(
            record,
            key=ResolutionIdempotencyKey("caller-key"),
            resolution_digest=digest,
        )
        is ResolutionIdempotencyDisposition.TERMINAL_CONFLICT
    )
    with pytest.raises(InputValidationError):
        InteractionStoreReplayed(
            command=_resolve_command(record, "caller-key"),
            record=record,
            replay_kind=InteractionReplayKind.SAME_KEY,
        )


def test_deadline_outcomes_never_enter_candidate_commit_or_replay() -> None:
    """Keep the sealed deadline path outside candidate keys and replay."""
    previous = _pending_record()
    resolution = ExpiredResolution(
        request_id=previous.request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=previous.absolute_expires_at,
    )
    record = _deadline_record(previous, resolution)
    digest = record.resolution_digest
    assert digest is not None

    assert (
        evaluate_resolution_idempotency(
            record,
            key=ResolutionIdempotencyKey("caller-key"),
            resolution_digest=digest,
        )
        is ResolutionIdempotencyDisposition.TERMINAL_CONFLICT
    )
    with pytest.raises(InputValidationError):
        ResolveInteractionApplied(
            record=record,
            previous=previous,
            decision_stage=ResolutionDecisionStage.COMMIT,
            observed_at=_observation(86_400),
            policy=_POLICY,
            idempotency_key=ResolutionIdempotencyKey("caller-key"),
        )
    with pytest.raises(InputValidationError):
        InteractionStoreReplayed(
            command=_resolve_command(record, "caller-key"),
            record=record,
            replay_kind=InteractionReplayKind.SAME_KEY,
        )


def test_direct_and_scope_cancellation_have_distinct_fixed_scopes() -> None:
    """Prevent direct cancellation from masquerading as run cancellation."""
    pending = _pending()
    actor = InteractionActor(principal=_principal())
    correlation = InteractionCorrelation.from_request(pending)
    direct = CancelInteractionCommand(
        actor=actor,
        correlation=correlation,
        provenance=AnswerProvenance.HUMAN,
        expected_state_revision=pending.state_revision,
    )
    scope = InteractionExecutionScope(
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        include_descendants=True,
    )
    scope_cancel = TerminalizeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )

    assert direct.status is ResolutionStatus.CANCELLED
    assert direct.cancellation_scope is CancellationScope.REQUEST
    assert scope_cancel.status is ResolutionStatus.CANCELLED
    assert scope_cancel.cancellation_scope is CancellationScope.CONTAINING_RUN
    assert (
        "status"
        not in signature(TerminalizeInteractionScopeCommand).parameters
    )
    assert (
        "cancellation_scope"
        not in signature(TerminalizeInteractionScopeCommand).parameters
    )
    with pytest.raises(TypeError):
        TerminalizeInteractionScopeCommand(
            actor=actor,
            scope=scope,
            provenance=AnswerProvenance.HUMAN,
            status=ResolutionStatus.SUPERSEDED,  # type: ignore[call-arg]
        )
    with pytest.raises(InputValidationError):
        TerminalizeInteractionCommand(
            actor=actor,
            correlation=correlation,
            status=ResolutionStatus.CANCELLED,  # type: ignore[arg-type]
            provenance=AnswerProvenance.HUMAN,
        )


def test_create_result_exactly_binds_its_create_and_admit_command() -> None:
    """Reject create results that change the admitted canonical request."""
    command = CreateInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        request=_created(),
    )
    applied = apply_create_interaction(command, _POLICY)

    assert isinstance(applied, CreateInteractionApplied)
    assert applied.record.request.state is RequestState.PENDING
    assert applied.record.request.state_revision == 1
    assert applied.record.store_revision == 1
    changed_request = replace(
        applied.record.request,
        reason="A forged admitted reason.",
    )
    changed = replace(
        applied.record,
        request=changed_request,
        semantic_fingerprint=semantic_request_fingerprint(changed_request),
    )
    with pytest.raises(InputValidationError):
        CreateInteractionApplied(
            command=command,
            record=changed,
            policy=_POLICY,
        )


def test_create_atomically_carries_broker_minted_resumer_binding() -> None:
    """Bind a bare resumer to the request continuation during creation."""
    resumer = _RecordingResumer()
    command = CreateInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        request=_created(),
        resumer=resumer,
    )

    registration = command.resumer_registration
    assert isinstance(registration, InputResumerRegistration)
    assert registration.continuation_id == command.request.continuation_id
    assert registration.resumer is resumer
    assert "_RecordingResumer" not in repr(command)
    assert isinstance(
        apply_create_interaction(command, _POLICY), CreateInteractionApplied
    )


def test_equivalent_request_limit_counts_terminal_branch_lifetime() -> None:
    """Keep terminal equivalents in the logical-branch loop budget."""
    terminal_records: list[InteractionRecord] = []
    actor = InteractionActor(principal=_principal())
    for index in range(3):
        pending = _pending_record_on_branch(
            f"terminal-{index}",
            "branch-1",
        )
        cancelled = apply_request_cancellation(
            pending,
            CancelInteractionCommand(
                actor=actor,
                correlation=pending.correlation,
                provenance=AnswerProvenance.HUMAN,
            ),
            _observation(1),
            _POLICY,
        )
        assert isinstance(cancelled, CancelInteractionApplied)
        terminal_records.append(cancelled.record)
    candidate = replace(
        _created(),
        request_id=InputRequestId("candidate-4"),
        continuation_id=ContinuationId("candidate-4-continuation"),
    )
    command = CreateInteractionCommand(actor=actor, request=candidate)

    with pytest.raises(InputValidationError) as loop_limit:
        validate_interaction_admission(
            tuple(terminal_records),
            command,
            _POLICY,
        )
    assert loop_limit.value.code is InputErrorCode.INTERACTION_LOOP_LIMIT

    other_origin = replace(candidate.origin, branch_id=BranchId("branch-2"))
    other_branch = replace(candidate, origin=other_origin)
    validate_interaction_admission(
        tuple(terminal_records),
        CreateInteractionCommand(actor=actor, request=other_branch),
        _POLICY,
    )


def test_direct_cancel_result_is_request_scoped_and_command_bound() -> None:
    """Fix direct cancellation status, scope, authority, and timestamp."""
    previous = _pending_record()
    command = CancelInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=previous.correlation,
        provenance=AnswerProvenance.HUMAN,
        expected_state_revision=previous.request.state_revision,
    )
    applied = apply_request_cancellation(
        previous,
        command,
        _observation(1),
        _POLICY,
    )
    resolution = applied.record.request.resolution

    assert isinstance(applied, CancelInteractionApplied)
    assert isinstance(resolution, CancelledResolution)
    assert resolution.scope is CancellationScope.REQUEST
    assert resolution.provenance is command.provenance
    assert resolution.resolved_at == _NOW + timedelta(seconds=1)
    assert applied.record.resolved_by == command.actor.principal
    assert applied.record.idempotency_ledger == ()

    forged = replace(
        applied.record,
        resolved_by=PrincipalScope(user_id=UserId("other-user")),
    )
    with pytest.raises(InputValidationError):
        CancelInteractionApplied(
            command=command,
            previous=previous,
            record=forged,
            observed_at=_observation(1),
            policy=_POLICY,
        )


@pytest.mark.parametrize(
    "status",
    (ResolutionStatus.UNAVAILABLE, ResolutionStatus.SUPERSEDED),
)
def test_direct_terminalization_is_exactly_command_bound(
    status: ResolutionStatus,
) -> None:
    """Bind direct terminal status, provenance, authority, and revisions."""
    previous = _pending_record()
    command = TerminalizeInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=previous.correlation,
        status=status,  # type: ignore[arg-type]
        provenance=AnswerProvenance.HUMAN,
        expected_state_revision=previous.request.state_revision,
    )
    applied = apply_request_terminalization(
        previous,
        command,
        _observation(2),
        _POLICY,
    )
    resolution = applied.record.request.resolution

    assert isinstance(applied, TerminalizeInteractionApplied)
    assert resolution is not None
    assert resolution.status is status
    assert resolution.provenance is command.provenance
    assert resolution.resolved_at == _NOW + timedelta(seconds=2)
    assert applied.record.request.state_revision == (
        previous.request.state_revision + 1
    )
    assert applied.record.store_revision == previous.store_revision + 1
    assert applied.record.resolved_by == command.actor.principal


def test_cancel_and_terminalize_report_committed_precedence_winners() -> None:
    """Return deadline or lease commits instead of a lying rejection."""
    previous = _pending_record()
    actor = InteractionActor(principal=_principal())
    cancel = CancelInteractionCommand(
        actor=actor,
        correlation=previous.correlation,
        provenance=AnswerProvenance.HUMAN,
    )
    terminalize = TerminalizeInteractionCommand(
        actor=actor,
        correlation=previous.correlation,
        status=ResolutionStatus.UNAVAILABLE,
        provenance=AnswerProvenance.HUMAN,
    )
    cancelled_at_expiry = apply_request_cancellation(
        previous,
        cancel,
        _observation(86_400),
        _POLICY,
    )
    terminalized_at_expiry = apply_request_terminalization(
        previous,
        terminalize,
        _observation(86_400),
        _POLICY,
    )
    for result in (cancelled_at_expiry, terminalized_at_expiry):
        assert isinstance(result, ResolveInteractionApplied)
        assert result.decision_stage is ResolutionDecisionStage.DEADLINE
        assert result.record.request.state is RequestState.EXPIRED

    presented = _presented_advisory_record()
    acquisition = apply_controller_activity(
        presented,
        _controller_command(
            presented,
            AcquireControllerActivity(
                request_id=presented.request.request_id,
                controller_id=ControllerId("controller-precedence"),
            ),
        ),
        _observation(25),
        _POLICY,
        lease_nonce=ActiveControlLeaseNonce("precedence-nonce"),
    )
    assert isinstance(acquisition, ControllerActivityApplied)
    leased = acquisition.record
    lease_cancel = replace(
        cancel,
        correlation=leased.correlation,
    )
    lease_terminalize = replace(
        terminalize,
        correlation=leased.correlation,
    )
    cancelled_at_lease = apply_request_cancellation(
        leased,
        lease_cancel,
        _observation(55),
        _POLICY,
    )
    terminalized_at_lease = apply_request_terminalization(
        leased,
        lease_terminalize,
        _observation(55),
        _POLICY,
    )
    for result in (cancelled_at_lease, terminalized_at_lease):
        assert isinstance(result, ControllerLeaseExpiredApplied)
        assert result.record.request.state is RequestState.PENDING
        assert result.record.advisory_wait is not None
        assert result.record.advisory_wait.status is AdvisoryWaitStatus.RUNNING


def test_scope_cancellation_and_supersession_are_distinct_atomic_batches() -> (
    None
):
    """Apply fixed containing-run cancellation or descendant supersession."""
    root = _pending_record()
    child = _pending_record_on_branch(
        "child-request",
        "child-branch",
        parent_branch_id="branch-1",
    )
    branch_record = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("child-branch"),
            parent_branch_id=BranchId("branch-1"),
            principal=_principal(),
        ),
        store_revision=InteractionStoreRevision(1),
    )
    scope = InteractionExecutionScope(
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        include_descendants=True,
    )
    actor = InteractionActor(principal=_principal())
    cancel_command = TerminalizeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )
    supersede_command = SupersedeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    )
    for target in (
        cancel_command.authorization_target,
        supersede_command.authorization_target,
        ListInteractionsCommand(
            actor=actor,
            scope=scope,
        ).authorization_target,
    ):
        assert isinstance(target, InteractionScopeAuthorizationTarget)
        assert target.principal == actor.principal
        assert target.branch_id == scope.branch_id
    previous = (root, child)
    generation = InteractionStoreGeneration(8)
    cancellation_backing = _new_interaction_store_backing(
        records=previous,
        branch_records=(branch_record,),
        store_generation=generation,
    )
    supersession_backing = _new_interaction_store_backing(
        records=previous,
        branch_records=(branch_record,),
        store_generation=generation,
    )
    cancellation_transaction = _begin_scope_transaction(
        cancellation_backing,
        cancel_command,
    )
    supersession_transaction = _begin_scope_transaction(
        supersession_backing,
        supersede_command,
    )
    cancellation = _apply_scope_cancellation(
        cancellation_transaction,
        cancel_command,
        _observation(3),
        _POLICY,
        backing=cancellation_backing,
    )
    supersession = _apply_scope_supersession(
        supersession_transaction,
        supersede_command,
        _observation(3),
        _POLICY,
        backing=supersession_backing,
    )

    assert isinstance(cancellation, ScopeCancellationApplied)
    assert isinstance(supersession, ScopeSupersessionApplied)
    for record in cancellation.records:
        resolution = record.request.resolution
        assert isinstance(resolution, CancelledResolution)
        assert resolution.scope is CancellationScope.CONTAINING_RUN
        assert resolution.resolved_at == _NOW + timedelta(seconds=3)
    for record in supersession.records:
        resolution = record.request.resolution
        assert isinstance(resolution, SupersededResolution)
        assert resolution.resolved_at == _NOW + timedelta(seconds=3)
    assert "supersede_scope" in InteractionStore.__dict__
    assert (
        "status" not in signature(SupersedeInteractionScopeCommand).parameters
    )
    with pytest.raises(TypeError):
        SupersedeInteractionScopeCommand(
            actor=actor,
            scope=scope,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            status=ResolutionStatus.CANCELLED,  # type: ignore[call-arg]
        )

    with pytest.raises(InputValidationError):
        ScopeSupersessionApplied(
            command=supersede_command,
            previous=supersession_transaction.selected_records,
            records=cancellation.records,
            observed_at=_observation(3),
            policy=_POLICY,
            _token=object(),
        )
    changed_request = replace(
        supersession.records[0].request,
        reason="A forged scope mutation.",
    )
    forged_record = replace(
        supersession.records[0],
        request=changed_request,
        semantic_fingerprint=semantic_request_fingerprint(changed_request),
    )
    with pytest.raises(InputValidationError):
        ScopeSupersessionApplied(
            command=supersede_command,
            previous=supersession_transaction.selected_records,
            records=(forged_record, supersession.records[1]),
            observed_at=_observation(3),
            policy=_POLICY,
            _token=object(),
        )


def test_scope_batch_reports_mixed_deadline_lease_and_command_winners() -> (
    None
):
    """Represent every mutation winner in one atomic selected batch."""
    normal = _pending_record()
    timed = _with_request_identity(
        _presented_advisory_record(),
        "timed-request",
    )
    lease_base = _with_request_identity(
        _presented_advisory_record(),
        "leased-request",
    )
    acquisition = apply_controller_activity(
        lease_base,
        _controller_command(
            lease_base,
            AcquireControllerActivity(
                request_id=lease_base.request.request_id,
                controller_id=ControllerId("scope-controller"),
            ),
        ),
        _observation(25),
        _POLICY,
        lease_nonce=ActiveControlLeaseNonce("scope-lease"),
    )
    assert isinstance(acquisition, ControllerActivityApplied)
    leased = acquisition.record
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=InteractionExecutionScope(run_id=RunId("run-1")),
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    )
    generation = InteractionStoreGeneration(21)
    snapshot = (timed, normal, leased)
    backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=generation,
    )
    transaction = _begin_scope_transaction(backing, command)

    result = _apply_scope_cancellation(
        transaction,
        command,
        _observation(70),
        _POLICY,
        backing=backing,
    )

    assert isinstance(result, ScopeCancellationApplied)
    records = {record.request.request_id: record for record in result.records}
    assert (
        records[normal.request.request_id].request.state
        is RequestState.CANCELLED
    )
    assert (
        records[timed.request.request_id].request.state
        is RequestState.TIMED_OUT
    )
    lease_result = records[leased.request.request_id]
    assert lease_result.request.state is RequestState.PENDING
    assert lease_result.advisory_wait is not None
    assert lease_result.advisory_wait.status is AdvisoryWaitStatus.RUNNING


def test_private_scope_transaction_closes_disclosure_exploits() -> None:
    """Keep complete scope proof private and public results scope-local."""
    root = _pending_record()
    child = _pending_record_on_branch(
        "child-request",
        "child-branch",
        parent_branch_id="branch-1",
    )
    leaf = _pending_record_on_branch(
        "leaf-request",
        "leaf-branch",
        parent_branch_id="middle-branch",
    )
    sibling = _pending_record_on_branch(
        "sibling-request",
        "sibling-branch",
        parent_branch_id="branch-1",
    )
    branch_records = tuple(
        InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=RunId("run-1"),
                branch_id=BranchId(branch_id),
                parent_branch_id=BranchId(parent_branch_id),
                principal=_principal(),
            ),
            store_revision=InteractionStoreRevision(index),
        )
        for index, (branch_id, parent_branch_id) in enumerate(
            (
                ("child-branch", "branch-1"),
                ("middle-branch", "child-branch"),
                ("leaf-branch", "middle-branch"),
                ("sibling-branch", "branch-1"),
            ),
            start=1,
        )
    )
    snapshot = (sibling, leaf, root, child)
    scope = InteractionExecutionScope(
        run_id=RunId("run-1"),
        branch_id=BranchId("child-branch"),
        include_descendants=True,
    )
    generation = InteractionStoreGeneration(12)
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=scope,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    )
    backing = _new_interaction_store_backing(
        records=snapshot,
        branch_records=tuple(reversed(branch_records)),
        store_generation=generation,
    )
    reordered_backing = _new_interaction_store_backing(
        records=tuple(reversed(snapshot)),
        branch_records=branch_records,
        store_generation=generation,
    )
    transaction = _begin_scope_transaction(backing, command)
    repeated = _begin_scope_transaction(reordered_backing, command)

    assert transaction.snapshot_digest == repeated.snapshot_digest
    assert transaction.selected_records == repeated.selected_records
    assert tuple(
        record.request.request_id for record in transaction.selected_records
    ) == (InputRequestId("child-request"), InputRequestId("leaf-request"))
    assert "request-1" not in repr(transaction)
    assert "sibling-request" not in repr(transaction)
    root_scope = replace(
        scope,
        branch_id=BranchId("branch-1"),
    )
    root_command = replace(command, scope=root_scope)
    root_transaction = _begin_scope_transaction(backing, root_command)
    assert {
        record.request.request_id
        for record in root_transaction.selected_records
    } == {
        InputRequestId("request-1"),
        InputRequestId("child-request"),
        InputRequestId("leaf-request"),
        InputRequestId("sibling-request"),
    }
    applied = _apply_scope_cancellation(
        transaction,
        command,
        _observation(4),
        _POLICY,
        backing=backing,
    )
    assert isinstance(applied, ScopeCancellationApplied)
    assert len(applied.records) == 2
    assert tuple(field.name for field in fields(applied)) == (
        "command",
        "previous",
        "records",
        "observed_at",
        "policy",
        "store_mutation_applied",
        "kind",
    )
    public_repr = repr(applied)
    for private_content in (
        "request-1",
        "sibling-request",
        "snapshot_records",
        "branch_records",
        "transaction",
    ):
        assert private_content not in public_repr

    for hostile_records in (
        applied.records[:1],
        applied.records + (sibling,),
        (applied.records[0], applied.records[0]),
    ):
        with pytest.raises(InputValidationError):
            ScopeCancellationApplied(
                command=command,
                previous=transaction.selected_records,
                records=hostile_records,
                observed_at=_observation(4),
                policy=_POLICY,
                _token=object(),
            )
    with pytest.raises(InputValidationError) as stale:
        _apply_scope_cancellation(
            transaction,
            command,
            _observation(4),
            _POLICY,
            backing=backing,
        )
    assert stale.value.code is InputErrorCode.STALE_REVISION
    with pytest.raises(InputValidationError) as mismatched:
        _apply_scope_cancellation(
            transaction,
            replace(command, provenance=AnswerProvenance.HUMAN),
            _observation(4),
            _POLICY,
            backing=backing,
        )
    assert mismatched.value.code is InputErrorCode.CORRELATION_MISMATCH

    missing_scope = replace(
        scope,
        branch_id=BranchId("missing-branch"),
    )
    empty_cancel_command = replace(command, scope=missing_scope)
    empty_supersede_command = SupersedeInteractionScopeCommand(
        actor=command.actor,
        scope=missing_scope,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    )
    empty_cancel_transaction = _begin_scope_transaction(
        backing,
        empty_cancel_command,
    )
    empty_supersede_transaction = _begin_scope_transaction(
        backing,
        empty_supersede_command,
    )
    empty_cancel = _apply_scope_cancellation(
        empty_cancel_transaction,
        empty_cancel_command,
        _observation(4),
        _POLICY,
        backing=backing,
    )
    empty_supersede = _apply_scope_supersession(
        empty_supersede_transaction,
        empty_supersede_command,
        _observation(4),
        _POLICY,
        backing=backing,
    )
    assert isinstance(empty_cancel, ScopeCancellationReplayed)
    assert isinstance(empty_supersede, ScopeSupersessionReplayed)
    assert empty_cancel.records == ()
    assert empty_supersede.records == ()
    assert tuple(field.name for field in fields(empty_cancel)) == (
        "command",
        "store_mutation_applied",
        "kind",
    )
    for private_content in (
        "request-1",
        "child-request",
        "leaf-request",
        "sibling-request",
        "snapshot_records",
        "branch_records",
    ):
        assert private_content not in repr(empty_cancel)
    with pytest.raises(InputValidationError):
        ScopeCancellationReplayed(
            command=command,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _InteractionScopeTransaction(
            command=command,
            scope=scope,
            principal=command.actor.principal,
            store_generation=generation,
            snapshot_digest=transaction.snapshot_digest,
            selected_records=transaction.selected_records,
            backing=backing,
            _token=object(),
        )
    assert not hasattr(_InteractionScopeTransaction, "_from_snapshot")
    for removed_name in (
        "InteractionScopeSelectionPlan",
        "select_interaction_scope",
        "apply_scope_cancellation",
        "apply_scope_supersession",
    ):
        assert not hasattr(interaction_api, removed_name)


def test_scope_transaction_rejects_ancestry_ownership_and_graph_drift() -> (
    None
):
    """Fail closed on actor, branch, ancestry, and graph ownership drift."""
    root = _pending_record()
    child = _pending_record_on_branch(
        "child-request",
        "child-branch",
        parent_branch_id="branch-1",
    )
    registration = InteractionBranchRegistration(
        run_id=RunId("run-1"),
        branch_id=BranchId("child-branch"),
        parent_branch_id=BranchId("branch-1"),
        principal=_principal(),
    )
    branch_record = InteractionBranchRecord(
        registration=registration,
        store_revision=InteractionStoreRevision(1),
    )
    scope = InteractionExecutionScope(
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        include_descendants=True,
    )
    generation = InteractionStoreGeneration(1)
    command = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )
    wrong_parent_origin = replace(
        child.request.origin,
        parent_branch_id=BranchId("other-parent"),
    )
    wrong_parent_request = replace(
        child.request,
        origin=wrong_parent_origin,
    )
    wrong_parent = replace(
        child,
        request=wrong_parent_request,
        semantic_fingerprint=semantic_request_fingerprint(
            wrong_parent_request
        ),
    )
    other_principal = PrincipalScope(user_id=UserId("other-user"))
    wrong_owner_origin = replace(
        child.request.origin,
        principal=other_principal,
    )
    wrong_owner_request = replace(
        child.request,
        origin=wrong_owner_origin,
    )
    wrong_owner = replace(
        child,
        request=wrong_owner_request,
        semantic_fingerprint=semantic_request_fingerprint(wrong_owner_request),
    )
    cross_run_record = replace(
        branch_record,
        registration=replace(
            registration,
            run_id=RunId("other-run"),
        ),
    )
    cycle_records = (
        replace(
            branch_record,
            registration=replace(
                registration,
                parent_branch_id=BranchId("grandchild-branch"),
            ),
        ),
        InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=RunId("run-1"),
                branch_id=BranchId("grandchild-branch"),
                parent_branch_id=BranchId("child-branch"),
                principal=_principal(),
            ),
            store_revision=InteractionStoreRevision(2),
        ),
    )
    cases = (
        ((wrong_parent,), (branch_record,)),
        ((wrong_owner,), (branch_record,)),
        ((child,), (cross_run_record,)),
        ((child,), cycle_records),
        ((child, child), (branch_record,)),
        ((child,), (branch_record, branch_record)),
    )
    for records, branches in cases:
        with pytest.raises(InputValidationError):
            _new_interaction_store_backing(
                records=records,
                branch_records=branches,
                store_generation=generation,
            )

    other_principal = PrincipalScope(user_id=UserId("other-user"))
    other_actor_command = replace(
        command,
        actor=InteractionActor(principal=other_principal),
    )
    with pytest.raises(InputValidationError) as actor_mismatch:
        _begin_scope_transaction(
            _new_interaction_store_backing(
                records=(root, child),
                branch_records=(branch_record,),
                store_generation=generation,
            ),
            other_actor_command,
        )
    assert actor_mismatch.value.code is InputErrorCode.FORBIDDEN

    other_root_origin = replace(
        root.request.origin,
        principal=other_principal,
    )
    other_root_request = replace(root.request, origin=other_root_origin)
    other_root = replace(
        root,
        request=other_root_request,
        semantic_fingerprint=semantic_request_fingerprint(other_root_request),
    )
    with pytest.raises(InputValidationError) as root_mismatch:
        _begin_scope_transaction(
            _new_interaction_store_backing(
                records=(other_root, child),
                branch_records=(branch_record,),
                store_generation=generation,
            ),
            command,
        )
    assert root_mismatch.value.code is InputErrorCode.FORBIDDEN

    leaf = _pending_record_on_branch(
        "leaf-request",
        "leaf-branch",
        parent_branch_id="middle-branch",
    )
    requestless_middle = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("middle-branch"),
            parent_branch_id=BranchId("branch-1"),
            principal=other_principal,
        ),
        store_revision=InteractionStoreRevision(2),
    )
    leaf_branch = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("leaf-branch"),
            parent_branch_id=BranchId("middle-branch"),
            principal=_principal(),
        ),
        store_revision=InteractionStoreRevision(3),
    )
    with pytest.raises(InputValidationError) as intermediate_mismatch:
        _begin_scope_transaction(
            _new_interaction_store_backing(
                records=(root, leaf),
                branch_records=(requestless_middle, leaf_branch),
                store_generation=generation,
            ),
            command,
        )
    assert intermediate_mismatch.value.code is InputErrorCode.FORBIDDEN

    other_sibling = _pending_record_on_branch(
        "other-sibling-request",
        "other-sibling-branch",
        parent_branch_id="branch-1",
    )
    other_sibling_origin = replace(
        other_sibling.request.origin,
        principal=other_principal,
    )
    other_sibling_request = replace(
        other_sibling.request,
        origin=other_sibling_origin,
    )
    other_sibling = replace(
        other_sibling,
        request=other_sibling_request,
        semantic_fingerprint=semantic_request_fingerprint(
            other_sibling_request
        ),
    )
    other_sibling_branch = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("other-sibling-branch"),
            parent_branch_id=BranchId("branch-1"),
            principal=other_principal,
        ),
        store_revision=InteractionStoreRevision(4),
    )
    child_scope_command = replace(
        command,
        scope=replace(scope, branch_id=BranchId("child-branch")),
    )
    with pytest.raises(InputValidationError) as sibling_mismatch:
        _begin_scope_transaction(
            _new_interaction_store_backing(
                records=(root, child, other_sibling),
                branch_records=(branch_record, other_sibling_branch),
                store_generation=generation,
            ),
            child_scope_command,
        )
    assert sibling_mismatch.value.code is InputErrorCode.FORBIDDEN

    same_branch_other = _pending_record_on_branch(
        "same-branch-other-request",
        "branch-1",
    )
    same_branch_other_origin = replace(
        same_branch_other.request.origin,
        principal=other_principal,
    )
    same_branch_other_request = replace(
        same_branch_other.request,
        origin=same_branch_other_origin,
    )
    same_branch_other = replace(
        same_branch_other,
        request=same_branch_other_request,
        semantic_fingerprint=semantic_request_fingerprint(
            same_branch_other_request
        ),
    )
    with pytest.raises(InputValidationError) as mixed_branch:
        _begin_scope_transaction(
            _new_interaction_store_backing(
                records=(root, same_branch_other),
                store_generation=generation,
            ),
            command,
        )
    assert mixed_branch.value.code is InputErrorCode.FORBIDDEN


def test_rejected_results_are_runtime_bound_to_their_operation() -> None:
    """Reject cross-operation commands in every specialized result class."""
    previous = _pending_record()
    actor = InteractionActor(principal=_principal())
    create = CreateInteractionCommand(actor=actor, request=_created())
    cancel = CancelInteractionCommand(
        actor=actor,
        correlation=previous.correlation,
        provenance=AnswerProvenance.HUMAN,
    )
    terminalize = TerminalizeInteractionCommand(
        actor=actor,
        correlation=previous.correlation,
        status=ResolutionStatus.UNAVAILABLE,
        provenance=AnswerProvenance.HUMAN,
    )
    scope = InteractionExecutionScope(run_id=RunId("run-1"))
    cancel_scope = TerminalizeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )
    supersede_scope = SupersedeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )
    due = TerminalizeDueInteractionsCommand()
    presentation = _detachment_command(previous)
    activity = _controller_command(
        previous,
        AcquireControllerActivity(
            request_id=previous.request.request_id,
            controller_id=ControllerId("controller-1"),
        ),
    )
    error = InputTransitionError(
        code=InputErrorCode.FORBIDDEN,
        path="authorization",
        message="operation is not authorized",
    )
    cases: tuple[tuple[Callable[..., object], object, object], ...] = (
        (CreateInteractionRejected, create, cancel),
        (CancelInteractionRejected, cancel, create),
        (TerminalizeInteractionRejected, terminalize, cancel),
        (ScopeCancellationRejected, cancel_scope, supersede_scope),
        (ScopeSupersessionRejected, supersede_scope, cancel_scope),
        (DueInteractionsRejected, due, create),
        (InteractionPresentationRejected, presentation, activity),
        (ControllerActivityRejected, activity, presentation),
    )

    for factory, command, wrong_command in cases:
        rejected = factory(command=command, error=error)
        assert getattr(rejected, "command") is command
        with pytest.raises(InputValidationError) as mismatch:
            factory(command=wrong_command, error=error)
        assert mismatch.value.path == "result.command"
    assert not hasattr(interaction_api, "InteractionStoreRejected")


def test_due_batch_uses_one_observation_and_exact_bound() -> None:
    """Settle a bounded batch from one shared trusted observation."""
    first = _pending_record()
    second = _pending_record_on_branch("second-request", "branch-1")
    command = TerminalizeDueInteractionsCommand(maximum_results=1)
    observed_at = _observation(86_400)
    batch = apply_due_interactions(
        (first, second),
        command,
        observed_at,
        _POLICY,
    )

    assert isinstance(batch, DueInteractionsApplied)
    assert batch.store_mutation_applied
    assert len(batch.records) == 1
    resolution = batch.records[0].request.resolution
    assert isinstance(resolution, ExpiredResolution)
    assert resolution.resolved_at == observed_at.wall_time
    with pytest.raises(InputValidationError):
        DueInteractionsApplied(
            command=command,
            previous=(first, second),
            records=batch.records,
            observed_at=_observation(86_401),
            policy=_POLICY,
        )


def test_next_deadline_helper_uses_lease_advisory_and_absolute_schedule() -> (
    None
):
    """Select one deterministic next wake from a complete snapshot."""
    presented = _with_request_identity(
        _presented_advisory_record(),
        "presented-deadline",
    )
    lease_base = _with_request_identity(
        _presented_advisory_record(),
        "leased-deadline",
    )
    acquisition = apply_controller_activity(
        lease_base,
        _controller_command(
            lease_base,
            AcquireControllerActivity(
                request_id=lease_base.request.request_id,
                controller_id=ControllerId("deadline-controller"),
            ),
        ),
        _observation(25),
        _POLICY,
        lease_nonce=ActiveControlLeaseNonce("deadline-lease"),
    )
    assert isinstance(acquisition, ControllerActivityApplied)

    snapshot = select_next_interaction_deadline(
        (_terminal_record(), presented, acquisition.record),
        _observation(30),
        DeadlineScheduleRevision(7),
    )

    assert snapshot.schedule_revision == 7
    assert snapshot.deadline == InteractionDeadline(
        request_id=acquisition.record.request.request_id,
        monotonic_deadline=55,
    )
    terminal_only = select_next_interaction_deadline(
        (_terminal_record(),),
        _observation(30),
        DeadlineScheduleRevision(8),
    )
    assert terminal_only.deadline is None


def test_detached_presentation_is_a_public_metadata_mutation() -> None:
    """Represent detached handling without inventing a lifecycle state."""
    previous = _queued_advisory_record()
    command = DetachInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=previous.correlation,
        expected_store_revision=previous.store_revision,
    )
    detached = replace(
        previous,
        presentation=InteractionPresentationState.DETACHED,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
    )
    applied = InteractionPresentationApplied(
        command=command,
        record=detached,
        previous=previous,
        observed_at=_observation(5),
        policy=_POLICY,
    )

    assert command.expected_store_revision == previous.store_revision
    assert applied.record.request.advisory_deadline is None
    assert applied.presentation is InteractionPresentationState.DETACHED
    assert "mark_detached" in InteractionStore.__dict__


def test_detachment_rejects_observations_before_request_creation() -> None:
    """Apply the creation-time lower bound to required and advisory queues."""
    required = replace(
        _pending_record(),
        presentation=InteractionPresentationState.QUEUED,
    )
    for previous in (required, _queued_advisory_record()):
        command = _detachment_command(previous)
        with pytest.raises(InputValidationError) as predates:
            apply_interaction_detachment(
                previous,
                command,
                _observation(0, wall_seconds=-1),
                _POLICY,
            )
        assert predates.value.path == "observed_at.wall_time"
        at_creation = apply_interaction_detachment(
            previous,
            command,
            _observation(0, wall_seconds=0),
            _POLICY,
        )
        assert isinstance(at_creation, InteractionPresentationApplied)
        assert (
            at_creation.record.presentation
            is InteractionPresentationState.DETACHED
        )


@pytest.mark.parametrize(
    ("operation", "initial_presentation"),
    (
        ("present", InteractionPresentationState.QUEUED),
        ("detach", InteractionPresentationState.QUEUED),
        ("detach", InteractionPresentationState.PRESENTED),
    ),
)
def test_presentation_operations_settle_absolute_expiry_before_mutation(
    operation: str,
    initial_presentation: InteractionPresentationState,
) -> None:
    """Make absolute-expiry equality win queued and presented operations."""
    previous = replace(
        _pending_record(),
        presentation=initial_presentation,
    )
    before: (
        InteractionPresentationApplied
        | ResolveInteractionApplied
        | ControllerLeaseExpiredApplied
    )
    if operation == "present":
        command: PresentInteractionCommand | DetachInteractionCommand = (
            _presentation_command(previous)
        )
        before = apply_interaction_presentation(
            previous,
            cast(PresentInteractionCommand, command),
            _observation(86_399),
            _POLICY,
        )
    else:
        command = _detachment_command(previous)
        before = apply_interaction_detachment(
            previous,
            command,
            _observation(86_399),
            _POLICY,
        )
    assert isinstance(before, InteractionPresentationApplied)
    assert before.command == command

    for second in (86_400, 86_401):
        result: (
            InteractionPresentationApplied
            | ResolveInteractionApplied
            | ControllerLeaseExpiredApplied
        )
        if operation == "present":
            result = apply_interaction_presentation(
                previous,
                cast(PresentInteractionCommand, command),
                _observation(second),
                _POLICY,
            )
        else:
            result = apply_interaction_detachment(
                previous,
                cast(DetachInteractionCommand, command),
                _observation(second),
                _POLICY,
            )
        assert isinstance(result, ResolveInteractionApplied)
        assert result.command == command
        resolution = result.record.request.resolution
        assert isinstance(resolution, ExpiredResolution)
        assert resolution.resolved_at == _NOW + timedelta(seconds=second)
    with pytest.raises(InputValidationError):
        InteractionPresentationApplied(
            command=command,
            record=before.record,
            previous=previous,
            observed_at=_observation(86_400),
            policy=_POLICY,
        )


def test_presentation_commands_bind_correlation_revision_and_actor() -> None:
    """Reject presentation results detached from exact command authority."""
    previous = replace(
        _pending_record(),
        presentation=InteractionPresentationState.QUEUED,
    )
    valid = _presentation_command(previous)
    applied = apply_interaction_presentation(
        previous,
        valid,
        _observation(1),
        _POLICY,
    )
    assert isinstance(applied, InteractionPresentationApplied)

    stale = replace(
        valid,
        expected_store_revision=InteractionStoreRevision(
            previous.store_revision + 1
        ),
    )
    wrong_actor = replace(
        valid,
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("other-user"))
        ),
    )
    other = _pending_record_on_branch("other-request", "branch-1")
    wrong_correlation = replace(valid, correlation=other.correlation)
    for command in (stale, wrong_actor, wrong_correlation):
        with pytest.raises(InputValidationError):
            InteractionPresentationApplied(
                command=command,
                record=applied.record,
                previous=previous,
                observed_at=_observation(1),
                policy=_POLICY,
            )


def test_presentation_result_rejects_every_unrelated_record_mutation() -> None:
    """Constrain presentation to its exact lifecycle-neutral envelope."""
    previous = _queued_advisory_record()
    presented = _present_interaction_record(
        previous,
        _presentation_command(previous),
        _observation(5),
        _POLICY,
    )
    assert isinstance(presented, InteractionPresentationApplied)
    presented_record = presented.record

    changed_reason = replace(
        presented_record.request,
        reason="A different response is required.",
    )
    reason_record = replace(
        presented_record,
        request=changed_reason,
        semantic_fingerprint=semantic_request_fingerprint(changed_reason),
    )
    changed_origin = replace(
        presented_record.request.origin,
        branch_id=BranchId("other-branch"),
    )
    origin_request = replace(presented_record.request, origin=changed_origin)
    origin_record = replace(
        presented_record,
        request=origin_request,
        semantic_fingerprint=semantic_request_fingerprint(origin_request),
    )
    resolution = DeclinedResolution(
        request_id=presented_record.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=6),
    )
    transition = resolve_request(
        presented_record.request,
        resolution,
        expected_state_revision=presented_record.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    digest = canonical_resolution_digest(resolution)
    terminal_record = replace(
        presented_record,
        request=transition.request,
        semantic_fingerprint=semantic_request_fingerprint(transition.request),
        resolution_digest=digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("presentation-key"),
                resolution_digest=digest,
            ),
        ),
        resolved_by=_principal(),
    )
    skipped_revision = replace(
        presented_record,
        store_revision=InteractionStoreRevision(
            presented_record.store_revision + 1
        ),
    )

    for mutated in (
        reason_record,
        origin_record,
        terminal_record,
        skipped_revision,
    ):
        with pytest.raises(InputValidationError):
            InteractionPresentationApplied(
                command=_presentation_command(previous),
                record=mutated,
                previous=previous,
                observed_at=_observation(5),
                policy=_POLICY,
            )
    with pytest.raises(InputValidationError):
        replace(presented_record, semantic_fingerprint="0" * 64)


def test_detach_result_preserves_the_exact_queued_or_presented_record() -> (
    None
):
    """Allow detach to change only presentation and store revision."""
    for previous in (
        _queued_advisory_record(),
        _presented_advisory_record(),
    ):
        detached = replace(
            previous,
            presentation=InteractionPresentationState.DETACHED,
            store_revision=InteractionStoreRevision(
                previous.store_revision + 1
            ),
        )
        InteractionPresentationApplied(
            command=_detachment_command(previous),
            record=detached,
            previous=previous,
            observed_at=_observation(6),
            policy=_POLICY,
        )
        changed_request = replace(
            previous.request,
            reason="Detached after an unrelated mutation.",
        )
        changed = replace(
            detached,
            request=changed_request,
            semantic_fingerprint=semantic_request_fingerprint(changed_request),
        )

        with pytest.raises(InputValidationError):
            InteractionPresentationApplied(
                command=_detachment_command(previous),
                record=changed,
                previous=previous,
                observed_at=_observation(6),
                policy=_POLICY,
            )


def test_presentation_result_rejects_queued_as_an_applied_mutation() -> None:
    """Reserve queued state for admission rather than presentation."""
    previous = _queued_advisory_record()
    record = replace(
        previous,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
    )

    with pytest.raises(InputValidationError):
        InteractionPresentationApplied(
            command=_presentation_command(previous),
            record=record,
            previous=previous,
            observed_at=_observation(5),
            policy=_POLICY,
        )


def test_terminal_queued_or_presented_records_cannot_change_presentation() -> (
    None
):
    """Reject terminal records before any present or detach mutation."""
    queued = _queued_advisory_record()
    resolution = DeclinedResolution(
        request_id=queued.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )
    transition = resolve_request(
        queued.request,
        resolution,
        expected_state_revision=queued.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    digest = canonical_resolution_digest(resolution)
    terminal_queued = replace(
        queued,
        request=transition.request,
        resolution_digest=digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("terminal-queued-key"),
                resolution_digest=digest,
            ),
        ),
        resolved_by=_principal(),
    )

    for previous in (terminal_queued, _terminal_record()):
        with pytest.raises(InputValidationError):
            apply_interaction_presentation(
                previous,
                _presentation_command(previous),
                _observation(5),
                _POLICY,
            )
        with pytest.raises(InputValidationError):
            apply_interaction_detachment(
                previous,
                _detachment_command(previous),
                _observation(5),
                _POLICY,
            )


def test_acquire_returns_only_the_server_minted_lease_nonce() -> None:
    """Return a new nonce on acquire and omit it from later result shapes."""
    previous, acquired_record, evidence, nonce = (
        _acquired_controller_transition()
    )
    acquired_wait = acquired_record.advisory_wait
    assert acquired_wait is not None
    acquired = ControllerActivityApplied(
        command=_controller_command(previous, evidence),
        record=acquired_record,
        previous=previous,
        observed_at=_observation(10),
        policy=_POLICY,
        lease_nonce=nonce,
    )
    pulsed_record = replace(
        acquired_record,
        advisory_wait=replace(
            acquired_wait,
            activity_sequence=1,
            lease_expires_at_monotonic=40,
        ),
        store_revision=InteractionStoreRevision(
            acquired_record.store_revision + 1
        ),
    )
    pulse = PulseControllerActivity(
        request_id=previous.request.request_id,
        controller_id=ControllerId("controller-1"),
        lease_nonce=nonce,
        sequence=1,
    )
    pulsed = ControllerActivityApplied(
        command=_controller_command(acquired_record, pulse),
        record=pulsed_record,
        previous=acquired_record,
        observed_at=_observation(10),
        policy=_POLICY,
    )

    assert acquired.lease_nonce == nonce
    assert pulsed.lease_nonce is None
    with pytest.raises(InputValidationError):
        ControllerActivityApplied(
            command=_controller_command(previous, evidence),
            record=acquired_record,
            previous=previous,
            observed_at=_observation(11),
            policy=_POLICY,
            lease_nonce=nonce,
        )
    with pytest.raises(InputValidationError):
        ControllerActivityApplied(
            command=_controller_command(previous, evidence),
            record=acquired_record,
            previous=previous,
            observed_at=_observation(10),
            policy=_POLICY,
        )


def test_controller_results_bind_full_command_authority_and_evidence() -> None:
    """Reject controller results with altered actor, scope, or evidence."""
    previous, acquired_record, evidence, nonce = (
        _acquired_controller_transition()
    )
    command = _controller_command(previous, evidence)
    valid = ControllerActivityApplied(
        command=command,
        record=acquired_record,
        previous=previous,
        observed_at=_observation(10),
        policy=_POLICY,
        lease_nonce=nonce,
    )
    assert valid.command == command

    wrong_actor = replace(
        command,
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("other-user"))
        ),
    )
    wrong_correlation = replace(
        command,
        correlation=replace(
            command.correlation,
            run_id=RunId("other-run"),
        ),
    )
    wrong_evidence = replace(
        command,
        evidence=replace(
            evidence,
            controller_id=ControllerId("other-controller"),
        ),
    )
    for hostile in (wrong_actor, wrong_correlation, wrong_evidence):
        with pytest.raises(InputValidationError):
            ControllerActivityApplied(
                command=hostile,
                record=acquired_record,
                previous=previous,
                observed_at=_observation(10),
                policy=_POLICY,
                lease_nonce=nonce,
            )


def test_acquire_debits_exact_elapsed_budget_before_equal_after_due() -> None:
    """Debit running time and make deadline equality win acquisition."""
    previous = _presented_advisory_record()
    evidence = AcquireControllerActivity(
        request_id=previous.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    nonce = ActiveControlLeaseNonce("boundary-nonce")

    before = apply_controller_activity(
        previous,
        _controller_command(previous, evidence),
        _observation(64),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(before, ControllerActivityApplied)
    assert before.record.advisory_wait is not None
    assert before.record.advisory_wait.remaining_seconds == 1
    assert before.record.advisory_wait.lease_expires_at_monotonic == 94

    for second in (65, 66):
        due = apply_controller_activity(
            previous,
            _controller_command(previous, evidence),
            _observation(second),
            _POLICY,
            lease_nonce=nonce,
        )
        assert isinstance(due, ResolveInteractionApplied)
        assert due.decision_stage is ResolutionDecisionStage.DEADLINE
        assert due.record.request.state is RequestState.TIMED_OUT
        resolution = due.record.request.resolution
        assert isinstance(resolution, TimedOutResolution)
        assert resolution.resolved_at == _NOW + timedelta(seconds=second)


def test_controller_evidence_requires_strictly_unexpired_lease() -> None:
    """Make lease-expiry settlement win at equality and afterward."""
    presented = _presented_advisory_record()
    acquire = AcquireControllerActivity(
        request_id=presented.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    nonce = ActiveControlLeaseNonce("lease-boundary-nonce")
    acquired = apply_controller_activity(
        presented,
        _controller_command(presented, acquire),
        _observation(25),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(acquired, ControllerActivityApplied)
    wait = acquired.record.advisory_wait
    assert wait is not None
    assert wait.remaining_seconds == 40
    assert wait.lease_expires_at_monotonic == 55

    pulse = PulseControllerActivity(
        request_id=presented.request.request_id,
        controller_id=ControllerId("controller-1"),
        lease_nonce=nonce,
        sequence=1,
    )
    before = apply_controller_activity(
        acquired.record,
        _controller_command(acquired.record, pulse),
        _observation(54),
        _POLICY,
    )
    assert isinstance(before, ControllerActivityApplied)
    assert before.record.advisory_wait is not None
    assert before.record.advisory_wait.remaining_seconds == 40
    assert before.record.advisory_wait.lease_expires_at_monotonic == 84

    for second in (55, 56):
        expired = apply_controller_activity(
            acquired.record,
            _controller_command(acquired.record, pulse),
            _observation(second),
            _POLICY,
        )
        assert isinstance(expired, ControllerLeaseExpiredApplied)
        assert expired.record.advisory_wait is not None
        assert (
            expired.record.advisory_wait.status is AdvisoryWaitStatus.RUNNING
        )
        assert expired.record.advisory_wait.running_since_monotonic == 55
        assert expired.record.advisory_wait.remaining_seconds == 40


@pytest.mark.parametrize(
    "action_type",
    (ReleaseControllerActivity, DisconnectControllerActivity),
)
def test_release_and_disconnect_use_exact_trusted_now_before_expiry(
    action_type: type[
        ReleaseControllerActivity | DisconnectControllerActivity
    ],
) -> None:
    """Resume at trusted now only while the authenticated lease is active."""
    presented = _presented_advisory_record()
    nonce = ActiveControlLeaseNonce("resume-boundary-nonce")
    acquired = apply_controller_activity(
        presented,
        _controller_command(
            presented,
            AcquireControllerActivity(
                request_id=presented.request.request_id,
                controller_id=ControllerId("controller-1"),
            ),
        ),
        _observation(25),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(acquired, ControllerActivityApplied)
    evidence = action_type(
        request_id=presented.request.request_id,
        controller_id=ControllerId("controller-1"),
        lease_nonce=nonce,
        sequence=1,
    )

    before = apply_controller_activity(
        acquired.record,
        _controller_command(acquired.record, evidence),
        _observation(54),
        _POLICY,
    )
    assert isinstance(before, ControllerActivityApplied)
    assert before.record.advisory_wait is not None
    assert before.record.advisory_wait.running_since_monotonic == 54
    assert before.record.advisory_wait.remaining_seconds == 40
    for second in (55, 56):
        expired = apply_controller_activity(
            acquired.record,
            _controller_command(acquired.record, evidence),
            _observation(second),
            _POLICY,
        )
        assert isinstance(expired, ControllerLeaseExpiredApplied)


def test_operational_wait_prevents_twenty_second_early_timeout() -> None:
    """Ignore the original wall deadline after a legitimate pause."""
    presented = _presented_advisory_record()
    nonce = ActiveControlLeaseNonce("pause-nonce")
    acquired = apply_controller_activity(
        presented,
        _controller_command(
            presented,
            AcquireControllerActivity(
                request_id=presented.request.request_id,
                controller_id=ControllerId("controller-1"),
            ),
        ),
        _observation(25),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(acquired, ControllerActivityApplied)
    released = apply_controller_activity(
        acquired.record,
        _controller_command(
            acquired.record,
            ReleaseControllerActivity(
                request_id=presented.request.request_id,
                controller_id=ControllerId("controller-1"),
                lease_nonce=nonce,
                sequence=1,
            ),
        ),
        _observation(50),
        _POLICY,
    )
    assert isinstance(released, ControllerActivityApplied)
    assert released.record.request.advisory_deadline == _NOW + timedelta(
        seconds=65
    )

    early = apply_due_interaction(
        released.record,
        _observation(70),
        _POLICY,
    )
    assert early is None
    before = apply_due_interaction(
        released.record,
        _observation(89),
        _POLICY,
    )
    assert before is None
    for second in (90, 91):
        due = apply_due_interaction(
            released.record,
            _observation(second, wall_seconds=70 + second - 90),
            _POLICY,
        )
        assert isinstance(due, ResolveInteractionApplied)
        assert due.record.request.state is RequestState.TIMED_OUT
        resolution = due.record.request.resolution
        assert isinstance(resolution, TimedOutResolution)
        assert resolution.resolved_at == _NOW + timedelta(
            seconds=70 + second - 90
        )


def test_paused_budget_cannot_timeout_before_lease_and_effective_due() -> None:
    """Keep paused time free and settle lease expiry before later timeout."""
    presented = _presented_advisory_record()
    nonce = ActiveControlLeaseNonce("paused-nonce")
    acquired = apply_controller_activity(
        presented,
        _controller_command(
            presented,
            AcquireControllerActivity(
                request_id=presented.request.request_id,
                controller_id=ControllerId("controller-1"),
            ),
        ),
        _observation(25),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(acquired, ControllerActivityApplied)

    assert (
        apply_due_interaction(
            acquired.record,
            _observation(54, wall_seconds=100),
            _POLICY,
        )
        is None
    )
    at_lease = apply_due_interaction(
        acquired.record,
        _observation(55, wall_seconds=101),
        _POLICY,
    )
    assert isinstance(at_lease, ControllerLeaseExpiredApplied)
    assert at_lease.internal_authority is _DEADLINE_RESOLVER
    with pytest.raises(InputValidationError):
        replace(at_lease, internal_authority=None)
    at_due = apply_due_interaction(
        acquired.record,
        _observation(95, wall_seconds=141),
        _POLICY,
    )
    assert isinstance(at_due, ResolveInteractionApplied)
    assert at_due.record.request.state is RequestState.TIMED_OUT


def test_trusted_monotonic_time_cannot_precede_running_anchor() -> None:
    """Reject time travel instead of crediting advisory budget."""
    previous = _presented_advisory_record()
    with pytest.raises(InputValidationError) as error:
        apply_controller_activity(
            previous,
            _controller_command(
                previous,
                AcquireControllerActivity(
                    request_id=previous.request.request_id,
                    controller_id=ControllerId("controller-1"),
                ),
            ),
            _observation(4),
            _POLICY,
            lease_nonce=ActiveControlLeaseNonce("time-travel-nonce"),
        )

    assert error.value.path == "observed_at.monotonic_seconds"


def test_candidate_commit_uses_only_trusted_observation_time() -> None:
    """Ignore caller time and reject a mismatched result observation."""
    previous = _pending_record()
    command = ResolveInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("trusted-time-key"),
        proposed_resolution=DeclinedResolution(
            request_id=previous.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(days=10),
        ),
    )
    applied = apply_candidate_resolution(
        previous,
        command,
        _observation(7, wall_seconds=3),
        _POLICY,
    )
    assert isinstance(applied, ResolveInteractionApplied)
    resolution = applied.record.request.resolution
    assert isinstance(resolution, DeclinedResolution)
    assert resolution.resolved_at == _NOW + timedelta(seconds=3)

    with pytest.raises(InputValidationError):
        ResolveInteractionApplied(
            record=applied.record,
            previous=previous,
            decision_stage=ResolutionDecisionStage.COMMIT,
            observed_at=_observation(8, wall_seconds=4),
            policy=_POLICY,
            command=command,
            idempotency_key=command.idempotency_key,
        )


def test_release_and_disconnect_resume_the_exact_remaining_budget() -> None:
    """Resume a lease without refunding or spending advisory budget."""
    _, acquired_record, _, nonce = _acquired_controller_transition()
    acquired_wait = acquired_record.advisory_wait
    assert acquired_wait is not None
    for evidence in (
        ReleaseControllerActivity(
            request_id=acquired_record.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=nonce,
            sequence=1,
        ),
        DisconnectControllerActivity(
            request_id=acquired_record.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=nonce,
            sequence=1,
        ),
    ):
        resumed_wait = replace(
            acquired_wait,
            status=AdvisoryWaitStatus.RUNNING,
            running_since_monotonic=30,
            controller_id=None,
            lease_nonce=None,
            activity_sequence=None,
            lease_expires_at_monotonic=None,
        )
        resumed_record = replace(
            acquired_record,
            advisory_wait=resumed_wait,
            store_revision=InteractionStoreRevision(
                acquired_record.store_revision + 1
            ),
        )
        applied = ControllerActivityApplied(
            command=_controller_command(acquired_record, evidence),
            record=resumed_record,
            previous=acquired_record,
            observed_at=_observation(30),
            policy=_POLICY,
        )

        assert applied.lease_nonce is None
        assert applied.record.advisory_wait is not None
        assert (
            applied.record.advisory_wait.remaining_seconds
            == acquired_wait.remaining_seconds
        )
        assert applied.record.advisory_wait.controller_id is None


def test_controller_activity_requires_a_presented_advisory_lease() -> None:
    """Reject activity against required requests and unacquired leases."""
    required = _pending_record()
    required_record = replace(
        required,
        store_revision=InteractionStoreRevision(required.store_revision + 1),
    )
    acquire = AcquireControllerActivity(
        request_id=required.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    with pytest.raises(InputValidationError):
        ControllerActivityApplied(
            command=_controller_command(required, acquire),
            record=required_record,
            previous=required,
            observed_at=_observation(1),
            policy=_POLICY,
            lease_nonce=ActiveControlLeaseNonce("nonce"),
        )

    presented = _presented_advisory_record()
    unacquired_record = replace(
        presented,
        store_revision=InteractionStoreRevision(presented.store_revision + 1),
    )
    with pytest.raises(InputValidationError):
        pulse = PulseControllerActivity(
            request_id=presented.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("nonce"),
            sequence=1,
        )
        ControllerActivityApplied(
            command=_controller_command(presented, pulse),
            record=unacquired_record,
            previous=presented,
            observed_at=_observation(10),
            policy=_POLICY,
        )


def test_controller_activity_authenticates_identity_nonce_and_sequence() -> (
    None
):
    """Reject stale, skipped, or controller-swapped lease evidence."""
    _, acquired_record, _, nonce = _acquired_controller_transition()
    acquired_wait = acquired_record.advisory_wait
    assert acquired_wait is not None
    pulsed_wait = replace(
        acquired_wait,
        activity_sequence=1,
        lease_expires_at_monotonic=40,
    )
    pulsed_record = replace(
        acquired_record,
        advisory_wait=pulsed_wait,
        store_revision=InteractionStoreRevision(
            acquired_record.store_revision + 1
        ),
    )
    for evidence in (
        PulseControllerActivity(
            request_id=acquired_record.request.request_id,
            controller_id=ControllerId("other-controller"),
            lease_nonce=nonce,
            sequence=1,
        ),
        PulseControllerActivity(
            request_id=acquired_record.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("other-nonce"),
            sequence=1,
        ),
        PulseControllerActivity(
            request_id=acquired_record.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=nonce,
            sequence=2,
        ),
    ):
        with pytest.raises(InputValidationError):
            ControllerActivityApplied(
                command=_controller_command(acquired_record, evidence),
                record=pulsed_record,
                previous=acquired_record,
                observed_at=_observation(10),
                policy=_POLICY,
            )


def test_controller_pulse_preserves_budget_and_never_shortens_lease() -> None:
    """Limit pulse to the next sequence and a nondecreasing expiry."""
    _, acquired_record, _, nonce = _acquired_controller_transition()
    acquired_wait = acquired_record.advisory_wait
    assert acquired_wait is not None
    evidence = PulseControllerActivity(
        request_id=acquired_record.request.request_id,
        controller_id=ControllerId("controller-1"),
        lease_nonce=nonce,
        sequence=1,
    )
    changed_budget = replace(
        acquired_wait,
        remaining_seconds=54,
        activity_sequence=1,
        lease_expires_at_monotonic=40,
    )
    shortened_lease = replace(
        acquired_wait,
        activity_sequence=1,
        lease_expires_at_monotonic=34,
    )

    for wait in (changed_budget, shortened_lease):
        record = replace(
            acquired_record,
            advisory_wait=wait,
            store_revision=InteractionStoreRevision(
                acquired_record.store_revision + 1
            ),
        )
        with pytest.raises(InputValidationError):
            ControllerActivityApplied(
                command=_controller_command(acquired_record, evidence),
                record=record,
                previous=acquired_record,
                observed_at=_observation(10),
                policy=_POLICY,
            )


def test_release_must_clear_control_and_preserve_remaining_budget() -> None:
    """Reject release records that retain lease state or refund budget."""
    _, acquired_record, _, nonce = _acquired_controller_transition()
    acquired_wait = acquired_record.advisory_wait
    assert acquired_wait is not None
    evidence = ReleaseControllerActivity(
        request_id=acquired_record.request.request_id,
        controller_id=ControllerId("controller-1"),
        lease_nonce=nonce,
        sequence=1,
    )
    retained = replace(acquired_wait, activity_sequence=1)
    refunded = replace(
        acquired_wait,
        status=AdvisoryWaitStatus.RUNNING,
        remaining_seconds=56,
        running_since_monotonic=30,
        controller_id=None,
        lease_nonce=None,
        activity_sequence=None,
        lease_expires_at_monotonic=None,
    )

    for wait in (retained, refunded):
        record = replace(
            acquired_record,
            advisory_wait=wait,
            store_revision=InteractionStoreRevision(
                acquired_record.store_revision + 1
            ),
        )
        with pytest.raises(InputValidationError):
            ControllerActivityApplied(
                command=_controller_command(acquired_record, evidence),
                record=record,
                previous=acquired_record,
                observed_at=_observation(30),
                policy=_POLICY,
            )


def test_controller_result_rejects_unrelated_record_mutations() -> None:
    """Constrain activity results to advisory lease metadata only."""
    _, acquired_record, _, nonce = _acquired_controller_transition()
    acquired_wait = acquired_record.advisory_wait
    assert acquired_wait is not None
    evidence = PulseControllerActivity(
        request_id=acquired_record.request.request_id,
        controller_id=ControllerId("controller-1"),
        lease_nonce=nonce,
        sequence=1,
    )
    pulsed = replace(
        acquired_record,
        advisory_wait=replace(
            acquired_wait,
            activity_sequence=1,
            lease_expires_at_monotonic=40,
        ),
        store_revision=InteractionStoreRevision(
            acquired_record.store_revision + 1
        ),
    )
    changed_request = replace(
        pulsed.request,
        reason="A controller cannot change this reason.",
    )
    changed_reason = replace(
        pulsed,
        request=changed_request,
        semantic_fingerprint=semantic_request_fingerprint(changed_request),
    )
    changed_origin = replace(
        pulsed.request.origin,
        branch_id=BranchId("controller-swapped-branch"),
    )
    origin_request = replace(pulsed.request, origin=changed_origin)
    origin_record = replace(
        pulsed,
        request=origin_request,
        semantic_fingerprint=semantic_request_fingerprint(origin_request),
    )
    resolution = DeclinedResolution(
        request_id=pulsed.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=6),
    )
    transition = resolve_request(
        pulsed.request,
        resolution,
        expected_state_revision=pulsed.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    digest = canonical_resolution_digest(resolution)
    terminal_record = replace(
        pulsed,
        request=transition.request,
        semantic_fingerprint=semantic_request_fingerprint(transition.request),
        resolution_digest=digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("controller-key"),
                resolution_digest=digest,
            ),
        ),
        resolved_by=_principal(),
    )

    for mutated in (changed_reason, origin_record, terminal_record):
        with pytest.raises(InputValidationError):
            ControllerActivityApplied(
                command=_controller_command(acquired_record, evidence),
                record=mutated,
                previous=acquired_record,
                observed_at=_observation(10),
                policy=_POLICY,
            )
    with pytest.raises(InputValidationError):
        replace(pulsed, semantic_fingerprint="0" * 64)


@pytest.mark.parametrize(
    "evidence",
    (
        PulseControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=1,
        ),
        ReleaseControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=2,
        ),
        DisconnectControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=3,
        ),
    ),
)
def test_existing_controller_lease_actions_require_nonce_and_sequence(
    evidence: (
        PulseControllerActivity
        | ReleaseControllerActivity
        | DisconnectControllerActivity
    ),
) -> None:
    """Accept only authenticated sequenced evidence after acquisition."""
    pending = _pending()
    command = RecordControllerActivityCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=InteractionCorrelation.from_request(pending),
        evidence=evidence,
    )

    assert command.evidence is evidence
    assert evidence.lease_nonce == "lease-1"
    assert evidence.sequence >= 1


def test_authorized_projection_never_leaks_terminal_content() -> None:
    """Project content-free metadata for a bound non-owner authorization."""
    record = _terminal_record()
    actor = InteractionActor(
        principal=PrincipalScope(user_id=UserId("auditor"))
    )
    target = InteractionRequestAuthorizationTarget(
        request_id=record.request.request_id,
        origin=record.request.origin,
    )
    metadata_decision = InteractionAuthorizationDecision(
        actor=actor,
        operation=InteractionOperation.INSPECT,
        target=target,
        allowed=True,
        disclosure=InteractionDisclosure.TERMINAL_METADATA,
    )
    full_decision = replace(
        metadata_decision,
        disclosure=InteractionDisclosure.FULL,
    )
    owner_full_decision = replace(
        full_decision,
        actor=InteractionActor(principal=_principal()),
    )
    denied = replace(
        metadata_decision,
        allowed=False,
        disclosure=InteractionDisclosure.NONE,
    )

    metadata = project_authorized_interaction(record, metadata_decision)
    assert isinstance(metadata, InteractionTerminalMetadata)
    assert metadata.status is ResolutionStatus.DECLINED
    assert metadata.resolved_at == _NOW + timedelta(seconds=1)
    assert tuple(field.name for field in fields(metadata)) == (
        "status",
        "resolved_at",
    )
    for leaked in (
        "correlation",
        "state",
        "state_revision",
        "request_id",
        "continuation_id",
        "run_id",
        "turn_id",
        "task_id",
        "agent_id",
        "branch_id",
        "model_call_id",
        "resolved_by",
        "resolution_digest",
        "resolution",
        "answers",
    ):
        assert not hasattr(metadata, leaked)
    capped = project_authorized_interaction(record, full_decision)
    assert isinstance(capped, InteractionTerminalMetadata)
    assert tuple(field.name for field in fields(capped)) == (
        "status",
        "resolved_at",
    )
    assert (
        project_authorized_interaction(
            record,
            owner_full_decision,
        )
        is record
    )
    with pytest.raises(InputValidationError) as forbidden:
        project_authorized_interaction(record, denied)
    assert forbidden.value.code is InputErrorCode.FORBIDDEN


def test_advisory_record_matrix_rejects_crossed_timing_states() -> None:
    """Keep queued, presented, detached, and exhausted metadata coherent."""
    queued = _queued_advisory_record()
    presented = _presented_advisory_record()

    with pytest.raises(InputValidationError):
        replace(queued, presentation=InteractionPresentationState.PRESENTED)
    with pytest.raises(InputValidationError):
        replace(presented, presentation=InteractionPresentationState.QUEUED)
    assert presented.advisory_wait is not None
    with pytest.raises(InputValidationError):
        replace(
            presented,
            advisory_wait=replace(
                presented.advisory_wait,
                status=AdvisoryWaitStatus.EXHAUSTED,
                remaining_seconds=0,
                running_since_monotonic=None,
            ),
        )


def test_queued_advisory_can_terminalize_before_presentation() -> None:
    """Allow non-timeout settlement while advisory presentation is queued."""
    previous = _queued_advisory_record()
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
    record = InteractionRecord(
        request=transition.request,
        semantic_fingerprint=previous.semantic_fingerprint,
        absolute_expires_at=previous.absolute_expires_at,
        presentation=InteractionPresentationState.QUEUED,
        store_revision=InteractionStoreRevision(previous.store_revision + 1),
        advisory_wait=previous.advisory_wait,
        resolution_digest=digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("queued-key"),
                resolution_digest=digest,
            ),
        ),
        resolved_by=_principal(),
    )

    assert record.request.state is RequestState.DECLINED
    assert record.request.advisory_deadline is None
    assert record.presentation is InteractionPresentationState.QUEUED


def test_closed_handle_error_and_protocol_semantics_are_explicit() -> None:
    """Freeze post-close, waiter, linearization, and reopen guarantees."""
    error = InteractionStoreClosedError()
    missing = InteractionNotFoundError()
    unauthorized = InteractionNotFoundError()
    wait_doc = InteractionStore.wait_for_change.__doc__ or ""
    deadline_wait_doc = InteractionStore.wait_for_deadline_change.__doc__ or ""
    close_doc = InteractionStore.aclose.__doc__ or ""

    assert error.code is InputErrorCode.STORE_CLOSED
    assert error.path == "store.handle"
    assert "InteractionStoreClosedError" in wait_doc
    assert "InteractionNotFoundError" in wait_doc
    assert "InteractionStoreClosedError" in deadline_wait_doc
    assert "atomic commit before close" in close_doc
    assert "separately reopened handle" in close_doc
    assert "backing-state changes may still wake" in close_doc
    assert "open" in InteractionStoreFactory.__dict__
    assert missing.code is unauthorized.code is InputErrorCode.NOT_FOUND
    assert missing.path == unauthorized.path == "interaction"
    assert str(missing) == str(unauthorized)
    assert get_type_hints(InteractionStore.lookup_scoped)["return"] == (
        InteractionDisclosureProjection | None
    )
    assert get_type_hints(InteractionStore.lookup_branch_root)["return"] == (
        InteractionBranchRoot | None
    )
    assert (
        get_type_hints(InteractionStore.list_scoped)["return"]
        == tuple[InteractionDisclosureProjection, ...]
    )
    assert get_type_hints(InteractionStore.wait_for_change)["return"] == (
        InteractionDisclosureProjection
    )


def test_admission_cleanup_capability_is_sealed_private_and_content_free() -> (
    None
):
    """Keep cleanup authority opaque and absent from the public surface."""
    request = _created()
    actor = InteractionActor(principal=request.origin.principal)
    resumer = _RecordingResumer()
    create, cleanup = _new_interaction_admission_commands(
        actor=actor,
        request=request,
        resumer=resumer,
    )

    assert type(create) is _InteractionAdmissionCreateCommand
    assert type(cleanup) is _InteractionAdmissionCleanupCommand
    assert create._command == CreateInteractionCommand(
        actor=actor,
        request=request,
        resumer=resumer,
    )
    assert create._capability is cleanup._capability
    assert request.reason not in repr(create)
    assert str(request.request_id) not in repr(create)
    assert str(request.request_id) not in repr(cleanup)
    assert "_RecordingResumer" not in repr(create)
    for public_name in (
        "InteractionAdmissionCapability",
        "InteractionAdmissionCleanupCommand",
        "InteractionAdmissionCleanupResult",
        "InteractionAdmissionCreateCommand",
    ):
        assert not hasattr(interaction_api, public_name)

    with pytest.raises(InputValidationError):
        interaction_store._InteractionAdmissionCapability(_token=object())
    with pytest.raises(InputValidationError):
        _InteractionAdmissionCreateCommand(
            command=create._command,
            capability=create._capability,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _InteractionAdmissionCleanupCommand(
            capability=create._capability,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _InteractionAdmissionCleanupResult(
            disposition=_InteractionAdmissionCleanupDisposition.ABSENT,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _new_interaction_admission_cleanup_result(
            cast(_InteractionAdmissionCleanupDisposition, "absent")
        )
    proof = _new_interaction_admission_cleanup_result(
        _InteractionAdmissionCleanupDisposition.ABSENT
    )

    unsealed_capability = object.__new__(
        interaction_store._InteractionAdmissionCapability
    )
    with pytest.raises(InputValidationError) as capability_type_error:
        interaction_store._validate_interaction_admission_capability(
            object(),
            "admission.test.capability",
        )
    assert capability_type_error.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as capability_error:
        interaction_store._validate_interaction_admission_capability(
            unsealed_capability,
            "admission.test.capability",
        )
    assert capability_error.value.code is InputErrorCode.FORBIDDEN
    assert capability_error.value.path == "admission.test.capability"

    unsealed_create = object.__new__(_InteractionAdmissionCreateCommand)
    with pytest.raises(InputValidationError) as create_error:
        interaction_store._validate_interaction_admission_create_command(
            unsealed_create
        )
    assert create_error.value.code is InputErrorCode.FORBIDDEN

    nested_create = object.__new__(_InteractionAdmissionCreateCommand)
    object.__setattr__(nested_create, "_seal", create._seal)
    object.__setattr__(nested_create, "_command", create._command)
    object.__setattr__(
        nested_create,
        "_capability",
        unsealed_capability,
    )
    with pytest.raises(InputValidationError) as nested_create_error:
        interaction_store._validate_interaction_admission_create_command(
            nested_create
        )
    assert nested_create_error.value.path == "admission.create.capability"

    missing_create_command = object.__new__(_InteractionAdmissionCreateCommand)
    object.__setattr__(missing_create_command, "_seal", create._seal)
    object.__setattr__(
        missing_create_command,
        "_capability",
        create._capability,
    )
    with pytest.raises(InputValidationError) as missing_command_error:
        interaction_store._validate_interaction_admission_create_command(
            missing_create_command
        )
    assert missing_command_error.value.path == "admission.create.command"

    unbound_create = object.__new__(_InteractionAdmissionCreateCommand)
    object.__setattr__(unbound_create, "_seal", create._seal)
    object.__setattr__(
        unbound_create,
        "_capability",
        create._capability,
    )
    object.__setattr__(
        unbound_create,
        "_command",
        CreateInteractionCommand(actor=actor, request=request),
    )
    with pytest.raises(InputValidationError) as unbound_create_error:
        interaction_store._validate_interaction_admission_create_command(
            unbound_create
        )
    assert (
        unbound_create_error.value.path == "admission.create.command.resumer"
    )

    unsealed_cleanup = object.__new__(_InteractionAdmissionCleanupCommand)
    with pytest.raises(InputValidationError) as cleanup_error:
        interaction_store._validate_interaction_admission_cleanup_command(
            unsealed_cleanup
        )
    assert cleanup_error.value.code is InputErrorCode.FORBIDDEN

    nested_cleanup = object.__new__(_InteractionAdmissionCleanupCommand)
    object.__setattr__(nested_cleanup, "_seal", cleanup._seal)
    object.__setattr__(
        nested_cleanup,
        "_capability",
        unsealed_capability,
    )
    with pytest.raises(InputValidationError) as nested_cleanup_error:
        interaction_store._validate_interaction_admission_cleanup_command(
            nested_cleanup
        )
    assert nested_cleanup_error.value.path == "admission.cleanup.capability"

    unsealed_proof = object.__new__(_InteractionAdmissionCleanupResult)
    with pytest.raises(InputValidationError) as proof_type_error:
        interaction_store._validate_interaction_admission_cleanup_result(
            object()
        )
    assert proof_type_error.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as proof_error:
        interaction_store._validate_interaction_admission_cleanup_result(
            unsealed_proof
        )
    assert proof_error.value.code is InputErrorCode.FORBIDDEN

    malformed_proof = object.__new__(_InteractionAdmissionCleanupResult)
    object.__setattr__(malformed_proof, "_seal", proof._seal)
    object.__setattr__(malformed_proof, "disposition", "absent")
    with pytest.raises(InputValidationError) as disposition_error:
        interaction_store._validate_interaction_admission_cleanup_result(
            malformed_proof
        )
    assert (
        disposition_error.value.path == "admission.cleanup_result.disposition"
    )

    assert (
        interaction_store._validate_interaction_admission_capability(
            create._capability,
            "admission.test.capability",
        )
        is create._capability
    )
    assert (
        interaction_store._validate_interaction_admission_create_command(
            create
        )
        is create
    )
    assert (
        interaction_store._validate_interaction_admission_cleanup_command(
            cleanup
        )
        is cleanup
    )
    assert (
        interaction_store._validate_interaction_admission_cleanup_result(proof)
        is proof
    )
    assert proof.disposition is _InteractionAdmissionCleanupDisposition.ABSENT
    assert str(request.request_id) not in repr(proof)


def test_admission_cleanup_preserves_exact_temporal_precedence() -> None:
    """Let equal deadlines and lease expiry win before unavailable cleanup."""
    pending = _pending_record()
    unavailable = interaction_store._reduce_interaction_admission_cleanup(
        pending,
        _observation(1),
        _POLICY,
    )
    assert unavailable.request.state is RequestState.UNAVAILABLE
    assert isinstance(unavailable.request.resolution, UnavailableResolution)
    assert unavailable.request.resolution.provenance is AnswerProvenance.POLICY
    assert unavailable.resolved_by is _ADMISSION_CLEANUP_RESOLVER
    assert unavailable.request.resolution is not None
    forged_resolution = replace(
        unavailable.request.resolution,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    )
    with pytest.raises(InputValidationError):
        replace(
            unavailable,
            request=replace(
                unavailable.request,
                resolution=forged_resolution,
            ),
            resolution_digest=canonical_resolution_digest(forged_resolution),
        )

    expired = interaction_store._reduce_interaction_admission_cleanup(
        pending,
        _observation(86_400),
        _POLICY,
    )
    assert expired.request.state is RequestState.EXPIRED
    assert isinstance(expired.request.resolution, ExpiredResolution)
    assert expired.resolved_by is _DEADLINE_RESOLVER

    advisory = _presented_advisory_record()
    timed_out = interaction_store._reduce_interaction_admission_cleanup(
        advisory,
        _observation(65),
        _POLICY,
    )
    assert timed_out.request.state is RequestState.TIMED_OUT
    assert isinstance(timed_out.request.resolution, TimedOutResolution)
    assert timed_out.resolved_by is _DEADLINE_RESOLVER

    _, acquired, _, _ = _acquired_controller_transition()
    after_lease = interaction_store._reduce_interaction_admission_cleanup(
        acquired,
        _observation(40),
        _POLICY,
    )
    assert after_lease.request.state is RequestState.UNAVAILABLE
    assert after_lease.store_revision == acquired.store_revision + 2
    assert after_lease.advisory_wait is not None
    assert after_lease.advisory_wait.status is AdvisoryWaitStatus.RUNNING
    assert after_lease.advisory_wait.controller_id is None
    assert after_lease.resolved_by is _ADMISSION_CLEANUP_RESOLVER


def test_store_value_objects_and_public_helpers_fail_closed() -> None:
    """Exercise malformed store values without weakening their contracts."""
    actor = InteractionActor(principal=_principal())
    pending = _pending_record()
    terminal = _terminal_record()

    with pytest.raises(InputValidationError):
        interaction_store._InteractionSystemResolver(_token=object())
    correlation_with_task = replace(
        pending.correlation,
        task_id=cast(interaction_api.TaskId, "task-1"),
    )
    assert correlation_with_task.task_id == "task-1"
    with pytest.raises(InputValidationError):
        InteractionCorrelation.from_request(cast(InputRequest, object()))
    with pytest.raises(InputValidationError):
        InteractionExecutionScope(
            run_id=RunId("run-1"),
            include_descendants=True,
        )
    with pytest.raises(InputValidationError):
        InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("same"),
            parent_branch_id=BranchId("same"),
            principal=_principal(),
        )
    with pytest.raises(InputValidationError):
        InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("child"),
            parent_branch_id=BranchId("parent"),
            principal=cast(PrincipalScope, object()),
        )
    with pytest.raises(InputValidationError):
        RegisterInteractionBranchCommand(
            actor=actor,
            registration=cast(InteractionBranchRegistration, object()),
        )
    with pytest.raises(InputValidationError):
        InteractionBranchRecord(
            registration=cast(InteractionBranchRegistration, object()),
            store_revision=InteractionStoreRevision(1),
        )
    with pytest.raises(InputValidationError):
        InteractionBranchRoot(
            run_id=RunId("run-1"),
            branch_id=BranchId("same"),
            root_branch_id=BranchId("same"),
        )
    lookup = InteractionBranchRootLookup(
        actor=actor,
        run_id=RunId("run-1"),
        branch_id=BranchId("child"),
    )
    with pytest.raises(InputValidationError):
        _resolve_interaction_branch_root(
            cast(tuple[InteractionBranchRecord, ...], []),
            lookup,
        )
    with pytest.raises(InputValidationError):
        _resolve_interaction_branch_root(
            (),
            cast(InteractionBranchRootLookup, object()),
        )

    with pytest.raises(InputValidationError):
        AdvisoryWaitState(
            status=cast(AdvisoryWaitStatus, "invalid"),
            budget_seconds=60,
            remaining_seconds=60,
        )
    with pytest.raises(InputValidationError):
        AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=60,
            remaining_seconds=61,
        )
    with pytest.raises(InputValidationError):
        AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=60,
            remaining_seconds=60,
            running_since_monotonic=-1,
        )
    with pytest.raises(InputValidationError):
        AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=60,
            remaining_seconds=60,
            lease_expires_at_monotonic=-1,
        )

    with pytest.raises(InputValidationError):
        replace(pending, request=cast(InputRequest, object()))
    with pytest.raises(InputValidationError):
        replace(
            pending,
            absolute_expires_at=pending.absolute_expires_at
            + timedelta(seconds=1),
        )
    with pytest.raises(InputValidationError):
        replace(
            pending,
            presentation=cast(InteractionPresentationState, "invalid"),
        )
    with pytest.raises(InputValidationError):
        InteractionRecord(
            request=_created(),
            semantic_fingerprint=semantic_request_fingerprint(_created()),
            absolute_expires_at=_NOW + timedelta(days=1),
            presentation=InteractionPresentationState.QUEUED,
            store_revision=InteractionStoreRevision(0),
        )
    with pytest.raises(InputValidationError):
        InteractionTerminalMetadata(
            status=cast(ResolutionStatus, "invalid"),
            resolved_at=_NOW,
        )

    target = InteractionRequestAuthorizationTarget(
        request_id=terminal.request.request_id,
        origin=terminal.request.origin,
    )
    decision = InteractionAuthorizationDecision(
        actor=actor,
        operation=InteractionOperation.INSPECT,
        target=target,
        allowed=True,
        disclosure=InteractionDisclosure.TERMINAL_METADATA,
    )
    with pytest.raises(InputValidationError):
        project_authorized_interaction(
            terminal,
            cast(InteractionAuthorizationDecision, object()),
        )
    scope_decision = InteractionAuthorizationDecision(
        actor=actor,
        operation=InteractionOperation.LIST,
        target=InteractionScopeAuthorizationTarget(
            run_id=RunId("run-1"),
            principal=_principal(),
        ),
        allowed=True,
        disclosure=InteractionDisclosure.TERMINAL_METADATA,
    )
    with pytest.raises(InputValidationError):
        project_authorized_interaction(terminal, scope_decision)
    forbidden_operation = replace(decision)
    object.__setattr__(
        forbidden_operation,
        "operation",
        InteractionOperation.RESOLVE,
    )
    with pytest.raises(InputValidationError):
        project_authorized_interaction(terminal, forbidden_operation)
    mismatched = replace(
        decision,
        target=InteractionRequestAuthorizationTarget(
            request_id=InputRequestId("other-request"),
            origin=terminal.request.origin,
        ),
    )
    with pytest.raises(InputValidationError):
        project_authorized_interaction(terminal, mismatched)
    pending_decision = replace(
        decision,
        target=InteractionRequestAuthorizationTarget(
            request_id=pending.request.request_id,
            origin=pending.request.origin,
        ),
    )
    with pytest.raises(InputValidationError):
        project_authorized_interaction(pending, pending_decision)

    candidate = _resolve_command(pending, "new-key")
    assert (
        evaluate_resolution_idempotency(
            pending,
            key=candidate.idempotency_key,
            resolution_digest=candidate.resolution_digest,
        )
        is ResolutionIdempotencyDisposition.NEW_RESOLUTION
    )
    assert (
        evaluate_resolution_idempotency(
            terminal,
            key=ResolutionIdempotencyKey("different-key"),
            resolution_digest="0" * 64,
        )
        is ResolutionIdempotencyDisposition.TERMINAL_CONFLICT
    )


def test_store_admission_commands_and_deadlines_reject_bad_boundaries() -> (
    None
):
    """Cover capacity, command, and scheduler boundary validation."""
    actor = InteractionActor(principal=_principal())
    pending = _pending_record()
    create = CreateInteractionCommand(actor=actor, request=_created())
    assert create.resumer_registration is None
    with pytest.raises(InputValidationError):
        validate_interaction_admission(
            cast(tuple[InteractionRecord, ...], []),
            create,
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        validate_interaction_admission(
            (),
            cast(CreateInteractionCommand, object()),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        validate_interaction_admission((pending, pending), create, _POLICY)
    with pytest.raises(InputValidationError):
        validate_interaction_admission((pending,), create, _POLICY)

    other_created = replace(
        _created(),
        request_id=InputRequestId("request-2"),
        continuation_id=ContinuationId("continuation-2"),
    )
    other_create = CreateInteractionCommand(actor=actor, request=other_created)
    for policy in (
        replace(_POLICY, maximum_pending_interactions_per_process=1),
        replace(_POLICY, maximum_unresolved_interactions_per_run=1),
        replace(
            _POLICY,
            maximum_unresolved_required_interactions_per_branch=1,
        ),
    ):
        with pytest.raises(InputValidationError):
            validate_interaction_admission((pending,), other_create, policy)

    with pytest.raises(InputValidationError):
        CreateInteractionCommand(
            actor=actor,
            request=cast(InputRequest, object()),
        )
    with pytest.raises(InputValidationError):
        CreateInteractionCommand(actor=actor, request=_pending())
    lookup = interaction_store.ScopedInteractionLookup(
        actor=actor,
        correlation=pending.correlation,
    )
    assert lookup.correlation == pending.correlation
    with pytest.raises(InputValidationError):
        ListInteractionsCommand(
            actor=actor,
            scope=cast(InteractionExecutionScope, object()),
        )

    mismatched_resolution = DeclinedResolution(
        request_id=InputRequestId("other-request"),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    with pytest.raises(InputValidationError):
        _resolve_command(
            pending,
            "mismatch",
            resolution=mismatched_resolution,
        )
    with pytest.raises(TypeError):
        TrustedDefaultResolutionRequest(
            actor=actor,
            correlation=pending.correlation,
            expected_state_revision=pending.request.state_revision,
            proposed_resolution=cast(
                InputCandidateResolution,
                CancelledResolution(
                    request_id=pending.request.request_id,
                    provenance=AnswerProvenance.POLICY,
                    resolved_at=_NOW,
                    scope=CancellationScope.REQUEST,
                ),
            ),
            # The public trigger deliberately carries no resolution.
            # type: ignore[call-arg]
        )
    trusted_command = _trusted_default_command(
        actor=actor,
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
    )
    assert not hasattr(trusted_command, "proposed_resolution")

    with pytest.raises(InputValidationError):
        CancelInteractionCommand(
            actor=actor,
            correlation=pending.correlation,
            provenance=cast(AnswerProvenance, "invalid"),  # type: ignore[arg-type]
        )
    scope = InteractionExecutionScope(run_id=RunId("run-1"))
    with pytest.raises(InputValidationError):
        TerminalizeInteractionScopeCommand(
            actor=actor,
            scope=cast(InteractionExecutionScope, object()),
            provenance=AnswerProvenance.HUMAN,
        )
    with pytest.raises(InputValidationError):
        TerminalizeInteractionScopeCommand(
            actor=actor,
            scope=scope,
            provenance=cast(AnswerProvenance, "invalid"),  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        SupersedeInteractionScopeCommand(
            actor=actor,
            scope=cast(InteractionExecutionScope, object()),
            provenance=AnswerProvenance.HUMAN,
        )
    with pytest.raises(InputValidationError):
        SupersedeInteractionScopeCommand(
            actor=actor,
            scope=scope,
            provenance=cast(AnswerProvenance, "invalid"),  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        RecordControllerActivityCommand(
            actor=actor,
            correlation=pending.correlation,
            evidence=cast(AcquireControllerActivity, object()),
        )
    with pytest.raises(InputValidationError):
        RecordControllerActivityCommand(
            actor=actor,
            correlation=pending.correlation,
            evidence=AcquireControllerActivity(
                request_id=InputRequestId("other-request"),
                controller_id=ControllerId("controller-1"),
            ),
        )

    with pytest.raises(InputValidationError):
        InteractionDeadline(
            request_id=pending.request.request_id,
            monotonic_deadline=-1,
        )
    with pytest.raises(InputValidationError):
        InteractionDeadlineSnapshot(
            schedule_revision=DeadlineScheduleRevision(1),
            deadline=cast(InteractionDeadline, object()),
        )
    with pytest.raises(InputValidationError):
        select_next_interaction_deadline(
            cast(tuple[InteractionRecord, ...], []),
            _observation(0),
            DeadlineScheduleRevision(1),
        )
    with pytest.raises(InputValidationError):
        select_next_interaction_deadline(
            (),
            cast(InteractionTime, object()),
            DeadlineScheduleRevision(1),
        )
    with pytest.raises(InputValidationError):
        select_next_interaction_deadline(
            (pending, pending),
            _observation(0),
            DeadlineScheduleRevision(1),
        )


def test_store_result_objects_reject_unbound_or_forged_evidence() -> None:
    """Cover operation-specific result seals and exact evidence bindings."""
    actor = InteractionActor(principal=_principal())
    pending = _pending_record()
    resolve = _resolve_command(pending, "result-key")
    error = InputTransitionError(
        code=InputErrorCode.INVALID_TYPE,
        path="command",
        message="rejected",
    )

    with pytest.raises(InputValidationError):
        ResolveInteractionCommand(
            actor=actor,
            correlation=pending.correlation,
            expected_state_revision=pending.request.state_revision,
            idempotency_key=ResolutionIdempotencyKey("trusted-answer"),
            proposed_resolution=AnsweredResolution(
                request_id=pending.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
                answers=(
                    TextAnswer(
                        question_id=QuestionId("text"),
                        provenance=AnswerProvenance.TRUSTED_DEFAULT,
                        value="trusted",
                    ),
                ),
            ),
        )
    with pytest.raises(TypeError):
        TrustedDefaultResolutionRequest(
            actor=actor,
            correlation=pending.correlation,
            expected_state_revision=pending.request.state_revision,
            proposed_resolution=AnsweredResolution(
                request_id=pending.request.request_id,
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                resolved_at=_NOW,
                answers=(
                    TextAnswer(
                        question_id=QuestionId("text"),
                        provenance=AnswerProvenance.HUMAN,
                        value="untrusted",
                    ),
                ),
            ),
            # type: ignore[call-arg]
        )

    with pytest.raises(InputValidationError):
        CreateInteractionApplied(
            command=cast(CreateInteractionCommand, object()),
            record=pending,
            policy=_POLICY,
        )
    with pytest.raises(InputValidationError):
        CancelInteractionApplied(
            command=cast(CancelInteractionCommand, object()),
            previous=pending,
            record=pending,
            observed_at=_observation(1),
            policy=_POLICY,
        )
    with pytest.raises(InputValidationError):
        TerminalizeInteractionApplied(
            command=cast(TerminalizeInteractionCommand, object()),
            previous=pending,
            record=pending,
            observed_at=_observation(1),
            policy=_POLICY,
        )
    supersede_scope = SupersedeInteractionScopeCommand(
        actor=actor,
        scope=InteractionExecutionScope(run_id=RunId("run-1")),
        provenance=AnswerProvenance.HUMAN,
    )
    with pytest.raises(InputValidationError):
        ScopeSupersessionReplayed(
            command=supersede_scope,
            _token=object(),
        )

    evidence = AcquireControllerActivity(
        request_id=pending.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    controller_command = _controller_command(pending, evidence)
    with pytest.raises(InputValidationError):
        ControllerActivityApplied(
            command=cast(RecordControllerActivityCommand, object()),
            previous=pending,
            record=pending,
            observed_at=_observation(1),
            policy=_POLICY,
        )
    non_acquire = _controller_command(
        pending,
        DisconnectControllerActivity(
            request_id=pending.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease"),
            sequence=1,
        ),
    )
    with pytest.raises(InputValidationError):
        ControllerActivityApplied(
            command=non_acquire,
            previous=pending,
            record=pending,
            observed_at=_observation(1),
            policy=_POLICY,
            lease_nonce=ActiveControlLeaseNonce("unexpected"),
        )
    assert controller_command.evidence is evidence
    acquired_previous, acquired, _, _ = _acquired_controller_transition()
    applied = apply_controller_activity(
        acquired_previous,
        _controller_command(
            acquired_previous,
            AcquireControllerActivity(
                request_id=acquired_previous.request.request_id,
                controller_id=ControllerId("controller-2"),
            ),
        ),
        _observation(11),
        _POLICY,
        lease_nonce=ActiveControlLeaseNonce("second-nonce"),
    )
    assert isinstance(applied, ControllerActivityApplied)
    assert applied.action is ControllerActivityAction.ACQUIRE
    assert applied.evidence.action is ControllerActivityAction.ACQUIRE

    with pytest.raises(InputValidationError):
        ControllerLeaseExpiredApplied(
            command=resolve,
            internal_authority=interaction_store._DEADLINE_RESOLVER,
            previous=acquired,
            record=acquired,
            observed_at=_observation(40),
            policy=_POLICY,
        )
    with pytest.raises(InputValidationError):
        ControllerLeaseExpiredApplied(
            internal_authority=interaction_store._DEADLINE_RESOLVER,
            previous=acquired,
            record=acquired,
            observed_at=_observation(40),
            policy=_POLICY,
        )

    with pytest.raises(InputValidationError):
        InteractionStoreReplayed(
            command=cast(ResolveInteractionCommand, object()),
            record=_terminal_record(),
            replay_kind=InteractionReplayKind.SAME_KEY,
        )
    terminal = _terminal_record()
    terminal_command = _resolve_command(terminal, "key-1")
    with pytest.raises(InputValidationError):
        InteractionStoreReplayed(
            command=terminal_command,
            record=terminal,
            replay_kind=cast(InteractionReplayKind, "invalid"),
        )
    with pytest.raises(InputValidationError):
        InteractionStoreReplayed(
            command=terminal_command,
            record=terminal,
            replay_kind=InteractionReplayKind.SAME_KEY,
            previous=terminal,
        )
    semantic_command = _resolve_command(terminal, "new-semantic-key")
    with pytest.raises(InputValidationError):
        InteractionStoreReplayed(
            command=semantic_command,
            record=terminal,
            replay_kind=InteractionReplayKind.SEMANTIC_NEW_KEY,
        )
    replayed = InteractionStoreReplayed(
        command=terminal_command,
        record=terminal,
        replay_kind=InteractionReplayKind.SAME_KEY,
    )
    assert replayed.idempotency_key == ResolutionIdempotencyKey("key-1")
    with pytest.raises(InputValidationError):
        apply_semantic_resolution_replay(
            terminal,
            cast(ResolveInteractionCommand, object()),
        )

    with pytest.raises(InputValidationError):
        CreateInteractionRejected(
            command=CreateInteractionCommand(actor=actor, request=_created()),
            error=cast(InputTransitionError, object()),
        )
    with pytest.raises(InputValidationError):
        ResolveInteractionRejected(
            command=cast(ResolveInteractionCommand, object()),
            error=error,
            decision_stage=ResolutionDecisionStage.AUTHORIZATION,
        )
    with pytest.raises(InputValidationError):
        ResolveInteractionRejected(
            command=resolve,
            error=cast(InputTransitionError, object()),
            decision_stage=ResolutionDecisionStage.AUTHORIZATION,
        )
    with pytest.raises(InputValidationError):
        ResolveInteractionRejected(
            command=resolve,
            error=error,
            decision_stage=cast(ResolutionDecisionStage, "invalid"),
        )
    registration = InteractionBranchRegistration(
        run_id=RunId("run-1"),
        branch_id=BranchId("child"),
        parent_branch_id=BranchId("parent"),
        principal=_principal(),
    )
    branch_command = RegisterInteractionBranchCommand(
        actor=actor,
        registration=registration,
    )
    with pytest.raises(InputValidationError):
        InteractionBranchRegistrationRejected(
            command=cast(RegisterInteractionBranchCommand, object()),
            error=error,
        )
    with pytest.raises(InputValidationError):
        InteractionBranchRegistrationRejected(
            command=branch_command,
            error=cast(InputTransitionError, object()),
        )


def test_store_reducer_precedence_and_private_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover deadline, lease, reducer, and trusted-command guard paths."""
    pending = _pending_record()
    candidate = _resolve_command(pending, "candidate")
    deadline_result = apply_candidate_resolution(
        pending,
        candidate,
        _observation(86_400),
        _POLICY,
    )
    assert isinstance(deadline_result, ResolveInteractionApplied)

    _, acquired, _, nonce = _acquired_controller_transition()
    acquired_candidate = _resolve_command(acquired, "lease-candidate")
    lease_result = apply_candidate_resolution(
        acquired,
        acquired_candidate,
        _observation(40),
        _POLICY,
    )
    assert isinstance(lease_result, ControllerLeaseExpiredApplied)

    deadline_default = _trusted_default_command(
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
    )
    assert isinstance(
        apply_trusted_default_resolution(
            pending,
            deadline_default,
            _observation(86_400),
            _POLICY,
        ),
        ResolveInteractionApplied,
    )
    lease_default = _trusted_default_command(
        correlation=acquired.correlation,
        expected_state_revision=acquired.request.state_revision,
    )
    assert isinstance(
        apply_trusted_default_resolution(
            acquired,
            lease_default,
            _observation(40),
            _POLICY,
        ),
        ControllerLeaseExpiredApplied,
    )

    scope_cancel = TerminalizeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=InteractionExecutionScope(run_id=RunId("run-1")),
        provenance=AnswerProvenance.HUMAN,
    )
    scope_supersede = SupersedeInteractionScopeCommand(
        actor=InteractionActor(principal=_principal()),
        scope=scope_cancel.scope,
        provenance=AnswerProvenance.HUMAN,
    )
    snapshot = (pending,)
    backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=InteractionStoreGeneration(1),
    )
    transaction = _begin_scope_transaction(backing, scope_cancel)
    with pytest.raises(InputValidationError):
        _begin_scope_transaction(
            backing,
            cast(TerminalizeInteractionScopeCommand, object()),
        )
    with pytest.raises(InputValidationError):
        _apply_scope_cancellation(
            transaction,
            cast(TerminalizeInteractionScopeCommand, object()),
            _observation(1),
            _POLICY,
            backing=backing,
        )
    supersede_transaction = _begin_scope_transaction(backing, scope_supersede)
    with pytest.raises(InputValidationError):
        _apply_scope_supersession(
            supersede_transaction,
            cast(SupersedeInteractionScopeCommand, object()),
            _observation(1),
            _POLICY,
            backing=backing,
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_policy(object())
    with pytest.raises(InputValidationError):
        interaction_store._validate_temporal_context(object(), _POLICY)

    with pytest.raises(InputValidationError):
        interaction_store._reduce_create_interaction(
            cast(CreateInteractionCommand, object()),
            _POLICY,
        )
    advisory_create = CreateInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        request=_created(RequirementMode.ADVISORY),
    )
    advisory_record = apply_create_interaction(advisory_create, _POLICY)
    assert advisory_record.record.advisory_wait is not None
    rejected = InputTransitionRejected(
        previous=_created(),
        error=InputTransitionError(
            code=InputErrorCode.ILLEGAL_TRANSITION,
            path="request.state",
            message="rejected",
        ),
    )
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store,
            "mark_request_pending",
            lambda *args, **kwargs: rejected,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_create_interaction(
                CreateInteractionCommand(
                    actor=InteractionActor(principal=_principal()),
                    request=_created(),
                ),
                _POLICY,
            )

    detached = apply_interaction_detachment(
        pending,
        _detachment_command(pending),
        _observation(1),
        _POLICY,
    )
    assert isinstance(detached, InteractionPresentationApplied)
    with pytest.raises(InputValidationError):
        interaction_store._reduce_interaction_presentation(
            detached.record,
            _presentation_command(detached.record),
            _observation(2),
            _POLICY,
        )
    presented = _presented_advisory_record()
    with pytest.raises(InputValidationError):
        interaction_store._reduce_interaction_presentation(
            presented,
            _presentation_command(presented),
            _observation(6),
            _POLICY,
        )
    invalid_queued = _queued_advisory_record()
    assert invalid_queued.advisory_wait is not None
    assert presented.advisory_wait is not None
    object.__setattr__(
        invalid_queued,
        "advisory_wait",
        replace(
            presented.advisory_wait,
            status=AdvisoryWaitStatus.RUNNING,
        ),
    )
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store, "_validate_record", lambda *args: None
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_interaction_presentation(
                invalid_queued,
                _presentation_command(invalid_queued),
                _observation(6),
                _POLICY,
            )

    assert (
        interaction_store._reduce_expired_controller_lease(
            acquired,
            _observation(86_400),
            _POLICY,
        )
        is None
    )
    acquire = _controller_command(
        presented,
        AcquireControllerActivity(
            request_id=presented.request.request_id,
            controller_id=ControllerId("controller-1"),
        ),
    )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_controller_activity(
            presented,
            acquire,
            _observation(65),
            _POLICY,
            lease_nonce=ActiveControlLeaseNonce("late"),
        )
    pulse = _controller_command(
        acquired,
        PulseControllerActivity(
            request_id=acquired.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=nonce,
            sequence=1,
        ),
    )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_controller_activity(
            acquired,
            pulse,
            _observation(40),
            _POLICY,
            lease_nonce=None,
        )
    reacquire = _controller_command(
        acquired,
        AcquireControllerActivity(
            request_id=acquired.request.request_id,
            controller_id=ControllerId("controller-2"),
        ),
    )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_controller_activity(
            acquired,
            reacquire,
            _observation(20),
            _POLICY,
            lease_nonce=ActiveControlLeaseNonce("reacquire"),
        )
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store,
            "_due_resolution_status",
            lambda *args: None,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_controller_activity(
                presented,
                acquire,
                _observation(65),
                _POLICY,
                lease_nonce=ActiveControlLeaseNonce("empty-budget"),
            )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_controller_activity(
            acquired,
            pulse,
            _observation(20),
            _POLICY,
            lease_nonce=ActiveControlLeaseNonce("unexpected"),
        )
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store,
            "_reduce_expired_controller_lease",
            lambda *args: None,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_controller_activity(
                acquired,
                pulse,
                _observation(40),
                _POLICY,
                lease_nonce=None,
            )
    unsupported = replace(pulse)
    object.__setattr__(unsupported, "evidence", object())
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store,
            "_validate_controller_command",
            lambda *args: None,
        )
        context.setattr(
            interaction_store,
            "_validate_existing_controller_lease",
            lambda *args: None,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_controller_activity(
                acquired,
                unsupported,
                _observation(20),
                _POLICY,
                lease_nonce=None,
            )


def test_store_classification_and_resolution_reducers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover broker-bound classification and lifecycle-CAS rejection."""
    pending = _pending_record()
    base = _resolve_command(pending, "base")
    text_answer = TextAnswer(
        question_id=QuestionId("text"),
        provenance=AnswerProvenance.HUMAN,
        value="text",
    )
    single_answer = interaction_api.SingleSelectionAnswer(
        question_id=QuestionId("single"),
        provenance=AnswerProvenance.HUMAN,
        value=interaction_api.FreeFormOther(text="other"),
    )
    multiple_answer = interaction_api.MultipleSelectionAnswer(
        question_id=QuestionId("multiple"),
        provenance=AnswerProvenance.HUMAN,
        values=(interaction_api.FreeFormOther(text="another"),),
    )
    proposed = AnsweredResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(text_answer, single_answer, multiple_answer),
    )
    unproved = _resolve_command(
        pending,
        "classified",
        resolution=proposed,
    )
    binding = _classifier_binding()
    requests = _task_input_classification_requests(
        pending,
        unproved,
        _POLICY,
    )
    outputs = tuple(
        TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=binding.classifier_id,
            classification_id=f"proof-{index}",
            policy_revision=binding.policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )
        for index, request in enumerate(requests)
    )
    proof = _bind_task_input_classifications(
        binding,
        pending,
        unproved,
        outputs,
        _POLICY,
    )
    interaction_store._validate_candidate_classifications(
        pending,
        unproved,
        _POLICY,
        binding,
        proof,
    )
    with pytest.raises(InputValidationError):
        _bind_task_input_classifications(
            binding,
            pending,
            unproved,
            cast(tuple[TaskInputClassification, ...], []),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        _bind_task_input_classifications(
            binding,
            pending,
            unproved,
            cast(tuple[TaskInputClassification, ...], (object(),) * 3),
            _POLICY,
        )
    duplicate_outputs = (
        outputs[0],
        replace(outputs[1], classification_id=outputs[0].classification_id),
        outputs[2],
    )
    with pytest.raises(InputValidationError):
        _bind_task_input_classifications(
            binding,
            pending,
            unproved,
            duplicate_outputs,
            _POLICY,
        )
    mixed_policy = replace(outputs[1], policy_revision="policy-2")
    with pytest.raises(InputValidationError):
        _bind_task_input_classifications(
            binding,
            pending,
            unproved,
            (outputs[0], mixed_policy, outputs[2]),
            _POLICY,
        )
    wrong_semantic = replace(
        outputs[0],
        semantic_type=interaction_api.QuestionType.MULTILINE_TEXT,
    )
    with pytest.raises(InputValidationError):
        _bind_task_input_classifications(
            binding,
            pending,
            unproved,
            (wrong_semantic, outputs[1], outputs[2]),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        _task_input_classification_requests(
            _with_request_identity(pending, "other-request"),
            base,
            _POLICY,
        )

    with pytest.raises(InputValidationError):
        interaction_store._reduce_candidate_resolution(
            pending,
            cast(ResolveInteractionCommand, object()),
            _observation(1),
            _POLICY,
        )
    other = _with_request_identity(pending, "other-request")
    with pytest.raises(InputValidationError):
        interaction_store._reduce_candidate_resolution(
            pending,
            _resolve_command(other, "other"),
            _observation(1),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_candidate_resolution(
            pending,
            base,
            _observation(86_400),
            _POLICY,
        )
    _, acquired, _, _ = _acquired_controller_transition()
    with pytest.raises(InputValidationError):
        interaction_store._reduce_candidate_resolution(
            acquired,
            _resolve_command(acquired, "lease"),
            _observation(40),
            _POLICY,
        )
    ledger_record = _pending_record()
    object.__setattr__(
        ledger_record,
        "idempotency_ledger",
        (
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("existing"),
                resolution_digest="0" * 64,
            ),
        ),
    )
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store,
            "_validate_pending_record",
            lambda *args: None,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_candidate_resolution(
                ledger_record,
                _resolve_command(ledger_record, "full"),
                _observation(1),
                replace(_POLICY, maximum_idempotency_keys_per_request=1),
            )

    transition_error = InputTransitionError(
        code=InputErrorCode.STALE_REVISION,
        path="expected_state_revision",
        message="stale",
    )
    rejected = InputTransitionRejected(
        previous=pending.request,
        error=transition_error,
    )
    with monkeypatch.context() as context:
        context.setattr(
            interaction_store,
            "resolve_request",
            lambda *args, **kwargs: rejected,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_candidate_resolution(
                pending,
                base,
                _observation(1),
                _POLICY,
            )
        trusted = _trusted_default_command(
            correlation=pending.correlation,
            expected_state_revision=pending.request.state_revision,
        )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_trusted_default_resolution(
                pending,
                trusted,
                _observation(1),
                _POLICY,
            )
        with pytest.raises(InputValidationError):
            interaction_store._reduce_terminal_resolution(
                pending,
                DeclinedResolution(
                    request_id=pending.request.request_id,
                    provenance=AnswerProvenance.HUMAN,
                    resolved_at=_NOW,
                ),
                _principal(),
            )

    with pytest.raises(InputValidationError):
        interaction_store._validate_trusted_default_command(pending, object())
    trusted = _trusted_default_command(
        correlation=other.correlation,
        expected_state_revision=other.request.state_revision,
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_trusted_default_command(pending, trusted)
    stale_trusted = _trusted_default_command(
        correlation=pending.correlation,
        expected_state_revision=interaction_api.StateRevision(0),
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_trusted_default_command(
            pending,
            stale_trusted,
        )
    with pytest.raises(InputValidationError):
        interaction_store._require_no_prior_settlement(
            pending,
            _observation(86_400),
            _POLICY,
        )

    cancel = CancelInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=pending.correlation,
        provenance=AnswerProvenance.HUMAN,
    )
    terminalize = TerminalizeInteractionCommand(
        actor=InteractionActor(principal=_principal()),
        correlation=pending.correlation,
        status=ResolutionStatus.UNAVAILABLE,
        provenance=AnswerProvenance.HUMAN,
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_request_cancellation_command(
            pending,
            object(),
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_request_cancellation_command(
            pending,
            replace(cancel, correlation=other.correlation),
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_request_cancellation_command(
            pending,
            replace(
                cancel,
                expected_state_revision=interaction_api.StateRevision(0),
            ),
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_request_terminalization_command(
            pending,
            object(),
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_request_terminalization_command(
            pending,
            replace(terminalize, correlation=other.correlation),
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_request_terminalization_command(
            pending,
            replace(
                terminalize,
                expected_state_revision=interaction_api.StateRevision(0),
            ),
        )


def test_store_scope_snapshot_and_batch_reducers_fail_closed() -> None:
    """Cover private graph, ownership, and due-batch guard branches."""
    root = _pending_record()
    child = _pending_record_on_branch(
        "child-request",
        "child-branch",
        parent_branch_id="branch-1",
    )
    child_registration = InteractionBranchRegistration(
        run_id=RunId("run-1"),
        branch_id=BranchId("child-branch"),
        parent_branch_id=BranchId("branch-1"),
        principal=_principal(),
    )
    child_branch = InteractionBranchRecord(
        registration=child_registration,
        store_revision=InteractionStoreRevision(1),
    )
    scope = InteractionExecutionScope(run_id=RunId("run-1"))
    actor = InteractionActor(principal=_principal())

    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_snapshot(
            cast(tuple[InteractionRecord, ...], []),
            (),
        )
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_snapshot(
            (),
            cast(tuple[InteractionBranchRecord, ...], []),
        )
    root_registration = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-1"),
            branch_id=BranchId("branch-1"),
            parent_branch_id=BranchId("unexpected-parent"),
            principal=_principal(),
        ),
        store_revision=InteractionStoreRevision(2),
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_snapshot(
            (root,),
            (root_registration,),
        )
    with pytest.raises(InputValidationError):
        interaction_store._select_scope_records(
            (root,),
            cast(InteractionExecutionScope, object()),
            (),
        )

    other_principal = PrincipalScope(user_id=UserId("other-user"))
    conflicting_branch = replace(
        child_branch,
        registration=replace(
            child_registration,
            principal=other_principal,
        ),
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_ownership(
            (child,),
            (conflicting_branch,),
            scope,
            _principal(),
            (child,),
        )
    interaction_store._validate_scope_ownership(
        (root, child),
        (child_branch,),
        scope,
        _principal(),
        (child,),
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_ownership(
            (),
            (),
            scope,
            _principal(),
            (root,),
        )
    other_owned_branch = replace(
        child_branch,
        registration=replace(
            child_registration,
            principal=other_principal,
        ),
    )
    child_scope = InteractionExecutionScope(
        run_id=RunId("run-1"),
        branch_id=BranchId("child-branch"),
    )
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_ownership(
            (),
            (other_owned_branch,),
            child_scope,
            _principal(),
            (),
        )

    command = TerminalizeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )
    snapshot = (root,)
    backing = _new_interaction_store_backing(
        records=snapshot,
        store_generation=InteractionStoreGeneration(1),
    )
    transaction = _begin_scope_transaction(backing, command)
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_transaction_commit(
            object(),
            command,
            backing,
        )
    object.__setattr__(transaction, "principal", other_principal)
    with pytest.raises(InputValidationError):
        interaction_store._validate_scope_transaction_commit(
            transaction,
            command,
            backing,
        )
    assert interaction_store._scope_descendant_branches(
        child_scope,
        (child_branch,),
    ) == frozenset({BranchId("child-branch")})

    supersede = SupersedeInteractionScopeCommand(
        actor=actor,
        scope=scope,
        provenance=AnswerProvenance.HUMAN,
    )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_scope_cancellation(
            (root,),
            cast(TerminalizeInteractionScopeCommand, object()),
            _observation(1),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_scope_cancellation(
            (),
            command,
            _observation(1),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_scope_supersession(
            (root,),
            cast(SupersedeInteractionScopeCommand, object()),
            _observation(1),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_scope_supersession(
            (),
            supersede,
            _observation(1),
            _POLICY,
        )
    superseded_due = interaction_store._reduce_scope_supersession(
        (root,),
        supersede,
        _observation(86_400),
        _POLICY,
    )
    assert isinstance(superseded_due[0].request.resolution, ExpiredResolution)

    due_command = TerminalizeDueInteractionsCommand(maximum_results=10)
    with pytest.raises(InputValidationError):
        interaction_store._reduce_due_interactions(
            (root,),
            cast(TerminalizeDueInteractionsCommand, object()),
            _observation(1),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_due_interactions(
            cast(tuple[InteractionRecord, ...], []),
            due_command,
            _observation(1),
            _POLICY,
        )
    with pytest.raises(InputValidationError):
        interaction_store._reduce_due_interactions(
            (root, root),
            due_command,
            _observation(1),
            _POLICY,
        )
    assert (
        interaction_store._reduce_due_interactions(
            (_terminal_record(),),
            due_command,
            _observation(1),
            _POLICY,
        )
        == ()
    )
    _, acquired, _, _ = _acquired_controller_transition()
    lease_due = interaction_store._reduce_due_interactions(
        (acquired,),
        due_command,
        _observation(40),
        _POLICY,
    )
    assert len(lease_due) == 1
    assert lease_due[0].advisory_wait is not None
    assert lease_due[0].advisory_wait.status is AdvisoryWaitStatus.RUNNING
