"""Exercise fail-closed interaction store validation boundaries."""

from copy import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from avalan.interaction import (
    MAX_STATE_REVISION,
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AdvisoryWaitState,
    AdvisoryWaitStatus,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelInteractionCommand,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ControllerActivityApplied,
    ControllerId,
    CreateInteractionCommand,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputErrorCode,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputTransitionApplied,
    InputTransitionError,
    InputTransitionRejected,
    InputValidationError,
    InteractionActor,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionCorrelation,
    InteractionExecutionScope,
    InteractionPolicy,
    InteractionPresentationApplied,
    InteractionPresentationState,
    InteractionRecord,
    InteractionStoreGeneration,
    InteractionStoreRevision,
    InteractionTime,
    ModelCallId,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PresentInteractionCommand,
    PrincipalScope,
    PulseControllerActivity,
    QuestionId,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    RequirementMode,
    ResolutionDecisionStage,
    ResolutionIdempotencyEntry,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    RunId,
    SelectedChoice,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    TerminalizeInteractionCommand,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TrustedDefaultResolutionRequest,
    TurnId,
    UnavailableResolution,
    UserId,
    apply_candidate_resolution,
    apply_controller_activity,
    apply_create_interaction,
    apply_interaction_presentation,
    apply_semantic_resolution_replay,
    canonical_resolution_digest,
    create_input_request,
    resolve_request,
    validate_controller_activity_transition,
    validate_interaction_presentation_transition,
    validate_resolution_commit_transition,
)
from avalan.interaction import store as interaction_store
from avalan.interaction.store import (
    _DEADLINE_RESOLVER,
    _TRUSTED_DEFAULT_RESOLVER,
    TrustedDefaultResolutionCommand,
    _bind_task_input_classifications,
    _insert_interaction_store_backing_branch_record,
    _insert_interaction_store_backing_record,
    _InteractionStoreBacking,
    _new_interaction_store_backing,
    _new_task_input_classifier_binding,
    _new_trusted_default_resolution_command,
    _replace_interaction_store_backing_branch_record,
    _replace_interaction_store_backing_records,
    _snapshot_interaction_store_backing,
    _TaskInputClassifierBinding,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
_POLICY = InteractionPolicy()


def _observation(seconds: float) -> InteractionTime:
    return InteractionTime.from_clock(
        wall_time=_NOW + timedelta(seconds=seconds),
        monotonic_seconds=seconds,
    )


def _principal(user_id: str = "user-1") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(user_id))


def _actor(user_id: str = "user-1") -> InteractionActor:
    return InteractionActor(principal=_principal(user_id))


def _trusted_default_command(
    *,
    correlation: InteractionCorrelation,
    expected_state_revision: StateRevision,
    actor: InteractionActor | None = None,
) -> TrustedDefaultResolutionCommand:
    """Mint one sealed command as the trusted broker boundary."""
    return _new_trusted_default_resolution_command(
        TrustedDefaultResolutionRequest(
            actor=actor or _actor(),
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


def _created(mode: RequirementMode = RequirementMode.REQUIRED) -> InputRequest:
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


def _pending_record(
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InteractionRecord:
    result = apply_create_interaction(
        CreateInteractionCommand(actor=_actor(), request=_created(mode)),
        _POLICY,
    )
    return result.record


def _presented_advisory_record() -> InteractionRecord:
    previous = _pending_record(RequirementMode.ADVISORY)
    command = PresentInteractionCommand(
        actor=_actor(),
        correlation=previous.correlation,
        expected_store_revision=previous.store_revision,
    )
    result = apply_interaction_presentation(
        previous,
        command,
        _observation(5),
        _POLICY,
    )
    assert isinstance(result, InteractionPresentationApplied)
    return result.record


def _resolve_command(
    record: InteractionRecord,
    key: str,
    *,
    resolution: DeclinedResolution | AnsweredResolution | None = None,
    actor: InteractionActor | None = None,
) -> ResolveInteractionCommand:
    proposed = resolution or DeclinedResolution(
        request_id=record.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    return ResolveInteractionCommand(
        actor=actor or _actor(),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=proposed,
    )


def _candidate_transition() -> tuple[
    InteractionRecord,
    InteractionRecord,
    ResolveInteractionCommand,
]:
    previous = _pending_record()
    command = _resolve_command(previous, "key-1")
    result = apply_candidate_resolution(
        previous,
        command,
        _observation(10),
        _POLICY,
    )
    assert isinstance(result, ResolveInteractionApplied)
    return previous, result.record, command


def _forged_record(
    source: InteractionRecord,
    **changes: object,
) -> InteractionRecord:
    record = copy(source)
    for name, value in changes.items():
        object.__setattr__(record, name, value)
    return record


def _resolved_request(
    previous: InteractionRecord,
    resolution: (
        DeclinedResolution
        | AnsweredResolution
        | TimedOutResolution
        | UnavailableResolution
    ),
) -> InputRequest:
    transition = resolve_request(
        previous.request,
        resolution,
        expected_state_revision=previous.request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    return transition.request


def _unchecked_resolution_record(
    resolution: (
        DeclinedResolution
        | AnsweredResolution
        | TimedOutResolution
        | UnavailableResolution
    ),
    *,
    resolved_by: object,
    digest: str | None = None,
) -> InteractionRecord:
    previous = _pending_record(
        RequirementMode.ADVISORY
        if isinstance(resolution, TimedOutResolution)
        else RequirementMode.REQUIRED
    )
    request = _resolved_request(previous, resolution)
    return _forged_record(
        previous,
        request=request,
        resolution_digest=(
            canonical_resolution_digest(resolution)
            if digest is None
            else digest
        ),
        resolved_by=resolved_by,
    )


def _assert_record_error(
    record: InteractionRecord,
    code: InputErrorCode,
    path: str,
) -> None:
    with pytest.raises(InputValidationError) as raised:
        interaction_store._validate_record_resolution(record)
    assert raised.value.code is code
    assert raised.value.path == path


def test_exact_result_guards_reject_wrong_shapes_and_mutations() -> None:
    """Reject malformed or inexact single-record and batch results."""
    record = _pending_record()
    changed = replace(
        record,
        store_revision=InteractionStoreRevision(record.store_revision + 1),
    )

    interaction_store._require_exact_result_record(record, record, "lookup")
    interaction_store._require_exact_result_records(
        (record,),
        (record,),
        "cancel",
    )
    with pytest.raises(InputValidationError) as wrong_record:
        interaction_store._require_exact_result_record(
            cast(InteractionRecord, object()),
            record,
            "lookup",
        )
    assert wrong_record.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as changed_record:
        interaction_store._require_exact_result_record(
            changed,
            record,
            "lookup",
        )
    assert changed_record.value.code is InputErrorCode.ILLEGAL_TRANSITION
    with pytest.raises(InputValidationError) as wrong_batch:
        interaction_store._require_exact_result_records(
            cast(tuple[InteractionRecord, ...], [record]),
            (record,),
            "cancel",
        )
    assert wrong_batch.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as changed_batch:
        interaction_store._require_exact_result_records(
            (changed,),
            (record,),
            "cancel",
        )
    assert changed_batch.value.code is InputErrorCode.ILLEGAL_TRANSITION


def test_public_transition_validators_reject_inexact_results() -> None:
    """Bind presentation, controller, and resolution commits exactly."""
    queued = _pending_record(RequirementMode.ADVISORY)
    presentation = PresentInteractionCommand(
        actor=_actor(),
        correlation=queued.correlation,
        expected_store_revision=queued.store_revision,
    )
    presented_result = apply_interaction_presentation(
        queued,
        presentation,
        _observation(5),
        _POLICY,
    )
    assert isinstance(presented_result, InteractionPresentationApplied)
    validate_interaction_presentation_transition(
        queued,
        presented_result.record,
        presentation,
        _observation(5),
        _POLICY,
    )
    with pytest.raises(InputValidationError):
        validate_interaction_presentation_transition(
            queued,
            replace(
                presented_result.record,
                store_revision=InteractionStoreRevision(
                    presented_result.record.store_revision + 1
                ),
            ),
            presentation,
            _observation(5),
            _POLICY,
        )

    evidence = AcquireControllerActivity(
        request_id=presented_result.record.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    controller = RecordControllerActivityCommand(
        actor=_actor(),
        correlation=presented_result.record.correlation,
        evidence=evidence,
    )
    nonce = ActiveControlLeaseNonce("lease-1")
    controller_result = apply_controller_activity(
        presented_result.record,
        controller,
        _observation(10),
        _POLICY,
        lease_nonce=nonce,
    )
    assert isinstance(controller_result, ControllerActivityApplied)
    validate_controller_activity_transition(
        presented_result.record,
        controller_result.record,
        controller,
        _observation(10),
        _POLICY,
        lease_nonce=nonce,
    )
    with pytest.raises(InputValidationError):
        validate_controller_activity_transition(
            presented_result.record,
            replace(
                controller_result.record,
                store_revision=InteractionStoreRevision(
                    controller_result.record.store_revision + 1
                ),
            ),
            controller,
            _observation(10),
            _POLICY,
            lease_nonce=nonce,
        )

    previous, committed, command = _candidate_transition()
    validate_resolution_commit_transition(
        previous,
        committed,
        ResolutionDecisionStage.COMMIT,
        _observation(10),
        _POLICY,
        command=command,
        idempotency_key=command.idempotency_key,
    )
    with pytest.raises(InputValidationError) as invalid_stage:
        validate_resolution_commit_transition(
            previous,
            committed,
            ResolutionDecisionStage.AUTHORIZATION,
            _observation(10),
            _POLICY,
            command=command,
            idempotency_key=command.idempotency_key,
        )
    assert invalid_stage.value.code is InputErrorCode.INVALID_FORMAT
    with pytest.raises(InputValidationError) as wrong_key:
        validate_resolution_commit_transition(
            previous,
            committed,
            ResolutionDecisionStage.COMMIT,
            _observation(10),
            _POLICY,
            command=command,
            idempotency_key=ResolutionIdempotencyKey("wrong-key"),
        )
    assert wrong_key.value.code is InputErrorCode.IDEMPOTENCY_CONFLICT
    with pytest.raises(InputValidationError) as not_due:
        validate_resolution_commit_transition(
            previous,
            previous,
            ResolutionDecisionStage.DEADLINE,
            _observation(10),
            _POLICY,
            command=None,
            idempotency_key=None,
        )
    assert not_due.value.code is InputErrorCode.ILLEGAL_TRANSITION


def test_revision_and_existing_lease_validation_fail_closed() -> None:
    """Reject exhausted revisions and unsequenced lease evidence."""
    assert (
        interaction_store._next_store_revision(InteractionStoreRevision(7))
        == 8
    )
    with pytest.raises(InputValidationError) as exhausted:
        interaction_store._next_store_revision(
            InteractionStoreRevision(MAX_STATE_REVISION)
        )
    assert exhausted.value.code is InputErrorCode.STATE_REVISION_EXHAUSTED

    presented = _presented_advisory_record()
    acquire = AcquireControllerActivity(
        request_id=presented.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    acquired = apply_controller_activity(
        presented,
        RecordControllerActivityCommand(
            actor=_actor(),
            correlation=presented.correlation,
            evidence=acquire,
        ),
        _observation(10),
        _POLICY,
        lease_nonce=ActiveControlLeaseNonce("lease-1"),
    )
    assert isinstance(acquired, ControllerActivityApplied)
    assert acquired.record.advisory_wait is not None
    with pytest.raises(InputValidationError) as unsequenced:
        interaction_store._validate_existing_controller_lease(
            acquired.record.advisory_wait,
            acquire,
        )
    assert unsequenced.value.code is InputErrorCode.INVALID_TYPE
    interaction_store._validate_existing_controller_lease(
        acquired.record.advisory_wait,
        PulseControllerActivity(
            request_id=acquired.record.request.request_id,
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=1,
        ),
    )
    assert interaction_store._is_candidate_resolution_record(
        _candidate_transition()[1]
    )
    assert not interaction_store._is_candidate_resolution_record(
        _pending_record()
    )


def test_digest_and_advisory_state_validators_reject_corruption() -> None:
    """Reject malformed digests and crossed advisory state fields."""
    assert interaction_store._validate_sha256("0" * 64, "digest") == "0" * 64
    with pytest.raises(InputValidationError) as malformed:
        interaction_store._validate_sha256("A" * 64, "digest")
    assert malformed.value.code is InputErrorCode.INVALID_FORMAT

    AdvisoryWaitState(
        status=AdvisoryWaitStatus.QUEUED,
        budget_seconds=60,
        remaining_seconds=60,
    )
    AdvisoryWaitState(
        status=AdvisoryWaitStatus.RUNNING,
        budget_seconds=60,
        remaining_seconds=55,
        presented_at=_NOW,
        running_since_monotonic=5,
    )
    AdvisoryWaitState(
        status=AdvisoryWaitStatus.PAUSED,
        budget_seconds=60,
        remaining_seconds=50,
        presented_at=_NOW,
        controller_id=ControllerId("controller-1"),
        lease_nonce=ActiveControlLeaseNonce("lease-1"),
        activity_sequence=1,
        lease_expires_at_monotonic=20,
    )
    AdvisoryWaitState(
        status=AdvisoryWaitStatus.EXHAUSTED,
        budget_seconds=60,
        remaining_seconds=0,
        presented_at=_NOW,
    )
    with pytest.raises(InputValidationError) as crossed:
        AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=60,
            remaining_seconds=60,
            presented_at=_NOW,
        )
    assert crossed.value.path == "advisory"


def test_record_advisory_validation_rejects_inconsistent_persistence() -> None:
    """Reject advisory metadata inconsistent with mode, budget, or anchor."""
    required = _pending_record()
    queued = _pending_record(RequirementMode.ADVISORY)
    presented = _presented_advisory_record()
    valid_wait = AdvisoryWaitState(
        status=AdvisoryWaitStatus.QUEUED,
        budget_seconds=60,
        remaining_seconds=60,
    )

    interaction_store._validate_record_advisory(required)
    interaction_store._validate_record_advisory(queued)
    with pytest.raises(InputValidationError) as required_timing:
        interaction_store._validate_record_advisory(
            _forged_record(required, advisory_wait=valid_wait)
        )
    assert required_timing.value.path == "record.advisory_wait"
    with pytest.raises(InputValidationError) as missing_timing:
        interaction_store._validate_record_advisory(
            _forged_record(queued, advisory_wait=None)
        )
    assert missing_timing.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as wrong_budget:
        interaction_store._validate_record_advisory(
            _forged_record(
                queued,
                advisory_wait=AdvisoryWaitState(
                    status=AdvisoryWaitStatus.QUEUED,
                    budget_seconds=61,
                    remaining_seconds=61,
                ),
            )
        )
    assert wrong_budget.value.code is InputErrorCode.INVALID_FORMAT

    wrong_request = replace(presented.request)
    assert wrong_request.advisory_deadline is not None
    object.__setattr__(
        wrong_request,
        "advisory_deadline",
        wrong_request.advisory_deadline + timedelta(seconds=1),
    )
    with pytest.raises(InputValidationError) as wrong_anchor:
        interaction_store._validate_record_advisory(
            _forged_record(presented, request=wrong_request)
        )
    assert wrong_anchor.value.path == "record.advisory_wait"


def test_record_resolution_validation_rejects_corrupt_authority() -> None:
    """Reject incomplete, mismatched, or falsely privileged resolutions."""
    pending = _pending_record()
    _assert_record_error(
        _forged_record(pending, resolved_by=_principal()),
        InputErrorCode.INVALID_FORMAT,
        "record.resolution",
    )

    human_decline = DeclinedResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    decline = _unchecked_resolution_record(
        human_decline,
        resolved_by=_principal(),
    )
    _assert_record_error(
        _forged_record(decline, resolution_digest=None),
        InputErrorCode.INVALID_FORMAT,
        "record.resolution_digest",
    )
    _assert_record_error(
        _forged_record(decline, resolution_digest="0" * 64),
        InputErrorCode.INVALID_FORMAT,
        "record.resolution_digest",
    )

    presented = _presented_advisory_record()
    assert presented.request.advisory_deadline is not None
    timeout = TimedOutResolution(
        request_id=presented.request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=presented.request.advisory_deadline,
    )
    timeout_request = _resolved_request(presented, timeout)
    _assert_record_error(
        _forged_record(
            presented,
            request=timeout_request,
            presentation=InteractionPresentationState.QUEUED,
            resolution_digest=canonical_resolution_digest(timeout),
            resolved_by=_DEADLINE_RESOLVER,
        ),
        InputErrorCode.ILLEGAL_TRANSITION,
        "record.resolution",
    )

    trusted_decline = DeclinedResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.TRUSTED_DEFAULT,
        resolved_at=_NOW,
    )
    _assert_record_error(
        _unchecked_resolution_record(
            trusted_decline,
            resolved_by=_principal(),
        ),
        InputErrorCode.FORBIDDEN,
        "record.resolved_by",
    )

    trusted_with_human_answer = AnsweredResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.TRUSTED_DEFAULT,
        resolved_at=_NOW,
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
    )
    _assert_record_error(
        _unchecked_resolution_record(
            trusted_with_human_answer,
            resolved_by=_TRUSTED_DEFAULT_RESOLVER,
        ),
        InputErrorCode.FORBIDDEN,
        "record.resolution.answers",
    )

    policy_candidate = replace(
        human_decline,
        provenance=AnswerProvenance.POLICY,
    )
    _assert_record_error(
        _unchecked_resolution_record(
            policy_candidate,
            resolved_by=_principal(),
        ),
        InputErrorCode.FORBIDDEN,
        "record.resolved_by",
    )
    interaction_store._validate_record_resolution(
        _unchecked_resolution_record(
            policy_candidate,
            resolved_by=interaction_store._TRUSTED_POLICY_RESOLVER,
        )
    )
    policy_with_human_answer = replace(
        trusted_with_human_answer,
        provenance=AnswerProvenance.POLICY,
    )
    _assert_record_error(
        _unchecked_resolution_record(
            policy_with_human_answer,
            resolved_by=interaction_store._TRUSTED_POLICY_RESOLVER,
        ),
        InputErrorCode.FORBIDDEN,
        "record.resolution.answers",
    )
    _assert_record_error(
        _unchecked_resolution_record(
            human_decline,
            resolved_by=None,
        ),
        InputErrorCode.INVALID_TYPE,
        "record.resolved_by",
    )

    human_with_trusted_answer = AnsweredResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value=True,
            ),
        ),
    )
    valid_human_answer = replace(
        human_with_trusted_answer,
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
    )
    forged_answer_record = _unchecked_resolution_record(
        valid_human_answer,
        resolved_by=_principal(),
    )
    forged_answer_request = copy(forged_answer_record.request)
    object.__setattr__(
        forged_answer_request,
        "resolution",
        human_with_trusted_answer,
    )
    _assert_record_error(
        _forged_record(
            forged_answer_record,
            request=forged_answer_request,
            resolution_digest=canonical_resolution_digest(
                human_with_trusted_answer
            ),
        ),
        InputErrorCode.FORBIDDEN,
        "record.resolution.answers",
    )

    unavailable = UnavailableResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    _assert_record_error(
        _unchecked_resolution_record(
            unavailable,
            resolved_by=interaction_store._ADMISSION_CLEANUP_RESOLVER,
        ),
        InputErrorCode.FORBIDDEN,
        "record.resolution.provenance",
    )
    _assert_record_error(
        _unchecked_resolution_record(unavailable, resolved_by=None),
        InputErrorCode.INVALID_TYPE,
        "record.resolved_by",
    )


def test_idempotency_ledger_validation_rejects_corrupt_bindings() -> None:
    """Reject malformed, duplicate, unbound, or conflicting key ledgers."""
    pending = _pending_record()
    _, candidate, _ = _candidate_transition()
    assert candidate.resolution_digest is not None
    entry = ResolutionIdempotencyEntry(
        key=ResolutionIdempotencyKey("key-1"),
        resolution_digest=candidate.resolution_digest,
    )

    with pytest.raises(InputValidationError) as wrong_container:
        interaction_store._validate_idempotency_ledger(
            _forged_record(
                pending,
                idempotency_ledger=cast(
                    tuple[ResolutionIdempotencyEntry, ...],
                    [entry],
                ),
            )
        )
    assert wrong_container.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as wrong_entry:
        interaction_store._validate_idempotency_ledger(
            _forged_record(
                pending,
                idempotency_ledger=cast(
                    tuple[ResolutionIdempotencyEntry, ...],
                    (object(),),
                ),
            )
        )
    assert wrong_entry.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as duplicate:
        interaction_store._validate_idempotency_ledger(
            _forged_record(
                candidate,
                idempotency_ledger=(entry, entry),
            )
        )
    assert duplicate.value.code is InputErrorCode.DUPLICATE
    with pytest.raises(InputValidationError) as unresolved:
        interaction_store._validate_idempotency_ledger(
            _forged_record(pending, idempotency_ledger=(entry,))
        )
    assert unresolved.value.code is InputErrorCode.INVALID_FORMAT
    with pytest.raises(InputValidationError) as missing_key:
        interaction_store._validate_idempotency_ledger(
            _forged_record(candidate, idempotency_ledger=())
        )
    assert missing_key.value.code is InputErrorCode.INVALID_FORMAT

    unavailable = UnavailableResolution(
        request_id=pending.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    terminal = _unchecked_resolution_record(
        unavailable,
        resolved_by=_principal(),
    )
    with pytest.raises(InputValidationError) as terminal_key:
        interaction_store._validate_idempotency_ledger(
            _forged_record(terminal, idempotency_ledger=(entry,))
        )
    assert terminal_key.value.code is InputErrorCode.INVALID_FORMAT

    conflicting = ResolutionIdempotencyEntry(
        key=ResolutionIdempotencyKey("key-1"),
        resolution_digest="0" * 64,
    )
    with pytest.raises(InputValidationError) as conflict:
        interaction_store._validate_idempotency_ledger(
            _forged_record(candidate, idempotency_ledger=(conflicting,))
        )
    assert conflict.value.code is InputErrorCode.IDEMPOTENCY_CONFLICT


def test_field_and_command_validators_reject_wrong_runtime_types() -> None:
    """Reject invalid actor, provenance, command, and evidence identities."""
    pending = _pending_record()
    with pytest.raises(InputValidationError) as provenance:
        interaction_store._validate_terminalization_fields(
            ResolutionStatus.UNAVAILABLE,
            object(),
        )
    assert provenance.value.path == "terminalize.provenance"
    with pytest.raises(InputValidationError) as actor:
        interaction_store._validate_actor(object())
    assert actor.value.path == "actor"
    with pytest.raises(InputValidationError) as presentation:
        interaction_store._validate_presentation_command_identity(
            pending,
            object(),
        )
    assert presentation.value.path == "command"
    with pytest.raises(InputValidationError) as controller:
        interaction_store._validate_controller_command(pending, object())
    assert controller.value.path == "command"

    evidence = AcquireControllerActivity(
        request_id=pending.request.request_id,
        controller_id=ControllerId("controller-1"),
    )
    command = RecordControllerActivityCommand(
        actor=_actor(),
        correlation=pending.correlation,
        evidence=evidence,
    )
    object.__setattr__(
        command,
        "evidence",
        replace(evidence, request_id=InputRequestId("other-request")),
    )
    with pytest.raises(InputValidationError) as wrong_evidence:
        interaction_store._validate_controller_command(pending, command)
    assert wrong_evidence.value.path == "command.evidence.request_id"


def test_trigger_validators_cover_resolve_default_and_unknown_commands() -> (
    None
):
    """Validate supported deadline triggers and reject mismatched commands."""
    pending = _pending_record()
    resolve = _resolve_command(pending, "trigger-key")
    interaction_store._validate_deadline_trigger_command(pending, resolve)
    interaction_store._validate_lease_expiry_trigger_command(pending, resolve)

    other = replace(
        resolve.correlation,
        request_id=InputRequestId("other-request"),
    )
    wrong_resolve = replace(resolve)
    object.__setattr__(wrong_resolve, "correlation", other)
    with pytest.raises(InputValidationError) as deadline_mismatch:
        interaction_store._validate_deadline_trigger_command(
            pending,
            wrong_resolve,
        )
    assert deadline_mismatch.value.path == "result.command.correlation"
    with pytest.raises(InputValidationError) as lease_mismatch:
        interaction_store._validate_lease_expiry_trigger_command(
            pending,
            wrong_resolve,
        )
    assert lease_mismatch.value.path == "result.command.correlation"

    trusted = _trusted_default_command(
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
    )
    interaction_store._validate_deadline_trigger_command(pending, trusted)
    interaction_store._validate_lease_expiry_trigger_command(pending, trusted)

    for validator in (
        interaction_store._validate_deadline_trigger_command,
        interaction_store._validate_lease_expiry_trigger_command,
    ):
        with pytest.raises(InputValidationError) as unknown:
            validator(pending, object())
        assert unknown.value.code is InputErrorCode.INVALID_TYPE


def test_replay_branch_and_basic_identity_validators_fail_closed() -> None:
    """Reject replay, branch, correlation, and persisted-record drift."""
    _, terminal, _ = _candidate_transition()
    different = AnsweredResolution(
        request_id=terminal.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
    )
    command = _resolve_command(
        terminal,
        "different-key",
        resolution=different,
    )
    with pytest.raises(InputValidationError) as replay:
        interaction_store._validate_replay_command(terminal, command)
    assert replay.value.code is InputErrorCode.IDEMPOTENCY_CONFLICT

    policy_resolution = AnsweredResolution(
        request_id=terminal.request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.POLICY,
                value=True,
            ),
        ),
    )
    forged_policy_record = _unchecked_resolution_record(
        policy_resolution,
        resolved_by=_principal(),
    )
    policy_command = interaction_store._new_trusted_policy_resolution_command(
        actor=_actor(),
        correlation=forged_policy_record.correlation,
        expected_state_revision=forged_policy_record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("policy-replay-key"),
        proposed_resolution=policy_resolution,
    )
    with pytest.raises(InputValidationError) as policy_authority:
        interaction_store._validate_replay_command(
            forged_policy_record,
            policy_command,
        )
    assert policy_authority.value.code is InputErrorCode.FORBIDDEN
    assert policy_authority.value.path == "result.command.actor"

    registration = InteractionBranchRegistration(
        run_id=RunId("run-1"),
        branch_id=BranchId("child"),
        parent_branch_id=BranchId("parent"),
        principal=_principal(),
    )
    branch = InteractionBranchRecord(
        registration=registration,
        store_revision=InteractionStoreRevision(1),
    )
    valid_branch = RegisterInteractionBranchCommand(
        actor=_actor(),
        registration=registration,
    )
    interaction_store._validate_branch_result_command(branch, valid_branch)
    with pytest.raises(InputValidationError) as branch_command:
        interaction_store._validate_branch_result_command(branch, object())
    assert branch_command.value.code is InputErrorCode.INVALID_TYPE

    interaction_store._validate_correlation(terminal.correlation)
    interaction_store._validate_record(terminal, "record")
    interaction_store._validate_branch_record(branch, "branch")
    with pytest.raises(InputValidationError) as invalid_correlation:
        interaction_store._validate_correlation(object())
    assert invalid_correlation.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as invalid_record:
        interaction_store._validate_record(object(), "record")
    assert invalid_record.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as invalid_branch:
        interaction_store._validate_branch_record(object(), "branch")
    assert invalid_branch.value.code is InputErrorCode.INVALID_TYPE


def test_semantic_replay_binding_requires_original_candidate_commit() -> None:
    """Allow only an exact new-key binding on a candidate terminal record."""
    pending = _pending_record()
    with pytest.raises(InputValidationError) as noncandidate:
        interaction_store._validate_semantic_replay_binding(
            pending,
            pending,
            _resolve_command(pending, "new-key"),
        )
    assert noncandidate.value.path == "result.previous"

    _, terminal, _ = _candidate_transition()
    command = _resolve_command(terminal, "new-key")
    replay = apply_semantic_resolution_replay(terminal, command)
    interaction_store._validate_semantic_replay_binding(
        terminal,
        replay.record,
        command,
    )
    with pytest.raises(InputValidationError) as mutation:
        interaction_store._validate_semantic_replay_binding(
            terminal,
            replace(
                replay.record,
                store_revision=InteractionStoreRevision(
                    replay.record.store_revision + 1
                ),
            ),
            command,
        )
    assert mutation.value.path == "result.record"


def test_optional_revision_normalizes_only_present_values() -> None:
    """Normalize optional revisions without inventing absent concurrency."""
    pending = _pending_record()
    command = TerminalizeInteractionCommand(
        actor=_actor(),
        correlation=pending.correlation,
        status=ResolutionStatus.UNAVAILABLE,
        provenance=AnswerProvenance.HUMAN,
        expected_state_revision=StateRevision(7),
    )
    assert command.expected_state_revision == 7
    cancellation = CancelInteractionCommand(
        actor=_actor(),
        correlation=pending.correlation,
        provenance=AnswerProvenance.HUMAN,
    )
    assert cancellation.expected_state_revision is None


def test_private_authority_constructors_and_validators_fail_closed() -> None:
    """Reject direct construction of backing and classifier capabilities."""
    with pytest.raises(InputValidationError) as classifier:
        _TaskInputClassifierBinding(
            classifier_id=_POLICY.task_input_classifier_id,
            policy_revision=_POLICY.task_input_policy_revision,
            _token=object(),
        )
    with pytest.raises(InputValidationError) as backing:
        _InteractionStoreBacking(
            records=(),
            branch_records=(),
            store_generation=InteractionStoreGeneration(0),
            _token=object(),
        )
    with pytest.raises(InputValidationError) as invalid_binding:
        interaction_store._validate_classifier_binding_policy(
            object(),
            _POLICY,
        )
    with pytest.raises(InputValidationError) as invalid_backing:
        interaction_store._validate_store_backing(object())

    assert classifier.value.code is InputErrorCode.FORBIDDEN
    assert backing.value.code is InputErrorCode.FORBIDDEN
    assert invalid_binding.value.code is InputErrorCode.INVALID_TYPE
    assert invalid_backing.value.code is InputErrorCode.INVALID_TYPE


def test_private_scope_attestation_constructors_fail_closed() -> None:
    """Reject forged branch-closure and scope-presence capabilities."""
    scope = InteractionExecutionScope(run_id=RunId("run-1"))
    principal = _principal()

    with pytest.raises(InputValidationError) as branch_token:
        interaction_store._InteractionBranchClosureAttestation(
            frozenset(),
            _token=object(),
        )
    with pytest.raises(InputValidationError) as scope_token:
        interaction_store._InteractionScopeOwnershipAttestation(
            scope=scope,
            principal=principal,
            actor_owned_record_match=False,
            foreign_owned_record_match=False,
            actor_owned_branch_match=False,
            foreign_owned_branch_match=False,
            _token=object(),
        )
    with pytest.raises(InputValidationError) as invalid_scope:
        interaction_store._InteractionScopeOwnershipAttestation(
            scope=cast(InteractionExecutionScope, object()),
            principal=principal,
            actor_owned_record_match=False,
            foreign_owned_record_match=False,
            actor_owned_branch_match=False,
            foreign_owned_branch_match=False,
            _token=interaction_store._SCOPE_OWNERSHIP_ATTESTATION_TOKEN,
        )
    with pytest.raises(InputValidationError) as invalid_principal:
        interaction_store._InteractionScopeOwnershipAttestation(
            scope=scope,
            principal=cast(PrincipalScope, object()),
            actor_owned_record_match=False,
            foreign_owned_record_match=False,
            actor_owned_branch_match=False,
            foreign_owned_branch_match=False,
            _token=interaction_store._SCOPE_OWNERSHIP_ATTESTATION_TOKEN,
        )

    attestation = interaction_store._InteractionScopeOwnershipAttestation(
        scope=scope,
        principal=principal,
        actor_owned_record_match=False,
        foreign_owned_record_match=False,
        actor_owned_branch_match=False,
        foreign_owned_branch_match=False,
        _token=interaction_store._SCOPE_OWNERSHIP_ATTESTATION_TOKEN,
    )
    with pytest.raises(InputValidationError) as combined:
        _InteractionStoreBacking(
            records=(),
            branch_records=(),
            store_generation=InteractionStoreGeneration(0),
            scope_ownership_attestation=attestation,
            _token=interaction_store._PARTIAL_STORE_BACKING_TOKEN,
        )

    assert branch_token.value.code is InputErrorCode.FORBIDDEN
    assert scope_token.value.code is InputErrorCode.FORBIDDEN
    assert invalid_scope.value.code is InputErrorCode.INVALID_TYPE
    assert invalid_principal.value.code is InputErrorCode.INVALID_TYPE
    assert combined.value.code is InputErrorCode.INVALID_FORMAT


def test_branch_closure_attestation_rejects_graph_drift() -> None:
    """Reject duplicate, missing, cyclic, and unrelated closure edges."""
    principal = _principal()
    pending = _pending_record()
    child_origin = replace(
        pending.request.origin,
        branch_id=BranchId("child"),
        parent_branch_id=BranchId("root"),
    )
    child = replace(
        pending,
        request=replace(pending.request, origin=child_origin),
    )
    child_registration = InteractionBranchRegistration(
        run_id=child_origin.run_id,
        branch_id=child_origin.branch_id,
        parent_branch_id=BranchId("root"),
        principal=principal,
    )
    child_branch = InteractionBranchRecord(
        registration=child_registration,
        store_revision=InteractionStoreRevision(0),
    )
    root_branch = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=child_origin.run_id,
            branch_id=BranchId("root"),
            parent_branch_id=BranchId("child"),
            principal=principal,
        ),
        store_revision=InteractionStoreRevision(0),
    )
    unrelated_branch = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=child_origin.run_id,
            branch_id=BranchId("unrelated"),
            parent_branch_id=BranchId("other-root"),
            principal=principal,
        ),
        store_revision=InteractionStoreRevision(0),
    )

    for records, branches, code in (
        (
            (child,),
            (
                child_branch,
                replace(
                    child_branch,
                    store_revision=InteractionStoreRevision(1),
                ),
            ),
            InputErrorCode.DUPLICATE,
        ),
        ((child,), (), InputErrorCode.CORRELATION_MISMATCH),
        (
            (child,),
            (child_branch, root_branch),
            InputErrorCode.INVALID_FORMAT,
        ),
        (
            (child,),
            (child_branch, unrelated_branch),
            InputErrorCode.CORRELATION_MISMATCH,
        ),
    ):
        with pytest.raises(InputValidationError) as error:
            interaction_store._derive_interaction_branch_closure_roots(
                records,
                branches,
            )
        assert error.value.code is code

    closure = interaction_store._InteractionBranchClosureAttestation(
        frozenset(),
        _token=interaction_store._BRANCH_CLOSURE_ATTESTATION_TOKEN,
    )
    with pytest.raises(InputValidationError) as wrong_type:
        interaction_store._validate_interaction_branch_closure_attestation(
            cast(
                interaction_store._InteractionBranchClosureAttestation,
                object(),
            ),
            (),
            (),
        )
    object.__setattr__(
        closure,
        "authoritative_branch_roots",
        frozenset(
            {
                (
                    RunId("run-1"),
                    principal,
                    BranchId("unexpected-root"),
                )
            }
        ),
    )
    with pytest.raises(InputValidationError) as roots_changed:
        interaction_store._validate_interaction_branch_closure_attestation(
            closure,
            (),
            (),
        )
    assert wrong_type.value.code is InputErrorCode.INVALID_TYPE
    assert roots_changed.value.code is InputErrorCode.CORRELATION_MISMATCH


def test_scope_ownership_attestation_rejects_forgery_and_drift() -> None:
    """Reject forged fields, mismatched scope, and false actor presence."""
    scope = InteractionExecutionScope(run_id=RunId("run-1"))
    principal = _principal()

    def attestation() -> (
        interaction_store._InteractionScopeOwnershipAttestation
    ):
        return interaction_store._InteractionScopeOwnershipAttestation(
            scope=scope,
            principal=principal,
            actor_owned_record_match=False,
            foreign_owned_record_match=False,
            actor_owned_branch_match=False,
            foreign_owned_branch_match=False,
            _token=interaction_store._SCOPE_OWNERSHIP_ATTESTATION_TOKEN,
        )

    with pytest.raises(InputValidationError) as wrong_type:
        interaction_store._validate_interaction_scope_ownership_attestation(
            cast(
                interaction_store._InteractionScopeOwnershipAttestation,
                object(),
            )
        )
    invalid_scope = attestation()
    object.__setattr__(invalid_scope, "scope", object())
    with pytest.raises(InputValidationError) as forged_scope:
        interaction_store._validate_interaction_scope_ownership_attestation(
            invalid_scope
        )
    invalid_principal = attestation()
    object.__setattr__(invalid_principal, "principal", object())
    with pytest.raises(InputValidationError) as forged_principal:
        interaction_store._validate_interaction_scope_ownership_attestation(
            invalid_principal
        )

    mismatched = _InteractionStoreBacking(
        records=(),
        branch_records=(),
        store_generation=InteractionStoreGeneration(0),
        scope_ownership_attestation=attestation(),
        _token=interaction_store._STORE_BACKING_TOKEN,
    )
    with pytest.raises(InputValidationError) as scope_drift:
        interaction_store._validate_scope_ownership_presence(
            _snapshot_interaction_store_backing(mismatched),
            InteractionExecutionScope(run_id=RunId("other-run")),
            principal,
            (),
        )

    pending = _pending_record()
    false_presence = _InteractionStoreBacking(
        records=(pending,),
        branch_records=(),
        store_generation=InteractionStoreGeneration(0),
        scope_ownership_attestation=attestation(),
        _token=interaction_store._STORE_BACKING_TOKEN,
    )
    with pytest.raises(InputValidationError) as presence_drift:
        interaction_store._validate_scope_ownership_presence(
            _snapshot_interaction_store_backing(false_presence),
            scope,
            principal,
            (pending,),
        )

    foreign = replace(
        pending,
        request=replace(
            pending.request,
            origin=replace(
                pending.request.origin,
                principal=_principal("other-user"),
            ),
        ),
    )
    with pytest.raises(InputValidationError) as foreign_selected:
        interaction_store._validate_scope_ownership(
            (foreign,),
            (),
            frozenset(),
            scope,
            principal,
            (foreign,),
        )

    assert wrong_type.value.code is InputErrorCode.INVALID_TYPE
    assert forged_scope.value.code is InputErrorCode.INVALID_TYPE
    assert forged_principal.value.code is InputErrorCode.INVALID_TYPE
    assert scope_drift.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert presence_drift.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert foreign_selected.value.code is InputErrorCode.FORBIDDEN


def test_backing_inventory_mutations_are_exact_cas_operations() -> None:
    """Centralize record and branch inventory updates on one backing."""
    first = _pending_record()
    second_created = replace(
        _created(),
        request_id=InputRequestId("request-2"),
        continuation_id=ContinuationId("continuation-2"),
    )
    second = apply_create_interaction(
        CreateInteractionCommand(actor=_actor(), request=second_created),
        _POLICY,
    ).record
    backing = _new_interaction_store_backing(records=(second, first))
    initial = _snapshot_interaction_store_backing(backing)
    assert initial.records == (first, second)
    with pytest.raises(InputValidationError):
        backing._advance(
            records=initial.records,
            branch_records=(),
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _insert_interaction_store_backing_record(backing, first)
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_records(
            backing,
            cast(tuple[InteractionRecord, ...], []),
            (first,),
        )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_records(backing, (), ())

    updated = replace(
        first,
        store_revision=InteractionStoreRevision(first.store_revision + 1),
    )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_records(
            backing,
            (first,),
            (second,),
        )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_records(
            backing,
            (first, first),
            (updated, updated),
        )
    stale = replace(
        first,
        store_revision=InteractionStoreRevision(first.store_revision + 9),
    )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_records(
            backing,
            (stale,),
            (
                replace(
                    stale,
                    store_revision=InteractionStoreRevision(
                        stale.store_revision + 1
                    ),
                ),
            ),
        )
    replaced = _replace_interaction_store_backing_records(
        backing,
        (first,),
        (updated,),
    )
    assert replaced.records == (updated, second)
    assert replaced.store_generation == initial.store_generation + 1

    branch_backing = _new_interaction_store_backing()
    first_registration = InteractionBranchRegistration(
        run_id=RunId("run-1"),
        branch_id=BranchId("child-1"),
        parent_branch_id=BranchId("root-1"),
        principal=_principal(),
    )
    first_branch = InteractionBranchRecord(
        registration=first_registration,
        store_revision=InteractionStoreRevision(1),
    )
    second_branch = InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId("run-2"),
            branch_id=BranchId("child-2"),
            parent_branch_id=BranchId("root-2"),
            principal=_principal(),
        ),
        store_revision=InteractionStoreRevision(1),
    )
    _insert_interaction_store_backing_branch_record(
        branch_backing,
        first_branch,
    )
    inserted_branches = _insert_interaction_store_backing_branch_record(
        branch_backing,
        second_branch,
    )
    assert inserted_branches.branch_records == (first_branch, second_branch)
    with pytest.raises(InputValidationError):
        _insert_interaction_store_backing_branch_record(
            branch_backing,
            first_branch,
        )

    changed_identity = replace(
        first_branch,
        registration=replace(
            first_registration,
            branch_id=BranchId("changed-child"),
        ),
    )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_branch_record(
            branch_backing,
            first_branch,
            changed_identity,
        )
    stale_branch = replace(
        first_branch,
        store_revision=InteractionStoreRevision(9),
    )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_branch_record(
            branch_backing,
            stale_branch,
            replace(stale_branch, store_revision=InteractionStoreRevision(10)),
        )
    missing_branch = replace(
        first_branch,
        registration=replace(
            first_registration,
            branch_id=BranchId("missing-child"),
        ),
    )
    with pytest.raises(InputValidationError):
        _replace_interaction_store_backing_branch_record(
            branch_backing,
            missing_branch,
            replace(
                missing_branch,
                store_revision=InteractionStoreRevision(2),
            ),
        )
    updated_branch = replace(
        first_branch,
        store_revision=InteractionStoreRevision(2),
    )
    replaced_branch = _replace_interaction_store_backing_branch_record(
        branch_backing,
        first_branch,
        updated_branch,
    )
    assert replaced_branch.branch_records == (updated_branch, second_branch)


def test_requestless_branch_replacement_rejects_registration_drift() -> None:
    """Preserve exact ancestry and ownership for a requestless branch."""
    registration = InteractionBranchRegistration(
        run_id=RunId("requestless-run"),
        branch_id=BranchId("requestless-child"),
        parent_branch_id=BranchId("requestless-parent"),
        principal=_principal(),
    )
    branch_record = InteractionBranchRecord(
        registration=registration,
        store_revision=InteractionStoreRevision(4),
    )
    backing = _new_interaction_store_backing(
        branch_records=(branch_record,),
        store_generation=InteractionStoreGeneration(7),
    )
    initial = _snapshot_interaction_store_backing(backing)
    assert initial.records == ()

    changed_parent = replace(
        branch_record,
        registration=replace(
            registration,
            parent_branch_id=BranchId("hostile-parent"),
        ),
        store_revision=InteractionStoreRevision(5),
    )
    changed_principal = replace(
        branch_record,
        registration=replace(
            registration,
            principal=PrincipalScope(user_id=UserId("hostile-owner")),
        ),
        store_revision=InteractionStoreRevision(5),
    )
    for hostile, code, path in (
        (
            changed_parent,
            InputErrorCode.CORRELATION_MISMATCH,
            "branch_record.registration",
        ),
        (
            changed_principal,
            InputErrorCode.FORBIDDEN,
            "branch_record.registration.principal",
        ),
    ):
        with pytest.raises(InputValidationError) as raised:
            _replace_interaction_store_backing_branch_record(
                backing,
                branch_record,
                hostile,
            )
        assert raised.value.code is code
        assert raised.value.path == path
        current = _snapshot_interaction_store_backing(backing)
        assert current == initial
        assert current.store_generation == InteractionStoreGeneration(7)
        safe_error = str(raised.value)
        assert "hostile-parent" not in safe_error
        assert "hostile-owner" not in safe_error


def test_classification_work_and_empty_envelopes_fail_closed() -> None:
    """Cover non-free-form work and exact classifier tuple cardinality."""
    previous = _pending_record()
    with pytest.raises(InputValidationError):
        interaction_store._task_input_classification_requests(
            previous,
            cast(ResolveInteractionCommand, object()),
            _POLICY,
        )

    resolution = AnsweredResolution(
        request_id=previous.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
            MultipleSelectionAnswer(
                question_id=QuestionId("selected"),
                provenance=AnswerProvenance.HUMAN,
                values=(SelectedChoice(value=ChoiceValue("choice")),),
            ),
        ),
    )
    no_work = _resolve_command(
        previous,
        "no-classification-work",
        resolution=resolution,
    )
    assert (
        interaction_store._task_input_classification_requests(
            previous,
            no_work,
            _POLICY,
        )
        == ()
    )
    binding = _new_task_input_classifier_binding(
        classifier_id=_POLICY.task_input_classifier_id,
        policy_revision=_POLICY.task_input_policy_revision,
    )
    empty_proof = _bind_task_input_classifications(
        binding,
        previous,
        no_work,
        (),
        _POLICY,
    )
    with pytest.raises(InputValidationError) as unexpected:
        interaction_store._validate_candidate_classifications(
            previous,
            no_work,
            _POLICY,
            binding,
            empty_proof,
        )
    assert unexpected.value.path == "classification_proof"

    classified = _resolve_command(
        previous,
        "missing-classification-output",
        resolution=AnsweredResolution(
            request_id=previous.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(
                TextAnswer(
                    question_id=QuestionId("text"),
                    provenance=AnswerProvenance.HUMAN,
                    value="classify me",
                ),
            ),
        ),
    )
    with pytest.raises(InputValidationError) as missing:
        _bind_task_input_classifications(
            binding,
            previous,
            classified,
            (),
            _POLICY,
        )
    assert (
        missing.value.code is InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE
    )


@pytest.mark.parametrize(
    "question",
    (
        ConfirmationQuestion(
            question_id=QuestionId("optional-confirm"),
            prompt="Confirm?",
            required=False,
        ),
        TextQuestion(
            question_id=QuestionId("optional-text"),
            prompt="Text?",
            required=False,
        ),
        MultilineTextQuestion(
            question_id=QuestionId("optional-multiline"),
            prompt="Details?",
            required=False,
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("optional-single"),
            prompt="Select?",
            required=False,
            choices=(
                Choice(value=ChoiceValue("first"), label="First"),
                Choice(value=ChoiceValue("second"), label="Second"),
            ),
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("optional-multiple"),
            prompt="Select any?",
            required=False,
            choices=(
                Choice(value=ChoiceValue("first"), label="First"),
                Choice(value=ChoiceValue("second"), label="Second"),
            ),
        ),
    ),
)
def test_trusted_default_omits_each_optional_question_without_default(
    question: InputQuestion,
) -> None:
    """Omit absent optional defaults without converting them to decline."""
    created = replace(
        _created(),
        questions=(question,),
    )
    resolution = interaction_store._declared_trusted_default_resolution(
        created,
        _NOW,
    )
    assert resolution.answers == ()
    assert resolution.provenance is AnswerProvenance.TRUSTED_DEFAULT


def test_trusted_default_reducer_surfaces_transition_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate exact transition failure after deriving a valid default."""
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
    previous = apply_create_interaction(
        CreateInteractionCommand(actor=_actor(), request=created),
        _POLICY,
    ).record
    command = _trusted_default_command(
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
    )
    rejected = InputTransitionRejected(
        previous=previous.request,
        error=InputTransitionError(
            code=InputErrorCode.STALE_REVISION,
            path="expected_state_revision",
            message="stale",
        ),
    )
    monkeypatch.setattr(
        interaction_store,
        "resolve_request",
        lambda *args, **kwargs: rejected,
    )

    with pytest.raises(InputValidationError) as error:
        interaction_store._reduce_trusted_default_resolution(
            previous,
            command,
            _observation(1),
            _POLICY,
        )
    assert error.value.code is InputErrorCode.STALE_REVISION


def test_scope_digest_canonicalizes_mappings_and_rejects_unknown_values() -> (
    None
):
    """Canonicalize mapping order and reject unsupported snapshot members."""
    first = interaction_store._canonical_scope_snapshot_value({"b": 2, "a": 1})
    second = interaction_store._canonical_scope_snapshot_value(
        {
            "a": 1,
            "b": 2,
        }
    )
    assert first == second
    with pytest.raises(InputValidationError) as error:
        interaction_store._canonical_scope_snapshot_value(object())
    assert error.value.code is InputErrorCode.INVALID_TYPE


def test_deadline_result_rejects_candidate_classification_metadata() -> None:
    """Keep candidate-only classifier evidence off deadline commits."""
    previous = _pending_record()
    command = _resolve_command(previous, "deadline-with-proof")
    observed_at = _observation(86_400)
    due = interaction_store._reduce_due_resolution(
        previous,
        observed_at,
        _POLICY,
    )
    assert due is not None
    binding = _new_task_input_classifier_binding(
        classifier_id=_POLICY.task_input_classifier_id,
        policy_revision=_POLICY.task_input_policy_revision,
    )
    with pytest.raises(InputValidationError) as error:
        validate_resolution_commit_transition(
            previous,
            due,
            ResolutionDecisionStage.DEADLINE,
            observed_at,
            _POLICY,
            command=command,
            idempotency_key=None,
            classifier_binding=binding,
        )
    assert error.value.path == "result.classification_proof"
