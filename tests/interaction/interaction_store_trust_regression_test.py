"""Regress store-owned classification and resolution trust boundaries."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from inspect import signature
from typing import cast

import pytest

import avalan.interaction as interaction_api
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelInteractionApplied,
    CancelInteractionCommand,
    CancelledResolution,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputErrorCode,
    InputQuestion,
    InputRequestId,
    InputTransitionApplied,
    InputValidationError,
    InteractionActor,
    InteractionCorrelation,
    InteractionExecutionScope,
    InteractionPolicy,
    InteractionPresentationState,
    InteractionRecord,
    InteractionStoreRevision,
    InteractionTime,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalAuthoredProvenance,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    RunId,
    ScopeCancellationApplied,
    ScopeSupersessionApplied,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TerminalizeInteractionApplied,
    TerminalizeInteractionCommand,
    TerminalizeInteractionScopeCommand,
    TextAnswer,
    TextQuestion,
    TrustedDefaultResolutionApplied,
    TrustedDefaultResolutionRequest,
    TurnId,
    UnavailableResolution,
    UserId,
    apply_candidate_resolution,
    apply_request_cancellation,
    apply_request_terminalization,
    apply_trusted_default_resolution,
    create_input_request,
    mark_request_pending,
    semantic_request_fingerprint,
)
from avalan.interaction import store as interaction_store
from avalan.interaction.store import (
    TrustedDefaultResolutionCommand,
    _apply_scope_cancellation,
    _apply_scope_supersession,
    _begin_scope_transaction,
    _bind_task_input_classifications,
    _BoundTaskInputClassifications,
    _insert_interaction_store_backing_record,
    _InteractionStoreBacking,
    _new_interaction_store_backing,
    _new_task_input_classifier_binding,
    _new_trusted_default_resolution_command,
    _task_input_classification_requests,
    _TaskInputClassifierBinding,
    _validate_trusted_default_resolution_command,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
_POLICY = InteractionPolicy()


def _principal() -> PrincipalScope:
    return PrincipalScope(user_id=UserId("trust-user"))


def _actor() -> InteractionActor:
    return InteractionActor(principal=_principal())


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
        run_id=RunId("trust-run"),
        turn_id=TurnId("trust-turn"),
        agent_id=AgentId("trust-agent"),
        branch_id=BranchId("trust-branch"),
        model_call_id=ModelCallId("trust-call"),
        stream_session_id=StreamSessionId("trust-stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://trust-regression",
            agent_definition_revision="agent-r1",
            operation_id="trust-operation",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capability-r1",
        ),
        principal=_principal(),
    )


def _observation(seconds: float = 1) -> InteractionTime:
    return InteractionTime.from_clock(
        wall_time=_NOW + timedelta(seconds=seconds),
        monotonic_seconds=seconds,
    )


def _pending_record(
    questions: tuple[InputQuestion, ...],
    *,
    request_id: str = "trust-request",
) -> InteractionRecord:
    created = create_input_request(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"{request_id}-continuation"),
        origin=_origin(),
        mode=RequirementMode.REQUIRED,
        reason="Exercise the store trust boundary.",
        questions=questions,
        created_at=_NOW,
    )
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
        presentation=InteractionPresentationState.PRESENTED,
        store_revision=InteractionStoreRevision(1),
    )


def _classified_candidate() -> tuple[
    InteractionRecord,
    ResolveInteractionCommand,
]:
    previous = _pending_record(
        (
            TextQuestion(
                question_id=QuestionId("first-text"),
                prompt="First value?",
                required=True,
            ),
            MultilineTextQuestion(
                question_id=QuestionId("second-text"),
                prompt="Second value?",
                required=True,
            ),
        )
    )
    resolution = AnsweredResolution(
        request_id=previous.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(
            TextAnswer(
                question_id=QuestionId("first-text"),
                provenance=AnswerProvenance.HUMAN,
                value="first classified value",
            ),
            MultilineTextAnswer(
                question_id=QuestionId("second-text"),
                provenance=AnswerProvenance.HUMAN,
                value="second classified value",
            ),
        ),
    )
    return previous, ResolveInteractionCommand(
        actor=_actor(),
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("trust-resolution"),
        proposed_resolution=resolution,
    )


def _classifier_binding(
    *,
    classifier_id: str | None = None,
    policy_revision: str | None = None,
) -> _TaskInputClassifierBinding:
    return _new_task_input_classifier_binding(
        classifier_id=(
            _POLICY.task_input_classifier_id
            if classifier_id is None
            else classifier_id
        ),
        policy_revision=(
            _POLICY.task_input_policy_revision
            if policy_revision is None
            else policy_revision
        ),
    )


def _classification_outputs(
    previous: InteractionRecord,
    command: ResolveInteractionCommand,
    binding: _TaskInputClassifierBinding,
) -> tuple[TaskInputClassification, ...]:
    requests = _task_input_classification_requests(
        previous,
        command,
        _POLICY,
    )
    return tuple(
        TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=binding.classifier_id,
            classification_id=f"classification-{index}",
            policy_revision=binding.policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )
        for index, request in enumerate(requests, start=1)
    )


def _bound_proof(
    previous: InteractionRecord,
    command: ResolveInteractionCommand,
    binding: _TaskInputClassifierBinding,
) -> _BoundTaskInputClassifications:
    return _bind_task_input_classifications(
        binding,
        previous,
        command,
        _classification_outputs(previous, command, binding),
        _POLICY,
    )


def test_current_classifier_binding_and_exact_proof_commit() -> None:
    """Accept only current output sealed to the exact candidate."""
    previous, command = _classified_candidate()
    binding = _classifier_binding()
    proof = _bound_proof(previous, command, binding)

    result = apply_candidate_resolution(
        previous,
        command,
        _observation(),
        _POLICY,
        classifier_binding=binding,
        classification_proof=proof,
    )

    assert isinstance(result, ResolveInteractionApplied)
    resolution = result.record.request.resolution
    assert isinstance(resolution, AnsweredResolution)
    assert isinstance(command.proposed_resolution, AnsweredResolution)
    assert resolution.answers == command.proposed_resolution.answers
    assert resolution.provenance is command.proposed_resolution.provenance
    assert resolution.resolved_at == _observation().wall_time
    assert result.record.resolved_by == _principal()


@pytest.mark.parametrize(
    ("classifier_id", "policy_revision"),
    (
        ("retired-classifier", _POLICY.task_input_policy_revision),
        (_POLICY.task_input_classifier_id, "retired-policy"),
    ),
)
def test_stale_classifier_binding_identity_or_revision_fails_closed(
    classifier_id: str,
    policy_revision: str,
) -> None:
    """Reject bindings that no longer match current policy identity."""
    previous, command = _classified_candidate()
    binding = _classifier_binding(
        classifier_id=classifier_id,
        policy_revision=policy_revision,
    )

    with pytest.raises(InputValidationError) as error:
        _bound_proof(previous, command, binding)

    assert error.value.code is InputErrorCode.STALE_REVISION
    assert error.value.path == "classifier_binding"


@pytest.mark.parametrize(
    ("field_name", "stale_value"),
    (
        ("classifier_id", "retired-classifier"),
        ("policy_revision", "retired-policy"),
    ),
)
def test_stale_classifier_output_identity_or_revision_fails_closed(
    field_name: str,
    stale_value: str,
) -> None:
    """Revalidate untrusted output identity at the sealing boundary."""
    previous, command = _classified_candidate()
    binding = _classifier_binding()
    outputs = _classification_outputs(previous, command, binding)
    if field_name == "classifier_id":
        stale = replace(outputs[0], classifier_id=stale_value)
    else:
        assert field_name == "policy_revision"
        stale = replace(outputs[0], policy_revision=stale_value)

    with pytest.raises(InputValidationError) as error:
        _bind_task_input_classifications(
            binding,
            previous,
            command,
            (stale, outputs[1]),
            _POLICY,
        )

    assert error.value.code is InputErrorCode.STALE_REVISION
    assert error.value.path == "classifications"


def test_classifier_proof_capability_cannot_cross_equal_bindings() -> None:
    """Bind proof authority by capability identity, not string equality."""
    previous, command = _classified_candidate()
    original = _classifier_binding()
    equal_but_distinct = _classifier_binding()
    proof = _bound_proof(previous, command, original)

    with pytest.raises(InputValidationError) as error:
        apply_candidate_resolution(
            previous,
            command,
            _observation(),
            _POLICY,
            classifier_binding=equal_but_distinct,
            classification_proof=proof,
        )

    assert error.value.code is InputErrorCode.FORBIDDEN
    assert error.value.path == "classification_proof"


def test_swapped_classifier_outputs_are_not_sealed() -> None:
    """Reject correctly signed values paired with another question."""
    previous, command = _classified_candidate()
    binding = _classifier_binding()
    outputs = _classification_outputs(previous, command, binding)

    with pytest.raises(InputValidationError) as error:
        _bind_task_input_classifications(
            binding,
            previous,
            command,
            tuple(reversed(outputs)),
            _POLICY,
        )

    assert error.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert error.value.path == "classifications"


def test_forged_classifier_proof_cannot_reach_candidate_commit() -> None:
    """Reject both token forgery and an unsealed public output envelope."""
    previous, command = _classified_candidate()
    binding = _classifier_binding()
    outputs = _classification_outputs(previous, command, binding)

    with pytest.raises(InputValidationError) as token_error:
        _BoundTaskInputClassifications(
            binding=binding,
            request_id=command.correlation.request_id,
            candidate_digest=command.resolution_digest,
            classifications=outputs,
            _token=object(),
        )
    assert token_error.value.code is InputErrorCode.FORBIDDEN

    with pytest.raises(InputValidationError) as envelope_error:
        apply_candidate_resolution(
            previous,
            command,
            _observation(),
            _POLICY,
            classifier_binding=binding,
            classification_proof=cast(
                _BoundTaskInputClassifications,
                outputs[0],
            ),
        )
    assert (
        envelope_error.value.code
        is InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE
    )
    assert envelope_error.value.path == "classification_proof"


def test_classifier_proof_rejects_candidate_content_drift() -> None:
    """Prevent replaying genuine proof after candidate content changes."""
    previous, command = _classified_candidate()
    binding = _classifier_binding()
    proof = _bound_proof(previous, command, binding)
    resolution = command.proposed_resolution
    assert isinstance(resolution, AnsweredResolution)
    first = resolution.answers[0]
    assert isinstance(first, TextAnswer)
    changed = replace(
        command,
        proposed_resolution=replace(
            resolution,
            answers=(
                replace(first, value="candidate changed after proof"),
                resolution.answers[1],
            ),
        ),
    )

    with pytest.raises(InputValidationError) as error:
        apply_candidate_resolution(
            previous,
            changed,
            _observation(),
            _POLICY,
            classifier_binding=binding,
            classification_proof=proof,
        )

    assert error.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert error.value.path == "classification_proof"


def test_public_types_expose_no_self_sealing_factories() -> None:
    """Keep trust minting absent from the public value and trigger types."""
    assert not hasattr(TaskInputClassification, "_from_classifier")
    assert not hasattr(interaction_api, "TrustedDefaultResolutionCommand")
    assert not hasattr(
        interaction_api, "_new_trusted_default_resolution_command"
    )
    assert not hasattr(interaction_api, "_bind_task_input_classifications")
    assert not hasattr(interaction_api, "_new_task_input_classifier_binding")
    assert (
        "proposed_resolution"
        not in signature(TrustedDefaultResolutionRequest).parameters
    )

    constructor = cast(
        Callable[..., TrustedDefaultResolutionRequest],
        TrustedDefaultResolutionRequest,
    )
    previous = _pending_record(
        (
            ConfirmationQuestion(
                question_id=QuestionId("default-confirm"),
                prompt="Continue?",
                required=True,
                default_value=False,
            ),
        )
    )
    with pytest.raises(TypeError):
        constructor(
            actor=_actor(),
            correlation=previous.correlation,
            expected_state_revision=previous.request.state_revision,
            proposed_resolution=DeclinedResolution(
                request_id=previous.request.request_id,
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                resolved_at=_NOW,
            ),
        )


def test_trusted_default_seal_rejects_invalid_internal_values() -> None:
    """Reject invalid factory input and a command with a corrupted seal."""
    with pytest.raises(InputValidationError) as invalid_request:
        _new_trusted_default_resolution_command(
            cast(TrustedDefaultResolutionRequest, object())
        )
    assert invalid_request.value.code is InputErrorCode.INVALID_TYPE
    assert invalid_request.value.path == "trusted_default.request"

    previous = _pending_record(
        (
            ConfirmationQuestion(
                question_id=QuestionId("default-confirm"),
                prompt="Continue?",
                required=True,
                default_value=False,
            ),
        )
    )
    command = _trusted_default_command(
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
    )
    object.__setattr__(command, "_authority", object())

    with pytest.raises(InputValidationError) as invalid_authority:
        _validate_trusted_default_resolution_command(command)
    assert invalid_authority.value.code is InputErrorCode.FORBIDDEN
    assert invalid_authority.value.path == "trusted_default.authority"


def _apply_defaults(
    questions: tuple[InputQuestion, ...],
    *,
    request_id: str,
) -> TrustedDefaultResolutionApplied:
    previous = _pending_record(questions, request_id=request_id)
    command = _trusted_default_command(
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
    )
    result = apply_trusted_default_resolution(
        previous,
        command,
        _observation(),
        _POLICY,
    )
    assert isinstance(result, TrustedDefaultResolutionApplied)
    return result


def test_trusted_default_trigger_derives_false_and_empty_text_in_order() -> (
    None
):
    """Derive false and empty text from stored questions in their order."""
    result = _apply_defaults(
        (
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=False,
            ),
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Optional label?",
                required=False,
                default_value="",
            ),
            MultilineTextQuestion(
                question_id=QuestionId("multiline"),
                prompt="Optional details?",
                required=False,
                default_value="",
            ),
        ),
        request_id="scalar-defaults",
    )

    resolution = result.record.request.resolution
    assert isinstance(resolution, AnsweredResolution)
    assert resolution.provenance is AnswerProvenance.TRUSTED_DEFAULT
    assert tuple(answer.question_id for answer in resolution.answers) == (
        QuestionId("confirm"),
        QuestionId("text"),
        QuestionId("multiline"),
    )
    confirmation, text, multiline = resolution.answers
    assert isinstance(confirmation, ConfirmationAnswer)
    assert confirmation.value is False
    assert isinstance(text, TextAnswer)
    assert text.value == ""
    assert isinstance(multiline, MultilineTextAnswer)
    assert multiline.value == ""
    assert all(
        answer.provenance is AnswerProvenance.TRUSTED_DEFAULT
        for answer in resolution.answers
    )
    assert result.record.idempotency_ledger == ()
    assert (
        result.record.resolved_by
        is interaction_store._TRUSTED_DEFAULT_RESOLVER
    )


def test_trusted_default_trigger_derives_selection_defaults_in_order() -> None:
    """Derive one selected choice and an empty ordered choice tuple."""
    choices = (
        Choice(value=ChoiceValue("alpha"), label="Alpha"),
        Choice(value=ChoiceValue("beta"), label="Beta"),
    )
    result = _apply_defaults(
        (
            SingleSelectionQuestion(
                question_id=QuestionId("single"),
                prompt="Select one.",
                required=True,
                choices=choices,
                default_value=ChoiceValue("beta"),
            ),
            MultipleSelectionQuestion(
                question_id=QuestionId("multiple"),
                prompt="Select any.",
                required=False,
                choices=choices,
                default_value=(),
            ),
        ),
        request_id="selection-defaults",
    )

    resolution = result.record.request.resolution
    assert isinstance(resolution, AnsweredResolution)
    assert tuple(answer.question_id for answer in resolution.answers) == (
        QuestionId("single"),
        QuestionId("multiple"),
    )
    single, multiple = resolution.answers
    assert isinstance(single, SingleSelectionAnswer)
    assert single.value == SelectedChoice(value=ChoiceValue("beta"))
    assert isinstance(multiple, MultipleSelectionAnswer)
    assert multiple.values == ()
    assert all(
        answer.provenance is AnswerProvenance.TRUSTED_DEFAULT
        for answer in resolution.answers
    )


def _questions_without_required_default() -> tuple[InputQuestion, ...]:
    choices = (
        Choice(value=ChoiceValue("alpha"), label="Alpha"),
        Choice(value=ChoiceValue("beta"), label="Beta"),
    )
    return (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        ),
        TextQuestion(
            question_id=QuestionId("text"),
            prompt="Text?",
            required=True,
        ),
        MultilineTextQuestion(
            question_id=QuestionId("multiline"),
            prompt="Details?",
            required=True,
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("single"),
            prompt="Select one.",
            required=True,
            choices=choices,
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("multiple"),
            prompt="Select one or more.",
            required=True,
            choices=choices,
        ),
    )


@pytest.mark.parametrize("question", _questions_without_required_default())
def test_trusted_default_trigger_rejects_each_missing_required_default(
    question: InputQuestion,
) -> None:
    """Reject rather than infer when a required default is absent."""
    previous = _pending_record(
        (question,),
        request_id=f"missing-{question.question_id}",
    )
    command = _trusted_default_command(
        correlation=previous.correlation,
        expected_state_revision=previous.request.state_revision,
    )
    original = previous

    with pytest.raises(InputValidationError) as error:
        apply_trusted_default_resolution(
            previous,
            command,
            _observation(),
            _POLICY,
        )

    assert error.value.code is InputErrorCode.INVALID_DEFAULT
    assert error.value.path == "request.questions"
    assert previous == original


PrincipalCommandFactory = Callable[
    [PrincipalAuthoredProvenance],
    object,
]


def _principal_command_factories(
    previous: InteractionRecord,
) -> tuple[tuple[str, PrincipalCommandFactory], ...]:
    scope = InteractionExecutionScope(run_id=previous.request.origin.run_id)
    return (
        (
            "cancel",
            lambda provenance: CancelInteractionCommand(
                actor=_actor(),
                correlation=previous.correlation,
                provenance=provenance,
            ),
        ),
        (
            "terminalize",
            lambda provenance: TerminalizeInteractionCommand(
                actor=_actor(),
                correlation=previous.correlation,
                status=ResolutionStatus.UNAVAILABLE,
                provenance=provenance,
            ),
        ),
        (
            "cancel-scope",
            lambda provenance: TerminalizeInteractionScopeCommand(
                actor=_actor(),
                scope=scope,
                provenance=provenance,
            ),
        ),
        (
            "supersede-scope",
            lambda provenance: SupersedeInteractionScopeCommand(
                actor=_actor(),
                scope=scope,
                provenance=provenance,
            ),
        ),
    )


@pytest.mark.parametrize(
    "provenance",
    (AnswerProvenance.POLICY, AnswerProvenance.TRUSTED_DEFAULT),
)
def test_principal_commands_reject_trusted_provenance_before_any_write(
    provenance: AnswerProvenance,
) -> None:
    """Fail every principal mutation before it can reach persistence."""
    previous = _pending_record(
        (
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        )
    )
    writes: list[object] = []

    for operation, factory in _principal_command_factories(previous):
        with pytest.raises(InputValidationError) as error:
            command = factory(cast(PrincipalAuthoredProvenance, provenance))
            writes.append(command)
        assert error.value.code is InputErrorCode.FORBIDDEN, operation
        assert error.value.path.endswith("provenance"), operation

    assert writes == []
    assert previous.request.resolution is None
    assert previous.store_revision == InteractionStoreRevision(1)


@pytest.mark.parametrize(
    "provenance",
    (AnswerProvenance.HUMAN, AnswerProvenance.EXTERNAL_CONTROLLER),
)
def test_valid_principal_provenance_constructs_every_command(
    provenance: PrincipalAuthoredProvenance,
) -> None:
    """Keep both supported external authority categories available."""
    previous = _pending_record(
        (
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        )
    )

    for _, factory in _principal_command_factories(previous):
        command = factory(provenance)
        assert getattr(command, "provenance") is provenance


def test_valid_direct_and_internal_resolution_paths_keep_authority() -> None:
    """Persist principal and trusted-system authority without relabeling."""
    questions: tuple[InputQuestion, ...] = (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        ),
    )
    cancelled_previous = _pending_record(
        questions,
        request_id="human-cancel",
    )
    cancelled = apply_request_cancellation(
        cancelled_previous,
        CancelInteractionCommand(
            actor=_actor(),
            correlation=cancelled_previous.correlation,
            provenance=AnswerProvenance.HUMAN,
        ),
        _observation(),
        _POLICY,
    )
    assert isinstance(cancelled, CancelInteractionApplied)
    assert isinstance(cancelled.record.request.resolution, CancelledResolution)
    assert (
        cancelled.record.request.resolution.provenance
        is AnswerProvenance.HUMAN
    )
    assert cancelled.record.resolved_by == _principal()

    unavailable_previous = _pending_record(
        questions,
        request_id="external-unavailable",
    )
    unavailable = apply_request_terminalization(
        unavailable_previous,
        TerminalizeInteractionCommand(
            actor=_actor(),
            correlation=unavailable_previous.correlation,
            status=ResolutionStatus.UNAVAILABLE,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        ),
        _observation(),
        _POLICY,
    )
    assert isinstance(unavailable, TerminalizeInteractionApplied)
    assert isinstance(
        unavailable.record.request.resolution,
        UnavailableResolution,
    )
    assert (
        unavailable.record.request.resolution.provenance
        is AnswerProvenance.EXTERNAL_CONTROLLER
    )
    assert unavailable.record.resolved_by == _principal()


def _apply_scope_command(
    previous: InteractionRecord,
    command: (
        TerminalizeInteractionScopeCommand | SupersedeInteractionScopeCommand
    ),
    backing: _InteractionStoreBacking,
) -> ScopeCancellationApplied | ScopeSupersessionApplied:
    _insert_interaction_store_backing_record(backing, previous)
    transaction = _begin_scope_transaction(backing, command)
    if isinstance(command, TerminalizeInteractionScopeCommand):
        cancellation_result = _apply_scope_cancellation(
            transaction,
            command,
            _observation(),
            _POLICY,
            backing=backing,
        )
        assert isinstance(cancellation_result, ScopeCancellationApplied)
        return cancellation_result
    supersession_result = _apply_scope_supersession(
        transaction,
        command,
        _observation(),
        _POLICY,
        backing=backing,
    )
    assert isinstance(supersession_result, ScopeSupersessionApplied)
    return supersession_result


def test_valid_scope_paths_keep_human_and_external_provenance() -> None:
    """Apply both scope operations with supported principal provenance."""
    questions: tuple[InputQuestion, ...] = (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        ),
    )
    scope = InteractionExecutionScope(run_id=RunId("trust-run"))
    cancellation_previous = _pending_record(
        questions,
        request_id="scope-cancel",
    )
    cancellation = _apply_scope_command(
        cancellation_previous,
        TerminalizeInteractionScopeCommand(
            actor=_actor(),
            scope=scope,
            provenance=AnswerProvenance.HUMAN,
        ),
        _new_interaction_store_backing(),
    )
    assert isinstance(cancellation, ScopeCancellationApplied)
    cancelled = cancellation.records[0].request.resolution
    assert isinstance(cancelled, CancelledResolution)
    assert cancelled.provenance is AnswerProvenance.HUMAN

    supersession_previous = _pending_record(
        questions,
        request_id="scope-supersede",
    )
    supersession = _apply_scope_command(
        supersession_previous,
        SupersedeInteractionScopeCommand(
            actor=_actor(),
            scope=scope,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        ),
        _new_interaction_store_backing(),
    )
    assert isinstance(supersession, ScopeSupersessionApplied)
    superseded = supersession.records[0].request.resolution
    assert superseded is not None
    assert superseded.provenance is AnswerProvenance.EXTERNAL_CONTROLLER
