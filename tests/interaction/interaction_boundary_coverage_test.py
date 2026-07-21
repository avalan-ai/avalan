"""Cover defensive interaction handler, policy, and state boundaries."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from avalan.interaction import (
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AgentId,
    BranchId,
    ConfirmationQuestion,
    ContinuationId,
    ControllerId,
    DeadlineTiePolicy,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    HandlerLossDisposition,
    InputContinuationOutcome,
    InputDisconnectReason,
    InputErrorCode,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputRequest,
    InputRequestId,
    InputResumer,
    InputResumerRegistration,
    InputResumptionNotification,
    InputTransitionApplied,
    InputTransitionError,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchAuthorizationTarget,
    InteractionDisclosure,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestAuthorizationTarget,
    InteractionScopeAuthorizationTarget,
    InteractionTime,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    QuestionType,
    RequirementMode,
    RunId,
    StateRevision,
    StreamSessionId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TurnId,
    UserId,
    create_input_request,
    mark_request_pending,
)
from avalan.interaction.policy import (
    _SequencedControllerActivity,
    is_controller_activity_evidence,
)
from avalan.interaction.state import (
    _anchor_request_presentation,
    _InputTransitionResultBase,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _principal() -> PrincipalScope:
    return PrincipalScope(user_id=UserId("user-1"))


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


def _request(
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
    )


def _pending(
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    created = _request(mode)
    transition = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    return transition.request


def _classification_request() -> TaskInputClassificationRequest:
    return TaskInputClassificationRequest(
        value="candidate",
        request_id=InputRequestId("request-1"),
        candidate_digest="0" * 64,
        question_id=QuestionId("question-1"),
        semantic_type=QuestionType.TEXT,
        policy_revision="policy-1",
    )


def test_handler_context_rejects_invalid_request_and_feedback() -> None:
    """Reject unchecked handler context members before presentation."""
    with pytest.raises(InputValidationError) as invalid_request:
        InputHandlerContext(request=cast(InputRequest, object()))
    with pytest.raises(InputValidationError) as invalid_feedback:
        InputHandlerContext(
            request=_request(),
            validation_error=cast(InputTransitionError, object()),
        )

    assert invalid_request.value.path == "handler.request"
    assert invalid_feedback.value.path == "handler.validation_error"


def test_handler_outcomes_reject_runtime_subclasses() -> None:
    """Reject subclasses that bypass the closed handler outcome union."""
    detached_type = type(
        "UncheckedDetached",
        (InputHandlerDetached,),
        {},
    )
    disconnected_type = type(
        "UncheckedDisconnected",
        (InputHandlerDisconnected,),
        {},
    )

    with pytest.raises(InputValidationError) as detached:
        detached_type()
    with pytest.raises(InputValidationError) as disconnected:
        disconnected_type(reason=InputDisconnectReason.HANDLER_UNAVAILABLE)

    assert detached.value.path == "handler.outcome"
    assert disconnected.value.path == "handler.outcome"


def test_resumption_rejects_invalid_outcome_and_non_callable() -> None:
    """Reject malformed continuation outcomes and non-callable resumers."""
    with pytest.raises(InputValidationError) as invalid_outcome:
        InputResumptionNotification(
            continuation_id=ContinuationId("continuation-1"),
            state_revision=StateRevision(1),
            outcome=cast(InputContinuationOutcome, object()),
        )
    with pytest.raises(InputValidationError) as invalid_resumer:
        InputResumerRegistration(
            continuation_id=ContinuationId("continuation-1"),
            resumer=cast(InputResumer, object()),
        )

    assert invalid_outcome.value.path == "resumption.outcome"
    assert invalid_resumer.value.path == "resumer.callback"


def test_resumer_registration_accepts_async_function() -> None:
    """Accept coroutine functions in addition to async callable objects."""

    async def resumer(notification: InputResumptionNotification) -> None:
        del notification

    registration = InputResumerRegistration(
        continuation_id=ContinuationId("continuation-1"),
        resumer=resumer,
    )

    assert registration.resumer is resumer


def test_time_rejects_negative_deadline_argument() -> None:
    """Reject a negative monotonic deadline before comparing it."""
    observed = InteractionTime.from_clock(
        wall_time=_NOW,
        monotonic_seconds=10,
    )

    with pytest.raises(InputValidationError) as captured:
        observed.monotonic_deadline_reached(-0.1)

    assert captured.value.code is InputErrorCode.OUT_OF_BOUNDS
    assert captured.value.path == "deadline.monotonic_seconds"


def test_authorization_targets_reject_invalid_identity_shapes() -> None:
    """Reject malformed request, branch, and execution-scope targets."""
    with pytest.raises(InputValidationError) as actor:
        InteractionActor(principal=cast(PrincipalScope, object()))
    with pytest.raises(InputValidationError) as request:
        InteractionRequestAuthorizationTarget(
            request_id=InputRequestId("request-1"),
            origin=cast(ExecutionOrigin, object()),
        )
    with pytest.raises(InputValidationError) as self_parent:
        InteractionBranchAuthorizationTarget(
            run_id=RunId("run-1"),
            branch_id=BranchId("branch-1"),
            parent_branch_id=BranchId("branch-1"),
            principal=_principal(),
        )
    with pytest.raises(InputValidationError) as branch_principal:
        InteractionBranchAuthorizationTarget(
            run_id=RunId("run-1"),
            branch_id=BranchId("branch-1"),
            parent_branch_id=BranchId("root"),
            principal=cast(PrincipalScope, object()),
        )
    with pytest.raises(InputValidationError) as scope_principal:
        InteractionScopeAuthorizationTarget(
            run_id=RunId("run-1"),
            principal=cast(PrincipalScope, object()),
        )
    with pytest.raises(InputValidationError) as descendants:
        InteractionScopeAuthorizationTarget(
            run_id=RunId("run-1"),
            principal=_principal(),
            include_descendants=True,
        )

    assert actor.value.path == "actor.principal"
    assert request.value.path == "authorization.origin"
    assert self_parent.value.path == "authorization.parent_branch_id"
    assert branch_principal.value.path == "authorization.principal"
    assert scope_principal.value.path == "authorization.principal"
    assert descendants.value.path == "authorization.include_descendants"


def test_authorization_decision_rejects_invalid_core_members() -> None:
    """Reject malformed actor, operation, target, and disclosure members."""
    actor = InteractionActor(principal=_principal())
    target = InteractionRequestAuthorizationTarget(
        request_id=InputRequestId("request-1"),
        origin=_origin(),
    )

    with pytest.raises(InputValidationError) as invalid_actor:
        InteractionAuthorizationDecision(
            actor=cast(InteractionActor, object()),
            operation=InteractionOperation.INSPECT,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )
    with pytest.raises(InputValidationError) as invalid_operation:
        InteractionAuthorizationDecision(
            actor=actor,
            operation=cast(InteractionOperation, object()),
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )
    with pytest.raises(InputValidationError) as invalid_target:
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.INSPECT,
            target=cast(InteractionAuthorizationTarget, object()),
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )
    with pytest.raises(InputValidationError) as invalid_disclosure:
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.INSPECT,
            target=target,
            allowed=True,
            disclosure=cast(InteractionDisclosure, object()),
        )

    assert invalid_actor.value.path == "authorization.actor"
    assert invalid_operation.value.path == "authorization.operation"
    assert invalid_target.value.path == "authorization.target"
    assert invalid_disclosure.value.path == "authorization.disclosure"


def test_authorization_decision_rejects_operation_target_mismatches() -> None:
    """Bind branch, scope, list, and request operations to exact targets."""
    actor = InteractionActor(principal=_principal())
    request_target = InteractionRequestAuthorizationTarget(
        request_id=InputRequestId("request-1"),
        origin=_origin(),
    )
    branch_target = InteractionBranchAuthorizationTarget(
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        parent_branch_id=BranchId("root"),
        principal=_principal(),
    )
    scope_target = InteractionScopeAuthorizationTarget(
        run_id=RunId("run-1"),
        principal=_principal(),
    )

    with pytest.raises(InputValidationError) as branch:
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.REGISTER_BRANCH,
            target=request_target,
            allowed=True,
            disclosure=InteractionDisclosure.NONE,
        )
    with pytest.raises(InputValidationError) as listing:
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.LIST,
            target=branch_target,
            allowed=True,
            disclosure=InteractionDisclosure.NONE,
        )
    with pytest.raises(InputValidationError) as request:
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.RESOLVE,
            target=scope_target,
            allowed=True,
            disclosure=InteractionDisclosure.NONE,
        )

    assert branch.value.path == "authorization.target"
    assert listing.value.path == "authorization.target"
    assert request.value.path == "authorization.target"


def test_classification_request_rejects_invalid_content_metadata() -> None:
    """Reject malformed classifier value, digest, and semantic type."""
    valid = _classification_request()

    with pytest.raises(InputValidationError) as value:
        replace(valid, value=cast(str, object()))
    with pytest.raises(InputValidationError) as digest:
        replace(valid, candidate_digest="A" * 64)
    with pytest.raises(InputValidationError) as semantic_type:
        replace(valid, semantic_type=cast(QuestionType, object()))

    assert value.value.path == "classification.value"
    assert digest.value.path == "classification.candidate_digest"
    assert semantic_type.value.path == "classification.semantic_type"


def test_classification_output_rejects_invalid_metadata() -> None:
    """Reject malformed metadata while keeping classifier output unsealed."""
    request = _classification_request()

    with pytest.raises(InputValidationError) as invalid_decision:
        TaskInputClassification(
            classifier_id="classifier-1",
            decision=TaskInputClassificationDecision.ALLOW,
            classification_id="classification-1",
            policy_revision=request.policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=cast(QuestionType, object()),
        )
    with pytest.raises(InputValidationError) as invalid_classifier:
        TaskInputClassification(
            classifier_id="",
            decision=cast(TaskInputClassificationDecision, object()),
            classification_id="classification-1",
            policy_revision=request.policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )

    assert invalid_decision.value.path == "classification.semantic_type"
    assert invalid_classifier.value.path == "classification.decision"


def test_sequenced_activity_base_is_not_a_supported_variant() -> None:
    """Reject direct construction of the untagged sequenced activity base."""
    with pytest.raises(InputValidationError) as captured:
        _SequencedControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=1,
        )

    assert captured.value.path == "activity"


def test_controller_activity_evidence_rejects_unrelated_objects() -> None:
    """Recognize exact acquisition evidence and reject unrelated objects."""
    acquire = AcquireControllerActivity(
        request_id=InputRequestId("request-1"),
        controller_id=ControllerId("controller-1"),
    )

    assert is_controller_activity_evidence(acquire)
    assert not is_controller_activity_evidence(object())


@pytest.mark.parametrize(
    "field_name",
    (
        "attached_loss_with_resumer",
        "attached_loss_without_resumer",
    ),
)
def test_policy_rejects_invalid_handler_loss_type(field_name: str) -> None:
    """Reject malformed handler-loss dispositions in either policy slot."""
    if field_name == "attached_loss_with_resumer":
        with pytest.raises(InputValidationError) as captured:
            InteractionPolicy(
                attached_loss_with_resumer=cast(
                    HandlerLossDisposition,
                    object(),
                )
            )
    else:
        with pytest.raises(InputValidationError) as captured:
            InteractionPolicy(
                attached_loss_without_resumer=cast(
                    HandlerLossDisposition,
                    object(),
                )
            )

    assert captured.value.path == f"policy.{field_name}"


def test_policy_rejects_invalid_resumer_and_deadline_invariants() -> None:
    """Reject non-detach resumer handling and non-deadline-first equality."""
    with pytest.raises(InputValidationError) as resumer:
        InteractionPolicy(
            attached_loss_with_resumer=HandlerLossDisposition.UNAVAILABLE
        )
    with pytest.raises(InputValidationError) as deadline:
        InteractionPolicy(deadline_tie=cast(DeadlineTiePolicy, object()))

    assert resumer.value.path == "policy.attached_loss_with_resumer"
    assert deadline.value.path == "policy.deadline_tie"


def test_presentation_anchor_rejects_invalid_lifecycle_inputs() -> None:
    """Reject invalid, non-pending, and already-anchored presentation input."""
    with pytest.raises(InputValidationError) as invalid_request:
        _anchor_request_presentation(cast(InputRequest, object()), _NOW)
    with pytest.raises(InputValidationError) as created:
        _anchor_request_presentation(_request(), _NOW)

    pending = _pending(RequirementMode.ADVISORY)
    anchored = _anchor_request_presentation(pending, _NOW)
    with pytest.raises(InputValidationError) as repeated:
        _anchor_request_presentation(anchored, _NOW)

    assert invalid_request.value.path == "request"
    assert created.value.path == "request.state"
    assert repeated.value.path == "request.advisory_deadline"


def test_transition_result_base_rejects_direct_instantiation() -> None:
    """Reject construction outside the closed transition-result variants."""
    with pytest.raises(InputValidationError) as captured:
        _InputTransitionResultBase(previous=_request())

    assert captured.value.path == "transition"
