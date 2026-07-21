"""Exercise the provider-free structured-input domain lifecycle."""

from datetime import UTC, datetime, timedelta

from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    Choice,
    ChoiceValue,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputAnsweredResult,
    InputRequestId,
    InputTransitionApplied,
    ModelCallId,
    QuestionId,
    RequirementMode,
    ResumeInputContinuation,
    RunId,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    TurnId,
    create_input_request,
    decode_input_model_result,
    decode_input_request,
    decode_input_resolution,
    encode_input_model_result,
    encode_input_request,
    encode_input_resolution,
    mark_request_pending,
    project_resolution_to_model,
    resolve_request,
)


def test_provider_free_request_resolution_lifecycle() -> None:
    """Round-trip, resolve, and project one correlated typed request."""
    created_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    origin = ExecutionOrigin(
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
            model_config_reference="model-config-1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )
    request = create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Choose the execution strategy.",
        questions=(
            SingleSelectionQuestion(
                question_id=QuestionId("strategy"),
                prompt="Which strategy?",
                required=True,
                choices=(
                    Choice(value=ChoiceValue("fast"), label="Fast"),
                    Choice(value=ChoiceValue("safe"), label="Safe"),
                ),
            ),
        ),
        created_at=created_at,
    )

    restored = decode_input_request(encode_input_request(request))
    pending_transition = mark_request_pending(
        restored,
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(pending_transition, InputTransitionApplied)

    resolution = AnsweredResolution(
        request_id=restored.request_id,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        resolved_at=created_at + timedelta(seconds=1),
        answers=(
            SingleSelectionAnswer(
                question_id=QuestionId("strategy"),
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("safe")),
            ),
        ),
    )
    restored_resolution = decode_input_resolution(
        encode_input_resolution(resolution)
    )
    resolved_transition = resolve_request(
        pending_transition.request,
        restored_resolution,
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(resolved_transition, InputTransitionApplied)

    outcome = project_resolution_to_model(
        resolved_transition.request,
        containing_run_exists=True,
    )
    assert isinstance(outcome, ResumeInputContinuation)
    assert isinstance(outcome.result, InputAnsweredResult)
    assert (
        decode_input_model_result(encode_input_model_result(outcome.result))
        == outcome.result
    )
