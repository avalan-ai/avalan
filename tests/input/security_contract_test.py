"""Exercise interaction-class security separation."""

from datetime import UTC, datetime

import pytest

from avalan.interaction import (
    AgentId,
    BranchId,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputCodecError,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InteractionClass,
    ModelCallId,
    QuestionId,
    RequirementMode,
    RunId,
    StreamSessionId,
    TextQuestion,
    TurnId,
    create_input_request,
    decode_input_request,
    encode_input_request,
)


def test_requirement_input_n_015() -> None:
    """Keep task input separate from approval, steering, and authentication."""
    origin = ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://safe",
            agent_definition_revision="r1",
            operation_id="op",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )
    request = create_input_request(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Need non-secret task context.",
        questions=(
            TextQuestion(
                question_id=QuestionId("context"),
                prompt="Which public environment?",
                required=True,
            ),
        ),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert request.interaction_class is InteractionClass.TASK_INPUT
    assert "interaction_class" not in {
        field.name
        for field in InputRequest.__dataclass_fields__.values()
        if field.init
    }
    forged = encode_input_request(request)
    for prohibited in (
        InteractionClass.ACTION_APPROVAL,
        InteractionClass.STEERING,
        InteractionClass.AUTHENTICATION,
    ):
        forged["interaction_class"] = prohibited.value
        with pytest.raises(InputCodecError):
            decode_input_request(forged)


def test_prohibited_tags_are_rejected_without_scanning_free_text() -> None:
    """Reject explicit secret semantics while leaving text to host policy."""
    origin = ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://safe",
            agent_definition_revision="r1",
            operation_id="op",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )
    request = create_input_request(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Explain the boundary without submitting a secret.",
        questions=(
            TextQuestion(
                question_id=QuestionId("context"),
                prompt="Explain why password and token collection is unsafe.",
                required=True,
            ),
        ),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    wire = encode_input_request(request)

    assert decode_input_request(wire) == request
    questions = wire["questions"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    for field_name, tag in (
        ("kind", "password"),
        ("kind", "api_key"),
        ("kind", "token"),
        ("kind", "private_key"),
        ("kind", "payment"),
        ("kind", "mfa"),
        ("semantic_type", "authentication_challenge"),
    ):
        forged_question = dict(question)
        forged_question[field_name] = tag
        forged = dict(wire)
        forged["questions"] = [forged_question]
        with pytest.raises(InputCodecError) as error:
            decode_input_request(forged)
        assert error.value.code is InputErrorCode.PROHIBITED_INPUT
