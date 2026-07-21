"""Exercise the public canonical interaction contract."""

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

from avalan.interaction import (
    RESERVED_INPUT_CAPABILITY_NAME,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelledResolution,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    FreeFormOther,
    HostCapabilities,
    HostHandling,
    InputAnsweredResult,
    InputCodecError,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputTimedOutResult,
    InputTransitionApplied,
    InputTransitionRejected,
    InteractionClass,
    ModelCallId,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PresentationHint,
    QuestionId,
    QuestionType,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    RunId,
    SelectedChoice,
    SelectionValidationConstraints,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TaskId,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    create_input_request,
    decode_input_question,
    decode_input_request,
    encode_input_question,
    encode_input_request,
    mark_request_pending,
    project_resolution_to_model,
    resolve_request,
    semantic_request_fingerprint,
)

_CREATED_AT = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        parent_branch_id=BranchId("branch-parent"),
        model_call_id=ModelCallId("model-call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://support",
            agent_definition_revision="agent-r1",
            operation_id="operation-1",
            operation_index=2,
            model_config_reference="model-config-1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )


def _choices(*, second_label: str = "Careful") -> tuple[Choice, ...]:
    return (
        Choice(
            value=ChoiceValue("fast"),
            label="Fast",
            description="Finish sooner with fewer checks.",
        ),
        Choice(
            value=ChoiceValue("careful"),
            label=second_label,
            description="Run the complete validation set.",
        ),
    )


def _questions() -> tuple[
    ConfirmationQuestion,
    SingleSelectionQuestion,
    MultipleSelectionQuestion,
]:
    choices = _choices()
    return (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
            default_value=True,
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Choose a strategy.",
            required=True,
            header="Strategy",
            help_text="This controls validation depth.",
            presentation_hint=PresentationHint.RADIO,
            choices=choices,
            recommended_choice=ChoiceValue("careful"),
            default_value=ChoiceValue("careful"),
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Choose checks.",
            required=False,
            choices=choices,
            allow_other=True,
            default_value=(ChoiceValue("fast"),),
            constraints=SelectionValidationConstraints(
                minimum=0,
                maximum=3,
            ),
            presentation_hint=PresentationHint.CHECKBOX,
        ),
    )


def _request(
    *,
    questions: tuple[InputQuestion, ...] | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    typed_questions = _questions() if questions is None else questions
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=mode,
        reason="A decision is required to continue safely.",
        questions=typed_questions,
        created_at=_CREATED_AT,
    )


def _pending(request: InputRequest) -> InputRequest:
    result = mark_request_pending(
        request,
        expected_state_revision=StateRevision(0),
        presented_at=(
            request.created_at
            if request.mode is RequirementMode.ADVISORY
            else None
        ),
    )
    assert isinstance(result, InputTransitionApplied)
    assert result.mutation_applied
    return result.request


def _answered(request: InputRequest) -> AnsweredResolution:
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        resolved_at=_CREATED_AT + timedelta(seconds=1),
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
            SingleSelectionAnswer(
                question_id=QuestionId("strategy"),
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("careful")),
            ),
            MultipleSelectionAnswer(
                question_id=QuestionId("checks"),
                provenance=AnswerProvenance.POLICY,
                values=(
                    SelectedChoice(value=ChoiceValue("fast")),
                    FreeFormOther(text="security"),
                ),
            ),
        ),
    )


def test_requirement_input_n_004() -> None:
    """Describe every supported semantic input type."""
    question_types = {
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        ).kind,
        TextQuestion(
            question_id=QuestionId("name"),
            prompt="Name?",
            required=True,
        ).kind,
        MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Notes?",
            required=False,
        ).kind,
        _questions()[1].kind,
        _questions()[2].kind,
    }
    assert question_types == set(QuestionType)


def test_requirement_input_n_005() -> None:
    """Keep stable values independent of native display labels."""
    first = _request(
        questions=(
            SingleSelectionQuestion(
                question_id=QuestionId("strategy"),
                prompt="Choose a strategy.",
                required=True,
                choices=_choices(),
            ),
        )
    )
    second = _request(
        questions=(
            SingleSelectionQuestion(
                question_id=QuestionId("strategy"),
                prompt="Choose a strategy.",
                required=True,
                choices=_choices(second_label="Thorough"),
            ),
        )
    )
    assert semantic_request_fingerprint(first) == semantic_request_fingerprint(
        second
    )


def test_requirement_input_n_006() -> None:
    """Resume an answered request with the same logical origin."""
    pending = _pending(_request())
    transition = resolve_request(
        pending,
        _answered(pending),
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(transition, InputTransitionApplied)
    outcome = project_resolution_to_model(
        transition.request,
        containing_run_exists=True,
    )
    assert isinstance(outcome.result, InputAnsweredResult)
    assert transition.request.origin == pending.origin


def test_requirement_input_n_007() -> None:
    """Keep the canonical core independent of provider call syntax."""
    encoded = encode_input_request(_request())
    assert RESERVED_INPUT_CAPABILITY_NAME == "request_user_input"
    assert not ({"provider", "tool_call", "function_call"} & set(encoded))
    assert decode_input_request(encoded) == _request()


def test_requirement_input_n_008() -> None:
    """Advertise only for attached or durable host handling."""
    assert HostCapabilities(attached_resolution=True).can_advertise
    assert HostCapabilities(durable_resolution=True).can_advertise
    assert not HostCapabilities().can_advertise
    assert HostCapabilities().handling is HostHandling.UNAVAILABLE


def test_requirement_input_n_009() -> None:
    """Keep host availability independent of domain-tool permissions."""
    names = {item.name for item in fields(HostCapabilities)}
    assert names == {"attached_resolution", "durable_resolution"}


def test_requirement_input_n_010() -> None:
    """Represent every terminal outcome explicitly."""
    assert {status.value for status in ResolutionStatus} == {
        "answered",
        "declined",
        "cancelled",
        "timed_out",
        "unavailable",
        "expired",
        "superseded",
    }


def test_requirement_input_n_011() -> None:
    """Never manufacture a human answer for a timeout."""
    request = _pending(_request(mode=RequirementMode.ADVISORY))
    resolution = TimedOutResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_CREATED_AT + timedelta(seconds=60),
    )
    transition = resolve_request(
        request,
        resolution,
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(transition, InputTransitionApplied)
    outcome = project_resolution_to_model(
        transition.request,
        containing_run_exists=True,
    )
    assert isinstance(outcome.result, InputTimedOutResult)
    assert outcome.result.provenance is AnswerProvenance.POLICY
    assert not hasattr(outcome.result, "answers")


def test_requirement_input_n_012() -> None:
    """Expose distinct interaction classes while hardcoding task input."""
    assert set(InteractionClass) == {
        InteractionClass.TASK_INPUT,
        InteractionClass.ACTION_APPROVAL,
        InteractionClass.STEERING,
        InteractionClass.AUTHENTICATION,
    }
    assert _request().interaction_class is InteractionClass.TASK_INPUT


def test_requirement_input_n_013() -> None:
    """Require a material user-facing reason for interruption."""
    request = _request()
    assert request.reason == "A decision is required to continue safely."


def test_requirement_input_n_014() -> None:
    """Support one focused question and consequential choice descriptions."""
    question = _questions()[1]
    request = _request(questions=(question,))
    assert len(request.questions) == 1
    assert all(choice.description for choice in question.choices)


def test_requirement_input_n_029() -> None:
    """Carry complete immutable correlation and lifecycle identity."""
    request = _request()
    assert request.request_id == InputRequestId("request-1")
    assert request.origin.task_id == TaskId("task-1")
    assert request.origin.model_call_id == ModelCallId("model-call-1")
    assert request.mode is RequirementMode.REQUIRED
    assert request.created_at.tzinfo is UTC
    assert request.state is RequestState.CREATED


def test_requirement_input_n_030() -> None:
    """Carry the required fields for every question."""
    question = _questions()[0]
    assert question.question_id == QuestionId("confirm")
    assert question.prompt == "Continue?"
    assert question.kind is QuestionType.CONFIRMATION
    assert question.required


def test_requirement_input_n_031() -> None:
    """Round-trip every optional request presentation field."""
    request = _request()
    assert decode_input_request(encode_input_request(request)) == request
    selection = request.questions[1]
    assert isinstance(selection, SingleSelectionQuestion)
    assert selection.header == "Strategy"
    assert selection.help_text
    assert selection.recommended_choice == ChoiceValue("careful")
    assert selection.default_value == ChoiceValue("careful")
    assert selection.presentation_hint is PresentationHint.RADIO


def test_requirement_input_n_032() -> None:
    """Key answers by stable IDs rather than displayed wording."""
    answer = _answered(_request()).answers[1]
    assert answer.question_id == QuestionId("strategy")
    assert (
        "Choose a strategy." not in encode_input_request(_request())["reason"]
    )


def test_requirement_input_n_033() -> None:
    """Keep question variants closed to the supported flat types."""
    for question in (
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
            required=False,
        ),
        _questions()[1],
        _questions()[2],
    ):
        assert (
            decode_input_question(encode_input_question(question)) == question
        )
    invalid = encode_input_question(_questions()[0])
    invalid["kind"] = "nested_form"
    with pytest.raises(InputCodecError):
        decode_input_question(invalid)


def test_requirement_input_n_034() -> None:
    """Give every choice a stable value and readable label."""
    assert [(choice.value, choice.label) for choice in _choices()] == [
        (ChoiceValue("fast"), "Fast"),
        (ChoiceValue("careful"), "Careful"),
    ]


def test_requirement_input_n_035() -> None:
    """Allow display labels to change without changing returned values."""
    original = _choices()[1]
    changed = _choices(second_label="Thorough")[1]
    assert original.value == changed.value
    assert original.label != changed.label


def test_requirement_input_n_036() -> None:
    """Represent free-form Other as a tagged alternative."""
    other = FreeFormOther(text="A custom safe choice")
    assert other.kind.value == "free_form_other"
    assert not isinstance(other, SelectedChoice)


def test_requirement_input_n_037() -> None:
    """Encode structured choices before opening a free-form alternative."""
    encoded = encode_input_question(_questions()[2])
    encoded_choices = encoded["choices"]
    assert isinstance(encoded_choices, list)
    assert encoded_choices
    assert encoded["allow_other"] is True
    labels: set[object] = set()
    for choice in encoded_choices:
        assert isinstance(choice, dict)
        labels.add(choice["label"])
    assert "Other" not in labels


def test_requirement_input_n_038() -> None:
    """Express compact and native-control preferences as typed hints."""
    assert {hint.value for hint in PresentationHint} == {
        "compact",
        "expanded",
        "radio",
        "list",
        "checkbox",
        "single_line",
        "editor",
    }


def test_requirement_input_n_039() -> None:
    """Keep hints advisory and outside semantic fingerprinting."""
    base = _questions()[1]
    alternate = SingleSelectionQuestion(
        question_id=base.question_id,
        prompt=base.prompt,
        required=base.required,
        choices=base.choices,
        allow_other=base.allow_other,
        recommended_choice=base.recommended_choice,
        default_value=base.default_value,
        presentation_hint=PresentationHint.LIST,
    )
    assert semantic_request_fingerprint(
        _request(questions=(base,))
    ) == semantic_request_fingerprint(_request(questions=(alternate,)))


def test_requirement_input_n_040() -> None:
    """Resolve each request once into one of seven terminal states."""
    constructors = (
        DeclinedResolution,
        CancelledResolution,
        UnavailableResolution,
        ExpiredResolution,
        SupersededResolution,
    )
    for constructor in constructors:
        pending = _pending(_request())
        resolution = constructor(
            request_id=pending.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_CREATED_AT + timedelta(seconds=1),
        )
        result = resolve_request(
            pending,
            resolution,
            expected_state_revision=StateRevision(1),
        )
        assert isinstance(result, InputTransitionApplied)
        assert result.request.state.value == resolution.status.value
        replay = resolve_request(
            result.request,
            resolution,
            expected_state_revision=StateRevision(2),
        )
        assert isinstance(replay, InputTransitionApplied)
        assert not replay.mutation_applied
    assert ResolutionStatus.ANSWERED.value == "answered"
    assert ResolutionStatus.TIMED_OUT.value == "timed_out"


def test_requirement_input_n_041() -> None:
    """Identify request, keyed answers, provenance, and resolution time."""
    request = _request()
    resolution = _answered(request)
    assert resolution.request_id == request.request_id
    assert {answer.question_id for answer in resolution.answers} == {
        QuestionId("confirm"),
        QuestionId("strategy"),
        QuestionId("checks"),
    }
    assert all(answer.provenance for answer in resolution.answers)
    assert resolution.resolved_at == _CREATED_AT + timedelta(seconds=1)


def test_requirement_input_n_042() -> None:
    """Distinguish human, trusted-default, and policy provenance."""
    question = ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Continue?",
        required=True,
        default_value=True,
    )
    pending = _pending(_request(questions=(question,)))
    for provenance in (
        AnswerProvenance.HUMAN,
        AnswerProvenance.TRUSTED_DEFAULT,
        AnswerProvenance.POLICY,
    ):
        answer = ConfirmationAnswer(
            question_id=question.question_id,
            provenance=provenance,
            value=True,
        )
        resolution = AnsweredResolution(
            request_id=pending.request_id,
            provenance=provenance,
            resolved_at=_CREATED_AT + timedelta(seconds=1),
            answers=(answer,),
        )
        result = resolve_request(
            pending,
            resolution,
            expected_state_revision=StateRevision(1),
        )
        assert isinstance(result, InputTransitionApplied)


def test_requirement_input_n_043() -> None:
    """Reject invalid answers before creating a continuation result."""
    pending = _pending(_request())
    invalid = AnsweredResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_CREATED_AT + timedelta(seconds=1),
        answers=(
            TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value="yes",
            ),
        ),
    )
    result = resolve_request(
        pending,
        invalid,
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(result, InputTransitionRejected)
    assert result.previous is pending
    assert pending.state is RequestState.PENDING
