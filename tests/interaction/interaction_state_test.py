"""Exercise pure structured-interaction domain state transitions."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import ClassVar, cast

import pytest

from avalan.interaction import (
    MAX_STATE_REVISION,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancellationScope,
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
    InputAnsweredResult,
    InputCancelledResult,
    InputDeclinedResult,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InputTimedOutResult,
    InputTransitionApplied,
    InputTransitionError,
    InputTransitionRejected,
    InputUnavailableResult,
    InputValidationError,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    QuestionId,
    QuestionType,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    ResumeInputContinuation,
    RunId,
    SelectedChoice,
    SelectionValidationConstraints,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TerminateInputContinuation,
    TextAnswer,
    TextQuestion,
    TextValidationConstraints,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    create_input_request,
    mark_request_pending,
    project_resolution_to_model,
    resolve_request,
)
from avalan.interaction.entities import (
    InputAnswer,
    InputQuestion,
    InputResolution,
    _validate_answers_against_request,
    _validate_resolution_against_request,
    _validate_selection_value,
    _validate_trusted_default,
)
from avalan.interaction.state import (
    _anchor_request_presentation,
    _model_result,
)

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


class _UncheckedQuestion(InputQuestion):
    """Bypass base validation to probe state-boundary closure."""

    kind: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        pass


class _UncheckedAnswer(InputAnswer):
    """Bypass base validation to probe state-boundary closure."""

    question_type: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        pass


class _UncheckedResolution(InputResolution):
    """Bypass base validation to probe state-boundary closure."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.DECLINED

    def __post_init__(self) -> None:
        pass


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
            model_config_reference="model-config-1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )


def _choices(count: int = 2) -> tuple[Choice, ...]:
    return tuple(
        Choice(value=ChoiceValue(f"choice-{index}"), label=f"Choice {index}")
        for index in range(count)
    )


def _confirmation(*, required: bool = True) -> ConfirmationQuestion:
    return ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Continue?",
        required=required,
        default_value=True,
    )


def _request(
    *,
    questions: tuple[InputQuestion, ...] | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=mode,
        reason="A response is required to continue.",
        questions=questions or (_confirmation(),),
        created_at=_NOW,
        advisory_wait_seconds=60 if mode is RequirementMode.ADVISORY else None,
    )


def _pending(request: InputRequest) -> InputRequest:
    transition = mark_request_pending(
        request,
        expected_state_revision=request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    return transition.request


def _answered(
    request: InputRequest,
    answers: tuple[InputAnswer, ...],
) -> AnsweredResolution:
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        resolved_at=_NOW + timedelta(seconds=1),
        answers=answers,
    )


def _apply(
    request: InputRequest,
    resolution: InputResolution,
) -> InputRequest:
    transition = resolve_request(
        request,
        resolution,
        expected_state_revision=request.state_revision,
    )
    assert isinstance(transition, InputTransitionApplied)
    assert transition.mutation_applied
    return transition.request


def _rejected_code(result: object) -> InputErrorCode:
    assert isinstance(result, InputTransitionRejected)
    return result.error.code


def test_transition_result_enforces_revision_semantics() -> None:
    """Require one revision increment per mutation and none per replay."""
    created = _request()
    pending = _pending(created)

    assert pending.state_revision == 1
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=created,
            request=pending,
            mutation_applied=cast(bool, 1),
        )
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=created,
            request=created,
            mutation_applied=True,
        )
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=created,
            request=replace(created),
            mutation_applied=False,
        )
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=cast(InputRequest, object()),
            request=pending,
            mutation_applied=True,
        )
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=created,
            request=cast(InputRequest, object()),
            mutation_applied=True,
        )
    other_pending = _pending(
        create_input_request(
            request_id=InputRequestId("request-2"),
            continuation_id=ContinuationId("continuation-2"),
            origin=_origin(),
            mode=RequirementMode.REQUIRED,
            reason="A response is required to continue.",
            questions=(_confirmation(),),
            created_at=_NOW,
        )
    )
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=created,
            request=other_pending,
            mutation_applied=True,
        )
    with pytest.raises(InputValidationError):
        InputTransitionRejected(
            previous=created,
            error=cast(InputTransitionError, object()),
        )

    advisory_pending = _anchor_request_presentation(
        _pending(_request(mode=RequirementMode.ADVISORY)),
        _NOW,
    )
    advisory_terminal = _apply(
        advisory_pending,
        DeclinedResolution(
            request_id=advisory_pending.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
    )
    assert advisory_terminal.advisory_deadline is not None
    changed_deadline = replace(
        advisory_terminal,
        advisory_deadline=advisory_terminal.advisory_deadline
        + timedelta(seconds=1),
    )
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=advisory_pending,
            request=changed_deadline,
            mutation_applied=True,
        )

    illegal_state = _pending(created)
    object.__setattr__(illegal_state, "state", RequestState.CREATED)
    with pytest.raises(InputValidationError):
        InputTransitionApplied(
            previous=created,
            request=illegal_state,
            mutation_applied=True,
        )


def test_transition_error_rejects_invalid_public_constructor_values() -> None:
    """Validate all public transition-error constructor fields."""
    with pytest.raises(InputValidationError):
        InputTransitionError(
            code=cast(InputErrorCode, "input.invalid_type"),
            path="request",
            message="invalid",
        )
    with pytest.raises(InputValidationError):
        InputTransitionError(
            code=InputErrorCode.INVALID_TYPE,
            path="",
            message="invalid",
        )
    with pytest.raises(InputValidationError):
        InputTransitionError(
            code=InputErrorCode.INVALID_TYPE,
            path="request",
            message="",
        )


def test_pending_transition_rejects_invalid_preconditions_and_overflow() -> (
    None
):
    """Reject invalid preconditions and revision overflow."""
    created = _request()
    with pytest.raises(InputValidationError):
        mark_request_pending(
            cast(InputRequest, object()),
            expected_state_revision=StateRevision(0),
        )
    invalid_revision = mark_request_pending(
        created,
        expected_state_revision=cast(StateRevision, "zero"),
    )
    stale = mark_request_pending(
        created,
        expected_state_revision=StateRevision(1),
    )
    pending = _pending(created)
    wrong_state = mark_request_pending(
        pending,
        expected_state_revision=pending.state_revision,
    )
    exhausted = _request()
    object.__setattr__(
        exhausted,
        "state_revision",
        StateRevision(MAX_STATE_REVISION),
    )
    overflow = mark_request_pending(
        exhausted,
        expected_state_revision=exhausted.state_revision,
    )
    assert _rejected_code(invalid_revision) is InputErrorCode.INVALID_TYPE
    assert _rejected_code(stale) is InputErrorCode.STALE_REVISION
    assert _rejected_code(wrong_state) is InputErrorCode.ILLEGAL_TRANSITION
    assert _rejected_code(overflow) is InputErrorCode.STATE_REVISION_EXHAUSTED


def test_resolution_transition_is_exactly_once() -> None:
    """Accept one terminal mutation and make semantic replay idempotent."""
    created = _request()
    resolution = DeclinedResolution(
        request_id=created.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )

    from_created = resolve_request(
        created,
        resolution,
        expected_state_revision=created.state_revision,
    )
    assert _rejected_code(from_created) is InputErrorCode.ILLEGAL_TRANSITION

    pending = _pending(created)
    invalid_resolution = resolve_request(
        pending,
        cast(InputResolution, object()),
        expected_state_revision=pending.state_revision,
    )
    assert _rejected_code(invalid_resolution) is InputErrorCode.INVALID_TYPE

    resolved = _apply(pending, resolution)
    replay = resolve_request(
        resolved,
        resolution,
        expected_state_revision=resolved.state_revision,
    )
    assert isinstance(replay, InputTransitionApplied)
    assert not replay.mutation_applied
    assert replay.request is resolved

    conflict = resolve_request(
        resolved,
        UnavailableResolution(
            request_id=resolved.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=2),
        ),
        expected_state_revision=resolved.state_revision,
    )
    assert _rejected_code(conflict) is InputErrorCode.ILLEGAL_TRANSITION
    assert conflict.previous.state_revision == resolved.state_revision


def test_resolution_transition_rejects_revision_and_resolution_errors() -> (
    None
):
    """Return typed rejections without mutating pending requests."""
    pending = _pending(_request())
    declined = DeclinedResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )

    with pytest.raises(InputValidationError):
        resolve_request(
            cast(InputRequest, object()),
            declined,
            expected_state_revision=StateRevision(0),
        )
    invalid_revision = resolve_request(
        pending,
        declined,
        expected_state_revision=cast(StateRevision, "one"),
    )
    stale = resolve_request(
        pending,
        declined,
        expected_state_revision=StateRevision(0),
    )
    mismatch = resolve_request(
        pending,
        DeclinedResolution(
            request_id=InputRequestId("other-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
        expected_state_revision=pending.state_revision,
    )
    predates = resolve_request(
        pending,
        DeclinedResolution(
            request_id=pending.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW - timedelta(seconds=1),
        ),
        expected_state_revision=pending.state_revision,
    )
    required_timeout = resolve_request(
        pending,
        TimedOutResolution(
            request_id=pending.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=1),
        ),
        expected_state_revision=pending.state_revision,
    )
    exhausted = _pending(_request())
    object.__setattr__(
        exhausted,
        "state_revision",
        StateRevision(MAX_STATE_REVISION),
    )
    overflow = resolve_request(
        exhausted,
        declined,
        expected_state_revision=exhausted.state_revision,
    )

    assert _rejected_code(invalid_revision) is InputErrorCode.INVALID_TYPE
    assert _rejected_code(stale) is InputErrorCode.STALE_REVISION
    assert _rejected_code(mismatch) is InputErrorCode.CORRELATION_MISMATCH
    assert _rejected_code(predates) is InputErrorCode.INVALID_FORMAT
    assert (
        _rejected_code(required_timeout) is InputErrorCode.TIMED_OUT_REQUIRED
    )
    assert _rejected_code(overflow) is InputErrorCode.STATE_REVISION_EXHAUSTED


def test_advisory_timing_is_derived_once_from_trusted_presentation() -> None:
    """Admit queued first, then anchor trusted presentation exactly once."""
    created = _request(mode=RequirementMode.ADVISORY)
    admission = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(admission, InputTransitionApplied)
    queued = admission.request
    assert queued.advisory_deadline is None

    with pytest.raises(InputValidationError) as predating:
        _anchor_request_presentation(
            queued,
            _NOW - timedelta(microseconds=1),
        )
    with pytest.raises(InputValidationError) as naive:
        _anchor_request_presentation(queued, _NOW.replace(tzinfo=None))
    exact = _anchor_request_presentation(queued, _NOW)
    late = _anchor_request_presentation(
        queued,
        _NOW + timedelta(seconds=1),
    )
    required_admission = mark_request_pending(
        _request(),
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(required_admission, InputTransitionApplied)
    with pytest.raises(InputValidationError) as required_predating:
        _anchor_request_presentation(
            required_admission.request,
            _NOW - timedelta(microseconds=1),
        )
    with pytest.raises(InputValidationError) as required_naive:
        _anchor_request_presentation(
            required_admission.request,
            _NOW.replace(tzinfo=None),
        )
    required_timing = _anchor_request_presentation(
        required_admission.request,
        _NOW,
    )
    overflow_request = replace(
        created,
        created_at=datetime.max.replace(tzinfo=UTC),
    )
    overflow_admission = mark_request_pending(
        overflow_request,
        expected_state_revision=overflow_request.state_revision,
    )
    assert isinstance(overflow_admission, InputTransitionApplied)
    with pytest.raises(InputValidationError) as overflow:
        _anchor_request_presentation(
            overflow_admission.request,
            overflow_request.created_at,
        )

    assert predating.value.code is InputErrorCode.INVALID_FORMAT
    assert naive.value.code is InputErrorCode.NAIVE_TIMESTAMP
    assert required_predating.value.code is InputErrorCode.INVALID_FORMAT
    assert required_naive.value.code is InputErrorCode.NAIVE_TIMESTAMP
    assert overflow.value.code is InputErrorCode.OUT_OF_BOUNDS
    assert required_timing is required_admission.request
    assert exact.advisory_deadline == _NOW + timedelta(seconds=60)
    assert late.advisory_deadline == _NOW + timedelta(seconds=61)
    assert exact.state_revision == queued.state_revision

    pending = exact
    assert pending.advisory_deadline is not None

    def transition_at(resolved_at: datetime) -> object:
        return resolve_request(
            pending,
            TimedOutResolution(
                request_id=pending.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=resolved_at,
            ),
            expected_state_revision=pending.state_revision,
        )

    early_timeout = transition_at(
        pending.advisory_deadline - timedelta(microseconds=1)
    )
    exact_timeout = transition_at(pending.advisory_deadline)
    late_timeout = transition_at(
        pending.advisory_deadline + timedelta(seconds=1)
    )

    assert _rejected_code(early_timeout) is InputErrorCode.INVALID_FORMAT
    assert isinstance(exact_timeout, InputTransitionApplied)
    assert isinstance(late_timeout, InputTransitionApplied)


def test_queued_advisory_can_resolve_before_presentation() -> None:
    """Allow every non-timeout terminal outcome before presentation."""
    created = _request(mode=RequirementMode.ADVISORY)
    resolutions: tuple[InputResolution, ...] = (
        AnsweredResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=_NOW,
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    value=True,
                ),
            ),
        ),
        DeclinedResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=_NOW,
        ),
        UnavailableResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        CancelledResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=_NOW,
            scope=CancellationScope.REQUEST,
        ),
        SupersededResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        ExpiredResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(days=1),
        ),
    )

    for resolution in resolutions:
        pending = _pending(created)
        result = resolve_request(
            pending,
            resolution,
            expected_state_revision=pending.state_revision,
        )
        assert isinstance(result, InputTransitionApplied)
        assert result.request.advisory_deadline is None

    pending = _pending(created)
    timeout = resolve_request(
        pending,
        TimedOutResolution(
            request_id=pending.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=60),
        ),
        expected_state_revision=pending.state_revision,
    )
    assert _rejected_code(timeout) is InputErrorCode.INVALID_FORMAT


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (
            _confirmation(),
            TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value="yes",
            ),
        ),
        (
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            ),
            ConfirmationAnswer(
                question_id=QuestionId("text"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
        (
            MultilineTextQuestion(
                question_id=QuestionId("multiline"),
                prompt="Text?",
                required=True,
            ),
            ConfirmationAnswer(
                question_id=QuestionId("multiline"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
        (
            SingleSelectionQuestion(
                question_id=QuestionId("single"),
                prompt="Choose.",
                required=True,
                choices=_choices(),
            ),
            ConfirmationAnswer(
                question_id=QuestionId("single"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
        (
            MultipleSelectionQuestion(
                question_id=QuestionId("multiple"),
                prompt="Choose.",
                required=True,
                choices=_choices(),
                constraints=SelectionValidationConstraints(maximum=2),
            ),
            ConfirmationAnswer(
                question_id=QuestionId("multiple"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
    ],
)
def test_answer_type_must_match_question(
    question: InputQuestion,
    answer: InputAnswer,
) -> None:
    """Reject every cross-variant answer type mismatch."""
    pending = _pending(_request(questions=(question,)))
    result = resolve_request(
        pending,
        _answered(pending, (answer,)),
        expected_state_revision=pending.state_revision,
    )
    assert _rejected_code(result) is InputErrorCode.ANSWER_TYPE_MISMATCH


def test_answer_set_rejects_unknown_missing_and_invalid_text() -> None:
    """Reject unknown identifiers, missing required values, and text bounds."""
    text_question = TextQuestion(
        question_id=QuestionId("text"),
        prompt="Text?",
        required=True,
        constraints=TextValidationConstraints(
            minimum_length=2, maximum_length=3
        ),
    )
    multiline_question = MultilineTextQuestion(
        question_id=QuestionId("multiline"),
        prompt="Text?",
        required=True,
        constraints=TextValidationConstraints(
            minimum_length=2, maximum_length=3
        ),
    )

    pending = _pending(_request(questions=(_confirmation(),)))
    unknown = resolve_request(
        pending,
        _answered(
            pending,
            (
                ConfirmationAnswer(
                    question_id=QuestionId("unknown"),
                    provenance=AnswerProvenance.HUMAN,
                    value=True,
                ),
            ),
        ),
        expected_state_revision=pending.state_revision,
    )
    missing = resolve_request(
        pending,
        _answered(pending, ()),
        expected_state_revision=pending.state_revision,
    )
    assert _rejected_code(unknown) is InputErrorCode.UNKNOWN_QUESTION
    assert _rejected_code(missing) is InputErrorCode.MISSING_REQUIRED_ANSWER

    for question, answer in (
        (
            text_question,
            TextAnswer(
                question_id=question_id(text_question),
                provenance=AnswerProvenance.HUMAN,
                value="x",
            ),
        ),
        (
            multiline_question,
            MultilineTextAnswer(
                question_id=question_id(multiline_question),
                provenance=AnswerProvenance.HUMAN,
                value="long",
            ),
        ),
    ):
        text_pending = _pending(_request(questions=(question,)))
        result = resolve_request(
            text_pending,
            _answered(text_pending, (answer,)),
            expected_state_revision=text_pending.state_revision,
        )
        assert _rejected_code(result) is InputErrorCode.INVALID_CARDINALITY


def question_id(question: InputQuestion) -> QuestionId:
    """Return a typed identifier for parametrized question fixtures."""
    return question.question_id


def test_selection_answers_validate_values_and_cardinality() -> None:
    """Reject unknown, forbidden, malformed, and mis-sized selections."""
    choices = _choices(3)
    single = SingleSelectionQuestion(
        question_id=QuestionId("single"),
        prompt="Choose.",
        required=True,
        choices=choices,
    )
    multiple = MultipleSelectionQuestion(
        question_id=QuestionId("multiple"),
        prompt="Choose.",
        required=True,
        choices=choices,
        constraints=SelectionValidationConstraints(minimum=1, maximum=2),
    )

    cases: tuple[tuple[InputQuestion, InputAnswer, InputErrorCode], ...] = (
        (
            single,
            SingleSelectionAnswer(
                question_id=single.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("unknown")),
            ),
            InputErrorCode.UNKNOWN_CHOICE,
        ),
        (
            single,
            SingleSelectionAnswer(
                question_id=single.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=FreeFormOther(text="custom"),
            ),
            InputErrorCode.OTHER_NOT_ALLOWED,
        ),
        (
            multiple,
            MultipleSelectionAnswer(
                question_id=multiple.question_id,
                provenance=AnswerProvenance.HUMAN,
                values=(),
            ),
            InputErrorCode.INVALID_CARDINALITY,
        ),
        (
            multiple,
            MultipleSelectionAnswer(
                question_id=multiple.question_id,
                provenance=AnswerProvenance.HUMAN,
                values=tuple(
                    SelectedChoice(value=choice.value) for choice in choices
                ),
            ),
            InputErrorCode.INVALID_CARDINALITY,
        ),
    )
    for question, answer, code in cases:
        pending = _pending(_request(questions=(question,)))
        result = resolve_request(
            pending,
            _answered(pending, (answer,)),
            expected_state_revision=pending.state_revision,
        )
        assert _rejected_code(result) is code

    with pytest.raises(InputValidationError) as error:
        _validate_selection_value(single, cast(SelectedChoice, object()))
    assert error.value.code is InputErrorCode.INVALID_TYPE


@pytest.mark.parametrize(
    ("question", "matching", "mismatching"),
    [
        (
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=True,
            ),
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value=True,
            ),
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value=False,
            ),
        ),
        (
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
                default_value="Ada",
            ),
            TextAnswer(
                question_id=QuestionId("text"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value="Ada",
            ),
            TextAnswer(
                question_id=QuestionId("text"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value="Grace",
            ),
        ),
        (
            MultilineTextQuestion(
                question_id=QuestionId("multiline"),
                prompt="Text?",
                required=True,
                default_value="one\ntwo",
            ),
            MultilineTextAnswer(
                question_id=QuestionId("multiline"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value="one\ntwo",
            ),
            MultilineTextAnswer(
                question_id=QuestionId("multiline"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value="other",
            ),
        ),
        (
            SingleSelectionQuestion(
                question_id=QuestionId("single"),
                prompt="Choose.",
                required=True,
                choices=_choices(),
                allow_other=True,
                default_value=ChoiceValue("choice-0"),
            ),
            SingleSelectionAnswer(
                question_id=QuestionId("single"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value=SelectedChoice(value=ChoiceValue("choice-0")),
            ),
            SingleSelectionAnswer(
                question_id=QuestionId("single"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value=FreeFormOther(text="custom"),
            ),
        ),
        (
            MultipleSelectionQuestion(
                question_id=QuestionId("multiple"),
                prompt="Choose.",
                required=True,
                choices=_choices(),
                allow_other=True,
                default_value=(
                    ChoiceValue("choice-0"),
                    ChoiceValue("choice-1"),
                ),
                constraints=SelectionValidationConstraints(
                    minimum=1, maximum=3
                ),
            ),
            MultipleSelectionAnswer(
                question_id=QuestionId("multiple"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                values=(
                    SelectedChoice(value=ChoiceValue("choice-0")),
                    SelectedChoice(value=ChoiceValue("choice-1")),
                ),
            ),
            MultipleSelectionAnswer(
                question_id=QuestionId("multiple"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                values=(
                    SelectedChoice(value=ChoiceValue("choice-0")),
                    FreeFormOther(text="custom"),
                ),
            ),
        ),
    ],
)
def test_trusted_defaults_must_equal_declared_defaults(
    question: InputQuestion,
    matching: InputAnswer,
    mismatching: InputAnswer,
) -> None:
    """Accept exact trusted defaults and reject manufactured values."""
    pending = _pending(_request(questions=(question,)))
    accepted = resolve_request(
        pending,
        _answered(pending, (matching,)),
        expected_state_revision=pending.state_revision,
    )
    rejected = resolve_request(
        pending,
        _answered(pending, (mismatching,)),
        expected_state_revision=pending.state_revision,
    )

    assert isinstance(accepted, InputTransitionApplied)
    assert _rejected_code(rejected) is InputErrorCode.INVALID_DEFAULT


def test_outcome_projection_covers_resume_and_termination_matrix() -> None:
    """Project all terminal statuses according to run existence and scope."""
    confirmation = _confirmation()
    answered_pending = _pending(_request(questions=(confirmation,)))
    answered = _apply(
        answered_pending,
        _answered(
            answered_pending,
            (
                ConfirmationAnswer(
                    question_id=confirmation.question_id,
                    provenance=AnswerProvenance.HUMAN,
                    value=True,
                ),
            ),
        ),
    )
    answered_outcome = project_resolution_to_model(
        answered,
        containing_run_exists=True,
    )
    assert isinstance(answered_outcome, ResumeInputContinuation)
    assert isinstance(answered_outcome.result, InputAnsweredResult)

    resolution_types: tuple[
        tuple[InputResolution, type[object] | None], ...
    ] = (
        (
            DeclinedResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW + timedelta(seconds=1),
            ),
            InputDeclinedResult,
        ),
        (
            CancelledResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW + timedelta(seconds=1),
                scope=CancellationScope.REQUEST,
            ),
            InputCancelledResult,
        ),
        (
            TimedOutResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW + timedelta(seconds=1),
            ),
            InputTimedOutResult,
        ),
        (
            UnavailableResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW + timedelta(seconds=1),
            ),
            InputUnavailableResult,
        ),
        (
            ExpiredResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW + timedelta(seconds=1),
            ),
            None,
        ),
        (
            SupersededResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW + timedelta(seconds=1),
            ),
            None,
        ),
        (
            CancelledResolution(
                request_id=answered.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW + timedelta(seconds=1),
                scope=CancellationScope.CONTAINING_RUN,
            ),
            None,
        ),
    )
    for resolution, result_type in resolution_types:
        mode = (
            RequirementMode.ADVISORY
            if isinstance(resolution, TimedOutResolution)
            else RequirementMode.REQUIRED
        )
        pending = _pending(_request(mode=mode))
        if isinstance(resolution, TimedOutResolution):
            pending = _anchor_request_presentation(pending, _NOW)
            assert pending.advisory_deadline is not None
            resolution = replace(
                resolution,
                resolved_at=pending.advisory_deadline,
            )
        terminal = _apply(pending, resolution)
        outcome = project_resolution_to_model(
            terminal,
            containing_run_exists=True,
        )
        if result_type is None:
            assert isinstance(outcome, TerminateInputContinuation)
        else:
            assert isinstance(outcome, ResumeInputContinuation)
            assert isinstance(outcome.result, result_type)

    no_run = project_resolution_to_model(answered, containing_run_exists=False)
    assert isinstance(no_run, TerminateInputContinuation)


def test_outcome_projection_rejects_nonterminal_and_invalid_inputs() -> None:
    """Reject projection before resolution and unreachable model statuses."""
    request = _request()
    with pytest.raises(InputValidationError):
        project_resolution_to_model(
            request,
            containing_run_exists=cast(bool, 1),
        )
    with pytest.raises(InputValidationError):
        project_resolution_to_model(request, containing_run_exists=True)
    with pytest.raises(InputValidationError):
        _model_result(
            ExpiredResolution(
                request_id=request.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW,
            )
        )
    with pytest.raises(InputValidationError):
        _validate_resolution_against_request(
            request,
            cast(InputResolution, object()),
        )
    advisory = _request(mode=RequirementMode.ADVISORY)
    with pytest.raises(InputValidationError):
        _validate_resolution_against_request(
            advisory,
            TimedOutResolution(
                request_id=advisory.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW + timedelta(seconds=60),
            ),
        )
    with pytest.raises(InputValidationError):
        _validate_trusted_default(
            object(),
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value=True,
            ),
        )


def test_state_boundaries_reject_unchecked_union_subclasses() -> None:
    """Reject rogue variants before reading their declared discriminator."""
    private_value = "PRIVATE_ROGUE_STATE_SENTINEL"
    question = _UncheckedQuestion(
        question_id=QuestionId(private_value),
        prompt=private_value,
        required=True,
    )
    answer = _UncheckedAnswer(
        question_id=QuestionId(private_value),
        provenance=AnswerProvenance.TRUSTED_DEFAULT,
    )
    resolution = _UncheckedResolution(
        request_id=InputRequestId(private_value),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    pending = _pending(_request())

    transition = resolve_request(
        pending,
        resolution,
        expected_state_revision=pending.state_revision,
    )
    assert _rejected_code(transition) is InputErrorCode.INVALID_TYPE
    assert isinstance(transition, InputTransitionRejected)
    assert private_value not in str(transition.error)

    with pytest.raises(InputValidationError) as captured:
        _model_result(resolution)
    assert captured.value.code is InputErrorCode.INVALID_TYPE
    assert private_value not in str(captured.value)

    rogue_request_type = type("RogueInputRequest", (InputRequest,), {})
    rogue_request: InputRequest = object.__new__(rogue_request_type)
    with pytest.raises(InputValidationError) as captured:
        project_resolution_to_model(
            rogue_request,
            containing_run_exists=True,
        )
    assert captured.value.code is InputErrorCode.INVALID_TYPE

    forged_request = _request()
    object.__setattr__(forged_request, "state", RequestState.DECLINED)
    object.__setattr__(forged_request, "resolution", resolution)
    with pytest.raises(InputValidationError) as captured:
        project_resolution_to_model(
            forged_request,
            containing_run_exists=True,
        )
    assert captured.value.code is InputErrorCode.INVALID_TYPE
    assert private_value not in str(captured.value)

    forged_questions = _request()
    object.__setattr__(forged_questions, "questions", (question,))
    with pytest.raises(InputValidationError):
        _validate_answers_against_request(forged_questions, ())
    with pytest.raises(InputValidationError):
        _validate_answers_against_request(_request(), (answer,))
    with pytest.raises(InputValidationError):
        _validate_trusted_default(_confirmation(), answer)
    with pytest.raises(InputValidationError):
        _validate_trusted_default(
            _confirmation(),
            TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.TRUSTED_DEFAULT,
                value="not a confirmation",
            ),
        )
