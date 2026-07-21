"""Lock public canonical interaction types and result discrimination."""

from datetime import UTC, datetime, timedelta
from typing import Literal, assert_never, assert_type

from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelledResolution,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationDisposition,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    FreeFormOther,
    InputAnswer,
    InputAnsweredResult,
    InputCancelledResult,
    InputContinuationOutcome,
    InputDeclinedResult,
    InputModelResult,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    InputResultKind,
    InputTimedOutResult,
    InputTransitionApplied,
    InputTransitionError,
    InputTransitionRejected,
    InputTransitionResult,
    InputUnavailableResult,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    QuestionId,
    RequirementMode,
    ResolutionStatus,
    ResumeInputContinuation,
    RunId,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TaskId,
    TerminateInputContinuation,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TransitionResultType,
    TurnId,
    UnavailableResolution,
    create_input_request,
    mark_request_pending,
    project_resolution_to_model,
    resolve_request,
)

now = datetime(2026, 7, 20, tzinfo=UTC)
run_id = RunId("run")
turn_id = TurnId("turn")
task_id = TaskId("task")
agent_id = AgentId("agent")
branch_id = BranchId("branch")
model_call_id = ModelCallId("model-call")
request_id = InputRequestId("request")
continuation_id = ContinuationId("continuation")
stream_session_id = StreamSessionId("stream")
question_id = QuestionId("question")
choice_value = ChoiceValue("choice")
revision = StateRevision(0)

assert_type(run_id, RunId)
assert_type(turn_id, TurnId)
assert_type(task_id, TaskId)
assert_type(agent_id, AgentId)
assert_type(branch_id, BranchId)
assert_type(model_call_id, ModelCallId)
assert_type(request_id, InputRequestId)
assert_type(continuation_id, ContinuationId)
assert_type(stream_session_id, StreamSessionId)
assert_type(question_id, QuestionId)
assert_type(choice_value, ChoiceValue)
assert_type(revision, StateRevision)

choice = Choice(value=choice_value, label="Choice")
confirmation_question = ConfirmationQuestion(
    question_id=QuestionId("confirmation"),
    prompt="Continue?",
    required=True,
)
text_question = TextQuestion(
    question_id=QuestionId("text"),
    prompt="Name?",
    required=True,
)
multiline_question = MultilineTextQuestion(
    question_id=QuestionId("multiline"),
    prompt="Notes?",
    required=False,
)
single_question = SingleSelectionQuestion(
    question_id=QuestionId("single"),
    prompt="Choose one.",
    required=True,
    choices=(choice,),
)
multiple_question = MultipleSelectionQuestion(
    question_id=QuestionId("multiple"),
    prompt="Choose values.",
    required=False,
    choices=(choice,),
)

assert_type(confirmation_question, ConfirmationQuestion)
assert_type(text_question, TextQuestion)
assert_type(multiline_question, MultilineTextQuestion)
assert_type(single_question, SingleSelectionQuestion)
assert_type(multiple_question, MultipleSelectionQuestion)

confirmation_answer = ConfirmationAnswer(
    question_id=confirmation_question.question_id,
    provenance=AnswerProvenance.HUMAN,
    value=True,
)
text_answer = TextAnswer(
    question_id=text_question.question_id,
    provenance=AnswerProvenance.TRUSTED_DEFAULT,
    value="name",
)
multiline_answer = MultilineTextAnswer(
    question_id=multiline_question.question_id,
    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    value="line one\nline two",
)
single_answer = SingleSelectionAnswer(
    question_id=single_question.question_id,
    provenance=AnswerProvenance.HUMAN,
    value=SelectedChoice(value=choice_value),
)
multiple_answer = MultipleSelectionAnswer(
    question_id=multiple_question.question_id,
    provenance=AnswerProvenance.POLICY,
    values=(FreeFormOther(text="other"),),
)

assert_type(confirmation_answer, ConfirmationAnswer)
assert_type(text_answer, TextAnswer)
assert_type(multiline_answer, MultilineTextAnswer)
assert_type(single_answer, SingleSelectionAnswer)
assert_type(multiple_answer, MultipleSelectionAnswer)

origin = ExecutionOrigin(
    run_id=run_id,
    turn_id=turn_id,
    task_id=task_id,
    agent_id=agent_id,
    branch_id=branch_id,
    model_call_id=model_call_id,
    stream_session_id=stream_session_id,
    definition=ExecutionDefinitionRef(
        agent_definition_locator="agent://fixture",
        agent_definition_revision="r1",
        operation_id="operation",
        operation_index=0,
        model_config_reference="model-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    ),
)
request = create_input_request(
    request_id=request_id,
    continuation_id=continuation_id,
    origin=origin,
    mode=RequirementMode.REQUIRED,
    reason="Need one answer.",
    questions=(confirmation_question,),
    created_at=now,
)
assert_type(request, InputRequest)
advisory_request = create_input_request(
    request_id=request_id,
    continuation_id=continuation_id,
    origin=origin,
    mode=RequirementMode.ADVISORY,
    reason="Offer one answer.",
    questions=(confirmation_question,),
    created_at=now,
)
assert_type(advisory_request, InputRequest)
input_required = InputRequiredResult(
    request_id=request_id,
    continuation_id=continuation_id,
    detached_resumption_available=True,
)
assert_type(input_required, InputRequiredResult)
assert_type(
    input_required.kind,
    Literal[InputResultKind.INPUT_REQUIRED],
)

answered = AnsweredResolution(
    request_id=request_id,
    provenance=AnswerProvenance.HUMAN,
    resolved_at=now,
    answers=(confirmation_answer,),
)
declined = DeclinedResolution(
    request_id=request_id,
    provenance=AnswerProvenance.HUMAN,
    resolved_at=now,
)
cancelled = CancelledResolution(
    request_id=request_id,
    provenance=AnswerProvenance.HUMAN,
    resolved_at=now,
)
timed_out = TimedOutResolution(
    request_id=request_id,
    provenance=AnswerProvenance.POLICY,
    resolved_at=now,
)
unavailable = UnavailableResolution(
    request_id=request_id,
    provenance=AnswerProvenance.POLICY,
    resolved_at=now,
)
expired = ExpiredResolution(
    request_id=request_id,
    provenance=AnswerProvenance.POLICY,
    resolved_at=now,
)
superseded = SupersededResolution(
    request_id=request_id,
    provenance=AnswerProvenance.POLICY,
    resolved_at=now,
)

assert_type(answered, AnsweredResolution)
assert_type(declined, DeclinedResolution)
assert_type(cancelled, CancelledResolution)
assert_type(timed_out, TimedOutResolution)
assert_type(unavailable, UnavailableResolution)
assert_type(expired, ExpiredResolution)
assert_type(superseded, SupersededResolution)
advisory_transition = mark_request_pending(
    advisory_request,
    expected_state_revision=revision,
    presented_at=now,
)
assert_type(advisory_transition, InputTransitionResult)
if isinstance(advisory_transition, InputTransitionApplied):
    assert_type(advisory_transition.request, InputRequest)
    advisory_deadline = advisory_transition.request.advisory_deadline
    assert advisory_deadline is not None
    assert advisory_deadline == now + timedelta(seconds=60)
    advisory_timeout = TimedOutResolution(
        request_id=request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=advisory_deadline,
    )
    assert_type(
        resolve_request(
            advisory_transition.request,
            advisory_timeout,
            expected_state_revision=StateRevision(1),
        ),
        InputTransitionResult,
    )

transition = mark_request_pending(
    request,
    expected_state_revision=revision,
)
assert_type(transition, InputTransitionResult)
if isinstance(transition, InputTransitionApplied):
    assert_type(transition.request, InputRequest)
    resolved = resolve_request(
        transition.request,
        answered,
        expected_state_revision=StateRevision(1),
    )
    assert_type(resolved, InputTransitionResult)
    if isinstance(resolved, InputTransitionApplied):
        assert_type(resolved.request, InputRequest)
        outcome = project_resolution_to_model(
            resolved.request,
            containing_run_exists=True,
        )
        assert_type(outcome, InputContinuationOutcome)
        if isinstance(outcome, ResumeInputContinuation):
            assert_type(outcome.result, InputModelResult)
        elif isinstance(outcome, TerminateInputContinuation):
            assert_type(outcome.status, ResolutionStatus)
    elif isinstance(resolved, InputTransitionRejected):
        assert_type(resolved.error, InputTransitionError)
elif isinstance(transition, InputTransitionRejected):
    assert_type(transition.error, InputTransitionError)


def discriminate_model_result(result: InputModelResult) -> None:
    """Lock every model-result variant under strict narrowing."""
    if result.kind is InputResultKind.ANSWERED:
        assert_type(result, InputAnsweredResult)
        assert_type(result.answers, tuple[InputAnswer, ...])
    elif result.kind is InputResultKind.DECLINED:
        assert_type(result, InputDeclinedResult)
    elif result.kind is InputResultKind.CANCELLED:
        assert_type(result, InputCancelledResult)
    elif result.kind is InputResultKind.TIMED_OUT:
        assert_type(result, InputTimedOutResult)
    elif result.kind is InputResultKind.UNAVAILABLE:
        assert_type(result, InputUnavailableResult)
    else:
        assert_never(result)


def discriminate_continuation(result: InputContinuationOutcome) -> None:
    """Lock both continuation variants under their literal discriminator."""
    if result.disposition is ContinuationDisposition.RESUME:
        assert_type(result, ResumeInputContinuation)
        assert_type(result.result, InputModelResult)
    elif result.disposition is ContinuationDisposition.TERMINATE:
        assert_type(result, TerminateInputContinuation)
        assert_type(result.status, ResolutionStatus)
    else:
        assert_never(result)


def discriminate_transition(result: InputTransitionResult) -> None:
    """Lock both transition variants under their literal discriminator."""
    if result.kind is TransitionResultType.APPLIED:
        assert_type(result, InputTransitionApplied)
        assert_type(result.request, InputRequest)
    elif result.kind is TransitionResultType.REJECTED:
        assert_type(result, InputTransitionRejected)
        assert_type(result.error, InputTransitionError)
    else:
        assert_never(result)
