"""Exercise the canonical interaction boundary without a provider or UI."""

from datetime import UTC, datetime, timedelta

import pytest

from avalan.event import Event, InteractionLifecyclePayload
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancellationScope,
    CancelledResolution,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    InputRequest,
    InputRequestId,
    InputResolution,
    InputTransitionApplied,
    InputTransitionRejected,
    InteractionSnapshot,
    ModelCallId,
    QuestionId,
    RequestState,
    RequirementMode,
    ResumeInputContinuation,
    RunId,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TaskId,
    TerminateInputContinuation,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    create_input_request,
    decode_input_request,
    decode_input_resolution,
    decode_interaction_snapshot,
    encode_continuation_outcome,
    encode_input_request,
    encode_input_resolution,
    encode_interaction_snapshot,
    mark_request_pending,
    project_resolution_to_model,
    resolve_request,
)
from avalan.interaction.state import _anchor_request_presentation
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    validate_canonical_stream_items,
)

_NOW = datetime(2026, 7, 20, 20, 0, tzinfo=UTC)


def _request(
    *,
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=ExecutionOrigin(
            run_id=RunId("run-1"),
            turn_id=TurnId("turn-1"),
            task_id=TaskId("task-1"),
            agent_id=AgentId("agent-1"),
            branch_id=BranchId("branch-1"),
            model_call_id=ModelCallId("model-call-1"),
            stream_session_id=StreamSessionId("stream-1"),
            definition=ExecutionDefinitionRef(
                agent_definition_locator="agent://support",
                agent_definition_revision="agent-r1",
                operation_id="operation-1",
                operation_index=0,
                model_config_reference="model-r1",
                tool_revision="tools-r1",
                capability_revision="capabilities-r1",
            ),
        ),
        mode=mode,
        reason="Choose a safe deployment target.",
        questions=(
            TextQuestion(
                question_id=QuestionId("target"),
                prompt="Which target?",
                required=True,
            ),
        ),
        created_at=_NOW,
    )


def _pending(request: InputRequest) -> InputRequest:
    transition = mark_request_pending(
        request,
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(transition, InputTransitionApplied)
    if request.mode is RequirementMode.ADVISORY:
        return _anchor_request_presentation(
            transition.request,
            request.created_at,
        )
    return transition.request


def _correlation(request: InputRequest) -> StreamItemCorrelation:
    return StreamItemCorrelation(
        request_id=request.request_id,
        continuation_id=request.continuation_id,
        task_id=request.origin.task_id,
        agent_id=request.origin.agent_id,
        branch_id=request.origin.branch_id,
    )


def _stream_item(
    request: InputRequest,
    kind: StreamItemKind,
    sequence: int,
    *,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=request.origin.stream_session_id,
        run_id=request.origin.run_id,
        turn_id=request.origin.turn_id,
        sequence=sequence,
        kind=kind,
        channel=(
            StreamChannel.INTERACTION
            if kind.value.startswith("interaction.")
            else StreamChannel.CONTROL
        ),
        correlation=(
            StreamItemCorrelation()
            if kind
            in {
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CLOSED,
            }
            else _correlation(request)
        ),
        terminal_outcome=terminal_outcome,
    )


def test_canonical_interaction_boundary_e2e() -> None:
    """Serialize, suspend, observe, resolve, and resume one request."""
    request = decode_input_request(encode_input_request(_request()))
    pending = _pending(request)
    lifecycle_events = (
        Event.from_interaction_lifecycle(
            InteractionLifecyclePayload.from_canonical_ids(
                request_id=request.request_id,
                run_id=request.origin.run_id,
                turn_id=request.origin.turn_id,
                task_id=request.origin.task_id,
                agent_id=request.origin.agent_id,
                branch_id=request.origin.branch_id,
                state=RequestState.CREATED,
            )
        ),
        Event.from_interaction_lifecycle(
            InteractionLifecyclePayload.from_canonical_ids(
                request_id=pending.request_id,
                run_id=pending.origin.run_id,
                turn_id=pending.origin.turn_id,
                task_id=pending.origin.task_id,
                agent_id=pending.origin.agent_id,
                branch_id=pending.origin.branch_id,
                state=RequestState.PENDING,
            )
        ),
    )
    stream = (
        _stream_item(request, StreamItemKind.STREAM_STARTED, 0),
        _stream_item(request, StreamItemKind.INTERACTION_CREATED, 1),
        _stream_item(pending, StreamItemKind.INTERACTION_PENDING, 2),
        _stream_item(
            pending,
            StreamItemKind.STREAM_INPUT_REQUIRED,
            3,
            terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
        ),
        _stream_item(pending, StreamItemKind.STREAM_CLOSED, 4),
    )
    resolution = AnsweredResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=2),
        answers=(
            TextAnswer(
                question_id=QuestionId("target"),
                provenance=AnswerProvenance.HUMAN,
                value="staging",
            ),
        ),
    )
    decoded_resolution = decode_input_resolution(
        encode_input_resolution(resolution)
    )
    transition = resolve_request(
        pending,
        decoded_resolution,
        expected_state_revision=StateRevision(1),
    )

    assert all(
        "Which target?" not in repr(event) for event in lifecycle_events
    )
    assert validate_canonical_stream_items(stream) == stream
    assert isinstance(transition, InputTransitionApplied)
    outcome = project_resolution_to_model(
        transition.request,
        containing_run_exists=True,
    )
    assert isinstance(outcome, ResumeInputContinuation)
    assert encode_continuation_outcome(outcome) == {
        "disposition": "resume",
        "request_id": "request-1",
        "result": {
            "kind": "answered",
            "request_id": "request-1",
            "provenance": "human",
            "resolved_at": "2026-07-20T20:00:02.000000Z",
            "answers": [
                {
                    "question_id": "target",
                    "kind": "text",
                    "provenance": "human",
                    "value": "staging",
                }
            ],
        },
    }
    for nonanswer_resolution, mode, resumes in _TERMINAL_NONANSWER_CASES:
        _assert_terminal_nonanswer_outcome(
            nonanswer_resolution,
            mode,
            resumes,
        )


def test_codec_and_snapshot_round_trip() -> None:
    """Round-trip the canonical request and its versioned snapshot."""
    request = _request()
    encoded_request = encode_input_request(request)
    snapshot = InteractionSnapshot(request=request)
    encoded_snapshot = encode_interaction_snapshot(snapshot)

    assert decode_input_request(encoded_request) == request
    assert decode_interaction_snapshot(encoded_snapshot) == snapshot
    assert decode_interaction_snapshot(encoded_snapshot.encode()) == snapshot


def test_illegal_transition_preserves_prior_request() -> None:
    """Reject an illegal mutation without changing the prior value."""
    pending = _pending(_request())
    before = encode_input_request(pending)

    rejected = mark_request_pending(
        pending,
        expected_state_revision=StateRevision(1),
    )

    assert isinstance(rejected, InputTransitionRejected)
    assert rejected.previous is pending
    assert encode_input_request(pending) == before


_TERMINAL_NONANSWER_CASES = (
    (
        DeclinedResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=2),
        ),
        RequirementMode.REQUIRED,
        True,
    ),
    (
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=2),
        ),
        RequirementMode.REQUIRED,
        True,
    ),
    (
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=2),
            scope=CancellationScope.CONTAINING_RUN,
        ),
        RequirementMode.REQUIRED,
        False,
    ),
    (
        TimedOutResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=60),
        ),
        RequirementMode.ADVISORY,
        True,
    ),
    (
        UnavailableResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=2),
        ),
        RequirementMode.REQUIRED,
        True,
    ),
    (
        ExpiredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=2),
        ),
        RequirementMode.REQUIRED,
        False,
    ),
    (
        SupersededResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=2),
        ),
        RequirementMode.REQUIRED,
        False,
    ),
)


def _assert_terminal_nonanswer_outcome(
    resolution: InputResolution,
    mode: RequirementMode,
    resumes: bool,
) -> None:
    pending = _pending(_request(mode=mode))
    decoded = decode_input_resolution(encode_input_resolution(resolution))
    if isinstance(decoded, TimedOutResolution):
        assert pending.advisory_deadline == decoded.resolved_at
    transition = resolve_request(
        pending,
        decoded,
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(transition, InputTransitionApplied)

    outcome = project_resolution_to_model(
        transition.request,
        containing_run_exists=True,
    )

    if resumes:
        assert isinstance(outcome, ResumeInputContinuation)
        assert outcome.result.kind.value == decoded.status.value
    else:
        assert isinstance(outcome, TerminateInputContinuation)
        assert outcome.status is decoded.status


@pytest.mark.parametrize(
    ("resolution", "mode", "resumes"),
    _TERMINAL_NONANSWER_CASES,
)
def test_terminal_nonanswer_outcome_matrix(
    resolution: InputResolution,
    mode: RequirementMode,
    resumes: bool,
) -> None:
    """Project every explicit terminal non-answer outcome."""
    assert not isinstance(resolution, AnsweredResolution)
    _assert_terminal_nonanswer_outcome(resolution, mode, resumes)
