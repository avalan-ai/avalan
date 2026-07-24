"""Exercise strict async interaction handler and resumer contracts."""

from asyncio import run as asyncio_run
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import cast

import pytest

from avalan.interaction import (
    AgentId,
    AnswerProvenance,
    BranchId,
    CancellationScope,
    CancelledResolution,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    InputCandidateResolution,
    InputDeclinedResult,
    InputDisconnectReason,
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputRequest,
    InputRequestId,
    InputResolution,
    InputResumer,
    InputResumerRegistration,
    InputResumptionNotification,
    InputValidationError,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    ResumeInputContinuation,
    RunId,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    create_input_request,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _request() -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=ExecutionOrigin(
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
            principal=PrincipalScope(),
        ),
        mode=RequirementMode.REQUIRED,
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


class _Handler:
    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        return InputHandlerResolution(
            resolution=DeclinedResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
            )
        )


class _Resumer:
    def __init__(self) -> None:
        self.notification: InputResumptionNotification | None = None

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        self.notification = notification


class _SyncResumer:
    def __call__(self, notification: InputResumptionNotification) -> None:
        del notification


def test_async_handler_and_in_process_resumer_are_typed() -> None:
    """Invoke typed async boundaries without persisting callback objects."""
    asyncio_run(_exercise_async_handler_and_resumer())


async def _exercise_async_handler_and_resumer() -> None:
    request = _request()
    handler: InputHandler = _Handler()
    outcome = await handler(InputHandlerContext(request=request))
    assert isinstance(outcome, InputHandlerResolution)

    resumer = _Resumer()
    typed_resumer: InputResumer = resumer
    registration = InputResumerRegistration(
        continuation_id=request.continuation_id,
        resumer=typed_resumer,
    )
    notification = InputResumptionNotification(
        continuation_id=request.continuation_id,
        state_revision=StateRevision(2),
        outcome=ResumeInputContinuation(
            request_id=request.request_id,
            result=InputDeclinedResult(
                request_id=request.request_id,
                provenance=outcome.resolution.provenance,
                resolved_at=outcome.resolution.resolved_at,
            ),
        ),
    )
    await registration.resumer(notification)

    assert resumer.notification == notification


def test_handler_outcomes_distinguish_detach_and_disconnect() -> None:
    """Keep intentional detach separate from attached-channel loss."""
    detached = InputHandlerDetached()
    disconnected = InputHandlerDisconnected(
        reason=InputDisconnectReason.CONTROL_CHANNEL_CLOSED
    )

    assert detached.kind.value == "detached"
    assert disconnected.kind.value == "disconnected"
    with pytest.raises(FrozenInstanceError):
        setattr(detached, "kind", disconnected.kind)
    with pytest.raises(InputValidationError):
        InputHandlerDisconnected(reason="closed")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "resolution",
    (
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            scope=CancellationScope.REQUEST,
        ),
        TimedOutResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        UnavailableResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        ExpiredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        SupersededResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
    ),
)
def test_handler_resolution_rejects_non_candidate_outcomes(
    resolution: InputResolution,
) -> None:
    """Keep cancellation and system settlement out of attached handlers."""
    with pytest.raises(InputValidationError) as error:
        InputHandlerResolution(
            resolution=cast(InputCandidateResolution, resolution)
        )

    assert error.value.path == "handler.resolution"


def test_resumer_registration_rejects_synchronous_callables() -> None:
    """Reject callable objects that cannot be awaited at runtime."""
    with pytest.raises(InputValidationError) as error:
        InputResumerRegistration(
            continuation_id=ContinuationId("continuation-1"),
            resumer=cast(InputResumer, _SyncResumer()),
        )

    assert error.value.path == "resumer.callback"
