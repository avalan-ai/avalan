"""Exercise the explicit headless policy contract."""

from asyncio import run as run_async
from asyncio import wait_for
from builtins import input as builtin_input
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from avalan import (
    AgentRunInputRequired,
    InputContinuationRef,
    InputRequestRef,
    InputRequestView,
)
from avalan.interaction import (
    AgentId,
    AnswerProvenance,
    BranchId,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DurableInteractionSuspension,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputRequest,
    InputRequestId,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    RunId,
    StateRevision,
    StreamSessionId,
    TurnId,
    create_input_request,
)
from avalan.interaction.error import InputValidationError
from avalan.interaction.headless import (
    DeclineInputPolicy,
    DurableHandoffInputPolicy,
    ExternalControllerInputPolicy,
    PolicyValueInputPolicy,
    PredeclaredInputPolicy,
    TrustedDefaultInputPolicy,
    UnavailableInputPolicy,
)

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def _context() -> InputHandlerContext:
    request = create_input_request(
        request_id=InputRequestId("request-headless"),
        continuation_id=ContinuationId("continuation-headless"),
        origin=ExecutionOrigin(
            run_id=RunId("run-headless"),
            turn_id=TurnId("turn-headless"),
            agent_id=AgentId("agent-headless"),
            branch_id=BranchId("branch-headless"),
            model_call_id=ModelCallId("call-headless"),
            stream_session_id=StreamSessionId("stream-headless"),
            definition=ExecutionDefinitionRef(
                agent_definition_locator="agent://headless-contract",
                agent_definition_revision="agent-r1",
                operation_id="operation",
                operation_index=0,
                model_config_reference="model-r1",
                tool_revision="tools-r1",
                capability_revision="capability-r1",
            ),
            principal=PrincipalScope(),
        ),
        mode=RequirementMode.REQUIRED,
        reason="Choose whether to continue.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=True,
            ),
        ),
        created_at=_NOW,
    )
    return InputHandlerContext(
        request=replace(
            request,
            state=RequestState.PENDING,
            state_revision=StateRevision(1),
        )
    )


def _policy_answer() -> ConfirmationAnswer:
    return ConfirmationAnswer(
        question_id=QuestionId("confirm"),
        provenance=AnswerProvenance.POLICY,
        value=True,
    )


def test_requirement_input_n_068() -> None:
    """Cover every explicit headless policy family with async effects."""

    async def exercise() -> None:
        context = _context()

        async def controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            assert received is context
            return InputHandlerDetached()

        async def provider(
            received: InputHandlerContext,
        ) -> tuple[ConfirmationAnswer, ...]:
            assert received is context
            return (_policy_answer(),)

        async def handoff(
            value: DurableInteractionSuspension,
        ) -> InputRequest:
            assert value is not None
            return context.request

        policies = (
            PredeclaredInputPolicy(answers=(_policy_answer(),)),
            ExternalControllerInputPolicy(controller=controller),
            DurableHandoffInputPolicy(handoff=handoff),
            TrustedDefaultInputPolicy(),
            PolicyValueInputPolicy(provider=provider),
            DeclineInputPolicy(),
            UnavailableInputPolicy(),
        )
        outcomes = tuple([await policy(context) for policy in policies])
        assert len(outcomes) == 7
        assert isinstance(outcomes[1], InputHandlerDetached)
        assert isinstance(outcomes[2], InputHandlerDetached)
        assert isinstance(outcomes[-1], InputHandlerDisconnected)
        assert outcomes[-1].reason is InputDisconnectReason.HANDLER_UNAVAILABLE

    run_async(exercise())


def test_requirement_input_n_069() -> None:
    """Fail fast without stdin, sync callbacks, or implicit answers."""

    async def exercise() -> None:
        context = _context()
        waits: list[int] = []

        async def handoff(
            value: DurableInteractionSuspension,
        ) -> InputRequest:
            assert value is not None
            return context.request

        async def waiter(seconds: int) -> None:
            waits.append(seconds)

        with patch(
            "builtins.input",
            side_effect=AssertionError("headless policy read stdin"),
        ):
            unavailable = await wait_for(
                UnavailableInputPolicy()(context),
                timeout=0.1,
            )
            durable = DurableHandoffInputPolicy(
                handoff=handoff,
                durable_handoff_wait_seconds=3,
                waiter=waiter,
            )
            detached = await wait_for(durable(context), timeout=0.1)
            await wait_for(durable.wait(), timeout=0.1)
        assert builtin_input is not None
        assert isinstance(unavailable, InputHandlerDisconnected)
        assert isinstance(detached, InputHandlerDetached)
        assert waits == [3]

    def synchronous(value: DurableInteractionSuspension) -> InputRequest:
        assert value is not None
        return _context().request

    with pytest.raises(InputValidationError):
        DurableHandoffInputPolicy(
            handoff=synchronous,  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        PolicyValueInputPolicy(
            provider=synchronous,  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        ExternalControllerInputPolicy(
            controller=synchronous,  # type: ignore[arg-type]
        )
    run_async(exercise())


def test_requirement_input_n_070() -> None:
    """Expose a machine-readable pause with opaque durable correlation."""
    request = _context().request
    result = AgentRunInputRequired(
        request=InputRequestView(
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=request.created_at,
            state=request.state,
            state_revision=request.state_revision,
        ),
        request_id=InputRequestRef("opaque-request-reference"),
        continuation_id=InputContinuationRef("opaque-continuation-reference"),
        detached_resumption_available=True,
    )
    envelope = {
        "kind": result.kind.value,
        "request_id": result.request_id,
        "continuation_id": result.continuation_id,
        "detached_resumption_available": result.detached_resumption_available,
        "channel": result.channel,
    }
    assert envelope == {
        "kind": "input_required",
        "request_id": "opaque-request-reference",
        "continuation_id": "opaque-continuation-reference",
        "detached_resumption_available": True,
        "channel": "typed",
    }
