"""Pin exact transcript materialization and interaction correlation."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from unittest import IsolatedAsyncioTestCase

from avalan.agent.execution import (
    AgentExecution,
    ExecutionCorrelationError,
    UuidExecutionIdFactory,
)
from avalan.entities import (
    Message,
    MessageRole,
    MessageToolCall,
    normalize_tool_arguments,
)
from avalan.interaction.codec import encode_input_model_result
from avalan.interaction.entities import (
    AgentId,
    AnswerProvenance,
    BranchId,
    ConfirmationQuestion,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    InputRequestId,
    InputUnavailableResult,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    RunId,
    StateRevision,
    StreamSessionId,
    TaskId,
    TurnId,
    UnavailableResolution,
    UserId,
    create_input_request,
)
from avalan.interaction.policy import InteractionActor, InteractionPolicy
from avalan.interaction.store import (
    CreateInteractionCommand,
    apply_create_interaction,
)
from avalan.model.capability import (
    CorrelatedCapabilityResult,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.types import JsonValue

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _origin() -> ExecutionOrigin:
    principal = PrincipalScope(user_id=UserId("exact-user"))
    return ExecutionOrigin(
        run_id=RunId("exact-run"),
        turn_id=TurnId("exact-turn"),
        task_id=TaskId("exact-task"),
        agent_id=AgentId("exact-agent"),
        branch_id=BranchId("exact-branch"),
        model_call_id=ModelCallId("exact-model-call"),
        stream_session_id=StreamSessionId("exact-stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://exact",
            agent_definition_revision="agent-r1",
            operation_id="operation-exact",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
        principal=principal,
    )


def _questions() -> tuple[ConfirmationQuestion, ...]:
    return (
        ConfirmationQuestion(
            question_id=QuestionId("continue"),
            prompt="Continue?",
            required=True,
        ),
    )


def _call() -> TaskInputCapabilityCall:
    return TaskInputCapabilityCall(
        call_id="exact-input-call",
        provider_name="request_user_input",
        arguments={
            "mode": "required",
            "reason": "Choose exactly.",
            "questions": (
                {
                    "question_id": "continue",
                    "kind": "confirmation",
                    "prompt": "Continue?",
                    "required": True,
                    "choices": (),
                    "allow_other": False,
                },
            ),
        },
        mode=RequirementMode.REQUIRED,
        reason="Choose exactly.",
        questions=_questions(),
        advertisement=TaskInputCapabilityAdvertisement.ATTACHED,
    )


def _pending(origin: ExecutionOrigin) -> InputRequest:
    created = create_input_request(
        request_id=InputRequestId("exact-request"),
        continuation_id=ContinuationId("exact-continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Choose exactly.",
        questions=_questions(),
        created_at=_NOW,
    )
    return apply_create_interaction(
        CreateInteractionCommand(
            actor=InteractionActor(principal=origin.principal),
            request=created,
        ),
        InteractionPolicy(),
    ).record.request


def _terminal(pending: InputRequest) -> InputRequest:
    resolution = UnavailableResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
    )
    return replace(
        pending,
        state=RequestState.UNAVAILABLE,
        state_revision=StateRevision(pending.state_revision + 1),
        resolution=resolution,
    )


def _result(request: InputRequest) -> InputUnavailableResult:
    return InputUnavailableResult(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
    )


def _assistant_message(content: str | None = None) -> Message:
    call = _call()
    return Message(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=[
            MessageToolCall(
                id=str(call.call_id),
                name=call.provider_name,
                arguments=normalize_tool_arguments(call.arguments),
            )
        ],
    )


def _correlated_messages(
    result: InputUnavailableResult,
    *,
    assistant_content: str | None = None,
) -> tuple[Message, Message]:
    call = _call()
    correlated = CorrelatedCapabilityResult(
        call_id=call.call_id,
        canonical_name=call.canonical_name,
        provider_name=call.provider_name,
        payload=cast(
            Mapping[str, JsonValue],
            encode_input_model_result(result),
        ),
    )
    return (
        _assistant_message(assistant_content),
        correlated.local_message(),
    )


class ResponseMaterializationTest(IsolatedAsyncioTestCase):
    """Reject a transcript that substitutes the recorded response."""

    async def test_transcript_response_must_be_one_exact_message(self) -> None:
        invalid_messages = (
            (Message(role=MessageRole.ASSISTANT, content="substituted"),),
            (
                Message(role=MessageRole.ASSISTANT, content="answer"),
                Message(role=MessageRole.TOOL, content="extra"),
            ),
            (
                Message(
                    role=MessageRole.ASSISTANT,
                    content="answer",
                    name="substituted",
                ),
            ),
            (Message(role=MessageRole.TOOL, content="forged extra"),),
        )
        for messages in invalid_messages:
            with (
                self.subTest(messages=messages),
                self.assertRaises(ExecutionCorrelationError),
            ):
                execution = AgentExecution(
                    origin=_origin(),
                    id_factory=UuidExecutionIdFactory(),
                    initial_messages=(),
                )
                await execution.record_response("answer", messages=messages)


class CorrelatedInteractionMessagesTest(IsolatedAsyncioTestCase):
    """Require the exact ordered call/result pair and nothing else."""

    async def _pending_execution(
        self,
        *,
        assistant_content: str | None = None,
    ) -> tuple[AgentExecution, InputRequest, InputUnavailableResult]:
        origin = _origin()
        execution = AgentExecution(
            origin=origin,
            id_factory=UuidExecutionIdFactory(),
            initial_messages=(),
        )
        await execution.begin_interaction(
            "exact-fingerprint",
            _call(),
            _assistant_message(assistant_content),
        )
        pending = _pending(origin)
        await execution.mark_interaction_pending(pending)
        terminal = _terminal(pending)
        return execution, terminal, _result(terminal)

    async def test_result_messages_reject_order_substitution_and_extras(
        self,
    ) -> None:
        _, _, result = await self._pending_execution()
        assistant, tool = _correlated_messages(result)
        invalid_messages = (
            (tool, assistant),
            (replace(assistant, content="forged preamble"), tool),
            (
                replace(assistant, content="preamble"),
                tool,
                Message(role=MessageRole.USER, content="extra"),
            ),
            (
                replace(
                    assistant,
                    tool_calls=[
                        *(assistant.tool_calls or ()),
                        MessageToolCall(name="extra", arguments={}),
                    ],
                ),
                tool,
            ),
        )
        for messages in invalid_messages:
            with (
                self.subTest(messages=messages),
                self.assertRaises(ExecutionCorrelationError),
            ):
                execution, terminal, result = await self._pending_execution()
                await execution.record_interaction_result(
                    terminal,
                    result,
                    messages,
                )

        execution, terminal, result = await self._pending_execution(
            assistant_content="trusted preamble"
        )
        assistant, tool = _correlated_messages(
            result,
            assistant_content="trusted preamble",
        )
        self.assertTrue(
            await execution.record_interaction_result(
                terminal,
                result,
                (assistant, tool),
            )
        )
