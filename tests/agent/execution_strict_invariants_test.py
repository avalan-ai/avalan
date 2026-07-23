"""Pin fail-closed execution, broker, and tool-argument invariants."""

from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal, cast, get_type_hints
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    ExecutionBranchInteractionBroker,
    ExecutionCorrelationError,
    ExecutionLedgerEntry,
    ExecutionLedgerEntryKind,
    ExecutionMemoryComponent,
    ExecutionMemoryEntry,
    ExecutionRevisionError,
    ExecutionStateError,
    ModelPromptRecord,
    UuidExecutionIdFactory,
)
from avalan.entities import (
    Message,
    MessageRole,
    MessageToolCall,
    ToolValue,
    normalize_tool_arguments,
)
from avalan.interaction import semantic_request_fingerprint
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerRequest,
    InteractionDelivery,
    InteractionRequestResult,
)
from avalan.interaction.entities import (
    AgentId,
    BranchId,
    ConfirmationQuestion,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    InputRequestId,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    RunId,
    StreamSessionId,
    TaskId,
    TurnId,
    UserId,
    create_input_request,
)
from avalan.interaction.policy import InteractionActor, InteractionPolicy
from avalan.interaction.store import (
    CreateInteractionCommand,
    TerminalizeInteractionScopeCommand,
    apply_create_interaction,
)
from avalan.model.capability import (
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _principal(user_id: str = "strict-user") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(user_id))


def _definition() -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://strict",
        agent_definition_revision="agent-r1",
        operation_id="operation-strict",
        operation_index=0,
        model_config_reference="model-config-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("strict-run"),
        turn_id=TurnId("strict-turn"),
        task_id=TaskId("strict-task"),
        agent_id=AgentId("strict-agent"),
        branch_id=BranchId("strict-branch"),
        model_call_id=ModelCallId("strict-model-call"),
        stream_session_id=StreamSessionId("strict-stream"),
        definition=_definition(),
        principal=_principal(),
    )


def _questions(
    *, prompt: str = "Continue?"
) -> tuple[ConfirmationQuestion, ...]:
    return (
        ConfirmationQuestion(
            question_id=QuestionId("continue"),
            prompt=prompt,
            required=True,
        ),
    )


def _broker_request(
    origin: ExecutionOrigin,
    actor: InteractionActor,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=actor,
        origin=origin,
        mode=RequirementMode.ADVISORY,
        reason="Choose how execution should continue.",
        questions=_questions(),
        continuation_ttl_seconds=3_600,
        advisory_wait_seconds=30,
    )


def _task_input_call() -> TaskInputCapabilityCall:
    return TaskInputCapabilityCall(
        call_id="strict-input-call",
        provider_name="request_user_input",
        arguments={
            "mode": "required",
            "reason": "Choose how execution should continue.",
            "questions": {
                "question_id": "continue",
                "kind": "confirmation",
                "prompt": "Continue?",
                "required": True,
                "choices": (),
                "allow_other": False,
            },
        },
        mode=RequirementMode.REQUIRED,
        reason="Choose how execution should continue.",
        questions=_questions(),
        advertisement=TaskInputCapabilityAdvertisement.ATTACHED,
    )


def _task_input_message() -> Message:
    call = _task_input_call()
    return Message(
        role=MessageRole.ASSISTANT,
        tool_calls=[
            MessageToolCall(
                id=str(call.call_id),
                name=call.provider_name,
                arguments=normalize_tool_arguments(call.arguments),
            )
        ],
    )


def _pending_request(
    origin: ExecutionOrigin,
    *,
    reason: str = "Choose how execution should continue.",
) -> InputRequest:
    actor = InteractionActor(principal=origin.principal)
    created = create_input_request(
        request_id=InputRequestId("strict-request"),
        continuation_id=ContinuationId("strict-continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason=reason,
        questions=_questions(),
        created_at=_NOW,
    )
    return apply_create_interaction(
        CreateInteractionCommand(actor=actor, request=created),
        InteractionPolicy(),
    ).record.request


class _SubstitutingBroker:
    """Return internally valid admission state for a different contract."""

    def __init__(self, substitution: str) -> None:
        self.substitution = substitution

    async def request(self, request: InteractionBrokerRequest) -> object:
        if self.substitution == "result_type":
            return object()
        actor = request.actor
        origin = request.origin
        mode = request.mode
        reason = request.reason
        questions = request.questions
        continuation_ttl_seconds = request.continuation_ttl_seconds
        advisory_wait_seconds = request.advisory_wait_seconds
        if self.substitution == "actor":
            actor = InteractionActor(principal=_principal("other-user"))
        elif self.substitution == "origin":
            origin = replace(origin, branch_id=BranchId("other-branch"))
        elif self.substitution == "mode":
            mode = RequirementMode.REQUIRED
            advisory_wait_seconds = None
        elif self.substitution == "reason":
            reason = "Use a substituted reason."
        elif self.substitution == "questions":
            questions = _questions(prompt="Use the substituted question?")
        elif self.substitution == "continuation_ttl_seconds":
            continuation_ttl_seconds += 1
        elif self.substitution == "advisory_wait_seconds":
            assert advisory_wait_seconds is not None
            advisory_wait_seconds += 1
        created = create_input_request(
            request_id=InputRequestId("broker-request"),
            continuation_id=ContinuationId("broker-continuation"),
            origin=origin,
            mode=mode,
            reason=reason,
            questions=questions,
            created_at=_NOW,
            continuation_ttl_seconds=continuation_ttl_seconds,
            advisory_wait_seconds=advisory_wait_seconds,
        )
        applied = apply_create_interaction(
            CreateInteractionCommand(actor=actor, request=created),
            InteractionPolicy(),
        )
        delivery_record = applied.record
        if self.substitution == "delivery_record":
            altered_request = replace(
                delivery_record.request,
                reason="Delivery substituted this reason.",
            )
            delivery_record = replace(
                delivery_record,
                request=altered_request,
                semantic_fingerprint=semantic_request_fingerprint(
                    altered_request
                ),
            )
        return InteractionRequestResult(
            create_result=applied,
            delivery=InteractionDelivery(
                correlation=delivery_record.correlation,
                record=delivery_record,
                handler_attempts=0,
            ),
        )

    async def cancel_scope(
        self,
        _command: TerminalizeInteractionScopeCommand,
    ) -> object:
        raise AssertionError("cancellation is outside this test")


class BrokerContractValidationTest(IsolatedAsyncioTestCase):
    """Reject broker substitution before exposing authoritative state."""

    async def test_broker_result_must_match_every_requested_field(
        self,
    ) -> None:
        origin = _origin()
        actor = InteractionActor(principal=origin.principal)
        request = _broker_request(origin, actor)
        substitutions = (
            "result_type",
            "actor",
            "origin",
            "mode",
            "reason",
            "questions",
            "continuation_ttl_seconds",
            "advisory_wait_seconds",
            "delivery_record",
        )
        for substitution in substitutions:
            with (
                self.subTest(substitution=substitution),
                self.assertRaises(ExecutionCorrelationError),
            ):
                broker = ExecutionBranchInteractionBroker(
                    broker=cast(
                        InteractionBroker,
                        _SubstitutingBroker(substitution),
                    ),
                    actor=actor,
                    current_origin=lambda: origin,
                )
                await broker.request(request)

    async def test_pending_request_must_match_reserved_call(self) -> None:
        origin = _origin()
        execution = AgentExecution(
            origin=origin,
            id_factory=UuidExecutionIdFactory(),
            initial_messages=(),
        )
        await execution.begin_interaction(
            "strict-fingerprint",
            _task_input_call(),
            _task_input_message(),
        )
        with self.assertRaises(ExecutionCorrelationError):
            await execution.mark_interaction_pending(
                _pending_request(origin, reason="Substituted reason."),
            )
        self.assertEqual(
            execution.status,
            AgentExecutionStatus.PREPARING_INPUT,
        )


class SnapshotLedgerReplayTest(TestCase):
    """Reject public snapshots that disagree with legal ledger replay."""

    def setUp(self) -> None:
        self.origin = _origin()
        self.execution = AgentExecution(
            origin=self.origin,
            id_factory=UuidExecutionIdFactory(),
            initial_messages=(
                Message(role=MessageRole.USER, content="Initial input."),
            ),
        )

    def test_snapshot_rejects_forged_derived_state(self) -> None:
        snapshot = self.execution.snapshot
        pending = _pending_request(self.origin)
        forged_states = (
            {"status": AgentExecutionStatus.COMPLETED},
            {
                "messages": (
                    *snapshot.messages,
                    Message(role=MessageRole.ASSISTANT, content="Forged."),
                )
            },
            {
                "status": AgentExecutionStatus.WAITING_FOR_INPUT,
                "pending_request": pending,
                "active_interaction_fingerprint": "strict-fingerprint",
                "interaction_fingerprint_counts": (("strict-fingerprint", 1),),
                "interaction_count": 1,
            },
        )
        for fields in forged_states:
            with (
                self.subTest(fields=tuple(fields)),
                self.assertRaises(ExecutionStateError),
            ):
                replace(snapshot, **fields)

    def test_snapshot_rejects_illegal_transition_and_revision(self) -> None:
        snapshot = self.execution.snapshot
        pending = _pending_request(self.origin)
        illegal_pending = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.INTERACTION_PENDING,
            origin=self.origin,
            request=pending,
        )
        with self.assertRaises(ExecutionStateError):
            replace(
                snapshot,
                revision=1,
                status=AgentExecutionStatus.WAITING_FOR_INPUT,
                ledger=(*snapshot.ledger, illegal_pending),
                pending_request=pending,
                active_interaction_fingerprint="strict-fingerprint",
                interaction_fingerprint_counts=(("strict-fingerprint", 1),),
                interaction_count=1,
            )
        with self.assertRaises(ExecutionRevisionError):
            replace(snapshot, revision=1)


class SnapshotRevisionReplayTest(IsolatedAsyncioTestCase):
    """Require revisions reachable through exact cursor advancement."""

    async def test_response_cursor_tail_requires_its_own_revision(
        self,
    ) -> None:
        execution = AgentExecution(
            origin=_origin(),
            id_factory=UuidExecutionIdFactory(),
            initial_messages=(),
        )
        await execution.record_response("unsaved response")
        await execution.record_prompt(
            ModelPromptRecord(
                input="next prompt",
                instructions=None,
                system_prompt=None,
                developer_prompt=None,
            )
        )

        class _Sink:
            async def append_execution_memory_entry(
                _self,
                entry: ExecutionMemoryEntry,
            ) -> None:
                self.assertIs(
                    entry.component,
                    ExecutionMemoryComponent.RESPONSE,
                )

        await execution.sync_memory(_Sink())
        snapshot = execution.snapshot
        self.assertEqual(snapshot.revision, 4)
        self.assertEqual(snapshot.response_sync_cursor, 3)
        with self.assertRaises(ExecutionRevisionError):
            replace(snapshot, revision=3)


class MessageToolCallArgumentsTest(TestCase):
    """Keep message tool-call arguments inside a detached JSON object."""

    def test_arguments_are_closed_normalized_and_detached(self) -> None:
        source: dict[str, object] = {
            "nested": [{"value": "before"}],
            "tuple": (
                MappingProxyType({"enabled": True}),
                None,
            ),
        }
        call = MessageToolCall(
            name="strict_tool",
            arguments=normalize_tool_arguments(source),
        )
        source_nested = cast(list[object], source["nested"])
        source_item = cast(dict[str, object], source_nested[0])
        source_item["value"] = "after"
        self.assertEqual(
            call.arguments,
            {
                "nested": [{"value": "before"}],
                "tuple": [{"enabled": True}, None],
            },
        )
        self.assertEqual(
            get_type_hints(MessageToolCall)["arguments"],
            dict[str, ToolValue],
        )

    def test_arguments_reject_non_json_shapes(self) -> None:
        invalid_values: tuple[object, ...] = (
            {1: "non-string key"},
            {"nested": {1: "non-string key"}},
            {"bytes": b"not-json"},
            {"object": object()},
            {"number": float("nan")},
            {"number": float("inf")},
        )
        for value in invalid_values:
            with (
                self.subTest(value=value),
                self.assertRaises((TypeError, ValueError)),
            ):
                normalize_tool_arguments(value)
        with self.assertRaises(TypeError):
            MessageToolCall(
                name="strict_tool",
                arguments=cast(dict[str, ToolValue], object()),
            )
        with self.assertRaises(ValueError):
            MessageToolCall(
                name="strict_tool",
                arguments={},
                content_type=cast(Literal["json"], "text"),
            )
