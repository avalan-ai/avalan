"""Exercise defensive execution branches at explicit trust boundaries."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.agent import execution as execution_module
from avalan.agent.execution import (
    MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
    AgentExecution,
    AgentExecutionStatus,
    ExecutionBranchInteractionBroker,
    ExecutionCorrelationError,
    ExecutionLedgerEntry,
    ExecutionLedgerEntryKind,
    ExecutionMemoryEntry,
    ExecutionRevisionError,
    ExecutionStateError,
)
from avalan.entities import (
    Message,
    MessageRole,
    MessageToolCall,
    normalize_tool_arguments,
)
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionRequestResult,
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
    ExpiredResolution,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    InputUnavailableResult,
    InteractionStoreRevision,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    RunId,
    StateRevision,
    StreamSessionId,
    TaskId,
    TerminateInputContinuation,
    TurnId,
    UnavailableResolution,
    UserId,
    create_input_request,
)
from avalan.interaction.error import InputErrorCode
from avalan.interaction.policy import InteractionActor
from avalan.interaction.state import (
    InputTransitionError,
)
from avalan.interaction.store import (
    CreateInteractionApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionBranchRegistrationRejected,
    InteractionBranchRegistrationReplayed,
    InteractionExecutionScope,
    RegisterInteractionBranchCommand,
    ScopeCancellationRejected,
    ScopeCancellationReplayed,
    TerminalizeInteractionScopeCommand,
)
from avalan.model.capability import (
    CorrelatedCapabilityResult,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.types import JsonValue

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
_REASON = "Choose how the execution should continue."


def _definition() -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://coverage",
        agent_definition_revision="agent-r1",
        operation_id="operation-coverage",
        operation_index=1,
        model_config_reference="model-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )


def _principal(user_id: str = "user-1") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(user_id))


def _origin(
    *,
    parent_branch_id: BranchId | None = None,
    principal: PrincipalScope | None = None,
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        parent_branch_id=parent_branch_id,
        model_call_id=ModelCallId("model-call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=_definition(),
        principal=principal or _principal(),
    )


def _question() -> ConfirmationQuestion:
    return ConfirmationQuestion(
        question_id=QuestionId("continue"),
        prompt="Continue?",
        required=True,
    )


def _task_input_call() -> TaskInputCapabilityCall:
    return TaskInputCapabilityCall(
        call_id="input-call",
        provider_name="request_user_input",
        arguments={
            "mode": "required",
            "reason": _REASON,
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
        reason=_REASON,
        questions=(_question(),),
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


def _created_request(
    origin: ExecutionOrigin,
    *,
    request_id: str = "request-1",
    continuation_id: str = "continuation-1",
    reason: str = _REASON,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(continuation_id),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason=reason,
        questions=(_question(),),
        created_at=_NOW,
    )


def _pending_request(
    origin: ExecutionOrigin,
    *,
    request_id: str = "request-1",
    continuation_id: str = "continuation-1",
) -> InputRequest:
    return replace(
        _created_request(
            origin,
            request_id=request_id,
            continuation_id=continuation_id,
        ),
        state=RequestState.PENDING,
        state_revision=StateRevision(1),
    )


def _resolved_request(request: InputRequest) -> InputRequest:
    return replace(
        request,
        state=RequestState.UNAVAILABLE,
        state_revision=StateRevision(request.state_revision + 1),
        resolution=UnavailableResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
    )


def _terminated_request(request: InputRequest) -> InputRequest:
    return replace(
        request,
        state=RequestState.EXPIRED,
        state_revision=StateRevision(request.state_revision + 1),
        resolution=ExpiredResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
    )


def _result(
    request: InputRequest,
    *,
    provenance: AnswerProvenance = AnswerProvenance.POLICY,
) -> InputUnavailableResult:
    return InputUnavailableResult(
        request_id=request.request_id,
        provenance=provenance,
        resolved_at=_NOW,
    )


def _correlated_messages(
    result: InputUnavailableResult,
) -> tuple[Message, ...]:
    call = _task_input_call()
    correlated = CorrelatedCapabilityResult(
        call_id=call.call_id,
        canonical_name=call.canonical_name,
        provider_name=call.provider_name,
        payload=cast(
            Mapping[str, JsonValue],
            encode_input_model_result(result),
        ),
    )
    return (_task_input_message(), correlated.local_message())


def _input_entry(origin: ExecutionOrigin) -> ExecutionLedgerEntry:
    return ExecutionLedgerEntry(
        sequence=0,
        kind=ExecutionLedgerEntryKind.INPUT,
        origin=origin,
    )


def _reserved_entry(
    origin: ExecutionOrigin,
    sequence: int,
    fingerprint: str = "fingerprint",
) -> ExecutionLedgerEntry:
    return ExecutionLedgerEntry(
        sequence=sequence,
        kind=ExecutionLedgerEntryKind.INTERACTION_RESERVED,
        origin=origin,
        semantic_fingerprint=fingerprint,
        task_input_call=_task_input_call(),
        interaction_assistant_message=_task_input_message(),
    )


def _abandoned_entry(
    origin: ExecutionOrigin,
    sequence: int,
    fingerprint: str = "fingerprint",
) -> ExecutionLedgerEntry:
    return ExecutionLedgerEntry(
        sequence=sequence,
        kind=ExecutionLedgerEntryKind.INTERACTION_ABANDONED,
        origin=origin,
        semantic_fingerprint=fingerprint,
    )


def _replay(*entries: ExecutionLedgerEntry) -> None:
    execution_module._replay_execution_ledger(entries)


def _broker_request(
    origin: ExecutionOrigin,
    actor: InteractionActor,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=actor,
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason=_REASON,
        questions=(_question(),),
    )


def _create_command(
    request: InteractionBrokerRequest,
) -> CreateInteractionCommand:
    return CreateInteractionCommand(
        actor=request.actor,
        request=_created_request(request.origin),
    )


def _transition_error() -> InputTransitionError:
    return InputTransitionError(
        code=InputErrorCode.ILLEGAL_TRANSITION,
        path="request",
        message="The request was rejected.",
    )


def _scope_command(
    origin: ExecutionOrigin,
    actor: InteractionActor,
    *,
    turn_id: TurnId | None = None,
) -> TerminalizeInteractionScopeCommand:
    return TerminalizeInteractionScopeCommand(
        actor=actor,
        scope=InteractionExecutionScope(
            run_id=origin.run_id,
            turn_id=turn_id,
            branch_id=origin.branch_id,
        ),
        provenance=AnswerProvenance.HUMAN,
    )


def _registration_command(
    origin: ExecutionOrigin,
    actor: InteractionActor,
) -> RegisterInteractionBranchCommand:
    parent_branch_id = origin.parent_branch_id or BranchId("parent-1")
    return RegisterInteractionBranchCommand(
        actor=actor,
        registration=InteractionBranchRegistration(
            run_id=origin.run_id,
            branch_id=origin.branch_id,
            parent_branch_id=parent_branch_id,
            principal=origin.principal,
        ),
    )


class _IdFactory:
    """Mint stable provider-turn identities for focused tests."""

    async def new_run_id(self) -> RunId:
        return RunId("run-new")

    async def new_turn_id(self) -> TurnId:
        return TurnId("turn-new")

    async def new_task_id(self) -> TaskId:
        return TaskId("task-new")

    async def new_model_call_id(self) -> ModelCallId:
        return ModelCallId("model-call-new")

    async def new_branch_id(self) -> BranchId:
        return BranchId("branch-new")

    async def new_stream_session_id(self) -> StreamSessionId:
        return StreamSessionId("stream-new")


class _CancellationBroker:
    """Return one scripted cancellation result."""

    def __init__(self, result: object) -> None:
        self._result = result

    async def cancel_scope(
        self,
        _command: TerminalizeInteractionScopeCommand,
    ) -> object:
        return self._result


class _RegistrationBroker:
    """Return one scripted branch-registration result."""

    def __init__(
        self,
        result_factory: Callable[[RegisterInteractionBranchCommand], object],
    ) -> None:
        self._result_factory = result_factory

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> object:
        return self._result_factory(command)


class _TranscriptCursorConflictSink:
    """Simulate an out-of-band transcript cursor writer."""

    def __init__(self, execution: AgentExecution) -> None:
        self._execution = execution

    async def append_execution_memory_entry(
        self,
        _entry: ExecutionMemoryEntry,
    ) -> None:
        current = self._execution._state
        self._execution._state = replace(
            current,
            revision=current.revision + 1,
            memory_sync_cursor=current.memory_sync_cursor + 1,
        )


class _ResponseCursorConflictSink:
    """Simulate an out-of-band response cursor writer."""

    def __init__(self, execution: AgentExecution) -> None:
        self._execution = execution

    async def append_execution_memory_entry(
        self,
        _entry: ExecutionMemoryEntry,
    ) -> None:
        current = self._execution._state
        self._execution._state = replace(
            current,
            revision=current.revision + 1,
            response_sync_cursor=current.response_sync_cursor + 1,
        )


class LedgerReplayDefenseTest(TestCase):
    """Reject every cross-entry ledger inconsistency."""

    def test_ledger_origin_guards_reject_empty_misordered_and_stale_tails(
        self,
    ) -> None:
        origin = _origin()
        with self.assertRaisesRegex(
            ExecutionStateError,
            "must not be empty",
        ):
            execution_module._validate_ledger_origins(origin, ())

        completed = ExecutionLedgerEntry(
            sequence=0,
            kind=ExecutionLedgerEntryKind.COMPLETED,
            origin=origin,
        )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "must begin with input",
        ):
            execution_module._validate_ledger_origins(origin, (completed,))

        advanced_origin = replace(
            origin,
            turn_id=TurnId("turn-new"),
            model_call_id=ModelCallId("model-call-new"),
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "does not match the ledger tail",
        ):
            execution_module._validate_ledger_origins(
                advanced_origin,
                (_input_entry(origin),),
            )

    def test_terminal_ledger_rejects_wrong_kind_and_post_terminal_work(
        self,
    ) -> None:
        origin = _origin()
        cancelled = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.CANCELLED,
            origin=origin,
        )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "does not match its status",
        ):
            execution_module._validate_terminal_ledger(
                AgentExecutionStatus.COMPLETED,
                (_input_entry(origin), cancelled),
                False,
            )

        completed = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.COMPLETED,
            origin=origin,
        )
        trailing = ExecutionLedgerEntry(
            sequence=2,
            kind=ExecutionLedgerEntryKind.TRANSCRIPT,
            origin=origin,
            messages=(Message(role=MessageRole.USER, content="late"),),
        )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "cannot follow termination",
        ):
            execution_module._validate_terminal_ledger(
                AgentExecutionStatus.COMPLETED,
                (_input_entry(origin), completed, trailing),
                False,
            )

    def test_replay_rejects_terminal_and_cleanup_ordering(self) -> None:
        origin = _origin()
        cancelled = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.CANCELLED,
            origin=origin,
        )
        cleanup = ExecutionLedgerEntry(
            sequence=2,
            kind=ExecutionLedgerEntryKind.CLEANUP_CLAIMED,
            origin=origin,
        )
        second_cleanup = replace(cleanup, sequence=3)
        with self.assertRaisesRegex(
            ExecutionStateError,
            "work after cleanup",
        ):
            _replay(
                _input_entry(origin),
                cancelled,
                cleanup,
                second_cleanup,
            )

        trailing = ExecutionLedgerEntry(
            sequence=2,
            kind=ExecutionLedgerEntryKind.TRANSCRIPT,
            origin=origin,
            messages=(Message(role=MessageRole.USER, content="late"),),
        )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "work after termination",
        ):
            _replay(_input_entry(origin), cancelled, trailing)

        with self.assertRaisesRegex(
            ExecutionStateError,
            "cleanup cannot precede",
        ):
            _replay(_input_entry(origin), cleanup)

    def test_replay_rejects_repeated_input_and_interaction_loop_overflow(
        self,
    ) -> None:
        origin = _origin()
        with self.assertRaisesRegex(
            ExecutionStateError,
            "repeated input",
        ):
            _replay(
                _input_entry(origin),
                replace(_input_entry(origin), sequence=1),
            )

        entries = [_input_entry(origin)]
        for index in range(MAXIMUM_EQUIVALENT_INPUT_REQUESTS + 1):
            entries.append(
                _reserved_entry(origin, len(entries), "same-fingerprint")
            )
            if index < MAXIMUM_EQUIVALENT_INPUT_REQUESTS:
                entries.append(
                    _abandoned_entry(
                        origin,
                        len(entries),
                        "same-fingerprint",
                    )
                )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "loop limit",
        ):
            execution_module._replay_execution_ledger(tuple(entries))

    def test_replay_rejects_interaction_correlation_substitutions(
        self,
    ) -> None:
        origin = _origin()
        reservation = _reserved_entry(origin, 1)
        mismatched_abandonment = _abandoned_entry(
            origin,
            2,
            "different-fingerprint",
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed its reservation",
        ):
            _replay(
                _input_entry(origin),
                reservation,
                mismatched_abandonment,
            )

        other_origin = replace(
            origin,
            turn_id=TurnId("turn-other"),
            model_call_id=ModelCallId("model-call-other"),
        )
        wrong_origin_pending = _pending_request(other_origin)
        pending_entry = ExecutionLedgerEntry(
            sequence=2,
            kind=ExecutionLedgerEntryKind.INTERACTION_PENDING,
            origin=origin,
            request=wrong_origin_pending,
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed execution origin",
        ):
            _replay(_input_entry(origin), reservation, pending_entry)

        pending = _pending_request(origin)
        pending_entry = replace(pending_entry, request=pending)
        wrong_required = InputRequiredResult(
            request_id=InputRequestId("request-other"),
            continuation_id=pending.continuation_id,
            detached_resumption_available=True,
        )
        required_entry = ExecutionLedgerEntry(
            sequence=3,
            kind=ExecutionLedgerEntryKind.INPUT_REQUIRED,
            origin=origin,
            request=pending,
            input_required=wrong_required,
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "not correlated",
        ):
            _replay(
                _input_entry(origin),
                reservation,
                pending_entry,
                required_entry,
            )

        resolved = _resolved_request(pending)
        wrong_result = _result(
            resolved,
            provenance=AnswerProvenance.HUMAN,
        )
        result_entry = ExecutionLedgerEntry(
            sequence=3,
            kind=ExecutionLedgerEntryKind.INTERACTION_RESULT,
            origin=origin,
            messages=_correlated_messages(wrong_result),
            request=resolved,
            result=wrong_result,
            task_input_call=_task_input_call(),
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed its continuation",
        ):
            _replay(
                _input_entry(origin),
                reservation,
                pending_entry,
                result_entry,
            )

        terminated = _terminated_request(pending)
        wrong_termination = TerminateInputContinuation(
            request_id=terminated.request_id,
            status=ResolutionStatus.CANCELLED,
        )
        termination_entry = ExecutionLedgerEntry(
            sequence=3,
            kind=ExecutionLedgerEntryKind.INTERACTION_TERMINATED,
            origin=origin,
            request=terminated,
            termination_outcome=wrong_termination,
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed its continuation",
        ):
            _replay(
                _input_entry(origin),
                reservation,
                pending_entry,
                termination_entry,
            )

    def test_replay_rejects_model_turn_while_interaction_is_reserved(
        self,
    ) -> None:
        origin = _origin()
        turn = ExecutionLedgerEntry(
            sequence=2,
            kind=ExecutionLedgerEntryKind.MODEL_TURN,
            origin=replace(
                origin,
                turn_id=TurnId("turn-new"),
                model_call_id=ModelCallId("model-call-new"),
            ),
        )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "illegal execution state",
        ):
            _replay(
                _input_entry(origin),
                _reserved_entry(origin, 1),
                turn,
            )


class BranchBrokerDefenseTest(IsolatedAsyncioTestCase):
    """Reject invalid branch broker values and substituted store results."""

    def _wrapper(
        self,
        broker: object,
        origin: ExecutionOrigin,
        actor: InteractionActor,
    ) -> ExecutionBranchInteractionBroker:
        return ExecutionBranchInteractionBroker(
            broker=cast(InteractionBroker, broker),
            actor=actor,
            current_origin=lambda: origin,
        )

    async def test_public_type_actor_and_scope_guards(self) -> None:
        origin = _origin()
        actor = InteractionActor(principal=origin.principal)
        wrapper = self._wrapper(object(), origin, actor)
        with self.assertRaisesRegex(
            TypeError,
            "interaction broker request",
        ):
            await wrapper.request(cast(InteractionBrokerRequest, object()))
        with self.assertRaisesRegex(
            TypeError,
            "terminalize an interaction scope",
        ):
            await wrapper.cancel_scope(
                cast(TerminalizeInteractionScopeCommand, object())
            )

        other_actor = InteractionActor(principal=_principal("user-other"))
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "actor does not match",
        ):
            wrapper._validate_actor(other_actor)

        mismatched_scope = InteractionExecutionScope(
            run_id=origin.run_id,
            turn_id=TurnId("turn-other"),
            branch_id=origin.branch_id,
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "scope does not match",
        ):
            wrapper._validate_scope(mismatched_scope, origin)

    async def test_cancellation_rejects_invalid_store_results(self) -> None:
        origin = _origin()
        actor = InteractionActor(principal=origin.principal)
        command = _scope_command(origin, actor)
        rejected = InteractionBrokerResult(
            store_result=ScopeCancellationRejected(
                command=command,
                error=_transition_error(),
            )
        )
        registration_command = _registration_command(origin, actor)
        unrelated = InteractionBrokerResult(
            store_result=InteractionBranchRegistrationRejected(
                command=registration_command,
                error=_transition_error(),
            )
        )
        other_command = _scope_command(
            origin,
            actor,
            turn_id=TurnId("turn-other"),
        )
        replayed = object.__new__(ScopeCancellationReplayed)
        object.__setattr__(replayed, "command", other_command)
        mismatched = InteractionBrokerResult(store_result=replayed)

        cases = (
            (object(), "invalid state"),
            (rejected, "was rejected"),
            (unrelated, "unrelated state"),
            (mismatched, "mismatched state"),
        )
        for result, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ExecutionCorrelationError, message),
            ):
                wrapper = self._wrapper(
                    _CancellationBroker(result),
                    origin,
                    actor,
                )
                await wrapper.cancel_scope(command)

    async def test_child_registration_rejects_every_invalid_result(
        self,
    ) -> None:
        origin = _origin(parent_branch_id=BranchId("parent-1"))
        actor = InteractionActor(principal=origin.principal)
        scope_rejection = InteractionBrokerResult(
            store_result=ScopeCancellationRejected(
                command=_scope_command(origin, actor),
                error=_transition_error(),
            )
        )

        def rejected(
            command: RegisterInteractionBranchCommand,
        ) -> object:
            return InteractionBrokerResult(
                store_result=InteractionBranchRegistrationRejected(
                    command=command,
                    error=_transition_error(),
                )
            )

        def mismatched(
            command: RegisterInteractionBranchCommand,
        ) -> object:
            registration = replace(
                command.registration,
                branch_id=BranchId("branch-other"),
            )
            other_command = RegisterInteractionBranchCommand(
                actor=command.actor,
                registration=registration,
            )
            record = InteractionBranchRecord(
                registration=registration,
                store_revision=InteractionStoreRevision(1),
            )
            return InteractionBrokerResult(
                store_result=InteractionBranchRegistrationReplayed(
                    command=other_command,
                    record=record,
                )
            )

        cases = (
            (object(), "cannot register"),
            (
                _RegistrationBroker(lambda _command: object()),
                "invalid state",
            ),
            (_RegistrationBroker(rejected), "was rejected"),
            (
                _RegistrationBroker(lambda _command: scope_rejection),
                "unrelated state",
            ),
            (_RegistrationBroker(mismatched), "mismatched state"),
        )
        for broker, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ExecutionCorrelationError, message),
            ):
                wrapper = self._wrapper(broker, origin, actor)
                await wrapper._ensure_branch_registered(origin)


class BrokerResultValidationTest(TestCase):
    """Validate broker admission results independently from constructors."""

    @staticmethod
    def _forged_result(
        create_result: object,
        delivery: object | None,
    ) -> InteractionRequestResult:
        result = object.__new__(InteractionRequestResult)
        object.__setattr__(result, "create_result", create_result)
        object.__setattr__(result, "delivery", delivery)
        return result

    @staticmethod
    def _forged_applied(
        command: CreateInteractionCommand,
        admitted_request: InputRequest,
        *,
        correlation: object = "correlation",
    ) -> CreateInteractionApplied:
        result = object.__new__(CreateInteractionApplied)
        object.__setattr__(result, "command", command)
        object.__setattr__(
            result,
            "record",
            SimpleNamespace(
                request=admitted_request,
                correlation=correlation,
            ),
        )
        return result

    def test_rejected_admission_rejects_delivery_then_returns_without_one(
        self,
    ) -> None:
        origin = _origin()
        actor = InteractionActor(principal=origin.principal)
        request = _broker_request(origin, actor)
        command = _create_command(request)
        rejected = CreateInteractionRejected(
            command=command,
            error=_transition_error(),
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "unrelated state",
        ):
            execution_module._validate_broker_request_result(
                request,
                self._forged_result(object(), None),
            )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "returned a delivery",
        ):
            execution_module._validate_broker_request_result(
                request,
                self._forged_result(rejected, object()),
            )

        valid_rejection = InteractionRequestResult(
            create_result=rejected,
            delivery=None,
        )
        execution_module._validate_broker_request_result(
            request,
            valid_rejection,
        )
        self.assertIsNone(valid_rejection.delivery)

    def test_applied_admission_rejects_contract_and_delivery_substitution(
        self,
    ) -> None:
        origin = _origin()
        actor = InteractionActor(principal=origin.principal)
        request = _broker_request(origin, actor)
        command = _create_command(request)
        created = command.request
        changed = replace(created, reason="Substituted reason.")
        changed_applied = self._forged_applied(command, changed)
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed its created request",
        ):
            execution_module._validate_broker_request_result(
                request,
                self._forged_result(changed_applied, None),
            )

        applied = self._forged_applied(command, created)
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "omitted its delivery",
        ):
            execution_module._validate_broker_request_result(
                request,
                self._forged_result(applied, None),
            )

        delivery = SimpleNamespace(
            correlation="different-correlation",
            record=SimpleNamespace(request=created),
        )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed admission correlation",
        ):
            execution_module._validate_broker_request_result(
                request,
                self._forged_result(applied, delivery),
            )


class ExecutionHelperDefenseTest(TestCase):
    """Exercise terminal, memory, and replay helper guards."""

    def test_terminal_request_rejects_changed_and_nonterminal_contracts(
        self,
    ) -> None:
        pending = _pending_request(_origin())
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "changed its immutable contract",
        ):
            execution_module._validate_terminal_request(
                pending,
                replace(pending, reason="Substituted reason."),
            )
        with self.assertRaisesRegex(
            ExecutionCorrelationError,
            "is not terminal",
        ):
            execution_module._validate_terminal_request(pending, pending)

    def test_missing_active_input_and_memory_message_fail_closed(self) -> None:
        origin = _origin()
        input_entry = _input_entry(origin)
        with self.assertRaisesRegex(
            ExecutionStateError,
            "no originating task-input",
        ):
            execution_module._active_task_input((input_entry,))

        execution = AgentExecution(
            origin=origin,
            id_factory=_IdFactory(),
            initial_messages=(),
        )
        with self.assertRaisesRegex(
            ExecutionStateError,
            "no ledger message",
        ):
            execution_module._memory_entry_for_message(
                execution.snapshot,
                0,
            )

    def test_result_replay_skips_an_unrelated_interaction(self) -> None:
        origin = _origin()
        first_request = _resolved_request(
            _pending_request(
                origin,
                request_id="request-first",
                continuation_id="continuation-first",
            )
        )
        first_result = _result(first_request)
        entry = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.INTERACTION_RESULT,
            origin=origin,
            messages=_correlated_messages(first_result),
            request=first_request,
            result=first_result,
            task_input_call=_task_input_call(),
        )
        other_request = _resolved_request(
            _pending_request(
                origin,
                request_id="request-other",
                continuation_id="continuation-other",
            )
        )
        other_result = _result(other_request)
        self.assertFalse(
            execution_module._find_result_replay(
                (entry,),
                other_request,
                other_result,
                _correlated_messages(other_result),
            )
        )


class ExecutionMutationDefenseTest(IsolatedAsyncioTestCase):
    """Exercise positive identity renewal and cursor-race detection."""

    async def test_new_stream_turn_mints_a_stream_session(self) -> None:
        execution = AgentExecution(
            origin=_origin(),
            id_factory=_IdFactory(),
            initial_messages=(),
        )
        previous = execution.origin
        advanced = await execution.advance_model_turn(new_stream_session=True)
        self.assertEqual(advanced.turn_id, TurnId("turn-new"))
        self.assertEqual(
            advanced.model_call_id,
            ModelCallId("model-call-new"),
        )
        self.assertEqual(
            advanced.stream_session_id,
            StreamSessionId("stream-new"),
        )
        self.assertNotEqual(
            advanced.stream_session_id,
            previous.stream_session_id,
        )

    async def test_transcript_cursor_conflict_is_detected(self) -> None:
        execution = AgentExecution(
            origin=_origin(),
            id_factory=_IdFactory(),
            initial_messages=(
                Message(role=MessageRole.USER, content="initial"),
            ),
        )
        with self.assertRaisesRegex(
            ExecutionRevisionError,
            "memory synchronization cursor changed",
        ):
            await execution.sync_memory(
                _TranscriptCursorConflictSink(execution)
            )

    async def test_response_cursor_conflict_is_detected(self) -> None:
        execution = AgentExecution(
            origin=_origin(),
            id_factory=_IdFactory(),
            initial_messages=(),
        )
        await execution.record_response("not in the transcript")
        with self.assertRaisesRegex(
            ExecutionRevisionError,
            "response synchronization cursor changed",
        ):
            await execution.sync_memory(_ResponseCursorConflictSink(execution))
