"""Exercise invocation-scoped agent execution state."""

from asyncio import CancelledError, Event, create_task, gather, sleep
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, NoReturn, cast
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.agent.execution import (
    MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
    AgentExecution,
    AgentExecutionSnapshot,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionCorrelationError,
    ExecutionInputRequiredError,
    ExecutionLedgerEntry,
    ExecutionLedgerEntryKind,
    ExecutionMemoryComponent,
    ExecutionMemoryEntry,
    ExecutionRevisionError,
    ExecutionStateError,
    ExecutionTerminatedError,
    InteractionLoopLimitError,
    ModelPromptRecord,
    UuidExecutionIdFactory,
    create_agent_execution,
)
from avalan.entities import (
    Message,
    MessageRole,
    MessageToolCall,
    normalize_tool_arguments,
)
from avalan.interaction.broker import InteractionBroker
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
from avalan.interaction.handler import (
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
)
from avalan.interaction.policy import InteractionActor
from avalan.interaction.state import project_resolution_to_model
from avalan.interaction.validation import MAX_STATE_REVISION
from avalan.model.capability import (
    CorrelatedCapabilityResult,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.types import JsonValue

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


class _ExecutionMemorySink:
    """Adapt an explicit execution-memory entry callback for tests."""

    def __init__(
        self,
        append: Callable[[ExecutionMemoryEntry], Awaitable[None]],
    ) -> None:
        self._append = append

    async def append_execution_memory_entry(
        self,
        entry: ExecutionMemoryEntry,
    ) -> None:
        await self._append(entry)


class _IdFactory:
    """Mint deterministic identities for execution tests."""

    def __init__(self) -> None:
        self.run = 0
        self.turn = 0
        self.task = 0
        self.model_call = 0
        self.branch = 0
        self.stream = 0

    async def new_run_id(self) -> RunId:
        self.run += 1
        return RunId(f"run-{self.run}")

    async def new_turn_id(self) -> TurnId:
        self.turn += 1
        return TurnId(f"turn-{self.turn}")

    async def new_task_id(self) -> TaskId:
        self.task += 1
        return TaskId(f"task-{self.task}")

    async def new_model_call_id(self) -> ModelCallId:
        self.model_call += 1
        return ModelCallId(f"model-call-{self.model_call}")

    async def new_branch_id(self) -> BranchId:
        self.branch += 1
        return BranchId(f"branch-{self.branch}")

    async def new_stream_session_id(self) -> StreamSessionId:
        self.stream += 1
        return StreamSessionId(f"stream-{self.stream}")


class _BlockingIdFactory(_IdFactory):
    """Pause turn identity generation at a deterministic boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.block_turn = False
        self.started = Event()
        self.release = Event()

    async def new_turn_id(self) -> TurnId:
        if self.block_turn:
            self.started.set()
            await self.release.wait()
        return await super().new_turn_id()


class _PersistentTerminalFailure(BaseException):
    """Represent a public terminal transition that always fails."""


class _Broker:
    """Provide only the async methods validated by attached state."""

    async def request(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def cancel_scope(self, *_args: object, **_kwargs: object) -> None:
        return None


async def _handler(_context: InputHandlerContext) -> InputHandlerOutcome:
    return InputHandlerDetached()


def _definition(*, operation_index: int = 2) -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://support",
        agent_definition_revision="agent-r1",
        operation_id="operation-support",
        operation_index=operation_index,
        model_config_reference="model-config-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )


def _principal(*, user_id: str = "user-1") -> PrincipalScope:
    return PrincipalScope(user_id=UserId(user_id))


def _message(content: str, role: MessageRole = MessageRole.USER) -> Message:
    return Message(role=role, content=content)


def _task_input_call() -> TaskInputCapabilityCall:
    return TaskInputCapabilityCall(
        call_id="input-call",
        provider_name="request_user_input",
        arguments={
            "mode": "required",
            "reason": "Choose how the execution should continue.",
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
        reason="Choose how the execution should continue.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("continue"),
                prompt="Continue?",
                required=True,
            ),
        ),
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
    return (
        _task_input_message(),
        correlated.tool_result_message(call),
    )


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        parent_branch_id=BranchId("parent-1"),
        model_call_id=ModelCallId("model-call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=_definition(),
        principal=_principal(),
    )


def _request(
    origin: ExecutionOrigin,
    *,
    request_id: str = "request-1",
    continuation_id: str = "continuation-1",
    pending: bool = True,
) -> InputRequest:
    return (
        create_input_request(
            request_id=InputRequestId(request_id),
            continuation_id=ContinuationId(continuation_id),
            origin=origin,
            mode=RequirementMode.REQUIRED,
            reason="Choose how the execution should continue.",
            questions=(
                ConfirmationQuestion(
                    question_id=QuestionId("continue"),
                    prompt="Continue?",
                    required=True,
                ),
            ),
            created_at=_NOW,
        )
        if not pending
        else InputRequest(
            request_id=InputRequestId(request_id),
            continuation_id=ContinuationId(continuation_id),
            origin=origin,
            mode=RequirementMode.REQUIRED,
            reason="Choose how the execution should continue.",
            questions=(
                ConfirmationQuestion(
                    question_id=QuestionId("continue"),
                    prompt="Continue?",
                    required=True,
                ),
            ),
            created_at=_NOW,
            state=RequestState.PENDING,
            state_revision=StateRevision(1),
        )
    )


def _resolved_request(request: InputRequest) -> InputRequest:
    resolution = UnavailableResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
    )
    return replace(
        request,
        state=RequestState.UNAVAILABLE,
        state_revision=StateRevision(request.state_revision + 1),
        resolution=resolution,
    )


def _terminated_request(request: InputRequest) -> InputRequest:
    resolution = ExpiredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
    )
    return replace(
        request,
        state=RequestState.EXPIRED,
        state_revision=StateRevision(request.state_revision + 1),
        resolution=resolution,
    )


def _result(request: InputRequest) -> InputUnavailableResult:
    return InputUnavailableResult(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
    )


async def _execution(
    *,
    factory: _IdFactory | None = None,
    runtime: AttachedInteractionRuntime | None = None,
) -> AgentExecution:
    selected = factory or _IdFactory()
    return await create_agent_execution(
        definition=_definition(),
        agent_id=AgentId("agent-1"),
        principal=_principal(),
        initial_messages=(_message("initial"),),
        id_factory=selected,
        interaction_runtime=runtime,
    )


class PromptAndLedgerValidationTest(TestCase):
    """Exercise immutable prompt and ledger value objects."""

    def test_memory_entry_rejects_every_untyped_identity_field(self) -> None:
        base: dict[str, Any] = {
            "origin": _origin(),
            "ledger_sequence": 0,
            "component": ExecutionMemoryComponent.MESSAGE,
            "component_index": 0,
            "message": _message("memory"),
        }
        invalid = (
            ("origin", object()),
            ("ledger_sequence", -1),
            ("component_index", True),
            ("component", "message"),
            ("message", object()),
        )
        for name, value in invalid:
            with self.subTest(name=name), self.assertRaises(TypeError):
                cast(Any, ExecutionMemoryEntry)(**{**base, name: value})

    def test_prompt_freezes_list_input_and_validates_text_fields(self) -> None:
        messages = [_message("one"), _message("two")]
        prompt = ModelPromptRecord(
            input=messages,
            instructions="instructions",
            system_prompt=None,
            developer_prompt="developer",
        )
        messages.append(_message("three"))
        self.assertEqual(prompt.input, tuple(messages[:2]))
        with self.assertRaises(FrozenInstanceError):
            prompt.instructions = "changed"  # type: ignore[misc]
        for name in ("instructions", "system_prompt", "developer_prompt"):
            values: dict[str, Any] = {
                "input": "input",
                "instructions": None,
                "system_prompt": None,
                "developer_prompt": None,
            }
            values[name] = 1
            with self.subTest(name=name), self.assertRaises(TypeError):
                ModelPromptRecord(**values)

    def test_prompt_accepts_every_input_variant_and_rejects_mixed_input(
        self,
    ) -> None:
        message = _message("message")
        for value, expected in (
            ("text", "text"),
            (message, message),
            (["one", "two"], ("one", "two")),
            ([], ()),
        ):
            with self.subTest(value=value):
                prompt = ModelPromptRecord(
                    input=cast(Any, value),
                    instructions=None,
                    system_prompt=None,
                    developer_prompt=None,
                )
                self.assertEqual(prompt.input, expected)
        for invalid_value in cast(
            tuple[object, ...], (("tuple",), ["one", message])
        ):
            with (
                self.subTest(value=invalid_value),
                self.assertRaises(TypeError),
            ):
                ModelPromptRecord(
                    input=cast(Any, invalid_value),
                    instructions=None,
                    system_prompt=None,
                    developer_prompt=None,
                )

    def test_ledger_records_each_typed_payload(self) -> None:
        origin = _origin()
        prompt = ModelPromptRecord(
            input="prompt",
            instructions=None,
            system_prompt=None,
            developer_prompt=None,
        )
        pending_request = _request(origin)
        request = _resolved_request(pending_request)
        result = _result(request)
        required = InputRequiredResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            detached_resumption_available=True,
        )
        terminated_request = _terminated_request(pending_request)
        termination = project_resolution_to_model(
            terminated_request,
            containing_run_exists=True,
        )
        assert isinstance(termination, TerminateInputContinuation)
        entries = (
            ExecutionLedgerEntry(
                sequence=0,
                kind=ExecutionLedgerEntryKind.INPUT,
                origin=origin,
                messages=(_message("input"),),
            ),
            ExecutionLedgerEntry(
                sequence=1,
                kind=ExecutionLedgerEntryKind.TRANSCRIPT,
                origin=origin,
                messages=(_message("ordinary"),),
            ),
            ExecutionLedgerEntry(
                sequence=2,
                kind=ExecutionLedgerEntryKind.MODEL_PROMPT,
                origin=origin,
                prompt=prompt,
            ),
            ExecutionLedgerEntry(
                sequence=3,
                kind=ExecutionLedgerEntryKind.MODEL_RESPONSE,
                origin=origin,
                response="response",
                messages=(_message("response", MessageRole.ASSISTANT),),
                response_in_transcript=True,
            ),
            ExecutionLedgerEntry(
                sequence=4,
                kind=ExecutionLedgerEntryKind.INTERACTION_RESERVED,
                origin=origin,
                semantic_fingerprint="fingerprint",
                task_input_call=_task_input_call(),
                interaction_assistant_message=_task_input_message(),
            ),
            ExecutionLedgerEntry(
                sequence=5,
                kind=ExecutionLedgerEntryKind.INTERACTION_PENDING,
                origin=origin,
                request=pending_request,
            ),
            ExecutionLedgerEntry(
                sequence=6,
                kind=ExecutionLedgerEntryKind.INTERACTION_RESULT,
                origin=origin,
                request=request,
                result=result,
                task_input_call=_task_input_call(),
                messages=_correlated_messages(result),
            ),
            ExecutionLedgerEntry(
                sequence=7,
                kind=ExecutionLedgerEntryKind.INPUT_REQUIRED,
                origin=origin,
                request=request,
                input_required=required,
            ),
            ExecutionLedgerEntry(
                sequence=8,
                kind=ExecutionLedgerEntryKind.INTERACTION_TERMINATED,
                origin=origin,
                request=terminated_request,
                termination_outcome=termination,
            ),
            ExecutionLedgerEntry(
                sequence=9,
                kind=ExecutionLedgerEntryKind.COMPLETED,
                origin=origin,
            ),
        )
        self.assertEqual(
            tuple(entry.sequence for entry in entries), tuple(range(10))
        )

    def test_ledger_rejects_invalid_fields_and_payload_shapes(self) -> None:
        origin = _origin()
        base: dict[str, Any] = {
            "sequence": 0,
            "kind": ExecutionLedgerEntryKind.COMPLETED,
            "origin": origin,
        }
        invalid_fields = (
            ("sequence", True),
            ("kind", "completed"),
            ("origin", object()),
            ("messages", (object(),)),
            ("prompt", object()),
            ("response", object()),
            ("response_in_transcript", 1),
            ("task_input_call", object()),
            ("interaction_assistant_message", object()),
            ("request", object()),
            ("result", object()),
            ("input_required", object()),
            ("termination_outcome", object()),
        )
        for name, value in invalid_fields:
            with self.subTest(name=name), self.assertRaises(TypeError):
                cast(Any, ExecutionLedgerEntry)(**{**base, name: value})
        with self.assertRaises(ValueError):
            ExecutionLedgerEntry(
                **base,
                semantic_fingerprint="",
            )
        with self.assertRaises(ExecutionStateError):
            ExecutionLedgerEntry(
                sequence=0,
                kind=ExecutionLedgerEntryKind.MODEL_PROMPT,
                origin=origin,
            )
        with self.assertRaises(ExecutionStateError):
            ExecutionLedgerEntry(
                sequence=0,
                kind=ExecutionLedgerEntryKind.MODEL_RESPONSE,
                origin=origin,
                response="response",
                response_in_transcript=True,
            )
        with self.assertRaises(ExecutionCorrelationError):
            ExecutionLedgerEntry(
                sequence=0,
                kind=ExecutionLedgerEntryKind.MODEL_RESPONSE,
                origin=origin,
                response="response",
                messages=(_message("substituted", MessageRole.ASSISTANT),),
                response_in_transcript=True,
            )


class ExecutionCreationTest(IsolatedAsyncioTestCase):
    """Exercise identity creation and immutable initial state."""

    async def test_creation_mints_genuine_ids_and_stable_definition(
        self,
    ) -> None:
        execution = await _execution()
        self.assertEqual(execution.origin.run_id, "run-1")
        self.assertEqual(execution.origin.turn_id, "turn-1")
        self.assertEqual(execution.origin.model_call_id, "model-call-1")
        self.assertEqual(execution.origin.branch_id, "branch-1")
        self.assertEqual(execution.origin.stream_session_id, "stream-1")
        self.assertEqual(execution.origin.agent_id, "agent-1")
        self.assertEqual(execution.origin.task_id, "task-1")
        self.assertEqual(execution.initial_origin, execution.origin)
        self.assertEqual(execution.definition, _definition())
        self.assertEqual(execution.operation_id, "operation-support")
        self.assertEqual(execution.operation_index, 2)
        self.assertEqual(execution.status, AgentExecutionStatus.RUNNING)
        self.assertEqual(execution.revision, 0)
        self.assertEqual(execution.snapshot.memory_sync_cursor, 0)
        self.assertEqual(execution.messages, (_message("initial"),))
        self.assertEqual(execution.ledger, execution.snapshot.ledger)
        self.assertIsNone(execution.last_prompt)
        self.assertIsNone(execution.last_response)
        self.assertIsNone(execution.pending_request)
        self.assertIsNone(execution.interaction_runtime)
        with self.assertRaises(FrozenInstanceError):
            execution.snapshot.revision = 1  # type: ignore[misc]

    async def test_creation_tracks_an_already_synchronized_prefix(
        self,
    ) -> None:
        messages = (_message("history"), _message("current"))
        execution = await create_agent_execution(
            definition=_definition(),
            agent_id=AgentId("agent-1"),
            principal=_principal(),
            initial_messages=messages,
            synced_message_prefix=1,
            id_factory=_IdFactory(),
        )
        synchronized: list[Message] = []

        async def append(entry: ExecutionMemoryEntry) -> None:
            self.assertIs(entry.component, ExecutionMemoryComponent.MESSAGE)
            synchronized.append(entry.message)

        await execution.sync_memory(_ExecutionMemorySink(append))

        self.assertEqual(synchronized, [_message("current")])
        self.assertEqual(execution.snapshot.memory_sync_cursor, 2)

    async def test_creation_rejects_invalid_synchronized_prefixes(
        self,
    ) -> None:
        common: dict[str, Any] = {
            "definition": _definition(),
            "agent_id": AgentId("agent-1"),
            "principal": _principal(),
            "initial_messages": (_message("initial"),),
            "id_factory": _IdFactory(),
        }
        for value, error in (
            (True, TypeError),
            ("1", TypeError),
            (-1, ValueError),
            (2, ValueError),
        ):
            with self.subTest(value=value), self.assertRaises(error):
                await cast(Any, create_agent_execution)(
                    **common,
                    synced_message_prefix=value,
                )

    async def test_attached_runtime_preserves_real_task_and_branch_ids(
        self,
    ) -> None:
        factory = _IdFactory()
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, _Broker()),
            actor=InteractionActor(principal=_principal()),
            handler=cast(InputHandler, _handler),
            id_factory=factory,
            task_id=TaskId("task-real"),
            branch_id=BranchId("branch-real"),
            parent_branch_id=BranchId("branch-parent"),
        )
        execution = await _execution(factory=factory, runtime=runtime)
        self.assertIs(execution.interaction_runtime, runtime)
        self.assertEqual(execution.origin.task_id, "task-real")
        self.assertEqual(execution.origin.branch_id, "branch-real")
        self.assertEqual(execution.origin.parent_branch_id, "branch-parent")
        self.assertEqual(factory.branch, 0)
        self.assertEqual(factory.task, 0)

    async def test_creation_validates_runtime_trust_and_types(self) -> None:
        factory = _IdFactory()
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, _Broker()),
            actor=InteractionActor(principal=_principal()),
            handler=cast(InputHandler, _handler),
            id_factory=factory,
        )
        common: dict[str, Any] = {
            "definition": _definition(),
            "agent_id": AgentId("agent-1"),
            "principal": _principal(),
            "initial_messages": (_message("initial"),),
        }
        for field_name, value in (
            ("definition", object()),
            ("principal", object()),
            ("initial_messages", [object()]),
            ("interaction_runtime", object()),
        ):
            with self.subTest(field=field_name), self.assertRaises(TypeError):
                await cast(Any, create_agent_execution)(
                    **{
                        **common,
                        field_name: value,
                    }
                )
        with self.assertRaises(ExecutionCorrelationError):
            await cast(Any, create_agent_execution)(
                **{
                    **common,
                    "principal": _principal(user_id="other"),
                    "interaction_runtime": runtime,
                }
            )
        with self.assertRaises(ExecutionCorrelationError):
            await cast(Any, create_agent_execution)(
                **{
                    **common,
                    "id_factory": _IdFactory(),
                    "interaction_runtime": runtime,
                }
            )

    async def test_uuid_factory_mints_distinct_ids(self) -> None:
        factory = UuidExecutionIdFactory()
        values = {
            await factory.new_run_id(),
            await factory.new_turn_id(),
            await factory.new_task_id(),
            await factory.new_model_call_id(),
            await factory.new_branch_id(),
            await factory.new_stream_session_id(),
        }
        self.assertEqual(len(values), 6)

    async def test_transcript_and_prompt_snapshot_nested_mutable_payloads(
        self,
    ) -> None:
        arguments = {"nested": ["before"]}
        message = Message(
            role=MessageRole.USER,
            content="initial",
            arguments=arguments,
            tool_calls=[
                MessageToolCall(
                    name="request_user_input",
                    arguments=cast(
                        Any,
                        {
                            "mode": "required",
                            "questions": (
                                MappingProxyType(
                                    {
                                        "question_id": "continue",
                                        "kind": "confirmation",
                                        "prompt": "Continue?",
                                        "required": True,
                                        "choices": (),
                                        "allow_other": False,
                                    }
                                ),
                            ),
                        },
                    ),
                )
            ],
        )
        execution = await create_agent_execution(
            definition=_definition(),
            agent_id=AgentId("agent-1"),
            principal=_principal(),
            initial_messages=(message,),
            id_factory=_IdFactory(),
        )
        arguments["nested"].append("caller mutation")
        self.assertEqual(
            execution.messages[0].arguments,
            {"nested": ["before"]},
        )
        snapshotted_tool_calls = execution.messages[0].tool_calls
        assert snapshotted_tool_calls is not None
        self.assertEqual(
            snapshotted_tool_calls[0].arguments,
            {
                "mode": "required",
                "questions": [
                    {
                        "question_id": "continue",
                        "kind": "confirmation",
                        "prompt": "Continue?",
                        "required": True,
                        "choices": [],
                        "allow_other": False,
                    },
                ],
            },
        )

        exposed = execution.ledger[0].messages[0]
        assert exposed.arguments is not None
        exposed.arguments["nested"] = ["public mutation"]
        self.assertEqual(
            execution.ledger[0].messages[0].arguments,
            {"nested": ["before"]},
        )

        prompt_arguments = {"prompt": ["before"]}
        prompt = ModelPromptRecord(
            input=Message(
                role=MessageRole.USER,
                content="prompt",
                arguments=prompt_arguments,
            ),
            instructions=None,
            system_prompt=None,
            developer_prompt=None,
        )
        await execution.record_prompt(prompt)
        prompt_arguments["prompt"].append("caller mutation")
        recorded = execution.last_prompt
        assert recorded is not None and isinstance(recorded.input, Message)
        self.assertEqual(recorded.input.arguments, {"prompt": ["before"]})
        assert recorded.input.arguments is not None
        recorded.input.arguments["prompt"] = ["public mutation"]
        refreshed = execution.last_prompt
        assert refreshed is not None and isinstance(refreshed.input, Message)
        self.assertEqual(refreshed.input.arguments, {"prompt": ["before"]})

    async def test_direct_constructor_rejects_invalid_dependencies(
        self,
    ) -> None:
        factory = _IdFactory()
        with self.assertRaises(TypeError):
            AgentExecution(
                origin=cast(Any, object()),
                id_factory=factory,
                initial_messages=(),
            )
        with self.assertRaises(TypeError):
            AgentExecution(
                origin=_origin(),
                id_factory=factory,
                initial_messages=(),
                interaction_runtime=cast(Any, object()),
            )

    async def test_direct_constructor_enforces_attached_runtime_origin(
        self,
    ) -> None:
        factory = _IdFactory()
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, _Broker()),
            actor=InteractionActor(principal=_principal()),
            handler=cast(InputHandler, _handler),
            id_factory=factory,
            task_id=TaskId("task-1"),
            branch_id=BranchId("branch-1"),
            parent_branch_id=BranchId("parent-1"),
        )
        execution = AgentExecution(
            origin=_origin(),
            id_factory=factory,
            initial_messages=(),
            interaction_runtime=runtime,
        )
        self.assertIs(execution.interaction_runtime, runtime)

        mismatches = (
            replace(
                runtime,
                actor=InteractionActor(principal=_principal(user_id="other")),
            ),
            replace(runtime, id_factory=_IdFactory()),
            replace(runtime, task_id=TaskId("task-other")),
            replace(runtime, branch_id=BranchId("branch-other")),
            replace(
                runtime,
                parent_branch_id=BranchId("parent-other"),
            ),
        )
        for mismatch in mismatches:
            with (
                self.subTest(runtime=mismatch),
                self.assertRaises(ExecutionCorrelationError),
            ):
                AgentExecution(
                    origin=_origin(),
                    id_factory=factory,
                    initial_messages=(),
                    interaction_runtime=mismatch,
                )


class ExecutionMutationTest(IsolatedAsyncioTestCase):
    """Exercise prompt, response, sync, and terminal mutations."""

    async def asyncSetUp(self) -> None:
        self.execution = await _execution()

    async def test_prompt_response_and_sync_are_ledger_authoritative(
        self,
    ) -> None:
        prompt = ModelPromptRecord(
            input=[_message("initial")],
            instructions="instruction",
            system_prompt="system",
            developer_prompt="developer",
        )
        first_revision = await self.execution.record_prompt(
            prompt,
            expected_revision=0,
        )
        assistant = _message("answer", MessageRole.ASSISTANT)
        tool = _message("tool", MessageRole.TOOL)
        transcript_revision = await self.execution.record_messages(
            (tool,),
            expected_revision=first_revision,
        )
        second_revision = await self.execution.record_response(
            "answer",
            messages=(assistant,),
            expected_revision=transcript_revision,
        )
        self.assertEqual(second_revision, 3)
        self.assertEqual(self.execution.last_prompt, prompt)
        self.assertEqual(self.execution.last_response, "answer")
        self.assertEqual(
            self.execution.messages,
            (_message("initial"), tool, assistant),
        )
        self.assertEqual(
            tuple(entry.kind for entry in self.execution.ledger),
            (
                ExecutionLedgerEntryKind.INPUT,
                ExecutionLedgerEntryKind.MODEL_PROMPT,
                ExecutionLedgerEntryKind.TRANSCRIPT,
                ExecutionLedgerEntryKind.MODEL_RESPONSE,
            ),
        )
        self.assertTrue(self.execution.ledger[-1].response_in_transcript)
        synchronized_messages: list[Message] = []
        synchronized_responses: list[object] = []

        async def append(entry: ExecutionMemoryEntry) -> None:
            if entry.component is ExecutionMemoryComponent.MESSAGE:
                synchronized_messages.append(entry.message)
                return
            synchronized_responses.append(entry.message.content)

        sink = _ExecutionMemorySink(append)
        await self.execution.sync_memory(sink)
        self.assertEqual(
            synchronized_messages,
            [_message("initial"), tool, assistant],
        )
        self.assertEqual(synchronized_responses, [])
        await self.execution.sync_memory(sink)
        self.assertEqual(len(synchronized_messages), 3)
        self.assertEqual(
            self.execution.snapshot.response_sync_cursor,
            len(self.execution.ledger),
        )

    async def test_repeated_sync_never_duplicates_assistant_output(
        self,
    ) -> None:
        synchronized: list[object] = []

        async def append(entry: ExecutionMemoryEntry) -> None:
            if entry.component is ExecutionMemoryComponent.RESPONSE:
                synchronized.append(entry.message.content)

        await self.execution.record_response("first")
        sink = _ExecutionMemorySink(append)
        await self.execution.sync_memory(sink)
        await self.execution.sync_memory(sink)
        self.assertEqual(synchronized, ["first"])
        await self.execution.record_response("second")
        await self.execution.sync_memory(sink)
        await self.execution.sync_memory(sink)
        self.assertEqual(synchronized, ["first", "second"])

    async def test_failed_memory_sink_retries_only_the_unacknowledged_tail(
        self,
    ) -> None:
        second = _message("second")
        await self.execution.record_messages((second,))
        synchronized: list[str] = []
        fail_second = True

        async def append(entry: ExecutionMemoryEntry) -> None:
            nonlocal fail_second
            self.assertIs(entry.component, ExecutionMemoryComponent.MESSAGE)
            message = entry.message
            content = cast(str, message.content)
            if content == "second" and fail_second:
                fail_second = False
                raise RuntimeError("memory unavailable")
            synchronized.append(content)

        sink = _ExecutionMemorySink(append)

        with self.assertRaisesRegex(RuntimeError, "memory unavailable"):
            await self.execution.sync_memory(sink)
        self.assertEqual(synchronized, ["initial"])

        await self.execution.sync_memory(sink)
        await self.execution.sync_memory(sink)

        self.assertEqual(synchronized, ["initial", "second"])

    async def test_mutations_reject_bad_types_and_stale_revisions(
        self,
    ) -> None:
        prompt = ModelPromptRecord(
            input="prompt",
            instructions=None,
            system_prompt=None,
            developer_prompt=None,
        )
        for action in (
            self.execution.record_prompt(cast(Any, object())),
            self.execution.record_messages(cast(Any, [])),
            self.execution.record_response(cast(Any, object())),
            self.execution.record_response("ok", messages=cast(Any, [])),
            self.execution.complete_with_response(cast(Any, object())),
            self.execution.complete_with_response(
                "ok",
                messages=cast(Any, []),
            ),
        ):
            with self.assertRaises(TypeError):
                await action
        with self.assertRaises(ValueError):
            await self.execution.record_messages(())
        await self.execution.record_prompt(prompt)
        with self.assertRaises(ExecutionRevisionError):
            await self.execution.record_response(
                "response",
                expected_revision=0,
            )
        for expected in (-1, True):
            with self.subTest(expected=expected), self.assertRaises(TypeError):
                await self.execution.record_response(
                    "response",
                    expected_revision=cast(Any, expected),
                )

    async def test_response_and_completion_commit_atomically(self) -> None:
        assistant = _message("answer", MessageRole.ASSISTANT)

        self.assertTrue(
            await self.execution.complete_with_response(
                "answer",
                messages=(assistant,),
                expected_revision=0,
            )
        )

        self.assertIs(
            self.execution.status,
            AgentExecutionStatus.COMPLETED,
        )
        self.assertEqual(self.execution.revision, 2)
        self.assertEqual(self.execution.last_response, "answer")
        self.assertEqual(
            self.execution.messages,
            (_message("initial"), assistant),
        )
        self.assertEqual(
            tuple(entry.kind for entry in self.execution.ledger[-2:]),
            (
                ExecutionLedgerEntryKind.MODEL_RESPONSE,
                ExecutionLedgerEntryKind.COMPLETED,
            ),
        )
        terminal = self.execution.snapshot
        self.assertFalse(
            await self.execution.complete_with_response(
                "different",
                messages=(_message("different", MessageRole.ASSISTANT),),
            )
        )
        self.assertEqual(self.execution.snapshot, terminal)

    async def test_response_completion_loser_records_nothing(self) -> None:
        self.assertTrue(await self.execution.cancel())
        terminal = self.execution.snapshot

        self.assertFalse(
            await self.execution.complete_with_response(
                "answer",
                messages=(_message("answer", MessageRole.ASSISTANT),),
            )
        )

        self.assertEqual(self.execution.snapshot, terminal)
        self.assertIsNone(self.execution.last_response)
        self.assertEqual(self.execution.messages, (_message("initial"),))

    async def test_response_completion_rolls_back_partial_failure(
        self,
    ) -> None:
        before = self.execution.snapshot
        failure = RuntimeError("completion commit failed")
        original_commit = self.execution._commit
        commit_count = 0

        def commit_then_fail(*args: Any, **kwargs: Any) -> Any:
            nonlocal commit_count
            commit_count += 1
            result = original_commit(*args, **kwargs)
            if commit_count == 2:
                raise failure
            return result

        self.execution._commit = commit_then_fail  # type: ignore[method-assign]
        try:
            with self.assertRaises(RuntimeError) as raised:
                await self.execution.complete_with_response(
                    "answer",
                    messages=(_message("answer", MessageRole.ASSISTANT),),
                )
        finally:
            self.execution._commit = original_commit  # type: ignore[method-assign]

        self.assertIs(raised.exception, failure)
        self.assertEqual(self.execution.snapshot, before)
        self.assertIsNone(self.execution.last_response)

    async def test_terminal_transitions_and_cleanup_are_exactly_once(
        self,
    ) -> None:
        with self.assertRaises(ExecutionStateError):
            await self.execution.claim_cleanup()
        self.assertTrue(await self.execution.complete())
        self.assertFalse(await self.execution.complete())
        self.assertFalse(await self.execution.cancel())
        self.assertFalse(await self.execution.fail())
        self.assertEqual(self.execution.status, AgentExecutionStatus.COMPLETED)
        self.assertTrue(await self.execution.claim_cleanup())
        self.assertFalse(await self.execution.claim_cleanup())
        self.assertEqual(
            tuple(entry.kind for entry in self.execution.ledger[-2:]),
            (
                ExecutionLedgerEntryKind.COMPLETED,
                ExecutionLedgerEntryKind.CLEANUP_CLAIMED,
            ),
        )
        with self.assertRaises(ExecutionStateError):
            await self.execution.record_response("late")

    async def test_cancel_and_fail_accept_nonrunning_nonterminal_states(
        self,
    ) -> None:
        preparing = await _execution()
        await preparing.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        self.assertTrue(await preparing.cancel())
        self.assertEqual(preparing.status, AgentExecutionStatus.CANCELLED)

        waiting = await _execution()
        await waiting.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        await waiting.mark_interaction_pending(_request(waiting.origin))
        self.assertTrue(await waiting.fail())
        self.assertEqual(waiting.status, AgentExecutionStatus.ERRORED)

    async def test_revision_exhaustion_fails_without_mutation(self) -> None:
        object.__setattr__(  # noqa: SLF001
            self.execution._state,
            "revision",
            MAX_STATE_REVISION,
        )
        with self.assertRaises(ExecutionRevisionError):
            await self.execution.record_response("response")
        self.assertEqual(self.execution.revision, MAX_STATE_REVISION)

    async def test_reducer_rejects_logical_identity_drift(self) -> None:
        for drifted in (
            replace(
                self.execution.origin,
                run_id=RunId("run-drifted"),
            ),
            replace(
                self.execution.origin,
                parent_branch_id=BranchId("parent-drifted"),
            ),
        ):
            with self.assertRaises(ExecutionCorrelationError):
                self.execution._commit(  # noqa: SLF001
                    self.execution.snapshot,
                    kind=ExecutionLedgerEntryKind.MODEL_TURN,
                    origin=drifted,
                )
        self.assertEqual(self.execution.revision, 0)


class InteractionStateTest(IsolatedAsyncioTestCase):
    """Exercise reservation, suspension, and continuation transitions."""

    async def asyncSetUp(self) -> None:
        self.factory = _IdFactory()
        self.execution = await _execution(factory=self.factory)

    async def _pending(self, fingerprint: str = "fingerprint") -> InputRequest:
        await self.execution.begin_interaction(
            fingerprint,
            _task_input_call(),
            _task_input_message(),
        )
        request = _request(self.execution.origin)
        await self.execution.mark_interaction_pending(request)
        return request

    async def _resolve_and_advance(
        self,
        fingerprint: str = "fingerprint",
    ) -> None:
        request = await self._pending(fingerprint)
        request = _resolved_request(request)
        result = _result(request)
        await self.execution.record_interaction_result(
            request,
            result,
            _correlated_messages(result),
        )
        await self.execution.advance_model_turn()

    async def test_provider_exit_fallback_is_terminal_and_idempotent(
        self,
    ) -> None:
        await self._pending()
        before = self.execution.snapshot
        failure = _PersistentTerminalFailure("persistent fail")
        fail_calls = 0

        async def persistent_fail(
            *,
            expected_revision: int | None = None,
        ) -> bool:
            nonlocal fail_calls
            del expected_revision
            fail_calls += 1
            raise failure

        self.execution.fail = persistent_fail  # type: ignore[method-assign]
        secondary_failures = await self.execution.settle_provider_exit(
            cancelled=False
        )

        self.assertEqual(secondary_failures, (failure,))
        self.assertEqual(fail_calls, 1)
        self.assertIs(
            self.execution.status,
            AgentExecutionStatus.ERRORED,
        )
        self.assertIsNone(self.execution.pending_request)
        self.assertEqual(self.execution.revision, before.revision + 1)
        self.assertEqual(
            self.execution.ledger[:-1],
            before.ledger,
        )
        self.assertIs(
            self.execution.ledger[-1].kind,
            ExecutionLedgerEntryKind.ERRORED,
        )

        terminal = self.execution.snapshot
        self.assertEqual(
            await self.execution.settle_provider_exit(cancelled=False),
            (),
        )
        self.assertEqual(self.execution.snapshot, terminal)
        self.assertEqual(fail_calls, 1)
        with self.assertRaises(TypeError):
            await self.execution.settle_provider_exit(cancelled=cast(Any, 1))

    async def test_provider_exit_retains_both_transition_failures_for_retry(
        self,
    ) -> None:
        public_failure = _PersistentTerminalFailure("public fail failed")
        fallback_failure = RuntimeError("terminal fallback failed")
        fail_calls = 0
        original_commit = self.execution._commit

        async def persistent_fail(
            *,
            expected_revision: int | None = None,
        ) -> bool:
            nonlocal fail_calls
            del expected_revision
            fail_calls += 1
            raise public_failure

        def failing_commit(*args: Any, **kwargs: Any) -> NoReturn:
            del args, kwargs
            raise fallback_failure

        self.execution.fail = persistent_fail  # type: ignore[method-assign]
        self.execution._commit = failing_commit  # type: ignore[method-assign]
        try:
            self.assertEqual(
                await self.execution.settle_provider_exit(cancelled=False),
                (public_failure, fallback_failure),
            )
            self.assertIs(
                self.execution.status,
                AgentExecutionStatus.RUNNING,
            )
            self.assertEqual(fail_calls, 1)

            self.execution._commit = original_commit  # type: ignore[method-assign]
            await sleep(0)
            self.assertEqual(
                await self.execution.settle_provider_exit(cancelled=False),
                (public_failure,),
            )
            self.assertEqual(fail_calls, 2)
            self.assertIs(
                self.execution.status,
                AgentExecutionStatus.ERRORED,
            )
        finally:
            self.execution._commit = original_commit  # type: ignore[method-assign]

    async def test_concurrent_provider_exit_settlement_is_coalesced(
        self,
    ) -> None:
        before = self.execution.snapshot
        failure = _PersistentTerminalFailure("persistent concurrent fail")
        fail_started = Event()
        fail_release = Event()
        fail_calls = 0

        async def persistent_fail(
            *,
            expected_revision: int | None = None,
        ) -> bool:
            nonlocal fail_calls
            del expected_revision
            fail_calls += 1
            fail_started.set()
            await fail_release.wait()
            raise failure

        self.execution.fail = persistent_fail  # type: ignore[method-assign]
        first = create_task(
            self.execution.settle_provider_exit(cancelled=False)
        )
        await fail_started.wait()
        second = create_task(
            self.execution.settle_provider_exit(cancelled=False)
        )
        await sleep(0)

        self.assertEqual(fail_calls, 1)
        fail_release.set()
        first_result, second_result = await gather(first, second)

        self.assertIs(first_result, second_result)
        self.assertEqual(first_result, (failure,))
        self.assertEqual(fail_calls, 1)
        self.assertIs(
            self.execution.status,
            AgentExecutionStatus.ERRORED,
        )
        self.assertEqual(self.execution.revision, before.revision + 1)
        self.assertEqual(self.execution.ledger[:-1], before.ledger)
        self.assertIs(
            self.execution.ledger[-1].kind,
            ExecutionLedgerEntryKind.ERRORED,
        )

    async def test_concurrent_provider_exit_uses_first_owner_classification(
        self,
    ) -> None:
        cases = (
            (
                False,
                AgentExecutionStatus.ERRORED,
                ExecutionLedgerEntryKind.ERRORED,
                "fail",
            ),
            (
                True,
                AgentExecutionStatus.CANCELLED,
                ExecutionLedgerEntryKind.CANCELLED,
                "cancel",
            ),
        )
        for (
            first_cancelled,
            expected_status,
            expected_kind,
            call_name,
        ) in cases:
            with self.subTest(first_cancelled=first_cancelled):
                execution = await _execution(factory=_IdFactory())
                before = execution.snapshot
                failure = _PersistentTerminalFailure(f"persistent {call_name}")
                transition_started = Event()
                transition_release = Event()
                calls: list[str] = []

                async def blocked_transition(
                    *,
                    expected_revision: int | None = None,
                ) -> bool:
                    del expected_revision
                    calls.append(call_name)
                    transition_started.set()
                    await transition_release.wait()
                    raise failure

                async def unexpected_transition(
                    *,
                    expected_revision: int | None = None,
                ) -> bool:
                    del expected_revision
                    calls.append("unexpected")
                    raise AssertionError(
                        "second owner ran a public transition"
                    )

                if first_cancelled:
                    execution.cancel = blocked_transition  # type: ignore[method-assign]
                    execution.fail = unexpected_transition  # type: ignore[method-assign]
                else:
                    execution.fail = blocked_transition  # type: ignore[method-assign]
                    execution.cancel = unexpected_transition  # type: ignore[method-assign]

                first = create_task(
                    execution.settle_provider_exit(cancelled=first_cancelled)
                )
                await transition_started.wait()
                second = create_task(
                    execution.settle_provider_exit(
                        cancelled=not first_cancelled
                    )
                )
                await sleep(0)
                self.assertEqual(calls, [call_name])

                transition_release.set()
                first_result, second_result = await gather(first, second)

                self.assertIs(first_result, second_result)
                self.assertEqual(first_result, (failure,))
                self.assertEqual(calls, [call_name])
                self.assertIs(execution.status, expected_status)
                self.assertEqual(execution.revision, before.revision + 1)
                self.assertEqual(execution.ledger[:-1], before.ledger)
                self.assertIs(execution.ledger[-1].kind, expected_kind)

    async def test_attached_result_resumes_same_run_on_new_turn(self) -> None:
        initial = self.execution.origin
        reserved_revision = await self.execution.begin_interaction(
            "fingerprint",
            _task_input_call(),
            _task_input_message(),
            expected_revision=0,
        )
        self.assertEqual(reserved_revision, 1)
        self.assertEqual(
            self.execution.status, AgentExecutionStatus.PREPARING_INPUT
        )
        self.assertEqual(self.execution.interaction_count, 1)
        request = _request(self.execution.origin)
        await self.execution.mark_interaction_pending(
            request,
            expected_revision=reserved_revision,
        )
        self.assertEqual(
            self.execution.status, AgentExecutionStatus.WAITING_FOR_INPUT
        )
        self.assertEqual(self.execution.pending_request, request)
        request = _resolved_request(request)
        result = _result(request)
        correlated = _correlated_messages(result)
        self.assertTrue(
            await self.execution.record_interaction_result(
                request,
                result,
                correlated,
            )
        )
        self.assertEqual(self.execution.status, AgentExecutionStatus.RESUMING)
        self.assertIsNone(self.execution.pending_request)
        self.assertFalse(
            await self.execution.record_interaction_result(
                request,
                result,
                correlated,
            )
        )
        with self.assertRaises(ExecutionStateError):
            await self.execution.record_response("too early")
        continued = await self.execution.advance_model_turn()
        self.assertEqual(self.execution.status, AgentExecutionStatus.RUNNING)
        self.assertEqual(continued.run_id, initial.run_id)
        self.assertEqual(continued.agent_id, initial.agent_id)
        self.assertEqual(continued.branch_id, initial.branch_id)
        self.assertEqual(continued.task_id, initial.task_id)
        self.assertEqual(continued.definition, initial.definition)
        self.assertEqual(continued.principal, initial.principal)
        self.assertEqual(
            continued.stream_session_id, initial.stream_session_id
        )
        self.assertNotEqual(continued.turn_id, initial.turn_id)
        self.assertNotEqual(continued.model_call_id, initial.model_call_id)
        self.assertEqual(self.execution.messages[-2:], correlated)

    async def test_result_compare_and_swap_allows_one_concurrent_winner(
        self,
    ) -> None:
        request = await self._pending()
        request = _resolved_request(request)
        result = _result(request)
        messages = _correlated_messages(result)
        outcomes = await gather(
            self.execution.record_interaction_result(
                request,
                result,
                messages,
            ),
            self.execution.record_interaction_result(
                request,
                result,
                messages,
            ),
        )
        self.assertEqual(sorted(outcomes), [False, True])
        self.assertEqual(
            sum(
                entry.kind is ExecutionLedgerEntryKind.INTERACTION_RESULT
                for entry in self.execution.ledger
            ),
            1,
        )

    async def test_conflicting_replay_and_wrong_correlation_fail_closed(
        self,
    ) -> None:
        request = await self._pending()
        request = _resolved_request(request)
        result = _result(request)
        messages = _correlated_messages(result)
        await self.execution.record_interaction_result(
            request,
            result,
            messages,
        )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                request,
                result,
                (_message("different", MessageRole.TOOL),),
            )

        other = await _execution()
        await other.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        wrong_request = _request(
            replace(request.origin, run_id=RunId("run-other"))
        )
        with self.assertRaises(ExecutionCorrelationError):
            await other.mark_interaction_pending(wrong_request)

    async def test_result_requires_waiting_state_and_exact_request(
        self,
    ) -> None:
        request = _request(self.execution.origin)
        with self.assertRaises(ExecutionStateError):
            await self.execution.record_interaction_result(
                request,
                _result(request),
                (),
            )
        terminal_without_reservation = _resolved_request(request)
        terminal_result = _result(terminal_without_reservation)
        with self.assertRaises(ExecutionStateError):
            await self.execution.record_interaction_result(
                terminal_without_reservation,
                terminal_result,
                _correlated_messages(terminal_result),
            )

        request = await self._pending()
        wrong_request = _resolved_request(
            _request(
                self.execution.origin,
                request_id="request-other",
                continuation_id="continuation-other",
            )
        )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                wrong_request,
                _result(wrong_request),
                (),
            )
        wrong_result = InputUnavailableResult(
            request_id=InputRequestId("request-other"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        )
        request = _resolved_request(request)
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                request,
                wrong_result,
                _correlated_messages(wrong_result),
            )

    async def test_request_lifecycle_and_immutable_contract_fail_closed(
        self,
    ) -> None:
        await self.execution.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        created = _request(self.execution.origin, pending=False)
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.mark_interaction_pending(created)
        terminal_before_pending = _resolved_request(
            _request(self.execution.origin)
        )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.mark_interaction_pending(
                terminal_before_pending
            )
        self.assertEqual(self.execution.revision, 1)

        pending = _request(self.execution.origin)
        await self.execution.mark_interaction_pending(pending)
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                pending,
                _result(pending),
                (),
            )

        terminal = _resolved_request(pending)
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                replace(terminal, reason="Changed immutable reason."),
                _result(terminal),
                (),
            )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                terminal,
                InputUnavailableResult(
                    request_id=InputRequestId("request-other"),
                    provenance=AnswerProvenance.POLICY,
                    resolved_at=_NOW,
                ),
                (),
            )

        result = _result(terminal)
        self.assertTrue(
            await self.execution.record_interaction_result(
                terminal,
                result,
                _correlated_messages(result),
            )
        )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_result(
                replace(terminal, reason="Changed replay contract."),
                result,
                (),
            )

    async def test_terminal_request_is_recorded_and_replayed_exactly_once(
        self,
    ) -> None:
        pending = await self._pending()
        terminal = _terminated_request(pending)
        outcome = project_resolution_to_model(
            terminal,
            containing_run_exists=True,
        )
        assert isinstance(outcome, TerminateInputContinuation)
        mismatched = TerminateInputContinuation(
            request_id=terminal.request_id,
            status=ResolutionStatus.SUPERSEDED,
        )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_termination(
                terminal,
                mismatched,
            )

        self.assertTrue(
            await self.execution.record_interaction_termination(
                terminal,
                outcome,
            )
        )
        self.assertFalse(
            await self.execution.record_interaction_termination(
                terminal,
                outcome,
            )
        )
        self.assertFalse(await self.execution.cancel())
        self.assertEqual(self.execution.status, AgentExecutionStatus.CANCELLED)
        self.assertIsNone(self.execution.pending_request)
        entry = self.execution.ledger[-1]
        self.assertEqual(
            entry.kind,
            ExecutionLedgerEntryKind.INTERACTION_TERMINATED,
        )
        self.assertEqual(entry.request, terminal)
        self.assertEqual(entry.termination_outcome, outcome)
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.record_interaction_termination(
                terminal,
                mismatched,
            )

        unrelated = _terminated_request(
            _request(
                self.execution.initial_origin,
                request_id="request-unrelated",
                continuation_id="continuation-unrelated",
            )
        )
        unrelated_outcome = project_resolution_to_model(
            unrelated,
            containing_run_exists=True,
        )
        assert isinstance(unrelated_outcome, TerminateInputContinuation)
        with self.assertRaises(ExecutionStateError):
            await self.execution.record_interaction_termination(
                unrelated,
                unrelated_outcome,
            )

    async def test_nonmatching_committed_result_is_not_an_idempotent_replay(
        self,
    ) -> None:
        request = await self._pending()
        request = _resolved_request(request)
        result = _result(request)
        await self.execution.record_interaction_result(
            request,
            result,
            _correlated_messages(result),
        )
        other = _resolved_request(
            _request(
                self.execution.initial_origin,
                request_id="request-other",
                continuation_id="continuation-other",
            )
        )
        with self.assertRaises(ExecutionStateError):
            await self.execution.record_interaction_result(
                other,
                _result(other),
                (),
            )

    async def test_abandon_releases_reservation_without_rolling_back_count(
        self,
    ) -> None:
        await self.execution.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        self.assertTrue(await self.execution.abandon_interaction())
        self.assertFalse(await self.execution.abandon_interaction())
        self.assertEqual(self.execution.status, AgentExecutionStatus.RUNNING)
        self.assertEqual(self.execution.interaction_count, 1)
        self.assertEqual(
            self.execution.ledger[-1].kind,
            ExecutionLedgerEntryKind.INTERACTION_ABANDONED,
        )

    async def test_equivalent_request_limit_is_separate_and_bounded(
        self,
    ) -> None:
        for _ in range(MAXIMUM_EQUIVALENT_INPUT_REQUESTS):
            await self._resolve_and_advance()
        with self.assertRaises(InteractionLoopLimitError):
            await self.execution.begin_interaction(
                "fingerprint", _task_input_call(), _task_input_message()
            )
        self.assertEqual(
            self.execution.interaction_count,
            MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
        )
        await self.execution.begin_interaction(
            "different", _task_input_call(), _task_input_message()
        )
        self.assertEqual(
            self.execution.interaction_count,
            MAXIMUM_EQUIVALENT_INPUT_REQUESTS + 1,
        )

    async def test_input_required_replay_and_new_stream_resume(self) -> None:
        request = await self._pending()
        initial = self.execution.origin
        required = InputRequiredResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            detached_resumption_available=True,
        )
        self.assertTrue(await self.execution.mark_input_required(required))
        self.assertFalse(await self.execution.mark_input_required(required))
        conflicting = InputRequiredResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            detached_resumption_available=False,
        )
        with self.assertRaises(ExecutionStateError):
            await self.execution.mark_input_required(conflicting)
        self.assertEqual(
            self.execution.status, AgentExecutionStatus.INPUT_REQUIRED
        )
        with self.assertRaises(ExecutionStateError):
            await self.execution.advance_model_turn()
        with self.assertRaises(ExecutionStateError):
            await self.execution.advance_model_turn(new_stream_session=True)
        self.assertEqual(self.execution.origin, initial)
        self.assertEqual(self.execution.pending_request, request)
        self.assertEqual(
            self.execution.status, AgentExecutionStatus.INPUT_REQUIRED
        )

    async def test_input_required_rejects_wrong_correlation(self) -> None:
        request = await self._pending()
        wrong = InputRequiredResult(
            request_id=InputRequestId("request-other"),
            continuation_id=request.continuation_id,
            detached_resumption_available=True,
        )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.mark_input_required(wrong)

    async def test_turn_advance_rejects_an_interaction_boundary(self) -> None:
        await self.execution.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        with self.assertRaises(ExecutionStateError):
            await self.execution.advance_model_turn()

    async def test_invalid_interaction_transitions_leave_state_unchanged(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            await self.execution.begin_interaction(
                cast(Any, 1), _task_input_call(), _task_input_message()
            )
        with self.assertRaises(TypeError):
            await self.execution.begin_interaction(
                "fingerprint",
                cast(Any, object()),
                _task_input_message(),
            )
        with self.assertRaises(TypeError):
            await self.execution.begin_interaction(
                "fingerprint",
                _task_input_call(),
                cast(Any, object()),
            )
        with self.assertRaises(ExecutionCorrelationError):
            await self.execution.begin_interaction(
                "fingerprint",
                _task_input_call(),
                _message("forged", MessageRole.ASSISTANT),
            )
        with self.assertRaises(TypeError):
            await self.execution.mark_interaction_pending(cast(Any, object()))
        with self.assertRaises(TypeError):
            await self.execution.record_interaction_result(
                cast(Any, object()),
                cast(Any, object()),
                cast(Any, []),
            )
        with self.assertRaises(TypeError):
            await self.execution.record_interaction_result(
                _request(self.execution.origin),
                cast(Any, object()),
                (),
            )
        with self.assertRaises(TypeError):
            await self.execution.mark_input_required(cast(Any, object()))
        with self.assertRaises(TypeError):
            await self.execution.record_interaction_termination(
                cast(Any, object()),
                cast(Any, object()),
            )
        with self.assertRaises(TypeError):
            await self.execution.record_interaction_termination(
                _terminated_request(_request(self.execution.origin)),
                cast(Any, object()),
            )
        with self.assertRaises(TypeError):
            await self.execution.advance_model_turn(
                new_stream_session=cast(Any, 1)
            )
        self.assertFalse(await self.execution.abandon_interaction())
        self.assertEqual(self.execution.revision, 0)


class ExecutionConcurrencyTest(IsolatedAsyncioTestCase):
    """Prove external identity awaits never hold the state lock."""

    async def test_cancellation_wins_while_turn_identity_awaits(self) -> None:
        factory = _BlockingIdFactory()
        execution = await _execution(factory=factory)
        factory.block_turn = True
        advance = create_task(execution.advance_model_turn())
        await factory.started.wait()
        self.assertTrue(await execution.cancel())
        factory.release.set()
        with self.assertRaises(ExecutionRevisionError):
            await advance
        self.assertEqual(execution.status, AgentExecutionStatus.CANCELLED)

    async def test_cancelled_identity_await_leaves_snapshot_unchanged(
        self,
    ) -> None:
        factory = _BlockingIdFactory()
        execution = await _execution(factory=factory)
        before = execution.snapshot
        factory.block_turn = True
        advance = create_task(execution.advance_model_turn())
        await factory.started.wait()
        advance.cancel()
        with self.assertRaises(CancelledError):
            await advance
        self.assertEqual(execution.snapshot, before)
        self.assertEqual(execution.status, AgentExecutionStatus.RUNNING)


class RuntimeValidationTest(TestCase):
    """Exercise attached runtime and snapshot validation boundaries."""

    def test_runtime_accepts_defaults_and_rejects_invalid_dependencies(
        self,
    ) -> None:
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, _Broker()),
            actor=InteractionActor(principal=_principal()),
            handler=cast(InputHandler, _handler),
            id_factory=_IdFactory(),
        )
        self.assertIsNone(runtime.branch_id)
        invalid: tuple[dict[str, Any], ...] = (
            {"actor": object()},
            {"broker": object()},
            {"handler": lambda _context: None},
            {"id_factory": object()},
            {"task_id": ""},
        )
        base: dict[str, Any] = {
            "broker": cast(InteractionBroker, _Broker()),
            "actor": InteractionActor(principal=_principal()),
            "handler": cast(InputHandler, _handler),
            "id_factory": _IdFactory(),
        }
        for values in invalid:
            with (
                self.subTest(values=values),
                self.assertRaises((TypeError, ValueError)),
            ):
                cast(Any, AttachedInteractionRuntime)(**{**base, **values})
        with self.assertRaises(ExecutionCorrelationError):
            AttachedInteractionRuntime(
                **base,
                branch_id=BranchId("same"),
                parent_branch_id=BranchId("same"),
            )

    def test_runtime_accepts_an_async_callable_object(self) -> None:
        class Handler:
            async def __call__(
                self,
                _context: InputHandlerContext,
            ) -> InputHandlerOutcome:
                return InputHandlerDetached()

        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, _Broker()),
            actor=InteractionActor(principal=_principal()),
            handler=cast(InputHandler, Handler()),
            id_factory=_IdFactory(),
        )
        self.assertIsInstance(runtime.actor, InteractionActor)

    def test_snapshot_rejects_inconsistent_state(self) -> None:
        origin = _origin()
        entry = ExecutionLedgerEntry(
            sequence=0,
            kind=ExecutionLedgerEntryKind.INPUT,
            origin=origin,
        )
        base: dict[str, Any] = {
            "revision": 0,
            "status": AgentExecutionStatus.RUNNING,
            "origin": origin,
            "ledger": (entry,),
            "messages": (),
            "pending_request": None,
            "active_interaction_fingerprint": None,
            "interaction_fingerprint_counts": (),
            "interaction_count": 0,
            "memory_sync_cursor": 0,
            "response_sync_cursor": 0,
            "cleanup_started": False,
        }
        invalid = (
            ("revision", True, TypeError),
            ("status", "running", TypeError),
            ("origin", object(), TypeError),
            ("ledger", (object(),), TypeError),
            (
                "ledger",
                (replace(entry, sequence=1),),
                ExecutionStateError,
            ),
            ("messages", (object(),), TypeError),
            ("pending_request", object(), TypeError),
            (
                "pending_request",
                _resolved_request(_request(origin)),
                ExecutionStateError,
            ),
            ("active_interaction_fingerprint", "", ValueError),
            ("interaction_fingerprint_counts", (("x", 0),), TypeError),
            (
                "interaction_fingerprint_counts",
                (("z", 1), ("a", 1)),
                ExecutionStateError,
            ),
            ("interaction_count", True, TypeError),
            (
                "interaction_fingerprint_counts",
                (("x", 1),),
                ExecutionStateError,
            ),
            ("memory_sync_cursor", 1, TypeError),
            ("response_sync_cursor", 2, TypeError),
            ("cleanup_started", 1, TypeError),
        )
        for name, value, error in invalid:
            with (
                self.subTest(name=name, value=value),
                self.assertRaises(error),
            ):
                cast(Any, AgentExecutionSnapshot)(**{**base, name: value})

    def test_snapshot_rejects_post_terminal_work_and_origin_drift(
        self,
    ) -> None:
        origin = _origin()
        input_entry = ExecutionLedgerEntry(
            sequence=0,
            kind=ExecutionLedgerEntryKind.INPUT,
            origin=origin,
        )
        completed = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.COMPLETED,
            origin=origin,
        )
        post_terminal = ExecutionLedgerEntry(
            sequence=2,
            kind=ExecutionLedgerEntryKind.MODEL_TURN,
            origin=origin,
        )
        drifted = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.MODEL_TURN,
            origin=replace(origin, run_id=RunId("drifted-run")),
        )
        volatile_drift = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.TRANSCRIPT,
            origin=replace(origin, turn_id=TurnId("drifted-turn")),
            messages=(_message("drifted"),),
        )
        unadvanced_turn = ExecutionLedgerEntry(
            sequence=1,
            kind=ExecutionLedgerEntryKind.MODEL_TURN,
            origin=origin,
        )
        base: dict[str, Any] = {
            "revision": 2,
            "status": AgentExecutionStatus.COMPLETED,
            "origin": origin,
            "ledger": (input_entry, completed),
            "messages": (),
            "pending_request": None,
            "active_interaction_fingerprint": None,
            "interaction_fingerprint_counts": (),
            "interaction_count": 0,
            "memory_sync_cursor": 0,
            "response_sync_cursor": 0,
            "cleanup_started": False,
        }

        with self.assertRaises(ExecutionStateError):
            AgentExecutionSnapshot(
                **cast(
                    Any,
                    {
                        **base,
                        "revision": 3,
                        "ledger": (
                            input_entry,
                            completed,
                            post_terminal,
                        ),
                    },
                )
            )
        with self.assertRaises(ExecutionCorrelationError):
            AgentExecutionSnapshot(
                **cast(
                    Any,
                    {
                        **base,
                        "status": AgentExecutionStatus.RUNNING,
                        "ledger": (input_entry, drifted),
                    },
                )
            )
        for invalid_entry in (volatile_drift, unadvanced_turn):
            with (
                self.subTest(kind=invalid_entry.kind),
                self.assertRaises(ExecutionCorrelationError),
            ):
                AgentExecutionSnapshot(
                    **cast(
                        Any,
                        {
                            **base,
                            "status": AgentExecutionStatus.RUNNING,
                            "origin": invalid_entry.origin,
                            "ledger": (input_entry, invalid_entry),
                        },
                    )
                )
        with self.assertRaises(ExecutionStateError):
            AgentExecutionSnapshot(
                **cast(Any, {**base, "cleanup_started": True})
            )

    def test_typed_suspension_and_termination_errors(self) -> None:
        required = InputRequiredResult(
            request_id=InputRequestId("request-1"),
            continuation_id=ContinuationId("continuation-1"),
            detached_resumption_available=True,
        )
        suspended = ExecutionInputRequiredError(required)
        self.assertIs(suspended.result, required)
        terminated_outcome = TerminateInputContinuation(
            request_id=InputRequestId("request-1"),
            status=ResolutionStatus.EXPIRED,
        )
        terminated = ExecutionTerminatedError(terminated_outcome)
        self.assertIs(terminated.outcome, terminated_outcome)
        with self.assertRaises(TypeError):
            ExecutionInputRequiredError(cast(Any, object()))
        with self.assertRaises(TypeError):
            ExecutionTerminatedError(cast(Any, object()))
