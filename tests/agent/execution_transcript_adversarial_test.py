"""Exercise adversarial execution transcript and synchronization invariants."""

from asyncio import CancelledError, Event, create_task
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    ExecutionCorrelationError,
    ExecutionIdFactory,
    ExecutionLedgerEntryKind,
    ExecutionMemoryComponent,
    ExecutionMemoryEntry,
    ExecutionStateError,
    create_agent_execution,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
    _ToolExecutionOutcome,
)
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    MessageToolCall,
    ToolCall,
    ToolCallContext,
    TransformerEngineSettings,
    normalize_tool_arguments,
)
from avalan.event.manager import EventManager, EventManagerMode
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
    InputRequiredResult,
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
)
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.capability import (
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    validate_canonical_stream_items,
)
from avalan.tool.manager import ToolManager

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


class _Ids:
    """Mint deterministic execution identifiers."""

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


def _definition() -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://adversarial",
        agent_definition_revision="agent-r1",
        operation_id="operation-adversarial",
        operation_index=0,
        model_config_reference="model-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )


def _principal() -> PrincipalScope:
    return PrincipalScope(user_id=UserId("user-adversarial"))


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-adversarial"),
        turn_id=TurnId("turn-adversarial"),
        task_id=TaskId("task-adversarial"),
        agent_id=AgentId("agent-adversarial"),
        branch_id=BranchId("branch-adversarial"),
        model_call_id=ModelCallId("model-call-adversarial"),
        stream_session_id=StreamSessionId("stream-adversarial"),
        definition=_definition(),
        principal=_principal(),
    )


def _message(content: str, role: MessageRole = MessageRole.USER) -> Message:
    return Message(role=role, content=content)


def _task_input_call() -> TaskInputCapabilityCall:
    return TaskInputCapabilityCall(
        call_id="input-call",
        provider_name="request_user_input",
        arguments={"mode": "required"},
        mode=RequirementMode.REQUIRED,
        reason="Need confirmation.",
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


async def _execution(
    *messages: Message,
    synced_message_prefix: int = 0,
) -> AgentExecution:
    return await create_agent_execution(
        definition=_definition(),
        agent_id=AgentId("agent-adversarial"),
        principal=_principal(),
        initial_messages=messages,
        synced_message_prefix=synced_message_prefix,
        id_factory=_Ids(),
    )


def _pending_request(origin: ExecutionOrigin) -> InputRequest:
    return InputRequest(
        request_id=InputRequestId("request-adversarial"),
        continuation_id=ContinuationId("continuation-adversarial"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Need confirmation.",
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


def _result(request: InputRequest) -> InputUnavailableResult:
    return InputUnavailableResult(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
    )


class _Engine:
    """Expose the minimum engine surface required by EngineAgent."""

    model_id = "adversarial-model"
    model_type = "fake"

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<adversarial-eos>")


class _Agent(EngineAgent):
    """Use the production engine-agent execution path."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


def _last_text(input_value: object) -> str:
    messages = (
        [input_value]
        if isinstance(input_value, Message)
        else cast(list[Message], input_value)
    )
    for message in reversed(messages):
        if isinstance(message.content, str):
            return message.content
    raise AssertionError("model input has no text")


def _text_response(text: str) -> TextGenerationResponse:
    return TextGenerationResponse(
        lambda **_: text,
        logger=getLogger(),
        use_async_generator=False,
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )


class _ModelManager:
    """Return a distinct mutable provider response for every invocation."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.responses: list[TextGenerationResponse] = []

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        response = _text_response(f"answer:{_last_text(call.operation.input)}")
        self.responses.append(response)
        return response


def _operation() -> AgentOperation:
    environment = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="adversarial-model",
            params={},
        ),
        settings=TransformerEngineSettings(),
    )
    return AgentOperation(
        specification=Specification(instructions="answer exactly"),
        environment=environment,
    )


class _Harness:
    """Own one shared real EngineAgent and Orchestrator."""

    def __init__(self) -> None:
        self.logger = getLogger()
        self.events = EventManager(mode=EventManagerMode.TEST)
        self.memory = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=RecentMessageMemory(),
            text_partitioner=None,
            logger=self.logger,
            event_manager=self.events,
        )
        self.tool = ToolManager.create_instance()
        self.engine = _Engine()
        self.manager = _ModelManager()
        self.operation = _operation()
        uri = self.operation.environment.engine_uri
        self.agent = _Agent(
            cast(Any, self.engine),
            self.memory,
            self.tool,
            self.events,
            cast(ModelManager, self.manager),
            uri,
        )
        self.orchestrator = Orchestrator(
            self.logger,
            cast(ModelManager, self.manager),
            self.memory,
            self.tool,
            self.events,
            self.operation,
        )
        environment_hash = dumps(asdict(self.operation.environment))
        self.orchestrator._engine_agents[environment_hash] = self.agent
        self.ids = _Ids()

    async def call(self, prompt: str) -> OrchestratorResponse:
        return await self.orchestrator(
            prompt,
            execution_id_factory=cast(ExecutionIdFactory, self.ids),
        )

    async def close(self) -> None:
        await self.events.aclose()


class TranscriptTruthTest(IsolatedAsyncioTestCase):
    """Require immutable final transcript truth after provider completion."""

    async def asyncSetUp(self) -> None:
        self.harness = _Harness()

    async def asyncTearDown(self) -> None:
        await self.harness.close()

    async def test_completion_materializes_final_assistant_transcript(
        self,
    ) -> None:
        response = await self.harness.call("alpha")
        provider_response = self.harness.manager.responses[-1]
        self.assertEqual(await response.to_str(), "answer:alpha")
        execution = response._execution
        assert execution is not None

        self.assertIs(execution.status, AgentExecutionStatus.COMPLETED)
        self.assertEqual(
            tuple(
                (message.role, message.content)
                for message in execution.messages
            ),
            (
                (MessageRole.USER, "alpha"),
                (MessageRole.ASSISTANT, "answer:alpha"),
            ),
        )
        self.assertEqual(execution.last_response, "answer:alpha")
        self.assertFalse(
            any(
                isinstance(entry.response, TextGenerationResponse)
                for entry in execution.ledger
            )
        )
        self.assertEqual(
            sum(
                entry.kind is ExecutionLedgerEntryKind.COMPLETED
                for entry in execution.ledger
            ),
            1,
        )

        provider_response.set_thinking(True)
        self.assertEqual(execution.last_response, "answer:alpha")
        self.assertIsNot(execution.last_response, provider_response)

    async def test_same_task_overlapping_runs_sync_explicit_executions(
        self,
    ) -> None:
        response_a = await self.harness.call("alpha")
        response_b = await self.harness.call("beta")
        self.assertEqual(await response_a.to_str(), "answer:alpha")
        self.assertEqual(await response_b.to_str(), "answer:beta")
        execution_a = response_a._execution
        execution_b = response_b._execution
        assert execution_a is not None and execution_b is not None

        await self.harness.agent.sync_messages(execution_b)
        await self.harness.agent.sync_messages(execution_a)
        await self.harness.agent.sync_messages(execution_b)
        await self.harness.agent.sync_messages(execution_a)

        self.assertEqual(
            tuple(
                item.message.content
                for item in self.harness.memory.recent_messages or ()
            ),
            ("beta", "answer:beta", "alpha", "answer:alpha"),
        )


class RetrySafeSynchronizationTest(IsolatedAsyncioTestCase):
    """Require per-entry acknowledgement under failure and cancellation."""

    async def test_failed_message_and_response_append_retry_without_loss(
        self,
    ) -> None:
        execution = await _execution(
            _message("one"),
            _message("two"),
        )
        await execution.record_response("answer")
        saved_messages: list[str] = []
        saved_responses: list[str] = []
        fail_message = True
        fail_response = True

        async def append(entry: ExecutionMemoryEntry) -> None:
            nonlocal fail_message, fail_response
            if entry.component is ExecutionMemoryComponent.RESPONSE:
                assert isinstance(entry.message.content, str)
                if fail_response:
                    fail_response = False
                    raise RuntimeError("response append failed before commit")
                saved_responses.append(entry.message.content)
                return
            message = entry.message
            assert isinstance(message.content, str)
            if message.content == "two" and fail_message:
                fail_message = False
                raise RuntimeError("message append failed before commit")
            saved_messages.append(message.content)

        sink = _ExecutionMemorySink(append)

        with self.assertRaisesRegex(RuntimeError, "message append failed"):
            await execution.sync_memory(sink)
        self.assertEqual(saved_messages, ["one"])

        with self.assertRaisesRegex(RuntimeError, "response append failed"):
            await execution.sync_memory(sink)
        self.assertEqual(saved_messages, ["one", "two"])
        self.assertEqual(saved_responses, [])

        await execution.sync_memory(sink)
        await execution.sync_memory(sink)
        self.assertEqual(saved_messages, ["one", "two"])
        self.assertEqual(saved_responses, ["answer"])

    async def test_cancelled_append_keeps_entry_retryable(self) -> None:
        execution = await _execution(_message("retry me"))
        started = Event()
        blocked = Event()
        saved: list[str] = []

        async def blocked_append(entry: ExecutionMemoryEntry) -> None:
            self.assertIs(entry.component, ExecutionMemoryComponent.MESSAGE)
            started.set()
            await blocked.wait()
            message = entry.message
            assert isinstance(message.content, str)
            saved.append(message.content)

        task = create_task(
            execution.sync_memory(_ExecutionMemorySink(blocked_append))
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(CancelledError):
            await task

        async def retry_append(entry: ExecutionMemoryEntry) -> None:
            self.assertIs(entry.component, ExecutionMemoryComponent.MESSAGE)
            message = entry.message
            assert isinstance(message.content, str)
            saved.append(message.content)

        retry_sink = _ExecutionMemorySink(retry_append)
        await execution.sync_memory(retry_sink)
        await execution.sync_memory(retry_sink)
        self.assertEqual(saved, ["retry me"])


def _empty_stream_response() -> TextGenerationResponse:
    async def items() -> AsyncIterator[CanonicalStreamItem]:
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=1,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=2,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda **_: items(),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )


class StreamTerminalTruthTest(IsolatedAsyncioTestCase):
    """Require guard termination to agree across stream and execution state."""

    async def test_empty_tool_observation_completes_execution_once(
        self,
    ) -> None:
        execution = await _execution(_message("initial"))
        operation = _operation()
        response = _empty_stream_response()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = _Engine()
        events = MagicMock(spec=EventManager)
        events.trigger = AsyncMock()
        context = ModelCallContext(
            specification=operation.specification,
            input=_message("initial"),
            execution=execution,
            execution_origin=execution.origin,
        )
        orchestrator_response = OrchestratorResponse(
            _message("initial"),
            response,
            agent,
            operation,
            {},
            context,
            event_manager=events,
            enable_tool_parsing=False,
        )
        orchestrator_response.__aiter__()
        orchestrator_response._tool_result_outcomes.put(
            _ToolExecutionOutcome(
                call=ToolCall(id="call-1", name="tool", arguments=None),
                context=ToolCallContext(
                    execution=execution,
                    execution_origin=execution.origin,
                ),
                planned_index=0,
                result=None,
            )
        )

        items = [item async for item in orchestrator_response]
        validate_canonical_stream_items(items)
        self.assertIs(execution.status, AgentExecutionStatus.COMPLETED)
        self.assertEqual(
            sum(
                item.kind is StreamItemKind.STREAM_COMPLETED for item in items
            ),
            1,
        )
        self.assertEqual(
            sum(
                entry.kind is ExecutionLedgerEntryKind.COMPLETED
                for entry in execution.ledger
            ),
            1,
        )


class InteractionResumeAdversarialTest(IsolatedAsyncioTestCase):
    """Require resolution, not identity generation, to unblock a branch."""

    async def test_input_required_cannot_advance_without_resolution(
        self,
    ) -> None:
        execution = await _execution(_message("initial"))
        await execution.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        request = _pending_request(execution.origin)
        await execution.mark_interaction_pending(request)
        required = InputRequiredResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            detached_resumption_available=True,
        )
        await execution.mark_input_required(required)
        before = execution.snapshot

        with self.assertRaises(ExecutionStateError):
            await execution.advance_model_turn(new_stream_session=True)

        self.assertEqual(execution.snapshot, before)
        self.assertIs(execution.status, AgentExecutionStatus.INPUT_REQUIRED)
        self.assertEqual(execution.pending_request, request)

    async def test_resume_rejects_missing_correlated_result_message(
        self,
    ) -> None:
        execution = await _execution(_message("initial"))
        await execution.begin_interaction(
            "fingerprint", _task_input_call(), _task_input_message()
        )
        pending = _pending_request(execution.origin)
        await execution.mark_interaction_pending(pending)
        terminal = _resolved_request(pending)

        with self.assertRaises(ExecutionCorrelationError):
            await execution.record_interaction_result(
                terminal,
                _result(terminal),
                (),
            )

        self.assertIs(
            execution.status,
            AgentExecutionStatus.WAITING_FOR_INPUT,
        )
        self.assertEqual(execution.pending_request, pending)


class SnapshotInvariantTest(TestCase):
    """Reject public execution snapshots that encode impossible states."""

    def test_public_snapshot_rejects_impossible_status_shapes(self) -> None:
        origin = _origin()
        running = AgentExecution(
            origin=origin,
            id_factory=_Ids(),
            initial_messages=(_message("initial"),),
        ).snapshot
        pending = _pending_request(origin)
        cases = (
            {
                "status": AgentExecutionStatus.WAITING_FOR_INPUT,
                "pending_request": None,
                "active_interaction_fingerprint": "fingerprint",
            },
            {
                "status": AgentExecutionStatus.RUNNING,
                "pending_request": pending,
                "active_interaction_fingerprint": "fingerprint",
                "interaction_fingerprint_counts": (("fingerprint", 1),),
                "interaction_count": 1,
            },
            {
                "status": AgentExecutionStatus.PREPARING_INPUT,
                "active_interaction_fingerprint": None,
            },
            {
                "status": AgentExecutionStatus.COMPLETED,
            },
            {
                "status": AgentExecutionStatus.RUNNING,
                "cleanup_started": True,
            },
        )
        for values in cases:
            with (
                self.subTest(values=values),
                self.assertRaises(ExecutionStateError),
            ):
                replace(running, **values)
