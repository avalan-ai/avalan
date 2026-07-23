"""Exercise adversarial response ownership and runtime isolation boundaries."""

from asyncio import CancelledError, Event, create_task, gather, sleep, wait_for
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent import engine as engine_module
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionCorrelationError,
    ExecutionLedgerEntryKind,
    create_agent_execution,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineMessageIdempotencyKey,
    EngineSettings,
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event import Event as AgentEvent
from avalan.event import EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.interaction import (
    AgentId,
    AnswerProvenance,
    BranchId,
    ConfirmationQuestion,
    ContinuationId,
    CreateInteractionCommand,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
    InputRequestId,
    InteractionActor,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionDelivery,
    InteractionExecutionScope,
    InteractionPolicy,
    InteractionRequestResult,
    InteractionStoreRevision,
    PrincipalScope,
    QuestionId,
    RegisterInteractionBranchCommand,
    RequirementMode,
    RunId,
    TaskId,
    TerminalizeInteractionScopeCommand,
    UserId,
    apply_create_interaction,
    create_input_request,
)
from avalan.interaction.broker import InteractionBroker
from avalan.interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.model import (
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
)
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.engine import Engine
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.tool.manager import ToolManager

if TYPE_CHECKING:
    from avalan.server.routers.mcp import StreamResponse
else:
    StreamResponse = Any

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


class _HarnessEngine:
    """Expose one explicit provider-support trust surface."""

    model_id = "response-ownership-model"
    model_type = "fake"

    def __init__(self, support: ProviderCapabilitySupport) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<ownership-eos>")
        self.provider_capability_support = support


class _Agent(EngineAgent):
    """Use the production model-dispatch implementation unchanged."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


class _FailingSyncAgent(_Agent):
    """Fail one assistant append before its acknowledgement commits."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sync_attempts: list[tuple[MessageRole, str]] = []
        self._fail_assistant_once = True
        super().__init__(*args, **kwargs)

    async def sync_message(
        self,
        message: Message,
        *,
        idempotency_key: EngineMessageIdempotencyKey | None = None,
    ) -> None:
        """Record each attempt and inject one pre-commit sink failure."""
        assert isinstance(message.content, str)
        self.sync_attempts.append((message.role, message.content))
        if message.role is MessageRole.ASSISTANT and self._fail_assistant_once:
            self._fail_assistant_once = False
            raise RuntimeError("injected assistant memory failure")
        await super().sync_message(
            message,
            idempotency_key=idempotency_key,
        )


def _last_text(input_value: object) -> str:
    """Return the last text message supplied to one model call."""
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
    """Return one materialized deterministic provider response."""
    return TextGenerationResponse(
        lambda **_: text,
        logger=getLogger(),
        use_async_generator=False,
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )


class _ModelManager:
    """Capture model contexts and answer from the last user message."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        return _text_response(f"answer:{_last_text(call.operation.input)}")


class _ConcurrentModelManager(_ModelManager):
    """Hold two overlapping calls behind independently released barriers."""

    def __init__(self) -> None:
        super().__init__()
        self.both_started = Event()
        self.releases = {"alpha": Event(), "beta": Event()}

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        prompt = _last_text(call.operation.input)
        self.calls.append(call)
        if len(self.calls) == 2:
            self.both_started.set()
        await self.releases[prompt].wait()
        return _text_response(f"answer:{prompt}")


class _TrackedTextResponse(TextGenerationResponse):
    """Record provider cleanup without changing response semantics."""

    def __init__(self, text: str) -> None:
        super().__init__(
            lambda **_: text,
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        self.cleanup_calls: list[str] = []

    async def cancel(self) -> None:
        """Record and delegate provider cancellation."""
        self.cleanup_calls.append("cancel")
        await super().cancel()

    async def aclose(self) -> None:
        """Record and delegate provider closure."""
        self.cleanup_calls.append("close")
        await super().aclose()


class _TrackedModelManager(_ModelManager):
    """Return one response whose ownership cleanup is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.responses: list[_TrackedTextResponse] = []

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        response = _TrackedTextResponse(
            f"answer:{_last_text(call.operation.input)}"
        )
        self.responses.append(response)
        return response


class _FailingCloseResponse(TextGenerationResponse):
    """Fail two close operations before one exact successful retry."""

    def __init__(self) -> None:
        super().__init__(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
        )
        self.close_attempts = 0
        self.events: list[str] = []

    async def cancel(self) -> None:
        """Attempt cancellation through the same provider close primitive."""
        self.events.append("cancel")
        await self._attempt_close()

    async def aclose(self) -> None:
        """Attempt an explicit provider close."""
        self.events.append("close")
        await self._attempt_close()

    async def _attempt_close(self) -> None:
        self.close_attempts += 1
        if self.close_attempts <= 2:
            raise RuntimeError(
                f"provider close attempt {self.close_attempts} failed"
            )
        self._output_closed = True


class _NeverReturningCancelResponse(TextGenerationResponse):
    """Close promptly while retaining a blocked cancellation operation."""

    def __init__(self) -> None:
        super().__init__(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
        )
        self.cancel_started = Event()
        self.cancel_release = Event()
        self.cancel_failure = RuntimeError("late provider cancel failure")
        self.close_completed = Event()
        self.close_calls = 0

    async def cancel(self) -> None:
        """Block until the test releases one late cancellation failure."""
        self.cancel_started.set()
        await self.cancel_release.wait()
        raise self.cancel_failure

    async def aclose(self) -> None:
        """Close independently of the blocked cancellation operation."""
        self.close_calls += 1
        self._output_closed = True
        self.close_completed.set()


class _ImmediateCleanupResponse(TextGenerationResponse):
    """Complete cancellation immediately for no-head-of-line probes."""

    def __init__(self) -> None:
        super().__init__(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
        )
        self.closed = Event()
        self.cancel_calls = 0
        self.close_calls = 0

    async def cancel(self) -> None:
        """Close the response immediately."""
        self.cancel_calls += 1
        await self.aclose()

    async def aclose(self) -> None:
        """Record immediate provider closure."""
        self.close_calls += 1
        self._output_closed = True
        self.closed.set()


class _CallbackRegistrationFailureResponse(_ImmediateCleanupResponse):
    """Fail ownership callback setup after the registry write."""

    def __init__(self) -> None:
        super().__init__()
        self.callback_failure = RuntimeError(
            "provider callback registration failed"
        )

    def add_done_callback(self, callback: Any) -> NoReturn:
        """Reject provider-consumption callback registration."""
        del callback
        raise self.callback_failure


class _NeverReturningCloseResponse(TextGenerationResponse):
    """Fail cancellation once and retain a late successful close."""

    def __init__(self) -> None:
        super().__init__(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
        )
        self.cancel_failure = RuntimeError("provider cancel failed once")
        self.cancel_calls = 0
        self.close_started = Event()
        self.close_release = Event()

    async def cancel(self) -> None:
        """Fail the first cancellation without closing the provider."""
        self.cancel_calls += 1
        if self.cancel_calls == 1:
            raise self.cancel_failure

    async def aclose(self) -> None:
        """Close only after the test releases the retained operation."""
        self.close_started.set()
        await self.close_release.wait()
        self._output_closed = True


class _NoClosureResponse(TextGenerationResponse):
    """Return from close without satisfying the provider close contract."""

    def __init__(self) -> None:
        super().__init__(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
        )
        self.cancel_failure = RuntimeError("provider cancel failed")

    async def cancel(self) -> NoReturn:
        """Fail cancellation before the explicit close attempt."""
        raise self.cancel_failure

    async def aclose(self) -> None:
        """Return without marking the provider closed."""


class _PersistentFailModelManager(_TrackedModelManager):
    """Inject a public execution transition that always raises."""

    def __init__(self) -> None:
        super().__init__()
        self.failure = _ProviderBoundaryExit("persistent terminal failure")
        self.fail_calls = 0

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        response = await super().__call__(call)
        execution = call.context.execution
        assert execution is not None

        async def persistent_fail(
            *,
            expected_revision: int | None = None,
        ) -> bool:
            del expected_revision
            self.fail_calls += 1
            raise self.failure

        execution.fail = persistent_fail  # type: ignore[method-assign]
        return response


class _AfterProviderFailureEvents(EventManager):
    """Raise one selected exit after the provider has returned."""

    def __init__(
        self,
        failure: BaseException,
        *,
        event_type: EventType = EventType.MODEL_EXECUTE_AFTER,
    ) -> None:
        super().__init__(mode=EventManagerMode.TEST)
        self.failure = failure
        self.event_type = event_type

    async def trigger(self, event: AgentEvent) -> None:
        """Raise the configured primary exit at its selected boundary."""
        if event.type is self.event_type:
            raise self.failure
        await super().trigger(event)


def _operation() -> AgentOperation:
    """Return one deterministic operation for the real orchestrator path."""
    environment = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id=_HarnessEngine.model_id,
            params={},
        ),
        settings=TransformerEngineSettings(),
    )
    return AgentOperation(
        specification=Specification(instructions="answer exactly"),
        environment=environment,
    )


async def _new_execution() -> AgentExecution:
    """Return one real running execution for cleanup ownership probes."""
    return await create_agent_execution(
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://cleanup-test",
            agent_definition_revision="definition-r1",
            operation_id="cleanup-operation",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tool-r1",
            capability_revision="capability-r1",
        ),
        agent_id=AgentId("cleanup-agent"),
        principal=PrincipalScope(),
        initial_messages=(
            Message(role=MessageRole.USER, content="cleanup boundary"),
        ),
    )


class _Harness:
    """Own one real EngineAgent and Orchestrator integration boundary."""

    def __init__(
        self,
        *,
        events: EventManager | None = None,
        manager: _ModelManager | None = None,
        support: ProviderCapabilitySupport | None = None,
        agent_type: type[_Agent] = _Agent,
    ) -> None:
        self.logger = getLogger()
        self.events = events or EventManager(mode=EventManagerMode.TEST)
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
        self.manager = manager or _ModelManager()
        self.engine = _HarnessEngine(support or ProviderCapabilitySupport())
        self.operation = _operation()
        engine_uri = self.operation.environment.engine_uri
        self.agent = agent_type(
            cast(Any, self.engine),
            self.memory,
            self.tool,
            self.events,
            cast(ModelManager, self.manager),
            engine_uri,
        )
        self.orchestrator = Orchestrator(
            self.logger,
            cast(ModelManager, self.manager),
            self.memory,
            self.tool,
            self.events,
            self.operation,
            exit_memory=False,
        )
        environment_hash = dumps(asdict(self.operation.environment))
        self.orchestrator._engine_agents[environment_hash] = self.agent
        self.exited = False

    async def exit(self) -> None:
        """Exit the orchestrator exactly once."""
        if self.exited:
            return
        await self.orchestrator.__aexit__(None, None, None)
        self.exited = True

    async def close(self) -> None:
        """Close test events when the orchestrator was not exited."""
        if not self.exited:
            await self.events.aclose()

    def recent(self) -> tuple[tuple[MessageRole, str], ...]:
        """Return exact role/content pairs from recent memory."""
        return tuple(
            (item.message.role, cast(str, item.message.content))
            for item in self.memory.recent_messages or ()
        )


async def _sync_server_response(
    orchestrator: Orchestrator,
    response: "StreamResponse",
) -> None:
    """Synchronize the exact response accepted by a server stream helper."""
    await orchestrator.sync_messages(response)


class _RecordingGlobalBroker:
    """Record valid facade delegation and reject unexpected cancellation."""

    def __init__(self) -> None:
        self.requests: list[InteractionBrokerRequest] = []
        self.cancellations: list[TerminalizeInteractionScopeCommand] = []
        self.registrations: list[RegisterInteractionBranchCommand] = []
        self.events: list[str] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.events.append("request")
        self.requests.append(request)
        request_number = len(self.requests)
        policy = InteractionPolicy()
        created = create_input_request(
            request_id=InputRequestId(f"ownership-request-{request_number}"),
            continuation_id=ContinuationId(
                f"ownership-continuation-{request_number}"
            ),
            origin=request.origin,
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=_NOW,
            continuation_ttl_seconds=request.continuation_ttl_seconds,
            advisory_wait_seconds=request.advisory_wait_seconds,
        )
        applied = apply_create_interaction(
            CreateInteractionCommand(actor=request.actor, request=created),
            policy,
        )
        return InteractionRequestResult(
            create_result=applied,
            delivery=InteractionDelivery(
                correlation=applied.record.correlation,
                record=applied.record,
                handler_attempts=0,
            ),
        )

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> NoReturn:
        self.cancellations.append(command)
        raise AssertionError("unscoped cancellation reached the global broker")

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> InteractionBrokerResult:
        self.events.append("register")
        self.registrations.append(command)
        return InteractionBrokerResult(
            store_result=InteractionBranchRegistrationApplied(
                command=command,
                record=InteractionBranchRecord(
                    registration=command.registration,
                    store_revision=InteractionStoreRevision(
                        len(self.registrations)
                    ),
                ),
            )
        )


class _DetachedHandler:
    """Return a detached outcome if a valid request is ever presented."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Detach without fabricating an interaction result."""
        assert isinstance(context, InputHandlerContext)
        return InputHandlerDetached()


class _ProviderBoundaryExit(BaseException):
    """Represent a non-cancellation exit outside ``Exception``."""


def _runtime(
    broker: _RecordingGlobalBroker,
) -> AttachedInteractionRuntime:
    """Return one attached runtime with explicit branch ancestry."""
    return AttachedInteractionRuntime(
        broker=cast(InteractionBroker, broker),
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("ownership-user"))
        ),
        handler=_DetachedHandler(),
        task_id=TaskId("ownership-task"),
        branch_id=BranchId("ownership-branch"),
        parent_branch_id=BranchId("ownership-parent"),
    )


def _interaction_request(
    runtime: AttachedInteractionRuntime,
    origin: ExecutionOrigin,
) -> InteractionBrokerRequest:
    """Return one valid request carrying a caller-selected origin."""
    return InteractionBrokerRequest(
        actor=runtime.actor,
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Need one scoped decision.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("continue"),
                prompt="Continue?",
                required=True,
            ),
        ),
    )


class ResponseOwnershipAdversarialTest(IsolatedAsyncioTestCase):
    """Pin exact response ownership, retry, and broker-scope behavior."""

    async def _assert_post_provider_exit_converges(
        self,
        failure: BaseException,
        expected_status: AgentExecutionStatus,
        *,
        event_type: EventType = EventType.MODEL_EXECUTE_AFTER,
    ) -> tuple[
        type[BaseException],
        bool,
        tuple[str, ...],
        AgentExecutionStatus,
        bool,
        bool,
        EventType,
    ]:
        manager = _TrackedModelManager()
        events = _AfterProviderFailureEvents(
            failure,
            event_type=event_type,
        )
        harness = _Harness(events=events, manager=manager)
        try:
            with self.assertRaises(type(failure)) as raised:
                await harness.orchestrator("provider-boundary")

            self.assertIs(raised.exception, failure)
            self.assertEqual(len(manager.responses), 1)
            self.assertEqual(
                manager.responses[0].cleanup_calls,
                ["cancel", "close"],
            )
            execution = manager.calls[0].context.execution
            assert execution is not None
            self.assertIs(execution.status, expected_status)
            self.assertEqual(harness.orchestrator._pending_responses, {})
            self.assertIsNone(harness.agent.output)
            return (
                type(raised.exception),
                raised.exception is failure,
                tuple(manager.responses[0].cleanup_calls),
                execution.status,
                not harness.orchestrator._pending_responses,
                harness.agent.output is None,
                events.event_type,
            )
        finally:
            await harness.close()

    async def test_runtime_error_after_provider_fails_and_closes(self) -> None:
        failure = RuntimeError("after-provider runtime failure")
        summary = await self._assert_post_provider_exit_converges(
            failure,
            AgentExecutionStatus.ERRORED,
        )
        self.assertEqual(
            summary,
            (
                RuntimeError,
                True,
                ("cancel", "close"),
                AgentExecutionStatus.ERRORED,
                True,
                True,
                EventType.MODEL_EXECUTE_AFTER,
            ),
        )

    async def test_cancellation_after_provider_cancels_and_closes(
        self,
    ) -> None:
        failure = CancelledError("after-provider cancellation")
        summary = await self._assert_post_provider_exit_converges(
            failure,
            AgentExecutionStatus.CANCELLED,
        )
        self.assertEqual(
            summary,
            (
                CancelledError,
                True,
                ("cancel", "close"),
                AgentExecutionStatus.CANCELLED,
                True,
                True,
                EventType.MODEL_EXECUTE_AFTER,
            ),
        )

    async def test_base_exit_after_provider_fails_and_closes(self) -> None:
        failure = _ProviderBoundaryExit("after-provider base exit")
        summary = await self._assert_post_provider_exit_converges(
            failure,
            AgentExecutionStatus.ERRORED,
        )
        self.assertEqual(
            summary,
            (
                _ProviderBoundaryExit,
                True,
                ("cancel", "close"),
                AgentExecutionStatus.ERRORED,
                True,
                True,
                EventType.MODEL_EXECUTE_AFTER,
            ),
        )

    async def test_outer_agent_event_keeps_provider_owned(self) -> None:
        failure = RuntimeError("outer agent handoff failure")
        summary = await self._assert_post_provider_exit_converges(
            failure,
            AgentExecutionStatus.ERRORED,
            event_type=EventType.ENGINE_AGENT_CALL_AFTER,
        )
        self.assertEqual(
            summary,
            (
                RuntimeError,
                True,
                ("cancel", "close"),
                AgentExecutionStatus.ERRORED,
                True,
                True,
                EventType.ENGINE_AGENT_CALL_AFTER,
            ),
        )

    async def test_persistent_public_fail_preserves_primary_and_falls_back(
        self,
    ) -> None:
        manager = _PersistentFailModelManager()
        primary = RuntimeError("after-provider primary failure")
        harness = _Harness(
            events=_AfterProviderFailureEvents(primary),
            manager=manager,
        )
        try:
            with self.assertRaises(RuntimeError) as raised:
                await harness.orchestrator("persistent-fail")

            self.assertIs(raised.exception, primary)
            self.assertEqual(manager.fail_calls, 1)
            self.assertEqual(
                primary.__notes__,
                [
                    "post-provider cleanup failure: _ProviderBoundaryExit: "
                    "persistent terminal failure"
                ],
            )
            self.assertEqual(
                manager.responses[0].cleanup_calls,
                ["cancel", "close"],
            )
            execution = manager.calls[0].context.execution
            assert execution is not None
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertIs(
                execution.ledger[-1].kind,
                ExecutionLedgerEntryKind.ERRORED,
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
            self.assertEqual(
                getattr(
                    harness.orchestrator, "_pending_provider_cleanups", {}
                ),
                {},
            )
            self.assertIsNone(harness.agent.output)
        finally:
            await harness.close()

    async def test_cleanup_failures_are_notes_on_the_primary_exit(
        self,
    ) -> None:
        harness = _Harness()
        response = MagicMock(spec=TextGenerationResponse)
        response.cancel = AsyncMock(
            side_effect=RuntimeError("provider cancel failed")
        )
        response.aclose = AsyncMock(
            side_effect=_ProviderBoundaryExit("provider close failed")
        )
        response.cleanup_complete = False
        execution = MagicMock()
        execution.settle_provider_exit = AsyncMock(
            return_value=(
                KeyboardInterrupt("execution terminalization failed"),
            )
        )
        primary = RuntimeError("primary exit")
        try:
            await harness.agent._settle_unhanded_provider_response(
                response,
                execution,
                primary,
                cancelled=False,
            )

            self.assertEqual(
                primary.__notes__,
                [
                    (
                        "post-provider cleanup failure: RuntimeError: "
                        "provider cancel failed"
                    ),
                    (
                        "post-provider cleanup failure: _ProviderBoundaryExit:"
                        " provider close failed"
                    ),
                    (
                        "post-provider cleanup failure: KeyboardInterrupt: "
                        "execution terminalization failed"
                    ),
                ],
            )
        finally:
            await harness.close()

    async def test_engine_cleanup_retries_two_close_failures_to_third_success(
        self,
    ) -> None:
        harness = _Harness()
        response = _FailingCloseResponse()
        execution = await _new_execution()
        primary = RuntimeError("engine handoff failed")

        async def fail_boundary() -> NoReturn:
            try:
                raise primary
            except BaseException as error:
                await harness.agent._settle_unhanded_provider_response(
                    response,
                    execution,
                    error,
                    cancelled=False,
                )
                raise

        try:
            with self.assertRaises(RuntimeError) as raised:
                await fail_boundary()

            self.assertIs(raised.exception, primary)
            self.assertEqual(response.close_attempts, 2)
            self.assertFalse(response.cleanup_complete)
            self.assertEqual(
                tuple(primary.__notes__),
                (
                    (
                        "post-provider cleanup failure: RuntimeError: "
                        "provider close attempt 1 failed"
                    ),
                    (
                        "post-provider cleanup failure: RuntimeError: "
                        "provider close attempt 2 failed"
                    ),
                ),
            )
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertIs(
                execution.ledger[-1].kind,
                ExecutionLedgerEntryKind.ERRORED,
            )
            self.assertIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )

            failures = await harness.agent.drain_pending_provider_cleanups(
                execution
            )

            self.assertEqual(failures, ())
            self.assertEqual(response.close_attempts, 3)
            self.assertEqual(response.events, ["cancel", "close", "cancel"])
            self.assertTrue(response.cleanup_complete)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            await harness.close()

    async def test_nonreturning_cancel_is_owned_until_late_failure_drains(
        self,
    ) -> None:
        harness = _Harness()
        harness.agent._PROVIDER_CLEANUP_TIMEOUT_SECONDS = 0.01
        response = _NeverReturningCancelResponse()
        execution = await _new_execution()
        primary = RuntimeError("engine callback failed")

        async def fail_boundary() -> NoReturn:
            try:
                raise primary
            except BaseException as error:
                await harness.agent._settle_unhanded_provider_response(
                    response,
                    execution,
                    error,
                    cancelled=False,
                )
                raise

        try:
            with self.assertRaises(RuntimeError) as raised:
                await wait_for(fail_boundary(), timeout=0.2)

            self.assertIs(raised.exception, primary)
            await wait_for(response.cancel_started.wait(), timeout=0.1)
            await wait_for(response.close_completed.wait(), timeout=0.1)
            self.assertTrue(response.cleanup_complete)
            self.assertEqual(response.close_calls, 1)
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertEqual(len(primary.__notes__), 1)
            self.assertIn("provider cancel exceeded", primary.__notes__[0])
            owner = harness.agent._pending_provider_cleanups[id(response)]
            cancel_task = owner._cancel_task
            assert cancel_task is not None
            self.assertFalse(cancel_task.done())

            response.cancel_release.set()
            failures = await harness.agent.drain_pending_provider_cleanups(
                execution
            )

            self.assertEqual(failures, (response.cancel_failure,))
            self.assertTrue(cancel_task.done())
            self.assertIs(cancel_task.exception(), response.cancel_failure)
            self.assertIsNone(owner._cancel_task)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            response.cancel_release.set()
            await harness.close()

    async def test_engine_cleanup_drain_has_no_owner_head_of_line_blocking(
        self,
    ) -> None:
        harness = _Harness()
        harness.agent._PROVIDER_CLEANUP_TIMEOUT_SECONDS = 0.05
        slow_response = _NeverReturningCancelResponse()
        fast_response = _ImmediateCleanupResponse()
        slow_execution = await _new_execution()
        fast_execution = await _new_execution()
        slow_owner = harness.agent._retain_provider_cleanup(
            slow_response,
            slow_execution,
        )
        fast_owner = harness.agent._retain_provider_cleanup(
            fast_response,
            fast_execution,
        )
        slow_owner.require_cleanup(cancelled=False)
        fast_owner.require_cleanup(cancelled=False)
        drain = create_task(harness.agent.drain_pending_provider_cleanups())
        try:
            await wait_for(fast_response.closed.wait(), timeout=0.02)
            for _ in range(10):
                if (
                    id(fast_response)
                    not in harness.agent._pending_provider_cleanups
                ):
                    break
                await sleep(0)

            self.assertNotIn(
                id(fast_response),
                harness.agent._pending_provider_cleanups,
            )
            self.assertIn(
                id(slow_response),
                harness.agent._pending_provider_cleanups,
            )
            self.assertFalse(drain.done())
            await wait_for(slow_response.close_completed.wait(), timeout=0.1)
            first_failures = await wait_for(drain, timeout=0.1)
            self.assertEqual(len(first_failures), 1)
            self.assertIsInstance(first_failures[0], TimeoutError)
            self.assertIs(
                slow_execution.status,
                AgentExecutionStatus.ERRORED,
            )
            self.assertIs(
                fast_execution.status,
                AgentExecutionStatus.ERRORED,
            )

            slow_response.cancel_release.set()
            retry_failures = (
                await harness.agent.drain_pending_provider_cleanups()
            )

            self.assertEqual(
                retry_failures,
                (slow_response.cancel_failure,),
            )
            self.assertEqual(harness.agent._pending_provider_cleanups, {})
        finally:
            slow_response.cancel_release.set()
            if not drain.done():
                drain.cancel()
                await gather(drain, return_exceptions=True)
            await harness.close()

    async def test_engine_drain_skips_healthy_handoff_until_abandoned(
        self,
    ) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        harness.agent._retain_provider_cleanup(response, execution)
        try:
            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(),
                (),
            )
            self.assertFalse(response.cleanup_complete)
            self.assertIs(execution.status, AgentExecutionStatus.RUNNING)
            self.assertIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )

            failures = await harness.agent.drain_pending_provider_cleanups(
                execution,
                abandon_unclaimed=True,
            )

            self.assertEqual(failures, ())
            self.assertTrue(response.cleanup_complete)
            self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
            self.assertIs(
                execution.ledger[-1].kind,
                ExecutionLedgerEntryKind.CANCELLED,
            )
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            await harness.close()

    async def test_engine_cleanup_first_owner_wins_mixed_intent(self) -> None:
        for first_cancelled, expected_status in (
            (False, AgentExecutionStatus.ERRORED),
            (True, AgentExecutionStatus.CANCELLED),
        ):
            with self.subTest(first_cancelled=first_cancelled):
                harness = _Harness()
                response = _ImmediateCleanupResponse()
                execution = await _new_execution()
                owner = harness.agent._retain_provider_cleanup(
                    response,
                    execution,
                )
                owner.require_cleanup(cancelled=first_cancelled)
                owner.require_cleanup(cancelled=not first_cancelled)
                try:
                    first_drain, second_drain = await gather(
                        harness.agent.drain_pending_provider_cleanups(
                            execution
                        ),
                        harness.agent.drain_pending_provider_cleanups(
                            execution
                        ),
                    )

                    self.assertEqual(first_drain, ())
                    self.assertEqual(second_drain, ())
                    self.assertIs(execution.status, expected_status)
                    self.assertEqual(response.cancel_calls, 1)
                    self.assertEqual(response.close_calls, 1)
                    self.assertNotIn(
                        id(response),
                        harness.agent._pending_provider_cleanups,
                    )
                finally:
                    await harness.close()

    async def test_engine_close_timeout_remains_owned_for_late_success(
        self,
    ) -> None:
        harness = _Harness()
        harness.agent._PROVIDER_CLEANUP_TIMEOUT_SECONDS = 0.01
        response = _NeverReturningCloseResponse()
        execution = await _new_execution()
        owner = harness.agent._retain_provider_cleanup(response, execution)
        owner.require_cleanup(cancelled=False)
        try:
            first_failures = (
                await harness.agent.drain_pending_provider_cleanups(execution)
            )

            self.assertIs(first_failures[0], response.cancel_failure)
            self.assertIsInstance(first_failures[1], TimeoutError)
            self.assertIn("provider close exceeded", str(first_failures[1]))
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
            close_task = owner._close_task
            assert close_task is not None
            self.assertFalse(close_task.done())

            response.close_release.set()
            second_failures = (
                await harness.agent.drain_pending_provider_cleanups(execution)
            )

            self.assertEqual(second_failures, ())
            self.assertTrue(close_task.done())
            self.assertTrue(response.cleanup_complete)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            response.close_release.set()
            await harness.close()

    async def test_engine_close_must_confirm_provider_closure(self) -> None:
        harness = _Harness()
        response = _NoClosureResponse()
        execution = await _new_execution()
        owner = harness.agent._retain_provider_cleanup(response, execution)
        owner.require_cleanup(cancelled=False)
        try:
            first_failures = (
                await harness.agent.drain_pending_provider_cleanups(execution)
            )

            self.assertIs(first_failures[0], response.cancel_failure)
            self.assertIsInstance(first_failures[1], RuntimeError)
            self.assertEqual(
                str(first_failures[1]),
                "provider close completed without closure",
            )
            self.assertIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )

            response._output_closed = True
            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(execution),
                (),
            )
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            response._output_closed = True
            await harness.close()

    async def test_engine_settlement_timeout_retains_late_failure_for_retry(
        self,
    ) -> None:
        harness = _Harness()
        harness.agent._PROVIDER_CLEANUP_TIMEOUT_SECONDS = 0.01
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        settlement_started = Event()
        settlement_release = Event()
        settlement_failure = RuntimeError("late execution settlement failure")
        original_settlement = execution.settle_provider_exit

        async def blocked_settlement(
            *,
            cancelled: bool,
        ) -> NoReturn:
            del cancelled
            settlement_started.set()
            await settlement_release.wait()
            raise settlement_failure

        execution.settle_provider_exit = blocked_settlement  # type: ignore[method-assign]
        owner = harness.agent._retain_provider_cleanup(response, execution)
        owner.require_cleanup(cancelled=False)
        try:
            first_failures = (
                await harness.agent.drain_pending_provider_cleanups(execution)
            )

            await settlement_started.wait()
            self.assertEqual(len(first_failures), 1)
            self.assertIsInstance(first_failures[0], TimeoutError)
            self.assertIn(
                "provider execution settlement exceeded",
                str(first_failures[0]),
            )
            self.assertIs(execution.status, AgentExecutionStatus.RUNNING)
            self.assertIsNotNone(owner._settlement_task)

            settlement_release.set()
            second_failures = (
                await harness.agent.drain_pending_provider_cleanups(execution)
            )

            self.assertEqual(second_failures, (settlement_failure,))
            self.assertIs(execution.status, AgentExecutionStatus.RUNNING)
            self.assertIsNone(owner._settlement_task)
            execution.settle_provider_exit = original_settlement  # type: ignore[method-assign]

            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(execution),
                (),
            )
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            settlement_release.set()
            execution.settle_provider_exit = original_settlement  # type: ignore[method-assign]
            await harness.close()

    async def test_engine_owner_handles_preclosed_executionless_response(
        self,
    ) -> None:
        response = _ImmediateCleanupResponse()
        owner = engine_module._EngineProviderCleanup(
            response,
            None,
            timeout_seconds=0.01,
        )

        self.assertEqual(await owner.converge(), ())
        response._output_closed = True
        owner.require_cleanup(cancelled=False)
        self.assertEqual(await owner.converge(), ())
        self.assertTrue(owner.cleanup_complete)

    async def test_engine_owner_handles_timeout_completion_races(self) -> None:
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        owner = engine_module._EngineProviderCleanup(
            response,
            execution,
            timeout_seconds=0.01,
        )

        cancel_task = create_task(sleep(0))
        await cancel_task
        owner._cancel_task = cast(Any, cancel_task)
        cancel_timeout = TimeoutError("cancel completion race")
        with patch.object(
            engine_module,
            "wait_for",
            AsyncMock(side_effect=cancel_timeout),
        ):
            self.assertEqual(await owner._await_cancel(), (cancel_timeout,))
        self.assertIsNone(owner._cancel_task)

        cancel_failure = RuntimeError("cancel failed at timeout")

        async def fail_cancel() -> NoReturn:
            raise cancel_failure

        failed_cancel_task = create_task(fail_cancel())
        await gather(failed_cancel_task, return_exceptions=True)
        owner._cancel_task = failed_cancel_task
        second_cancel_timeout = TimeoutError("cancel failure race")
        with patch.object(
            engine_module,
            "wait_for",
            AsyncMock(side_effect=second_cancel_timeout),
        ):
            self.assertEqual(
                await owner._await_cancel(),
                (second_cancel_timeout, cancel_failure),
            )
        self.assertIsNone(owner._cancel_task)

        response._output_closed = True
        close_failure = RuntimeError("close failed at timeout")

        async def fail_close() -> NoReturn:
            raise close_failure

        close_task = create_task(fail_close())
        await gather(close_task, return_exceptions=True)
        owner._close_task = cast(Any, close_task)
        close_timeout = TimeoutError("close completion race")
        with patch.object(
            engine_module,
            "wait_for",
            AsyncMock(side_effect=close_timeout),
        ):
            self.assertEqual(
                await owner._await_close(),
                (close_timeout, close_failure),
            )
        self.assertIsNone(owner._close_task)

        successful_close_task = create_task(sleep(0))
        await successful_close_task
        owner._close_task = cast(Any, successful_close_task)
        second_close_timeout = TimeoutError("close success race")
        with patch.object(
            engine_module,
            "wait_for",
            AsyncMock(side_effect=second_close_timeout),
        ):
            self.assertEqual(
                await owner._await_close(),
                (second_close_timeout,),
            )
        self.assertIsNone(owner._close_task)

        settlement_failure = RuntimeError("settlement failed at timeout")

        async def settled() -> tuple[BaseException, ...]:
            return (settlement_failure,)

        settlement_task = create_task(settled())
        await settlement_task
        owner._settlement_task = settlement_task
        settlement_timeout = TimeoutError("settlement completion race")
        with patch.object(
            engine_module,
            "wait_for",
            AsyncMock(side_effect=settlement_timeout),
        ):
            self.assertEqual(
                await owner._await_settlement(),
                (settlement_timeout, settlement_failure),
            )
        self.assertIsNone(owner._settlement_task)

        raised_settlement_failure = RuntimeError(
            "settlement task failed at timeout"
        )

        async def fail_settlement() -> NoReturn:
            raise raised_settlement_failure

        failed_settlement_task = create_task(fail_settlement())
        await gather(failed_settlement_task, return_exceptions=True)
        owner._settlement_task = cast(Any, failed_settlement_task)
        raised_settlement_timeout = TimeoutError("settlement failure race")
        with patch.object(
            engine_module,
            "wait_for",
            AsyncMock(side_effect=raised_settlement_timeout),
        ):
            self.assertEqual(
                await owner._await_settlement(),
                (raised_settlement_timeout, raised_settlement_failure),
            )
        self.assertIsNone(owner._settlement_task)

        cancelled_task = create_task(sleep(1))
        cancelled_task.cancel()
        await gather(cancelled_task, return_exceptions=True)
        owner._observe_task_completion(cancelled_task)

    async def test_engine_nonprovider_settlement_preserves_primary(
        self,
    ) -> None:
        harness = _Harness()
        primary = RuntimeError("engine failed before provider")
        execution = await _new_execution()
        settlement_failure = RuntimeError("execution settlement failed")
        original_settlement = execution.settle_provider_exit

        async def failing_settlement(
            *,
            cancelled: bool,
        ) -> NoReturn:
            del cancelled
            raise settlement_failure

        try:
            await harness.agent._settle_failed_output(
                "no provider",
                None,
                primary,
                cancelled=False,
            )
            execution.settle_provider_exit = failing_settlement  # type: ignore[method-assign]
            await harness.agent._settle_failed_output(
                "no provider",
                execution,
                primary,
                cancelled=False,
            )

            self.assertEqual(
                primary.__notes__,
                [
                    "post-provider cleanup failure: RuntimeError: "
                    "execution settlement failed"
                ],
            )
        finally:
            execution.settle_provider_exit = original_settlement  # type: ignore[method-assign]
            await execution.settle_provider_exit(cancelled=False)
            await harness.close()

    def test_engine_cleanup_notes_exclude_primary_and_duplicates(self) -> None:
        primary = RuntimeError("primary exit")
        secondary = RuntimeError("secondary cleanup failure")

        EngineAgent._attach_cleanup_failures(
            primary,
            (primary, secondary, secondary),
        )

        self.assertEqual(
            primary.__notes__,
            [
                "post-provider cleanup failure: RuntimeError: "
                "secondary cleanup failure"
            ],
        )

    async def test_engine_cleanup_setup_failure_preserves_primary(
        self,
    ) -> None:
        harness = _Harness()
        response = _CallbackRegistrationFailureResponse()
        execution = await _new_execution()
        try:
            with self.assertRaises(RuntimeError) as raised:
                try:
                    harness.agent._retain_provider_cleanup(
                        response,
                        execution,
                    )
                except BaseException as primary:
                    await harness.agent._settle_failed_output(
                        response,
                        execution,
                        primary,
                        cancelled=False,
                    )
                    raise

            self.assertIs(raised.exception, response.callback_failure)
            self.assertEqual(
                getattr(raised.exception, "__notes__", []),
                [],
            )
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertEqual(response.cancel_calls, 1)
            self.assertEqual(response.close_calls, 1)
            self.assertTrue(response.cleanup_complete)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            await harness.close()

    async def test_unexpected_cleanup_failure_cannot_mask_event_exit(
        self,
    ) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        primary = RuntimeError("engine event failed")
        cleanup_failure = RuntimeError("cleanup setup failed")
        harness.agent._retain_provider_cleanup(response, execution)
        try:
            with patch.object(
                harness.agent,
                "_settle_unhanded_provider_response",
                AsyncMock(side_effect=cleanup_failure),
            ):
                await harness.agent._settle_failed_output(
                    response,
                    execution,
                    primary,
                    cancelled=False,
                )

            self.assertEqual(
                primary.__notes__,
                [
                    "post-provider cleanup failure: RuntimeError: "
                    "cleanup setup failed"
                ],
            )
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertEqual(response.cancel_calls, 1)
            self.assertEqual(response.close_calls, 1)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            await harness.close()

    async def test_recovery_failure_is_only_a_secondary_note(self) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        primary = RuntimeError("engine event failed")
        cleanup_failure = RuntimeError("cleanup setup failed")
        recovery_failure = RuntimeError("cleanup recovery failed")
        owner = harness.agent._retain_provider_cleanup(response, execution)
        try:
            with (
                patch.object(
                    harness.agent,
                    "_settle_unhanded_provider_response",
                    AsyncMock(side_effect=cleanup_failure),
                ),
                patch.object(
                    harness.agent,
                    "_recover_failed_provider_cleanup",
                    AsyncMock(side_effect=recovery_failure),
                ),
            ):
                await harness.agent._settle_failed_output(
                    response,
                    execution,
                    primary,
                    cancelled=False,
                )

            self.assertEqual(
                primary.__notes__,
                [
                    (
                        "post-provider cleanup failure: RuntimeError: "
                        "cleanup setup failed"
                    ),
                    (
                        "post-provider cleanup failure: RuntimeError: "
                        "cleanup recovery failed"
                    ),
                ],
            )
            self.assertIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )

            owner.require_cleanup(cancelled=False)
            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(execution),
                (),
            )
        finally:
            await harness.close()

    async def test_recovery_creates_and_releases_a_durable_owner(self) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        try:
            self.assertEqual(
                await harness.agent._recover_failed_provider_cleanup(
                    response,
                    execution,
                    cancelled=False,
                ),
                (),
            )
            self.assertTrue(response.cleanup_complete)
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertNotIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
        finally:
            await harness.close()

    async def test_recovery_captures_owner_creation_and_convergence_failures(
        self,
    ) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        creation_failure = RuntimeError("owner construction failed")
        convergence_failure = RuntimeError("owner convergence failed")
        try:
            with patch.object(
                engine_module,
                "_EngineProviderCleanup",
                side_effect=creation_failure,
            ):
                self.assertEqual(
                    await harness.agent._recover_failed_provider_cleanup(
                        response,
                        execution,
                        cancelled=False,
                    ),
                    (creation_failure,),
                )

            owner = harness.agent._retain_provider_cleanup(
                response,
                execution,
            )
            with patch.object(
                owner,
                "converge",
                AsyncMock(side_effect=convergence_failure),
            ):
                self.assertEqual(
                    await harness.agent._recover_failed_provider_cleanup(
                        response,
                        execution,
                        cancelled=False,
                    ),
                    (convergence_failure,),
                )

            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(execution),
                (),
            )
        finally:
            await harness.close()

    async def test_unhanded_cleanup_captures_unexpected_convergence_exit(
        self,
    ) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        primary = RuntimeError("engine event failed")
        convergence_failure = RuntimeError("owner convergence escaped")
        owner = harness.agent._retain_provider_cleanup(response, execution)
        try:
            with patch.object(
                owner,
                "converge",
                AsyncMock(side_effect=convergence_failure),
            ):
                await harness.agent._settle_unhanded_provider_response(
                    response,
                    execution,
                    primary,
                    cancelled=False,
                )

            self.assertEqual(
                primary.__notes__,
                [
                    "post-provider cleanup failure: RuntimeError: "
                    "owner convergence escaped"
                ],
            )
            self.assertIn(
                id(response),
                harness.agent._pending_provider_cleanups,
            )
            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(execution),
                (),
            )
        finally:
            await harness.close()

    async def test_exact_execution_drain_skips_other_owner(self) -> None:
        harness = _Harness()
        first_response = _ImmediateCleanupResponse()
        second_response = _ImmediateCleanupResponse()
        first_execution = await _new_execution()
        second_execution = await _new_execution()
        first_owner = harness.agent._retain_provider_cleanup(
            first_response,
            first_execution,
        )
        second_owner = harness.agent._retain_provider_cleanup(
            second_response,
            second_execution,
        )
        first_owner.require_cleanup(cancelled=False)
        second_owner.require_cleanup(cancelled=False)
        try:
            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(
                    first_execution
                ),
                (),
            )
            self.assertNotIn(
                id(first_response),
                harness.agent._pending_provider_cleanups,
            )
            self.assertIn(
                id(second_response),
                harness.agent._pending_provider_cleanups,
            )
            self.assertIs(
                second_execution.status,
                AgentExecutionStatus.RUNNING,
            )

            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(
                    second_execution
                ),
                (),
            )
        finally:
            await harness.close()

    async def test_drain_aggregates_unexpected_owner_exit(self) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        owner = harness.agent._retain_provider_cleanup(response, execution)
        owner.require_cleanup(cancelled=False)
        drain_failure = RuntimeError("owner drain escaped")
        try:
            with patch.object(
                harness.agent,
                "_drain_provider_cleanup",
                AsyncMock(side_effect=drain_failure),
            ):
                self.assertEqual(
                    await harness.agent.drain_pending_provider_cleanups(
                        execution
                    ),
                    (drain_failure,),
                )

            self.assertEqual(
                await harness.agent.drain_pending_provider_cleanups(execution),
                (),
            )
        finally:
            await harness.close()

    async def test_retain_rejects_provider_ownership_collision(self) -> None:
        harness = _Harness()
        response = _ImmediateCleanupResponse()
        other_response = _ImmediateCleanupResponse()
        execution = await _new_execution()
        other_execution = await _new_execution()
        owner = harness.agent._retain_provider_cleanup(
            other_response,
            other_execution,
        )
        harness.agent._pending_provider_cleanups[id(response)] = owner
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "provider cleanup ownership collision",
            ):
                harness.agent._retain_provider_cleanup(response, execution)
            recovery_failures = (
                await harness.agent._recover_failed_provider_cleanup(
                    response,
                    execution,
                    cancelled=False,
                )
            )
            self.assertEqual(len(recovery_failures), 1)
            self.assertIsInstance(recovery_failures[0], RuntimeError)
            self.assertEqual(
                str(recovery_failures[0]),
                "provider cleanup ownership collision",
            )
        finally:
            harness.agent._pending_provider_cleanups.pop(id(response), None)
            owner.require_cleanup(cancelled=False)
            await harness.agent.drain_pending_provider_cleanups(
                other_execution
            )
            await execution.settle_provider_exit(cancelled=False)
            await harness.close()

    async def test_cancelled_cleanup_task_is_preserved_as_a_note(self) -> None:
        harness = _Harness()
        response = MagicMock(spec=TextGenerationResponse)
        primary = RuntimeError("primary exit")

        def cancelled_task(coroutine: Any) -> Any:
            task = create_task(coroutine)
            task.cancel()
            return task

        try:
            with patch.object(
                engine_module,
                "create_task",
                side_effect=cancelled_task,
            ):
                await harness.agent._settle_unhanded_provider_response(
                    response,
                    None,
                    primary,
                    cancelled=False,
                )
            self.assertEqual(
                primary.__notes__,
                ["post-provider cleanup failure: CancelledError: "],
            )
        finally:
            await harness.close()

    async def test_parent_exit_syncs_two_concurrent_owned_responses(
        self,
    ) -> None:
        manager = _ConcurrentModelManager()
        harness = _Harness(manager=manager)

        async def child(prompt: str) -> tuple[OrchestratorResponse, str]:
            response = await harness.orchestrator(prompt)
            return response, await response.to_str()

        alpha_task = create_task(child("alpha"))
        beta_task = create_task(child("beta"))
        try:
            await wait_for(manager.both_started.wait(), timeout=1)
            manager.releases["alpha"].set()
            alpha_response, alpha_output = await wait_for(
                alpha_task, timeout=1
            )
            manager.releases["beta"].set()
            beta_response, beta_output = await wait_for(beta_task, timeout=1)

            self.assertEqual(
                (alpha_output, beta_output),
                (
                    "answer:alpha",
                    "answer:beta",
                ),
            )
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (alpha_response, beta_response),
            )

            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "alpha"),
                    (MessageRole.ASSISTANT, "answer:alpha"),
                    (MessageRole.USER, "beta"),
                    (MessageRole.ASSISTANT, "answer:beta"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            unfinished = []
            for task in (alpha_task, beta_task):
                if not task.done():
                    task.cancel()
                    unfinished.append(task)
            if unfinished:
                await gather(*unfinished, return_exceptions=True)
            await harness.close()

    async def test_failed_sink_stays_owned_and_retries_without_duplicates(
        self,
    ) -> None:
        harness = _Harness(agent_type=_FailingSyncAgent)
        try:
            response = await harness.orchestrator("retry")
            self.assertEqual(await response.to_str(), "answer:retry")

            with self.assertRaisesRegex(
                RuntimeError,
                "injected assistant memory failure",
            ):
                await harness.orchestrator.sync_messages(response)

            self.assertEqual(harness.recent(), ((MessageRole.USER, "retry"),))
            self.assertEqual(
                tuple(harness.orchestrator._pending_responses.values()),
                (response,),
            )

            await harness.exit()

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "retry"),
                    (MessageRole.ASSISTANT, "answer:retry"),
                ),
            )
            agent = cast(_FailingSyncAgent, harness.agent)
            self.assertEqual(
                agent.sync_attempts,
                [
                    (MessageRole.USER, "retry"),
                    (MessageRole.ASSISTANT, "answer:retry"),
                    (MessageRole.ASSISTANT, "answer:retry"),
                ],
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            await harness.close()

    async def test_server_boundary_syncs_its_exact_response(self) -> None:
        harness = _Harness()
        try:
            response = await harness.orchestrator("server")
            self.assertEqual(await response.to_str(), "answer:server")

            await _sync_server_response(
                harness.orchestrator,
                cast("StreamResponse", response),
            )

            self.assertEqual(
                harness.recent(),
                (
                    (MessageRole.USER, "server"),
                    (MessageRole.ASSISTANT, "answer:server"),
                ),
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            await harness.close()

    async def test_host_support_intersects_model_adapter_proof(self) -> None:
        broker = _RecordingGlobalBroker()
        runtime = _runtime(broker)

        async def catalog_for(
            support: ProviderCapabilitySupport,
        ) -> ModelCapabilityCatalog:
            harness = _Harness(support=support)
            try:
                response = await harness.orchestrator(
                    "capability",
                    interaction_runtime=runtime,
                )
                self.assertEqual(await response.to_str(), "answer:capability")
                capability = harness.manager.calls[-1].context.capability
                assert capability is not None
                await harness.orchestrator.sync_messages(response)
                return capability
            finally:
                await harness.close()

        incapable = await catalog_for(ProviderCapabilitySupport())
        proven = await catalog_for(
            ProviderCapabilitySupport(
                structured_invocation=True,
                stable_call_ids=True,
                correlated_results=True,
            )
        )

        self.assertFalse(incapable.support.structured_invocation)
        self.assertTrue(incapable.support.attached_resolution)
        self.assertIs(
            incapable.task_input_advertisement,
            TaskInputCapabilityAdvertisement.INCAPABLE,
        )
        self.assertNotIn(
            RESERVED_INPUT_CAPABILITY_NAME,
            {item.canonical_name for item in incapable.descriptors},
        )
        self.assertTrue(proven.support.structured_invocation)
        self.assertTrue(proven.support.attached_resolution)
        self.assertIs(
            proven.task_input_advertisement,
            TaskInputCapabilityAdvertisement.ATTACHED,
        )
        self.assertIn(
            RESERVED_INPUT_CAPABILITY_NAME,
            {item.canonical_name for item in proven.descriptors},
        )

    async def test_contexts_reject_broker_calls_outside_their_branch(
        self,
    ) -> None:
        raw_broker = _RecordingGlobalBroker()
        runtime = _runtime(raw_broker)
        support = ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
        )
        harness = _Harness(support=support)
        try:
            response = await harness.orchestrator(
                "scoped",
                interaction_runtime=runtime,
            )
            self.assertEqual(await response.to_str(), "answer:scoped")
            model_context = harness.manager.calls[-1].context
            tool_context = response._new_tool_context(response._input)
            execution = response.execution
            assert execution is not None
            origin = execution.origin
            self.assertEqual(model_context.execution_origin, origin)
            self.assertEqual(tool_context.execution_origin, origin)
            model_broker = model_context.interaction_broker
            tool_broker = tool_context.interaction_broker
            assert model_broker is not None and tool_broker is not None
            self.assertIsNot(model_broker, raw_broker)
            self.assertIsNot(tool_broker, raw_broker)

            wrong_origins = (
                replace(origin, run_id=RunId("wrong-run")),
                replace(origin, branch_id=BranchId("wrong-branch")),
                replace(
                    origin,
                    parent_branch_id=BranchId("wrong-parent"),
                ),
            )
            for broker in (model_broker, tool_broker):
                for wrong_origin in wrong_origins:
                    with self.assertRaises(ExecutionCorrelationError):
                        await broker.request(
                            _interaction_request(runtime, wrong_origin)
                        )

                for scope in (
                    InteractionExecutionScope(run_id=RunId("wrong-run")),
                    InteractionExecutionScope(
                        run_id=origin.run_id,
                        branch_id=BranchId("wrong-branch"),
                    ),
                ):
                    command = TerminalizeInteractionScopeCommand(
                        actor=runtime.actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    )
                    with self.assertRaises(ExecutionCorrelationError):
                        await broker.cancel_scope(command)

            self.assertEqual(raw_broker.requests, [])
            self.assertEqual(raw_broker.cancellations, [])

            valid_request = _interaction_request(runtime, origin)
            first_result = await model_broker.request(valid_request)
            replay_result = await tool_broker.request(valid_request)

            self.assertIsInstance(first_result, InteractionRequestResult)
            self.assertIsInstance(replay_result, InteractionRequestResult)
            self.assertEqual(
                raw_broker.events, ["register", "request", "request"]
            )
            self.assertEqual(
                raw_broker.registrations,
                [
                    RegisterInteractionBranchCommand(
                        actor=runtime.actor,
                        registration=InteractionBranchRegistration(
                            run_id=origin.run_id,
                            branch_id=origin.branch_id,
                            parent_branch_id=BranchId("ownership-parent"),
                            principal=origin.principal,
                        ),
                    )
                ],
            )
            self.assertEqual(
                raw_broker.requests, [valid_request, valid_request]
            )
            self.assertEqual(raw_broker.cancellations, [])
            await harness.orchestrator.sync_messages(response)
        finally:
            await harness.close()


class _LifetimeEngine(Engine):
    """Provide a load-free concrete Engine for lifetime isolation tests."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            model_id,
            EngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )

    def _load_model(self) -> object:
        return object()


class _TrackedResource:
    """Record which engine-owned async stack closes this resource."""

    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    async def __aenter__(self) -> "_TrackedResource":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: Any | None,
    ) -> None:
        self.closed.append(self.name)


class EngineLifetimeAdversarialTest(IsolatedAsyncioTestCase):
    """Pin instance ownership of every mutable Engine lifetime field."""

    async def test_engines_own_independent_lifecycle_state(self) -> None:
        first = _LifetimeEngine("first")
        second = _LifetimeEngine("second")

        for name in (
            "_exit_stack",
            "_pending_exit_task",
            "_loaded_model",
            "_loaded_tokenizer",
        ):
            self.assertIn(name, vars(first))
            self.assertIn(name, vars(second))
        self.assertIsInstance(first._exit_stack, AsyncExitStack)
        self.assertIsInstance(second._exit_stack, AsyncExitStack)
        self.assertIsNot(first._exit_stack, second._exit_stack)
        self.assertIsNone(first._pending_exit_task)
        self.assertIsNone(second._pending_exit_task)
        self.assertFalse(first._loaded_model)
        self.assertFalse(second._loaded_model)
        self.assertFalse(first._loaded_tokenizer)
        self.assertFalse(second._loaded_tokenizer)

        first._loaded_model = True
        first._loaded_tokenizer = True
        self.assertFalse(second._loaded_model)
        self.assertFalse(second._loaded_tokenizer)

    async def test_engine_provider_support_defaults_to_incapable(self) -> None:
        engine = _LifetimeEngine("default-support")

        self.assertEqual(
            engine.provider_capability_support,
            ProviderCapabilitySupport(),
        )

    async def test_closing_one_engine_does_not_close_another(self) -> None:
        first = _LifetimeEngine("first-close")
        second = _LifetimeEngine("second-close")
        closed: list[str] = []
        await first._exit_stack.enter_async_context(
            _TrackedResource("first", closed)
        )
        await second._exit_stack.enter_async_context(
            _TrackedResource("second", closed)
        )
        second._loaded_model = True
        second._loaded_tokenizer = True

        try:
            first.__exit__(None, None, None)
            await first.wait_closed()

            self.assertEqual(closed, ["first"])
            self.assertIsNone(first._pending_exit_task)
            self.assertIsNone(second._pending_exit_task)
            self.assertTrue(second._loaded_model)
            self.assertTrue(second._loaded_tokenizer)
        finally:
            second.__exit__(None, None, None)
            await second.wait_closed()

        self.assertEqual(closed, ["first", "second"])
