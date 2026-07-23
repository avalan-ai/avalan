"""Pin cancellation cleanup ownership across orchestration boundaries."""

from asyncio import CancelledError, create_task, gather, sleep, wait_for
from asyncio import Event as AsyncioEvent
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionLedgerEntryKind,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import EngineUri, TransformerEngineSettings
from avalan.event import Event, EventType
from avalan.event.manager import (
    EventManager,
    EventManagerMode,
    EventSubscriberClass,
)
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionRequestResult,
)
from avalan.interaction.entities import (
    RESERVED_INPUT_CAPABILITY_NAME,
    BranchId,
    ContinuationId,
    InputRequestId,
    InputRequiredResult,
    PrincipalScope,
    TaskId,
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
from avalan.interaction.state import (
    InputTransitionApplied,
    mark_request_pending,
)
from avalan.interaction.store import (
    _SCOPE_RESULT_TOKEN,
    ScopeCancellationReplayed,
    TerminalizeInteractionScopeCommand,
)
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.tool.manager import ToolManager

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


class _Engine:
    """Expose the provider support required by the test orchestrator."""

    model_id = "cleanup-ownership-model"
    model_type = "fake"

    def __init__(self, support: ProviderCapabilitySupport) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<cleanup-eos>")
        self.provider_capability_support = support


class _Agent(EngineAgent):
    """Use the production model dispatch implementation unchanged."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


class _ModelManager:
    """Capture model calls and dispatch them to one response factory."""

    def __init__(
        self,
        factory: Callable[[ModelCall], TextGenerationResponse],
    ) -> None:
        self.factory = factory
        self.calls: list[ModelCall] = []

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        return self.factory(call)


def _operation() -> AgentOperation:
    """Return one deterministic operation for the real orchestrator path."""
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id=_Engine.model_id,
        params={},
    )
    return AgentOperation(
        specification=Specification(instructions="clean up exactly once"),
        environment=EngineEnvironment(
            engine_uri=engine_uri,
            settings=TransformerEngineSettings(),
        ),
    )


class _Harness:
    """Own one real orchestrator integration boundary."""

    def __init__(
        self,
        manager: _ModelManager,
        *,
        support: ProviderCapabilitySupport | None = None,
    ) -> None:
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
        self.manager = manager
        self.engine = _Engine(support or ProviderCapabilitySupport())
        self.operation = _operation()
        engine_uri = self.operation.environment.engine_uri
        self.agent = _Agent(
            cast(Any, self.engine),
            self.memory,
            self.tool,
            self.events,
            cast(ModelManager, manager),
            engine_uri,
        )
        self.orchestrator = Orchestrator(
            self.logger,
            cast(ModelManager, manager),
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
        """Close event delivery when the orchestrator was not exited."""
        if not self.exited:
            await self.events.aclose()


class _CleanupSource(AsyncIterator[CanonicalStreamItem]):
    """Act as a lazy provider factory and record cleanup ordering."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self._never = AsyncioEvent()

    def __call__(self, **_: object) -> "_CleanupSource":
        return self

    def __aiter__(self) -> "_CleanupSource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        await self._never.wait()
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.events.append("cancel")

    async def aclose(self) -> None:
        self.events.append("aclose")


class _CleanupFault(BaseException):
    """Represent a provider cleanup failure outside Exception."""


class _FaultingCleanupSource(_CleanupSource):
    """Fail one requested cleanup stage while recording later attempts."""

    def __init__(
        self,
        *,
        fail_cancel: bool = False,
        close_failures: int = 0,
    ) -> None:
        super().__init__()
        self.fail_cancel = fail_cancel
        self.close_failures = close_failures
        self.close_attempts = 0
        self.closed = False

    async def cancel(self) -> None:
        self.events.append("cancel")
        if self.fail_cancel:
            raise _CleanupFault("provider cancel escaped Exception")

    async def aclose(self) -> None:
        self.events.append("aclose")
        self.close_attempts += 1
        if self.close_attempts <= self.close_failures:
            raise _CleanupFault("provider close escaped Exception")
        self.closed = True


class _RetryingCloseSource(_CleanupSource):
    """Fail two provider closes and hold the successful public retry."""

    def __init__(self) -> None:
        super().__init__()
        self.close_attempts = 0
        self.closed = False
        self.retry_started = AsyncioEvent()
        self.retry_release = AsyncioEvent()

    async def aclose(self) -> None:
        self.events.append("aclose")
        self.close_attempts += 1
        if self.close_attempts <= 2:
            raise RuntimeError(
                f"provider close attempt {self.close_attempts} failed"
            )
        self.retry_started.set()
        await self.retry_release.wait()
        self.closed = True


class _BlockingCleanupSource(_CleanupSource):
    """Hold provider cancellation so repeated task cancellation can race it."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = AsyncioEvent()
        self.cancel_release = AsyncioEvent()
        self.cancel_completed = False

    async def cancel(self) -> None:
        self.events.append("cancel")
        self.cancel_started.set()
        await self.cancel_release.wait()
        self.cancel_completed = True


class _ResistantCancelSource(_CleanupSource):
    """Ignore task cancellation until a late provider failure is released."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = AsyncioEvent()
        self.cancel_release = AsyncioEvent()
        self.close_completed = AsyncioEvent()
        self.cancel_failure = _CleanupFault("late provider cancel failure")

    async def cancel(self) -> None:
        self.events.append("cancel")
        self.cancel_started.set()
        while not self.cancel_release.is_set():
            try:
                await self.cancel_release.wait()
            except CancelledError:
                continue
        raise self.cancel_failure

    async def aclose(self) -> None:
        self.events.append("aclose")
        self.close_completed.set()


class _ReadFailureSource(_CleanupSource):
    """Raise one selected provider exit and optionally fail close attempts."""

    def __init__(
        self,
        failure: BaseException,
        *,
        block_success: bool = False,
        close_failures: int = 0,
    ) -> None:
        super().__init__()
        self.failure = failure
        self.close_failures = close_failures
        self.block_success = block_success
        self.close_attempts = 0
        self.closed = False
        self.retry_started = AsyncioEvent()
        self.retry_release = AsyncioEvent()

    async def __anext__(self) -> CanonicalStreamItem:
        raise self.failure

    async def aclose(self) -> None:
        self.events.append("aclose")
        self.close_attempts += 1
        if self.close_attempts <= self.close_failures:
            raise _CleanupFault(
                f"provider close attempt {self.close_attempts} failed"
            )
        if self.block_success:
            self.retry_started.set()
            await self.retry_release.wait()
        self.closed = True


class _AnswerSource(_CleanupSource):
    """Emit one complete canonical answer for terminal race tests."""

    def __init__(self) -> None:
        super().__init__()
        self.items = iter(
            (
                CanonicalStreamItem(
                    stream_session_id="answer-stream",
                    run_id="answer-run",
                    turn_id="answer-turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="answer-stream",
                    run_id="answer-run",
                    turn_id="answer-turn",
                    sequence=1,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="answer",
                ),
                CanonicalStreamItem(
                    stream_session_id="answer-stream",
                    run_id="answer-run",
                    turn_id="answer-turn",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="answer-stream",
                    run_id="answer-run",
                    turn_id="answer-turn",
                    sequence=3,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            )
        )

    async def __anext__(self) -> CanonicalStreamItem:
        try:
            return next(self.items)
        except StopIteration as error:
            raise StopAsyncIteration from error


def _cleanup_response(source: _CleanupSource) -> TextGenerationResponse:
    """Return a lazy response whose factory itself owns provider cleanup."""
    return TextGenerationResponse(
        source,
        logger=getLogger(),
        use_async_generator=True,
    )


class _FailingRegistration(dict[int, OrchestratorResponse]):
    """Reject the ownership-transfer write after wrapper construction."""

    def __setitem__(
        self,
        key: int,
        value: OrchestratorResponse,
    ) -> None:
        raise RuntimeError("registration failed")


class _CommitThenRaiseRegistration(dict[int, object]):
    """Commit an ownership entry and then raise its selected primary."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__()
        self.failure = failure

    def __setitem__(self, key: int, value: object) -> None:
        super().__setitem__(key, value)
        raise self.failure


class _WaitingHandler(InputHandler):
    """Hold one attached interaction until its consumer is cancelled."""

    def __init__(self) -> None:
        self.started = AsyncioEvent()
        self._never = AsyncioEvent()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait at the handler boundary for cancellation."""
        assert isinstance(context, InputHandlerContext)
        self.started.set()
        await self._never.wait()
        return InputHandlerDetached()


class _FlakyWaitingBroker:
    """Fail the first branch cleanup and succeed exactly once on retry."""

    def __init__(self, *, cleanup_failures: int = 1) -> None:
        assert cleanup_failures >= 0
        self.cleanup_failures = cleanup_failures
        self.requests: list[InteractionBrokerRequest] = []
        self.cancel_scope_attempts = 0
        self.cancel_scope_commands: list[
            TerminalizeInteractionScopeCommand
        ] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        canonical = create_input_request(
            request_id=InputRequestId("cleanup-request"),
            continuation_id=ContinuationId("cleanup-continuation"),
            origin=request.origin,
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=_NOW,
            continuation_ttl_seconds=request.continuation_ttl_seconds,
            advisory_wait_seconds=request.advisory_wait_seconds,
        )
        admitted = mark_request_pending(
            canonical,
            expected_state_revision=canonical.state_revision,
        )
        assert isinstance(admitted, InputTransitionApplied)
        assert request.handler is not None
        await request.handler(InputHandlerContext(request=admitted.request))
        raise AssertionError("waiting broker must be cancelled")

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        self.cancel_scope_attempts += 1
        if self.cancel_scope_attempts <= self.cleanup_failures:
            raise RuntimeError("flaky scope cleanup")
        self.cancel_scope_commands.append(command)
        return InteractionBrokerResult(
            store_result=ScopeCancellationReplayed(
                command=command,
                _token=_SCOPE_RESULT_TOKEN,
            )
        )


class _CapabilitySource(AsyncIterator[CanonicalStreamItem]):
    """Emit one reserved call and record its provider-source cleanup."""

    def __init__(self, provider_name: str) -> None:
        correlation = StreamItemCorrelation(tool_call_id="cleanup-input")
        self.items = iter(
            (
                CanonicalStreamItem(
                    stream_session_id="provider-stream",
                    run_id="provider-run",
                    turn_id="provider-turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                    provider_family="openai",
                ),
                CanonicalStreamItem(
                    stream_session_id="provider-stream",
                    run_id="provider-run",
                    turn_id="provider-turn",
                    sequence=1,
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    channel=StreamChannel.TOOL_CALL,
                    text_delta=dumps(_input_arguments()),
                    correlation=correlation,
                    provider_family="openai",
                ),
                CanonicalStreamItem(
                    stream_session_id="provider-stream",
                    run_id="provider-run",
                    turn_id="provider-turn",
                    sequence=2,
                    kind=StreamItemKind.TOOL_CALL_READY,
                    channel=StreamChannel.TOOL_CALL,
                    data={"name": provider_name},
                    correlation=correlation,
                    provider_family="openai",
                ),
                CanonicalStreamItem(
                    stream_session_id="provider-stream",
                    run_id="provider-run",
                    turn_id="provider-turn",
                    sequence=3,
                    kind=StreamItemKind.TOOL_CALL_DONE,
                    channel=StreamChannel.TOOL_CALL,
                    correlation=correlation,
                    provider_family="openai",
                ),
            )
        )
        self.aclose_calls = 0
        self.cancel_calls = 0

    def __call__(self, **_: object) -> "_CapabilitySource":
        return self

    def __aiter__(self) -> "_CapabilitySource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        try:
            return next(self.items)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def cancel(self) -> None:
        self.cancel_calls += 1

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _input_arguments() -> dict[str, object]:
    """Return one valid required-input request."""
    return {
        "mode": "required",
        "reason": "Need one bounded decision.",
        "questions": [
            {
                "question_id": "continue",
                "kind": "confirmation",
                "prompt": "Continue?",
                "required": True,
                "choices": [],
                "allow_other": False,
            }
        ],
    }


class OrchestratorCleanupOwnershipTest(IsolatedAsyncioTestCase):
    """Exercise unowned and retained cleanup failure boundaries."""

    def _assert_unowned_settled(
        self,
        harness: _Harness,
        source: _CleanupSource,
        execution: AgentExecution,
        status: AgentExecutionStatus,
    ) -> None:
        self.assertEqual(source.events, ["cancel", "aclose"])
        self.assertEqual(execution.status, status)
        self.assertEqual(harness.orchestrator._pending_responses, {})

    async def test_blocked_after_listener_cancellation_settles_provider(
        self,
    ) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        started = AsyncioEvent()
        release = AsyncioEvent()

        async def listener(event: Event) -> None:
            assert event.type is EventType.ENGINE_RUN_AFTER
            started.set()
            await release.wait()

        harness.events.add_listener(
            listener,
            [EventType.ENGINE_RUN_AFTER],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        task = create_task(harness.orchestrator("cancel after provider"))
        try:
            await wait_for(started.wait(), timeout=1)
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            task.cancel()
            with self.assertRaises(CancelledError):
                await task
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.CANCELLED,
            )
        finally:
            release.set()
            await harness.close()

    async def test_after_listener_error_settles_provider(self) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )

        async def listener(event: Event) -> None:
            assert event.type is EventType.ENGINE_RUN_AFTER
            raise RuntimeError("listener failed")

        harness.events.add_listener(
            listener,
            [EventType.ENGINE_RUN_AFTER],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "listener failed"):
                await harness.orchestrator("fail after provider")
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.ERRORED,
            )
        finally:
            await harness.close()

    async def test_after_event_keyboard_interrupt_settles_provider(
        self,
    ) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )

        async def trigger(event: Event) -> None:
            if event.type is EventType.ENGINE_RUN_AFTER:
                raise KeyboardInterrupt("after event interrupted")

        try:
            with patch.object(
                harness.events,
                "trigger",
                side_effect=trigger,
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt,
                    "after event interrupted",
                ):
                    await harness.orchestrator("interrupt after provider")
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.CANCELLED,
            )
        finally:
            await harness.close()

    async def test_repeated_cancellation_cannot_interrupt_local_cleanup(
        self,
    ) -> None:
        source = _BlockingCleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        started = AsyncioEvent()
        release = AsyncioEvent()

        async def listener(event: Event) -> None:
            assert event.type is EventType.ENGINE_RUN_AFTER
            started.set()
            await release.wait()

        harness.events.add_listener(
            listener,
            [EventType.ENGINE_RUN_AFTER],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        task = create_task(harness.orchestrator("cancel cleanup twice"))
        try:
            await wait_for(started.wait(), timeout=1)
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            task.cancel()
            await wait_for(source.cancel_started.wait(), timeout=1)
            task.cancel()
            await sleep(0)
            self.assertFalse(task.done())
            source.cancel_release.set()
            with self.assertRaises(CancelledError):
                await task
            self.assertTrue(source.cancel_completed)
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.CANCELLED,
            )
        finally:
            release.set()
            source.cancel_release.set()
            if not task.done():
                task.cancel()
                await task
            await harness.close()

    async def test_wrapper_error_settles_provider(self) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        try:
            with patch(
                "avalan.agent.orchestrator.OrchestratorResponse",
                side_effect=RuntimeError("wrapper failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "wrapper failed"):
                    await harness.orchestrator("fail wrapper")
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.ERRORED,
            )
        finally:
            await harness.close()

    async def test_registration_error_settles_provider(self) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        harness.orchestrator._pending_responses = cast(
            Any,
            _FailingRegistration(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                await harness.orchestrator("fail registration")
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.ERRORED,
            )
        finally:
            await harness.close()

    async def _assert_registration_error_survives_cleanup_fault(
        self,
        source: _FaultingCleanupSource,
        expected_events: list[str],
    ) -> tuple[tuple[str, ...], bool, AgentExecutionStatus, int, bool]:
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        harness.orchestrator._pending_responses = cast(
            Any,
            _FailingRegistration(),
        )
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "registration failed",
            ) as raised:
                await harness.orchestrator("preserve registration failure")
            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self.assertEqual(source.events, expected_events)
            self.assertTrue(source.closed)
            self.assertEqual(
                execution.status,
                AgentExecutionStatus.ERRORED,
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
            notes = cast(
                list[str],
                getattr(raised.exception, "__notes__", []),
            )
            self.assertEqual(len(notes), 1)
            self.assertIn("_CleanupFault", notes[0])
            return (
                tuple(source.events),
                source.closed,
                execution.status,
                len(notes),
                "_CleanupFault" in notes[0],
            )
        finally:
            await harness.close()

    async def test_cancel_base_exception_does_not_skip_unowned_close(
        self,
    ) -> None:
        source = _FaultingCleanupSource(fail_cancel=True)

        cleanup_summary = (
            await self._assert_registration_error_survives_cleanup_fault(
                source,
                ["cancel", "aclose"],
            )
        )
        self.assertEqual(
            cleanup_summary,
            (
                ("cancel", "aclose"),
                True,
                AgentExecutionStatus.ERRORED,
                1,
                True,
            ),
        )

    async def test_close_base_exception_does_not_skip_terminalization(
        self,
    ) -> None:
        source = _FaultingCleanupSource(close_failures=1)

        cleanup_summary = (
            await self._assert_registration_error_survives_cleanup_fault(
                source,
                ["cancel", "aclose", "aclose"],
            )
        )
        self.assertEqual(
            cleanup_summary,
            (
                ("cancel", "aclose", "aclose"),
                True,
                AgentExecutionStatus.ERRORED,
                1,
                True,
            ),
        )

    async def test_terminalization_base_exception_is_secondary(self) -> None:
        source = _CleanupSource()

        async def settle_execution(
            *,
            cancelled: bool,
        ) -> tuple[BaseException, ...]:
            self.assertFalse(cancelled)
            return (_CleanupFault("execution terminalization failed"),)

        execution = cast(
            AgentExecution,
            SimpleNamespace(settle_provider_exit=settle_execution),
        )

        failures = await Orchestrator._cleanup_unowned_provider_response(
            _cleanup_response(source),
            execution,
            cancelled=False,
        )

        self.assertEqual(source.events, ["cancel", "aclose"])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], _CleanupFault)
        self.assertEqual(str(failures[0]), "execution terminalization failed")

    async def test_explicit_close_base_exception_keeps_closed_provider(
        self,
    ) -> None:
        source = _FaultingCleanupSource(fail_cancel=True)
        response = _cleanup_response(source)
        terminalized = False

        async def fail_after_close() -> None:
            raise _CleanupFault("post-close callback failed")

        async def settle_execution(
            *,
            cancelled: bool,
        ) -> tuple[BaseException, ...]:
            nonlocal terminalized
            self.assertFalse(cancelled)
            terminalized = True
            return ()

        response.add_done_callback(fail_after_close)
        execution = cast(
            AgentExecution,
            SimpleNamespace(settle_provider_exit=settle_execution),
        )

        failures = await Orchestrator._cleanup_unowned_provider_response(
            response,
            execution,
            cancelled=False,
        )

        self.assertEqual(source.events, ["cancel", "aclose"])
        self.assertTrue(source.closed)
        self.assertTrue(terminalized)
        self.assertEqual(len(failures), 2)
        self.assertEqual(
            tuple(str(failure) for failure in failures),
            (
                "provider cancel escaped Exception",
                "post-close callback failed",
            ),
        )

    async def test_registration_cancellation_settles_provider(self) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        after_started = AsyncioEvent()
        after_release = AsyncioEvent()
        wrapper_constructed = AsyncioEvent()
        original_response_type = OrchestratorResponse

        async def listener(event: Event) -> None:
            assert event.type is EventType.ENGINE_RUN_AFTER
            after_started.set()
            await after_release.wait()

        def construct_response(
            *args: Any, **kwargs: Any
        ) -> OrchestratorResponse:
            response = original_response_type(*args, **kwargs)
            wrapper_constructed.set()
            return response

        harness.events.add_listener(
            listener,
            [EventType.ENGINE_RUN_AFTER],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        task = create_task(harness.orchestrator("cancel registration"))
        try:
            await wait_for(after_started.wait(), timeout=1)
            await harness.orchestrator._pending_responses_lock.acquire()
            with patch(
                "avalan.agent.orchestrator.OrchestratorResponse",
                side_effect=construct_response,
            ):
                after_release.set()
                await wait_for(wrapper_constructed.wait(), timeout=1)
                execution = harness.manager.calls[0].context.execution
                assert execution is not None
                task.cancel()
                with self.assertRaises(CancelledError):
                    await task
            self._assert_unowned_settled(
                harness,
                source,
                execution,
                AgentExecutionStatus.CANCELLED,
            )
        finally:
            if harness.orchestrator._pending_responses_lock.locked():
                harness.orchestrator._pending_responses_lock.release()
            after_release.set()
            if not task.done():
                task.cancel()
                await task
            await harness.close()

    async def test_public_sync_skips_provider_still_transferring(self) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        after_started = AsyncioEvent()
        after_release = AsyncioEvent()

        async def listener(event: Event) -> None:
            assert event.type is EventType.ENGINE_RUN_AFTER
            after_started.set()
            await after_release.wait()

        harness.events.add_listener(
            listener,
            [EventType.ENGINE_RUN_AFTER],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        call = create_task(harness.orchestrator("healthy transfer"))
        response = None
        try:
            await wait_for(after_started.wait(), timeout=1)
            execution = harness.manager.calls[0].context.execution
            assert execution is not None

            await wait_for(harness.orchestrator.sync_messages(), timeout=1)

            self.assertEqual(source.events, [])
            self.assertIs(execution.status, AgentExecutionStatus.RUNNING)
            self.assertEqual(
                len(harness.orchestrator._pending_provider_cleanups),
                1,
            )
            owner = next(
                iter(harness.orchestrator._pending_provider_cleanups.values())
            )
            self.assertFalse(owner.cleanup_required)

            after_release.set()
            response = await wait_for(call, timeout=1)
            self.assertIn(
                id(response),
                harness.orchestrator._pending_responses,
            )
            self.assertEqual(
                harness.orchestrator._pending_provider_cleanups,
                {},
            )
            self.assertEqual(
                harness.agent._pending_provider_cleanups,
                {},
            )
        finally:
            after_release.set()
            if not call.done():
                call.cancel()
                await gather(call, return_exceptions=True)
            if response is not None:
                await response.aclose()
                await harness.orchestrator.sync_messages(response)
            await harness.close()

    async def test_exit_claims_transferring_owner_before_registration(
        self,
    ) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        after_started = AsyncioEvent()
        after_release = AsyncioEvent()
        wrapper_constructed = AsyncioEvent()
        original_response_type = OrchestratorResponse

        async def listener(event: Event) -> None:
            assert event.type is EventType.ENGINE_RUN_AFTER
            after_started.set()
            await after_release.wait()

        def construct_response(
            *args: Any,
            **kwargs: Any,
        ) -> OrchestratorResponse:
            response = original_response_type(*args, **kwargs)
            wrapper_constructed.set()
            return response

        harness.events.add_listener(
            listener,
            [EventType.ENGINE_RUN_AFTER],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        call = create_task(harness.orchestrator("exit transfer"))
        exit_task = None
        try:
            await wait_for(after_started.wait(), timeout=1)
            await harness.orchestrator._pending_responses_lock.acquire()
            exit_task = create_task(
                harness.orchestrator.__aexit__(None, None, None)
            )
            await sleep(0)
            with patch(
                "avalan.agent.orchestrator.OrchestratorResponse",
                side_effect=construct_response,
            ):
                after_release.set()
                await wait_for(wrapper_constructed.wait(), timeout=1)
                harness.orchestrator._pending_responses_lock.release()
                await wait_for(exit_task, timeout=1)
                harness.exited = True
                with self.assertRaisesRegex(
                    RuntimeError,
                    "orchestrator is closing",
                ):
                    await wait_for(call, timeout=1)

            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.CANCELLED
                    for entry in execution.ledger
                ),
                1,
            )
            self.assertEqual(source.events, ["cancel", "aclose"])
            self.assertEqual(
                harness.orchestrator._pending_provider_cleanups,
                {},
            )
            self.assertEqual(harness.orchestrator._pending_responses, {})
        finally:
            after_release.set()
            if harness.orchestrator._pending_responses_lock.locked():
                harness.orchestrator._pending_responses_lock.release()
            for task in (call, exit_task):
                if task is not None and not task.done():
                    task.cancel()
            await gather(
                *(task for task in (call, exit_task) if task),
                return_exceptions=True,
            )
            if not harness.exited:
                await harness.close()

    async def test_exit_closes_running_registered_response(self) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        response = await harness.orchestrator("running at exit")
        execution = response.execution
        assert execution is not None

        await harness.exit()

        self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
        self.assertEqual(source.events, ["cancel", "aclose"])
        self.assertTrue(response.ownership_cleanup_complete)
        self.assertNotIn(
            id(response),
            harness.orchestrator._pending_responses,
        )

    async def test_exit_closes_input_required_response_and_branch(
        self,
    ) -> None:
        broker = _FlakyWaitingBroker(cleanup_failures=0)
        handler = _WaitingHandler()
        source: _CapabilitySource | None = None

        def factory(call: ModelCall) -> TextGenerationResponse:
            nonlocal source
            capability = call.context.capability
            assert capability is not None
            provider_name = capability.provider_name(
                RESERVED_INPUT_CAPABILITY_NAME,
                provider_family="openai",
            )
            source = _CapabilitySource(provider_name)
            return TextGenerationResponse(
                source,
                logger=getLogger(),
                use_async_generator=True,
            )

        support = ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
        )
        harness = _Harness(_ModelManager(factory), support=support)
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("exit-user"))
            ),
            handler=cast(InputHandler, handler),
            task_id=TaskId("exit-task"),
            branch_id=BranchId("exit-branch"),
        )
        response = await harness.orchestrator(
            "input required at exit",
            interaction_runtime=runtime,
        )
        execution = response.execution
        assert execution is not None
        consumer = create_task(response.to_str())
        try:
            await wait_for(handler.started.wait(), timeout=1)
            pending_request = execution.pending_request
            assert pending_request is not None
            await execution.mark_input_required(
                InputRequiredResult(
                    request_id=pending_request.request_id,
                    continuation_id=pending_request.continuation_id,
                    detached_resumption_available=True,
                )
            )
            self.assertIs(
                execution.status,
                AgentExecutionStatus.INPUT_REQUIRED,
            )

            await harness.exit()
            await gather(consumer, return_exceptions=True)

            assert source is not None
            self.assertIs(
                execution.status,
                AgentExecutionStatus.CANCELLED,
            )
            self.assertEqual(source.aclose_calls, 1)
            self.assertEqual(broker.cancel_scope_attempts, 1)
            self.assertEqual(len(broker.cancel_scope_commands), 1)
            self.assertTrue(response.ownership_cleanup_complete)
            self.assertNotIn(
                id(response),
                harness.orchestrator._pending_responses,
            )
        finally:
            if not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            if not harness.exited:
                await harness.close()

    async def test_unowned_close_retry_retains_owner_until_public_sync(
        self,
    ) -> None:
        source = _RetryingCloseSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        harness.orchestrator._pending_responses = cast(
            Any,
            _FailingRegistration(),
        )
        retry_one = None
        retry_two = None
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "registration failed",
            ) as raised:
                await harness.orchestrator("retry unowned close")
            execution = harness.manager.calls[0].context.execution
            assert execution is not None

            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertEqual(source.close_attempts, 2)
            self.assertFalse(source.closed)
            notes = cast(
                list[str],
                getattr(raised.exception, "__notes__", []),
            )
            self.assertEqual(len(notes), 2)
            self.assertIn("provider close attempt 1 failed", notes[0])
            self.assertIn("provider close attempt 2 failed", notes[1])
            self.assertEqual(
                len(harness.orchestrator._pending_provider_cleanups),
                1,
            )

            retry_one = create_task(harness.orchestrator.sync_messages())
            retry_two = create_task(harness.orchestrator.sync_messages())
            await wait_for(source.retry_started.wait(), timeout=1)
            await wait_for(
                harness.orchestrator._pending_responses_lock.acquire(),
                timeout=1,
            )
            harness.orchestrator._pending_responses_lock.release()
            source.retry_release.set()
            await gather(retry_one, retry_two)

            self.assertTrue(source.closed)
            self.assertEqual(source.close_attempts, 3)
            self.assertEqual(
                harness.orchestrator._pending_provider_cleanups,
                {},
            )
        finally:
            source.retry_release.set()
            for task in (retry_one, retry_two):
                if task is not None and not task.done():
                    task.cancel()
            await gather(
                *(task for task in (retry_one, retry_two) if task),
                return_exceptions=True,
            )
            await harness.close()

    async def test_nonreturning_cancel_does_not_block_later_owner(
        self,
    ) -> None:
        slow_source = _ResistantCancelSource()
        fast_source = _FaultingCleanupSource(close_failures=2)
        sources = iter((slow_source, fast_source))
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(next(sources)))
        )
        harness.orchestrator._pending_responses = cast(
            Any,
            _FailingRegistration(),
        )
        timeout_patches = (
            patch(
                "avalan.model.response.text._PROVIDER_CLEANUP_TIMEOUT_SECONDS",
                0.01,
            ),
            patch(
                "avalan.agent.orchestrator._PROVIDER_CLEANUP_TIMEOUT_SECONDS",
                0.02,
            ),
        )
        try:
            with timeout_patches[0], timeout_patches[1]:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "registration failed",
                ) as slow_raised:
                    await wait_for(
                        harness.orchestrator("slow failed handoff"),
                        timeout=0.2,
                    )
                await wait_for(slow_source.cancel_started.wait(), timeout=0.1)
                await wait_for(slow_source.close_completed.wait(), timeout=0.1)
                slow_execution = harness.manager.calls[0].context.execution
                assert slow_execution is not None
                self.assertIs(
                    slow_execution.status,
                    AgentExecutionStatus.ERRORED,
                )
                self.assertIn("exceeded", slow_raised.exception.__notes__[0])

                with (
                    patch.object(
                        harness.orchestrator,
                        "_sync_terminal_responses_and_snapshot",
                        AsyncMock(return_value=()),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "registration failed",
                    ),
                ):
                    await harness.orchestrator("fast failed handoff")
                self.assertEqual(fast_source.close_attempts, 2)
                self.assertEqual(
                    len(harness.orchestrator._pending_provider_cleanups),
                    2,
                )

                with self.assertRaises(RuntimeError):
                    await wait_for(
                        harness.orchestrator.sync_messages(),
                        timeout=0.1,
                    )
                self.assertTrue(fast_source.closed)
                self.assertEqual(fast_source.close_attempts, 3)
                self.assertEqual(
                    len(harness.orchestrator._pending_provider_cleanups),
                    1,
                )

                slow_source.cancel_release.set()
                await sleep(0)
                with self.assertRaises(_CleanupFault) as late_raised:
                    await harness.orchestrator.sync_messages()
                self.assertIs(
                    late_raised.exception,
                    slow_source.cancel_failure,
                )
                self.assertEqual(
                    harness.orchestrator._pending_provider_cleanups,
                    {},
                )
        finally:
            slow_source.cancel_release.set()
            await harness.close()

    async def test_provider_base_exits_cleanup_all_consumption_boundaries(
        self,
    ) -> None:
        cases = (
            (KeyboardInterrupt, AgentExecutionStatus.CANCELLED),
            (SystemExit, AgentExecutionStatus.ERRORED),
        )
        for failure_type, expected_status in cases:
            for boundary in (
                "text_iteration",
                "text_to_str",
                "orchestrator_iteration",
                "orchestrator_to_str",
            ):
                with self.subTest(
                    failure_type=failure_type.__name__,
                    boundary=boundary,
                ):
                    primary = failure_type(
                        f"{failure_type.__name__} at {boundary}"
                    )
                    source = _ReadFailureSource(primary)
                    if boundary.startswith("text_"):
                        text_response = _cleanup_response(source)
                        with self.assertRaises(failure_type) as raised:
                            if boundary == "text_iteration":
                                await anext(aiter(text_response))
                            else:
                                await text_response.to_str()
                        self.assertIs(raised.exception, primary)
                        self.assertEqual(source.events, ["cancel", "aclose"])
                        self.assertTrue(text_response.cleanup_complete)
                        continue

                    harness = _Harness(
                        _ModelManager(lambda _call: _cleanup_response(source))
                    )
                    try:
                        orchestrator_response = await harness.orchestrator(
                            boundary
                        )

                        async def consume() -> None:
                            async for _ in orchestrator_response:
                                pass

                        with self.assertRaises(failure_type) as raised:
                            if boundary == "orchestrator_iteration":
                                await consume()
                            else:
                                await orchestrator_response.to_str()
                        self.assertIs(raised.exception, primary)
                        execution = orchestrator_response.execution
                        assert execution is not None
                        self.assertIs(execution.status, expected_status)
                        terminal_kind = (
                            StreamItemKind.STREAM_CANCELLED
                            if expected_status
                            is AgentExecutionStatus.CANCELLED
                            else StreamItemKind.STREAM_ERRORED
                        )
                        self.assertEqual(
                            sum(
                                item.kind is terminal_kind
                                for item in (
                                    orchestrator_response.canonical_items
                                )
                            ),
                            1,
                        )
                        self.assertEqual(source.events, ["cancel", "aclose"])
                        self.assertTrue(
                            orchestrator_response.ownership_cleanup_complete
                        )
                        await harness.orchestrator.sync_messages(
                            orchestrator_response
                        )
                        self.assertNotIn(
                            id(orchestrator_response),
                            harness.orchestrator._pending_responses,
                        )
                    finally:
                        await harness.close()

    async def test_errored_provider_close_retry_coalesces_before_release(
        self,
    ) -> None:
        primary = _CleanupFault("provider read escaped Exception")
        source = _ReadFailureSource(
            primary,
            block_success=True,
            close_failures=1,
        )
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        retry_close = None
        retry_sync = None
        try:
            response = await harness.orchestrator("errored close retry")
            with self.assertRaises(_CleanupFault) as raised:
                await response.to_str()
            self.assertIs(raised.exception, primary)
            notes = cast(
                list[str],
                getattr(primary, "__notes__", []),
            )
            self.assertEqual(len(notes), 1)
            self.assertIn("provider close attempt 1 failed", notes[0])
            execution = response.execution
            assert execution is not None
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertFalse(response.ownership_cleanup_complete)
            self.assertIs(
                harness.orchestrator._pending_responses[id(response)],
                response,
            )

            retry_close = create_task(response.aclose())
            retry_sync = create_task(
                harness.orchestrator.sync_messages(response)
            )
            await wait_for(source.retry_started.wait(), timeout=1)
            await wait_for(
                harness.orchestrator._pending_responses_lock.acquire(),
                timeout=1,
            )
            harness.orchestrator._pending_responses_lock.release()
            source.retry_release.set()
            await gather(retry_close, retry_sync)

            self.assertTrue(source.closed)
            self.assertEqual(source.close_attempts, 2)
            self.assertTrue(response.ownership_cleanup_complete)
            self.assertNotIn(
                id(response),
                harness.orchestrator._pending_responses,
            )
        finally:
            source.retry_release.set()
            for task in (retry_close, retry_sync):
                if task is not None and not task.done():
                    task.cancel()
            await gather(
                *(task for task in (retry_close, retry_sync) if task),
                return_exceptions=True,
            )
            await harness.close()

    async def test_exit_preserves_body_primary_and_runs_every_stage(
        self,
    ) -> None:
        slow_source = _ResistantCancelSource()
        fast_source = _FaultingCleanupSource(close_failures=2)
        sources = iter((slow_source, fast_source))
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(next(sources)))
        )
        harness.orchestrator._pending_responses = cast(
            Any,
            _FailingRegistration(),
        )
        stage_events: list[str] = []

        class ExitMemory:
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: object | None,
            ) -> None:
                stage_events.append("memory")
                raise _CleanupFault("memory exit failed")

        class ExitStack:
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: object | None,
            ) -> None:
                stage_events.append("engine-stack")
                raise _CleanupFault("engine stack exit failed")

        class ExitEngine:
            def __init__(self, name: str, *, fail: bool) -> None:
                self.name = name
                self.fail = fail

            async def wait_closed(self) -> None:
                stage_events.append(self.name)
                if self.fail:
                    raise _CleanupFault(f"{self.name} failed")

        original_event_close = harness.events.aclose

        async def close_events() -> None:
            stage_events.append("events")
            await original_event_close()
            raise _CleanupFault("event close failed")

        timeout_patches = (
            patch(
                "avalan.model.response.text._PROVIDER_CLEANUP_TIMEOUT_SECONDS",
                0.01,
            ),
            patch(
                "avalan.agent.orchestrator._PROVIDER_CLEANUP_TIMEOUT_SECONDS",
                0.02,
            ),
        )
        body_primary = RuntimeError("context body failed")
        slow_owner = None
        slow_owner_id: int | None = None
        try:
            with timeout_patches[0], timeout_patches[1]:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "registration failed",
                ):
                    await harness.orchestrator("slow exit owner")
                with (
                    patch.object(
                        harness.orchestrator,
                        "_sync_terminal_responses_and_snapshot",
                        AsyncMock(return_value=()),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "registration failed",
                    ),
                ):
                    await harness.orchestrator("fast exit owner")
                slow_owner_id, slow_owner = next(
                    iter(
                        harness.orchestrator._pending_provider_cleanups.items()
                    )
                )

                harness.orchestrator._exit_memory = True
                harness.orchestrator._memory = cast(Any, ExitMemory())
                harness.orchestrator._engines_stack = cast(Any, ExitStack())
                harness.orchestrator._engines = cast(
                    Any,
                    [
                        ExitEngine("engine-one", fail=True),
                        ExitEngine("engine-two", fail=False),
                    ],
                )
                with patch.object(
                    harness.events,
                    "aclose",
                    side_effect=close_events,
                ):
                    result = await harness.orchestrator.__aexit__(
                        RuntimeError,
                        body_primary,
                        None,
                    )
                harness.exited = True

                self.assertFalse(result)
                self.assertEqual(
                    stage_events,
                    [
                        "memory",
                        "engine-stack",
                        "engine-one",
                        "engine-two",
                        "events",
                    ],
                )
                notes = cast(
                    list[str],
                    getattr(body_primary, "__notes__", []),
                )
                self.assertGreaterEqual(len(notes), 5)
                self.assertTrue(
                    any(
                        "provider cleanup did not converge" in note
                        for note in notes
                    )
                )
                self.assertTrue(
                    any("memory exit failed" in note for note in notes)
                )
                self.assertTrue(
                    any("engine stack exit failed" in note for note in notes)
                )
                self.assertTrue(
                    any("engine-one failed" in note for note in notes)
                )
                self.assertTrue(
                    any("event close failed" in note for note in notes)
                )
                self.assertTrue(fast_source.closed)
                self.assertEqual(fast_source.close_attempts, 3)
                self.assertEqual(
                    len(harness.orchestrator._pending_provider_cleanups),
                    1,
                )
        finally:
            slow_source.cancel_release.set()
            await sleep(0)
            if slow_owner is not None:
                await slow_owner.converge()
                assert slow_owner_id is not None
                harness.orchestrator._pending_provider_cleanups.pop(
                    slow_owner_id,
                    None,
                )
            if not harness.exited:
                await harness.close()

    async def test_pre_provider_base_exits_terminalize_without_owner(
        self,
    ) -> None:
        cases = (
            (
                RuntimeError("pre-provider runtime failure"),
                AgentExecutionStatus.ERRORED,
            ),
            (
                CancelledError("pre-provider cancellation"),
                AgentExecutionStatus.CANCELLED,
            ),
            (
                _CleanupFault("pre-provider base failure"),
                AgentExecutionStatus.ERRORED,
            ),
        )
        for primary, expected_status in cases:
            with self.subTest(primary=type(primary).__name__):
                harness = _Harness(
                    _ModelManager(
                        lambda _call: _cleanup_response(_CleanupSource())
                    )
                )
                executions: list[AgentExecution] = []

                async def capture_execution(event: Event) -> None:
                    payload = cast(dict[str, Any], event.payload)
                    context = cast(
                        ModelCallContext,
                        payload["context"],
                    )
                    assert context.execution is not None
                    executions.append(context.execution)

                harness.events.add_listener(
                    capture_execution,
                    [EventType.ENGINE_AGENT_CALL_BEFORE],
                    subscriber_class=EventSubscriberClass.CRITICAL,
                )
                try:
                    with (
                        patch.object(
                            harness.agent,
                            "_prepare_call",
                            side_effect=primary,
                        ),
                        self.assertRaises(type(primary)) as raised,
                    ):
                        await harness.orchestrator("pre-provider")

                    self.assertIs(raised.exception, primary)
                    self.assertEqual(len(executions), 1)
                    self.assertIs(executions[0].status, expected_status)
                    self.assertEqual(harness.manager.calls, [])
                    self.assertEqual(
                        harness.orchestrator._pending_provider_cleanups,
                        {},
                    )
                    self.assertEqual(
                        harness.agent._pending_provider_cleanups,
                        {},
                    )
                finally:
                    await harness.close()

    async def test_handoff_installation_failures_preserve_owner(self) -> None:
        cases = (
            "owner_constructor",
            "local_registration",
            "engine_ack",
            "response_registration",
        )
        for boundary in cases:
            with self.subTest(boundary=boundary):
                primary = _CleanupFault(f"{boundary} failed")
                source = _CleanupSource()
                harness = _Harness(
                    _ModelManager(lambda _call: _cleanup_response(source))
                )
                patches = []
                if boundary == "owner_constructor":
                    patches.append(
                        patch(
                            "avalan.agent.orchestrator."
                            "_PendingProviderCleanup",
                            side_effect=primary,
                        )
                    )
                elif boundary == "local_registration":
                    harness.orchestrator._pending_provider_cleanups = cast(
                        Any,
                        _CommitThenRaiseRegistration(primary),
                    )
                elif boundary == "engine_ack":
                    patches.append(
                        patch.object(
                            harness.agent,
                            "acknowledge_provider_handoff",
                            side_effect=primary,
                        )
                    )
                else:
                    harness.orchestrator._pending_responses = cast(
                        Any,
                        _CommitThenRaiseRegistration(primary),
                    )

                try:
                    if patches:
                        patches[0].start()
                    with self.assertRaises(_CleanupFault) as raised:
                        await harness.orchestrator(boundary)

                    self.assertIs(raised.exception, primary)
                    execution = harness.manager.calls[0].context.execution
                    assert execution is not None
                    self.assertIs(
                        execution.status,
                        AgentExecutionStatus.ERRORED,
                    )
                    self.assertEqual(source.events, ["cancel", "aclose"])
                    self.assertEqual(
                        harness.orchestrator._pending_provider_cleanups,
                        {},
                    )
                    self.assertEqual(
                        harness.orchestrator._pending_responses,
                        {},
                    )
                    self.assertEqual(
                        harness.agent._pending_provider_cleanups,
                        {},
                    )
                finally:
                    for active_patch in patches:
                        active_patch.stop()
                    await harness.close()

    async def test_unexpected_owner_convergence_cannot_mask_primary(
        self,
    ) -> None:
        source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        harness.orchestrator._pending_responses = cast(
            Any,
            _FailingRegistration(),
        )
        cleanup_failure = _CleanupFault("owner convergence failed")
        try:
            with (
                patch(
                    "avalan.agent.orchestrator."
                    "_PendingProviderCleanup.converge",
                    AsyncMock(side_effect=cleanup_failure),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "registration failed",
                ) as raised,
            ):
                await harness.orchestrator("convergence failure")

            notes = cast(
                list[str],
                getattr(raised.exception, "__notes__", []),
            )
            self.assertEqual(len(notes), 1)
            self.assertIn("owner convergence failed", notes[0])
            self.assertEqual(source.events, [])
            self.assertEqual(
                len(harness.orchestrator._pending_provider_cleanups),
                1,
            )

            await harness.orchestrator.sync_messages()

            execution = harness.manager.calls[0].context.execution
            assert execution is not None
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertEqual(source.events, ["cancel", "aclose"])
            self.assertEqual(
                harness.orchestrator._pending_provider_cleanups,
                {},
            )
        finally:
            await harness.close()

    async def test_commit_then_raise_completion_keeps_canonical_winner(
        self,
    ) -> None:
        source = _AnswerSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        try:
            response = await harness.orchestrator("commit then raise")
            execution = response.execution
            assert execution is not None
            original_complete = execution.complete_with_response
            primary = _CleanupFault("completion callback failed")

            async def commit_then_raise(
                output: str,
                *,
                messages: tuple[Any, ...] = (),
                expected_revision: int | None = None,
            ) -> bool:
                await original_complete(
                    output,
                    messages=cast(Any, messages),
                    expected_revision=expected_revision,
                )
                raise primary

            with (
                patch.object(
                    execution,
                    "complete_with_response",
                    side_effect=commit_then_raise,
                ),
                self.assertRaises(_CleanupFault) as raised,
            ):
                await response.to_str()

            self.assertIs(raised.exception, primary)
            self.assertIs(execution.status, AgentExecutionStatus.COMPLETED)
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.COMPLETED
                    for entry in execution.ledger
                ),
                1,
            )
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.ERRORED
                    for entry in execution.ledger
                ),
                0,
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.STREAM_COMPLETED
                    for item in response.canonical_items
                ),
                1,
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.STREAM_ERRORED
                    for item in response.canonical_items
                ),
                0,
            )
            await harness.orchestrator.sync_messages(response)
        finally:
            await harness.close()

    async def test_to_str_cancellation_winner_has_no_success_projection(
        self,
    ) -> None:
        source = _AnswerSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        response = await harness.orchestrator("cancel wins to str")
        execution = response.execution
        assert execution is not None
        original_complete = execution.complete_with_response
        completion_started = AsyncioEvent()
        completion_release = AsyncioEvent()
        consumer = None

        async def blocked_complete(
            output: str,
            *,
            messages: tuple[Any, ...] = (),
            expected_revision: int | None = None,
        ) -> bool:
            completion_started.set()
            await completion_release.wait()
            return await original_complete(
                output,
                messages=cast(Any, messages),
                expected_revision=expected_revision,
            )

        try:
            with patch.object(
                execution,
                "complete_with_response",
                side_effect=blocked_complete,
            ):
                consumer = create_task(response.to_str())
                await wait_for(completion_started.wait(), timeout=1)
                await wait_for(response.aclose(), timeout=1)
                completion_release.set()
                with self.assertRaises(CancelledError):
                    await wait_for(consumer, timeout=1)

            with self.assertRaises(CancelledError):
                await wait_for(response.to_str(), timeout=1)
            with self.assertRaises(CancelledError):
                await wait_for(response.to_json(), timeout=1)
            await harness.orchestrator.sync_messages(response)

            self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
            self.assertIsNone(execution.last_response)
            self.assertIsNone(response._final_response_text)
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.MODEL_RESPONSE
                    for entry in execution.ledger
                ),
                0,
            )
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.COMPLETED
                    for entry in execution.ledger
                ),
                0,
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.STREAM_CANCELLED
                    for item in response.canonical_items
                ),
                1,
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.STREAM_COMPLETED
                    for item in response.canonical_items
                ),
                0,
            )
            self.assertNotIn(
                "answer",
                tuple(
                    item.message.content
                    for item in harness.memory.recent_messages or ()
                ),
            )
        finally:
            completion_release.set()
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()

    async def test_to_str_completion_winner_is_cached_success(self) -> None:
        source = _AnswerSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        response = await harness.orchestrator("complete wins to str")
        execution = response.execution
        assert execution is not None
        original_complete = execution.complete_with_response
        completion_committed = AsyncioEvent()
        completion_release = AsyncioEvent()
        consumer = None

        async def committed_complete(
            output: str,
            *,
            messages: tuple[Any, ...] = (),
            expected_revision: int | None = None,
        ) -> bool:
            result = await original_complete(
                output,
                messages=cast(Any, messages),
                expected_revision=expected_revision,
            )
            completion_committed.set()
            await completion_release.wait()
            return result

        try:
            with patch.object(
                execution,
                "complete_with_response",
                side_effect=committed_complete,
            ):
                consumer = create_task(response.to_str())
                await wait_for(completion_committed.wait(), timeout=1)
                await wait_for(response.aclose(), timeout=1)
                completion_release.set()
                self.assertEqual(
                    await wait_for(consumer, timeout=1),
                    "answer",
                )

            self.assertEqual(await response.to_str(), "answer")
            self.assertIs(execution.status, AgentExecutionStatus.COMPLETED)
            self.assertEqual(execution.last_response, "answer")
            self.assertEqual(response._final_response_text, "answer")
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.MODEL_RESPONSE
                    for entry in execution.ledger
                ),
                1,
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.STREAM_COMPLETED
                    for item in response.canonical_items
                ),
                1,
            )
        finally:
            completion_release.set()
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()

    async def test_iteration_cancellation_winner_is_typed_terminal(
        self,
    ) -> None:
        source = _AnswerSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        response = await harness.orchestrator("cancel wins iteration")
        execution = response.execution
        assert execution is not None
        original_complete = execution.complete_with_response
        completion_started = AsyncioEvent()
        completion_release = AsyncioEvent()
        consumer = None

        async def blocked_complete(
            output: str,
            *,
            messages: tuple[Any, ...] = (),
            expected_revision: int | None = None,
        ) -> bool:
            completion_started.set()
            await completion_release.wait()
            return await original_complete(
                output,
                messages=cast(Any, messages),
                expected_revision=expected_revision,
            )

        async def collect_items() -> list[CanonicalStreamItem]:
            return [item async for item in response]

        try:
            with patch.object(
                execution,
                "complete_with_response",
                side_effect=blocked_complete,
            ):
                consumer = create_task(collect_items())
                await wait_for(completion_started.wait(), timeout=1)
                await wait_for(response.aclose(), timeout=1)
                completion_release.set()
                items = await wait_for(consumer, timeout=1)

            self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
            self.assertIsNone(execution.last_response)
            self.assertEqual(
                sum(
                    entry.kind is ExecutionLedgerEntryKind.MODEL_RESPONSE
                    for entry in execution.ledger
                ),
                0,
            )
            self.assertEqual(
                [
                    item.kind
                    for item in items
                    if item.kind
                    in {
                        StreamItemKind.STREAM_COMPLETED,
                        StreamItemKind.STREAM_CANCELLED,
                        StreamItemKind.STREAM_ERRORED,
                    }
                ],
                [StreamItemKind.STREAM_CANCELLED],
            )
        finally:
            completion_release.set()
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()

    async def test_iteration_completion_winner_is_typed_terminal(
        self,
    ) -> None:
        source = _AnswerSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source))
        )
        response = await harness.orchestrator("complete wins iteration")
        execution = response.execution
        assert execution is not None
        original_complete = execution.complete_with_response
        completion_committed = AsyncioEvent()
        completion_release = AsyncioEvent()
        consumer = None

        async def committed_complete(
            output: str,
            *,
            messages: tuple[Any, ...] = (),
            expected_revision: int | None = None,
        ) -> bool:
            result = await original_complete(
                output,
                messages=cast(Any, messages),
                expected_revision=expected_revision,
            )
            completion_committed.set()
            await completion_release.wait()
            return result

        async def collect_items() -> list[CanonicalStreamItem]:
            return [item async for item in response]

        try:
            with patch.object(
                execution,
                "complete_with_response",
                side_effect=committed_complete,
            ):
                consumer = create_task(collect_items())
                await wait_for(completion_committed.wait(), timeout=1)
                await wait_for(response.aclose(), timeout=1)
                completion_release.set()
                items = await wait_for(consumer, timeout=1)

            self.assertIs(execution.status, AgentExecutionStatus.COMPLETED)
            self.assertEqual(execution.last_response, "answer")
            self.assertEqual(
                [
                    item.kind
                    for item in items
                    if item.kind
                    in {
                        StreamItemKind.STREAM_COMPLETED,
                        StreamItemKind.STREAM_CANCELLED,
                        StreamItemKind.STREAM_ERRORED,
                    }
                ],
                [StreamItemKind.STREAM_COMPLETED],
            )
        finally:
            completion_release.set()
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()

    async def test_acknowledged_continuation_has_one_exit_owner(self) -> None:
        initial_source = _CleanupSource()
        continuation_source = _CleanupSource()
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(initial_source))
        )
        response = await harness.orchestrator("continuation owner")
        execution = response.execution
        assert execution is not None
        continuation = _cleanup_response(continuation_source)
        harness.agent._retain_provider_cleanup(continuation, execution)
        await response._response.aclose()

        await response._install_continuation_response(
            continuation,
            "exit-continuation",
            activate=True,
        )

        self.assertEqual(
            harness.agent._pending_provider_cleanups,
            {},
        )
        self.assertEqual(
            await harness.agent.drain_pending_provider_cleanups(
                abandon_unclaimed=True,
            ),
            (),
        )
        self.assertEqual(continuation_source.events, [])

        await harness.exit()

        self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
        self.assertEqual(
            continuation_source.events,
            ["cancel", "aclose"],
        )
        self.assertTrue(response.ownership_cleanup_complete)

    async def _start_failed_attached_cleanup(
        self,
        *,
        iterate: bool,
    ) -> tuple[
        _Harness,
        OrchestratorResponse,
        AgentExecution,
        _FlakyWaitingBroker,
        _CapabilitySource,
    ]:
        broker = _FlakyWaitingBroker()
        handler = _WaitingHandler()
        source: _CapabilitySource | None = None

        def factory(call: ModelCall) -> TextGenerationResponse:
            nonlocal source
            capability = call.context.capability
            assert capability is not None
            provider_name = capability.provider_name(
                RESERVED_INPUT_CAPABILITY_NAME,
                provider_family="openai",
            )
            source = _CapabilitySource(provider_name)
            return TextGenerationResponse(
                source,
                logger=getLogger(),
                use_async_generator=True,
            )

        support = ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
        )
        harness = _Harness(_ModelManager(factory), support=support)
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("cleanup-user"))
            ),
            handler=cast(InputHandler, handler),
            task_id=TaskId("cleanup-task"),
            branch_id=BranchId("cleanup-branch"),
        )
        response = await harness.orchestrator(
            "attached cleanup",
            interaction_runtime=runtime,
        )
        execution = response.execution
        assert execution is not None

        async def consume_items() -> None:
            async for _ in response:
                pass

        consumer = create_task(
            consume_items() if iterate else response.to_str()
        )
        await wait_for(handler.started.wait(), timeout=1)
        assert source is not None
        self.assertEqual(source.aclose_calls, 1)
        consumer.cancel()
        with self.assertRaises(CancelledError) as raised:
            await consumer
        notes = cast(
            list[str],
            getattr(raised.exception, "__notes__", []),
        )
        self.assertEqual(len(notes), 1)
        self.assertIn("RuntimeError: flaky scope cleanup", notes[0])

        self.assertEqual(execution.status, AgentExecutionStatus.CANCELLED)
        self.assertFalse(response.ownership_cleanup_complete)
        self.assertIs(
            harness.orchestrator._pending_responses[id(response)],
            response,
        )
        self.assertEqual(broker.cancel_scope_attempts, 1)
        self.assertEqual(broker.cancel_scope_commands, [])
        self.assertEqual(
            sum(
                item.kind is StreamItemKind.STREAM_CANCELLED
                for item in response.canonical_items
            ),
            1,
        )
        self.assertEqual(
            sum(
                item.kind is StreamItemKind.STREAM_CLOSED
                for item in response.canonical_items
            ),
            1,
        )
        return harness, response, execution, broker, source

    def _assert_attached_cleanup_retried_once(
        self,
        harness: _Harness,
        response: OrchestratorResponse,
        execution: AgentExecution,
        broker: _FlakyWaitingBroker,
        source: _CapabilitySource,
    ) -> tuple[bool, bool, int, int, int, int, bool]:
        self.assertTrue(response.ownership_cleanup_complete)
        self.assertTrue(execution.snapshot.cleanup_started)
        self.assertEqual(broker.cancel_scope_attempts, 2)
        self.assertEqual(len(broker.cancel_scope_commands), 1)
        self.assertEqual(source.aclose_calls, 1)
        self.assertEqual(source.cancel_calls, 0)
        self.assertEqual(
            sum(
                item.kind is StreamItemKind.STREAM_CANCELLED
                for item in response.canonical_items
            ),
            1,
        )
        self.assertEqual(
            sum(
                item.kind is StreamItemKind.STREAM_CLOSED
                for item in response.canonical_items
            ),
            1,
        )
        self.assertNotIn(id(response), harness.orchestrator._pending_responses)
        return (
            response.ownership_cleanup_complete,
            execution.snapshot.cleanup_started,
            broker.cancel_scope_attempts,
            len(broker.cancel_scope_commands),
            source.aclose_calls,
            source.cancel_calls,
            id(response) not in harness.orchestrator._pending_responses,
        )

    async def test_public_aclose_retries_before_explicit_sync_release(
        self,
    ) -> None:
        (
            harness,
            response,
            execution,
            broker,
            source,
        ) = await self._start_failed_attached_cleanup(iterate=False)
        try:
            await response.aclose()
            self.assertIn(
                id(response),
                harness.orchestrator._pending_responses,
            )
            await harness.orchestrator.sync_messages(response)
            self._assert_attached_cleanup_retried_once(
                harness,
                response,
                execution,
                broker,
                source,
            )
        finally:
            await harness.close()

    async def test_explicit_sync_retries_failed_iteration_cleanup(
        self,
    ) -> None:
        (
            harness,
            response,
            execution,
            broker,
            source,
        ) = await self._start_failed_attached_cleanup(iterate=True)
        try:
            await harness.orchestrator.sync_messages(response)
            cleanup_summary = self._assert_attached_cleanup_retried_once(
                harness,
                response,
                execution,
                broker,
                source,
            )
            self.assertEqual(
                cleanup_summary,
                (True, True, 2, 1, 1, 0, True),
            )
        finally:
            await harness.close()

    async def test_all_response_sync_retries_failed_cleanup(self) -> None:
        (
            harness,
            response,
            execution,
            broker,
            source,
        ) = await self._start_failed_attached_cleanup(iterate=False)
        try:
            await harness.orchestrator.sync_messages()
            cleanup_summary = self._assert_attached_cleanup_retried_once(
                harness,
                response,
                execution,
                broker,
                source,
            )
            self.assertEqual(
                cleanup_summary,
                (True, True, 2, 1, 1, 0, True),
            )
        finally:
            await harness.close()

    async def test_orchestrator_exit_retries_failed_cleanup(self) -> None:
        (
            harness,
            response,
            execution,
            broker,
            source,
        ) = await self._start_failed_attached_cleanup(iterate=False)
        await harness.exit()
        cleanup_summary = self._assert_attached_cleanup_retried_once(
            harness,
            response,
            execution,
            broker,
            source,
        )
        self.assertEqual(
            cleanup_summary,
            (True, True, 2, 1, 1, 0, True),
        )

    async def test_provider_close_retry_coalesces_before_ownership_release(
        self,
    ) -> None:
        source = _RetryingCloseSource()
        broker = _FlakyWaitingBroker(cleanup_failures=0)
        handler = _WaitingHandler()
        support = ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
        )
        harness = _Harness(
            _ModelManager(lambda _call: _cleanup_response(source)),
            support=support,
        )
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("cleanup-user"))
            ),
            handler=cast(InputHandler, handler),
            task_id=TaskId("cleanup-task"),
            branch_id=BranchId("cleanup-branch"),
        )
        response = await harness.orchestrator(
            "retry provider cleanup",
            interaction_runtime=runtime,
        )
        execution = response.execution
        assert execution is not None
        retry_close = None
        retry_sync = None
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "provider close attempt 2 failed",
            ):
                await response.aclose()

            self.assertEqual(
                execution.status,
                AgentExecutionStatus.CANCELLED,
            )
            self.assertFalse(source.closed)
            self.assertEqual(source.close_attempts, 2)
            self.assertFalse(response.ownership_cleanup_complete)
            self.assertEqual(broker.cancel_scope_attempts, 1)
            self.assertIs(
                harness.orchestrator._pending_responses[id(response)],
                response,
            )

            retry_close = create_task(response.aclose())
            retry_sync = create_task(
                harness.orchestrator.sync_messages(response)
            )
            await wait_for(source.retry_started.wait(), timeout=1)
            await wait_for(
                harness.orchestrator._pending_responses_lock.acquire(),
                timeout=1,
            )
            harness.orchestrator._pending_responses_lock.release()
            self.assertFalse(source.closed)
            self.assertIn(
                id(response),
                harness.orchestrator._pending_responses,
            )

            source.retry_release.set()
            await gather(retry_close, retry_sync)

            self.assertTrue(source.closed)
            self.assertEqual(source.close_attempts, 3)
            self.assertEqual(
                source.events,
                ["cancel", "aclose", "aclose", "aclose"],
            )
            self.assertTrue(response.ownership_cleanup_complete)
            self.assertEqual(broker.cancel_scope_attempts, 1)
            self.assertNotIn(
                id(response),
                harness.orchestrator._pending_responses,
            )
        finally:
            source.retry_release.set()
            for task in (retry_close, retry_sync):
                if task is not None and not task.done():
                    task.cancel()
            await gather(
                *(task for task in (retry_close, retry_sync) if task),
                return_exceptions=True,
            )
            await harness.close()


if __name__ == "__main__":
    from unittest import main

    main()
