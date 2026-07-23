"""Exercise adversarial suspension and cancellation runtime boundaries."""

from asyncio import (
    CancelledError,
    Event,
    create_task,
    gather,
    wait_for,
)
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    UuidExecutionIdFactory,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import EngineUri, TransformerEngineSettings
from avalan.event.manager import EventManager, EventManagerMode
from avalan.interaction import (
    AnswerProvenance,
    BranchId,
    ContinuationId,
    CreateInteractionApplied,
    CreateInteractionCommand,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
    InputRequestId,
    InteractionActor,
    InteractionBrokerRequest,
    InteractionDelivery,
    InteractionPolicy,
    InteractionRequestResult,
    InteractionTime,
    PrincipalScope,
    RequestState,
    ResolutionStatus,
    TaskId,
    TerminalizeInteractionCommand,
    UserId,
    apply_create_interaction,
    apply_request_terminalization,
    create_input_request,
)
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerResult,
)
from avalan.interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
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
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _scope_cancellation_result(
    command: TerminalizeInteractionScopeCommand,
) -> InteractionBrokerResult:
    return InteractionBrokerResult(
        store_result=ScopeCancellationReplayed(
            command=command,
            _token=_SCOPE_RESULT_TOKEN,
        )
    )


def echo(value: str) -> str:
    """Echo one value.

    Args:
        value: Value to echo.

    Returns:
        The provided value.
    """
    return value


class _Engine:
    """Expose the model attributes consumed by a real engine agent."""

    model_id = "suspension-model"
    model_type = "fake"
    provider_capability_support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
    )

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<suspension-eos>")


class _Agent(EngineAgent):
    """Use the production model-dispatch implementation unchanged."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


class _ScriptedModelManager:
    """Dispatch real model calls through a test-owned response factory."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.factory: Callable[[ModelCall], TextGenerationResponse] | None = (
            None
        )

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        assert self.factory is not None
        return self.factory(call)


class _BlockingSource(AsyncIterator[CanonicalStreamItem]):
    """Block provider reads and count source cleanup."""

    def __init__(self) -> None:
        self.started = Event()
        self._never = Event()
        self.aclose_calls = 0

    def __aiter__(self) -> "_BlockingSource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        self.started.set()
        await self._never.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _blocking_response(source: _BlockingSource) -> TextGenerationResponse:
    """Return a response backed by one deliberately blocked source."""
    return TextGenerationResponse(
        lambda: source,
        logger=getLogger(),
        use_async_generator=True,
    )


def _input_arguments() -> dict[str, object]:
    """Return one valid bounded task-input request."""
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


def _capability_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> TextGenerationResponse:
    """Return one complete structured capability-call stream."""

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        common = {
            "stream_session_id": "provider-stream",
            "run_id": "provider-run",
            "turn_id": "provider-turn",
            "provider_family": "openai",
        }
        yield CanonicalStreamItem(
            **common,
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        correlation = StreamItemCorrelation(tool_call_id=call_id)
        yield CanonicalStreamItem(
            **common,
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            text_delta=dumps(arguments),
            correlation=correlation,
        )
        yield CanonicalStreamItem(
            **common,
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            data={"name": name},
            correlation=correlation,
        )
        yield CanonicalStreamItem(
            **common,
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
        )
        yield CanonicalStreamItem(
            **common,
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            **common,
            sequence=5,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda: items(),
        logger=getLogger(),
        use_async_generator=True,
    )


class _WaitingHandler:
    """Hold one attached interaction at the handler boundary."""

    def __init__(self) -> None:
        self.started = Event()
        self._release = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait until containing-run cancellation interrupts delivery."""
        assert context.request.state is RequestState.PENDING
        self.started.set()
        await self._release.wait()
        return InputHandlerDetached()


def _admit_request(
    request: InteractionBrokerRequest,
    *,
    request_id: str,
) -> CreateInteractionApplied:
    """Apply one exact broker-style request admission."""
    created = create_input_request(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"{request_id}-continuation"),
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
        InteractionPolicy(),
    )
    assert isinstance(applied, CreateInteractionApplied)
    return applied


class _WaitingBroker:
    """Publish one request and wait at the attached handler."""

    def __init__(self) -> None:
        self.cancel_scope_commands: list[
            TerminalizeInteractionScopeCommand
        ] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Wait until containing-run cancellation interrupts the handler."""
        applied = _admit_request(request, request_id="waiting-request")
        assert request.handler is not None
        await request.handler(
            InputHandlerContext(request=applied.record.request)
        )
        raise AssertionError("waiting broker must be cancelled")

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Record branch-scoped cancellation."""
        self.cancel_scope_commands.append(command)
        return _scope_cancellation_result(command)


class _OpenCapabilitySource(AsyncIterator[CanonicalStreamItem]):
    """Leave a provider open after one complete reserved capability call."""

    def __init__(self, provider_name: str) -> None:
        common = {
            "stream_session_id": "provider-stream",
            "run_id": "provider-run",
            "turn_id": "provider-turn",
            "provider_family": "openai",
        }
        correlation = StreamItemCorrelation(tool_call_id="input-call")
        self._items = (
            CanonicalStreamItem(
                **common,
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                **common,
                sequence=1,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta=dumps(_input_arguments()),
                correlation=correlation,
            ),
            CanonicalStreamItem(
                **common,
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": provider_name},
                correlation=correlation,
            ),
            CanonicalStreamItem(
                **common,
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            ),
        )
        self._index = 0
        self._never = Event()
        self.trailing_read_attempted = Event()
        self.aclose_calls = 0

    @property
    def read_count(self) -> int:
        """Return the number of provider reads attempted."""
        return self._index

    def __aiter__(self) -> "_OpenCapabilitySource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        self._index += 1
        if self._index <= len(self._items):
            return self._items[self._index - 1]
        self.trailing_read_attempted.set()
        await self._never.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _open_capability_response(
    provider_name: str,
) -> tuple[TextGenerationResponse, _OpenCapabilitySource]:
    """Return a response deliberately left open after task-input completion."""
    source = _OpenCapabilitySource(provider_name)
    return (
        TextGenerationResponse(
            lambda: source,
            logger=getLogger(),
            use_async_generator=True,
        ),
        source,
    )


class _DetachedHandler:
    """Return control to a broker-owned pending continuation."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Detach without resolving the authoritative request."""
        assert context.request.state is RequestState.PENDING
        return InputHandlerDetached()


class _PendingBroker:
    """Return a valid authoritative pending delivery for input-required."""

    def __init__(self) -> None:
        self.cancel_scope_calls = 0

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Admit, publish, and return one still-pending request."""
        applied = _admit_request(request, request_id="pending-request")
        if request.handler is not None:
            await request.handler(
                InputHandlerContext(request=applied.record.request)
            )
        delivery = InteractionDelivery(
            correlation=applied.record.correlation,
            record=applied.record,
            handler_attempts=1,
        )
        return InteractionRequestResult(
            create_result=applied,
            delivery=delivery,
        )

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Record branch cleanup for the test runtime."""
        assert isinstance(command, TerminalizeInteractionScopeCommand)
        self.cancel_scope_calls += 1
        return _scope_cancellation_result(command)


class _TerminalBroker:
    """Return one explicit model-visible unavailable outcome."""

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Admit, publish, and terminalize one exact request."""
        policy = InteractionPolicy()
        applied = _admit_request(request, request_id="terminal-request")
        if request.handler is not None:
            await request.handler(
                InputHandlerContext(request=applied.record.request)
            )
        terminal = apply_request_terminalization(
            applied.record,
            TerminalizeInteractionCommand(
                actor=request.actor,
                correlation=applied.record.correlation,
                status=cast(Any, ResolutionStatus.UNAVAILABLE),
                provenance=AnswerProvenance.HUMAN,
                expected_state_revision=(
                    applied.record.request.state_revision
                ),
            ),
            InteractionTime.from_clock(
                wall_time=_NOW,
                monotonic_seconds=2.0,
            ),
            policy,
        )
        return InteractionRequestResult(
            create_result=applied,
            delivery=InteractionDelivery(
                correlation=terminal.record.correlation,
                record=terminal.record,
                handler_attempts=1,
            ),
        )

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Accept branch cleanup after a terminal delivery."""
        assert isinstance(command, TerminalizeInteractionScopeCommand)
        return _scope_cancellation_result(command)


class _FlakyCleanupBroker(_WaitingBroker):
    """Fail the first branch cleanup and accept an explicit retry."""

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Raise once, then complete branch cleanup."""
        self.cancel_scope_commands.append(command)
        if len(self.cancel_scope_commands) == 1:
            raise RuntimeError("injected broker cleanup failure")
        return _scope_cancellation_result(command)


class ExecutionSuspensionAdversarialTest(IsolatedAsyncioTestCase):
    """Exercise failure-prone boundaries through one shared real runtime."""

    async def asyncSetUp(self) -> None:
        self.logger = getLogger()
        self.event_manager = EventManager(mode=EventManagerMode.TEST)
        self.memory = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=RecentMessageMemory(),
            text_partitioner=None,
            logger=self.logger,
            event_manager=self.event_manager,
        )
        self.tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(namespace="runtime", tools=[echo])],
            enable_tools=["runtime.echo"],
        )
        self.engine = _Engine()
        self.model_manager = _ScriptedModelManager()
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id=self.engine.model_id,
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri,
            settings=TransformerEngineSettings(),
        )
        operation = AgentOperation(
            specification=Specification(instructions="suspend safely"),
            environment=environment,
        )
        self.agent = _Agent(
            cast(Any, self.engine),
            self.memory,
            self.tool,
            self.event_manager,
            cast(ModelManager, self.model_manager),
            engine_uri,
        )
        self.orchestrator = Orchestrator(
            self.logger,
            cast(ModelManager, self.model_manager),
            self.memory,
            self.tool,
            self.event_manager,
            [operation],
        )
        self.orchestrator._engine_agents[dumps(asdict(environment))] = (
            self.agent
        )

    async def asyncTearDown(self) -> None:
        await self.event_manager.aclose()

    @staticmethod
    def _runtime(
        broker: object,
        handler: object,
    ) -> AttachedInteractionRuntime:
        return AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("runtime-user"))
            ),
            handler=cast(Any, handler),
            id_factory=UuidExecutionIdFactory(),
            task_id=TaskId("runtime-task"),
            branch_id=BranchId("runtime-branch"),
        )

    def _reserved_response(self, call: ModelCall) -> TextGenerationResponse:
        capability = call.context.capability
        assert capability is not None
        provider_name = capability.provider_name(
            RESERVED_INPUT_CAPABILITY_NAME,
            provider_family="openai",
        )
        return _capability_response(
            call_id="input-call",
            name=provider_name,
            arguments=_input_arguments(),
        )

    async def test_initial_checker_cancellation_closes_provider_once(
        self,
    ) -> None:
        source = _BlockingSource()
        self.model_manager.factory = lambda _call: _blocking_response(source)
        response = await self.orchestrator("blocked initial provider")
        cancellation = Event()

        async def check_cancellation() -> None:
            if cancellation.is_set():
                raise CancelledError()

        response.set_cancellation_checker(check_cancellation)
        consumer = create_task(response.to_str())
        await wait_for(source.started.wait(), timeout=1)
        cancellation.set()
        failure: BaseException | None = None
        try:
            await wait_for(consumer, timeout=1)
        except BaseException as exc:
            failure = exc

        self.assertEqual(
            (type(failure), source.aclose_calls),
            (CancelledError, 1),
        )

    async def test_complete_reserved_call_suspends_without_read_ahead(
        self,
    ) -> None:
        broker = _WaitingBroker()
        handler = _WaitingHandler()
        runtime = self._runtime(broker, handler)
        source: _OpenCapabilitySource | None = None

        def factory(call: ModelCall) -> TextGenerationResponse:
            nonlocal source
            capability = call.context.capability
            assert capability is not None
            provider_name = capability.provider_name(
                RESERVED_INPUT_CAPABILITY_NAME,
                provider_family="openai",
            )
            response, source = _open_capability_response(provider_name)
            return response

        self.model_manager.factory = factory
        response = await self.orchestrator(
            "suspend at complete input call",
            interaction_runtime=runtime,
        )
        consumer = create_task(response.to_str())
        failure: BaseException | None = None
        try:
            await wait_for(handler.started.wait(), timeout=1)
        except BaseException as exc:
            failure = exc
        assert source is not None
        observed = (
            type(failure) if failure is not None else None,
            source.read_count,
            source.trailing_read_attempted.is_set(),
        )
        consumer.cancel()
        await gather(consumer, return_exceptions=True)

        self.assertEqual(observed, (None, 4, False))

    async def test_streaming_input_required_terminates_cleanly(self) -> None:
        broker = _PendingBroker()
        runtime = self._runtime(broker, _DetachedHandler())
        self.model_manager.factory = self._reserved_response
        response = await self.orchestrator(
            "return input required",
            interaction_runtime=runtime,
        )

        items = await wait_for(_consume(response), timeout=1)

        self.assertEqual(
            tuple(item.kind for item in items[-2:]),
            (
                StreamItemKind.STREAM_INPUT_REQUIRED,
                StreamItemKind.STREAM_CLOSED,
            ),
        )
        assert response._execution is not None
        self.assertEqual(
            response._execution.status,
            AgentExecutionStatus.INPUT_REQUIRED,
        )

    async def test_streaming_continuation_failure_is_terminal(self) -> None:
        broker = _TerminalBroker()
        runtime = self._runtime(broker, _DetachedHandler())

        def factory(call: ModelCall) -> TextGenerationResponse:
            if len(self.model_manager.calls) == 1:
                return self._reserved_response(call)
            raise RuntimeError("injected continuation failure")

        self.model_manager.factory = factory
        response = await self.orchestrator(
            "fail the attached continuation",
            interaction_runtime=runtime,
        )
        failure: BaseException | None = None
        try:
            await wait_for(_consume(response), timeout=1)
        except BaseException as exc:
            failure = exc

        assert response._execution is not None
        terminal_kinds = tuple(
            item.kind for item in response.canonical_items[-2:]
        )
        self.assertEqual(
            (
                type(failure),
                str(failure),
                response._execution.status,
                terminal_kinds,
            ),
            (
                RuntimeError,
                "injected continuation failure",
                AgentExecutionStatus.ERRORED,
                (
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cleanup_failure_is_surfaced_and_retryable(self) -> None:
        broker = _FlakyCleanupBroker()
        handler = _WaitingHandler()
        runtime = self._runtime(broker, handler)
        self.model_manager.factory = self._reserved_response
        response = await self.orchestrator(
            "cancel while input is pending",
            interaction_runtime=runtime,
        )
        consumer = create_task(response.to_str())
        await wait_for(handler.started.wait(), timeout=1)
        consumer.cancel()
        cleanup_failure: BaseException | None = None
        try:
            await consumer
        except BaseException as exc:
            cleanup_failure = exc

        retry_failure: BaseException | None = None
        try:
            await response._cancel_pending_interaction()
        except BaseException as exc:
            retry_failure = exc

        assert response._execution is not None
        assert cleanup_failure is not None
        self.assertEqual(
            (
                type(cleanup_failure),
                str(cleanup_failure),
                retry_failure,
                len(broker.cancel_scope_commands),
                response._execution.status,
            ),
            (
                CancelledError,
                "",
                None,
                2,
                AgentExecutionStatus.CANCELLED,
            ),
        )
        self.assertTrue(
            any(
                "injected broker cleanup failure" in note
                for note in getattr(cleanup_failure, "__notes__", ())
            )
        )


async def _consume(response: Any) -> list[CanonicalStreamItem]:
    """Collect one canonical response through public async iteration."""
    return [item async for item in response]


if __name__ == "__main__":
    from unittest import main

    main()
