"""Exercise invocation cancellation through real orchestration boundaries."""

from asyncio import CancelledError, Event, create_task, wait_for
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
)
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import EngineUri, Message, TransformerEngineSettings
from avalan.event.manager import EventManager, EventManagerMode
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
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def echo(value: str) -> str:
    """Echo one value.

    Args:
        value: Value to echo.

    Returns:
        The provided value.
    """
    return value


class _Engine:
    """Expose the model attributes required by a real engine agent."""

    model_id = "cancellation-model"
    model_type = "fake"
    provider_capability_support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
    )

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<cancellation-eos>")


class _Agent(EngineAgent):
    """Use the production model dispatch implementation unchanged."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


class _ScriptedModelManager:
    """Dispatch real model calls to a test-owned response factory."""

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
    """Block provider reads and count every cleanup operation."""

    def __init__(self) -> None:
        self.started = Event()
        self._never = Event()
        self.aclose_calls = 0
        self.cancel_calls = 0

    def __aiter__(self) -> "_BlockingSource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        self.started.set()
        await self._never.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _WaitingHandler(InputHandler):
    """Hold one attached interaction at the asynchronous handler boundary."""

    def __init__(self) -> None:
        self.started = Event()
        self._release = Event()
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait until the containing invocation is cancelled."""
        self.contexts.append(context)
        self.started.set()
        await self._release.wait()
        return InputHandlerDetached()


class _WaitingBroker:
    """Publish a canonical request and retain branch cancellation calls."""

    def __init__(self) -> None:
        self.requests: list[InteractionBrokerRequest] = []
        self.cancel_scope_commands: list[
            TerminalizeInteractionScopeCommand
        ] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        canonical = create_input_request(
            request_id=InputRequestId("request-attached"),
            continuation_id=ContinuationId("continuation-attached"),
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
        raise AssertionError(
            "waiting broker must be cancelled before delivery"
        )

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        self.cancel_scope_commands.append(command)
        return InteractionBrokerResult(
            store_result=ScopeCancellationReplayed(
                command=command,
                _token=_SCOPE_RESULT_TOKEN,
            )
        )


def _message_text(input_value: object) -> str:
    """Return the final plain-text user message in one model call."""
    messages = (
        [input_value]
        if isinstance(input_value, Message)
        else cast(list[Message], input_value)
    )
    for message in reversed(messages):
        if isinstance(message.content, str):
            return message.content
    raise AssertionError("model call has no text message")


def _text_response(text: str) -> TextGenerationResponse:
    """Return one repo-native materialized model response."""
    return TextGenerationResponse(
        lambda: text,
        logger=getLogger(),
        use_async_generator=False,
    )


def _blocking_response(source: _BlockingSource) -> TextGenerationResponse:
    """Return a repo-native response backed by one blocking source."""
    return TextGenerationResponse(
        lambda: source,
        logger=getLogger(),
        use_async_generator=True,
    )


def _capability_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> TextGenerationResponse:
    """Return one complete structured capability-call stream."""

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
            provider_family="openai",
        )
        correlation = StreamItemCorrelation(tool_call_id=call_id)
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            text_delta=dumps(arguments),
            correlation=correlation,
            provider_family="openai",
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            data={"name": name},
            correlation=correlation,
            provider_family="openai",
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
            provider_family="openai",
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
            provider_family="openai",
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=5,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
            provider_family="openai",
        )

    return TextGenerationResponse(
        lambda: items(),
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


class ExecutionCancellationIntegrationTest(IsolatedAsyncioTestCase):
    """Exercise source and branch cleanup under real orchestration."""

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
            specification=Specification(instructions="cancel safely"),
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
        environment_hash = dumps(asdict(environment))
        self.orchestrator._engine_agents[environment_hash] = self.agent

    async def asyncTearDown(self) -> None:
        await self.event_manager.aclose()

    async def test_initial_provider_read_closes_exactly_once(self) -> None:
        source = _BlockingSource()
        self.model_manager.factory = lambda _call: _blocking_response(source)
        response = await self.orchestrator("initial read")
        execution = response._execution
        assert execution is not None

        consumer = create_task(response.to_str())
        await wait_for(source.started.wait(), timeout=1)
        consumer.cancel()
        with self.assertRaises(CancelledError):
            await consumer

        self.assertEqual(source.aclose_calls, 1)
        self.assertEqual(source.cancel_calls, 1)
        self.assertEqual(execution.status, AgentExecutionStatus.CANCELLED)

    async def test_continuation_read_closes_exactly_once(self) -> None:
        source = _BlockingSource()

        def factory(call: ModelCall) -> TextGenerationResponse:
            if len(self.model_manager.calls) == 1:
                capability = call.context.capability
                assert capability is not None
                provider_name = capability.provider_name(
                    "runtime.echo",
                    provider_family="openai",
                )
                return _capability_response(
                    call_id="echo-call",
                    name=provider_name,
                    arguments={"value": "continue"},
                )
            return _blocking_response(source)

        self.model_manager.factory = factory
        response = await self.orchestrator("continue through tool")
        execution = response._execution
        assert execution is not None
        cancellation = Event()

        async def check_cancellation() -> None:
            if cancellation.is_set():
                raise CancelledError()

        response.set_cancellation_checker(check_cancellation)
        consumer = create_task(response.to_str())
        await wait_for(source.started.wait(), timeout=1)
        cancellation.set()
        with self.assertRaises(CancelledError):
            await wait_for(consumer, timeout=1)

        self.assertEqual(len(self.model_manager.calls), 2)
        self.assertEqual(source.aclose_calls, 1)
        self.assertEqual(source.cancel_calls, 1)
        self.assertEqual(execution.status, AgentExecutionStatus.CANCELLED)

    async def test_waiting_attached_branch_does_not_block_unrelated_run(
        self,
    ) -> None:
        broker = _WaitingBroker()
        handler = _WaitingHandler()
        actor = InteractionActor(
            principal=PrincipalScope(user_id=UserId("attached-user"))
        )
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=actor,
            handler=cast(InputHandler, handler),
            task_id=TaskId("attached-task"),
            branch_id=BranchId("attached-branch"),
        )

        def factory(call: ModelCall) -> TextGenerationResponse:
            prompt = _message_text(call.context.input)
            if prompt == "attached":
                capability = call.context.capability
                assert capability is not None
                provider_name = capability.provider_name(
                    RESERVED_INPUT_CAPABILITY_NAME,
                    provider_family="openai",
                )
                return _capability_response(
                    call_id="attached-input-call",
                    name=provider_name,
                    arguments=_input_arguments(),
                )
            return _text_response(f"answer:{prompt}")

        self.model_manager.factory = factory
        attached = await self.orchestrator(
            "attached",
            interaction_runtime=runtime,
        )
        execution = attached._execution
        assert execution is not None
        attached_consumer = create_task(attached.to_str())
        await wait_for(handler.started.wait(), timeout=1)

        unrelated = await self.orchestrator("unrelated")
        self.assertEqual(await unrelated.to_str(), "answer:unrelated")
        self.assertFalse(attached_consumer.done())
        self.assertEqual(
            execution.status,
            AgentExecutionStatus.WAITING_FOR_INPUT,
        )
        self.assertEqual(len(self.model_manager.calls), 2)

        attached_consumer.cancel()
        with self.assertRaises(CancelledError):
            await attached_consumer
        await attached._cancel_pending_interaction()
        self.assertEqual(len(broker.requests), 1)
        self.assertEqual(len(broker.cancel_scope_commands), 1)
        self.assertTrue(execution.snapshot.cleanup_started)
        self.assertEqual(execution.status, AgentExecutionStatus.CANCELLED)


if __name__ == "__main__":
    from unittest import main

    main()
