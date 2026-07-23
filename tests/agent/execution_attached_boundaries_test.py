"""Exercise cancellation and isolation at attached interaction boundaries."""

from asyncio import CancelledError, Event, Task, create_task, gather, wait_for
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionIdFactory,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.interaction import (
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    ConfirmationAnswer,
    ContinuationId,
    CreateInteractionCommand,
    InputHandlerContext,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputModelResult,
    InputRequest,
    InputRequestId,
    InteractionActor,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionDelivery,
    InteractionPolicy,
    InteractionRecord,
    InteractionRequestResult,
    InteractionTime,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequestState,
    ResolutionIdempotencyKey,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    RunId,
    StreamSessionId,
    TaskId,
    TerminalizeInteractionScopeCommand,
    TurnId,
    UserId,
    apply_candidate_resolution,
    apply_create_interaction,
    create_input_request,
)
from avalan.interaction.broker import InteractionBroker
from avalan.interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
from avalan.interaction.store import (
    _apply_scope_cancellation,
    _begin_scope_transaction,
    _new_interaction_store_backing,
)
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.model import ProviderCapabilitySupport
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    validate_canonical_stream_items,
)
from avalan.tool.manager import ToolManager

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


class _Boundary(StrEnum):
    """Identify one cancellation boundary in the attached state machine."""

    REQUEST_CREATION = "request_creation"
    PUBLICATION = "publication"
    HANDLER_WAIT = "handler_wait"
    RESOLUTION_COMMIT = "resolution_commit"
    RESULT_APPEND = "result_append"
    NEXT_PROVIDER_CALL = "next_provider_call"


class _Gate:
    """Expose entry and hold a boundary without timing assumptions."""

    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()

    async def hold(self) -> None:
        """Signal boundary entry and wait for release or cancellation."""
        self.entered.set()
        await self.release.wait()


class _Engine:
    """Expose explicit structured-call support for attached task input."""

    model_id = "attached-boundaries-model"
    model_type = "fake"
    provider_capability_support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
    )

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<attached-boundary-eos>")


class _Agent(EngineAgent):
    """Use the real model-dispatch implementation unchanged."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


class _Ids:
    """Mint deterministic execution identities for one attached run."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.counts: dict[str, int] = {}

    def _next(self, kind: str) -> str:
        value = self.counts.get(kind, 0) + 1
        self.counts[kind] = value
        return f"{self.prefix}-{kind}-{value}"

    async def new_run_id(self) -> RunId:
        return RunId(self._next("run"))

    async def new_turn_id(self) -> TurnId:
        return TurnId(self._next("turn"))

    async def new_task_id(self) -> TaskId:
        return TaskId(self._next("task"))

    async def new_model_call_id(self) -> ModelCallId:
        return ModelCallId(self._next("model-call"))

    async def new_branch_id(self) -> BranchId:
        return BranchId(self._next("branch"))

    async def new_stream_session_id(self) -> StreamSessionId:
        return StreamSessionId(self._next("stream"))


def _input_arguments() -> dict[str, object]:
    """Return one valid confirmation request payload."""
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


class _CapabilitySource(AsyncIterator[CanonicalStreamItem]):
    """Yield one complete reserved call and expose exact source cleanup."""

    def __init__(self, label: str, provider_name: str) -> None:
        self.label = label
        self.call_id = f"input-call-{label}"
        stream_session_id = f"provider-stream-{label}"
        run_id = f"provider-run-{label}"
        turn_id = f"provider-turn-{label}"
        correlation = StreamItemCorrelation(tool_call_id=self.call_id)
        self.items = (
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                provider_family="openai",
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=1,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta=dumps(_input_arguments()),
                correlation=correlation,
                provider_family="openai",
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": provider_name},
                correlation=correlation,
                provider_family="openai",
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                provider_family="openai",
            ),
        )
        self.index = 0
        self.trailing_read = Event()
        self._never = Event()
        self.aclose_calls = 0
        self.cancel_calls = 0

    def __aiter__(self) -> "_CapabilitySource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        self.trailing_read.set()
        await self._never.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _BlockingProviderSource(AsyncIterator[CanonicalStreamItem]):
    """Hold the resumed provider call and expose cancellation cleanup."""

    def __init__(self, gate: _Gate) -> None:
        self.gate = gate
        self.aclose_calls = 0
        self.cancel_calls = 0

    def __aiter__(self) -> "_BlockingProviderSource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        await self.gate.hold()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1

    async def cancel(self) -> None:
        self.cancel_calls += 1


def _source_response(
    source: AsyncIterator[CanonicalStreamItem],
) -> TextGenerationResponse:
    """Return a response backed by one test-owned canonical source."""
    return TextGenerationResponse(
        lambda: source,
        logger=getLogger(),
        use_async_generator=True,
    )


def _text_response(text: str) -> TextGenerationResponse:
    """Return one deterministic materialized continuation response."""
    return TextGenerationResponse(
        lambda: text,
        logger=getLogger(),
        use_async_generator=False,
    )


def _user_prompt(input_value: object) -> str:
    """Return the original text user prompt from one model context."""
    messages = (
        [input_value]
        if isinstance(input_value, Message)
        else cast(list[Message], input_value)
    )
    for message in reversed(messages):
        if message.role is MessageRole.USER and isinstance(
            message.content, str
        ):
            return message.content
    raise AssertionError("model input has no user prompt")


class _ModelManager:
    """Return a reserved call first and a deterministic continuation next."""

    def __init__(
        self,
        *,
        blocking_continuation_gate: _Gate | None = None,
    ) -> None:
        self.calls: list[ModelCall] = []
        self.initial_sources: dict[str, _CapabilitySource] = {}
        self.continuation_sources: dict[str, _BlockingProviderSource] = {}
        self.blocking_continuation_gate = blocking_continuation_gate

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        prompt = _user_prompt(call.context.input)
        if call.context.parent is None:
            capability = call.context.capability
            assert capability is not None
            provider_name = capability.provider_name(
                RESERVED_INPUT_CAPABILITY_NAME,
                provider_family="openai",
            )
            initial_source = _CapabilitySource(prompt, provider_name)
            self.initial_sources[prompt] = initial_source
            return _source_response(initial_source)
        if self.blocking_continuation_gate is not None:
            continuation_source = _BlockingProviderSource(
                self.blocking_continuation_gate
            )
            self.continuation_sources[prompt] = continuation_source
            return _source_response(continuation_source)
        return _text_response(f"done:{prompt}")


class _AnswerHandler:
    """Answer one confirmation, optionally after an explicit wait gate."""

    def __init__(self, wait_gate: _Gate | None = None) -> None:
        self.wait_gate = wait_gate
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return one exact human-authored confirmation answer."""
        self.contexts.append(context)
        if self.wait_gate is not None:
            await self.wait_gate.hold()
        return InputHandlerResolution(
            resolution=AnsweredResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
                answers=(
                    ConfirmationAnswer(
                        question_id=QuestionId("continue"),
                        provenance=AnswerProvenance.HUMAN,
                        value=True,
                    ),
                ),
            )
        )


class _BoundaryBroker:
    """Apply canonical request state around deterministic boundary gates."""

    def __init__(
        self,
        *,
        boundary: _Boundary | None = None,
        gate: _Gate | None = None,
        fail_cleanup_once: bool = False,
    ) -> None:
        self.boundary = boundary
        self.gate = gate
        self.fail_cleanup_once = fail_cleanup_once
        self.requests: list[InteractionBrokerRequest] = []
        self.results: list[InteractionRequestResult] = []
        self.records: list[InteractionRecord] = []
        self.cancel_scope_commands: list[
            TerminalizeInteractionScopeCommand
        ] = []
        self.cleanup_successes = 0

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        request_number = len(self.requests)
        if self.boundary is _Boundary.REQUEST_CREATION:
            assert self.gate is not None
            await self.gate.hold()

        policy = InteractionPolicy()
        canonical = create_input_request(
            request_id=InputRequestId(f"attached-request-{request_number}"),
            continuation_id=ContinuationId(
                f"attached-continuation-{request_number}"
            ),
            origin=request.origin,
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=_NOW,
            continuation_ttl_seconds=request.continuation_ttl_seconds,
            advisory_wait_seconds=request.advisory_wait_seconds,
        )
        admitted = apply_create_interaction(
            CreateInteractionCommand(actor=request.actor, request=canonical),
            policy,
        )
        record_index = len(self.records)
        self.records.append(admitted.record)
        assert request.handler is not None
        handler_outcome = await request.handler(
            InputHandlerContext(request=admitted.record.request)
        )
        assert isinstance(handler_outcome, InputHandlerResolution)

        if self.boundary is _Boundary.RESOLUTION_COMMIT:
            assert self.gate is not None
            await self.gate.hold()

        resolution = apply_candidate_resolution(
            admitted.record,
            ResolveInteractionCommand(
                actor=request.actor,
                correlation=admitted.record.correlation,
                expected_state_revision=(
                    admitted.record.request.state_revision
                ),
                idempotency_key=ResolutionIdempotencyKey(
                    f"attached-resolution-{request_number}"
                ),
                proposed_resolution=handler_outcome.resolution,
            ),
            InteractionTime.from_clock(
                wall_time=_NOW,
                monotonic_seconds=float(request_number),
            ),
            policy,
        )
        assert isinstance(resolution, ResolveInteractionApplied)
        self.records[record_index] = resolution.record
        result = InteractionRequestResult(
            create_result=admitted,
            delivery=InteractionDelivery(
                correlation=resolution.record.correlation,
                record=resolution.record,
                handler_attempts=1,
            ),
        )
        self.results.append(result)
        return result

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        self.cancel_scope_commands.append(command)
        if self.fail_cleanup_once and len(self.cancel_scope_commands) == 1:
            raise RuntimeError("injected scope cleanup failure")
        self.cleanup_successes += 1
        backing = _new_interaction_store_backing(records=tuple(self.records))
        store_result = _apply_scope_cancellation(
            _begin_scope_transaction(backing, command),
            command,
            InteractionTime.from_clock(
                wall_time=_NOW,
                monotonic_seconds=float(len(self.cancel_scope_commands)),
            ),
            InteractionPolicy(),
            backing=backing,
        )
        self.records = list(store_result.records)
        return InteractionBrokerResult(store_result=store_result)


class _BoundaryEventManager(EventManager):
    """Hold canonical interaction publication at its first emitted item."""

    def __init__(
        self,
        boundary: _Boundary | None,
        gate: _Gate | None,
    ) -> None:
        super().__init__(mode=EventManagerMode.TEST)
        self.boundary = boundary
        self.gate = gate

    async def trigger_stream_item(
        self,
        item: CanonicalStreamItem,
        *,
        event_type: EventType = EventType.TOKEN_GENERATED,
    ) -> None:
        if (
            self.boundary is _Boundary.PUBLICATION
            and item.kind is StreamItemKind.INTERACTION_CREATED
        ):
            assert self.gate is not None
            await self.gate.hold()
        await super().trigger_stream_item(item, event_type=event_type)


def _operation() -> AgentOperation:
    """Return one operation for the real orchestration path."""
    environment = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id=_Engine.model_id,
            params={},
        ),
        settings=TransformerEngineSettings(),
    )
    return AgentOperation(
        specification=Specification(instructions="handle attached input"),
        environment=environment,
    )


class _Harness:
    """Own one loaded Orchestrator and its deterministic fake boundaries."""

    def __init__(
        self,
        *,
        broker: _BoundaryBroker,
        manager: _ModelManager,
        boundary: _Boundary | None = None,
        gate: _Gate | None = None,
    ) -> None:
        self.logger = getLogger()
        self.events = _BoundaryEventManager(boundary, gate)
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
        self.manager = manager
        self.broker = broker
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

    def runtime(
        self,
        *,
        prefix: str,
        handler: _AnswerHandler,
    ) -> AttachedInteractionRuntime:
        """Return one runtime with deterministic task and branch identity."""
        ids = _Ids(prefix)
        return AttachedInteractionRuntime(
            broker=cast(InteractionBroker, self.broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId(f"{prefix}-user"))
            ),
            handler=handler,
            id_factory=cast(ExecutionIdFactory, ids),
            task_id=TaskId(f"{prefix}-task"),
            branch_id=BranchId(f"{prefix}-branch"),
        )

    async def exit(self) -> None:
        """Exit the loaded orchestrator exactly once."""
        if self.exited:
            return
        await self.orchestrator.__aexit__(None, None, None)
        self.exited = True

    async def close(self) -> None:
        """Close events when the orchestrator has not been exited."""
        if not self.exited:
            await self.events.aclose()


def _install_result_append_gate(
    execution: AgentExecution,
    gate: _Gate,
) -> None:
    """Hold immediately after the correlated result append commits."""
    original = execution.record_interaction_result

    async def gated_record(
        request: InputRequest,
        result: InputModelResult,
        messages: tuple[Message, ...],
        *,
        expected_revision: int | None = None,
    ) -> bool:
        committed = await original(
            request,
            result,
            messages,
            expected_revision=expected_revision,
        )
        await gate.hold()
        return committed

    setattr(execution, "record_interaction_result", gated_record)


async def _task_failure(task: Task[str]) -> BaseException | None:
    """Return one task failure without swallowing cancellation identity."""
    try:
        await task
    except BaseException as exc:
        return exc
    return None


class AttachedCancellationBoundaryTest(IsolatedAsyncioTestCase):
    """Require cancellation convergence at every attached boundary."""

    async def _exercise_boundary(
        self,
        boundary: _Boundary,
        *,
        fail_cleanup_once: bool = False,
    ) -> tuple[
        _Boundary,
        AgentExecutionStatus,
        int,
        int,
        bool,
        bool,
        bool,
        tuple[StreamItemKind, ...],
    ]:
        gate = _Gate()
        broker = _BoundaryBroker(
            boundary=boundary,
            gate=gate,
            fail_cleanup_once=fail_cleanup_once,
        )
        manager = _ModelManager(
            blocking_continuation_gate=(
                gate if boundary is _Boundary.NEXT_PROVIDER_CALL else None
            )
        )
        harness = _Harness(
            broker=broker,
            manager=manager,
            boundary=boundary,
            gate=gate,
        )
        handler = _AnswerHandler(
            gate if boundary is _Boundary.HANDLER_WAIT else None
        )
        prompt = boundary.value
        runtime = harness.runtime(prefix=prompt, handler=handler)
        response = await harness.orchestrator(
            prompt,
            interaction_runtime=runtime,
        )
        execution = response.execution
        assert execution is not None
        if boundary is _Boundary.RESULT_APPEND:
            _install_result_append_gate(execution, gate)

        consumer = create_task(response.to_str())
        try:
            await wait_for(gate.entered.wait(), timeout=1)
            self.assertFalse(consumer.done())
            consumer.cancel()
            failure = await wait_for(_task_failure(consumer), timeout=1)
            automatic_attempts = len(broker.cancel_scope_commands)
            automatic_successes = broker.cleanup_successes

            retry_failures: list[BaseException] = []
            for _ in range(2):
                try:
                    await response._cancel_pending_interaction()
                except BaseException as exc:
                    retry_failures.append(exc)

            self.assertIs(type(failure), CancelledError)
            if fail_cleanup_once:
                assert failure is not None
                self.assertTrue(
                    any(
                        "injected scope cleanup failure" in note
                        for note in getattr(failure, "__notes__", ())
                    )
                )
                self.assertEqual(
                    (automatic_attempts, automatic_successes),
                    (1, 0),
                )
                self.assertEqual(
                    (
                        len(broker.cancel_scope_commands),
                        broker.cleanup_successes,
                    ),
                    (2, 1),
                )
            else:
                self.assertEqual(
                    (automatic_attempts, automatic_successes),
                    (1, 1),
                )
                self.assertEqual(
                    (
                        len(broker.cancel_scope_commands),
                        broker.cleanup_successes,
                    ),
                    (1, 1),
                )
            self.assertEqual(retry_failures, [])

            initial_source = manager.initial_sources[prompt]
            self.assertEqual(
                (
                    initial_source.aclose_calls,
                    initial_source.cancel_calls,
                    initial_source.trailing_read.is_set(),
                ),
                (1, 0, False),
            )
            continuation_source = manager.continuation_sources.get(prompt)
            if boundary is _Boundary.NEXT_PROVIDER_CALL:
                assert continuation_source is not None
                self.assertEqual(
                    (
                        continuation_source.cancel_calls,
                        continuation_source.aclose_calls,
                    ),
                    (1, 1),
                )
            else:
                self.assertIsNone(continuation_source)

            self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
            self.assertIsNone(execution.snapshot.pending_request)
            self.assertTrue(execution.snapshot.cleanup_started)
            self.assertTrue(response._interaction_cleanup_complete)
            self.assertIsNone(response._pending_interaction_task)
            self.assertIsNone(response._pending_tool_batch_task)
            self.assertFalse(
                any(
                    record.request.state
                    in {RequestState.CREATED, RequestState.PENDING}
                    for record in broker.records
                )
            )
            if boundary in {
                _Boundary.PUBLICATION,
                _Boundary.HANDLER_WAIT,
                _Boundary.RESOLUTION_COMMIT,
            }:
                self.assertEqual(
                    {record.request.state for record in broker.records},
                    {RequestState.CANCELLED},
                )
            self.assertEqual(
                tuple(item.kind for item in response.canonical_items[-2:]),
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            )
            self.assertNotIn(
                StreamItemKind.STREAM_ERRORED,
                {item.kind for item in response.canonical_items},
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.STREAM_CANCELLED
                    for item in response.canonical_items
                ),
                1,
            )
            validate_canonical_stream_items(response.canonical_items)

            self.assertEqual(
                len(broker.cancel_scope_commands),
                2 if fail_cleanup_once else 1,
            )
            cleanup = broker.cancel_scope_commands[-1]
            self.assertEqual(cleanup.actor, runtime.actor)
            self.assertEqual(cleanup.scope.run_id, execution.origin.run_id)
            self.assertEqual(
                cleanup.scope.branch_id,
                execution.origin.branch_id,
            )
            return (
                boundary,
                execution.status,
                len(broker.cancel_scope_commands),
                broker.cleanup_successes,
                response._interaction_cleanup_complete,
                response._pending_interaction_task is None,
                response._pending_tool_batch_task is None,
                tuple(item.kind for item in response.canonical_items[-2:]),
            )
        finally:
            if not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()

    async def test_cancel_during_request_creation(self) -> None:
        summary = await self._exercise_boundary(_Boundary.REQUEST_CREATION)
        self.assertEqual(
            summary,
            (
                _Boundary.REQUEST_CREATION,
                AgentExecutionStatus.CANCELLED,
                1,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cancel_during_publication(self) -> None:
        summary = await self._exercise_boundary(_Boundary.PUBLICATION)
        self.assertEqual(
            summary,
            (
                _Boundary.PUBLICATION,
                AgentExecutionStatus.CANCELLED,
                1,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cancel_during_handler_wait(self) -> None:
        summary = await self._exercise_boundary(_Boundary.HANDLER_WAIT)
        self.assertEqual(
            summary,
            (
                _Boundary.HANDLER_WAIT,
                AgentExecutionStatus.CANCELLED,
                1,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cancel_during_resolution_commit(self) -> None:
        summary = await self._exercise_boundary(_Boundary.RESOLUTION_COMMIT)
        self.assertEqual(
            summary,
            (
                _Boundary.RESOLUTION_COMMIT,
                AgentExecutionStatus.CANCELLED,
                1,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cancel_during_result_append(self) -> None:
        summary = await self._exercise_boundary(_Boundary.RESULT_APPEND)
        self.assertEqual(
            summary,
            (
                _Boundary.RESULT_APPEND,
                AgentExecutionStatus.CANCELLED,
                1,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cancel_during_next_provider_call(self) -> None:
        summary = await self._exercise_boundary(_Boundary.NEXT_PROVIDER_CALL)
        self.assertEqual(
            summary,
            (
                _Boundary.NEXT_PROVIDER_CALL,
                AgentExecutionStatus.CANCELLED,
                1,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )

    async def test_cleanup_failure_retries_once_without_rewriting_cancel(
        self,
    ) -> None:
        summary = await self._exercise_boundary(
            _Boundary.HANDLER_WAIT,
            fail_cleanup_once=True,
        )
        self.assertEqual(
            summary,
            (
                _Boundary.HANDLER_WAIT,
                AgentExecutionStatus.CANCELLED,
                2,
                1,
                True,
                True,
                True,
                (
                    StreamItemKind.STREAM_CANCELLED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            ),
        )


class AttachedConcurrencyIsolationTest(IsolatedAsyncioTestCase):
    """Require independent attached runs on one loaded orchestrator."""

    async def test_waiting_run_does_not_block_second_attached_run(
        self,
    ) -> None:
        slow_gate = _Gate()
        broker = _BoundaryBroker()
        manager = _ModelManager()
        harness = _Harness(broker=broker, manager=manager)
        slow_runtime = harness.runtime(
            prefix="slow",
            handler=_AnswerHandler(slow_gate),
        )
        fast_runtime = harness.runtime(
            prefix="fast",
            handler=_AnswerHandler(),
        )
        slow = await harness.orchestrator(
            "slow",
            interaction_runtime=slow_runtime,
        )
        slow_consumer = create_task(slow.to_str())
        fast_consumer: Task[str] | None = None
        try:
            await wait_for(slow_gate.entered.wait(), timeout=1)
            self.assertFalse(slow_consumer.done())

            fast = await harness.orchestrator(
                "fast",
                interaction_runtime=fast_runtime,
            )
            fast_consumer = create_task(fast.to_str())
            self.assertEqual(
                await wait_for(fast_consumer, timeout=1),
                "done:fast",
            )
            self.assertFalse(slow_consumer.done())
            self.assertIs(
                slow.execution.status if slow.execution else None,
                AgentExecutionStatus.WAITING_FOR_INPUT,
            )

            slow_gate.release.set()
            self.assertEqual(
                await wait_for(slow_consumer, timeout=1),
                "done:slow",
            )
            assert slow.execution is not None and fast.execution is not None
            self.assertIs(
                slow.execution.status,
                AgentExecutionStatus.COMPLETED,
            )
            self.assertIs(
                fast.execution.status,
                AgentExecutionStatus.COMPLETED,
            )
            self.assertNotEqual(
                slow.execution.origin.run_id,
                fast.execution.origin.run_id,
            )
            self.assertEqual(
                (
                    slow.execution.origin.branch_id,
                    fast.execution.origin.branch_id,
                ),
                (BranchId("slow-branch"), BranchId("fast-branch")),
            )

            slow_calls = [
                call
                for call in manager.calls
                if _user_prompt(call.context.input) == "slow"
            ]
            fast_calls = [
                call
                for call in manager.calls
                if _user_prompt(call.context.input) == "fast"
            ]
            self.assertEqual(len(slow_calls), 2)
            self.assertEqual(len(fast_calls), 2)
            self.assertIsNone(slow_calls[0].context.parent)
            self.assertIsNone(fast_calls[0].context.parent)
            self.assertIsNotNone(slow_calls[1].context.parent)
            self.assertIsNotNone(fast_calls[1].context.parent)
            slow_origins = [
                call.context.execution_origin for call in slow_calls
            ]
            fast_origins = [
                call.context.execution_origin for call in fast_calls
            ]
            assert all(
                origin is not None for origin in (*slow_origins, *fast_origins)
            )
            self.assertEqual(
                {
                    origin.run_id
                    for origin in slow_origins
                    if origin is not None
                },
                {slow.execution.origin.run_id},
            )
            self.assertEqual(
                {
                    origin.run_id
                    for origin in fast_origins
                    if origin is not None
                },
                {fast.execution.origin.run_id},
            )
            model_call_ids = {
                origin.model_call_id
                for origin in (*slow_origins, *fast_origins)
                if origin is not None
            }
            self.assertEqual(len(model_call_ids), 4)

            request_ids = {
                result.delivery.record.request.request_id
                for result in broker.results
                if result.delivery is not None
            }
            self.assertEqual(len(request_ids), 2)
            self.assertEqual(
                {request.origin.run_id for request in broker.requests},
                {
                    slow.execution.origin.run_id,
                    fast.execution.origin.run_id,
                },
            )
            self.assertEqual(
                {request.origin.branch_id for request in broker.requests},
                {BranchId("slow-branch"), BranchId("fast-branch")},
            )
            self.assertNotEqual(
                manager.initial_sources["slow"].call_id,
                manager.initial_sources["fast"].call_id,
            )
            for source in manager.initial_sources.values():
                self.assertEqual(
                    (
                        source.aclose_calls,
                        source.cancel_calls,
                        source.trailing_read.is_set(),
                    ),
                    (1, 0, False),
                )
            self.assertEqual(broker.cancel_scope_commands, [])
            self.assertEqual(len(harness.orchestrator._engine_agents), 1)
            validate_canonical_stream_items(slow.canonical_items)
            validate_canonical_stream_items(fast.canonical_items)

            await harness.exit()
        finally:
            pending = [
                task
                for task in (slow_consumer, fast_consumer)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await gather(*pending, return_exceptions=True)
            await harness.close()
