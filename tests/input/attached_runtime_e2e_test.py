"""Exercise attached structured input through the public runtime path."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    create_task,
    get_running_loop,
    run,
)
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import UTC, datetime
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    InputType,
    Specification,
)
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionIdFactory,
    ModelPromptRecord,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    EngineUri,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.interaction import (
    ActiveControlLeaseNonce,
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    BranchId,
    ConfirmationAnswer,
    ContinuationId,
    InputHandlerContext,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputRequestId,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBrokerRequest,
    InteractionClock,
    InteractionDelivery,
    InteractionDisclosure,
    InteractionIdFactory,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestResult,
    InteractionTime,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequestState,
    ResolutionIdempotencyKey,
    RunId,
    StreamSessionId,
    TaskId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TurnId,
    UserId,
)
from avalan.interaction.broker import InteractionBrokerResult
from avalan.interaction.store import TerminalizeInteractionScopeCommand
from avalan.interaction.stores import MemoryInteractionStoreFactory
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCallContext
from avalan.model.capability import ProviderCapabilitySupport
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


class _Clock(InteractionClock):
    """Hold deterministic broker time while deadline work is idle."""

    def __init__(self) -> None:
        self._waiters: list[Future[None]] = []

    async def read(self) -> InteractionTime:
        """Return one coherent trusted time observation."""
        return InteractionTime.from_clock(
            wall_time=_NOW,
            monotonic_seconds=1.0,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait until broker shutdown cancels the dormant deadline pump."""
        assert monotonic_deadline >= 1.0
        future = get_running_loop().create_future()
        self._waiters.append(future)
        try:
            await future
        except CancelledError:
            raise
        finally:
            self._waiters.remove(future)


class _BrokerIds(InteractionIdFactory):
    """Mint deterministic broker-owned identifiers."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next(self, kind: str) -> str:
        self._sequence += 1
        return f"attached-{kind}-{self._sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return a request identifier."""
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        """Return a continuation identifier."""
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return a resolution idempotency key."""
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return an active-control lease nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _ExecutionIds(ExecutionIdFactory):
    """Mint deterministic execution and provider-call identifiers."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next(self, kind: str) -> str:
        self._sequence += 1
        return f"execution-{kind}-{self._sequence}"

    async def new_run_id(self) -> RunId:
        """Return a logical run identifier."""
        return RunId(self._next("run"))

    async def new_turn_id(self) -> TurnId:
        """Return a model-turn identifier."""
        return TurnId(self._next("turn"))

    async def new_model_call_id(self) -> ModelCallId:
        """Return a provider-call identifier."""
        return ModelCallId(self._next("call"))

    async def new_task_id(self) -> TaskId:
        """Return a logical task identifier."""
        return TaskId(self._next("task"))

    async def new_branch_id(self) -> BranchId:
        """Return an execution-branch identifier."""
        return BranchId(self._next("branch"))

    async def new_stream_session_id(self) -> StreamSessionId:
        """Return a stream-session identifier."""
        return StreamSessionId(self._next("stream"))


class _Classifier(TaskInputClassifier):
    """Allow the broker-normalized confirmation under one policy."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self._policy = policy

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return an exact allow decision for the normalized request."""
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self._policy.task_input_classifier_id,
            classification_id="attached-classification",
            policy_revision=self._policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Authorizer:
    """Authorize the test actor for every exact broker operation."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return full disclosure bound to the exact target."""
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


class _GateAnswerHandler:
    """Hold one attached request open before returning a human answer."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait for release and answer the exact confirmation question."""
        self.contexts.append(context)
        self.started.set()
        await self.release.wait()
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


class _BrokerProbe:
    """Count runtime operations while delegating to the real broker."""

    def __init__(self, broker: AsyncInteractionBroker) -> None:
        self._broker = broker
        self.requests: list[InteractionBrokerRequest] = []
        self.results: list[InteractionRequestResult] = []
        self.cancel_scope_calls = 0

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Delegate one request and retain its authoritative result."""
        self.requests.append(request)
        result = await self._broker.request(request)
        self.results.append(result)
        return result

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Delegate branch-scoped cancellation and count it."""
        self.cancel_scope_calls += 1
        return await self._broker.cancel_scope(command)


async def _broker() -> AsyncInteractionBroker:
    """Open the real broker over its public memory-store factory."""
    policy = InteractionPolicy()
    clock = _Clock()
    ids = _BrokerIds()
    classifier = _Classifier(policy)
    factory = MemoryInteractionStoreFactory(
        policy=policy,
        clock=clock,
        authorizer=_Authorizer(),
        id_factory=ids,
        classifier=classifier,
    )
    return AsyncInteractionBroker(
        store=await factory.open(),
        clock=clock,
        id_factory=ids,
        policy=policy,
        classifier=classifier,
    )


def _provider_response(
    *,
    request_input: bool = False,
    answer: str | None = None,
) -> TextGenerationResponse:
    """Return one deterministic provider stream."""

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        sequence = 0
        common = {
            "stream_session_id": "provider-stream",
            "run_id": "provider-run",
            "turn_id": "provider-turn",
        }
        yield CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
            provider_family="openai",
        )
        sequence += 1
        if request_input:
            correlation = StreamItemCorrelation(tool_call_id="input-call")
            arguments = {
                "mode": "required",
                "reason": "Confirm before continuing.",
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
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta=dumps(arguments),
                correlation=correlation,
                provider_family="openai",
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": "request_user_input"},
                correlation=correlation,
                provider_family="openai",
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                provider_family="openai",
            )
            sequence += 1
        if answer is not None:
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=answer,
                provider_family="openai",
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                provider_family="openai",
            )
            sequence += 1
        yield CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
            provider_family="openai",
        )
        yield CanonicalStreamItem(
            **common,
            sequence=sequence + 1,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
            provider_family="openai",
        )

    return TextGenerationResponse(
        lambda **_: items(),
        logger=getLogger(),
        use_async_generator=True,
    )


def _operation() -> AgentOperation:
    """Return one real orchestrator operation for the fake provider."""
    return AgentOperation(
        specification=Specification(
            role=None,
            goal=None,
            input_type=InputType.TEXT,
        ),
        environment=EngineEnvironment(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="attached-model",
                params={},
            ),
            settings=TransformerEngineSettings(),
        ),
    )


def test_requirement_input_n_002_n_019_model_call_to_input_required() -> None:
    """Suspend one real run and resume it once with broker-owned identity."""

    async def exercise() -> tuple[object, ...]:
        broker = await _broker()
        probe = _BrokerProbe(broker)
        handler = _GateAnswerHandler()
        ids = _ExecutionIds()
        principal = PrincipalScope(user_id=UserId("attached-user"))
        runtime = AttachedInteractionRuntime(
            broker=cast(Any, probe),
            actor=InteractionActor(principal=principal),
            handler=handler,
            id_factory=ids,
        )
        operation = _operation()
        tool = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.trigger_stream_item = AsyncMock()
        event_manager.should_emit.return_value = False
        memory = MagicMock(spec=MemoryManager)
        memory.participant_id = None
        memory.permanent_message = None
        model_manager = MagicMock()
        orchestrator = Orchestrator(
            MagicMock(),
            model_manager,
            memory,
            tool,
            event_manager,
            operation,
        )
        responses = iter(
            (
                _provider_response(request_input=True),
                _provider_response(answer="Confirmed."),
            )
        )
        contexts: list[ModelCallContext] = []

        async def model_call(
            context: ModelCallContext,
        ) -> TextGenerationResponse:
            contexts.append(context)
            execution = context.execution
            assert execution is not None
            assert context.input is not None
            await execution.record_prompt(
                ModelPromptRecord(
                    input=context.input,
                    instructions=None,
                    system_prompt=None,
                    developer_prompt=None,
                )
            )
            response = next(responses)
            return response

        engine_agent = AsyncMock(spec=EngineAgent, side_effect=model_call)
        engine_agent.engine = SimpleNamespace(
            model_id="attached-model",
            tokenizer=SimpleNamespace(eos_token="<eos>"),
            provider_capability_support=ProviderCapabilitySupport(
                structured_invocation=True,
                stable_call_ids=True,
                correlated_results=True,
            ),
        )
        environment_hash = dumps(asdict(operation.environment))
        orchestrator._engine_agents[environment_hash] = engine_agent

        try:
            response = await orchestrator(
                "Start the operation.",
                interaction_runtime=runtime,
            )
            consume = create_task(
                _consume(response),
                name="attached-runtime-consumer",
            )
            await handler.started.wait()

            waiting_kinds = tuple(
                item.kind for item in response.canonical_items
            )
            assert StreamItemKind.INTERACTION_CREATED in waiting_kinds
            assert StreamItemKind.INTERACTION_PENDING in waiting_kinds
            assert StreamItemKind.STREAM_COMPLETED not in waiting_kinds
            assert StreamItemKind.STREAM_INPUT_REQUIRED not in waiting_kinds
            assert engine_agent.await_count == 1
            assert len(probe.requests) == 1

            handler.release.set()
            items = await consume

            assert engine_agent.await_count == 2
            assert len(contexts) == 2
            first_origin = contexts[0].execution_origin
            second_origin = contexts[1].execution_origin
            assert first_origin is not None and second_origin is not None
            assert second_origin.run_id == first_origin.run_id
            assert second_origin.branch_id == first_origin.branch_id
            assert second_origin.turn_id != first_origin.turn_id
            assert second_origin.model_call_id != first_origin.model_call_id
            assert len(probe.results) == 1
            delivery = probe.results[0].delivery
            assert isinstance(delivery, InteractionDelivery)
            assert delivery.record.request.state is RequestState.ANSWERED
            assert len(handler.contexts) == 1
            assert probe.cancel_scope_calls == 0
            assert response._execution is not None
            assert response._execution.status is AgentExecutionStatus.COMPLETED
            assert response._execution.interaction_count == 1
            kinds = tuple(item.kind for item in items)
            assert kinds.count(StreamItemKind.INTERACTION_CREATED) == 1
            assert kinds.count(StreamItemKind.INTERACTION_PENDING) == 1
            assert kinds.count(StreamItemKind.INTERACTION_ANSWERED) == 1
            assert kinds.count(StreamItemKind.MODEL_CONTINUATION_STARTED) == 1
            assert (
                kinds.count(StreamItemKind.MODEL_CONTINUATION_COMPLETED) == 1
            )
            assert kinds[-2:] == (
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            )
            assert (
                "".join(
                    item.text_delta or ""
                    for item in items
                    if item.kind is StreamItemKind.ANSWER_DELTA
                )
                == "Confirmed."
            )
            return (
                response._execution.status,
                response._execution.interaction_count,
                engine_agent.await_count,
                len(probe.requests),
                len(probe.results),
                probe.cancel_scope_calls,
                kinds[-2:],
                "".join(
                    item.text_delta or ""
                    for item in items
                    if item.kind is StreamItemKind.ANSWER_DELTA
                ),
            )
        finally:
            await broker.aclose()

    runtime_summary = run(exercise())
    assert runtime_summary == (
        AgentExecutionStatus.COMPLETED,
        1,
        2,
        1,
        1,
        0,
        (
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_CLOSED,
        ),
        "Confirmed.",
    )


async def _consume(response: Any) -> list[CanonicalStreamItem]:
    """Collect one real orchestrator response."""
    return [item async for item in response]
