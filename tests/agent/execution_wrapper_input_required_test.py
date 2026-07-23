"""Exercise input-required semantics through public orchestrator wrappers."""

from asyncio import wait_for
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from json import dumps, loads
from logging import getLogger
from types import SimpleNamespace
from typing import Annotated, Any, cast
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionInputRequiredError,
    UuidExecutionIdFactory,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.orchestrators.default import (
    DefaultOrchestrator,
)
from avalan.agent.orchestrator.orchestrators.json import JsonOrchestrator
from avalan.agent.orchestrator.orchestrators.reasoning.cot import (
    ReasoningOrchestrator,
)
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event import EventPayloadKind, EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.interaction import (
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    ContinuationId,
    CreateInteractionApplied,
    CreateInteractionCommand,
    DeclinedResolution,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputRequestId,
    InputTransitionRejected,
    InteractionActor,
    InteractionBrokerRequest,
    InteractionDelivery,
    InteractionPolicy,
    InteractionRequestResult,
    InteractionTime,
    PrincipalScope,
    RequestState,
    ResolutionIdempotencyKey,
    TaskId,
    TextAnswer,
    UserId,
    apply_candidate_resolution,
    apply_create_interaction,
    create_input_request,
    resolve_request,
)
from avalan.interaction.broker import InteractionBroker
from avalan.interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
from avalan.interaction.store import (
    ResolveInteractionApplied,
    ResolveInteractionCommand,
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

_NOW = datetime(2026, 7, 22, 18, 0, tzinfo=UTC)
_PROMPT = "Begin the attached task."
_PREAMBLE = "I need one decision before continuing. "
_FINAL_ANSWER = "The task is finished."


@dataclass(frozen=True, slots=True)
class _JsonResult:
    """Provide a concrete target for public JSON conversions."""

    answer: Annotated[str, "Final answer"]


@dataclass(frozen=True, slots=True)
class _ResponsePlan:
    """Describe one deterministic fake-provider response."""

    arguments: dict[str, object] | None = None
    preamble: str | None = None
    answer: str | None = None


class _Engine:
    """Expose model attributes consumed by a real engine agent."""

    model_id = "wrapper-input-required"
    model_type = "fake"
    provider_capability_support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        attached_resolution=True,
    )

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<wrapper-eos>")


class _Agent(EngineAgent):
    """Use production model dispatch with deterministic preparation."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        return {"instructions": context.specification.instructions}


class _ScriptedModelManager:
    """Dispatch a finite sequence of deterministic provider responses."""

    def __init__(self, plans: list[_ResponsePlan]) -> None:
        self.plans = plans
        self.calls: list[ModelCall] = []

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        index = len(self.calls)
        self.calls.append(call)
        if index >= len(self.plans):
            raise AssertionError("unexpected model continuation")
        return _provider_response(call, index, self.plans[index])


class _DetachedHandler:
    """Leave one request pending for an explicit input-required result."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Detach without fabricating a resolution."""
        self.contexts.append(context)
        return InputHandlerDetached()


class _InvalidThenDetachedHandler:
    """Return one type-mismatched answer, then detach after correction."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Expose validation feedback after one invalid resolution."""
        self.contexts.append(context)
        if len(self.contexts) == 1:
            question = context.request.questions[0]
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=context.request.request_id,
                    provenance=AnswerProvenance.HUMAN,
                    resolved_at=_NOW,
                    answers=(
                        TextAnswer(
                            question_id=question.question_id,
                            provenance=AnswerProvenance.HUMAN,
                            value="yes",
                        ),
                    ),
                )
            )
        return InputHandlerDetached()


class _DecliningHandler:
    """Return one valid model-visible decline."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Decline the exact pending request."""
        return InputHandlerResolution(
            resolution=DeclinedResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
            )
        )


class _PendingBroker:
    """Admit a request and keep it pending after bounded handler attempts."""

    def __init__(self) -> None:
        self.requests: list[InteractionBrokerRequest] = []
        self.validation_errors: list[object] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Return an authoritative pending delivery."""
        self.requests.append(request)
        applied = _admit_request(request)
        attempts = 0
        if request.handler is not None:
            attempts += 1
            outcome = await request.handler(
                InputHandlerContext(request=applied.record.request)
            )
            if isinstance(outcome, InputHandlerResolution):
                transition = resolve_request(
                    applied.record.request,
                    outcome.resolution,
                    expected_state_revision=(
                        applied.record.request.state_revision
                    ),
                )
                assert isinstance(transition, InputTransitionRejected)
                self.validation_errors.append(transition.error)
                attempts += 1
                outcome = await request.handler(
                    InputHandlerContext(
                        request=applied.record.request,
                        validation_error=transition.error,
                    )
                )
            assert isinstance(outcome, InputHandlerDetached)
        return InteractionRequestResult(
            create_result=applied,
            delivery=InteractionDelivery(
                correlation=applied.record.correlation,
                record=applied.record,
                handler_attempts=attempts,
            ),
        )

    async def cancel_scope(self, command: object) -> object:
        """Reject unexpected cleanup in this non-cancellation harness."""
        del command
        raise AssertionError("pending wrapper test requested scope cleanup")


class _DecliningBroker:
    """Commit a valid decline and expose its correlated model result."""

    def __init__(self) -> None:
        self.requests: list[InteractionBrokerRequest] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Resolve one admitted request through its attached handler."""
        self.requests.append(request)
        applied = _admit_request(request)
        assert request.handler is not None
        outcome = await request.handler(
            InputHandlerContext(request=applied.record.request)
        )
        assert isinstance(outcome, InputHandlerResolution)
        resolved = apply_candidate_resolution(
            applied.record,
            ResolveInteractionCommand(
                actor=request.actor,
                correlation=applied.record.correlation,
                expected_state_revision=(
                    applied.record.request.state_revision
                ),
                idempotency_key=ResolutionIdempotencyKey("decline-key"),
                proposed_resolution=outcome.resolution,
            ),
            InteractionTime.from_clock(
                wall_time=_NOW,
                monotonic_seconds=2.0,
            ),
            InteractionPolicy(),
        )
        assert isinstance(resolved, ResolveInteractionApplied)
        return InteractionRequestResult(
            create_result=applied,
            delivery=InteractionDelivery(
                correlation=resolved.record.correlation,
                record=resolved.record,
                handler_attempts=1,
            ),
        )

    async def cancel_scope(self, command: object) -> object:
        """Reject unexpected cleanup in this non-cancellation harness."""
        del command
        raise AssertionError("declining wrapper test requested scope cleanup")


class _Harness:
    """Wire public wrappers to a real agent and deterministic provider."""

    def __init__(
        self,
        *,
        wrapper: str,
        plans: list[_ResponsePlan],
        broker: object,
        handler: Callable[[InputHandlerContext], Any],
    ) -> None:
        self.logger = getLogger(__name__)
        self.events = EventManager(mode=EventManagerMode.TEST)
        self.recent = RecentMessageMemory()
        self.memory = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=self.recent,
            text_partitioner=None,
            logger=self.logger,
            event_manager=self.events,
        )
        self.tool = ToolManager.create_instance(enable_tools=[])
        self.model_manager = _ScriptedModelManager(plans)
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id=_Engine.model_id,
            params={},
        )
        base = self._base_orchestrator(wrapper)
        operation = base.operations[0]
        self.agent = _Agent(
            cast(Any, _Engine()),
            self.memory,
            self.tool,
            self.events,
            cast(ModelManager, self.model_manager),
            self.engine_uri,
        )
        base._engine_agents[dumps(asdict(operation.environment))] = self.agent
        self.public: Any = (
            ReasoningOrchestrator(base) if wrapper == "reasoning" else base
        )
        self.runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("wrapper-user"))
            ),
            handler=cast(Any, handler),
            id_factory=UuidExecutionIdFactory(),
            task_id=TaskId("wrapper-task"),
            branch_id=BranchId("wrapper-branch"),
        )

    def _base_orchestrator(self, wrapper: str) -> Orchestrator:
        common = {
            "engine_uri": self.engine_uri,
            "logger": self.logger,
            "model_manager": cast(ModelManager, self.model_manager),
            "memory": self.memory,
            "tool": self.tool,
            "event_manager": self.events,
            "settings": TransformerEngineSettings(),
        }
        if wrapper == "json":
            return JsonOrchestrator(
                **common,
                output=_JsonResult,
                instructions="Return structured output.",
            )
        return DefaultOrchestrator(
            **common,
            name=None,
            role=None,
            task=None,
            instructions="Handle attached input safely.",
        )

    async def response(self) -> Any:
        """Invoke the configured public orchestrator."""
        return await self.public(
            _PROMPT,
            interaction_runtime=self.runtime,
        )

    async def close(self) -> None:
        """Close event delivery resources."""
        await self.events.aclose()


def _input_arguments() -> dict[str, object]:
    """Return one valid required confirmation request."""
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


def _malformed_input_arguments() -> dict[str, object]:
    """Return a structurally invalid request with no questions."""
    return {
        "mode": "required",
        "reason": "This request is malformed.",
        "questions": [],
    }


def _admit_request(
    request: InteractionBrokerRequest,
) -> CreateInteractionApplied:
    """Apply one deterministic broker-style admission."""
    created = create_input_request(
        request_id=InputRequestId("wrapper-request"),
        continuation_id=ContinuationId("wrapper-continuation"),
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


def _provider_response(
    call: ModelCall,
    index: int,
    plan: _ResponsePlan,
) -> TextGenerationResponse:
    """Return one canonical provider stream for a scripted model call."""
    capability = call.context.capability
    assert capability is not None
    provider_name = capability.provider_name(
        RESERVED_INPUT_CAPABILITY_NAME,
        provider_family="openai",
    )

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        common = {
            "stream_session_id": f"provider-stream-{index}",
            "run_id": f"provider-run-{index}",
            "turn_id": f"provider-turn-{index}",
            "provider_family": "openai",
        }
        sequence = 0
        yield CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        sequence += 1
        if plan.preamble is not None:
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=plan.preamble,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            sequence += 1
        if plan.arguments is not None:
            correlation = StreamItemCorrelation(
                tool_call_id=f"input-call-{index}"
            )
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta=dumps(plan.arguments),
                correlation=correlation,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": provider_name},
                correlation=correlation,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            )
            sequence += 1
        if plan.answer is not None:
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=plan.answer,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            sequence += 1
        yield CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            **common,
            sequence=sequence + 1,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda: items(),
        logger=getLogger(__name__),
        use_async_generator=True,
    )


async def _consume(response: Any) -> list[CanonicalStreamItem]:
    """Collect one public orchestrator response."""
    return [item async for item in response]


def _answer_text(items: list[CanonicalStreamItem]) -> str:
    """Join public answer deltas in canonical order."""
    return "".join(
        item.text_delta or ""
        for item in items
        if item.kind is StreamItemKind.ANSWER_DELTA
    )


def _messages(response: Any) -> tuple[Message, ...]:
    """Return the response's public immutable execution transcript."""
    execution = response.execution
    assert execution is not None
    return execution.messages


class ExecutionWrapperInputRequiredTest(IsolatedAsyncioTestCase):
    """Require explicit suspension semantics across public wrappers."""

    async def test_default_stream_has_exact_input_required_order(self) -> None:
        broker = _PendingBroker()
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=broker,
            handler=_DetachedHandler(),
        )
        try:
            response = await harness.response()
            items = await wait_for(_consume(response), timeout=1)
            lifecycle = tuple(
                item.kind
                for item in items
                if item.kind
                in {
                    StreamItemKind.INTERACTION_CREATED,
                    StreamItemKind.INTERACTION_PENDING,
                    StreamItemKind.STREAM_INPUT_REQUIRED,
                    StreamItemKind.STREAM_CLOSED,
                }
            )
            event_states = tuple(
                event.observability.data["state"]
                for event in harness.events.history
                if event.type is EventType.INTERACTION_LIFECYCLE
                and event.observability.kind
                is EventPayloadKind.CANONICAL_STREAM
                and "state" in event.observability.data
            )

            self.assertEqual(
                lifecycle,
                (
                    StreamItemKind.INTERACTION_CREATED,
                    StreamItemKind.INTERACTION_PENDING,
                    StreamItemKind.STREAM_INPUT_REQUIRED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            )
            self.assertEqual(
                event_states,
                (
                    RequestState.CREATED.value,
                    RequestState.PENDING.value,
                    RequestState.PENDING.value,
                ),
            )
            self.assertNotIn(StreamItemKind.STREAM_COMPLETED, lifecycle)
            self.assertEqual(len(harness.model_manager.calls), 1)
            assert response.execution is not None
            self.assertIs(
                response.execution.status,
                AgentExecutionStatus.INPUT_REQUIRED,
            )
        finally:
            await harness.close()

    async def test_conversions_and_wrappers_expose_input_required(
        self,
    ) -> None:
        cases = (
            ("default", "to_str"),
            ("default", "to_json"),
            ("default", "to_entity"),
            ("json", "call"),
            ("reasoning", "call"),
        )
        for wrapper, conversion in cases:
            with self.subTest(wrapper=wrapper, conversion=conversion):
                broker = _PendingBroker()
                harness = _Harness(
                    wrapper=wrapper,
                    plans=[_ResponsePlan(arguments=_input_arguments())],
                    broker=broker,
                    handler=_DetachedHandler(),
                )
                try:
                    failure: ExecutionInputRequiredError | None = None
                    try:
                        if conversion == "call":
                            await harness.response()
                        else:
                            response = await harness.response()
                            if conversion == "to_str":
                                await response.to_str()
                            elif conversion == "to_json":
                                await response.to_json()
                            else:
                                await response.to(_JsonResult)
                    except ExecutionInputRequiredError as exc:
                        failure = exc

                    self.assertIsNotNone(failure)
                    assert failure is not None
                    self.assertEqual(
                        str(failure),
                        "execution requires correlated input",
                    )
                    self.assertEqual(
                        str(failure.result.request_id),
                        "wrapper-request",
                    )
                    self.assertEqual(
                        str(failure.result.continuation_id),
                        "wrapper-continuation",
                    )
                    self.assertFalse(
                        failure.result.detached_resumption_available
                    )
                    self.assertEqual(len(harness.model_manager.calls), 1)
                finally:
                    await harness.close()

    async def test_malformed_request_invokes_no_broker_or_continuation(
        self,
    ) -> None:
        broker = _PendingBroker()
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_malformed_input_arguments())],
            broker=broker,
            handler=_DetachedHandler(),
        )
        try:
            response = await harness.response()
            items = await wait_for(_consume(response), timeout=1)
            await response.sync_messages()

            self.assertEqual(len(harness.model_manager.calls), 1)
            self.assertEqual(broker.requests, [])
            self.assertIn(
                StreamItemKind.STREAM_DIAGNOSTIC,
                {item.kind for item in items},
            )
            self.assertEqual(
                tuple(item.kind for item in items[-2:]),
                (
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            )
            transcript = _messages(response)
            self.assertEqual(
                tuple(message.role for message in transcript),
                (MessageRole.USER,),
            )
            self.assertEqual(
                tuple(item.message for item in harness.recent.data),
                transcript,
            )
        finally:
            await harness.close()

    async def test_invalid_resolution_retries_without_continuation(
        self,
    ) -> None:
        broker = _PendingBroker()
        handler = _InvalidThenDetachedHandler()
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=broker,
            handler=handler,
        )
        try:
            response = await harness.response()
            items = await wait_for(_consume(response), timeout=1)
            await response.sync_messages()

            self.assertEqual(len(harness.model_manager.calls), 1)
            self.assertEqual(len(broker.requests), 1)
            self.assertEqual(len(broker.validation_errors), 1)
            self.assertEqual(len(handler.contexts), 2)
            self.assertIsNone(handler.contexts[0].validation_error)
            self.assertIsNotNone(handler.contexts[1].validation_error)
            self.assertEqual(
                tuple(item.kind for item in items[-2:]),
                (
                    StreamItemKind.STREAM_INPUT_REQUIRED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            )
            transcript = _messages(response)
            self.assertEqual(
                tuple(message.role for message in transcript),
                (MessageRole.USER,),
            )
            self.assertEqual(
                tuple(item.message for item in harness.recent.data),
                transcript,
            )
        finally:
            await harness.close()

    async def test_preamble_order_and_materialization_match_streaming(
        self,
    ) -> None:
        plans = [
            _ResponsePlan(
                arguments=_input_arguments(),
                preamble=_PREAMBLE,
            ),
            _ResponsePlan(answer=_FINAL_ANSWER),
        ]
        materialized = _Harness(
            wrapper="default",
            plans=list(plans),
            broker=_DecliningBroker(),
            handler=_DecliningHandler(),
        )
        streamed = _Harness(
            wrapper="default",
            plans=list(plans),
            broker=_DecliningBroker(),
            handler=_DecliningHandler(),
        )
        try:
            materialized_response = await materialized.response()
            materialized_text = await materialized_response.to_str()
            await materialized_response.sync_messages()

            streamed_response = await streamed.response()
            stream_items = await wait_for(
                _consume(streamed_response),
                timeout=1,
            )
            await streamed_response.sync_messages()

            materialized_messages = _messages(materialized_response)
            streamed_messages = _messages(streamed_response)
            self.assertEqual(
                materialized_text,
                _answer_text(list(materialized_response.canonical_items)),
            )
            self.assertEqual(materialized_text, _answer_text(stream_items))
            self.assertEqual(
                materialized_text,
                f"{_PREAMBLE}{_FINAL_ANSWER}",
            )
            self.assertEqual(materialized_messages, streamed_messages)
            self.assertEqual(
                tuple(message.role for message in materialized_messages),
                (
                    MessageRole.USER,
                    MessageRole.ASSISTANT,
                    MessageRole.TOOL,
                    MessageRole.ASSISTANT,
                ),
            )
            assistant_call = materialized_messages[1]
            correlated_result = materialized_messages[2]
            self.assertEqual(assistant_call.content, _PREAMBLE)
            self.assertIsNotNone(assistant_call.tool_calls)
            assert assistant_call.tool_calls is not None
            assert isinstance(correlated_result.content, str)
            result_envelope = loads(correlated_result.content)
            self.assertEqual(
                result_envelope["call_id"],
                assistant_call.tool_calls[0].id,
            )
            self.assertEqual(
                materialized_messages[3].content,
                _FINAL_ANSWER,
            )
            self.assertEqual(
                tuple(item.message for item in materialized.recent.data),
                materialized_messages,
            )
            self.assertEqual(
                tuple(item.message for item in streamed.recent.data),
                streamed_messages,
            )
            self.assertEqual(len(materialized.model_manager.calls), 2)
            self.assertEqual(len(streamed.model_manager.calls), 2)
        finally:
            await materialized.close()
            await streamed.close()
