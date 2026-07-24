"""Exercise input-required semantics through public orchestrator wrappers."""

from asyncio import wait_for
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from logging import getLogger
from types import SimpleNamespace
from typing import Annotated, Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch
from uuid import uuid4

from avalan import (
    AgentRunCancelled,
    AgentRunCompleted,
    AgentRunFailed,
    AgentRunInputRequired,
    run_agent,
)
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    DurableInteractionStagingContext,
    ExecutionCorrelationError,
    ExecutionInputRequiredError,
    ExecutionTerminatedError,
    UuidExecutionIdFactory,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.orchestrators.default import (
    DefaultOrchestrator,
)
from avalan.agent.orchestrator.orchestrators.json import (
    JsonOrchestrator,
    JsonOrchestratorOutput,
)
from avalan.agent.orchestrator.orchestrators.reasoning.cot import (
    ReasoningOrchestrator,
)
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    ReasoningOrchestratorResponse,
    TransformerEngineSettings,
)
from avalan.event import EventPayloadKind, EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.interaction import (
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CapabilityRevision,
    ConfirmationQuestion,
    ContinuationClaim,
    ContinuationFencingToken,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ContinuationStoreRevision,
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
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PortableContinuation,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderContinuationSnapshotAdapter,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    StateRevision,
    TaskId,
    TerminateInputContinuation,
    TextAnswer,
    UserId,
    apply_candidate_resolution,
    apply_create_interaction,
    create_input_request,
    decode_continuation_snapshot,
    encode_continuation_snapshot,
    resolve_request,
)
from avalan.interaction.broker import InteractionBroker
from avalan.interaction.continuation import (
    derive_continuation_dispatch_id,
    derive_provider_idempotency_key,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
from avalan.interaction.store import (
    ResolveInteractionApplied,
    ResolveInteractionCommand,
)
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.capability import (
    ContinuationSnapshotCodecRegistry,
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.model.manager import ModelManager
from avalan.model.nlp.text.vendor import openai as openai_module
from avalan.model.nlp.text.vendor.openai import OpenAIClient
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
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
    failure: BaseException | None = None


def _durable_binding() -> ContinuationRevisionBinding:
    """Return the exact provider revision used by the durable harness."""
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("wrapper-input-required"),
        provider_config_revision=ProviderConfigRevision("provider-v1"),
        model_config_revision=ModelConfigRevision("model-v1"),
        capability_revision=CapabilityRevision("capability-v1"),
    )


def _durable_support(
    binding: ContinuationRevisionBinding,
) -> ProviderCapabilitySupport:
    """Return fully registered durable provider support."""
    registry = ContinuationSnapshotCodecRegistry("wrapper-codec-registry")
    registry.register(
        codec_id="wrapper-codec-v1",
        revision_binding=binding,
        snapshot_kind="wrapper.provider-response",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=lambda value, expected: decode_continuation_snapshot(
            value,
            expected_binding=expected,
        ),
    )
    return ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        durable_store=True,
        registered_resumer=True,
        continuation_snapshot_codec_registry=registry,
        continuation_snapshot_codec=registry.reference("wrapper-codec-v1"),
    )


def _openai_durable_support(
    binding: ContinuationRevisionBinding,
) -> ProviderCapabilitySupport:
    """Return registered native OpenAI continuation support."""
    registry = ContinuationSnapshotCodecRegistry("openai-codec-registry")
    codec = OpenAIClient.register_continuation_snapshot_codec(
        registry,
        codec_id="openai-responses-v1",
        revision_binding=binding,
    )
    return ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        durable_store=True,
        registered_resumer=True,
        continuation_snapshot_codec_registry=registry,
        continuation_snapshot_codec=codec,
    )


def _durable_staging_context() -> DurableInteractionStagingContext:
    """Return one fully correlated provider staging context."""
    binding = _durable_binding()
    support = _durable_support(binding)
    registry = support.continuation_snapshot_codec_registry
    codec = support.continuation_snapshot_codec
    assert registry is not None and codec is not None
    continuation_id = ContinuationId("staging-continuation")
    dispatch_id = derive_continuation_dispatch_id(continuation_id)
    provider_idempotency_key = derive_provider_idempotency_key(
        continuation_id,
        dispatch_id,
    )
    task_input_call = TaskInputCapabilityCall(
        call_id="staging-input-call",
        provider_name=RESERVED_INPUT_CAPABILITY_NAME,
        arguments=_input_arguments(),
        mode=RequirementMode.REQUIRED,
        reason="Need one bounded decision.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("continue"),
                prompt="Continue?",
                required=True,
            ),
        ),
        advertisement=TaskInputCapabilityAdvertisement.DURABLE,
    )
    snapshot = ContinuationSnapshot(
        snapshot_kind="wrapper.provider-response",
        revision_binding=binding,
        model_call_id=ModelCallId("staging-model-call"),
        provider_idempotency_key=provider_idempotency_key,
        payload={
            "reserved_capability_call_id": str(task_input_call.call_id),
            "replay_items": (),
        },
    )
    return DurableInteractionStagingContext(
        task_input_call=task_input_call,
        continuation_id=continuation_id,
        dispatch_id=dispatch_id,
        revision_binding=binding,
        codec_registry=registry,
        codec=codec,
        provider_snapshot=snapshot,
        provider_idempotency_key=provider_idempotency_key,
        provider_call_correlation_id=str(task_input_call.call_id),
    )


def _openai_client() -> OpenAIClient:
    """Return an unconnected native client with real replay ownership."""
    client = object.__new__(OpenAIClient)
    client._base_url = "https://api.openai.com/v1"  # noqa: SLF001
    client._is_azure = False  # noqa: SLF001
    client._stream_retention_policy = StreamRetentionPolicy()  # noqa: SLF001
    client._replay_owners_by_call_id = {}  # noqa: SLF001
    client._active_replay_owners = {}  # noqa: SLF001
    client._active_replay_streams = {}  # noqa: SLF001
    client._active_replay_call_ids = {}  # noqa: SLF001
    client._ambiguous_replay_call_ids = {}  # noqa: SLF001
    client._replay_association_poisoned = False  # noqa: SLF001
    client._closed = False  # noqa: SLF001
    return client


class DurableInteractionStagingContextValidationTest(TestCase):
    """Exercise every fail-closed provider staging boundary."""

    def test_rejects_invalid_identity_and_codec_components(self) -> None:
        valid = _durable_staging_context()
        invalid_values = (
            (
                "task_input_call",
                object(),
                TypeError,
                "task_input_call must",
            ),
            (
                "task_input_call",
                replace(
                    valid.task_input_call,
                    advertisement=TaskInputCapabilityAdvertisement.ATTACHED,
                ),
                ExecutionCorrelationError,
                "durable reserved call",
            ),
            (
                "dispatch_id",
                "wrong-dispatch",
                ExecutionCorrelationError,
                "dispatch does not match",
            ),
            (
                "revision_binding",
                object(),
                TypeError,
                "revision_binding must",
            ),
            (
                "codec_registry",
                object(),
                TypeError,
                "codec registry",
            ),
            (
                "codec",
                object(),
                ExecutionCorrelationError,
                "codec is not registered",
            ),
            (
                "provider_snapshot",
                object(),
                TypeError,
                "continuation snapshot",
            ),
            (
                "provider_snapshot",
                replace(
                    valid.provider_snapshot,
                    snapshot_kind="other.provider-response",
                ),
                ExecutionCorrelationError,
                "snapshot does not match",
            ),
        )
        for field_name, value, error_type, message in invalid_values:
            with (
                self.subTest(field_name=field_name, value=value),
                self.assertRaisesRegex(error_type, message),
            ):
                replace(valid, **cast(Any, {field_name: value}))

    def test_rejects_invalid_provider_correlations(self) -> None:
        valid = _durable_staging_context()
        invalid_values = (
            (
                "provider_idempotency_key",
                1,
                TypeError,
                "must be non-empty",
            ),
            (
                "provider_idempotency_key",
                ProviderIdempotencyKey("wrong-key"),
                ExecutionCorrelationError,
                "does not match durable dispatch",
            ),
            (
                "provider_call_correlation_id",
                "wrong-call",
                ExecutionCorrelationError,
                "does not match the reserved call",
            ),
            (
                "provider_snapshot",
                replace(
                    valid.provider_snapshot,
                    provider_idempotency_key=ProviderIdempotencyKey(
                        "wrong-key"
                    ),
                ),
                ExecutionCorrelationError,
                "changed its idempotency key",
            ),
            (
                "provider_snapshot",
                replace(
                    valid.provider_snapshot,
                    payload={
                        "reserved_capability_call_id": "wrong-call",
                        "replay_items": (),
                    },
                ),
                ExecutionCorrelationError,
                "changed the reserved call",
            ),
        )
        for field_name, value, error_type, message in invalid_values:
            with (
                self.subTest(field_name=field_name, value=value),
                self.assertRaisesRegex(error_type, message),
            ):
                replace(valid, **cast(Any, {field_name: value}))

    def test_rejects_codec_round_trip_drift(self) -> None:
        valid = _durable_staging_context()
        drifted = replace(
            valid.provider_snapshot,
            payload={
                "reserved_capability_call_id": (
                    valid.provider_call_correlation_id
                ),
                "replay_items": ({"type": "reasoning"},),
            },
        )
        with (
            patch.object(
                ContinuationSnapshotCodecRegistry,
                "restore_snapshot",
                return_value=drifted,
            ),
            self.assertRaisesRegex(
                ExecutionCorrelationError,
                "changed durable replay state",
            ),
        ):
            replace(valid)


class _DurableStager:
    """Build one uncommitted portable interaction or fail before staging."""

    def __init__(
        self,
        binding: ContinuationRevisionBinding,
        *,
        failure: BaseException | None = None,
        mutation: str | None = None,
    ) -> None:
        self.binding = binding
        self.failure = failure
        self.mutation = mutation
        self.requests: list[InteractionBrokerRequest] = []
        self.suspensions: list[DurableInteractionSuspension] = []
        self.staging_contexts: list[DurableInteractionStagingContext] = []

    async def __call__(
        self,
        request: InteractionBrokerRequest,
        *,
        execution: AgentExecution,
        response: object,
        stream_sequence: int,
        staging: DurableInteractionStagingContext,
    ) -> DurableInteractionSuspension:
        """Return an exact unpersisted request and continuation."""
        del response
        self.requests.append(request)
        self.staging_contexts.append(staging)
        if self.failure is not None:
            raise self.failure
        created = create_input_request(
            request_id=InputRequestId("durable-wrapper-request"),
            continuation_id=staging.continuation_id,
            origin=request.origin,
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=_NOW,
            continuation_ttl_seconds=request.continuation_ttl_seconds,
            advisory_wait_seconds=request.advisory_wait_seconds,
        )
        continuation = PortableContinuation(
            continuation_id=created.continuation_id,
            request_id=created.request_id,
            origin=created.origin,
            provider_call_id=created.origin.model_call_id,
            provider_call_correlation_id=(
                staging.provider_call_correlation_id
            ),
            definition=created.origin.definition,
            operation_cursor=execution.operation_index,
            generation_settings={},
            transcript=(),
            observations=(),
            revision_binding=staging.revision_binding,
            interaction_count=execution.interaction_count,
            tool_loop_count=0,
            stream_sequence=stream_sequence,
            state_revision=StateRevision(execution.revision),
            store_revision=ContinuationStoreRevision(0),
            created_at=_NOW,
            updated_at=_NOW,
            expires_at=_NOW + timedelta(days=1),
            claim=ContinuationClaim(),
            fencing_token=ContinuationFencingToken(0),
            provider_snapshot=staging.provider_snapshot,
        )
        if self.mutation == "missing_snapshot":
            continuation = replace(
                continuation,
                provider_snapshot=None,
            )
        elif self.mutation == "wrong_snapshot":
            continuation = replace(
                continuation,
                provider_snapshot=replace(
                    staging.provider_snapshot,
                    payload={
                        "reserved_capability_call_id": "wrong-call",
                        "replay_items": (),
                    },
                ),
            )
        elif self.mutation == "wrong_call":
            continuation = replace(
                continuation,
                provider_call_correlation_id="wrong-call",
            )
        elif self.mutation is not None:
            raise AssertionError("unknown durable staging mutation")
        suspension = DurableInteractionSuspension(
            command=CreateInteractionCommand(
                actor=request.actor,
                request=created,
            ),
            continuation=continuation,
        )
        self.suspensions.append(suspension)
        return suspension


class _WrapperSnapshotAdapter:
    """Export a provider-owned snapshot for the exact reserved call."""

    def export_continuation_snapshot(
        self,
        *,
        revision_binding: ContinuationRevisionBinding,
        model_call_id: ModelCallId,
        provider_idempotency_key: ProviderIdempotencyKey,
        provider_call_correlation_id: str,
    ) -> ContinuationSnapshot:
        return ContinuationSnapshot(
            snapshot_kind="wrapper.provider-response",
            revision_binding=revision_binding,
            model_call_id=model_call_id,
            provider_idempotency_key=provider_idempotency_key,
            payload={
                "reserved_capability_call_id": provider_call_correlation_id,
                "replay_items": (),
            },
        )

    def import_continuation_snapshot(
        self,
        snapshot: ContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
    ) -> None:
        assert snapshot.revision_binding == expected_binding
        assert (
            snapshot.payload["reserved_capability_call_id"]
            == provider_call_correlation_id
        )

    def validate_continuation_snapshot_call(
        self,
        snapshot: ContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
        expected_provider_name: str,
        expected_arguments: Mapping[str, object],
    ) -> None:
        assert snapshot.revision_binding == expected_binding
        assert (
            snapshot.payload["reserved_capability_call_id"]
            == provider_call_correlation_id
        )
        assert expected_provider_name == RESERVED_INPUT_CAPABILITY_NAME
        assert set(expected_arguments) == {"mode", "reason", "questions"}


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

    def __init__(
        self,
        plans: list[_ResponsePlan],
        *,
        snapshot_adapter: ProviderContinuationSnapshotAdapter | None = None,
        source_close: Callable[[], None] | None = None,
    ) -> None:
        self.plans = plans
        self.calls: list[ModelCall] = []
        self.snapshot_adapter = snapshot_adapter
        self.source_close = source_close

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        index = len(self.calls)
        self.calls.append(call)
        if index >= len(self.plans):
            raise AssertionError("unexpected model continuation")
        plan = self.plans[index]
        if plan.failure is not None:
            raise plan.failure
        return _provider_response(
            call,
            index,
            plan,
            snapshot_adapter=self.snapshot_adapter,
            source_close=self.source_close,
        )


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
        durable_stager: _DurableStager | None = None,
        durable_support: ProviderCapabilitySupport | None = None,
        snapshot_adapter: ProviderContinuationSnapshotAdapter | None = None,
        source_close: Callable[[], None] | None = None,
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
        self.model_manager = _ScriptedModelManager(
            plans,
            snapshot_adapter=snapshot_adapter,
            source_close=source_close,
        )
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
        engine = _Engine()
        if durable_stager is not None:
            engine.provider_capability_support = (
                durable_support or _durable_support(durable_stager.binding)
            )
        self.agent = _Agent(
            cast(Any, engine),
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
        actor = InteractionActor(
            principal=PrincipalScope(user_id=UserId("wrapper-user"))
        )
        self.runtime = (
            DurableInteractionRuntime(
                actor=actor,
                stager=durable_stager,
                id_factory=UuidExecutionIdFactory(),
                task_id=TaskId("wrapper-task"),
                branch_id=BranchId("wrapper-branch"),
            )
            if durable_stager is not None
            else AttachedInteractionRuntime(
                broker=cast(InteractionBroker, broker),
                actor=actor,
                handler=cast(Any, handler),
                id_factory=UuidExecutionIdFactory(),
                task_id=TaskId("wrapper-task"),
                branch_id=BranchId("wrapper-branch"),
            )
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
    *,
    snapshot_adapter: ProviderContinuationSnapshotAdapter | None = None,
    source_close: Callable[[], None] | None = None,
) -> TextGenerationResponse:
    """Return one canonical provider stream for a scripted model call."""
    capability = call.context.capability
    assert capability is not None
    provider_name = capability.provider_name(
        RESERVED_INPUT_CAPABILITY_NAME,
        provider_family="openai",
    )

    async def provider_items() -> AsyncIterator[CanonicalStreamItem]:
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

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        try:
            async for item in provider_items():
                yield item
        finally:
            if source_close is not None:
                source_close()

    return TextGenerationResponse(
        lambda: items(),
        logger=getLogger(__name__),
        use_async_generator=True,
        continuation_snapshot_adapter=(
            snapshot_adapter or _WrapperSnapshotAdapter()
        ),
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

    async def test_public_sdk_preserves_noncompleted_wrapper_results(
        self,
    ) -> None:
        for wrapper in ("json", "reasoning"):
            for outcome in ("input_required", "cancelled", "failed"):
                with self.subTest(wrapper=wrapper, outcome=outcome):
                    plan = _ResponsePlan(arguments=_input_arguments())
                    if outcome == "cancelled":
                        plan = _ResponsePlan(
                            failure=ExecutionTerminatedError(
                                TerminateInputContinuation(
                                    request_id=InputRequestId(
                                        "wrapper-cancelled"
                                    ),
                                    status=ResolutionStatus.CANCELLED,
                                )
                            )
                        )
                    elif outcome == "failed":
                        plan = _ResponsePlan(
                            failure=RuntimeError("private provider failure")
                        )
                    harness = _Harness(
                        wrapper=wrapper,
                        plans=[plan],
                        broker=_PendingBroker(),
                        handler=_DetachedHandler(),
                    )
                    try:
                        result = await run_agent(
                            cast(Orchestrator, harness.public),
                            _PROMPT,
                            interaction_runtime=harness.runtime,
                        )
                        if outcome == "input_required":
                            self.assertIsInstance(
                                result,
                                AgentRunInputRequired,
                            )
                            assert isinstance(result, AgentRunInputRequired)
                            self.assertFalse(
                                result.detached_resumption_available
                            )
                            self.assertIsNone(result.request_id)
                            self.assertIsNone(result.continuation_id)
                            self.assertEqual(
                                result.request.reason,
                                "Need one bounded decision.",
                            )
                        elif outcome == "cancelled":
                            self.assertIsInstance(result, AgentRunCancelled)
                        else:
                            self.assertIsInstance(result, AgentRunFailed)
                            assert isinstance(result, AgentRunFailed)
                            self.assertEqual(
                                result.code,
                                "agent.execution_failed",
                            )
                            self.assertEqual(
                                result.message,
                                "agent execution failed",
                            )
                            self.assertFalse(result.retryable)
                        self.assertEqual(len(harness.model_manager.calls), 1)
                    finally:
                        await harness.close()

    async def test_public_sdk_preserves_completed_wrapper_values(
        self,
    ) -> None:
        cases = (
            ("json", '{"answer":"structured"}'),
            ("reasoning", "Reasoning: checked\nAnswer: explained"),
        )
        for wrapper, answer in cases:
            with self.subTest(wrapper=wrapper):
                harness = _Harness(
                    wrapper=wrapper,
                    plans=[_ResponsePlan(answer=answer)],
                    broker=_PendingBroker(),
                    handler=_DetachedHandler(),
                )
                try:
                    result = await run_agent(
                        cast(Orchestrator, harness.public),
                        _PROMPT,
                        interaction_runtime=harness.runtime,
                    )
                    self.assertIsInstance(result, AgentRunCompleted)
                    assert isinstance(result, AgentRunCompleted)
                    if wrapper == "json":
                        self.assertIsInstance(
                            result.value,
                            JsonOrchestratorOutput,
                        )
                        self.assertEqual(
                            result.value,
                            '{"answer":"structured"}',
                        )
                        self.assertIs(result.to_str(), result.value)
                        self.assertIs(result.to_json(), result.value)
                    else:
                        self.assertEqual(
                            result.value,
                            ReasoningOrchestratorResponse(
                                answer="explained",
                                reasoning="checked",
                            ),
                        )
                        with self.assertRaises(TypeError):
                            result.to_str()
                        with self.assertRaises(TypeError):
                            result.to_json()
                    self.assertEqual(len(harness.model_manager.calls), 1)
                finally:
                    await harness.close()

    async def test_durable_materialization_stages_without_broker_mutation(
        self,
    ) -> None:
        binding = _durable_binding()
        stager = _DurableStager(binding)
        broker = _PendingBroker()
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=broker,
            handler=_DetachedHandler(),
            durable_stager=stager,
        )
        response: Any | None = None
        try:
            response = await harness.response()
            with self.assertRaises(ExecutionInputRequiredError) as raised:
                await response.to_str()

            failure = raised.exception
            self.assertIs(failure.durable, stager.suspensions[0])
            self.assertTrue(failure.result.detached_resumption_available)
            self.assertEqual(broker.requests, [])
            self.assertEqual(len(stager.requests), 1)
            self.assertEqual(len(stager.suspensions), 1)
            suspension = stager.suspensions[0]
            staging = stager.staging_contexts[0]
            self.assertIs(
                suspension.command.request.state,
                RequestState.CREATED,
            )
            self.assertEqual(
                int(suspension.continuation.store_revision),
                0,
            )
            self.assertEqual(
                suspension.continuation.provider_snapshot,
                staging.provider_snapshot,
            )
            self.assertEqual(
                suspension.command.request.continuation_id,
                staging.continuation_id,
            )
            self.assertEqual(
                staging.provider_idempotency_key,
                derive_provider_idempotency_key(
                    staging.continuation_id,
                    staging.dispatch_id,
                ),
            )
            self.assertEqual(
                staging.provider_snapshot.provider_idempotency_key,
                staging.provider_idempotency_key,
            )
            self.assertEqual(
                suspension.continuation.provider_call_correlation_id,
                str(staging.task_input_call.call_id),
            )
            execution = response.execution
            assert execution is not None
            self.assertIsNone(execution.interaction_broker)
            self.assertIs(
                execution.status,
                AgentExecutionStatus.INPUT_REQUIRED,
            )
            self.assertEqual(
                execution.pending_request,
                suspension.command.request,
            )
            capability = harness.model_manager.calls[0].context.capability
            assert capability is not None
            self.assertIs(
                capability.task_input_advertisement,
                TaskInputCapabilityAdvertisement.DURABLE,
            )
            kinds = {item.kind for item in response.canonical_items}
            self.assertNotIn(StreamItemKind.INTERACTION_CREATED, kinds)
            self.assertNotIn(StreamItemKind.INTERACTION_PENDING, kinds)
            self.assertIn(StreamItemKind.STREAM_INPUT_REQUIRED, kinds)
        finally:
            if response is not None:
                await response.aclose()
            await harness.close()

    async def test_durable_stages_real_openai_owner_before_source_close(
        self,
    ) -> None:
        binding = _durable_binding()
        first_client = _openai_client()
        first_owner = openai_module._OpenAIReplayOwner(  # noqa: SLF001
            first_client._stream_retention_policy  # noqa: SLF001
        )
        first_owner.begin_attempt()
        prior_items = (
            {
                "id": "reasoning-prior",
                "type": "reasoning",
                "encrypted_content": "encrypted-prior-reasoning",
                "summary": [],
            },
            {
                "id": "function-prior",
                "type": "function_call",
                "call_id": "prior-input-call",
                "name": RESERVED_INPUT_CAPABILITY_NAME,
                "arguments": dumps(
                    _input_arguments(),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        )
        for item in prior_items:
            self.assertTrue(first_owner.admit(cast(dict[str, Any], item)))
        first_client._activate_replay_owner(first_owner)  # noqa: SLF001
        first_snapshot = first_client.export_continuation_snapshot(
            revision_binding=binding,
            model_call_id=ModelCallId("prior-model-call"),
            provider_idempotency_key=ProviderIdempotencyKey(
                "prior-provider-retry"
            ),
            provider_call_correlation_id="prior-input-call",
        )
        first_client.validate_continuation_snapshot_call(
            first_snapshot,
            expected_binding=binding,
            provider_call_correlation_id="prior-input-call",
            expected_provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            expected_arguments=cast(
                Mapping[str, Any],
                _input_arguments(),
            ),
        )

        client = _openai_client()
        client.import_continuation_snapshot(
            first_snapshot,
            expected_binding=binding,
            provider_call_correlation_id="prior-input-call",
        )
        owner = client._replay_owners_by_call_id.pop(  # noqa: SLF001
            "prior-input-call"
        )
        client._activate_replay_owner(owner)  # noqa: SLF001
        client._active_replay_call_ids["prior-input-call"] = (
            owner  # noqa: SLF001
        )
        owner.begin_attempt()
        current_items = (
            {
                "id": "reasoning-current",
                "type": "reasoning",
                "encrypted_content": "encrypted-current-reasoning",
                "summary": [],
            },
            {
                "id": "function-current",
                "type": "function_call",
                "call_id": "input-call-0",
                "name": RESERVED_INPUT_CAPABILITY_NAME,
                "arguments": dumps(
                    _input_arguments(),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        )
        for item in current_items:
            self.assertTrue(owner.admit(cast(dict[str, Any], item)))
        stager = _DurableStager(binding)
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=_PendingBroker(),
            handler=_DetachedHandler(),
            durable_stager=stager,
            durable_support=_openai_durable_support(binding),
            snapshot_adapter=client,
            source_close=owner.release,
        )
        response: Any | None = None
        try:
            response = await harness.response()
            with self.assertRaises(ExecutionInputRequiredError):
                await response.to_str()

            self.assertTrue(owner.released)
            self.assertEqual(len(stager.staging_contexts), 1)
            snapshot = stager.staging_contexts[0].provider_snapshot
            items = snapshot.payload["replay_items"]
            self.assertIsInstance(items, tuple)
            assert isinstance(items, tuple)
            prior_reasoning = items[0]
            prior_call = items[1]
            current_reasoning = items[2]
            current_call = items[3]
            assert isinstance(prior_reasoning, Mapping)
            assert isinstance(prior_call, Mapping)
            assert isinstance(current_reasoning, Mapping)
            assert isinstance(current_call, Mapping)
            self.assertEqual(
                prior_reasoning["encrypted_content"],
                "encrypted-prior-reasoning",
            )
            self.assertEqual(prior_call["call_id"], "prior-input-call")
            self.assertEqual(
                current_reasoning["encrypted_content"],
                "encrypted-current-reasoning",
            )
            self.assertEqual(current_call["call_id"], "input-call-0")

            fresh = _openai_client()
            fresh.import_continuation_snapshot(
                snapshot,
                expected_binding=binding,
                provider_call_correlation_id="input-call-0",
            )
            restored = fresh._replay_owners_by_call_id[  # noqa: SLF001
                "input-call-0"
            ].replay_items()
            self.assertEqual(
                restored[0]["encrypted_content"],
                "encrypted-prior-reasoning",
            )
            self.assertEqual(restored[1]["call_id"], "prior-input-call")
            self.assertEqual(
                restored[2]["encrypted_content"],
                "encrypted-current-reasoning",
            )
            self.assertEqual(restored[3]["call_id"], "input-call-0")
        finally:
            if response is not None:
                await response.aclose()
            await harness.close()

    async def test_durable_stream_preserves_exact_staged_payload(
        self,
    ) -> None:
        stager = _DurableStager(_durable_binding())
        broker = _PendingBroker()
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=broker,
            handler=_DetachedHandler(),
            durable_stager=stager,
        )
        response: Any | None = None
        try:
            response = await harness.response()
            with self.assertRaises(ExecutionInputRequiredError) as raised:
                await _consume(response)

            self.assertIs(
                raised.exception.durable,
                stager.suspensions[0],
            )
            self.assertEqual(broker.requests, [])
            self.assertEqual(
                tuple(item.kind for item in response.canonical_items[-2:]),
                (
                    StreamItemKind.STREAM_INPUT_REQUIRED,
                    StreamItemKind.STREAM_CLOSED,
                ),
            )
            execution = response.execution
            assert execution is not None
            self.assertEqual(
                execution.pending_request,
                stager.suspensions[0].command.request,
            )
        finally:
            if response is not None:
                await response.aclose()
            await harness.close()

    async def test_durable_stager_failure_cleans_reservation_and_source(
        self,
    ) -> None:
        staging_failure = RuntimeError("durable stager failed")
        stager = _DurableStager(
            _durable_binding(),
            failure=staging_failure,
        )
        broker = _PendingBroker()
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=broker,
            handler=_DetachedHandler(),
            durable_stager=stager,
        )
        response: Any | None = None
        try:
            response = await harness.response()
            with self.assertRaises(RuntimeError) as raised:
                await response.to_str()

            self.assertIs(raised.exception, staging_failure)
            self.assertEqual(broker.requests, [])
            self.assertEqual(stager.suspensions, [])
            execution = response.execution
            assert execution is not None
            self.assertIs(
                execution.status,
                AgentExecutionStatus.ERRORED,
            )
            self.assertIsNone(execution.pending_request)
            self.assertIsNone(
                execution.snapshot.active_interaction_fingerprint
            )
            self.assertTrue(execution.snapshot.cleanup_started)
            self.assertTrue(response.ownership_cleanup_complete)
            self.assertIn(
                StreamItemKind.STREAM_ERRORED,
                {item.kind for item in response.canonical_items},
            )
        finally:
            if response is not None:
                await response.aclose()
            await harness.close()

    async def test_durable_staging_rejects_incomplete_provider_replay(
        self,
    ) -> None:
        for mutation in (
            "missing_snapshot",
            "wrong_snapshot",
            "wrong_call",
        ):
            with self.subTest(mutation=mutation):
                stager = _DurableStager(
                    _durable_binding(),
                    mutation=mutation,
                )
                broker = _PendingBroker()
                harness = _Harness(
                    wrapper="default",
                    plans=[_ResponsePlan(arguments=_input_arguments())],
                    broker=broker,
                    handler=_DetachedHandler(),
                    durable_stager=stager,
                )
                response: Any | None = None
                try:
                    response = await harness.response()
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "changed provider replay state",
                    ):
                        await response.to_str()

                    self.assertEqual(broker.requests, [])
                    self.assertEqual(len(stager.suspensions), 1)
                    execution = response.execution
                    assert execution is not None
                    self.assertIs(
                        execution.status,
                        AgentExecutionStatus.ERRORED,
                    )
                    self.assertIsNone(execution.pending_request)
                    self.assertIsNone(
                        execution.snapshot.active_interaction_fingerprint
                    )
                    self.assertTrue(response.ownership_cleanup_complete)
                finally:
                    if response is not None:
                        await response.aclose()
                    await harness.close()

    async def test_durable_staging_rejects_missing_provider_adapter(
        self,
    ) -> None:
        stager = _DurableStager(_durable_binding())
        harness = _Harness(
            wrapper="default",
            plans=[_ResponsePlan(arguments=_input_arguments())],
            broker=_PendingBroker(),
            handler=_DetachedHandler(),
            durable_stager=stager,
        )
        response: Any | None = None
        try:
            response = await harness.response()
            response._response._continuation_snapshot_adapter = (
                None  # noqa: SLF001
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "registered provider replay",
            ):
                await response.to_str()

            self.assertEqual(stager.requests, [])
            execution = response.execution
            assert execution is not None
            self.assertIs(execution.status, AgentExecutionStatus.ERRORED)
            self.assertIsNone(execution.pending_request)
            self.assertTrue(response.ownership_cleanup_complete)
        finally:
            if response is not None:
                await response.aclose()
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
