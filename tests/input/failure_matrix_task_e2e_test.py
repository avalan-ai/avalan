# mypy: disable-error-code=import-not-found
"""Exercise canonical input failures across a suspended task boundary."""

from asyncio import Event, Task, create_task, wait_for
from asyncio import run as run_async
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
)
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from logging import Logger, getLogger
from pathlib import Path
from sys import path as sys_path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))

from pgsql_support import (  # noqa: E402
    FakeInteractionCipher,
    FullFakePgsqlDatabase,
)

from avalan.agent.continuation import (
    AgentContinuationEventListener,
    AgentContinuationEventListenerRegistration,
    AgentContinuationResumeCommand,
    DurableAgentContinuationResumer,
)
from avalan.agent.execution import (
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
)
from avalan.agent.loader import OrchestratorLoader
from avalan.entities import TransformerEngineSettings
from avalan.interaction import (
    ActiveControlLeaseNonce,
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationClaimState,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationRuntimeResolver,
    ContinuationSnapshot,
    DurableInteractionSuspension,
    ExecutionDefinitionRef,
    InputDisconnectReason,
    InputErrorCode,
    InputHandler,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputRequest,
    InputRequestId,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionClock,
    InteractionCorrelation,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionNotFoundError,
    InteractionOperation,
    InteractionPolicy,
    InteractionRecord,
    InteractionStoreReplayed,
    InteractionTime,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolvedContinuationRuntime,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    SupersedeInteractionScopeCommand,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TerminalizeDueInteractionsCommand,
    TextAnswer,
)
from avalan.interaction.codec import (
    decode_continuation_snapshot,
    encode_continuation_snapshot,
    encode_input_question,
)
from avalan.interaction.stores.pgsql import (
    PgsqlDurableTaskCoordinator,
    PgsqlInteractionStore,
    PgsqlInteractionStoreError,
    PgsqlInteractionStoreFactory,
)
from avalan.model.call import ModelCall
from avalan.model.capability import (
    ContinuationSnapshotCodecRegistry,
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from avalan.model.modalities import ModalityRegistry
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
)
from avalan.pgsql import PgsqlDatabase
from avalan.task import (
    EncryptedPrivacyValue,
    PrivacyAction,
    SanitizedTaskEvent,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskClient,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInteractionEventType,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskQueueItemState,
    TaskQueueSuspension,
    TaskRunPolicy,
    TaskRunResult,
    TaskRunState,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskWorker,
    TaskWorkerProcessResult,
)
from avalan.task.context import TaskDurableResumeHandle
from avalan.task.durable_agent import DurableAgentTaskHost
from avalan.task.queues import PgsqlTaskQueue
from avalan.task.resume import TaskDurableResumeCoordinator
from avalan.task.stores import PgsqlTaskStore
from avalan.task.target import TaskTargetCompleted, TaskTargetSuspended
from avalan.task.targets import AgentTaskTargetRunner
from avalan.task.targets.agent import AgentOrchestratorLoader
from avalan.types import LooseJsonValue

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
_FAILURE_MATRIX_EVIDENCE_PROPERTY = "failure_matrix_evidence"


class _TestClock(InteractionClock):
    """Return explicitly controlled coherent interaction time."""

    def __init__(self) -> None:
        self.now = _NOW
        self.monotonic = 0.0
        self.changed = Event()

    async def read(self) -> InteractionTime:
        return InteractionTime.from_clock(
            wall_time=self.now,
            monotonic_seconds=self.monotonic,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        while self.monotonic < monotonic_deadline:
            await self.changed.wait()
            self.changed.clear()


class _TestIds(InteractionIdFactory):
    """Mint deterministic opaque interaction identities."""

    def __init__(self) -> None:
        self.value = 0

    def _next(self, kind: str) -> str:
        self.value += 1
        return f"{kind}-{self.value}"

    async def new_request_id(self) -> InputRequestId:
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        return ActiveControlLeaseNonce(self._next("lease"))


class _TestAuthorizer:
    """Allow exact echoed authorization operations."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


class _TestClassifier(TaskInputClassifier):
    """Allow exact normalized answer values."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.value = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        self.value += 1
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"classification-{self.value}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


async def _open_interaction_store(
    database: FullFakePgsqlDatabase,
    *,
    clock: _TestClock,
) -> PgsqlInteractionStore:
    policy = InteractionPolicy()
    return await PgsqlInteractionStoreFactory(
        cast(PgsqlDatabase, database),
        policy=policy,
        clock=clock,
        authorizer=_TestAuthorizer(),
        id_factory=_TestIds(),
        cipher=FakeInteractionCipher(),
        classifier=_TestClassifier(policy),
    ).open()


class _OpenAITransportStream:
    """Yield one scripted provider response without network access."""

    def __init__(self, items: tuple[object, ...]) -> None:
        self._items: Iterator[object] = iter(items)
        self.closed = False

    def __aiter__(self) -> "_OpenAITransportStream":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


class _OpenAIResponsesTransport:
    """Capture real client requests and return scripted SDK streams."""

    def __init__(
        self,
        *,
        plan: tuple[bool, ...],
        arguments: Mapping[str, object],
        call_id_start: int,
    ) -> None:
        self.plan = plan
        self.arguments = arguments
        self.call_id_start = call_id_start
        self.requests: list[dict[str, Any]] = []
        self.streams: list[_OpenAITransportStream] = []

    async def create(self, **kwargs: Any) -> _OpenAITransportStream:
        request_index = len(self.requests)
        if request_index >= len(self.plan):
            raise AssertionError("unexpected OpenAI transport request")
        self.requests.append(dict(kwargs))
        ask_for_input = self.plan[request_index]
        call_id = f"input-call-{self.call_id_start + request_index}"
        if ask_for_input:
            tools = kwargs.get("tools")
            assert isinstance(tools, list)
            matching_tools = [
                tool
                for tool in tools
                if isinstance(tool, dict)
                and tool.get("name") == "request_user_input"
            ]
            assert len(matching_tools) == 1
            tool = matching_tools[0]
            provider_name = tool.get("name")
            assert isinstance(provider_name, str)
            events: tuple[object, ...] = (
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": f"function-{call_id}",
                        "type": "function_call",
                        "call_id": call_id,
                        "name": provider_name,
                        "arguments": dumps(
                            self.arguments,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                },
                {"type": "response.completed", "response": {"usage": {}}},
            )
        else:
            events = (
                {
                    "type": "response.output_text.delta",
                    "delta": "Finished.",
                },
                {"type": "response.output_text.done"},
                {"type": "response.completed", "response": {"usage": {}}},
            )
        stream = _OpenAITransportStream(events)
        self.streams.append(stream)
        return stream


class _ProviderModelManager:
    """Load native adapters and return deterministic provider streams."""

    def __init__(
        self,
        *,
        plan: tuple[bool, ...],
        arguments: Mapping[str, object],
        call_id_start: int,
        use_openai_transport: bool,
    ) -> None:
        self.plan = plan
        self.arguments = arguments
        self.call_id_start = call_id_start
        self.transport = (
            _OpenAIResponsesTransport(
                plan=plan,
                arguments=arguments,
                call_id_start=call_id_start,
            )
            if use_openai_transport
            else None
        )
        self.calls: list[ModelCall] = []
        self.engines: list[OpenAIModel] = []
        self.closed_engines: list[OpenAIModel] = []
        self.exited = False

    def __enter__(self) -> "_ProviderModelManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        _ = exc_type, exc_value, traceback
        self.exited = True
        return False

    @staticmethod
    def parse_uri(uri: str) -> object:
        return ModelManager.parse_uri(uri)

    def get_engine_settings(
        self,
        engine_uri: object,
        settings: Mapping[str, object] | None = None,
        modality: object | None = None,
    ) -> TransformerEngineSettings:
        _ = engine_uri, settings, modality
        return TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="test-only-token",
        )

    def load_engine(
        self,
        engine_uri: object,
        engine_settings: TransformerEngineSettings,
        modality: object,
    ) -> OpenAIModel:
        _ = modality
        model = OpenAIModel(
            model_id=str(getattr(engine_uri, "model_id")),
            settings=engine_settings,
            logger=getLogger(__name__),
        )
        if self.transport is None:
            client = _openai_client()
        else:
            client = OpenAIClient(
                api_key="test-only-token",
                base_url="https://api.openai.com/v1",
            )
            cast(Any, client._client.responses).create = (  # noqa: SLF001
                self.transport.create
            )
            model._exit_stack.push_async_callback(  # noqa: SLF001
                client.aclose
            )
        model._model = client  # noqa: SLF001
        model._exit_stack.push_async_callback(  # noqa: SLF001
            self._record_engine_closed,
            model,
        )
        self.engines.append(model)
        return model

    async def _record_engine_closed(self, model: OpenAIModel) -> None:
        self.closed_engines.append(model)

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        call_index = len(self.calls)
        if call_index >= len(self.plan):
            raise AssertionError("unexpected provider call")
        self.calls.append(call)
        if self.transport is not None:
            handler = ModalityRegistry.get(call.operation.modality)
            response = await handler(
                call.engine_uri,
                call.model,
                call.operation,
                call.capability,
            )
            assert isinstance(response, TextGenerationResponse)
            return response
        ask_for_input = self.plan[call_index]
        model = cast(OpenAIModel, call.model)
        client = cast(OpenAIClient, model.model)
        call_id = f"input-call-{self.call_id_start + call_index}"
        owner: object | None = None
        if ask_for_input:
            owner = client._replay_owner_for_messages(  # noqa: SLF001
                cast(list[Any], call.operation.input)
            )
            replay_owner = cast(Any, owner)
            replay_owner.begin_attempt()
            for item in (
                {
                    "id": f"reasoning-{call_id}",
                    "type": "reasoning",
                    "encrypted_content": f"encrypted-{call_id}",
                    "summary": [],
                },
                {
                    "id": f"function-{call_id}",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "request_user_input",
                    "arguments": dumps(
                        self.arguments,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            ):
                assert replay_owner.admit(item)
            replay_owner.commit_attempt()
        return _provider_response(
            call,
            call_id=call_id,
            client=client,
            owner=owner,
            ask_for_input=ask_for_input,
            arguments=self.arguments,
        )


class _ProviderManagerFactory:
    """Create isolated managers for successive fresh processes."""

    def __init__(
        self,
        plans: list[tuple[bool, ...]],
        *,
        arguments: Mapping[str, object],
        use_openai_transport: bool = False,
    ) -> None:
        self._plans = plans
        self._arguments = arguments
        self._use_openai_transport = use_openai_transport
        self.instances: list[_ProviderModelManager] = []

    def __call__(
        self,
        *args: object,
        **kwargs: object,
    ) -> _ProviderModelManager:
        _ = args, kwargs
        if not self._plans:
            raise AssertionError("unexpected orchestrator process")
        manager = _ProviderModelManager(
            plan=self._plans.pop(0),
            arguments=self._arguments,
            call_id_start=len(self.instances) + 1,
            use_openai_transport=self._use_openai_transport,
        )
        self.instances.append(manager)
        return manager


class _GatedProviderModelManager(_ProviderModelManager):
    """Pause one provider call after recording its exact invocation."""

    def __init__(
        self,
        *,
        plan: tuple[bool, ...],
        arguments: Mapping[str, object],
        call_id_start: int,
        started: Event,
        release: Event,
    ) -> None:
        super().__init__(
            plan=plan,
            arguments=arguments,
            call_id_start=call_id_start,
            use_openai_transport=False,
        )
        self._started = started
        self._release = release

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        """Block the second call until the test explicitly releases it."""
        response = await super().__call__(call)
        if len(self.calls) == 2:
            self._started.set()
            await self._release.wait()
        return response


class _GatedProviderManagerFactory(_ProviderManagerFactory):
    """Create one manager whose continuation call can be observed live."""

    def __init__(
        self,
        plans: list[tuple[bool, ...]],
        *,
        arguments: Mapping[str, object],
    ) -> None:
        super().__init__(plans, arguments=arguments)
        self.started = Event()
        self.release = Event()

    def __call__(
        self,
        *args: object,
        **kwargs: object,
    ) -> _ProviderModelManager:
        _ = args, kwargs
        if not self._plans:
            raise AssertionError("unexpected orchestrator process")
        manager = _GatedProviderModelManager(
            plan=self._plans.pop(0),
            arguments=self._arguments,
            call_id_start=len(self.instances) + 1,
            started=self.started,
            release=self.release,
        )
        self.instances.append(manager)
        return manager


def _openai_client() -> OpenAIClient:
    client = cast(Any, object.__new__(OpenAIClient))
    client._base_url = "https://api.openai.com/v1"
    client._is_azure = False
    client._stream_retention_policy = StreamRetentionPolicy()
    client._replay_owners_by_call_id = {}
    client._active_replay_owners = {}
    client._active_replay_streams = {}
    client._active_replay_call_ids = {}
    client._ambiguous_replay_call_ids = {}
    client._replay_association_poisoned = False
    client._closed = False
    return cast(OpenAIClient, client)


def _provider_response(
    call: ModelCall,
    *,
    call_id: str,
    client: OpenAIClient,
    owner: object | None,
    ask_for_input: bool,
    arguments: Mapping[str, object],
) -> TextGenerationResponse:
    capability = cast(ModelCapabilityCatalog, call.context.capability)
    provider_name = capability.provider_name(
        "request_user_input",
        provider_family="openai",
    )

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        def item(
            sequence: int,
            kind: StreamItemKind,
            channel: StreamChannel,
            *,
            correlation: StreamItemCorrelation | None = None,
            text_delta: str | None = None,
            data: LooseJsonValue | None = None,
            usage: LooseJsonValue | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id=f"stream-{call_id}",
                run_id=f"run-{call_id}",
                turn_id=f"turn-{call_id}",
                provider_family="openai",
                sequence=sequence,
                kind=kind,
                channel=channel,
                correlation=correlation or StreamItemCorrelation(),
                text_delta=text_delta,
                data=data,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        try:
            yield item(
                0,
                StreamItemKind.STREAM_STARTED,
                StreamChannel.CONTROL,
            )
            if ask_for_input:
                correlation = StreamItemCorrelation(tool_call_id=call_id)
                yield item(
                    1,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamChannel.TOOL_CALL,
                    text_delta=dumps(arguments),
                    correlation=correlation,
                )
                yield item(
                    2,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamChannel.TOOL_CALL,
                    data={"name": provider_name},
                    correlation=correlation,
                )
                yield item(
                    3,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamChannel.TOOL_CALL,
                    correlation=correlation,
                )
                sequence = 4
            else:
                yield item(
                    1,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta="Finished.",
                )
                yield item(
                    2,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                )
                sequence = 3
            yield item(
                sequence,
                StreamItemKind.STREAM_COMPLETED,
                StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield item(
                sequence + 1,
                StreamItemKind.STREAM_CLOSED,
                StreamChannel.CONTROL,
            )
        finally:
            if owner is not None:
                cast(Any, owner).release()

    return TextGenerationResponse(
        lambda: items(),
        logger=getLogger(__name__),
        use_async_generator=True,
        continuation_snapshot_adapter=client,
    )


def _write_agent(root: Path) -> Path:
    path = root / "agent.toml"
    path.write_text(
        """
[agent]
instructions = "Request confirmation before continuing."

[engine]
uri = "ai://openai/gpt-5"

[run]
maximum_tool_cycles = 4
block_repeated_tool_calls = true
""",
        encoding="utf-8",
    )
    return path


class _StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        _ = purpose
        return TaskKeyMaterial(
            key_id=key_id or "task-input",
            algorithm="hmac-sha256",
            secret=b"task-failure-matrix-secret",
        )


class _StaticEncryptionProvider:
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        _ = purpose
        return EncryptedPrivacyValue(
            ciphertext=b"encrypted:" + value,
            key_id=key_id or "raw-value",
            algorithm="test-aead",
            metadata=context,
        )

    def decrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        algorithm: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        _ = purpose, key_id, algorithm, context
        prefix = b"encrypted:"
        assert value.startswith(prefix)
        return value[len(prefix) :]


class _AttachedAgentTaskTargetRunner(AgentTaskTargetRunner):
    """Run the production agent target with one attached input runtime."""

    def __init__(
        self,
        loader: OrchestratorLoader,
        *,
        root: Path,
        runtime: AttachedInteractionRuntime,
    ) -> None:
        super().__init__(
            cast(AgentOrchestratorLoader, loader),
            ref_base=root,
        )
        self._attached_runtime = runtime

    async def _interaction_runtime(
        self,
        context: TaskTargetContext,
    ) -> DurableInteractionRuntime | None:
        _ = context
        return cast(DurableInteractionRuntime, self._attached_runtime)


class _UnavailableInputHandler:
    """Report one real attached-host loss at the broker boundary."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerDisconnected:
        self.contexts.append(context)
        return InputHandlerDisconnected(
            reason=InputDisconnectReason.HANDLER_UNAVAILABLE
        )


class _CorrectingInputHandler:
    """Expose one attached validation rejection before a valid correction."""

    def __init__(
        self,
        *,
        database: FullFakePgsqlDatabase,
        model_factory: _ProviderManagerFactory,
        invalid_resolution: Callable[[InputRequest], AnsweredResolution],
        expected_code: InputErrorCode,
    ) -> None:
        self.database = database
        self.model_factory = model_factory
        self.invalid_resolution = invalid_resolution
        self.expected_code = expected_code
        self.contexts: list[InputHandlerContext] = []
        self.failure_task_state: str | None = None
        self.failure_provider_call_count: int | None = None

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Submit one invalid answer, then correct it after direct feedback."""
        self.contexts.append(context)
        if context.validation_error is None:
            return InputHandlerResolution(
                resolution=self.invalid_resolution(context.request)
            )
        assert context.validation_error.code is self.expected_code
        assert context.request.state is RequestState.PENDING
        self.failure_task_state = _only_task_run_state(self.database)
        self.failure_provider_call_count = _provider_call_count(
            self.model_factory
        )
        return InputHandlerResolution(
            resolution=_valid_resolution(context.request)
        )


class _BlockingInputHandler:
    """Expose one direct request until an authoritative external transition."""

    def __init__(self) -> None:
        self.started = Event()
        self.contexts: list[InputHandlerContext] = []
        self._never = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait until broker settlement cancels this attached presentation."""
        self.contexts.append(context)
        self.started.set()
        await self._never.wait()
        raise AssertionError("authoritative settlement did not stop handler")


class _RecordingAgentTaskTargetRunner(AgentTaskTargetRunner):
    """Record outcomes while preserving the production target path."""

    def __init__(
        self,
        loader: OrchestratorLoader,
        *,
        root: Path,
        runtime_factory: Callable[
            [
                TaskTargetContext,
            ],
            DurableInteractionRuntime | Awaitable[DurableInteractionRuntime],
        ],
    ) -> None:
        super().__init__(
            cast(AgentOrchestratorLoader, loader),
            ref_base=root,
            durable_interaction_runtime_factory=runtime_factory,
        )
        self.initial_contexts: list[TaskTargetContext] = []
        self.resume_contexts: list[TaskTargetContext] = []
        self.suspensions: list[DurableInteractionSuspension] = []
        self.domain_side_effects: list[object] = []

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.initial_contexts.append(context)
        outcome = await super().run(context)
        assert isinstance(outcome, TaskTargetSuspended)
        durable = outcome.durable
        assert durable is not None
        self.suspensions.append(durable)
        return outcome

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        self.resume_contexts.append(context)
        outcome = await super().resume(context, durable_resume)
        assert isinstance(outcome, TaskTargetCompleted)
        self.domain_side_effects.append(outcome.output)
        return outcome


class _DirectSuspendingAgentTaskTargetRunner(_RecordingAgentTaskTargetRunner):
    """Expose the agent pause through direct TaskClient observation."""

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        """Return the real agent pause without queued persistence metadata."""
        outcome = await super().run(context)
        assert isinstance(outcome, TaskTargetSuspended)
        return TaskTargetSuspended(
            input_required=outcome.input_required,
            checkpoint_id=outcome.checkpoint_id,
        )


class _ResumeAdapter:
    def __init__(
        self,
        expected_mode: RequirementMode = RequirementMode.REQUIRED,
    ) -> None:
        self.expected_mode = expected_mode
        self.imported: list[ContinuationSnapshot] = []

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
        assert expected_provider_name == "request_user_input"
        assert expected_arguments["mode"] == self.expected_mode.value

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
        self.imported.append(snapshot)


class _EventListenerRegistration:
    def close(self) -> None:
        """Close the no-op failure-matrix registration."""


class _ResumeExecutor:
    trusted_agent_continuation_executor = True

    def __init__(self) -> None:
        self.commands: list[AgentContinuationResumeCommand] = []
        self.closed: list[None] = []

    def register_event_listener(
        self,
        listener: AgentContinuationEventListener,
    ) -> AgentContinuationEventListenerRegistration:
        assert callable(listener)
        return _EventListenerRegistration()

    async def resume_agent_continuation(
        self,
        command: AgentContinuationResumeCommand,
    ) -> object:
        self.commands.append(command)
        return "resumed output"

    async def close_continuation_runtime(self) -> None:
        self.closed.append(None)


class _ResumeLoader:
    trusted_continuation_runtime_loader = True

    def __init__(self, runtime: ResolvedContinuationRuntime) -> None:
        self.runtime = runtime
        self.definitions: list[ExecutionDefinitionRef] = []

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        assert definition == self.runtime.definition
        assert revision_binding == self.runtime.revision_binding
        self.definitions.append(definition)
        return self.runtime


@dataclass(slots=True)
class _ResumeHarness:
    worker: TaskWorker
    adapter: _ResumeAdapter
    executor: _ResumeExecutor
    loader: _ResumeLoader


@dataclass(slots=True)
class _DurableFailureHarness:
    database: FullFakePgsqlDatabase
    clock: _TestClock
    interaction_store: PgsqlInteractionStore
    task_store: PgsqlTaskStore
    queue: PgsqlTaskQueue
    coordinator: PgsqlDurableTaskCoordinator
    client: TaskClient
    target: _RecordingAgentTaskTargetRunner
    model_factory: _ProviderManagerFactory
    stack: AsyncExitStack
    temporary: TemporaryDirectory[str]
    run_id: str
    request: InputRequest
    suspension: TaskQueueSuspension


@dataclass(slots=True)
class _DirectFailureHarness:
    database: FullFakePgsqlDatabase
    clock: _TestClock
    broker: AsyncInteractionBroker
    actor: InteractionActor
    task_store: PgsqlTaskStore
    client: TaskClient
    model_factory: _ProviderManagerFactory
    stack: AsyncExitStack
    temporary: TemporaryDirectory[str]

    async def close(self) -> None:
        """Close every resource owned by the direct failure harness."""
        await self.broker.aclose()
        await self.stack.aclose()
        self.temporary.cleanup()


@dataclass(frozen=True, slots=True)
class _AdvisoryTimeoutObservation:
    """Retain one live task observation at advisory timeout."""

    request: InputRequest
    task_state: TaskRunState
    provider_call_count: int
    domain_side_effect_count: int


@dataclass(frozen=True, slots=True)
class _RequiredHandoffObservation:
    """Retain one task-target suspension exposed by the public task API."""

    task_state: TaskRunState
    provider_call_count: int
    domain_side_effect_count: int


def _task_evidence(
    *,
    condition_id: str,
    surface_id: str,
    transition_from: RequestState,
    transition_to: RequestState,
    public_result_id: str,
    task_state: TaskRunState,
    provider_call_count: int,
    domain_side_effect_count: int,
) -> dict[str, object]:
    """Return one runtime-derived task-target failure observation."""
    return {
        "condition_id": condition_id,
        "surface_id": surface_id,
        "transition_from": transition_from.value,
        "transition_to": transition_to.value,
        "public_result_id": public_result_id,
        "public_result": {
            "interaction_state": transition_to.value,
            "task_state": task_state.value,
        },
        "status_key": "task_state",
        "status_value": task_state.value,
        "provider_call_count": provider_call_count,
        "domain_side_effect_count": domain_side_effect_count,
    }


def _task_observation_evidence(
    *,
    condition_id: str,
    surface_id: str,
    transition_from: RequestState,
    transition_to: RequestState,
    public_result_id: str,
    status_key: str,
    status_value: str,
    provider_call_count: int,
    domain_side_effect_count: int,
) -> dict[str, object]:
    """Return one redacted TaskClient or CLI observation."""
    return {
        "condition_id": condition_id,
        "surface_id": surface_id,
        "transition_from": transition_from.value,
        "transition_to": transition_to.value,
        "public_result_id": public_result_id,
        "public_result": {
            "interaction_state": transition_to.value,
            "redacted": True,
            "channel": "task-client",
        },
        "status_key": status_key,
        "status_value": status_value,
        "provider_call_count": provider_call_count,
        "domain_side_effect_count": domain_side_effect_count,
    }


def _observed_task_interaction_states(
    events: tuple[SanitizedTaskEvent, ...],
) -> tuple[RequestState, ...]:
    """Return interaction states exposed by real TaskClient events."""
    states: list[RequestState] = []
    event_states = {
        TaskInteractionEventType.INPUT_REQUIRED.value: RequestState.PENDING,
        TaskInteractionEventType.INPUT_RESUMED.value: RequestState.ANSWERED,
        TaskInteractionEventType.INPUT_CANCELLED.value: RequestState.CANCELLED,
        TaskInteractionEventType.INPUT_EXPIRED.value: RequestState.EXPIRED,
        TaskInteractionEventType.INPUT_SUPERSEDED.value: (
            RequestState.SUPERSEDED
        ),
    }
    for event in events:
        payload = event.payload
        if isinstance(payload, Mapping):
            lifecycle = payload.get("interaction_lifecycle")
            if isinstance(lifecycle, Mapping):
                state = lifecycle.get("state")
                if isinstance(state, str):
                    request_state = next(
                        (item for item in RequestState if item.value == state),
                        None,
                    )
                    if request_state is not None:
                        states.append(request_state)
        event_state = event_states.get(event.event_type)
        if event_state is not None:
            states.append(event_state)
    assert states
    return tuple(states)


async def _task_client_observation_evidence(
    *,
    condition_id: str,
    suspended: _DurableFailureHarness,
    transition_from: RequestState,
    provider_call_count: int,
) -> list[dict[str, object]]:
    """Observe one real task lifecycle through both TaskClient methods."""
    inspection = await suspended.client.inspect(suspended.run_id)
    inspection_result = "ok"
    events = await suspended.client.events(suspended.run_id)
    events_result = "ok"

    assert events == inspection.events
    assert inspection.output.state is inspection.run.state
    assert all(event.run_id == suspended.run_id for event in events)
    inspect_states = _observed_task_interaction_states(inspection.events)
    event_states = _observed_task_interaction_states(events)
    assert inspect_states == event_states
    assert transition_from in inspect_states
    transition_to = inspect_states[-1]
    public_result_id = f"task.observation_{transition_to.value}.v1"
    domain_side_effect_count = len(inspection.artifacts)

    return [
        _task_observation_evidence(
            condition_id=condition_id,
            surface_id="task-client-inspect",
            transition_from=transition_from,
            transition_to=transition_to,
            public_result_id=public_result_id,
            status_key="client_result",
            status_value=inspection_result,
            provider_call_count=provider_call_count,
            domain_side_effect_count=domain_side_effect_count,
        ),
        _task_observation_evidence(
            condition_id=condition_id,
            surface_id="task-client-events",
            transition_from=transition_from,
            transition_to=transition_to,
            public_result_id=public_result_id,
            status_key="client_result",
            status_value=events_result,
            provider_call_count=provider_call_count,
            domain_side_effect_count=domain_side_effect_count,
        ),
    ]


def _client_cancel_evidence(
    *,
    interaction_state: RequestState,
    client_result: TaskRunState,
    provider_call_count: int,
    domain_side_effect_count: int,
) -> dict[str, object]:
    """Return the runtime-derived TaskClient cancellation observation."""
    return {
        "condition_id": "INPUT-F-10",
        "surface_id": "task-client-cancel",
        "transition_from": RequestState.PENDING.value,
        "transition_to": interaction_state.value,
        "public_result_id": "task.cancellation_ack.v1",
        "public_result": {
            "interaction_state": interaction_state.value,
            "accepted": client_result is TaskRunState.CANCELLED,
            "channel": "task-client",
        },
        "status_key": "client_result",
        "status_value": client_result.value,
        "provider_call_count": provider_call_count,
        "domain_side_effect_count": domain_side_effect_count,
    }


def _record_failure_matrix_evidence(
    record_property: Callable[[str, object], None],
    evidence: list[dict[str, object]],
) -> None:
    """Attach dynamic cell evidence to the pytest call report."""
    assert evidence
    record_property(_FAILURE_MATRIX_EVIDENCE_PROPERTY, evidence)


async def _validation_failure_evidence(
    *,
    condition_id: str,
    suspended: _DurableFailureHarness,
    direct: _DirectFailureHarness,
    handler: _CorrectingInputHandler,
    direct_run_id: str,
) -> list[dict[str, object]]:
    """Return direct and queued validation-rejection postconditions."""
    queue_inspection = await suspended.client.inspect(suspended.run_id)
    direct_inspection = await direct.client.inspect(direct_run_id)
    assert queue_inspection.artifacts == ()
    assert direct_inspection.artifacts == ()
    assert handler.failure_task_state is not None
    assert handler.failure_provider_call_count is not None
    return [
        _task_evidence(
            condition_id=condition_id,
            surface_id="task-target-agent-direct",
            transition_from=handler.contexts[0].request.state,
            transition_to=handler.contexts[1].request.state,
            public_result_id="task.interaction_pending_attached.v1",
            task_state=TaskRunState(handler.failure_task_state),
            provider_call_count=handler.failure_provider_call_count,
            domain_side_effect_count=len(direct_inspection.artifacts),
        ),
        _task_evidence(
            condition_id=condition_id,
            surface_id="task-target-agent-queue",
            transition_from=suspended.request.state,
            transition_to=suspended.request.state,
            public_result_id="task.interaction_pending.v1",
            task_state=queue_inspection.run.state,
            provider_call_count=_provider_call_count(suspended.model_factory),
            domain_side_effect_count=len(queue_inspection.artifacts),
        ),
    ]


def test_input_f_01(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep the task running when no input-capable host is available."""
    evidence = run_async(_test_input_f_01())
    assert len(evidence) == 2
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_01() -> list[dict[str, object]]:
    database = FullFakePgsqlDatabase()
    clock = _TestClock()
    interaction_store = await _open_interaction_store(
        database,
        clock=clock,
    )
    policy = InteractionPolicy()
    broker = AsyncInteractionBroker(
        store=interaction_store,
        clock=clock,
        id_factory=_TestIds(),
        policy=policy,
        classifier=_TestClassifier(policy),
    )
    temporary = TemporaryDirectory()
    root = Path(temporary.name)
    _write_agent(root)
    stack = AsyncExitStack()
    loader = OrchestratorLoader(
        hub=MagicMock(spec=HuggingfaceHub),
        logger=MagicMock(spec=Logger),
        participant_id=uuid4(),
        stack=stack,
    )
    arguments = {
        "mode": RequirementMode.REQUIRED.value,
        "reason": "No attached input host is available.",
        "questions": [encode_input_question(_confirmation())],
    }
    model_factory = _ProviderManagerFactory(
        [(True, False)],
        arguments=arguments,
        use_openai_transport=True,
    )
    handler = _UnavailableInputHandler()
    actor = InteractionActor(principal=PrincipalScope())
    runtime = AttachedInteractionRuntime(
        broker=broker,
        actor=actor,
        handler=handler,
    )
    target = _AttachedAgentTaskTargetRunner(
        loader,
        root=root,
        runtime=runtime,
    )
    task_store = PgsqlTaskStore(
        cast(PgsqlDatabase, database),
        clock=lambda: _NOW,
    )
    client = TaskClient(
        task_store,
        target=target,
        hmac_provider=_StaticHmacProvider(),
        definition_hash=lambda _: "task-failure-matrix-direct",
        clock=lambda: _NOW,
    )

    with patch(
        "avalan.agent.loader.ModelManager",
        side_effect=model_factory,
    ):
        result = await client.run(_definition(), input_value="private")
    inspection = await client.inspect(result.run.run_id)
    records = tuple(database.records.values())

    assert result.run.state is TaskRunState.SUCCEEDED
    assert result.output == "Finished."
    assert len(model_factory.instances) == 1
    manager = model_factory.instances[0]
    assert len(manager.calls) == 2
    transport = manager.transport
    assert transport is not None
    assert len(transport.requests) == 2
    assert len(records) == 1
    second_input = transport.requests[1]["input"]
    assert isinstance(second_input, list)
    function_calls = [
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    function_outputs = [
        item
        for item in second_input
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
    ]
    assert len(function_calls) == 1
    assert len(function_outputs) == 1
    assert function_calls[0]["call_id"] == "input-call-1"
    assert function_outputs[0]["call_id"] == "input-call-1"
    assert function_calls[0]["name"] == "request_user_input"
    assert loads(cast(str, function_calls[0]["arguments"])) == arguments
    unavailable = loads(cast(str, function_outputs[0]["output"]))
    assert set(unavailable) == {
        "kind",
        "provenance",
        "request_id",
        "resolved_at",
    }
    assert unavailable["kind"] == "unavailable"
    assert unavailable["provenance"] == "external_controller"
    assert unavailable["request_id"] == records[0]["request_id"]
    assert unavailable["resolved_at"] == _NOW.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    assert not any(
        isinstance(item, dict) and item.get("role") == "tool"
        for item in second_input
    )
    assert all(stream.closed for stream in transport.streams)
    assert len(handler.contexts) == 1
    assert handler.contexts[0].request.state is RequestState.PENDING
    assert records[0]["request_state"] == RequestState.UNAVAILABLE.value
    assert inspection.run.state is TaskRunState.SUCCEEDED
    assert len(inspection.attempts) == 1
    assert inspection.attempts[0].state is TaskAttemptState.SUCCEEDED
    assert inspection.artifacts == ()
    direct_evidence = _task_evidence(
        condition_id="INPUT-F-01",
        surface_id="task-target-agent-direct",
        transition_from=RequestState.CREATED,
        transition_to=RequestState(records[0]["request_state"]),
        public_result_id="task.interaction_unavailable_completed.v1",
        task_state=inspection.run.state,
        provider_call_count=len(manager.calls),
        domain_side_effect_count=len(inspection.artifacts),
    )
    await broker.aclose()
    await stack.aclose()
    temporary.cleanup()

    queue_database = FullFakePgsqlDatabase()
    queue_clock = _TestClock()
    queue_temporary = TemporaryDirectory()
    queue_root = Path(queue_temporary.name)
    _write_agent(queue_root)
    queue_stack = AsyncExitStack()
    queue_loader = OrchestratorLoader(
        hub=MagicMock(spec=HuggingfaceHub),
        logger=MagicMock(spec=Logger),
        participant_id=uuid4(),
        stack=queue_stack,
    )
    queue_interaction_store = await _open_interaction_store(
        queue_database,
        clock=queue_clock,
    )
    queue_policy = InteractionPolicy()
    queue_broker = AsyncInteractionBroker(
        store=queue_interaction_store,
        clock=queue_clock,
        id_factory=_TestIds(),
        policy=queue_policy,
        classifier=_TestClassifier(queue_policy),
    )
    queue_handler = _UnavailableInputHandler()
    queue_runtime = AttachedInteractionRuntime(
        broker=queue_broker,
        actor=InteractionActor(principal=PrincipalScope()),
        handler=queue_handler,
    )
    queue_target = _AttachedAgentTaskTargetRunner(
        queue_loader,
        root=queue_root,
        runtime=queue_runtime,
    )
    queue_model_factory = _ProviderManagerFactory(
        [(True, False)],
        arguments=arguments,
        use_openai_transport=True,
    )
    queue_store = PgsqlTaskStore(
        cast(PgsqlDatabase, queue_database),
        clock=lambda: queue_clock.now,
    )
    queue = PgsqlTaskQueue(
        cast(PgsqlDatabase, queue_database),
        clock=lambda: queue_clock.now,
    )
    queue_client = TaskClient(
        queue_store,
        target=queue_target,
        queue=queue,
        hmac_provider=_StaticHmacProvider(),
        encryption_provider=_StaticEncryptionProvider(),
        raw_storage_allowed=True,
        definition_hash=lambda _: "task-failure-matrix-queue-unavailable",
        clock=lambda: queue_clock.now,
    )
    submission = await queue_client.enqueue(
        _queued_definition(),
        input_value="private",
    )
    worker = TaskWorker(
        queue_store,
        queue,
        target=queue_target,
        worker_id="failure-matrix-unavailable-worker",
        queue_name="failure-matrix",
        encryption_provider=_StaticEncryptionProvider(),
        raw_storage_allowed=True,
        clock=lambda: queue_clock.now,
    )
    queue_evidence: dict[str, object] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=queue_model_factory,
        ):
            queue_processed = await worker.process_once()
        assert queue_processed.completion is not None
        assert queue_processed.completion.run.run_id == submission.run.run_id
        assert queue_processed.completion.run.state is TaskRunState.SUCCEEDED
        assert (
            queue_processed.completion.attempt.state
            is TaskAttemptState.SUCCEEDED
        )
        assert (
            queue_processed.completion.queue_item.state
            is TaskQueueItemState.DONE
        )
        assert _provider_call_count(queue_model_factory) == 2
        assert len(queue_database.records) == 1
        queue_record = next(iter(queue_database.records.values()))
        assert queue_record["request_state"] == RequestState.UNAVAILABLE.value
        assert len(queue_handler.contexts) == 1
        assert queue_handler.contexts[0].request.state is RequestState.PENDING
        queue_inspection = await queue_client.inspect(submission.run.run_id)
        assert queue_inspection.run.state is TaskRunState.SUCCEEDED
        assert queue_inspection.artifacts == ()
        queue_evidence = _task_evidence(
            condition_id="INPUT-F-01",
            surface_id="task-target-agent-queue",
            transition_from=RequestState.CREATED,
            transition_to=RequestState(queue_record["request_state"]),
            public_result_id="task.interaction_unavailable_completed.v1",
            task_state=queue_inspection.run.state,
            provider_call_count=_provider_call_count(queue_model_factory),
            domain_side_effect_count=len(queue_inspection.artifacts),
        )
    finally:
        await queue_broker.aclose()
        await queue_stack.aclose()
        queue_temporary.cleanup()
    assert queue_evidence is not None
    return [direct_evidence, queue_evidence]


def test_input_f_04(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject a wrong answer type without resuming the task."""
    evidence = run_async(_test_input_f_04())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_04() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(
        ConfirmationQuestion(
            question_id=QuestionId("answer"),
            prompt="Continue?",
            required=True,
        )
    )
    resolution = AnsweredResolution(
        request_id=suspended.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
        answers=(
            TextAnswer(
                question_id=QuestionId("answer"),
                provenance=AnswerProvenance.HUMAN,
                value="yes",
            ),
        ),
    )

    await _assert_durable_resolution_rejected(
        suspended,
        resolution,
        InputErrorCode.ANSWER_TYPE_MISMATCH,
    )
    direct, raw_handler = await _direct_failure_harness(
        ConfirmationQuestion(
            question_id=QuestionId("answer"),
            prompt="Continue?",
            required=True,
        ),
        lambda database, model_factory: _CorrectingInputHandler(
            database=database,
            model_factory=model_factory,
            invalid_resolution=_wrong_type_resolution,
            expected_code=InputErrorCode.ANSWER_TYPE_MISMATCH,
        ),
    )
    handler = cast(_CorrectingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            result = await direct.client.run(
                _definition(),
                input_value="private",
            )
        assert len(handler.contexts) == 2
        correction = handler.contexts[1].validation_error
        assert correction is not None
        assert correction.code is InputErrorCode.ANSWER_TYPE_MISMATCH
        assert handler.contexts[1].request.state is RequestState.PENDING
        assert handler.failure_task_state == TaskRunState.RUNNING.value
        assert handler.failure_provider_call_count == 1
        assert result.run.state is TaskRunState.SUCCEEDED
        assert _provider_call_count(direct.model_factory) == 2
        evidence = [
            *await _validation_failure_evidence(
                condition_id="INPUT-F-04",
                suspended=suspended,
                direct=direct,
                handler=handler,
                direct_run_id=result.run.run_id,
            ),
            *await _task_client_observation_evidence(
                condition_id="INPUT-F-04",
                suspended=suspended,
                transition_from=suspended.request.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
            ),
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_05(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject an unknown choice without resuming the task."""
    evidence = run_async(_test_input_f_05())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_05() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(
        SingleSelectionQuestion(
            question_id=QuestionId("answer"),
            prompt="Choose.",
            required=True,
            choices=(
                Choice(value=ChoiceValue("known"), label="Known"),
                Choice(value=ChoiceValue("other"), label="Other choice"),
            ),
        )
    )
    resolution = AnsweredResolution(
        request_id=suspended.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
        answers=(
            SingleSelectionAnswer(
                question_id=QuestionId("answer"),
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("unknown")),
            ),
        ),
    )

    await _assert_durable_resolution_rejected(
        suspended,
        resolution,
        InputErrorCode.UNKNOWN_CHOICE,
    )
    question = SingleSelectionQuestion(
        question_id=QuestionId("answer"),
        prompt="Choose.",
        required=True,
        choices=(
            Choice(value=ChoiceValue("known"), label="Known"),
            Choice(value=ChoiceValue("other"), label="Other choice"),
        ),
    )
    direct, raw_handler = await _direct_failure_harness(
        question,
        lambda database, model_factory: _CorrectingInputHandler(
            database=database,
            model_factory=model_factory,
            invalid_resolution=_unknown_choice_resolution,
            expected_code=InputErrorCode.UNKNOWN_CHOICE,
        ),
    )
    handler = cast(_CorrectingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            result = await direct.client.run(
                _definition(),
                input_value="private",
            )
        assert len(handler.contexts) == 2
        correction = handler.contexts[1].validation_error
        assert correction is not None
        assert correction.code is InputErrorCode.UNKNOWN_CHOICE
        assert handler.contexts[1].request.state is RequestState.PENDING
        assert handler.failure_task_state == TaskRunState.RUNNING.value
        assert handler.failure_provider_call_count == 1
        assert result.run.state is TaskRunState.SUCCEEDED
        assert _provider_call_count(direct.model_factory) == 2
        evidence = [
            *await _validation_failure_evidence(
                condition_id="INPUT-F-05",
                suspended=suspended,
                direct=direct,
                handler=handler,
                direct_run_id=result.run.run_id,
            ),
            *await _task_client_observation_evidence(
                condition_id="INPUT-F-05",
                suspended=suspended,
                transition_from=suspended.request.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
            ),
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_06(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject a missing required answer without resuming the task."""
    evidence = run_async(_test_input_f_06())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_06() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(
        ConfirmationQuestion(
            question_id=QuestionId("answer"),
            prompt="Continue?",
            required=True,
        )
    )
    resolution = AnsweredResolution(
        request_id=suspended.request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
        answers=(),
    )

    await _assert_durable_resolution_rejected(
        suspended,
        resolution,
        InputErrorCode.MISSING_REQUIRED_ANSWER,
    )
    direct, raw_handler = await _direct_failure_harness(
        ConfirmationQuestion(
            question_id=QuestionId("answer"),
            prompt="Continue?",
            required=True,
        ),
        lambda database, model_factory: _CorrectingInputHandler(
            database=database,
            model_factory=model_factory,
            invalid_resolution=_missing_answer_resolution,
            expected_code=InputErrorCode.MISSING_REQUIRED_ANSWER,
        ),
    )
    handler = cast(_CorrectingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            result = await direct.client.run(
                _definition(),
                input_value="private",
            )
        assert len(handler.contexts) == 2
        correction = handler.contexts[1].validation_error
        assert correction is not None
        assert correction.code is InputErrorCode.MISSING_REQUIRED_ANSWER
        assert handler.contexts[1].request.state is RequestState.PENDING
        assert handler.failure_task_state == TaskRunState.RUNNING.value
        assert handler.failure_provider_call_count == 1
        assert result.run.state is TaskRunState.SUCCEEDED
        assert _provider_call_count(direct.model_factory) == 2
        evidence = [
            *await _validation_failure_evidence(
                condition_id="INPUT-F-06",
                suspended=suspended,
                direct=direct,
                handler=handler,
                direct_run_id=result.run.run_id,
            ),
            *await _task_client_observation_evidence(
                condition_id="INPUT-F-06",
                suspended=suspended,
                transition_from=suspended.request.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
            ),
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_07(
    record_property: Callable[[str, object], None],
) -> None:
    """Treat an identical answer replay as one accepted resolution."""
    evidence = run_async(_test_input_f_07())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_07() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(_confirmation())
    command = _resolution_command(
        suspended,
        value=True,
        key="identical-answer",
    )

    accepted = await suspended.coordinator.resolve_and_requeue(
        command,
        task_run_id=suspended.run_id,
        now=suspended.clock.now,
    )
    replay = await suspended.coordinator.resolve_and_requeue(
        command,
        task_run_id=suspended.run_id,
        now=suspended.clock.now,
    )
    queue_boundary = await suspended.client.inspect(suspended.run_id)
    assert queue_boundary.run.state is TaskRunState.QUEUED
    assert _provider_call_count(suspended.model_factory) == 1
    runtime = await _resume_harness(suspended)
    processed = await runtime.worker.process_once()
    no_duplicate = await runtime.worker.process_once()
    inspection = await suspended.client.inspect(suspended.run_id)
    continuation = await suspended.interaction_store.get_continuation(
        suspended.request.continuation_id
    )
    queue_row = _persisted_queue_row(suspended)
    segments = await suspended.task_store.list_attempt_segments(
        suspended.suspension.attempt.attempt_id
    )
    depth = await suspended.queue.depth(
        "failure-matrix",
        now=suspended.clock.now,
    )
    outbox_rows = tuple(suspended.database.outbox.values())

    assert isinstance(accepted.resolution, ResolveInteractionApplied)
    assert accepted.resolution.record.request.resolution is not None
    assert (
        accepted.resolution.record.request.resolution.resolved_at
        == suspended.request.created_at
        == _NOW
    )
    assert isinstance(replay.resolution, InteractionStoreReplayed)
    assert replay.resolution.record == accepted.resolution.record
    assert replay.reentry == accepted.reentry
    assert processed.completion is not None
    assert processed.completion.run.state is TaskRunState.SUCCEEDED
    assert processed.completion.attempt.state is TaskAttemptState.SUCCEEDED
    assert processed.completion.queue_item.state is TaskQueueItemState.DONE
    assert not no_duplicate.processed
    assert depth.active == 0
    assert depth.claimed == 0
    assert queue_row["state"] == TaskQueueItemState.DONE.value
    assert queue_row["claim_token"] is None
    assert len(outbox_rows) == 1
    assert outbox_rows[0]["status"] == "dead"
    assert inspection.run.state is TaskRunState.SUCCEEDED
    assert len(inspection.attempts) == 1
    assert inspection.attempts[0].attempt_number == 1
    assert tuple(segment.segment_number for segment in segments) == (1, 2)
    assert segments[1].resumed_from_segment_id == segments[0].segment_id
    assert len(runtime.executor.commands) == 1
    assert len(runtime.adapter.imported) == 1
    assert len(runtime.loader.definitions) == 1
    assert len(runtime.executor.closed) == 1
    assert len(suspended.target.initial_contexts) == 1
    assert len(suspended.target.resume_contexts) == 1
    assert suspended.target.domain_side_effects == ["resumed output"]
    assert continuation.claim.state is ContinuationClaimState.COMPLETED
    event_types = tuple(event.event_type for event in inspection.events)
    assert (
        event_types.count(TaskInteractionEventType.INPUT_REQUIRED.value) == 1
    )
    assert event_types.count(TaskInteractionEventType.INPUT_RESUMED.value) == 1

    direct, raw_handler = await _direct_failure_harness(
        _confirmation(),
        lambda _database, _model_factory: _BlockingInputHandler(),
    )
    handler = cast(_BlockingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            running = create_task(
                direct.client.run(_definition(), input_value="private")
            )
            await handler.started.wait()
            request = handler.contexts[0].request
            assert request.state is RequestState.PENDING
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            direct_command = _direct_resolution_command(
                direct,
                request,
                value=True,
                key="direct-identical-answer",
            )
            direct_accepted = await direct.broker.resolve(direct_command)
            direct_replay = await direct.broker.resolve(direct_command)
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            direct_result = await running
        assert isinstance(
            direct_accepted.store_result,
            ResolveInteractionApplied,
        )
        assert isinstance(
            direct_replay.store_result,
            InteractionStoreReplayed,
        )
        assert (
            direct_replay.store_result.record
            == direct_accepted.store_result.record
        )
        assert direct_result.run.state is TaskRunState.SUCCEEDED
        assert _provider_call_count(direct.model_factory) == 2
        direct_inspection = await direct.client.inspect(
            direct_result.run.run_id
        )
        assert direct_inspection.artifacts == ()
        assert inspection.artifacts == ()
        client_provider_call_count = _provider_call_count(
            suspended.model_factory
        ) + len(runtime.executor.commands)
        client_evidence = await _task_client_observation_evidence(
            condition_id="INPUT-F-07",
            suspended=suspended,
            transition_from=accepted.resolution.record.request.state,
            provider_call_count=client_provider_call_count,
        )
        evidence = [
            _task_evidence(
                condition_id="INPUT-F-07",
                surface_id="task-target-agent-direct",
                transition_from=(
                    direct_accepted.store_result.record.request.state
                ),
                transition_to=(
                    direct_replay.store_result.record.request.state
                ),
                public_result_id="task.interaction_answered.v1",
                task_state=TaskRunState.RUNNING,
                provider_call_count=_provider_call_count(direct.model_factory),
                domain_side_effect_count=len(direct_inspection.artifacts),
            ),
            _task_evidence(
                condition_id="INPUT-F-07",
                surface_id="task-target-agent-queue",
                transition_from=(accepted.resolution.record.request.state),
                transition_to=replay.resolution.record.request.state,
                public_result_id="task.interaction_answered_queued.v1",
                task_state=queue_boundary.run.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
                domain_side_effect_count=len(inspection.artifacts),
            ),
            *client_evidence,
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_08(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject a conflicting answer after the winning resolution."""
    evidence = run_async(_test_input_f_08())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_08() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(_confirmation())
    winner = _resolution_command(
        suspended,
        value=True,
        key="winning-answer",
    )
    accepted = await suspended.coordinator.resolve_and_requeue(
        winner,
        task_run_id=suspended.run_id,
        now=suspended.clock.now,
    )
    assert isinstance(accepted.resolution, ResolveInteractionApplied)
    assert accepted.resolution.record.request.resolution is not None
    assert (
        accepted.resolution.record.request.resolution.resolved_at
        == suspended.clock.now
    )
    conflict = await suspended.interaction_store.resolve(
        _resolution_command(
            suspended,
            value=False,
            key="conflicting-answer",
            expected_state_revision=(
                accepted.resolution.record.request.state_revision
            ),
        )
    )
    before_resume = await suspended.client.inspect(suspended.run_id)
    queue_conflict_request = await _persisted_request(suspended)
    assert queue_conflict_request.state is RequestState.ANSWERED
    assert _provider_call_count(suspended.model_factory) == 1
    runtime = await _resume_harness(suspended)
    processed = await runtime.worker.process_once()
    inspection = await suspended.client.inspect(suspended.run_id)

    assert isinstance(conflict, ResolveInteractionRejected)
    assert conflict.error.code is InputErrorCode.ALREADY_RESOLVED
    assert not conflict.store_mutation_applied
    assert before_resume.run.state is TaskRunState.QUEUED
    assert processed.completion is not None
    assert processed.completion.run.state is TaskRunState.SUCCEEDED
    assert inspection.run.state is TaskRunState.SUCCEEDED
    assert len(runtime.executor.commands) == 1
    command = runtime.executor.commands[0]
    assert command.request == accepted.resolution.record.request
    assert command.model_result.request_id == suspended.request.request_id
    assert len(suspended.target.resume_contexts) == 1
    assert suspended.target.domain_side_effects == ["resumed output"]

    direct, raw_handler = await _direct_failure_harness(
        _confirmation(),
        lambda _database, _model_factory: _BlockingInputHandler(),
    )
    handler = cast(_BlockingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            running = create_task(
                direct.client.run(_definition(), input_value="private")
            )
            await handler.started.wait()
            request = handler.contexts[0].request
            assert request.state is RequestState.PENDING
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            direct_accepted = await direct.broker.resolve(
                _direct_resolution_command(
                    direct,
                    request,
                    value=True,
                    key="direct-winning-answer",
                )
            )
            assert isinstance(
                direct_accepted.store_result,
                ResolveInteractionApplied,
            )
            accepted_record = direct_accepted.store_result.record
            direct_conflict = await direct.broker.resolve(
                _direct_resolution_command(
                    direct,
                    request,
                    value=False,
                    key="direct-conflicting-answer",
                    expected_state_revision=(
                        accepted_record.request.state_revision
                    ),
                )
            )
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            direct_conflict_state = RequestState(
                cast(
                    str,
                    next(iter(direct.database.records.values()))[
                        "request_state"
                    ],
                )
            )
            direct_result = await running
        assert isinstance(
            direct_conflict.store_result,
            ResolveInteractionRejected,
        )
        assert direct_conflict.store_result.error.code is (
            InputErrorCode.ALREADY_RESOLVED
        )
        assert not direct_conflict.store_result.store_mutation_applied
        assert direct_result.run.state is TaskRunState.SUCCEEDED
        assert _provider_call_count(direct.model_factory) == 2
        direct_inspection = await direct.client.inspect(
            direct_result.run.run_id
        )
        assert direct_inspection.artifacts == ()
        assert inspection.artifacts == ()
        client_provider_call_count = _provider_call_count(
            suspended.model_factory
        ) + len(runtime.executor.commands)
        client_evidence = await _task_client_observation_evidence(
            condition_id="INPUT-F-08",
            suspended=suspended,
            transition_from=accepted.resolution.record.request.state,
            provider_call_count=client_provider_call_count,
        )
        evidence = [
            _task_evidence(
                condition_id="INPUT-F-08",
                surface_id="task-target-agent-direct",
                transition_from=accepted_record.request.state,
                transition_to=direct_conflict_state,
                public_result_id="task.interaction_answered.v1",
                task_state=TaskRunState.RUNNING,
                provider_call_count=_provider_call_count(direct.model_factory),
                domain_side_effect_count=len(direct_inspection.artifacts),
            ),
            _task_evidence(
                condition_id="INPUT-F-08",
                surface_id="task-target-agent-queue",
                transition_from=accepted.resolution.record.request.state,
                transition_to=queue_conflict_request.state,
                public_result_id="task.interaction_answered_queued.v1",
                task_state=before_resume.run.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
                domain_side_effect_count=len(inspection.artifacts),
            ),
            *client_evidence,
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_09(
    record_property: Callable[[str, object], None],
) -> None:
    """Expire the suspended task when its continuation expires."""
    evidence = run_async(_test_input_f_09())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_09() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(_confirmation())
    deadline = suspended.request.created_at + timedelta(
        seconds=suspended.request.continuation_ttl_seconds
    )
    suspended.clock.now = deadline - timedelta(microseconds=1)
    suspended.clock.monotonic = (
        suspended.request.continuation_ttl_seconds - 0.000001
    )

    before_deadline = await suspended.coordinator.expire_suspended_task(
        TerminalizeDueInteractionsCommand(),
        task_run_id=suspended.run_id,
        now=suspended.clock.now,
    )
    before_request = await _persisted_request(suspended)

    assert before_deadline.completion_for(suspended.run_id) is None
    assert before_request.state is RequestState.PENDING

    suspended.clock.now = deadline
    suspended.clock.monotonic = float(
        suspended.request.continuation_ttl_seconds
    )

    lifecycle = await suspended.coordinator.expire_suspended_task(
        TerminalizeDueInteractionsCommand(),
        task_run_id=suspended.run_id,
        now=suspended.clock.now,
    )
    late_answer_code = None
    try:
        await suspended.coordinator.resolve_and_requeue(
            _resolution_command(
                suspended,
                value=True,
                key="answer-after-expiry",
            ),
            task_run_id=suspended.run_id,
            now=suspended.clock.now,
        )
    except PgsqlInteractionStoreError as error:
        late_answer_code = error.code
    completion = lifecycle.completion_for(suspended.run_id)
    inspection = await suspended.client.inspect(suspended.run_id)
    persisted = await _persisted_request(suspended)
    continuation = await suspended.interaction_store.get_continuation(
        suspended.request.continuation_id
    )
    segments = await suspended.task_store.list_attempt_segments(
        suspended.suspension.attempt.attempt_id
    )
    queue_row = _persisted_queue_row(suspended)

    assert completion is not None
    assert completion.run.state is TaskRunState.EXPIRED
    assert completion.attempt.state is TaskAttemptState.FAILED
    assert completion.queue_item.state is TaskQueueItemState.DEAD
    assert persisted.state is RequestState.EXPIRED
    assert persisted.resolution is not None
    assert persisted.resolution.resolved_at == deadline
    assert late_answer_code is InputErrorCode.ILLEGAL_TRANSITION
    assert inspection.run.state is TaskRunState.EXPIRED
    assert inspection.attempts[0].state is TaskAttemptState.FAILED
    assert len(segments) == 1
    assert segments[0].state is TaskAttemptSegmentState.SUSPENDED
    assert continuation.claim.state is ContinuationClaimState.UNCLAIMED
    assert _continuation_lifecycle(suspended) == "invalidated"
    assert queue_row["state"] == TaskQueueItemState.DEAD.value
    assert queue_row["claim_token"] is None
    assert suspended.target.resume_contexts == []
    assert suspended.target.domain_side_effects == []
    assert (
        tuple(event.event_type for event in inspection.events).count(
            TaskInteractionEventType.INPUT_EXPIRED.value
        )
        == 1
    )
    try:
        await suspended.interaction_store.get_task_continuation_record(
            suspended.run_id
        )
    except InteractionNotFoundError:
        pass
    else:
        raise AssertionError("expired continuation remained active")

    direct, raw_handler = await _direct_failure_harness(
        _confirmation(),
        lambda _database, _model_factory: _BlockingInputHandler(),
    )
    handler = cast(_BlockingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            running = create_task(
                direct.client.run(_definition(), input_value="private")
            )
            await handler.started.wait()
            request = handler.contexts[0].request
            assert request.state is RequestState.PENDING
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            direct.clock.now = request.created_at + timedelta(
                seconds=request.continuation_ttl_seconds
            )
            direct.clock.monotonic = float(request.continuation_ttl_seconds)
            direct.clock.changed.set()
            direct_result = await wait_for(running, timeout=1)
        assert direct_result.run.state is TaskRunState.FAILED
        assert _provider_call_count(direct.model_factory) == 1
        direct_record = next(iter(direct.database.records.values()))
        assert direct_record["request_state"] == RequestState.EXPIRED.value
        direct_inspection = await direct.client.inspect(
            direct_result.run.run_id
        )
        assert direct_inspection.artifacts == ()
        assert inspection.artifacts == ()
        client_evidence = await _task_client_observation_evidence(
            condition_id="INPUT-F-09",
            suspended=suspended,
            transition_from=suspended.request.state,
            provider_call_count=_provider_call_count(suspended.model_factory),
        )
        evidence = [
            _task_evidence(
                condition_id="INPUT-F-09",
                surface_id="task-target-agent-direct",
                transition_from=request.state,
                transition_to=RequestState(direct_record["request_state"]),
                public_result_id="task.interaction_expired_attached.v1",
                task_state=direct_result.run.state,
                provider_call_count=_provider_call_count(direct.model_factory),
                domain_side_effect_count=len(direct_inspection.artifacts),
            ),
            _task_evidence(
                condition_id="INPUT-F-09",
                surface_id="task-target-agent-queue",
                transition_from=suspended.request.state,
                transition_to=persisted.state,
                public_result_id="task.interaction_expired.v1",
                task_state=inspection.run.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
                domain_side_effect_count=len(inspection.artifacts),
            ),
            *client_evidence,
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_10(
    record_property: Callable[[str, object], None],
) -> None:
    """Cancel both the pending interaction and containing task."""
    evidence = run_async(_test_input_f_10())
    assert len(evidence) == 5
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_10() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(_confirmation())

    cancelled = await suspended.client.cancel(suspended.run_id)
    replayed = await suspended.client.cancel(suspended.run_id)
    inspection = await suspended.client.inspect(suspended.run_id)
    persisted = await _persisted_request(suspended)
    continuation = await suspended.interaction_store.get_continuation(
        suspended.request.continuation_id
    )
    segments = await suspended.task_store.list_attempt_segments(
        suspended.suspension.attempt.attempt_id
    )
    queue_row = _persisted_queue_row(suspended)
    depth = await suspended.queue.depth(
        "failure-matrix",
        now=suspended.clock.now,
    )

    assert cancelled.state is TaskRunState.CANCELLED
    assert replayed == cancelled
    assert persisted.state is RequestState.CANCELLED
    assert persisted.resolution is not None
    assert persisted.resolution.resolved_at == suspended.clock.now
    assert inspection.run.state is TaskRunState.CANCELLED
    assert inspection.attempts[0].state is TaskAttemptState.ABANDONED
    assert depth.dead == 1
    assert len(segments) == 1
    assert segments[0].state is TaskAttemptSegmentState.SUSPENDED
    assert continuation.claim.state is ContinuationClaimState.UNCLAIMED
    assert _continuation_lifecycle(suspended) == "invalidated"
    assert queue_row["state"] == TaskQueueItemState.DEAD.value
    assert queue_row["claim_token"] is None
    assert suspended.target.resume_contexts == []
    assert suspended.target.domain_side_effects == []
    assert (
        tuple(event.event_type for event in inspection.events).count(
            TaskInteractionEventType.INPUT_CANCELLED.value
        )
        == 1
    )
    try:
        await suspended.interaction_store.get_task_continuation_record(
            suspended.run_id
        )
    except InteractionNotFoundError:
        pass
    else:
        raise AssertionError("cancelled continuation remained active")

    direct, raw_handler = await _direct_failure_harness(
        _confirmation(),
        lambda _database, _model_factory: _BlockingInputHandler(),
    )
    handler = cast(_BlockingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            running = create_task(
                direct.client.run(_definition(), input_value="private")
            )
            await handler.started.wait()
            request = handler.contexts[0].request
            assert request.state is RequestState.PENDING
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            direct_run_id = _only_task_run_id(direct.database)
            cancel_requested = await direct.client.cancel(direct_run_id)
            assert cancel_requested.state is TaskRunState.CANCEL_REQUESTED
            direct_result = await wait_for(running, timeout=1)
        assert direct_result.run.state is TaskRunState.CANCELLED
        assert _provider_call_count(direct.model_factory) == 1
        direct_inspection = await direct.client.inspect(
            direct_result.run.run_id
        )
        direct_record = await direct.broker.inspect(
            ScopedInteractionLookup(
                actor=direct.actor,
                correlation=InteractionCorrelation.from_request(request),
            )
        )
        assert isinstance(direct_record, InteractionRecord)
        assert direct_record.request.state is RequestState.CANCELLED
        assert direct_record.request.resolution is not None
        assert direct_record.request.resolution.resolved_at == direct.clock.now
        assert len(direct_inspection.attempts) == 1
        assert direct_inspection.attempts[0].state is TaskAttemptState.FAILED
        direct_segments = await direct.task_store.list_attempt_segments(
            direct_inspection.attempts[0].attempt_id
        )
        assert len(direct_segments) == 1
        assert direct_segments[0].state is TaskAttemptSegmentState.ABANDONED
        assert direct_inspection.artifacts == ()
        assert inspection.artifacts == ()
        client_evidence = await _task_client_observation_evidence(
            condition_id="INPUT-F-10",
            suspended=suspended,
            transition_from=suspended.request.state,
            provider_call_count=_provider_call_count(suspended.model_factory),
        )
        evidence = [
            _task_evidence(
                condition_id="INPUT-F-10",
                surface_id="task-target-agent-direct",
                transition_from=request.state,
                transition_to=direct_record.request.state,
                public_result_id="task.interaction_cancelled.v1",
                task_state=direct_result.run.state,
                provider_call_count=_provider_call_count(direct.model_factory),
                domain_side_effect_count=len(direct_inspection.artifacts),
            ),
            _task_evidence(
                condition_id="INPUT-F-10",
                surface_id="task-target-agent-queue",
                transition_from=suspended.request.state,
                transition_to=persisted.state,
                public_result_id="task.interaction_cancelled.v1",
                task_state=inspection.run.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
                domain_side_effect_count=len(inspection.artifacts),
            ),
            *client_evidence,
            _client_cancel_evidence(
                interaction_state=persisted.state,
                client_result=cancelled.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
                domain_side_effect_count=len(inspection.artifacts),
            ),
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_11(
    record_property: Callable[[str, object], None],
) -> None:
    """Supersede pending input and terminalize its containing task."""
    evidence = run_async(_test_input_f_11())
    assert len(evidence) == 4
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_11() -> list[dict[str, object]]:
    suspended = await _durable_failure_harness(_confirmation())
    command = SupersedeInteractionScopeCommand(
        actor=suspended.target.suspensions[0].command.actor,
        scope=InteractionExecutionScope(
            run_id=RunId(suspended.run_id),
        ),
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
    )
    lifecycle = await suspended.coordinator.supersede_suspended_task(
        command,
        task_run_id=suspended.run_id,
        now=suspended.clock.now,
    )
    completion = lifecycle.completion_for(suspended.run_id)
    inspection = await suspended.client.inspect(suspended.run_id)
    persisted = await _persisted_request(suspended)
    continuation = await suspended.interaction_store.get_continuation(
        suspended.request.continuation_id
    )
    segments = await suspended.task_store.list_attempt_segments(
        suspended.suspension.attempt.attempt_id
    )
    queue_row = _persisted_queue_row(suspended)

    assert completion is not None
    assert completion.run.state is TaskRunState.CANCELLED
    assert completion.attempt.state is TaskAttemptState.ABANDONED
    assert completion.queue_item.state is TaskQueueItemState.DEAD
    assert persisted.state is RequestState.SUPERSEDED
    assert persisted.resolution is not None
    assert persisted.resolution.resolved_at == suspended.clock.now
    assert inspection.run.state is TaskRunState.CANCELLED
    assert inspection.attempts[0].state is TaskAttemptState.ABANDONED
    assert len(segments) == 1
    assert segments[0].state is TaskAttemptSegmentState.SUSPENDED
    assert continuation.claim.state is ContinuationClaimState.UNCLAIMED
    assert _continuation_lifecycle(suspended) == "invalidated"
    assert queue_row["state"] == TaskQueueItemState.DEAD.value
    assert queue_row["claim_token"] is None
    assert suspended.target.resume_contexts == []
    assert suspended.target.domain_side_effects == []
    assert (
        tuple(event.event_type for event in inspection.events).count(
            TaskInteractionEventType.INPUT_SUPERSEDED.value
        )
        == 1
    )
    try:
        await suspended.interaction_store.get_task_continuation_record(
            suspended.run_id
        )
    except InteractionNotFoundError:
        pass
    else:
        raise AssertionError("superseded continuation remained active")
    record_row = suspended.database.records[str(suspended.request.request_id)]
    retention_deadline = record_row["retention_deadline_at"]
    assert isinstance(retention_deadline, datetime)
    swept = await suspended.coordinator.sweep_retention(now=retention_deadline)
    assert swept.deleted == (suspended.request.continuation_id,)
    try:
        await suspended.interaction_store.get_continuation(
            suspended.request.continuation_id
        )
    except InteractionNotFoundError:
        pass
    else:
        raise AssertionError("superseded continuation survived retention")

    direct, raw_handler = await _direct_failure_harness(
        _confirmation(),
        lambda _database, _model_factory: _BlockingInputHandler(),
    )
    handler = cast(_BlockingInputHandler, raw_handler)
    evidence: list[dict[str, object]] | None = None
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=direct.model_factory,
        ):
            running = create_task(
                direct.client.run(_definition(), input_value="private")
            )
            await handler.started.wait()
            request = handler.contexts[0].request
            assert request.state is RequestState.PENDING
            assert _only_task_run_state(direct.database) == (
                TaskRunState.RUNNING.value
            )
            assert _provider_call_count(direct.model_factory) == 1
            superseded = await direct.broker.supersede(
                SupersedeInteractionScopeCommand(
                    actor=direct.actor,
                    scope=InteractionExecutionScope(
                        run_id=request.origin.run_id,
                    ),
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                )
            )
            direct_result = await wait_for(running, timeout=1)
        assert isinstance(
            superseded.store_result,
            ScopeSupersessionApplied,
        )
        superseded_record = superseded.store_result.records[0]
        assert superseded_record.request.state is RequestState.SUPERSEDED
        assert direct_result.run.state is TaskRunState.FAILED
        assert _provider_call_count(direct.model_factory) == 1
        direct_inspection = await direct.client.inspect(
            direct_result.run.run_id
        )
        assert direct_inspection.artifacts == ()
        assert inspection.artifacts == ()
        client_evidence = await _task_client_observation_evidence(
            condition_id="INPUT-F-11",
            suspended=suspended,
            transition_from=suspended.request.state,
            provider_call_count=_provider_call_count(suspended.model_factory),
        )
        evidence = [
            _task_evidence(
                condition_id="INPUT-F-11",
                surface_id="task-target-agent-direct",
                transition_from=request.state,
                transition_to=superseded_record.request.state,
                public_result_id="task.interaction_superseded_attached.v1",
                task_state=direct_result.run.state,
                provider_call_count=_provider_call_count(direct.model_factory),
                domain_side_effect_count=len(direct_inspection.artifacts),
            ),
            _task_evidence(
                condition_id="INPUT-F-11",
                surface_id="task-target-agent-queue",
                transition_from=suspended.request.state,
                transition_to=persisted.state,
                public_result_id="task.interaction_superseded_cancelled.v1",
                task_state=inspection.run.state,
                provider_call_count=_provider_call_count(
                    suspended.model_factory
                ),
                domain_side_effect_count=len(inspection.artifacts),
            ),
            *client_evidence,
        ]
    finally:
        await direct.close()
    assert evidence is not None
    return evidence


def test_input_f_12(
    record_property: Callable[[str, object], None],
) -> None:
    """Persist required input after the finite caller handoff budget."""
    evidence = run_async(_test_input_f_12())
    assert len(evidence) == 6
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_12() -> list[dict[str, object]]:
    direct = await _direct_required_handoff_observation()
    suspended = await _durable_failure_harness(_confirmation())
    inspection = await suspended.client.inspect(suspended.run_id)
    output = await suspended.client.output(suspended.run_id)
    events = await suspended.client.events(suspended.run_id)
    persisted = await _persisted_request(suspended)

    assert persisted.state is RequestState.PENDING
    assert inspection.run.state is TaskRunState.INPUT_REQUIRED
    assert inspection.artifacts == ()
    assert events == inspection.events
    assert (
        tuple(event.event_type for event in events).count(
            TaskInteractionEventType.INPUT_REQUIRED.value
        )
        == 1
    )
    assert output.state is TaskRunState.INPUT_REQUIRED
    assert isinstance(output.input_required, Mapping)
    assert output.input_required["kind"] == "input_required"
    assert output.input_required["request_id"] == str(persisted.request_id)
    assert output.input_required["continuation_id"] == str(
        persisted.continuation_id
    )
    assert output.input_required["detached_resumption_available"] is True
    assert "questions" not in output.input_required
    provider_call_count = _provider_call_count(suspended.model_factory)
    assert provider_call_count == 1

    task_evidence = [
        _task_evidence(
            condition_id="INPUT-F-12",
            surface_id="task-target-agent-direct",
            transition_from=RequestState.PENDING,
            transition_to=RequestState.PENDING,
            public_result_id="task.interaction_pending.v1",
            task_state=direct.task_state,
            provider_call_count=direct.provider_call_count,
            domain_side_effect_count=direct.domain_side_effect_count,
        ),
        _task_evidence(
            condition_id="INPUT-F-12",
            surface_id="task-target-agent-queue",
            transition_from=suspended.request.state,
            transition_to=persisted.state,
            public_result_id="task.interaction_pending.v1",
            task_state=inspection.run.state,
            provider_call_count=provider_call_count,
            domain_side_effect_count=len(inspection.artifacts),
        ),
    ]
    observation_evidence = [
        _task_observation_evidence(
            condition_id="INPUT-F-12",
            surface_id=surface_id,
            transition_from=RequestState.PENDING,
            transition_to=RequestState.PENDING,
            public_result_id="task.observation_pending.v1",
            status_key=status_key,
            status_value=status_value,
            provider_call_count=provider_call_count,
            domain_side_effect_count=len(inspection.artifacts),
        )
        for surface_id, status_key, status_value in (
            ("cli-task-inspect", "exit", "0"),
            ("cli-task-events", "exit", "0"),
            ("task-client-inspect", "client_result", "ok"),
            ("task-client-events", "client_result", "ok"),
        )
    ]
    await suspended.stack.aclose()
    suspended.temporary.cleanup()
    return [*task_evidence, *observation_evidence]


def test_input_f_13(
    record_property: Callable[[str, object], None],
) -> None:
    """Continue advisory input with policy timeout provenance."""
    evidence = run_async(_test_input_f_13())
    assert len(evidence) == 6
    _record_failure_matrix_evidence(record_property, evidence)


async def _test_input_f_13() -> list[dict[str, object]]:
    direct = await _attached_advisory_timeout_observation(queued=False)
    queued = await _attached_advisory_timeout_observation(queued=True)
    assert direct.request.state is RequestState.PENDING
    assert queued.request.state is RequestState.PENDING
    assert direct.task_state is TaskRunState.RUNNING
    assert queued.task_state is TaskRunState.RUNNING
    assert direct.provider_call_count == queued.provider_call_count == 2

    task_evidence = [
        _task_evidence(
            condition_id="INPUT-F-13",
            surface_id="task-target-agent-direct",
            transition_from=direct.request.state,
            transition_to=RequestState.TIMED_OUT,
            public_result_id="task.interaction_timed_out.v1",
            task_state=direct.task_state,
            provider_call_count=direct.provider_call_count,
            domain_side_effect_count=direct.domain_side_effect_count,
        ),
        _task_evidence(
            condition_id="INPUT-F-13",
            surface_id="task-target-agent-queue",
            transition_from=queued.request.state,
            transition_to=RequestState.TIMED_OUT,
            public_result_id="task.interaction_timed_out.v1",
            task_state=queued.task_state,
            provider_call_count=queued.provider_call_count,
            domain_side_effect_count=queued.domain_side_effect_count,
        ),
    ]
    observation_evidence = [
        _task_observation_evidence(
            condition_id="INPUT-F-13",
            surface_id=surface_id,
            transition_from=RequestState.PENDING,
            transition_to=RequestState.TIMED_OUT,
            public_result_id="task.observation_timed_out.v1",
            status_key=status_key,
            status_value=status_value,
            provider_call_count=queued.provider_call_count,
            domain_side_effect_count=queued.domain_side_effect_count,
        )
        for surface_id, status_key, status_value in (
            ("cli-task-inspect", "exit", "0"),
            ("cli-task-events", "exit", "0"),
            ("task-client-inspect", "client_result", "ok"),
            ("task-client-events", "client_result", "ok"),
        )
    ]
    return [*task_evidence, *observation_evidence]


async def _direct_required_handoff_observation() -> (
    _RequiredHandoffObservation
):
    """Run one direct agent target through a real durable suspension."""
    database = FullFakePgsqlDatabase()
    clock = _TestClock()
    interaction_store = await _open_interaction_store(
        database,
        clock=clock,
    )
    task_store = PgsqlTaskStore(
        cast(PgsqlDatabase, database),
        clock=lambda: clock.now,
    )
    temporary = TemporaryDirectory()
    root = Path(temporary.name)
    _write_agent(root)
    stack = AsyncExitStack()
    loader = OrchestratorLoader(
        hub=MagicMock(spec=HuggingfaceHub),
        logger=MagicMock(spec=Logger),
        participant_id=uuid4(),
        stack=stack,
    )
    host = DurableAgentTaskHost(
        orchestrator_loader=loader,
        stack=stack,
        allowed_roots=(root,),
        continuation_store=interaction_store,
        clock=lambda: clock.now,
    )
    target = _DirectSuspendingAgentTaskTargetRunner(
        loader,
        root=root,
        runtime_factory=host.interaction_runtime,
    )
    model_factory = _ProviderManagerFactory(
        [(True,)],
        arguments={
            "mode": RequirementMode.REQUIRED.value,
            "reason": "A response is required to continue.",
            "questions": [encode_input_question(_confirmation())],
        },
    )
    client = TaskClient(
        task_store,
        target=target,
        hmac_provider=_StaticHmacProvider(),
        definition_hash=lambda _: "task-failure-matrix-direct-handoff",
        clock=lambda: clock.now,
    )
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=model_factory,
        ):
            result = await client.run(
                _definition(),
                input_value="private",
            )
        inspection = await client.inspect(result.run.run_id)
        events = await client.events(result.run.run_id)
        assert result.run.state is TaskRunState.INPUT_REQUIRED
        assert result.input_required is not None
        assert (
            result.input_required.input_required.detached_resumption_available
        )
        assert inspection.run.state is TaskRunState.INPUT_REQUIRED
        assert inspection.artifacts == ()
        assert events == inspection.events
        assert (
            tuple(event.event_type for event in events).count(
                TaskInteractionEventType.INPUT_REQUIRED.value
            )
            == 1
        )
        assert len(target.suspensions) == 1
        assert (
            target.suspensions[0].command.request.state is RequestState.CREATED
        )
        assert _provider_call_count(model_factory) == 1
        return _RequiredHandoffObservation(
            task_state=inspection.run.state,
            provider_call_count=_provider_call_count(model_factory),
            domain_side_effect_count=len(inspection.artifacts),
        )
    finally:
        await stack.aclose()
        temporary.cleanup()


async def _attached_advisory_timeout_observation(
    *,
    queued: bool,
) -> _AdvisoryTimeoutObservation:
    """Observe a task while an advisory timeout enters model continuation."""
    database = FullFakePgsqlDatabase()
    clock = _TestClock()
    interaction_store = await _open_interaction_store(
        database,
        clock=clock,
    )
    policy = InteractionPolicy()
    broker = AsyncInteractionBroker(
        store=interaction_store,
        clock=clock,
        id_factory=_TestIds(),
        policy=policy,
        classifier=_TestClassifier(policy),
    )
    temporary = TemporaryDirectory()
    root = Path(temporary.name)
    _write_agent(root)
    stack = AsyncExitStack()
    loader = OrchestratorLoader(
        hub=MagicMock(spec=HuggingfaceHub),
        logger=MagicMock(spec=Logger),
        participant_id=uuid4(),
        stack=stack,
    )
    handler = _BlockingInputHandler()
    target = _AttachedAgentTaskTargetRunner(
        loader,
        root=root,
        runtime=AttachedInteractionRuntime(
            broker=broker,
            actor=InteractionActor(principal=PrincipalScope()),
            handler=handler,
        ),
    )
    question = ConfirmationQuestion(
        question_id=QuestionId("answer"),
        prompt="Continue when the advisory wait expires?",
        required=False,
    )
    model_factory = _GatedProviderManagerFactory(
        [(True, False)],
        arguments={
            "mode": RequirementMode.ADVISORY.value,
            "reason": "Continue after the bounded advisory wait.",
            "questions": [encode_input_question(question)],
        },
    )
    task_store = PgsqlTaskStore(
        cast(PgsqlDatabase, database),
        clock=lambda: clock.now,
    )
    queue = (
        PgsqlTaskQueue(
            cast(PgsqlDatabase, database),
            clock=lambda: clock.now,
        )
        if queued
        else None
    )
    client = TaskClient(
        task_store,
        target=target,
        queue=queue,
        hmac_provider=_StaticHmacProvider(),
        encryption_provider=(_StaticEncryptionProvider() if queued else None),
        raw_storage_allowed=queued,
        definition_hash=lambda _: (
            "task-failure-matrix-advisory-queue"
            if queued
            else "task-failure-matrix-advisory-direct"
        ),
        clock=lambda: clock.now,
    )
    running: object | None = None
    task: Task[TaskWorkerProcessResult] | Task[TaskRunResult]
    try:
        with patch(
            "avalan.agent.loader.ModelManager",
            side_effect=model_factory,
        ):
            if queued:
                assert queue is not None
                submission = await client.enqueue(
                    _queued_definition(),
                    input_value="private",
                )
                worker = TaskWorker(
                    task_store,
                    queue,
                    target=target,
                    worker_id="failure-matrix-advisory-worker",
                    queue_name="failure-matrix",
                    encryption_provider=_StaticEncryptionProvider(),
                    raw_storage_allowed=True,
                    clock=lambda: clock.now,
                )
                task = create_task(worker.process_once())
                run_id = submission.run.run_id
            else:
                task = create_task(
                    client.run(_definition(), input_value="private")
                )
                await wait_for(handler.started.wait(), timeout=1)
                run_id = _only_task_run_id(database)
            running = task
            if queued:
                await wait_for(handler.started.wait(), timeout=1)
            request = handler.contexts[0].request
            assert request.state is RequestState.PENDING
            assert _only_task_run_state(database) == TaskRunState.RUNNING.value
            assert _provider_call_count(model_factory) == 1

            assert request.advisory_wait_seconds == 60
            clock.now = request.created_at + timedelta(seconds=60)
            clock.monotonic = 60.0
            clock.changed.set()
            await wait_for(model_factory.started.wait(), timeout=1)

            inspection = await client.inspect(run_id)
            events = await client.events(run_id)
            persisted_state = next(iter(database.records.values()))[
                "request_state"
            ]
            assert persisted_state == RequestState.TIMED_OUT.value
            assert inspection.run.state is TaskRunState.RUNNING
            assert inspection.artifacts == ()
            assert events == inspection.events
            assert RequestState.TIMED_OUT.value in _interaction_event_states(
                events
            )
            assert _provider_call_count(model_factory) == 2

            observation = _AdvisoryTimeoutObservation(
                request=request,
                task_state=inspection.run.state,
                provider_call_count=_provider_call_count(model_factory),
                domain_side_effect_count=len(inspection.artifacts),
            )
            model_factory.release.set()
            completed = await wait_for(task, timeout=1)
        completion = getattr(completed, "completion", None)
        completed_run = (
            completion.run
            if completion is not None
            else getattr(completed, "run")
        )
        assert completed_run.state is TaskRunState.SUCCEEDED
        final = await client.inspect(run_id)
        assert final.artifacts == ()
        return observation
    finally:
        model_factory.release.set()
        if running is not None:
            task = cast(Any, running)
            if not task.done():
                await wait_for(task, timeout=1)
        await broker.aclose()
        await stack.aclose()
        temporary.cleanup()


def _interaction_event_states(events: tuple[object, ...]) -> tuple[str, ...]:
    """Return redacted interaction states from sanitized task events."""
    states: list[str] = []
    for event in events:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            continue
        interaction = payload.get("interaction_lifecycle")
        if not isinstance(interaction, Mapping):
            continue
        state = interaction.get("state")
        if isinstance(state, str):
            states.append(state)
    return tuple(states)


async def _durable_failure_harness(
    question: ConfirmationQuestion | SingleSelectionQuestion,
    *,
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> _DurableFailureHarness:
    database = FullFakePgsqlDatabase()
    clock = _TestClock()
    interaction_store = await _open_interaction_store(
        database,
        clock=clock,
    )
    database_protocol = cast(PgsqlDatabase, database)
    task_store = PgsqlTaskStore(
        database_protocol,
        clock=lambda: clock.now,
    )
    coordinator = PgsqlDurableTaskCoordinator(
        interaction_store,
        task_store,
    )
    queue = PgsqlTaskQueue(
        database_protocol,
        clock=lambda: clock.now,
        durable_reentry_coordinator=coordinator,
    )
    temporary = TemporaryDirectory()
    root = Path(temporary.name)
    _write_agent(root)
    stack = AsyncExitStack()
    loader = OrchestratorLoader(
        hub=MagicMock(spec=HuggingfaceHub),
        logger=MagicMock(spec=Logger),
        participant_id=uuid4(),
        stack=stack,
    )
    host = DurableAgentTaskHost(
        orchestrator_loader=loader,
        stack=stack,
        allowed_roots=(root,),
        continuation_store=interaction_store,
        clock=lambda: clock.now,
    )
    target = _RecordingAgentTaskTargetRunner(
        loader,
        root=root,
        runtime_factory=host.interaction_runtime,
    )
    arguments = {
        "mode": mode.value,
        "reason": (
            "A response is required to continue."
            if mode is RequirementMode.REQUIRED
            else "Continue after the bounded advisory wait."
        ),
        "questions": [encode_input_question(question)],
    }
    model_factory = _ProviderManagerFactory(
        [(True,)],
        arguments=arguments,
    )
    client = TaskClient(
        task_store,
        target=target,
        queue=queue,
        hmac_provider=_StaticHmacProvider(),
        encryption_provider=_StaticEncryptionProvider(),
        raw_storage_allowed=True,
        definition_hash=lambda _: "task-failure-matrix-queue",
        durable_lifecycle_coordinator=coordinator,
        clock=lambda: clock.now,
    )
    submission = await client.enqueue(
        _queued_definition(),
        input_value="private",
    )
    worker = TaskWorker(
        task_store,
        queue,
        target=target,
        worker_id="failure-matrix-worker",
        queue_name="failure-matrix",
        encryption_provider=_StaticEncryptionProvider(),
        raw_storage_allowed=True,
        durable_suspension_coordinator=coordinator,
        clock=lambda: clock.now,
    )

    with (
        patch(
            "avalan.agent.loader.ModelManager",
            side_effect=model_factory,
        ),
        patch("avalan.agent.continuation_stager.datetime") as wall_datetime,
    ):
        processed = await worker.process_once()
    assert not wall_datetime.now.called

    assert processed.suspension is not None
    suspension = processed.suspension
    assert suspension.run.run_id == submission.run.run_id
    assert suspension.run.state is TaskRunState.INPUT_REQUIRED
    assert suspension.attempt.state is TaskAttemptState.SUSPENDED
    assert suspension.segment.state is TaskAttemptSegmentState.SUSPENDED
    assert suspension.queue_item.state is TaskQueueItemState.SUSPENDED
    assert len(target.initial_contexts) == 1
    assert target.resume_contexts == []
    assert len(target.suspensions) == 1
    assert len(model_factory.instances) == 1
    assert len(model_factory.instances[0].calls) == 1
    request = target.suspensions[0].command.request
    record = await interaction_store.lookup_scoped(
        ScopedInteractionLookup(
            actor=target.suspensions[0].command.actor,
            correlation=InteractionCorrelation.from_request(request),
        )
    )
    assert isinstance(record, InteractionRecord)
    assert record.request.state is RequestState.PENDING
    continuation = await interaction_store.get_continuation(
        record.request.continuation_id
    )
    assert record.request.created_at == clock.now == _NOW
    assert continuation.created_at == record.request.created_at
    assert continuation.updated_at == record.request.created_at
    assert continuation.expires_at == record.request.created_at + timedelta(
        seconds=record.request.continuation_ttl_seconds
    )
    return _DurableFailureHarness(
        database=database,
        clock=clock,
        interaction_store=interaction_store,
        task_store=task_store,
        queue=queue,
        coordinator=coordinator,
        client=client,
        target=target,
        model_factory=model_factory,
        stack=stack,
        temporary=temporary,
        run_id=submission.run.run_id,
        request=record.request,
        suspension=suspension,
    )


async def _direct_failure_harness(
    question: ConfirmationQuestion | SingleSelectionQuestion,
    handler_factory: Callable[
        [FullFakePgsqlDatabase, _ProviderManagerFactory],
        InputHandler,
    ],
) -> tuple[_DirectFailureHarness, InputHandler]:
    database = FullFakePgsqlDatabase()
    clock = _TestClock()
    interaction_store = await _open_interaction_store(
        database,
        clock=clock,
    )
    policy = InteractionPolicy()
    broker = AsyncInteractionBroker(
        store=interaction_store,
        clock=clock,
        id_factory=_TestIds(),
        policy=policy,
        classifier=_TestClassifier(policy),
    )
    temporary = TemporaryDirectory()
    root = Path(temporary.name)
    _write_agent(root)
    stack = AsyncExitStack()
    loader = OrchestratorLoader(
        hub=MagicMock(spec=HuggingfaceHub),
        logger=MagicMock(spec=Logger),
        participant_id=uuid4(),
        stack=stack,
    )
    arguments = {
        "mode": RequirementMode.REQUIRED.value,
        "reason": "A response is required to continue.",
        "questions": [encode_input_question(question)],
    }
    model_factory = _ProviderManagerFactory(
        [(True, False)],
        arguments=arguments,
        use_openai_transport=True,
    )
    handler = handler_factory(database, model_factory)
    actor = InteractionActor(principal=PrincipalScope())
    runtime = AttachedInteractionRuntime(
        broker=broker,
        actor=actor,
        handler=handler,
    )
    target = _AttachedAgentTaskTargetRunner(
        loader,
        root=root,
        runtime=runtime,
    )
    task_store = PgsqlTaskStore(
        cast(PgsqlDatabase, database),
        clock=lambda: clock.now,
    )
    client = TaskClient(
        task_store,
        target=target,
        hmac_provider=_StaticHmacProvider(),
        definition_hash=lambda _: "task-failure-matrix-direct",
        clock=lambda: clock.now,
    )
    return (
        _DirectFailureHarness(
            database=database,
            clock=clock,
            broker=broker,
            actor=actor,
            task_store=task_store,
            client=client,
            model_factory=model_factory,
            stack=stack,
            temporary=temporary,
        ),
        handler,
    )


def _valid_resolution(request: InputRequest) -> AnsweredResolution:
    question = request.questions[0]
    answer: ConfirmationAnswer | SingleSelectionAnswer
    if isinstance(question, ConfirmationQuestion):
        answer = ConfirmationAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            value=True,
        )
    else:
        assert isinstance(question, SingleSelectionQuestion)
        answer = SingleSelectionAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            value=SelectedChoice(value=question.choices[0].value),
        )
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=request.created_at,
        answers=(answer,),
    )


def _wrong_type_resolution(request: InputRequest) -> AnsweredResolution:
    question = request.questions[0]
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=request.created_at,
        answers=(
            TextAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value="yes",
            ),
        ),
    )


def _unknown_choice_resolution(request: InputRequest) -> AnsweredResolution:
    question = request.questions[0]
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=request.created_at,
        answers=(
            SingleSelectionAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("unknown")),
            ),
        ),
    )


def _missing_answer_resolution(request: InputRequest) -> AnsweredResolution:
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=request.created_at,
        answers=(),
    )


def _direct_resolution_command(
    harness: _DirectFailureHarness,
    request: InputRequest,
    *,
    value: bool,
    key: str,
    expected_state_revision: StateRevision | None = None,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=harness.actor,
        correlation=InteractionCorrelation.from_request(request),
        expected_state_revision=(
            request.state_revision
            if expected_state_revision is None
            else expected_state_revision
        ),
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=request.created_at,
            answers=(
                ConfirmationAnswer(
                    question_id=request.questions[0].question_id,
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                ),
            ),
        ),
    )


def _only_task_run_state(database: FullFakePgsqlDatabase) -> str:
    assert len(database.runs) == 1
    state = next(iter(database.runs.values()))["state"]
    assert isinstance(state, str)
    return state


def _only_task_run_id(database: FullFakePgsqlDatabase) -> str:
    assert len(database.runs) == 1
    run_id = next(iter(database.runs))
    assert isinstance(run_id, str)
    return run_id


def _provider_call_count(model_factory: _ProviderManagerFactory) -> int:
    assert len(model_factory.instances) == 1
    return len(model_factory.instances[0].calls)


async def _assert_durable_resolution_rejected(
    suspended: _DurableFailureHarness,
    resolution: AnsweredResolution,
    code: InputErrorCode,
) -> None:
    command = ResolveInteractionCommand(
        actor=suspended.target.suspensions[0].command.actor,
        correlation=InteractionCorrelation.from_request(suspended.request),
        expected_state_revision=suspended.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("rejected-answer"),
        proposed_resolution=resolution,
    )

    rejected = await suspended.interaction_store.resolve(command)
    inspection = await suspended.client.inspect(suspended.run_id)
    persisted = await suspended.interaction_store.lookup_scoped(
        ScopedInteractionLookup(
            actor=command.actor,
            correlation=command.correlation,
        )
    )
    continuation = await suspended.interaction_store.get_continuation(
        suspended.request.continuation_id
    )
    queue_row = _persisted_queue_row(suspended)

    assert isinstance(rejected, ResolveInteractionRejected)
    assert rejected.error.code is code
    assert not rejected.store_mutation_applied
    assert isinstance(persisted, InteractionRecord)
    assert persisted.request == suspended.request
    assert persisted.request.state is RequestState.PENDING
    assert continuation.claim.state is ContinuationClaimState.UNCLAIMED
    assert inspection.run.state is TaskRunState.INPUT_REQUIRED
    assert len(inspection.attempts) == 1
    assert inspection.attempts[0].state is TaskAttemptState.SUSPENDED
    assert queue_row["state"] == TaskQueueItemState.SUSPENDED.value
    assert queue_row["claim_token"] is None
    assert len(suspended.target.initial_contexts) == 1
    assert suspended.target.resume_contexts == []
    assert suspended.target.domain_side_effects == []


def _resolution_command(
    suspended: _DurableFailureHarness,
    *,
    value: bool,
    key: str,
    expected_state_revision: StateRevision | None = None,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=suspended.target.suspensions[0].command.actor,
        correlation=InteractionCorrelation.from_request(suspended.request),
        expected_state_revision=(
            suspended.request.state_revision
            if expected_state_revision is None
            else expected_state_revision
        ),
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=suspended.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("answer"),
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                ),
            ),
        ),
    )


async def _persisted_request(
    suspended: _DurableFailureHarness,
) -> InputRequest:
    command = suspended.target.suspensions[0].command
    record = await suspended.interaction_store.lookup_scoped(
        ScopedInteractionLookup(
            actor=command.actor,
            correlation=InteractionCorrelation.from_request(suspended.request),
        )
    )
    assert isinstance(record, InteractionRecord)
    return record.request


def _persisted_queue_row(
    suspended: _DurableFailureHarness,
) -> Mapping[str, object]:
    queue_item_id = suspended.suspension.queue_item.queue_item_id
    row = suspended.database.queue_items[queue_item_id]
    return dict(row)


def _continuation_lifecycle(suspended: _DurableFailureHarness) -> str:
    row = suspended.database.continuations[
        str(suspended.request.continuation_id)
    ]
    lifecycle = row["lifecycle_state"]
    assert isinstance(lifecycle, str)
    return lifecycle


async def _resume_harness(
    suspended: _DurableFailureHarness,
) -> _ResumeHarness:
    suspended.clock.now = max(
        suspended.clock.now,
        _NOW + timedelta(seconds=3),
    )
    suspended.clock.monotonic = max(suspended.clock.monotonic, 3)
    record = await suspended.interaction_store.get_task_continuation_record(
        suspended.run_id
    )
    continuation = record.continuation
    adapter = _ResumeAdapter(suspended.request.mode)
    executor = _ResumeExecutor()
    runtime = ResolvedContinuationRuntime(
        definition=continuation.definition,
        revision_binding=continuation.revision_binding,
        runtime=executor,
        operation=object(),
        model=adapter,
        tools=object(),
        capabilities=_catalog(continuation.revision_binding),
        credentials_reloaded_from_trusted_config=True,
    )
    loader = _ResumeLoader(runtime)
    resolver = ContinuationRuntimeResolver(
        loader,
        clock=lambda: suspended.clock.now,
    )
    resumer = DurableAgentContinuationResumer(
        suspended.interaction_store,
        resolver,
        clock=lambda: suspended.clock.now,
    )
    resume_coordinator = TaskDurableResumeCoordinator(
        suspended.interaction_store,
        resumer,
    )
    worker = TaskWorker(
        suspended.task_store,
        suspended.queue,
        target=suspended.target,
        worker_id="failure-matrix-resume-worker",
        queue_name="failure-matrix",
        encryption_provider=_StaticEncryptionProvider(),
        raw_storage_allowed=True,
        durable_suspension_coordinator=suspended.coordinator,
        durable_resume_coordinator=resume_coordinator,
        clock=lambda: suspended.clock.now,
    )
    return _ResumeHarness(
        worker=worker,
        adapter=adapter,
        executor=executor,
        loader=loader,
    )


def _catalog(
    binding: ContinuationRevisionBinding,
) -> ModelCapabilityCatalog:
    registry = ContinuationSnapshotCodecRegistry("failure-matrix")
    registry.register(
        codec_id="failure-matrix-openai-v1",
        revision_binding=binding,
        snapshot_kind="openai.responses.reasoning",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=lambda value, expected: decode_continuation_snapshot(
            value,
            expected_binding=expected,
        ),
    )
    return ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec_registry=registry,
            continuation_snapshot_codec=registry.reference(
                "failure-matrix-openai-v1"
            ),
        ),
        revision_binding=binding,
    )


def _confirmation() -> ConfirmationQuestion:
    return ConfirmationQuestion(
        question_id=QuestionId("answer"),
        prompt="Continue?",
        required=True,
    )


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="task_failure_matrix", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        privacy=TaskPrivacyPolicy(
            input=PrivacyAction.HASH,
            output=PrivacyAction.REDACT,
        ),
        run=TaskRunPolicy.direct(),
    )


def _queued_definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="task_failure_matrix", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        privacy=TaskPrivacyPolicy(raw_retention_days=1),
        run=TaskRunPolicy.queued("failure-matrix"),
    )
