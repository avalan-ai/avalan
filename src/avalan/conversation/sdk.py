"""Expose the typed asynchronous direct conversation SDK."""

from .binding import ProviderFamily, ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CheckpointId,
    CheckpointIdentity,
    CheckpointSequence,
    ConversationBranchId,
    ConversationId,
    ConversationOperation,
    ExecutionSegmentId,
    LocalDeletionState,
    LogicalTurnId,
    ProviderLaneStorage,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyKey,
    RetentionLimits,
)
from .coordinator import RunScopedConversationCoordinator
from .errors import (
    ConversationAuthorizationError,
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationConflictError,
    ConversationError,
    ConversationTransitionError,
    ConversationValidationError,
)
from .items import (
    ProviderItem,
    VisibleTranscriptEntry,
    VisibleTranscriptRole,
    public_provider_item_projection,
)
from .lifecycle import (
    AmbiguousDispatchReconciliationDisposition,
    AmbiguousDispatchReconciliationRequest,
    AmbiguousDispatchReconciliationResult,
    AmbiguousDispatchResolution,
    DirectDeletionResult,
    ProviderLifecycleReconciler,
    ProviderLifecycleStore,
    StoredProviderResolver,
    UpstreamAvailability,
)
from .observability import ConversationRequestSemantics
from .protocols import ConversationProviderStateSink, ConversationStore
from .runtime import (
    AtomicCommitReceipt,
    ConversationCommitBoundary,
    ConversationLaneRequest,
    ConversationRunRequest,
    ExplicitBranchAdvance,
    FirstTurnAdvance,
    NamedHeadAdvance,
    OrdinaryChildAdvance,
    ProviderLaneOutputCandidate,
    ResetAdvance,
)
from .settings import (
    ConversationHandle,
    ConversationMode,
    ConversationModeConversion,
    ConversationParent,
    ConversationResetIntent,
    ConversationResult,
    DisabledCompaction,
    EffectiveReasoningMetadata,
    ProviderUsage,
    StandaloneCompactRequest,
    StandaloneCompactResult,
    StatelessConversationHandle,
    StatelessConversationSettings,
    StatelessParent,
    StoredConversationHandle,
    StoredConversationSettings,
    StoredParent,
    validate_mode_transition_authority,
)
from .state import (
    ConversationCheckpoint,
    StoredProviderLaneSnapshot,
    validate_upstream_identifier_separation,
)
from .value import validate_identifier

from asyncio import CancelledError, Queue, Task, create_task, shield
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, TypeAlias, cast, final, overload
from uuid import uuid4


class DirectConversationStreamState(StrEnum):
    """Identify the stable caller-visible lifecycle of a direct stream."""

    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLOSED_INCOMPLETE = "closed_incomplete"


class ConversationHandleUnavailableError(RuntimeError):
    """Report that a stream has no successfully committed terminal handle."""

    def __init__(self, state: DirectConversationStreamState) -> None:
        if not isinstance(state, DirectConversationStreamState):
            raise ConversationValidationError()
        self.state = state
        super().__init__(
            "conversation handle is unavailable before successful commit"
        )

    def __repr__(self) -> str:
        """Return a content-safe stream-state diagnostic."""
        return (
            f"ConversationHandleUnavailableError(state={self.state.value!r})"
        )


class DirectConversationStreamError(RuntimeError):
    """Report a content-safe non-domain direct stream failure."""

    def __init__(self) -> None:
        super().__init__("direct conversation stream failed")


class DirectConversationCancelledError(DirectConversationStreamError):
    """Report an unexpected provider or worker stream cancellation."""

    def __init__(self) -> None:
        RuntimeError.__init__(
            self,
            "direct conversation stream was cancelled unexpectedly",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectConversationResult:
    """Return visible output and metadata from one committed direct turn."""

    output: str
    usage: ProviderUsage
    reasoning: EffectiveReasoningMetadata
    handle: ConversationHandle

    def __post_init__(self) -> None:
        if type(self.output) is not str:
            raise ConversationValidationError()
        if type(self.usage) is not ProviderUsage:
            raise ConversationValidationError()
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        if not isinstance(
            self.handle,
            StatelessConversationHandle | StoredConversationHandle,
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectConversationOutputDelta:
    """Publish one visible provider-authored text segment."""

    text_delta: str

    def __post_init__(self) -> None:
        if type(self.text_delta) is not str or not self.text_delta:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectConversationStreamTerminal:
    """Publish one committed result as the sole terminal stream event."""

    result: DirectConversationResult

    def __post_init__(self) -> None:
        if type(self.result) is not DirectConversationResult:
            raise ConversationValidationError()


DirectConversationStreamItem: TypeAlias = (
    DirectConversationOutputDelta | DirectConversationStreamTerminal
)
ActiveConversationSettings: TypeAlias = (
    StatelessConversationSettings | StoredConversationSettings
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectConversationRuntime:
    """Bind direct operations to trusted run-scoped lane authority."""

    coordinator: RunScopedConversationCoordinator
    store: ConversationStore
    authority: AuthorityScope
    lane: ProviderLaneBinding
    retention: RetentionLimits
    id_namespace: str | None = None
    provider_resolver: StoredProviderResolver | None = None
    lifecycle_reconciler: ProviderLifecycleReconciler | None = None

    def __post_init__(self) -> None:
        if type(self.coordinator) is not RunScopedConversationCoordinator:
            raise ConversationValidationError()
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        if type(self.lane) is not ProviderLaneBinding:
            raise ConversationValidationError()
        if type(self.retention) is not RetentionLimits:
            raise ConversationValidationError()
        if (
            self.lane.provider_family
            not in {
                ProviderFamily.SYNTHETIC,
                ProviderFamily.OPENAI,
                ProviderFamily.AZURE_OPENAI,
            }
            or self.lane.agent_id != self.authority.agent_id
        ):
            raise ConversationValidationError()
        if self.id_namespace is not None:
            validate_identifier(self.id_namespace, "id_namespace")
        if (
            self.lifecycle_reconciler is not None
            and self.provider_resolver is None
        ):
            raise ConversationValidationError()
        if (
            self.provider_resolver is not None
            and type(self.provider_resolver) is not StoredProviderResolver
        ):
            raise ConversationValidationError()
        if self.lifecycle_reconciler is not None:
            if (
                type(self.lifecycle_reconciler)
                is not ProviderLifecycleReconciler
            ):
                raise ConversationValidationError()
            assert self.provider_resolver is not None
            self.lifecycle_reconciler.assert_runtime(
                store=cast(ProviderLifecycleStore, self.store),
                resolver=self.provider_resolver,
                authority=self.authority,
            )
        self.coordinator.validate_direct_runtime(self.store, self.lane)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _RequestIds:
    conversation_id: ConversationId
    logical_turn_id: LogicalTurnId
    execution_segment_id: ExecutionSegmentId
    checkpoint_id: CheckpointId
    branch_id: ConversationBranchId
    provisional_response_id: ProvisionalResponseId
    public_response_id: PublicResponseId
    idempotency_key: RequestIdempotencyKey


@final
class _DirectStreamSink(ConversationProviderStateSink):
    def __init__(self, queue: Queue[object]) -> None:
        self._queue = queue
        self._finalized = False
        self._cleaned = False

    async def stage(self, item: ProviderItem) -> None:
        if self._finalized or self._cleaned or type(item) is not ProviderItem:
            raise ConversationValidationError()
        for entry in public_provider_item_projection((item,)):
            self._queue.put_nowait(
                DirectConversationOutputDelta(text_delta=entry.content)
            )

    async def finalize(
        self,
        outputs: tuple[ProviderLaneOutputCandidate, ...],
    ) -> None:
        if (
            self._finalized
            or self._cleaned
            or type(outputs) is not tuple
            or not outputs
            or any(
                type(item) is not ProviderLaneOutputCandidate
                for item in outputs
            )
        ):
            raise ConversationValidationError()
        self._finalized = True

    async def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True


@final
class DirectConversationStream(AsyncIterator[DirectConversationStreamItem]):
    """Yield visible output and one post-commit terminal result."""

    _END = object()

    def __init__(
        self,
        coordinator: RunScopedConversationCoordinator,
        request: ConversationRunRequest,
        provider_resolver: StoredProviderResolver | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._request = request
        self._provider_resolver = provider_resolver
        self._queue: Queue[object] = Queue()
        self._task: Task[None] | None = None
        self._state = DirectConversationStreamState.PENDING
        self._terminal: DirectConversationStreamTerminal | None = None
        self._failure: BaseException | None = None
        self._iteration_claimed = False
        self._terminal_yielded = False
        self._closing_state: DirectConversationStreamState | None = None

    @property
    def state(self) -> DirectConversationStreamState:
        """Return the current stable stream lifecycle state."""
        return self._state

    @property
    def terminal(self) -> DirectConversationStreamTerminal:
        """Return the terminal result only after successful commit."""
        if (
            self._state is not DirectConversationStreamState.COMMITTED
            or self._terminal is None
        ):
            raise ConversationHandleUnavailableError(self._state)
        return self._terminal

    @property
    def committed_handle(self) -> ConversationHandle:
        """Return the handle only after successful terminal commit."""
        return self.terminal.result.handle

    def __aiter__(self) -> AsyncIterator[DirectConversationStreamItem]:
        if self._iteration_claimed:
            raise RuntimeError("direct conversation stream is single-use")
        self._iteration_claimed = True
        self._start()
        return self

    async def __anext__(self) -> DirectConversationStreamItem:
        self._start()
        try:
            item = await self._queue.get()
        except CancelledError:
            await self.cancel()
            raise
        if item is self._END:
            if self._failure is not None:
                raise self._failure
            raise StopAsyncIteration
        if isinstance(item, DirectConversationStreamTerminal):
            if self._terminal_yielded:
                raise StopAsyncIteration
            self._terminal_yielded = True
            return item
        if type(item) is DirectConversationOutputDelta:
            return item
        self._failure = DirectConversationStreamError()
        self._state = DirectConversationStreamState.FAILED
        raise self._failure

    async def cancel(self) -> None:
        """Cancel, await cleanup, and leave no resumable child handle."""
        await self._close(DirectConversationStreamState.CANCELLED)

    async def aclose(self) -> None:
        """Close an unconsumed or incomplete stream asynchronously."""
        await self._close(DirectConversationStreamState.CLOSED_INCOMPLETE)

    def _start(self) -> None:
        if (
            self._task is not None
            or self._state is not DirectConversationStreamState.PENDING
        ):
            return
        self._state = DirectConversationStreamState.ACTIVE
        self._task = create_task(self._run())

    async def _run(self) -> None:
        sink = _DirectStreamSink(self._queue)
        try:
            if self._provider_resolver is None:
                receipt = await self._coordinator.stream_with_sink(
                    self._request,
                    sink,
                )
            else:
                receipt = await self._coordinator.stream_with_sink(
                    self._request,
                    sink,
                    stored_provider_resolver=self._provider_resolver,
                )
            result = _direct_result(receipt)
            terminal = DirectConversationStreamTerminal(result=result)
            self._terminal = terminal
            self._state = DirectConversationStreamState.COMMITTED
            self._queue.put_nowait(terminal)
        except CancelledError:
            closing_state = self._closing_state
            if closing_state is None:
                self._failure = DirectConversationCancelledError()
                self._state = DirectConversationStreamState.FAILED
            else:
                self._state = closing_state
                raise
        except ConversationError as error:
            self._failure = error
            self._state = DirectConversationStreamState.FAILED
        except BaseException:
            self._failure = DirectConversationStreamError()
            self._state = DirectConversationStreamState.FAILED
        finally:
            self._queue.put_nowait(self._END)

    async def _close(self, state: DirectConversationStreamState) -> None:
        if self._state in {
            DirectConversationStreamState.COMMITTED,
            DirectConversationStreamState.FAILED,
            DirectConversationStreamState.CANCELLED,
            DirectConversationStreamState.CLOSED_INCOMPLETE,
        }:
            return
        self._closing_state = state
        task = self._task
        if task is None:
            self._state = state
            self._queue.put_nowait(self._END)
            return
        task.cancel()
        cancellation: CancelledError | None = None
        while not task.done():
            try:
                await shield(task)
            except CancelledError as error:
                if not task.done():
                    cancellation = cancellation or error
        try:
            task.result()
        except CancelledError:
            pass
        self._state = state
        self._terminal = None
        if cancellation is not None:
            raise cancellation

    def __repr__(self) -> str:
        """Return content-free direct stream lifecycle metadata."""
        return f"DirectConversationStream(state={self._state.value!r})"


@final
class DirectConversationClient:
    """Execute typed direct conversation operations."""

    def __init__(self, runtime: DirectConversationRuntime) -> None:
        if type(runtime) is not DirectConversationRuntime:
            raise ConversationValidationError()
        self._runtime = runtime
        self._namespace = runtime.id_namespace or uuid4().hex
        self._sequence = 0
        self._idempotency_ids: dict[
            tuple[str, RequestIdempotencyKey], _RequestIds
        ] = {}

    async def retrieve(
        self,
        public_response_id: PublicResponseId,
    ) -> DirectConversationResult:
        """Retrieve local output and prove stored upstream availability."""
        validate_identifier(public_response_id, "public_response_id")
        result = await self._runtime.store.retrieve(
            public_response_id,
            self._runtime.authority,
        )
        checkpoint = await self._runtime.store.load(
            result.handle.checkpoint_id,
            self._runtime.authority,
        )
        validate_upstream_identifier_separation(
            checkpoint,
            additional_public_identifiers=(str(public_response_id),),
        )
        stored_lanes = tuple(
            lane
            for lane in checkpoint.content.lanes
            if isinstance(lane, StoredProviderLaneSnapshot)
        )
        if stored_lanes:
            resolver = self._runtime.provider_resolver
            if resolver is None:
                raise ConversationCapabilityError()
            for lane in stored_lanes:
                adapter = await resolver.resolve(lane.binding.integrity_digest)
                lane.binding.assert_compatible(adapter.binding)
                if lane.binding.execution_definition_digest is None:
                    raise ConversationBindingDriftError()
                retrieved = await adapter.retrieve(lane.upstream_response_id)
                if (
                    retrieved.availability
                    is not UpstreamAvailability.AVAILABLE
                ):
                    raise ConversationAuthorizationError()
                if (
                    retrieved.upstream_response_id != lane.upstream_response_id
                    or retrieved.binding_digest
                    != lane.binding.integrity_digest
                    or retrieved.execution_definition_digest
                    != lane.binding.execution_definition_digest
                    or retrieved.effective_reasoning_context
                    != lane.reasoning.effective
                ):
                    raise ConversationBindingDriftError()
        return _direct_result_from_resource(result, checkpoint)

    async def reconcile_ambiguous_dispatch(
        self,
        operation: ConversationOperation,
        idempotency_key: RequestIdempotencyKey,
        resolution: AmbiguousDispatchResolution,
    ) -> AmbiguousDispatchReconciliationResult:
        """Apply one explicit durable decision to a fenced dispatch."""
        if not isinstance(operation, ConversationOperation) or not isinstance(
            resolution, AmbiguousDispatchResolution
        ):
            raise ConversationValidationError()
        validate_identifier(idempotency_key, "idempotency_key")
        result = await self._runtime.store.reconcile_ambiguous_dispatch(
            AmbiguousDispatchReconciliationRequest(
                authority=self._runtime.authority,
                operation=operation,
                idempotency_key=idempotency_key,
                resolution=resolution,
            )
        )
        if result.disposition is (
            AmbiguousDispatchReconciliationDisposition.NOT_FOUND_OR_UNAUTHORIZED
        ):
            raise ConversationAuthorizationError()
        return result

    async def delete(
        self,
        public_response_id: PublicResponseId,
    ) -> DirectDeletionResult:
        """Tombstone locally first and reconcile upstream deletion."""
        validate_identifier(public_response_id, "public_response_id")
        preparation = await self._runtime.store.prepare_deletion(
            public_response_id,
            self._runtime.authority,
        )
        if preparation.state is LocalDeletionState.DELETED:
            return DirectDeletionResult(
                public_response_id=public_response_id,
                local_tombstoned=True,
                upstream_pending=False,
            )
        checkpoint = preparation.checkpoint
        assert checkpoint is not None
        stored_lane_count = sum(
            isinstance(lane, StoredProviderLaneSnapshot)
            for lane in checkpoint.content.lanes
        )
        reconciler = self._runtime.lifecycle_reconciler
        at = datetime.now(UTC)
        if preparation.state is LocalDeletionState.ACTIVE:
            try:
                await self._runtime.store.tombstone(
                    public_response_id,
                    self._runtime.authority,
                    at,
                )
            except (
                ConversationAuthorizationError,
                ConversationConflictError,
            ):
                preparation = await self._runtime.store.prepare_deletion(
                    public_response_id,
                    self._runtime.authority,
                )
                if preparation.state is LocalDeletionState.DELETED:
                    return DirectDeletionResult(
                        public_response_id=public_response_id,
                        local_tombstoned=True,
                        upstream_pending=False,
                    )
                if preparation.state is LocalDeletionState.ACTIVE:
                    raise
        if reconciler is not None and stored_lane_count:
            await reconciler.run_once(limit=stored_lane_count)
        elif stored_lane_count:
            return DirectDeletionResult(
                public_response_id=public_response_id,
                local_tombstoned=True,
                upstream_pending=True,
            )
        upstream_pending = False
        try:
            await self._runtime.store.delete(
                public_response_id,
                self._runtime.authority,
                at,
            )
        except ConversationTransitionError:
            upstream_pending = True
        except ConversationAuthorizationError:
            preparation = await self._runtime.store.prepare_deletion(
                public_response_id,
                self._runtime.authority,
            )
            if preparation.state is not LocalDeletionState.DELETED:
                raise
        return DirectDeletionResult(
            public_response_id=public_response_id,
            local_tombstoned=True,
            upstream_pending=upstream_pending,
        )

    async def convert(
        self,
        input: str,
        transition: ConversationModeConversion,
        settings: ActiveConversationSettings,
    ) -> DirectConversationResult:
        """Reject unproven continuity-preserving mode conversion."""
        _validate_input(input)
        _validate_active_settings(settings)
        if type(transition) is not ConversationModeConversion:
            raise ConversationValidationError()
        authorization = transition.authorization
        validate_mode_transition_authority(
            transition,
            current_checkpoint_id=authorization.checkpoint_id,
            current_parent=authorization.parent,
            current_authority=self._runtime.authority,
            current_binding=self._runtime.lane,
        )
        raise ConversationCapabilityError()

    @overload
    async def create(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[False] = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult: ...

    @overload
    async def create(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[True],
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationStream: ...

    async def create(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: bool = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult | DirectConversationStream:
        """Create one new direct conversation."""
        _validate_input(input)
        _validate_active_settings(settings)
        _validate_explicit_idempotency_key(idempotency_key)
        if (
            settings.parent is not None
            or settings.branch is not None
            or settings.named_head is not None
        ):
            raise ConversationValidationError()
        request = self._root_request(
            input,
            settings,
            reset_parent=None,
            idempotency_key=idempotency_key,
        )
        return await self._dispatch(request, stream=stream)

    @overload
    async def continue_conversation(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[False] = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult: ...

    @overload
    async def continue_conversation(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[True],
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationStream: ...

    async def continue_conversation(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: bool = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult | DirectConversationStream:
        """Continue one immutable parent or advance one named head."""
        _validate_input(input)
        _validate_active_settings(settings)
        _validate_explicit_idempotency_key(idempotency_key)
        if settings.parent is None or settings.branch is not None:
            raise ConversationValidationError()
        parent = await self._load_parent(settings.parent, settings.mode)
        ids = self._ids("continue", idempotency_key=idempotency_key)
        advance = (
            NamedHeadAdvance(
                head_id=settings.named_head.head_id,
                parent_checkpoint_id=parent.identity.checkpoint_id,
                expected_revision=settings.named_head.expected_revision,
            )
            if settings.named_head is not None
            else OrdinaryChildAdvance(
                parent_checkpoint_id=parent.identity.checkpoint_id
            )
        )
        request = self._child_request(
            input,
            settings,
            parent,
            ids,
            advance=advance,
            operation=ConversationOperation.CONTINUE,
            branch_id=parent.identity.branch_id,
        )
        return await self._dispatch(request, stream=stream)

    @overload
    async def branch(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[False] = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult: ...

    @overload
    async def branch(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[True],
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationStream: ...

    async def branch(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        stream: bool = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult | DirectConversationStream:
        """Create an explicit child branch from one immutable parent."""
        _validate_input(input)
        _validate_active_settings(settings)
        _validate_explicit_idempotency_key(idempotency_key)
        intent = settings.branch
        if (
            settings.parent is None
            or intent is None
            or settings.named_head is not None
        ):
            raise ConversationValidationError()
        parent = await self._load_parent(settings.parent, settings.mode)
        ids = self._ids("branch", idempotency_key=idempotency_key)
        request = self._child_request(
            input,
            settings,
            parent,
            ids,
            advance=ExplicitBranchAdvance(
                parent_checkpoint_id=parent.identity.checkpoint_id,
                branch_id=intent.branch_id,
            ),
            operation=ConversationOperation.BRANCH,
            branch_id=intent.branch_id,
        )
        return await self._dispatch(request, stream=stream)

    @overload
    async def reset(
        self,
        input: str,
        intent: ConversationResetIntent,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[False] = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult: ...

    @overload
    async def reset(
        self,
        input: str,
        intent: ConversationResetIntent,
        settings: ActiveConversationSettings,
        *,
        stream: Literal[True],
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationStream: ...

    async def reset(
        self,
        input: str,
        intent: ConversationResetIntent,
        settings: ActiveConversationSettings,
        *,
        stream: bool = False,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> DirectConversationResult | DirectConversationStream:
        """Create an explicit new root after authorized parent resolution."""
        _validate_input(input)
        _validate_active_settings(settings)
        _validate_explicit_idempotency_key(idempotency_key)
        if (
            type(intent) is not ConversationResetIntent
            or intent.target_mode is not settings.mode
        ):
            raise ConversationValidationError()
        if (
            settings.parent is not None
            or settings.branch is not None
            or settings.named_head is not None
        ):
            raise ConversationValidationError()
        await self._load_parent(intent.parent, intent.parent.handle.mode)
        request = self._root_request(
            input,
            settings,
            reset_parent=intent.parent,
            idempotency_key=idempotency_key,
        )
        return await self._dispatch(request, stream=stream)

    async def compact(
        self,
        request: StandaloneCompactRequest,
        *,
        idempotency_key: RequestIdempotencyKey | None = None,
    ) -> StandaloneCompactResult:
        """Compact one stateless parent through the fake provider."""
        if type(request) is not StandaloneCompactRequest:
            raise ConversationValidationError()
        _validate_explicit_idempotency_key(idempotency_key)
        parent = await self._load_parent(
            request.parent,
            ConversationMode.STATELESS,
        )
        settings = StatelessConversationSettings(parent=request.parent)
        ids = self._ids("compact", idempotency_key=idempotency_key)
        run_request = self._child_request(
            "compact",
            settings,
            parent,
            ids,
            advance=OrdinaryChildAdvance(
                parent_checkpoint_id=parent.identity.checkpoint_id
            ),
            operation=ConversationOperation.COMPACT,
            branch_id=parent.identity.branch_id,
            outward=False,
        )
        receipt = await self._runtime.coordinator.compact(run_request)
        checkpoint = receipt.checkpoint
        if checkpoint.integrity is None:
            raise ConversationValidationError()
        return StandaloneCompactResult(
            handle=StatelessConversationHandle(
                conversation_id=checkpoint.identity.conversation_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                branch_id=checkpoint.identity.branch_id,
            ),
            canonical_context_digest=checkpoint.integrity.digest,
        )

    async def _dispatch(
        self,
        request: ConversationRunRequest,
        *,
        stream: bool,
    ) -> DirectConversationResult | DirectConversationStream:
        if type(stream) is not bool:
            raise ConversationValidationError()
        if stream:
            return DirectConversationStream(
                self._runtime.coordinator,
                request,
                self._runtime.provider_resolver,
            )
        resolver = self._runtime.provider_resolver
        if resolver is None:
            receipt = await self._runtime.coordinator.execute(request)
        else:
            receipt = await self._runtime.coordinator.execute(
                request,
                stored_provider_resolver=resolver,
            )
        return _direct_result(receipt)

    async def _load_parent(
        self,
        parent: ConversationParent,
        mode: ConversationMode,
    ) -> ConversationCheckpoint:
        if not isinstance(parent, StatelessParent | StoredParent):
            raise ConversationValidationError()
        handle = parent.handle
        if handle.mode is not mode:
            raise ConversationValidationError()
        checkpoint = await self._runtime.store.load(
            handle.checkpoint_id,
            self._runtime.authority,
        )
        if (
            checkpoint.identity.conversation_id != handle.conversation_id
            or checkpoint.identity.branch_id != handle.branch_id
        ):
            raise ConversationValidationError()
        lanes = {lane.lane_id: lane for lane in checkpoint.content.lanes}
        lane = lanes.get(self._runtime.lane.lane_id)
        if lane is None:
            raise ConversationValidationError()
        if lane.binding != self._runtime.lane:
            if (
                not isinstance(lane, StoredProviderLaneSnapshot)
                or mode is not ConversationMode.STORED
                or self._runtime.provider_resolver is None
            ):
                raise ConversationBindingDriftError()
            resolver = self._runtime.provider_resolver
            resolved = await resolver.resolve_continuation_runtime(
                lane.binding.integrity_digest,
            )
            if getattr(resolved, "binding", None) != lane.binding:
                raise ConversationBindingDriftError()
        return checkpoint

    def _root_request(
        self,
        input: str,
        settings: ActiveConversationSettings,
        *,
        reset_parent: ConversationParent | None,
        idempotency_key: RequestIdempotencyKey | None,
    ) -> ConversationRunRequest:
        ids = self._ids(
            "reset" if reset_parent is not None else "create",
            idempotency_key=idempotency_key,
        )
        parent_checkpoint_id = (
            reset_parent.handle.checkpoint_id
            if reset_parent is not None
            else None
        )
        advance = (
            ResetAdvance(parent_checkpoint_id=parent_checkpoint_id)
            if parent_checkpoint_id is not None
            else FirstTurnAdvance()
        )
        return self._request(
            input,
            settings,
            ids,
            identity=CheckpointIdentity(
                conversation_id=ids.conversation_id,
                logical_turn_id=ids.logical_turn_id,
                execution_segment_id=ids.execution_segment_id,
                checkpoint_id=ids.checkpoint_id,
                branch_id=ids.branch_id,
                sequence=CheckpointSequence(0),
            ),
            advance=advance,
            operation=ConversationOperation.CREATE,
            parent_checkpoint_id=parent_checkpoint_id,
            outward=True,
        )

    def _child_request(
        self,
        input: str,
        settings: ActiveConversationSettings,
        parent: ConversationCheckpoint,
        ids: _RequestIds,
        *,
        advance: (
            OrdinaryChildAdvance | ExplicitBranchAdvance | NamedHeadAdvance
        ),
        operation: ConversationOperation,
        branch_id: ConversationBranchId,
        outward: bool = True,
    ) -> ConversationRunRequest:
        identity = CheckpointIdentity(
            conversation_id=parent.identity.conversation_id,
            logical_turn_id=ids.logical_turn_id,
            execution_segment_id=ids.execution_segment_id,
            checkpoint_id=ids.checkpoint_id,
            branch_id=branch_id,
            sequence=CheckpointSequence(parent.identity.sequence + 1),
            parent_checkpoint_id=parent.identity.checkpoint_id,
            parent_sequence=parent.identity.sequence,
        )
        return self._request(
            input,
            settings,
            ids,
            identity=identity,
            advance=advance,
            operation=operation,
            parent_checkpoint_id=parent.identity.checkpoint_id,
            outward=outward,
        )

    def _request(
        self,
        input: str,
        settings: ActiveConversationSettings,
        ids: _RequestIds,
        *,
        identity: CheckpointIdentity,
        advance: (
            FirstTurnAdvance
            | ResetAdvance
            | OrdinaryChildAdvance
            | ExplicitBranchAdvance
            | NamedHeadAdvance
        ),
        operation: ConversationOperation,
        parent_checkpoint_id: CheckpointId | None,
        outward: bool,
    ) -> ConversationRunRequest:
        retention = _validated_retention(
            settings.retention,
            self._runtime.retention,
            settings.mode,
        )
        return ConversationRunRequest(
            semantics=ConversationRequestSemantics(
                authority=self._runtime.authority,
                operation=operation,
                mode=settings.mode,
                reasoning_context=settings.reasoning_context,
                semantic_input=(
                    {"operation": "compact"}
                    if operation is ConversationOperation.COMPACT
                    else {"text": input}
                ),
                parent_checkpoint_id=parent_checkpoint_id,
            ),
            identity=identity,
            advance=advance,
            lanes=(
                ConversationLaneRequest(
                    lane_id=self._runtime.lane.lane_id,
                    mode=settings.mode,
                    reasoning_context=settings.reasoning_context,
                    compaction=(
                        settings.compaction
                        if isinstance(settings, StatelessConversationSettings)
                        else DisabledCompaction()
                    ),
                ),
            ),
            visible_delta=(
                ()
                if operation is ConversationOperation.COMPACT
                else (
                    VisibleTranscriptEntry(
                        role=VisibleTranscriptRole.USER,
                        content=input,
                    ),
                )
            ),
            retention=retention,
            idempotency_key=ids.idempotency_key,
            boundary=(
                ConversationCommitBoundary.OUTWARD_TURN
                if outward
                else ConversationCommitBoundary.INTERNAL_SEGMENT
            ),
            provisional_response_id=(
                ids.provisional_response_id if outward else None
            ),
            public_response_id=ids.public_response_id if outward else None,
        )

    def _ids(
        self,
        operation: str,
        *,
        idempotency_key: RequestIdempotencyKey | None,
    ) -> _RequestIds:
        validate_identifier(operation, "operation")
        _validate_explicit_idempotency_key(idempotency_key)
        cache_key = (
            (operation, idempotency_key)
            if idempotency_key is not None
            else None
        )
        if cache_key is not None:
            prior = self._idempotency_ids.get(cache_key)
            if prior is not None:
                return prior
        self._sequence += 1
        prefix = f"direct-{self._namespace}-{self._sequence}-{operation}"
        key = (
            RequestIdempotencyKey(f"{prefix}-key")
            if idempotency_key is None
            else idempotency_key
        )
        validate_identifier(key, "idempotency_key")
        ids = _RequestIds(
            conversation_id=ConversationId(f"{prefix}-conversation"),
            logical_turn_id=LogicalTurnId(f"{prefix}-turn"),
            execution_segment_id=ExecutionSegmentId(f"{prefix}-segment"),
            checkpoint_id=CheckpointId(f"{prefix}-checkpoint"),
            branch_id=ConversationBranchId(f"{prefix}-branch"),
            provisional_response_id=ProvisionalResponseId(
                f"{prefix}-provisional"
            ),
            public_response_id=PublicResponseId(f"{prefix}-response"),
            idempotency_key=key,
        )
        if cache_key is not None:
            self._idempotency_ids[cache_key] = ids
        return ids


def _validate_input(value: object) -> str:
    if type(value) is not str or not value or not value.strip():
        raise ConversationValidationError()
    if len(value.encode("utf-8")) > 1_048_576:
        raise ConversationValidationError()
    return value


def _validate_explicit_idempotency_key(
    value: RequestIdempotencyKey | None,
) -> None:
    if value is not None:
        validate_identifier(value, "idempotency_key")


def _validate_active_settings(value: object) -> ActiveConversationSettings:
    if type(value) not in (
        StatelessConversationSettings,
        StoredConversationSettings,
    ):
        raise ConversationValidationError()
    return cast(ActiveConversationSettings, value)


def _validated_retention(
    requested: RetentionLimits | None,
    configured: RetentionLimits,
    mode: ConversationMode,
) -> RetentionLimits:
    selected = requested or configured
    if type(selected) is not RetentionLimits:
        raise ConversationValidationError()
    expected_upstream = (
        ProviderLaneStorage.STATELESS
        if mode is ConversationMode.STATELESS
        else ProviderLaneStorage.STORED
    )
    if selected.storage.upstream is not expected_upstream:
        raise ConversationValidationError()
    if selected.storage.local is not configured.storage.local:
        raise ConversationValidationError()
    if selected.storage.provider_storage_disclosed != (
        mode is ConversationMode.STORED
    ):
        raise ConversationValidationError()
    if (
        selected.upstream_lifetime_status
        is not configured.upstream_lifetime_status
    ):
        raise ConversationValidationError()
    for requested_ttl, configured_ttl in (
        (selected.local_ttl_seconds, configured.local_ttl_seconds),
        (selected.envelope_ttl_seconds, configured.envelope_ttl_seconds),
        (
            selected.known_upstream_ttl_seconds,
            configured.known_upstream_ttl_seconds,
        ),
    ):
        if configured_ttl is None:
            if requested_ttl is not None:
                raise ConversationValidationError()
        elif requested_ttl is None or requested_ttl > configured_ttl:
            raise ConversationValidationError()
    return selected


def _direct_result(receipt: AtomicCommitReceipt) -> DirectConversationResult:
    if type(receipt) is not AtomicCommitReceipt or receipt.result is None:
        raise ConversationValidationError()
    outputs = receipt.output_candidates
    if not outputs:
        raise ConversationValidationError()
    public_response_id = receipt.result.public_response_id
    additional_public_identifiers = (
        (str(public_response_id),) if public_response_id is not None else ()
    )
    validate_upstream_identifier_separation(
        receipt.checkpoint,
        additional_public_identifiers=additional_public_identifiers,
        additional_upstream_response_ids=tuple(
            str(output.upstream_response_id)
            for output in outputs
            if output.upstream_response_id is not None
        ),
    )
    usage = ProviderUsage(
        input_tokens=sum(item.usage.input_tokens for item in outputs),
        output_tokens=sum(item.usage.output_tokens for item in outputs),
    )
    return DirectConversationResult(
        output="".join(
            entry.content
            for output in outputs
            for entry in public_provider_item_projection(
                output.completed_items
            )
        ),
        usage=usage,
        reasoning=outputs[-1].reasoning,
        handle=receipt.result.handle,
    )


def _direct_result_from_resource(
    result: ConversationResult,
    checkpoint: ConversationCheckpoint,
) -> DirectConversationResult:
    """Project one retrieved local resource into the direct SDK result."""
    if (
        type(result) is not ConversationResult
        or type(checkpoint) is not ConversationCheckpoint
        or not result.lane_outputs
        or result.handle.checkpoint_id != checkpoint.identity.checkpoint_id
    ):
        raise ConversationValidationError()
    public_response_id = result.public_response_id
    validate_upstream_identifier_separation(
        checkpoint,
        additional_public_identifiers=(
            (str(public_response_id),)
            if public_response_id is not None
            else ()
        ),
    )
    return DirectConversationResult(
        output="".join(
            entry.content
            for lane in result.lane_outputs
            for entry in lane.items
        ),
        usage=ProviderUsage(
            input_tokens=sum(
                lane.usage.input_tokens for lane in result.lane_outputs
            ),
            output_tokens=sum(
                lane.usage.output_tokens for lane in result.lane_outputs
            ),
        ),
        reasoning=result.reasoning,
        handle=result.handle,
    )
