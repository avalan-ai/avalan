"""Coordinate deterministic fake-lane conversation execution."""

from .agent import (
    AgentConversationLaneInvocation,
    AgentConversationLaneInvocationResult,
    AgentConversationSuspensionBoundary,
)
from .binding import (
    ConversationCapability,
    ConversationCapabilityProfile,
    ProviderFamily,
    ProviderLaneBinding,
    ProviderTransport,
)
from .codec import with_checkpoint_integrity
from .contract import (
    FAILURE_FENCES,
    AuthorityScope,
    CheckpointId,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointSequence,
    ChildLaneRetentionPolicy,
    ConversationBranchId,
    ConversationId,
    ConversationOperation,
    ExecutionSegmentId,
    FailureBoundary,
    IdempotencyDisposition,
    LogicalTurnId,
    NamedHeadRevision,
    PortableContinuationReference,
    ProviderLaneId,
    ProviderLaneOwnerKind,
    PublicResponseId,
    RequestIdempotencyIdentity,
    RetryRule,
    UpstreamResponseId,
)
from .errors import (
    ConversationAmbiguousDispatchError,
    ConversationAuthorizationError,
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationCommitError,
    ConversationConflictError,
    ConversationError,
    ConversationErrorCode,
    ConversationLimitError,
    ConversationProviderResponseError,
    ConversationPublicationError,
    ConversationTransitionError,
    ConversationValidationError,
    DurableConversationErrorCode,
)
from .execution import (
    AgentStructuredInputRequested,
    ConversationExecutionReservation,
    DurableToolRecoveryAction,
    DurableToolRecoveryAdmission,
    ProviderExecutionSegment,
    ProviderExecutionSegmentPhase,
    ProviderLaneExecutionAttestation,
    ProviderLaneExecutionReceipt,
    ProviderLaneExecutionReservation,
    ProviderLaneExecutionStage,
    ProviderToolExecution,
    ToolEffectPolicy,
    ToolExecutionPhase,
    durable_tool_recovery_action,
    provider_lane_execution_receipt,
)
from .fakes import (
    DeterministicFakeProviderDiagnostics,
    DeterministicFakeProviderScript,
    _build_deterministic_fake_provider_runtime,
    _close_deterministic_fake_provider_stream,
    _deterministic_fake_provider_diagnostics,
    _DeterministicFakeProviderRuntime,
    _DeterministicFakeProviderStreamState,
    _dispatch_deterministic_fake_provider,
    _next_deterministic_fake_provider_item,
    _open_deterministic_fake_provider_stream,
    _terminal_deterministic_fake_provider_stream,
    _validate_deterministic_fake_provider_runtime,
    _validate_fake_provider_script,
)
from .items import (
    CompactionBoundary,
    ProviderItem,
    ProviderItemCaller,
    ProviderItemKind,
    ProviderItemLedger,
    ProviderItemPhase,
    VisibleTranscript,
    VisibleTranscriptEntry,
    provider_item_byte_count,
    validate_provider_item_sequence,
)
from .lifecycle import ProviderQuarantineRequest, StoredProviderResolver
from .observability import (
    authority_digest,
    canonical_request_digest,
    checkpoint_observation,
)
from .protocols import (
    ConversationAuthorityResolver,
    ConversationClock,
    ConversationObserver,
    ConversationProviderStateSink,
    ConversationProviderStream,
    ConversationPublisher,
    ConversationRetryWaiter,
    ConversationStore,
    ConversationUnitOfWork,
    CoordinatorBoundaryHook,
    FirstStoredProviderPlan,
    ProviderPlan,
    ProviderResult,
    StandaloneCompactProviderPlan,
    StatelessProviderPlan,
    StoredProviderPlan,
)
from .providers.openai import (
    NativeOpenAIConversationLaneRuntime,
    NativeOpenAIProviderDiagnostics,
    NativeOpenAIStatelessProvider,
)
from .providers.openai_stored import (
    NativeOpenAIStoredLaneRuntime,
    NativeOpenAIStoredProvider,
)
from .runtime import (
    AtomicCommitReceipt,
    AtomicConversationCommit,
    ConversationCommitBoundary,
    ConversationLaneRequest,
    ConversationRunRequest,
    CoordinatorAwaitBoundary,
    ExplicitBranchAdvance,
    FailureDisposition,
    FirstTurnAdvance,
    IdempotencySettlementDisposition,
    IdempotencySettlementResolution,
    NamedHeadAdvance,
    OrdinaryChildAdvance,
    OutboxClaimDisposition,
    OutboxClaimTarget,
    ProviderLaneOutputCandidate,
    ProvisionalPublicResponse,
    ResetAdvance,
    StoreCloseDisposition,
    StoreCloseResolution,
    conversation_run_request_from_recovery_payload,
    conversation_run_request_recovery_payload,
    request_operation,
)
from .security import is_trusted_conversation_hardening_hook
from .settings import (
    CompactionOperation,
    ConversationMode,
    EffectiveReasoningMetadata,
    InlineCompaction,
    ProviderLaneOutputScope,
    ProviderUsage,
    ReasoningContext,
    StandaloneCompactHandle,
)
from .state import (
    CheckpointCandidate,
    CheckpointLifecycle,
    CheckpointTimestamps,
    ConversationCheckpoint,
    ExecutionSegmentCheckpointCandidate,
    MultiLaneCheckpointContent,
    NamedHeadMetadata,
    OutwardTurnCheckpointCandidate,
    ProviderLaneLifecycle,
    ProviderLaneSnapshot,
    StandaloneCompactCheckpointCandidate,
    StatelessProviderLaneSnapshot,
    StoredProviderLaneSnapshot,
    SuspensionCheckpointCandidate,
    is_standalone_compact_bridge,
    validate_checkpoint_parent_kind,
)
from .value import (
    ProviderCallId,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from asyncio import CancelledError, Task, create_task, wait
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from inspect import isawaitable, iscoroutinefunction
from typing import TypeAlias, final


@final
class _NoopCoordinatorBoundaryHook:
    async def reach(self, boundary: CoordinatorAwaitBoundary) -> None:
        assert isinstance(boundary, CoordinatorAwaitBoundary)


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConversationLaneRuntime:
    """Bind one inert fake script to its exact lane contract."""

    binding: ProviderLaneBinding
    capability_profile: ConversationCapabilityProfile
    provider_script: DeterministicFakeProviderScript
    retention_policy: ChildLaneRetentionPolicy = (
        ChildLaneRetentionPolicy.RETAIN
    )
    max_output_items: int = 1_024
    _provider_runtime: _DeterministicFakeProviderRuntime = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
        _build_provider_runtime: Callable[
            [DeterministicFakeProviderScript],
            _DeterministicFakeProviderRuntime,
        ] = _build_deterministic_fake_provider_runtime,
    ) -> None:
        _validate_lane_runtime(self, require_state=False)
        object.__setattr__(
            self,
            "_provider_runtime",
            _build_provider_runtime(self.provider_script),
        )

    @property
    def provider(
        self,
        _provider_diagnostics: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
            ],
            DeterministicFakeProviderDiagnostics,
        ] = _deterministic_fake_provider_diagnostics,
    ) -> DeterministicFakeProviderDiagnostics:
        """Return an immutable snapshot of canonical fake diagnostics."""
        runtime = _validate_lane_runtime(self)
        return _provider_diagnostics(
            runtime._provider_runtime,
            runtime.provider_script,
        )


_DETERMINISTIC_FAKE_PROVIDER_SCRIPT_TYPE = DeterministicFakeProviderScript
_DETERMINISTIC_FAKE_RUNTIME_TYPE = _DeterministicFakeProviderRuntime


def _validate_lane_runtime(
    runtime: object,
    *,
    require_state: bool = True,
    _validate_provider_runtime: Callable[
        [object, DeterministicFakeProviderScript],
        _DeterministicFakeProviderRuntime,
    ] = _validate_deterministic_fake_provider_runtime,
    _validate_script: Callable[[object], None] = (
        _validate_fake_provider_script
    ),
) -> ConversationLaneRuntime:
    if type(runtime) is not ConversationLaneRuntime:
        raise ConversationValidationError()
    try:
        binding = runtime.binding
        capability_profile = runtime.capability_profile
        provider_script = runtime.provider_script
        retention_policy = runtime.retention_policy
        max_output_items = runtime.max_output_items
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if (
        type(binding) is not ProviderLaneBinding
        or type(capability_profile) is not ConversationCapabilityProfile
        or type(provider_script)
        is not _DETERMINISTIC_FAKE_PROVIDER_SCRIPT_TYPE
        or type(retention_policy) is not ChildLaneRetentionPolicy
        or type(max_output_items) is not int
        or max_output_items <= 0
    ):
        raise ConversationValidationError()
    _validate_script(provider_script)
    capability_profile.assert_binding(binding)
    if (
        binding.provider_family is not ProviderFamily.SYNTHETIC
        or not capability_profile.test_only
        or binding.adapter_type
        != "avalan.conversation.fakes.DeterministicFakeProviderScript"
    ):
        raise ConversationCapabilityError()
    if require_state:
        try:
            provider_runtime = runtime._provider_runtime
        except AttributeError as exc:
            raise ConversationValidationError() from exc
        _validate_provider_runtime(
            provider_runtime,
            provider_script,
        )
    return runtime


def _validate_native_lane_runtime(
    runtime: object,
) -> NativeOpenAIConversationLaneRuntime:
    if type(runtime) is not NativeOpenAIConversationLaneRuntime:
        raise ConversationValidationError()
    try:
        binding = runtime.binding
        capability_profile = runtime.capability_profile
        provider = runtime.provider
        retention_policy = runtime.retention_policy
        max_output_items = runtime.max_output_items
        max_output_bytes = runtime.max_output_bytes
        max_output_segments = runtime.max_output_segments
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if (
        type(binding) is not ProviderLaneBinding
        or type(capability_profile) is not ConversationCapabilityProfile
        or type(provider) is not NativeOpenAIStatelessProvider
        or provider.binding != binding
        or provider.capability_profile != capability_profile
        or type(retention_policy) is not ChildLaneRetentionPolicy
        or type(max_output_items) is not int
        or max_output_items <= 0
        or type(max_output_bytes) is not int
        or max_output_bytes <= 0
        or type(max_output_segments) is not int
        or max_output_segments <= 0
    ):
        raise ConversationValidationError()
    capability_profile.assert_binding(binding)
    if binding.provider_family not in {
        ProviderFamily.OPENAI,
        ProviderFamily.AZURE_OPENAI,
    }:
        raise ConversationCapabilityError()
    return runtime


def _validate_stored_native_lane_runtime(
    runtime: object,
) -> NativeOpenAIStoredLaneRuntime:
    if type(runtime) is not NativeOpenAIStoredLaneRuntime:
        raise ConversationValidationError()
    try:
        binding = runtime.binding
        capability_profile = runtime.capability_profile
        provider = runtime.provider
        retention_policy = runtime.retention_policy
        max_output_items = runtime.max_output_items
        max_output_bytes = runtime.max_output_bytes
        max_output_segments = runtime.max_output_segments
    except AttributeError as exc:
        raise ConversationValidationError() from exc
    if (
        type(binding) is not ProviderLaneBinding
        or type(capability_profile) is not ConversationCapabilityProfile
        or type(provider) is not NativeOpenAIStoredProvider
        or provider.binding != binding
        or provider.capability_profile != capability_profile
        or type(retention_policy) is not ChildLaneRetentionPolicy
        or type(max_output_items) is not int
        or max_output_items <= 0
        or type(max_output_bytes) is not int
        or max_output_bytes <= 0
        or type(max_output_segments) is not int
        or max_output_segments <= 0
    ):
        raise ConversationValidationError()
    capability_profile.assert_binding(binding)
    if binding.provider_family not in {
        ProviderFamily.OPENAI,
        ProviderFamily.AZURE_OPENAI,
    }:
        raise ConversationCapabilityError()
    return runtime


NativeLaneRuntime: TypeAlias = (
    NativeOpenAIConversationLaneRuntime | NativeOpenAIStoredLaneRuntime
)

_MAX_COMPACTION_FAILURE_RECORDS = 128


def _validate_any_native_lane_runtime(
    runtime: object,
) -> NativeLaneRuntime:
    if type(runtime) is NativeOpenAIConversationLaneRuntime:
        return _validate_native_lane_runtime(runtime)
    if type(runtime) is NativeOpenAIStoredLaneRuntime:
        return _validate_stored_native_lane_runtime(runtime)
    raise ConversationValidationError()


def _validate_any_lane_runtime(
    runtime: object,
) -> ConversationLaneRuntime | NativeLaneRuntime:
    if type(runtime) is ConversationLaneRuntime:
        return _validate_lane_runtime(runtime)
    if type(runtime) is NativeOpenAIConversationLaneRuntime:
        return _validate_native_lane_runtime(runtime)
    if type(runtime) is NativeOpenAIStoredLaneRuntime:
        return _validate_stored_native_lane_runtime(runtime)
    raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionFailureRecord:
    """Record one content-free failed compaction boundary."""

    operation: CompactionOperation
    boundary: FailureBoundary
    error_code: ConversationErrorCode | DurableConversationErrorCode | str
    cancelled: bool
    committed: bool
    streaming: bool

    def __post_init__(self) -> None:
        if (
            self.operation
            not in {
                CompactionOperation.INLINE,
                CompactionOperation.STANDALONE,
            }
            or not isinstance(self.boundary, FailureBoundary)
            or type(self.cancelled) is not bool
            or type(self.committed) is not bool
            or type(self.streaming) is not bool
        ):
            raise ConversationValidationError()
        if isinstance(
            self.error_code,
            ConversationErrorCode | DurableConversationErrorCode,
        ):
            return
        if self.error_code not in {
            "conversation_cancelled",
            "conversation_internal_failure",
        }:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CoordinatorDiagnostics:
    """Report content-free active attempt and resource counts."""

    active_attempts: int
    closed: bool
    compaction_failure_count: int
    compaction_failures: tuple[CompactionFailureRecord, ...]


@final
@dataclass(slots=True)
class _AttemptStaging:
    lane_id: ProviderLaneId
    items: list[ProviderItem]
    reasoning: EffectiveReasoningMetadata | None = None
    usage: ProviderUsage = ProviderUsage()
    upstream_response_id: str | None = None
    visible_output: bool = False
    tool_effect: bool = False
    provider_output: bool = False

    def accept(self, item: ProviderItem) -> None:
        if type(item) is not ProviderItem or item.lane_id != self.lane_id:
            raise ConversationValidationError()
        self.items.append(item)
        self.provider_output = True
        if item.kind is ProviderItemKind.MESSAGE and item.phase in {
            ProviderItemPhase.ASSISTANT,
            ProviderItemPhase.FINAL,
        }:
            self.visible_output = True
        if item.caller is ProviderItemCaller.TOOL:
            self.tool_effect = True

    def finish(self, result: ProviderResult) -> None:
        if (
            type(result) is not ProviderResult
            or tuple(self.items) != result.items
        ):
            raise ConversationValidationError()
        self.reasoning = result.reasoning
        self.usage = result.usage
        self.upstream_response_id = (
            str(result.upstream_response_id)
            if result.upstream_response_id is not None
            else None
        )

    def rollback(self) -> None:
        self.items.clear()
        self.reasoning = None
        self.usage = ProviderUsage()
        self.upstream_response_id = None
        self.visible_output = False
        self.tool_effect = False
        self.provider_output = False


@final
@dataclass(slots=True)
class _DispatchProgress:
    may_have_dispatched: bool = False
    provider_output: bool = False
    tool_effect: bool = False

    def mark_possible_dispatch(self) -> None:
        self.may_have_dispatched = True

    def mark_provider_output(self) -> None:
        self.provider_output = True

    def mark_tool_effect(self) -> None:
        self.tool_effect = True


@final
@dataclass(slots=True)
class _SegmentExecutionContext:
    """Own private durable segments for one outward execution attempt."""

    request: ConversationRunRequest
    idempotency: RequestIdempotencyIdentity
    segments: list[ProviderExecutionSegment]
    visible_transcript: VisibleTranscript
    lane_snapshots: dict[ProviderLaneId, ProviderLaneSnapshot]

    def __post_init__(self) -> None:
        if (
            type(self.request) is not ConversationRunRequest
            or type(self.idempotency) is not RequestIdempotencyIdentity
            or type(self.segments) is not list
            or type(self.visible_transcript) is not VisibleTranscript
            or type(self.lane_snapshots) is not dict
            or any(
                lane_id != snapshot.lane_id
                for lane_id, snapshot in self.lane_snapshots.items()
            )
            or self.idempotency.authority != self.request.semantics.authority
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _CompletedStoredProviderResponse:
    """Retain one completed private response until checkpoint commit."""

    binding: ProviderLaneBinding
    upstream_response_id: UpstreamResponseId

    def __post_init__(self) -> None:
        if type(self.binding) is not ProviderLaneBinding:
            raise ConversationValidationError()
        validate_identifier(
            self.upstream_response_id,
            "upstream_response_id",
        )


def _remember_completed_stored_response(
    completed: list[_CompletedStoredProviderResponse] | None,
    binding: ProviderLaneBinding,
    result: ProviderResult,
) -> None:
    """Retain one validated terminal stored response exactly once."""
    upstream_response_id = result.upstream_response_id
    if completed is None or upstream_response_id is None:
        return
    target = _CompletedStoredProviderResponse(
        binding=binding,
        upstream_response_id=upstream_response_id,
    )
    if target not in completed:
        completed.append(target)


@final
class _ProviderStateSinkOwner:
    """Own one private async stream sink through terminal cleanup."""

    def __init__(self, sink: ConversationProviderStateSink) -> None:
        for method_name in ("stage", "finalize", "cleanup"):
            method = getattr(sink, method_name, None)
            if not callable(method) or not iscoroutinefunction(method):
                raise ConversationValidationError()
        self._sink = sink
        self._finalized = False
        self._cleanup_task: Task[None] | None = None
        self._cleaned = False

    @property
    def cleaned(self) -> bool:
        """Return whether the owned sidecar cleanup completed."""
        return self._cleaned

    async def stage(self, item: ProviderItem) -> None:
        """Stage one item through the private async boundary."""
        if self._finalized or self._cleaned:
            raise ConversationValidationError()
        cancelled = False
        failed = False
        try:
            await self._sink.stage(item)
        except CancelledError:
            cancelled = True
        except BaseException:
            failed = True
        if cancelled:
            raise CancelledError() from None
        if failed:
            raise ConversationCommitError() from None

    async def finalize(
        self,
        outputs: tuple[ProviderLaneOutputCandidate, ...],
    ) -> None:
        """Finalize one complete staged response exactly once."""
        if self._finalized or self._cleaned:
            raise ConversationValidationError()
        self._finalized = True
        cancelled = False
        failed = False
        try:
            await self._sink.finalize(outputs)
        except CancelledError:
            cancelled = True
        except BaseException:
            failed = True
        if cancelled:
            raise CancelledError() from None
        if failed:
            raise ConversationCommitError() from None

    async def cleanup(self) -> None:
        """Complete cancellation-safe sidecar cleanup exactly once."""
        if self._cleaned:
            return
        task = self._cleanup_task
        if task is None:
            task = create_task(self._sink.cleanup())
            self._cleanup_task = task
        caller_cancelled = False
        while not task.done():
            try:
                await wait({task})
            except CancelledError:
                caller_cancelled = True
        task_cancelled = task.cancelled()
        task_failed = False
        if not task_cancelled:
            task_failed = task.exception() is not None
        if task_cancelled:
            raise CancelledError() from None
        if task_failed:
            raise ConversationCommitError() from None
        self._cleaned = True
        if caller_cancelled:
            raise CancelledError() from None


def _apply_provider_state_cleanup_failure(
    primary_failure: BaseException | None,
    cleanup_failure: BaseException,
) -> None:
    """Preserve a primary failure or propagate the safe cleanup failure."""
    if primary_failure is None:
        raise cleanup_failure
    primary_failure.add_note("conversation provider-state cleanup failed")


LaneRuntime: TypeAlias = ConversationLaneRuntime | NativeLaneRuntime


LanePlan: TypeAlias = tuple[
    ConversationLaneRequest,
    LaneRuntime,
    ProviderPlan,
]


@final
class RunScopedConversationCoordinator:
    """Execute one run at a time without holding locks across effects."""

    def __init__(
        self,
        *,
        store: ConversationStore,
        authority_resolver: ConversationAuthorityResolver,
        clock: ConversationClock,
        publisher: ConversationPublisher,
        observer: ConversationObserver,
        retry_waiter: ConversationRetryWaiter,
        lanes: tuple[LaneRuntime, ...],
        max_attempts: int = 2,
        max_active_executions: int = 128,
        boundary_hook: CoordinatorBoundaryHook | None = None,
        hardening_required: bool = False,
    ) -> None:
        if (
            type(lanes) is not tuple
            or not lanes
            or any(
                type(item)
                not in {
                    ConversationLaneRuntime,
                    NativeOpenAIConversationLaneRuntime,
                    NativeOpenAIStoredLaneRuntime,
                }
                for item in lanes
            )
        ):
            raise ConversationValidationError()
        for item in lanes:
            _validate_any_lane_runtime(item)
        lane_ids = tuple(item.binding.lane_id for item in lanes)
        if len(lane_ids) != len(set(lane_ids)):
            raise ConversationValidationError()
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ConversationValidationError()
        if (
            type(max_active_executions) is not int
            or max_active_executions <= 0
        ):
            raise ConversationValidationError()
        hardening_hook_valid = is_trusted_conversation_hardening_hook(
            boundary_hook
        )
        if type(hardening_required) is not bool or (
            hardening_required and not hardening_hook_valid
        ):
            raise ConversationValidationError()
        self._store = store
        self._resolver = authority_resolver
        self._clock = clock
        self._publisher = publisher
        self._observer = observer
        self._retry_waiter = retry_waiter
        self._lanes = {item.binding.lane_id: item for item in lanes}
        self._fake_runtimes = {
            item.binding.lane_id: item._provider_runtime
            for item in lanes
            if type(item) is ConversationLaneRuntime
        }
        self._max_attempts = max_attempts
        self._max_active_executions = max_active_executions
        self._hook = boundary_hook or _NoopCoordinatorBoundaryHook()
        self._hardening_required = hardening_required
        self._hardening_active = hardening_required and hardening_hook_valid
        self._active_attempts: set[str] = set()
        self._compaction_failure_count = 0
        self._compaction_failures: list[CompactionFailureRecord] = []
        self._execution_sequence = 0
        self._closed = False

    @property
    def diagnostics(self) -> CoordinatorDiagnostics:
        """Return current content-free coordinator resource counts."""
        return CoordinatorDiagnostics(
            active_attempts=len(self._active_attempts),
            closed=self._closed,
            compaction_failure_count=self._compaction_failure_count,
            compaction_failures=tuple(self._compaction_failures),
        )

    @property
    def hardening_active(self) -> bool:
        """Return whether the trusted hardening hook is installed."""
        return self._hardening_active

    def assert_hardening_hook(self, hook: CoordinatorBoundaryHook) -> None:
        """Require the exact trusted hook installed by server policy."""
        if (
            self._hook is not hook
            or not self._hardening_required
            or not self.hardening_active
        ):
            raise ConversationValidationError()

    def _record_compaction_failure(
        self,
        operation: CompactionOperation,
        boundary: FailureBoundary,
        error: BaseException,
        *,
        committed: bool,
        streaming: bool,
    ) -> None:
        """Retain only closed content-free compaction failure facts."""
        if operation is CompactionOperation.NONE:
            return
        error_code: ConversationErrorCode | DurableConversationErrorCode | str
        if isinstance(error, ConversationError):
            error_code = error.code
            if boundary not in {
                FailureBoundary.CHECKPOINT_COMMIT,
                FailureBoundary.OUTWARD_PUBLICATION,
            }:
                boundary = error.boundary
        elif isinstance(error, CancelledError):
            error_code = "conversation_cancelled"
        else:
            error_code = "conversation_internal_failure"
        self._compaction_failure_count += 1
        self._compaction_failures.append(
            CompactionFailureRecord(
                operation=operation,
                boundary=boundary,
                error_code=error_code,
                cancelled=isinstance(error, CancelledError),
                committed=committed,
                streaming=streaming,
            )
        )
        if len(self._compaction_failures) > _MAX_COMPACTION_FAILURE_RECORDS:
            del self._compaction_failures[0]

    def fake_provider_diagnostics(
        self,
        lane_id: ProviderLaneId,
        _provider_diagnostics: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
            ],
            DeterministicFakeProviderDiagnostics,
        ] = _deterministic_fake_provider_diagnostics,
        _validate_runtime: Callable[
            [object], ConversationLaneRuntime
        ] = _validate_lane_runtime,
    ) -> DeterministicFakeProviderDiagnostics:
        """Return immutable diagnostics for one canonical fake lane."""
        validate_identifier(str(lane_id), "lane_id")
        runtime = self._lanes.get(lane_id)
        state = self._fake_runtimes.get(lane_id)
        if type(runtime) is not ConversationLaneRuntime or state is None:
            raise ConversationValidationError()
        _validate_runtime(runtime)
        return _provider_diagnostics(state, runtime.provider_script)

    def validate_direct_runtime(
        self,
        store: ConversationStore,
        binding: ProviderLaneBinding,
    ) -> None:
        """Validate one fake-only direct SDK runtime without allocating."""
        if self._closed or store is not self._store:
            raise ConversationValidationError()
        runtime = self._lanes.get(binding.lane_id)
        if runtime is None or runtime.binding != binding:
            raise ConversationValidationError()
        _validate_any_lane_runtime(runtime)

    def native_provider_diagnostics(
        self,
        lane_id: ProviderLaneId,
    ) -> NativeOpenAIProviderDiagnostics:
        """Return content-free diagnostics for one exact native lane."""
        validate_identifier(str(lane_id), "lane_id")
        runtime = self._lanes.get(lane_id)
        if type(runtime) is NativeOpenAIConversationLaneRuntime:
            return _validate_native_lane_runtime(runtime).provider.diagnostics
        if type(runtime) is NativeOpenAIStoredLaneRuntime:
            return _validate_stored_native_lane_runtime(
                runtime
            ).provider.diagnostics
        raise ConversationValidationError()

    async def execute(
        self,
        request: ConversationRunRequest,
        *,
        stored_provider_resolver: StoredProviderResolver | None = None,
    ) -> AtomicCommitReceipt:
        """Execute a non-streaming fake-provider run."""
        return await self._run(
            request,
            streaming=False,
            stored_provider_resolver=stored_provider_resolver,
        )

    async def execute_agent(
        self,
        request: ConversationRunRequest,
        lane_invocations: tuple[AgentConversationLaneInvocation, ...],
    ) -> AtomicCommitReceipt:
        """Execute callbacks scoped to one prepared agent invocation."""
        if (
            type(lane_invocations) is not tuple
            or not lane_invocations
            or any(
                type(item) is not AgentConversationLaneInvocation
                for item in lane_invocations
            )
        ):
            raise ConversationValidationError()
        lane_ids = tuple(item.lane_id for item in lane_invocations)
        if len(lane_ids) != len(set(lane_ids)) or set(lane_ids) != {
            lane.lane_id for lane in request.lanes
        }:
            raise ConversationValidationError()
        for invocation in lane_invocations:
            runtime = self._lanes.get(invocation.lane_id)
            if runtime is None or invocation.binding != runtime.binding:
                raise ConversationBindingDriftError()
        return await self._run(
            request,
            streaming=False,
            lane_invocations={item.lane_id: item for item in lane_invocations},
        )

    async def recover_durable_tool_execution(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> AtomicCommitReceipt:
        """Recover one exact durable tool suffix through outward commit."""
        validate_identifier(checkpoint_id, "checkpoint_id")
        if self._closed or type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        execution_token = self._activate_execution()
        owner_token: str | None = None
        committed = False
        identity: RequestIdempotencyIdentity | None = None
        try:
            resolved_authority = await self._resolver.resolve()
            if resolved_authority != authority:
                raise ConversationAuthorizationError()
            checkpoint = await self._store.load(checkpoint_id, authority)
            segments = checkpoint.content.execution_segments
            integrity = checkpoint.integrity
            if (
                checkpoint.kind
                is not CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
                or checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
                or integrity is None
                or not segments
                or any(
                    segment.recovery_request is None for segment in segments
                )
            ):
                raise ConversationValidationError()
            recovery_request = segments[0].recovery_request
            assert recovery_request is not None
            if any(
                segment.recovery_request != recovery_request
                for segment in segments
            ):
                raise ConversationConflictError()
            action = durable_tool_recovery_action(segments)
            request = conversation_run_request_from_recovery_payload(
                recovery_request,
                authority=authority,
                retention=checkpoint.retention,
                lane_topology=checkpoint.content.lane_topology,
                idempotency_key=segments[0].idempotency_key,
            )
            identity = self._idempotency(request)
            if (
                identity.request_digest != segments[0].request_digest
                or checkpoint.identity.conversation_id
                != request.identity.conversation_id
                or checkpoint.identity.logical_turn_id
                != request.identity.logical_turn_id
            ):
                raise ConversationConflictError()
            self._validate_request_lane_authority(request, authority)
            resolution_parent = await self._parent_for_runtime_resolution(
                request,
                authority,
            )
            resolved_runtimes = await self._resolve_lane_runtimes(
                request,
                resolution_parent,
                None,
            )
            execution_reservation = self._execution_reservation(
                request,
                identity,
                resolved_runtimes,
            )
            admission = DurableToolRecoveryAdmission(
                checkpoint_id=checkpoint.identity.checkpoint_id,
                checkpoint_integrity=integrity.digest,
                idempotency=identity,
                binding=segments[-1].binding,
                action=action,
                segment_count=len(segments),
            )
            lease = await self._store.admit_tool_recovery(
                admission,
                execution_reservation,
            )
            owner_token = lease.owner_token
            parent = await self._resolve_parent(request, authority)
            plans = self._plan_lanes(
                request,
                parent,
                streaming=False,
                runtimes=resolved_runtimes,
            )
            self._validate_limits_before_dispatch(request, parent, plans)
            await self._allocate(request, owner_token, authority)
            receipt = await self._recover_native_tool_suffix(
                request=request,
                parent=parent,
                checkpoint=checkpoint,
                segments=segments,
                action=action,
                plans=plans,
                execution_reservation=execution_reservation,
                owner_token=owner_token,
            )
            committed = True
            return receipt
        except BaseException:
            if (
                owner_token is not None
                and not committed
                and identity is not None
            ):
                await self._rollback(
                    identity,
                    owner_token,
                    ambiguous=True,
                )
            raise
        finally:
            self._active_attempts.discard(execution_token)

    async def stage_structured_input_suspension(
        self,
        checkpoint: ConversationCheckpoint,
        continuation: PortableContinuationReference,
    ) -> ConversationUnitOfWork:
        """Stage one exact checkpoint/portable-continuation atomic unit."""
        if self._closed:
            raise ConversationTransitionError()
        return await self._store.stage(
            SuspensionCheckpointCandidate(
                checkpoint=checkpoint,
                continuation=continuation,
            )
        )

    async def stream(
        self,
        request: ConversationRunRequest,
        *,
        stored_provider_resolver: StoredProviderResolver | None = None,
    ) -> AtomicCommitReceipt:
        """Execute a streaming fake-provider run to its terminal boundary."""
        return await self._run(
            request,
            streaming=True,
            stored_provider_resolver=stored_provider_resolver,
        )

    async def stream_with_sink(
        self,
        request: ConversationRunRequest,
        sink: ConversationProviderStateSink,
        *,
        stored_provider_resolver: StoredProviderResolver | None = None,
    ) -> AtomicCommitReceipt:
        """Execute a streamed run through one private provider-state sink."""
        return await self._run(
            request,
            streaming=True,
            sink=sink,
            stored_provider_resolver=stored_provider_resolver,
        )

    async def compact(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        """Execute one standalone stateless fake-provider compaction."""
        if request.semantics.operation is not ConversationOperation.COMPACT:
            raise ConversationValidationError()
        return await self._run(request, streaming=False)

    async def commit_compact_result(
        self,
        source: ConversationCheckpoint,
        identity: CheckpointIdentity,
        authority: AuthorityScope,
        *,
        advance: NamedHeadAdvance | None = None,
    ) -> ConversationCheckpoint:
        """Commit one explicit continuable child of private compact state."""
        if (
            self._closed
            or type(source) is not ConversationCheckpoint
            or source.kind is not CheckpointKind.STANDALONE_COMPACT_RESULT
            or source.lifecycle is not CheckpointLifecycle.COMMITTED
            or type(identity) is not CheckpointIdentity
            or type(authority) is not AuthorityScope
            or (advance is not None and type(advance) is not NamedHeadAdvance)
            or source.authority != authority
            or identity.conversation_id != source.identity.conversation_id
            or identity.parent_checkpoint_id != source.identity.checkpoint_id
            or identity.parent_sequence != source.identity.sequence
            or identity.sequence != source.identity.sequence + 1
            or any(
                not isinstance(lane, StatelessProviderLaneSnapshot)
                or lane.compaction_boundary is None
                for lane in source.content.lanes
            )
        ):
            raise ConversationValidationError()
        if advance is not None and (
            source.head is None
            or source.head.head_id != advance.head_id
            or source.head.revision != advance.expected_revision + 1
            or source.identity.parent_checkpoint_id
            != advance.parent_checkpoint_id
        ):
            raise ConversationValidationError()
        now = await self._clock.now()
        expires_at = source.timestamps.expires_at
        if expires_at is not None and expires_at <= now:
            raise ConversationTransitionError()
        candidate = ExecutionSegmentCheckpointCandidate(
            checkpoint=with_checkpoint_integrity(
                ConversationCheckpoint(
                    identity=identity,
                    kind=CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
                    lifecycle=CheckpointLifecycle.STAGED,
                    authority=authority,
                    content=source.content,
                    timestamps=CheckpointTimestamps(
                        created_at=now,
                        expires_at=expires_at,
                    ),
                    retention=source.retention,
                    head=source.head if advance is not None else None,
                )
            )
        )
        try:
            if advance is None:
                return await self._store.create(candidate)
            return await self._store.create_with_named_head(
                candidate,
                advance,
            )
        except CancelledError as error:
            recovered = await self._recover_compact_commit(
                candidate.checkpoint,
                authority,
                advance=advance,
            )
            self._record_compaction_failure(
                CompactionOperation.STANDALONE,
                FailureBoundary.CHECKPOINT_COMMIT,
                error,
                committed=recovered is not None,
                streaming=False,
            )
            raise
        except ConversationConflictError as error:
            recovered = await self._recover_compact_commit(
                candidate.checkpoint,
                authority,
                advance=advance,
            )
            if recovered is not None:
                return recovered
            self._record_compaction_failure(
                CompactionOperation.STANDALONE,
                FailureBoundary.CHECKPOINT_COMMIT,
                error,
                committed=False,
                streaming=False,
            )
            raise
        except ConversationError as error:
            self._record_compaction_failure(
                CompactionOperation.STANDALONE,
                FailureBoundary.CHECKPOINT_COMMIT,
                error,
                committed=False,
                streaming=False,
            )
            raise
        except Exception:
            recovered = await self._recover_compact_commit(
                candidate.checkpoint,
                authority,
                advance=advance,
            )
            if recovered is not None:
                return recovered
            commit_failure = ConversationCommitError()
            self._record_compaction_failure(
                CompactionOperation.STANDALONE,
                FailureBoundary.CHECKPOINT_COMMIT,
                commit_failure,
                committed=False,
                streaming=False,
            )
            raise commit_failure from None

    async def _recover_compact_commit(
        self,
        expected: ConversationCheckpoint,
        authority: AuthorityScope,
        *,
        advance: NamedHeadAdvance | None = None,
    ) -> ConversationCheckpoint | None:
        """Recover only one fully committed exact compact child."""
        try:
            recovered = await self._store.load(
                expected.identity.checkpoint_id,
                authority,
            )
        except Exception:
            return None
        if (
            recovered.lifecycle is not CheckpointLifecycle.COMMITTED
            or recovered.kind is not expected.kind
            or recovered.identity != expected.identity
            or recovered.authority != expected.authority
            or recovered.content != expected.content
            or recovered.retention != expected.retention
            or recovered.head != expected.head
            or recovered.timestamps.created_at
            != expected.timestamps.created_at
            or recovered.timestamps.expires_at
            != expected.timestamps.expires_at
            or recovered.integrity is None
        ):
            return None
        if advance is not None:
            try:
                head = await self._store.load_head(
                    advance.head_id,
                    authority,
                )
            except Exception:
                return None
            if (
                head.revision != advance.expected_revision + 1
                or head.checkpoint_id != expected.identity.checkpoint_id
            ):
                return None
        return recovered

    async def close(self) -> None:
        """Close the owned store after all run-scoped attempts finish."""
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(CoordinatorAwaitBoundary.CLOSE)
        except CancelledError as exc:
            cancellation = exc
        if self._active_attempts:
            conflict = ConversationConflictError()
            if cancellation is not None:
                raise cancellation from conflict
            raise conflict
        if self._closed:
            if cancellation is not None:
                raise cancellation
            return
        provider_error: BaseException | None = None
        try:
            await self._close_native_providers()
        except CancelledError as exc:
            cancellation = cancellation or exc
            provider_error = exc.__cause__
        except BaseException as exc:
            provider_error = exc
        action: StoreCloseResolution | None = None
        action_error: BaseException | None = None
        try:
            action = await self._store.close()
        except CancelledError as exc:
            cancellation = cancellation or exc
            action_error = exc
        except BaseException as exc:
            action_error = exc
        probe: StoreCloseResolution | None = None
        probe_error: BaseException | None = None
        try:
            probe = await self._store.inspect_close()
        except CancelledError as exc:
            cancellation = cancellation or exc
            probe_error = exc
        except BaseException as exc:
            probe_error = exc
        consistency_error: BaseException | None = None
        if probe_error is not None:
            consistency_error = probe_error
        elif type(probe) is not StoreCloseResolution:
            consistency_error = ConversationValidationError()
        else:
            assert probe is not None
            if (
                probe.disposition is StoreCloseDisposition.CLOSED
                and provider_error is None
            ):
                self._closed = True
            if action_error is None:
                if type(action) is not StoreCloseResolution:
                    consistency_error = ConversationValidationError()
                elif action != probe:
                    consistency_error = ConversationConflictError()
                elif probe.disposition is not StoreCloseDisposition.CLOSED:
                    consistency_error = ConversationConflictError()
        if cancellation is not None:
            cause = (
                consistency_error
                if consistency_error is not cancellation
                else None
            )
            if cause is None and action_error is not cancellation:
                cause = action_error
            if cause is None:
                cause = provider_error
            if cause is not None:
                raise cancellation from cause
            raise cancellation
        if provider_error is not None:
            if action_error is not None:
                raise provider_error from action_error
            if consistency_error is not None:
                raise provider_error from consistency_error
            raise provider_error
        if action_error is not None:
            if consistency_error is not None:
                raise action_error from consistency_error
            raise action_error
        if consistency_error is not None:
            raise consistency_error

    async def _close_native_providers(self) -> None:
        closed: set[int] = set()
        cancellation: CancelledError | None = None
        cleanup_error: BaseException | None = None
        for runtime in self._lanes.values():
            provider: (
                NativeOpenAIStatelessProvider | NativeOpenAIStoredProvider
            )
            if type(runtime) is NativeOpenAIConversationLaneRuntime:
                provider = _validate_native_lane_runtime(runtime).provider
            elif type(runtime) is NativeOpenAIStoredLaneRuntime:
                provider = _validate_stored_native_lane_runtime(
                    runtime
                ).provider
            else:
                continue
            identity = id(provider)
            if identity in closed:
                continue
            closed.add(identity)
            try:
                await provider.aclose()
            except CancelledError as exc:
                cancellation = cancellation or exc
                try:
                    await provider.aclose()
                except CancelledError:
                    cleanup_error = cleanup_error or ConversationCommitError()
                except BaseException:
                    cleanup_error = cleanup_error or ConversationCommitError()
            except BaseException:
                cleanup_error = cleanup_error or ConversationCommitError()
        if cancellation is not None:
            if cleanup_error is not None:
                raise cancellation from cleanup_error
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error

    async def _run(
        self,
        request: ConversationRunRequest,
        *,
        streaming: bool,
        sink: ConversationProviderStateSink | None = None,
        stored_provider_resolver: StoredProviderResolver | None = None,
        lane_invocations: (
            Mapping[
                ProviderLaneId,
                AgentConversationLaneInvocation,
            ]
            | None
        ) = None,
    ) -> AtomicCommitReceipt:
        if self._closed or type(request) is not ConversationRunRequest:
            raise ConversationValidationError()
        if sink is not None and not streaming:
            raise ConversationValidationError()
        if lane_invocations is not None and (
            streaming
            or sink is not None
            or set(lane_invocations)
            != {lane.lane_id for lane in request.lanes}
        ):
            raise ConversationValidationError()
        if (
            stored_provider_resolver is not None
            and type(stored_provider_resolver) is not StoredProviderResolver
        ):
            raise ConversationValidationError()
        compaction_operation = (
            CompactionOperation.STANDALONE
            if request.semantics.operation is ConversationOperation.COMPACT
            else (
                CompactionOperation.INLINE
                if any(
                    type(lane.compaction) is InlineCompaction
                    for lane in request.lanes
                )
                else CompactionOperation.NONE
            )
        )
        failure_boundary = FailureBoundary.VALIDATION_BEFORE_DISPATCH
        sink_owner = (
            _ProviderStateSinkOwner(sink) if sink is not None else None
        )
        execution_token = self._activate_execution()
        identity = self._idempotency(request)
        owner_token: str | None = None
        committed = False
        progress = _DispatchProgress()
        completed_stored: list[_CompletedStoredProviderResponse] = []
        quarantine_at: datetime | None = None
        primary_failure: BaseException | None = None
        try:
            await self._hook.reach(CoordinatorAwaitBoundary.RESOLVE_AUTHORITY)
            authority = await self._resolver.resolve()
            if authority != request.semantics.authority:
                raise ConversationAuthorizationError()
            self._validate_request_lane_authority(request, authority)
            await self._hook.reach(
                CoordinatorAwaitBoundary.RESERVE_IDEMPOTENCY
            )
            resolution_parent = await self._parent_for_runtime_resolution(
                request,
                authority,
            )
            resolved_runtimes = await self._resolve_lane_runtimes(
                request,
                resolution_parent,
                stored_provider_resolver,
            )
            execution_reservation = self._execution_reservation(
                request,
                identity,
                resolved_runtimes,
            )
            resolution = await self._store.reserve_idempotency(
                identity,
                execution=execution_reservation,
            )
            if resolution.disposition is IdempotencyDisposition.CONFLICT:
                raise ConversationConflictError()
            if resolution.disposition is IdempotencyDisposition.FENCED:
                raise ConversationAmbiguousDispatchError()
            if (
                resolution.disposition
                is IdempotencyDisposition.REPLAY_COMMITTED
            ):
                committed = True
                assert resolution.checkpoint_id is not None
                checkpoint = await self._store.load(
                    resolution.checkpoint_id, authority
                )
                output_candidates = (
                    await self._store.retrieve_output_candidates(
                        resolution.checkpoint_id, authority
                    )
                )
                result = None
                if resolution.public_response_id is not None:
                    result = await self._store.retrieve(
                        resolution.public_response_id, authority
                    )
                    failure_boundary = FailureBoundary.OUTWARD_PUBLICATION
                    await self._publish_one(
                        checkpoint,
                        resolution.public_response_id,
                        f"publication-{resolution.public_response_id}",
                    )
                if sink_owner is not None:
                    await sink_owner.finalize(output_candidates)
                    await sink_owner.cleanup()
                return AtomicCommitReceipt(
                    checkpoint=checkpoint,
                    result=result,
                    outbox=None,
                    output_candidates=output_candidates,
                )
            assert resolution.owner_token is not None
            owner_token = resolution.owner_token
            await self._hook.reach(CoordinatorAwaitBoundary.RESOLVE_PARENT)
            parent = await self._resolve_parent(request, authority)
            await self._hook.reach(CoordinatorAwaitBoundary.VALIDATE_PLAN)
            plans = self._plan_lanes(
                request,
                parent,
                streaming=streaming,
                runtimes=resolved_runtimes,
            )
            self._validate_limits_before_dispatch(request, parent, plans)
            await self._allocate(request, owner_token, authority)
            quarantine_at = await self._clock.now()
            failure_boundary = FailureBoundary.FAILURE_BEFORE_OUTPUT
            (
                snapshots,
                output_candidates,
                execution_attestations,
                execution_segments,
            ) = await self._dispatch_lanes(
                plans,
                execution_reservation=execution_reservation,
                owner_token=owner_token,
                authority=authority,
                request=request,
                parent=parent,
                streaming=streaming,
                progress=progress,
                sink=sink_owner,
                completed_stored=completed_stored,
                lane_invocations=lane_invocations,
            )
            if request.semantics.operation is ConversationOperation.COMPACT:
                self._validate_compact_outputs(output_candidates)
            if sink_owner is not None:
                await sink_owner.finalize(output_candidates)
                await sink_owner.cleanup()
            now = await self._clock.now()
            agent_visible_delta = (
                self._agent_child_visible_delta(
                    request,
                    output_candidates,
                )
                if lane_invocations is not None
                else ()
            )
            candidate = build_checkpoint_candidate(
                request,
                parent=parent,
                completed_lanes=snapshots,
                created_at=now,
                execution_segments=execution_segments,
                additional_visible_delta=agent_visible_delta,
            )
            commit = AtomicConversationCommit(
                candidate=candidate,
                idempotency=identity,
                owner_token=owner_token,
                output_candidates=output_candidates,
                committed_at=now,
                result_mode=(
                    ConversationMode.STORED
                    if any(
                        isinstance(lane, StoredProviderLaneSnapshot)
                        for lane in candidate.checkpoint.content.lanes
                    )
                    else ConversationMode.STATELESS
                ),
                execution_attestations=execution_attestations,
                provisional_response_id=request.provisional_response_id,
                public_response_id=request.public_response_id,
                outbox_intent_id=(
                    f"publication-{request.public_response_id}"
                    if request.public_response_id is not None
                    else None
                ),
                head_id=(
                    request.advance.head_id
                    if isinstance(request.advance, NamedHeadAdvance)
                    and request.semantics.operation
                    is not ConversationOperation.COMPACT
                    else None
                ),
                expected_head_revision=(
                    request.advance.expected_revision
                    if isinstance(request.advance, NamedHeadAdvance)
                    and request.semantics.operation
                    is not ConversationOperation.COMPACT
                    else None
                ),
            )
            failure_boundary = FailureBoundary.CHECKPOINT_COMMIT
            await self._hook.reach(CoordinatorAwaitBoundary.COMMIT)
            recovered_commit = False
            try:
                receipt = await self._store.commit_atomic(commit)
            except (Exception, CancelledError) as commit_error:
                recovered = await self._recover_atomic_commit(
                    candidate,
                    output_candidates,
                    request.public_response_id,
                )
                if recovered is not None:
                    receipt = recovered
                    committed = True
                    recovered_commit = True
                    if isinstance(commit_error, CancelledError):
                        raise
                else:
                    if isinstance(
                        commit_error,
                        ConversationError | CancelledError,
                    ):
                        raise
                    raise ConversationCommitError() from commit_error
            committed = True
            await self._observe("checkpoint_committed", receipt.checkpoint)
            failure_boundary = FailureBoundary.OUTWARD_PUBLICATION
            if receipt.outbox is not None:
                await self._publish_one(
                    receipt.checkpoint,
                    receipt.outbox.intent.public_response_id,
                    receipt.outbox.intent.intent_id,
                )
            elif recovered_commit and request.public_response_id is not None:
                await self._publish_one(
                    receipt.checkpoint,
                    request.public_response_id,
                    f"publication-{request.public_response_id}",
                )
            return receipt
        except BaseException as exc:
            primary_failure = exc
            self._record_compaction_failure(
                compaction_operation,
                failure_boundary,
                exc,
                committed=committed,
                streaming=streaming,
            )
            quarantine_error: BaseException | None = None
            if owner_token is not None and not committed:
                if completed_stored and not isinstance(
                    exc,
                    AgentConversationSuspensionBoundary,
                ):
                    try:
                        await self._persist_completed_upstream_quarantine(
                            request,
                            tuple(completed_stored),
                            quarantine_at or datetime.now(UTC),
                        )
                    except BaseException as cleanup_exc:
                        quarantine_error = cleanup_exc
                try:
                    await self._rollback(
                        identity,
                        owner_token,
                        ambiguous=(
                            not isinstance(
                                exc,
                                AgentConversationSuspensionBoundary,
                            )
                            and (
                                isinstance(
                                    exc,
                                    ConversationAmbiguousDispatchError,
                                )
                                or progress.may_have_dispatched
                                and not isinstance(
                                    exc,
                                    ConversationProviderResponseError,
                                )
                                and not (
                                    isinstance(exc, ConversationError)
                                    and exc.boundary
                                    is FailureBoundary.PROVIDER_REJECTION
                                )
                            )
                        ),
                    )
                except BaseException as cleanup_exc:
                    if isinstance(exc, CancelledError):
                        raise exc from cleanup_exc
                    raise cleanup_exc from exc
            if quarantine_error is not None:
                if isinstance(exc, CancelledError):
                    raise exc from quarantine_error
                failure = ConversationCommitError()
                failure.add_note(
                    "provider cleanup quarantine could not be persisted"
                )
                raise failure from quarantine_error
            raise
        finally:
            cleanup_failure: BaseException | None = None
            if sink_owner is not None and not sink_owner.cleaned:
                try:
                    await sink_owner.cleanup()
                except CancelledError as error:
                    cleanup_failure = error
                except BaseException:
                    cleanup_failure = ConversationCommitError()
            self._active_attempts.discard(execution_token)
            if cleanup_failure is not None:
                _apply_provider_state_cleanup_failure(
                    primary_failure,
                    cleanup_failure,
                )

    async def _recover_atomic_commit(
        self,
        candidate: CheckpointCandidate,
        outputs: tuple[ProviderLaneOutputCandidate, ...],
        public_response_id: PublicResponseId | None,
    ) -> AtomicCommitReceipt | None:
        """Resolve a possible post-commit acknowledgement failure."""
        checkpoint = candidate.checkpoint
        try:
            committed = await self._store.load(
                checkpoint.identity.checkpoint_id,
                checkpoint.authority,
            )
        except ConversationAuthorizationError:
            return None
        recovered_outputs = await self._store.retrieve_output_candidates(
            checkpoint.identity.checkpoint_id,
            checkpoint.authority,
        )
        if recovered_outputs != outputs:
            raise ConversationConflictError()
        result = None
        if public_response_id is not None:
            result = await self._store.retrieve(
                public_response_id,
                checkpoint.authority,
            )
        return AtomicCommitReceipt(
            checkpoint=committed,
            result=result,
            outbox=None,
            output_candidates=recovered_outputs,
        )

    async def _quarantine_completed_upstream(
        self,
        request: ConversationRunRequest,
        completed: tuple[_CompletedStoredProviderResponse, ...],
        at: datetime,
    ) -> None:
        """Persist private cleanup work for completed stored outputs."""
        if not completed:
            return
        unique = tuple(
            {
                (
                    str(item.binding.integrity_digest),
                    str(item.upstream_response_id),
                ): item
                for item in completed
            }.values()
        )
        candidates: list[ExecutionSegmentCheckpointCandidate] = []
        for target in unique:
            digest = sha256(
                canonical_json_bytes(
                    {
                        "source_checkpoint_id": request.identity.checkpoint_id,
                        "binding_digest": target.binding.integrity_digest,
                        "upstream_response_id": target.upstream_response_id,
                    }
                )
            ).hexdigest()
            prefix = f"quarantine-{digest}"
            quarantine = with_checkpoint_integrity(
                ConversationCheckpoint(
                    identity=CheckpointIdentity(
                        conversation_id=ConversationId(
                            f"{prefix}-conversation"
                        ),
                        logical_turn_id=LogicalTurnId(f"{prefix}-turn"),
                        execution_segment_id=ExecutionSegmentId(
                            f"{prefix}-segment"
                        ),
                        checkpoint_id=CheckpointId(prefix),
                        branch_id=ConversationBranchId(f"{prefix}-branch"),
                        sequence=CheckpointSequence(0),
                    ),
                    kind=CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
                    lifecycle=CheckpointLifecycle.STAGED,
                    authority=request.semantics.authority,
                    content=MultiLaneCheckpointContent(
                        visible_transcript=VisibleTranscript(entries=()),
                        lanes=(
                            StoredProviderLaneSnapshot(
                                binding=target.binding,
                                upstream_response_id=(
                                    target.upstream_response_id
                                ),
                                reasoning=EffectiveReasoningMetadata(
                                    requested=ReasoningContext.AUTO,
                                    effective=None,
                                ),
                                lifecycle=ProviderLaneLifecycle.COMMITTED,
                                retention_policy=(
                                    ChildLaneRetentionPolicy.RETAIN
                                ),
                            ),
                        ),
                    ),
                    timestamps=CheckpointTimestamps(
                        created_at=at,
                        expires_at=at
                        + timedelta(
                            seconds=(
                                request.retention.effective_ttl_seconds
                                or 604_800
                            )
                        ),
                    ),
                    retention=request.retention,
                )
            )
            candidates.append(
                ExecutionSegmentCheckpointCandidate(checkpoint=quarantine)
            )
        await self._store.quarantine_provider_checkpoint(
            ProviderQuarantineRequest(
                candidate=candidates[0],
                created_at=at,
                additional_candidates=tuple(candidates[1:]),
            )
        )

    async def _persist_completed_upstream_quarantine(
        self,
        request: ConversationRunRequest,
        completed: tuple[_CompletedStoredProviderResponse, ...],
        at: datetime,
    ) -> None:
        """Own quarantine persistence through caller cancellation."""
        task = create_task(
            self._quarantine_completed_upstream(request, completed, at)
        )
        while not task.done():
            try:
                await wait({task})
            except CancelledError:
                continue
        task.result()

    @staticmethod
    def _validate_compact_outputs(
        outputs: tuple[ProviderLaneOutputCandidate, ...],
    ) -> None:
        if (
            type(outputs) is not tuple
            or not outputs
            or any(
                output.mode is not ConversationMode.STATELESS
                or not output.completed_items
                or output.completed_items[-1].kind
                is not ProviderItemKind.COMPACTION
                or sum(
                    item.kind is ProviderItemKind.COMPACTION
                    for item in output.completed_items
                )
                != 1
                or output.public_output.items
                for output in outputs
            )
        ):
            raise ConversationValidationError()

    def _execution_reservation(
        self,
        request: ConversationRunRequest,
        idempotency: RequestIdempotencyIdentity,
        runtimes: Mapping[ProviderLaneId, LaneRuntime],
    ) -> ConversationExecutionReservation:
        lanes: list[ProviderLaneExecutionReservation] = []
        for lane_request in request.lanes:
            runtime = runtimes.get(lane_request.lane_id)
            if runtime is None:
                raise ConversationCapabilityError()
            lanes.append(
                ProviderLaneExecutionReservation(
                    binding=runtime.binding,
                    mode=lane_request.mode,
                    scope=ProviderLaneOutputScope.CURRENT_CALL,
                )
            )
        return ConversationExecutionReservation(
            idempotency=idempotency,
            identity=request.identity,
            lanes=tuple(lanes),
            authorized_agent_ids=(
                ()
                if request.lane_topology is None
                else tuple(
                    entry.agent_id for entry in request.lane_topology.entries
                )
            ),
        )

    def _validate_request_lane_authority(
        self,
        request: ConversationRunRequest,
        authority: AuthorityScope,
    ) -> None:
        """Authorize child lanes only through exact persisted topology."""
        topology = request.lane_topology
        if topology is None:
            if any(
                (runtime := self._lanes.get(lane_request.lane_id)) is not None
                and runtime.binding.agent_id != authority.agent_id
                for lane_request in request.lanes
            ):
                raise ConversationAuthorizationError()
            return
        roots = tuple(
            entry
            for entry in topology.entries
            if entry.owner_kind
            in {
                ProviderLaneOwnerKind.DIRECT_MODEL,
                ProviderLaneOwnerKind.PARENT_AGENT,
            }
            and entry.agent_id == authority.agent_id
        )
        if not roots:
            raise ConversationAuthorizationError()
        for lane_request in request.lanes:
            runtime = self._lanes.get(lane_request.lane_id)
            if runtime is None:
                continue
            entry = topology.entry(lane_request.lane_id)
            if (
                runtime.binding.agent_id != entry.agent_id
                or runtime.binding.integrity_digest != entry.binding_digest
                or runtime.retention_policy != entry.retention_policy
            ):
                raise ConversationAuthorizationError()

    async def _parent_for_runtime_resolution(
        self,
        request: ConversationRunRequest,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint | None:
        """Load only the parent needed to select an exact stored runtime."""
        advance = request.advance
        if isinstance(advance, FirstTurnAdvance | ResetAdvance):
            return None
        return await self._store.load(advance.parent_checkpoint_id, authority)

    async def _resolve_lane_runtimes(
        self,
        request: ConversationRunRequest,
        parent: ConversationCheckpoint | None,
        resolver: StoredProviderResolver | None,
    ) -> dict[ProviderLaneId, LaneRuntime]:
        """Resolve current or retained exact execution runtimes."""
        prior_lanes = (
            {lane.lane_id: lane for lane in parent.content.lanes}
            if parent is not None
            else {}
        )
        selected: dict[ProviderLaneId, LaneRuntime] = {}
        for lane_request in request.lanes:
            current = self._lanes.get(lane_request.lane_id)
            if current is None:
                raise ConversationCapabilityError()
            prior = prior_lanes.get(lane_request.lane_id)
            if prior is None or prior.binding == current.binding:
                selected[lane_request.lane_id] = current
                continue
            if (
                not isinstance(prior, StoredProviderLaneSnapshot)
                or lane_request.mode is not ConversationMode.STORED
                or resolver is None
            ):
                raise ConversationBindingDriftError()
            resolved = await resolver.resolve_continuation_runtime(
                prior.binding.integrity_digest
            )
            retired = _validate_stored_native_lane_runtime(resolved)
            if (
                retired.binding != prior.binding
                or retired.binding.lane_id != lane_request.lane_id
            ):
                raise ConversationBindingDriftError()
            selected[lane_request.lane_id] = retired
        return selected

    async def _resolve_parent(
        self,
        request: ConversationRunRequest,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint | None:
        advance = request.advance
        if isinstance(advance, FirstTurnAdvance):
            return None
        parent_id = advance.parent_checkpoint_id
        parent = await self._store.load(parent_id, authority)
        if isinstance(advance, ResetAdvance):
            return None
        child_kind = (
            CheckpointKind.STANDALONE_COMPACT_RESULT
            if request.semantics.operation is ConversationOperation.COMPACT
            else (
                CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
                if request.boundary
                is ConversationCommitBoundary.INTERNAL_SEGMENT
                else CheckpointKind.COMPLETED_OUTWARD_TURN
            )
        )
        compact_source = None
        if (
            parent.kind is CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
            and parent.identity.parent_checkpoint_id is not None
        ):
            compact_source = await self._store.load(
                parent.identity.parent_checkpoint_id,
                authority,
            )
        validate_checkpoint_parent_kind(
            child_kind,
            parent.kind,
            compact_continuation=is_standalone_compact_bridge(
                parent,
                compact_source,
            ),
        )
        identity = request.identity
        if (
            identity.conversation_id != parent.identity.conversation_id
            or identity.parent_sequence != parent.identity.sequence
            or identity.sequence != parent.identity.sequence + 1
        ):
            raise ConversationValidationError()
        if isinstance(advance, OrdinaryChildAdvance | NamedHeadAdvance):
            if identity.branch_id != parent.identity.branch_id:
                raise ConversationValidationError()
        elif isinstance(advance, ExplicitBranchAdvance) and (
            identity.branch_id == parent.identity.branch_id
        ):
            raise ConversationValidationError()
        if isinstance(advance, NamedHeadAdvance):
            head = await self._store.load_head(advance.head_id, authority)
            if (
                head.checkpoint_id != parent_id
                or head.revision != advance.expected_revision
            ):
                raise ConversationConflictError()
        return parent

    def _plan_lanes(
        self,
        request: ConversationRunRequest,
        parent: ConversationCheckpoint | None,
        *,
        streaming: bool,
        runtimes: Mapping[ProviderLaneId, LaneRuntime] | None = None,
    ) -> tuple[LanePlan, ...]:
        parent_lanes = (
            {lane.lane_id: lane for lane in parent.content.lanes}
            if parent is not None
            else {}
        )
        plans: list[LanePlan] = []
        for lane_request in request.lanes:
            runtime = (
                runtimes.get(lane_request.lane_id)
                if runtimes is not None
                else self._lanes.get(lane_request.lane_id)
            )
            if runtime is None:
                raise ConversationCapabilityError()
            runtime.capability_profile.assert_binding(runtime.binding)
            self._require_capabilities(
                lane_request,
                runtime,
                streaming=streaming,
                standalone_compaction=(
                    request.semantics.operation
                    is ConversationOperation.COMPACT
                ),
            )
            prior = parent_lanes.get(lane_request.lane_id)
            if prior is not None:
                prior.binding.assert_compatible(runtime.binding)
            reasoning = EffectiveReasoningMetadata(
                requested=lane_request.reasoning_context,
                effective=None,
            )
            if lane_request.mode is ConversationMode.STATELESS:
                if prior is not None and not isinstance(
                    prior, StatelessProviderLaneSnapshot
                ):
                    raise ConversationValidationError()
                ledger = (
                    prior.ledger
                    if isinstance(prior, StatelessProviderLaneSnapshot)
                    else ProviderItemLedger(
                        lane_id=lane_request.lane_id,
                        normalization_version=runtime.binding.continuation_codec_version,
                        items=(),
                    )
                )
                new_input = None
                if type(runtime) in {
                    NativeOpenAIConversationLaneRuntime,
                    NativeOpenAIStoredLaneRuntime,
                }:
                    semantic_input = request.semantics.semantic_input
                    if not isinstance(semantic_input, Mapping):
                        raise ConversationValidationError()
                    new_input = semantic_input
                plan: ProviderPlan
                if (
                    request.semantics.operation
                    is ConversationOperation.COMPACT
                ):
                    plan = StandaloneCompactProviderPlan(
                        binding=runtime.binding,
                        ledger=ledger,
                        reasoning=(
                            prior.reasoning
                            if isinstance(prior, StatelessProviderLaneSnapshot)
                            else reasoning
                        ),
                    )
                else:
                    plan = StatelessProviderPlan(
                        binding=runtime.binding,
                        ledger=ledger,
                        reasoning=reasoning,
                        compaction=lane_request.compaction,
                        new_input=new_input,
                    )
            else:
                if prior is not None and not isinstance(
                    prior, StoredProviderLaneSnapshot
                ):
                    raise ConversationValidationError()
                stored_new_input = None
                if type(runtime) is NativeOpenAIStoredLaneRuntime:
                    semantic_input = request.semantics.semantic_input
                    if not isinstance(semantic_input, Mapping):
                        raise ConversationValidationError()
                    stored_new_input = semantic_input
                if isinstance(prior, StoredProviderLaneSnapshot):
                    plan = StoredProviderPlan(
                        binding=runtime.binding,
                        upstream_response_id=prior.upstream_response_id,
                        reasoning=reasoning,
                        compaction=lane_request.compaction,
                        new_input=stored_new_input,
                    )
                else:
                    plan = FirstStoredProviderPlan(
                        binding=runtime.binding,
                        reasoning=reasoning,
                        compaction=lane_request.compaction,
                        new_input=stored_new_input,
                    )
            plans.append((lane_request, runtime, plan))
        return tuple(plans)

    @staticmethod
    def _require_capabilities(
        lane: ConversationLaneRequest,
        runtime: LaneRuntime,
        *,
        streaming: bool,
        standalone_compaction: bool,
    ) -> None:
        expected_transport = (
            ProviderTransport.STREAMING
            if streaming
            else ProviderTransport.NON_STREAMING
        )
        if (
            type(runtime)
            in {
                NativeOpenAIConversationLaneRuntime,
                NativeOpenAIStoredLaneRuntime,
            }
            and runtime.binding.transport is not expected_transport
        ):
            raise ConversationBindingDriftError()
        required = [
            (
                ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY
                if lane.mode is ConversationMode.STATELESS
                else ConversationCapability.STORED_RESPONSES_CHAINING
            )
        ]
        if lane.reasoning_context is ReasoningContext.CURRENT_TURN:
            required.append(
                ConversationCapability.REASONING_CONTEXT_CURRENT_TURN
            )
        elif lane.reasoning_context is ReasoningContext.ALL_TURNS:
            required.append(ConversationCapability.REASONING_CONTEXT_ALL_TURNS)
        if streaming:
            required.append(ConversationCapability.STREAMING_ITEM_FIDELITY)
        if isinstance(lane.compaction, InlineCompaction):
            required.append(ConversationCapability.INLINE_COMPACTION)
        if standalone_compaction:
            required.append(ConversationCapability.STANDALONE_COMPACTION)
        for capability in required:
            runtime.capability_profile.require(capability)

    @staticmethod
    def _validate_limits_before_dispatch(
        request: ConversationRunRequest,
        parent: ConversationCheckpoint | None,
        plans: tuple[LanePlan, ...],
    ) -> None:
        if (
            len(canonical_json_bytes(request.semantics.semantic_input))
            > 1_048_576
        ):
            raise ConversationValidationError()
        if sum(
            len(entry.content.encode("utf-8"))
            for entry in request.visible_delta
        ) > (1_048_576):
            raise ConversationValidationError()
        parent_items = (
            parent.content.safe_counts.provider_item_count if parent else 0
        )
        if (
            parent_items + sum(item[1].max_output_items for item in plans)
            > 10_000
        ):
            raise ConversationValidationError()
        for lane, runtime, plan in plans:
            if type(plan) is StandaloneCompactProviderPlan:
                if type(runtime) is not NativeOpenAIConversationLaneRuntime:
                    continue
                runtime.provider.validate_compaction_request(
                    plan,
                    runtime.binding.transport,
                )
            elif type(lane.compaction) is InlineCompaction:
                if type(runtime) is NativeOpenAIConversationLaneRuntime:
                    runtime.provider.validate_compaction_request(
                        plan,
                        runtime.binding.transport,
                    )
                elif type(runtime) is NativeOpenAIStoredLaneRuntime:
                    runtime.provider.validate_compaction_request(
                        plan,
                        runtime.binding.transport,
                    )

    @staticmethod
    def _agent_child_visible_delta(
        request: ConversationRunRequest,
        outputs: tuple[ProviderLaneOutputCandidate, ...],
    ) -> tuple[VisibleTranscriptEntry, ...]:
        """Project only child final messages in frozen topology order."""
        topology = request.lane_topology
        if topology is None:
            raise ConversationValidationError()
        by_lane = {output.lane_id: output for output in outputs}
        if len(by_lane) != len(outputs):
            raise ConversationValidationError()
        entries: list[VisibleTranscriptEntry] = []
        for lane in topology.entries:
            if lane.owner_kind is not ProviderLaneOwnerKind.CHILD_AGENT:
                continue
            output = by_lane.get(lane.lane_id)
            if (
                output is None
                or output.binding.integrity_digest != lane.binding_digest
            ):
                raise ConversationBindingDriftError()
            entries.extend(output.public_output.items)
        visible = tuple(entries)
        if (
            sum(len(entry.content.encode("utf-8")) for entry in visible)
            > 1_048_576
        ):
            raise ConversationValidationError()
        return visible

    async def _allocate(
        self,
        request: ConversationRunRequest,
        owner_token: str,
        authority: AuthorityScope,
    ) -> None:
        if request.provisional_response_id is None:
            return
        assert request.public_response_id is not None
        await self._hook.reach(CoordinatorAwaitBoundary.ALLOCATE_RESPONSE)
        await self._store.allocate_public_response(
            ProvisionalPublicResponse(
                provisional_response_id=request.provisional_response_id,
                public_response_id=request.public_response_id,
                owner_token=owner_token,
                authority_digest=str(authority_digest(authority)),
            )
        )

    async def _dispatch_lanes(
        self,
        plans: tuple[LanePlan, ...],
        *,
        execution_reservation: ConversationExecutionReservation,
        owner_token: str,
        authority: AuthorityScope,
        request: ConversationRunRequest,
        parent: ConversationCheckpoint | None,
        streaming: bool,
        progress: _DispatchProgress,
        sink: _ProviderStateSinkOwner | None,
        completed_stored: list[_CompletedStoredProviderResponse],
        lane_invocations: (
            Mapping[
                ProviderLaneId,
                AgentConversationLaneInvocation,
            ]
            | None
        ) = None,
    ) -> tuple[
        tuple[ProviderLaneSnapshot, ...],
        tuple[ProviderLaneOutputCandidate, ...],
        tuple[ProviderLaneExecutionAttestation, ...],
        tuple[ProviderExecutionSegment, ...],
    ]:
        snapshots: list[ProviderLaneSnapshot] = []
        outputs: list[ProviderLaneOutputCandidate] = []
        attestations: list[ProviderLaneExecutionAttestation] = []
        segment_context = _SegmentExecutionContext(
            request=request,
            idempotency=execution_reservation.idempotency,
            segments=[],
            visible_transcript=VisibleTranscript(
                entries=(
                    request.visible_delta
                    if parent is None
                    else parent.content.visible_transcript.entries
                    + request.visible_delta
                )
            ),
            lane_snapshots={
                lane.lane_id: lane
                for lane in (
                    parent.content.lanes if parent is not None else ()
                )
                if lane.retention_policy is ChildLaneRetentionPolicy.RETAIN
            },
        )
        for lane_request, runtime, plan in plans:
            invocation = (
                lane_invocations.get(lane_request.lane_id)
                if lane_invocations is not None
                else None
            )
            result = await self._dispatch_complete_lane(
                runtime,
                plan,
                streaming=streaming,
                progress=progress,
                sink=sink,
                completed_stored=completed_stored,
                segment_context=segment_context,
                lane_invocation=invocation,
            )
            if lane_request.mode is ConversationMode.STORED:
                _remember_completed_stored_response(
                    completed_stored,
                    runtime.binding,
                    result,
                )
            scope = ProviderLaneOutputScope.CURRENT_CALL
            execution_receipt = provider_lane_execution_receipt(
                authority=authority,
                identity=request.identity,
                binding=runtime.binding,
                mode=lane_request.mode,
                scope=scope,
                completed_items=result.items,
                reasoning=result.reasoning,
                usage=result.usage,
                upstream_response_id=result.upstream_response_id,
            )
            snapshot = self._lane_snapshot(
                lane_request,
                runtime,
                plan,
                result,
                execution_receipt,
            )
            snapshots.append(snapshot)
            segment_context.lane_snapshots[snapshot.lane_id] = snapshot
            output = ProviderLaneOutputCandidate(
                lane_id=lane_request.lane_id,
                binding=runtime.binding,
                mode=lane_request.mode,
                scope=scope,
                completed_items=result.items,
                reasoning=result.reasoning,
                usage=result.usage,
                execution_receipt=execution_receipt,
                upstream_response_id=result.upstream_response_id,
            )
            outputs.append(output)
            await self._hook.reach(CoordinatorAwaitBoundary.STAGE_EXECUTION)
            attestations.append(
                await self._store.stage_execution(
                    ProviderLaneExecutionStage(
                        idempotency=execution_reservation.idempotency,
                        owner_token=owner_token,
                        identity=request.identity,
                        binding=output.binding,
                        mode=output.mode,
                        scope=output.scope,
                        completed_items=output.completed_items,
                        reasoning=output.reasoning,
                        usage=output.usage,
                        execution_receipt=output.execution_receipt,
                        upstream_response_id=output.upstream_response_id,
                    )
                )
            )
        return (
            tuple(snapshots),
            tuple(outputs),
            tuple(attestations),
            tuple(segment_context.segments),
        )

    async def _dispatch_complete_lane(
        self,
        runtime: LaneRuntime,
        plan: ProviderPlan,
        *,
        streaming: bool,
        progress: _DispatchProgress,
        sink: _ProviderStateSinkOwner | None,
        completed_stored: list[_CompletedStoredProviderResponse] | None = None,
        segment_context: _SegmentExecutionContext | None = None,
        lane_invocation: AgentConversationLaneInvocation | None = None,
    ) -> ProviderResult:
        completed_targets = (
            completed_stored if completed_stored is not None else []
        )
        if lane_invocation is not None:
            if (
                streaming
                or sink is not None
                or type(plan) is not StatelessProviderPlan
                or lane_invocation.binding != runtime.binding
            ):
                raise ConversationCapabilityError()
            await self._hook.reach(CoordinatorAwaitBoundary.PROVIDER_DISPATCH)
            progress.mark_possible_dispatch()
            pending = lane_invocation.dispatch(plan)
            if not isawaitable(pending):
                raise ConversationCapabilityError()
            invocation_result = await pending
            if (
                type(invocation_result)
                is not AgentConversationLaneInvocationResult
            ):
                raise ConversationProviderResponseError()
            result = invocation_result.result
            if segment_context is None:
                raise ConversationCapabilityError()
            for candidate in invocation_result.segments:
                segment = ProviderExecutionSegment(
                    schema_version=1,
                    idempotency_key=segment_context.idempotency.key,
                    request_digest=(
                        segment_context.idempotency.request_digest
                    ),
                    binding=runtime.binding,
                    mode=ConversationMode.STATELESS,
                    segment_index=candidate.segment_index,
                    phase=candidate.phase,
                    items=candidate.items,
                    reasoning=candidate.reasoning,
                    usage=candidate.usage,
                    tools=candidate.tools,
                    recovery_request=conversation_run_request_recovery_payload(
                        segment_context.request
                    ),
                )
                if any(
                    prior.lane_id == segment.lane_id
                    and prior.segment_index == segment.segment_index
                    and prior.phase is segment.phase
                    for prior in segment_context.segments
                ):
                    raise ConversationConflictError()
                segment_context.segments.append(segment)
            staging = _AttemptStaging(
                lane_id=runtime.binding.lane_id,
                items=[],
            )
            try:
                for item in result.items:
                    staging.accept(item)
                staging.finish(result)
            except (AttributeError, ConversationValidationError):
                raise ConversationProviderResponseError() from None
            progress.mark_provider_output()
            return result
        if type(plan) is StandaloneCompactProviderPlan:
            if streaming or type(runtime) is NativeOpenAIStoredLaneRuntime:
                raise ConversationCapabilityError()
            staging = _AttemptStaging(
                lane_id=runtime.binding.lane_id,
                items=[],
            )
            result = await self._dispatch_with_retry(
                runtime,
                plan,
                staging,
                streaming=False,
                progress=progress,
                sink=sink,
            )
            self._validate_standalone_provider_result(plan, result)
            return result
        if type(runtime) is ConversationLaneRuntime:
            staging = _AttemptStaging(
                lane_id=runtime.binding.lane_id,
                items=[],
            )
            return await self._dispatch_with_retry(
                runtime,
                plan,
                staging,
                streaming=streaming,
                progress=progress,
                sink=sink,
            )
        if type(runtime) is NativeOpenAIStoredLaneRuntime:
            stored_native = _validate_stored_native_lane_runtime(runtime)
            if not isinstance(
                plan,
                FirstStoredProviderPlan | StoredProviderPlan,
            ):
                raise ConversationCapabilityError()
            return await self._dispatch_complete_stored_native_lane(
                stored_native,
                plan,
                streaming=streaming,
                progress=progress,
                sink=sink,
                completed_stored=completed_targets,
                segment_context=segment_context,
            )
        native = _validate_native_lane_runtime(runtime)
        if type(plan) is not StatelessProviderPlan:
            raise ConversationCapabilityError()
        original = plan
        current = plan
        completed: list[ProviderItem] = []
        input_tokens = 0
        output_tokens = 0
        reasoning = plan.reasoning
        prior_lane_segments = (
            tuple(
                segment
                for segment in segment_context.segments
                if segment.lane_id == native.binding.lane_id
            )
            if segment_context is not None
            else ()
        )
        recovered_items = _durable_execution_items(prior_lane_segments)
        segments = (
            max(segment.segment_index for segment in prior_lane_segments) + 1
            if prior_lane_segments
            else 0
        )
        output_item_count = len(recovered_items)
        output_byte_count = sum(
            provider_item_byte_count(item) for item in recovered_items
        )
        while True:
            segments += 1
            if segments > native.max_output_segments:
                raise ConversationLimitError()
            staging = _AttemptStaging(
                lane_id=native.binding.lane_id,
                items=[],
            )
            result = await self._dispatch_with_retry(
                native,
                current,
                staging,
                streaming=streaming,
                progress=progress,
                sink=sink,
            )
            next_item_count = output_item_count + len(result.items)
            next_byte_count = output_byte_count + sum(
                provider_item_byte_count(item) for item in result.items
            )
            if (
                next_item_count > native.max_output_items
                or next_byte_count > native.max_output_bytes
            ):
                raise ConversationLimitError()
            self._validate_native_provider_segment(
                original,
                tuple(completed),
                result,
            )
            calls = tuple(
                item
                for item in result.items
                if item.kind is ProviderItemKind.FUNCTION_CALL
            )
            requested_tools = self._requested_tool_executions(
                native,
                calls,
                segment_context,
            )
            recovery_items = original.ledger.items + tuple(completed)
            if not calls:
                recovery_items += result.items
            recovery_snapshot = self._stateless_lane_snapshot(
                binding=native.binding,
                retention_policy=native.retention_policy,
                items=recovery_items,
                reasoning=result.reasoning,
            )
            recovery_checkpoint = await self._persist_native_execution_segment(
                native=native,
                mode=ConversationMode.STATELESS,
                segment_index=segments - 1,
                phase=ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
                items=result.items,
                result=result,
                tools=requested_tools,
                segment_context=segment_context,
                lane_snapshot=recovery_snapshot,
            )
            next_item_count += len(calls)
            if next_item_count > native.max_output_items:
                raise ConversationLimitError()
            (
                tool_items,
                next_byte_count,
                completed_tools,
            ) = await self._execute_native_tools(
                native,
                calls=calls,
                completed=tuple(completed),
                result=result,
                order_base=len(original.ledger.items),
                current_byte_count=next_byte_count,
                progress=progress,
                segment_context=segment_context,
                recovery_checkpoint=recovery_checkpoint,
                requested_tools=requested_tools,
            )
            if calls:
                output_snapshot = self._stateless_lane_snapshot(
                    binding=native.binding,
                    retention_policy=native.retention_policy,
                    items=(
                        original.ledger.items
                        + tuple(completed)
                        + result.items
                        + tool_items
                    ),
                    reasoning=result.reasoning,
                )
                await self._persist_native_execution_segment(
                    native=native,
                    mode=ConversationMode.STATELESS,
                    segment_index=segments - 1,
                    phase=ProviderExecutionSegmentPhase.TOOL_OUTPUT,
                    items=result.items + tool_items,
                    result=result,
                    tools=completed_tools,
                    segment_context=segment_context,
                    lane_snapshot=output_snapshot,
                )
            output_item_count = next_item_count
            output_byte_count = next_byte_count
            segment_items = result.items + tool_items
            completed.extend(segment_items)
            if sink is not None:
                for item in segment_items:
                    await sink.stage(item)
            input_tokens += result.usage.input_tokens
            output_tokens += result.usage.output_tokens
            reasoning = result.reasoning
            if not calls:
                return ProviderResult(
                    items=tuple(completed),
                    reasoning=reasoning,
                    usage=ProviderUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                )
            ledger = ProviderItemLedger(
                lane_id=native.binding.lane_id,
                normalization_version=(
                    native.binding.continuation_codec_version
                ),
                items=original.ledger.items + tuple(completed),
            )
            current = StatelessProviderPlan(
                binding=native.binding,
                ledger=ledger,
                reasoning=original.reasoning,
                compaction=original.compaction,
                new_input=None,
            )

    async def _recover_native_tool_suffix(
        self,
        *,
        request: ConversationRunRequest,
        parent: ConversationCheckpoint | None,
        checkpoint: ConversationCheckpoint,
        segments: tuple[ProviderExecutionSegment, ...],
        action: DurableToolRecoveryAction,
        plans: tuple[LanePlan, ...],
        execution_reservation: ConversationExecutionReservation,
        owner_token: str,
    ) -> AtomicCommitReceipt:
        """Execute an admitted native stateless suffix to outward commit."""
        if (
            len(plans) != 1
            or len(request.lanes) != 1
            or len(checkpoint.content.lanes) != 1
        ):
            raise ConversationCapabilityError()
        lane_request, runtime, _ = plans[0]
        native = _validate_native_lane_runtime(runtime)
        if (
            lane_request.mode is not ConversationMode.STATELESS
            or native.binding != segments[-1].binding
        ):
            raise ConversationCapabilityError()
        snapshot = checkpoint.content.lanes[0]
        if (
            type(snapshot) is not StatelessProviderLaneSnapshot
            or snapshot.binding != native.binding
        ):
            raise ConversationConflictError()
        segment_context = _SegmentExecutionContext(
            request=request,
            idempotency=execution_reservation.idempotency,
            segments=list(segments),
            visible_transcript=checkpoint.content.visible_transcript,
            lane_snapshots={snapshot.lane_id: snapshot},
        )
        progress = _DispatchProgress()
        reconciled_outputs: dict[ProviderCallId, str] = {}
        if action is DurableToolRecoveryAction.REQUIRE_RECONCILIATION:
            latest = segments[-1]
            calls = tuple(
                item
                for item in latest.items
                if item.kind is ProviderItemKind.FUNCTION_CALL
            )
            by_call = {tool.call_id: tool for tool in latest.tools}
            for call in calls:
                call_id = call.call_id
                assert call_id is not None
                tool = by_call.get(call_id)
                if tool is None:
                    raise ConversationConflictError()
                if tool.effect_policy is ToolEffectPolicy.FENCED_UNPROTECTED:
                    reconciliation = await native.provider.reconcile_tool(call)
                    if reconciliation.applied:
                        assert reconciliation.output is not None
                        reconciled_outputs[call_id] = reconciliation.output
            action = DurableToolRecoveryAction.REEXECUTE_PURE
        if action in {
            DurableToolRecoveryAction.REEXECUTE_PURE,
            DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT,
        }:
            latest = segments[-1]
            calls = tuple(
                item
                for item in latest.items
                if item.kind is ProviderItemKind.FUNCTION_CALL
            )
            requested_tools = self._requested_tool_executions(
                native,
                calls,
                segment_context,
            )
            if not calls or requested_tools != latest.tools:
                raise ConversationConflictError()
            prior_items = _durable_execution_items(segments[:-1])
            result = ProviderResult(
                items=latest.items,
                reasoning=latest.reasoning,
                usage=latest.usage,
            )
            order_base = min(
                (
                    int(item.order)
                    for item in _durable_execution_items(segments)
                ),
                default=0,
            )
            tool_items, _, completed_tools = await self._execute_native_tools(
                native,
                calls=calls,
                completed=prior_items,
                result=result,
                order_base=order_base,
                current_byte_count=sum(
                    provider_item_byte_count(item)
                    for item in (*prior_items, *latest.items)
                ),
                progress=progress,
                segment_context=segment_context,
                recovery_checkpoint=checkpoint,
                requested_tools=requested_tools,
                recovered_outputs=reconciled_outputs,
            )
            snapshot = self._stateless_lane_snapshot(
                binding=native.binding,
                retention_policy=native.retention_policy,
                items=snapshot.ledger.items + latest.items + tool_items,
                reasoning=latest.reasoning,
            )
            await self._persist_native_execution_segment(
                native=native,
                mode=ConversationMode.STATELESS,
                segment_index=latest.segment_index,
                phase=ProviderExecutionSegmentPhase.TOOL_OUTPUT,
                items=latest.items + tool_items,
                result=result,
                tools=completed_tools,
                segment_context=segment_context,
                lane_snapshot=snapshot,
            )
            action = DurableToolRecoveryAction.RESUME_PROVIDER
        if action is DurableToolRecoveryAction.RESUME_PROVIDER:
            snapshot = segment_context.lane_snapshots[native.binding.lane_id]
            assert type(snapshot) is StatelessProviderLaneSnapshot
            await self._dispatch_complete_lane(
                native,
                StatelessProviderPlan(
                    binding=native.binding,
                    ledger=snapshot.ledger,
                    reasoning=snapshot.reasoning,
                    compaction=lane_request.compaction,
                ),
                streaming=False,
                progress=progress,
                sink=None,
                segment_context=segment_context,
            )
        durable_suffix = tuple(segment_context.segments)
        if durable_tool_recovery_action(durable_suffix) is not (
            DurableToolRecoveryAction.COMMIT_OUTWARD
        ):
            raise ConversationConflictError()
        final_snapshot = segment_context.lane_snapshots[native.binding.lane_id]
        if type(final_snapshot) is not StatelessProviderLaneSnapshot:
            raise ConversationConflictError()
        completed_items = _durable_execution_items(durable_suffix)
        provider_segments = tuple(
            segment
            for segment in durable_suffix
            if segment.phase is ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
        )
        final_segment = provider_segments[-1]
        usage = ProviderUsage(
            input_tokens=sum(
                segment.usage.input_tokens for segment in provider_segments
            ),
            output_tokens=sum(
                segment.usage.output_tokens for segment in provider_segments
            ),
        )
        execution_receipt = provider_lane_execution_receipt(
            authority=request.semantics.authority,
            identity=request.identity,
            binding=native.binding,
            mode=ConversationMode.STATELESS,
            scope=ProviderLaneOutputScope.CURRENT_CALL,
            completed_items=completed_items,
            reasoning=final_segment.reasoning,
            usage=usage,
            upstream_response_id=None,
        )
        completed_snapshot = replace(
            final_snapshot,
            reasoning=final_segment.reasoning,
            execution_receipt=execution_receipt,
        )
        output = ProviderLaneOutputCandidate(
            lane_id=native.binding.lane_id,
            binding=native.binding,
            mode=ConversationMode.STATELESS,
            scope=ProviderLaneOutputScope.CURRENT_CALL,
            completed_items=completed_items,
            reasoning=final_segment.reasoning,
            usage=usage,
            execution_receipt=execution_receipt,
        )
        await self._hook.reach(CoordinatorAwaitBoundary.STAGE_EXECUTION)
        attestation = await self._store.stage_execution(
            ProviderLaneExecutionStage(
                idempotency=execution_reservation.idempotency,
                owner_token=owner_token,
                identity=request.identity,
                binding=output.binding,
                mode=output.mode,
                scope=output.scope,
                completed_items=output.completed_items,
                reasoning=output.reasoning,
                usage=output.usage,
                execution_receipt=output.execution_receipt,
            )
        )
        now = await self._clock.now()
        candidate = build_checkpoint_candidate(
            request,
            parent=parent,
            completed_lanes=(completed_snapshot,),
            created_at=now,
            execution_segments=durable_suffix,
        )
        commit = AtomicConversationCommit(
            candidate=candidate,
            idempotency=execution_reservation.idempotency,
            owner_token=owner_token,
            output_candidates=(output,),
            committed_at=now,
            result_mode=ConversationMode.STATELESS,
            execution_attestations=(attestation,),
            provisional_response_id=request.provisional_response_id,
            public_response_id=request.public_response_id,
            outbox_intent_id=(
                f"publication-{request.public_response_id}"
                if request.public_response_id is not None
                else None
            ),
            head_id=(
                request.advance.head_id
                if isinstance(request.advance, NamedHeadAdvance)
                else None
            ),
            expected_head_revision=(
                request.advance.expected_revision
                if isinstance(request.advance, NamedHeadAdvance)
                else None
            ),
        )
        await self._hook.reach(CoordinatorAwaitBoundary.COMMIT)
        receipt = await self._store.commit_atomic(commit)
        await self._observe("checkpoint_committed", receipt.checkpoint)
        if receipt.outbox is not None:
            await self._publish_one(
                receipt.checkpoint,
                receipt.outbox.intent.public_response_id,
                receipt.outbox.intent.intent_id,
            )
        return receipt

    @staticmethod
    def _validate_standalone_provider_result(
        plan: StandaloneCompactProviderPlan,
        result: ProviderResult,
    ) -> None:
        """Validate one complete canonical provider-private next context."""
        if (
            type(plan) is not StandaloneCompactProviderPlan
            or type(result) is not ProviderResult
            or result.upstream_response_id is not None
            or result.reasoning != plan.reasoning
            or not result.items
            or result.items[-1].kind is not ProviderItemKind.COMPACTION
            or sum(
                item.kind is ProviderItemKind.COMPACTION
                for item in result.items
            )
            != 1
            or any(
                item.lane_id != plan.binding.lane_id for item in result.items
            )
            or any(
                item.kind is not ProviderItemKind.MESSAGE
                or item.phase is not ProviderItemPhase.INPUT
                or item.caller is not ProviderItemCaller.CALLER
                for item in result.items[:-1]
            )
            or result.items[-1].caller is not ProviderItemCaller.PROVIDER
        ):
            raise ConversationProviderResponseError()
        try:
            ProviderItemLedger(
                lane_id=plan.binding.lane_id,
                normalization_version=(
                    plan.binding.continuation_codec_version
                ),
                items=result.items,
            )
        except ConversationValidationError:
            raise ConversationProviderResponseError() from None

    async def _dispatch_complete_stored_native_lane(
        self,
        native: NativeOpenAIStoredLaneRuntime,
        original: FirstStoredProviderPlan | StoredProviderPlan,
        *,
        streaming: bool,
        progress: _DispatchProgress,
        sink: _ProviderStateSinkOwner | None,
        completed_stored: list[_CompletedStoredProviderResponse] | None = None,
        segment_context: _SegmentExecutionContext | None = None,
    ) -> ProviderResult:
        """Complete one stored lane using immediate response-ID chaining."""
        completed_targets = (
            completed_stored if completed_stored is not None else []
        )
        current: FirstStoredProviderPlan | StoredProviderPlan = original
        completed: list[ProviderItem] = []
        input_tokens = 0
        output_tokens = 0
        reasoning = original.reasoning
        segments = 0
        output_item_count = 0
        output_byte_count = 0
        while True:
            segments += 1
            if segments > native.max_output_segments:
                raise ConversationLimitError()
            staging = _AttemptStaging(
                lane_id=native.binding.lane_id,
                items=[],
            )
            result = await self._dispatch_with_retry(
                native,
                current,
                staging,
                streaming=streaming,
                progress=progress,
                sink=sink,
                completed_stored=completed_targets,
            )
            _remember_completed_stored_response(
                completed_targets,
                native.binding,
                result,
            )
            next_item_count = output_item_count + len(result.items)
            next_byte_count = output_byte_count + sum(
                provider_item_byte_count(item) for item in result.items
            )
            if (
                next_item_count > native.max_output_items
                or next_byte_count > native.max_output_bytes
            ):
                raise ConversationLimitError()
            self._validate_native_stored_provider_segment(
                current,
                tuple(completed),
                result,
            )
            calls = tuple(
                item
                for item in result.items
                if item.kind is ProviderItemKind.FUNCTION_CALL
            )
            requested_tools = self._requested_tool_executions(
                native,
                calls,
                segment_context,
            )
            recovery_snapshot = self._stored_lane_snapshot(
                binding=native.binding,
                retention_policy=native.retention_policy,
                result=result,
            )
            recovery_checkpoint = await self._persist_native_execution_segment(
                native=native,
                mode=ConversationMode.STORED,
                segment_index=segments - 1,
                phase=ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
                items=result.items,
                result=result,
                tools=requested_tools,
                segment_context=segment_context,
                lane_snapshot=recovery_snapshot,
            )
            next_item_count += len(calls)
            if next_item_count > native.max_output_items:
                raise ConversationLimitError()
            (
                tool_items,
                next_byte_count,
                completed_tools,
            ) = await self._execute_native_tools(
                native,
                calls=calls,
                completed=tuple(completed),
                result=result,
                order_base=0,
                current_byte_count=next_byte_count,
                progress=progress,
                segment_context=segment_context,
                recovery_checkpoint=recovery_checkpoint,
                requested_tools=requested_tools,
            )
            if calls:
                await self._persist_native_execution_segment(
                    native=native,
                    mode=ConversationMode.STORED,
                    segment_index=segments - 1,
                    phase=ProviderExecutionSegmentPhase.TOOL_OUTPUT,
                    items=result.items + tool_items,
                    result=result,
                    tools=completed_tools,
                    segment_context=segment_context,
                    lane_snapshot=recovery_snapshot,
                )
            output_item_count = next_item_count
            output_byte_count = next_byte_count
            segment_items = result.items + tool_items
            completed.extend(segment_items)
            if sink is not None:
                for item in segment_items:
                    await sink.stage(item)
            input_tokens += result.usage.input_tokens
            output_tokens += result.usage.output_tokens
            reasoning = result.reasoning
            upstream_response_id = result.upstream_response_id
            assert upstream_response_id is not None
            if not calls:
                return ProviderResult(
                    items=tuple(completed),
                    reasoning=reasoning,
                    usage=ProviderUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                    upstream_response_id=upstream_response_id,
                )
            current = StoredProviderPlan(
                binding=native.binding,
                upstream_response_id=upstream_response_id,
                reasoning=original.reasoning,
                compaction=original.compaction,
                new_input={
                    "items": tuple(item.canonical_input for item in tool_items)
                },
                model_call_index=current.model_call_index + 1,
                item_order_offset=len(completed),
            )

    @staticmethod
    def _requested_tool_executions(
        native: NativeLaneRuntime,
        calls: tuple[ProviderItem, ...],
        segment_context: _SegmentExecutionContext | None,
    ) -> tuple[ProviderToolExecution, ...]:
        """Return canonical pre-effect metadata for one provider segment."""
        if segment_context is None:
            return ()
        return tuple(
            native.provider.tool_execution_metadata(
                call,
                request_idempotency_key=segment_context.idempotency.key,
                phase=ToolExecutionPhase.REQUESTED,
            )
            for call in calls
        )

    @staticmethod
    def _stateless_lane_snapshot(
        *,
        binding: ProviderLaneBinding,
        retention_policy: ChildLaneRetentionPolicy,
        items: tuple[ProviderItem, ...],
        reasoning: EffectiveReasoningMetadata,
        execution_receipt: ProviderLaneExecutionReceipt | None = None,
    ) -> StatelessProviderLaneSnapshot:
        """Return one complete closed stateless recovery ledger."""
        ledger = ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=binding.continuation_codec_version,
            items=items,
        )
        compactions = tuple(
            item
            for item in ledger.items
            if item.kind is ProviderItemKind.COMPACTION
        )
        boundary = None
        if compactions:
            latest = compactions[-1]
            boundary = CompactionBoundary(
                boundary_item_id=latest.item_id,
                boundary_order=latest.order,
                retained_suffix=tuple(
                    item.item_id
                    for item in ledger.items
                    if item.order > latest.order
                ),
            )
        return StatelessProviderLaneSnapshot(
            binding=binding,
            ledger=ledger,
            reasoning=reasoning,
            lifecycle=ProviderLaneLifecycle.COMMITTED,
            retention_policy=retention_policy,
            compaction_boundary=boundary,
            execution_receipt=execution_receipt,
        )

    @staticmethod
    def _stored_lane_snapshot(
        *,
        binding: ProviderLaneBinding,
        retention_policy: ChildLaneRetentionPolicy,
        result: ProviderResult,
        execution_receipt: ProviderLaneExecutionReceipt | None = None,
    ) -> StoredProviderLaneSnapshot:
        """Return one immediate stored response recovery pointer."""
        upstream_response_id = result.upstream_response_id
        if upstream_response_id is None:
            raise ConversationValidationError()
        return StoredProviderLaneSnapshot(
            binding=binding,
            upstream_response_id=upstream_response_id,
            reasoning=result.reasoning,
            lifecycle=ProviderLaneLifecycle.COMMITTED,
            retention_policy=retention_policy,
            execution_receipt=execution_receipt,
        )

    async def _persist_native_execution_segment(
        self,
        *,
        native: NativeLaneRuntime,
        mode: ConversationMode,
        segment_index: int,
        phase: ProviderExecutionSegmentPhase,
        items: tuple[ProviderItem, ...],
        result: ProviderResult,
        tools: tuple[ProviderToolExecution, ...],
        segment_context: _SegmentExecutionContext | None,
        lane_snapshot: ProviderLaneSnapshot,
    ) -> ConversationCheckpoint | None:
        """Commit one private recovery boundary before advancing effects."""
        if segment_context is None:
            return None
        if lane_snapshot.lane_id != native.binding.lane_id:
            raise ConversationValidationError()
        segment = ProviderExecutionSegment(
            schema_version=1,
            idempotency_key=segment_context.idempotency.key,
            request_digest=segment_context.idempotency.request_digest,
            binding=native.binding,
            mode=mode,
            segment_index=segment_index,
            phase=phase,
            items=items,
            reasoning=result.reasoning,
            usage=result.usage,
            tools=tools,
            upstream_response_id=result.upstream_response_id,
            recovery_request=conversation_run_request_recovery_payload(
                segment_context.request
            ),
        )
        key = (segment.lane_id, segment.segment_index, segment.phase)
        if any(
            (prior.lane_id, prior.segment_index, prior.phase) == key
            for prior in segment_context.segments
        ):
            raise ConversationConflictError()
        has_internal_cycle = bool(tools) or any(
            prior.lane_id == segment.lane_id
            for prior in segment_context.segments
        )
        checkpoint = None
        if has_internal_cycle:
            durable_suffix = (*segment_context.segments, segment)
            lane_snapshots = dict(segment_context.lane_snapshots)
            lane_snapshots[lane_snapshot.lane_id] = lane_snapshot
            checkpoint = await self._commit_execution_segment_checkpoint(
                segment_context.request,
                segment,
                durable_suffix,
                tuple(lane_snapshots.values()),
                segment_context.visible_transcript,
            )
            if (
                checkpoint.content.execution_segments != durable_suffix
                or checkpoint.content.lanes != tuple(lane_snapshots.values())
            ):
                raise ConversationConflictError()
            segment_context.lane_snapshots = lane_snapshots
        segment_context.segments.append(segment)
        return checkpoint

    async def _commit_execution_segment_checkpoint(
        self,
        request: ConversationRunRequest,
        segment: ProviderExecutionSegment,
        durable_suffix: tuple[ProviderExecutionSegment, ...],
        lanes: tuple[ProviderLaneSnapshot, ...],
        visible_transcript: VisibleTranscript,
    ) -> ConversationCheckpoint:
        """Create or exactly recover one deterministic private checkpoint."""
        seed = canonical_json_bytes(
            {
                "checkpoint_id": request.identity.checkpoint_id,
                "lane_id": segment.lane_id,
                "phase": segment.phase.value,
                "segment_index": segment.segment_index,
            }
        )
        suffix = sha256(seed).hexdigest()
        checkpoint_id = CheckpointId(f"internal-segment-{suffix}")
        branch_id = ConversationBranchId(f"internal-segment-{suffix}")
        created_at = await self._clock.now()
        ttl = request.retention.effective_ttl_seconds
        candidate = ExecutionSegmentCheckpointCandidate(
            checkpoint=with_checkpoint_integrity(
                ConversationCheckpoint(
                    identity=CheckpointIdentity(
                        conversation_id=request.identity.conversation_id,
                        logical_turn_id=request.identity.logical_turn_id,
                        execution_segment_id=ExecutionSegmentId(
                            f"internal-segment-{suffix}"
                        ),
                        checkpoint_id=checkpoint_id,
                        branch_id=branch_id,
                        sequence=CheckpointSequence(0),
                    ),
                    kind=CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
                    lifecycle=CheckpointLifecycle.STAGED,
                    authority=request.semantics.authority,
                    content=MultiLaneCheckpointContent(
                        visible_transcript=visible_transcript,
                        lanes=lanes,
                        execution_segments=durable_suffix,
                        lane_topology=request.lane_topology,
                    ),
                    timestamps=CheckpointTimestamps(
                        created_at=created_at,
                        expires_at=(
                            created_at + timedelta(seconds=ttl)
                            if ttl is not None
                            else None
                        ),
                    ),
                    retention=request.retention,
                )
            )
        )
        try:
            committed = await self._store.commit(candidate)
        except ConversationConflictError:
            committed = await self._store.load(
                checkpoint_id,
                request.semantics.authority,
            )
        if (
            committed.kind is not CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
            or committed.authority != request.semantics.authority
            or committed.identity.conversation_id
            != request.identity.conversation_id
            or committed.identity.logical_turn_id
            != request.identity.logical_turn_id
            or committed.identity.branch_id != branch_id
            or committed.content.visible_transcript != visible_transcript
            or committed.content.lanes != lanes
            or committed.content.execution_segments != durable_suffix
        ):
            raise ConversationConflictError()
        return committed

    async def _execute_native_tools(
        self,
        native: NativeLaneRuntime,
        *,
        calls: tuple[ProviderItem, ...],
        completed: tuple[ProviderItem, ...],
        result: ProviderResult,
        order_base: int,
        current_byte_count: int,
        progress: _DispatchProgress,
        segment_context: _SegmentExecutionContext | None = None,
        recovery_checkpoint: ConversationCheckpoint | None = None,
        requested_tools: tuple[ProviderToolExecution, ...] = (),
        recovered_outputs: Mapping[ProviderCallId, str] | None = None,
    ) -> tuple[
        tuple[ProviderItem, ...],
        int,
        tuple[ProviderToolExecution, ...],
    ]:
        """Execute validated native calls and return canonical tool items."""
        tool_items: list[ProviderItem] = []
        tool_executions: list[ProviderToolExecution] = []
        byte_count = current_byte_count
        recovered = recovered_outputs or {}
        call_ids = {call.call_id for call in calls}
        if (
            not isinstance(recovered, Mapping)
            or not set(recovered) <= call_ids
            or any(
                type(call_id) is not str or type(output) is not str
                for call_id, output in recovered.items()
            )
        ):
            raise ConversationValidationError()
        for call in calls:
            call_id = call.call_id
            assert call_id is not None
            if call_id in recovered:
                output = recovered[call_id]
            else:
                previous_tool_effect = progress.tool_effect
                progress.mark_tool_effect()
                try:
                    output = await native.provider.execute_tool(call)
                except AgentStructuredInputRequested as request:
                    progress.tool_effect = previous_tool_effect
                    if segment_context is None or recovery_checkpoint is None:
                        raise ConversationCapabilityError() from None
                    tool = next(
                        (
                            candidate
                            for candidate in requested_tools
                            if candidate.call_id == call_id
                        ),
                        None,
                    )
                    if (
                        tool is None
                        or request.arguments != tool.arguments
                        or recovery_checkpoint.content.execution_segments
                        != tuple(segment_context.segments)
                    ):
                        raise ConversationValidationError() from None
                    checkpoint = self._structured_input_checkpoint(
                        segment_context,
                        recovery_checkpoint,
                        call,
                    )
                    raise AgentConversationSuspensionBoundary(
                        request=request,
                        call=call,
                        tool=tool,
                        checkpoint=checkpoint,
                    ) from None
            output_seed = canonical_json_bytes(
                {
                    "call_id": call_id,
                    "checkpoint_id": (
                        segment_context.request.identity.checkpoint_id
                        if segment_context is not None
                        else "uncoordinated"
                    ),
                    "lane_id": native.binding.lane_id,
                }
            )
            tool_item = ProviderItem(
                item_id=ProviderItemId(
                    "tool-output-" + sha256(output_seed).hexdigest()[:24]
                ),
                lane_id=native.binding.lane_id,
                model_call_id=call.model_call_id,
                kind=ProviderItemKind.FUNCTION_CALL_OUTPUT,
                order=ProviderItemOrder(
                    order_base
                    + len(completed)
                    + len(result.items)
                    + len(tool_items)
                ),
                provider_index=ProviderItemIndex(
                    sum(
                        item.model_call_id == call.model_call_id
                        for item in (
                            *completed,
                            *result.items,
                            *tool_items,
                        )
                    )
                ),
                phase=ProviderItemPhase.TOOL,
                caller=ProviderItemCaller.TOOL,
                canonical_input={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
                normalization_version=(
                    native.binding.continuation_codec_version
                ),
                call_id=call_id,
            )
            byte_count += provider_item_byte_count(tool_item)
            if byte_count > native.max_output_bytes:
                raise ConversationLimitError()
            tool_items.append(tool_item)
            if segment_context is not None:
                tool_executions.append(
                    native.provider.tool_execution_metadata(
                        call,
                        request_idempotency_key=(
                            segment_context.idempotency.key
                        ),
                        phase=ToolExecutionPhase.OUTPUT_PERSISTED,
                        output_id=tool_item.item_id,
                    )
                )
        return tuple(tool_items), byte_count, tuple(tool_executions)

    @staticmethod
    def _structured_input_checkpoint(
        segment_context: _SegmentExecutionContext,
        recovery_checkpoint: ConversationCheckpoint,
        call: ProviderItem,
    ) -> ConversationCheckpoint:
        """Return one deterministic staged suspension after a durable fence."""
        if (
            recovery_checkpoint.kind
            is not CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
            or recovery_checkpoint.lifecycle
            is not CheckpointLifecycle.COMMITTED
            or recovery_checkpoint.authority
            != segment_context.request.semantics.authority
            or call.call_id is None
        ):
            raise ConversationValidationError()
        seed = canonical_json_bytes(
            {
                "call_id": call.call_id,
                "checkpoint_id": recovery_checkpoint.identity.checkpoint_id,
                "execution_segment_id": (
                    recovery_checkpoint.identity.execution_segment_id
                ),
            }
        )
        suffix = sha256(seed).hexdigest()
        identity = recovery_checkpoint.identity
        lanes = tuple(
            replace(
                lane,
                lifecycle=(
                    ProviderLaneLifecycle.SUSPENDED
                    if lane.lane_id == call.lane_id
                    else lane.lifecycle
                ),
            )
            for lane in recovery_checkpoint.content.lanes
        )
        created_at = recovery_checkpoint.timestamps.created_at
        return with_checkpoint_integrity(
            ConversationCheckpoint(
                identity=CheckpointIdentity(
                    conversation_id=identity.conversation_id,
                    logical_turn_id=identity.logical_turn_id,
                    execution_segment_id=ExecutionSegmentId(
                        f"structured-input-segment-{suffix}"
                    ),
                    checkpoint_id=CheckpointId(
                        f"structured-input-checkpoint-{suffix}"
                    ),
                    branch_id=identity.branch_id,
                    sequence=CheckpointSequence(identity.sequence + 1),
                    parent_checkpoint_id=identity.checkpoint_id,
                    parent_sequence=identity.sequence,
                ),
                kind=CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
                lifecycle=CheckpointLifecycle.STAGED,
                authority=recovery_checkpoint.authority,
                content=MultiLaneCheckpointContent(
                    visible_transcript=(
                        recovery_checkpoint.content.visible_transcript
                    ),
                    lanes=lanes,
                    execution_segments=tuple(segment_context.segments),
                    lane_topology=(recovery_checkpoint.content.lane_topology),
                ),
                timestamps=CheckpointTimestamps(
                    created_at=created_at,
                    expires_at=recovery_checkpoint.timestamps.expires_at,
                ),
                retention=recovery_checkpoint.retention,
            )
        )

    @staticmethod
    def _validate_native_provider_segment(
        original: StatelessProviderPlan,
        completed: tuple[ProviderItem, ...],
        result: ProviderResult,
    ) -> None:
        """Validate a complete provider segment before any tool effect."""
        if (
            type(original) is not StatelessProviderPlan
            or type(completed) is not tuple
            or type(result) is not ProviderResult
            or result.upstream_response_id is not None
            or result.reasoning.requested is not original.reasoning.requested
            or any(
                item.caller is not ProviderItemCaller.PROVIDER
                for item in result.items
            )
        ):
            raise ConversationProviderResponseError()
        prior = original.ledger.items + completed
        prior_model_calls = {item.model_call_id for item in prior}
        segment_model_calls = {item.model_call_id for item in result.items}
        if result.items and (
            len(segment_model_calls) != 1
            or not segment_model_calls.isdisjoint(prior_model_calls)
            or tuple(item.provider_index for item in result.items)
            != tuple(
                ProviderItemIndex(index) for index in range(len(result.items))
            )
        ):
            raise ConversationProviderResponseError()
        permitted_open_calls = frozenset(
            item.call_id
            for item in result.items
            if item.kind is ProviderItemKind.FUNCTION_CALL
            and item.call_id is not None
        )
        try:
            validate_provider_item_sequence(
                lane_id=original.binding.lane_id,
                normalization_version=(
                    original.binding.continuation_codec_version
                ),
                items=prior + result.items,
                permitted_open_call_ids=permitted_open_calls,
            )
        except ConversationValidationError:
            raise ConversationProviderResponseError() from None

    @staticmethod
    def _validate_native_stored_provider_segment(
        plan: FirstStoredProviderPlan | StoredProviderPlan,
        completed: tuple[ProviderItem, ...],
        result: ProviderResult,
    ) -> None:
        """Validate one stored segment and its immediate response ID."""
        if (
            not isinstance(plan, FirstStoredProviderPlan | StoredProviderPlan)
            or type(completed) is not tuple
            or type(result) is not ProviderResult
            or result.upstream_response_id is None
            or result.reasoning.requested is not plan.reasoning.requested
            or any(
                item.caller is not ProviderItemCaller.PROVIDER
                for item in result.items
            )
        ):
            raise ConversationProviderResponseError()
        expected_call_id = f"native-model-call-{plan.model_call_index}"
        if result.items and (
            {str(item.model_call_id) for item in result.items}
            != {expected_call_id}
            or tuple(item.provider_index for item in result.items)
            != tuple(
                ProviderItemIndex(index) for index in range(len(result.items))
            )
            or tuple(item.order for item in result.items)
            != tuple(
                ProviderItemOrder(plan.item_order_offset + index)
                for index in range(len(result.items))
            )
        ):
            raise ConversationProviderResponseError()
        permitted_open_calls = frozenset(
            item.call_id
            for item in result.items
            if item.kind is ProviderItemKind.FUNCTION_CALL
            and item.call_id is not None
        )
        try:
            validate_provider_item_sequence(
                lane_id=plan.binding.lane_id,
                normalization_version=plan.binding.continuation_codec_version,
                items=completed + result.items,
                permitted_open_call_ids=permitted_open_calls,
            )
        except ConversationValidationError:
            raise ConversationProviderResponseError() from None

    async def _dispatch_with_retry(
        self,
        runtime: LaneRuntime,
        plan: ProviderPlan,
        staging: _AttemptStaging,
        *,
        streaming: bool,
        progress: _DispatchProgress,
        sink: _ProviderStateSinkOwner | None,
        completed_stored: list[_CompletedStoredProviderResponse] | None = None,
        _dispatch_provider: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
                ProviderPlan,
            ],
            Awaitable[ProviderResult],
        ] = _dispatch_deterministic_fake_provider,
        _validate_runtime: Callable[
            [object], ConversationLaneRuntime
        ] = _validate_lane_runtime,
    ) -> ProviderResult:
        attempt = 1
        while True:
            try:
                if type(runtime) is ConversationLaneRuntime:
                    _validate_runtime(runtime)
                    provider_runtime = self._fake_runtimes.get(
                        runtime.binding.lane_id
                    )
                    if (
                        type(provider_runtime)
                        is not _DETERMINISTIC_FAKE_RUNTIME_TYPE
                    ):
                        raise ConversationValidationError()
                    # The lane contributes only inert script data. Provider
                    # effects always cross the closed repository executor;
                    # no awaitable is resolved from caller-owned data.
                    if streaming:
                        result = await self._stream_once(
                            runtime,
                            provider_runtime,
                            plan,
                            staging,
                            progress,
                            sink,
                        )
                    else:
                        await self._hook.reach(
                            CoordinatorAwaitBoundary.PROVIDER_DISPATCH
                        )
                        progress.mark_possible_dispatch()
                        result = await _dispatch_provider(
                            provider_runtime,
                            runtime.provider_script,
                            plan,
                        )
                        for item in result.items:
                            staging.accept(item)
                        staging.finish(result)
                else:
                    native = _validate_any_native_lane_runtime(runtime)
                    if streaming:
                        result = await self._stream_native_once(
                            native,
                            plan,
                            staging,
                            progress,
                            sink,
                            completed_stored,
                        )
                    else:
                        await self._hook.reach(
                            CoordinatorAwaitBoundary.PROVIDER_DISPATCH
                        )
                        progress.mark_possible_dispatch()
                        if type(plan) is StandaloneCompactProviderPlan:
                            if (
                                type(native)
                                is not NativeOpenAIConversationLaneRuntime
                            ):
                                raise ConversationCapabilityError()
                            result = await native.provider.compact(plan)
                        else:
                            result = await native.provider.dispatch(plan)
                        try:
                            for item in result.items:
                                staging.accept(item)
                            staging.finish(result)
                        except ConversationValidationError:
                            raise ConversationProviderResponseError() from None
                progress.mark_provider_output()
                return result
            except ConversationError as exc:
                disposition = reduce_failure(
                    exc.boundary,
                    visible_output=staging.visible_output,
                    tool_effect=(
                        staging.tool_effect
                        or staging.provider_output
                        or progress.tool_effect
                        or progress.provider_output
                    ),
                    committed=False,
                    ambiguous=isinstance(
                        exc, ConversationAmbiguousDispatchError
                    )
                    or (
                        progress.may_have_dispatched
                        and not isinstance(
                            exc,
                            ConversationProviderResponseError,
                        )
                    ),
                )
                staging.rollback()
                if (
                    disposition.retry_rule is not RetryRule.BOUNDED_EFFECT_FREE
                    or type(runtime) is NativeOpenAIStoredLaneRuntime
                    or attempt == self._max_attempts
                ):
                    raise
                await self._hook.reach(CoordinatorAwaitBoundary.RETRY_WAIT)
                await self._retry_waiter.wait(attempt)
                attempt += 1

    async def _stream_native_once(
        self,
        runtime: NativeLaneRuntime,
        plan: ProviderPlan,
        staging: _AttemptStaging,
        progress: _DispatchProgress,
        _sink: _ProviderStateSinkOwner | None,
        completed_stored: list[_CompletedStoredProviderResponse] | None,
    ) -> ProviderResult:
        await self._hook.reach(CoordinatorAwaitBoundary.PROVIDER_STREAM_OPEN)
        progress.mark_possible_dispatch()
        stream = await runtime.provider.stream(plan)
        try:
            async for item in stream:
                await self._hook.reach(
                    CoordinatorAwaitBoundary.PROVIDER_STREAM_ITEM
                )
                try:
                    staging.accept(item)
                except ConversationValidationError:
                    raise ConversationProviderResponseError() from None
            await self._hook.reach(
                CoordinatorAwaitBoundary.PROVIDER_STREAM_TERMINAL
            )
            result = await stream.terminal()
            try:
                staging.finish(result)
            except ConversationValidationError:
                raise ConversationProviderResponseError() from None
            if type(runtime) is NativeOpenAIStoredLaneRuntime:
                _remember_completed_stored_response(
                    completed_stored,
                    runtime.binding,
                    result,
                )
            return result
        finally:
            await self._close_native_stream(stream)

    async def _close_native_stream(
        self,
        stream: ConversationProviderStream,
    ) -> None:
        """Settle one native stream close while preserving cancellation."""
        primary_error: BaseException | None = None
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(
                CoordinatorAwaitBoundary.PROVIDER_STREAM_CLOSE
            )
        except CancelledError as exc:
            cancellation = exc
        except BaseException as exc:
            primary_error = exc
        cleanup_error: BaseException | None = None
        try:
            await stream.aclose()
        except CancelledError as exc:
            cancellation = cancellation or exc
            try:
                await stream.aclose()
            except BaseException as retry_exc:
                cleanup_error = retry_exc
        except BaseException as exc:
            cleanup_error = exc
        if cancellation is not None:
            cause = primary_error or cleanup_error
            if primary_error is not None and cleanup_error is not None:
                primary_error.__cause__ = cleanup_error
            if cause is not None:
                raise cancellation from cause
            raise cancellation
        if primary_error is not None:
            if cleanup_error is not None:
                raise primary_error from cleanup_error
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error

    async def _stream_once(
        self,
        runtime: ConversationLaneRuntime,
        provider_runtime: _DeterministicFakeProviderRuntime,
        plan: ProviderPlan,
        staging: _AttemptStaging,
        progress: _DispatchProgress,
        sink: _ProviderStateSinkOwner | None,
        _next_provider_item: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
                _DeterministicFakeProviderStreamState,
            ],
            Awaitable[ProviderItem],
        ] = _next_deterministic_fake_provider_item,
        _open_provider_stream: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
                ProviderPlan,
            ],
            Awaitable[_DeterministicFakeProviderStreamState],
        ] = _open_deterministic_fake_provider_stream,
        _terminal_provider_stream: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
                _DeterministicFakeProviderStreamState,
            ],
            Awaitable[ProviderResult],
        ] = _terminal_deterministic_fake_provider_stream,
    ) -> ProviderResult:
        await self._hook.reach(CoordinatorAwaitBoundary.PROVIDER_STREAM_OPEN)
        progress.mark_possible_dispatch()
        stream = await _open_provider_stream(
            provider_runtime,
            runtime.provider_script,
            plan,
        )
        try:
            while True:
                try:
                    item = await _next_provider_item(
                        provider_runtime,
                        runtime.provider_script,
                        stream,
                    )
                except StopAsyncIteration:
                    break
                await self._hook.reach(
                    CoordinatorAwaitBoundary.PROVIDER_STREAM_ITEM
                )
                staging.accept(item)
                if sink is not None:
                    await sink.stage(item)
            await self._hook.reach(
                CoordinatorAwaitBoundary.PROVIDER_STREAM_TERMINAL
            )
            result = await _terminal_provider_stream(
                provider_runtime,
                runtime.provider_script,
                stream,
            )
            staging.finish(result)
            return result
        finally:
            await self._close_stream(runtime, provider_runtime, stream)

    async def _close_stream(
        self,
        runtime: ConversationLaneRuntime,
        provider_runtime: _DeterministicFakeProviderRuntime,
        stream: _DeterministicFakeProviderStreamState,
        _close_provider_stream: Callable[
            [
                _DeterministicFakeProviderRuntime,
                DeterministicFakeProviderScript,
                _DeterministicFakeProviderStreamState,
            ],
            Awaitable[None],
        ] = _close_deterministic_fake_provider_stream,
    ) -> None:
        primary_error: BaseException | None = None
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(
                CoordinatorAwaitBoundary.PROVIDER_STREAM_CLOSE
            )
        except CancelledError as exc:
            cancellation = exc
        except BaseException as exc:
            primary_error = exc
        cleanup_error: BaseException | None = None
        try:
            await _close_provider_stream(
                provider_runtime,
                runtime.provider_script,
                stream,
            )
        except CancelledError as exc:
            cancellation = cancellation or exc
            try:
                await _close_provider_stream(
                    provider_runtime,
                    runtime.provider_script,
                    stream,
                )
            except BaseException as retry_exc:
                cleanup_error = retry_exc
        except BaseException as exc:
            cleanup_error = exc
        if cancellation is not None:
            cause = primary_error or cleanup_error
            if primary_error is not None and cleanup_error is not None:
                primary_error.__cause__ = cleanup_error
            if cause is not None:
                raise cancellation from cause
            raise cancellation
        if primary_error is not None:
            if cleanup_error is not None:
                raise primary_error from cleanup_error
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error

    @staticmethod
    def _lane_snapshot(
        lane_request: ConversationLaneRequest,
        runtime: LaneRuntime,
        plan: ProviderPlan,
        result: ProviderResult,
        execution_receipt: ProviderLaneExecutionReceipt,
    ) -> ProviderLaneSnapshot:
        for item in result.items:
            if item.lane_id != lane_request.lane_id:
                raise ConversationValidationError()
        if lane_request.mode is ConversationMode.STATELESS:
            if not isinstance(
                plan,
                StatelessProviderPlan | StandaloneCompactProviderPlan,
            ):
                raise ConversationValidationError()
            prior_items = (
                ()
                if type(plan) is StandaloneCompactProviderPlan
                else plan.ledger.items
            )
            return RunScopedConversationCoordinator._stateless_lane_snapshot(
                binding=runtime.binding,
                retention_policy=runtime.retention_policy,
                items=prior_items + result.items,
                reasoning=result.reasoning,
                execution_receipt=execution_receipt,
            )
        ProviderItemLedger(
            lane_id=lane_request.lane_id,
            normalization_version=runtime.binding.continuation_codec_version,
            items=result.items,
        )
        return RunScopedConversationCoordinator._stored_lane_snapshot(
            binding=runtime.binding,
            retention_policy=runtime.retention_policy,
            result=result,
            execution_receipt=execution_receipt,
        )

    async def _observe(
        self,
        event: str,
        checkpoint: ConversationCheckpoint,
    ) -> None:
        await self._hook.reach(CoordinatorAwaitBoundary.OBSERVE)
        try:
            await self._observer.publish(
                checkpoint_observation(event, checkpoint)
            )
        except Exception:
            return

    async def _publish_one(
        self,
        checkpoint: ConversationCheckpoint,
        public_response_id: PublicResponseId,
        intent_id: str,
    ) -> None:
        target = OutboxClaimTarget(
            authority=checkpoint.authority,
            checkpoint_id=checkpoint.identity.checkpoint_id,
            public_response_id=public_response_id,
            intent_id=intent_id,
        )
        attempt = 1
        while True:
            resolution = await self._store.claim_outbox(target)
            if (
                resolution.disposition
                is OutboxClaimDisposition.ALREADY_PUBLISHED
            ):
                return
            if resolution.disposition is OutboxClaimDisposition.CLAIMED:
                assert resolution.record is not None
                record = resolution.record
                break
            if (
                resolution.disposition
                is OutboxClaimDisposition.ACTIVELY_LEASED
                and attempt < self._max_attempts
            ):
                await self._hook.reach(CoordinatorAwaitBoundary.RETRY_WAIT)
                await self._retry_waiter.wait(attempt)
                attempt += 1
                continue
            raise ConversationPublicationError()
        assert record.lease_owner is not None
        owner_token = record.lease_owner
        try:
            await self._hook.reach(CoordinatorAwaitBoundary.PUBLISH)
            await self._publisher.publish(record.intent)
        except BaseException as exc:
            release_error = await self._release_claim(target, owner_token)
            if isinstance(exc, CancelledError):
                if release_error is not None:
                    raise exc from release_error
                raise
            if release_error is not None:
                raise ConversationPublicationError() from release_error
            raise ConversationPublicationError() from exc
        try:
            await self._store.acknowledge_outbox(
                target,
                owner_token,
            )
        except BaseException as exc:
            release_error = await self._release_claim(target, owner_token)
            if isinstance(exc, CancelledError):
                if release_error is not None:
                    raise exc from release_error
                raise
            if release_error is not None:
                raise ConversationPublicationError() from release_error
            raise ConversationPublicationError() from exc
        await self._observe("outbox_published", checkpoint)

    async def _release_claim(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> BaseException | None:
        try:
            await self._store.release_outbox(target, owner_token)
        except BaseException as exc:
            return exc
        return None

    async def _rollback(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> None:
        cancellation: CancelledError | None = None
        cleanup_failure: BaseException | None = None
        try:
            await self._hook.reach(CoordinatorAwaitBoundary.ROLLBACK)
        except CancelledError as exc:
            cancellation = exc
        except BaseException as exc:
            cleanup_failure = exc
        (
            resolution,
            corroborated,
            attempt_cancellation,
            attempt_failure,
        ) = await self._settlement_attempt(
            identity,
            owner_token,
            ambiguous=ambiguous,
            reconcile=False,
        )
        cancellation = cancellation or attempt_cancellation
        cleanup_failure = attempt_failure or cleanup_failure
        if not corroborated or (
            resolution is not None
            and resolution.disposition
            is IdempotencySettlementDisposition.LEASED
        ):
            for _attempt in range(self._max_attempts):
                (
                    candidate,
                    candidate_corroborated,
                    attempt_cancellation,
                    attempt_failure,
                ) = await self._settlement_attempt(
                    identity,
                    owner_token,
                    ambiguous=ambiguous,
                    reconcile=True,
                )
                cancellation = cancellation or attempt_cancellation
                cleanup_failure = attempt_failure or cleanup_failure
                if not candidate_corroborated:
                    resolution = None
                    corroborated = False
                    continue
                resolution = candidate
                corroborated = True
                if (
                    resolution is not None
                    and resolution.disposition
                    is not IdempotencySettlementDisposition.LEASED
                ):
                    break
        safe = (
            corroborated
            and resolution is not None
            and resolution.disposition
            in {
                IdempotencySettlementDisposition.SETTLED,
                IdempotencySettlementDisposition.LEASED,
            }
        )
        if not safe:
            if (
                corroborated
                and resolution is not None
                and resolution.disposition
                is IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
            ):
                cleanup_failure = ConversationConflictError()
            cleanup_failure = cleanup_failure or ConversationConflictError()
            if cancellation is not None:
                raise cancellation from cleanup_failure
            raise cleanup_failure
        if cancellation is not None:
            if (
                resolution is not None
                and resolution.disposition
                is IdempotencySettlementDisposition.LEASED
                and cleanup_failure is not None
                and cleanup_failure is not cancellation
            ):
                raise cancellation from cleanup_failure
            raise cancellation
        if (
            resolution is not None
            and resolution.disposition
            is IdempotencySettlementDisposition.LEASED
            and cleanup_failure is not None
        ):
            raise cleanup_failure

    async def _settlement_attempt(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
        reconcile: bool,
    ) -> tuple[
        IdempotencySettlementResolution | None,
        bool,
        CancelledError | None,
        BaseException | None,
    ]:
        action: IdempotencySettlementResolution | None = None
        action_error: BaseException | None = None
        cancellation: CancelledError | None = None
        try:
            if reconcile:
                action = await self._store.reconcile_idempotency(
                    identity,
                    owner_token,
                    ambiguous=ambiguous,
                )
            else:
                action = await self._store.abandon_idempotency(
                    identity,
                    owner_token,
                    ambiguous=ambiguous,
                )
        except CancelledError as exc:
            cancellation = exc
            action_error = exc
        except BaseException as exc:
            action_error = exc
        probe: IdempotencySettlementResolution | None = None
        probe_error: BaseException | None = None
        try:
            probe = await self._store.inspect_idempotency_settlement(
                identity,
                owner_token,
            )
        except CancelledError as exc:
            cancellation = cancellation or exc
            probe_error = exc
        except BaseException as exc:
            probe_error = exc
        if probe_error is not None:
            return None, False, cancellation, probe_error
        probe_failure = self._settlement_validation_error(
            probe,
            owner_token,
        )
        if probe_failure is not None:
            return None, False, cancellation, probe_failure
        assert probe is not None
        if action_error is not None:
            return probe, True, cancellation, action_error
        action_failure = self._settlement_validation_error(
            action,
            owner_token,
        )
        if action_failure is not None:
            return probe, False, cancellation, action_failure
        if action != probe:
            return probe, False, cancellation, ConversationConflictError()
        return probe, True, cancellation, None

    @staticmethod
    def _settlement_validation_error(
        resolution: IdempotencySettlementResolution | None,
        owner_token: str,
    ) -> BaseException | None:
        if type(resolution) is not IdempotencySettlementResolution:
            return ConversationValidationError()
        if resolution.disposition is IdempotencySettlementDisposition.LEASED:
            expires_at = resolution.lease_expires_at
            if (
                resolution.lease_owner_token != owner_token
                or expires_at is None
                or expires_at.utcoffset() is None
                or expires_at.year >= datetime.max.year
            ):
                return ConversationConflictError()
        return None

    @staticmethod
    def _idempotency(
        request: ConversationRunRequest,
    ) -> RequestIdempotencyIdentity:
        base = canonical_request_digest(request.semantics)
        lane_payload = tuple(
            {
                "lane_id": lane.lane_id,
                "mode": lane.mode.value,
                "reasoning_context": lane.reasoning_context.value,
                "compaction": lane.compaction.operation.value,
                "compact_threshold": getattr(
                    lane.compaction,
                    "compact_threshold",
                    None,
                ),
            }
            for lane in request.lanes
        )
        advance = request.advance
        advance_payload = {
            "kind": type(advance).__name__,
            "branch_id": (
                advance.branch_id
                if isinstance(advance, ExplicitBranchAdvance)
                else None
            ),
            "head_id": (
                advance.head_id
                if isinstance(advance, NamedHeadAdvance)
                else None
            ),
            "head_revision": (
                advance.expected_revision
                if isinstance(advance, NamedHeadAdvance)
                else None
            ),
        }
        payload = freeze_json_value(
            {
                "base": base,
                "lanes": lane_payload,
                "advance": advance_payload,
                "boundary": request.boundary.value,
                "lane_topology": (
                    None
                    if request.lane_topology is None
                    else tuple(
                        {
                            "agent_id": entry.agent_id,
                            "binding_digest": entry.binding_digest,
                            "lane_id": entry.lane_id,
                            "model_slot": entry.model_slot,
                            "owner_kind": entry.owner_kind.value,
                            "parent_lane_id": entry.parent_lane_id,
                            "retention_policy": entry.retention_policy.value,
                            "topology_path": entry.topology_path,
                        }
                        for entry in request.lane_topology.entries
                    )
                ),
            }
        )
        digest = sha256(canonical_json_bytes(payload)).hexdigest()
        return RequestIdempotencyIdentity(
            authority=request.semantics.authority,
            operation=request_operation(request),
            key=request.idempotency_key,
            request_digest=type(base)(digest),
        )

    def _activate_execution(self) -> str:
        if len(self._active_attempts) >= self._max_active_executions:
            raise ConversationConflictError()
        self._execution_sequence += 1
        token = f"coordinator-execution-{self._execution_sequence}"
        validate_identifier(token, "execution_token")
        self._active_attempts.add(token)
        return token


def _durable_execution_items(
    segments: tuple[ProviderExecutionSegment, ...],
) -> tuple[ProviderItem, ...]:
    """Return one non-duplicated canonical item sequence from a suffix."""
    items: list[ProviderItem] = []
    for segment in segments:
        if segment.phase is ProviderExecutionSegmentPhase.PROVIDER_RESPONSE:
            items.extend(segment.items)
        else:
            items.extend(
                item
                for item in segment.items
                if item.caller is ProviderItemCaller.TOOL
            )
    return tuple(items)


def build_checkpoint_candidate(
    request: ConversationRunRequest,
    *,
    parent: ConversationCheckpoint | None,
    completed_lanes: tuple[ProviderLaneSnapshot, ...],
    created_at: datetime,
    execution_segments: tuple[ProviderExecutionSegment, ...] = (),
    additional_visible_delta: tuple[VisibleTranscriptEntry, ...] = (),
) -> CheckpointCandidate:
    """Build one immutable staged candidate at a validated boundary."""
    if (
        type(request) is not ConversationRunRequest
        or not isinstance(created_at, datetime)
        or created_at.utcoffset() is None
        or type(completed_lanes) is not tuple
        or not completed_lanes
        or type(execution_segments) is not tuple
        or type(additional_visible_delta) is not tuple
        or any(
            type(segment) is not ProviderExecutionSegment
            for segment in execution_segments
        )
        or any(
            type(entry) is not VisibleTranscriptEntry
            for entry in additional_visible_delta
        )
    ):
        raise ConversationValidationError()
    selected = {lane.lane_id for lane in completed_lanes}
    retained: tuple[ProviderLaneSnapshot, ...] = ()
    transcript_entries = request.visible_delta + additional_visible_delta
    if parent is not None:
        transcript_entries = (
            parent.content.visible_transcript.entries
            + request.visible_delta
            + additional_visible_delta
        )
        retained = tuple(
            lane
            for lane in parent.content.lanes
            if lane.lane_id not in selected
            and lane.retention_policy is ChildLaneRetentionPolicy.RETAIN
        )
    committed_lanes = tuple(
        lane
        for lane in completed_lanes
        if request.boundary is not ConversationCommitBoundary.OUTWARD_TURN
        or lane.retention_policy is ChildLaneRetentionPolicy.RETAIN
    )
    lanes = committed_lanes + retained
    ttl = request.retention.effective_ttl_seconds
    head = None
    if isinstance(request.advance, NamedHeadAdvance):
        head = NamedHeadMetadata(
            head_id=request.advance.head_id,
            revision=NamedHeadRevision(request.advance.expected_revision + 1),
        )
    kind = (
        CheckpointKind.STANDALONE_COMPACT_RESULT
        if request.semantics.operation is ConversationOperation.COMPACT
        else (
            CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
            if request.boundary is ConversationCommitBoundary.INTERNAL_SEGMENT
            else CheckpointKind.COMPLETED_OUTWARD_TURN
        )
    )
    checkpoint = with_checkpoint_integrity(
        ConversationCheckpoint(
            identity=request.identity,
            kind=kind,
            lifecycle=CheckpointLifecycle.STAGED,
            authority=request.semantics.authority,
            content=MultiLaneCheckpointContent(
                visible_transcript=VisibleTranscript(
                    entries=transcript_entries
                ),
                lanes=lanes,
                execution_segments=execution_segments,
                lane_topology=request.lane_topology,
            ),
            timestamps=CheckpointTimestamps(
                created_at=created_at,
                expires_at=(
                    created_at + timedelta(seconds=ttl)
                    if ttl is not None
                    else None
                ),
            ),
            retention=request.retention,
            head=head,
        )
    )
    if request.semantics.operation is ConversationOperation.COMPACT:
        assert checkpoint.identity.parent_checkpoint_id is not None
        return StandaloneCompactCheckpointCandidate(
            checkpoint=checkpoint,
            handle=StandaloneCompactHandle(
                conversation_id=checkpoint.identity.conversation_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                branch_id=checkpoint.identity.branch_id,
                parent_checkpoint_id=checkpoint.identity.parent_checkpoint_id,
                head_id=(
                    request.advance.head_id
                    if isinstance(request.advance, NamedHeadAdvance)
                    else None
                ),
                expected_head_revision=(
                    request.advance.expected_revision
                    if isinstance(request.advance, NamedHeadAdvance)
                    else None
                ),
            ),
        )
    if request.boundary is ConversationCommitBoundary.INTERNAL_SEGMENT:
        return ExecutionSegmentCheckpointCandidate(checkpoint=checkpoint)
    assert request.public_response_id is not None
    return OutwardTurnCheckpointCandidate(
        checkpoint=checkpoint,
        public_response_id=request.public_response_id,
    )


def reduce_failure(
    boundary: FailureBoundary,
    *,
    visible_output: bool,
    tool_effect: bool,
    committed: bool,
    ambiguous: bool,
) -> FailureDisposition:
    """Reduce exact failure facts without admitting unsafe retries."""
    if not isinstance(boundary, FailureBoundary):
        raise ConversationValidationError()
    for value in (visible_output, tool_effect, committed, ambiguous):
        if type(value) is not bool:
            raise ConversationValidationError()
    fence = FAILURE_FENCES[boundary]
    retry_rule = fence.retry_rule
    must_fence = fence.fence_duplicate_dispatch
    reconciliation = fence.reconciliation_required
    if visible_output or tool_effect or committed:
        retry_rule = RetryRule.NEVER
        must_fence = True
        reconciliation = True
    if ambiguous:
        retry_rule = RetryRule.FENCED_RECONCILIATION
        must_fence = True
        reconciliation = True
    return FailureDisposition(
        boundary=boundary,
        retry_rule=retry_rule,
        fence_dispatch=must_fence,
        preserve_parent=fence.preserve_parent,
        reconciliation_required=reconciliation,
    )
