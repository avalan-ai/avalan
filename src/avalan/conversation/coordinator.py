"""Coordinate deterministic fake-lane conversation execution."""

from .binding import (
    ConversationCapability,
    ConversationCapabilityProfile,
    ProviderFamily,
    ProviderLaneBinding,
)
from .codec import with_checkpoint_integrity
from .contract import (
    FAILURE_FENCES,
    AuthorityScope,
    CheckpointIdentity,
    CheckpointKind,
    ChildLaneRetentionPolicy,
    FailureBoundary,
    IdempotencyDisposition,
    NamedHeadRevision,
    ProviderLaneId,
    PublicResponseId,
    RequestIdempotencyIdentity,
    RetryRule,
)
from .errors import (
    ConversationAmbiguousDispatchError,
    ConversationAuthorizationError,
    ConversationCapabilityError,
    ConversationCommitError,
    ConversationConflictError,
    ConversationError,
    ConversationPublicationError,
    ConversationValidationError,
)
from .execution import (
    ConversationExecutionReservation,
    ProviderLaneExecutionAttestation,
    ProviderLaneExecutionReceipt,
    ProviderLaneExecutionReservation,
    ProviderLaneExecutionStage,
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
)
from .observability import (
    authority_digest,
    canonical_request_digest,
    checkpoint_observation,
)
from .protocols import (
    ConversationAuthorityResolver,
    ConversationClock,
    ConversationObserver,
    ConversationPublisher,
    ConversationRetryWaiter,
    ConversationStore,
    CoordinatorBoundaryHook,
    FirstStoredProviderPlan,
    ProviderPlan,
    ProviderResult,
    StatelessProviderPlan,
    StoredProviderPlan,
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
    request_operation,
)
from .settings import (
    ConversationMode,
    EffectiveReasoningMetadata,
    ProviderLaneOutputScope,
    ProviderUsage,
    ReasoningContext,
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
    StatelessProviderLaneSnapshot,
    StoredProviderLaneSnapshot,
)
from .value import (
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from asyncio import CancelledError
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
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


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CoordinatorDiagnostics:
    """Report content-free active attempt and resource counts."""

    active_attempts: int
    closed: bool


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

    def accept(self, item: ProviderItem) -> None:
        if type(item) is not ProviderItem or item.lane_id != self.lane_id:
            raise ConversationValidationError()
        self.items.append(item)
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


@final
@dataclass(slots=True)
class _DispatchProgress:
    may_have_dispatched: bool = False

    def mark_possible_dispatch(self) -> None:
        self.may_have_dispatched = True


LanePlan: TypeAlias = tuple[
    ConversationLaneRequest,
    ConversationLaneRuntime,
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
        lanes: tuple[ConversationLaneRuntime, ...],
        max_attempts: int = 2,
        max_active_executions: int = 128,
        boundary_hook: CoordinatorBoundaryHook | None = None,
    ) -> None:
        if (
            type(lanes) is not tuple
            or not lanes
            or any(type(item) is not ConversationLaneRuntime for item in lanes)
        ):
            raise ConversationValidationError()
        for item in lanes:
            _validate_lane_runtime(item)
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
        self._store = store
        self._resolver = authority_resolver
        self._clock = clock
        self._publisher = publisher
        self._observer = observer
        self._retry_waiter = retry_waiter
        self._lanes = {item.binding.lane_id: item for item in lanes}
        self._fake_runtimes = {
            item.binding.lane_id: item._provider_runtime for item in lanes
        }
        self._max_attempts = max_attempts
        self._max_active_executions = max_active_executions
        self._hook = boundary_hook or _NoopCoordinatorBoundaryHook()
        self._active_attempts: set[str] = set()
        self._execution_sequence = 0
        self._closed = False

    @property
    def diagnostics(self) -> CoordinatorDiagnostics:
        """Return current content-free coordinator resource counts."""
        return CoordinatorDiagnostics(
            active_attempts=len(self._active_attempts),
            closed=self._closed,
        )

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
        if runtime is None or state is None:
            raise ConversationValidationError()
        _validate_runtime(runtime)
        return _provider_diagnostics(state, runtime.provider_script)

    async def execute(
        self, request: ConversationRunRequest
    ) -> AtomicCommitReceipt:
        """Execute a non-streaming fake-provider run."""
        return await self._run(request, streaming=False)

    async def stream(
        self, request: ConversationRunRequest
    ) -> AtomicCommitReceipt:
        """Execute a streaming fake-provider run to its terminal boundary."""
        return await self._run(request, streaming=True)

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
            if probe.disposition is StoreCloseDisposition.CLOSED:
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
            if cause is not None:
                raise cancellation from cause
            raise cancellation
        if action_error is not None:
            if consistency_error is not None:
                raise action_error from consistency_error
            raise action_error
        if consistency_error is not None:
            raise consistency_error

    async def _run(
        self,
        request: ConversationRunRequest,
        *,
        streaming: bool,
    ) -> AtomicCommitReceipt:
        if self._closed or type(request) is not ConversationRunRequest:
            raise ConversationValidationError()
        execution_token = self._activate_execution()
        identity = self._idempotency(request)
        owner_token: str | None = None
        committed = False
        progress = _DispatchProgress()
        try:
            await self._hook.reach(CoordinatorAwaitBoundary.RESOLVE_AUTHORITY)
            authority = await self._resolver.resolve()
            if authority != request.semantics.authority:
                raise ConversationAuthorizationError()
            if any(
                (runtime := self._lanes.get(lane_request.lane_id)) is not None
                and runtime.binding.agent_id != authority.agent_id
                for lane_request in request.lanes
            ):
                raise ConversationAuthorizationError()
            await self._hook.reach(
                CoordinatorAwaitBoundary.RESERVE_IDEMPOTENCY
            )
            execution_reservation = self._execution_reservation(
                request,
                identity,
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
                    await self._publish_one(
                        checkpoint,
                        resolution.public_response_id,
                        f"publication-{resolution.public_response_id}",
                    )
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
            plans = self._plan_lanes(request, parent, streaming=streaming)
            self._validate_limits_before_dispatch(request, parent, plans)
            await self._allocate(request, owner_token, authority)
            (
                snapshots,
                output_candidates,
                execution_attestations,
            ) = await self._dispatch_lanes(
                plans,
                execution_reservation=execution_reservation,
                owner_token=owner_token,
                authority=authority,
                identity=request.identity,
                streaming=streaming,
                progress=progress,
            )
            now = await self._clock.now()
            candidate = build_checkpoint_candidate(
                request,
                parent=parent,
                completed_lanes=snapshots,
                created_at=now,
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
                    else None
                ),
                expected_head_revision=(
                    request.advance.expected_revision
                    if isinstance(request.advance, NamedHeadAdvance)
                    else None
                ),
            )
            await self._hook.reach(CoordinatorAwaitBoundary.COMMIT)
            try:
                receipt = await self._store.commit_atomic(commit)
            except ConversationError:
                raise
            except Exception as exc:
                raise ConversationCommitError() from exc
            committed = True
            await self._observe("checkpoint_committed", receipt.checkpoint)
            if receipt.outbox is not None:
                await self._publish_one(
                    receipt.checkpoint,
                    receipt.outbox.intent.public_response_id,
                    receipt.outbox.intent.intent_id,
                )
            return receipt
        except BaseException as exc:
            if owner_token is not None and not committed:
                try:
                    await self._rollback(
                        identity,
                        owner_token,
                        ambiguous=(
                            progress.may_have_dispatched
                            or isinstance(
                                exc, ConversationAmbiguousDispatchError
                            )
                        ),
                    )
                except BaseException as cleanup_exc:
                    if isinstance(exc, CancelledError):
                        raise exc from cleanup_exc
                    raise cleanup_exc from exc
            raise
        finally:
            self._active_attempts.discard(execution_token)

    def _execution_reservation(
        self,
        request: ConversationRunRequest,
        idempotency: RequestIdempotencyIdentity,
    ) -> ConversationExecutionReservation:
        lanes: list[ProviderLaneExecutionReservation] = []
        for lane_request in request.lanes:
            runtime = self._lanes.get(lane_request.lane_id)
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
        )

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
    ) -> tuple[LanePlan, ...]:
        parent_lanes = (
            {lane.lane_id: lane for lane in parent.content.lanes}
            if parent is not None
            else {}
        )
        plans: list[LanePlan] = []
        for lane_request in request.lanes:
            runtime = self._lanes.get(lane_request.lane_id)
            if runtime is None:
                raise ConversationCapabilityError()
            runtime.capability_profile.assert_binding(runtime.binding)
            self._require_capabilities(
                lane_request, runtime, streaming=streaming
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
                plan: ProviderPlan = StatelessProviderPlan(
                    binding=runtime.binding,
                    ledger=ledger,
                    reasoning=reasoning,
                )
            else:
                if prior is not None and not isinstance(
                    prior, StoredProviderLaneSnapshot
                ):
                    raise ConversationValidationError()
                if isinstance(prior, StoredProviderLaneSnapshot):
                    plan = StoredProviderPlan(
                        binding=runtime.binding,
                        upstream_response_id=prior.upstream_response_id,
                        reasoning=reasoning,
                    )
                else:
                    plan = FirstStoredProviderPlan(
                        binding=runtime.binding,
                        reasoning=reasoning,
                    )
            plans.append((lane_request, runtime, plan))
        return tuple(plans)

    @staticmethod
    def _require_capabilities(
        lane: ConversationLaneRequest,
        runtime: ConversationLaneRuntime,
        *,
        streaming: bool,
    ) -> None:
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
        identity: CheckpointIdentity,
        streaming: bool,
        progress: _DispatchProgress,
    ) -> tuple[
        tuple[ProviderLaneSnapshot, ...],
        tuple[ProviderLaneOutputCandidate, ...],
        tuple[ProviderLaneExecutionAttestation, ...],
    ]:
        snapshots: list[ProviderLaneSnapshot] = []
        outputs: list[ProviderLaneOutputCandidate] = []
        attestations: list[ProviderLaneExecutionAttestation] = []
        for lane_request, runtime, plan in plans:
            staging = _AttemptStaging(lane_id=lane_request.lane_id, items=[])
            result = await self._dispatch_with_retry(
                runtime,
                plan,
                staging,
                streaming=streaming,
                progress=progress,
            )
            scope = ProviderLaneOutputScope.CURRENT_CALL
            execution_receipt = provider_lane_execution_receipt(
                authority=authority,
                identity=identity,
                binding=runtime.binding,
                mode=lane_request.mode,
                scope=scope,
                completed_items=result.items,
                reasoning=result.reasoning,
                usage=result.usage,
                upstream_response_id=result.upstream_response_id,
            )
            snapshots.append(
                self._lane_snapshot(
                    lane_request,
                    runtime,
                    plan,
                    result,
                    execution_receipt,
                )
            )
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
                        identity=identity,
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
        return tuple(snapshots), tuple(outputs), tuple(attestations)

    async def _dispatch_with_retry(
        self,
        runtime: ConversationLaneRuntime,
        plan: ProviderPlan,
        staging: _AttemptStaging,
        *,
        streaming: bool,
        progress: _DispatchProgress,
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
                # effects always cross the closed repository executor below;
                # no method or awaitable is resolved from caller-owned data.
                if streaming:
                    result = await self._stream_once(
                        runtime,
                        provider_runtime,
                        plan,
                        staging,
                        progress,
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
                return result
            except ConversationError as exc:
                disposition = reduce_failure(
                    exc.boundary,
                    visible_output=staging.visible_output,
                    tool_effect=staging.tool_effect,
                    committed=False,
                    ambiguous=isinstance(
                        exc, ConversationAmbiguousDispatchError
                    )
                    or progress.may_have_dispatched,
                )
                staging.rollback()
                if (
                    disposition.retry_rule is not RetryRule.BOUNDED_EFFECT_FREE
                    or attempt == self._max_attempts
                ):
                    raise
                await self._hook.reach(CoordinatorAwaitBoundary.RETRY_WAIT)
                await self._retry_waiter.wait(attempt)
                attempt += 1

    async def _stream_once(
        self,
        runtime: ConversationLaneRuntime,
        provider_runtime: _DeterministicFakeProviderRuntime,
        plan: ProviderPlan,
        staging: _AttemptStaging,
        progress: _DispatchProgress,
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
        runtime: ConversationLaneRuntime,
        plan: ProviderPlan,
        result: ProviderResult,
        execution_receipt: ProviderLaneExecutionReceipt,
    ) -> ProviderLaneSnapshot:
        for item in result.items:
            if item.lane_id != lane_request.lane_id:
                raise ConversationValidationError()
        if lane_request.mode is ConversationMode.STATELESS:
            if type(plan) is not StatelessProviderPlan:
                raise ConversationValidationError()
            ledger = ProviderItemLedger(
                lane_id=lane_request.lane_id,
                normalization_version=runtime.binding.continuation_codec_version,
                items=plan.ledger.items + result.items,
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
                binding=runtime.binding,
                ledger=ledger,
                reasoning=result.reasoning,
                lifecycle=ProviderLaneLifecycle.COMMITTED,
                retention_policy=runtime.retention_policy,
                compaction_boundary=boundary,
                execution_receipt=execution_receipt,
            )
        ProviderItemLedger(
            lane_id=lane_request.lane_id,
            normalization_version=runtime.binding.continuation_codec_version,
            items=result.items,
        )
        if result.upstream_response_id is None:
            raise ConversationValidationError()
        return StoredProviderLaneSnapshot(
            binding=runtime.binding,
            upstream_response_id=result.upstream_response_id,
            reasoning=result.reasoning,
            lifecycle=ProviderLaneLifecycle.COMMITTED,
            retention_policy=runtime.retention_policy,
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


def build_checkpoint_candidate(
    request: ConversationRunRequest,
    *,
    parent: ConversationCheckpoint | None,
    completed_lanes: tuple[ProviderLaneSnapshot, ...],
    created_at: datetime,
) -> CheckpointCandidate:
    """Build one immutable staged candidate at a validated boundary."""
    if (
        type(request) is not ConversationRunRequest
        or not isinstance(created_at, datetime)
        or created_at.utcoffset() is None
        or type(completed_lanes) is not tuple
        or not completed_lanes
    ):
        raise ConversationValidationError()
    selected = {lane.lane_id for lane in completed_lanes}
    retained: tuple[ProviderLaneSnapshot, ...] = ()
    transcript_entries = request.visible_delta
    if parent is not None:
        transcript_entries = (
            parent.content.visible_transcript.entries + request.visible_delta
        )
        retained = tuple(
            lane
            for lane in parent.content.lanes
            if lane.lane_id not in selected
            and lane.retention_policy is ChildLaneRetentionPolicy.RETAIN
        )
    lanes = completed_lanes + retained
    ttl = request.retention.effective_ttl_seconds
    head = None
    if isinstance(request.advance, NamedHeadAdvance):
        head = NamedHeadMetadata(
            head_id=request.advance.head_id,
            revision=NamedHeadRevision(request.advance.expected_revision + 1),
        )
    kind = (
        CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        if request.boundary is ConversationCommitBoundary.INTERNAL_SEGMENT
        else CheckpointKind.COMPLETED_OUTWARD_TURN
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
