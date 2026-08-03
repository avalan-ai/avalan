"""Implement the bounded asynchronous in-memory conversation store."""

from .codec import ConversationCheckpointCodec, with_checkpoint_integrity
from .contract import (
    AuthorityScope,
    CheckpointId,
    CheckpointIdentity,
    CheckpointKind,
    IdempotencyDisposition,
    IdempotencyRecordState,
    LocalDeletionState,
    NamedHeadId,
    NamedHeadRevision,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyIdentity,
)
from .errors import (
    ConversationAuthorizationError,
    ConversationConflictError,
    ConversationLimitError,
    ConversationStorageError,
    ConversationTransitionError,
    ConversationValidationError,
)
from .execution import (
    ConversationExecutionReservation,
    ProviderLaneExecutionAttestation,
    ProviderLaneExecutionReservation,
    ProviderLaneExecutionStage,
    provider_lane_execution_receipt,
)
from .items import ProviderItemLedger
from .lifecycle import (
    AmbiguousDispatchReconciliationDisposition,
    AmbiguousDispatchReconciliationRequest,
    AmbiguousDispatchReconciliationResult,
    AmbiguousDispatchResolution,
    LocalDeletionPreparation,
    ProviderLifecycleOrigin,
    ProviderLifecycleWorkRecord,
    ProviderLifecycleWorkState,
    ProviderQuarantineReceipt,
    ProviderQuarantineRequest,
)
from .observability import authority_digest
from .protocols import (
    ConversationClock,
    ConversationOutboxRecoveryWorker,
    ConversationUnitOfWork,
)
from .runtime import (
    AtomicCommitReceipt,
    AtomicConversationCommit,
    CheckpointPage,
    IdempotencyResolution,
    IdempotencySettlementDisposition,
    IdempotencySettlementResolution,
    NamedHeadAdvance,
    OutboxClaimDisposition,
    OutboxClaimResolution,
    OutboxClaimTarget,
    OutboxRecord,
    OutboxRecoveryBatch,
    OutboxRecoveryDisposition,
    OutboxState,
    ProviderLaneOutputCandidate,
    ProvisionalPublicResponse,
    PruneReceipt,
    PublicationIntent,
    PublicResponseRecord,
    StoreCloseDisposition,
    StoreCloseResolution,
    StoreLimits,
    SweepReceipt,
)
from .settings import (
    ConversationHandle,
    ConversationMode,
    ConversationResult,
    ProviderLaneOutputScope,
    StatelessConversationHandle,
    StoredConversationHandle,
)
from .state import (
    CheckpointCandidate,
    CheckpointLifecycle,
    ConversationCheckpoint,
    ExecutionSegmentCheckpointCandidate,
    NamedHeadLifecycle,
    NamedHeadMetadata,
    NamedHeadSnapshot,
    OutwardTurnCheckpointCandidate,
    StandaloneCompactCheckpointCandidate,
    StatelessProviderLaneSnapshot,
    StoredProviderLaneSnapshot,
    SuspensionCheckpointCandidate,
)
from .value import AuthorityDigest, IntegrityDigest, validate_identifier

from asyncio import CancelledError, Condition, Lock
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hmac import compare_digest
from typing import Protocol, final


class StoreAwaitBoundary(StrEnum):
    """Identify every injectable asynchronous store boundary."""

    CREATE = "create"
    LOAD = "load"
    AUTHORIZE = "authorize"
    STAGE = "stage"
    COMMIT = "commit"
    COMMIT_ATOMIC = "commit_atomic"
    EXECUTION_STAGE = "execution_stage"
    HEAD = "head"
    BRANCH = "branch"
    IDEMPOTENCY = "idempotency"
    IDEMPOTENCY_RECONCILE_BEGIN = "idempotency_reconcile_begin"
    IDEMPOTENCY_RECONCILE = "idempotency_reconcile"
    IDEMPOTENCY_RECONCILE_SETTLED = "idempotency_reconcile_settled"
    IDEMPOTENCY_SETTLEMENT = "idempotency_settlement"
    ALLOCATE = "allocate"
    ROLLBACK_BEGIN = "rollback_begin"
    ROLLBACK = "rollback"
    ROLLBACK_SETTLED = "rollback_settled"
    RETRIEVE = "retrieve"
    RETRIEVE_OUTPUTS = "retrieve_outputs"
    PREPARE_DELETE = "prepare_delete"
    TOMBSTONE = "tombstone"
    DELETE = "delete"
    LIST = "list"
    SWEEP = "sweep"
    PRUNE = "prune"
    OUTBOX_CLAIM = "outbox_claim"
    OUTBOX_RECOVERY_CLAIM = "outbox_recovery_claim"
    OUTBOX_ACKNOWLEDGE = "outbox_acknowledge"
    OUTBOX_RELEASE = "outbox_release"
    CLOSE_BEGIN = "close_begin"
    CLOSE = "close"
    CLOSE_SETTLED = "close_settled"
    CLOSE_STATUS = "close_status"


class StoreBoundaryHook(Protocol):
    """Inject deterministic behavior before an asynchronous store boundary."""

    async def reach(self, boundary: StoreAwaitBoundary) -> None:
        """Reach one named boundary before store state is locked."""
        ...


@final
class _NoopStoreBoundaryHook:
    async def reach(self, boundary: StoreAwaitBoundary) -> None:
        assert isinstance(boundary, StoreAwaitBoundary)


@final
class _UtcStoreClock:
    async def now(self) -> datetime:
        return datetime.now(UTC)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoreDiagnostics:
    """Report bounded content-free in-memory store resource counts."""

    checkpoints: int
    provisional_responses: int
    public_responses: int
    idempotency_records: int
    idempotency_waiters: int
    heads: int
    outbox_records: int
    output_records: int
    terminal_metadata: int
    staged_executions: int
    locked: bool
    closed: bool


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _StoredCheckpoint:
    checkpoint: ConversationCheckpoint
    encoded: bytes
    authority_digest: str


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _IdempotencyEntry:
    identity: RequestIdempotencyIdentity
    state: IdempotencyRecordState
    owner_token: str
    lease_expires_at: datetime
    execution: "_ExecutionReservationRecord | None" = None
    checkpoint_id: CheckpointId | None = None
    public_response_id: PublicResponseId | None = None


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _ExecutionLaneReservationRecord:
    lane_id: str
    binding_digest: str
    mode: ConversationMode
    scope: ProviderLaneOutputScope


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _ExecutionReservationRecord:
    checkpoint_identity: tuple[
        str,
        str,
        str,
        str,
        str,
        int,
        str | None,
        int | None,
    ]
    lanes: tuple[_ExecutionLaneReservationRecord, ...]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _StagedExecutionRecord:
    staging_id: str
    idempotency_key: tuple[str, str, str]
    request_digest: str
    authority_digest: str
    owner_token: str
    checkpoint_identity: tuple[
        str,
        str,
        str,
        str,
        str,
        int,
        str | None,
        int | None,
    ]
    lane_id: str
    binding_digest: str
    mode: ConversationMode
    scope: ProviderLaneOutputScope
    execution_digest: str
    item_count: int
    opaque_byte_count: int


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _TerminalMetadata:
    checkpoint_id: CheckpointId
    public_response_id: PublicResponseId | None
    authority_digest: str
    state: CheckpointLifecycle
    at: datetime


@final
class InMemoryConversationUnitOfWork:
    """Own one staged candidate until explicit commit or rollback."""

    def __init__(
        self,
        store: "InMemoryConversationStore",
        candidate: CheckpointCandidate,
    ) -> None:
        self._store = store
        self._candidate = candidate
        self._finished = False

    async def __aenter__(self) -> ConversationUnitOfWork:
        if self._finished:
            raise ConversationStorageError()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if not self._finished:
            await self.rollback()

    async def commit(self) -> ConversationCheckpoint:
        if self._finished:
            raise ConversationStorageError()
        checkpoint = await self._store.commit(self._candidate)
        self._finished = True
        return checkpoint

    async def rollback(self) -> None:
        self._finished = True


@final
class InMemoryConversationStore:
    """Persist immutable checkpoints under a bounded async API."""

    _CONCEALED_DIGEST = "0" * 64

    def __init__(
        self,
        *,
        limits: StoreLimits = StoreLimits(),
        codec: ConversationCheckpointCodec = ConversationCheckpointCodec(),
        clock: ConversationClock | None = None,
        boundary_hook: StoreBoundaryHook | None = None,
    ) -> None:
        if (
            type(limits) is not StoreLimits
            or type(codec) is not ConversationCheckpointCodec
        ):
            raise ConversationValidationError()
        self._limits = limits
        self._codec = codec
        self._clock = clock or _UtcStoreClock()
        self._hook = boundary_hook or _NoopStoreBoundaryHook()
        self._lock = Lock()
        self._idempotency_changed = Condition(self._lock)
        self._checkpoints: dict[CheckpointId, _StoredCheckpoint] = {}
        self._children: dict[CheckpointId, set[CheckpointId]] = {}
        self._provisional: dict[
            ProvisionalResponseId, ProvisionalPublicResponse
        ] = {}
        self._public: dict[PublicResponseId, PublicResponseRecord] = {}
        self._results: dict[PublicResponseId, ConversationResult] = {}
        self._outputs: dict[
            CheckpointId, tuple[ProviderLaneOutputCandidate, ...]
        ] = {}
        self._idempotency: dict[tuple[str, str, str], _IdempotencyEntry] = {}
        self._execution_staging: dict[str, _StagedExecutionRecord] = {}
        self._execution_stage_keys: dict[tuple[str, str, str], str] = {}
        self._heads: dict[tuple[str, NamedHeadId], NamedHeadSnapshot] = {}
        self._outbox: dict[str, OutboxRecord] = {}
        self._outbox_ready_order: dict[str, int] = {}
        self._terminal: dict[CheckpointId, _TerminalMetadata] = {}
        self._provider_lifecycle: dict[str, ProviderLifecycleWorkRecord] = {}
        self._owner_sequence = 0
        self._execution_stage_sequence = 0
        self._outbox_ready_sequence = 0
        self._idempotency_waiters = 0
        self._closed = False

    @property
    def diagnostics(self) -> StoreDiagnostics:
        """Return current content-free resource counts without awaiting."""
        return StoreDiagnostics(
            checkpoints=len(self._checkpoints),
            provisional_responses=len(self._provisional),
            public_responses=len(self._public),
            idempotency_records=len(self._idempotency),
            idempotency_waiters=self._idempotency_waiters,
            heads=len(self._heads),
            outbox_records=len(self._outbox),
            output_records=len(self._outputs),
            terminal_metadata=len(self._terminal),
            staged_executions=len(self._execution_staging),
            locked=self._lock.locked(),
            closed=self._closed,
        )

    async def create(
        self, candidate: CheckpointCandidate
    ) -> ConversationCheckpoint:
        await self._hook.reach(StoreAwaitBoundary.CREATE)
        return await self._commit_candidate(candidate)

    async def create_with_named_head(
        self,
        candidate: CheckpointCandidate,
        advance: NamedHeadAdvance,
    ) -> ConversationCheckpoint:
        """Create a checkpoint and advance an exact head atomically."""
        if type(advance) is not NamedHeadAdvance:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.CREATE)
        staged = self._candidate_checkpoint(candidate)
        committed = self._committed_checkpoint(
            staged,
            staged.timestamps.created_at,
        )
        expected_head = NamedHeadMetadata(
            head_id=advance.head_id,
            revision=NamedHeadRevision(advance.expected_revision + 1),
        )
        if (
            committed.head != expected_head
            or committed.identity.parent_checkpoint_id
            == advance.parent_checkpoint_id
        ):
            raise ConversationValidationError()
        encoded = self._codec.encode(committed)
        authority_key = str(authority_digest(committed.authority))
        async with self._lock:
            self._ensure_open_locked()
            self._validate_checkpoint_write_locked(committed, encoded)
            direct_parent_id = committed.identity.parent_checkpoint_id
            assert direct_parent_id is not None
            direct_parent = self._checkpoints[direct_parent_id].checkpoint
            if (
                direct_parent.kind
                is not CheckpointKind.STANDALONE_COMPACT_RESULT
                or direct_parent.identity.parent_checkpoint_id
                != advance.parent_checkpoint_id
            ):
                raise ConversationValidationError()
            current = self._heads.get((authority_key, advance.head_id))
            if (
                current is None
                or current.lifecycle is not NamedHeadLifecycle.ACTIVE
                or current.revision != advance.expected_revision
                or current.checkpoint_id != advance.parent_checkpoint_id
            ):
                raise ConversationConflictError()
            checkpoint_id = committed.identity.checkpoint_id
            self._checkpoints[checkpoint_id] = _StoredCheckpoint(
                checkpoint=committed,
                encoded=encoded,
                authority_digest=authority_key,
            )
            self._register_child_locked(committed)
            self._heads[(authority_key, advance.head_id)] = NamedHeadSnapshot(
                head_id=advance.head_id,
                revision=NamedHeadRevision(advance.expected_revision + 1),
                checkpoint_id=checkpoint_id,
            )
        return committed

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        await self._hook.reach(StoreAwaitBoundary.LOAD)
        return await self._authorized_checkpoint(checkpoint_id, authority)

    async def authorize(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        await self._hook.reach(StoreAwaitBoundary.AUTHORIZE)
        return await self._authorized_checkpoint(checkpoint_id, authority)

    async def stage(
        self, candidate: CheckpointCandidate
    ) -> ConversationUnitOfWork:
        await self._hook.reach(StoreAwaitBoundary.STAGE)
        self._candidate_checkpoint(candidate)
        async with self._lock:
            self._ensure_open_locked()
        return InMemoryConversationUnitOfWork(self, candidate)

    async def commit(
        self, candidate: CheckpointCandidate
    ) -> ConversationCheckpoint:
        await self._hook.reach(StoreAwaitBoundary.COMMIT)
        return await self._commit_candidate(candidate)

    async def stage_execution(
        self,
        stage: ProviderLaneExecutionStage,
    ) -> ProviderLaneExecutionAttestation:
        """Stage one exact owner-bound lane result for atomic consumption."""
        if type(stage) is not ProviderLaneExecutionStage:
            raise ConversationValidationError()
        expected_receipt = provider_lane_execution_receipt(
            authority=stage.idempotency.authority,
            identity=stage.identity,
            binding=stage.binding,
            mode=stage.mode,
            scope=stage.scope,
            completed_items=stage.completed_items,
            reasoning=stage.reasoning,
            usage=stage.usage,
            upstream_response_id=stage.upstream_response_id,
        )
        if stage.execution_receipt != expected_receipt:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.EXECUTION_STAGE)
        async with self._lock:
            self._ensure_open_locked()
            key = self._idempotency_key(stage.idempotency)
            current = self._idempotency.get(key)
            checkpoint_identity = self._checkpoint_identity_key(stage.identity)
            expected_lane = (
                next(
                    (
                        lane
                        for lane in current.execution.lanes
                        if lane.lane_id == stage.binding.lane_id
                    ),
                    None,
                )
                if current is not None and current.execution is not None
                else None
            )
            if (
                current is None
                or current.identity != stage.idempotency
                or current.owner_token != stage.owner_token
                or current.state is not IdempotencyRecordState.IN_PROGRESS
                or current.execution is None
                or current.execution.checkpoint_identity != checkpoint_identity
                or expected_lane is None
                or expected_lane.binding_digest
                != str(stage.binding.integrity_digest)
                or expected_lane.mode is not stage.mode
                or expected_lane.scope is not stage.scope
            ):
                raise ConversationConflictError()
            stage_key = (
                stage.owner_token,
                str(stage.identity.checkpoint_id),
                str(stage.binding.lane_id),
            )
            if stage_key in self._execution_stage_keys:
                raise ConversationConflictError()
            if (
                len(self._execution_staging)
                >= self._limits.max_staged_execution_records
            ):
                raise ConversationLimitError()
            staging_id = self._next_execution_stage_id_locked()
            record = _StagedExecutionRecord(
                staging_id=staging_id,
                idempotency_key=key,
                request_digest=str(stage.idempotency.request_digest),
                authority_digest=str(
                    authority_digest(stage.idempotency.authority)
                ),
                owner_token=stage.owner_token,
                checkpoint_identity=checkpoint_identity,
                lane_id=str(stage.binding.lane_id),
                binding_digest=str(stage.binding.integrity_digest),
                mode=stage.mode,
                scope=stage.scope,
                execution_digest=str(expected_receipt.digest),
                item_count=expected_receipt.item_count,
                opaque_byte_count=expected_receipt.opaque_byte_count,
            )
            self._execution_staging[staging_id] = record
            self._execution_stage_keys[stage_key] = staging_id
            return ProviderLaneExecutionAttestation(
                schema_version=1,
                staging_id=staging_id,
                lane_id=stage.binding.lane_id,
            )

    async def commit_atomic(
        self, commit: AtomicConversationCommit
    ) -> AtomicCommitReceipt:
        if type(commit) is not AtomicConversationCommit:
            raise ConversationValidationError()
        self._validate_atomic_commit_value(commit)
        await self._hook.reach(StoreAwaitBoundary.COMMIT_ATOMIC)
        staged = self._candidate_checkpoint(commit.candidate)
        committed = self._committed_checkpoint(staged, commit.committed_at)
        encoded = self._codec.encode(committed)
        authority_key = str(authority_digest(committed.authority))
        async with self._lock:
            self._ensure_open_locked()
            self._validate_checkpoint_write_locked(committed, encoded)
            parent_id = committed.identity.parent_checkpoint_id
            parent = (
                self._checkpoints[parent_id].checkpoint
                if parent_id is not None
                else None
            )
            self._validate_output_candidates(
                committed,
                commit.output_candidates,
                parent=parent,
            )
            key = self._idempotency_key(commit.idempotency)
            entry = self._idempotency.get(key)
            if (
                entry is None
                or entry.identity != commit.idempotency
                or entry.owner_token != commit.owner_token
                or entry.state is not IdempotencyRecordState.IN_PROGRESS
            ):
                raise ConversationConflictError()
            staging_ids = self._validate_staged_executions_locked(
                commit,
                committed,
                entry,
            )
            provisional = self._validate_provisional_locked(
                commit, authority_key
            )
            head = self._validate_head_locked(commit, committed)
            result = self._build_result(commit, committed)
            outbox = self._build_outbox_locked(commit, committed, result)
            checkpoint_id = committed.identity.checkpoint_id
            self._checkpoints[checkpoint_id] = _StoredCheckpoint(
                checkpoint=committed,
                encoded=encoded,
                authority_digest=authority_key,
            )
            self._register_child_locked(committed)
            self._outputs[checkpoint_id] = commit.output_candidates
            if provisional is not None:
                del self._provisional[provisional.provisional_response_id]
            if commit.public_response_id is not None:
                assert result is not None
                self._public[commit.public_response_id] = PublicResponseRecord(
                    public_response_id=commit.public_response_id,
                    checkpoint_id=checkpoint_id,
                    authority_digest=authority_key,
                )
                self._results[commit.public_response_id] = result
            if head is not None and commit.head_id is not None:
                self._heads[(authority_key, commit.head_id)] = head
            if outbox is not None:
                self._outbox[outbox.intent.intent_id] = outbox
                self._append_outbox_ready_locked(outbox.intent.intent_id)
            self._consume_staged_executions_locked(staging_ids)
            self._idempotency[key] = replace(
                entry,
                state=IdempotencyRecordState.COMMITTED,
                checkpoint_id=checkpoint_id,
                public_response_id=commit.public_response_id,
            )
            self._idempotency_changed.notify_all()
        return AtomicCommitReceipt(
            checkpoint=committed,
            result=result,
            outbox=outbox,
            output_candidates=commit.output_candidates,
        )

    async def create_head(
        self,
        head: NamedHeadSnapshot,
        authority: AuthorityScope,
    ) -> None:
        if type(head) is not NamedHeadSnapshot:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.HEAD)
        async with self._lock:
            self._ensure_open_locked()
            self._authorize_entry_locked(head.checkpoint_id, authority)
            key = self._head_key(authority, head.head_id)
            if key in self._heads:
                raise ConversationConflictError()
            if len(self._heads) >= self._limits.max_heads:
                raise ConversationLimitError()
            self._heads[key] = head

    async def load_head(
        self,
        head_id: NamedHeadId,
        authority: AuthorityScope,
    ) -> NamedHeadSnapshot:
        validate_identifier(head_id, "head_id")
        await self._hook.reach(StoreAwaitBoundary.HEAD)
        async with self._lock:
            self._ensure_open_locked()
            head = self._heads.get(self._head_key(authority, head_id))
            checkpoint_id = (
                head.checkpoint_id
                if head is not None
                else CheckpointId("concealed-checkpoint")
            )
            self._authorize_entry_locked(checkpoint_id, authority)
            if head is None or head.lifecycle is not NamedHeadLifecycle.ACTIVE:
                raise ConversationAuthorizationError()
            return head

    async def branch_count(
        self,
        parent_checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> int:
        await self._hook.reach(StoreAwaitBoundary.BRANCH)
        async with self._lock:
            self._ensure_open_locked()
            self._authorize_entry_locked(parent_checkpoint_id, authority)
            return len(self._children.get(parent_checkpoint_id, set()))

    async def reserve_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        *,
        execution: ConversationExecutionReservation | None = None,
    ) -> IdempotencyResolution:
        if type(identity) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        execution_record = self._execution_reservation_record(
            identity,
            execution,
        )
        await self._hook.reach(StoreAwaitBoundary.IDEMPOTENCY)
        now = await self._clock.now()
        self._validate_time(now)
        async with self._idempotency_changed:
            key = self._idempotency_key(identity)
            while True:
                self._ensure_open_locked()
                current = self._idempotency.get(key)
                if (
                    current is not None
                    and current.identity.request_digest
                    != (identity.request_digest)
                ):
                    return IdempotencyResolution(
                        disposition=IdempotencyDisposition.CONFLICT
                    )
                if (
                    current is not None
                    and current.execution != execution_record
                ):
                    return IdempotencyResolution(
                        disposition=IdempotencyDisposition.CONFLICT
                    )
                if current is not None:
                    if current.state is IdempotencyRecordState.COMMITTED:
                        assert current.checkpoint_id is not None
                        return IdempotencyResolution(
                            disposition=(
                                IdempotencyDisposition.REPLAY_COMMITTED
                            ),
                            checkpoint_id=current.checkpoint_id,
                            public_response_id=current.public_response_id,
                        )
                    if current.state is IdempotencyRecordState.AMBIGUOUS:
                        return IdempotencyResolution(
                            disposition=IdempotencyDisposition.FENCED
                        )
                    if current.state is IdempotencyRecordState.IN_PROGRESS:
                        if current.lease_expires_at <= now:
                            self._remove_provisional_owner_locked(
                                current.owner_token
                            )
                            self._remove_staged_execution_owner_locked(
                                current.owner_token
                            )
                            self._idempotency[key] = replace(
                                current,
                                state=IdempotencyRecordState.AMBIGUOUS,
                            )
                            self._idempotency_changed.notify_all()
                            return IdempotencyResolution(
                                disposition=IdempotencyDisposition.FENCED
                            )
                        # The closed idempotency contract fences every matching
                        # in-progress record until its authoritative owner
                        # settles; duplicate callers never join owner state.
                        return IdempotencyResolution(
                            disposition=IdempotencyDisposition.FENCED
                        )
                in_flight = sum(
                    item.state
                    in {
                        IdempotencyRecordState.IN_PROGRESS,
                        IdempotencyRecordState.AMBIGUOUS,
                    }
                    for item in self._idempotency.values()
                )
                if in_flight >= self._limits.max_in_flight:
                    raise ConversationLimitError()
                if (
                    current is None
                    and len(self._idempotency)
                    >= self._limits.max_idempotency_records
                ):
                    raise ConversationLimitError()
                owner_token = self._next_owner_token_locked()
                self._idempotency[key] = _IdempotencyEntry(
                    identity=identity,
                    state=IdempotencyRecordState.IN_PROGRESS,
                    owner_token=owner_token,
                    lease_expires_at=now
                    + timedelta(
                        seconds=self._limits.idempotency_lease_seconds
                    ),
                    execution=execution_record,
                )
                return IdempotencyResolution(
                    disposition=IdempotencyDisposition.EXECUTE,
                    owner_token=owner_token,
                )

    async def fence_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> None:
        if type(ambiguous) is not bool:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.IDEMPOTENCY)
        except CancelledError as exc:
            cancellation = exc
        async with self._idempotency_changed:
            self._ensure_open_locked()
            key = self._idempotency_key(identity)
            current = self._idempotency.get(key)
            if (
                current is None
                or current.identity != identity
                or current.owner_token != owner_token
                or current.state is not IdempotencyRecordState.IN_PROGRESS
            ):
                raise ConversationConflictError()
            self._idempotency[key] = replace(
                current,
                state=(
                    IdempotencyRecordState.AMBIGUOUS
                    if ambiguous
                    else IdempotencyRecordState.FAILED_NO_DISPATCH
                ),
            )
            self._idempotency_changed.notify_all()
        if cancellation is not None:
            raise cancellation

    async def abandon_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> IdempotencySettlementResolution:
        if type(identity) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        if type(ambiguous) is not bool:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.ROLLBACK_BEGIN)
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.ROLLBACK)
        except CancelledError as exc:
            cancellation = exc
        async with self._idempotency_changed:
            self._ensure_open_locked()
            key = self._idempotency_key(identity)
            current = self._idempotency.get(key)
            if (
                current is None
                or current.identity != identity
                or current.owner_token != owner_token
                or current.state is not IdempotencyRecordState.IN_PROGRESS
            ):
                raise ConversationConflictError()
            self._remove_provisional_owner_locked(owner_token)
            self._remove_staged_execution_owner_locked(owner_token)
            if ambiguous:
                self._idempotency[key] = replace(
                    current,
                    state=IdempotencyRecordState.AMBIGUOUS,
                )
            else:
                del self._idempotency[key]
            self._idempotency_changed.notify_all()
            resolution = self._idempotency_settlement_locked(
                identity,
                owner_token,
            )
        try:
            await self._hook.reach(StoreAwaitBoundary.ROLLBACK_SETTLED)
        except CancelledError as exc:
            cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation
        return resolution

    async def reconcile_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> IdempotencySettlementResolution:
        if type(identity) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        if type(ambiguous) is not bool:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.IDEMPOTENCY_RECONCILE_BEGIN)
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.IDEMPOTENCY_RECONCILE)
        except CancelledError as exc:
            cancellation = exc
        async with self._idempotency_changed:
            self._ensure_open_locked()
            key = self._idempotency_key(identity)
            current = self._idempotency.get(key)
            if current is None:
                resolution = IdempotencySettlementResolution(
                    disposition=IdempotencySettlementDisposition.SETTLED
                )
            elif (
                current.identity != identity
                or current.owner_token != owner_token
            ):
                raise ConversationConflictError()
            else:
                self._remove_provisional_owner_locked(owner_token)
                self._remove_staged_execution_owner_locked(owner_token)
                if current.state is IdempotencyRecordState.IN_PROGRESS:
                    if ambiguous:
                        self._idempotency[key] = replace(
                            current,
                            state=IdempotencyRecordState.AMBIGUOUS,
                        )
                    else:
                        del self._idempotency[key]
                    self._idempotency_changed.notify_all()
                resolution = self._idempotency_settlement_locked(
                    identity,
                    owner_token,
                )
        try:
            await self._hook.reach(
                StoreAwaitBoundary.IDEMPOTENCY_RECONCILE_SETTLED
            )
        except CancelledError as exc:
            cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation
        return resolution

    async def reconcile_ambiguous_dispatch(
        self,
        request: AmbiguousDispatchReconciliationRequest,
    ) -> AmbiguousDispatchReconciliationResult:
        """Apply one explicit durable ambiguity decision."""
        if type(request) is not AmbiguousDispatchReconciliationRequest:
            raise ConversationValidationError()
        dispositions = AmbiguousDispatchReconciliationDisposition
        key = (
            str(authority_digest(request.authority)),
            request.operation.value,
            str(request.idempotency_key),
        )
        async with self._idempotency_changed:
            self._ensure_open_locked()
            current = self._idempotency.get(key)
            if current is None:
                return AmbiguousDispatchReconciliationResult(
                    disposition=(dispositions.NOT_FOUND_OR_UNAUTHORIZED)
                )
            if current.state is IdempotencyRecordState.FAILED_NO_DISPATCH:
                disposition = dispositions.ALREADY_RESOLVED_NO_DISPATCH
            elif current.state is not IdempotencyRecordState.AMBIGUOUS:
                raise ConversationConflictError()
            elif (
                request.resolution is AmbiguousDispatchResolution.RETAIN_FENCE
            ):
                disposition = dispositions.FENCE_RETAINED
            else:
                self._idempotency[key] = replace(
                    current,
                    state=IdempotencyRecordState.FAILED_NO_DISPATCH,
                )
                self._idempotency_changed.notify_all()
                disposition = dispositions.RESOLVED_NO_DISPATCH
        return AmbiguousDispatchReconciliationResult(disposition=disposition)

    async def inspect_idempotency_settlement(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
    ) -> IdempotencySettlementResolution:
        if type(identity) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        await self._hook.reach(StoreAwaitBoundary.IDEMPOTENCY_SETTLEMENT)
        async with self._lock:
            return self._idempotency_settlement_locked(identity, owner_token)

    async def allocate_public_response(
        self, allocation: ProvisionalPublicResponse
    ) -> None:
        if type(allocation) is not ProvisionalPublicResponse:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.ALLOCATE)
        async with self._lock:
            self._ensure_open_locked()
            owner = next(
                (
                    entry
                    for entry in self._idempotency.values()
                    if entry.owner_token == allocation.owner_token
                    and entry.state is IdempotencyRecordState.IN_PROGRESS
                ),
                None,
            )
            if owner is None or not compare_digest(
                str(authority_digest(owner.identity.authority)),
                allocation.authority_digest,
            ):
                raise ConversationConflictError()
            if (
                allocation.provisional_response_id in self._provisional
                or allocation.public_response_id in self._public
                or any(
                    item.public_response_id == allocation.public_response_id
                    for item in self._provisional.values()
                )
            ):
                raise ConversationConflictError()
            if (
                len(self._provisional)
                >= self._limits.max_provisional_responses
                or len(self._public) + len(self._provisional)
                >= self._limits.max_public_responses
                or len(self._outbox) + len(self._provisional)
                >= self._limits.max_outbox_records
            ):
                raise ConversationLimitError()
            self._provisional[allocation.provisional_response_id] = allocation

    async def rollback_attempt(self, owner_token: str) -> None:
        validate_identifier(owner_token, "owner_token")
        await self._hook.reach(StoreAwaitBoundary.ROLLBACK_BEGIN)
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.ROLLBACK)
        except CancelledError as exc:
            cancellation = exc
        async with self._lock:
            self._ensure_open_locked()
            self._remove_provisional_owner_locked(owner_token)
            self._remove_staged_execution_owner_locked(owner_token)
        try:
            await self._hook.reach(StoreAwaitBoundary.ROLLBACK_SETTLED)
        except CancelledError as exc:
            cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation

    async def retrieve_output_candidates(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> tuple[ProviderLaneOutputCandidate, ...]:
        validate_identifier(checkpoint_id, "checkpoint_id")
        await self._hook.reach(StoreAwaitBoundary.RETRIEVE_OUTPUTS)
        async with self._lock:
            self._ensure_open_locked()
            self._authorize_entry_locked(checkpoint_id, authority)
            candidates = self._outputs.get(checkpoint_id)
            if candidates is None:
                raise ConversationAuthorizationError()
            return candidates

    async def retrieve(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> ConversationResult:
        validate_identifier(public_response_id, "public_response_id")
        await self._hook.reach(StoreAwaitBoundary.RETRIEVE)
        async with self._lock:
            self._ensure_open_locked()
            record = self._authorized_public_locked(
                public_response_id, authority, allow_tombstone=False
            )
            result = self._results.get(record.public_response_id)
            if result is None:
                raise ConversationAuthorizationError()
            return result

    async def prepare_deletion(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> LocalDeletionPreparation:
        """Resolve one authorized deletion without disclosing private state."""
        validate_identifier(public_response_id, "public_response_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        await self._hook.reach(StoreAwaitBoundary.PREPARE_DELETE)
        supplied = str(authority_digest(authority))
        async with self._lock:
            self._ensure_open_locked()
            record = self._public.get(public_response_id)
            if record is not None:
                authorized = compare_digest(supplied, record.authority_digest)
                stored = self._checkpoints.get(record.checkpoint_id)
                available = stored is not None and (
                    (
                        not record.tombstoned
                        and stored.checkpoint.lifecycle
                        is CheckpointLifecycle.COMMITTED
                    )
                    or (
                        record.tombstoned
                        and stored.checkpoint.lifecycle
                        is CheckpointLifecycle.TOMBSTONED
                    )
                )
                if not authorized or not available:
                    raise ConversationAuthorizationError()
                assert stored is not None
                return LocalDeletionPreparation(
                    state=(
                        LocalDeletionState.TOMBSTONED
                        if record.tombstoned
                        else LocalDeletionState.ACTIVE
                    ),
                    checkpoint=stored.checkpoint,
                )
            terminal = next(
                (
                    item
                    for item in self._terminal.values()
                    if item.public_response_id == public_response_id
                ),
                None,
            )
            expected = (
                terminal.authority_digest
                if terminal is not None
                else self._CONCEALED_DIGEST
            )
            authorized = compare_digest(supplied, expected)
            deleted = (
                terminal is not None
                and terminal.state is CheckpointLifecycle.DELETED
            )
            if not authorized or not deleted:
                raise ConversationAuthorizationError()
            return LocalDeletionPreparation(
                state=LocalDeletionState.DELETED,
                checkpoint=None,
            )

    async def tombstone(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: datetime,
    ) -> ConversationCheckpoint:
        self._validate_time(at)
        await self._hook.reach(StoreAwaitBoundary.TOMBSTONE)
        async with self._lock:
            self._ensure_open_locked()
            record = self._authorized_public_locked(
                public_response_id, authority, allow_tombstone=False
            )
            stored = self._checkpoints[record.checkpoint_id]
            checkpoint = stored.checkpoint
            if checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED:
                raise ConversationTransitionError()
            timestamps = replace(checkpoint.timestamps, tombstoned_at=at)
            tombstone = with_checkpoint_integrity(
                replace(
                    checkpoint,
                    lifecycle=CheckpointLifecycle.TOMBSTONED,
                    timestamps=timestamps,
                )
            )
            self._checkpoints[record.checkpoint_id] = _StoredCheckpoint(
                checkpoint=tombstone,
                encoded=self._codec.encode(tombstone),
                authority_digest=stored.authority_digest,
            )
            self._public[public_response_id] = replace(record, tombstoned=True)
            self._results.pop(public_response_id, None)
            self._retire_outbox_for_checkpoint_locked(record.checkpoint_id)
            self._record_terminal_locked(
                tombstone.identity.checkpoint_id,
                public_response_id,
                stored.authority_digest,
                CheckpointLifecycle.TOMBSTONED,
                at,
            )
            self._enqueue_provider_lifecycle_locked(
                tombstone,
                ProviderLifecycleOrigin.LOCAL_TOMBSTONE,
            )
            return tombstone

    async def delete(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: datetime,
    ) -> None:
        self._validate_time(at)
        await self._hook.reach(StoreAwaitBoundary.DELETE)
        async with self._lock:
            self._ensure_open_locked()
            record = self._authorized_public_locked(
                public_response_id, authority, allow_tombstone=True
            )
            if not record.tombstoned:
                raise ConversationTransitionError()
            stored = self._checkpoints.get(record.checkpoint_id)
            if (
                stored is None
                or stored.checkpoint.lifecycle
                is not CheckpointLifecycle.TOMBSTONED
            ):
                raise ConversationAuthorizationError()
            if self._provider_lifecycle_pending_locked(record.checkpoint_id):
                raise ConversationTransitionError()
            del self._checkpoints[record.checkpoint_id]
            del self._public[public_response_id]
            self._results.pop(public_response_id, None)
            for head_key, head in tuple(self._heads.items()):
                if head.checkpoint_id == record.checkpoint_id:
                    self._heads[head_key] = replace(
                        head, lifecycle=NamedHeadLifecycle.TOMBSTONED
                    )
            self._retire_checkpoint_operational_locked(record.checkpoint_id)
            self._remove_checkpoint_graph_locked(record.checkpoint_id)
            self._record_terminal_locked(
                record.checkpoint_id,
                public_response_id,
                stored.authority_digest,
                CheckpointLifecycle.DELETED,
                at,
            )
            self._retire_provider_lifecycle_locked(record.checkpoint_id)

    async def list_checkpoints(
        self,
        authority: AuthorityScope,
        *,
        cursor: CheckpointId | None,
        limit: int,
    ) -> CheckpointPage:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if cursor is not None:
            validate_identifier(cursor, "cursor")
        if (
            type(limit) is not int
            or not 1 <= limit <= self._limits.max_page_size
        ):
            raise ConversationLimitError()
        await self._hook.reach(StoreAwaitBoundary.LIST)
        supplied = str(authority_digest(authority))
        async with self._lock:
            self._ensure_open_locked()
            ordered = tuple(
                key
                for key in sorted(self._checkpoints, key=str)
                if self._checkpoints[key].authority_digest == supplied
                and self._checkpoints[key].checkpoint.lifecycle
                is CheckpointLifecycle.COMMITTED
                and (cursor is None or str(key) > str(cursor))
            )
            selected = ordered[:limit]
            values = tuple(
                self._codec.decode(self._checkpoints[key].encoded)
                for key in selected
            )
            next_cursor = (
                selected[-1] if len(ordered) > len(selected) else None
            )
            return CheckpointPage(
                checkpoints=values,
                next_cursor=next_cursor,
            )

    async def sweep(self, now: datetime, *, limit: int) -> SweepReceipt:
        self._validate_time(now)
        if type(limit) is not int or limit <= 0:
            raise ConversationLimitError()
        await self._hook.reach(StoreAwaitBoundary.SWEEP)
        async with self._lock:
            self._ensure_open_locked()
            expired = 0
            deleted = 0
            preexisting_expired = tuple(
                key
                for key, stored in self._checkpoints.items()
                if stored.checkpoint.lifecycle is CheckpointLifecycle.EXPIRED
                and not self._provider_lifecycle_pending_locked(key)
            )
            for checkpoint_id in preexisting_expired[:limit]:
                self._remove_expired_locked(checkpoint_id, now)
                deleted += 1
            remaining = max(0, limit - deleted)
            eligible = tuple(
                (key, stored)
                for key, stored in self._checkpoints.items()
                if stored.checkpoint.lifecycle is CheckpointLifecycle.COMMITTED
                and stored.checkpoint.timestamps.expires_at is not None
                and stored.checkpoint.timestamps.expires_at <= now
            )
            for checkpoint_id, stored in eligible[:remaining]:
                checkpoint = with_checkpoint_integrity(
                    replace(
                        stored.checkpoint,
                        lifecycle=CheckpointLifecycle.EXPIRED,
                    )
                )
                self._checkpoints[checkpoint_id] = _StoredCheckpoint(
                    checkpoint=checkpoint,
                    encoded=self._codec.encode(checkpoint),
                    authority_digest=stored.authority_digest,
                )
                for public_id, record in tuple(self._public.items()):
                    if record.checkpoint_id == checkpoint_id:
                        self._public[public_id] = replace(
                            record, tombstoned=True
                        )
                        self._results.pop(public_id, None)
                self._retire_outbox_for_checkpoint_locked(checkpoint_id)
                self._record_terminal_locked(
                    checkpoint_id,
                    None,
                    stored.authority_digest,
                    CheckpointLifecycle.EXPIRED,
                    now,
                )
                self._enqueue_provider_lifecycle_locked(
                    checkpoint,
                    ProviderLifecycleOrigin.LOCAL_EXPIRY,
                )
                expired += 1
            return SweepReceipt(expired=expired, deleted=deleted)

    async def claim_provider_lifecycle(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> tuple[ProviderLifecycleWorkRecord, ...]:
        """Claim bounded provider lifecycle work for one authority."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if (
            type(limit) is not int
            or not 1 <= limit <= self._limits.max_page_size
        ):
            raise ConversationLimitError()
        now = await self._clock.now()
        self._validate_time(now)
        authority_key = str(authority_digest(authority))
        async with self._lock:
            self._ensure_open_locked()
            selected: list[ProviderLifecycleWorkRecord] = []
            for work_id, record in self._provider_lifecycle.items():
                stored = self._checkpoints.get(record.checkpoint_id)
                if (
                    stored is None
                    or stored.authority_digest != authority_key
                    or record.state
                    not in {
                        ProviderLifecycleWorkState.PENDING,
                        ProviderLifecycleWorkState.FAILED,
                    }
                    and not (
                        record.state is ProviderLifecycleWorkState.CLAIMED
                        and record.lease_expires_at is not None
                        and record.lease_expires_at <= now
                    )
                ):
                    continue
                claimed = replace(
                    record,
                    state=ProviderLifecycleWorkState.CLAIMED,
                    attempts=record.attempts + 1,
                    lease_owner=self._next_owner_token_locked(),
                    lease_expires_at=now
                    + timedelta(seconds=self._limits.outbox_lease_seconds),
                )
                self._provider_lifecycle[work_id] = claimed
                selected.append(claimed)
                if len(selected) == limit:
                    break
            return tuple(selected)

    async def acknowledge_provider_lifecycle(
        self,
        record: ProviderLifecycleWorkRecord,
        *,
        succeeded: bool,
    ) -> None:
        """Settle one exact provider lifecycle attempt."""
        if (
            type(record) is not ProviderLifecycleWorkRecord
            or record.state is not ProviderLifecycleWorkState.CLAIMED
            or record.lease_owner is None
            or type(succeeded) is not bool
        ):
            raise ConversationValidationError()
        async with self._lock:
            self._ensure_open_locked()
            current = self._provider_lifecycle.get(record.work_id)
            if current != record:
                raise ConversationConflictError()
            self._provider_lifecycle[record.work_id] = replace(
                record,
                state=(
                    ProviderLifecycleWorkState.COMPLETED
                    if succeeded
                    else ProviderLifecycleWorkState.FAILED
                ),
                lease_owner=None,
                lease_expires_at=None,
            )

    async def quarantine_provider_checkpoint(
        self,
        request: ProviderQuarantineRequest,
    ) -> ProviderQuarantineReceipt:
        """Persist one private cleanup checkpoint transactionally."""
        if type(request) is not ProviderQuarantineRequest:
            raise ConversationValidationError()
        candidates = (request.candidate, *request.additional_candidates)
        prepared = tuple(
            (
                committed,
                self._codec.encode(committed),
            )
            for candidate in candidates
            for committed in (
                self._committed_checkpoint(
                    self._candidate_checkpoint(candidate),
                    request.created_at,
                ),
            )
        )
        async with self._lock:
            self._ensure_open_locked()
            for committed, encoded in prepared:
                checkpoint_id = committed.identity.checkpoint_id
                existing = self._checkpoints.get(checkpoint_id)
                if existing is not None:
                    if existing.encoded != encoded:
                        raise ConversationConflictError()
                    continue
                self._validate_checkpoint_write_locked(
                    committed,
                    encoded,
                    enforce_capacity=False,
                )
            for committed, encoded in prepared:
                checkpoint_id = committed.identity.checkpoint_id
                if checkpoint_id in self._checkpoints:
                    continue
                self._checkpoints[checkpoint_id] = _StoredCheckpoint(
                    checkpoint=committed,
                    encoded=encoded,
                    authority_digest=str(
                        authority_digest(committed.authority)
                    ),
                )
                self._enqueue_provider_lifecycle_locked(
                    committed,
                    ProviderLifecycleOrigin.COMMIT_QUARANTINE,
                )
        return ProviderQuarantineReceipt(
            checkpoint_id=prepared[0][0].identity.checkpoint_id,
            target_count=len(prepared),
        )

    async def prune(self, now: datetime, *, limit: int) -> PruneReceipt:
        self._validate_time(now)
        if type(limit) is not int or limit <= 0:
            raise ConversationLimitError()
        await self._hook.reach(StoreAwaitBoundary.PRUNE)
        async with self._idempotency_changed:
            self._ensure_open_locked()
            outbox_ids = tuple(
                intent_id
                for intent_id, record in self._outbox.items()
                if record.state is OutboxState.PUBLISHED
                and record.published_at is not None
                and record.published_at <= now
            )[:limit]
            for intent_id in outbox_ids:
                del self._outbox[intent_id]
                self._outbox_ready_order.pop(intent_id, None)
            remaining = max(0, limit - len(outbox_ids))
            idempotency_keys = tuple(
                key
                for key, entry in self._idempotency.items()
                if entry.state is IdempotencyRecordState.FAILED_NO_DISPATCH
                or (
                    entry.state is IdempotencyRecordState.COMMITTED
                    and entry.checkpoint_id not in self._checkpoints
                )
            )[:remaining]
            for key in idempotency_keys:
                del self._idempotency[key]
            if idempotency_keys:
                self._idempotency_changed.notify_all()
            return PruneReceipt(
                outbox_records=len(outbox_ids),
                idempotency_records=len(idempotency_keys),
            )

    async def claim_outbox(
        self,
        target: OutboxClaimTarget,
    ) -> OutboxClaimResolution:
        if type(target) is not OutboxClaimTarget:
            raise ConversationValidationError()
        now = await self._clock.now()
        self._validate_time(now)
        await self._hook.reach(StoreAwaitBoundary.OUTBOX_CLAIM)
        async with self._lock:
            self._ensure_open_locked()
            record = self._outbox.get(target.intent_id)
            if not self._outbox_target_matches(record, target):
                return OutboxClaimResolution(
                    disposition=(
                        OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
                    )
                )
            assert record is not None
            if record.state is OutboxState.PUBLISHED:
                return OutboxClaimResolution(
                    disposition=OutboxClaimDisposition.ALREADY_PUBLISHED
                )
            if (
                record.state is OutboxState.CLAIMED
                and record.lease_expires_at is not None
                and record.lease_expires_at > now
            ):
                return OutboxClaimResolution(
                    disposition=OutboxClaimDisposition.ACTIVELY_LEASED
                )
            claimed = replace(
                record,
                state=OutboxState.CLAIMED,
                attempts=record.attempts + 1,
                lease_owner=self._next_owner_token_locked(),
                lease_expires_at=now
                + timedelta(seconds=self._limits.outbox_lease_seconds),
                published_at=None,
            )
            self._outbox[target.intent_id] = claimed
            return OutboxClaimResolution(
                disposition=OutboxClaimDisposition.CLAIMED,
                record=claimed,
            )

    def create_outbox_recovery_worker(
        self,
        authority: AuthorityScope,
    ) -> ConversationOutboxRecoveryWorker:
        """Create one trusted authority-isolated recovery worker."""
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        store = self

        @final
        class AuthorityOutboxRecoveryWorker:
            async def claim(self, *, limit: int) -> OutboxRecoveryBatch:
                return await store._claim_pending_outbox(
                    authority,
                    limit=limit,
                )

            async def acknowledge(self, record: OutboxRecord) -> None:
                target, owner_token = self._settlement(record)
                await store.acknowledge_outbox(target, owner_token)

            async def release(self, record: OutboxRecord) -> None:
                target, owner_token = self._settlement(record)
                await store.release_outbox(target, owner_token)

            @staticmethod
            def _settlement(
                record: OutboxRecord,
            ) -> tuple[OutboxClaimTarget, str]:
                if (
                    type(record) is not OutboxRecord
                    or record.state is not OutboxState.CLAIMED
                    or record.lease_owner is None
                ):
                    raise ConversationValidationError()
                return (
                    OutboxClaimTarget(
                        authority=authority,
                        checkpoint_id=record.intent.checkpoint_id,
                        public_response_id=record.intent.public_response_id,
                        intent_id=record.intent.intent_id,
                    ),
                    record.lease_owner,
                )

        return AuthorityOutboxRecoveryWorker()

    async def _claim_pending_outbox(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> OutboxRecoveryBatch:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if (
            type(limit) is not int
            or limit <= 0
            or limit > self._limits.max_page_size
        ):
            raise ConversationLimitError()
        now = await self._clock.now()
        self._validate_time(now)
        await self._hook.reach(StoreAwaitBoundary.OUTBOX_RECOVERY_CLAIM)
        authority_key = str(authority_digest(authority))
        async with self._lock:
            self._ensure_open_locked()
            self._rotate_expired_outbox_locked(authority_key, now)
            ordered = sorted(
                (
                    record
                    for record in self._outbox.values()
                    if compare_digest(
                        str(record.authority_digest),
                        authority_key,
                    )
                    and record.state is not OutboxState.PUBLISHED
                ),
                key=self._outbox_recovery_order_locked,
            )
            claimed: list[OutboxRecord] = []
            for record in ordered:
                if len(claimed) == limit:
                    break
                if (
                    record.state is OutboxState.CLAIMED
                    and record.lease_expires_at is not None
                    and record.lease_expires_at > now
                ):
                    continue
                recovered = replace(
                    record,
                    state=OutboxState.CLAIMED,
                    attempts=record.attempts + 1,
                    lease_owner=self._next_owner_token_locked(),
                    lease_expires_at=now
                    + timedelta(seconds=self._limits.outbox_lease_seconds),
                    published_at=None,
                )
                self._outbox[record.intent.intent_id] = recovered
                claimed.append(recovered)
            return OutboxRecoveryBatch(
                disposition=(
                    OutboxRecoveryDisposition.CLAIMED
                    if claimed
                    else OutboxRecoveryDisposition.EMPTY
                ),
                limit=limit,
                records=tuple(claimed),
            )

    async def acknowledge_outbox(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        if type(target) is not OutboxClaimTarget:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        at = await self._clock.now()
        self._validate_time(at)
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.OUTBOX_ACKNOWLEDGE)
        except CancelledError as exc:
            cancellation = exc
        async with self._lock:
            self._ensure_open_locked()
            record = self._outbox.get(target.intent_id)
            if (
                not self._outbox_target_matches(record, target)
                or record is None
            ):
                raise ConversationConflictError()
            if record.state is OutboxState.PUBLISHED:
                if cancellation is not None:
                    raise cancellation
                return
            if (
                record.state is not OutboxState.CLAIMED
                or record.lease_owner != owner_token
            ):
                raise ConversationConflictError()
            self._outbox[target.intent_id] = replace(
                record,
                state=OutboxState.PUBLISHED,
                lease_owner=None,
                lease_expires_at=None,
                published_at=at,
            )
            self._outbox_ready_order.pop(target.intent_id, None)
        if cancellation is not None:
            raise cancellation

    async def release_outbox(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        if type(target) is not OutboxClaimTarget:
            raise ConversationValidationError()
        validate_identifier(owner_token, "owner_token")
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.OUTBOX_RELEASE)
        except CancelledError as exc:
            cancellation = exc
        async with self._lock:
            self._ensure_open_locked()
            record = self._outbox.get(target.intent_id)
            if (
                not self._outbox_target_matches(record, target)
                or record is None
            ):
                raise ConversationConflictError()
            if record.state is OutboxState.PUBLISHED:
                if cancellation is not None:
                    raise cancellation
                return
            if (
                record.state is not OutboxState.CLAIMED
                or record.lease_owner != owner_token
            ):
                raise ConversationConflictError()
            self._outbox[target.intent_id] = replace(
                record,
                state=OutboxState.PENDING,
                lease_owner=None,
                lease_expires_at=None,
            )
            self._requeue_outbox_ready_locked(target.intent_id)
        if cancellation is not None:
            raise cancellation

    async def close(self) -> StoreCloseResolution:
        await self._hook.reach(StoreAwaitBoundary.CLOSE_BEGIN)
        cancellation: CancelledError | None = None
        try:
            await self._hook.reach(StoreAwaitBoundary.CLOSE)
        except CancelledError as exc:
            cancellation = exc
        async with self._idempotency_changed:
            if self._closed:
                resolution = StoreCloseResolution(
                    disposition=StoreCloseDisposition.CLOSED
                )
            else:
                self._closed = True
                self._checkpoints.clear()
                self._children.clear()
                self._provisional.clear()
                self._public.clear()
                self._results.clear()
                self._outputs.clear()
                self._idempotency.clear()
                self._execution_staging.clear()
                self._execution_stage_keys.clear()
                self._heads.clear()
                self._outbox.clear()
                self._outbox_ready_order.clear()
                self._terminal.clear()
                self._idempotency_changed.notify_all()
                resolution = StoreCloseResolution(
                    disposition=StoreCloseDisposition.CLOSED
                )
        try:
            await self._hook.reach(StoreAwaitBoundary.CLOSE_SETTLED)
        except CancelledError as exc:
            cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation
        return resolution

    async def inspect_close(self) -> StoreCloseResolution:
        await self._hook.reach(StoreAwaitBoundary.CLOSE_STATUS)
        async with self._lock:
            return StoreCloseResolution(
                disposition=(
                    StoreCloseDisposition.CLOSED
                    if self._closed
                    else StoreCloseDisposition.OPEN
                )
            )

    async def _authorized_checkpoint(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        validate_identifier(checkpoint_id, "checkpoint_id")
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        async with self._lock:
            self._ensure_open_locked()
            stored = self._authorize_entry_locked(checkpoint_id, authority)
            return self._codec.decode(stored.encoded)

    async def _commit_candidate(
        self, candidate: CheckpointCandidate
    ) -> ConversationCheckpoint:
        staged = self._candidate_checkpoint(candidate)
        committed_at = staged.timestamps.created_at
        committed = self._committed_checkpoint(staged, committed_at)
        encoded = self._codec.encode(committed)
        async with self._lock:
            self._ensure_open_locked()
            self._validate_checkpoint_write_locked(committed, encoded)
            checkpoint_id = committed.identity.checkpoint_id
            self._checkpoints[checkpoint_id] = _StoredCheckpoint(
                checkpoint=committed,
                encoded=encoded,
                authority_digest=str(authority_digest(committed.authority)),
            )
            self._register_child_locked(committed)
        return committed

    @staticmethod
    def _validate_atomic_commit_value(
        commit: AtomicConversationCommit,
    ) -> None:
        AtomicConversationCommit(
            candidate=commit.candidate,
            idempotency=commit.idempotency,
            owner_token=commit.owner_token,
            output_candidates=commit.output_candidates,
            committed_at=commit.committed_at,
            result_mode=commit.result_mode,
            execution_attestations=commit.execution_attestations,
            provisional_response_id=commit.provisional_response_id,
            public_response_id=commit.public_response_id,
            outbox_intent_id=commit.outbox_intent_id,
            head_id=commit.head_id,
            expected_head_revision=commit.expected_head_revision,
        )

    @staticmethod
    def _candidate_checkpoint(
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        if not isinstance(
            candidate,
            ExecutionSegmentCheckpointCandidate
            | SuspensionCheckpointCandidate
            | OutwardTurnCheckpointCandidate
            | StandaloneCompactCheckpointCandidate,
        ):
            raise ConversationValidationError()
        checkpoint = candidate.checkpoint
        if any(
            lane.binding.agent_id != checkpoint.authority.agent_id
            for lane in checkpoint.content.lanes
        ):
            raise ConversationAuthorizationError()
        return checkpoint

    @staticmethod
    def _committed_checkpoint(
        checkpoint: ConversationCheckpoint,
        at: datetime,
    ) -> ConversationCheckpoint:
        if checkpoint.lifecycle is not CheckpointLifecycle.STAGED:
            raise ConversationTransitionError()
        timestamps = replace(checkpoint.timestamps, committed_at=at)
        return with_checkpoint_integrity(
            replace(
                checkpoint,
                lifecycle=CheckpointLifecycle.COMMITTED,
                timestamps=timestamps,
            )
        )

    def _validate_checkpoint_write_locked(
        self,
        checkpoint: ConversationCheckpoint,
        encoded: bytes,
        *,
        enforce_capacity: bool = True,
    ) -> None:
        checkpoint_id = checkpoint.identity.checkpoint_id
        if checkpoint_id in self._checkpoints:
            raise ConversationConflictError()
        if enforce_capacity and str(checkpoint_id).startswith("quarantine-"):
            raise ConversationValidationError()
        ordinary_checkpoint_count = sum(
            not str(checkpoint_id).startswith("quarantine-")
            for checkpoint_id in self._checkpoints
        )
        if (
            enforce_capacity
            and ordinary_checkpoint_count >= self._limits.max_checkpoints
        ):
            raise ConversationLimitError()
        if len(encoded) > self._limits.max_checkpoint_bytes:
            raise ConversationLimitError()
        if checkpoint.identity.sequence > self._limits.max_depth:
            raise ConversationLimitError()
        counts = checkpoint.content.safe_counts
        if counts.provider_item_count > self._limits.max_provider_items:
            raise ConversationLimitError()
        parent_id = checkpoint.identity.parent_checkpoint_id
        if parent_id is not None:
            parent = self._checkpoints.get(parent_id)
            if (
                parent is None
                or parent.checkpoint.lifecycle
                is not CheckpointLifecycle.COMMITTED
                or parent.authority_digest
                != str(authority_digest(checkpoint.authority))
            ):
                raise ConversationAuthorizationError()
            children = self._children.get(parent_id, set())
            if len(children) >= self._limits.max_children_per_parent:
                raise ConversationLimitError()

    @staticmethod
    def _validate_output_candidates(
        checkpoint: ConversationCheckpoint,
        candidates: tuple[ProviderLaneOutputCandidate, ...],
        *,
        parent: ConversationCheckpoint | None = None,
    ) -> None:
        lanes = {lane.lane_id: lane for lane in checkpoint.content.lanes}
        candidate_ids = tuple(candidate.lane_id for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ConversationValidationError()
        parent_lanes = (
            {lane.lane_id: lane for lane in parent.content.lanes}
            if parent is not None
            else {}
        )
        for lane_id, checkpoint_lane in lanes.items():
            if lane_id not in candidate_ids and (
                parent_lanes.get(lane_id) != checkpoint_lane
            ):
                raise ConversationValidationError()
        for candidate in candidates:
            lane = lanes.get(candidate.lane_id)
            if (
                lane is None
                or lane.binding != candidate.binding
                or lane.reasoning != candidate.reasoning
                or lane.execution_receipt != candidate.execution_receipt
            ):
                raise ConversationValidationError()
            expected_receipt = provider_lane_execution_receipt(
                authority=checkpoint.authority,
                identity=checkpoint.identity,
                binding=candidate.binding,
                mode=candidate.mode,
                scope=candidate.scope,
                completed_items=candidate.completed_items,
                reasoning=candidate.reasoning,
                usage=candidate.usage,
                upstream_response_id=candidate.upstream_response_id,
            )
            if candidate.execution_receipt != expected_receipt:
                raise ConversationValidationError()
            prior = parent_lanes.get(candidate.lane_id)
            if prior is not None:
                prior.binding.assert_compatible(lane.binding)
            if isinstance(lane, StatelessProviderLaneSnapshot):
                if (
                    candidate.mode is not ConversationMode.STATELESS
                    or candidate.upstream_response_id is not None
                ):
                    raise ConversationValidationError()
                if candidate.scope is ProviderLaneOutputScope.CUMULATIVE:
                    expected_items = lane.ledger.items
                elif (
                    checkpoint.kind is CheckpointKind.STANDALONE_COMPACT_RESULT
                ):
                    expected_items = lane.ledger.items
                elif prior is None:
                    expected_items = lane.ledger.items
                elif isinstance(prior, StatelessProviderLaneSnapshot) and (
                    lane.ledger.items[: len(prior.ledger.items)]
                    == prior.ledger.items
                ):
                    expected_items = lane.ledger.items[
                        len(prior.ledger.items) :
                    ]
                else:
                    raise ConversationValidationError()
                if candidate.completed_items != expected_items:
                    raise ConversationValidationError()
            elif (
                candidate.mode is not ConversationMode.STORED
                or candidate.scope is not ProviderLaneOutputScope.CURRENT_CALL
                or candidate.upstream_response_id != lane.upstream_response_id
            ):
                raise ConversationValidationError()
            else:
                ProviderItemLedger(
                    lane_id=candidate.lane_id,
                    normalization_version=(
                        candidate.binding.continuation_codec_version
                    ),
                    items=candidate.completed_items,
                )

    def _validate_staged_executions_locked(
        self,
        commit: AtomicConversationCommit,
        checkpoint: ConversationCheckpoint,
        entry: _IdempotencyEntry,
    ) -> tuple[str, ...]:
        execution = entry.execution
        if execution is None or not commit.execution_attestations:
            raise ConversationConflictError()
        candidate_ids = {
            str(candidate.lane_id) for candidate in commit.output_candidates
        }
        reservation_ids = {lane.lane_id for lane in execution.lanes}
        attestation_ids = {
            str(attestation.lane_id)
            for attestation in commit.execution_attestations
        }
        if (
            candidate_ids != reservation_ids
            or attestation_ids != candidate_ids
            or execution.checkpoint_identity
            != self._checkpoint_identity_key(checkpoint.identity)
        ):
            raise ConversationConflictError()
        attestations = {
            str(attestation.lane_id): attestation
            for attestation in commit.execution_attestations
        }
        expected_key = self._idempotency_key(commit.idempotency)
        expected_authority = str(authority_digest(checkpoint.authority))
        staging_ids: list[str] = []
        for candidate in commit.output_candidates:
            attestation = attestations[str(candidate.lane_id)]
            record = self._execution_staging.get(attestation.staging_id)
            expected_receipt = provider_lane_execution_receipt(
                authority=checkpoint.authority,
                identity=checkpoint.identity,
                binding=candidate.binding,
                mode=candidate.mode,
                scope=candidate.scope,
                completed_items=candidate.completed_items,
                reasoning=candidate.reasoning,
                usage=candidate.usage,
                upstream_response_id=candidate.upstream_response_id,
            )
            stage_key = (
                commit.owner_token,
                str(checkpoint.identity.checkpoint_id),
                str(candidate.lane_id),
            )
            if (
                record is None
                or record.staging_id != attestation.staging_id
                or self._execution_stage_keys.get(stage_key)
                != attestation.staging_id
                or record.idempotency_key != expected_key
                or record.request_digest
                != str(commit.idempotency.request_digest)
                or not compare_digest(
                    record.authority_digest,
                    expected_authority,
                )
                or record.owner_token != commit.owner_token
                or record.checkpoint_identity != execution.checkpoint_identity
                or record.lane_id != str(candidate.lane_id)
                or record.binding_digest
                != str(candidate.binding.integrity_digest)
                or record.mode is not candidate.mode
                or record.scope is not candidate.scope
                or not compare_digest(
                    record.execution_digest,
                    str(expected_receipt.digest),
                )
                or record.item_count != expected_receipt.item_count
                or record.opaque_byte_count
                != expected_receipt.opaque_byte_count
            ):
                raise ConversationConflictError()
            staging_ids.append(record.staging_id)
        owned_staging_ids = {
            staging_id
            for staging_id, record in self._execution_staging.items()
            if record.owner_token == commit.owner_token
            and record.checkpoint_identity == execution.checkpoint_identity
        }
        if owned_staging_ids != set(staging_ids):
            raise ConversationConflictError()
        return tuple(staging_ids)

    def _consume_staged_executions_locked(
        self,
        staging_ids: tuple[str, ...],
    ) -> None:
        for staging_id in staging_ids:
            record = self._execution_staging.pop(staging_id)
            stage_key = (
                record.owner_token,
                record.checkpoint_identity[3],
                record.lane_id,
            )
            if self._execution_stage_keys.get(stage_key) != staging_id:
                raise ConversationStorageError()
            del self._execution_stage_keys[stage_key]

    def _validate_provisional_locked(
        self,
        commit: AtomicConversationCommit,
        authority_key: str,
    ) -> ProvisionalPublicResponse | None:
        if commit.provisional_response_id is None:
            return None
        provisional = self._provisional.get(commit.provisional_response_id)
        if (
            provisional is None
            or provisional.owner_token != commit.owner_token
            or provisional.public_response_id != commit.public_response_id
            or not compare_digest(provisional.authority_digest, authority_key)
        ):
            raise ConversationConflictError()
        assert commit.public_response_id is not None
        if commit.public_response_id in self._public:
            raise ConversationConflictError()
        if len(self._public) >= self._limits.max_public_responses:
            raise ConversationLimitError()
        return provisional

    def _validate_head_locked(
        self,
        commit: AtomicConversationCommit,
        checkpoint: ConversationCheckpoint,
    ) -> NamedHeadSnapshot | None:
        if commit.head_id is None:
            return None
        authority_key = str(authority_digest(checkpoint.authority))
        current = self._heads.get((authority_key, commit.head_id))
        assert commit.expected_head_revision is not None
        if (
            current is None
            or current.lifecycle is not NamedHeadLifecycle.ACTIVE
            or current.revision != commit.expected_head_revision
            or current.checkpoint_id
            != checkpoint.identity.parent_checkpoint_id
        ):
            raise ConversationConflictError()
        return NamedHeadSnapshot(
            head_id=current.head_id,
            revision=NamedHeadRevision(current.revision + 1),
            checkpoint_id=checkpoint.identity.checkpoint_id,
        )

    def _build_outbox_locked(
        self,
        commit: AtomicConversationCommit,
        checkpoint: ConversationCheckpoint,
        result: ConversationResult | None,
    ) -> OutboxRecord | None:
        if commit.outbox_intent_id is None:
            return None
        if len(self._outbox) >= self._limits.max_outbox_records:
            raise ConversationLimitError()
        if commit.outbox_intent_id in self._outbox:
            raise ConversationConflictError()
        assert commit.public_response_id is not None
        assert result is not None
        return OutboxRecord(
            intent=PublicationIntent(
                intent_id=commit.outbox_intent_id,
                public_response_id=commit.public_response_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                lane_outputs=result.lane_outputs,
            ),
            authority_digest=AuthorityDigest(
                str(authority_digest(checkpoint.authority))
            ),
            state=OutboxState.PENDING,
        )

    def _outbox_recovery_order_locked(
        self,
        record: OutboxRecord,
    ) -> tuple[int, str]:
        stored = self._checkpoints.get(record.intent.checkpoint_id)
        if stored is None:
            raise ConversationStorageError()
        committed_at = stored.checkpoint.timestamps.committed_at
        if committed_at is None:
            raise ConversationStorageError()
        ready_order = self._outbox_ready_order.get(record.intent.intent_id)
        if ready_order is None:
            raise ConversationStorageError()
        return ready_order, record.intent.intent_id

    def _append_outbox_ready_locked(self, intent_id: str) -> None:
        if intent_id in self._outbox_ready_order:
            raise ConversationStorageError()
        self._outbox_ready_sequence += 1
        self._outbox_ready_order[intent_id] = self._outbox_ready_sequence

    def _requeue_outbox_ready_locked(self, intent_id: str) -> None:
        record = self._outbox.get(intent_id)
        if record is None or record.state is OutboxState.PUBLISHED:
            raise ConversationStorageError()
        self._outbox_ready_sequence += 1
        self._outbox_ready_order[intent_id] = self._outbox_ready_sequence

    def _rotate_expired_outbox_locked(
        self,
        authority_key: str,
        now: datetime,
    ) -> None:
        expired = sorted(
            (
                record
                for record in self._outbox.values()
                if compare_digest(
                    str(record.authority_digest),
                    authority_key,
                )
                and record.state is OutboxState.CLAIMED
                and record.lease_expires_at is not None
                and record.lease_expires_at <= now
            ),
            key=self._outbox_recovery_order_locked,
        )
        for record in expired:
            self._requeue_outbox_ready_locked(record.intent.intent_id)

    @classmethod
    def _outbox_target_matches(
        cls,
        record: OutboxRecord | None,
        target: OutboxClaimTarget,
    ) -> bool:
        supplied = str(authority_digest(target.authority))
        expected = (
            str(record.authority_digest)
            if record is not None
            else cls._CONCEALED_DIGEST
        )
        authorized = compare_digest(supplied, expected)
        identity_matches = (
            record is not None
            and record.intent.intent_id == target.intent_id
            and record.intent.checkpoint_id == target.checkpoint_id
            and record.intent.public_response_id == target.public_response_id
        )
        return authorized and identity_matches

    @staticmethod
    def _build_result(
        commit: AtomicConversationCommit,
        checkpoint: ConversationCheckpoint,
    ) -> ConversationResult | None:
        if commit.public_response_id is None:
            return None
        lane = commit.output_candidates[-1]
        handle: ConversationHandle
        if commit.result_mode is ConversationMode.STATELESS:
            handle = StatelessConversationHandle(
                conversation_id=checkpoint.identity.conversation_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                branch_id=checkpoint.identity.branch_id,
            )
        else:
            handle = StoredConversationHandle(
                conversation_id=checkpoint.identity.conversation_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                branch_id=checkpoint.identity.branch_id,
                public_response_id=commit.public_response_id,
            )
        assert checkpoint.integrity is not None
        return ConversationResult(
            handle=handle,
            reasoning=lane.reasoning,
            checkpoint_digest=checkpoint.integrity.digest,
            lane_outputs=tuple(
                candidate.public_output
                for candidate in commit.output_candidates
            ),
            public_response_id=commit.public_response_id,
        )

    def _authorize_entry_locked(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> _StoredCheckpoint:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        supplied = str(authority_digest(authority))
        stored = self._checkpoints.get(checkpoint_id)
        expected = (
            stored.authority_digest
            if stored is not None
            else self._CONCEALED_DIGEST
        )
        authorized = compare_digest(supplied, expected)
        active = (
            stored is not None
            and stored.checkpoint.lifecycle is CheckpointLifecycle.COMMITTED
        )
        if not authorized or not active:
            raise ConversationAuthorizationError()
        assert stored is not None
        return stored

    def _authorized_public_locked(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        *,
        allow_tombstone: bool,
    ) -> PublicResponseRecord:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        supplied = str(authority_digest(authority))
        record = self._public.get(public_response_id)
        expected = (
            record.authority_digest
            if record is not None
            else self._CONCEALED_DIGEST
        )
        authorized = compare_digest(supplied, expected)
        available = record is not None and (
            allow_tombstone or not record.tombstoned
        )
        if not authorized or not available:
            raise ConversationAuthorizationError()
        assert record is not None
        return record

    def _register_child_locked(
        self, checkpoint: ConversationCheckpoint
    ) -> None:
        parent_id = checkpoint.identity.parent_checkpoint_id
        if parent_id is not None:
            self._children.setdefault(parent_id, set()).add(
                checkpoint.identity.checkpoint_id
            )

    def _record_terminal_locked(
        self,
        checkpoint_id: CheckpointId,
        public_response_id: PublicResponseId | None,
        authority_key: str,
        state: CheckpointLifecycle,
        at: datetime,
    ) -> None:
        while len(self._terminal) >= self._limits.max_terminal_metadata:
            oldest = next(iter(self._terminal))
            del self._terminal[oldest]
        self._terminal[checkpoint_id] = _TerminalMetadata(
            checkpoint_id=checkpoint_id,
            public_response_id=public_response_id,
            authority_digest=authority_key,
            state=state,
            at=at,
        )

    def _remove_expired_locked(
        self, checkpoint_id: CheckpointId, at: datetime
    ) -> None:
        stored = self._checkpoints.pop(checkpoint_id)
        for public_id, record in tuple(self._public.items()):
            if record.checkpoint_id == checkpoint_id:
                del self._public[public_id]
                self._results.pop(public_id, None)
        self._retire_checkpoint_operational_locked(checkpoint_id)
        self._remove_checkpoint_graph_locked(checkpoint_id)
        self._record_terminal_locked(
            checkpoint_id,
            None,
            stored.authority_digest,
            CheckpointLifecycle.DELETED,
            at,
        )
        self._retire_provider_lifecycle_locked(checkpoint_id)

    def _enqueue_provider_lifecycle_locked(
        self,
        checkpoint: ConversationCheckpoint,
        origin: ProviderLifecycleOrigin,
    ) -> None:
        for lane in checkpoint.content.lanes:
            if not isinstance(lane, StoredProviderLaneSnapshot):
                continue
            work_id = (
                f"provider-lifecycle-{checkpoint.identity.checkpoint_id}-"
                f"{lane.lane_id}"
            )
            if work_id in self._provider_lifecycle:
                continue
            self._provider_lifecycle[work_id] = ProviderLifecycleWorkRecord(
                work_id=work_id,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                lane_id=lane.lane_id,
                binding_digest=IntegrityDigest(lane.binding.integrity_digest),
                upstream_response_id=lane.upstream_response_id,
                origin=origin,
                state=ProviderLifecycleWorkState.PENDING,
                attempts=0,
            )

    def _provider_lifecycle_pending_locked(
        self,
        checkpoint_id: CheckpointId,
    ) -> bool:
        return any(
            record.checkpoint_id == checkpoint_id
            and record.state is not ProviderLifecycleWorkState.COMPLETED
            for record in self._provider_lifecycle.values()
        )

    def _retire_provider_lifecycle_locked(
        self,
        checkpoint_id: CheckpointId,
    ) -> None:
        for work_id, record in tuple(self._provider_lifecycle.items()):
            if record.checkpoint_id == checkpoint_id:
                del self._provider_lifecycle[work_id]

    def _retire_outbox_for_checkpoint_locked(
        self, checkpoint_id: CheckpointId
    ) -> None:
        stale = tuple(
            intent_id
            for intent_id, record in self._outbox.items()
            if record.intent.checkpoint_id == checkpoint_id
        )
        for intent_id in stale:
            del self._outbox[intent_id]
            self._outbox_ready_order.pop(intent_id, None)

    def _retire_checkpoint_operational_locked(
        self, checkpoint_id: CheckpointId
    ) -> None:
        self._outputs.pop(checkpoint_id, None)
        self._retire_outbox_for_checkpoint_locked(checkpoint_id)
        stale = tuple(
            key
            for key, entry in self._idempotency.items()
            if entry.checkpoint_id == checkpoint_id
            and entry.state is IdempotencyRecordState.COMMITTED
        )
        for key in stale:
            del self._idempotency[key]
        if stale:
            self._idempotency_changed.notify_all()

    def _remove_checkpoint_graph_locked(
        self, checkpoint_id: CheckpointId
    ) -> None:
        self._children.pop(checkpoint_id, None)
        for parent_id, children in tuple(self._children.items()):
            children.discard(checkpoint_id)
            if not children:
                del self._children[parent_id]

    def _remove_provisional_owner_locked(self, owner_token: str) -> None:
        stale = tuple(
            key
            for key, value in self._provisional.items()
            if value.owner_token == owner_token
        )
        for key in stale:
            del self._provisional[key]

    def _remove_staged_execution_owner_locked(self, owner_token: str) -> None:
        staging_ids = tuple(
            staging_id
            for staging_id, record in self._execution_staging.items()
            if record.owner_token == owner_token
        )
        for staging_id in staging_ids:
            record = self._execution_staging.pop(staging_id)
            stage_key = (
                record.owner_token,
                record.checkpoint_identity[3],
                record.lane_id,
            )
            if self._execution_stage_keys.get(stage_key) == staging_id:
                del self._execution_stage_keys[stage_key]

    @classmethod
    def _execution_reservation_record(
        cls,
        identity: RequestIdempotencyIdentity,
        execution: ConversationExecutionReservation | None,
    ) -> _ExecutionReservationRecord | None:
        if execution is None:
            return None
        if (
            type(execution) is not ConversationExecutionReservation
            or execution.idempotency != identity
        ):
            raise ConversationValidationError()
        return _ExecutionReservationRecord(
            checkpoint_identity=cls._checkpoint_identity_key(
                execution.identity
            ),
            lanes=tuple(
                cls._execution_lane_reservation_record(lane)
                for lane in execution.lanes
            ),
        )

    @staticmethod
    def _execution_lane_reservation_record(
        lane: ProviderLaneExecutionReservation,
    ) -> _ExecutionLaneReservationRecord:
        return _ExecutionLaneReservationRecord(
            lane_id=str(lane.binding.lane_id),
            binding_digest=str(lane.binding.integrity_digest),
            mode=lane.mode,
            scope=lane.scope,
        )

    @staticmethod
    def _checkpoint_identity_key(
        identity: CheckpointIdentity,
    ) -> tuple[
        str,
        str,
        str,
        str,
        str,
        int,
        str | None,
        int | None,
    ]:
        if type(identity) is not CheckpointIdentity:
            raise ConversationValidationError()
        return (
            str(identity.conversation_id),
            str(identity.logical_turn_id),
            str(identity.execution_segment_id),
            str(identity.checkpoint_id),
            str(identity.branch_id),
            int(identity.sequence),
            (
                str(identity.parent_checkpoint_id)
                if identity.parent_checkpoint_id is not None
                else None
            ),
            (
                int(identity.parent_sequence)
                if identity.parent_sequence is not None
                else None
            ),
        )

    def _idempotency_settlement_locked(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
    ) -> IdempotencySettlementResolution:
        key = self._idempotency_key(identity)
        current = self._idempotency.get(key)
        has_provisional = any(
            value.owner_token == owner_token
            for value in self._provisional.values()
        )
        if current is None:
            return IdempotencySettlementResolution(
                disposition=(
                    IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
                    if has_provisional
                    else IdempotencySettlementDisposition.SETTLED
                )
            )
        if current.identity != identity or current.owner_token != owner_token:
            return IdempotencySettlementResolution(
                disposition=(
                    IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
                )
            )
        if current.state is IdempotencyRecordState.IN_PROGRESS:
            return IdempotencySettlementResolution(
                disposition=IdempotencySettlementDisposition.LEASED,
                lease_expires_at=current.lease_expires_at,
                lease_owner_token=current.owner_token,
            )
        if has_provisional:
            return IdempotencySettlementResolution(
                disposition=(
                    IdempotencySettlementDisposition.OWNERSHIP_CONFLICT
                )
            )
        return IdempotencySettlementResolution(
            disposition=IdempotencySettlementDisposition.SETTLED
        )

    def _next_owner_token_locked(self) -> str:
        self._owner_sequence += 1
        owner_token = f"reservation-owner-{self._owner_sequence}"
        validate_identifier(owner_token, "owner_token")
        return owner_token

    def _next_execution_stage_id_locked(self) -> str:
        self._execution_stage_sequence += 1
        staging_id = f"execution-stage-{self._execution_stage_sequence}"
        validate_identifier(staging_id, "staging_id")
        return staging_id

    @staticmethod
    def _head_key(
        authority: AuthorityScope,
        head_id: NamedHeadId,
    ) -> tuple[str, NamedHeadId]:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        return str(authority_digest(authority)), head_id

    @staticmethod
    def _idempotency_key(
        identity: RequestIdempotencyIdentity,
    ) -> tuple[str, str, str]:
        return (
            str(authority_digest(identity.authority)),
            identity.operation.value,
            str(identity.key),
        )

    @staticmethod
    def _validate_time(value: datetime) -> None:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ConversationValidationError()

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise ConversationStorageError()
