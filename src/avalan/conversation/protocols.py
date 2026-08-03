"""Define async-only coordinator, store, and provider protocols."""

from ..types import JsonValue
from .binding import ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CheckpointId,
    CheckpointIdentity,
    NamedHeadId,
    PublicResponseId,
    RequestIdempotencyIdentity,
    UpstreamResponseId,
)
from .errors import ConversationValidationError
from .execution import (
    ConversationExecutionReservation,
    DurableToolRecoveryAdmission,
    DurableToolRecoveryLease,
    ProviderLaneExecutionAttestation,
    ProviderLaneExecutionStage,
)
from .items import ProviderItem, ProviderItemLedger
from .lifecycle import (
    AmbiguousDispatchReconciliationRequest,
    AmbiguousDispatchReconciliationResult,
    LocalDeletionPreparation,
    ProviderQuarantineReceipt,
    ProviderQuarantineRequest,
)
from .observability import (
    ConversationObservation,
)
from .runtime import (
    AtomicCommitReceipt,
    AtomicConversationCommit,
    CheckpointPage,
    ConversationRunRequest,
    CoordinatorAwaitBoundary,
    IdempotencyResolution,
    IdempotencySettlementResolution,
    NamedHeadAdvance,
    OutboxClaimResolution,
    OutboxClaimTarget,
    OutboxRecord,
    OutboxRecoveryBatch,
    ProviderLaneOutputCandidate,
    ProvisionalPublicResponse,
    PruneReceipt,
    PublicationIntent,
    StoreCloseResolution,
    SweepReceipt,
)
from .settings import (
    CompactionPolicy,
    ConversationResult,
    DisabledCompaction,
    EffectiveReasoningMetadata,
    InlineCompaction,
    ProviderUsage,
)
from .state import (
    CheckpointCandidate,
    ConversationCheckpoint,
    NamedHeadSnapshot,
)
from .value import freeze_json_value, validate_identifier

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias, final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessProviderPlan:
    """Dispatch canonical stateless context without an upstream ID."""

    binding: ProviderLaneBinding
    ledger: ProviderItemLedger
    reasoning: EffectiveReasoningMetadata
    compaction: CompactionPolicy = DisabledCompaction()
    new_input: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.ledger) is not ProviderItemLedger
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or self.binding.lane_id != self.ledger.lane_id
            or not isinstance(
                self.compaction,
                DisabledCompaction | InlineCompaction,
            )
        ):
            raise ConversationValidationError()
        if self.new_input is not None:
            frozen = freeze_json_value(self.new_input)
            if not isinstance(frozen, Mapping):
                raise ConversationValidationError()
            object.__setattr__(self, "new_input", frozen)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StandaloneCompactProviderPlan:
    """Dispatch complete canonical context to a native compact operation."""

    binding: ProviderLaneBinding
    ledger: ProviderItemLedger
    reasoning: EffectiveReasoningMetadata

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.ledger) is not ProviderItemLedger
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or self.binding.lane_id != self.ledger.lane_id
            or not self.ledger.items
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StoredProviderPlan:
    """Dispatch new input using one private upstream response ID."""

    binding: ProviderLaneBinding
    upstream_response_id: UpstreamResponseId
    reasoning: EffectiveReasoningMetadata
    compaction: CompactionPolicy = DisabledCompaction()
    new_input: Mapping[str, JsonValue] | None = None
    model_call_index: int = 1
    item_order_offset: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or not isinstance(
                self.compaction,
                DisabledCompaction | InlineCompaction,
            )
            or type(self.model_call_index) is not int
            or self.model_call_index <= 0
            or type(self.item_order_offset) is not int
            or self.item_order_offset < 0
        ):
            raise ConversationValidationError()
        validate_identifier(self.upstream_response_id, "upstream_response_id")
        _freeze_stored_input(self)

    def __repr__(self) -> str:
        """Return plan metadata without private provider state or input."""
        return (
            "StoredProviderPlan("
            f"binding_alias={self.binding.safe_alias!r}, "
            "upstream_response_id=<redacted>, new_input=<redacted>, "
            f"model_call_index={self.model_call_index}, "
            f"item_order_offset={self.item_order_offset})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FirstStoredProviderPlan:
    """Dispatch the first provider-stored request without a previous ID."""

    binding: ProviderLaneBinding
    reasoning: EffectiveReasoningMetadata
    compaction: CompactionPolicy = DisabledCompaction()
    new_input: Mapping[str, JsonValue] | None = None
    model_call_index: int = 1
    item_order_offset: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or not isinstance(
                self.compaction,
                DisabledCompaction | InlineCompaction,
            )
            or type(self.model_call_index) is not int
            or self.model_call_index <= 0
            or type(self.item_order_offset) is not int
            or self.item_order_offset < 0
        ):
            raise ConversationValidationError()
        _freeze_stored_input(self)


def _freeze_stored_input(
    plan: FirstStoredProviderPlan | StoredProviderPlan,
) -> None:
    """Freeze one stored request input before provider dispatch."""
    if plan.new_input is None:
        return
    frozen = freeze_json_value(plan.new_input)
    if not isinstance(frozen, Mapping):
        raise ConversationValidationError()
    object.__setattr__(plan, "new_input", frozen)


ProviderPlan: TypeAlias = (
    StatelessProviderPlan
    | StandaloneCompactProviderPlan
    | FirstStoredProviderPlan
    | StoredProviderPlan
)


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderResult:
    """Return complete validated items and effective reasoning metadata."""

    items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    upstream_response_id: UpstreamResponseId | None = None
    usage: ProviderUsage = ProviderUsage()

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or any(
            type(item) is not ProviderItem for item in self.items
        ):
            raise ConversationValidationError()
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        if type(self.usage) is not ProviderUsage:
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id,
                "upstream_response_id",
            )

    def __repr__(self) -> str:
        """Return result metadata without private provider state."""
        return (
            "ProviderResult("
            f"item_count={len(self.items)}, reasoning={self.reasoning!r}, "
            f"usage={self.usage!r}, upstream_response_id=<redacted>)"
        )


class ConversationProviderStream(Protocol):
    """Yield complete provider items and close asynchronously."""

    def __aiter__(self) -> AsyncIterator[ProviderItem]:
        """Return the asynchronous item iterator."""
        ...

    async def terminal(self) -> ProviderResult:
        """Return validated terminal provider metadata."""
        ...

    async def aclose(self) -> None:
        """Close and await the owned provider stream."""
        ...


class ConversationProvider(Protocol):
    """Dispatch typed provider plans using asynchronous effects only."""

    async def dispatch(self, plan: ProviderPlan) -> ProviderResult:
        """Dispatch one non-streaming provider request."""
        ...

    async def stream(self, plan: ProviderPlan) -> ConversationProviderStream:
        """Open one owned asynchronous provider stream."""
        ...


class ConversationProviderStateSink(Protocol):
    """Privately stage provider items for one streamed logical turn."""

    async def stage(self, item: ProviderItem) -> None:
        """Stage one complete provider item without publishing it."""
        ...

    async def finalize(
        self,
        outputs: tuple[ProviderLaneOutputCandidate, ...],
    ) -> None:
        """Finalize complete private state before checkpoint commit."""
        ...

    async def cleanup(self) -> None:
        """Release the single-owner private staging sidecar."""
        ...


class CoordinatorBoundaryHook(Protocol):
    """Inject deterministic behavior before coordinator awaits."""

    async def reach(self, boundary: CoordinatorAwaitBoundary) -> None:
        """Reach one named boundary before its asynchronous effect."""
        ...


class ConversationOutboxRecoveryWorker(Protocol):
    """Recover pending publication work within one trusted authority."""

    async def claim(self, *, limit: int) -> OutboxRecoveryBatch:
        """Claim the oldest bounded set of currently available work."""
        ...

    async def acknowledge(self, record: OutboxRecord) -> None:
        """Acknowledge one exact worker-leased publication record."""
        ...

    async def release(self, record: OutboxRecord) -> None:
        """Release one exact worker-leased publication record."""
        ...


class ConversationStore(Protocol):
    """Persist and resolve immutable checkpoints asynchronously."""

    async def create(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        """Create one checkpoint without a public mapping."""
        ...

    async def create_with_named_head(
        self,
        candidate: CheckpointCandidate,
        advance: NamedHeadAdvance,
    ) -> ConversationCheckpoint:
        """Create one checkpoint and advance one named head atomically."""
        ...

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Load one authorized immutable checkpoint."""
        ...

    async def authorize(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Resolve one checkpoint using constant-disclosure authorization."""
        ...

    async def stage(
        self,
        candidate: CheckpointCandidate,
    ) -> "ConversationUnitOfWork":
        """Create one async unit of work for a validated candidate."""
        ...

    async def commit(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        """Commit one validated immutable checkpoint candidate."""
        ...

    async def commit_atomic(
        self,
        commit: AtomicConversationCommit,
    ) -> AtomicCommitReceipt:
        """Commit every authoritative outward artifact atomically."""
        ...

    async def quarantine_provider_checkpoint(
        self,
        request: ProviderQuarantineRequest,
    ) -> ProviderQuarantineReceipt:
        """Persist one private cleanup checkpoint transactionally."""
        ...

    async def reconcile_ambiguous_dispatch(
        self,
        request: AmbiguousDispatchReconciliationRequest,
    ) -> AmbiguousDispatchReconciliationResult:
        """Apply one explicit durable ambiguity decision."""
        ...

    async def stage_execution(
        self,
        stage: ProviderLaneExecutionStage,
    ) -> ProviderLaneExecutionAttestation:
        """Persist an owner-bound staging row for transactional consumption.

        A durable implementation stores the opaque identity and canonical
        result digest in a reservation-owned row, then consumes that row in
        the same transaction as the checkpoint commit.
        """
        ...

    async def create_head(
        self,
        head: NamedHeadSnapshot,
        authority: AuthorityScope,
    ) -> None:
        """Create one authority-scoped named head."""
        ...

    async def load_head(
        self,
        head_id: NamedHeadId,
        authority: AuthorityScope,
    ) -> NamedHeadSnapshot:
        """Load one authorized named-head snapshot."""
        ...

    async def branch_count(
        self,
        parent_checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> int:
        """Return the bounded committed child count for one parent."""
        ...

    async def reserve_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        *,
        execution: ConversationExecutionReservation | None = None,
    ) -> IdempotencyResolution:
        """Reserve, await, or replay one scoped idempotent operation."""
        ...

    async def admit_tool_recovery(
        self,
        admission: DurableToolRecoveryAdmission,
        execution: ConversationExecutionReservation,
    ) -> DurableToolRecoveryLease:
        """Atomically lease one exact fenced durable tool suffix."""
        ...

    async def fence_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> None:
        """Record an ambiguous fence or known no-dispatch failure."""
        ...

    async def abandon_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> IdempotencySettlementResolution:
        """Atomically clean and settle one owned failed reservation."""
        ...

    async def reconcile_idempotency(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
        *,
        ambiguous: bool,
    ) -> IdempotencySettlementResolution:
        """Idempotently reconcile a failed owned reservation."""
        ...

    async def inspect_idempotency_settlement(
        self,
        identity: RequestIdempotencyIdentity,
        owner_token: str,
    ) -> IdempotencySettlementResolution:
        """Inspect cleanup state without mutating the reservation."""
        ...

    async def allocate_public_response(
        self,
        allocation: ProvisionalPublicResponse,
    ) -> None:
        """Allocate a private provisional response for one attempt."""
        ...

    async def rollback_attempt(self, owner_token: str) -> None:
        """Remove every non-authoritative artifact owned by one reservation."""
        ...

    async def retrieve_output_candidates(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> tuple[ProviderLaneOutputCandidate, ...]:
        """Retrieve authorized typed output metadata for one commit."""
        ...

    async def retrieve(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> ConversationResult:
        """Retrieve one authorized committed public result."""
        ...

    async def prepare_deletion(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> LocalDeletionPreparation:
        """Resolve active, tombstoned, or deleted local state privately."""
        ...

    async def tombstone(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: "datetime",
    ) -> ConversationCheckpoint:
        """Atomically conceal one response and freeze its checkpoint."""
        ...

    async def delete(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: "datetime",
    ) -> None:
        """Delete tombstoned local content while retaining bounded metadata."""
        ...

    async def list_checkpoints(
        self,
        authority: AuthorityScope,
        *,
        cursor: CheckpointId | None,
        limit: int,
    ) -> CheckpointPage:
        """List one deterministic bounded authorized checkpoint page."""
        ...

    async def sweep(self, now: "datetime", *, limit: int) -> SweepReceipt:
        """Expire and delete a bounded set of eligible checkpoints."""
        ...

    async def prune(self, now: "datetime", *, limit: int) -> PruneReceipt:
        """Retire a bounded set of safe terminal operational records."""
        ...

    async def claim_outbox(
        self,
        target: OutboxClaimTarget,
    ) -> OutboxClaimResolution:
        """Resolve one authority-bound exact publication claim."""
        ...

    def create_outbox_recovery_worker(
        self,
        authority: AuthorityScope,
    ) -> ConversationOutboxRecoveryWorker:
        """Create one trusted authority-isolated recovery worker."""
        ...

    async def acknowledge_outbox(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        """Mark one owner-leased publication intent delivered."""
        ...

    async def release_outbox(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        """Return one owner-leased publication intent to pending state."""
        ...

    async def close(self) -> StoreCloseResolution:
        """Close and await every owned storage resource."""
        ...

    async def inspect_close(self) -> StoreCloseResolution:
        """Inspect whether every owned storage resource is closed."""
        ...


class ConversationCoordinator(Protocol):
    """Coordinate one typed conversation operation asynchronously."""

    async def execute(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        """Execute and commit one non-streaming fake-lane operation."""
        ...

    async def stream(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        """Execute and commit one streaming fake-lane operation."""
        ...

    async def stream_with_sink(
        self,
        request: ConversationRunRequest,
        sink: ConversationProviderStateSink,
    ) -> AtomicCommitReceipt:
        """Stream through one private sink and commit after finalization."""
        ...

    async def compact(
        self,
        request: ConversationRunRequest,
    ) -> AtomicCommitReceipt:
        """Execute one standalone fake-provider compaction."""
        ...

    async def commit_compact_result(
        self,
        source: ConversationCheckpoint,
        identity: CheckpointIdentity,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Commit one explicit continuable child of compact state."""
        ...


class ConversationObserver(Protocol):
    """Publish content-safe lifecycle observations asynchronously."""

    async def publish(self, observation: ConversationObservation) -> None:
        """Publish one content-safe post-transition observation."""
        ...


class ConversationAuthorityResolver(Protocol):
    """Resolve trusted run authority asynchronously."""

    async def resolve(self) -> AuthorityScope:
        """Return trusted authority for the current run."""
        ...


class ConversationClock(Protocol):
    """Read an aware coordinator time source asynchronously."""

    async def now(self) -> "datetime":
        """Return the current aware instant."""
        ...


class ConversationRetryWaiter(Protocol):
    """Wait between bounded effect-free retries asynchronously."""

    async def wait(self, attempt: int) -> None:
        """Wait before the numbered retry attempt."""
        ...


class ConversationPublisher(Protocol):
    """Publish one idempotent outward intent asynchronously."""

    async def publish(self, intent: PublicationIntent) -> None:
        """Publish one committed content-safe intent."""
        ...


class ConversationOutbox(Protocol):
    """Claim and settle publication intents asynchronously."""

    def create_outbox_recovery_worker(
        self,
        authority: AuthorityScope,
    ) -> ConversationOutboxRecoveryWorker:
        """Create one trusted authority-isolated recovery worker."""
        ...

    async def claim(
        self,
        target: OutboxClaimTarget,
    ) -> OutboxClaimResolution:
        """Resolve one authority-bound exact publication claim."""
        ...

    async def acknowledge(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        """Acknowledge one delivered owner-leased publication intent."""
        ...

    async def release(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        """Release one owner-leased failed publication intent for retry."""
        ...


class ConversationUnitOfWork(Protocol):
    """Stage and atomically commit one conversation transaction."""

    async def __aenter__(self) -> "ConversationUnitOfWork":
        """Enter the owned asynchronous transaction."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Roll back an uncommitted transaction and release resources."""
        ...

    async def commit(self) -> ConversationCheckpoint:
        """Commit the staged candidate exactly once."""
        ...

    async def rollback(self) -> None:
        """Discard the staged candidate idempotently."""
        ...
