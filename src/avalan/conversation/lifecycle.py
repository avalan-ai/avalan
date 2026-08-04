"""Define private provider-stored lifecycle and reconciliation contracts."""

from .binding import ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CheckpointId,
    ConversationOperation,
    LocalDeletionState,
    ProviderLaneId,
    PublicResponseId,
    RequestIdempotencyKey,
    UpstreamLifetimeStatus,
    UpstreamResponseId,
)
from .errors import ConversationValidationError
from .settings import EffectiveReasoningContext
from .state import (
    CheckpointLifecycle,
    ConversationCheckpoint,
    ExecutionSegmentCheckpointCandidate,
    StoredProviderLaneSnapshot,
)
from .value import IntegrityDigest, validate_identifier

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, final, runtime_checkable


class UpstreamAvailability(StrEnum):
    """Identify what is known about one private upstream response."""

    AVAILABLE = "available"
    EXPIRED = "expired"
    DELETED = "deleted"
    UNKNOWN_UNAVAILABLE = "unknown_unavailable"


class UpstreamDeleteDisposition(StrEnum):
    """Identify an idempotent upstream deletion outcome."""

    DELETED = "deleted"
    ALREADY_ABSENT = "already_absent"


class ProviderLifecycleWorkState(StrEnum):
    """Identify one durable provider lifecycle outbox state."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


class ProviderLifecycleOrigin(StrEnum):
    """Identify why an upstream response requires lifecycle work."""

    LOCAL_TOMBSTONE = "local_tombstone"
    LOCAL_EXPIRY = "local_expiry"
    COMMIT_QUARANTINE = "commit_quarantine"


class AmbiguousDispatchResolution(StrEnum):
    """Identify an explicit operator decision for a fenced dispatch."""

    CONFIRMED_NO_DISPATCH = "confirmed_no_dispatch"
    RETAIN_FENCE = "retain_fence"


class AmbiguousDispatchReconciliationDisposition(StrEnum):
    """Identify the durable result of one explicit fence decision."""

    RESOLVED_NO_DISPATCH = "resolved_no_dispatch"
    ALREADY_RESOLVED_NO_DISPATCH = "already_resolved_no_dispatch"
    FENCE_RETAINED = "fence_retained"
    NOT_FOUND_OR_UNAUTHORIZED = "not_found_or_unauthorized"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AmbiguousDispatchReconciliationRequest:
    """Authorize one exact idempotency fence decision."""

    authority: AuthorityScope
    operation: ConversationOperation
    idempotency_key: RequestIdempotencyKey
    resolution: AmbiguousDispatchResolution

    def __post_init__(self) -> None:
        if (
            type(self.authority) is not AuthorityScope
            or not isinstance(self.operation, ConversationOperation)
            or not isinstance(self.resolution, AmbiguousDispatchResolution)
        ):
            raise ConversationValidationError()
        validate_identifier(self.idempotency_key, "idempotency_key")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AmbiguousDispatchReconciliationResult:
    """Report a content-free durable ambiguity transition."""

    disposition: AmbiguousDispatchReconciliationDisposition

    def __post_init__(self) -> None:
        if not isinstance(
            self.disposition,
            AmbiguousDispatchReconciliationDisposition,
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class UpstreamRetentionMetadata:
    """Record provider-reported retention without inventing a lifetime."""

    status: UpstreamLifetimeStatus
    expires_at: datetime | None = None
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, UpstreamLifetimeStatus):
            raise ConversationValidationError()
        if self.status is UpstreamLifetimeStatus.NOT_APPLICABLE:
            raise ConversationValidationError()
        if self.expires_at is not None and (
            not isinstance(self.expires_at, datetime)
            or self.expires_at.utcoffset() is None
        ):
            raise ConversationValidationError()
        if self.ttl_seconds is not None and (
            type(self.ttl_seconds) is not int or self.ttl_seconds <= 0
        ):
            raise ConversationValidationError()
        if self.status is UpstreamLifetimeStatus.UNKNOWN:
            if self.expires_at is not None or self.ttl_seconds is not None:
                raise ConversationValidationError()
        elif self.expires_at is None and self.ttl_seconds is None:
            raise ConversationValidationError()

    @classmethod
    def unknown(cls) -> "UpstreamRetentionMetadata":
        """Return an explicit typed unknown upstream lifetime."""
        return cls(status=UpstreamLifetimeStatus.UNKNOWN)


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RetrievedUpstreamResponse:
    """Carry one proven private upstream response retrieval result."""

    upstream_response_id: UpstreamResponseId
    availability: UpstreamAvailability
    retention: UpstreamRetentionMetadata
    binding_digest: IntegrityDigest | None = None
    execution_definition_digest: IntegrityDigest | None = None
    effective_reasoning_context: EffectiveReasoningContext | None = None

    def __post_init__(self) -> None:
        validate_identifier(
            self.upstream_response_id,
            "upstream_response_id",
        )
        if not isinstance(self.availability, UpstreamAvailability):
            raise ConversationValidationError()
        if type(self.retention) is not UpstreamRetentionMetadata:
            raise ConversationValidationError()
        for value, name in (
            (self.binding_digest, "binding_digest"),
            (
                self.execution_definition_digest,
                "execution_definition_digest",
            ),
        ):
            if value is not None:
                validate_identifier(value, name)
        if self.effective_reasoning_context is not None and not isinstance(
            self.effective_reasoning_context,
            EffectiveReasoningContext,
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return lifecycle metadata without the private upstream ID."""
        return (
            "RetrievedUpstreamResponse("
            "upstream_response_id=<redacted>, "
            f"availability={self.availability.value!r}, "
            f"retention={self.retention!r}, "
            f"binding_digest={self.binding_digest!r}, "
            "execution_definition_digest="
            f"{self.execution_definition_digest!r}, "
            "effective_reasoning_context="
            f"{self.effective_reasoning_context!r})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class UpstreamDeleteResult:
    """Return one content-free idempotent provider deletion result."""

    disposition: UpstreamDeleteDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, UpstreamDeleteDisposition):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLifecycleWorkRecord:
    """Carry one leased private upstream target from the durable outbox."""

    work_id: str
    checkpoint_id: CheckpointId
    lane_id: ProviderLaneId
    binding_digest: IntegrityDigest
    upstream_response_id: UpstreamResponseId
    origin: ProviderLifecycleOrigin
    state: ProviderLifecycleWorkState
    attempts: int
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.work_id, "work_id"),
            (self.checkpoint_id, "checkpoint_id"),
            (self.lane_id, "lane_id"),
            (self.binding_digest, "binding_digest"),
            (self.upstream_response_id, "upstream_response_id"),
        ):
            validate_identifier(value, name)
        if not isinstance(
            self.origin, ProviderLifecycleOrigin
        ) or not isinstance(
            self.state,
            ProviderLifecycleWorkState,
        ):
            raise ConversationValidationError()
        if type(self.attempts) is not int or self.attempts < 0:
            raise ConversationValidationError()
        claimed = self.state is ProviderLifecycleWorkState.CLAIMED
        if claimed != (
            self.lease_owner is not None and self.lease_expires_at is not None
        ):
            raise ConversationValidationError()
        if self.lease_owner is not None:
            validate_identifier(self.lease_owner, "lease_owner")
        if self.lease_expires_at is not None and (
            not isinstance(self.lease_expires_at, datetime)
            or self.lease_expires_at.utcoffset() is None
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return leased metadata without the private upstream target."""
        return (
            "ProviderLifecycleWorkRecord("
            f"work_id={self.work_id!r}, "
            f"checkpoint_id={self.checkpoint_id!r}, "
            f"lane_id={self.lane_id!r}, "
            f"binding_digest={self.binding_digest!r}, "
            f"origin={self.origin.value!r}, state={self.state.value!r}, "
            f"attempts={self.attempts}, upstream_response_id=<redacted>, "
            f"lease_owner={self.lease_owner!r}, "
            f"lease_expires_at={self.lease_expires_at!r})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderQuarantineReceipt:
    """Report a durable private quarantine without exposing provider IDs."""

    checkpoint_id: CheckpointId
    target_count: int

    def __post_init__(self) -> None:
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        if type(self.target_count) is not int or self.target_count <= 0:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderQuarantineRequest:
    """Request durable cleanup for completed but uncommitted provider state."""

    candidate: ExecutionSegmentCheckpointCandidate
    created_at: datetime
    additional_candidates: tuple[ExecutionSegmentCheckpointCandidate, ...] = ()

    def __post_init__(self) -> None:
        if type(self.candidate) is not ExecutionSegmentCheckpointCandidate:
            raise ConversationValidationError()
        if type(self.additional_candidates) is not tuple or any(
            type(candidate) is not ExecutionSegmentCheckpointCandidate
            for candidate in self.additional_candidates
        ):
            raise ConversationValidationError()
        candidates = (self.candidate, *self.additional_candidates)
        checkpoint_ids: list[CheckpointId] = []
        for candidate in candidates:
            checkpoint = candidate.checkpoint
            if (
                not str(checkpoint.identity.checkpoint_id).startswith(
                    "quarantine-"
                )
                or checkpoint.identity.parent_checkpoint_id is not None
                or checkpoint.identity.sequence != 0
                or len(checkpoint.content.lanes) != 1
                or not isinstance(
                    checkpoint.content.lanes[0],
                    StoredProviderLaneSnapshot,
                )
            ):
                raise ConversationValidationError()
            checkpoint_ids.append(checkpoint.identity.checkpoint_id)
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ConversationValidationError()
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.utcoffset() is None
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectDeletionResult:
    """Report local-first deletion without exposing private provider state."""

    public_response_id: PublicResponseId
    local_tombstoned: bool
    upstream_pending: bool

    def __post_init__(self) -> None:
        validate_identifier(self.public_response_id, "public_response_id")
        if (
            type(self.local_tombstoned) is not bool
            or type(self.upstream_pending) is not bool
            or not self.local_tombstoned
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LocalDeletionPreparation:
    """Carry only the local state needed for idempotent deletion."""

    state: LocalDeletionState
    checkpoint: ConversationCheckpoint | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, LocalDeletionState):
            raise ConversationValidationError()
        if self.state is LocalDeletionState.DELETED:
            if self.checkpoint is not None:
                raise ConversationValidationError()
            return
        if type(self.checkpoint) is not ConversationCheckpoint:
            raise ConversationValidationError()
        expected = (
            CheckpointLifecycle.COMMITTED
            if self.state is LocalDeletionState.ACTIVE
            else CheckpointLifecycle.TOMBSTONED
        )
        if self.checkpoint.lifecycle is not expected:
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return content-free local deletion metadata."""
        return (
            "LocalDeletionPreparation("
            f"state={self.state.value!r}, "
            f"checkpoint_available={self.checkpoint is not None!r})"
        )


class StoredResponseLifecycleAdapter(Protocol):
    """Retrieve and delete private provider responses asynchronously."""

    @property
    def binding(self) -> ProviderLaneBinding:
        """Return the exact immutable provider binding."""
        raise NotImplementedError

    async def retrieve(
        self,
        upstream_response_id: UpstreamResponseId,
    ) -> RetrievedUpstreamResponse:
        """Retrieve one proven private upstream response."""
        raise NotImplementedError

    async def delete(
        self,
        upstream_response_id: UpstreamResponseId,
    ) -> UpstreamDeleteResult:
        """Delete one proven private upstream response idempotently."""
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoredProviderResolverEntry:
    """Retain one exact adapter through an explicit rotation window."""

    adapter: StoredResponseLifecycleAdapter
    revision: str
    valid_from: datetime
    valid_until: datetime | None = None
    continuation_runtime: object | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.revision, "resolver_revision")
        if (
            not isinstance(self.valid_from, datetime)
            or self.valid_from.utcoffset() is None
            or self.valid_until is not None
            and (
                not isinstance(self.valid_until, datetime)
                or self.valid_until.utcoffset() is None
                or self.valid_until <= self.valid_from
            )
        ):
            raise ConversationValidationError()
        binding = getattr(self.adapter, "binding", None)
        retrieve = getattr(self.adapter, "retrieve", None)
        delete = getattr(self.adapter, "delete", None)
        if (
            type(binding) is not ProviderLaneBinding
            or not callable(retrieve)
            or not callable(delete)
        ):
            raise ConversationValidationError()
        if (
            self.continuation_runtime is not None
            and getattr(
                self.continuation_runtime,
                "binding",
                None,
            )
            != binding
        ):
            raise ConversationValidationError()


@final
class StoredProviderResolver:
    """Resolve current and retired exact adapters by private binding digest."""

    def __init__(
        self,
        entries: tuple[StoredProviderResolverEntry, ...],
        *,
        clock: Callable[[], Awaitable[datetime]],
    ) -> None:
        if type(entries) is not tuple or not entries or not callable(clock):
            raise ConversationValidationError()
        if any(
            type(entry) is not StoredProviderResolverEntry for entry in entries
        ):
            raise ConversationValidationError()
        digests = tuple(
            entry.adapter.binding.integrity_digest for entry in entries
        )
        if len(digests) != len(set(digests)):
            raise ConversationValidationError()
        self._entries = {
            entry.adapter.binding.integrity_digest: entry for entry in entries
        }
        self._clock = clock

    async def resolve(
        self,
        binding_digest: IntegrityDigest,
    ) -> StoredResponseLifecycleAdapter:
        """Return the exact adapter while its resolver window is valid."""
        validate_identifier(binding_digest, "binding_digest")
        now = await self._clock()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ConversationValidationError()
        entry = self._entries.get(binding_digest)
        if (
            entry is None
            or now < entry.valid_from
            or entry.valid_until is not None
            and now >= entry.valid_until
        ):
            raise ConversationValidationError()
        return entry.adapter

    async def resolve_continuation_runtime(
        self,
        binding_digest: IntegrityDigest,
    ) -> object:
        """Return the exact retired execution runtime in its valid window."""
        await self.resolve(binding_digest)
        entry = self._entries[binding_digest]
        if entry.continuation_runtime is None:
            raise ConversationValidationError()
        return entry.continuation_runtime


@final
class ProviderLifecycleReconciler:
    """Deliver provider deletion attempts through the durable store outbox."""

    def __init__(
        self,
        *,
        store: "ProviderLifecycleStore",
        resolver: StoredProviderResolver,
        authority: AuthorityScope,
    ) -> None:
        if (
            not isinstance(store, ProviderLifecycleStore)
            or type(resolver) is not StoredProviderResolver
            or type(authority) is not AuthorityScope
        ):
            raise ConversationValidationError()
        self._store = store
        self._resolver = resolver
        self._authority = authority

    def assert_runtime(
        self,
        *,
        store: "ProviderLifecycleStore",
        resolver: StoredProviderResolver,
        authority: AuthorityScope,
    ) -> None:
        """Reject a reconciler paired with different runtime authority."""
        if (
            store is not self._store
            or resolver is not self._resolver
            or authority != self._authority
        ):
            raise ConversationValidationError()

    async def run_once(self, *, limit: int) -> int:
        """Attempt and settle a bounded batch of lifecycle work."""
        if type(limit) is not int or limit <= 0:
            raise ConversationValidationError()
        records = await self._store.claim_provider_lifecycle(
            self._authority,
            limit=limit,
        )
        settled = 0
        for record in records:
            succeeded = False
            try:
                adapter = await self._resolver.resolve(record.binding_digest)
                result = await adapter.delete(record.upstream_response_id)
                succeeded = result.disposition in {
                    UpstreamDeleteDisposition.DELETED,
                    UpstreamDeleteDisposition.ALREADY_ABSENT,
                }
            except Exception:
                succeeded = False
            await self._store.acknowledge_provider_lifecycle(
                record,
                succeeded=succeeded,
            )
            settled += int(succeeded)
        return settled


@runtime_checkable
class ProviderLifecycleStore(Protocol):
    """Persist provider lifecycle attempts in one transactional outbox."""

    async def claim_provider_lifecycle(
        self,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> tuple[ProviderLifecycleWorkRecord, ...]:
        """Claim bounded provider lifecycle work for one authority."""
        raise NotImplementedError

    async def acknowledge_provider_lifecycle(
        self,
        record: ProviderLifecycleWorkRecord,
        *,
        succeeded: bool,
    ) -> None:
        """Settle one exact provider lifecycle attempt."""
        raise NotImplementedError

    async def quarantine_provider_checkpoint(
        self,
        request: ProviderQuarantineRequest,
    ) -> ProviderQuarantineReceipt:
        """Persist one private cleanup checkpoint transactionally."""
        raise NotImplementedError

    async def reconcile_ambiguous_dispatch(
        self,
        request: AmbiguousDispatchReconciliationRequest,
    ) -> AmbiguousDispatchReconciliationResult:
        """Apply one explicit durable ambiguity decision."""
        raise NotImplementedError
