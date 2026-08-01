"""Define closed runtime contracts for coordinated conversation execution."""

from ..types import JsonValue
from .binding import ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CanonicalRequestDigest,
    CheckpointId,
    CheckpointIdentity,
    ConversationBranchId,
    ConversationOperation,
    FailureBoundary,
    IdempotencyDisposition,
    NamedHeadId,
    NamedHeadRevision,
    ProviderLaneId,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyIdentity,
    RequestIdempotencyKey,
    RetentionLimits,
    RetryRule,
    UpstreamResponseId,
)
from .errors import ConversationValidationError
from .execution import (
    ProviderLaneExecutionAttestation,
    ProviderLaneExecutionReceipt,
)
from .items import ProviderItem, VisibleTranscriptEntry
from .observability import ConversationRequestSemantics
from .settings import (
    ConversationMode,
    ConversationResult,
    EffectiveReasoningMetadata,
    ProviderLaneOutput,
    ProviderLaneOutputScope,
    ProviderUsage,
    ReasoningContext,
)
from .state import (
    CheckpointCandidate,
    ConversationCheckpoint,
    ExecutionSegmentCheckpointCandidate,
    OutwardTurnCheckpointCandidate,
    StandaloneCompactCheckpointCandidate,
    StoredProviderLaneSnapshot,
    SuspensionCheckpointCandidate,
)
from .value import AuthorityDigest, validate_identifier, validate_revision

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias, final


class ConversationCommitBoundary(StrEnum):
    """Identify one valid coordinator commit boundary."""

    INTERNAL_SEGMENT = "internal_segment"
    OUTWARD_TURN = "outward_turn"


class CoordinatorAwaitBoundary(StrEnum):
    """Identify every injectable coordinator await boundary."""

    RESOLVE_AUTHORITY = "resolve_authority"
    RESERVE_IDEMPOTENCY = "reserve_idempotency"
    RESOLVE_PARENT = "resolve_parent"
    VALIDATE_PLAN = "validate_plan"
    ALLOCATE_RESPONSE = "allocate_response"
    PROVIDER_DISPATCH = "provider_dispatch"
    PROVIDER_STREAM_OPEN = "provider_stream_open"
    PROVIDER_STREAM_ITEM = "provider_stream_item"
    PROVIDER_STREAM_TERMINAL = "provider_stream_terminal"
    PROVIDER_STREAM_CLOSE = "provider_stream_close"
    STAGE_EXECUTION = "stage_execution"
    RETRY_WAIT = "retry_wait"
    COMMIT = "commit"
    OBSERVE = "observe"
    PUBLISH = "publish"
    ROLLBACK = "rollback"
    CLOSE = "close"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FirstTurnAdvance:
    """Start a new conversation without a parent checkpoint."""

    def __post_init__(self) -> None:
        return None


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OrdinaryChildAdvance:
    """Create an ordinary immutable child on its parent's branch."""

    parent_checkpoint_id: CheckpointId

    def __post_init__(self) -> None:
        validate_identifier(self.parent_checkpoint_id, "parent_checkpoint_id")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExplicitBranchAdvance:
    """Create an intentional child on a distinct branch."""

    parent_checkpoint_id: CheckpointId
    branch_id: ConversationBranchId

    def __post_init__(self) -> None:
        validate_identifier(self.parent_checkpoint_id, "parent_checkpoint_id")
        validate_identifier(self.branch_id, "branch_id")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NamedHeadAdvance:
    """Advance one named head from an exact parent and revision."""

    head_id: NamedHeadId
    parent_checkpoint_id: CheckpointId
    expected_revision: NamedHeadRevision

    def __post_init__(self) -> None:
        validate_identifier(self.head_id, "head_id")
        validate_identifier(self.parent_checkpoint_id, "parent_checkpoint_id")
        validate_revision(self.expected_revision, "expected_revision")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResetAdvance:
    """Authorize a new root while deliberately discarding prior continuity."""

    parent_checkpoint_id: CheckpointId

    def __post_init__(self) -> None:
        validate_identifier(self.parent_checkpoint_id, "parent_checkpoint_id")


ConversationAdvance: TypeAlias = (
    FirstTurnAdvance
    | OrdinaryChildAdvance
    | ExplicitBranchAdvance
    | NamedHeadAdvance
    | ResetAdvance
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationLaneRequest:
    """Select one exact lane mode and reasoning scope for a run."""

    lane_id: ProviderLaneId
    mode: ConversationMode
    reasoning_context: ReasoningContext = ReasoningContext.AUTO

    def __post_init__(self) -> None:
        validate_identifier(self.lane_id, "lane_id")
        if self.mode is ConversationMode.OFF or not isinstance(
            self.mode, ConversationMode
        ):
            raise ConversationValidationError()
        if not isinstance(self.reasoning_context, ReasoningContext):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationRunRequest:
    """Describe one fully typed run-scoped coordinator operation."""

    semantics: ConversationRequestSemantics
    identity: CheckpointIdentity
    advance: ConversationAdvance
    lanes: tuple[ConversationLaneRequest, ...]
    visible_delta: tuple[VisibleTranscriptEntry, ...]
    retention: RetentionLimits
    idempotency_key: RequestIdempotencyKey
    boundary: ConversationCommitBoundary
    provisional_response_id: ProvisionalResponseId | None = None
    public_response_id: PublicResponseId | None = None

    def __post_init__(self) -> None:
        if type(self.semantics) is not ConversationRequestSemantics:
            raise ConversationValidationError()
        if type(self.identity) is not CheckpointIdentity:
            raise ConversationValidationError()
        if not isinstance(
            self.advance,
            FirstTurnAdvance
            | OrdinaryChildAdvance
            | ExplicitBranchAdvance
            | NamedHeadAdvance
            | ResetAdvance,
        ):
            raise ConversationValidationError()
        if type(self.lanes) is not tuple or not self.lanes:
            raise ConversationValidationError()
        if any(
            type(item) is not ConversationLaneRequest for item in self.lanes
        ):
            raise ConversationValidationError()
        lane_ids = tuple(item.lane_id for item in self.lanes)
        if len(lane_ids) != len(set(lane_ids)):
            raise ConversationValidationError()
        if type(self.visible_delta) is not tuple or any(
            type(item) is not VisibleTranscriptEntry
            for item in self.visible_delta
        ):
            raise ConversationValidationError()
        if type(self.retention) is not RetentionLimits:
            raise ConversationValidationError()
        validate_identifier(self.idempotency_key, "idempotency_key")
        if not isinstance(self.boundary, ConversationCommitBoundary):
            raise ConversationValidationError()
        outward = self.boundary is ConversationCommitBoundary.OUTWARD_TURN
        if outward != (
            self.provisional_response_id is not None
            and self.public_response_id is not None
        ):
            raise ConversationValidationError()
        if self.provisional_response_id is not None:
            validate_identifier(
                self.provisional_response_id, "provisional_response_id"
            )
        if self.public_response_id is not None:
            validate_identifier(self.public_response_id, "public_response_id")
        _validate_advance_identity(self)


def _validate_advance_identity(request: ConversationRunRequest) -> None:
    identity = request.identity
    advance = request.advance
    if isinstance(advance, FirstTurnAdvance | ResetAdvance):
        if (
            identity.parent_checkpoint_id is not None
            or identity.parent_sequence is not None
            or identity.sequence != 0
        ):
            raise ConversationValidationError()
        if request.semantics.parent_checkpoint_id is not None and isinstance(
            advance, FirstTurnAdvance
        ):
            raise ConversationValidationError()
        if isinstance(advance, ResetAdvance) and (
            request.semantics.parent_checkpoint_id
            != advance.parent_checkpoint_id
        ):
            raise ConversationValidationError()
        return
    parent_id = advance.parent_checkpoint_id
    if (
        identity.parent_checkpoint_id != parent_id
        or identity.parent_sequence is None
        or request.semantics.parent_checkpoint_id != parent_id
    ):
        raise ConversationValidationError()
    if isinstance(advance, ExplicitBranchAdvance):
        if identity.branch_id != advance.branch_id:
            raise ConversationValidationError()


class OutboxState(StrEnum):
    """Identify one durable publication intent state."""

    PENDING = "pending"
    CLAIMED = "claimed"
    PUBLISHED = "published"


class OutboxClaimDisposition(StrEnum):
    """Describe one authority-bound targeted outbox claim decision."""

    CLAIMED = "claimed"
    ALREADY_PUBLISHED = "already_published"
    ACTIVELY_LEASED = "actively_leased"
    NOT_FOUND_OR_UNAUTHORIZED = "not_found_or_unauthorized"


class OutboxRecoveryDisposition(StrEnum):
    """Describe one bounded generic recovery scan decision."""

    CLAIMED = "claimed"
    EMPTY = "empty"


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneOutputCandidate:
    """Preserve one completed lane attempt without changing retained state."""

    lane_id: ProviderLaneId
    binding: ProviderLaneBinding
    mode: ConversationMode
    scope: ProviderLaneOutputScope
    completed_items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    usage: ProviderUsage
    execution_receipt: ProviderLaneExecutionReceipt
    upstream_response_id: UpstreamResponseId | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.lane_id, "lane_id")
        if (
            type(self.binding) is not ProviderLaneBinding
            or self.binding.lane_id != self.lane_id
        ):
            raise ConversationValidationError()
        if self.mode is ConversationMode.OFF or not isinstance(
            self.mode, ConversationMode
        ):
            raise ConversationValidationError()
        if not isinstance(self.scope, ProviderLaneOutputScope):
            raise ConversationValidationError()
        if (
            self.mode is ConversationMode.STORED
            and self.scope is not ProviderLaneOutputScope.CURRENT_CALL
        ):
            raise ConversationValidationError()
        if type(self.completed_items) is not tuple or any(
            type(item) is not ProviderItem for item in self.completed_items
        ):
            raise ConversationValidationError()
        if any(item.lane_id != self.lane_id for item in self.completed_items):
            raise ConversationValidationError()
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        if type(self.usage) is not ProviderUsage:
            raise ConversationValidationError()
        if type(self.execution_receipt) is not ProviderLaneExecutionReceipt:
            raise ConversationValidationError()
        stored = self.mode is ConversationMode.STORED
        if stored != (self.upstream_response_id is not None):
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id, "upstream_response_id"
            )

    @property
    def public_output(self) -> ProviderLaneOutput:
        """Return the outward-safe output without private continuation IDs."""
        return ProviderLaneOutput(
            lane_id=self.lane_id,
            binding_alias=self.binding.safe_alias,
            mode=self.mode,
            scope=self.scope,
            items=self.completed_items,
            reasoning=self.reasoning,
            usage=self.usage,
        )

    def __repr__(self) -> str:
        """Return a representation without the private upstream ID."""
        return (
            "ProviderLaneOutputCandidate("
            f"lane_id={self.lane_id!r}, binding_alias="
            f"{self.binding.safe_alias!r}, mode={self.mode.value!r}, "
            f"scope={self.scope.value!r}, "
            f"completed_items={len(self.completed_items)}, "
            f"usage={self.usage!r}, "
            f"execution_receipt={self.execution_receipt!r}, "
            "upstream_response_id=<redacted>)"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PublicationIntent:
    """Describe one content-safe idempotent outward publication."""

    intent_id: str
    public_response_id: PublicResponseId
    checkpoint_id: CheckpointId
    lane_outputs: tuple[ProviderLaneOutput, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.intent_id, "intent_id")
        validate_identifier(self.public_response_id, "public_response_id")
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        if type(self.lane_outputs) is not tuple or not self.lane_outputs:
            raise ConversationValidationError()
        if any(
            type(item) is not ProviderLaneOutput for item in self.lane_outputs
        ):
            raise ConversationValidationError()
        lane_ids = tuple(item.lane_id for item in self.lane_outputs)
        if len(lane_ids) != len(set(lane_ids)):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxRecord:
    """Record the immutable publication intent and current delivery state."""

    intent: PublicationIntent
    authority_digest: AuthorityDigest
    state: OutboxState
    attempts: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        if type(self.intent) is not PublicationIntent:
            raise ConversationValidationError()
        validate_identifier(self.authority_digest, "authority_digest")
        if not isinstance(self.state, OutboxState):
            raise ConversationValidationError()
        if type(self.attempts) is not int or self.attempts < 0:
            raise ConversationValidationError()
        if self.lease_owner is not None:
            validate_identifier(self.lease_owner, "lease_owner")
        for value in (self.lease_expires_at, self.published_at):
            if value is not None and (
                not isinstance(value, datetime) or value.utcoffset() is None
            ):
                raise ConversationValidationError()
        if self.state is OutboxState.PENDING and any(
            value is not None
            for value in (
                self.lease_owner,
                self.lease_expires_at,
                self.published_at,
            )
        ):
            raise ConversationValidationError()
        if self.state is OutboxState.CLAIMED and (
            self.lease_owner is None
            or self.lease_expires_at is None
            or self.published_at is not None
        ):
            raise ConversationValidationError()
        if self.state is OutboxState.PUBLISHED and (
            self.lease_owner is not None
            or self.lease_expires_at is not None
            or self.published_at is None
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxClaimTarget:
    """Bind one publication claim to trusted authority and exact IDs."""

    authority: AuthorityScope
    checkpoint_id: CheckpointId
    public_response_id: PublicResponseId
    intent_id: str

    def __post_init__(self) -> None:
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        for value, name in (
            (self.checkpoint_id, "checkpoint_id"),
            (self.public_response_id, "public_response_id"),
            (self.intent_id, "intent_id"),
        ):
            validate_identifier(value, name)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxClaimResolution:
    """Return one closed constant-disclosure outbox claim decision."""

    disposition: OutboxClaimDisposition
    record: OutboxRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, OutboxClaimDisposition):
            raise ConversationValidationError()
        claimed = self.disposition is OutboxClaimDisposition.CLAIMED
        if claimed != (self.record is not None):
            raise ConversationValidationError()
        if self.record is not None and (
            type(self.record) is not OutboxRecord
            or self.record.state is not OutboxState.CLAIMED
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxRecoveryBatch:
    """Return bounded authority-isolated publication work."""

    disposition: OutboxRecoveryDisposition
    limit: int
    records: tuple[OutboxRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, OutboxRecoveryDisposition):
            raise ConversationValidationError()
        if type(self.limit) is not int or self.limit <= 0:
            raise ConversationValidationError()
        if (
            type(self.records) is not tuple
            or len(self.records) > self.limit
            or any(
                type(record) is not OutboxRecord
                or record.state is not OutboxState.CLAIMED
                for record in self.records
            )
        ):
            raise ConversationValidationError()
        claimed = self.disposition is OutboxRecoveryDisposition.CLAIMED
        if claimed != bool(self.records):
            raise ConversationValidationError()
        intent_ids = tuple(record.intent.intent_id for record in self.records)
        if len(intent_ids) != len(set(intent_ids)):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProvisionalPublicResponse:
    """Keep an uncommitted public allocation private to one attempt."""

    provisional_response_id: ProvisionalResponseId
    public_response_id: PublicResponseId
    owner_token: str
    authority_digest: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.provisional_response_id, "provisional_response_id"),
            (self.public_response_id, "public_response_id"),
            (self.owner_token, "owner_token"),
            (self.authority_digest, "authority_digest"),
        ):
            validate_identifier(value, name)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PublicResponseRecord:
    """Map one committed public response to its authoritative checkpoint."""

    public_response_id: PublicResponseId
    checkpoint_id: CheckpointId
    authority_digest: str
    tombstoned: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.public_response_id, "public_response_id")
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        validate_identifier(self.authority_digest, "authority_digest")
        if type(self.tombstoned) is not bool:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoreLimits:
    """Bound every in-memory store allocation and traversal."""

    max_checkpoints: int = 1_024
    max_checkpoint_bytes: int = 8_388_608
    max_provider_items: int = 10_000
    max_depth: int = 256
    max_children_per_parent: int = 64
    max_in_flight: int = 128
    max_idempotency_records: int = 1_024
    max_provisional_responses: int = 128
    max_public_responses: int = 1_024
    max_heads: int = 1_024
    max_outbox_records: int = 1_024
    max_terminal_metadata: int = 1_024
    max_staged_execution_records: int = 4_096
    max_page_size: int = 100
    outbox_lease_seconds: int = 30
    idempotency_lease_seconds: int = 30

    def __post_init__(self) -> None:
        for value in (
            self.max_checkpoints,
            self.max_checkpoint_bytes,
            self.max_provider_items,
            self.max_depth,
            self.max_children_per_parent,
            self.max_in_flight,
            self.max_idempotency_records,
            self.max_provisional_responses,
            self.max_public_responses,
            self.max_heads,
            self.max_outbox_records,
            self.max_terminal_metadata,
            self.max_staged_execution_records,
            self.max_page_size,
            self.outbox_lease_seconds,
            self.idempotency_lease_seconds,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AtomicConversationCommit:
    """Commit checkpoint, mapping, head, idempotency, and outbox atomically."""

    candidate: CheckpointCandidate
    idempotency: RequestIdempotencyIdentity
    owner_token: str
    output_candidates: tuple[ProviderLaneOutputCandidate, ...]
    committed_at: datetime
    result_mode: ConversationMode
    execution_attestations: tuple[ProviderLaneExecutionAttestation, ...] = ()
    provisional_response_id: ProvisionalResponseId | None = None
    public_response_id: PublicResponseId | None = None
    outbox_intent_id: str | None = None
    head_id: NamedHeadId | None = None
    expected_head_revision: NamedHeadRevision | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.candidate,
            ExecutionSegmentCheckpointCandidate
            | SuspensionCheckpointCandidate
            | OutwardTurnCheckpointCandidate
            | StandaloneCompactCheckpointCandidate,
        ):
            raise ConversationValidationError()
        if type(self.idempotency) is not RequestIdempotencyIdentity:
            raise ConversationValidationError()
        validate_identifier(self.owner_token, "owner_token")
        if (
            type(self.output_candidates) is not tuple
            or not self.output_candidates
        ):
            raise ConversationValidationError()
        if any(
            type(item) is not ProviderLaneOutputCandidate
            for item in self.output_candidates
        ):
            raise ConversationValidationError()
        lane_ids = tuple(item.lane_id for item in self.output_candidates)
        if len(lane_ids) != len(set(lane_ids)):
            raise ConversationValidationError()
        if type(self.execution_attestations) is not tuple or any(
            type(item) is not ProviderLaneExecutionAttestation
            for item in self.execution_attestations
        ):
            raise ConversationValidationError()
        attested_lane_ids = tuple(
            item.lane_id for item in self.execution_attestations
        )
        if len(attested_lane_ids) != len(set(attested_lane_ids)) or (
            attested_lane_ids and set(attested_lane_ids) != set(lane_ids)
        ):
            raise ConversationValidationError()
        checkpoint_lane_ids = tuple(
            item.lane_id for item in self.candidate.checkpoint.content.lanes
        )
        if not set(lane_ids) <= set(checkpoint_lane_ids):
            raise ConversationValidationError()
        lanes = {
            item.lane_id: item
            for item in self.candidate.checkpoint.content.lanes
        }
        if any(
            lanes[item.lane_id].binding != item.binding
            for item in self.output_candidates
        ):
            raise ConversationValidationError()
        if any(
            lanes[item.lane_id].execution_receipt != item.execution_receipt
            for item in self.output_candidates
        ):
            raise ConversationValidationError()
        if self.candidate.checkpoint.authority != self.idempotency.authority:
            raise ConversationValidationError()
        if (
            not isinstance(self.committed_at, datetime)
            or self.committed_at.utcoffset() is None
        ):
            raise ConversationValidationError()
        if self.result_mode is ConversationMode.OFF or not isinstance(
            self.result_mode, ConversationMode
        ):
            raise ConversationValidationError()
        outward = self.public_response_id is not None
        if outward != (
            self.provisional_response_id is not None
            and self.outbox_intent_id is not None
        ):
            raise ConversationValidationError()
        candidate_outward = isinstance(
            self.candidate,
            OutwardTurnCheckpointCandidate,
        )
        if candidate_outward != outward:
            raise ConversationValidationError()
        if candidate_outward:
            assert isinstance(
                self.candidate,
                OutwardTurnCheckpointCandidate,
            )
            if self.candidate.public_response_id != self.public_response_id:
                raise ConversationValidationError()
        if self.provisional_response_id is not None:
            validate_identifier(
                self.provisional_response_id, "provisional_response_id"
            )
        if self.public_response_id is not None:
            validate_identifier(self.public_response_id, "public_response_id")
        if self.outbox_intent_id is not None:
            validate_identifier(self.outbox_intent_id, "outbox_intent_id")
        expected_result_mode = (
            ConversationMode.STORED
            if any(
                isinstance(item, StoredProviderLaneSnapshot)
                for item in self.candidate.checkpoint.content.lanes
            )
            else ConversationMode.STATELESS
        )
        if self.result_mode is not expected_result_mode:
            raise ConversationValidationError()
        if (self.head_id is None) != (self.expected_head_revision is None):
            raise ConversationValidationError()
        if self.head_id is not None:
            validate_identifier(self.head_id, "head_id")
            assert self.expected_head_revision is not None
            validate_revision(
                self.expected_head_revision, "expected_head_revision"
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AtomicCommitReceipt:
    """Return one authoritative commit and optional public result."""

    checkpoint: ConversationCheckpoint
    result: ConversationResult | None
    outbox: OutboxRecord | None
    output_candidates: tuple[ProviderLaneOutputCandidate, ...]

    def __post_init__(self) -> None:
        if type(self.checkpoint) is not ConversationCheckpoint:
            raise ConversationValidationError()
        if (
            self.result is not None
            and type(self.result) is not ConversationResult
        ):
            raise ConversationValidationError()
        if self.outbox is not None and type(self.outbox) is not OutboxRecord:
            raise ConversationValidationError()
        if (
            type(self.output_candidates) is not tuple
            or not self.output_candidates
        ):
            raise ConversationValidationError()
        if any(
            type(item) is not ProviderLaneOutputCandidate
            for item in self.output_candidates
        ):
            raise ConversationValidationError()
        public_outputs = tuple(
            candidate.public_output for candidate in self.output_candidates
        )
        if self.result is None:
            if self.outbox is not None:
                raise ConversationValidationError()
            return
        if self.result.handle.checkpoint_id != (
            self.checkpoint.identity.checkpoint_id
        ) or (
            self.result.lane_outputs != public_outputs
            or self.result.public_response_id is None
        ):
            raise ConversationValidationError()
        if self.outbox is None:
            return
        intent = self.outbox.intent
        if (
            self.result.public_response_id != intent.public_response_id
            or intent.checkpoint_id != self.checkpoint.identity.checkpoint_id
            or intent.lane_outputs != public_outputs
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class IdempotencyResolution:
    """Return the closed decision for one idempotency reservation."""

    disposition: IdempotencyDisposition
    owner_token: str | None = None
    checkpoint_id: CheckpointId | None = None
    public_response_id: PublicResponseId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, IdempotencyDisposition):
            raise ConversationValidationError()
        if self.checkpoint_id is not None:
            validate_identifier(self.checkpoint_id, "checkpoint_id")
        if self.public_response_id is not None:
            validate_identifier(self.public_response_id, "public_response_id")
        if self.owner_token is not None:
            validate_identifier(self.owner_token, "owner_token")
        replay = self.disposition is IdempotencyDisposition.REPLAY_COMMITTED
        if replay != (self.checkpoint_id is not None):
            raise ConversationValidationError()
        execute = self.disposition is IdempotencyDisposition.EXECUTE
        if execute != (self.owner_token is not None):
            raise ConversationValidationError()


class IdempotencySettlementDisposition(StrEnum):
    """Identify the exact post-cleanup state of one owned reservation."""

    SETTLED = "settled"
    LEASED = "leased"
    OWNERSHIP_CONFLICT = "ownership_conflict"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class IdempotencySettlementResolution:
    """Report whether cleanup settled or remains finitely reclaimable."""

    disposition: IdempotencySettlementDisposition
    lease_expires_at: datetime | None = None
    lease_owner_token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, IdempotencySettlementDisposition):
            raise ConversationValidationError()
        leased = self.disposition is IdempotencySettlementDisposition.LEASED
        has_lease = (
            self.lease_expires_at is not None
            and self.lease_owner_token is not None
        )
        if leased != has_lease or (
            (self.lease_expires_at is None) != (self.lease_owner_token is None)
        ):
            raise ConversationValidationError()
        if self.lease_expires_at is not None and (
            not isinstance(self.lease_expires_at, datetime)
            or self.lease_expires_at.utcoffset() is None
            or self.lease_expires_at.year >= datetime.max.year
        ):
            raise ConversationValidationError()
        if self.lease_owner_token is not None:
            validate_identifier(
                self.lease_owner_token,
                "lease_owner_token",
            )


class StoreCloseDisposition(StrEnum):
    """Identify whether an owned store is observably open or closed."""

    OPEN = "open"
    CLOSED = "closed"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoreCloseResolution:
    """Report the exact observable lifecycle state of one store."""

    disposition: StoreCloseDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, StoreCloseDisposition):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointPage:
    """Return one bounded checkpoint listing page."""

    checkpoints: tuple[ConversationCheckpoint, ...]
    next_cursor: CheckpointId | None

    def __post_init__(self) -> None:
        if type(self.checkpoints) is not tuple or any(
            type(item) is not ConversationCheckpoint
            for item in self.checkpoints
        ):
            raise ConversationValidationError()
        if self.next_cursor is not None:
            validate_identifier(self.next_cursor, "next_cursor")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SweepReceipt:
    """Report bounded content-free retention sweep counts."""

    expired: int
    deleted: int

    def __post_init__(self) -> None:
        if (
            type(self.expired) is not int
            or self.expired < 0
            or type(self.deleted) is not int
            or self.deleted < 0
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PruneReceipt:
    """Report bounded retirement of terminal operational metadata."""

    outbox_records: int
    idempotency_records: int

    def __post_init__(self) -> None:
        if (
            type(self.outbox_records) is not int
            or self.outbox_records < 0
            or type(self.idempotency_records) is not int
            or self.idempotency_records < 0
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FailureDisposition:
    """Reduce one failure to its deterministic retry and fencing policy."""

    boundary: FailureBoundary
    retry_rule: RetryRule
    fence_dispatch: bool
    preserve_parent: bool
    reconciliation_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, FailureBoundary) or not isinstance(
            self.retry_rule, RetryRule
        ):
            raise ConversationValidationError()
        for value in (
            self.fence_dispatch,
            self.preserve_parent,
            self.reconciliation_required,
        ):
            if type(value) is not bool:
                raise ConversationValidationError()


def request_operation(
    request: ConversationRunRequest,
) -> ConversationOperation:
    """Return the exact idempotency operation for one advance."""
    if isinstance(request.advance, FirstTurnAdvance | ResetAdvance):
        return ConversationOperation.CREATE
    if isinstance(request.advance, ExplicitBranchAdvance):
        return ConversationOperation.BRANCH
    return ConversationOperation.CONTINUE


def request_digest_value(value: JsonValue) -> CanonicalRequestDigest:
    """Narrow one validated canonical request digest string."""
    if type(value) is not str:
        raise ConversationValidationError()
    validate_identifier(value, "request_digest")
    return CanonicalRequestDigest(value)
