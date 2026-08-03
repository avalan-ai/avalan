"""Define immutable checkpoints, lane snapshots, and pure reducers."""

from .binding import ProviderLaneBinding
from .contract import (
    LOCAL_DELETION_TRANSITIONS,
    UPSTREAM_DELETION_TRANSITIONS,
    AuthorityScope,
    CheckpointId,
    CheckpointIdentity,
    CheckpointKind,
    ChildLaneRetentionPolicy,
    LocalDeletionState,
    NamedHeadId,
    NamedHeadRevision,
    PortableContinuationReference,
    ProviderLaneId,
    PublicResponseId,
    ResponseResourceState,
    RetentionLimits,
    UpstreamDeletionState,
    UpstreamResponseId,
    response_transition_allowed,
)
from .errors import ConversationTransitionError, ConversationValidationError
from .execution import ProviderLaneExecutionReceipt
from .items import CompactionBoundary, ProviderItemLedger, VisibleTranscript
from .settings import EffectiveReasoningMetadata, StandaloneCompactHandle
from .value import (
    ConversationCodecVersion,
    IntegrityDigest,
    validate_identifier,
    validate_revision,
)

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, final


class CheckpointLifecycle(StrEnum):
    """Identify one immutable checkpoint lifecycle state."""

    STAGED = "staged"
    COMMITTED = "committed"
    QUARANTINED = "quarantined"
    TOMBSTONED = "tombstoned"
    EXPIRED = "expired"
    DELETED = "deleted"
    SUPERSEDED = "superseded"


class ProviderLaneLifecycle(StrEnum):
    """Identify one provider-lane snapshot lifecycle state."""

    STAGED = "staged"
    COMMITTED = "committed"
    SUSPENDED = "suspended"
    TOMBSTONED = "tombstoned"


class NamedHeadLifecycle(StrEnum):
    """Identify whether a named head can advance."""

    ACTIVE = "active"
    TOMBSTONED = "tombstoned"


CHECKPOINT_LIFECYCLE_TRANSITIONS: Mapping[
    CheckpointLifecycle,
    frozenset[CheckpointLifecycle],
] = MappingProxyType(
    {
        CheckpointLifecycle.STAGED: frozenset(
            {
                CheckpointLifecycle.COMMITTED,
                CheckpointLifecycle.QUARANTINED,
            }
        ),
        CheckpointLifecycle.COMMITTED: frozenset(
            {
                CheckpointLifecycle.TOMBSTONED,
                CheckpointLifecycle.EXPIRED,
                CheckpointLifecycle.SUPERSEDED,
            }
        ),
        CheckpointLifecycle.QUARANTINED: frozenset(
            {CheckpointLifecycle.TOMBSTONED}
        ),
        CheckpointLifecycle.TOMBSTONED: frozenset(
            {CheckpointLifecycle.DELETED}
        ),
        CheckpointLifecycle.EXPIRED: frozenset({CheckpointLifecycle.DELETED}),
        CheckpointLifecycle.SUPERSEDED: frozenset(
            {CheckpointLifecycle.DELETED}
        ),
        CheckpointLifecycle.DELETED: frozenset(),
    }
)

PROVIDER_LANE_TRANSITIONS: Mapping[
    ProviderLaneLifecycle,
    frozenset[ProviderLaneLifecycle],
] = MappingProxyType(
    {
        ProviderLaneLifecycle.STAGED: frozenset(
            {
                ProviderLaneLifecycle.COMMITTED,
                ProviderLaneLifecycle.SUSPENDED,
                ProviderLaneLifecycle.TOMBSTONED,
            }
        ),
        ProviderLaneLifecycle.COMMITTED: frozenset(
            {
                ProviderLaneLifecycle.SUSPENDED,
                ProviderLaneLifecycle.TOMBSTONED,
            }
        ),
        ProviderLaneLifecycle.SUSPENDED: frozenset(
            {
                ProviderLaneLifecycle.STAGED,
                ProviderLaneLifecycle.TOMBSTONED,
            }
        ),
        ProviderLaneLifecycle.TOMBSTONED: frozenset(),
    }
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointTimestamps:
    """Record ordered aware checkpoint lifecycle timestamps."""

    created_at: datetime
    committed_at: datetime | None = None
    expires_at: datetime | None = None
    tombstoned_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        for value in (
            self.created_at,
            self.committed_at,
            self.expires_at,
            self.tombstoned_at,
            self.deleted_at,
        ):
            if value is not None and (
                not isinstance(value, datetime) or value.utcoffset() is None
            ):
                raise ConversationValidationError()
        if (
            self.committed_at is not None
            and self.committed_at < self.created_at
        ):
            raise ConversationValidationError()
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ConversationValidationError()
        if (
            self.tombstoned_at is not None
            and self.tombstoned_at < self.created_at
        ):
            raise ConversationValidationError()
        if self.deleted_at is not None and self.deleted_at < self.created_at:
            raise ConversationValidationError()
        if (
            self.committed_at is not None
            and self.expires_at is not None
            and self.expires_at <= self.committed_at
        ):
            raise ConversationValidationError()
        if (
            self.committed_at is not None
            and self.tombstoned_at is not None
            and self.tombstoned_at < self.committed_at
        ):
            raise ConversationValidationError()
        if (
            self.committed_at is not None
            and self.deleted_at is not None
            and self.deleted_at < self.committed_at
        ):
            raise ConversationValidationError()
        if (
            self.tombstoned_at is not None
            and self.deleted_at is not None
            and self.deleted_at < self.tombstoned_at
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NamedHeadMetadata:
    """Record an optional exact named-head revision at checkpoint creation."""

    head_id: NamedHeadId
    revision: NamedHeadRevision

    def __post_init__(self) -> None:
        validate_identifier(self.head_id, "head_id")
        validate_revision(self.revision, "head revision")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointIntegrityMetadata:
    """Bind one encoded checkpoint to its codec, size, and exact digest."""

    codec_version: ConversationCodecVersion
    digest: IntegrityDigest
    encoded_byte_count: int

    def __post_init__(self) -> None:
        validate_revision(self.codec_version, "codec_version")
        if self.codec_version == 0:
            raise ConversationValidationError()
        validate_identifier(self.digest, "digest")
        if (
            type(self.encoded_byte_count) is not int
            or self.encoded_byte_count <= 0
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SafeCheckpointCounts:
    """Store only bounded content-free checkpoint counters."""

    lane_count: int
    provider_item_count: int
    transcript_entry_count: int
    opaque_byte_count: int

    def __post_init__(self) -> None:
        for value in (
            self.lane_count,
            self.provider_item_count,
            self.transcript_entry_count,
            self.opaque_byte_count,
        ):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessProviderLaneSnapshot:
    """Store one exact stateless ledger without an upstream response ID."""

    binding: ProviderLaneBinding
    ledger: ProviderItemLedger
    reasoning: EffectiveReasoningMetadata
    lifecycle: ProviderLaneLifecycle
    retention_policy: ChildLaneRetentionPolicy
    compaction_boundary: CompactionBoundary | None = None
    execution_receipt: ProviderLaneExecutionReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.binding) is not ProviderLaneBinding:
            raise ConversationValidationError()
        if type(self.ledger) is not ProviderItemLedger:
            raise ConversationValidationError()
        if self.binding.lane_id != self.ledger.lane_id:
            raise ConversationValidationError()
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        if not isinstance(
            self.lifecycle, ProviderLaneLifecycle
        ) or not isinstance(
            self.retention_policy,
            ChildLaneRetentionPolicy,
        ):
            raise ConversationValidationError()
        if self.compaction_boundary is not None:
            if type(self.compaction_boundary) is not CompactionBoundary:
                raise ConversationValidationError()
            self.compaction_boundary.validate_latest(self.ledger)
        if (
            self.execution_receipt is not None
            and type(self.execution_receipt)
            is not ProviderLaneExecutionReceipt
        ):
            raise ConversationValidationError()

    @property
    def lane_id(self) -> ProviderLaneId:
        """Return the bound provider lane identifier."""
        return self.binding.lane_id

    def __repr__(self) -> str:
        """Return a representation without private provider identifiers."""
        opaque_bytes = sum(
            item.opaque_state.byte_count
            for item in self.ledger.items
            if item.opaque_state is not None
        )
        return (
            "StatelessProviderLaneSnapshot("
            f"lane_id={self.lane_id!r}, lifecycle={self.lifecycle.value!r}, "
            f"provider_item_count={self.ledger.item_count}, "
            f"opaque_byte_count={opaque_bytes}, "
            "provider_items=<redacted>, "
            f"execution_receipt={self.execution_receipt!r})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StoredProviderLaneSnapshot:
    """Store one private upstream response ID without a stateless ledger."""

    binding: ProviderLaneBinding
    upstream_response_id: UpstreamResponseId
    reasoning: EffectiveReasoningMetadata
    lifecycle: ProviderLaneLifecycle
    retention_policy: ChildLaneRetentionPolicy
    execution_receipt: ProviderLaneExecutionReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.binding) is not ProviderLaneBinding:
            raise ConversationValidationError()
        validate_identifier(self.upstream_response_id, "upstream_response_id")
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        if not isinstance(
            self.lifecycle, ProviderLaneLifecycle
        ) or not isinstance(
            self.retention_policy,
            ChildLaneRetentionPolicy,
        ):
            raise ConversationValidationError()
        if (
            self.execution_receipt is not None
            and type(self.execution_receipt)
            is not ProviderLaneExecutionReceipt
        ):
            raise ConversationValidationError()

    @property
    def lane_id(self) -> ProviderLaneId:
        """Return the bound provider lane identifier."""
        return self.binding.lane_id

    def __repr__(self) -> str:
        """Return a representation without the private upstream ID."""
        return (
            "StoredProviderLaneSnapshot("
            f"lane_id={self.lane_id!r}, lifecycle={self.lifecycle.value!r}, "
            "upstream_response_id=<redacted>, "
            f"execution_receipt={self.execution_receipt!r})"
        )


ProviderLaneSnapshot: TypeAlias = (
    StatelessProviderLaneSnapshot | StoredProviderLaneSnapshot
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultiLaneCheckpointContent:
    """Keep shared visible transcript separate from bound provider lanes."""

    visible_transcript: VisibleTranscript
    lanes: tuple[ProviderLaneSnapshot, ...]

    def __post_init__(self) -> None:
        if type(self.visible_transcript) is not VisibleTranscript:
            raise ConversationValidationError()
        if type(self.lanes) is not tuple or not self.lanes:
            raise ConversationValidationError()
        lane_ids: list[ProviderLaneId] = []
        for lane in self.lanes:
            if not isinstance(
                lane,
                StatelessProviderLaneSnapshot | StoredProviderLaneSnapshot,
            ):
                raise ConversationValidationError()
            lane_ids.append(lane.lane_id)
        if len(lane_ids) != len(set(lane_ids)):
            raise ConversationValidationError()

    @property
    def safe_counts(self) -> SafeCheckpointCounts:
        """Return content-free aggregate counts for observability."""
        provider_items = 0
        opaque_bytes = 0
        for lane in self.lanes:
            if isinstance(lane, StatelessProviderLaneSnapshot):
                provider_items += lane.ledger.item_count
                opaque_bytes += sum(
                    item.opaque_state.byte_count
                    for item in lane.ledger.items
                    if item.opaque_state is not None
                )
        return SafeCheckpointCounts(
            lane_count=len(self.lanes),
            provider_item_count=provider_items,
            transcript_entry_count=self.visible_transcript.entry_count,
            opaque_byte_count=opaque_bytes,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCheckpoint:
    """Store one immutable authoritative conversation boundary."""

    identity: CheckpointIdentity
    kind: CheckpointKind
    lifecycle: CheckpointLifecycle
    authority: AuthorityScope
    content: MultiLaneCheckpointContent
    timestamps: CheckpointTimestamps
    retention: RetentionLimits
    head: NamedHeadMetadata | None = None
    integrity: CheckpointIntegrityMetadata | None = None

    def __post_init__(self) -> None:
        if type(self.identity) is not CheckpointIdentity:
            raise ConversationValidationError()
        if not isinstance(self.kind, CheckpointKind) or not isinstance(
            self.lifecycle,
            CheckpointLifecycle,
        ):
            raise ConversationValidationError()
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        if type(self.content) is not MultiLaneCheckpointContent:
            raise ConversationValidationError()
        if type(self.timestamps) is not CheckpointTimestamps:
            raise ConversationValidationError()
        if type(self.retention) is not RetentionLimits:
            raise ConversationValidationError()
        if self.head is not None and type(self.head) is not NamedHeadMetadata:
            raise ConversationValidationError()
        if (
            self.integrity is not None
            and type(self.integrity) is not CheckpointIntegrityMetadata
        ):
            raise ConversationValidationError()
        committed_lifecycles = {
            CheckpointLifecycle.COMMITTED,
            CheckpointLifecycle.TOMBSTONED,
            CheckpointLifecycle.EXPIRED,
            CheckpointLifecycle.DELETED,
            CheckpointLifecycle.SUPERSEDED,
        }
        if (self.lifecycle in committed_lifecycles) != (
            self.timestamps.committed_at is not None
        ):
            raise ConversationValidationError()
        if self.lifecycle is CheckpointLifecycle.TOMBSTONED and (
            self.timestamps.tombstoned_at is None
        ):
            raise ConversationValidationError()
        if (
            self.timestamps.tombstoned_at is not None
            and self.lifecycle
            not in {
                CheckpointLifecycle.TOMBSTONED,
                CheckpointLifecycle.DELETED,
            }
        ):
            raise ConversationValidationError()
        if self.lifecycle is CheckpointLifecycle.EXPIRED and (
            self.timestamps.expires_at is None
        ):
            raise ConversationValidationError()
        if (self.lifecycle is CheckpointLifecycle.DELETED) != (
            self.timestamps.deleted_at is not None
        ):
            raise ConversationValidationError()
        validate_upstream_identifier_separation(self)


def validate_upstream_identifier_separation(
    checkpoint: ConversationCheckpoint,
    *,
    additional_public_identifiers: tuple[str, ...] = (),
    additional_upstream_response_ids: tuple[str, ...] = (),
) -> None:
    """Reject private upstream IDs that alias Avalan public identifiers."""
    if (
        type(checkpoint) is not ConversationCheckpoint
        or type(additional_public_identifiers) is not tuple
        or type(additional_upstream_response_ids) is not tuple
    ):
        raise ConversationValidationError()
    identity = checkpoint.identity
    public_identifiers = {
        str(identity.conversation_id),
        str(identity.logical_turn_id),
        str(identity.execution_segment_id),
        str(identity.checkpoint_id),
        str(identity.branch_id),
        str(checkpoint.authority.principal_id),
        str(checkpoint.authority.agent_id),
        str(checkpoint.authority.endpoint_id),
    }
    if identity.parent_checkpoint_id is not None:
        public_identifiers.add(str(identity.parent_checkpoint_id))
    if checkpoint.authority.tenant_id is not None:
        public_identifiers.add(str(checkpoint.authority.tenant_id))
    if checkpoint.head is not None:
        public_identifiers.add(str(checkpoint.head.head_id))
    for lane in checkpoint.content.lanes:
        public_identifiers.add(str(lane.lane_id))
        public_identifiers.add(str(lane.binding.safe_alias))
    for identifier in additional_public_identifiers:
        validate_identifier(identifier, "public_identifier")
        public_identifiers.add(identifier)
    upstream_response_ids = {
        str(lane.upstream_response_id)
        for lane in checkpoint.content.lanes
        if isinstance(lane, StoredProviderLaneSnapshot)
    }
    for identifier in additional_upstream_response_ids:
        validate_identifier(identifier, "upstream_response_id")
        upstream_response_ids.add(identifier)
    if public_identifiers & upstream_response_ids:
        raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionSegmentCheckpointCandidate:
    """Stage a private completed provider execution segment."""

    checkpoint: ConversationCheckpoint

    def __post_init__(self) -> None:
        _validate_candidate(
            self.checkpoint,
            CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SuspensionCheckpointCandidate:
    """Stage a private structured-input suspension boundary."""

    checkpoint: ConversationCheckpoint
    continuation: PortableContinuationReference

    def __post_init__(self) -> None:
        _validate_candidate(
            self.checkpoint,
            CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
        )
        if type(self.continuation) is not PortableContinuationReference:
            raise ConversationValidationError()
        validate_upstream_identifier_separation(
            self.checkpoint,
            additional_public_identifiers=(
                str(self.continuation.continuation_id),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OutwardTurnCheckpointCandidate:
    """Stage an outward logical turn and its Avalan-owned public ID."""

    checkpoint: ConversationCheckpoint
    public_response_id: PublicResponseId

    def __post_init__(self) -> None:
        _validate_candidate(
            self.checkpoint,
            CheckpointKind.COMPLETED_OUTWARD_TURN,
        )
        validate_identifier(self.public_response_id, "public_response_id")
        validate_upstream_identifier_separation(
            self.checkpoint,
            additional_public_identifiers=(str(self.public_response_id),),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StandaloneCompactCheckpointCandidate:
    """Stage a canonical standalone compact result."""

    checkpoint: ConversationCheckpoint
    handle: StandaloneCompactHandle

    def __post_init__(self) -> None:
        _validate_candidate(
            self.checkpoint,
            CheckpointKind.STANDALONE_COMPACT_RESULT,
        )
        if (
            type(self.handle) is not StandaloneCompactHandle
            or self.handle.conversation_id
            != self.checkpoint.identity.conversation_id
            or self.handle.checkpoint_id
            != self.checkpoint.identity.checkpoint_id
            or self.handle.branch_id != self.checkpoint.identity.branch_id
            or self.handle.parent_checkpoint_id
            != self.checkpoint.identity.parent_checkpoint_id
        ):
            raise ConversationValidationError()


CheckpointCandidate: TypeAlias = (
    ExecutionSegmentCheckpointCandidate
    | SuspensionCheckpointCandidate
    | OutwardTurnCheckpointCandidate
    | StandaloneCompactCheckpointCandidate
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NamedHeadSnapshot:
    """Store one immutable named-head compare-and-swap snapshot."""

    head_id: NamedHeadId
    revision: NamedHeadRevision
    checkpoint_id: CheckpointId
    lifecycle: NamedHeadLifecycle = NamedHeadLifecycle.ACTIVE

    def __post_init__(self) -> None:
        validate_identifier(self.head_id, "head_id")
        validate_revision(self.revision, "revision")
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        if not isinstance(self.lifecycle, NamedHeadLifecycle):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DeletionSnapshot:
    """Keep local tombstone authority separate from upstream cleanup."""

    local: LocalDeletionState
    upstream: UpstreamDeletionState

    def __post_init__(self) -> None:
        if not isinstance(self.local, LocalDeletionState) or not isinstance(
            self.upstream,
            UpstreamDeletionState,
        ):
            raise ConversationValidationError()
        if self.local is LocalDeletionState.ACTIVE and self.upstream not in {
            UpstreamDeletionState.NOT_APPLICABLE,
            UpstreamDeletionState.PENDING,
        }:
            raise ConversationValidationError()


def reduce_checkpoint_lifecycle(
    source: CheckpointLifecycle,
    target: CheckpointLifecycle,
) -> CheckpointLifecycle:
    """Return a legal checkpoint lifecycle transition."""
    if (
        not isinstance(source, CheckpointLifecycle)
        or not isinstance(target, CheckpointLifecycle)
        or target not in CHECKPOINT_LIFECYCLE_TRANSITIONS[source]
    ):
        raise ConversationTransitionError()
    return target


def reduce_response_resource(
    source: ResponseResourceState,
    target: ResponseResourceState,
) -> ResponseResourceState:
    """Return a legal outward response-resource transition."""
    if (
        not isinstance(source, ResponseResourceState)
        or not isinstance(target, ResponseResourceState)
        or not response_transition_allowed(source, target)
    ):
        raise ConversationTransitionError()
    return target


def reduce_provider_lane(
    source: ProviderLaneLifecycle,
    target: ProviderLaneLifecycle,
) -> ProviderLaneLifecycle:
    """Return a legal provider-lane lifecycle transition."""
    if (
        not isinstance(source, ProviderLaneLifecycle)
        or not isinstance(target, ProviderLaneLifecycle)
        or target not in PROVIDER_LANE_TRANSITIONS[source]
    ):
        raise ConversationTransitionError()
    return target


def reduce_named_head(
    source: NamedHeadSnapshot,
    *,
    expected_revision: NamedHeadRevision,
    checkpoint_id: CheckpointId,
) -> NamedHeadSnapshot:
    """Advance one active named head using exact compare-and-swap semantics."""
    if type(source) is not NamedHeadSnapshot:
        raise ConversationValidationError()
    validate_revision(expected_revision, "expected_revision")
    validate_identifier(checkpoint_id, "checkpoint_id")
    if (
        source.lifecycle is not NamedHeadLifecycle.ACTIVE
        or expected_revision != source.revision
    ):
        raise ConversationTransitionError()
    return replace(
        source,
        revision=NamedHeadRevision(source.revision + 1),
        checkpoint_id=checkpoint_id,
    )


def reduce_deletion(
    source: DeletionSnapshot,
    *,
    local: LocalDeletionState | None = None,
    upstream: UpstreamDeletionState | None = None,
) -> DeletionSnapshot:
    """Return one legal local or upstream deletion transition."""
    if type(source) is not DeletionSnapshot:
        raise ConversationValidationError()
    if (local is None) == (upstream is None):
        raise ConversationTransitionError()
    if local is not None:
        if (
            not isinstance(local, LocalDeletionState)
            or local not in LOCAL_DELETION_TRANSITIONS[source.local]
        ):
            raise ConversationTransitionError()
        return replace(source, local=local)
    assert upstream is not None
    if (
        not isinstance(upstream, UpstreamDeletionState)
        or upstream not in UPSTREAM_DELETION_TRANSITIONS[source.upstream]
    ):
        raise ConversationTransitionError()
    return replace(source, upstream=upstream)


def _validate_candidate(
    checkpoint: ConversationCheckpoint,
    kind: CheckpointKind,
) -> None:
    if (
        type(checkpoint) is not ConversationCheckpoint
        or checkpoint.kind is not kind
        or checkpoint.lifecycle is not CheckpointLifecycle.STAGED
        or checkpoint.integrity is None
    ):
        raise ConversationValidationError()
