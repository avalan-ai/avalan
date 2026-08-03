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
    ConversationAgentId,
    LocalDeletionState,
    NamedHeadId,
    NamedHeadRevision,
    PortableContinuationReference,
    ProviderLaneId,
    ProviderLaneOwnerKind,
    PublicResponseId,
    ResponseResourceState,
    RetentionLimits,
    UpstreamDeletionState,
    UpstreamResponseId,
    response_transition_allowed,
)
from .errors import ConversationTransitionError, ConversationValidationError
from .execution import ProviderExecutionSegment, ProviderLaneExecutionReceipt
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
class ProviderLaneTopologyEntry:
    """Persist one exact agent/model lane ownership relationship."""

    lane_id: ProviderLaneId
    owner_kind: ProviderLaneOwnerKind
    agent_id: ConversationAgentId
    topology_path: str
    model_slot: str
    retention_policy: ChildLaneRetentionPolicy
    binding_digest: IntegrityDigest
    parent_lane_id: ProviderLaneId | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.lane_id, "lane_id")
        validate_identifier(self.agent_id, "agent_id")
        validate_identifier(self.topology_path, "topology_path")
        validate_identifier(self.model_slot, "model_slot")
        if (
            not isinstance(self.owner_kind, ProviderLaneOwnerKind)
            or not isinstance(
                self.retention_policy,
                ChildLaneRetentionPolicy,
            )
            or type(self.binding_digest) is not str
            or len(self.binding_digest) != 64
        ):
            raise ConversationValidationError()
        try:
            int(self.binding_digest, 16)
        except ValueError as exc:
            raise ConversationValidationError() from exc
        child = self.owner_kind is ProviderLaneOwnerKind.CHILD_AGENT
        if child != (self.parent_lane_id is not None):
            raise ConversationValidationError()
        if self.parent_lane_id is not None:
            validate_identifier(self.parent_lane_id, "parent_lane_id")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderLaneTopology:
    """Persist deterministic lane topology without provider-private state."""

    schema_version: int
    entries: tuple[ProviderLaneTopologyEntry, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.entries) is not tuple
            or not self.entries
            or any(
                type(entry) is not ProviderLaneTopologyEntry
                for entry in self.entries
            )
        ):
            raise ConversationValidationError()
        lane_ids = tuple(entry.lane_id for entry in self.entries)
        paths = tuple(entry.topology_path for entry in self.entries)
        if len(lane_ids) != len(set(lane_ids)) or len(paths) != len(
            set(paths)
        ):
            raise ConversationValidationError()
        by_id = {entry.lane_id: entry for entry in self.entries}
        for entry in self.entries:
            if entry.parent_lane_id is None:
                continue
            parent = by_id.get(entry.parent_lane_id)
            if (
                parent is None
                or parent.owner_kind is not ProviderLaneOwnerKind.PARENT_AGENT
                or not entry.topology_path.startswith(
                    f"{parent.topology_path}/child/"
                )
            ):
                raise ConversationValidationError()

    @property
    def lane_ids(self) -> frozenset[ProviderLaneId]:
        """Return every deterministic lane in this topology."""
        return frozenset(entry.lane_id for entry in self.entries)

    @property
    def agent_ids(self) -> frozenset[ConversationAgentId]:
        """Return every agent authorized by this exact topology."""
        return frozenset(entry.agent_id for entry in self.entries)

    def entry(self, lane_id: ProviderLaneId) -> ProviderLaneTopologyEntry:
        """Return one exact lane entry or reject the missing lane."""
        validate_identifier(lane_id, "lane_id")
        for entry in self.entries:
            if entry.lane_id == lane_id:
                return entry
        raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultiLaneCheckpointContent:
    """Keep shared visible transcript separate from bound provider lanes."""

    visible_transcript: VisibleTranscript
    lanes: tuple[ProviderLaneSnapshot, ...]
    execution_segments: tuple[ProviderExecutionSegment, ...] = ()
    lane_topology: ProviderLaneTopology | None = None

    def __post_init__(self) -> None:
        if type(self.visible_transcript) is not VisibleTranscript:
            raise ConversationValidationError()
        if (
            type(self.lanes) is not tuple
            or type(self.execution_segments) is not tuple
            or (not self.lanes and not self.execution_segments)
            or (
                self.lane_topology is not None
                and type(self.lane_topology) is not ProviderLaneTopology
            )
        ):
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
        if any(
            type(segment) is not ProviderExecutionSegment
            for segment in self.execution_segments
        ):
            raise ConversationValidationError()
        segment_keys = tuple(
            (segment.lane_id, segment.segment_index, segment.phase)
            for segment in self.execution_segments
        )
        if len(segment_keys) != len(set(segment_keys)):
            raise ConversationValidationError()
        if self.lane_topology is not None:
            represented = {
                *lane_ids,
                *(segment.lane_id for segment in self.execution_segments),
            }
            if not represented <= self.lane_topology.lane_ids:
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
        for segment in self.execution_segments:
            provider_items += len(segment.items)
            opaque_bytes += sum(
                item.opaque_state.byte_count
                for item in segment.items
                if item.opaque_state is not None
            )
        return SafeCheckpointCounts(
            lane_count=len(
                {
                    *(lane.lane_id for lane in self.lanes),
                    *(segment.lane_id for segment in self.execution_segments),
                }
            ),
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
        if self.content.execution_segments and self.kind not in {
            CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            CheckpointKind.COMPLETED_OUTWARD_TURN,
            CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
        }:
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
class SuspensionContinuationCheckpointCandidate:
    """Stage an outward child through the structured-input resume fence."""

    checkpoint: ConversationCheckpoint
    public_response_id: PublicResponseId
    suspension_checkpoint_id: CheckpointId

    def __post_init__(self) -> None:
        _validate_candidate(
            self.checkpoint,
            CheckpointKind.COMPLETED_OUTWARD_TURN,
        )
        validate_identifier(self.public_response_id, "public_response_id")
        validate_identifier(
            self.suspension_checkpoint_id,
            "suspension_checkpoint_id",
        )
        if (
            self.checkpoint.identity.parent_checkpoint_id
            != self.suspension_checkpoint_id
        ):
            raise ConversationValidationError()
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
    | SuspensionContinuationCheckpointCandidate
    | StandaloneCompactCheckpointCandidate
)


def validate_checkpoint_parent_kind(
    child_kind: CheckpointKind,
    parent_kind: CheckpointKind | None,
    *,
    suspension_continuation: bool = False,
    compact_continuation: bool = False,
) -> None:
    """Validate one closed checkpoint parent-kind transition."""
    if not isinstance(child_kind, CheckpointKind) or (
        parent_kind is not None and not isinstance(parent_kind, CheckpointKind)
    ):
        raise ConversationValidationError()
    allowed: dict[CheckpointKind, frozenset[CheckpointKind | None]] = {
        CheckpointKind.COMPLETED_OUTWARD_TURN: frozenset(
            {
                None,
                CheckpointKind.COMPLETED_OUTWARD_TURN,
            }
        ),
        CheckpointKind.STANDALONE_COMPACT_RESULT: frozenset(
            {CheckpointKind.COMPLETED_OUTWARD_TURN}
        ),
        CheckpointKind.INTERNAL_PROVIDER_BOUNDARY: frozenset(
            {
                None,
                CheckpointKind.COMPLETED_OUTWARD_TURN,
                CheckpointKind.STANDALONE_COMPACT_RESULT,
            }
        ),
        CheckpointKind.STRUCTURED_INPUT_SUSPENSION: frozenset(
            {
                None,
                CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            }
        ),
    }
    if suspension_continuation:
        if (
            child_kind is CheckpointKind.COMPLETED_OUTWARD_TURN
            and parent_kind is CheckpointKind.STRUCTURED_INPUT_SUSPENSION
        ):
            return
        raise ConversationTransitionError()
    if compact_continuation:
        if (
            child_kind is CheckpointKind.COMPLETED_OUTWARD_TURN
            and parent_kind is CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        ):
            return
        raise ConversationTransitionError()
    if parent_kind not in allowed.get(child_kind, frozenset()):
        raise ConversationTransitionError()


def is_standalone_compact_bridge(
    checkpoint: ConversationCheckpoint | None,
    source: ConversationCheckpoint | None,
) -> bool:
    """Return whether one internal checkpoint bridges compact state."""
    return (
        type(checkpoint) is ConversationCheckpoint
        and type(source) is ConversationCheckpoint
        and checkpoint.kind is CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        and source.kind is CheckpointKind.STANDALONE_COMPACT_RESULT
        and checkpoint.lifecycle is CheckpointLifecycle.COMMITTED
        and source.lifecycle is CheckpointLifecycle.COMMITTED
        and checkpoint.authority == source.authority
        and checkpoint.identity.conversation_id
        == source.identity.conversation_id
        and checkpoint.identity.parent_checkpoint_id
        == source.identity.checkpoint_id
        and checkpoint.identity.parent_sequence == source.identity.sequence
        and checkpoint.identity.sequence == source.identity.sequence + 1
        and checkpoint.content == source.content
        and bool(checkpoint.content.lanes)
        and all(
            type(lane) is StatelessProviderLaneSnapshot
            and lane.compaction_boundary is not None
            for lane in checkpoint.content.lanes
        )
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
