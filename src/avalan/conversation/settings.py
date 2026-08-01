"""Define immutable conversation settings, parents, handles, and results."""

from .binding import ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CheckpointId,
    ConversationBranchId,
    ConversationId,
    NamedHeadId,
    NamedHeadRevision,
    PublicResponseId,
)
from .errors import (
    ConversationAuthorizationError,
    ConversationValidationError,
)
from .value import CallerHeldState, IntegrityDigest, validate_identifier

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, final


class ConversationMode(StrEnum):
    """Identify the provider continuation mode for one lane."""

    OFF = "off"
    STATELESS = "stateless"
    STORED = "stored"


class ReasoningContext(StrEnum):
    """Identify the requested reasoning-context scope."""

    AUTO = "auto"
    CURRENT_TURN = "current_turn"
    ALL_TURNS = "all_turns"


class EffectiveReasoningContext(StrEnum):
    """Identify a provider-reported effective reasoning context."""

    CURRENT_TURN = "current_turn"
    ALL_TURNS = "all_turns"


class CompactionOperation(StrEnum):
    """Identify an explicit compaction operation."""

    NONE = "none"
    INLINE = "inline"
    STANDALONE = "standalone"


class ConversationResetDisposition(StrEnum):
    """Identify whether opaque continuity survives an explicit reset."""

    PRESERVED = "preserved"
    OPAQUE_STATE_LOST = "opaque_state_lost"


class ConversationModeChangeOperation(StrEnum):
    """Identify an explicit reset or continuity-preserving conversion."""

    RESET = "reset"
    CONVERT = "convert"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ModeTransitionAuthority:
    """Bind one mode change to exact current continuity authority."""

    authority: AuthorityScope
    binding: ProviderLaneBinding
    checkpoint_id: CheckpointId | None
    parent: "ConversationParent | None"
    source_mode: ConversationMode
    target_mode: ConversationMode
    operation: ConversationModeChangeOperation

    def __post_init__(self) -> None:
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        if type(self.binding) is not ProviderLaneBinding:
            raise ConversationValidationError()
        if not isinstance(
            self.source_mode, ConversationMode
        ) or not isinstance(self.target_mode, ConversationMode):
            raise ConversationValidationError()
        if not isinstance(self.operation, ConversationModeChangeOperation):
            raise ConversationValidationError()
        pair = (self.source_mode, self.target_mode)
        if self.operation is ConversationModeChangeOperation.RESET:
            if pair == (ConversationMode.OFF, ConversationMode.OFF):
                raise ConversationValidationError()
        elif pair not in {
            (ConversationMode.STATELESS, ConversationMode.STORED),
            (ConversationMode.STORED, ConversationMode.STATELESS),
        }:
            raise ConversationValidationError()
        _validate_mode_transition_parent(self)


ConversationModeChangeAuthorization: TypeAlias = ModeTransitionAuthority


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationModeReset:
    """Reset visible and opaque continuity under exact trusted authority."""

    authorization: ModeTransitionAuthority
    disposition: ConversationResetDisposition = (
        ConversationResetDisposition.OPAQUE_STATE_LOST
    )

    def __post_init__(self) -> None:
        if type(self.authorization) is not ModeTransitionAuthority:
            raise ConversationValidationError()
        if (
            self.authorization.operation
            is not ConversationModeChangeOperation.RESET
            or self.disposition
            is not ConversationResetDisposition.OPAQUE_STATE_LOST
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationModeConversion:
    """Convert stored and stateless continuity without silent state loss."""

    authorization: ModeTransitionAuthority
    disposition: ConversationResetDisposition = (
        ConversationResetDisposition.PRESERVED
    )

    def __post_init__(self) -> None:
        if type(self.authorization) is not ModeTransitionAuthority:
            raise ConversationValidationError()
        if (
            self.authorization.operation
            is not ConversationModeChangeOperation.CONVERT
            or self.disposition is not ConversationResetDisposition.PRESERVED
        ):
            raise ConversationValidationError()


ConversationModeTransition: TypeAlias = (
    ConversationModeReset | ConversationModeConversion
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DisabledCompaction:
    """Disable inline compaction explicitly."""

    operation: CompactionOperation = CompactionOperation.NONE

    def __post_init__(self) -> None:
        if self.operation is not CompactionOperation.NONE:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InlineCompaction:
    """Request provider inline compaction at a positive token threshold."""

    compact_threshold: int
    operation: CompactionOperation = CompactionOperation.INLINE

    def __post_init__(self) -> None:
        if (
            self.operation is not CompactionOperation.INLINE
            or type(self.compact_threshold) is not int
            or self.compact_threshold <= 0
        ):
            raise ConversationValidationError()


CompactionPolicy: TypeAlias = DisabledCompaction | InlineCompaction


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessConversationHandle:
    """Return one stateless checkpoint and optional caller-held envelope."""

    conversation_id: ConversationId
    checkpoint_id: CheckpointId
    branch_id: ConversationBranchId
    envelope: CallerHeldState | None = None
    mode: ConversationMode = ConversationMode.STATELESS

    def __post_init__(self) -> None:
        if self.mode is not ConversationMode.STATELESS:
            raise ConversationValidationError()
        for value, name in (
            (self.conversation_id, "conversation_id"),
            (self.checkpoint_id, "checkpoint_id"),
            (self.branch_id, "branch_id"),
        ):
            validate_identifier(value, name)
        if (
            self.envelope is not None
            and type(self.envelope) is not CallerHeldState
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoredConversationHandle:
    """Return one Avalan-owned handle for provider-stored continuation."""

    conversation_id: ConversationId
    checkpoint_id: CheckpointId
    branch_id: ConversationBranchId
    public_response_id: PublicResponseId | None = None
    mode: ConversationMode = ConversationMode.STORED

    def __post_init__(self) -> None:
        if self.mode is not ConversationMode.STORED:
            raise ConversationValidationError()
        for value, name in (
            (self.conversation_id, "conversation_id"),
            (self.checkpoint_id, "checkpoint_id"),
            (self.branch_id, "branch_id"),
        ):
            validate_identifier(value, name)
        if self.public_response_id is not None:
            validate_identifier(self.public_response_id, "public_response_id")


ConversationHandle: TypeAlias = (
    StatelessConversationHandle | StoredConversationHandle
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessParent:
    """Select a stateless parent without exposing provider state separately."""

    handle: StatelessConversationHandle

    def __post_init__(self) -> None:
        if type(self.handle) is not StatelessConversationHandle:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoredParent:
    """Select a stored parent without exposing its upstream response ID."""

    handle: StoredConversationHandle

    def __post_init__(self) -> None:
        if type(self.handle) is not StoredConversationHandle:
            raise ConversationValidationError()


ConversationParent: TypeAlias = StatelessParent | StoredParent


def _validate_mode_transition_parent(
    authorization: ModeTransitionAuthority,
) -> None:
    if authorization.source_mode is ConversationMode.OFF:
        if (
            authorization.checkpoint_id is not None
            or authorization.parent is not None
        ):
            raise ConversationValidationError()
        return
    if authorization.checkpoint_id is None:
        raise ConversationValidationError()
    if authorization.source_mode is ConversationMode.STATELESS:
        if (
            type(authorization.parent) is not StatelessParent
            or authorization.parent.handle.checkpoint_id
            != authorization.checkpoint_id
        ):
            raise ConversationValidationError()
    elif (
        type(authorization.parent) is not StoredParent
        or authorization.parent.handle.checkpoint_id
        != authorization.checkpoint_id
    ):
        raise ConversationValidationError()


def validate_mode_transition_authority(
    transition: ConversationModeTransition,
    *,
    current_checkpoint_id: CheckpointId | None,
    current_parent: ConversationParent | None,
    current_authority: AuthorityScope,
    current_binding: ProviderLaneBinding,
) -> None:
    """Validate one mode transition against exact trusted current state."""
    if not isinstance(
        transition,
        ConversationModeReset | ConversationModeConversion,
    ):
        raise ConversationValidationError()
    authorization = transition.authorization
    if (
        type(current_authority) is not AuthorityScope
        or type(current_binding) is not ProviderLaneBinding
    ):
        raise ConversationValidationError()
    if current_checkpoint_id is not None:
        validate_identifier(current_checkpoint_id, "current_checkpoint_id")
    if current_parent is not None and not isinstance(
        current_parent,
        StatelessParent | StoredParent,
    ):
        raise ConversationValidationError()
    if (current_checkpoint_id is None) != (current_parent is None):
        raise ConversationValidationError()
    if authorization.authority != current_authority:
        raise ConversationAuthorizationError()
    authorization.binding.assert_compatible(current_binding)
    if (
        authorization.checkpoint_id != current_checkpoint_id
        or authorization.parent != current_parent
    ):
        raise ConversationAuthorizationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OneShotConversationSettings:
    """Preserve existing one-shot behavior with no continuation state."""

    mode: ConversationMode = ConversationMode.OFF

    def __post_init__(self) -> None:
        if self.mode is not ConversationMode.OFF:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessConversationSettings:
    """Configure stateless replay with a stateless parent only."""

    parent: StatelessParent | None = None
    reasoning_context: ReasoningContext = ReasoningContext.AUTO
    compaction: CompactionPolicy = DisabledCompaction()
    mode: ConversationMode = ConversationMode.STATELESS

    def __post_init__(self) -> None:
        if self.mode is not ConversationMode.STATELESS:
            raise ConversationValidationError()
        if (
            self.parent is not None
            and type(self.parent) is not StatelessParent
        ):
            raise ConversationValidationError()
        if not isinstance(self.reasoning_context, ReasoningContext):
            raise ConversationValidationError()
        if not isinstance(
            self.compaction, DisabledCompaction | InlineCompaction
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoredConversationSettings:
    """Configure disclosed provider-stored continuation."""

    provider_storage_disclosed: bool
    parent: StoredParent | None = None
    reasoning_context: ReasoningContext = ReasoningContext.AUTO
    mode: ConversationMode = ConversationMode.STORED

    def __post_init__(self) -> None:
        if (
            self.mode is not ConversationMode.STORED
            or type(self.provider_storage_disclosed) is not bool
            or not self.provider_storage_disclosed
        ):
            raise ConversationValidationError()
        if self.parent is not None and type(self.parent) is not StoredParent:
            raise ConversationValidationError()
        if not isinstance(self.reasoning_context, ReasoningContext):
            raise ConversationValidationError()


ConversationSettings: TypeAlias = (
    OneShotConversationSettings
    | StatelessConversationSettings
    | StoredConversationSettings
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveReasoningMetadata:
    """Record requested and provider-reported reasoning context separately."""

    requested: ReasoningContext
    effective: EffectiveReasoningContext | None

    def __post_init__(self) -> None:
        if not isinstance(self.requested, ReasoningContext):
            raise ConversationValidationError()
        if self.effective is not None and not isinstance(
            self.effective,
            EffectiveReasoningContext,
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationResult:
    """Return a completed conversation handle and safe integrity metadata."""

    handle: ConversationHandle
    reasoning: EffectiveReasoningMetadata
    checkpoint_digest: IntegrityDigest

    def __post_init__(self) -> None:
        if not isinstance(
            self.handle,
            StatelessConversationHandle | StoredConversationHandle,
        ):
            raise ConversationValidationError()
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        validate_identifier(self.checkpoint_digest, "checkpoint_digest")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationStreamTerminal:
    """Expose the same terminal result contract for streamed execution."""

    result: ConversationResult

    def __post_init__(self) -> None:
        if type(self.result) is not ConversationResult:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NamedHeadParent:
    """Bind a named-head advance to an exact expected revision."""

    head_id: NamedHeadId
    expected_revision: NamedHeadRevision
    parent: ConversationParent

    def __post_init__(self) -> None:
        validate_identifier(self.head_id, "head_id")
        if (
            type(self.expected_revision) is not int
            or self.expected_revision < 0
        ):
            raise ConversationValidationError()
        if not isinstance(self.parent, StatelessParent | StoredParent):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StandaloneCompactRequest:
    """Request standalone compaction from a stateless parent only."""

    parent: StatelessParent
    operation: CompactionOperation = CompactionOperation.STANDALONE

    def __post_init__(self) -> None:
        if (
            type(self.parent) is not StatelessParent
            or self.operation is not CompactionOperation.STANDALONE
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StandaloneCompactResult:
    """Return canonical caller-held compact state and its checkpoint handle."""

    handle: StatelessConversationHandle
    canonical_context_digest: IntegrityDigest

    def __post_init__(self) -> None:
        if type(self.handle) is not StatelessConversationHandle:
            raise ConversationValidationError()
        validate_identifier(
            self.canonical_context_digest,
            "canonical_context_digest",
        )
