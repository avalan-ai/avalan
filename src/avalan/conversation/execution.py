"""Bind completed provider output to one exact checkpoint execution."""

from ..types import JsonValue
from .binding import ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CheckpointIdentity,
    ProviderLaneId,
    RequestIdempotencyIdentity,
    UpstreamResponseId,
)
from .errors import ConversationValidationError
from .items import ProviderItem
from .settings import (
    ConversationMode,
    EffectiveReasoningMetadata,
    ProviderLaneOutputScope,
    ProviderUsage,
)
from .value import (
    IntegrityDigest,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from dataclasses import dataclass
from hashlib import sha256
from typing import final


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionReceipt:
    """Store content-safe proof of one exact completed lane execution."""

    schema_version: int
    digest: IntegrityDigest
    item_count: int
    opaque_byte_count: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConversationValidationError()
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise ConversationValidationError()
        try:
            int(self.digest, 16)
        except ValueError as exc:
            raise ConversationValidationError() from exc
        if (
            type(self.item_count) is not int
            or self.item_count < 0
            or type(self.opaque_byte_count) is not int
            or self.opaque_byte_count < 0
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free receipt metadata."""
        return (
            "ProviderLaneExecutionReceipt("
            f"schema_version={self.schema_version}, "
            f"digest={self.digest!r}, item_count={self.item_count}, "
            f"opaque_byte_count={self.opaque_byte_count})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionReservation:
    """Bind one expected lane to an owned execution attempt."""

    binding: ProviderLaneBinding
    mode: ConversationMode
    scope: ProviderLaneOutputScope

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or not isinstance(self.mode, ConversationMode)
            or self.mode is ConversationMode.OFF
            or not isinstance(self.scope, ProviderLaneOutputScope)
            or (
                self.mode is ConversationMode.STORED
                and self.scope is not ProviderLaneOutputScope.CURRENT_CALL
            )
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free reservation metadata."""
        return (
            "ProviderLaneExecutionReservation("
            f"lane_id={self.binding.lane_id!r}, mode={self.mode.value!r}, "
            f"scope={self.scope.value!r})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConversationExecutionReservation:
    """Bind an idempotency reservation to one exact turn and lane set."""

    idempotency: RequestIdempotencyIdentity
    identity: CheckpointIdentity
    lanes: tuple[ProviderLaneExecutionReservation, ...]

    def __post_init__(self) -> None:
        if (
            type(self.idempotency) is not RequestIdempotencyIdentity
            or type(self.identity) is not CheckpointIdentity
            or type(self.lanes) is not tuple
            or not self.lanes
            or any(
                type(lane) is not ProviderLaneExecutionReservation
                for lane in self.lanes
            )
        ):
            raise ConversationValidationError()
        lane_ids = tuple(lane.binding.lane_id for lane in self.lanes)
        if len(lane_ids) != len(set(lane_ids)) or any(
            lane.binding.agent_id != self.idempotency.authority.agent_id
            for lane in self.lanes
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free reservation metadata."""
        return (
            "ConversationExecutionReservation("
            f"checkpoint_id={self.identity.checkpoint_id!r}, "
            f"lane_count={len(self.lanes)})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionStage:
    """Request authoritative staging for one exact completed lane."""

    idempotency: RequestIdempotencyIdentity
    owner_token: str
    identity: CheckpointIdentity
    binding: ProviderLaneBinding
    mode: ConversationMode
    scope: ProviderLaneOutputScope
    completed_items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    usage: ProviderUsage
    execution_receipt: ProviderLaneExecutionReceipt
    upstream_response_id: UpstreamResponseId | None = None

    def __post_init__(self) -> None:
        if (
            type(self.idempotency) is not RequestIdempotencyIdentity
            or type(self.identity) is not CheckpointIdentity
            or type(self.binding) is not ProviderLaneBinding
            or not isinstance(self.mode, ConversationMode)
            or self.mode is ConversationMode.OFF
            or not isinstance(self.scope, ProviderLaneOutputScope)
            or type(self.completed_items) is not tuple
            or any(
                type(item) is not ProviderItem for item in self.completed_items
            )
            or any(
                item.lane_id != self.binding.lane_id
                for item in self.completed_items
            )
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or type(self.usage) is not ProviderUsage
            or type(self.execution_receipt) is not ProviderLaneExecutionReceipt
        ):
            raise ConversationValidationError()
        validate_identifier(self.owner_token, "owner_token")
        if (self.mode is ConversationMode.STORED) != (
            self.upstream_response_id is not None
        ):
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id,
                "upstream_response_id",
            )
        expected = provider_lane_execution_receipt(
            authority=self.idempotency.authority,
            identity=self.identity,
            binding=self.binding,
            mode=self.mode,
            scope=self.scope,
            completed_items=self.completed_items,
            reasoning=self.reasoning,
            usage=self.usage,
            upstream_response_id=self.upstream_response_id,
        )
        if self.execution_receipt != expected:
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free staging metadata."""
        return (
            "ProviderLaneExecutionStage("
            f"checkpoint_id={self.identity.checkpoint_id!r}, "
            f"lane_id={self.binding.lane_id!r}, "
            f"item_count={len(self.completed_items)})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionAttestation:
    """Carry one opaque store-owned staged-execution identity."""

    schema_version: int
    staging_id: str
    lane_id: ProviderLaneId

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConversationValidationError()
        validate_identifier(self.staging_id, "staging_id")
        validate_identifier(self.lane_id, "lane_id")

    def __repr__(self) -> str:
        """Return metadata without the opaque staging identity."""
        return (
            "ProviderLaneExecutionAttestation("
            f"schema_version={self.schema_version}, "
            "staging_id=<redacted>, "
            f"lane_id={self.lane_id!r})"
        )


def provider_lane_execution_receipt(
    *,
    authority: AuthorityScope,
    identity: CheckpointIdentity,
    binding: ProviderLaneBinding,
    mode: ConversationMode,
    scope: ProviderLaneOutputScope,
    completed_items: tuple[ProviderItem, ...],
    reasoning: EffectiveReasoningMetadata,
    usage: ProviderUsage,
    upstream_response_id: UpstreamResponseId | None,
) -> ProviderLaneExecutionReceipt:
    """Return a digest binding exact output to its execution authority."""
    if (
        type(authority) is not AuthorityScope
        or type(identity) is not CheckpointIdentity
        or type(binding) is not ProviderLaneBinding
        or not isinstance(mode, ConversationMode)
        or mode is ConversationMode.OFF
        or not isinstance(scope, ProviderLaneOutputScope)
        or type(completed_items) is not tuple
        or any(type(item) is not ProviderItem for item in completed_items)
        or type(reasoning) is not EffectiveReasoningMetadata
        or type(usage) is not ProviderUsage
    ):
        raise ConversationValidationError()
    if any(item.lane_id != binding.lane_id for item in completed_items):
        raise ConversationValidationError()
    if (mode is ConversationMode.STORED) != (upstream_response_id is not None):
        raise ConversationValidationError()
    payload = freeze_json_value(
        {
            "domain": "avalan.provider-lane-execution",
            "schema_version": 1,
            "authority": {
                "source": authority.source.value,
                "principal_id": authority.principal_id,
                "agent_id": authority.agent_id,
                "endpoint_id": authority.endpoint_id,
                "tenant_id": authority.tenant_id,
                "local_single_user_configured": (
                    authority.local_single_user_configured
                ),
                "network_exposed": authority.network_exposed,
            },
            "identity": {
                "conversation_id": identity.conversation_id,
                "logical_turn_id": identity.logical_turn_id,
                "execution_segment_id": identity.execution_segment_id,
                "checkpoint_id": identity.checkpoint_id,
                "branch_id": identity.branch_id,
                "sequence": identity.sequence,
                "parent_checkpoint_id": identity.parent_checkpoint_id,
                "parent_sequence": identity.parent_sequence,
            },
            "binding_digest": binding.integrity_digest,
            "lane_id": binding.lane_id,
            "mode": mode.value,
            "scope": scope.value,
            "completed_items": tuple(
                _provider_item_value(item) for item in completed_items
            ),
            "reasoning": {
                "requested": reasoning.requested.value,
                "effective": (
                    reasoning.effective.value
                    if reasoning.effective is not None
                    else None
                ),
            },
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "upstream_response_id": upstream_response_id,
        }
    )
    return ProviderLaneExecutionReceipt(
        schema_version=1,
        digest=IntegrityDigest(
            sha256(canonical_json_bytes(payload)).hexdigest()
        ),
        item_count=len(completed_items),
        opaque_byte_count=sum(
            item.opaque_state.byte_count
            for item in completed_items
            if item.opaque_state is not None
        ),
    )


def _provider_item_value(item: ProviderItem) -> JsonValue:
    opaque = item.opaque_state
    return freeze_json_value(
        {
            "item_id": item.item_id,
            "lane_id": item.lane_id,
            "model_call_id": item.model_call_id,
            "kind": item.kind.value,
            "order": item.order,
            "provider_index": item.provider_index,
            "phase": item.phase.value,
            "caller": item.caller.value,
            "canonical_input": item.canonical_input,
            "normalization_version": item.normalization_version,
            "call_id": item.call_id,
            "opaque_state": (
                {
                    "digest": opaque.digest,
                    "byte_count": opaque.byte_count,
                }
                if opaque is not None
                else None
            ),
            "complete": item.complete,
        }
    )
