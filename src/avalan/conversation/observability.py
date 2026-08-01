"""Create secret-free conversation digests and observability projections."""

from ..types import JsonValue
from .contract import (
    AuthorityScope,
    CanonicalRequestDigest,
    CheckpointId,
    ConversationOperation,
    RequestIdempotencyKey,
)
from .errors import ConversationValidationError
from .settings import ConversationMode, ReasoningContext
from .state import ConversationCheckpoint
from .value import (
    AuthorityDigest,
    IntegrityDigest,
    RequestSemanticDigest,
    SafeAlias,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationRequestSemantics:
    """Describe digestable request semantics without retaining secrets."""

    authority: AuthorityScope
    operation: ConversationOperation
    mode: ConversationMode
    reasoning_context: ReasoningContext
    semantic_input: JsonValue
    parent_checkpoint_id: CheckpointId | None = None
    opaque_digests: tuple[IntegrityDigest, ...] = ()

    def __post_init__(self) -> None:
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        if not isinstance(self.operation, ConversationOperation):
            raise ConversationValidationError()
        if not isinstance(self.mode, ConversationMode) or not isinstance(
            self.reasoning_context,
            ReasoningContext,
        ):
            raise ConversationValidationError()
        object.__setattr__(
            self,
            "semantic_input",
            freeze_json_value(self.semantic_input),
        )
        if self.parent_checkpoint_id is not None:
            validate_identifier(
                self.parent_checkpoint_id,
                "parent_checkpoint_id",
            )
        if type(self.opaque_digests) is not tuple:
            raise ConversationValidationError()
        for value in self.opaque_digests:
            validate_identifier(value, "opaque_digest")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationObservation:
    """Project only content-free checkpoint state for telemetry."""

    event: str
    checkpoint_id: CheckpointId
    lane_count: int
    provider_item_count: int
    transcript_entry_count: int
    opaque_byte_count: int
    checkpoint_state: str
    codec_version: int | None
    integrity_digest: IntegrityDigest | None
    binding_aliases: tuple[SafeAlias, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.event, "event")
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        for value in (
            self.lane_count,
            self.provider_item_count,
            self.transcript_entry_count,
            self.opaque_byte_count,
        ):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()
        validate_identifier(self.checkpoint_state, "checkpoint_state")
        if self.codec_version is not None and (
            type(self.codec_version) is not int or self.codec_version <= 0
        ):
            raise ConversationValidationError()
        if self.integrity_digest is not None:
            validate_identifier(self.integrity_digest, "integrity_digest")
        if type(self.binding_aliases) is not tuple:
            raise ConversationValidationError()
        for alias in self.binding_aliases:
            validate_identifier(alias, "binding_alias")

    def to_mapping(self) -> Mapping[str, JsonValue]:
        """Return a read-only mapping containing permitted fields only."""
        value: dict[str, JsonValue] = {
            "event": self.event,
            "checkpoint_id": self.checkpoint_id,
            "lane_count": self.lane_count,
            "provider_item_count": self.provider_item_count,
            "transcript_entry_count": self.transcript_entry_count,
            "opaque_byte_count": self.opaque_byte_count,
            "checkpoint_state": self.checkpoint_state,
            "codec_version": self.codec_version,
            "integrity_digest": self.integrity_digest,
            "binding_aliases": tuple(self.binding_aliases),
        }
        return MappingProxyType(value)


def authority_digest(authority: AuthorityScope) -> AuthorityDigest:
    """Digest the complete trusted authority scope deterministically."""
    if type(authority) is not AuthorityScope:
        raise ConversationValidationError()
    value = freeze_json_value(
        {
            "source": authority.source.value,
            "tenant_id": authority.tenant_id,
            "principal_id": authority.principal_id,
            "agent_id": authority.agent_id,
            "endpoint_id": authority.endpoint_id,
            "local_single_user_configured": (
                authority.local_single_user_configured
            ),
            "network_exposed": authority.network_exposed,
        }
    )
    return AuthorityDigest(sha256(canonical_json_bytes(value)).hexdigest())


def canonical_request_digest(
    request: ConversationRequestSemantics,
) -> CanonicalRequestDigest:
    """Digest semantic request content and authority without secret bytes."""
    if type(request) is not ConversationRequestSemantics:
        raise ConversationValidationError()
    value = freeze_json_value(
        {
            "authority_digest": authority_digest(request.authority),
            "operation": request.operation.value,
            "mode": request.mode.value,
            "reasoning_context": request.reasoning_context.value,
            "parent_checkpoint_id": request.parent_checkpoint_id,
            "semantic_input": request.semantic_input,
            "opaque_digests": tuple(request.opaque_digests),
        }
    )
    return CanonicalRequestDigest(
        sha256(canonical_json_bytes(value)).hexdigest()
    )


def idempotency_digest(
    request: ConversationRequestSemantics,
    key: RequestIdempotencyKey,
) -> RequestSemanticDigest:
    """Bind one request digest to its idempotency namespace key."""
    validate_identifier(key, "idempotency_key")
    digest = canonical_request_digest(request)
    value = freeze_json_value(
        {
            "request_digest": digest,
            "idempotency_key": key,
        }
    )
    return RequestSemanticDigest(
        sha256(canonical_json_bytes(value)).hexdigest()
    )


def checkpoint_observation(
    event: str,
    checkpoint: ConversationCheckpoint,
) -> ConversationObservation:
    """Return a safe immutable observability projection."""
    if type(checkpoint) is not ConversationCheckpoint:
        raise ConversationValidationError()
    counts = checkpoint.content.safe_counts
    integrity = checkpoint.integrity
    return ConversationObservation(
        event=event,
        checkpoint_id=checkpoint.identity.checkpoint_id,
        lane_count=counts.lane_count,
        provider_item_count=counts.provider_item_count,
        transcript_entry_count=counts.transcript_entry_count,
        opaque_byte_count=counts.opaque_byte_count,
        checkpoint_state=checkpoint.lifecycle.value,
        codec_version=(integrity.codec_version if integrity else None),
        integrity_digest=(integrity.digest if integrity else None),
        binding_aliases=tuple(
            lane.binding.safe_alias for lane in checkpoint.content.lanes
        ),
    )
