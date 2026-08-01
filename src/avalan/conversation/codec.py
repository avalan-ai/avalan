"""Encode and decode canonical versioned conversation checkpoints."""

from ..types import JsonValue
from .binding import ProviderFamily, ProviderLaneBinding, ProviderTransport
from .contract import (
    AuthorityEndpointId,
    AuthorityPrincipalId,
    AuthorityScope,
    AuthoritySource,
    AuthorityTenantId,
    CheckpointId,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointSequence,
    ChildLaneRetentionPolicy,
    ConversationAgentId,
    ConversationBranchId,
    ConversationId,
    ConversationModelCallId,
    ExecutionSegmentId,
    LocalResponseStorage,
    LogicalTurnId,
    NamedHeadId,
    NamedHeadRevision,
    ProviderLaneId,
    ProviderLaneStorage,
    RetentionLimits,
    StoragePolicy,
    UpstreamLifetimeStatus,
    UpstreamResponseId,
)
from .errors import (
    ConversationCodecError,
    ConversationIntegrityError,
    ConversationLimitError,
    ConversationValidationError,
)
from .items import (
    CompactionBoundary,
    ProviderItem,
    ProviderItemCaller,
    ProviderItemKind,
    ProviderItemLedger,
    ProviderItemPhase,
    VisibleTranscript,
    VisibleTranscriptEntry,
    VisibleTranscriptRole,
)
from .settings import (
    EffectiveReasoningContext,
    EffectiveReasoningMetadata,
    ReasoningContext,
)
from .state import (
    CheckpointIntegrityMetadata,
    CheckpointLifecycle,
    CheckpointTimestamps,
    ConversationCheckpoint,
    MultiLaneCheckpointContent,
    NamedHeadMetadata,
    ProviderLaneLifecycle,
    StatelessProviderLaneSnapshot,
    StoredProviderLaneSnapshot,
)
from .value import (
    CapabilityProfileRevision,
    ConversationCodecVersion,
    ExecutionDefinitionRevision,
    IntegrityDigest,
    JsonLimits,
    ModelConfigurationRevision,
    OpaqueProviderState,
    ProviderApiRevision,
    ProviderCallId,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
    ProviderSdkRevision,
    ToolSchemaRevision,
    freeze_json_value,
    thaw_json_value,
)

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from typing import final

CHECKPOINT_CODEC_VERSION = ConversationCodecVersion(1)
_ENVELOPE_KIND = "avalan.conversation.checkpoint"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointCodecLimits:
    """Bound encoded checkpoint bytes and decoded JSON complexity."""

    max_bytes: int = 4_194_304
    max_depth: int = 48
    max_items: int = 100_000
    max_string_bytes: int = 2_097_152

    def __post_init__(self) -> None:
        for value in (
            self.max_bytes,
            self.max_depth,
            self.max_items,
            self.max_string_bytes,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()

    @property
    def json_limits(self) -> JsonLimits:
        """Return the equivalent recursive JSON limits."""
        return JsonLimits(
            max_depth=self.max_depth,
            max_items=self.max_items,
            max_string_bytes=self.max_string_bytes,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCheckpointCodec:
    """Round-trip canonical version-one checkpoint envelopes strictly."""

    limits: CheckpointCodecLimits = CheckpointCodecLimits()

    def __post_init__(self) -> None:
        if type(self.limits) is not CheckpointCodecLimits:
            raise ConversationValidationError()

    def encode(self, checkpoint: ConversationCheckpoint) -> bytes:
        """Return canonical UTF-8 bytes for one immutable checkpoint."""
        if type(checkpoint) is not ConversationCheckpoint:
            raise ConversationValidationError()
        if checkpoint.integrity is None:
            raise ConversationIntegrityError()
        try:
            payload = freeze_json_value(
                {
                    "kind": _ENVELOPE_KIND,
                    "version": CHECKPOINT_CODEC_VERSION,
                    "checkpoint": _encode_checkpoint(checkpoint),
                },
                limits=self.limits.json_limits,
            )
            envelope = _mapping(
                payload,
                {"kind", "version", "checkpoint"},
            )
            rebuilt = _decode_checkpoint(envelope["checkpoint"])
            if rebuilt != checkpoint:
                raise ConversationCodecError()
            _verify_integrity(checkpoint)
            encoded = dumps(
                thaw_json_value(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except ConversationIntegrityError:
            raise
        except ConversationLimitError:
            raise
        except ConversationCodecError:
            raise
        except (ConversationValidationError, TypeError, ValueError) as exc:
            raise ConversationCodecError() from exc
        if len(encoded) > self.limits.max_bytes:
            raise ConversationLimitError()
        return encoded

    def decode(self, encoded: bytes) -> ConversationCheckpoint:
        """Return one checkpoint from strict canonical envelope bytes."""
        if type(encoded) is not bytes or not encoded:
            raise ConversationCodecError()
        if len(encoded) > self.limits.max_bytes:
            raise ConversationLimitError()
        try:
            text = encoded.decode("utf-8")
            raw: object = loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConversationCodecError() from exc
        frozen = freeze_json_value(raw, limits=self.limits.json_limits)
        envelope = _mapping(
            frozen,
            {"kind", "version", "checkpoint"},
        )
        if (
            _string(envelope["kind"]) != _ENVELOPE_KIND
            or _integer(envelope["version"]) != CHECKPOINT_CODEC_VERSION
        ):
            raise ConversationCodecError()
        checkpoint = _decode_checkpoint(envelope["checkpoint"])
        if checkpoint.integrity is None:
            raise ConversationIntegrityError()
        canonical = self.encode(checkpoint)
        if canonical != encoded:
            raise ConversationCodecError()
        _verify_integrity(checkpoint)
        return checkpoint


def checkpoint_payload_digest(
    checkpoint: ConversationCheckpoint,
) -> IntegrityDigest:
    """Digest exact checkpoint content with integrity metadata omitted."""
    if type(checkpoint) is not ConversationCheckpoint:
        raise ConversationValidationError()
    payload = _encode_unsigned_checkpoint_payload(checkpoint)
    return IntegrityDigest(sha256(payload).hexdigest())


def with_checkpoint_integrity(
    checkpoint: ConversationCheckpoint,
) -> ConversationCheckpoint:
    """Return a checkpoint bound to its exact canonical unsigned payload."""
    if type(checkpoint) is not ConversationCheckpoint:
        raise ConversationValidationError()
    payload = _encode_unsigned_checkpoint_payload(checkpoint)
    integrity = CheckpointIntegrityMetadata(
        codec_version=CHECKPOINT_CODEC_VERSION,
        digest=IntegrityDigest(sha256(payload).hexdigest()),
        encoded_byte_count=len(payload),
    )
    return replace(checkpoint, integrity=integrity)


def _encode_unsigned_checkpoint_payload(
    checkpoint: ConversationCheckpoint,
) -> bytes:
    """Return private canonical bytes with integrity metadata omitted."""
    unsigned = replace(checkpoint, integrity=None)
    return dumps(
        _encode_checkpoint(unsigned),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_checkpoint(
    checkpoint: ConversationCheckpoint,
) -> dict[str, object]:
    return {
        "identity": _encode_identity(checkpoint.identity),
        "kind": checkpoint.kind.value,
        "lifecycle": checkpoint.lifecycle.value,
        "authority": _encode_authority(checkpoint.authority),
        "content": _encode_content(checkpoint.content),
        "timestamps": _encode_timestamps(checkpoint.timestamps),
        "retention": _encode_retention(checkpoint.retention),
        "head": _encode_head(checkpoint.head),
        "integrity": _encode_integrity(checkpoint.integrity),
    }


def _decode_checkpoint(value: JsonValue) -> ConversationCheckpoint:
    item = _mapping(
        value,
        {
            "identity",
            "kind",
            "lifecycle",
            "authority",
            "content",
            "timestamps",
            "retention",
            "head",
            "integrity",
        },
    )
    try:
        return ConversationCheckpoint(
            identity=_decode_identity(item["identity"]),
            kind=CheckpointKind(_string(item["kind"])),
            lifecycle=CheckpointLifecycle(_string(item["lifecycle"])),
            authority=_decode_authority(item["authority"]),
            content=_decode_content(item["content"]),
            timestamps=_decode_timestamps(item["timestamps"]),
            retention=_decode_retention(item["retention"]),
            head=_decode_head(item["head"]),
            integrity=_decode_integrity(item["integrity"]),
        )
    except (AssertionError, ConversationValidationError, ValueError) as exc:
        raise ConversationCodecError() from exc


def _encode_identity(value: CheckpointIdentity) -> dict[str, object]:
    return {
        "conversation_id": value.conversation_id,
        "logical_turn_id": value.logical_turn_id,
        "execution_segment_id": value.execution_segment_id,
        "checkpoint_id": value.checkpoint_id,
        "branch_id": value.branch_id,
        "sequence": value.sequence,
        "parent_checkpoint_id": value.parent_checkpoint_id,
        "parent_sequence": value.parent_sequence,
    }


def _decode_identity(value: JsonValue) -> CheckpointIdentity:
    item = _mapping(
        value,
        {
            "conversation_id",
            "logical_turn_id",
            "execution_segment_id",
            "checkpoint_id",
            "branch_id",
            "sequence",
            "parent_checkpoint_id",
            "parent_sequence",
        },
    )
    parent_id = _optional_string(item["parent_checkpoint_id"])
    parent_sequence = _optional_integer(item["parent_sequence"])
    return CheckpointIdentity(
        conversation_id=ConversationId(_string(item["conversation_id"])),
        logical_turn_id=LogicalTurnId(_string(item["logical_turn_id"])),
        execution_segment_id=ExecutionSegmentId(
            _string(item["execution_segment_id"])
        ),
        checkpoint_id=CheckpointId(_string(item["checkpoint_id"])),
        branch_id=ConversationBranchId(_string(item["branch_id"])),
        sequence=CheckpointSequence(_integer(item["sequence"])),
        parent_checkpoint_id=CheckpointId(parent_id) if parent_id else None,
        parent_sequence=(
            CheckpointSequence(parent_sequence)
            if parent_sequence is not None
            else None
        ),
    )


def _encode_authority(value: AuthorityScope) -> dict[str, object]:
    return {
        "source": value.source.value,
        "principal_id": value.principal_id,
        "agent_id": value.agent_id,
        "endpoint_id": value.endpoint_id,
        "tenant_id": value.tenant_id,
        "local_single_user_configured": value.local_single_user_configured,
        "network_exposed": value.network_exposed,
    }


def _decode_authority(value: JsonValue) -> AuthorityScope:
    item = _mapping(
        value,
        {
            "source",
            "principal_id",
            "agent_id",
            "endpoint_id",
            "tenant_id",
            "local_single_user_configured",
            "network_exposed",
        },
    )
    tenant = _optional_string(item["tenant_id"])
    return AuthorityScope(
        source=AuthoritySource(_string(item["source"])),
        principal_id=AuthorityPrincipalId(_string(item["principal_id"])),
        agent_id=ConversationAgentId(_string(item["agent_id"])),
        endpoint_id=AuthorityEndpointId(_string(item["endpoint_id"])),
        tenant_id=AuthorityTenantId(tenant) if tenant else None,
        local_single_user_configured=_boolean(
            item["local_single_user_configured"]
        ),
        network_exposed=_boolean(item["network_exposed"]),
    )


def _encode_content(value: MultiLaneCheckpointContent) -> dict[str, object]:
    return {
        "visible_transcript": [
            {"role": entry.role.value, "content": entry.content}
            for entry in value.visible_transcript.entries
        ],
        "lanes": [_encode_lane(lane) for lane in value.lanes],
    }


def _decode_content(value: JsonValue) -> MultiLaneCheckpointContent:
    item = _mapping(value, {"visible_transcript", "lanes"})
    entries = tuple(
        _decode_transcript_entry(entry)
        for entry in _sequence(item["visible_transcript"])
    )
    lanes = tuple(_decode_lane(lane) for lane in _sequence(item["lanes"]))
    return MultiLaneCheckpointContent(
        visible_transcript=VisibleTranscript(entries=entries),
        lanes=lanes,
    )


def _decode_transcript_entry(value: JsonValue) -> VisibleTranscriptEntry:
    item = _mapping(value, {"role", "content"})
    return VisibleTranscriptEntry(
        role=VisibleTranscriptRole(_string(item["role"])),
        content=_string(item["content"]),
    )


def _encode_lane(
    value: StatelessProviderLaneSnapshot | StoredProviderLaneSnapshot,
) -> dict[str, object]:
    common: dict[str, object] = {
        "binding": _encode_binding(value.binding),
        "reasoning": _encode_reasoning(value.reasoning),
        "lifecycle": value.lifecycle.value,
        "retention_policy": value.retention_policy.value,
    }
    if isinstance(value, StatelessProviderLaneSnapshot):
        return {
            "mode": "stateless",
            **common,
            "ledger": _encode_ledger(value.ledger),
            "compaction_boundary": _encode_boundary(value.compaction_boundary),
        }
    return {
        "mode": "stored",
        **common,
        "upstream_response_id": value.upstream_response_id,
    }


def _decode_lane(
    value: JsonValue,
) -> StatelessProviderLaneSnapshot | StoredProviderLaneSnapshot:
    raw = _mapping_unchecked(value)
    mode = _string(raw.get("mode"))
    binding = _decode_binding(raw.get("binding"))
    reasoning = _decode_reasoning(raw.get("reasoning"))
    lifecycle = ProviderLaneLifecycle(_string(raw.get("lifecycle")))
    retention_policy = ChildLaneRetentionPolicy(
        _string(raw.get("retention_policy"))
    )
    if mode == "stateless":
        _exact_keys(
            raw,
            {
                "mode",
                "binding",
                "reasoning",
                "lifecycle",
                "retention_policy",
                "ledger",
                "compaction_boundary",
            },
        )
        return StatelessProviderLaneSnapshot(
            binding=binding,
            reasoning=reasoning,
            lifecycle=lifecycle,
            retention_policy=retention_policy,
            ledger=_decode_ledger(raw["ledger"]),
            compaction_boundary=_decode_boundary(raw["compaction_boundary"]),
        )
    if mode == "stored":
        _exact_keys(
            raw,
            {
                "mode",
                "binding",
                "reasoning",
                "lifecycle",
                "retention_policy",
                "upstream_response_id",
            },
        )
        return StoredProviderLaneSnapshot(
            binding=binding,
            reasoning=reasoning,
            lifecycle=lifecycle,
            retention_policy=retention_policy,
            upstream_response_id=UpstreamResponseId(
                _string(raw["upstream_response_id"])
            ),
        )
    raise ConversationCodecError()


def _encode_binding(value: ProviderLaneBinding) -> dict[str, object]:
    return {
        "lane_id": value.lane_id,
        "adapter_type": value.adapter_type,
        "provider_family": value.provider_family.value,
        "normalized_endpoint": value.normalized_endpoint,
        "azure_resource_identity": value.azure_resource_identity,
        "model_or_deployment": value.model_or_deployment,
        "provider_api_revision": value.provider_api_revision,
        "sdk_revision": value.sdk_revision,
        "model_configuration_revision": value.model_configuration_revision,
        "capability_profile_revision": value.capability_profile_revision,
        "tool_schema_revision": value.tool_schema_revision,
        "execution_definition_revision": value.execution_definition_revision,
        "continuation_codec_version": value.continuation_codec_version,
        "transport": value.transport.value,
        "agent_id": value.agent_id,
    }


def _decode_binding(value: object) -> ProviderLaneBinding:
    item = _mapping(
        _json(value),
        {
            "lane_id",
            "adapter_type",
            "provider_family",
            "normalized_endpoint",
            "azure_resource_identity",
            "model_or_deployment",
            "provider_api_revision",
            "sdk_revision",
            "model_configuration_revision",
            "capability_profile_revision",
            "tool_schema_revision",
            "execution_definition_revision",
            "continuation_codec_version",
            "transport",
            "agent_id",
        },
    )
    return ProviderLaneBinding(
        lane_id=ProviderLaneId(_string(item["lane_id"])),
        adapter_type=_string(item["adapter_type"]),
        provider_family=ProviderFamily(_string(item["provider_family"])),
        normalized_endpoint=_string(item["normalized_endpoint"]),
        azure_resource_identity=_optional_string(
            item["azure_resource_identity"]
        ),
        model_or_deployment=_string(item["model_or_deployment"]),
        provider_api_revision=ProviderApiRevision(
            _string(item["provider_api_revision"])
        ),
        sdk_revision=ProviderSdkRevision(_string(item["sdk_revision"])),
        model_configuration_revision=ModelConfigurationRevision(
            _string(item["model_configuration_revision"])
        ),
        capability_profile_revision=CapabilityProfileRevision(
            _string(item["capability_profile_revision"])
        ),
        tool_schema_revision=ToolSchemaRevision(
            _string(item["tool_schema_revision"])
        ),
        execution_definition_revision=ExecutionDefinitionRevision(
            _string(item["execution_definition_revision"])
        ),
        continuation_codec_version=ConversationCodecVersion(
            _integer(item["continuation_codec_version"])
        ),
        transport=ProviderTransport(_string(item["transport"])),
        agent_id=ConversationAgentId(_string(item["agent_id"])),
    )


def _encode_reasoning(value: EffectiveReasoningMetadata) -> dict[str, object]:
    return {
        "requested": value.requested.value,
        "effective": value.effective.value if value.effective else None,
    }


def _decode_reasoning(value: object) -> EffectiveReasoningMetadata:
    item = _mapping(_json(value), {"requested", "effective"})
    effective = _optional_string(item["effective"])
    return EffectiveReasoningMetadata(
        requested=ReasoningContext(_string(item["requested"])),
        effective=(
            EffectiveReasoningContext(effective) if effective else None
        ),
    )


def _encode_ledger(value: ProviderItemLedger) -> dict[str, object]:
    return {
        "lane_id": value.lane_id,
        "normalization_version": value.normalization_version,
        "items": [_encode_item(item) for item in value.items],
    }


def _decode_ledger(value: object) -> ProviderItemLedger:
    item = _mapping(
        _json(value),
        {"lane_id", "normalization_version", "items"},
    )
    return ProviderItemLedger(
        lane_id=ProviderLaneId(_string(item["lane_id"])),
        normalization_version=ConversationCodecVersion(
            _integer(item["normalization_version"])
        ),
        items=tuple(_decode_item(raw) for raw in _sequence(item["items"])),
    )


def _encode_item(value: ProviderItem) -> dict[str, object]:
    return {
        "item_id": value.item_id,
        "lane_id": value.lane_id,
        "model_call_id": value.model_call_id,
        "kind": value.kind.value,
        "order": value.order,
        "provider_index": value.provider_index,
        "phase": value.phase.value,
        "caller": value.caller.value,
        "canonical_input": thaw_json_value(value.canonical_input),
        "normalization_version": value.normalization_version,
        "call_id": value.call_id,
        "opaque_state": (
            b64encode(value.opaque_state._codec_bytes()).decode("ascii")
            if value.opaque_state
            else None
        ),
        "complete": value.complete,
    }


def _decode_item(value: JsonValue) -> ProviderItem:
    item = _mapping(
        value,
        {
            "item_id",
            "lane_id",
            "model_call_id",
            "kind",
            "order",
            "provider_index",
            "phase",
            "caller",
            "canonical_input",
            "normalization_version",
            "call_id",
            "opaque_state",
            "complete",
        },
    )
    call_id = _optional_string(item["call_id"])
    opaque = _optional_string(item["opaque_state"])
    try:
        opaque_state = (
            OpaqueProviderState(_value=b64decode(opaque, validate=True))
            if opaque is not None
            else None
        )
    except ValueError as exc:
        raise ConversationCodecError() from exc
    return ProviderItem(
        item_id=ProviderItemId(_string(item["item_id"])),
        lane_id=ProviderLaneId(_string(item["lane_id"])),
        model_call_id=ConversationModelCallId(_string(item["model_call_id"])),
        kind=ProviderItemKind(_string(item["kind"])),
        order=ProviderItemOrder(_integer(item["order"])),
        provider_index=ProviderItemIndex(_integer(item["provider_index"])),
        phase=ProviderItemPhase(_string(item["phase"])),
        caller=ProviderItemCaller(_string(item["caller"])),
        canonical_input=_mapping_unchecked(item["canonical_input"]),
        normalization_version=ConversationCodecVersion(
            _integer(item["normalization_version"])
        ),
        call_id=ProviderCallId(call_id) if call_id else None,
        opaque_state=opaque_state,
        complete=_boolean(item["complete"]),
    )


def _encode_boundary(value: CompactionBoundary | None) -> object:
    if value is None:
        return None
    return {
        "boundary_item_id": value.boundary_item_id,
        "boundary_order": value.boundary_order,
        "retained_suffix": list(value.retained_suffix),
    }


def _decode_boundary(value: object) -> CompactionBoundary | None:
    if value is None:
        return None
    item = _mapping(
        _json(value),
        {"boundary_item_id", "boundary_order", "retained_suffix"},
    )
    return CompactionBoundary(
        boundary_item_id=ProviderItemId(_string(item["boundary_item_id"])),
        boundary_order=ProviderItemOrder(_integer(item["boundary_order"])),
        retained_suffix=tuple(
            ProviderItemId(_string(raw))
            for raw in _sequence(item["retained_suffix"])
        ),
    )


def _encode_timestamps(value: CheckpointTimestamps) -> dict[str, object]:
    return {
        "created_at": value.created_at.isoformat(),
        "committed_at": (
            value.committed_at.isoformat() if value.committed_at else None
        ),
        "expires_at": (
            value.expires_at.isoformat() if value.expires_at else None
        ),
        "tombstoned_at": (
            value.tombstoned_at.isoformat() if value.tombstoned_at else None
        ),
        "deleted_at": (
            value.deleted_at.isoformat() if value.deleted_at else None
        ),
    }


def _decode_timestamps(value: JsonValue) -> CheckpointTimestamps:
    item = _mapping(
        value,
        {
            "created_at",
            "committed_at",
            "expires_at",
            "tombstoned_at",
            "deleted_at",
        },
    )
    return CheckpointTimestamps(
        created_at=_datetime(item["created_at"]),
        committed_at=_optional_datetime(item["committed_at"]),
        expires_at=_optional_datetime(item["expires_at"]),
        tombstoned_at=_optional_datetime(item["tombstoned_at"]),
        deleted_at=_optional_datetime(item["deleted_at"]),
    )


def _encode_retention(value: RetentionLimits) -> dict[str, object]:
    return {
        "storage": {
            "local": value.storage.local.value,
            "upstream": value.storage.upstream.value,
            "provider_storage_disclosed": (
                value.storage.provider_storage_disclosed
            ),
        },
        "upstream_lifetime_status": value.upstream_lifetime_status.value,
        "local_ttl_seconds": value.local_ttl_seconds,
        "envelope_ttl_seconds": value.envelope_ttl_seconds,
        "known_upstream_ttl_seconds": value.known_upstream_ttl_seconds,
    }


def _decode_retention(value: JsonValue) -> RetentionLimits:
    item = _mapping(
        value,
        {
            "storage",
            "upstream_lifetime_status",
            "local_ttl_seconds",
            "envelope_ttl_seconds",
            "known_upstream_ttl_seconds",
        },
    )
    storage = _mapping(
        item["storage"],
        {"local", "upstream", "provider_storage_disclosed"},
    )
    return RetentionLimits(
        storage=StoragePolicy(
            local=LocalResponseStorage(_string(storage["local"])),
            upstream=ProviderLaneStorage(_string(storage["upstream"])),
            provider_storage_disclosed=_boolean(
                storage["provider_storage_disclosed"]
            ),
        ),
        upstream_lifetime_status=UpstreamLifetimeStatus(
            _string(item["upstream_lifetime_status"])
        ),
        local_ttl_seconds=_optional_integer(item["local_ttl_seconds"]),
        envelope_ttl_seconds=_optional_integer(item["envelope_ttl_seconds"]),
        known_upstream_ttl_seconds=_optional_integer(
            item["known_upstream_ttl_seconds"]
        ),
    )


def _encode_head(value: NamedHeadMetadata | None) -> object:
    if value is None:
        return None
    return {"head_id": value.head_id, "revision": value.revision}


def _decode_head(value: JsonValue) -> NamedHeadMetadata | None:
    if value is None:
        return None
    item = _mapping(value, {"head_id", "revision"})
    return NamedHeadMetadata(
        head_id=NamedHeadId(_string(item["head_id"])),
        revision=NamedHeadRevision(_integer(item["revision"])),
    )


def _encode_integrity(value: CheckpointIntegrityMetadata | None) -> object:
    if value is None:
        return None
    return {
        "codec_version": value.codec_version,
        "digest": value.digest,
        "encoded_byte_count": value.encoded_byte_count,
    }


def _decode_integrity(value: JsonValue) -> CheckpointIntegrityMetadata | None:
    if value is None:
        return None
    item = _mapping(
        value,
        {"codec_version", "digest", "encoded_byte_count"},
    )
    return CheckpointIntegrityMetadata(
        codec_version=ConversationCodecVersion(
            _integer(item["codec_version"])
        ),
        digest=IntegrityDigest(_string(item["digest"])),
        encoded_byte_count=_integer(item["encoded_byte_count"]),
    )


def _verify_integrity(checkpoint: ConversationCheckpoint) -> None:
    if checkpoint.integrity is None:
        raise ConversationIntegrityError()
    payload = _encode_unsigned_checkpoint_payload(checkpoint)
    if (
        checkpoint.integrity.codec_version != CHECKPOINT_CODEC_VERSION
        or checkpoint.integrity.encoded_byte_count != len(payload)
        or checkpoint.integrity.digest
        != IntegrityDigest(sha256(payload).hexdigest())
    ):
        raise ConversationIntegrityError()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConversationCodecError()
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise ConversationCodecError()


def _mapping(
    value: JsonValue,
    keys: set[str],
) -> Mapping[str, JsonValue]:
    result = _mapping_unchecked(value)
    _exact_keys(result, keys)
    return result


def _mapping_unchecked(value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConversationCodecError()
    return value


def _exact_keys(value: Mapping[str, object], keys: set[str]) -> None:
    if set(value) != keys:
        raise ConversationCodecError()


def _sequence(value: JsonValue) -> tuple[JsonValue, ...]:
    if not isinstance(value, tuple):
        raise ConversationCodecError()
    return value


def _json(value: object) -> JsonValue:
    try:
        return freeze_json_value(value)
    except (ConversationLimitError, ConversationValidationError) as exc:
        raise ConversationCodecError() from exc


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ConversationCodecError()
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ConversationCodecError()
    return value


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    return _integer(value)


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ConversationCodecError()
    return value


def _datetime(value: object) -> datetime:
    text = _string(value)
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ConversationCodecError() from exc
    if result.utcoffset() is None or result.isoformat() != text:
        raise ConversationCodecError()
    return result


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)
