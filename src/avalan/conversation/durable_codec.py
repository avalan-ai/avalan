"""Encode encrypted durable conversation payload contents strictly."""

from ..interaction.entities import (
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    StateRevision,
)
from ..types import JsonValue
from .codec import (
    _decode_binding,
    _decode_execution_receipt,
    _decode_item,
    _decode_reasoning,
    _encode_binding,
    _encode_execution_receipt,
    _encode_item,
    _encode_reasoning,
)
from .contract import (
    ContinuationDigest,
    PortableContinuationReference,
    UpstreamResponseId,
)
from .errors import (
    ConversationCodecError,
    ConversationLimitError,
    ConversationValidationError,
)
from .execution import ConversationExecutionReservation
from .runtime import ProviderLaneOutputCandidate
from .settings import (
    ConversationMode,
    ProviderLaneOutputScope,
    ProviderUsage,
)
from .value import (
    JsonLimits,
    canonical_json_bytes,
    freeze_json_value,
    thaw_json_value,
)

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from typing import final

DURABLE_PAYLOAD_CODEC_VERSION = 1
_OUTPUT_KIND = "avalan.conversation.lane-output"
_CONTINUATION_KIND = "avalan.conversation.continuation-reference"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableConversationCodecLimits:
    """Bound every standalone encrypted durable payload codec."""

    max_bytes: int = 8_388_608
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
        """Return recursive bounds for one payload."""
        return JsonLimits(
            max_depth=self.max_depth,
            max_items=self.max_items,
            max_string_bytes=self.max_string_bytes,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableConversationCodec:
    """Round-trip encrypted durable auxiliary payloads canonically."""

    limits: DurableConversationCodecLimits = DurableConversationCodecLimits()

    def __post_init__(self) -> None:
        if type(self.limits) is not DurableConversationCodecLimits:
            raise ConversationValidationError()

    def encode_output(self, value: ProviderLaneOutputCandidate) -> bytes:
        """Encode one private provider-lane output candidate."""
        if type(value) is not ProviderLaneOutputCandidate:
            raise ConversationValidationError()
        return self._encode(
            {
                "binding": _encode_binding(value.binding),
                "execution_receipt": _encode_execution_receipt(
                    value.execution_receipt
                ),
                "items": [
                    _encode_item(item) for item in value.completed_items
                ],
                "kind": _OUTPUT_KIND,
                "mode": value.mode.value,
                "reasoning": _encode_reasoning(value.reasoning),
                "scope": value.scope.value,
                "upstream_response_id": value.upstream_response_id,
                "usage": {
                    "input_tokens": value.usage.input_tokens,
                    "output_tokens": value.usage.output_tokens,
                },
                "version": DURABLE_PAYLOAD_CODEC_VERSION,
            }
        )

    def decode_output(self, encoded: bytes) -> ProviderLaneOutputCandidate:
        """Decode one private provider-lane output candidate."""
        value = self._decode(encoded)
        item = _exact_mapping(
            value,
            {
                "binding",
                "execution_receipt",
                "items",
                "kind",
                "mode",
                "reasoning",
                "scope",
                "upstream_response_id",
                "usage",
                "version",
            },
        )
        if (
            _string(item["kind"]) != _OUTPUT_KIND
            or _integer(item["version"]) != DURABLE_PAYLOAD_CODEC_VERSION
        ):
            raise ConversationCodecError()
        binding = _decode_binding(item["binding"])
        items = tuple(
            _decode_item(value) for value in _sequence(item["items"])
        )
        usage = _exact_mapping(
            item["usage"], {"input_tokens", "output_tokens"}
        )
        upstream = _optional_string(item["upstream_response_id"])
        try:
            result = ProviderLaneOutputCandidate(
                lane_id=binding.lane_id,
                binding=binding,
                mode=ConversationMode(_string(item["mode"])),
                scope=ProviderLaneOutputScope(_string(item["scope"])),
                completed_items=items,
                reasoning=_decode_reasoning(item["reasoning"]),
                usage=ProviderUsage(
                    input_tokens=_integer(usage["input_tokens"]),
                    output_tokens=_integer(usage["output_tokens"]),
                ),
                execution_receipt=_decode_execution_receipt(
                    item["execution_receipt"]
                ),
                upstream_response_id=(
                    UpstreamResponseId(upstream)
                    if upstream is not None
                    else None
                ),
            )
        except (ConversationValidationError, ValueError) as exc:
            raise ConversationCodecError() from exc
        if self.encode_output(result) != encoded:
            raise ConversationCodecError()
        return result

    def encode_continuation_reference(
        self,
        value: PortableContinuationReference,
    ) -> bytes:
        """Encode one structured-input reference without provider snapshots."""
        if type(value) is not PortableContinuationReference:
            raise ConversationValidationError()
        return self._encode(
            {
                "continuation_id": value.continuation_id,
                "definition": _encode_definition(value.definition),
                "digest": value.digest,
                "kind": _CONTINUATION_KIND,
                "revision_binding": _encode_revision_binding(
                    value.revision_binding
                ),
                "state_revision": value.state_revision,
                "version": DURABLE_PAYLOAD_CODEC_VERSION,
            }
        )

    def decode_continuation_reference(
        self,
        encoded: bytes,
    ) -> PortableContinuationReference:
        """Decode one exact structured-input continuation reference."""
        value = self._decode(encoded)
        item = _exact_mapping(
            value,
            {
                "continuation_id",
                "definition",
                "digest",
                "kind",
                "revision_binding",
                "state_revision",
                "version",
            },
        )
        if (
            _string(item["kind"]) != _CONTINUATION_KIND
            or _integer(item["version"]) != DURABLE_PAYLOAD_CODEC_VERSION
        ):
            raise ConversationCodecError()
        try:
            result = PortableContinuationReference(
                continuation_id=ContinuationId(
                    _string(item["continuation_id"])
                ),
                state_revision=StateRevision(_integer(item["state_revision"])),
                digest=ContinuationDigest(_string(item["digest"])),
                definition=_decode_definition(item["definition"]),
                revision_binding=_decode_revision_binding(
                    item["revision_binding"]
                ),
            )
        except (AssertionError, ValueError) as exc:
            raise ConversationCodecError() from exc
        if self.encode_continuation_reference(result) != encoded:
            raise ConversationCodecError()
        return result

    def _encode(self, value: object) -> bytes:
        try:
            frozen = freeze_json_value(value, limits=self.limits.json_limits)
            encoded = dumps(
                thaw_json_value(frozen),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (ConversationLimitError, ConversationValidationError):
            raise
        except (TypeError, ValueError) as exc:
            raise ConversationCodecError() from exc
        if not encoded or len(encoded) > self.limits.max_bytes:
            raise ConversationLimitError()
        return encoded

    def _decode(self, encoded: bytes) -> JsonValue:
        if type(encoded) is not bytes or not encoded:
            raise ConversationCodecError()
        if len(encoded) > self.limits.max_bytes:
            raise ConversationLimitError()
        try:
            text = encoded.decode("utf-8")
            raw: object = loads(text, object_pairs_hook=_unique_object)
        except (JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConversationCodecError() from exc
        value = freeze_json_value(raw, limits=self.limits.json_limits)
        if self._encode(value) != encoded:
            raise ConversationCodecError()
        return value


def execution_reservation_digest(
    value: ConversationExecutionReservation | None,
) -> str | None:
    """Return the exact canonical execution-reservation digest."""
    if value is None:
        return None
    if type(value) is not ConversationExecutionReservation:
        raise ConversationValidationError()
    identity = value.identity
    payload = freeze_json_value(
        {
            "checkpoint_identity": {
                "branch_id": identity.branch_id,
                "checkpoint_id": identity.checkpoint_id,
                "conversation_id": identity.conversation_id,
                "execution_segment_id": identity.execution_segment_id,
                "logical_turn_id": identity.logical_turn_id,
                "parent_checkpoint_id": identity.parent_checkpoint_id,
                "parent_sequence": identity.parent_sequence,
                "sequence": identity.sequence,
            },
            "lanes": tuple(
                {
                    "binding_digest": lane.binding.integrity_digest,
                    "lane_id": lane.binding.lane_id,
                    "mode": lane.mode.value,
                    "scope": lane.scope.value,
                }
                for lane in value.lanes
            ),
            "schema_version": 1,
        }
    )
    return sha256(canonical_json_bytes(payload)).hexdigest()


def continuation_definition_digest(value: ExecutionDefinitionRef) -> str:
    """Return a stable digest of one resolvable execution definition."""
    if type(value) is not ExecutionDefinitionRef:
        raise ConversationValidationError()
    return sha256(
        canonical_json_bytes(freeze_json_value(_encode_definition(value)))
    ).hexdigest()


def continuation_revision_binding_digest(
    value: ContinuationRevisionBinding,
) -> str:
    """Return a stable digest of one continuation revision binding."""
    if type(value) is not ContinuationRevisionBinding:
        raise ConversationValidationError()
    return sha256(
        canonical_json_bytes(
            freeze_json_value(_encode_revision_binding(value))
        )
    ).hexdigest()


def _encode_definition(value: ExecutionDefinitionRef) -> dict[str, object]:
    return {
        "agent_definition_locator": value.agent_definition_locator,
        "agent_definition_revision": value.agent_definition_revision,
        "operation_id": value.operation_id,
        "operation_index": value.operation_index,
        "model_config_reference": value.model_config_reference,
        "tool_revision": value.tool_revision,
        "capability_revision": value.capability_revision,
    }


def _decode_definition(value: JsonValue) -> ExecutionDefinitionRef:
    item = _exact_mapping(
        value,
        {
            "agent_definition_locator",
            "agent_definition_revision",
            "operation_id",
            "operation_index",
            "model_config_reference",
            "tool_revision",
            "capability_revision",
        },
    )
    return ExecutionDefinitionRef(
        agent_definition_locator=_string(item["agent_definition_locator"]),
        agent_definition_revision=_string(item["agent_definition_revision"]),
        operation_id=_string(item["operation_id"]),
        operation_index=_integer(item["operation_index"]),
        model_config_reference=_string(item["model_config_reference"]),
        tool_revision=_string(item["tool_revision"]),
        capability_revision=_string(item["capability_revision"]),
    )


def _encode_revision_binding(
    value: ContinuationRevisionBinding,
) -> dict[str, object]:
    return {
        "provider_family": value.provider_family,
        "model_id": value.model_id,
        "provider_config_revision": value.provider_config_revision,
        "model_config_revision": value.model_config_revision,
        "capability_revision": value.capability_revision,
    }


def _decode_revision_binding(value: JsonValue) -> ContinuationRevisionBinding:
    item = _exact_mapping(
        value,
        {
            "provider_family",
            "model_id",
            "provider_config_revision",
            "model_config_revision",
            "capability_revision",
        },
    )
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName(_string(item["provider_family"])),
        model_id=ModelId(_string(item["model_id"])),
        provider_config_revision=ProviderConfigRevision(
            _string(item["provider_config_revision"])
        ),
        model_config_revision=ModelConfigRevision(
            _string(item["model_config_revision"])
        ),
        capability_revision=CapabilityRevision(
            _string(item["capability_revision"])
        ),
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ConversationCodecError()
        value[key] = item
    return value


def _exact_mapping(
    value: JsonValue,
    keys: set[str],
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConversationCodecError()
    return value


def _sequence(value: JsonValue) -> tuple[JsonValue, ...]:
    if not isinstance(value, tuple):
        raise ConversationCodecError()
    return value


def _string(value: JsonValue) -> str:
    if not isinstance(value, str) or not value:
        raise ConversationCodecError()
    return value


def _optional_string(value: JsonValue) -> str | None:
    if value is None:
        return None
    return _string(value)


def _integer(value: JsonValue) -> int:
    if type(value) is not int or value < 0:
        raise ConversationCodecError()
    return value
