"""Validate immutable JSON and redaction-safe conversation values."""

from ..types import JsonValue
from .errors import ConversationLimitError, ConversationValidationError

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from types import MappingProxyType
from typing import NewType, final

ProviderItemId = NewType("ProviderItemId", str)
ProviderCallId = NewType("ProviderCallId", str)
CapabilityProfileId = NewType("CapabilityProfileId", str)
CapabilityProfileRevision = NewType("CapabilityProfileRevision", str)
ProviderApiRevision = NewType("ProviderApiRevision", str)
ProviderSdkRevision = NewType("ProviderSdkRevision", str)
ModelConfigurationRevision = NewType("ModelConfigurationRevision", str)
ToolSchemaRevision = NewType("ToolSchemaRevision", str)
ExecutionDefinitionRevision = NewType("ExecutionDefinitionRevision", str)
ConversationCodecVersion = NewType("ConversationCodecVersion", int)
ProviderItemIndex = NewType("ProviderItemIndex", int)
ProviderItemOrder = NewType("ProviderItemOrder", int)
IntegrityDigest = NewType("IntegrityDigest", str)
AuthorityDigest = NewType("AuthorityDigest", str)
RequestSemanticDigest = NewType("RequestSemanticDigest", str)
SafeAlias = NewType("SafeAlias", str)

_MAX_IDENTIFIER_LENGTH = 512


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class JsonLimits:
    """Bound immutable JSON validation before recursive allocation."""

    max_depth: int = 32
    max_items: int = 10_000
    max_string_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        for value in (self.max_depth, self.max_items, self.max_string_bytes):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OpaqueProviderState:
    """Hold provider-owned bytes without rendering their content."""

    _value: bytes

    def __post_init__(self) -> None:
        if type(self._value) is not bytes or not self._value:
            raise ConversationValidationError()

    @property
    def byte_count(self) -> int:
        """Return the opaque byte count without revealing content."""
        return len(self._value)

    @property
    def digest(self) -> IntegrityDigest:
        """Return a one-way digest of the exact opaque bytes."""
        return IntegrityDigest(sha256(self._value).hexdigest())

    def __repr__(self) -> str:
        """Return a redacted representation."""
        return (
            f"OpaqueProviderState(byte_count={self.byte_count}, redacted=True)"
        )

    def __str__(self) -> str:
        """Return a redacted display value."""
        return "<opaque-provider-state>"

    def _codec_bytes(self) -> bytes:
        """Return bytes only to the owned checkpoint codec."""
        return self._value


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CallerHeldState:
    """Hold authenticated caller state without rendering the token."""

    _value: str

    def __post_init__(self) -> None:
        validate_identifier(
            self._value, "caller-held state", max_length=65_536
        )

    @property
    def byte_count(self) -> int:
        """Return the UTF-8 token size without revealing content."""
        return len(self._value.encode("utf-8"))

    @property
    def digest(self) -> IntegrityDigest:
        """Return a one-way digest of the caller-held token."""
        return IntegrityDigest(sha256(self._value.encode("utf-8")).hexdigest())

    def __repr__(self) -> str:
        """Return a redacted representation."""
        return f"CallerHeldState(byte_count={self.byte_count}, redacted=True)"

    def __str__(self) -> str:
        """Return a redacted display value."""
        return "<caller-held-state>"

    def _codec_text(self) -> str:
        """Return text only to an owned envelope codec."""
        return self._value


def validate_identifier(
    value: object,
    field_name: str,
    *,
    max_length: int = _MAX_IDENTIFIER_LENGTH,
) -> str:
    """Return one bounded, normalized identifier."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > max_length
    ):
        raise ConversationValidationError()
    return value


def validate_revision(value: object, field_name: str) -> int:
    """Return one non-negative integer revision."""
    if type(value) is not int or value < 0:
        raise ConversationValidationError()
    return value


def freeze_json_value(
    value: object,
    *,
    limits: JsonLimits = JsonLimits(),
) -> JsonValue:
    """Return deeply immutable, finite, bounded JSON data."""
    if type(limits) is not JsonLimits:
        raise ConversationValidationError()
    remaining = [limits.max_items]
    return _freeze_json_value(value, limits, remaining, 0)


def thaw_json_value(value: JsonValue) -> object:
    """Return a mutable JSON-compatible tree for canonical encoding."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Return deterministic UTF-8 JSON for one validated value."""
    return dumps(
        thaw_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_digest(value: JsonValue) -> IntegrityDigest:
    """Return the SHA-256 digest of canonical validated JSON."""
    return IntegrityDigest(sha256(canonical_json_bytes(value)).hexdigest())


def _freeze_json_value(
    value: object,
    limits: JsonLimits,
    remaining: list[int],
    depth: int,
) -> JsonValue:
    if depth > limits.max_depth:
        raise ConversationLimitError()
    remaining[0] -= 1
    if remaining[0] < 0:
        raise ConversationLimitError()
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ConversationValidationError()
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > limits.max_string_bytes:
            raise ConversationLimitError()
        return value
    if isinstance(value, tuple | list):
        return tuple(
            _freeze_json_value(item, limits, remaining, depth + 1)
            for item in value
        )
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConversationValidationError()
            if key in frozen:
                raise ConversationValidationError()
            frozen[key] = _freeze_json_value(
                item,
                limits,
                remaining,
                depth + 1,
            )
        return MappingProxyType(frozen)
    raise ConversationValidationError()
