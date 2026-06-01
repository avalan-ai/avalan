from ..types import MutableJsonValue
from .definition import PrivacyAction, TaskPrivacyPolicy

from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from hmac import new as hmac_new
from json import dumps
from math import isfinite
from typing import Protocol, TypeAlias, cast

PrivacySafeValue: TypeAlias = MutableJsonValue

REDACTED_MARKER = "<redacted>"
DROPPED_MARKER = "<dropped>"
HASHED_MARKER = "<hmac-sha256>"
ENCRYPTED_MARKER = "<encrypted>"
STORED_MARKER = "<stored>"

_CANONICAL_JSON_SEPARATORS = (",", ":")
_COMMON_SAFE_FIELDS = frozenset(
    {
        "attempt",
        "attempt_number",
        "category",
        "code",
        "count",
        "counts",
        "created_at",
        "details",
        "duration_ms",
        "elapsed_ms",
        "ended_at",
        "error_category",
        "event_type",
        "failed_attempt_count",
        "finished_at",
        "hint",
        "issues",
        "max_attempts",
        "message",
        "name",
        "path",
        "retry_delay_seconds",
        "retry_exhausted",
        "retryable",
        "run_state",
        "scope",
        "severity",
        "started_at",
        "state",
        "status",
        "timestamp",
        "type",
    }
)
_EVENT_COMMON_SAFE_FIELDS = frozenset(
    _COMMON_SAFE_FIELDS
    - {
        "details",
        "hint",
        "issues",
        "message",
        "path",
    }
)
_TOKEN_EVENT_TYPES = frozenset(
    {
        "input_token_count_after",
        "input_token_count_before",
        "token_generated",
    }
)
_ENGINE_EVENT_TYPES = frozenset(
    {
        "call_prepare_after",
        "call_prepare_before",
        "end",
        "start",
        "stream_end",
    }
)
_EVENT_PREFIX_SAFE_FIELDS = (
    ("tool_", frozenset({"name"})),
    ("model_", frozenset({"name"})),
    ("engine_", frozenset({"name"})),
    ("memory_", frozenset({"name"})),
)
_POLICY_FIELD_ORDER = (
    "input",
    "prompt",
    "output",
    "files",
    "file_bytes",
    "token_text",
    "tool_arguments",
    "tool_results",
    "events",
    "errors",
)
_POLICY_FIELD_NAMES = frozenset(_POLICY_FIELD_ORDER)


class TaskKeyPurpose(StrEnum):
    IDEMPOTENCY = "idempotency"
    PRIVACY_HASH = "privacy_hash"
    ARTIFACT_CONTENT = "artifact_content"
    RAW_VALUE = "raw_value"


class PrivacyField(StrEnum):
    INPUT = "input"
    PROMPT = "prompt"
    OUTPUT = "output"
    FILES = "files"
    FILE_BYTES = "file_bytes"
    TOKEN_TEXT = "token_text"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_RESULTS = "tool_results"
    EVENTS = "events"
    ERRORS = "errors"


_RAW_PRIVACY_ACTIONS = frozenset(
    {
        PrivacyAction.ENCRYPT,
        PrivacyAction.STORE,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskKeyMaterial:
    key_id: str
    algorithm: str
    secret: bytes

    def __post_init__(self) -> None:
        assert isinstance(self.key_id, str) and self.key_id.strip()
        assert isinstance(self.algorithm, str) and self.algorithm.strip()
        assert isinstance(self.secret, bytes) and self.secret

    def metadata(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EncryptedPrivacyValue:
    ciphertext: bytes
    key_id: str
    algorithm: str
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.ciphertext, bytes) and self.ciphertext
        assert isinstance(self.key_id, str) and self.key_id.strip()
        assert isinstance(self.algorithm, str) and self.algorithm.strip()
        if self.metadata is not None:
            assert isinstance(self.metadata, Mapping)
            for key, value in self.metadata.items():
                assert isinstance(key, str) and key.strip()
                assert isinstance(value, str)

    def as_dict(self) -> dict[str, PrivacySafeValue]:
        value: dict[str, PrivacySafeValue] = {
            "algorithm": self.algorithm,
            "ciphertext": b64encode(self.ciphertext).decode("ascii"),
            "key_id": self.key_id,
            "privacy": ENCRYPTED_MARKER,
        }
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value


class HmacProvider(Protocol):
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial: ...


class EncryptionProvider(Protocol):
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue: ...


class PrivacySanitizationError(ValueError):
    pass


def privacy_policy_with_defaults(
    overrides: Mapping[str, PrivacyAction | str | int] | None = None,
) -> TaskPrivacyPolicy:
    if overrides is None:
        return TaskPrivacyPolicy()
    assert isinstance(overrides, Mapping), "overrides must be a mapping"
    values: dict[str, PrivacyAction | int] = {}
    for key, value in overrides.items():
        assert key in _POLICY_FIELD_NAMES or key == "raw_retention_days"
        if key == "raw_retention_days":
            assert isinstance(value, int)
            values[key] = value
        else:
            values[key] = _privacy_action(value)
    default_policy = TaskPrivacyPolicy()
    return TaskPrivacyPolicy(
        input=cast(PrivacyAction, values.get("input", default_policy.input)),
        prompt=cast(
            PrivacyAction, values.get("prompt", default_policy.prompt)
        ),
        output=cast(
            PrivacyAction, values.get("output", default_policy.output)
        ),
        files=cast(PrivacyAction, values.get("files", default_policy.files)),
        file_bytes=cast(
            PrivacyAction,
            values.get("file_bytes", default_policy.file_bytes),
        ),
        token_text=cast(
            PrivacyAction,
            values.get("token_text", default_policy.token_text),
        ),
        tool_arguments=cast(
            PrivacyAction,
            values.get("tool_arguments", default_policy.tool_arguments),
        ),
        tool_results=cast(
            PrivacyAction,
            values.get("tool_results", default_policy.tool_results),
        ),
        events=cast(
            PrivacyAction, values.get("events", default_policy.events)
        ),
        errors=cast(
            PrivacyAction, values.get("errors", default_policy.errors)
        ),
        raw_retention_days=cast(
            int,
            values.get(
                "raw_retention_days",
                default_policy.raw_retention_days,
            ),
        ),
    )


def privacy_policy_fields(
    policy: TaskPrivacyPolicy,
) -> dict[str, PrivacyAction | object]:
    assert isinstance(policy, TaskPrivacyPolicy)
    return {
        field_name: getattr(policy, field_name)
        for field_name in _POLICY_FIELD_ORDER
    }


def privacy_policy_hash_fields(
    policy: TaskPrivacyPolicy,
) -> tuple[str, ...]:
    fields = privacy_policy_fields(policy)
    return tuple(
        field_name
        for field_name, action in fields.items()
        if action == PrivacyAction.HASH
    )


def privacy_policy_raw_fields(
    policy: TaskPrivacyPolicy,
) -> tuple[str, ...]:
    fields = privacy_policy_fields(policy)
    return tuple(
        field_name
        for field_name, action in fields.items()
        if action in _RAW_PRIVACY_ACTIONS
    )


def privacy_policy_store_fields(
    policy: TaskPrivacyPolicy,
) -> tuple[str, ...]:
    fields = privacy_policy_fields(policy)
    return tuple(
        field_name
        for field_name, action in fields.items()
        if action == PrivacyAction.STORE
    )


class PrivacySanitizer:
    policy: TaskPrivacyPolicy
    hmac_provider: HmacProvider | None
    encryption_provider: EncryptionProvider | None
    raw_storage_allowed: bool
    event_allowlists: Mapping[str, frozenset[str]]

    def __init__(
        self,
        policy: TaskPrivacyPolicy | None = None,
        *,
        hmac_provider: HmacProvider | None = None,
        encryption_provider: EncryptionProvider | None = None,
        raw_storage_allowed: bool = False,
        event_allowlists: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.policy = policy or TaskPrivacyPolicy()
        self.hmac_provider = hmac_provider
        self.encryption_provider = encryption_provider
        self.raw_storage_allowed = raw_storage_allowed
        self.event_allowlists = _freeze_allowlists(event_allowlists or {})

    def sanitize(
        self,
        field: PrivacyField | str,
        value: object,
        *,
        key_id: str | None = None,
    ) -> PrivacySafeValue:
        privacy_field = _privacy_field(field)
        action = cast(PrivacyAction, getattr(self.policy, privacy_field.value))
        return self.sanitize_with_action(action, value, key_id=key_id)

    def sanitize_event(
        self,
        event_type: str,
        payload: Mapping[str, object] | None,
        *,
        key_id: str | None = None,
    ) -> PrivacySafeValue:
        assert isinstance(event_type, str) and event_type.strip()
        action = self.policy.events
        if action != PrivacyAction.REDACT:
            event = {"event_type": event_type, "payload": payload or {}}
            return self.sanitize_with_action(action, event, key_id=key_id)
        return self._redact_event(event_type, payload or {})

    def sanitize_with_action(
        self,
        action: PrivacyAction,
        value: object,
        *,
        key_id: str | None = None,
    ) -> PrivacySafeValue:
        assert isinstance(action, PrivacyAction)
        match action:
            case PrivacyAction.DROP:
                return {"privacy": DROPPED_MARKER}
            case PrivacyAction.HASH:
                return self._hash(value, key_id=key_id)
            case PrivacyAction.REDACT:
                return self._redact(value)
            case PrivacyAction.STORE:
                return self._store(value)
            case PrivacyAction.ENCRYPT:
                return self._encrypt(value, key_id=key_id)

    def _hash(
        self, value: object, *, key_id: str | None
    ) -> dict[str, PrivacySafeValue]:
        if self.hmac_provider is None:
            raise PrivacySanitizationError("privacy HMAC key is unavailable")
        key = self.hmac_provider.hmac_key(
            purpose=TaskKeyPurpose.PRIVACY_HASH,
            key_id=key_id,
        )
        payload = _private_value_bytes(value)
        digest = hmac_new(key.secret, payload, sha256).hexdigest()
        return {
            "algorithm": key.algorithm,
            "digest": digest,
            "key_id": key.key_id,
            "privacy": HASHED_MARKER,
        }

    def _redact(self, value: object) -> PrivacySafeValue:
        if isinstance(value, Mapping):
            redacted: dict[str, PrivacySafeValue] = {}
            for key, item in value.items():
                if isinstance(key, str) and key in _COMMON_SAFE_FIELDS:
                    redacted[key] = _safe_metadata_value(item)
            if redacted:
                return redacted
        return {"privacy": REDACTED_MARKER}

    def _redact_event(
        self,
        event_type: str,
        payload: Mapping[str, object],
    ) -> dict[str, PrivacySafeValue]:
        allowed_fields = self.event_allowlists.get(event_type)
        if allowed_fields is None:
            allowed_fields = _default_event_allowed_fields(event_type)
        else:
            allowed_fields = frozenset(
                allowed_fields | _default_event_allowed_fields(event_type)
            )
        redacted: dict[str, PrivacySafeValue] = {"event_type": event_type}
        for key, item in payload.items():
            if key in allowed_fields:
                redacted[key] = _safe_event_metadata_value(item)
        return redacted

    def _store(self, value: object) -> dict[str, PrivacySafeValue]:
        if not self.raw_storage_allowed or self.policy.raw_retention_days <= 0:
            raise PrivacySanitizationError("raw privacy storage is disabled")
        return {
            "privacy": STORED_MARKER,
            "value": _private_json_value(value),
        }

    def _encrypt(
        self, value: object, *, key_id: str | None
    ) -> dict[str, PrivacySafeValue]:
        if self.encryption_provider is None:
            raise PrivacySanitizationError(
                "privacy encryption key is unavailable"
            )
        if not self.raw_storage_allowed or self.policy.raw_retention_days <= 0:
            raise PrivacySanitizationError("raw privacy storage is disabled")
        encrypted = self.encryption_provider.encrypt(
            _private_value_bytes(value),
            purpose=TaskKeyPurpose.RAW_VALUE,
            key_id=key_id,
        )
        return encrypted.as_dict()


def _privacy_action(value: PrivacyAction | str | int) -> PrivacyAction:
    if isinstance(value, PrivacyAction):
        return value
    assert isinstance(value, str), "privacy action must be a string"
    return PrivacyAction(value)


def _privacy_field(value: PrivacyField | str) -> PrivacyField:
    if isinstance(value, PrivacyField):
        return value
    assert isinstance(value, str), "privacy field must be a string"
    return PrivacyField(value)


def _freeze_allowlists(
    allowlists: Mapping[str, tuple[str, ...]],
) -> Mapping[str, frozenset[str]]:
    frozen: dict[str, frozenset[str]] = {}
    for event_type, fields in allowlists.items():
        assert isinstance(event_type, str) and event_type.strip()
        assert isinstance(fields, tuple)
        frozen[event_type] = frozenset(fields)
    return frozen


def _default_event_allowed_fields(event_type: str) -> frozenset[str]:
    allowed_fields = _EVENT_COMMON_SAFE_FIELDS
    if event_type in _TOKEN_EVENT_TYPES:
        return allowed_fields
    if event_type in _ENGINE_EVENT_TYPES:
        return allowed_fields
    for prefix, fields in _EVENT_PREFIX_SAFE_FIELDS:
        if event_type.startswith(prefix):
            return frozenset(allowed_fields | fields)
    return allowed_fields


def _private_value_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return dumps(
        _private_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=_CANONICAL_JSON_SEPARATORS,
        sort_keys=True,
    ).encode("utf-8")


def _private_json_value(value: object) -> PrivacySafeValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    if isinstance(value, bytes | bytearray):
        return {
            "encoding": "base64",
            "value": b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Mapping):
        private_value: dict[str, PrivacySafeValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PrivacySanitizationError(
                    "privacy value contains a non-string key"
                )
            private_value[key] = _private_json_value(item)
        return private_value
    if isinstance(value, list | tuple):
        return [_private_json_value(item) for item in value]
    raise PrivacySanitizationError("privacy value is not JSON-compatible")


def _safe_metadata_value(value: object) -> PrivacySafeValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if isfinite(value):
            return value
        return REDACTED_MARKER
    if isinstance(value, Mapping):
        safe: dict[str, PrivacySafeValue] = {}
        for key, item in value.items():
            if isinstance(key, str) and key in _COMMON_SAFE_FIELDS:
                safe[key] = _safe_metadata_value(item)
        if safe:
            return safe
        return REDACTED_MARKER
    if isinstance(value, list | tuple):
        return [_safe_metadata_value(item) for item in value]
    return REDACTED_MARKER


def _safe_event_metadata_value(value: object) -> PrivacySafeValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if isfinite(value):
            return value
        return REDACTED_MARKER
    if isinstance(value, Mapping):
        safe: dict[str, PrivacySafeValue] = {}
        for key, item in value.items():
            if isinstance(key, str) and key in _EVENT_COMMON_SAFE_FIELDS:
                safe[key] = _safe_event_metadata_value(item)
        if safe:
            return safe
        return REDACTED_MARKER
    if isinstance(value, list | tuple):
        return [_safe_event_metadata_value(item) for item in value]
    return REDACTED_MARKER
