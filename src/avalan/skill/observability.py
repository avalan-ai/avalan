from ..entities import ToolCallContext
from ..event import Event, EventObservabilityPayload, EventType
from ..types import LooseJsonValue
from ._async import (
    DEFAULT_SKILL_IO_TIMEOUT_SECONDS,
    skill_bounded_await,
    skill_cancellation_checkpoint,
)
from .contract import SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillProvenance,
    SkillRegistryVersion,
    SkillResourceContent,
    SkillSourceAuthority,
    SkillSourceAuthorityKind,
    model_dict,
)
from .path_policy import (
    sanitize_skill_resource_id,
    sanitize_skill_source_label,
)
from .settings import SKILL_SETTINGS_POLICY_VERSION, TrustedSkillSettings

from collections.abc import Awaitable, Mapping
from enum import StrEnum
from hashlib import sha256
from inspect import isawaitable
from json import dumps
from re import fullmatch, split
from secrets import token_hex
from typing import Protocol, cast, runtime_checkable
from uuid import UUID

SKILL_AUDIT_EVENT_SCHEMA = "skills.audit.v1"
SKILL_AUDIT_HASH_PREFIX_LENGTH = 16
SKILL_AUDIT_MAX_PAYLOAD_BYTES = 4096
_SKILL_AUDIT_MAX_TEXT_BYTES = 512
_SKILL_AUDIT_COMPACT_TEXT_BYTES = 192
_SKILL_AUDIT_MINIMAL_KEYS = (
    "schema",
    "event_type",
    "policy_version",
    "status",
    "payload_truncated",
)


@runtime_checkable
class SkillEventPublisher(Protocol):
    def trigger(self, event: Event) -> Awaitable[None]:
        pass  # pragma: no cover


class SkillAuditDeliveryError(RuntimeError):
    """Signal critical skill audit delivery failure."""


def assert_skill_event_publisher(
    publisher: object | None,
) -> None:
    """Assert a publisher exposes the skill audit trigger shape."""
    if publisher is None:
        return
    trigger = getattr(publisher, "trigger", None)
    assert callable(trigger)


def skill_audit_correlation_id(prefix: str) -> str:
    assert isinstance(prefix, str)
    assert fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", prefix)
    return f"{prefix}:{token_hex(8)}"


async def emit_skill_audit_event(
    publisher: SkillEventPublisher | None,
    settings: TrustedSkillSettings | None,
    event_type: EventType,
    fields: Mapping[str, object],
    *,
    delivery_timeout_seconds: float | None = DEFAULT_SKILL_IO_TIMEOUT_SECONDS,
) -> None:
    assert_skill_event_publisher(publisher)
    assert settings is None or isinstance(settings, TrustedSkillSettings)
    assert isinstance(event_type, EventType)
    assert isinstance(fields, Mapping)
    # Fail-closed applies to configured critical delivery failures. When no
    # publisher is configured there is no operator audit endpoint to require.
    if publisher is None or not skill_audit_events_enabled(settings):
        return

    await skill_cancellation_checkpoint()
    data = skill_audit_payload_data(event_type, fields, settings=settings)
    event = Event.from_observability_payload(
        type=event_type,
        observability_payload=EventObservabilityPayload.canonical_stream(data),
    )
    try:
        await skill_cancellation_checkpoint()
        result = publisher.trigger(event)
        assert isawaitable(result)
        await skill_bounded_await(
            result,
            timeout_seconds=delivery_timeout_seconds,
        )
        await skill_cancellation_checkpoint()
    except Exception:
        if skill_audit_fail_closed(settings):
            raise SkillAuditDeliveryError(
                "Critical skill audit delivery failed."
            ) from None


def skill_audit_events_enabled(
    settings: TrustedSkillSettings | None,
) -> bool:
    if settings is None:
        return True
    observability = settings.observability
    return observability.enabled and observability.emit_events


def skill_audit_fail_closed(settings: TrustedSkillSettings | None) -> bool:
    return (
        settings is not None
        and settings.observability.enabled
        and settings.observability.emit_events
        and settings.observability.audit_fail_closed
    )


def skill_audit_payload_data(
    event_type: EventType,
    fields: Mapping[str, object],
    *,
    settings: TrustedSkillSettings | None = None,
) -> dict[str, LooseJsonValue]:
    assert isinstance(event_type, EventType)
    assert isinstance(fields, Mapping)
    assert settings is None or isinstance(settings, TrustedSkillSettings)
    raw_data: dict[str, object] = {
        "schema": SKILL_AUDIT_EVENT_SCHEMA,
        "event_type": event_type.value,
        "policy_version": SKILL_SETTINGS_POLICY_VERSION,
    }
    for key, value in fields.items():
        assert isinstance(key, str)
        assert fullmatch(r"[a-z][a-z0-9_]*", key) is not None
        if value is None:
            continue
        audit_value = _audit_field_value(key, value, settings)
        if audit_value is None:
            continue
        raw_data[key] = audit_value
    data = _json_safe_model_dict(raw_data)
    return _bounded_payload(data)


def skill_audit_context_fields(
    context: ToolCallContext,
    *,
    tool_name: str,
) -> dict[str, object]:
    assert isinstance(context, ToolCallContext)
    assert isinstance(tool_name, str)
    fields: dict[str, object] = {}
    if context.agent_id is not None:
        fields["agent_id"] = _uuid_text(context.agent_id)
    if context.session_id is not None:
        fields["session_id"] = _uuid_text(context.session_id)
    tool_call_id = _tool_call_id(context, tool_name)
    if tool_call_id is not None:
        fields["tool_call_id"] = tool_call_id
    return fields


def skill_audit_registry_fields(
    registry_version: SkillRegistryVersion,
    *,
    status: SkillStatus | None = None,
) -> dict[str, object]:
    assert isinstance(registry_version, SkillRegistryVersion)
    fields: dict[str, object] = {
        "registry_version": registry_version.as_model_value()
    }
    if status is not None:
        fields["status"] = status.value
    return fields


def skill_audit_authority_value(
    authority: SkillSourceAuthority | SkillSourceAuthorityKind | None,
) -> str | None:
    if authority is None:
        return None
    if isinstance(authority, SkillSourceAuthority):
        return authority.kind.value
    assert isinstance(authority, SkillSourceAuthorityKind)
    return authority.value


def skill_audit_diagnostic_fields(
    diagnostic: SkillDiagnosticInfo | None,
) -> dict[str, object]:
    if diagnostic is None:
        return {}
    assert isinstance(diagnostic, SkillDiagnosticInfo)
    return {
        "status": diagnostic.status.value,
        "diagnostic_code": diagnostic.code.value,
    }


def skill_audit_diagnostics_fields(
    diagnostics: tuple[SkillDiagnosticInfo, ...],
) -> dict[str, object]:
    assert isinstance(diagnostics, tuple)
    for diagnostic in diagnostics:
        assert isinstance(diagnostic, SkillDiagnosticInfo)
    if not diagnostics:
        return {"diagnostic_count": 0}
    return {
        "status": diagnostics[0].status.value,
        "diagnostic_code": diagnostics[0].code.value,
        "diagnostic_count": len(diagnostics),
        "diagnostic_codes": [
            diagnostic.code.value for diagnostic in diagnostics[:8]
        ],
    }


def skill_audit_content_fields(
    content: SkillResourceContent,
    provenance: SkillProvenance | None,
) -> dict[str, object]:
    assert isinstance(content, SkillResourceContent)
    assert provenance is None or isinstance(provenance, SkillProvenance)
    fields: dict[str, object] = {
        "source_label": content.handle.source_label,
        "skill_id": content.handle.skill_id,
        "resource_id": content.handle.resource_id,
        "status": content.handle.status.value,
        "size_bytes": content.handle.size_bytes,
        "start_byte": content.start_byte,
        "end_byte": content.end_byte,
        "truncated": content.truncated,
        "stale": content.handle.stale,
    }
    if provenance is not None:
        fields["source_authority"] = skill_audit_authority_value(
            provenance.authority
        )
        fields["hash_prefix"] = skill_audit_hash_prefix(
            provenance.content_sha256_prefix
        )
    return fields


def skill_audit_hash_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    assert isinstance(value, str)
    if fullmatch(r"[a-f0-9]{8,64}", value) is None:
        return None
    return value[:SKILL_AUDIT_HASH_PREFIX_LENGTH]


def _tool_call_id(
    context: ToolCallContext,
    tool_name: str,
) -> str | None:
    calls = context.calls or []
    for call in reversed(calls):
        if call.name != tool_name:
            continue
        if call.id is None:
            return None
        return str(call.id)
    return None


def _uuid_text(value: UUID | str) -> str:
    assert isinstance(value, UUID | str)
    return str(value)


def _audit_field_value(
    key: str,
    value: object,
    settings: TrustedSkillSettings | None,
) -> object | None:
    assert isinstance(key, str)
    assert settings is None or isinstance(settings, TrustedSkillSettings)
    if key == "source_label":
        if not _include_source_labels(settings) or not isinstance(value, str):
            return None
        return _audit_value(sanitize_skill_source_label(value))
    if key == "source_labels":
        if not _include_source_labels(settings):
            return None
        return _audit_value(
            tuple(
                sanitize_skill_source_label(item)
                for item in _string_items(value)
            )
        )
    if key == "source_id":
        if not _include_source_labels(settings) or not isinstance(value, str):
            return None
        label = value.removeprefix("source:")
        return _audit_value(f"source:{sanitize_skill_source_label(label)}")
    if key == "source_authority" and not _include_source_authority(settings):
        return None
    if key == "resource_id":
        if not isinstance(value, str):
            return None
        return _audit_value(sanitize_skill_resource_id(value))
    if key == "resource_ids":
        return _audit_value(
            tuple(
                sanitize_skill_resource_id(item)
                for item in _string_items(value)
            )
        )
    if key == "skill_id":
        if not isinstance(value, str):
            return None
        return _audit_value(_sanitize_skill_audit_id(value))
    if key == "skill_ids":
        return _audit_value(
            tuple(
                _sanitize_skill_audit_id(item) for item in _string_items(value)
            )
        )
    if key in {
        "agent_id",
        "session_id",
        "tool_call_id",
        "operation_id",
    }:
        if isinstance(value, UUID):
            return str(value)
        if not isinstance(value, str):
            return None
        return _audit_value(_sanitize_audit_identifier(value, prefix="id"))
    return _audit_value(value)


def _audit_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        return _limit_text(_safe_text(value), _SKILL_AUDIT_MAX_TEXT_BYTES)
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_audit_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            assert isinstance(key, str)
            result[key] = _audit_value(item)
        return result
    return _limit_text(_safe_text(str(value)), _SKILL_AUDIT_MAX_TEXT_BYTES)


def _include_source_labels(settings: TrustedSkillSettings | None) -> bool:
    return settings is None or settings.privacy.include_source_labels


def _include_source_authority(settings: TrustedSkillSettings | None) -> bool:
    return settings is None or settings.privacy.include_authority


def _string_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _sanitize_skill_audit_id(value: str) -> str:
    assert isinstance(value, str)
    normalized = value.strip().lower()
    sanitized = sanitize_skill_source_label(value)
    if sanitized == normalized:
        return sanitized
    if sanitized.startswith("source-"):
        return f"skill-{sha256(value.encode('utf-8')).hexdigest()[:16]}"
    return sanitized


def _sanitize_audit_identifier(value: str, *, prefix: str) -> str:
    assert isinstance(value, str)
    assert isinstance(prefix, str) and prefix
    normalized = value.strip()
    if _is_safe_audit_identifier(normalized):
        return normalized
    return f"{prefix}-{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _is_safe_audit_identifier(value: str) -> bool:
    assert isinstance(value, str)
    if not value or len(value.encode("utf-8")) > _SKILL_AUDIT_MAX_TEXT_BYTES:
        return False
    if fullmatch(
        (
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}"
        ),
        value,
    ):
        return True
    if fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None:
        return False
    if ".." in value:
        return False
    if _has_hidden_audit_identifier_fragment(value):
        return False
    return all(
        _is_safe_audit_identifier_part(part)
        for part in split(r"[:.]", value)
        if part
    )


def _has_hidden_audit_identifier_fragment(value: str) -> bool:
    assert isinstance(value, str)
    return any(part.startswith(".") for part in value.split(":") if part)


def _is_safe_audit_identifier_part(value: str) -> bool:
    assert isinstance(value, str)
    sanitized = sanitize_skill_source_label(value)
    return sanitized == value.lower()


def _safe_text(value: str) -> str:
    assert isinstance(value, str)
    try:
        model_dict({"value": value})
    except AssertionError:
        return "redacted"
    return value


def _limit_text(value: str, max_bytes: int) -> str:
    assert isinstance(value, str)
    assert isinstance(max_bytes, int) and not isinstance(max_bytes, bool)
    assert max_bytes > 3
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return f"{encoded[: max_bytes - 3].decode('utf-8', errors='ignore')}..."


def _json_safe_model_dict(
    data: Mapping[str, object],
) -> dict[str, LooseJsonValue]:
    model_value = model_dict(data)
    json_value = _json_safe_value(model_value)
    assert isinstance(json_value, dict)
    return cast(dict[str, LooseJsonValue], json_value)


def _json_safe_value(value: SkillModelValue) -> LooseJsonValue:
    if isinstance(value, Mapping):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _bounded_payload(
    data: dict[str, LooseJsonValue],
) -> dict[str, LooseJsonValue]:
    if _payload_size(data) <= SKILL_AUDIT_MAX_PAYLOAD_BYTES:
        return data

    bounded: dict[str, LooseJsonValue] = dict(data)
    for key in (
        "diagnostic_codes",
        "resource_ids",
        "skill_ids",
        "source_labels",
    ):
        bounded.pop(key, None)
    bounded["payload_truncated"] = True
    if _payload_size(bounded) <= SKILL_AUDIT_MAX_PAYLOAD_BYTES:
        return bounded

    compact_keys = {
        "schema",
        "event_type",
        "policy_version",
        "agent_id",
        "session_id",
        "tool_call_id",
        "operation_id",
        "registry_version",
        "source_authority",
        "source_label",
        "skill_id",
        "resource_id",
        "status",
        "diagnostic_code",
        "size_bytes",
        "start_byte",
        "end_byte",
        "truncated",
        "stale",
        "hash_prefix",
    }
    compact: dict[str, LooseJsonValue] = {
        key: value for key, value in bounded.items() if key in compact_keys
    }
    compact["payload_truncated"] = True
    return _force_payload_bound(compact)


def _force_payload_bound(
    data: dict[str, LooseJsonValue],
) -> dict[str, LooseJsonValue]:
    bounded = {key: _compact_value(value) for key, value in data.items()}
    if _payload_size(bounded) <= SKILL_AUDIT_MAX_PAYLOAD_BYTES:
        return bounded

    for key in (
        "diagnostic_code",
        "hash_prefix",
        "source_label",
        "resource_id",
        "skill_id",
        "source_authority",
        "registry_version",
        "tool_call_id",
        "session_id",
        "agent_id",
        "operation_id",
    ):
        bounded.pop(key, None)
        if _payload_size(bounded) <= SKILL_AUDIT_MAX_PAYLOAD_BYTES:
            return bounded

    minimal = {
        key: value
        for key, value in bounded.items()
        if key in _SKILL_AUDIT_MINIMAL_KEYS
    }
    minimal["payload_truncated"] = True
    if _payload_size(minimal) <= SKILL_AUDIT_MAX_PAYLOAD_BYTES:
        return minimal

    return {
        "schema": SKILL_AUDIT_EVENT_SCHEMA,
        "event_type": "skills.audit.truncated",
        "policy_version": SKILL_SETTINGS_POLICY_VERSION,
        "payload_truncated": True,
    }


def _compact_value(value: object) -> LooseJsonValue:
    if isinstance(value, str):
        return _limit_text(value, _SKILL_AUDIT_COMPACT_TEXT_BYTES)
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:8]]
    if isinstance(value, Mapping):
        return {
            key: _compact_value(item) for key, item in tuple(value.items())[:8]
        }
    if isinstance(value, bool | int | float) or value is None:
        return value
    return "redacted"


def _payload_size(data: Mapping[str, object]) -> int:
    return len(dumps(data, allow_nan=False, sort_keys=True).encode("utf-8"))
