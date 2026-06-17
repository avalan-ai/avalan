"""Provide safe text and payload summaries for CLI display."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from json import JSONDecodeError, dumps, loads
from re import sub
from typing import cast
from uuid import UUID

from rich.markup import escape

MAX_SUMMARY_CHARS = 300
MAX_SUMMARY_DEPTH = 5
MAX_SUMMARY_ITEMS = 6
MAX_TEXT_CHARS = 120
ANSI_CONTROL_STRING_PATTERN = (
    r"(?:\x1b\][\s\S]*?(?:\x07|\x1b\\|$)"
    r"|\x9d[\s\S]*?(?:\x07|\x1b\\|\x9c|$)"
    r"|\x1b[PX^_][\s\S]*?(?:\x1b\\|$)"
    r"|[\x90\x98\x9e\x9f][\s\S]*?(?:\x1b\\|\x9c|$))"
)
ANSI_CSI_PATTERN = r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
ANSI_ESCAPE_PATTERN = r"\x1b[ -/]*[0-~]"
REDACTED = "<redacted>"
SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
    }
)


def truncate_text(text: str, limit: int) -> str:
    """Return text bounded to limit characters."""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def strip_terminal_controls(text: str) -> str:
    """Return text without ANSI and terminal control sequences."""
    text = sub(ANSI_CONTROL_STRING_PATTERN, "", text)
    text = sub(ANSI_CSI_PATTERN, "", text)
    return sub(ANSI_ESCAPE_PATTERN, "", text)


def safe_text(value: object, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Return escaped, printable text for CLI rendering."""
    try:
        text = str(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    text = strip_terminal_controls(text)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    text = text.replace("\t", "\\t")
    text = sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return escape(truncate_text(text, limit))


def is_sensitive_key(key: object) -> bool:
    """Return whether key appears to name sensitive data."""
    try:
        key_text = strip_terminal_controls(str(key)).lower()
    except Exception:
        return False
    key_text = sub(r"\[/?[^\]]+\]", "", key_text)
    compact_key = sub(r"[^a-z0-9]", "", key_text)
    return any(
        sensitive in key_text or sensitive.replace("_", "") in compact_key
        for sensitive in SENSITIVE_KEY_PARTS
    )


def safe_data(
    value: object,
    *,
    key: object | None = None,
    depth: int = 0,
    seen: set[int] | None = None,
) -> object:
    """Return redacted data made from display-safe values."""
    if key is not None and is_sensitive_key(key):
        return REDACTED
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, datetime | UUID):
        return safe_text(value)
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, bytes):
        return f"<bytes {len(value)}>"
    if depth >= MAX_SUMMARY_DEPTH:
        return f"<{type(value).__name__}>"

    seen = seen if seen is not None else set()
    tracks_identity = isinstance(
        value, Mapping | list | tuple | set | frozenset
    ) or (is_dataclass(value) and not isinstance(value, type))
    if tracks_identity:
        value_id = id(value)
        if value_id in seen:
            return "<cycle>"
        seen.add(value_id)
    try:
        if isinstance(value, Mapping):
            safe_mapping: dict[str, object] = {}
            try:
                items = value.items()
                for index, (child_key, child_value) in enumerate(items):
                    if index >= MAX_SUMMARY_ITEMS:
                        safe_mapping["..."] = "truncated"
                        break
                    safe_mapping[safe_text(child_key)] = safe_data(
                        child_value,
                        key=child_key,
                        depth=depth + 1,
                        seen=seen,
                    )
            except Exception:
                return f"<unreadable {type(value).__name__}>"
            return safe_mapping
        if isinstance(value, set | frozenset):
            safe_set = [
                safe_data(item, depth=depth + 1, seen=seen) for item in value
            ]
            safe_set.sort(key=_stable_data_sort_key)
            if len(safe_set) > MAX_SUMMARY_ITEMS:
                return safe_set[:MAX_SUMMARY_ITEMS] + ["truncated"]
            return safe_set
        if isinstance(value, list | tuple):
            safe_sequence: list[object] = []
            for index, item in enumerate(value):
                if index >= MAX_SUMMARY_ITEMS:
                    safe_sequence.append("truncated")
                    break
                safe_sequence.append(
                    safe_data(item, depth=depth + 1, seen=seen)
                )
            return safe_sequence
        if is_dataclass(value) and not isinstance(value, type):
            safe_object: dict[str, object] = {}
            for index, field in enumerate(fields(value)):
                if index >= MAX_SUMMARY_ITEMS:
                    safe_object["..."] = "truncated"
                    break
                try:
                    field_value = getattr(value, field.name)
                except Exception:
                    field_value = f"<unreadable {field.name}>"
                safe_object[field.name] = safe_data(
                    field_value,
                    key=field.name,
                    depth=depth + 1,
                    seen=seen,
                )
            return safe_object
    finally:
        if tracks_identity:
            seen.remove(id(value))
    return safe_text(value)


def safe_summary(
    value: object,
    *,
    limit: int = MAX_SUMMARY_CHARS,
) -> str:
    """Return a deterministic JSON summary for display."""
    safe_value = safe_data(value)
    try:
        text = dumps(safe_value, ensure_ascii=True, sort_keys=True)
    except Exception:
        text = safe_text(safe_value)
    return truncate_text(text, limit)


def safe_tool_call_request_text(text: str) -> str:
    """Return redacted streamed tool-call request text."""
    assert isinstance(text, str)
    if not text:
        return ""
    try:
        parsed = loads(text)
    except JSONDecodeError:
        if _contains_sensitive_marker(text):
            return REDACTED
        return safe_text(text, limit=len(text))
    return safe_summary(parsed, limit=max(len(text), MAX_SUMMARY_CHARS))


def value_from(value: object, name: str) -> object | None:
    """Return a mapping or attribute value by name."""
    if isinstance(value, Mapping):
        return value.get(name)
    try:
        return cast(object, getattr(value, name))
    except Exception:
        return None


def event_type_value(event_type: object) -> str:
    """Return a string event type value."""
    value = getattr(event_type, "value", event_type)
    return value if isinstance(value, str) else str(value)


def _contains_sensitive_marker(text: str) -> bool:
    key_text = strip_terminal_controls(text).lower()
    key_text = sub(r"\[/?[^\]]+\]", "", key_text)
    compact_key = sub(r"[^a-z0-9]", "", key_text)
    return any(
        sensitive in key_text or sensitive.replace("_", "") in compact_key
        for sensitive in SENSITIVE_KEY_PARTS
    )


def _stable_data_sort_key(value: object) -> str:
    return dumps(value, ensure_ascii=True, sort_keys=True)
