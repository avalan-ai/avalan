from .contract import SKILL_MAIN_RESOURCE_ID
from .path_policy import (
    sanitize_skill_source_label,
    skill_model_handle_denial_reason,
)

from re import findall, fullmatch, sub

_LOGICAL_ID_PATTERN = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
_PATHLIKE_NAME_FRAGMENTS = ("/", "\\", ":", "$", "~")
_RESOURCE_GLOB_FRAGMENTS = ("*", "?", "[", "]", "{", "}")


def normalize_skill_description(value: str) -> str | None:
    """Return compact model-facing skill description text."""
    assert isinstance(value, str), "description must be a string"
    normalized = sub(r"\s+", " ", value.strip())
    return normalized or None


def normalize_skill_name(value: str) -> str | None:
    """Return the stable logical skill ID for a manifest name.

    The Phase 3 convention is intentionally logical and path-independent:
    skill IDs are derived only from the manifest ``name`` value by lowering
    ASCII word tokens and joining them with hyphens.
    """
    assert isinstance(value, str), "name must be a string"
    if skill_name_denial_reason(value) is not None:
        return None
    tokens = findall(r"[a-z0-9]+", value.strip().lower())
    return "-".join(tokens)


def normalize_skill_resource_id(value: str) -> str | None:
    """Return a safe skill-relative resource ID."""
    assert isinstance(value, str), "resource ID must be a string"
    normalized = value.strip()
    if skill_resource_denial_reason(normalized) is not None:
        return None
    return normalized


def normalize_skill_source_label(value: str) -> str:
    """Return the model-facing source label."""
    assert isinstance(value, str), "source label must be a string"
    return sanitize_skill_source_label(value)


def normalize_skill_tag(value: str) -> str | None:
    """Return a normalized logical tag."""
    assert isinstance(value, str), "tag must be a string"
    return normalize_skill_name(value)


def normalize_skill_tags(values: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return sorted unique normalized tags."""
    assert isinstance(values, tuple), "tags must be a tuple"
    tags: set[str] = set()
    for value in values:
        tag = normalize_skill_tag(value)
        if tag is None:
            return None
        tags.add(tag)
    return tuple(sorted(tags))


def skill_name_denial_reason(value: str) -> str | None:
    """Return why a manifest name cannot become a stable skill ID."""
    assert isinstance(value, str), "name must be a string"
    stripped = value.strip()
    if not stripped:
        return "empty_name"
    if "\x00" in stripped:
        return "nul_byte"
    if any(fragment in stripped for fragment in _PATHLIKE_NAME_FRAGMENTS):
        return "path_like_name"
    if ".." in stripped or stripped.startswith("."):
        return "path_like_name"
    tokens = findall(r"[a-z0-9]+", stripped.lower())
    if not tokens:
        return "empty_name"
    normalized = "-".join(tokens)
    if fullmatch(_LOGICAL_ID_PATTERN, normalized) is None:
        return "invalid_name"
    return None


def skill_resource_denial_reason(value: str) -> str | None:
    """Return why a declared resource ID is unsafe."""
    assert isinstance(value, str), "resource ID must be a string"
    if not value:
        return "empty_resource"
    if value == SKILL_MAIN_RESOURCE_ID:
        return "reserved_resource"
    if value.endswith("/"):
        return "directory_resource"
    if any(fragment in value for fragment in _RESOURCE_GLOB_FRAGMENTS):
        return "recursive_resource"
    reason = skill_model_handle_denial_reason(value)
    if reason is not None:
        return reason
    return None
