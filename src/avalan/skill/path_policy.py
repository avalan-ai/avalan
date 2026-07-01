from hashlib import sha256
from pathlib import PurePath, PurePosixPath
from re import findall, fullmatch, match

_LOGICAL_ID_PATTERN = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
_URL_LIKE_PATTERN = r"[a-zA-Z][a-zA-Z0-9+.-]*://"
_URI_SCHEME_PATTERN = r"[a-zA-Z][a-zA-Z0-9+.-]*:"
_WINDOWS_ABSOLUTE_PATTERN = r"[a-zA-Z]:[\\/]"
_SENSITIVE_PARTS = frozenset(
    {
        ".aws",
        ".codex",
        ".config",
        ".env",
        ".ssh",
        "credentials",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)
_SENSITIVE_BASENAMES = frozenset(
    {
        "credentials",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)


def sanitize_skill_source_label(value: str) -> str:
    assert isinstance(value, str), "source label must be a string"
    normalized = value.strip().lower()
    if _is_safe_logical_label(normalized):
        return normalized
    if not _requires_digest(value):
        candidate = "-".join(findall(r"[a-z0-9]+", normalized))
        if _is_safe_logical_label(candidate):
            return candidate
    return f"source-{_digest(value)}"


def sanitize_skill_resource_id(value: str) -> str:
    assert isinstance(value, str), "resource ID must be a string"
    normalized = value.replace("\\", "/").strip()
    if _is_safe_resource_id(normalized):
        return normalized
    return f"resource/{_digest(value)}"


def skill_model_handle_denial_reason(
    value: str,
    *,
    allow_hidden_paths: bool = False,
) -> str | None:
    assert isinstance(value, str), "model handle must be a string"
    if value == "":
        return "empty_handle"
    common_reason = _common_denial_reason(value)
    if common_reason is not None:
        return common_reason
    if value.startswith("/"):
        return "absolute_handle"
    if match(_WINDOWS_ABSOLUTE_PATTERN, value) is not None:
        return "absolute_handle"
    if _has_empty_model_segment(value):
        return "empty_path_segment"
    path = PurePosixPath(value)
    return _relative_part_denial_reason(
        path.parts,
        allow_hidden_paths=allow_hidden_paths,
    )


def skill_source_root_denial_reason(
    value: str,
    *,
    allow_hidden_paths: bool = False,
) -> str | None:
    assert isinstance(value, str), "source root must be a string"
    if value == "":
        return "empty_root"
    common_reason = _common_denial_reason(value)
    if common_reason is not None:
        return common_reason
    if match(_WINDOWS_ABSOLUTE_PATTERN, value) is not None:
        return "windows_absolute_path"
    path = PurePosixPath(value.replace("\\", "/"))
    parts = tuple(part for part in path.parts if part != "/")
    return _relative_part_denial_reason(
        parts,
        allow_hidden_paths=allow_hidden_paths,
    )


def redact_host_path(value: str | PurePath) -> str:
    assert isinstance(value, str | PurePath), "path must be text-like"
    text = str(value).replace("\\", "/")
    if "\x00" in text:
        return "<host-path>"
    name = PurePosixPath(text).name
    if name and skill_model_handle_denial_reason(name) is None:
        return f"<host-path>/{name}"
    return "<host-path>"


def sanitize_source_label(value: str) -> str:
    return sanitize_skill_source_label(value)


class SkillPathPolicy:
    def __init__(self, *, allow_hidden_paths: bool = False) -> None:
        assert isinstance(allow_hidden_paths, bool)
        self._allow_hidden_paths = allow_hidden_paths

    def sanitize_source_label(self, value: str) -> str:
        return sanitize_skill_source_label(value)

    def sanitize_resource_id(self, value: str) -> str:
        return sanitize_skill_resource_id(value)

    def model_handle_denial_reason(self, value: str) -> str | None:
        return skill_model_handle_denial_reason(
            value,
            allow_hidden_paths=self._allow_hidden_paths,
        )

    def source_root_denial_reason(self, value: str) -> str | None:
        return skill_source_root_denial_reason(
            value,
            allow_hidden_paths=self._allow_hidden_paths,
        )


def _is_safe_logical_label(value: str) -> bool:
    return (
        fullmatch(_LOGICAL_ID_PATTERN, value) is not None
        and not _has_hidden_part(value.split("."))
        and not _has_sensitive_part(value.split("."))
    )


def _is_safe_resource_id(value: str) -> bool:
    if skill_model_handle_denial_reason(value) is not None:
        return False
    return True


def _common_denial_reason(value: str) -> str | None:
    if "\x00" in value:
        return "nul_byte"
    if "\\" in value:
        return "backslash"
    if value.startswith("~"):
        return "home_expansion"
    if "$" in value:
        return "environment_expansion"
    if match(_URL_LIKE_PATTERN, value) is not None:
        return "url_handle"
    if (
        match(_URI_SCHEME_PATTERN, value) is not None
        and match(_WINDOWS_ABSOLUTE_PATTERN, value) is None
    ):
        return "url_handle"
    if value.startswith("//"):
        return "url_handle"
    return None


def _requires_digest(value: str) -> bool:
    normalized = value.strip().lower().replace("\\", "/")
    if not normalized:
        return True
    if any(fragment in normalized for fragment in ("/", ":", "$", "~")):
        return True
    parts = tuple(part for part in normalized.split(".") if part)
    return _has_hidden_part(list(parts)) or _has_sensitive_part(list(parts))


def _relative_part_denial_reason(
    parts: tuple[str, ...],
    *,
    allow_hidden_paths: bool,
) -> str | None:
    for part in parts:
        if part == "..":
            return "traversal"
        if not allow_hidden_paths and part.startswith("."):
            return "hidden_path"
        if _is_sensitive_part(part):
            return "sensitive_path"
    return None


def _has_hidden_part(parts: list[str]) -> bool:
    return any(part.startswith(".") for part in parts if part)


def _has_sensitive_part(parts: list[str]) -> bool:
    return any(_is_sensitive_part(part) for part in parts)


def _is_sensitive_part(part: str) -> bool:
    lowered = part.lower()
    basename = lowered.split(".", 1)[0]
    return (
        lowered in _SENSITIVE_PARTS
        or lowered.startswith(".env")
        or basename in _SENSITIVE_BASENAMES
    )


def _has_empty_model_segment(value: str) -> bool:
    return any(part in {"", "."} for part in value.split("/"))


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
