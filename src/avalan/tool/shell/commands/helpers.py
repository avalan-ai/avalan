from ..entities import ShellExecutionErrorCode, ShellPolicyDenied
from .base import NormalizedPath

from collections.abc import Mapping
from fnmatch import fnmatchcase
from os.path import relpath
from pathlib import Path, PurePosixPath

_SENSITIVE_PATH_PATTERNS = (
    ".git",
    ".git/**",
    ".env*",
    "**/.env*",
    ".git-credentials",
    "**/.git-credentials",
    ".netrc",
    "**/.netrc",
    ".npmrc",
    "**/.npmrc",
    ".pypirc",
    "**/.pypirc",
    "*.pem",
    "**/*.pem",
    "*.key",
    "**/*.key",
    "*_rsa",
    "**/*_rsa",
    "id_dsa",
    "**/id_dsa",
    "id_ecdsa",
    "**/id_ecdsa",
    "id_ed25519",
    "**/id_ed25519",
    ".ssh",
    "**/.ssh",
    ".ssh/**",
    "**/.ssh/**",
    ".aws",
    "**/.aws",
    ".aws/**",
    "**/.aws/**",
    ".gcloud",
    "**/.gcloud",
    ".gcloud/**",
    "**/.gcloud/**",
    ".kube",
    "**/.kube",
    ".kube/**",
    "**/.kube/**",
    ".docker/config.json",
    "**/.docker/config.json",
    ".config/gh/hosts.yml",
    "**/.config/gh/hosts.yml",
    ".config/gcloud",
    "**/.config/gcloud",
    ".config/gcloud/**",
    "**/.config/gcloud/**",
    "credentials",
    "**/credentials",
)


def policy_denied(
    error_code: ShellExecutionErrorCode,
    message: str,
) -> ShellPolicyDenied:
    return ShellPolicyDenied(error_code, message)


def path_matches_sensitive_denylist(display_path: str) -> bool:
    assert isinstance(display_path, str), "display_path must be a string"
    canonical_path = _canonical_display_path(display_path).lower()
    return any(
        fnmatchcase(candidate_path, pattern)
        for candidate_path in _sensitive_path_candidates(canonical_path)
        for pattern in _SENSITIVE_PATH_PATTERNS
    )


def is_denied_display_path(path: str) -> bool:
    assert isinstance(path, str), "path must be a string"
    return bool(path) and (
        path_matches_sensitive_denylist(path)
        or _looks_like_policy_deny_glob(path)
    )


def _looks_like_policy_deny_glob(path: str) -> bool:
    return path.startswith("!") and (
        path_matches_sensitive_denylist(path[1:])
        or path in {"!.*", "!**/.*"}
        or path[1:] in _SENSITIVE_PATH_PATTERNS
    )


def _canonical_display_path(display_path: str) -> str:
    assert isinstance(display_path, str), "display_path must be a string"
    path = PurePosixPath(display_path)
    return "." if str(path) == "." else path.as_posix().lstrip("/")


def _sensitive_path_candidates(canonical_path: str) -> tuple[str, ...]:
    path = PurePosixPath(canonical_path)
    parts = path.parts
    candidates = [path.as_posix()]
    for index in range(len(parts) - 1, 0, -1):
        candidates.append(PurePosixPath(*parts[:index]).as_posix())
    return tuple(candidates)


def _validate_known_options(
    options: Mapping[str, object],
    *,
    allowed_options: set[str],
    forbidden_options: frozenset[str] = frozenset(),
    command: str,
    include_option_name: bool = False,
) -> None:
    for option in options:
        if option in forbidden_options:
            message = f"unsupported {command} option"
            if include_option_name:
                message = f"{message}: {option}"
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                message,
            )
        if option not in allowed_options:
            message = f"unknown {command} option"
            if include_option_name:
                message = f"{message}: {option}"
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                message,
            )


def _required_string_option(
    options: Mapping[str, object],
    name: str,
) -> str:
    value = options.get(name)
    if not isinstance(value, str) or not value.strip():
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be a non-empty string",
        )
    return value


def _literal_option(
    options: Mapping[str, object],
    name: str,
    *,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    value = options.get(name, default)
    if value not in allowed:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is unsupported",
        )
    assert isinstance(value, str)
    return value


def _bool_option(
    options: Mapping[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be a boolean",
        )
    return value


def _bounded_int_option(
    options: Mapping[str, object],
    name: str,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be an integer",
        )
    if value < min_value or value > max_value:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is out of range",
        )
    return value


def _optional_bounded_int_option(
    options: Mapping[str, object],
    name: str,
    *,
    min_value: int,
    max_value: int,
) -> int | None:
    if name not in options or options[name] is None:
        return None
    return _bounded_int_option(
        options,
        name,
        default=min_value,
        min_value=min_value,
        max_value=max_value,
    )


def _single_path(
    paths: tuple[NormalizedPath, ...],
    *,
    allowed_kinds: tuple[str, ...],
    command: str,
) -> NormalizedPath:
    if len(paths) != 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{command} requires exactly one path",
        )
    path = paths[0]
    _validate_path_kind(path, allowed_kinds=allowed_kinds, command=command)
    return path


def _validate_path_kind(
    path: NormalizedPath,
    *,
    allowed_kinds: tuple[str, ...],
    command: str,
) -> None:
    if path.operand.kind not in allowed_kinds:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"unsupported {command} path kind",
        )


def _validate_filter_paths(
    paths: tuple[NormalizedPath, ...],
    *,
    command: str,
    allowed_kinds: tuple[str, ...],
) -> None:
    if not paths:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{command} requires at least one path",
        )
    for path in paths:
        _validate_path_kind(path, allowed_kinds=allowed_kinds, command=command)


def _relative_argv_path(cwd: Path, path: Path) -> str:
    try:
        relative_path = path.relative_to(cwd)
    except ValueError:
        relative_path = Path(relpath(path, cwd))
    return relative_path.as_posix() or "."


def _contains_unsafe_control(value: str) -> bool:
    return any(
        ord(character) < 32 or ord(character) == 127 for character in value
    )


def _pdf_page_range(
    options: Mapping[str, object],
    *,
    max_pages: int,
) -> tuple[int, int]:
    first_page = _bounded_int_option(
        options,
        "first_page",
        default=1,
        min_value=1,
        max_value=2**31 - 1,
    )
    last_page = _optional_bounded_int_option(
        options,
        "last_page",
        min_value=1,
        max_value=2**31 - 1,
    )
    if last_page is None:
        last_page = first_page
    if first_page > last_page:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_PAGE_RANGE,
            "first_page must not exceed last_page",
        )
    page_count = last_page - first_page + 1
    if page_count > max_pages:
        raise policy_denied(
            ShellExecutionErrorCode.PDF_PAGE_CAP_EXCEEDED,
            "PDF page range is too large",
        )
    return first_page, last_page


def _option_safe_path_argument(cwd: Path, path: Path) -> str:
    argument = _relative_argv_path(cwd, path)
    if argument == "-":
        raise policy_denied(
            ShellExecutionErrorCode.DENIED_PATH,
            "path cannot read stdin",
        )
    if argument.startswith("-"):
        return f"./{argument}"
    return argument


def _option_safe_display_path_argument(display_path: str) -> str:
    if display_path.startswith("-"):
        return f"./{display_path}"
    return display_path


def _media_path_argument(cwd: Path, path: Path) -> str:
    return _option_safe_path_argument(cwd, path)


def _media_display_path_argument(display_path: str) -> str:
    return _option_safe_display_path_argument(display_path)


def _contains_traversal(value: str) -> bool:
    return any(part == ".." for part in PurePosixPath(value).parts)
