from ..entities import ShellExecutionErrorCode
from .base import (
    NormalizedPath,
    NormalizedWorkspace,
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _SENSITIVE_PATH_PATTERNS,
    _bool_option,
    _bounded_int_option,
    _contains_traversal,
    _literal_option,
    _optional_bounded_int_option,
    _relative_argv_path,
    _required_string_option,
    _validate_known_options,
    is_denied_display_path,
    path_matches_sensitive_denylist,
    policy_denied,
)

from collections.abc import Sequence
from pathlib import PurePosixPath
from re import compile as compile_pattern
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import ShellToolSettings

_OUTPUT_LINE_PATTERN = compile_pattern(
    r"^(?P<path>.*?)(?::\d+:\d+:|-\d+(?:-\d+)?-)"
)
_FORBIDDEN_OPTIONS = frozenset(
    (
        "hidden",
        "no_ignore",
        "follow",
        "text",
        "binary",
        "search_zip",
        "pre",
        "replace",
        "pcre2",
        "multiline",
        "preprocessor",
        "preprocessors",
    )
)


def _normalized_rg_globs(
    value: object,
    settings: "ShellToolSettings",
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "globs must be a sequence",
        )
    if len(value) > settings.max_glob_count:
        raise policy_denied(
            ShellExecutionErrorCode.GLOB_TOO_LARGE,
            "too many globs",
        )
    globs: list[str] = []
    total_bytes = 0
    for glob in value:
        if not isinstance(glob, str) or not glob.strip():
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "glob must be a non-empty string",
            )
        glob_bytes = len(glob.encode("utf-8"))
        total_bytes += glob_bytes
        if (
            glob_bytes > settings.max_glob_bytes_per_glob
            or total_bytes > settings.max_total_glob_bytes
        ):
            raise policy_denied(
                ShellExecutionErrorCode.GLOB_TOO_LARGE,
                "glob is too large",
            )
        _validate_rg_glob(glob, settings)
        globs.append(glob)
    return tuple(globs)


def _validate_rg_glob(glob: str, settings: "ShellToolSettings") -> None:
    glob_pattern = glob[1:] if glob.startswith("!") else glob
    if glob_pattern.startswith("/"):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "absolute globs are disabled",
        )
    if _contains_traversal(glob_pattern):
        raise policy_denied(
            ShellExecutionErrorCode.TRAVERSAL,
            "glob contains traversal",
        )
    if not settings.allow_hidden and _glob_matches_hidden(glob_pattern):
        raise policy_denied(
            ShellExecutionErrorCode.HIDDEN_PATH,
            "hidden globs are disabled",
        )
    if path_matches_sensitive_denylist(glob_pattern):
        raise policy_denied(
            ShellExecutionErrorCode.SENSITIVE_PATH,
            "glob is denied",
        )
    if glob_pattern.count("**") > 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "glob is too recursive",
        )


def _glob_matches_hidden(glob_pattern: str) -> bool:
    return any(
        part.startswith(".") or part in ("{.,*", "{,.}*")
        for part in PurePosixPath(glob_pattern).parts
    )


def _rg_policy_deny_globs(settings: "ShellToolSettings") -> tuple[str, ...]:
    globs = [f"!{pattern}" for pattern in _SENSITIVE_PATH_PATTERNS]
    if not settings.allow_hidden:
        globs.extend(("!.*", "!**/.*"))
    return tuple(globs)


def _validate_rg_paths(paths: tuple[NormalizedPath, ...]) -> None:
    for path in paths:
        if path.operand.kind not in ("file", "directory", "any", "text_file"):
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "unsupported rg path kind",
            )


def _rg_path_arguments(
    paths: tuple[NormalizedPath, ...],
    workspace: NormalizedWorkspace,
) -> tuple[str, ...]:
    if not paths:
        return (".",)
    return tuple(
        _relative_argv_path(workspace.cwd, path.path) for path in paths
    )


def _rg_display_path_arguments(
    paths: tuple[NormalizedPath, ...],
) -> tuple[str, ...]:
    if not paths:
        return (".",)
    return tuple(path.display_path for path in paths)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "pattern",
            "case",
            "fixed_strings",
            "context_lines",
            "before_context",
            "after_context",
            "max_matches_per_file",
            "max_depth",
            "max_filesize_bytes",
            "globs",
        },
        forbidden_options=_FORBIDDEN_OPTIONS,
        command="rg",
        include_option_name=True,
    )
    pattern = _required_string_option(request.options, "pattern")
    case_mode = _literal_option(
        request.options,
        "case",
        default="sensitive",
        allowed=("sensitive", "insensitive", "smart"),
    )
    fixed_strings = _bool_option(
        request.options,
        "fixed_strings",
        default=False,
    )
    context_lines = _bounded_int_option(
        request.options,
        "context_lines",
        default=0,
        min_value=0,
        max_value=settings.max_rg_context_lines,
    )
    before_context = _optional_bounded_int_option(
        request.options,
        "before_context",
        min_value=0,
        max_value=settings.max_rg_context_lines,
    )
    after_context = _optional_bounded_int_option(
        request.options,
        "after_context",
        min_value=0,
        max_value=settings.max_rg_context_lines,
    )
    max_matches_per_file = _optional_bounded_int_option(
        request.options,
        "max_matches_per_file",
        min_value=1,
        max_value=settings.max_rg_matches_per_file,
    )
    max_depth = _optional_non_negative_int_option(
        request.options,
        "max_depth",
    )
    max_filesize_bytes = _optional_positive_int_option(
        request.options, "max_filesize_bytes"
    )
    globs = _normalized_rg_globs(request.options.get("globs"), settings)
    _validate_rg_paths(context.paths)
    argv_parts = [
        context.executable_name,
        "--no-config",
        "--color=never",
        "--no-heading",
        "--line-number",
        "--column",
        "--max-columns",
        str(settings.max_rg_columns),
        "--max-columns-preview",
    ]
    if case_mode == "insensitive":
        argv_parts.append("--ignore-case")
    elif case_mode == "smart":
        argv_parts.append("--smart-case")
    if fixed_strings:
        argv_parts.append("--fixed-strings")
    if context_lines:
        argv_parts.extend(("--context", str(context_lines)))
    if before_context is not None:
        argv_parts.extend(("--before-context", str(before_context)))
    if after_context is not None:
        argv_parts.extend(("--after-context", str(after_context)))
    if max_matches_per_file is not None:
        argv_parts.extend(("--max-count", str(max_matches_per_file)))
    if max_depth is not None:
        argv_parts.extend(("--max-depth", str(max_depth)))
    if max_filesize_bytes is not None:
        argv_parts.extend(("--max-filesize", str(max_filesize_bytes)))
    for glob in globs:
        argv_parts.extend(("--glob", glob))
    display_parts = list(argv_parts)
    for glob in _rg_policy_deny_globs(settings):
        argv_parts.extend(("--glob", glob))
    argv_parts.extend(("-e", pattern, "--"))
    display_parts.extend(("-e", pattern, "--"))
    argv_parts.extend(_rg_path_arguments(context.paths, context.workspace))
    display_parts.extend(_rg_display_path_arguments(context.paths))
    context.metadata["exit_code_statuses"] = {1: "no_matches"}
    return tuple(argv_parts), tuple(display_parts), None


def filter_output(value: str) -> str:
    assert isinstance(value, str), "value must be a string"
    return "\n".join(_filtered_output_line(line) for line in value.split("\n"))


def _optional_positive_int_option(
    options: dict[str, object],
    name: str,
) -> int | None:
    if name not in options or options[name] is None:
        return None
    value = options[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be an integer",
        )
    if value < 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is out of range",
        )
    return value


def _optional_non_negative_int_option(
    options: dict[str, object],
    name: str,
) -> int | None:
    if name not in options or options[name] is None:
        return None
    value = options[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be an integer",
        )
    if value < 0:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is out of range",
        )
    return value


def _filtered_output_line(line: str) -> str:
    match = _OUTPUT_LINE_PATTERN.match(line)
    path = match.group("path") if match else line.split(":", 1)[0]
    if is_denied_display_path(path):
        return "[redacted_path]"
    return line


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="rg",
    executable_name="rg",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("ripgrep",),
    argv_builder=build_argv,
    output_filter=filter_output,
)
