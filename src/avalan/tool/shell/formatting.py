from ...types import assert_string_sequence as _assert_string_sequence
from .commands.helpers import (
    is_denied_display_path,
)
from .entities import (
    ExecutionResult,
    GeneratedFile,
    ShellExecutionStatus,
    ShellOutputKind,
)
from .registry import SHELL_COMMAND_DEFINITIONS
from .settings import ShellToolSettings

from collections.abc import Sequence
from pathlib import Path
from re import DOTALL, IGNORECASE, Match, Pattern
from re import compile as compile_pattern
from re import escape as escape_pattern
from shlex import join as shell_join
from tempfile import gettempdir
from unicodedata import category as unicode_category

_ANSI_ESCAPE_PATTERN = compile_pattern(
    r"\x1B(?:\][^\x07]*(?:\x07|\x1B\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])"
)
_GENERATED_OUTPUT_PREFIX_PATTERN = compile_pattern(
    r"avalan-shell-[0-9a-fA-F]{32}"
)
_POSIX_PATH_SEGMENT = r"[A-Za-z0-9._~+%-]+"
_SPACED_POSIX_PATH_SEGMENT = (
    r"[A-Za-z0-9._~+%-][A-Za-z0-9._~+% -]* "
    r"[A-Za-z0-9._~+% -]*[A-Za-z0-9._~+%-]"
)
_SPACED_POSIX_ABSOLUTE_PATH_PATTERN = compile_pattern(
    rf"(?<![:/A-Za-z0-9._-])/{_POSIX_PATH_SEGMENT}"
    rf"(?:/{_POSIX_PATH_SEGMENT})*/{_SPACED_POSIX_PATH_SEGMENT}"
    rf"(?:/(?:{_POSIX_PATH_SEGMENT}|{_SPACED_POSIX_PATH_SEGMENT}))+"
    r"(?=$|[\s:,'\")\]}])"
)
_POSIX_ABSOLUTE_PATH_PATTERN = compile_pattern(
    r"(?<![:/A-Za-z0-9._-])/"
    r"[A-Za-z0-9._~+%-]+"
    r"(?:/[A-Za-z0-9._~+%-]+)*"
    r"(?=$|[\s:,'\")\]}])"
)
_SENSITIVE_DISPLAY_PATH_TOKEN_PATTERN = compile_pattern(
    r"(?<![!*/A-Za-z0-9._~+%-])"
    r"!?(?:\.{1,2}/)?[A-Za-z0-9._~+%*-]+"
    r"(?:/[A-Za-z0-9._~+%*-]+)*/?"
    r"(?![*/A-Za-z0-9._~+%-])"
)
_PRIVATE_KEY_PATTERN = compile_pattern(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    flags=DOTALL,
)
_QUOTED_KEY_SECRET_PATTERN = compile_pattern(
    r"(?P<key_quote>[\"'])(?P<name>[A-Z_][A-Z0-9_]*)"
    r"(?P=key_quote)(?P<separator>\s*:\s*)"
    r"(?P<value>"
    r"\"(?:\\.|[^\"\\])*\"?"
    r"|'(?:\\.|[^'\\])*'?"
    r"|true|false|null"
    r"|-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    r"|[^\s,\]}]+"
    r")",
    flags=IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = compile_pattern(
    r"\b([A-Z_][A-Z0-9_]*)(\s*[=:]\s*)"
    r"(\"(?:\\.|[^\"\\])*\"?|'(?:\\.|[^'\\])*'?|[^\s]+)",
    flags=IGNORECASE,
)
_SECRET_NAME_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_MIN_REDACTION_VALUE_LENGTH = 4
_REDACTED_SECRET = "[redacted]"
_REDACTED_PATH = "[redacted_path]"


def format_shell_result(
    result: object,
    *,
    settings: ShellToolSettings | None = None,
    redaction_values: Sequence[str] = (),
) -> str:
    assert isinstance(
        result, ExecutionResult
    ), "result must be a shell execution result"
    if settings is not None:
        assert isinstance(
            settings,
            ShellToolSettings,
        ), "settings must be shell tool settings"
    _assert_string_sequence(redaction_values, "redaction_values")

    redactor = _Redactor(
        (
            *(settings.environment.values() if settings is not None else ()),
            *redaction_values,
        )
    )
    error_code = result.error_code.value if result.error_code else None
    stdout = _format_output(result.command, result.stdout, redactor)
    stderr = _format_output(result.command, result.stderr, redactor)
    lines = [
        f"tool: {redactor(result.tool_name)}",
        f"status: {result.status.value}",
        f"command: {redactor(shell_join(result.display_argv))}",
        f"cwd: {redactor(result.display_cwd)}",
        f"exit_code: {_scalar(result.exit_code, redactor)}",
        f"error_code: {_scalar(error_code, redactor)}",
        f"error_message: {_scalar(result.error_message, redactor)}",
        f"output_kind: {result.output_kind.value}",
        f"stdout_media_type: {redactor(result.stdout_media_type)}",
        f"timed_out: {_bool(result.timed_out)}",
        f"duration_ms: {result.duration_ms}",
        f"stdout_bytes: {result.stdout_bytes}",
        f"stderr_bytes: {result.stderr_bytes}",
        f"stdout_truncated: {_bool(result.stdout_truncated)}",
        f"stderr_truncated: {_bool(result.stderr_truncated)}",
        "",
        "stdout:",
        stdout,
    ]
    if _include_stderr(result):
        lines.extend(("", "stderr:", stderr))
    if result.output_kind is ShellOutputKind.GENERATED_FILES:
        lines.extend(("", "generated_files:"))
        lines.extend(_generated_file_lines(result.generated_files, redactor))
    return "\n".join(lines)


def _include_stderr(result: ExecutionResult) -> bool:
    return (
        bool(result.stderr)
        or result.status
        in {
            ShellExecutionStatus.NONZERO_EXIT,
            ShellExecutionStatus.NO_MATCHES,
            ShellExecutionStatus.TIMEOUT,
            ShellExecutionStatus.COMMAND_UNAVAILABLE,
            ShellExecutionStatus.SPAWN_FAILED,
        }
        or (result.exit_code is not None and result.exit_code != 0)
    )


def _generated_file_lines(
    generated_files: tuple[GeneratedFile, ...],
    redactor: "_Redactor",
) -> list[str]:
    if not generated_files:
        return ["[]"]
    lines: list[str] = []
    for generated_file in generated_files:
        lines.extend(
            [
                f"- display_path: {redactor(generated_file.display_path)}",
                f"  media_type: {redactor(generated_file.media_type)}",
                f"  suffix: {redactor(generated_file.suffix)}",
                f"  bytes: {generated_file.bytes}",
                f"  sha256: {_scalar(generated_file.sha256, redactor)}",
                f"  page: {_scalar(generated_file.page, redactor)}",
                f"  width: {_scalar(generated_file.width, redactor)}",
                f"  height: {_scalar(generated_file.height, redactor)}",
                f"  truncated: {_bool(generated_file.truncated)}",
            ]
        )
        if generated_file.content_base64 is not None:
            lines.append(
                f"  content_base64: {redactor(generated_file.content_base64)}"
            )
    return lines


def _scalar(value: object, redactor: "_Redactor") -> str:
    if value is None:
        return "null"
    return redactor(str(value)).replace("\n", "\\n")


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _format_output(
    command: str,
    value: str,
    redactor: "_Redactor",
) -> str:
    sanitized = _sanitize_controls(value)
    sanitized = _PRIVATE_KEY_PATTERN.sub(_REDACTED_SECRET, sanitized)
    return redactor(_post_filter_output(command, sanitized))


class _Redactor:
    _patterns: tuple[Pattern[str], ...]
    _path_patterns: tuple[Pattern[str], ...]
    _username_patterns: tuple[Pattern[str], ...]

    def __init__(self, values: Sequence[str]) -> None:
        redaction_values = tuple(
            sorted(
                {
                    value
                    for value in values
                    if len(value) >= _MIN_REDACTION_VALUE_LENGTH
                },
                key=len,
                reverse=True,
            )
        )
        self._patterns = tuple(
            compile_pattern(escape_pattern(value))
            for value in redaction_values
        )
        self._path_patterns = _host_path_patterns()
        self._username_patterns = _host_username_patterns()

    def __call__(self, value: str) -> str:
        redacted = _sanitize_controls(value)
        redacted = _PRIVATE_KEY_PATTERN.sub(_REDACTED_SECRET, redacted)
        redacted = _QUOTED_KEY_SECRET_PATTERN.sub(
            _redact_quoted_key_secret,
            redacted,
        )
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
            _redact_secret_assignment,
            redacted,
        )
        for pattern in self._patterns:
            redacted = pattern.sub(_REDACTED_SECRET, redacted)
        redacted = _SPACED_POSIX_ABSOLUTE_PATH_PATTERN.sub(
            _REDACTED_PATH,
            redacted,
        )
        redacted = _POSIX_ABSOLUTE_PATH_PATTERN.sub(
            _REDACTED_PATH,
            redacted,
        )
        for pattern in self._path_patterns:
            redacted = pattern.sub(_REDACTED_PATH, redacted)
        for pattern in self._username_patterns:
            redacted = pattern.sub(_REDACTED_SECRET, redacted)
        redacted = _SENSITIVE_DISPLAY_PATH_TOKEN_PATTERN.sub(
            _redact_sensitive_display_path,
            redacted,
        )
        redacted = _GENERATED_OUTPUT_PREFIX_PATTERN.sub(
            _REDACTED_PATH,
            redacted,
        )
        return redacted


def _redact_quoted_key_secret(match: Match[str]) -> str:
    key_quote = match.group("key_quote")
    name = match.group("name")
    separator = match.group("separator")
    value = match.group("value")
    if not _is_secret_name(name):
        return match.group(0)
    if value.startswith(("'", '"')):
        value_quote = value[0]
        return (
            f"{key_quote}{name}{key_quote}{separator}"
            f"{value_quote}{_REDACTED_SECRET}{value_quote}"
        )
    return f'{key_quote}{name}{key_quote}{separator}"{_REDACTED_SECRET}"'


def _host_path_patterns() -> tuple[Pattern[str], ...]:
    candidates = {
        str(Path.home()),
        gettempdir(),
        "/private/tmp",
        "/var/folders",
        "/private/var/folders",
    }
    return tuple(
        compile_pattern(escape_pattern(candidate))
        for candidate in sorted(candidates, key=len, reverse=True)
        if candidate and candidate != "/"
    )


def _host_username_patterns() -> tuple[Pattern[str], ...]:
    candidates = {Path.home().name}
    return tuple(
        compile_pattern(
            rf"(?<![A-Za-z0-9._-]){escape_pattern(candidate)}"
            r"(?![A-Za-z0-9._-])"
        )
        for candidate in sorted(candidates, key=len, reverse=True)
        if len(candidate) >= _MIN_REDACTION_VALUE_LENGTH
    )


def _redact_secret_assignment(match: Match[str]) -> str:
    name, separator, value = match.groups()
    if not _is_secret_name(name):
        return match.group(0)
    return f"{name}{separator}{_REDACTED_SECRET}"


def _redact_sensitive_display_path(match: Match[str]) -> str:
    value = match.group(0)
    if is_denied_display_path(value.rstrip("/")):
        return _REDACTED_PATH
    return value


def _is_secret_name(name: str) -> bool:
    upper_name = name.upper()
    return any(marker in upper_name for marker in _SECRET_NAME_MARKERS) or (
        "KEY" in upper_name.split("_")
    )


def _sanitize_controls(value: str) -> str:
    without_ansi = _ANSI_ESCAPE_PATTERN.sub("", value)
    return "".join(
        character
        for character in without_ansi
        if character in ("\n", "\t")
        or not unicode_category(character).startswith("C")
    )


def _post_filter_output(command: str, value: str) -> str:
    command_definition = SHELL_COMMAND_DEFINITIONS.get(command)
    if command_definition is None:
        return value
    return command_definition.output_filter(value)
