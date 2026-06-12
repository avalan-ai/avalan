from ...types import (
    assert_bool as _assert_bool,
)
from ...types import (
    assert_int as _assert_int,
)
from ...types import (
    assert_media_type as _assert_media_type,
)
from ...types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ...types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from ...types import (
    assert_optional_bounded_number as _assert_optional_bounded_number,
)
from ...types import (
    assert_optional_non_negative_int as _assert_optional_non_negative_int,
)
from ...types import (
    assert_safe_path_name as _assert_safe_path_name,
)
from ...types import (
    assert_safe_suffix as _assert_safe_suffix,
)
from ...types import (
    assert_safe_suffix_sequence as _assert_safe_suffix_sequence,
)
from ...types import (
    assert_sha256_hex as _assert_sha256_hex,
)
from ...types import (
    assert_string_tuple as _assert_string_tuple,
)
from ...types import (
    assert_suffix_media_type_mapping as _assert_suffix_media_type_mapping,
)

from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from typing import Literal, final

ShellPathKind = Literal[
    "file",
    "directory",
    "any",
    "text_file",
    "json_file",
    "pdf_file",
    "image_file",
]
ShellPathAccess = Literal["read", "create", "write"]
ShellResourceClass = Literal["standard", "heavy"]
GENERATED_OUTPUT_PREFIX_PLACEHOLDER = "__AVALAN_GENERATED_OUTPUT_PREFIX__"


class ShellExecutionStatus(StrEnum):
    COMPLETED = "completed"
    NO_MATCHES = "no_matches"
    NONZERO_EXIT = "nonzero_exit"
    POLICY_DENIED = "policy_denied"
    COMMAND_UNAVAILABLE = "command_unavailable"
    SPAWN_FAILED = "spawn_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BINARY_SKIPPED = "binary_skipped"
    TOO_LARGE = "too_large"
    TOOL_ERROR = "tool_error"


class ShellExecutionErrorCode(StrEnum):
    COMPLETED = "completed"
    NO_MATCHES = "no_matches"
    NONZERO_EXIT = "nonzero_exit"
    POLICY_DENIED = "policy_denied"
    COMMAND_UNAVAILABLE = "command_unavailable"
    SPAWN_FAILED = "spawn_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BINARY_SKIPPED = "binary_skipped"
    TOO_LARGE = "too_large"
    TOOL_ERROR = "tool_error"
    DENIED_COMMAND = "denied_command"
    DENIED_PATH = "denied_path"
    TRAVERSAL = "traversal"
    HIDDEN_PATH = "hidden_path"
    SENSITIVE_PATH = "sensitive_path"
    SYMLINK = "symlink"
    SPECIAL_FILE = "special_file"
    BINARY_CONTENT = "binary_content"
    TOO_MANY_ARGUMENTS = "too_many_arguments"
    ARGUMENT_TOO_LARGE = "argument_too_large"
    COMMAND_TOO_LARGE = "command_too_large"
    GLOB_TOO_LARGE = "glob_too_large"
    INVALID_OPTION = "invalid_option"
    INVALID_CWD = "invalid_cwd"
    WRITE_DENIED = "write_denied"
    SHELL_DENIED = "shell_denied"
    STDIN_DENIED = "stdin_denied"
    EXECUTABLE_UNAVAILABLE = "executable_unavailable"
    UNSAFE_FILTER = "unsafe_filter"
    UNSUPPORTED_JQ_FEATURE = "unsupported_jq_feature"
    INVALID_PAGE_RANGE = "invalid_page_range"
    PDF_PAGE_CAP_EXCEEDED = "pdf_page_cap_exceeded"
    RASTER_DPI_CAP_EXCEEDED = "raster_dpi_cap_exceeded"
    GENERATED_OUTPUT_CAP_EXCEEDED = "generated_output_cap_exceeded"
    UNSUPPORTED_OCR_LANGUAGE = "unsupported_ocr_language"
    INVALID_OCR_MODE = "invalid_ocr_mode"
    UNSUPPORTED_MEDIA_SIGNATURE = "unsupported_media_signature"


class ShellOutputKind(StrEnum):
    TEXT = "text"
    JSON = "json"
    GENERATED_FILES = "generated_files"


SHELL_STATUS_ERROR_CODES: dict[
    ShellExecutionStatus,
    ShellExecutionErrorCode,
] = {
    status: ShellExecutionErrorCode(status.value)
    for status in ShellExecutionStatus
}


class ShellToolError(Exception):
    error_code: ShellExecutionErrorCode

    def __init__(
        self,
        error_code: ShellExecutionErrorCode,
        message: str,
    ) -> None:
        assert isinstance(
            error_code,
            ShellExecutionErrorCode,
        ), "error_code must be a shell execution error code"
        _assert_non_empty_string(message, "message")
        super().__init__(message)
        self.error_code = error_code


class ShellPolicyDenied(ShellToolError):
    pass


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class PathOperand:
    name: str
    path: str
    kind: ShellPathKind
    access: ShellPathAccess
    required: bool = True

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        _assert_non_empty_string(self.path, "path")
        assert self.kind in (
            "file",
            "directory",
            "any",
            "text_file",
            "json_file",
            "pdf_file",
            "image_file",
        ), "kind must be a shell path kind"
        assert self.access in (
            "read",
            "create",
            "write",
        ), "access must be a shell path access"
        _assert_bool(self.required, "required")


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCommandRequest:
    tool_name: str
    command: str
    options: dict[str, object]
    paths: tuple[PathOperand, ...]
    cwd: str | None
    stdin: bytes | None = None
    timeout_seconds: float | None = None
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.tool_name, "tool_name")
        _assert_non_empty_string(self.command, "command")
        assert isinstance(self.options, dict), "options must be a dictionary"
        assert isinstance(self.paths, tuple), "paths must be a tuple"
        for path in self.paths:
            assert isinstance(path, PathOperand), "paths must contain operands"
        if self.cwd is not None:
            _assert_non_empty_string(self.cwd, "cwd")
        if self.stdin is not None:
            assert isinstance(self.stdin, bytes), "stdin must be bytes"
        _assert_optional_bounded_number(
            self.timeout_seconds,
            "timeout_seconds",
            min_value=0,
            min_inclusive=False,
        )
        _assert_optional_non_negative_int(
            self.max_stdout_bytes,
            "max_stdout_bytes",
        )
        _assert_optional_non_negative_int(
            self.max_stderr_bytes,
            "max_stderr_bytes",
        )
        assert isinstance(self.metadata, dict), "metadata must be a dictionary"
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "metadata", dict(self.metadata))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ExecutionSpec:
    _policy_owned: InitVar[object]
    backend: Literal["local"]
    tool_name: str
    command: str
    executable: str | None
    argv: tuple[str, ...]
    display_argv: tuple[str, ...]
    cwd: str
    display_cwd: str
    env: dict[str, str]
    stdin: bytes | None
    stdout_media_type: str
    output_kind: ShellOutputKind
    resource_class: ShellResourceClass
    output_plan: "GeneratedOutputPlan | None"
    timeout_seconds: float
    max_stdout_bytes: int
    max_stderr_bytes: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self, _policy_owned: object) -> None:
        assert (
            _policy_owned is _EXECUTION_SPEC_FACTORY_KEY
        ), "ExecutionSpec must be created by policy"
        assert self.backend == "local", "backend must be local"
        _assert_non_empty_string(self.tool_name, "tool_name")
        _assert_non_empty_string(self.command, "command")
        if self.executable is not None:
            _assert_non_empty_string(self.executable, "executable")
        _assert_string_tuple(self.argv, "argv")
        _assert_string_tuple(self.display_argv, "display_argv")
        _assert_non_empty_string(self.cwd, "cwd")
        _assert_non_empty_string(self.display_cwd, "display_cwd")
        assert isinstance(self.env, dict), "env must be a dictionary"
        for key, value in self.env.items():
            _assert_non_empty_string(key, "env key")
            _assert_non_empty_string(value, f"env.{key}")
        if self.stdin is not None:
            assert isinstance(self.stdin, bytes), "stdin must be bytes"
        _assert_media_type(
            self.stdout_media_type,
            "stdout_media_type",
        )
        assert isinstance(
            self.output_kind,
            ShellOutputKind,
        ), "output_kind must be a shell output kind"
        assert self.resource_class in (
            "standard",
            "heavy",
        ), "resource_class must be a shell resource class"
        if self.output_plan is not None:
            assert isinstance(
                self.output_plan,
                GeneratedOutputPlan,
            ), "output_plan must be a generated output plan"
            assert (
                self.argv[1:].count(GENERATED_OUTPUT_PREFIX_PLACEHOLDER) == 1
            ), "generated output argv must include one output placeholder"
        _assert_optional_bounded_number(
            self.timeout_seconds,
            "timeout_seconds",
            min_value=0,
            min_inclusive=False,
        )
        _assert_non_negative_int(self.max_stdout_bytes, "max_stdout_bytes")
        _assert_non_negative_int(self.max_stderr_bytes, "max_stderr_bytes")
        assert isinstance(self.metadata, dict), "metadata must be a dictionary"
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "display_argv", tuple(self.display_argv))
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "metadata", dict(self.metadata))


_EXECUTION_SPEC_FACTORY_KEY = object()


def _create_execution_spec_from_policy(
    *,
    backend: Literal["local"],
    tool_name: str,
    command: str,
    executable: str | None,
    argv: tuple[str, ...],
    display_argv: tuple[str, ...],
    cwd: str,
    display_cwd: str,
    env: dict[str, str],
    stdin: bytes | None,
    stdout_media_type: str,
    output_kind: ShellOutputKind,
    resource_class: ShellResourceClass,
    output_plan: "GeneratedOutputPlan | None",
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    metadata: dict[str, object] | None = None,
) -> ExecutionSpec:
    if metadata is not None:
        assert isinstance(
            metadata,
            dict,
        ), "metadata must be a dictionary"
    return ExecutionSpec(
        _policy_owned=_EXECUTION_SPEC_FACTORY_KEY,
        backend=backend,
        tool_name=tool_name,
        command=command,
        executable=executable,
        argv=argv,
        display_argv=display_argv,
        cwd=cwd,
        display_cwd=display_cwd,
        env=env,
        stdin=stdin,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        resource_class=resource_class,
        output_plan=output_plan,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        metadata=dict(metadata if metadata is not None else {}),
    )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class GeneratedOutputPlan:
    prefix_name: str
    display_prefix: str
    allowed_suffixes: tuple[str, ...]
    suffix_media_types: dict[str, str]
    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    max_inline_bytes: int
    max_raster_long_edge_pixels: int | None = None
    max_raster_pixels: int | None = None

    def __post_init__(self) -> None:
        _assert_safe_path_name(self.prefix_name, "prefix_name")
        _assert_non_empty_string(self.display_prefix, "display_prefix")
        assert isinstance(
            self.allowed_suffixes,
            tuple,
        ), "allowed_suffixes must be a tuple"
        _assert_safe_suffix_sequence(self.allowed_suffixes, "allowed_suffixes")
        _assert_suffix_media_type_mapping(
            self.suffix_media_types,
            "suffix_media_types",
        )
        _assert_non_negative_int(self.max_files, "max_files")
        _assert_non_negative_int(self.max_file_bytes, "max_file_bytes")
        _assert_non_negative_int(self.max_total_bytes, "max_total_bytes")
        _assert_non_negative_int(self.max_inline_bytes, "max_inline_bytes")
        _assert_optional_non_negative_int(
            self.max_raster_long_edge_pixels,
            "max_raster_long_edge_pixels",
        )
        _assert_optional_non_negative_int(
            self.max_raster_pixels,
            "max_raster_pixels",
        )
        object.__setattr__(
            self,
            "allowed_suffixes",
            tuple(self.allowed_suffixes),
        )
        object.__setattr__(
            self,
            "suffix_media_types",
            dict(self.suffix_media_types),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class GeneratedFile:
    display_path: str
    media_type: str
    suffix: str
    bytes: int
    sha256: str | None = None
    page: int | None = None
    width: int | None = None
    height: int | None = None
    content_base64: str | None = None
    truncated: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.display_path, "display_path")
        _assert_media_type(self.media_type, "media_type")
        _assert_safe_suffix(self.suffix, "suffix")
        _assert_non_negative_int(self.bytes, "bytes")
        if self.sha256 is not None:
            _assert_sha256_hex(self.sha256)
        _assert_optional_non_negative_int(self.page, "page")
        _assert_optional_non_negative_int(self.width, "width")
        _assert_optional_non_negative_int(self.height, "height")
        if self.content_base64 is not None:
            _assert_non_empty_string(self.content_base64, "content_base64")
        _assert_bool(self.truncated, "truncated")
        assert isinstance(self.metadata, dict), "metadata must be a dictionary"
        object.__setattr__(self, "metadata", dict(self.metadata))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ExecutionResult:
    backend: str
    tool_name: str
    command: str
    argv: tuple[str, ...]
    display_argv: tuple[str, ...]
    cwd: str
    display_cwd: str
    status: ShellExecutionStatus
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_media_type: str
    output_kind: ShellOutputKind
    generated_files: tuple[GeneratedFile, ...] = ()
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    duration_ms: int = 0
    error_code: ShellExecutionErrorCode | None = None
    error_message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.backend, "backend")
        _assert_non_empty_string(self.tool_name, "tool_name")
        _assert_non_empty_string(self.command, "command")
        _assert_string_tuple(self.argv, "argv")
        _assert_string_tuple(self.display_argv, "display_argv")
        _assert_non_empty_string(self.cwd, "cwd")
        _assert_non_empty_string(self.display_cwd, "display_cwd")
        assert isinstance(
            self.status,
            ShellExecutionStatus,
        ), "status must be a shell execution status"
        if self.exit_code is not None:
            _assert_int(self.exit_code, "exit_code")
        assert isinstance(self.stdout, str), "stdout must be a string"
        assert isinstance(self.stderr, str), "stderr must be a string"
        _assert_media_type(self.stdout_media_type, "stdout_media_type")
        assert isinstance(
            self.output_kind,
            ShellOutputKind,
        ), "output_kind must be a shell output kind"
        assert isinstance(
            self.generated_files,
            tuple,
        ), "generated_files must be a tuple"
        for generated_file in self.generated_files:
            assert isinstance(
                generated_file,
                GeneratedFile,
            ), "generated_files must contain generated files"
        _assert_non_negative_int(self.stdout_bytes, "stdout_bytes")
        _assert_non_negative_int(self.stderr_bytes, "stderr_bytes")
        _assert_bool(self.stdout_truncated, "stdout_truncated")
        _assert_bool(self.stderr_truncated, "stderr_truncated")
        _assert_bool(self.timed_out, "timed_out")
        _assert_bool(self.cancelled, "cancelled")
        _assert_non_negative_int(self.duration_ms, "duration_ms")
        if self.error_code is not None:
            assert isinstance(
                self.error_code,
                ShellExecutionErrorCode,
            ), "error_code must be a shell execution error code"
        if self.error_message is not None:
            _assert_non_empty_string(self.error_message, "error_message")
        assert isinstance(self.metadata, dict), "metadata must be a dictionary"
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "display_argv", tuple(self.display_argv))
        object.__setattr__(
            self,
            "generated_files",
            tuple(self.generated_files),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
