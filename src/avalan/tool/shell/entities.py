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
    assert_positive_int as _assert_positive_int,
)
from ...types import (
    assert_positive_number as _assert_positive_number,
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
ShellExecutionModeValue = Literal["local", "sandbox", "container"]
ShellCompositionMode = Literal["pipeline", "serial", "parallel"]
DEFAULT_MAX_PIPELINE_STAGES = 8
GENERATED_OUTPUT_PREFIX_PLACEHOLDER = "__AVALAN_GENERATED_OUTPUT_PREFIX__"
_SHELL_COMPOSITION_MODES: tuple[ShellCompositionMode, ...] = (
    "pipeline",
    "serial",
    "parallel",
)


def _assert_composition_mode(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert (
        value in _SHELL_COMPOSITION_MODES
    ), f"{field_name} must be pipeline, serial, or parallel"


def _assert_stage_count_at_most(
    stage_count: int,
    max_pipeline_stages: int,
) -> None:
    _assert_positive_int(max_pipeline_stages, "max_pipeline_stages")
    assert (
        stage_count <= max_pipeline_stages
    ), "steps must not exceed max_pipeline_stages"


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
class ShellStreamRef:
    step_id: str
    stream: Literal["stdout"]

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.step_id, "step_id")
        assert self.stream == "stdout", "stream must be stdout"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCommandStepRequest:
    id: str
    command: str
    options: dict[str, object] = field(default_factory=dict)
    paths: tuple[str, ...] = ()
    cwd: str | None = None
    stdin_from: ShellStreamRef | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        _assert_non_empty_string(self.command, "command")
        assert isinstance(self.options, dict), "options must be a dictionary"
        _assert_string_tuple(self.paths, "paths")
        if self.cwd is not None:
            _assert_non_empty_string(self.cwd, "cwd")
        if self.stdin_from is not None:
            assert isinstance(
                self.stdin_from,
                ShellStreamRef,
            ), "stdin_from must be a shell stream reference"
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "paths", tuple(self.paths))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCompositionRequest:
    mode: ShellCompositionMode = "pipeline"
    steps: tuple[ShellCommandStepRequest, ...]
    timeout_seconds: float | None = None
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    max_intermediate_bytes: int | None = None

    def __post_init__(self) -> None:
        _assert_composition_mode(self.mode, "mode")
        assert isinstance(self.steps, tuple), "steps must be a tuple"
        assert self.steps, "steps must not be empty"
        step_ids: set[str] = set()
        for step in self.steps:
            assert isinstance(
                step,
                ShellCommandStepRequest,
            ), "steps must contain shell command step requests"
            assert step.id not in step_ids, "steps must have unique ids"
            step_ids.add(step.id)
        for step in self.steps:
            if step.stdin_from is not None:
                assert (
                    step.stdin_from.step_id in step_ids
                ), "stdin_from must reference a known step"
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
        _assert_optional_non_negative_int(
            self.max_intermediate_bytes,
            "max_intermediate_bytes",
        )
        object.__setattr__(self, "steps", tuple(self.steps))

    def validate_stage_count(
        self,
        max_pipeline_stages: int = DEFAULT_MAX_PIPELINE_STAGES,
    ) -> None:
        _assert_stage_count_at_most(
            len(self.steps),
            max_pipeline_stages,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ExecutionSpec:
    _policy_owned: InitVar[object]
    backend: ShellExecutionModeValue
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
        assert self.backend in {
            "local",
            "sandbox",
            "container",
        }, "backend must be local, sandbox, or container"
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
    backend: ShellExecutionModeValue,
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
class ShellExecutionStepSpec:
    id: str
    spec: ExecutionSpec
    stdin_from: ShellStreamRef | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        assert isinstance(
            self.spec,
            ExecutionSpec,
        ), "spec must be a shell execution spec"
        if self.stdin_from is not None:
            assert isinstance(
                self.stdin_from,
                ShellStreamRef,
            ), "stdin_from must be a shell stream reference"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCompositionSpec:
    mode: ShellCompositionMode
    steps: tuple[ShellExecutionStepSpec, ...]
    timeout_seconds: float
    max_stdout_bytes: int
    max_stderr_bytes: int
    max_intermediate_bytes: int

    def __post_init__(self) -> None:
        _assert_composition_mode(self.mode, "mode")
        assert isinstance(self.steps, tuple), "steps must be a tuple"
        assert self.steps, "steps must not be empty"
        step_ids: set[str] = set()
        for step in self.steps:
            assert isinstance(
                step,
                ShellExecutionStepSpec,
            ), "steps must contain shell execution step specs"
            assert step.id not in step_ids, "steps must have unique ids"
            step_ids.add(step.id)
        for step in self.steps:
            if step.stdin_from is not None:
                assert (
                    step.stdin_from.step_id in step_ids
                ), "stdin_from must reference a known step"
        _assert_positive_number(self.timeout_seconds, "timeout_seconds")
        _assert_non_negative_int(self.max_stdout_bytes, "max_stdout_bytes")
        _assert_non_negative_int(self.max_stderr_bytes, "max_stderr_bytes")
        _assert_non_negative_int(
            self.max_intermediate_bytes,
            "max_intermediate_bytes",
        )
        object.__setattr__(self, "steps", tuple(self.steps))

    def validate_stage_count(
        self,
        max_pipeline_stages: int = DEFAULT_MAX_PIPELINE_STAGES,
    ) -> None:
        _assert_stage_count_at_most(
            len(self.steps),
            max_pipeline_stages,
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


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellExecutionStepResult:
    id: str
    command: str
    status: ShellExecutionStatus
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: int
    error_code: ShellExecutionErrorCode | None = None
    error_message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        _assert_non_empty_string(self.command, "command")
        assert isinstance(
            self.status,
            ShellExecutionStatus,
        ), "status must be a shell execution status"
        if self.exit_code is not None:
            _assert_int(self.exit_code, "exit_code")
        assert isinstance(self.stdout, str), "stdout must be a string"
        assert isinstance(self.stderr, str), "stderr must be a string"
        _assert_non_negative_int(self.stdout_bytes, "stdout_bytes")
        _assert_non_negative_int(self.stderr_bytes, "stderr_bytes")
        _assert_bool(self.stdout_truncated, "stdout_truncated")
        _assert_bool(self.stderr_truncated, "stderr_truncated")
        _assert_non_negative_int(self.duration_ms, "duration_ms")
        if self.error_code is not None:
            assert isinstance(
                self.error_code,
                ShellExecutionErrorCode,
            ), "error_code must be a shell execution error code"
        if self.error_message is not None:
            _assert_non_empty_string(self.error_message, "error_message")
        assert isinstance(self.metadata, dict), "metadata must be a dictionary"
        object.__setattr__(self, "metadata", dict(self.metadata))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCompositionResult:
    mode: ShellCompositionMode
    status: ShellExecutionStatus
    stdout: str
    stderr: str
    steps: tuple[ShellExecutionStepResult, ...]
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
        _assert_composition_mode(self.mode, "mode")
        assert isinstance(
            self.status,
            ShellExecutionStatus,
        ), "status must be a shell execution status"
        assert isinstance(self.stdout, str), "stdout must be a string"
        assert isinstance(self.stderr, str), "stderr must be a string"
        assert isinstance(self.steps, tuple), "steps must be a tuple"
        assert self.steps, "steps must not be empty"
        step_ids: set[str] = set()
        for step in self.steps:
            assert isinstance(
                step,
                ShellExecutionStepResult,
            ), "steps must contain shell execution step results"
            assert step.id not in step_ids, "steps must have unique ids"
            step_ids.add(step.id)
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
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "metadata", dict(self.metadata))


class ShellFormattedResult(str):
    execution_result: ExecutionResult

    def __new__(
        cls,
        value: str,
        execution_result: ExecutionResult,
    ) -> "ShellFormattedResult":
        assert isinstance(value, str), "value must be a string"
        assert isinstance(
            execution_result,
            ExecutionResult,
        ), "execution_result must be a shell execution result"
        formatted = str.__new__(cls, value)
        formatted.execution_result = execution_result
        return formatted

    def __copy__(self) -> "ShellFormattedResult":
        return self

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> "ShellFormattedResult":
        return self

    def __reduce__(
        self,
    ) -> tuple[type["ShellFormattedResult"], tuple[str, ExecutionResult]]:
        return (self.__class__, (str(self), self.execution_result))
