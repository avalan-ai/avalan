from ...container import ContainerProfileSelection
from ...isolation import SandboxProfileSelection
from ...types import (
    assert_absolute_path_mapping as _assert_absolute_path_mapping,
)
from ...types import (
    assert_absolute_path_sequence as _assert_absolute_path_sequence,
)
from ...types import (
    assert_bool as _assert_bool,
)
from ...types import (
    assert_env_name as _assert_env_name,
)
from ...types import (
    assert_known_string_sequence as _assert_known_string_sequence,
)
from ...types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ...types import (
    assert_non_empty_string_sequence as _assert_non_empty_string_sequence,
)
from ...types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from ...types import (
    assert_optional_bounded_number as _assert_optional_bounded_number,
)
from ...types import (
    assert_positive_int as _assert_positive_int,
)
from .entities import ShellExecutionModeValue
from .git import (
    SHELL_GIT_CAPABILITY_IDS,
    SHELL_GIT_COMMAND_IDS,
    SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
    ShellGitCapability,
    ShellGitCommandName,
)
from .registry import SHELL_COMMAND_IDS

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from re import compile as compile_pattern
from types import MappingProxyType
from typing import ClassVar, Literal, cast, final

_TESSERACT_LANGUAGE_PATTERN = compile_pattern(r"^[A-Za-z0-9_]+$")
ShellPipelineTransport = Literal["buffered", "native"]
_PIPELINE_TRANSPORTS: tuple[ShellPipelineTransport, ...] = (
    "buffered",
    "native",
)
ShellGitCredentialPolicy = Literal["deny", "allow_explicit"]
_GIT_CREDENTIAL_POLICIES: tuple[ShellGitCredentialPolicy, ...] = (
    "deny",
    "allow_explicit",
)
_GIT_REMOTE_PROTOCOLS: tuple[str, ...] = ("file", "https")
_GIT_REMOTE_MANAGEMENT_COMMAND_IDS: tuple[str, ...] = tuple(
    command.value
    for command in (
        ShellGitCommandName.REMOTE_LIST,
        ShellGitCommandName.REMOTE_ADD,
        ShellGitCommandName.REMOTE_SET_URL,
        ShellGitCommandName.REMOTE_REMOVE,
        ShellGitCommandName.REMOTE_RENAME,
    )
)
_GIT_HOST_PATTERN = compile_pattern(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellGitToolSettings:
    workspace_root: str = "."
    cwd: str = "."
    capabilities: Sequence[str] = field(
        default_factory=lambda: (ShellGitCapability.READ.value,),
    )
    allowed_commands: Sequence[str] = field(
        default_factory=lambda: SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
    )
    default_timeout_seconds: float = 10.0
    max_timeout_seconds: float = 60.0
    max_stdout_bytes: int = 65536
    max_stderr_bytes: int = 32768
    max_diff_bytes: int = 131072
    max_log_count: int = 50
    max_grep_matches: int = 1000
    max_pathspecs: int = 64
    max_pathspec_bytes: int = 4096
    max_revision_bytes: int = 256
    max_commit_message_bytes: int = 4096
    allow_external_diff: bool = False
    allow_textconv: bool = False
    allow_optional_locks: bool = False
    allow_submodules: bool = False
    allow_bare_repositories: bool = False
    allow_linked_worktrees: bool = False
    allow_alternates: bool = False
    allow_submodule_update: bool = False
    allowed_remote_protocols: Sequence[str] = field(
        default_factory=lambda: ("https",),
    )
    allowed_remote_hosts: Sequence[str] = field(default_factory=tuple)
    credential_policy: ShellGitCredentialPolicy = "deny"
    allow_remote_credentials: bool = False
    redact_remote_urls: bool = True
    redact_credentials: bool = True
    redact_author_emails: bool = False

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.workspace_root, "git.workspace_root")
        _assert_non_empty_string(self.cwd, "git.cwd")
        object.__setattr__(
            self,
            "capabilities",
            _normalized_git_capabilities(self.capabilities),
        )
        object.__setattr__(
            self,
            "allowed_commands",
            _normalized_git_commands(self.allowed_commands),
        )
        _assert_positive_timeout_order(
            self.default_timeout_seconds,
            self.max_timeout_seconds,
            "git.default_timeout_seconds",
            "git.max_timeout_seconds",
        )
        for field_name in _GIT_POSITIVE_INT_FIELDS:
            _assert_positive_int(
                getattr(self, field_name), f"git.{field_name}"
            )
        for field_name in _GIT_BOOLEAN_FIELDS:
            _assert_bool(getattr(self, field_name), f"git.{field_name}")
        object.__setattr__(
            self,
            "allowed_remote_protocols",
            _normalized_git_remote_protocols(
                self.allowed_remote_protocols,
            ),
        )
        object.__setattr__(
            self,
            "allowed_remote_hosts",
            _normalized_git_remote_hosts(self.allowed_remote_hosts),
        )
        _assert_git_credential_policy(self.credential_policy)
        if self.allow_remote_credentials:
            object.__setattr__(
                self,
                "credential_policy",
                "allow_explicit",
            )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellToolSettings:
    CLI_SCALAR_FIELDS: ClassVar[tuple[str, ...]] = (
        "backend",
        "execution_mode",
        "workspace_root",
        "cwd",
        "materialized_input_files_dir",
        "input_file_manifest_enabled",
        "input_file_manifest_message",
        "input_file_manifest_path_message",
        "default_timeout_seconds",
        "max_timeout_seconds",
        "max_stdout_bytes",
        "max_stderr_bytes",
        "max_stdin_bytes",
        "max_pipeline_stages",
        "max_pipeline_bytes",
        "max_intermediate_bytes",
        "pipeline_transport",
        "max_arguments",
        "max_argument_bytes",
        "max_command_bytes",
        "max_path_count",
        "max_glob_count",
        "max_glob_bytes_per_glob",
        "max_total_glob_bytes",
        "max_full_file_bytes",
        "max_rg_columns",
        "max_rg_context_lines",
        "max_rg_matches_per_file",
        "max_head_lines",
        "max_tail_lines",
        "max_text_filter_input_bytes",
        "max_filter_program_bytes",
        "max_filter_pattern_bytes",
        "max_filter_selectors",
        "max_awk_fields",
        "max_awk_separator_bytes",
        "max_json_input_bytes",
        "max_jq_filter_bytes",
        "max_pdf_input_bytes",
        "max_pdf_text_pages",
        "max_pdf_raster_pages",
        "max_pdf_raster_dpi",
        "max_raster_long_edge_pixels",
        "max_raster_pixels",
        "max_output_files",
        "max_output_file_bytes",
        "max_total_output_file_bytes",
        "max_inline_output_file_bytes",
        "max_ocr_input_bytes",
        "max_ocr_pixels",
        "max_ocr_languages",
        "max_tesseract_dpi",
        "stream_read_chunk_bytes",
        "max_concurrent_processes",
        "max_concurrent_heavy_processes",
        "default_pdf_timeout_seconds",
        "max_pdf_timeout_seconds",
        "default_ocr_timeout_seconds",
        "max_ocr_timeout_seconds",
        "tesseract_thread_limit",
        "allow_pipelines",
        "allow_media_tools",
        "allow_absolute_paths",
        "allow_symlinks",
        "allow_hidden",
    )

    backend: ShellExecutionModeValue | None = None
    execution_mode: ShellExecutionModeValue | None = None
    workspace_root: str = "."
    cwd: str = "."
    materialized_input_files_dir: str = "avalan-input-files"
    input_file_manifest_enabled: bool = True
    input_file_manifest_message: str = "Attached files available to tools:"
    input_file_manifest_path_message: str = (
        "Use these path values as tool arguments."
    )
    default_timeout_seconds: float = 10.0
    max_timeout_seconds: float = 60.0
    max_stdout_bytes: int = 65536
    max_stderr_bytes: int = 32768
    max_stdin_bytes: int = 0
    max_pipeline_stages: int = 8
    max_pipeline_bytes: int = 1048576
    max_intermediate_bytes: int = 1048576
    pipeline_transport: ShellPipelineTransport = "buffered"
    max_arguments: int = 128
    max_argument_bytes: int = 8192
    max_command_bytes: int = 32768
    max_path_count: int = 128
    max_glob_count: int = 32
    max_glob_bytes_per_glob: int = 2048
    max_total_glob_bytes: int = 8192
    max_full_file_bytes: int = 1048576
    max_rg_columns: int = 1000
    max_rg_context_lines: int = 10
    max_rg_matches_per_file: int = 1000
    max_head_lines: int = 500
    max_tail_lines: int = 500
    max_text_filter_input_bytes: int = 1048576
    max_filter_program_bytes: int = 8192
    max_filter_pattern_bytes: int = 2048
    max_filter_selectors: int = 32
    max_awk_fields: int = 64
    max_awk_separator_bytes: int = 16
    max_json_input_bytes: int = 5242880
    max_jq_filter_bytes: int = 4096
    max_pdf_input_bytes: int = 104857600
    max_pdf_text_pages: int = 50
    max_pdf_raster_pages: int = 8
    max_pdf_raster_dpi: int = 600
    max_raster_long_edge_pixels: int = 2048
    max_raster_pixels: int = 40000000
    max_output_files: int = 8
    max_output_file_bytes: int = 10485760
    max_total_output_file_bytes: int = 52428800
    max_inline_output_file_bytes: int = 2097152
    max_ocr_input_bytes: int = 26214400
    max_ocr_pixels: int = 20000000
    max_ocr_languages: int = 4
    max_tesseract_dpi: int = 600
    stream_read_chunk_bytes: int = 8192
    max_concurrent_processes: int = 4
    max_concurrent_heavy_processes: int = 1
    default_pdf_timeout_seconds: float = 30.0
    max_pdf_timeout_seconds: float = 120.0
    default_ocr_timeout_seconds: float = 60.0
    max_ocr_timeout_seconds: float = 300.0
    tesseract_thread_limit: int = 1
    allow_pipelines: bool = False
    allow_media_tools: bool = False
    allow_write: bool = False
    allow_shell: bool = False
    allow_absolute_paths: bool = False
    allow_symlinks: bool = False
    allow_hidden: bool = False
    allowed_commands: Sequence[str] = field(
        default_factory=lambda: SHELL_COMMAND_IDS,
    )
    allowed_pdf_raster_formats: Sequence[str] = field(
        default_factory=lambda: ("png",),
    )
    allowed_tesseract_output_formats: Sequence[str] = field(
        default_factory=lambda: ("txt",),
    )
    allowed_tesseract_languages: Sequence[str] = field(
        default_factory=lambda: ("eng",),
    )
    environment: Mapping[str, str] = field(default_factory=dict)
    environment_allowlist: Sequence[str] = field(default_factory=tuple)
    executable_paths: Mapping[str, str] = field(default_factory=dict)
    executable_search_paths: Sequence[str] = field(default_factory=tuple)
    git: ShellGitToolSettings | Mapping[str, object] = field(
        default_factory=ShellGitToolSettings,
    )
    container: ContainerProfileSelection | None = None
    sandbox: SandboxProfileSelection | None = None

    def __post_init__(self) -> None:
        execution_mode = _normalized_execution_mode(
            self.execution_mode,
            self.backend,
        )
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "backend", execution_mode)
        object.__setattr__(self, "git", _coerce_git_settings(self.git))
        _assert_non_empty_string(self.workspace_root, "workspace_root")
        _assert_non_empty_string(self.cwd, "cwd")
        _assert_relative_path(
            self.materialized_input_files_dir,
            "materialized_input_files_dir",
        )
        _assert_non_empty_string(
            self.input_file_manifest_message,
            "input_file_manifest_message",
        )
        _assert_non_empty_string(
            self.input_file_manifest_path_message,
            "input_file_manifest_path_message",
        )
        _assert_positive_timeout_order(
            self.default_timeout_seconds,
            self.max_timeout_seconds,
            "default_timeout_seconds",
            "max_timeout_seconds",
        )
        _assert_positive_timeout_order(
            self.default_pdf_timeout_seconds,
            self.max_pdf_timeout_seconds,
            "default_pdf_timeout_seconds",
            "max_pdf_timeout_seconds",
        )
        _assert_positive_timeout_order(
            self.default_ocr_timeout_seconds,
            self.max_ocr_timeout_seconds,
            "default_ocr_timeout_seconds",
            "max_ocr_timeout_seconds",
        )
        for field_name in _POSITIVE_INT_FIELDS:
            _assert_positive_int(getattr(self, field_name), field_name)
        _assert_non_negative_int(self.max_stdin_bytes, "max_stdin_bytes")
        assert self.max_stdin_bytes == 0, "max_stdin_bytes must be zero"
        for field_name in _BOOLEAN_FIELDS:
            _assert_bool(getattr(self, field_name), field_name)
        _assert_pipeline_transport(self.pipeline_transport)
        assert not self.allow_write, "allow_write is reserved"
        assert not self.allow_shell, "allow_shell is reserved"
        _assert_known_commands(self.allowed_commands)
        _assert_non_empty_known_values(
            self.allowed_pdf_raster_formats,
            "allowed_pdf_raster_formats",
            ("png",),
        )
        _assert_non_empty_known_values(
            self.allowed_tesseract_output_formats,
            "allowed_tesseract_output_formats",
            ("txt",),
        )
        _assert_tesseract_languages(self.allowed_tesseract_languages)
        assert isinstance(
            self.environment,
            Mapping,
        ), "environment must be a mapping"
        for name, value in self.environment.items():
            _assert_env_name(name, "environment key")
            _assert_non_empty_string(value, f"environment.{name}")
        if self.environment_allowlist:
            _assert_non_empty_string_sequence(
                self.environment_allowlist,
                "environment_allowlist",
            )
        for name in self.environment_allowlist:
            _assert_env_name(name, "environment_allowlist")
        _assert_absolute_path_mapping(
            self.executable_paths,
            "executable_paths",
        )
        for command in self.executable_paths:
            assert (
                command in SHELL_COMMAND_IDS
            ), "executable_paths must be known"
        _assert_absolute_path_sequence(
            self.executable_search_paths,
            "executable_search_paths",
        )
        assert isinstance(
            self.git,
            ShellGitToolSettings,
        ), "git must be shell Git tool settings"
        if self.container is not None:
            assert isinstance(self.container, ContainerProfileSelection)
        if self.sandbox is not None:
            assert isinstance(self.sandbox, SandboxProfileSelection)
        assert not (
            execution_mode != "container" and self.container is not None
        ), "container policy requires shell execution mode container"
        assert not (
            execution_mode != "sandbox" and self.sandbox is not None
        ), "sandbox policy requires shell execution mode sandbox"
        assert not (
            execution_mode == "container" and self.sandbox is not None
        ), "container mode cannot carry sandbox policy"
        assert not (
            execution_mode == "sandbox" and self.container is not None
        ), "sandbox mode cannot carry container policy"
        object.__setattr__(
            self,
            "allowed_commands",
            tuple(self.allowed_commands),
        )
        object.__setattr__(
            self,
            "allowed_pdf_raster_formats",
            tuple(self.allowed_pdf_raster_formats),
        )
        object.__setattr__(
            self,
            "allowed_tesseract_output_formats",
            tuple(self.allowed_tesseract_output_formats),
        )
        object.__setattr__(
            self,
            "allowed_tesseract_languages",
            tuple(self.allowed_tesseract_languages),
        )
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(self.environment)),
        )
        object.__setattr__(
            self,
            "environment_allowlist",
            tuple(self.environment_allowlist),
        )
        object.__setattr__(
            self,
            "executable_paths",
            MappingProxyType(dict(self.executable_paths)),
        )
        object.__setattr__(
            self,
            "executable_search_paths",
            tuple(self.executable_search_paths),
        )


_POSITIVE_INT_FIELDS = (
    "max_stdout_bytes",
    "max_stderr_bytes",
    "max_pipeline_stages",
    "max_pipeline_bytes",
    "max_intermediate_bytes",
    "max_arguments",
    "max_argument_bytes",
    "max_command_bytes",
    "max_path_count",
    "max_glob_count",
    "max_glob_bytes_per_glob",
    "max_total_glob_bytes",
    "max_full_file_bytes",
    "max_rg_columns",
    "max_rg_context_lines",
    "max_rg_matches_per_file",
    "max_head_lines",
    "max_tail_lines",
    "max_text_filter_input_bytes",
    "max_filter_program_bytes",
    "max_filter_pattern_bytes",
    "max_filter_selectors",
    "max_awk_fields",
    "max_awk_separator_bytes",
    "max_json_input_bytes",
    "max_jq_filter_bytes",
    "max_pdf_input_bytes",
    "max_pdf_text_pages",
    "max_pdf_raster_pages",
    "max_pdf_raster_dpi",
    "max_raster_long_edge_pixels",
    "max_raster_pixels",
    "max_output_files",
    "max_output_file_bytes",
    "max_total_output_file_bytes",
    "max_inline_output_file_bytes",
    "max_ocr_input_bytes",
    "max_ocr_pixels",
    "max_ocr_languages",
    "max_tesseract_dpi",
    "stream_read_chunk_bytes",
    "max_concurrent_processes",
    "max_concurrent_heavy_processes",
    "tesseract_thread_limit",
)

_BOOLEAN_FIELDS = (
    "input_file_manifest_enabled",
    "allow_pipelines",
    "allow_media_tools",
    "allow_write",
    "allow_shell",
    "allow_absolute_paths",
    "allow_symlinks",
    "allow_hidden",
)

_GIT_POSITIVE_INT_FIELDS = (
    "max_stdout_bytes",
    "max_stderr_bytes",
    "max_diff_bytes",
    "max_log_count",
    "max_grep_matches",
    "max_pathspecs",
    "max_pathspec_bytes",
    "max_revision_bytes",
    "max_commit_message_bytes",
)

_GIT_BOOLEAN_FIELDS = (
    "allow_external_diff",
    "allow_textconv",
    "allow_optional_locks",
    "allow_submodules",
    "allow_bare_repositories",
    "allow_linked_worktrees",
    "allow_alternates",
    "allow_submodule_update",
    "allow_remote_credentials",
    "redact_remote_urls",
    "redact_credentials",
    "redact_author_emails",
)


def _normalized_execution_mode(
    execution_mode: object,
    backend: object,
) -> ShellExecutionModeValue:
    if execution_mode is None and backend is None:
        return "local"
    if execution_mode is None:
        execution_mode = backend
    elif backend is not None:
        assert (
            execution_mode == backend
        ), "execution_mode and backend must match"
    assert isinstance(
        execution_mode,
        str,
    ), "execution_mode must be a string"
    assert execution_mode in (
        "local",
        "sandbox",
        "container",
    ), "execution_mode must be local, sandbox, or container"
    return cast(ShellExecutionModeValue, execution_mode)


def _assert_positive_timeout_order(
    default_value: object,
    max_value: object,
    default_field_name: str,
    max_field_name: str,
) -> None:
    _assert_optional_bounded_number(
        default_value,
        default_field_name,
        min_value=0,
        min_inclusive=False,
    )
    _assert_optional_bounded_number(
        max_value,
        max_field_name,
        min_value=0,
        min_inclusive=False,
    )
    assert isinstance(default_value, int | float)
    assert isinstance(max_value, int | float)
    assert (
        default_value <= max_value
    ), f"{default_field_name} must not exceed {max_field_name}"


def _assert_relative_path(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    path = Path(value)
    assert not path.is_absolute(), f"{field_name} must be relative"
    assert ".." not in path.parts, f"{field_name} must not contain .."


def _assert_known_commands(value: object) -> None:
    _assert_known_string_sequence(
        value,
        "allowed_commands",
        SHELL_COMMAND_IDS,
    )


def _assert_non_empty_known_values(
    value: object,
    field_name: str,
    known_values: tuple[str, ...],
) -> None:
    _assert_known_string_sequence(value, field_name, known_values)


def _assert_pipeline_transport(value: object) -> None:
    _assert_non_empty_string(value, "pipeline_transport")
    assert (
        value in _PIPELINE_TRANSPORTS
    ), "pipeline_transport must be buffered or native"


def _coerce_git_settings(value: object) -> ShellGitToolSettings:
    if isinstance(value, ShellGitToolSettings):
        return value
    assert isinstance(value, Mapping), "git must be shell Git tool settings"
    return ShellGitToolSettings(**dict(value))


def _normalized_git_capabilities(value: object) -> tuple[str, ...]:
    _assert_non_empty_string_sequence(value, "git.capabilities")
    assert isinstance(value, Sequence)
    capabilities: list[str] = []
    for item in value:
        capability = (
            item.value if isinstance(item, ShellGitCapability) else item
        )
        assert (
            capability in SHELL_GIT_CAPABILITY_IDS
        ), f"git.capabilities contains unsupported value: {item!r}"
        if capability not in capabilities:
            capabilities.append(capability)
    return tuple(capabilities)


def _normalized_git_commands(value: object) -> tuple[str, ...]:
    assert isinstance(
        value, Sequence
    ), "git.allowed_commands must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), "git.allowed_commands must be a sequence"
    commands: list[str] = []
    for item in value:
        command = item.value if isinstance(item, ShellGitCommandName) else item
        assert isinstance(
            command,
            str,
        ), "git.allowed_commands must contain strings"
        assert (
            command.strip()
        ), "git.allowed_commands must not contain empty values"
        if command == ShellGitCapability.REMOTE.value:
            for remote_command in _GIT_REMOTE_MANAGEMENT_COMMAND_IDS:
                if remote_command not in commands:
                    commands.append(remote_command)
            continue
        assert (
            command in SHELL_GIT_COMMAND_IDS
        ), f"git.allowed_commands contains unsupported value: {item!r}"
        if command not in commands:
            commands.append(command)
    return tuple(commands)


def _normalized_git_remote_protocols(value: object) -> tuple[str, ...]:
    assert isinstance(
        value,
        Sequence,
    ), "git.allowed_remote_protocols must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), "git.allowed_remote_protocols must be a sequence"
    protocols: list[str] = []
    for item in value:
        _assert_non_empty_string(item, "git.allowed_remote_protocols")
        assert isinstance(item, str)
        protocol = item.lower()
        assert (
            protocol in _GIT_REMOTE_PROTOCOLS
        ), f"git.allowed_remote_protocols contains unsafe value: {item!r}"
        if protocol not in protocols:
            protocols.append(protocol)
    return tuple(protocols)


def _normalized_git_remote_hosts(value: object) -> tuple[str, ...]:
    assert isinstance(
        value,
        Sequence,
    ), "git.allowed_remote_hosts must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), "git.allowed_remote_hosts must be a sequence"
    hosts: list[str] = []
    for item in value:
        _assert_non_empty_string(item, "git.allowed_remote_hosts")
        assert isinstance(item, str)
        assert not any(
            marker in item for marker in (":", "/", "@", "*")
        ), "git.allowed_remote_hosts contains unsafe value"
        assert _GIT_HOST_PATTERN.match(
            item,
        ), "git.allowed_remote_hosts contains unsafe value"
        host = item.lower()
        if host not in hosts:
            hosts.append(host)
    return tuple(hosts)


def _assert_git_credential_policy(value: object) -> None:
    _assert_non_empty_string(value, "git.credential_policy")
    assert (
        value in _GIT_CREDENTIAL_POLICIES
    ), "git.credential_policy must be deny or allow_explicit"


def _assert_tesseract_languages(value: object) -> None:
    _assert_non_empty_string_sequence(value, "allowed_tesseract_languages")
    assert isinstance(
        value,
        Sequence,
    ), "allowed_tesseract_languages must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), "allowed_tesseract_languages must be a sequence"
    for item in value:
        assert _TESSERACT_LANGUAGE_PATTERN.match(
            item,
        ), "allowed_tesseract_languages contains unsupported value"
