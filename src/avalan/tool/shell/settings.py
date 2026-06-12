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
from .registry import SHELL_COMMAND_IDS

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from re import compile as compile_pattern
from types import MappingProxyType
from typing import ClassVar, Literal, final

_TESSERACT_LANGUAGE_PATTERN = compile_pattern(r"^[A-Za-z0-9_]+$")


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellToolSettings:
    CLI_SCALAR_FIELDS: ClassVar[tuple[str, ...]] = (
        "backend",
        "workspace_root",
        "cwd",
        "default_timeout_seconds",
        "max_timeout_seconds",
        "max_stdout_bytes",
        "max_stderr_bytes",
        "max_stdin_bytes",
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
        "allow_media_tools",
        "allow_absolute_paths",
        "allow_symlinks",
        "allow_hidden",
    )

    backend: Literal["local"] = "local"
    workspace_root: str = "."
    cwd: str = "."
    default_timeout_seconds: float = 10.0
    max_timeout_seconds: float = 60.0
    max_stdout_bytes: int = 65536
    max_stderr_bytes: int = 32768
    max_stdin_bytes: int = 0
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
    max_pdf_raster_dpi: int = 300
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

    def __post_init__(self) -> None:
        assert self.backend == "local", "backend must be local"
        _assert_non_empty_string(self.workspace_root, "workspace_root")
        _assert_non_empty_string(self.cwd, "cwd")
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
    "allow_media_tools",
    "allow_write",
    "allow_shell",
    "allow_absolute_paths",
    "allow_symlinks",
    "allow_hidden",
)


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
