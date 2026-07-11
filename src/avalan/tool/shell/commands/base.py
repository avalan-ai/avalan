from ....types import (
    assert_bool as _assert_bool,
)
from ....types import (
    assert_media_type as _assert_media_type,
)
from ....types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ....types import (
    assert_string_tuple as _assert_string_tuple,
)
from ..entities import (
    GeneratedOutputPlan,
    PathOperand,
    ShellCommandRequest,
    ShellOutputKind,
)

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from ..filesystem import ShellPathMetadata
    from ..settings import ShellToolSettings


class ShellDependencyGroup(StrEnum):
    CORE = "core"
    TEXT_FILTERS = "text_filters"
    JSON = "json"
    PROCESS = "process"
    POPPLER = "poppler"
    PYTHON_PDF = "python_pdf"
    OCR = "ocr"


ShellCommandArgv = tuple[
    tuple[str, ...],
    tuple[str, ...],
    GeneratedOutputPlan | None,
]
ShellCommandArgvBuilder = Callable[
    ["ShellCommandPolicyContext"],
    ShellCommandArgv,
]
ShellCommandOutputContract = Callable[
    [ShellCommandRequest],
    tuple[str, ShellOutputKind],
]
ShellCommandOutputFilter = Callable[[str], str]


def default_output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    return "text/plain", ShellOutputKind.TEXT


def default_output_filter(value: str) -> str:
    assert isinstance(value, str), "value must be a string"
    return value


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellStreamContract:
    media_types: tuple[str, ...] = ()
    output_kinds: tuple[ShellOutputKind, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(
            self.media_types, tuple
        ), "media_types must be a tuple"
        assert isinstance(
            self.output_kinds, tuple
        ), "output_kinds must be a tuple"
        assert bool(self.media_types) == bool(
            self.output_kinds
        ), "media_types and output_kinds must both be empty or populated"
        assert len(set(self.media_types)) == len(
            self.media_types
        ), "media_types must not contain duplicates"
        assert len(set(self.output_kinds)) == len(
            self.output_kinds
        ), "output_kinds must not contain duplicates"
        for media_type in self.media_types:
            _assert_media_type(media_type, "media_types")
        for output_kind in self.output_kinds:
            assert isinstance(
                output_kind,
                ShellOutputKind,
            ), "output_kinds must contain shell output kinds"

    def accepts(
        self,
        *,
        media_type: str,
        output_kind: ShellOutputKind,
    ) -> bool:
        _assert_media_type(media_type, "media_type")
        assert isinstance(
            output_kind,
            ShellOutputKind,
        ), "output_kind must be a shell output kind"
        return (
            media_type in self.media_types and output_kind in self.output_kinds
        )

    @property
    def supports_stdin(self) -> bool:
        return bool(self.media_types)


DEFAULT_STDIN_CONTRACT = ShellStreamContract()


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class NormalizedWorkspace:
    root: Path
    cwd: Path
    display_cwd: str


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class NormalizedPath:
    operand: PathOperand
    path: Path
    display_path: str
    metadata: "ShellPathMetadata | None"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCommandPolicyContext:
    executable_name: str
    request: ShellCommandRequest
    paths: tuple[NormalizedPath, ...]
    workspace: NormalizedWorkspace
    settings: "ShellToolSettings"
    metadata: dict[str, object]
    stdin_mode: bool = False
    stdin_media_type: str | None = None
    stdin_output_kind: ShellOutputKind | None = None

    def __post_init__(self) -> None:
        _assert_bool(self.stdin_mode, "stdin_mode")
        if self.stdin_media_type is not None:
            _assert_media_type(self.stdin_media_type, "stdin_media_type")
        if self.stdin_output_kind is not None:
            assert isinstance(
                self.stdin_output_kind,
                ShellOutputKind,
            ), "stdin_output_kind must be a shell output kind"
        if self.stdin_mode:
            assert (
                self.stdin_media_type is not None
            ), "stdin_media_type is required in stdin mode"
            assert (
                self.stdin_output_kind is not None
            ), "stdin_output_kind is required in stdin mode"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCommandDefinition:
    logical_id: str
    executable_name: str
    dependency_group: ShellDependencyGroup
    container_package_hints: tuple[str, ...]
    argv_builder: ShellCommandArgvBuilder
    output_contract: ShellCommandOutputContract = default_output_contract
    stdin_contract: ShellStreamContract = DEFAULT_STDIN_CONTRACT
    output_filter: ShellCommandOutputFilter = default_output_filter
    public: bool = True
    media_risk: bool = False
    process_risk: bool = False
    supports_double_dash: bool = True

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.logical_id, "logical_id")
        _assert_non_empty_string(self.executable_name, "executable_name")
        assert isinstance(
            self.dependency_group,
            ShellDependencyGroup,
        ), "dependency_group must be a shell dependency group"
        _assert_string_tuple(
            self.container_package_hints,
            "container_package_hints",
        )
        assert (
            self.container_package_hints
        ), "container_package_hints must not be empty"
        assert callable(self.argv_builder), "argv_builder must be callable"
        assert callable(
            self.output_contract
        ), "output_contract must be callable"
        assert isinstance(
            self.stdin_contract,
            ShellStreamContract,
        ), "stdin_contract must be a shell stream contract"
        assert callable(self.output_filter), "output_filter must be callable"
        _assert_bool(self.public, "public")
        _assert_bool(self.media_risk, "media_risk")
        _assert_bool(self.process_risk, "process_risk")
        _assert_bool(self.supports_double_dash, "supports_double_dash")
