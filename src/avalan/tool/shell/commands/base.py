from ....types import (
    assert_bool as _assert_bool,
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
    POPPLER = "poppler"
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


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCommandDefinition:
    logical_id: str
    executable_name: str
    dependency_group: ShellDependencyGroup
    container_package_hints: tuple[str, ...]
    argv_builder: ShellCommandArgvBuilder
    output_contract: ShellCommandOutputContract = default_output_contract
    output_filter: ShellCommandOutputFilter = default_output_filter
    public: bool = True
    media_risk: bool = False
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
        assert callable(self.output_filter), "output_filter must be callable"
        _assert_bool(self.public, "public")
        _assert_bool(self.media_risk, "media_risk")
        _assert_bool(self.supports_double_dash, "supports_double_dash")
