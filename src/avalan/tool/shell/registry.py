from ...types import (
    assert_bool as _assert_bool,
)
from ...types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ...types import (
    assert_string_tuple as _assert_string_tuple,
)

from dataclasses import dataclass
from enum import StrEnum
from typing import final


class ShellDependencyGroup(StrEnum):
    CORE = "core"
    TEXT_FILTERS = "text_filters"
    JSON = "json"
    POPPLER = "poppler"
    OCR = "ocr"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellCommandDefinition:
    logical_id: str
    executable_name: str
    dependency_group: ShellDependencyGroup
    container_package_hints: tuple[str, ...]
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
        _assert_bool(self.public, "public")
        _assert_bool(self.media_risk, "media_risk")
        _assert_bool(self.supports_double_dash, "supports_double_dash")


SHELL_COMMANDS: tuple[ShellCommandDefinition, ...] = (
    ShellCommandDefinition(
        logical_id="rg",
        executable_name="rg",
        dependency_group=ShellDependencyGroup.CORE,
        container_package_hints=("ripgrep",),
    ),
    ShellCommandDefinition(
        logical_id="head",
        executable_name="head",
        dependency_group=ShellDependencyGroup.CORE,
        container_package_hints=("coreutils",),
    ),
    ShellCommandDefinition(
        logical_id="tail",
        executable_name="tail",
        dependency_group=ShellDependencyGroup.CORE,
        container_package_hints=("coreutils",),
    ),
    ShellCommandDefinition(
        logical_id="ls",
        executable_name="ls",
        dependency_group=ShellDependencyGroup.CORE,
        container_package_hints=("coreutils",),
    ),
    ShellCommandDefinition(
        logical_id="cat",
        executable_name="cat",
        dependency_group=ShellDependencyGroup.CORE,
        container_package_hints=("coreutils",),
    ),
    ShellCommandDefinition(
        logical_id="wc",
        executable_name="wc",
        dependency_group=ShellDependencyGroup.CORE,
        container_package_hints=("coreutils",),
    ),
    ShellCommandDefinition(
        logical_id="awk",
        executable_name="awk",
        dependency_group=ShellDependencyGroup.TEXT_FILTERS,
        container_package_hints=("gawk", "mawk"),
        supports_double_dash=False,
    ),
    ShellCommandDefinition(
        logical_id="sed",
        executable_name="sed",
        dependency_group=ShellDependencyGroup.TEXT_FILTERS,
        container_package_hints=("sed",),
        supports_double_dash=False,
    ),
    ShellCommandDefinition(
        logical_id="jq",
        executable_name="jq",
        dependency_group=ShellDependencyGroup.JSON,
        container_package_hints=("jq",),
    ),
    ShellCommandDefinition(
        logical_id="pdftotext",
        executable_name="pdftotext",
        dependency_group=ShellDependencyGroup.POPPLER,
        container_package_hints=("poppler-utils", "poppler"),
        media_risk=True,
        supports_double_dash=False,
    ),
    ShellCommandDefinition(
        logical_id="pdftoppm",
        executable_name="pdftoppm",
        dependency_group=ShellDependencyGroup.POPPLER,
        container_package_hints=("poppler-utils", "poppler"),
        media_risk=True,
        supports_double_dash=False,
    ),
    ShellCommandDefinition(
        logical_id="tesseract",
        executable_name="tesseract",
        dependency_group=ShellDependencyGroup.OCR,
        container_package_hints=("tesseract-ocr", "tesseract"),
        media_risk=True,
        supports_double_dash=False,
    ),
)
SHELL_COMMAND_IDS: tuple[str, ...] = tuple(
    command.logical_id for command in SHELL_COMMANDS
)
SHELL_COMMAND_DEFINITIONS: dict[str, ShellCommandDefinition] = {
    command.logical_id: command for command in SHELL_COMMANDS
}
