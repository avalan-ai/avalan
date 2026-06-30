from ...filesystem import which_executable as _which_executable
from ...types import (
    assert_absolute_path_mapping as _assert_absolute_path_mapping,
)
from ...types import (
    assert_absolute_path_sequence as _assert_absolute_path_sequence,
)
from ...types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .registry import SHELL_COMMAND_DEFINITIONS, ShellCommandDefinition

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from sys import executable as sys_executable
from types import MappingProxyType
from typing import Protocol, final

_PYTHON_RUNNER_COMMAND_IDS = frozenset(
    {
        "pdfplumber",
        "pypdf",
        "reportlab",
    }
)

ExecutableLookup = Callable[
    [ShellCommandDefinition, tuple[str, ...]],
    Awaitable[str | None],
]


class ExecutableResolver(Protocol):
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        raise NotImplementedError


async def unavailable_executable_lookup(
    command: ShellCommandDefinition,
    search_paths: tuple[str, ...],
) -> str | None:
    _assert_command_definition(command)
    _assert_absolute_path_sequence(search_paths, "search_paths")
    return None


async def trusted_search_path_executable_lookup(
    command: ShellCommandDefinition,
    search_paths: tuple[str, ...],
) -> str | None:
    _assert_command_definition(command)
    _assert_absolute_path_sequence(search_paths, "search_paths")
    if not search_paths:
        return None
    return await _which_executable(command.executable_name, search_paths)


def _default_executable_lookup() -> ExecutableLookup:
    return trusted_search_path_executable_lookup


@final
@dataclass(kw_only=True, slots=True)
class TrustedExecutableResolver:
    executable_paths: Mapping[str, str] = field(default_factory=dict)
    executable_search_paths: Sequence[str] = field(default_factory=tuple)
    lookup: ExecutableLookup = field(
        default_factory=_default_executable_lookup,
    )
    _cache: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _assert_absolute_path_mapping(
            self.executable_paths,
            "executable_paths",
        )
        for command_id in self.executable_paths:
            assert (
                command_id in SHELL_COMMAND_DEFINITIONS
            ), "executable_paths must be known"
        _assert_absolute_path_sequence(
            self.executable_search_paths,
            "executable_search_paths",
        )
        assert callable(self.lookup), "lookup must be callable"
        self.executable_paths = MappingProxyType(dict(self.executable_paths))
        self.executable_search_paths = tuple(self.executable_search_paths)

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        _assert_command_definition(command)
        if command.logical_id in self._cache:
            return self._cache[command.logical_id]

        executable = self.executable_paths.get(command.logical_id)
        if (
            executable is None
            and command.logical_id in _PYTHON_RUNNER_COMMAND_IDS
            and self.lookup is trusted_search_path_executable_lookup
        ):
            executable = await self.lookup(
                command,
                tuple(self.executable_search_paths),
            )
            if executable is None:
                executable = sys_executable
        elif executable is None:
            executable = await self.lookup(
                command,
                tuple(self.executable_search_paths),
            )
        if executable is not None:
            _assert_non_empty_string(executable, "executable")

        self._cache[command.logical_id] = executable
        return executable

    async def resolve_command(self, command_id: str) -> str | None:
        _assert_non_empty_string(command_id, "command_id")
        assert command_id in SHELL_COMMAND_DEFINITIONS, "command_id is unknown"
        return await self.resolve(SHELL_COMMAND_DEFINITIONS[command_id])

    def clear_cache(self) -> None:
        self._cache.clear()


def _assert_command_definition(command: object) -> None:
    assert isinstance(
        command,
        ShellCommandDefinition,
    ), "command must be a shell command definition"
