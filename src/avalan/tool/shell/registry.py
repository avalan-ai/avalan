from .commands import SHELL_COMMANDS as SHELL_COMMANDS
from .commands import (
    ShellCommandDefinition as ShellCommandDefinition,
)
from .commands import (
    ShellDependencyGroup as ShellDependencyGroup,
)

SHELL_COMMAND_IDS: tuple[str, ...] = tuple(
    command.logical_id for command in SHELL_COMMANDS
)
SHELL_COMMAND_DEFINITIONS: dict[str, ShellCommandDefinition] = {
    command.logical_id: command for command in SHELL_COMMANDS
}
