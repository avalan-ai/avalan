from .commands import SHELL_COMMANDS as SHELL_COMMANDS
from .commands import (
    ShellCommandDefinition as ShellCommandDefinition,
)
from .commands import (
    ShellDependencyGroup as ShellDependencyGroup,
)
from .git import (
    SHELL_GIT_COMMAND_CAPABILITIES as SHELL_GIT_COMMAND_CAPABILITIES,
)
from .git import SHELL_GIT_COMMAND_IDS as SHELL_GIT_COMMAND_IDS
from .git import SHELL_GIT_TOOL_COMMANDS as SHELL_GIT_TOOL_COMMANDS
from .git import SHELL_GIT_TOOL_NAMES as SHELL_GIT_TOOL_NAMES

SHELL_COMMAND_IDS: tuple[str, ...] = tuple(
    command.logical_id for command in SHELL_COMMANDS
)
SHELL_COMMAND_DEFINITIONS: dict[str, ShellCommandDefinition] = {
    command.logical_id: command for command in SHELL_COMMANDS
}
