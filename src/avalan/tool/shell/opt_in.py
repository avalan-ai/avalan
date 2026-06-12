from .settings import ShellToolSettings

from collections.abc import Sequence

SHELL_TOOL_NAMESPACE = "shell"
SHELL_TOOL_WILDCARD = "shell.*"


def normalize_shell_enabled_tools(
    enabled_tools: Sequence[str] | None,
) -> list[str] | None:
    """Return enabled tools with shell wildcard names normalized."""
    if enabled_tools is None:
        return None

    normalized: list[str] = []
    for tool_name in enabled_tools:
        assert isinstance(tool_name, str), "enabled tools must be strings"
        assert tool_name.strip(), "enabled tools must not be empty"
        normalized.append(
            SHELL_TOOL_NAMESPACE
            if tool_name == SHELL_TOOL_WILDCARD
            else tool_name
        )
    return normalized


def enables_shell_tools(enabled_tools: Sequence[str] | None) -> bool:
    """Return whether an enabled-tool selection targets shell tools."""
    if enabled_tools is None:
        return False

    return any(
        tool_name == SHELL_TOOL_NAMESPACE
        or tool_name == SHELL_TOOL_WILDCARD
        or (
            tool_name.startswith(f"{SHELL_TOOL_NAMESPACE}.")
            and len(tool_name) > len(f"{SHELL_TOOL_NAMESPACE}.")
        )
        for tool_name in enabled_tools
    )


def should_append_shell_toolset(
    *,
    shell_settings: ShellToolSettings | None,
    enabled_tools: Sequence[str] | None,
) -> bool:
    """Return whether shell should be an available toolset."""
    return shell_settings is not None or enables_shell_tools(enabled_tools)
