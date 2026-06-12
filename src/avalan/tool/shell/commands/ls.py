from ..entities import ShellExecutionErrorCode
from .base import (
    NormalizedPath,
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _relative_argv_path,
    _validate_known_options,
    _validate_path_kind,
    is_denied_display_path,
    policy_denied,
)


def _optional_single_path(
    paths: tuple[NormalizedPath, ...],
    *,
    allowed_kinds: tuple[str, ...],
    command: str,
) -> NormalizedPath | None:
    if len(paths) > 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{command} requires at most one path",
        )
    if not paths:
        return None
    path = paths[0]
    _validate_path_kind(path, allowed_kinds=allowed_kinds, command=command)
    return path


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    _validate_known_options(
        context.request.options,
        allowed_options=set(),
        command="ls",
    )
    path = _optional_single_path(
        context.paths,
        allowed_kinds=("file", "directory", "any"),
        command="ls",
    )
    path_argument = "."
    display_path_argument = "."
    if path is not None:
        path_argument = _relative_argv_path(context.workspace.cwd, path.path)
        display_path_argument = path.display_path
    argv_parts = [context.executable_name, "-1p"]
    if context.settings.allow_hidden:
        argv_parts.append("-A")
    argv_parts.extend(("--", path_argument))
    display_parts = list(argv_parts)
    display_parts[-1] = display_path_argument
    return tuple(argv_parts), tuple(display_parts), None


def filter_output(value: str) -> str:
    assert isinstance(value, str), "value must be a string"
    return "\n".join(_filtered_output_line(line) for line in value.split("\n"))


def _filtered_output_line(line: str) -> str:
    path = line.rstrip("/")
    if is_denied_display_path(path):
        return "[redacted_path]"
    return line


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="ls",
    executable_name="ls",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
    output_filter=filter_output,
)
