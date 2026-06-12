from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bounded_int_option,
    _relative_argv_path,
    _single_path,
    _validate_known_options,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"lines"},
        command="tail",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("file", "text_file"),
        command="tail",
    )
    lines = _bounded_int_option(
        request.options,
        "lines",
        default=80,
        min_value=1,
        max_value=context.settings.max_tail_lines,
    )
    path_argument = _relative_argv_path(context.workspace.cwd, path.path)
    argv = (context.executable_name, "-n", str(lines), "--", path_argument)
    display_argv = (
        context.executable_name,
        "-n",
        str(lines),
        "--",
        path.display_path,
    )
    return argv, display_argv, None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="tail",
    executable_name="tail",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
