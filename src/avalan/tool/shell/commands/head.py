from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bounded_int_option,
    _optional_bounded_int_option,
    _relative_argv_path,
    _single_path,
    _validate_known_options,
)

_MAX_SHADOWED_LINE_COUNT = 2**31 - 1


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"byte_count", "lines"},
        command="head",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("file", "text_file"),
        command="head",
    )
    byte_count = _optional_bounded_int_option(
        request.options,
        "byte_count",
        min_value=1,
        max_value=context.settings.max_stdout_bytes,
    )
    if byte_count is None:
        flag = "-n"
        count = _bounded_int_option(
            request.options,
            "lines",
            default=80,
            min_value=1,
            max_value=context.settings.max_head_lines,
        )
    else:
        if "lines" in request.options:
            _bounded_int_option(
                request.options,
                "lines",
                default=80,
                min_value=1,
                max_value=_MAX_SHADOWED_LINE_COUNT,
            )
        flag = "-c"
        count = byte_count
    path_argument = _relative_argv_path(context.workspace.cwd, path.path)
    argv = (context.executable_name, flag, str(count), "--", path_argument)
    display_argv = (
        context.executable_name,
        flag,
        str(count),
        "--",
        path.display_path,
    )
    return argv, display_argv, None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="head",
    executable_name="head",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
