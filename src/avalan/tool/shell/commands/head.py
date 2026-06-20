from ..entities import ShellExecutionErrorCode
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
    policy_denied,
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
    byte_count = _optional_positive_int_option(
        request.options,
        "byte_count",
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


def _optional_positive_int_option(
    options: dict[str, object],
    name: str,
) -> int | None:
    if name not in options or options[name] is None:
        return None
    value = options[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be an integer",
        )
    if value < 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is out of range",
        )
    return value


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="head",
    executable_name="head",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
