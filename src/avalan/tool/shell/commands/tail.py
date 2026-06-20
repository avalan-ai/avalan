from ..entities import ShellExecutionErrorCode
from .base import (
    NormalizedPath,
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
_TAIL_START_OPTIONS = ("start_line", "byte_count", "start_byte")


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"byte_count", "lines", "start_byte", "start_line"},
        command="tail",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("file", "text_file"),
        command="tail",
    )
    active_start_options = tuple(
        option
        for option in _TAIL_START_OPTIONS
        if request.options.get(option) is not None
    )
    if len(active_start_options) > 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "tail start options conflict",
        )
    if active_start_options:
        if "lines" in request.options:
            _bounded_int_option(
                request.options,
                "lines",
                default=80,
                min_value=1,
                max_value=_MAX_SHADOWED_LINE_COUNT,
            )
        flag, count = _tail_start_option(
            context,
            path,
            request.options,
            active_start_options[0],
        )
    else:
        lines = _bounded_int_option(
            request.options,
            "lines",
            default=80,
            min_value=1,
            max_value=context.settings.max_tail_lines,
        )
        flag = "-n"
        count = str(lines)
    path_argument = _relative_argv_path(context.workspace.cwd, path.path)
    argv = (context.executable_name, flag, count, "--", path_argument)
    display_argv = (
        context.executable_name,
        flag,
        count,
        "--",
        path.display_path,
    )
    return argv, display_argv, None


def _tail_start_option(
    context: ShellCommandPolicyContext,
    path: NormalizedPath,
    options: dict[str, object],
    option: str,
) -> tuple[str, str]:
    value = _positive_int_option(
        options,
        option,
        max_value=(
            context.settings.max_stdout_bytes
            if option == "byte_count"
            else _MAX_SHADOWED_LINE_COUNT
        ),
    )
    if option == "start_line":
        _validate_from_start_file_size(_required_file_size(path), context)
        return "-n", f"+{value}"
    if option == "start_byte":
        _validate_start_byte_suffix(_required_file_size(path), value, context)
        return "-c", f"+{value}"
    return "-c", str(value)


def _positive_int_option(
    options: dict[str, object],
    name: str,
    *,
    max_value: int,
) -> int:
    assert name in options, "active tail option must be present"
    value = options[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be an integer",
        )
    if value < 1 or value > max_value:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is out of range",
        )
    return value


def _required_file_size(path: NormalizedPath) -> int:
    if path.metadata is None:
        raise policy_denied(
            ShellExecutionErrorCode.DENIED_PATH,
            "input file is unavailable",
        )
    return path.metadata.size


def _validate_from_start_file_size(
    file_size: int,
    context: ShellCommandPolicyContext,
) -> None:
    if file_size > context.settings.max_stdout_bytes:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "tail from-start file is too large",
        )


def _validate_start_byte_suffix(
    file_size: int,
    start_byte: int,
    context: ShellCommandPolicyContext,
) -> None:
    suffix_bytes = max(0, file_size - start_byte + 1)
    if suffix_bytes > context.settings.max_stdout_bytes:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "tail byte suffix is too large",
        )


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="tail",
    executable_name="tail",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
