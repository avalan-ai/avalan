from ..entities import ShellExecutionErrorCode
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _relative_argv_path,
    _validate_known_options,
    _validate_path_kind,
    policy_denied,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"lines", "words", "count_bytes"},
        command="wc",
    )
    if not context.paths:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "wc requires at least one path",
        )
    for path in context.paths:
        _validate_path_kind(
            path,
            allowed_kinds=("file", "text_file"),
            command="wc",
        )
    lines = _bool_option(request.options, "lines", default=True)
    words = _bool_option(request.options, "words", default=False)
    count_bytes = _bool_option(
        request.options,
        "count_bytes",
        default=False,
    )
    if not lines and not words and not count_bytes:
        lines = True
    flags: list[str] = []
    if lines:
        flags.append("-l")
    if words:
        flags.append("-w")
    if count_bytes:
        flags.append("-c")
    path_arguments = tuple(
        _relative_argv_path(context.workspace.cwd, path.path)
        for path in context.paths
    )
    display_path_arguments = tuple(path.display_path for path in context.paths)
    argv = (context.executable_name, *flags, "--", *path_arguments)
    display_argv = (
        context.executable_name,
        *flags,
        "--",
        *display_path_arguments,
    )
    return argv, display_argv, None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="wc",
    executable_name="wc",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
