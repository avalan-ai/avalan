from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _option_safe_display_path_argument,
    _option_safe_path_argument,
    _validate_filter_paths,
    _validate_known_options,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"brief", "mime_type"},
        command="file",
    )
    _validate_filter_paths(
        context.paths,
        command="file",
        allowed_kinds=("file",),
    )
    brief = _bool_option(request.options, "brief", default=False)
    mime_type = _bool_option(request.options, "mime_type", default=False)
    argv_parts = [context.executable_name]
    if brief:
        argv_parts.append("--brief")
    if mime_type:
        argv_parts.append("--mime-type")
    argv_parts.append("--")
    argv_parts.extend(
        _option_safe_path_argument(context.workspace.cwd, path.path)
        for path in context.paths
    )
    display_parts = list(argv_parts[:])
    display_parts[-len(context.paths) :] = [
        _option_safe_display_path_argument(path.display_path)
        for path in context.paths
    ]
    return tuple(argv_parts), tuple(display_parts), None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="file",
    executable_name="file",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("file",),
    argv_builder=build_argv,
)
