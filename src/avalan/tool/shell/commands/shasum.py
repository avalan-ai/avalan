from ..entities import _ShellRuntimeDependency
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _literal_option,
    _option_safe_path_argument,
    _validate_filter_paths,
    _validate_known_options,
)

SHASUM_ALGORITHMS = (
    "1",
    "224",
    "256",
    "384",
    "512",
    "512224",
    "512256",
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"algorithm"},
        command="shasum",
        include_option_name=True,
    )
    _validate_filter_paths(
        context.paths,
        command="shasum",
        allowed_kinds=("file",),
    )
    algorithm = _literal_option(
        request.options,
        "algorithm",
        default="1",
        allowed=SHASUM_ALGORITHMS,
    )
    argv_parts = [context.executable_name, "-a", algorithm, "--"]
    argv_parts.extend(
        _option_safe_path_argument(context.workspace.cwd, path.path)
        for path in context.paths
    )
    argv = tuple(argv_parts)
    return argv, argv, None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="shasum",
    executable_name="shasum",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("perl-utils",),
    argv_builder=build_argv,
    runtime_dependencies=(_ShellRuntimeDependency.SYSTEM_PERL,),
)
