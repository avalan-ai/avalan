from ..entities import ShellCommandRequest, ShellOutputKind
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _relative_argv_path,
    _single_path,
    _validate_known_options,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    _validate_known_options(
        context.request.options,
        allowed_options=set(),
        command="cat",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("file", "json_file", "text_file"),
        command="cat",
    )
    path_argument = _relative_argv_path(context.workspace.cwd, path.path)
    return (
        (context.executable_name, "--", path_argument),
        (context.executable_name, "--", path.display_path),
        None,
    )


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    if any(path.kind == "json_file" for path in request.paths):
        return "application/json", ShellOutputKind.JSON
    return "text/plain", ShellOutputKind.TEXT


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="cat",
    executable_name="cat",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
    output_contract=output_contract,
)
