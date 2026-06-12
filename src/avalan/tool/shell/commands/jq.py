from ..entities import (
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellOutputKind,
)
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _contains_unsafe_control,
    _relative_argv_path,
    _required_string_option,
    _validate_filter_paths,
    _validate_known_options,
    policy_denied,
)

_FORBIDDEN_IDENTIFIERS = frozenset(
    (
        "import",
        "include",
        "module",
        "env",
        "input",
        "inputs",
    )
)


def _validate_jq_filter(jq_filter: str) -> None:
    index = 0
    length = len(jq_filter)
    while index < length:
        character = jq_filter[index]
        if character == "\x00" or (
            _contains_unsafe_control(character)
            and character not in ("\n", "\t")
        ):
            raise policy_denied(
                ShellExecutionErrorCode.UNSAFE_FILTER,
                "jq filter contains unsafe control characters",
            )
        if character == '"':
            index = _skip_jq_string(jq_filter, index)
            continue
        if character == "#":
            index = _skip_jq_comment(jq_filter, index)
            continue
        if character == "$":
            variable, next_index = _read_jq_identifier(jq_filter, index + 1)
            if variable == "ENV":
                raise policy_denied(
                    ShellExecutionErrorCode.UNSUPPORTED_JQ_FEATURE,
                    "jq environment access is disabled",
                )
            index = next_index if next_index > index + 1 else index + 1
            continue
        if character.isalpha() or character == "_":
            identifier, next_index = _read_jq_identifier(jq_filter, index)
            if (
                identifier in _FORBIDDEN_IDENTIFIERS
                and _previous_non_space(jq_filter, index) != "."
            ):
                raise policy_denied(
                    ShellExecutionErrorCode.UNSUPPORTED_JQ_FEATURE,
                    "jq feature is disabled",
                )
            index = next_index
            continue
        index += 1


def _skip_jq_string(jq_filter: str, start_index: int) -> int:
    index = start_index + 1
    while index < len(jq_filter):
        character = jq_filter[index]
        if character == "\\":
            index += 2
            continue
        if character == '"':
            return index + 1
        index += 1
    raise policy_denied(
        ShellExecutionErrorCode.UNSAFE_FILTER,
        "jq string is unterminated",
    )


def _skip_jq_comment(jq_filter: str, start_index: int) -> int:
    index = start_index
    while index < len(jq_filter) and jq_filter[index] != "\n":
        index += 1
    return index


def _read_jq_identifier(value: str, start_index: int) -> tuple[str, int]:
    index = start_index
    while index < len(value) and (
        value[index].isalnum() or value[index] == "_"
    ):
        index += 1
    return value[start_index:index], index


def _previous_non_space(value: str, index: int) -> str | None:
    previous_index = index - 1
    while previous_index >= 0:
        character = value[previous_index]
        if not character.isspace():
            return character
        previous_index -= 1
    return None


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "filter",
            "raw_output",
            "compact",
            "slurp",
            "sort_keys",
        },
        command="jq",
    )
    _validate_filter_paths(
        context.paths, command="jq", allowed_kinds=("json_file",)
    )
    jq_filter = _required_string_option(request.options, "filter")
    if len(jq_filter.encode("utf-8")) > settings.max_jq_filter_bytes:
        raise policy_denied(
            ShellExecutionErrorCode.UNSAFE_FILTER,
            "jq filter is too large",
        )
    _validate_jq_filter(jq_filter)
    raw_output = _bool_option(request.options, "raw_output", default=False)
    compact = _bool_option(request.options, "compact", default=False)
    slurp = _bool_option(request.options, "slurp", default=False)
    sort_keys = _bool_option(request.options, "sort_keys", default=False)
    argv_parts = [context.executable_name]
    if raw_output:
        argv_parts.append("--raw-output")
    if compact:
        argv_parts.append("--compact-output")
    if slurp:
        argv_parts.append("--slurp")
    if sort_keys:
        argv_parts.append("--sort-keys")
    path_arguments = tuple(
        _relative_argv_path(context.workspace.cwd, path.path)
        for path in context.paths
    )
    display_path_arguments = tuple(path.display_path for path in context.paths)
    argv_parts.extend(("--", jq_filter, *path_arguments))
    display_parts = list(argv_parts[: -len(path_arguments)])
    display_parts.extend(display_path_arguments)
    return tuple(argv_parts), tuple(display_parts), None


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    if request.options.get("raw_output") is True:
        return "text/plain", ShellOutputKind.TEXT
    return "application/json", ShellOutputKind.JSON


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="jq",
    executable_name="jq",
    dependency_group=ShellDependencyGroup.JSON,
    container_package_hints=("jq",),
    argv_builder=build_argv,
    output_contract=output_contract,
)
