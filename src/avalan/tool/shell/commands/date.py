from ..date import (
    DATE_CUSTOM_FORMAT_DIRECTIVES,
    DATE_CUSTOM_FORMAT_MAX_BYTES,
    DATE_FORMATS,
)
from ..entities import ShellExecutionErrorCode
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _literal_option,
    _validate_known_options,
    policy_denied,
)

_DATE_CUSTOM_FORMAT_DIRECTIVE_SET = frozenset(DATE_CUSTOM_FORMAT_DIRECTIVES)
_DATE_FORMAT_ARGUMENTS = {
    "date": "+%Y-%m-%d",
    "time": "+%H:%M:%S",
    "iso8601": "+%Y-%m-%dT%H:%M:%S%z",
    "unix": "+%s",
}


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"custom_format", "format", "utc"},
        command="date",
        include_option_name=True,
    )
    if context.paths:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "date does not accept paths",
        )
    utc = _bool_option(request.options, "utc", default=False)
    output_format = _literal_option(
        request.options,
        "format",
        default="default",
        allowed=DATE_FORMATS,
    )
    custom_format = _custom_format_option(request.options)
    if custom_format is not None and output_format != "default":
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "custom_format requires format=default",
        )
    argv_parts = [context.executable_name]
    if utc:
        argv_parts.append("-u")
    format_argument = (
        f"+{custom_format}"
        if custom_format is not None
        else _DATE_FORMAT_ARGUMENTS.get(output_format)
    )
    if format_argument is not None:
        argv_parts.append(format_argument)
    argv = tuple(argv_parts)
    return argv, argv, None


def _custom_format_option(options: dict[str, object]) -> str | None:
    value = options.get("custom_format")
    if value is None:
        return None
    if not isinstance(value, str):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "custom_format must be a string",
        )
    if not value:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "custom_format must not be empty",
        )
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "custom_format must contain only printable ASCII",
        ) from error
    if any(byte < 0x20 or byte > 0x7E for byte in encoded):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "custom_format must contain only printable ASCII",
        )
    if len(encoded) > DATE_CUSTOM_FORMAT_MAX_BYTES:
        raise policy_denied(
            ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            "custom_format is too large",
        )
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 1 >= len(value):
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "custom_format contains a dangling %",
            )
        if value[index + 1] not in _DATE_CUSTOM_FORMAT_DIRECTIVE_SET:
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "custom_format contains an unsupported directive",
            )
        index += 2
    return value


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="date",
    executable_name="date",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
