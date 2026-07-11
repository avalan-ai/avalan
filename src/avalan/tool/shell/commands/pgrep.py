from ..entities import ShellExecutionErrorCode
from ..pgrep import PGREP_MAX_PID, REDACTED_PGREP_PATTERN
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _optional_bounded_int_option,
    _required_string_option,
    _validate_known_options,
    policy_denied,
)

from unicodedata import category as unicode_category

_PGREP_OPTIONS = {
    "exact",
    "full",
    "ignore_case",
    "newest",
    "oldest",
    "parent_pid",
    "pattern",
}


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options=_PGREP_OPTIONS,
        command="pgrep",
        include_option_name=True,
    )
    pattern = _validated_pattern(
        request.options,
        max_bytes=context.settings.max_filter_pattern_bytes,
    )
    full = _bool_option(request.options, "full", default=False)
    exact = _bool_option(request.options, "exact", default=False)
    ignore_case = _bool_option(
        request.options,
        "ignore_case",
        default=False,
    )
    newest = _bool_option(request.options, "newest", default=False)
    oldest = _bool_option(request.options, "oldest", default=False)
    if newest and oldest:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "newest and oldest are mutually exclusive",
        )
    parent_pid = _optional_bounded_int_option(
        request.options,
        "parent_pid",
        min_value=1,
        max_value=PGREP_MAX_PID,
    )

    argv_parts = [context.executable_name]
    for enabled, option in (
        (full, "-f"),
        (exact, "-x"),
        (ignore_case, "-i"),
        (newest, "-n"),
        (oldest, "-o"),
    ):
        if enabled:
            argv_parts.append(option)
    if parent_pid is not None:
        argv_parts.extend(("-P", str(parent_pid)))
    argv_parts.extend(("--", pattern))
    context.metadata["exit_code_statuses"] = {1: "no_matches"}
    argv = tuple(argv_parts)
    return argv, (*argv[:-1], REDACTED_PGREP_PATTERN), None


def _validated_pattern(
    options: dict[str, object],
    *,
    max_bytes: int,
) -> str:
    pattern = _required_string_option(options, "pattern")
    if any(
        unicode_category(character).startswith("C") for character in pattern
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pattern contains control characters",
        )
    encoded = pattern.encode("utf-8")
    if len(encoded) > max_bytes:
        raise policy_denied(
            ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            "pattern is too large",
        )
    return pattern


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="pgrep",
    executable_name="pgrep",
    dependency_group=ShellDependencyGroup.PROCESS,
    container_package_hints=("procps-ng", "procps"),
    argv_builder=build_argv,
    process_risk=True,
)
