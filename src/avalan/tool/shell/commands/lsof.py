from ..entities import ShellExecutionErrorCode
from ..lsof import LSOF_DEFAULT_LIMIT, LSOF_MAX_LIMIT, LSOF_MAX_PID
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bounded_int_option,
    _validate_known_options,
    policy_denied,
)

_LSOF_OPTIONS = {"limit", "pid"}


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options=_LSOF_OPTIONS,
        command="lsof",
        include_option_name=True,
    )
    pid = _validated_pid(request.options.get("pid"))
    limit = _bounded_int_option(
        request.options,
        "limit",
        default=LSOF_DEFAULT_LIMIT,
        min_value=1,
        max_value=LSOF_MAX_LIMIT,
    )
    context.metadata["_lsof_requested_pid"] = pid
    context.metadata["_lsof_limit"] = limit
    argv = (
        context.executable_name,
        "-n",
        "-P",
        "-w",
        "-b",
        "-a",
        "-p",
        str(pid),
        "-F0pftaP",
    )
    return argv, argv, None


def _validated_pid(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= LSOF_MAX_PID
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pid must be a valid process identifier",
        )
    return value


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="lsof",
    executable_name="lsof",
    dependency_group=ShellDependencyGroup.PROCESS,
    container_package_hints=("lsof",),
    argv_builder=build_argv,
    process_risk=True,
)
