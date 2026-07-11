from ..entities import ShellExecutionErrorCode
from ..ps import PS_MAX_PID
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import _validate_known_options, policy_denied

from collections.abc import Sequence

_PS_OPTIONS = {"pids"}


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options=_PS_OPTIONS,
        command="ps",
        include_option_name=True,
    )
    pids = _validated_pids(request.options.get("pids"))
    context.metadata["_ps_requested_pids"] = pids
    argv = (
        context.executable_name,
        "-p",
        ",".join(str(pid) for pid in pids),
        "-o",
        "pid=",
        "-o",
        "ppid=",
        "-o",
        "state=",
        "-o",
        "etime=",
        "-o",
        "comm=",
    )
    context.metadata["exit_code_statuses"] = {1: "no_matches"}
    return argv, argv, None


def _validated_pids(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or not value
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pids must be a non-empty sequence",
        )
    if len(value) != 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pids must contain exactly one value",
        )
    pids: list[int] = []
    for pid in value:
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or not 1 <= pid <= PS_MAX_PID
        ):
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "pids must contain valid process identifiers",
            )
        pids.append(pid)
    return tuple(pids)


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="ps",
    executable_name="ps",
    dependency_group=ShellDependencyGroup.PROCESS,
    container_package_hints=("procps-ng", "procps"),
    argv_builder=build_argv,
    process_risk=True,
)
