from ..entities import ShellExecutionErrorCode
from ..kill import is_protected_pid
from ..ps import PS_MAX_PID
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _required_string_option,
    _validate_known_options,
    policy_denied,
)

KILL_SIGNALS = ("TERM", "INT", "KILL")
_KILL_OPTIONS = {"pid", "signal"}


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options=_KILL_OPTIONS,
        command="kill",
        include_option_name=True,
    )
    pid = _validated_pid(request.options.get("pid"))
    signal = _required_string_option(request.options, "signal")
    if signal not in KILL_SIGNALS:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "signal is not supported",
        )
    argv = (context.executable_name, "-s", signal, "--", str(pid))
    return argv, argv, None


def _validated_pid(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= PS_MAX_PID
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pid must be a valid process identifier",
        )
    if is_protected_pid(value):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pid is protected",
        )
    return value


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="kill",
    executable_name="kill",
    dependency_group=ShellDependencyGroup.PROCESS,
    container_package_hints=("procps-ng", "procps"),
    argv_builder=build_argv,
    process_risk=True,
)
