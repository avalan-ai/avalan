from ..entities import (
    ExecutionResult,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
)
from ..kill import redacted_stderr as _redacted_kill_stderr
from ..lsof import LSOF_MAX_LIMIT, LSOF_MAX_PID
from ..lsof import open_file_rows_stdout as _lsof_open_file_rows_stdout
from ..lsof import redacted_stderr as _redacted_lsof_stderr
from ..pgrep import pid_only_stdout as _pgrep_pid_only_stdout
from ..pgrep import redacted_stderr as _redacted_pgrep_stderr
from ..ps import PsView
from ..ps import process_rows_stdout as _ps_process_rows_stdout
from ..ps import redacted_stderr as _redacted_ps_stderr

from collections.abc import Mapping
from dataclasses import replace
from typing import cast


def _lsof_public_result(
    result: ExecutionResult,
    *,
    backend: str,
    tool_name: str,
    command: str,
    display_argv: tuple[str, ...],
    cwd: str,
    display_cwd: str,
    stdout_media_type: str,
    output_kind: ShellOutputKind,
    requested_pid: int | None,
    limit: int,
    max_stdout_bytes: int,
) -> ExecutionResult:
    no_matches = (
        result.status is ShellExecutionStatus.NONZERO_EXIT
        and result.exit_code == 1
        and not result.stdout
        and not result.stderr
        and not result.stdout_truncated
        and not result.stderr_truncated
    )
    status = ShellExecutionStatus.NO_MATCHES if no_matches else result.status
    error_code = (
        ShellExecutionErrorCode.NO_MATCHES if no_matches else result.error_code
    )
    safe_stdout = ""
    row_limit_truncated = False
    byte_limit_truncated = False
    malformed = False
    if requested_pid is not None:
        output = _lsof_open_file_rows_stdout(
            result.stdout,
            requested_pid=requested_pid,
            limit=limit,
            max_stdout_bytes=max_stdout_bytes,
            stdout_truncated=result.stdout_truncated,
        )
        safe_stdout = output.stdout
        row_limit_truncated = output.row_limit_truncated
        byte_limit_truncated = output.byte_limit_truncated
        malformed = output.malformed
    malformed_completed = (
        status is ShellExecutionStatus.COMPLETED and malformed
    )
    if malformed_completed:
        status = ShellExecutionStatus.TOOL_ERROR
        error_code = ShellExecutionErrorCode.TOOL_ERROR
        safe_stdout = ""
    safe_stderr = _redacted_lsof_stderr(result.stderr)
    return replace(
        result,
        backend=backend,
        tool_name=tool_name,
        command=command,
        argv=display_argv,
        display_argv=display_argv,
        cwd=cwd,
        display_cwd=display_cwd,
        status=status,
        stdout=safe_stdout,
        stderr=safe_stderr,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        generated_files=(),
        stdout_bytes=len(safe_stdout.encode("utf-8")),
        stderr_bytes=len(safe_stderr.encode("utf-8")),
        stdout_truncated=(
            result.stdout_truncated
            or row_limit_truncated
            or byte_limit_truncated
        ),
        error_code=error_code,
        error_message=(
            "lsof output was malformed"
            if malformed_completed
            else _lsof_public_error_message(status)
        ),
        metadata={},
    )


def _lsof_requested_pid(metadata: Mapping[str, object]) -> int:
    value = metadata.get("_lsof_requested_pid")
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), "lsof execution requires a requested PID"
    assert 1 <= value <= LSOF_MAX_PID, "lsof requested PID is invalid"
    return value


def _lsof_requested_limit(metadata: Mapping[str, object]) -> int:
    value = metadata.get("_lsof_limit")
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), "lsof execution requires a result limit"
    assert 1 <= value <= LSOF_MAX_LIMIT, "lsof result limit is invalid"
    return value


def _lsof_public_error_message(
    status: ShellExecutionStatus,
) -> str | None:
    if status in {
        ShellExecutionStatus.COMPLETED,
        ShellExecutionStatus.NO_MATCHES,
    }:
        return None
    messages = {
        ShellExecutionStatus.POLICY_DENIED: "lsof was denied by policy",
        ShellExecutionStatus.COMMAND_UNAVAILABLE: "lsof is unavailable",
        ShellExecutionStatus.SPAWN_FAILED: "lsof failed to start",
        ShellExecutionStatus.TIMEOUT: "lsof timed out",
        ShellExecutionStatus.CANCELLED: "lsof was cancelled",
        ShellExecutionStatus.NONZERO_EXIT: "lsof exited non-zero",
    }
    return messages.get(status, "lsof execution failed")


def _pgrep_public_result(
    result: ExecutionResult,
    *,
    backend: str,
    tool_name: str,
    command: str,
    display_argv: tuple[str, ...],
    cwd: str,
    display_cwd: str,
    stdout_media_type: str,
    output_kind: ShellOutputKind,
) -> ExecutionResult:
    safe_stdout = _pgrep_pid_only_stdout(
        result.stdout,
        stdout_truncated=result.stdout_truncated,
    )
    safe_stderr = _redacted_pgrep_stderr(result.stderr)
    return replace(
        result,
        backend=backend,
        tool_name=tool_name,
        command=command,
        argv=display_argv,
        display_argv=display_argv,
        cwd=cwd,
        display_cwd=display_cwd,
        stdout=safe_stdout,
        stderr=safe_stderr,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        generated_files=(),
        stdout_bytes=len(safe_stdout.encode("utf-8")),
        stderr_bytes=len(safe_stderr.encode("utf-8")),
        error_message=_pgrep_public_error_message(result),
        metadata={},
    )


def _pgrep_public_error_message(result: ExecutionResult) -> str | None:
    if result.status in {
        ShellExecutionStatus.COMPLETED,
        ShellExecutionStatus.NO_MATCHES,
    }:
        return None
    if result.error_message is None:
        return None
    messages = {
        ShellExecutionStatus.POLICY_DENIED: "pgrep was denied by policy",
        ShellExecutionStatus.COMMAND_UNAVAILABLE: "pgrep is unavailable",
        ShellExecutionStatus.SPAWN_FAILED: "pgrep failed to start",
        ShellExecutionStatus.TIMEOUT: "pgrep timed out",
        ShellExecutionStatus.CANCELLED: "pgrep was cancelled",
        ShellExecutionStatus.NONZERO_EXIT: "pgrep exited non-zero",
    }
    return messages.get(result.status, "pgrep execution failed")


def _kill_public_result(
    result: ExecutionResult,
    *,
    backend: str,
    tool_name: str,
    command: str,
    display_argv: tuple[str, ...],
    cwd: str,
    display_cwd: str,
    stdout_media_type: str,
    output_kind: ShellOutputKind,
) -> ExecutionResult:
    safe_stderr = _redacted_kill_stderr(result.stderr)
    return replace(
        result,
        backend=backend,
        tool_name=tool_name,
        command=command,
        argv=display_argv,
        display_argv=display_argv,
        cwd=cwd,
        display_cwd=display_cwd,
        stdout="",
        stderr=safe_stderr,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        generated_files=(),
        stdout_bytes=0,
        stderr_bytes=len(safe_stderr.encode("utf-8")),
        error_message=_kill_public_error_message(result),
        metadata={},
    )


def _kill_public_error_message(result: ExecutionResult) -> str | None:
    if result.status is ShellExecutionStatus.COMPLETED:
        return None
    messages = {
        ShellExecutionStatus.POLICY_DENIED: "kill was denied by policy",
        ShellExecutionStatus.COMMAND_UNAVAILABLE: "kill is unavailable",
        ShellExecutionStatus.SPAWN_FAILED: "kill failed to start",
        ShellExecutionStatus.TIMEOUT: "kill timed out",
        ShellExecutionStatus.CANCELLED: "kill was cancelled",
        ShellExecutionStatus.NONZERO_EXIT: "kill exited non-zero",
    }
    return messages.get(result.status, "kill execution failed")


def _ps_public_result(
    result: ExecutionResult,
    *,
    backend: str,
    tool_name: str,
    command: str,
    display_argv: tuple[str, ...],
    cwd: str,
    display_cwd: str,
    stdout_media_type: str,
    output_kind: ShellOutputKind,
    requested_pids: tuple[int, ...],
    view: PsView,
) -> ExecutionResult:
    safe_stdout = _ps_process_rows_stdout(
        result.stdout,
        requested_pids=requested_pids,
        view=view,
        stdout_truncated=result.stdout_truncated,
    )
    safe_stderr = _redacted_ps_stderr(result.stderr)
    return replace(
        result,
        backend=backend,
        tool_name=tool_name,
        command=command,
        argv=display_argv,
        display_argv=display_argv,
        cwd=cwd,
        display_cwd=display_cwd,
        stdout=safe_stdout,
        stderr=safe_stderr,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        generated_files=(),
        stdout_bytes=len(safe_stdout.encode("utf-8")),
        stderr_bytes=len(safe_stderr.encode("utf-8")),
        error_message=_ps_public_error_message(result),
        metadata={},
    )


def _ps_requested_pids(metadata: Mapping[str, object]) -> tuple[int, ...]:
    value = metadata.get("_ps_requested_pids")
    assert isinstance(value, tuple), "ps execution requires requested PIDs"
    assert len(value) == 1, "ps execution requires exactly one requested PID"
    assert all(
        isinstance(pid, int) and not isinstance(pid, bool) for pid in value
    ), "ps requested PIDs must be integers"
    return cast(tuple[int, ...], value)


def _ps_requested_view(metadata: Mapping[str, object]) -> PsView:
    value = metadata.get("_ps_view")
    assert value in {"summary", "resources"}, "ps view must be supported"
    return cast(PsView, value)


def _ps_public_error_message(result: ExecutionResult) -> str | None:
    if result.status in {
        ShellExecutionStatus.COMPLETED,
        ShellExecutionStatus.NO_MATCHES,
    }:
        return None
    messages = {
        ShellExecutionStatus.POLICY_DENIED: "ps was denied by policy",
        ShellExecutionStatus.COMMAND_UNAVAILABLE: "ps is unavailable",
        ShellExecutionStatus.SPAWN_FAILED: "ps failed to start",
        ShellExecutionStatus.TIMEOUT: "ps timed out",
        ShellExecutionStatus.CANCELLED: "ps was cancelled",
        ShellExecutionStatus.NONZERO_EXIT: "ps exited non-zero",
    }
    return messages.get(result.status, "ps execution failed")


def _policy_denied_result(
    request: ShellCommandRequest,
    error: ShellPolicyDenied,
) -> ExecutionResult:
    return ExecutionResult(
        backend="local",
        tool_name=request.tool_name,
        command=request.command,
        argv=(request.command,),
        display_argv=(request.command,),
        cwd=".",
        display_cwd=".",
        status=ShellExecutionStatus.POLICY_DENIED,
        exit_code=None,
        stdout="",
        stderr="",
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=0,
        error_code=error.error_code,
        error_message=str(error),
        metadata=request.metadata,
    )
