from ....entities import ToolCall, ToolCallContext, ToolCallOutcome
from ... import Tool
from ..display import project_shell_tool_display
from ..entities import (
    ExecutionResult,
    ShellExecutionStatus,
)
from ..executor import CommandExecutor
from ..git import (
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitCommandResult,
    ShellGitExecutionErrorCode,
    ShellGitExecutionMode,
    ShellGitExecutionStatus,
    ShellGitFormattedResult,
    ShellGitPolicyDenied,
    shell_git_capability_for_request,
)
from ..git_policy import (
    GitExecutionPolicy,
    git_remote_audit_metadata,
    redact_git_text,
)
from ..settings import ShellGitToolSettings, ShellToolSettings
from ._arguments import _optional_cwd, _string_tuple

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from inspect import signature
from typing import Any, Self, cast


class _ShellGitCommandTool(Tool, ABC):
    _command: ShellGitCommandName
    _executor: CommandExecutor | None
    _git_policy: GitExecutionPolicy
    _settings: ShellToolSettings
    supports_streaming = False

    def __init__(
        self,
        *,
        command: ShellGitCommandName,
        settings: ShellToolSettings,
    ) -> None:
        super().__init__()
        self.__name__ = f"git_{command.value.replace('-', '_')}"
        self._command = command
        self._settings = settings
        self._git_policy = GitExecutionPolicy(settings=settings)
        self._executor = None

    def bind_execution(
        self,
        *,
        git_policy: GitExecutionPolicy,
        executor: CommandExecutor,
    ) -> Self:
        assert isinstance(
            git_policy,
            GitExecutionPolicy,
        ), "git_policy must be a shell Git execution policy"
        self._git_policy = git_policy
        self._executor = executor
        return self

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        request = None if outcome is not None else self._display_request(call)
        return project_shell_tool_display(
            call=call,
            outcome=outcome,
            request=request,
        )

    def _display_request(
        self,
        call: ToolCall,
    ) -> ShellGitCommandRequest | None:
        if call.arguments is None:
            arguments = {}
        elif isinstance(call.arguments, dict):
            arguments = dict(call.arguments)
        else:
            return None
        builder = getattr(self, "_build_request")
        assert callable(builder), "shell Git tool must define _build_request"
        try:
            bound = signature(builder).bind(**arguments)
            bound.apply_defaults()
            request = builder(**bound.arguments)
        except (AssertionError, TypeError, ValueError):
            return None
        assert isinstance(
            request,
            ShellGitCommandRequest,
        ), "_build_request must return a shell Git command request"
        return request

    async def _execute_request(
        self,
        request: ShellGitCommandRequest,
        *,
        context: ToolCallContext,
    ) -> str:
        try:
            spec = await self._git_policy.normalize(request)
        except ShellGitPolicyDenied as error:
            result = _git_policy_denied_result(
                request,
                error,
                settings=self._settings,
            )
            return ShellGitFormattedResult(
                _format_shell_git_result(result), result
            )
        executor = self._executor
        assert executor is not None, "shell Git tools require an executor"
        try:
            execution_result = await executor.execute(spec)
        except BaseException as error:
            if error.__class__.__name__ != "CancelledError":
                raise
            result = _git_cancelled_result(
                request,
                spec,
                settings=_git_settings(self._settings),
            )
            return ShellGitFormattedResult(
                _format_shell_git_result(result), result
            )
        result = _git_execution_result(
            request,
            spec,
            execution_result,
            settings=_git_settings(self._settings),
        )
        return ShellGitFormattedResult(
            _format_shell_git_result(result), result
        )

    def _request(
        self,
        *,
        options: dict[str, object],
        pathspecs: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        request = ShellGitCommandRequest(
            tool_name=f"shell.{self.__name__}",
            command=self._command,
            capability_required=ShellGitCapability.READ,
            options=options,
            pathspecs=_string_tuple(pathspecs, "pathspecs"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )
        return ShellGitCommandRequest(
            tool_name=request.tool_name,
            command=request.command,
            capability_required=shell_git_capability_for_request(request),
            options=request.options,
            pathspecs=request.pathspecs,
            cwd=request.cwd,
            timeout_seconds=request.timeout_seconds,
            max_stdout_bytes=request.max_stdout_bytes,
            max_stderr_bytes=request.max_stderr_bytes,
            metadata=request.metadata,
        )

    @abstractmethod
    async def __call__(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError


def _git_settings(settings: ShellToolSettings) -> ShellGitToolSettings:
    git_settings = settings.git
    assert isinstance(
        git_settings,
        ShellGitToolSettings,
    ), "git must be shell Git tool settings"
    return git_settings


def _git_policy_denied_result(
    request: ShellGitCommandRequest,
    error: ShellGitPolicyDenied,
    *,
    settings: ShellToolSettings,
) -> ShellGitCommandResult:
    git_settings = _git_settings(settings)
    display_argv = _redacted_git_argv(
        ("git", request.command.value),
        settings=git_settings,
    )
    capability = shell_git_capability_for_request(request)
    request_options = _redacted_git_metadata(
        request.options,
        settings=git_settings,
    )
    request_pathspecs = _redacted_git_metadata(
        request.pathspecs,
        settings=git_settings,
    )
    audit_metadata: dict[str, object] = {
        "git_command": request.command.value,
        "git_capability_required": request.capability_required.value,
        "git_capability_used": capability.value,
        "git_request_options": request_options,
        "git_request_pathspecs": request_pathspecs,
        "request_options": request_options,
        "request_pathspecs": request_pathspecs,
    }
    if capability is ShellGitCapability.WORKTREE:
        audit_metadata.update(
            {
                "git_mutation_attempted": True,
                "git_mutation_scope": "worktree",
            }
        )
    elif capability is ShellGitCapability.HISTORY:
        audit_metadata.update(
            {
                "git_mutation_attempted": True,
                "git_mutation_scope": "history",
            }
        )
    elif capability is ShellGitCapability.REMOTE:
        audit_metadata.update(
            git_remote_audit_metadata(request, settings=git_settings)
        )
    return ShellGitCommandResult(
        tool_name=request.tool_name,
        command=request.command,
        display_argv=display_argv,
        effective_cwd=redact_git_text(
            request.cwd or git_settings.cwd,
            git_settings,
        ),
        resolved_repo_root=None,
        capability_required=request.capability_required,
        capability_used=None,
        execution_mode="policy",
        status=ShellGitExecutionStatus.POLICY_DENIED,
        exit_code=None,
        stdout_snippet="",
        stderr_snippet="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=0,
        error_code=error.error_code,
        error_message=redact_git_text(str(error), git_settings),
        audit_metadata=audit_metadata,
    )


def _git_cancelled_result(
    request: ShellGitCommandRequest,
    spec: Any,
    *,
    settings: ShellGitToolSettings,
) -> ShellGitCommandResult:
    return ShellGitCommandResult(
        tool_name=request.tool_name,
        command=request.command,
        display_argv=spec.display_argv,
        effective_cwd=_metadata_text(
            spec.metadata,
            "git_effective_cwd",
            spec.display_cwd,
        ),
        resolved_repo_root=_metadata_optional_text(
            spec.metadata,
            "git_repo_root",
        ),
        capability_required=request.capability_required,
        capability_used=_git_capability_used(spec),
        execution_mode=_git_execution_mode(spec.backend),
        status=ShellGitExecutionStatus.CANCELLED,
        exit_code=None,
        stdout_snippet="",
        stderr_snippet="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=True,
        duration_ms=0,
        error_code=None,
        error_message="shell Git execution was cancelled",
        audit_metadata=cast(
            dict[str, object],
            _redacted_git_metadata(
                spec.metadata,
                settings=settings,
            ),
        ),
    )


def _git_execution_result(
    request: ShellGitCommandRequest,
    spec: Any,
    execution_result: ExecutionResult,
    *,
    settings: ShellGitToolSettings,
) -> ShellGitCommandResult:
    status, error_code = _git_status_and_error(request, execution_result)
    stdout = redact_git_text(execution_result.stdout, settings)
    stderr = redact_git_text(execution_result.stderr, settings)
    error_message = _git_error_message(
        execution_result,
        status=status,
        error_code=error_code,
        settings=settings,
    )
    return ShellGitCommandResult(
        tool_name=request.tool_name,
        command=request.command,
        display_argv=spec.display_argv,
        effective_cwd=_metadata_text(
            spec.metadata,
            "git_effective_cwd",
            execution_result.display_cwd,
        ),
        resolved_repo_root=_metadata_optional_text(
            spec.metadata,
            "git_repo_root",
        ),
        capability_required=request.capability_required,
        capability_used=_git_capability_used(spec),
        execution_mode=_git_execution_mode(spec.backend),
        status=status,
        exit_code=execution_result.exit_code,
        stdout_snippet=stdout,
        stderr_snippet=stderr,
        stdout_bytes=execution_result.stdout_bytes,
        stderr_bytes=execution_result.stderr_bytes,
        stdout_truncated=execution_result.stdout_truncated,
        stderr_truncated=execution_result.stderr_truncated,
        timed_out=execution_result.timed_out,
        cancelled=execution_result.cancelled,
        duration_ms=execution_result.duration_ms,
        error_code=error_code,
        error_message=error_message,
        audit_metadata=cast(
            dict[str, object],
            _redacted_git_metadata(
                {
                    **spec.metadata,
                    "shell_status": execution_result.status.value,
                    "shell_error_code": (
                        None
                        if execution_result.error_code is None
                        else execution_result.error_code.value
                    ),
                    "shell_metadata": execution_result.metadata,
                },
                settings=settings,
            ),
        ),
    )


def _git_status_and_error(
    request: ShellGitCommandRequest,
    result: ExecutionResult,
) -> tuple[ShellGitExecutionStatus, ShellGitExecutionErrorCode | None]:
    if result.status is ShellExecutionStatus.COMPLETED:
        if result.stdout_truncated or result.stderr_truncated:
            return (
                ShellGitExecutionStatus.FAILED,
                ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
            )
        return ShellGitExecutionStatus.SUCCESS, None
    if result.status is ShellExecutionStatus.COMMAND_UNAVAILABLE:
        return (
            ShellGitExecutionStatus.COMMAND_UNAVAILABLE,
            ShellGitExecutionErrorCode.COMMAND_UNAVAILABLE,
        )
    if result.status is ShellExecutionStatus.TIMEOUT:
        return (
            ShellGitExecutionStatus.TIMEOUT,
            ShellGitExecutionErrorCode.TIMEOUT,
        )
    if result.status is ShellExecutionStatus.CANCELLED:
        return ShellGitExecutionStatus.CANCELLED, None
    if result.status is ShellExecutionStatus.NO_MATCHES:
        if (
            request.command is ShellGitCommandName.GREP
            and result.exit_code == 1
        ):
            return ShellGitExecutionStatus.SUCCESS, None
        return (
            ShellGitExecutionStatus.FAILED,
            ShellGitExecutionErrorCode.NONZERO_EXIT,
        )
    if result.status is ShellExecutionStatus.TOO_LARGE:
        return (
            ShellGitExecutionStatus.FAILED,
            ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
        )
    if result.status is ShellExecutionStatus.POLICY_DENIED:
        return (
            ShellGitExecutionStatus.POLICY_DENIED,
            ShellGitExecutionErrorCode.COMMAND_DISABLED,
        )
    if result.status is ShellExecutionStatus.SPAWN_FAILED:
        return (
            ShellGitExecutionStatus.COMMAND_UNAVAILABLE,
            ShellGitExecutionErrorCode.COMMAND_UNAVAILABLE,
        )
    if result.status is ShellExecutionStatus.NONZERO_EXIT:
        return ShellGitExecutionStatus.FAILED, _git_nonzero_error_code(result)
    return (
        ShellGitExecutionStatus.FAILED,
        ShellGitExecutionErrorCode.NONZERO_EXIT,
    )


def _git_nonzero_error_code(
    result: ExecutionResult,
) -> ShellGitExecutionErrorCode:
    diagnostic = " ".join(
        value.lower()
        for value in (
            result.error_message or "",
            result.stderr,
            result.stdout,
        )
        if value
    )
    if (
        "ambiguous object name" in diagnostic
        or "ambiguous revision" in diagnostic
        or ("refname" in diagnostic and "ambiguous" in diagnostic)
    ):
        return ShellGitExecutionErrorCode.AMBIGUOUS_REVISION
    if any(
        marker in diagnostic
        for marker in (
            "unknown revision",
            "bad revision",
            "invalid revision",
            "invalid object name",
            "not a valid object name",
            "needed a single revision",
            "does not point to a commit",
            "no names found",
        )
    ):
        return ShellGitExecutionErrorCode.REVISION_NOT_FOUND
    return ShellGitExecutionErrorCode.NONZERO_EXIT


def _git_error_message(
    result: ExecutionResult,
    *,
    status: ShellGitExecutionStatus,
    error_code: ShellGitExecutionErrorCode | None,
    settings: ShellGitToolSettings,
) -> str | None:
    if status is ShellGitExecutionStatus.CANCELLED:
        return "shell Git execution was cancelled"
    if error_code is None:
        return None
    if error_code is ShellGitExecutionErrorCode.REVISION_NOT_FOUND:
        return "Git revision was not found"
    if error_code is ShellGitExecutionErrorCode.AMBIGUOUS_REVISION:
        return "Git revision is ambiguous"
    if result.error_message is not None:
        return redact_git_text(result.error_message, settings)
    if error_code is ShellGitExecutionErrorCode.OUTPUT_TRUNCATED:
        return "shell Git output was truncated"
    if status is ShellGitExecutionStatus.COMMAND_UNAVAILABLE:
        return "git executable is unavailable"
    if status is ShellGitExecutionStatus.TIMEOUT:
        return "shell Git execution timed out"
    if status is ShellGitExecutionStatus.POLICY_DENIED:
        return "shell Git execution was denied"
    return "shell Git command failed"


def _git_capability_used(spec: Any) -> ShellGitCapability | None:
    value = spec.metadata.get("git_capability_used")
    assert isinstance(
        value,
        str,
    ), "Git execution spec must include a capability"
    return ShellGitCapability(value)


def _git_execution_mode(value: str) -> ShellGitExecutionMode:
    assert value in (
        "local",
        "sandbox",
        "container",
    ), "shell Git execution backend must be local, sandbox, or container"
    return cast(ShellGitExecutionMode, value)


def _metadata_text(
    metadata: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else default


def _metadata_optional_text(
    metadata: Mapping[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _redacted_git_argv(
    argv: Sequence[str],
    *,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    return tuple(redact_git_text(argument, settings) for argument in argv)


def _redacted_git_metadata(
    value: object,
    *,
    settings: ShellGitToolSettings,
) -> object:
    if isinstance(value, str):
        return redact_git_text(value, settings)
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = (
                "[redacted]"
                if key_text == "message" and isinstance(item, str)
                else _redacted_git_metadata(item, settings=settings)
            )
        return redacted
    if isinstance(value, tuple):
        return tuple(
            _redacted_git_metadata(item, settings=settings) for item in value
        )
    if isinstance(value, list):
        return [
            _redacted_git_metadata(item, settings=settings) for item in value
        ]
    return value


def _format_shell_git_result(result: ShellGitCommandResult) -> str:
    error_code = result.error_code.value if result.error_code else None
    capability_used = (
        result.capability_used.value if result.capability_used else None
    )
    lines = [
        f"tool: {result.tool_name}",
        f"status: {result.status.value}",
        f"git_command: {result.command.value}",
        f"command: {' '.join(result.display_argv)}",
        f"cwd: {result.effective_cwd}",
        f"repo_root: {_scalar_text(result.resolved_repo_root)}",
        f"capability_required: {result.capability_required.value}",
        f"capability_used: {_scalar_text(capability_used)}",
        f"execution_mode: {result.execution_mode}",
        f"exit_code: {_scalar_text(result.exit_code)}",
        f"error_code: {_scalar_text(error_code)}",
        f"error_message: {_scalar_text(result.error_message)}",
        f"timed_out: {_bool_text(result.timed_out)}",
        f"cancelled: {_bool_text(result.cancelled)}",
        f"duration_ms: {result.duration_ms}",
        f"stdout_bytes: {result.stdout_bytes}",
        f"stderr_bytes: {result.stderr_bytes}",
        f"stdout_truncated: {_bool_text(result.stdout_truncated)}",
        f"stderr_truncated: {_bool_text(result.stderr_truncated)}",
        "",
        "stdout:",
        result.stdout_snippet,
        "",
        "stderr:",
        result.stderr_snippet,
    ]
    return "\n".join(lines)


def _scalar_text(value: object) -> str:
    return "null" if value is None else str(value)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
