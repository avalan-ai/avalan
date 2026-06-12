from .entities import (
    ExecutionSpec,
    GeneratedOutputPlan,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellOutputKind,
    ShellPolicyDenied,
    _create_execution_spec_from_policy,
)
from .registry import SHELL_COMMAND_DEFINITIONS
from .resolver import ExecutableResolver, TrustedExecutableResolver
from .settings import ShellToolSettings

from typing import Literal


class ExecutionPolicy:
    _resolver: ExecutableResolver
    _settings: ShellToolSettings

    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        resolver: ExecutableResolver | None = None,
    ) -> None:
        self._settings = settings or ShellToolSettings()
        self._resolver = resolver or TrustedExecutableResolver(
            executable_paths=self._settings.executable_paths,
            executable_search_paths=self._settings.executable_search_paths,
        )

    async def normalize(self, request: ShellCommandRequest) -> ExecutionSpec:
        if any(path.access != "read" for path in request.paths):
            raise ShellPolicyDenied(
                ShellExecutionErrorCode.WRITE_DENIED,
                "write access is disabled",
            )
        command_definition = SHELL_COMMAND_DEFINITIONS.get(request.command)
        if (
            command_definition is None
            or request.command not in self._settings.allowed_commands
        ):
            raise ShellPolicyDenied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "command is not allowed",
            )
        if (
            command_definition.media_risk
            and not self._settings.allow_media_tools
        ):
            raise ShellPolicyDenied(
                ShellExecutionErrorCode.DENIED_COMMAND,
                "media tools are disabled",
            )
        raise NotImplementedError("shell execution policy is not implemented")

    def create_execution_spec(
        self,
        *,
        backend: Literal["local"],
        tool_name: str,
        command: str,
        executable: str | None,
        argv: tuple[str, ...],
        display_argv: tuple[str, ...],
        cwd: str,
        display_cwd: str,
        env: dict[str, str],
        stdin: bytes | None,
        stdout_media_type: str,
        output_kind: ShellOutputKind,
        resource_class: Literal["standard", "heavy"],
        output_plan: GeneratedOutputPlan | None,
        timeout_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        metadata: dict[str, object] | None = None,
    ) -> ExecutionSpec:
        return _create_execution_spec_from_policy(
            backend=backend,
            tool_name=tool_name,
            command=command,
            executable=executable,
            argv=argv,
            display_argv=display_argv,
            cwd=cwd,
            display_cwd=display_cwd,
            env=env,
            stdin=stdin,
            stdout_media_type=stdout_media_type,
            output_kind=output_kind,
            resource_class=resource_class,
            output_plan=output_plan,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            metadata=metadata,
        )
