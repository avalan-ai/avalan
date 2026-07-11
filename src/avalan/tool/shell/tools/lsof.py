from ....entities import ToolCallContext
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..lsof import LSOF_DEFAULT_LIMIT, LSOF_MAX_LIMIT, LSOF_MAX_PID
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _optional_cwd,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from typing import Any


class LsofTool(_ShellCommandTool):
    """Inspect bounded open descriptor metadata for one process.

    Args:
        pid: Process identifier to inspect.
        limit: Maximum number of numeric file descriptor rows to return.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell result containing bounded descriptor metadata rows.
    """

    supports_streaming = False

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="lsof",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        schema = super().json_schema(prefix)
        parameters = schema["function"]["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        pid_schema = properties["pid"]
        assert isinstance(pid_schema, dict)
        pid_schema["minimum"] = 1
        pid_schema["maximum"] = LSOF_MAX_PID
        limit_schema = properties["limit"]
        assert isinstance(limit_schema, dict)
        limit_schema["minimum"] = 1
        limit_schema["maximum"] = LSOF_MAX_LIMIT
        return schema

    def _build_request(
        self,
        pid: int,
        limit: int = LSOF_DEFAULT_LIMIT,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.lsof",
            command="lsof",
            options={"pid": pid, "limit": limit},
            paths=(),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        pid: int,
        limit: int = LSOF_DEFAULT_LIMIT,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                pid=pid,
                limit=limit,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
