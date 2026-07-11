from ....entities import ToolCallContext
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _line_reader_request,
    _optional_cwd,
)
from ._base import ShellResultFormatter, _ShellCommandTool


class TailTool(_ShellCommandTool):
    """Read the last lines of a workspace text file.

    Args:
        path: Workspace-relative file path to read.
        lines: Number of trailing lines to return.
        start_line: One-based line number to start at via tail -n +N.
        byte_count: Native byte count to read via tail -c.
        start_byte: One-based byte number to start at via tail -c +N.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="tail",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        lines: int = 80,
        start_line: int | None = None,
        byte_count: int | None = None,
        start_byte: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return _line_reader_request(
            command="tail",
            path=path,
            lines=lines,
            byte_count=byte_count,
            start_line=start_line,
            start_byte=start_byte,
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        lines: int = 80,
        start_line: int | None = None,
        byte_count: int | None = None,
        start_byte: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                lines=lines,
                start_line=start_line,
                byte_count=byte_count,
                start_byte=start_byte,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
