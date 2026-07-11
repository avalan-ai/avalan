from ....entities import ToolCallContext
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _optional_cwd,
    _path_operands,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence


class FileTool(_ShellCommandTool):
    """Identify workspace file types.

    Args:
        paths: Workspace-relative regular file paths to inspect.
        cwd: Workspace-relative working directory for the command.
        brief: Omit file names from command output.
        mime_type: Emit MIME type output.
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
            command="file",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        brief: bool = False,
        mime_type: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.file",
            command="file",
            options={"brief": brief, "mime_type": mime_type},
            paths=_path_operands(paths, kind="file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        brief: bool = False,
        mime_type: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                paths=paths,
                cwd=_optional_cwd(cwd),
                brief=brief,
                mime_type=mime_type,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
