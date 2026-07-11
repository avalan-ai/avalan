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

from typing import Literal


class PyPdfTool(_ShellCommandTool):
    """Inspect metadata or extract text from a workspace PDF with pypdf.

    Args:
        path: Workspace-relative PDF file path to read.
        mode: pypdf operation mode to run.
        first_page: First one-based page number to extract in text mode.
        last_page: Optional last one-based page number in text mode.
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
            command="pypdf",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        mode: Literal["metadata", "text"] = "metadata",
        first_page: int = 1,
        last_page: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        options: dict[str, object] = {"mode": mode}
        if mode == "text":
            options["first_page"] = first_page
            options["last_page"] = last_page
        elif first_page != 1 or last_page is not None:
            options["first_page"] = first_page
            options["last_page"] = last_page
        return ShellCommandRequest(
            tool_name="shell.pypdf",
            command="pypdf",
            options=options,
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        mode: Literal["metadata", "text"] = "metadata",
        first_page: int = 1,
        last_page: int | None = None,
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
                mode=mode,
                first_page=first_page,
                last_page=last_page,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
