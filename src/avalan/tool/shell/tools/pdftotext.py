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


class PdfToTextTool(_ShellCommandTool):
    """Extract text from a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to read.
        first_page: First one-based page number to extract.
        last_page: Optional last one-based page number to extract.
        layout: Preserve physical page layout where supported.
        no_page_breaks: Suppress page break markers in text output.
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
            command="pdftotext",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        no_page_breaks: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pdftotext",
            command="pdftotext",
            options={
                "first_page": first_page,
                "last_page": last_page,
                "layout": layout,
                "no_page_breaks": no_page_breaks,
            },
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        no_page_breaks: bool = False,
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
                first_page=first_page,
                last_page=last_page,
                layout=layout,
                no_page_breaks=no_page_breaks,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
