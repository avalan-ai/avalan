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


class PdfToPpmTool(_ShellCommandTool):
    """Rasterize pages from a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to rasterize.
        first_page: First one-based page number to rasterize.
        last_page: Optional last one-based page number to rasterize.
        dpi: Rasterization dots per inch.
        grayscale: Render grayscale output.
        format: Raster image format.
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
            command="pdftoppm",
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
        dpi: int | None = None,
        grayscale: bool = False,
        format: Literal["png"] = "png",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pdftoppm",
            command="pdftoppm",
            options={
                "first_page": first_page,
                "last_page": last_page,
                "dpi": dpi,
                "grayscale": grayscale,
                "format": format,
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
        dpi: int | None = None,
        grayscale: bool = False,
        format: Literal["png"] = "png",
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
                dpi=dpi,
                grayscale=grayscale,
                format=format,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
