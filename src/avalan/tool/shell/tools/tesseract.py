from ....entities import ToolCallContext
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _optional_cwd,
    _optional_string_tuple,
    _path_operands,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence
from typing import Literal


class TesseractTool(_ShellCommandTool):
    """Recognize text in a workspace image file.

    Args:
        path: Workspace-relative image file path to read.
        languages: OCR language identifiers to use.
        psm: Tesseract page segmentation mode.
        oem: Optional Tesseract OCR engine mode.
        dpi: Optional input image DPI hint.
        output_format: OCR output format.
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
            command="tesseract",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        languages: Sequence[str] | None = None,
        psm: int = 3,
        oem: int | None = None,
        dpi: int | None = None,
        output_format: Literal["txt"] = "txt",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.tesseract",
            command="tesseract",
            options={
                "languages": _optional_string_tuple(
                    languages,
                    "languages",
                ),
                "psm": psm,
                "oem": oem,
                "dpi": dpi,
                "output_format": output_format,
            },
            paths=_path_operands((path,), kind="image_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        languages: Sequence[str] | None = None,
        psm: int = 3,
        oem: int | None = None,
        dpi: int | None = None,
        output_format: Literal["txt"] = "txt",
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
                languages=languages,
                psm=psm,
                oem=oem,
                dpi=dpi,
                output_format=output_format,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
