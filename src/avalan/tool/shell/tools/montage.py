from ....entities import ToolCallContext
from ..entities import ShellCommandRequest
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import _optional_cwd, _path_operands
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence
from typing import Literal


class MontageTool(_ShellCommandTool):
    """Create one bounded composite image from workspace images.

    Args:
        paths: Explicit workspace-relative image paths in display order.
        thumbnail: Optional per-image dimensions as WIDTHxHEIGHT.
        tile: Optional tile dimensions as COLUMNSxROWS.
        geometry: Horizontal and vertical spacing as +X+Y.
        output_format: Generated image format.
        output_filename: Optional safe basename for the returned artifact.
        quality: Optional output quality from 1 through 100.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result with one generated image.
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
            command="montage",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        thumbnail: str | None = None,
        tile: str | None = None,
        geometry: str = "+0+0",
        output_format: Literal["jpg", "jpeg", "png"] = "jpg",
        output_filename: str | None = None,
        quality: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.montage",
            command="montage",
            options={
                "thumbnail": thumbnail,
                "tile": tile,
                "geometry": geometry,
                "output_format": output_format,
                "output_filename": output_filename,
                "quality": quality,
            },
            paths=_path_operands(paths, kind="image_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        thumbnail: str | None = None,
        tile: str | None = None,
        geometry: str = "+0+0",
        output_format: Literal["jpg", "jpeg", "png"] = "jpg",
        output_filename: str | None = None,
        quality: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                paths=paths,
                thumbnail=thumbnail,
                tile=tile,
                geometry=geometry,
                output_format=output_format,
                output_filename=output_filename,
                quality=quality,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
