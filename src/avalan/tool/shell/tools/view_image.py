from ....entities import ToolCallContext, ToolResult, ToolResultImageDetail
from ... import Tool
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ..tool_images import viewed_image_tool_result

from typing import Literal


class ViewImageTool(Tool):
    """Attach an allowed image file to the model's next tool continuation.

    Args:
        path: Workspace-relative image file to inspect.
        detail: Requested model image fidelity.
        cwd: Workspace-relative working directory for the path.

    Returns:
        Textual artifact metadata and genuine model-visible image content.
    """

    supports_streaming = True

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
    ) -> None:
        super().__init__()
        self.__name__ = "view_image"
        self._settings = settings
        self._policy = policy

    async def __call__(
        self,
        path: str,
        detail: Literal["auto", "low", "high", "original"] = "auto",
        cwd: str | None = None,
        *,
        context: ToolCallContext,
    ) -> ToolResult:
        resolved_path, display_path, metadata = (
            await self._policy.resolve_view_image_path(path, cwd=cwd)
        )
        return await viewed_image_tool_result(
            resolved_path,
            display_path,
            metadata,
            settings=self._settings,
            detail=ToolResultImageDetail(detail),
        )
