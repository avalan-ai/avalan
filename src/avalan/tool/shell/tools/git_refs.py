from ....entities import ToolCallContext
from ..git import (
    ShellGitCommandName,
    ShellGitCommandRequest,
)
from ..settings import ShellToolSettings
from .git_base import _ShellGitCommandTool


class GitBranchCreateTool(_ShellGitCommandTool):
    """Create a branch from an approved start point.

    Args:
        name: Branch name to create.
        start_point: Optional start point revision.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.BRANCH_CREATE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        start_point: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "start_point": start_point},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        start_point: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                start_point=start_point,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchDeleteTool(_ShellGitCommandTool):
    """Delete a constrained branch name.

    Args:
        name: Branch name to delete.
        confirm_name: Exact branch name confirmation.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.BRANCH_DELETE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        confirm_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "confirm_name": confirm_name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        confirm_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                confirm_name=confirm_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchRenameTool(_ShellGitCommandTool):
    """Rename one constrained branch to another.

    Args:
        old_name: Existing branch name.
        new_name: New branch name.
        confirm_old_name: Exact existing branch name confirmation.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.BRANCH_RENAME,
            settings=settings,
        )

    def _build_request(
        self,
        old_name: str,
        new_name: str,
        confirm_old_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "old_name": old_name,
                "new_name": new_name,
                "confirm_old_name": confirm_old_name,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        old_name: str,
        new_name: str,
        confirm_old_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                old_name=old_name,
                new_name=new_name,
                confirm_old_name=confirm_old_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitTagCreateTool(_ShellGitCommandTool):
    """Create a constrained tag reference.

    Args:
        name: Tag name to create.
        target: Optional target revision.
        message: Optional tag annotation message.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.TAG_CREATE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        target: str | None = None,
        message: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "target": target, "message": message},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        target: str | None = None,
        message: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                target=target,
                message=message,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitTagDeleteTool(_ShellGitCommandTool):
    """Delete a constrained tag reference.

    Args:
        name: Tag name to delete.
        confirm_name: Exact tag name confirmation.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.TAG_DELETE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        confirm_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "confirm_name": confirm_name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        confirm_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                confirm_name=confirm_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
