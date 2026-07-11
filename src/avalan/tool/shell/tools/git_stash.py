from ....entities import ToolCallContext
from ..git import (
    ShellGitCommandName,
    ShellGitCommandRequest,
)
from ..settings import ShellToolSettings
from .git_base import _ShellGitCommandTool

from collections.abc import Sequence
from typing import Literal


class GitStashListTool(_ShellGitCommandTool):
    """Inspect bounded stash metadata.

    Args:
        max_count: Maximum number of stash entries to return.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_LIST,
            settings=settings,
        )

    def _build_request(
        self,
        max_count: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"max_count": max_count},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        max_count: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                max_count=max_count,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashShowTool(_ShellGitCommandTool):
    """Inspect bounded stash details.

    Args:
        stash: Stash reference to inspect.
        mode: Fixed stash show mode to request.
        paths: Optional repo-relative pathspecs to inspect.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_SHOW,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        mode: Literal["stat", "patch"] = "stat",
        *,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash, "mode": mode},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        mode: Literal["stat", "patch"] = "stat",
        *,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                mode=mode,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashPushTool(_ShellGitCommandTool):
    """Create a bounded stash entry.

    Args:
        message: Optional stash message.
        paths: Repo-relative paths to include.
        include_untracked: Include untracked files in the stash request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_PUSH,
            settings=settings,
        )

    def _build_request(
        self,
        message: str | None = None,
        *,
        paths: Sequence[str],
        include_untracked: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "message": message,
                "include_untracked": include_untracked,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        message: str | None = None,
        *,
        paths: Sequence[str],
        include_untracked: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                message=message,
                paths=paths,
                include_untracked=include_untracked,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashApplyTool(_ShellGitCommandTool):
    """Restore explicit repo-relative paths from a stash entry.

    Args:
        stash: Stash reference to apply.
        paths: Repo-relative paths to restore from the stash.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_APPLY,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str,
        *,
        paths: Sequence[str],
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str,
        *,
        paths: Sequence[str],
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashPopTool(_ShellGitCommandTool):
    """Pop a bounded stash entry.

    Args:
        stash: Stash reference to pop.
        confirm_stash: Exact stash reference confirmation.
        index: Restore index state from the stash.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_POP,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        *,
        confirm_stash: str,
        index: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "stash": stash,
                "confirm_stash": confirm_stash,
                "index": index,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        *,
        confirm_stash: str,
        index: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                confirm_stash=confirm_stash,
                index=index,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashDropTool(_ShellGitCommandTool):
    """Drop a bounded stash entry.

    Args:
        stash: Stash reference to drop.
        confirm_stash: Exact stash reference confirmation.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_DROP,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        *,
        confirm_stash: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash, "confirm_stash": confirm_stash},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        *,
        confirm_stash: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                confirm_stash=confirm_stash,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
