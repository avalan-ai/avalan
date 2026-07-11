from ....entities import ToolCallContext
from ..git import (
    ShellGitCommandName,
    ShellGitCommandRequest,
)
from ..settings import ShellToolSettings
from .git_base import _ShellGitCommandTool

from typing import Literal


class GitCommitTool(_ShellGitCommandTool):
    """Create a commit from the existing index.

    Args:
        message: Commit message to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.COMMIT, settings=settings)

    def _build_request(
        self,
        message: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"message": message},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        message: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                message=message,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitMergeTool(_ShellGitCommandTool):
    """Merge a constrained revision.

    Args:
        revision: Revision to merge.
        confirm_revision: Exact revision confirmation.
        mode: Merge mode to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.MERGE, settings=settings)

    def _build_request(
        self,
        revision: str,
        confirm_revision: str,
        mode: Literal["ff_only", "no_ff"] = "ff_only",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "revision": revision,
                "confirm_revision": confirm_revision,
                "mode": mode,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        confirm_revision: str,
        mode: Literal["ff_only", "no_ff"] = "ff_only",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                confirm_revision=confirm_revision,
                mode=mode,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRebaseTool(_ShellGitCommandTool):
    """Rebase onto a constrained upstream.

    Args:
        upstream: Upstream revision to rebase onto.
        confirm_upstream: Exact upstream confirmation.
        branch: Optional branch to rebase.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.REBASE, settings=settings)

    def _build_request(
        self,
        upstream: str,
        confirm_upstream: str,
        branch: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "upstream": upstream,
                "confirm_upstream": confirm_upstream,
                "branch": branch,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        upstream: str,
        confirm_upstream: str,
        branch: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                upstream=upstream,
                confirm_upstream=confirm_upstream,
                branch=branch,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCherryPickTool(_ShellGitCommandTool):
    """Cherry-pick a constrained revision.

    Args:
        revision: Revision to cherry-pick.
        confirm_revision: Exact revision confirmation.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.CHERRY_PICK,
            settings=settings,
        )

    def _build_request(
        self,
        revision: str,
        confirm_revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "revision": revision,
                "confirm_revision": confirm_revision,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        confirm_revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                confirm_revision=confirm_revision,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRevertTool(_ShellGitCommandTool):
    """Revert a constrained revision.

    Args:
        revision: Revision to revert.
        confirm_revision: Exact revision confirmation.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.REVERT, settings=settings)

    def _build_request(
        self,
        revision: str,
        confirm_revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "revision": revision,
                "confirm_revision": confirm_revision,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        confirm_revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                confirm_revision=confirm_revision,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
