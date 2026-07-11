from ....entities import ToolCallContext
from ..git import (
    ShellGitCommandName,
    ShellGitCommandRequest,
)
from ..settings import ShellToolSettings
from .git_base import _ShellGitCommandTool

from collections.abc import Sequence
from typing import Literal


class GitFetchTool(_ShellGitCommandTool):
    """Fetch bounded remote refs.

    Args:
        remote: Remote name to fetch from.
        ref_type: Typed remote ref form to fetch.
        ref_name: Branch or tag name selected by ref_type.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.FETCH, settings=settings)

    def _build_request(
        self,
        remote: str = "origin",
        ref_type: Literal["branch", "tag"] = "branch",
        ref_name: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "remote": remote,
                "ref_type": ref_type,
                "ref_name": ref_name,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        remote: str = "origin",
        ref_type: Literal["branch", "tag"] = "branch",
        ref_name: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                remote=remote,
                ref_type=ref_type,
                ref_name=ref_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitPullTool(_ShellGitCommandTool):
    """Pull bounded remote refs.

    Args:
        remote: Remote name to pull from.
        branch: Optional branch name to pull.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.PULL, settings=settings)

    def _build_request(
        self,
        remote: str = "origin",
        branch: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"remote": remote, "branch": branch},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        remote: str = "origin",
        branch: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                remote=remote,
                branch=branch,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitPushTool(_ShellGitCommandTool):
    """Push bounded remote refs.

    Args:
        remote: Remote name to push to.
        ref_type: Typed remote ref form to push.
        ref_name: Branch or tag name selected by ref_type.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.PUSH, settings=settings)

    def _build_request(
        self,
        remote: str = "origin",
        ref_type: Literal["branch", "tag"] = "branch",
        ref_name: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "remote": remote,
                "ref_type": ref_type,
                "ref_name": ref_name,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        remote: str = "origin",
        ref_type: Literal["branch", "tag"] = "branch",
        ref_name: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                remote=remote,
                ref_type=ref_type,
                ref_name=ref_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCloneTool(_ShellGitCommandTool):
    """Clone a remote repository into the workspace.

    Args:
        url: Remote URL to clone.
        destination: Workspace-relative clone destination.
        branch: Remote branch to clone.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.CLONE, settings=settings)

    def _build_request(
        self,
        url: str,
        destination: str,
        branch: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"url": url, "destination": destination, "branch": branch},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        url: str,
        destination: str,
        branch: str = "main",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                url=url,
                destination=destination,
                branch=branch,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteListTool(_ShellGitCommandTool):
    """List approved remote entries.

    Args:
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_LIST,
            settings=settings,
        )

    def _build_request(
        self,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteAddTool(_ShellGitCommandTool):
    """Add an approved remote entry.

    Args:
        name: Remote name to add.
        url: Approved remote URL to add.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_ADD,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        url: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "url": url},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        url: str,
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
                url=url,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteSetUrlTool(_ShellGitCommandTool):
    """Set an approved remote URL.

    Args:
        name: Remote name to update.
        url: Approved remote URL to store.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_SET_URL,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        url: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "url": url},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        url: str,
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
                url=url,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteRemoveTool(_ShellGitCommandTool):
    """Remove an approved remote entry.

    Args:
        name: Remote name to remove.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_REMOVE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
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
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteRenameTool(_ShellGitCommandTool):
    """Rename one approved remote entry to another.

    Args:
        old_name: Existing remote name.
        new_name: New remote name.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_RENAME,
            settings=settings,
        )

    def _build_request(
        self,
        old_name: str,
        new_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"old_name": old_name, "new_name": new_name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        old_name: str,
        new_name: str,
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
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitSubmoduleUpdateTool(_ShellGitCommandTool):
    """Update gated submodule paths.

    Args:
        paths: Repo-relative submodule paths to update.
        init: Initialize submodules before updating.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.SUBMODULE_UPDATE,
            settings=settings,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        init: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"init": init},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        init: bool = False,
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
                init=init,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
