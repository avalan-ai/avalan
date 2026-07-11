from ....entities import ToolCallContext
from ..git import (
    ShellGitCommandName,
    ShellGitCommandRequest,
)
from ..settings import ShellToolSettings
from .git_base import _ShellGitCommandTool

from collections.abc import Sequence
from typing import Literal


class GitStatusTool(_ShellGitCommandTool):
    """Inspect repository status metadata.

    Args:
        mode: Status output mode to request.
        paths: Repo-relative pathspecs to inspect.
        cwd: Workspace-relative working directory for repository discovery.
        include_branch: Include branch metadata in the status request.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.STATUS, settings=settings)

    def _build_request(
        self,
        mode: Literal["porcelain_v2", "short"] = "porcelain_v2",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        include_branch: bool = True,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "include_branch": include_branch},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["porcelain_v2", "short"] = "porcelain_v2",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        include_branch: bool = True,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                paths=paths,
                cwd=cwd,
                include_branch=include_branch,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRevParseTool(_ShellGitCommandTool):
    """Inspect approved repository and revision facts.

    Args:
        fact: Approved repository or revision fact to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REV_PARSE,
            settings=settings,
        )

    def _build_request(
        self,
        fact: Literal[
            "head",
            "short_head",
            "current_branch",
            "repo_root",
        ] = "head",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"fact": fact},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        fact: Literal[
            "head",
            "short_head",
            "current_branch",
            "repo_root",
        ] = "head",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                fact=fact,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchTool(_ShellGitCommandTool):
    """Inspect current or listed branches.

    Args:
        mode: Branch read mode to request.
        contains: Optional revision used to filter listed branches.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.BRANCH, settings=settings)

    def _build_request(
        self,
        mode: Literal["current", "list"] = "current",
        contains: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "contains": contains},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["current", "list"] = "current",
        contains: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                contains=contains,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitTagTool(_ShellGitCommandTool):
    """Inspect listed or shown tags.

    Args:
        mode: Tag read mode to request.
        name: Optional tag name for show mode.
        max_count: Optional maximum number of tags to return.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.TAG, settings=settings)

    def _build_request(
        self,
        mode: Literal["list", "show"] = "list",
        name: str | None = None,
        max_count: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "name": name, "max_count": max_count},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["list", "show"] = "list",
        name: str | None = None,
        max_count: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                name=name,
                max_count=max_count,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitDescribeTool(_ShellGitCommandTool):
    """Inspect bounded describe metadata.

    Args:
        target: Optional revision or ref to describe.
        mode: Describe mode to request.
        max_candidates: Maximum tag candidates to consider.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.DESCRIBE,
            settings=settings,
        )

    def _build_request(
        self,
        target: str | None = None,
        mode: Literal["tags", "always"] = "tags",
        max_candidates: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "target": target,
                "mode": mode,
                "max_candidates": max_candidates,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        target: str | None = None,
        mode: Literal["tags", "always"] = "tags",
        max_candidates: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                target=target,
                mode=mode,
                max_candidates=max_candidates,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitLsFilesTool(_ShellGitCommandTool):
    """List repository paths with safe modes.

    Args:
        mode: File listing mode to request.
        paths: Repo-relative pathspecs to list.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.LS_FILES,
            settings=settings,
        )

    def _build_request(
        self,
        mode: Literal["tracked", "modified", "deleted", "others"] = "tracked",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["tracked", "modified", "deleted", "others"] = "tracked",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitLogTool(_ShellGitCommandTool):
    """Inspect bounded history summaries.

    Args:
        max_count: Maximum number of commits to return.
        revision: Optional revision range to inspect.
        paths: Repo-relative pathspecs to filter history.
        format: Fixed history format to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.LOG, settings=settings)

    def _build_request(
        self,
        max_count: int = 10,
        revision: str | None = None,
        paths: Sequence[str] = (),
        format: Literal["summary", "oneline"] = "summary",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "max_count": max_count,
                "revision": revision,
                "format": format,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        max_count: int = 10,
        revision: str | None = None,
        paths: Sequence[str] = (),
        format: Literal["summary", "oneline"] = "summary",
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
                revision=revision,
                paths=paths,
                format=format,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitDiffTool(_ShellGitCommandTool):
    """Inspect bounded repository diffs.

    Args:
        mode: Diff mode to request.
        base_revision: Optional base revision for range mode.
        head_revision: Optional head revision for range mode.
        paths: Optional repo-relative pathspecs to diff.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.DIFF, settings=settings)

    def _build_request(
        self,
        mode: Literal[
            "worktree", "staged", "range", "stat", "name_only"
        ] = "worktree",
        base_revision: str | None = None,
        head_revision: str | None = None,
        *,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "mode": mode,
                "base_revision": base_revision,
                "head_revision": head_revision,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal[
            "worktree", "staged", "range", "stat", "name_only"
        ] = "worktree",
        base_revision: str | None = None,
        head_revision: str | None = None,
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
                mode=mode,
                base_revision=base_revision,
                head_revision=head_revision,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitShowTool(_ShellGitCommandTool):
    """Inspect bounded commit or tag details.

    Args:
        revision: Revision or tag to inspect.
        mode: Fixed show mode to request.
        paths: Repo-relative file pathspecs for stat and patch modes.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.SHOW, settings=settings)

    def _build_request(
        self,
        revision: str,
        mode: Literal["summary", "stat", "patch"] = "summary",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"revision": revision, "mode": mode},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        mode: Literal["summary", "stat", "patch"] = "summary",
        paths: Sequence[str] = (),
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
                mode=mode,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBlameTool(_ShellGitCommandTool):
    """Inspect bounded line blame for one file.

    Args:
        path: Repo-relative file path to inspect.
        start_line: Optional first line to inspect.
        end_line: Optional final line to inspect.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.BLAME, settings=settings)

    def _build_request(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"start_line": start_line, "end_line": end_line},
            pathspecs=(path,),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
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
                start_line=start_line,
                end_line=end_line,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitGrepTool(_ShellGitCommandTool):
    """Search repository content with bounded grep.

    Args:
        pattern: Search pattern to request.
        paths: Explicit repo-relative file pathspecs to search.
        case: Case-sensitivity mode.
        max_matches: Maximum matches per file to return.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.GREP, settings=settings)

    def _build_request(
        self,
        pattern: str,
        paths: Sequence[str],
        case: Literal["sensitive", "insensitive"] = "sensitive",
        max_matches: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "pattern": pattern,
                "case": case,
                "max_matches": max_matches,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        pattern: str,
        paths: Sequence[str],
        case: Literal["sensitive", "insensitive"] = "sensitive",
        max_matches: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                pattern=pattern,
                paths=paths,
                case=case,
                max_matches=max_matches,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
