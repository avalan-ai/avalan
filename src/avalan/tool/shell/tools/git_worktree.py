from ....entities import ToolCallContext
from ..git import (
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
)
from ..settings import ShellToolSettings
from ._arguments import _optional_cwd, _string_tuple
from .git_base import _git_settings, _ShellGitCommandTool

from collections.abc import Sequence
from typing import Any, Literal, cast


class GitAddTool(_ShellGitCommandTool):
    """Stage repo-relative paths for addition.

    Args:
        paths: Repo-relative paths to stage.
        mode: Add mode to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.ADD, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str],
        mode: Literal["normal", "intent_to_add"] = "normal",
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
        paths: Sequence[str],
        mode: Literal["normal", "intent_to_add"] = "normal",
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
                mode=mode,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRestoreTool(_ShellGitCommandTool):
    """Restore repo-relative paths from a safe source.

    Args:
        paths: Repo-relative paths to restore.
        source_revision: Optional source revision.
        staged: Restore index state.
        worktree: Restore worktree state.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.RESTORE,
            settings=settings,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        source_revision: str | None = None,
        staged: bool = False,
        worktree: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "source_revision": source_revision,
                "staged": staged,
                "worktree": worktree,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        source_revision: str | None = None,
        staged: bool = False,
        worktree: bool = True,
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
                source_revision=source_revision,
                staged=staged,
                worktree=worktree,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCheckoutTool(_ShellGitCommandTool):
    """Checkout constrained repo-relative paths.

    Args:
        paths: Repo-relative paths to checkout.
        target: Optional source revision for path checkout.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.CHECKOUT,
            settings=settings,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        target: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"target": target},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        target: str | None = None,
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
                target=target,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitSwitchTool(_ShellGitCommandTool):
    """Switch to a constrained branch target.

    Args:
        branch: Branch name to switch to.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.SWITCH, settings=settings)

    def _build_request(
        self,
        branch: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"branch": branch},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        branch: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                branch=branch,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitResetTool(_ShellGitCommandTool):
    """Reset constrained worktree paths or refs.

    Args:
        paths: Repo-relative paths to reset.
        mode: Reset mode to request.
        revision: Revision for ref-moving reset modes.
        confirm_revision: Exact revision confirmation.
        confirm_hard: Confirm hard reset mode.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.RESET, settings=settings)

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        schema = super().json_schema(prefix)
        parameters = schema["function"]["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        git_settings = _git_settings(self._settings)
        has_worktree = ShellGitCapability.WORKTREE.value in (
            git_settings.capabilities
        )
        has_history = ShellGitCapability.HISTORY.value in (
            git_settings.capabilities
        )
        common_names = (
            "cwd",
            "timeout_seconds",
            "max_stdout_bytes",
            "max_stderr_bytes",
        )
        worktree_properties = {
            "paths": properties["paths"],
            **{name: properties[name] for name in common_names},
        }
        history_properties = {
            "mode": dict(cast(dict[str, object], properties["mode"])),
            "revision": properties["revision"],
            "confirm_revision": properties["confirm_revision"],
            "confirm_hard": properties["confirm_hard"],
            **{name: properties[name] for name in common_names},
        }
        cast(dict[str, object], history_properties["mode"])["enum"] = [
            "soft",
            "mixed",
            "hard",
        ]
        cast(dict[str, object], history_properties["mode"]).pop(
            "default",
            None,
        )
        if has_worktree and not has_history:
            parameters["properties"] = worktree_properties
            parameters["required"] = ["paths"]
            parameters.pop("anyOf", None)
        elif has_history and not has_worktree:
            parameters["properties"] = history_properties
            parameters["required"] = [
                "mode",
                "revision",
                "confirm_revision",
            ]
            parameters.pop("anyOf", None)
        elif has_worktree and has_history:
            parameters["anyOf"] = [
                {
                    "type": "object",
                    "properties": worktree_properties,
                    "required": ["paths"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": history_properties,
                    "required": [
                        "mode",
                        "revision",
                        "confirm_revision",
                    ],
                    "additionalProperties": False,
                },
            ]
        return schema

    def _build_request(
        self,
        paths: Sequence[str] = (),
        mode: Literal["paths", "soft", "mixed", "hard"] = "paths",
        revision: str | None = None,
        confirm_revision: str | None = None,
        confirm_hard: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        capability = (
            ShellGitCapability.HISTORY
            if mode in ("soft", "mixed", "hard")
            else ShellGitCapability.WORKTREE
        )
        return ShellGitCommandRequest(
            tool_name=f"shell.{self.__name__}",
            command=self._command,
            capability_required=capability,
            options={
                "mode": mode,
                "revision": revision,
                "confirm_revision": confirm_revision,
                "confirm_hard": confirm_hard,
            },
            pathspecs=_string_tuple(paths, "pathspecs"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str] = (),
        mode: Literal["paths", "soft", "mixed", "hard"] = "paths",
        revision: str | None = None,
        confirm_revision: str | None = None,
        confirm_hard: bool = False,
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
                mode=mode,
                revision=revision,
                confirm_revision=confirm_revision,
                confirm_hard=confirm_hard,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRmTool(_ShellGitCommandTool):
    """Remove repo-relative paths from the worktree and index.

    Args:
        paths: Repo-relative paths to remove.
        cached: Remove paths from the index only.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.RM, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str],
        cached: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"cached": cached},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        cached: bool = False,
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
                cached=cached,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitMvTool(_ShellGitCommandTool):
    """Move one repo-relative path to another.

    Args:
        source: Repo-relative source path.
        destination: Repo-relative destination path.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.MV, settings=settings)

    def _build_request(
        self,
        source: str,
        destination: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"source": source, "destination": destination},
            pathspecs=(source, destination),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        source: str,
        destination: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                source=source,
                destination=destination,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCleanTool(_ShellGitCommandTool):
    """Clean constrained untracked paths.

    Args:
        paths: Repo-relative paths to clean.
        dry_run: Request dry-run output only.
        confirm_paths: Exact paths confirmation for non-dry-run clean.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.CLEAN, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str],
        dry_run: bool = True,
        confirm_paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "dry_run": dry_run,
                "confirm_paths": tuple(confirm_paths),
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        dry_run: bool = True,
        confirm_paths: Sequence[str] = (),
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
                dry_run=dry_run,
                confirm_paths=confirm_paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
