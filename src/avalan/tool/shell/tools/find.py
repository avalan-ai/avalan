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

from collections.abc import Sequence
from typing import Literal


class FindTool(_ShellCommandTool):
    """Find workspace entries with constrained selectors.

    Args:
        paths: Workspace-relative file or directory roots to search.
        cwd: Workspace-relative working directory for the command.
        min_depth: Minimum traversal depth below each root.
        max_depth: Maximum traversal depth below each root.
        entry_type: Entry type to include.
        name: Optional exact basename to match.
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
            command="find",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        min_depth: int | None = None,
        max_depth: int = 3,
        entry_type: Literal["any", "file", "directory"] = "any",
        name: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.find",
            command="find",
            options={
                "min_depth": min_depth,
                "max_depth": max_depth,
                "entry_type": entry_type,
                "name": name,
            },
            paths=_path_operands(paths, kind="any"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        min_depth: int | None = None,
        max_depth: int = 3,
        entry_type: Literal["any", "file", "directory"] = "any",
        name: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                paths=paths,
                cwd=_optional_cwd(cwd),
                min_depth=min_depth,
                max_depth=max_depth,
                entry_type=entry_type,
                name=name,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
