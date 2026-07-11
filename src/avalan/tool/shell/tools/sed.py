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
    _string_tuple,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence


class SedTool(_ShellCommandTool):
    """Select line ranges and patterns from workspace text files.

    Args:
        paths: Workspace-relative text file paths to read.
        line_ranges: One-based line ranges such as "1" or "1,10".
        patterns: Literal patterns to select from input lines.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.
        start_line: Optional first one-based line number to select.
        end_line: Optional last one-based line number to select.

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
            command="sed",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        line_ranges: Sequence[str] = (),
        patterns: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.sed",
            command="sed",
            options={
                "line_ranges": _string_tuple(
                    line_ranges,
                    "line_ranges",
                ),
                "patterns": _string_tuple(patterns, "patterns"),
                "start_line": start_line,
                "end_line": end_line,
            },
            paths=_path_operands(paths, kind="text_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        line_ranges: Sequence[str] = (),
        patterns: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                paths=paths,
                line_ranges=line_ranges,
                patterns=patterns,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                start_line=start_line,
                end_line=end_line,
            ),
            context=context,
        )
