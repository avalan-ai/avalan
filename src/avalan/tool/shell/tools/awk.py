from ....entities import ToolCallContext
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _optional_cwd,
    _optional_int_tuple,
    _path_operands,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence
from typing import Literal


class AwkTool(_ShellCommandTool):
    """Select fields and lines from workspace text files.

    Args:
        paths: Workspace-relative text file paths to read.
        fields: Optional one-based field indexes to print.
        field_separator: Input field separator mode.
        output_separator: Separator to use between printed fields.
        pattern: Optional literal pattern to match whole input lines.
        start_line: Optional first one-based line number to include.
        end_line: Optional last one-based line number to include.
        cwd: Workspace-relative working directory for the command.
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
            command="awk",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        fields: Sequence[int] | None = None,
        field_separator: Literal[
            "whitespace",
            "tab",
            "comma",
            "pipe",
        ] = "whitespace",
        output_separator: str = " ",
        pattern: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.awk",
            command="awk",
            options={
                "fields": _optional_int_tuple(fields, "fields"),
                "field_separator": field_separator,
                "output_separator": output_separator,
                "pattern": pattern,
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
        fields: Sequence[int] | None = None,
        field_separator: Literal[
            "whitespace",
            "tab",
            "comma",
            "pipe",
        ] = "whitespace",
        output_separator: str = " ",
        pattern: str | None = None,
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
                paths=paths,
                fields=fields,
                field_separator=field_separator,
                output_separator=output_separator,
                pattern=pattern,
                start_line=start_line,
                end_line=end_line,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
