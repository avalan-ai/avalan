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

from typing import Literal


class NlTool(_ShellCommandTool):
    """Number lines in a workspace text file.

    Args:
        path: Workspace-relative text file path to number.
        cwd: Workspace-relative working directory for the command.
        body_numbering: Body line numbering style.
        number_format: Line number alignment and padding style.
        number_separator: Separator between line numbers and content.
        starting_line_number: First line number to emit.
        line_increment: Increment between emitted line numbers.
        number_width: Minimum line number field width.
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
            command="nl",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        cwd: str | None = None,
        body_numbering: Literal["all", "nonempty", "none"] = "all",
        number_format: Literal["left", "right", "right_zero"] = "right",
        number_separator: Literal[
            "colon_space",
            "space",
            "tab",
            "two_spaces",
        ] = "tab",
        starting_line_number: int = 1,
        line_increment: int = 1,
        number_width: int = 6,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.nl",
            command="nl",
            options={
                "body_numbering": body_numbering,
                "number_format": number_format,
                "number_separator": number_separator,
                "starting_line_number": starting_line_number,
                "line_increment": line_increment,
                "number_width": number_width,
            },
            paths=_path_operands((path,), kind="text_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        cwd: str | None = None,
        body_numbering: Literal["all", "nonempty", "none"] = "all",
        number_format: Literal["left", "right", "right_zero"] = "right",
        number_separator: Literal[
            "colon_space",
            "space",
            "tab",
            "two_spaces",
        ] = "tab",
        starting_line_number: int = 1,
        line_increment: int = 1,
        number_width: int = 6,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                cwd=_optional_cwd(cwd),
                body_numbering=body_numbering,
                number_format=number_format,
                number_separator=number_separator,
                starting_line_number=starting_line_number,
                line_increment=line_increment,
                number_width=number_width,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
