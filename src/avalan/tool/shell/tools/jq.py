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


class JqTool(_ShellCommandTool):
    """Transform workspace JSON files with a constrained jq filter.

    Args:
        filter: Constrained jq filter expression.
        paths: Workspace-relative JSON file paths to read.
        cwd: Workspace-relative working directory for the command.
        raw_output: Emit raw string output.
        compact: Emit compact JSON output.
        slurp: Read all inputs into an array.
        sort_keys: Sort JSON object keys in output.
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
            command="jq",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        filter: str,
        paths: Sequence[str],
        cwd: str | None = None,
        raw_output: bool = False,
        compact: bool = False,
        slurp: bool = False,
        sort_keys: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.jq",
            command="jq",
            options={
                "filter": filter,
                "raw_output": raw_output,
                "compact": compact,
                "slurp": slurp,
                "sort_keys": sort_keys,
            },
            paths=_path_operands(paths, kind="json_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        filter: str,
        paths: Sequence[str],
        cwd: str | None = None,
        raw_output: bool = False,
        compact: bool = False,
        slurp: bool = False,
        sort_keys: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                filter=filter,
                paths=paths,
                cwd=_optional_cwd(cwd),
                raw_output=raw_output,
                compact=compact,
                slurp=slurp,
                sort_keys=sort_keys,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
