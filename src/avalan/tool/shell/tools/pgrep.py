from ....entities import ToolCall, ToolCallContext, ToolCallOutcome
from ..display import project_shell_tool_display
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _optional_cwd,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from typing import Any


class PgrepTool(_ShellCommandTool):
    """Find process identifiers through a bounded process-table query.

    Args:
        pattern: Process-name or command-line regular expression to match.
        full: Match the full command line without returning it.
        exact: Require the selected process text to match exactly.
        ignore_case: Match process text without case sensitivity.
        newest: Return only the newest matching process identifier.
        oldest: Return only the oldest matching process identifier.
        parent_pid: Restrict matches to children of this process identifier.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result containing process identifiers only.
    """

    supports_streaming = False

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="pgrep",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        schema = super().json_schema(prefix)
        parameters = schema["function"]["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        pattern_schema = properties["pattern"]
        assert isinstance(pattern_schema, dict)
        pattern_schema["minLength"] = 1
        parent_pid_schema = properties["parent_pid"]
        assert isinstance(parent_pid_schema, dict)
        parent_pid_schema["minimum"] = 1
        parent_pid_schema["maximum"] = 2**31 - 1
        return schema

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        if outcome is not None:
            return super().tool_display_projector(call, outcome)
        return project_shell_tool_display(
            call=call,
            request=ShellCommandRequest(
                tool_name="shell.pgrep",
                command="pgrep",
                options={},
                paths=(),
                cwd=None,
            ),
        )

    def _build_request(
        self,
        pattern: str,
        full: bool = False,
        exact: bool = False,
        ignore_case: bool = False,
        newest: bool = False,
        oldest: bool = False,
        parent_pid: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pgrep",
            command="pgrep",
            options={
                "pattern": pattern,
                "full": full,
                "exact": exact,
                "ignore_case": ignore_case,
                "newest": newest,
                "oldest": oldest,
                "parent_pid": parent_pid,
            },
            paths=(),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        pattern: str,
        full: bool = False,
        exact: bool = False,
        ignore_case: bool = False,
        newest: bool = False,
        oldest: bool = False,
        parent_pid: int | None = None,
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
                full=full,
                exact=exact,
                ignore_case=ignore_case,
                newest=newest,
                oldest=oldest,
                parent_pid=parent_pid,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
