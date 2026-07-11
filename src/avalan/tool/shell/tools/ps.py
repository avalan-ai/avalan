from ....entities import ToolCallContext
from ....types import assert_int_sequence as _assert_int_sequence
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

from collections.abc import Sequence
from typing import Any, Literal


class PsTool(_ShellCommandTool):
    """Inspect one fixed view of a selected process identifier.

    Args:
        pids: A sequence containing exactly one process identifier to inspect.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.
        view: Fixed process fields to return. Summary returns PID, parent PID,
            state, elapsed time, and command name. Resources returns PID, CPU
            percent, memory percent, resident and virtual memory in KiB, CPU
            time, and nice value.

    Returns:
        Formatted shell result containing only fields from the selected view.
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
            command="ps",
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
        pids_schema = properties["pids"]
        assert isinstance(pids_schema, dict)
        pids_schema["minItems"] = 1
        pids_schema["maxItems"] = 1
        pids_schema["uniqueItems"] = True
        items = pids_schema["items"]
        assert isinstance(items, dict)
        items["minimum"] = 1
        items["maximum"] = 2**31 - 1
        return schema

    def _build_request(
        self,
        pids: Sequence[int],
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        view: Literal["summary", "resources"] = "summary",
    ) -> ShellCommandRequest:
        _assert_int_sequence(pids, "pids")
        return ShellCommandRequest(
            tool_name="shell.ps",
            command="ps",
            options={"pids": tuple(pids), "view": view},
            paths=(),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        pids: Sequence[int],
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        view: Literal["summary", "resources"] = "summary",
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                pids=pids,
                view=view,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
