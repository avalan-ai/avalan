from ....entities import ToolCallContext
from ..entities import ShellCommandRequest
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import _optional_cwd, _path_operands
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence
from typing import Any, Literal


class ShasumTool(_ShellCommandTool):
    """Hash workspace files with an allowlisted SHA algorithm.

    Args:
        paths: Workspace-relative regular file paths to hash.
        algorithm: SHA algorithm accepted by shasum.
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
            command="shasum",
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
        paths_schema = properties["paths"]
        assert isinstance(paths_schema, dict)
        paths_schema["minItems"] = 1
        paths_schema["maxItems"] = self._settings.max_path_count
        return schema

    def _build_request(
        self,
        paths: Sequence[str],
        algorithm: Literal[
            "1",
            "224",
            "256",
            "384",
            "512",
            "512224",
            "512256",
        ] = "1",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.shasum",
            command="shasum",
            options={"algorithm": algorithm},
            paths=_path_operands(paths, kind="file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        algorithm: Literal[
            "1",
            "224",
            "256",
            "384",
            "512",
            "512224",
            "512256",
        ] = "1",
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
                algorithm=algorithm,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
