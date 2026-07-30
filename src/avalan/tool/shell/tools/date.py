from ....entities import ToolCallContext
from ..date import (
    DATE_CUSTOM_FORMAT_DIRECTIVES,
    DATE_CUSTOM_FORMAT_MAX_BYTES,
    DATE_CUSTOM_FORMAT_PATTERN,
)
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

from typing import Any, Literal


class DateTool(_ShellCommandTool):
    """Read the current date and time in a bounded portable format.

    Args:
        utc: Read Coordinated Universal Time instead of local time.
        format: Fixed output format. Default preserves native date output.
            Date emits YYYY-MM-DD, time emits HH:MM:SS, ISO 8601 emits
            YYYY-MM-DDTHH:MM:SS followed by a numeric offset, and Unix emits
            epoch seconds.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.
        custom_format: Optional 1-128-byte printable ASCII format using only
            the documented two-character directives. Mutually exclusive with
            non-default format.

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
            command="date",
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
        custom_format_schema = properties["custom_format"]
        assert isinstance(custom_format_schema, dict)
        custom_format_schema["minLength"] = 1
        custom_format_schema["maxLength"] = DATE_CUSTOM_FORMAT_MAX_BYTES
        custom_format_schema["pattern"] = DATE_CUSTOM_FORMAT_PATTERN
        custom_format_schema["description"] = (
            "Optional printable ASCII date format using only these "
            "two-character directives: "
            + " ".join(
                f"%{directive}" for directive in DATE_CUSTOM_FORMAT_DIRECTIVES
            )
            + ". Mutually exclusive with non-default format."
        )
        parameters["not"] = {
            "type": "object",
            "properties": {
                "format": {
                    "enum": [
                        "date",
                        "time",
                        "iso8601",
                        "unix",
                    ]
                },
                "custom_format": {"type": "string"},
            },
            "required": ["format", "custom_format"],
        }
        return schema

    def _build_request(
        self,
        utc: bool = False,
        format: Literal[
            "default",
            "date",
            "time",
            "iso8601",
            "unix",
        ] = "default",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        custom_format: str | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.date",
            command="date",
            options={
                "utc": utc,
                "format": format,
                "custom_format": custom_format,
            },
            paths=(),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        utc: bool = False,
        format: Literal[
            "default",
            "date",
            "time",
            "iso8601",
            "unix",
        ] = "default",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        custom_format: str | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                utc=utc,
                format=format,
                custom_format=custom_format,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )
