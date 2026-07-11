from ....entities import ToolCallContext
from ..entities import (
    ShellCommandRequest,
)
from ..executor import CommandExecutor
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._arguments import (
    _copied_json_schema_properties,
    _optional_cwd,
    _path_operands,
    _string_tuple,
)
from ._base import ShellResultFormatter, _ShellCommandTool

from collections.abc import Sequence
from typing import Any, Literal


class RgTool(_ShellCommandTool):
    """Search workspace text or list workspace files with ripgrep.

    Args:
        pattern: Search pattern to pass as a literal tool argument.
            Required in search mode.
        paths: Workspace-relative files or directories to search or list.
        cwd: Workspace-relative working directory for the command.
        case: Case-sensitivity mode.
        fixed_strings: Treat the pattern as a fixed string.
        context_lines: Number of context lines around each match.
        before_context: Number of leading context lines before each match.
        after_context: Number of trailing context lines after each match.
        max_matches_per_file: Maximum matches to return per file.
        max_depth: Maximum directory traversal depth for ripgrep.
        max_filesize_bytes: Skip files larger than this byte count.
        globs: Include or exclude glob patterns for ripgrep.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.
        mode: Ripgrep operation mode.

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
            command="rg",
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
        search_properties = _copied_json_schema_properties(properties)
        files_properties = {
            name: schema_property
            for name, schema_property in _copied_json_schema_properties(
                properties
            ).items()
            if name
            in {
                "mode",
                "paths",
                "cwd",
                "max_depth",
                "max_filesize_bytes",
                "globs",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            }
        }
        pattern_schema = search_properties["pattern"]
        assert isinstance(pattern_schema, dict)
        pattern_schema["type"] = "string"
        pattern_schema["minLength"] = 1
        search_mode_schema = search_properties["mode"]
        assert isinstance(search_mode_schema, dict)
        search_mode_schema["enum"] = ["search"]
        files_mode_schema = files_properties["mode"]
        assert isinstance(files_mode_schema, dict)
        files_mode_schema["enum"] = ["files"]
        files_mode_schema.pop("default", None)
        parameters["anyOf"] = [
            {
                "type": "object",
                "properties": search_properties,
                "required": ["pattern"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": files_properties,
                "required": ["mode"],
                "additionalProperties": False,
            },
        ]
        return schema

    def _build_request(
        self,
        pattern: str | None = None,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        case: Literal["sensitive", "insensitive", "smart"] = "sensitive",
        fixed_strings: bool = False,
        context_lines: int = 0,
        before_context: int | None = None,
        after_context: int | None = None,
        max_matches_per_file: int | None = None,
        max_depth: int | None = None,
        max_filesize_bytes: int | None = None,
        globs: Sequence[str] = (),
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        mode: Literal["search", "files"] = "search",
    ) -> ShellCommandRequest:
        assert mode != "search" or pattern is not None
        options: dict[str, object] = {
            "max_depth": max_depth,
            "max_filesize_bytes": max_filesize_bytes,
            "globs": _string_tuple(globs, "globs"),
        }
        if mode == "search":
            options["pattern"] = pattern
            options["case"] = case
            options["fixed_strings"] = fixed_strings
            options["context_lines"] = context_lines
            options["before_context"] = before_context
            options["after_context"] = after_context
            options["max_matches_per_file"] = max_matches_per_file
        else:
            options["mode"] = mode
            if pattern is not None:
                options["pattern"] = pattern
            if case != "sensitive":
                options["case"] = case
            if fixed_strings:
                options["fixed_strings"] = fixed_strings
            if context_lines:
                options["context_lines"] = context_lines
            if before_context is not None:
                options["before_context"] = before_context
            if after_context is not None:
                options["after_context"] = after_context
            if max_matches_per_file is not None:
                options["max_matches_per_file"] = max_matches_per_file
        return ShellCommandRequest(
            tool_name="shell.rg",
            command="rg",
            options=options,
            paths=_path_operands(paths, kind="any"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        pattern: str | None = None,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        case: Literal["sensitive", "insensitive", "smart"] = "sensitive",
        fixed_strings: bool = False,
        context_lines: int = 0,
        before_context: int | None = None,
        after_context: int | None = None,
        max_matches_per_file: int | None = None,
        max_depth: int | None = None,
        max_filesize_bytes: int | None = None,
        globs: Sequence[str] = (),
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        mode: Literal["search", "files"] = "search",
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                pattern=pattern,
                paths=paths,
                cwd=_optional_cwd(cwd),
                case=case,
                fixed_strings=fixed_strings,
                context_lines=context_lines,
                before_context=before_context,
                after_context=after_context,
                max_matches_per_file=max_matches_per_file,
                max_depth=max_depth,
                max_filesize_bytes=max_filesize_bytes,
                globs=globs,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                mode=mode,
            ),
            context=context,
        )
