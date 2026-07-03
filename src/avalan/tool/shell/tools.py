from ...entities import ToolCall, ToolCallContext, ToolCallOutcome
from ...types import assert_int_sequence as _assert_int_sequence
from ...types import assert_string_sequence as _assert_string_sequence
from .. import Tool
from ..display import REDACTED_DISPLAY_VALUE
from .composition_executor import CompositionExecutor
from .display import project_shell_tool_display
from .entities import (
    ExecutionResult,
    PathOperand,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionMode,
    ShellCompositionRequest,
    ShellCompositionResult,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellFormattedCompositionResult,
    ShellFormattedResult,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellStreamRef,
)
from .executor import CommandExecutor
from .formatting import (
    format_shell_composition_result,
    format_shell_result,
)
from .policy import ExecutionPolicy
from .registry import SHELL_COMMAND_DEFINITIONS
from .settings import ShellToolSettings

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from inspect import signature
from typing import Any, Literal, TypeAlias, TypedDict, cast

ShellResultFormatter = Callable[[ExecutionResult], str]
ShellCompositionResultFormatter = Callable[[ShellCompositionResult], str]
ShellPipelineOptionValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list[bool]
    | list[int]
    | list[float]
    | list[str]
)


class ShellPipelineStdinRefArgument(TypedDict):
    step_id: str
    stream: Literal["stdout"]


class _RequiredShellPipelineStepArgument(TypedDict):
    id: str
    command: str


class ShellPipelineStepArgument(
    _RequiredShellPipelineStepArgument,
    total=False,
):
    options: dict[str, ShellPipelineOptionValue]
    paths: list[str]
    cwd: str | None
    stdin_from: ShellPipelineStdinRefArgument | None


class _ShellCommandTool(Tool, ABC):
    _executor: CommandExecutor
    _formatter: ShellResultFormatter
    _policy: ExecutionPolicy
    _settings: ShellToolSettings
    supports_streaming = True

    def __init__(
        self,
        *,
        command: str,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None,
    ) -> None:
        super().__init__()
        self.__name__ = command
        self._settings = settings
        self._policy = policy
        self._executor = executor
        self._formatter = formatter or self._format_result

    async def _execute_request(
        self,
        request: ShellCommandRequest,
        *,
        context: ToolCallContext,
    ) -> str:
        try:
            spec = await self._policy.normalize(request)
        except ShellPolicyDenied as error:
            result = _policy_denied_result(request, error)
            return ShellFormattedResult(self._formatter(result), result)
        if context.stream_event is not None:
            result = await self._executor.execute(
                spec,
                stream=context.stream_event,
            )
        else:
            result = await self._executor.execute(spec)
        return ShellFormattedResult(self._formatter(result), result)

    def _format_result(self, result: ExecutionResult) -> str:
        return format_shell_result(result, settings=self._settings)

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        request = None if outcome is not None else self._display_request(call)
        return project_shell_tool_display(
            call=call,
            outcome=outcome,
            request=request,
        )

    def _display_request(self, call: ToolCall) -> ShellCommandRequest | None:
        if call.arguments is None:
            arguments = {}
        elif isinstance(call.arguments, dict):
            arguments = dict(call.arguments)
        else:
            return None
        builder = getattr(self, "_build_request")
        assert callable(builder), "shell tool must define _build_request"
        try:
            bound = signature(builder).bind(**arguments)
            bound.apply_defaults()
            request = builder(**bound.arguments)
        except (AssertionError, TypeError, ValueError):
            return None
        assert isinstance(
            request,
            ShellCommandRequest,
        ), "_build_request must return a shell command request"
        return request

    @abstractmethod
    async def __call__(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError


class PipelineTool(Tool):
    """Run a structured shell command pipeline.

    Args:
        steps: Ordered command stages with ids, commands, options, paths,
            working directories, and optional stdin references.
        mode: Shell-local composition mode to execute.
        timeout_seconds: Optional composition timeout in seconds.
        max_stdout_bytes: Optional aggregate stdout byte cap.
        max_stderr_bytes: Optional aggregate stderr byte cap.
        max_intermediate_bytes: Optional routed stdout byte cap.

    Returns:
        Formatted shell composition result.
    """

    _executor: CompositionExecutor
    _formatter: ShellCompositionResultFormatter
    _policy: ExecutionPolicy
    _settings: ShellToolSettings
    supports_streaming = True

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CompositionExecutor,
        formatter: ShellCompositionResultFormatter | None = None,
    ) -> None:
        super().__init__()
        self.__name__ = "pipeline"
        self._settings = settings
        self._policy = policy
        self._executor = executor
        self._formatter = formatter or self._format_result

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        schema = super().json_schema(prefix)
        parameters = schema["function"]["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        steps_schema = properties["steps"]
        assert isinstance(steps_schema, dict)
        steps_schema["minItems"] = 1
        step_schema = steps_schema["items"]
        assert isinstance(step_schema, dict)
        step_properties = step_schema["properties"]
        assert isinstance(step_properties, dict)
        _set_min_length(step_properties, "id", 1)
        _set_min_length(step_properties, "command", 1)
        stdin_schema = step_properties["stdin_from"]
        assert isinstance(stdin_schema, dict)
        stdin_properties = stdin_schema["properties"]
        assert isinstance(stdin_properties, dict)
        _set_min_length(stdin_properties, "step_id", 1)
        return schema

    def _build_request(
        self,
        steps: Sequence[ShellPipelineStepArgument],
        mode: ShellCompositionMode = "pipeline",
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        max_intermediate_bytes: int | None = None,
    ) -> ShellCompositionRequest:
        return ShellCompositionRequest(
            mode=mode,
            steps=_composition_step_requests(steps),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            max_intermediate_bytes=max_intermediate_bytes,
        )

    async def _execute_request(
        self,
        request: ShellCompositionRequest,
        *,
        context: ToolCallContext,
    ) -> str:
        try:
            spec = await self._policy.normalize_composition(request)
        except ShellPolicyDenied as error:
            result = _composition_policy_denied_result(request, error)
            return ShellFormattedCompositionResult(
                self._formatter(result),
                result,
            )
        if context.stream_event is not None:
            result = await self._executor.execute_composition(
                spec,
                stream=context.stream_event,
            )
        else:
            result = await self._executor.execute_composition(spec)
        return ShellFormattedCompositionResult(self._formatter(result), result)

    def _format_result(self, result: ShellCompositionResult) -> str:
        return format_shell_composition_result(
            result,
            settings=self._settings,
        )

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        request = None if outcome is not None else self._display_request(call)
        return project_shell_tool_display(
            call=call,
            outcome=outcome,
            request=request,
        )

    def _display_request(
        self,
        call: ToolCall,
    ) -> ShellCompositionRequest | None:
        if call.arguments is None:
            arguments = {}
        elif isinstance(call.arguments, dict):
            arguments = dict(call.arguments)
        else:
            return None
        try:
            bound = signature(self._build_request).bind(**arguments)
            bound.apply_defaults()
            request = self._build_request(**bound.arguments)
        except (AssertionError, TypeError, ValueError):
            return None
        if not _composition_request_has_safe_commands(request):
            return None
        return request

    async def __call__(
        self,
        steps: Sequence[ShellPipelineStepArgument],
        mode: ShellCompositionMode = "pipeline",
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        max_intermediate_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                steps=steps,
                mode=mode,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                max_intermediate_bytes=max_intermediate_bytes,
            ),
            context=context,
        )


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
            options.update(
                {
                    "pattern": pattern,
                    "case": case,
                    "fixed_strings": fixed_strings,
                    "context_lines": context_lines,
                    "before_context": before_context,
                    "after_context": after_context,
                    "max_matches_per_file": max_matches_per_file,
                }
            )
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


class HeadTool(_ShellCommandTool):
    """Read the first lines of a workspace text file.

    Args:
        path: Workspace-relative file path to read.
        lines: Number of leading lines to return.
        byte_count: Native byte count to read via head -c.
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
            command="head",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        lines: int = 80,
        byte_count: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return _line_reader_request(
            command="head",
            path=path,
            lines=lines,
            byte_count=byte_count,
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        lines: int = 80,
        byte_count: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                lines=lines,
                byte_count=byte_count,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class TailTool(_ShellCommandTool):
    """Read the last lines of a workspace text file.

    Args:
        path: Workspace-relative file path to read.
        lines: Number of trailing lines to return.
        start_line: One-based line number to start at via tail -n +N.
        byte_count: Native byte count to read via tail -c.
        start_byte: One-based byte number to start at via tail -c +N.
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
            command="tail",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        lines: int = 80,
        start_line: int | None = None,
        byte_count: int | None = None,
        start_byte: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return _line_reader_request(
            command="tail",
            path=path,
            lines=lines,
            byte_count=byte_count,
            start_line=start_line,
            start_byte=start_byte,
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        lines: int = 80,
        start_line: int | None = None,
        byte_count: int | None = None,
        start_byte: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                lines=lines,
                start_line=start_line,
                byte_count=byte_count,
                start_byte=start_byte,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class LsTool(_ShellCommandTool):
    """List workspace directory entries or a single file path.

    Args:
        path: Optional workspace-relative file or directory path to list.
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
            command="ls",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        paths = () if path is None or path == "" else (path,)
        return ShellCommandRequest(
            tool_name="shell.ls",
            command="ls",
            options={},
            paths=_path_operands(paths, kind="any"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str | None = None,
        cwd: str | None = None,
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
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class CatTool(_ShellCommandTool):
    """Read a workspace text file.

    Args:
        path: Workspace-relative text file path to read.
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
            command="cat",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.cat",
            command="cat",
            options={},
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
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


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


class FileTool(_ShellCommandTool):
    """Identify workspace file types.

    Args:
        paths: Workspace-relative regular file paths to inspect.
        cwd: Workspace-relative working directory for the command.
        brief: Omit file names from command output.
        mime_type: Emit MIME type output.
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
            command="file",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        brief: bool = False,
        mime_type: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.file",
            command="file",
            options={"brief": brief, "mime_type": mime_type},
            paths=_path_operands(paths, kind="file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        brief: bool = False,
        mime_type: bool = False,
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
                brief=brief,
                mime_type=mime_type,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


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


class WcTool(_ShellCommandTool):
    """Count lines, words, or bytes in workspace text files.

    Args:
        paths: Workspace-relative text file paths to count.
        cwd: Workspace-relative working directory for the command.
        lines: Include line counts.
        words: Include word counts.
        count_bytes: Include byte counts.
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
            command="wc",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        lines: bool = True,
        words: bool = False,
        count_bytes: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.wc",
            command="wc",
            options={
                "lines": lines,
                "words": words,
                "count_bytes": count_bytes,
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
        cwd: str | None = None,
        lines: bool = True,
        words: bool = False,
        count_bytes: bool = False,
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
                lines=lines,
                words=words,
                count_bytes=count_bytes,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


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


class PdfInfoTool(_ShellCommandTool):
    """Inspect metadata for a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to inspect.
        first_page: Optional first one-based page number for page details.
        last_page: Optional last one-based page number for page details.
        boxes: Include page bounding boxes.
        iso_dates: Emit dates in ISO-8601 format.
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
            command="pdfinfo",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        first_page: int | None = None,
        last_page: int | None = None,
        boxes: bool = False,
        iso_dates: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pdfinfo",
            command="pdfinfo",
            options={
                "first_page": first_page,
                "last_page": last_page,
                "boxes": boxes,
                "iso_dates": iso_dates,
            },
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        first_page: int | None = None,
        last_page: int | None = None,
        boxes: bool = False,
        iso_dates: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                first_page=first_page,
                last_page=last_page,
                boxes=boxes,
                iso_dates=iso_dates,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PdfToTextTool(_ShellCommandTool):
    """Extract text from a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to read.
        first_page: First one-based page number to extract.
        last_page: Optional last one-based page number to extract.
        layout: Preserve physical page layout where supported.
        no_page_breaks: Suppress page break markers in text output.
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
            command="pdftotext",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        no_page_breaks: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pdftotext",
            command="pdftotext",
            options={
                "first_page": first_page,
                "last_page": last_page,
                "layout": layout,
                "no_page_breaks": no_page_breaks,
            },
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        no_page_breaks: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                first_page=first_page,
                last_page=last_page,
                layout=layout,
                no_page_breaks=no_page_breaks,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PdfToPpmTool(_ShellCommandTool):
    """Rasterize pages from a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to rasterize.
        first_page: First one-based page number to rasterize.
        last_page: Optional last one-based page number to rasterize.
        dpi: Rasterization dots per inch.
        grayscale: Render grayscale output.
        format: Raster image format.
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
            command="pdftoppm",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        dpi: int | None = None,
        grayscale: bool = False,
        format: Literal["png"] = "png",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pdftoppm",
            command="pdftoppm",
            options={
                "first_page": first_page,
                "last_page": last_page,
                "dpi": dpi,
                "grayscale": grayscale,
                "format": format,
            },
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        dpi: int | None = None,
        grayscale: bool = False,
        format: Literal["png"] = "png",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                first_page=first_page,
                last_page=last_page,
                dpi=dpi,
                grayscale=grayscale,
                format=format,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class ReportLabTool(_ShellCommandTool):
    """Create a bounded generated PDF with ReportLab.

    Args:
        text: Text content to place into the generated PDF.
        title: PDF document title.
        page_size: Output page size.
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
            command="reportlab",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        text: str,
        title: str = "Document",
        page_size: Literal["letter", "a4"] = "letter",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.reportlab",
            command="reportlab",
            options={
                "text": text,
                "title": title,
                "page_size": page_size,
            },
            paths=(),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        text: str,
        title: str = "Document",
        page_size: Literal["letter", "a4"] = "letter",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                text=text,
                title=title,
                page_size=page_size,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PdfPlumberTool(_ShellCommandTool):
    """Extract text or tables from a workspace PDF with pdfplumber.

    Args:
        path: Workspace-relative PDF file path to read.
        mode: Extraction mode to run.
        first_page: First one-based page number to extract.
        last_page: Optional last one-based page number to extract.
        layout: Preserve physical page layout for text extraction.
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
            command="pdfplumber",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        mode: Literal["text", "tables"] = "text",
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.pdfplumber",
            command="pdfplumber",
            options={
                "mode": mode,
                "first_page": first_page,
                "last_page": last_page,
                "layout": layout,
            },
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        mode: Literal["text", "tables"] = "text",
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                mode=mode,
                first_page=first_page,
                last_page=last_page,
                layout=layout,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PyPdfTool(_ShellCommandTool):
    """Inspect metadata or extract text from a workspace PDF with pypdf.

    Args:
        path: Workspace-relative PDF file path to read.
        mode: pypdf operation mode to run.
        first_page: First one-based page number to extract in text mode.
        last_page: Optional last one-based page number in text mode.
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
            command="pypdf",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        mode: Literal["metadata", "text"] = "metadata",
        first_page: int = 1,
        last_page: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        options: dict[str, object] = {"mode": mode}
        if mode == "text":
            options["first_page"] = first_page
            options["last_page"] = last_page
        elif first_page != 1 or last_page is not None:
            options["first_page"] = first_page
            options["last_page"] = last_page
        return ShellCommandRequest(
            tool_name="shell.pypdf",
            command="pypdf",
            options=options,
            paths=_path_operands((path,), kind="pdf_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        mode: Literal["metadata", "text"] = "metadata",
        first_page: int = 1,
        last_page: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                mode=mode,
                first_page=first_page,
                last_page=last_page,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class TesseractTool(_ShellCommandTool):
    """Recognize text in a workspace image file.

    Args:
        path: Workspace-relative image file path to read.
        languages: OCR language identifiers to use.
        psm: Tesseract page segmentation mode.
        oem: Optional Tesseract OCR engine mode.
        dpi: Optional input image DPI hint.
        output_format: OCR output format.
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
            command="tesseract",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    def _build_request(
        self,
        path: str,
        languages: Sequence[str] | None = None,
        psm: int = 3,
        oem: int | None = None,
        dpi: int | None = None,
        output_format: Literal["txt"] = "txt",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellCommandRequest:
        return ShellCommandRequest(
            tool_name="shell.tesseract",
            command="tesseract",
            options={
                "languages": _optional_string_tuple(
                    languages,
                    "languages",
                ),
                "psm": psm,
                "oem": oem,
                "dpi": dpi,
                "output_format": output_format,
            },
            paths=_path_operands((path,), kind="image_file"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
        languages: Sequence[str] | None = None,
        psm: int = 3,
        oem: int | None = None,
        dpi: int | None = None,
        output_format: Literal["txt"] = "txt",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                path=path,
                languages=languages,
                psm=psm,
                oem=oem,
                dpi=dpi,
                output_format=output_format,
                cwd=_optional_cwd(cwd),
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


def _line_reader_request(
    *,
    command: Literal["head", "tail"],
    path: str,
    lines: int,
    cwd: str | None,
    timeout_seconds: float | None,
    max_stdout_bytes: int | None,
    max_stderr_bytes: int | None,
    byte_count: int | None = None,
    start_line: int | None = None,
    start_byte: int | None = None,
) -> ShellCommandRequest:
    options: dict[str, object] = {"lines": lines}
    if command == "head":
        options["byte_count"] = byte_count
    else:
        options["start_line"] = start_line
        options["byte_count"] = byte_count
        options["start_byte"] = start_byte
    return ShellCommandRequest(
        tool_name=f"shell.{command}",
        command=command,
        options=options,
        paths=_path_operands((path,), kind="text_file"),
        cwd=_optional_cwd(cwd),
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _composition_step_requests(
    steps: Sequence[ShellPipelineStepArgument],
) -> tuple[ShellCommandStepRequest, ...]:
    assert isinstance(steps, Sequence) and not isinstance(
        steps,
        str | bytes | bytearray,
    ), "steps must be a sequence"
    return tuple(
        _composition_step_request(step, index)
        for index, step in enumerate(steps)
    )


def _composition_request_has_safe_commands(
    request: ShellCompositionRequest,
) -> bool:
    return all(
        step.command in SHELL_COMMAND_DEFINITIONS for step in request.steps
    )


def _set_min_length(
    properties: Mapping[str, object],
    key: str,
    length: int,
) -> None:
    field_schema = properties[key]
    assert isinstance(field_schema, dict)
    field_schema["minLength"] = length


def _composition_step_request(
    step: ShellPipelineStepArgument,
    index: int,
) -> ShellCommandStepRequest:
    assert isinstance(step, Mapping), "steps must contain objects"
    options = step.get("options", {})
    assert isinstance(options, dict), "step options must be a dictionary"
    paths = step.get("paths", ())
    cwd = step.get("cwd")
    assert cwd is None or isinstance(cwd, str), "step cwd must be a string"
    return ShellCommandStepRequest(
        id=_required_step_string(step, "id", index),
        command=_required_step_string(step, "command", index),
        options=dict(options),
        paths=_step_paths(paths, index),
        cwd=_optional_cwd(cwd),
        stdin_from=_step_stdin_from(step.get("stdin_from"), index),
    )


def _required_step_string(
    step: Mapping[str, object],
    key: str,
    index: int,
) -> str:
    value = step.get(key)
    assert isinstance(value, str), f"steps[{index}].{key} must be a string"
    return value


def _step_paths(value: object, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    assert isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ), f"steps[{index}].paths must be a sequence"
    return _string_tuple(cast(Sequence[str], value), f"steps[{index}].paths")


def _step_stdin_from(
    value: object,
    index: int,
) -> ShellStreamRef | None:
    if value is None:
        return None
    if isinstance(value, ShellStreamRef):
        return value
    assert isinstance(
        value,
        Mapping,
    ), f"steps[{index}].stdin_from must be an object"
    step_id = value.get("step_id")
    stream = value.get("stream")
    assert isinstance(
        step_id,
        str,
    ), f"steps[{index}].stdin_from.step_id must be a string"
    assert (
        stream == "stdout"
    ), f"steps[{index}].stdin_from.stream must be stdout"
    return ShellStreamRef(step_id=step_id, stream="stdout")


def _optional_cwd(cwd: str | None) -> str | None:
    return None if cwd == "" else cwd


def _path_operands(
    paths: Sequence[str],
    *,
    kind: Literal[
        "any",
        "file",
        "text_file",
        "json_file",
        "pdf_file",
        "image_file",
    ],
) -> tuple[PathOperand, ...]:
    normalized_paths = _string_tuple(paths, "paths")
    return tuple(
        PathOperand(
            name=f"path_{index}",
            path=path,
            kind=kind,
            access="read",
        )
        for index, path in enumerate(normalized_paths)
    )


def _string_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    _assert_string_sequence(value, name)
    return tuple(value)


def _copied_json_schema_properties(
    properties: Mapping[str, object],
) -> dict[str, object]:
    return {
        name: (
            dict(schema_property)
            if isinstance(schema_property, Mapping)
            else schema_property
        )
        for name, schema_property in properties.items()
    }


def _optional_int_tuple(
    value: Sequence[int] | None,
    name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    _assert_int_sequence(value, name)
    return tuple(value)


def _optional_string_tuple(
    value: Sequence[str] | None,
    name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _string_tuple(value, name)


def _policy_denied_result(
    request: ShellCommandRequest,
    error: ShellPolicyDenied,
) -> ExecutionResult:
    return ExecutionResult(
        backend="local",
        tool_name=request.tool_name,
        command=request.command,
        argv=(request.command,),
        display_argv=(request.command,),
        cwd=".",
        display_cwd=".",
        status=ShellExecutionStatus.POLICY_DENIED,
        exit_code=None,
        stdout="",
        stderr="",
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=0,
        error_code=error.error_code,
        error_message=str(error),
        metadata=request.metadata,
    )


def _composition_policy_denied_result(
    request: ShellCompositionRequest,
    error: ShellPolicyDenied,
) -> ShellCompositionResult:
    return ShellCompositionResult(
        mode=request.mode,
        status=ShellExecutionStatus.POLICY_DENIED,
        stdout="",
        stderr="",
        steps=tuple(
            ShellExecutionStepResult(
                id=_safe_policy_denied_step_id(index),
                command=_safe_policy_denied_step_command(step.command),
                status=ShellExecutionStatus.POLICY_DENIED,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_bytes=0,
                stderr_bytes=0,
                stdout_truncated=False,
                stderr_truncated=False,
                duration_ms=0,
                error_code=error.error_code,
                error_message=str(error),
                metadata={
                    "display_cwd": step.cwd or ".",
                    "stdout_visible": index == len(request.steps) - 1,
                },
            )
            for index, step in enumerate(request.steps)
        ),
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=0,
        error_code=error.error_code,
        error_message=str(error),
        metadata={"mode": request.mode},
    )


def _safe_policy_denied_step_command(command: str) -> str:
    if command in SHELL_COMMAND_DEFINITIONS:
        return command
    return REDACTED_DISPLAY_VALUE


def _safe_policy_denied_step_id(index: int) -> str:
    return f"{REDACTED_DISPLAY_VALUE}-{index}"
