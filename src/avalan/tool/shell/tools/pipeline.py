from ....entities import ToolCall, ToolCallContext, ToolCallOutcome
from ... import Tool
from ...display import REDACTED_DISPLAY_VALUE
from ..composition_executor import CompositionExecutor
from ..display import project_shell_tool_display
from ..entities import (
    ShellCommandStepRequest,
    ShellCompositionMode,
    ShellCompositionRequest,
    ShellCompositionResult,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellFormattedCompositionResult,
    ShellPolicyDenied,
    ShellStreamRef,
)
from ..formatting import (
    format_shell_composition_result,
)
from ..policy import ExecutionPolicy
from ..registry import SHELL_COMMAND_DEFINITIONS
from ..settings import ShellToolSettings
from ._arguments import _optional_cwd, _string_tuple

from collections.abc import Callable, Mapping, Sequence
from inspect import signature
from typing import Any, Literal, TypeAlias, TypedDict, cast

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
    message = f"steps[{index}].stdin_from.stream must be stdout"
    assert stream == "stdout", message
    return ShellStreamRef(step_id=step_id, stream="stdout")


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
