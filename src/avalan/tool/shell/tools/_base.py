from ....entities import ToolCall, ToolCallContext, ToolCallOutcome
from ... import Tool
from ..display import project_shell_tool_display
from ..entities import (
    ExecutionResult,
    ShellCommandRequest,
    ShellFormattedResult,
    ShellPolicyDenied,
)
from ..executor import CommandExecutor
from ..formatting import (
    format_shell_result,
)
from ..lsof import LSOF_DEFAULT_LIMIT
from ..policy import ExecutionPolicy
from ..settings import ShellToolSettings
from ._results import (
    _kill_public_result,
    _lsof_public_result,
    _lsof_requested_limit,
    _lsof_requested_pid,
    _pgrep_public_result,
    _policy_denied_result,
    _ps_public_result,
    _ps_requested_pids,
    _ps_requested_view,
)

from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import signature

ShellResultFormatter = Callable[[ExecutionResult], str]


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
            if request.command == "lsof":
                result = _lsof_public_result(
                    result,
                    backend=result.backend,
                    tool_name=result.tool_name,
                    command=result.command,
                    display_argv=result.display_argv,
                    cwd=result.cwd,
                    display_cwd=result.display_cwd,
                    stdout_media_type=result.stdout_media_type,
                    output_kind=result.output_kind,
                    requested_pid=None,
                    limit=LSOF_DEFAULT_LIMIT,
                    max_stdout_bytes=0,
                )
            elif request.command == "ps":
                result = _ps_public_result(
                    result,
                    backend=result.backend,
                    tool_name=result.tool_name,
                    command=result.command,
                    display_argv=result.display_argv,
                    cwd=result.cwd,
                    display_cwd=result.display_cwd,
                    stdout_media_type=result.stdout_media_type,
                    output_kind=result.output_kind,
                    requested_pids=(),
                    view="summary",
                )
            elif request.command == "kill":
                result = _kill_public_result(
                    result,
                    backend=result.backend,
                    tool_name=result.tool_name,
                    command=result.command,
                    display_argv=result.display_argv,
                    cwd=result.cwd,
                    display_cwd=result.display_cwd,
                    stdout_media_type=result.stdout_media_type,
                    output_kind=result.output_kind,
                )
            return ShellFormattedResult(self._formatter(result), result)
        if self.supports_streaming and context.stream_event is not None:
            result = await self._executor.execute(
                spec,
                stream=context.stream_event,
            )
        else:
            result = await self._executor.execute(spec)
        if spec.command == "lsof":
            result = _lsof_public_result(
                result,
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                stdout_media_type=spec.stdout_media_type,
                output_kind=spec.output_kind,
                requested_pid=_lsof_requested_pid(spec.metadata),
                limit=_lsof_requested_limit(spec.metadata),
                max_stdout_bytes=spec.max_stdout_bytes,
            )
        elif spec.command == "pgrep":
            result = _pgrep_public_result(
                result,
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                stdout_media_type=spec.stdout_media_type,
                output_kind=spec.output_kind,
            )
        elif spec.command == "ps":
            result = _ps_public_result(
                result,
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                stdout_media_type=spec.stdout_media_type,
                output_kind=spec.output_kind,
                requested_pids=_ps_requested_pids(spec.metadata),
                view=_ps_requested_view(spec.metadata),
            )
        elif spec.command == "kill":
            result = _kill_public_result(
                result,
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                stdout_media_type=spec.stdout_media_type,
                output_kind=spec.output_kind,
            )
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
