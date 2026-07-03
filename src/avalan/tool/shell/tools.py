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
from .git import (
    SHELL_GIT_COMMAND_CAPABILITIES,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitCommandResult,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitFormattedResult,
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


class _ShellGitCommandTool(Tool, ABC):
    _command: ShellGitCommandName
    _settings: ShellToolSettings
    supports_streaming = False

    def __init__(
        self,
        *,
        command: ShellGitCommandName,
        settings: ShellToolSettings,
    ) -> None:
        super().__init__()
        self.__name__ = f"git_{command.value.replace('-', '_')}"
        self._command = command
        self._settings = settings

    async def _execute_request(
        self,
        request: ShellGitCommandRequest,
        *,
        context: ToolCallContext,
    ) -> str:
        result = _git_disabled_result(request)
        return ShellGitFormattedResult(
            _format_shell_git_result(result), result
        )

    def _request(
        self,
        *,
        options: dict[str, object],
        pathspecs: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return ShellGitCommandRequest(
            tool_name=f"shell.{self.__name__}",
            command=self._command,
            capability_required=SHELL_GIT_COMMAND_CAPABILITIES[self._command],
            options=options,
            pathspecs=_string_tuple(pathspecs, "pathspecs"),
            cwd=_optional_cwd(cwd),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    @abstractmethod
    async def __call__(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError


class GitStatusTool(_ShellGitCommandTool):
    """Inspect repository status metadata.

    Args:
        mode: Status output mode to request.
        paths: Repo-relative pathspecs to inspect.
        cwd: Workspace-relative working directory for repository discovery.
        include_branch: Include branch metadata in the status request.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.STATUS, settings=settings)

    def _build_request(
        self,
        mode: Literal["porcelain_v2", "short"] = "porcelain_v2",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        include_branch: bool = True,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "include_branch": include_branch},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["porcelain_v2", "short"] = "porcelain_v2",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        include_branch: bool = True,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                paths=paths,
                cwd=cwd,
                include_branch=include_branch,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRevParseTool(_ShellGitCommandTool):
    """Inspect approved repository and revision facts.

    Args:
        fact: Approved repository or revision fact to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REV_PARSE,
            settings=settings,
        )

    def _build_request(
        self,
        fact: Literal[
            "head",
            "short_head",
            "current_branch",
            "repo_root",
        ] = "head",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"fact": fact},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        fact: Literal[
            "head",
            "short_head",
            "current_branch",
            "repo_root",
        ] = "head",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                fact=fact,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchTool(_ShellGitCommandTool):
    """Inspect current or listed branches.

    Args:
        mode: Branch read mode to request.
        contains: Optional revision used to filter listed branches.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.BRANCH, settings=settings)

    def _build_request(
        self,
        mode: Literal["current", "list"] = "current",
        contains: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "contains": contains},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["current", "list"] = "current",
        contains: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                contains=contains,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitTagTool(_ShellGitCommandTool):
    """Inspect listed or shown tags.

    Args:
        mode: Tag read mode to request.
        name: Optional tag name for show mode.
        max_count: Optional maximum number of tags to return.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.TAG, settings=settings)

    def _build_request(
        self,
        mode: Literal["list", "show"] = "list",
        name: str | None = None,
        max_count: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "name": name, "max_count": max_count},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["list", "show"] = "list",
        name: str | None = None,
        max_count: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                name=name,
                max_count=max_count,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitDescribeTool(_ShellGitCommandTool):
    """Inspect bounded describe metadata.

    Args:
        target: Optional revision or ref to describe.
        mode: Describe mode to request.
        max_candidates: Maximum tag candidates to consider.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.DESCRIBE,
            settings=settings,
        )

    def _build_request(
        self,
        target: str | None = None,
        mode: Literal["tags", "always"] = "tags",
        max_candidates: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "target": target,
                "mode": mode,
                "max_candidates": max_candidates,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        target: str | None = None,
        mode: Literal["tags", "always"] = "tags",
        max_candidates: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                target=target,
                mode=mode,
                max_candidates=max_candidates,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitLsFilesTool(_ShellGitCommandTool):
    """List repository paths with safe modes.

    Args:
        mode: File listing mode to request.
        paths: Repo-relative pathspecs to list.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.LS_FILES,
            settings=settings,
        )

    def _build_request(
        self,
        mode: Literal["tracked", "modified", "deleted", "others"] = "tracked",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["tracked", "modified", "deleted", "others"] = "tracked",
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitLogTool(_ShellGitCommandTool):
    """Inspect bounded history summaries.

    Args:
        max_count: Maximum number of commits to return.
        revision: Optional revision range to inspect.
        paths: Repo-relative pathspecs to filter history.
        format: Fixed history format to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.LOG, settings=settings)

    def _build_request(
        self,
        max_count: int = 10,
        revision: str | None = None,
        paths: Sequence[str] = (),
        format: Literal["summary", "oneline"] = "summary",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "max_count": max_count,
                "revision": revision,
                "format": format,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        max_count: int = 10,
        revision: str | None = None,
        paths: Sequence[str] = (),
        format: Literal["summary", "oneline"] = "summary",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                max_count=max_count,
                revision=revision,
                paths=paths,
                format=format,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitDiffTool(_ShellGitCommandTool):
    """Inspect bounded repository diffs.

    Args:
        mode: Diff mode to request.
        base_revision: Optional base revision for range mode.
        head_revision: Optional head revision for range mode.
        paths: Repo-relative pathspecs to diff.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.DIFF, settings=settings)

    def _build_request(
        self,
        mode: Literal[
            "worktree", "staged", "range", "stat", "name_only"
        ] = "worktree",
        base_revision: str | None = None,
        head_revision: str | None = None,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "mode": mode,
                "base_revision": base_revision,
                "head_revision": head_revision,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal[
            "worktree", "staged", "range", "stat", "name_only"
        ] = "worktree",
        base_revision: str | None = None,
        head_revision: str | None = None,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                base_revision=base_revision,
                head_revision=head_revision,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitShowTool(_ShellGitCommandTool):
    """Inspect bounded commit or tag details.

    Args:
        revision: Revision or tag to inspect.
        mode: Fixed show mode to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.SHOW, settings=settings)

    def _build_request(
        self,
        revision: str,
        mode: Literal["summary", "stat", "patch"] = "summary",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"revision": revision, "mode": mode},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        mode: Literal["summary", "stat", "patch"] = "summary",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                mode=mode,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBlameTool(_ShellGitCommandTool):
    """Inspect bounded line blame for one file.

    Args:
        path: Repo-relative file path to inspect.
        start_line: Optional first line to inspect.
        end_line: Optional final line to inspect.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.BLAME, settings=settings)

    def _build_request(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"start_line": start_line, "end_line": end_line},
            pathspecs=(path,),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        path: str,
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
                path=path,
                start_line=start_line,
                end_line=end_line,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitGrepTool(_ShellGitCommandTool):
    """Search repository content with bounded grep.

    Args:
        pattern: Search pattern to request.
        paths: Repo-relative pathspecs to search.
        case: Case-sensitivity mode.
        max_matches: Maximum total matches to return.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.GREP, settings=settings)

    def _build_request(
        self,
        pattern: str,
        paths: Sequence[str] = (),
        case: Literal["sensitive", "insensitive"] = "sensitive",
        max_matches: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "pattern": pattern,
                "case": case,
                "max_matches": max_matches,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        pattern: str,
        paths: Sequence[str] = (),
        case: Literal["sensitive", "insensitive"] = "sensitive",
        max_matches: int | None = None,
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
                paths=paths,
                case=case,
                max_matches=max_matches,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashListTool(_ShellGitCommandTool):
    """Inspect bounded stash metadata.

    Args:
        max_count: Maximum number of stash entries to return.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_LIST,
            settings=settings,
        )

    def _build_request(
        self,
        max_count: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"max_count": max_count},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        max_count: int = 10,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                max_count=max_count,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashShowTool(_ShellGitCommandTool):
    """Inspect bounded stash details.

    Args:
        stash: Stash reference to inspect.
        mode: Fixed stash show mode to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_SHOW,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        mode: Literal["stat", "patch"] = "stat",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash, "mode": mode},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        mode: Literal["stat", "patch"] = "stat",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                mode=mode,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitAddTool(_ShellGitCommandTool):
    """Stage repo-relative paths for addition.

    Args:
        paths: Repo-relative paths to stage.
        mode: Add mode to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.ADD, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str],
        mode: Literal["normal", "intent_to_add"] = "normal",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        mode: Literal["normal", "intent_to_add"] = "normal",
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
                mode=mode,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRestoreTool(_ShellGitCommandTool):
    """Restore repo-relative paths from a safe source.

    Args:
        paths: Repo-relative paths to restore.
        source_revision: Optional source revision.
        staged: Restore index state.
        worktree: Restore worktree state.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.RESTORE,
            settings=settings,
        )

    def _build_request(
        self,
        paths: Sequence[str],
        source_revision: str | None = None,
        staged: bool = False,
        worktree: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "source_revision": source_revision,
                "staged": staged,
                "worktree": worktree,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        source_revision: str | None = None,
        staged: bool = False,
        worktree: bool = True,
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
                source_revision=source_revision,
                staged=staged,
                worktree=worktree,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCheckoutTool(_ShellGitCommandTool):
    """Checkout constrained paths or branch targets.

    Args:
        mode: Checkout mode to request.
        target: Optional branch or revision target.
        paths: Repo-relative paths for path checkout.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.CHECKOUT,
            settings=settings,
        )

    def _build_request(
        self,
        mode: Literal["paths", "branch"],
        target: str | None = None,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": mode, "target": target},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        mode: Literal["paths", "branch"],
        target: str | None = None,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                mode=mode,
                target=target,
                paths=paths,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitSwitchTool(_ShellGitCommandTool):
    """Switch to a constrained branch target.

    Args:
        branch: Branch name to switch to.
        create: Create the branch before switching.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.SWITCH, settings=settings)

    def _build_request(
        self,
        branch: str,
        create: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"branch": branch, "create": create},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        branch: str,
        create: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                branch=branch,
                create=create,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitResetTool(_ShellGitCommandTool):
    """Reset constrained worktree paths.

    Args:
        paths: Repo-relative paths to reset.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.RESET, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"mode": "paths"},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str] = (),
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
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRmTool(_ShellGitCommandTool):
    """Remove repo-relative paths from the index.

    Args:
        paths: Repo-relative paths to remove.
        cached: Remove only from the index.
        recursive: Permit recursive directory removal.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.RM, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str],
        cached: bool = False,
        recursive: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"cached": cached, "recursive": recursive},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        cached: bool = False,
        recursive: bool = False,
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
                cached=cached,
                recursive=recursive,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitMvTool(_ShellGitCommandTool):
    """Move one repo-relative path to another.

    Args:
        source: Repo-relative source path.
        destination: Repo-relative destination path.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.MV, settings=settings)

    def _build_request(
        self,
        source: str,
        destination: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"source": source, "destination": destination},
            pathspecs=(source, destination),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        source: str,
        destination: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                source=source,
                destination=destination,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashPushTool(_ShellGitCommandTool):
    """Create a bounded stash entry.

    Args:
        message: Optional stash message.
        paths: Repo-relative paths to include.
        include_untracked: Include untracked files in the stash request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_PUSH,
            settings=settings,
        )

    def _build_request(
        self,
        message: str | None = None,
        paths: Sequence[str] = (),
        include_untracked: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "message": message,
                "include_untracked": include_untracked,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        message: str | None = None,
        paths: Sequence[str] = (),
        include_untracked: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                message=message,
                paths=paths,
                include_untracked=include_untracked,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashApplyTool(_ShellGitCommandTool):
    """Apply a bounded stash entry.

    Args:
        stash: Stash reference to apply.
        index: Restore index state from the stash.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_APPLY,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        index: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash, "index": index},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        index: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                index=index,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCommitTool(_ShellGitCommandTool):
    """Create a commit from the existing index.

    Args:
        message: Commit message to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.COMMIT, settings=settings)

    def _build_request(
        self,
        message: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"message": message},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        message: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                message=message,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchCreateTool(_ShellGitCommandTool):
    """Create a branch from an approved start point.

    Args:
        name: Branch name to create.
        start_point: Optional start point revision.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.BRANCH_CREATE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        start_point: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "start_point": start_point},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        start_point: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                start_point=start_point,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchDeleteTool(_ShellGitCommandTool):
    """Delete a constrained branch name.

    Args:
        name: Branch name to delete.
        force: Request force deletion.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.BRANCH_DELETE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        force: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "force": force},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        force: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                force=force,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitBranchRenameTool(_ShellGitCommandTool):
    """Rename one constrained branch to another.

    Args:
        old_name: Existing branch name.
        new_name: New branch name.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.BRANCH_RENAME,
            settings=settings,
        )

    def _build_request(
        self,
        old_name: str,
        new_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"old_name": old_name, "new_name": new_name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        old_name: str,
        new_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                old_name=old_name,
                new_name=new_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitTagCreateTool(_ShellGitCommandTool):
    """Create a constrained tag reference.

    Args:
        name: Tag name to create.
        target: Optional target revision.
        message: Optional tag annotation message.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.TAG_CREATE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        target: str | None = None,
        message: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "target": target, "message": message},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        target: str | None = None,
        message: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                target=target,
                message=message,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitTagDeleteTool(_ShellGitCommandTool):
    """Delete a constrained tag reference.

    Args:
        name: Tag name to delete.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.TAG_DELETE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitMergeTool(_ShellGitCommandTool):
    """Merge a constrained revision.

    Args:
        revision: Revision to merge.
        mode: Merge mode to request.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.MERGE, settings=settings)

    def _build_request(
        self,
        revision: str,
        mode: Literal["ff_only", "no_ff"] = "ff_only",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"revision": revision, "mode": mode},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        mode: Literal["ff_only", "no_ff"] = "ff_only",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                mode=mode,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRebaseTool(_ShellGitCommandTool):
    """Rebase onto a constrained upstream.

    Args:
        upstream: Upstream revision to rebase onto.
        branch: Optional branch to rebase.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.REBASE, settings=settings)

    def _build_request(
        self,
        upstream: str,
        branch: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"upstream": upstream, "branch": branch},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        upstream: str,
        branch: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                upstream=upstream,
                branch=branch,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCherryPickTool(_ShellGitCommandTool):
    """Cherry-pick a constrained revision.

    Args:
        revision: Revision to cherry-pick.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.CHERRY_PICK,
            settings=settings,
        )

    def _build_request(
        self,
        revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"revision": revision},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRevertTool(_ShellGitCommandTool):
    """Revert a constrained revision.

    Args:
        revision: Revision to revert.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.REVERT, settings=settings)

    def _build_request(
        self,
        revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"revision": revision},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        revision: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                revision=revision,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCleanTool(_ShellGitCommandTool):
    """Clean constrained untracked paths.

    Args:
        paths: Repo-relative paths to clean.
        directories: Include directories in the clean request.
        ignored: Include ignored files in the clean request.
        dry_run: Request dry-run output only.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.CLEAN, settings=settings)

    def _build_request(
        self,
        paths: Sequence[str] = (),
        directories: bool = False,
        ignored: bool = False,
        dry_run: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "directories": directories,
                "ignored": ignored,
                "dry_run": dry_run,
            },
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str] = (),
        directories: bool = False,
        ignored: bool = False,
        dry_run: bool = True,
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
                directories=directories,
                ignored=ignored,
                dry_run=dry_run,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashPopTool(_ShellGitCommandTool):
    """Pop a bounded stash entry.

    Args:
        stash: Stash reference to pop.
        index: Restore index state from the stash.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_POP,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        index: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash, "index": index},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        index: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                index=index,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitStashDropTool(_ShellGitCommandTool):
    """Drop a bounded stash entry.

    Args:
        stash: Stash reference to drop.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.STASH_DROP,
            settings=settings,
        )

    def _build_request(
        self,
        stash: str = "stash@{0}",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"stash": stash},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        stash: str = "stash@{0}",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                stash=stash,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitFetchTool(_ShellGitCommandTool):
    """Fetch bounded remote refs.

    Args:
        remote: Remote name to fetch from.
        refspecs: Bounded refspecs to fetch.
        prune: Request stale remote ref pruning.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.FETCH, settings=settings)

    def _build_request(
        self,
        remote: str = "origin",
        refspecs: Sequence[str] = (),
        prune: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "remote": remote,
                "refspecs": _string_tuple(refspecs, "refspecs"),
                "prune": prune,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        remote: str = "origin",
        refspecs: Sequence[str] = (),
        prune: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                remote=remote,
                refspecs=refspecs,
                prune=prune,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitPullTool(_ShellGitCommandTool):
    """Pull bounded remote refs.

    Args:
        remote: Remote name to pull from.
        branch: Optional branch name to pull.
        ff_only: Request fast-forward-only pull behavior.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.PULL, settings=settings)

    def _build_request(
        self,
        remote: str = "origin",
        branch: str | None = None,
        ff_only: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"remote": remote, "branch": branch, "ff_only": ff_only},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        remote: str = "origin",
        branch: str | None = None,
        ff_only: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                remote=remote,
                branch=branch,
                ff_only=ff_only,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitPushTool(_ShellGitCommandTool):
    """Push bounded remote refs.

    Args:
        remote: Remote name to push to.
        refspec: Optional bounded refspec to push.
        set_upstream: Set upstream for the pushed branch.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.PUSH, settings=settings)

    def _build_request(
        self,
        remote: str = "origin",
        refspec: str | None = None,
        set_upstream: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={
                "remote": remote,
                "refspec": refspec,
                "set_upstream": set_upstream,
            },
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        remote: str = "origin",
        refspec: str | None = None,
        set_upstream: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                remote=remote,
                refspec=refspec,
                set_upstream=set_upstream,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitCloneTool(_ShellGitCommandTool):
    """Clone a remote repository into the workspace.

    Args:
        url: Remote URL to clone.
        destination: Workspace-relative clone destination.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.CLONE, settings=settings)

    def _build_request(
        self,
        url: str,
        destination: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"url": url, "destination": destination},
            pathspecs=(destination,),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        url: str,
        destination: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                url=url,
                destination=destination,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteListTool(_ShellGitCommandTool):
    """List approved remote entries.

    Args:
        redact_urls: Redact remote URLs in formatted output.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_LIST,
            settings=settings,
        )

    def _build_request(
        self,
        redact_urls: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"redact_urls": redact_urls},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        redact_urls: bool = True,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                redact_urls=redact_urls,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteAddTool(_ShellGitCommandTool):
    """Add an approved remote entry.

    Args:
        name: Remote name to add.
        url: Approved remote URL to add.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_ADD,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        url: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "url": url},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        url: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                url=url,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteSetUrlTool(_ShellGitCommandTool):
    """Set an approved remote URL.

    Args:
        name: Remote name to update.
        url: Approved remote URL to store.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_SET_URL,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        url: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name, "url": url},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        url: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                url=url,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteRemoveTool(_ShellGitCommandTool):
    """Remove an approved remote entry.

    Args:
        name: Remote name to remove.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_REMOVE,
            settings=settings,
        )

    def _build_request(
        self,
        name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"name": name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                name=name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitRemoteRenameTool(_ShellGitCommandTool):
    """Rename one approved remote entry to another.

    Args:
        old_name: Existing remote name.
        new_name: New remote name.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.REMOTE_RENAME,
            settings=settings,
        )

    def _build_request(
        self,
        old_name: str,
        new_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"old_name": old_name, "new_name": new_name},
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        old_name: str,
        new_name: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            self._build_request(
                old_name=old_name,
                new_name=new_name,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class GitSubmoduleUpdateTool(_ShellGitCommandTool):
    """Update gated submodule paths.

    Args:
        paths: Repo-relative submodule paths to update.
        init: Initialize submodules before updating.
        recursive: Request recursive submodule updates.
        cwd: Workspace-relative working directory for repository discovery.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Stable non-executing shell Git result.
    """

    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(
            command=ShellGitCommandName.SUBMODULE_UPDATE,
            settings=settings,
        )

    def _build_request(
        self,
        paths: Sequence[str] = (),
        init: bool = False,
        recursive: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
    ) -> ShellGitCommandRequest:
        return self._request(
            options={"init": init, "recursive": recursive},
            pathspecs=paths,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )

    async def __call__(
        self,
        paths: Sequence[str] = (),
        init: bool = False,
        recursive: bool = False,
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
                init=init,
                recursive=recursive,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
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


def _git_disabled_result(
    request: ShellGitCommandRequest,
) -> ShellGitCommandResult:
    display_argv = ("git", request.command.value)
    return ShellGitCommandResult(
        tool_name=request.tool_name,
        command=request.command,
        display_argv=display_argv,
        effective_cwd=request.cwd or ".",
        resolved_repo_root=None,
        capability_required=request.capability_required,
        capability_used=None,
        execution_mode="disabled",
        status=ShellGitExecutionStatus.POLICY_DENIED,
        exit_code=None,
        stdout_snippet="",
        stderr_snippet="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=0,
        error_code=ShellGitExecutionErrorCode.COMMAND_DISABLED,
        error_message="shell Git execution is disabled in Phase 1",
        audit_metadata={"request_options": request.options},
    )


def _format_shell_git_result(result: ShellGitCommandResult) -> str:
    error_code = result.error_code.value if result.error_code else None
    capability_used = (
        result.capability_used.value if result.capability_used else None
    )
    return "\n".join(
        [
            f"tool: {result.tool_name}",
            f"status: {result.status.value}",
            f"git_command: {result.command.value}",
            f"command: {' '.join(result.display_argv)}",
            f"cwd: {result.effective_cwd}",
            f"repo_root: {_scalar_text(result.resolved_repo_root)}",
            f"capability_required: {result.capability_required.value}",
            f"capability_used: {_scalar_text(capability_used)}",
            f"execution_mode: {result.execution_mode}",
            f"exit_code: {_scalar_text(result.exit_code)}",
            f"error_code: {_scalar_text(error_code)}",
            f"error_message: {_scalar_text(result.error_message)}",
            f"timed_out: {_bool_text(result.timed_out)}",
            f"cancelled: {_bool_text(result.cancelled)}",
            f"duration_ms: {result.duration_ms}",
            f"stdout_bytes: {result.stdout_bytes}",
            f"stderr_bytes: {result.stderr_bytes}",
            f"stdout_truncated: {_bool_text(result.stdout_truncated)}",
            f"stderr_truncated: {_bool_text(result.stderr_truncated)}",
            "",
            "stdout:",
            result.stdout_snippet,
            "",
            "stderr:",
            result.stderr_snippet,
        ]
    )


def _scalar_text(value: object) -> str:
    return "null" if value is None else str(value)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


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
