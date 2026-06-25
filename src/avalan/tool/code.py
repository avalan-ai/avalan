from ..compat import override
from ..container import (
    ContainerAsyncBackend,
    ContainerBackend,
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendSelection,
    ContainerBackendStream,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerLifecycleDeadlines,
    ContainerManagedLifecycleResult,
    ContainerNormalizedRunPlan,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerResultStatus,
    ContainerStreamDrainPolicy,
    ContainerToolRuntimeSettings,
    normalize_container_run_plan,
    run_container_managed_lifecycle,
    select_container_backend,
)
from ..entities import (
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    ToolValue,
)
from . import Tool, ToolSet
from .builtin_display import (
    project_ast_grep_tool_display,
    project_code_run_tool_display,
)

from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any, cast

try:
    from RestrictedPython import (
        RestrictingNodeTransformer,
        compile_restricted,
        safe_globals,
    )

    HAS_CODE_DEPENDENCIES = True
except ImportError:
    HAS_CODE_DEPENDENCIES = False
    RestrictingNodeTransformer = None
    compile_restricted = None
    safe_globals = None


_AST_GREP_CLEANUP_SECONDS = 5.0
_AST_GREP_COMMAND = "ast-grep"
_AST_GREP_LOGICAL_NAME = "search.ast.grep"
_AST_GREP_MAX_STDERR_BYTES = 32768
_AST_GREP_MAX_STDOUT_BYTES = 65536
_AST_GREP_TIMEOUT_SECONDS = 10.0
_AST_GREP_WORKSPACE = "/workspace"


class CodeTool(Tool):
    """Execute Python code in a restricted environment.

    Args:
        code: Python source that defines a callable named `run`.
        args: Positional arguments forwarded to `run`.
        kwargs: Keyword arguments forwarded to `run`.

    Returns:
        Text representation of the value returned by `run`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "run"

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        return project_code_run_tool_display(call=call, outcome=outcome)

    async def __call__(
        self,
        code: str,
        *args: Any,
        context: ToolCallContext,
        **kwargs: Any,
    ) -> str:
        _ = context
        locals_dict: dict[str, Any] = {}
        byte_code = compile_restricted(
            code,
            filename="<avalan:tool:code>",
            mode="exec",
            flags=0,
            dont_inherit=False,
            policy=RestrictingNodeTransformer,
        )
        exec(byte_code, safe_globals, locals_dict)
        assert "run" in locals_dict

        function = locals_dict["run"]
        positional_args: tuple[Any, ...] = args
        keyword_args: dict[str, Any] = kwargs

        if (
            positional_args
            and not keyword_args
            and len(positional_args) == 2
            and isinstance(positional_args[1], dict)
        ):
            unpacked_args, unpacked_kwargs = positional_args
            if isinstance(unpacked_args, tuple):
                positional_args = unpacked_args
            elif isinstance(unpacked_args, dict):
                positional_args = ()
                unpacked_kwargs = unpacked_args
            keyword_args = unpacked_kwargs

        result = (
            function(*positional_args, **keyword_args)
            if positional_args and keyword_args
            else (
                function(*positional_args)
                if positional_args
                else function(**keyword_args) if keyword_args else function()
            )
        )

        return str(result)


class AstGrepTool(Tool):
    """Search or rewrite code using the ast-grep CLI.

    Args:
        pattern: Code pattern to search for.
        lang: Programming language of the files.
        rewrite: Template used to rewrite matches.
        paths: Files or directories to search.

    Returns:
        Output produced by ast-grep.
    """

    def __init__(
        self,
        *,
        container_settings: ContainerEffectiveSettings | None = None,
        container_backend: ContainerAsyncBackend | None = None,
        container_opt_in_backends: Sequence[ContainerBackend | str] = (),
        container_rootful_authorized: bool = False,
    ) -> None:
        if container_settings is not None:
            assert isinstance(container_settings, ContainerEffectiveSettings)
        if container_backend is not None:
            assert isinstance(container_backend, ContainerAsyncBackend)
        assert isinstance(container_rootful_authorized, bool)
        opt_in_backends = tuple(
            ContainerBackend(backend) for backend in container_opt_in_backends
        )
        super().__init__()
        self.__name__ = "search.ast.grep"
        self._container_settings = container_settings
        self._container_backend = container_backend
        self._container_opt_in_backends = opt_in_backends
        self._container_rootful_authorized = container_rootful_authorized

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        return project_ast_grep_tool_display(call=call, outcome=outcome)

    async def __call__(
        self,
        pattern: str,
        *,
        context: ToolCallContext,
        lang: str,
        rewrite: str | None = None,
        paths: list[str] | None = None,
    ) -> str:
        assert pattern
        assert lang

        if self._container_settings is not None:
            container_result = await self._run_container(
                pattern,
                lang=lang,
                rewrite=rewrite,
                paths=paths,
                context=context,
            )
            if container_result is not None:
                return container_result

        args = _ast_grep_argv(
            pattern,
            lang=lang,
            rewrite=rewrite,
            paths=paths,
        )
        process = await create_subprocess_exec(
            *args,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode() or stdout.decode())
        return stdout.decode()

    async def _run_container(
        self,
        pattern: str,
        *,
        lang: str,
        rewrite: str | None,
        paths: list[str] | None,
        context: ToolCallContext,
    ) -> str | None:
        assert isinstance(context, ToolCallContext)
        start_time = perf_counter()
        try:
            plan = _ast_grep_container_plan(
                pattern,
                lang=lang,
                rewrite=rewrite,
                paths=paths,
                container_settings=self._container_settings,
            )
        except AstGrepContainerExecutionError:
            raise
        except AssertionError as error:
            raise AstGrepContainerExecutionError(
                f"container ast-grep plan is invalid: {error}",
                status="tool_error",
                metadata={"execution_backend": "container"},
            ) from error
        if plan is None:
            return None
        if self._container_backend is None:
            raise AstGrepContainerExecutionError(
                "container execution is selected but no backend is configured",
                status="tool_error",
                metadata=_ast_grep_container_metadata(plan, ()),
            )
        selection = await _select_container_backend(
            plan,
            self._container_backend,
            opt_in_backends=self._container_opt_in_backends,
            rootful_authorized=self._container_rootful_authorized,
        )
        if not selection.ok:
            raise AstGrepContainerExecutionError(
                _diagnostic_summary(selection.diagnostics),
                status="tool_error",
                metadata=_ast_grep_container_metadata(
                    plan,
                    selection.diagnostics,
                ),
            )
        result = await run_container_managed_lifecycle(
            self._container_backend,
            plan.run_plan,
            deadlines=_ast_grep_deadlines(),
            stream_policy=_ast_grep_stream_policy(),
        )
        stdout = _container_stream_capture(
            result,
            ContainerBackendStream.STDOUT,
            _AST_GREP_MAX_STDOUT_BYTES,
        )
        stderr = _container_stream_capture(
            result,
            ContainerBackendStream.STDERR,
            _AST_GREP_MAX_STDERR_BYTES,
        )
        await _emit_container_streams(context, result, stdout, stderr)
        status, message = _ast_grep_container_status(
            result,
            stdout=stdout[0],
            stderr=stderr[0],
        )
        if status is not None:
            raise AstGrepContainerExecutionError(
                message,
                status=status,
                stdout=stdout[0],
                stderr=stderr[0],
                metadata=_ast_grep_container_metadata(
                    plan,
                    result.diagnostics,
                    duration_ms=_duration_ms(start_time),
                    exit_code=result.execution.exit_code,
                ),
            )
        return stdout[0]


class CodeToolSet(ToolSet):
    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
        container_runtime: ContainerToolRuntimeSettings | None = None,
        container_settings: ContainerEffectiveSettings | None = None,
        container_backend: ContainerAsyncBackend | None = None,
        container_opt_in_backends: Sequence[ContainerBackend | str] = (),
        container_rootful_authorized: bool = False,
    ) -> None:
        assert isinstance(container_rootful_authorized, bool)
        if container_runtime is not None:
            assert isinstance(container_runtime, ContainerToolRuntimeSettings)
            container_settings = (
                container_settings or container_runtime.effective_settings
            )
            container_backend = container_backend or container_runtime.backend
            container_opt_in_backends = (
                container_opt_in_backends or container_runtime.opt_in_backends
            )
            container_rootful_authorized = (
                container_rootful_authorized
                or container_runtime.rootful_authorized
            )
        tools = [
            CodeTool(),
            AstGrepTool(
                container_settings=container_settings,
                container_backend=container_backend,
                container_opt_in_backends=container_opt_in_backends,
                container_rootful_authorized=container_rootful_authorized,
            ),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )


class AstGrepContainerExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: str,
        metadata: dict[str, object] | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        assert message
        assert status
        if metadata is not None:
            assert isinstance(metadata, dict)
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)
        super().__init__(message)
        self.message = message
        self.status = status
        self.metadata = metadata or {}
        self.stdout = stdout
        self.stderr = stderr


def _ast_grep_argv(
    pattern: str,
    *,
    lang: str,
    rewrite: str | None = None,
    paths: Sequence[str] | None = None,
    path_separator: bool = False,
) -> tuple[str, ...]:
    assert pattern
    assert lang
    args = [_AST_GREP_COMMAND, "--pattern", pattern, "--lang", lang]
    if rewrite is not None:
        args.extend(["--rewrite", rewrite])
    if paths:
        path_args = tuple(
            _container_path(path) if path_separator else _path_arg(path)
            for path in paths
        )
        if path_separator:
            args.append("--")
        args.extend(path_args)
    return tuple(args)


def _ast_grep_container_plan(
    pattern: str,
    *,
    lang: str,
    rewrite: str | None,
    paths: Sequence[str] | None,
    container_settings: ContainerEffectiveSettings | None,
) -> ContainerNormalizedRunPlan | None:
    if container_settings is None:
        return None
    assert isinstance(container_settings, ContainerEffectiveSettings)
    if not container_settings.enabled:
        if container_settings.required:
            raise AstGrepContainerExecutionError(
                "container execution is required but no profile is enabled",
                status="policy_denied",
                metadata={"execution_backend": "container"},
            )
        return None
    return normalize_container_run_plan(
        container_settings,
        ContainerPlanRequest(
            request_kind=ContainerPlanRequestKind.TYPED_TOOL,
            logical_name=_AST_GREP_LOGICAL_NAME,
            command=_AST_GREP_COMMAND,
            argv=_ast_grep_argv(
                pattern,
                lang=lang,
                rewrite=rewrite,
                paths=paths,
                path_separator=True,
            ),
            cwd=_AST_GREP_WORKSPACE,
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
    )


async def _select_container_backend(
    plan: ContainerNormalizedRunPlan,
    backend: ContainerAsyncBackend,
    *,
    opt_in_backends: Sequence[ContainerBackend | str] = (),
    rootful_authorized: bool,
) -> ContainerBackendSelection:
    probe = await backend.probe()
    return select_container_backend(
        plan.run_plan,
        (probe,),
        rootful_authorized=rootful_authorized,
        opt_in_backends=opt_in_backends,
    )


def _container_path(path: str) -> str:
    path = _path_arg(path)
    assert "\x00" not in path, "paths must not contain null bytes"
    posix_path = PurePosixPath(path)
    assert (
        not posix_path.is_absolute()
    ), "container ast-grep paths must be workspace-relative"
    assert (
        ".." not in posix_path.parts
    ), "container ast-grep paths must not traverse outside the workspace"
    return path


def _path_arg(path: str) -> str:
    assert isinstance(path, str), "paths must be strings"
    assert path, "paths must not be empty"
    return path


def _ast_grep_deadlines() -> ContainerLifecycleDeadlines:
    return ContainerLifecycleDeadlines(
        execution_seconds=_AST_GREP_TIMEOUT_SECONDS,
        parent_seconds=_AST_GREP_TIMEOUT_SECONDS,
        cleanup_seconds=_AST_GREP_CLEANUP_SECONDS,
    )


def _ast_grep_stream_policy() -> ContainerStreamDrainPolicy:
    return ContainerStreamDrainPolicy(
        max_chunks=1024,
        max_bytes=_AST_GREP_MAX_STDOUT_BYTES + _AST_GREP_MAX_STDERR_BYTES + 2,
        max_chunk_bytes=max(
            _AST_GREP_MAX_STDOUT_BYTES + 1,
            _AST_GREP_MAX_STDERR_BYTES + 1,
        ),
        max_stdout_bytes=_AST_GREP_MAX_STDOUT_BYTES + 1,
        max_stderr_bytes=_AST_GREP_MAX_STDERR_BYTES + 1,
        preserve_truncated_prefix=True,
    )


def _container_stream_capture(
    result: ContainerManagedLifecycleResult,
    stream: ContainerBackendStream,
    byte_cap: int,
) -> tuple[str, int, bool]:
    raw = b"".join(
        chunk.content
        for chunk in result.stream.chunks
        if chunk.stream is stream
    )
    content = raw[:byte_cap]
    return (
        content.decode("utf-8", errors="replace"),
        len(content),
        len(raw) > byte_cap,
    )


async def _emit_container_streams(
    context: ToolCallContext,
    result: ContainerManagedLifecycleResult,
    stdout: tuple[str, int, bool],
    stderr: tuple[str, int, bool],
) -> None:
    stream = context.stream_event
    if stream is None:
        return
    stdout_emitted = False
    stderr_emitted = False
    for chunk in result.stream.chunks:
        stream_kind = _stream_kind(cast(ContainerBackendStream, chunk.stream))
        if stream_kind is ToolExecutionStreamKind.STDOUT:
            if stdout_emitted or not stdout[0]:
                continue
            stdout_emitted = True
            content = stdout[0]
            metadata: dict[str, ToolValue] = {
                "backend": "container",
                "truncated": stdout[2],
            }
        elif stream_kind is ToolExecutionStreamKind.STDERR:
            if stderr_emitted or not stderr[0]:
                continue
            stderr_emitted = True
            content = stderr[0]
            metadata = {"backend": "container", "truncated": stderr[2]}
        else:
            content = chunk.content.decode("utf-8", errors="replace")
            metadata = {"backend": "container"}
        await stream(
            ToolExecutionStreamEvent(
                kind=stream_kind,
                content=content,
                metadata=metadata,
            )
        )


def _stream_kind(stream: ContainerBackendStream) -> ToolExecutionStreamKind:
    if stream is ContainerBackendStream.STDOUT:
        return ToolExecutionStreamKind.STDOUT
    if stream is ContainerBackendStream.STDERR:
        return ToolExecutionStreamKind.STDERR
    return ToolExecutionStreamKind.PROGRESS


def _ast_grep_container_status(
    result: ContainerManagedLifecycleResult,
    *,
    stdout: str,
    stderr: str,
) -> tuple[str | None, str]:
    if result.cancelled_phase is not None:
        return "cancelled", "container ast-grep execution cancelled"
    if result.timed_out_phase is not None:
        return "timeout", "container ast-grep execution timed out"
    if result.execution.exit_code is not None and result.execution.exit_code:
        return (
            "nonzero_exit",
            stderr or stdout or "container ast-grep exited non-zero",
        )
    if result.execution.status is ContainerResultStatus.DENIED:
        return "policy_denied", _diagnostic_summary(result.diagnostics)
    if result.execution.status is not ContainerResultStatus.COMPLETED:
        return "tool_error", _diagnostic_summary(result.diagnostics)
    return None, ""


def _ast_grep_container_metadata(
    plan: ContainerNormalizedRunPlan,
    diagnostics: Sequence[ContainerBackendDiagnostic],
    *,
    duration_ms: int | None = None,
    exit_code: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "execution_backend": "container",
        "container_profile": plan.run_plan.profile_name,
        "container_policy_version": plan.run_plan.policy_version,
        "container_plan_fingerprint": plan.plan_fingerprint,
    }
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms
    if exit_code is not None:
        metadata["exit_code"] = exit_code
    if diagnostics:
        metadata["container_diagnostic_codes"] = tuple(
            cast(ContainerBackendDiagnosticCode, diagnostic.code).value
            for diagnostic in diagnostics
        )
    return metadata


def _diagnostic_summary(
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> str:
    if not diagnostics:
        return "container execution failed"
    codes = sorted(
        {
            cast(ContainerBackendDiagnosticCode, diagnostic.code).value
            for diagnostic in diagnostics
        }
    )
    return "container execution failed: " + ", ".join(codes)


def _duration_ms(start_time: float) -> int:
    return max(0, int((perf_counter() - start_time) * 1000))
