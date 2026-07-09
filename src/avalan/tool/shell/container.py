from ...container import (
    ContainerAsyncBackend,
    ContainerBackend,
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendSelection,
    ContainerBackendStream,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerLifecycleDeadlines,
    ContainerManagedLifecycleResult,
    ContainerNormalizedRunPlan,
    ContainerOutputArtifact,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputDiagnosticCode,
    ContainerOutputMediaPolicy,
    ContainerOutputValidationResult,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerResultStatus,
    ContainerStreamDrainPolicy,
    normalize_container_run_plan,
    run_container_managed_lifecycle,
    select_container_backend,
)
from ...entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    ToolValue,
)
from ...isolation import (
    SandboxEffectiveSettings,
    SandboxOutputPolicy,
    SandboxResourceLimits,
    isolation_diagnostic_codes,
)
from ...sandbox import (
    SandboxExecutionPlan,
    SandboxPlanRequest,
    SandboxPlanRequestKind,
)
from ...types import (
    assert_bool as _assert_bool,
)
from .entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
)
from .executor import (
    CommandExecutor,
    LocalCommandExecutor,
    _status_for_exit_code,
)
from .policy import ExecutionPolicy

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath
from time import perf_counter
from typing import cast, final

_CONTAINER_OUTPUT_ROOT = "/outputs"
_DEFAULT_CLEANUP_SECONDS = 5.0
_MAX_PROGRESS_STREAM_BYTES = 4096
_MAX_PROGRESS_STREAM_CHUNKS = 16


class _ContainerGeneratedOutputError(Exception):
    pass


class ShellExecutionMode(StrEnum):
    LOCAL = "local"
    SANDBOX = "sandbox"
    CONTAINER = "container"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellExecutionPlan:
    mode: ShellExecutionMode | str
    local_spec: ExecutionSpec
    sandbox_plan: SandboxExecutionPlan | None = None
    container_plan: ContainerNormalizedRunPlan | None = None
    output_contract: ContainerOutputContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ShellExecutionMode, "mode"),
        )
        assert isinstance(self.local_spec, ExecutionSpec)
        if self.sandbox_plan is not None:
            assert isinstance(self.sandbox_plan, SandboxExecutionPlan)
        if self.container_plan is not None:
            assert isinstance(self.container_plan, ContainerNormalizedRunPlan)
        if self.output_contract is not None:
            assert isinstance(self.output_contract, ContainerOutputContract)
        if self.mode is ShellExecutionMode.LOCAL:
            assert (
                self.sandbox_plan is None
            ), "local shell plans cannot carry sandbox plans"
            assert (
                self.container_plan is None
            ), "local shell plans cannot carry container plans"
        if self.mode is ShellExecutionMode.SANDBOX:
            assert (
                self.sandbox_plan is not None
            ), "sandbox shell plans require a sandbox plan"
            assert (
                self.container_plan is None
            ), "sandbox shell plans cannot carry container plans"
        if self.mode is ShellExecutionMode.CONTAINER:
            assert (
                self.container_plan is not None
            ), "container shell plans require a container plan"
            assert (
                self.sandbox_plan is None
            ), "container shell plans cannot carry sandbox plans"

    def to_dict(self) -> dict[str, object]:
        mode = cast(ShellExecutionMode, self.mode)
        return {
            "mode": mode.value,
            "local": {
                "tool_name": self.local_spec.tool_name,
                "command": self.local_spec.command,
                "display_argv": list(self.local_spec.display_argv),
                "display_cwd": self.local_spec.display_cwd,
                "timeout_seconds": self.local_spec.timeout_seconds,
            },
            "container": (
                None
                if self.container_plan is None
                else self.container_plan.to_dict()
            ),
            "sandbox": (
                None
                if self.sandbox_plan is None
                else self.sandbox_plan.to_dict()
            ),
            "output_contract": (
                None
                if self.output_contract is None
                else self.output_contract.to_dict()
            ),
        }


async def normalize_shell_execution_request(
    request: ShellCommandRequest,
    policy: ExecutionPolicy,
    *,
    sandbox_settings: SandboxEffectiveSettings | None = None,
    container_settings: ContainerEffectiveSettings | None = None,
) -> ShellExecutionPlan:
    assert isinstance(request, ShellCommandRequest)
    assert isinstance(policy, ExecutionPolicy)
    local_spec = await policy.normalize(request)
    return lower_shell_execution_spec(
        local_spec,
        sandbox_settings=sandbox_settings,
        container_settings=container_settings,
    )


def lower_shell_execution_spec(
    spec: ExecutionSpec,
    *,
    sandbox_settings: SandboxEffectiveSettings | None = None,
    sandbox_output_dir: str | None = None,
    container_settings: ContainerEffectiveSettings | None = None,
) -> ShellExecutionPlan:
    assert isinstance(spec, ExecutionSpec)
    assert not (
        sandbox_settings is not None and container_settings is not None
    ), "shell execution cannot mix sandbox and container settings"
    if spec.backend == ShellExecutionMode.LOCAL.value:
        assert sandbox_settings is None, "local shell plans cannot use sandbox"
        assert (
            sandbox_output_dir is None
        ), "local shell plans cannot use sandbox output dirs"
        assert (
            container_settings is None
        ), "local shell plans cannot use container settings"
        return ShellExecutionPlan(
            mode=ShellExecutionMode.LOCAL,
            local_spec=spec,
        )
    elif spec.backend == ShellExecutionMode.SANDBOX.value:
        assert (
            container_settings is None
        ), "sandbox shell plans cannot use container settings"
        assert (
            sandbox_settings is not None
        ), "sandbox execution is selected but no settings are configured"
    else:
        assert spec.backend == ShellExecutionMode.CONTAINER.value
        assert (
            sandbox_settings is None
        ), "container shell plans cannot use sandbox settings"
        assert (
            sandbox_output_dir is None
        ), "container shell plans cannot use sandbox output dirs"
        assert (
            container_settings is not None
        ), "container execution is selected but no settings are configured"
    if sandbox_settings is not None:
        assert isinstance(sandbox_settings, SandboxEffectiveSettings)
        return ShellExecutionPlan(
            mode=ShellExecutionMode.SANDBOX,
            local_spec=spec,
            sandbox_plan=_sandbox_plan_from_spec(
                spec,
                sandbox_settings,
                output_dir=sandbox_output_dir,
            ),
        )
    assert isinstance(container_settings, ContainerEffectiveSettings)
    if not container_settings.enabled:
        if container_settings.required:
            raise ShellPolicyDenied(
                ShellExecutionErrorCode.POLICY_DENIED,
                "container execution is required but no profile is enabled",
            )
        raise ShellPolicyDenied(
            ShellExecutionErrorCode.POLICY_DENIED,
            "container execution is selected but no profile is enabled",
        )
    container_plan = normalize_container_run_plan(
        container_settings,
        _container_request_from_spec(spec),
    )
    return ShellExecutionPlan(
        mode=ShellExecutionMode.CONTAINER,
        local_spec=spec,
        container_plan=container_plan,
        output_contract=_output_contract_from_spec(spec),
    )


def _sandbox_plan_from_spec(
    spec: ExecutionSpec,
    sandbox_settings: SandboxEffectiveSettings,
    *,
    output_dir: str | None = None,
) -> SandboxExecutionPlan:
    assert spec.executable is not None, "sandbox execution requires executable"
    effective_settings = _narrow_sandbox_settings(spec, sandbox_settings)
    output_dir = _sandbox_output_dir(
        spec,
        effective_settings,
        output_dir=output_dir,
    )
    return SandboxExecutionPlan(
        request=SandboxPlanRequest(
            request_kind=SandboxPlanRequestKind.TYPED_TOOL,
            logical_name=spec.tool_name,
            command=spec.executable,
            argv=_sandbox_argv(spec, output_dir),
            cwd=spec.cwd,
            request_id=_optional_metadata_string(spec, "tool_call_id"),
        ),
        settings=effective_settings,
        environment=_sandbox_environment(spec, effective_settings),
        temp_dir=None,
        output_dir=output_dir,
        collect_outputs=spec.output_kind is ShellOutputKind.GENERATED_FILES,
        cleanup_budget_seconds=_DEFAULT_CLEANUP_SECONDS,
        stream_buffer_bytes=max(
            1,
            spec.max_stdout_bytes,
            spec.max_stderr_bytes,
        ),
    )


def _narrow_sandbox_settings(
    spec: ExecutionSpec,
    sandbox_settings: SandboxEffectiveSettings,
) -> SandboxEffectiveSettings:
    profile = sandbox_settings.profile
    resources = replace(
        profile.resources,
        timeout_seconds=_sandbox_timeout_seconds(spec, profile.resources),
    )
    output = _sandbox_output_policy(spec, profile.output)
    narrowed_profile = replace(
        profile,
        resources=resources,
        output=output,
    )
    return replace(sandbox_settings, profile=narrowed_profile)


def _sandbox_timeout_seconds(
    spec: ExecutionSpec,
    resources: SandboxResourceLimits,
) -> int:
    timeout = max(1, int(spec.timeout_seconds))
    if resources.timeout_seconds is None:
        return timeout
    return min(resources.timeout_seconds, timeout)


def _sandbox_output_policy(
    spec: ExecutionSpec,
    output: SandboxOutputPolicy,
) -> SandboxOutputPolicy:
    max_stdout_bytes = max(
        1, min(output.max_stdout_bytes, spec.max_stdout_bytes)
    )
    max_stderr_bytes = max(
        1, min(output.max_stderr_bytes, spec.max_stderr_bytes)
    )
    if spec.output_kind is not ShellOutputKind.GENERATED_FILES:
        return replace(
            output,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )
    assert spec.output_plan is not None, "generated outputs require a plan"
    max_artifact_bytes = min(
        output.max_artifact_bytes,
        spec.output_plan.max_total_bytes,
    )
    return replace(
        output,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        max_artifact_bytes=max_artifact_bytes,
    )


def _sandbox_output_dir(
    spec: ExecutionSpec,
    sandbox_settings: SandboxEffectiveSettings,
    *,
    output_dir: str | None = None,
) -> str | None:
    if spec.output_kind is not ShellOutputKind.GENERATED_FILES:
        assert (
            output_dir is None
        ), "sandbox output dirs require generated outputs"
        return None
    assert spec.output_plan is not None, "generated outputs require a plan"
    if output_dir is not None:
        return output_dir
    output_roots = sandbox_settings.profile.output_roots
    assert output_roots, "sandbox generated outputs require an output root"
    return output_roots[0]


def _sandbox_argv(
    spec: ExecutionSpec,
    output_dir: str | None,
) -> tuple[str, ...]:
    assert spec.executable is not None, "sandbox execution requires executable"
    argv_tail = spec.argv[1:]
    if spec.output_kind is ShellOutputKind.GENERATED_FILES:
        assert spec.output_plan is not None, "generated outputs require a plan"
        assert output_dir is not None, "generated outputs require output_dir"
        output_prefix = PurePosixPath(
            output_dir,
            spec.output_plan.prefix_name,
        ).as_posix()
        argv_tail = tuple(
            (
                output_prefix
                if argument == GENERATED_OUTPUT_PREFIX_PLACEHOLDER
                else argument
            )
            for argument in argv_tail
        )
    return (spec.executable, *argv_tail)


def _sandbox_environment(
    spec: ExecutionSpec,
    sandbox_settings: SandboxEffectiveSettings,
) -> dict[str, str]:
    profile = sandbox_settings.profile
    environment = dict(profile.environment.variables)
    for name in profile.environment.allowlist:
        value = spec.env.get(name)
        if value is not None:
            environment[name] = value
    return environment


@final
class ShellContainerCommandExecutor(CommandExecutor):
    def __init__(
        self,
        *,
        container_settings: ContainerEffectiveSettings | None,
        container_backend: ContainerAsyncBackend | None = None,
        opt_in_backends: Sequence[ContainerBackend | str] = (),
        local_executor: CommandExecutor | None = None,
        rootful_authorized: bool = False,
    ) -> None:
        if container_settings is not None:
            assert isinstance(container_settings, ContainerEffectiveSettings)
        if container_backend is not None:
            assert isinstance(container_backend, ContainerAsyncBackend)
        if local_executor is not None:
            assert hasattr(local_executor, "execute")
        opt_in_backends = tuple(
            ContainerBackend(backend) for backend in opt_in_backends
        )
        _assert_bool(rootful_authorized, "rootful_authorized")
        self._container_settings = container_settings
        self._container_backend = container_backend
        self._opt_in_backends = opt_in_backends
        self._local_executor = local_executor or LocalCommandExecutor()
        self._rootful_authorized = rootful_authorized

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        assert isinstance(spec, ExecutionSpec)
        start_time = perf_counter()
        try:
            plan = lower_shell_execution_spec(
                spec,
                container_settings=self._container_settings,
            )
        except ShellPolicyDenied as error:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.POLICY_DENIED,
                error_code=error.error_code,
                error_message=str(error),
                backend="container",
            )
        except AssertionError as error:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=f"container shell plan is invalid: {error}",
                backend="container",
            )
        if plan.mode is ShellExecutionMode.LOCAL:
            return await self._local_executor.execute(spec, stream=stream)
        if self._container_backend is None:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=(
                    "container execution is selected but no backend is "
                    "configured"
                ),
                backend="container",
            )
        assert plan.container_plan is not None
        try:
            selection = await _select_backend(
                plan.container_plan,
                self._container_backend,
                opt_in_backends=self._opt_in_backends,
                rootful_authorized=self._rootful_authorized,
            )
        except Exception as error:
            diagnostics = _backend_exception_diagnostics(
                error,
                operation=ContainerBackendOperation.PROBE,
                fallback_code=ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                fallback_message="container backend probe failed",
            )
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=_diagnostic_summary(diagnostics),
                backend="container",
                metadata=_container_metadata(plan, diagnostics),
            )
        if not selection.ok:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=_diagnostic_summary(selection.diagnostics),
                backend="container",
                metadata=_container_metadata(plan, selection.diagnostics),
            )
        try:
            result = await run_container_managed_lifecycle(
                self._container_backend,
                plan.container_plan.run_plan,
                output_contract=plan.output_contract,
                deadlines=_deadlines(spec),
                stream_policy=_stream_policy(spec),
            )
        except Exception as error:
            diagnostics = _backend_exception_diagnostics(
                error,
                operation=ContainerBackendOperation.CREATE,
                fallback_code=ContainerBackendDiagnosticCode.CREATE_FAILED,
                fallback_message="container backend execution failed",
            )
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=_diagnostic_summary(diagnostics),
                backend="container",
                metadata=_container_metadata(plan, diagnostics),
            )
        stdout = _shell_stream_capture(
            result,
            ContainerBackendStream.STDOUT,
            spec.max_stdout_bytes,
        )
        stderr = _shell_stream_capture(
            result,
            ContainerBackendStream.STDERR,
            spec.max_stderr_bytes,
        )
        generated_output_replacements = _container_generated_replacements(spec)
        stdout = _scrub_capture(stdout, generated_output_replacements)
        stderr = _scrub_capture(stderr, generated_output_replacements)
        await _emit_container_streams(
            stream,
            result,
            stdout,
            stderr,
            generated_output_replacements=generated_output_replacements,
        )
        return _container_result_to_shell_result(
            spec,
            plan,
            result,
            stdout=stdout,
            stderr=stderr,
            start_time=start_time,
        )


async def _select_backend(
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


def _backend_exception_diagnostics(
    error: Exception,
    *,
    operation: ContainerBackendOperation,
    fallback_code: ContainerBackendDiagnosticCode,
    fallback_message: str,
) -> tuple[ContainerBackendDiagnostic, ...]:
    diagnostic = getattr(error, "diagnostic", None)
    if all(
        hasattr(diagnostic, attribute)
        for attribute in ("code", "operation", "message")
    ):
        return (cast(ContainerBackendDiagnostic, diagnostic),)
    return (
        ContainerBackendDiagnostic(
            code=fallback_code,
            operation=operation,
            message=fallback_message,
            retryable=True,
        ),
    )


def _container_request_from_spec(spec: ExecutionSpec) -> ContainerPlanRequest:
    return ContainerPlanRequest(
        request_kind=ContainerPlanRequestKind.TYPED_TOOL,
        logical_name=spec.tool_name,
        command=spec.command,
        argv=_container_argv(spec),
        cwd=_container_cwd(spec.display_cwd),
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        request_id=_optional_metadata_string(spec, "tool_call_id"),
    )


def _container_argv(spec: ExecutionSpec) -> tuple[str, ...]:
    if spec.output_kind is ShellOutputKind.GENERATED_FILES:
        assert spec.output_plan is not None, "generated outputs require a plan"
        assert len(spec.argv) == len(
            spec.display_argv
        ), "generated output display argv must mirror argv"
        replacement = PurePosixPath(
            _CONTAINER_OUTPUT_ROOT,
            spec.output_plan.prefix_name,
        ).as_posix()
        return tuple(
            (
                replacement
                if argv_item == GENERATED_OUTPUT_PREFIX_PLACEHOLDER
                else display_item
            )
            for argv_item, display_item in zip(spec.argv, spec.display_argv)
        )
    replacement = f"{_CONTAINER_OUTPUT_ROOT}/{spec.command}"
    return tuple(
        replacement if item == GENERATED_OUTPUT_PREFIX_PLACEHOLDER else item
        for item in spec.display_argv
    )


def _container_cwd(display_cwd: str) -> str:
    if display_cwd == ".":
        return "/workspace"
    path = PurePosixPath(display_cwd)
    assert not path.is_absolute(), "display cwd must be workspace-relative"
    return PurePosixPath("/workspace", path).as_posix()


def _output_contract_from_spec(
    spec: ExecutionSpec,
) -> ContainerOutputContract | None:
    if spec.output_kind is not ShellOutputKind.GENERATED_FILES:
        return None
    assert spec.output_plan is not None, "generated outputs require a plan"
    return ContainerOutputContract(
        contract_type=ContainerOutputContractType.GENERATED_FILE,
        max_bytes=spec.output_plan.max_total_bytes,
        max_files=spec.output_plan.max_files,
        per_file_bytes=spec.output_plan.max_file_bytes,
        media_policy=ContainerOutputMediaPolicy(
            allowed_media_types=tuple(
                sorted(set(spec.output_plan.suffix_media_types.values()))
            ),
        ),
    )


def _deadlines(spec: ExecutionSpec) -> ContainerLifecycleDeadlines:
    return ContainerLifecycleDeadlines(
        execution_seconds=spec.timeout_seconds,
        parent_seconds=spec.timeout_seconds,
        cleanup_seconds=_DEFAULT_CLEANUP_SECONDS,
    )


def _stream_policy(spec: ExecutionSpec) -> ContainerStreamDrainPolicy:
    stdout_limit = spec.max_stdout_bytes + 1
    stderr_limit = spec.max_stderr_bytes + 1
    return ContainerStreamDrainPolicy(
        max_chunks=max(1, stdout_limit + stderr_limit + 16),
        max_bytes=max(1, stdout_limit + stderr_limit),
        max_chunk_bytes=max(1, stdout_limit, stderr_limit),
        max_stdout_bytes=stdout_limit,
        max_stderr_bytes=stderr_limit,
        max_non_output_chunks=_MAX_PROGRESS_STREAM_CHUNKS,
        max_non_output_bytes=_MAX_PROGRESS_STREAM_BYTES,
        preserve_truncated_prefix=True,
    )


async def _emit_container_streams(
    stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    result: ContainerManagedLifecycleResult,
    stdout: tuple[str, int, bool],
    stderr: tuple[str, int, bool],
    *,
    generated_output_replacements: tuple[tuple[str, str], ...] = (),
) -> None:
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
            content = _scrub_generated_output_paths(
                content,
                generated_output_replacements,
            )
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


def _container_result_to_shell_result(
    spec: ExecutionSpec,
    plan: ShellExecutionPlan,
    result: ContainerManagedLifecycleResult,
    *,
    stdout: tuple[str, int, bool],
    stderr: tuple[str, int, bool],
    start_time: float,
) -> ExecutionResult:
    status, error_code, error_message = _shell_status(spec, result)
    try:
        generated_files = _generated_files(spec, result.output)
    except _ContainerGeneratedOutputError as error:
        status = ShellExecutionStatus.TOO_LARGE
        error_code = ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED
        error_message = str(error) or "container generated output was rejected"
        generated_files = ()
    return ExecutionResult(
        backend="container",
        tool_name=spec.tool_name,
        command=spec.command,
        argv=spec.display_argv,
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=result.execution.exit_code,
        stdout=stdout[0],
        stderr=stderr[0],
        stdout_media_type=spec.stdout_media_type,
        output_kind=spec.output_kind,
        generated_files=generated_files,
        stdout_bytes=stdout[1],
        stderr_bytes=stderr[1],
        stdout_truncated=stdout[2],
        stderr_truncated=stderr[2],
        timed_out=result.timed_out_phase is not None,
        cancelled=result.cancelled_phase is not None,
        duration_ms=_duration_ms(start_time),
        error_code=error_code,
        error_message=error_message,
        metadata=_container_metadata(plan, result.diagnostics),
    )


def _shell_status(
    spec: ExecutionSpec,
    result: ContainerManagedLifecycleResult,
) -> tuple[ShellExecutionStatus, ShellExecutionErrorCode | None, str | None]:
    if result.cancelled_phase is not None:
        return (
            ShellExecutionStatus.CANCELLED,
            ShellExecutionErrorCode.CANCELLED,
            "container execution cancelled",
        )
    if result.timed_out_phase is not None:
        return (
            ShellExecutionStatus.TIMEOUT,
            ShellExecutionErrorCode.TIMEOUT,
            "container execution timed out",
        )
    if result.output is not None and result.output.decision is not (
        ContainerOutputDecisionType.ACCEPT
    ):
        if _output_too_large(result.output):
            return (
                ShellExecutionStatus.TOO_LARGE,
                ShellExecutionErrorCode.TOO_LARGE,
                "container generated output exceeded limits",
            )
        return (
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "container generated output was rejected",
        )
    if result.execution.exit_code is not None and result.execution.exit_code:
        status = _status_for_exit_code(result.execution.exit_code, spec)
        if status is not ShellExecutionStatus.NONZERO_EXIT:
            return (
                status,
                ShellExecutionErrorCode(status.value),
                _exit_status_message(status),
            )
        return (
            ShellExecutionStatus.NONZERO_EXIT,
            ShellExecutionErrorCode.NONZERO_EXIT,
            "container command exited non-zero",
        )
    if result.execution.status is ContainerResultStatus.DENIED:
        return (
            ShellExecutionStatus.POLICY_DENIED,
            ShellExecutionErrorCode.POLICY_DENIED,
            _diagnostic_summary(result.diagnostics),
        )
    if result.execution.status is not ContainerResultStatus.COMPLETED:
        return (
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.TOOL_ERROR,
            _diagnostic_summary(result.diagnostics),
        )
    return ShellExecutionStatus.COMPLETED, None, None


def _exit_status_message(status: ShellExecutionStatus) -> str:
    if status is ShellExecutionStatus.COMMAND_UNAVAILABLE:
        return "command is unavailable"
    return f"container command exited with {status.value}"


def _shell_stream_capture(
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


def _scrub_capture(
    capture: tuple[str, int, bool],
    replacements: tuple[tuple[str, str], ...],
) -> tuple[str, int, bool]:
    content, byte_count, truncated = capture
    if not replacements:
        return capture
    return (
        _scrub_generated_output_paths(content, replacements),
        byte_count,
        truncated,
    )


def _container_generated_replacements(
    spec: ExecutionSpec,
) -> tuple[tuple[str, str], ...]:
    if spec.output_kind is not ShellOutputKind.GENERATED_FILES:
        return ()
    assert spec.output_plan is not None, "generated outputs require a plan"
    runtime_prefix = PurePosixPath(
        _CONTAINER_OUTPUT_ROOT,
        spec.output_plan.prefix_name,
    ).as_posix()
    replacements = (
        (runtime_prefix, spec.output_plan.display_prefix),
        (_CONTAINER_OUTPUT_ROOT, "[generated_output_directory]"),
    )
    return tuple(
        sorted(replacements, key=lambda item: len(item[0]), reverse=True)
    )


def _scrub_generated_output_paths(
    value: str,
    replacements: tuple[tuple[str, str], ...],
) -> str:
    for source, replacement in replacements:
        if source:
            value = value.replace(source, replacement)
    return value


def _generated_files(
    spec: ExecutionSpec,
    output: ContainerOutputValidationResult | None,
) -> tuple[GeneratedFile, ...]:
    if output is None:
        return ()
    if output.decision is not ContainerOutputDecisionType.ACCEPT:
        return ()
    if spec.output_kind is not ShellOutputKind.GENERATED_FILES:
        return ()
    assert spec.output_plan is not None, "generated outputs require a plan"
    total_bytes = 0
    generated_files: list[GeneratedFile] = []
    for artifact in output.artifacts:
        if len(generated_files) >= spec.output_plan.max_files:
            raise _ContainerGeneratedOutputError(
                "container generated output file count exceeded limit"
            )
        if artifact.size_bytes > spec.output_plan.max_file_bytes:
            raise _ContainerGeneratedOutputError(
                "container generated output file exceeded byte limit"
            )
        total_bytes += artifact.size_bytes
        if total_bytes > spec.output_plan.max_total_bytes:
            raise _ContainerGeneratedOutputError(
                "container generated output files exceeded total byte limit"
            )
        generated_files.append(_generated_file(artifact, spec.output_plan))
    return tuple(generated_files)


def _generated_file(
    artifact: ContainerOutputArtifact,
    plan: GeneratedOutputPlan,
) -> GeneratedFile:
    path = PurePosixPath(artifact.path)
    if len(path.parts) != 1 or not _matches_generated_output_prefix(
        path.name,
        plan.prefix_name,
    ):
        raise _ContainerGeneratedOutputError(
            "container generated output path did not match expected prefix"
        )
    suffix = path.suffix
    if suffix not in plan.allowed_suffixes:
        raise _ContainerGeneratedOutputError(
            "container generated output suffix is not allowed"
        )
    media_type = plan.suffix_media_types[suffix]
    if artifact.media_type != media_type:
        raise _ContainerGeneratedOutputError(
            "container generated output media type did not match suffix"
        )
    return GeneratedFile(
        display_path=_display_generated_path(path.name, plan),
        media_type=media_type,
        suffix=suffix,
        bytes=artifact.size_bytes,
        sha256=artifact.digest.removeprefix("sha256:"),
        page=_generated_page_number(path.name, plan.prefix_name),
        truncated=False,
        metadata={"quarantined": artifact.quarantined},
    )


def _display_generated_path(name: str, plan: GeneratedOutputPlan) -> str:
    return (
        f"{plan.display_prefix}"
        f"{_generated_display_suffix(name, plan.prefix_name)}"
    )


def _generated_display_suffix(name: str, prefix: str) -> str:
    suffix = name[len(prefix) :]
    path_suffix = PurePosixPath(name).suffix
    if suffix and suffix == path_suffix:
        return suffix
    stem = PurePosixPath(name).stem
    stem_suffix = stem[len(prefix) :]
    if (
        len(stem_suffix) > 1
        and stem_suffix[0] in ("-", "_")
        and stem_suffix[1:].isdecimal()
    ):
        return suffix
    raise _ContainerGeneratedOutputError(
        "container generated output suffix did not match display policy"
    )


def _matches_generated_output_prefix(name: str, prefix: str) -> bool:
    if name == prefix:
        return True
    if not name.startswith(prefix):
        return False
    remainder = name[len(prefix) :]
    return bool(remainder) and remainder[0] in (".", "-", "_")


def _generated_page_number(name: str, prefix: str) -> int | None:
    suffix = PurePosixPath(name).stem[len(prefix) :].lstrip("-_")
    if not suffix.isdecimal():
        return None
    return int(suffix)


def _container_metadata(
    plan: ShellExecutionPlan,
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> dict[str, object]:
    metadata = dict(plan.local_spec.metadata)
    metadata["execution_backend"] = "container"
    if plan.container_plan is not None:
        metadata["container_profile"] = (
            plan.container_plan.run_plan.profile_name
        )
        metadata["container_policy_version"] = (
            plan.container_plan.run_plan.policy_version
        )
        metadata["container_plan_fingerprint"] = (
            plan.container_plan.plan_fingerprint
        )
    if diagnostics:
        metadata["container_diagnostic_codes"] = tuple(
            cast(ContainerBackendDiagnosticCode, diagnostic.code).value
            for diagnostic in diagnostics
        )
        stable_codes = isolation_diagnostic_codes(diagnostics)
        if stable_codes:
            metadata["isolation_diagnostic_codes"] = stable_codes
    return metadata


def _closed_result(
    spec: ExecutionSpec,
    *,
    start_time: float,
    status: ShellExecutionStatus,
    error_code: ShellExecutionErrorCode,
    error_message: str,
    backend: str = "local",
    metadata: dict[str, object] | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        backend=backend,
        tool_name=spec.tool_name,
        command=spec.command,
        argv=spec.display_argv,
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=None,
        stdout="",
        stderr="",
        stdout_media_type=spec.stdout_media_type,
        output_kind=spec.output_kind,
        stdout_bytes=0,
        stderr_bytes=0,
        timed_out=status is ShellExecutionStatus.TIMEOUT,
        cancelled=status is ShellExecutionStatus.CANCELLED,
        duration_ms=_duration_ms(start_time),
        error_code=error_code,
        error_message=error_message,
        metadata=metadata or dict(spec.metadata),
    )


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


def _output_too_large(output: ContainerOutputValidationResult) -> bool:
    return any(
        diagnostic.code is ContainerOutputDiagnosticCode.TOO_LARGE
        for diagnostic in output.diagnostics
    )


def _optional_metadata_string(
    spec: ExecutionSpec,
    key: str,
) -> str | None:
    value = spec.metadata.get(key)
    if value is None:
        return None
    assert isinstance(value, str), f"{key} metadata must be a string"
    return value


def _duration_ms(start_time: float) -> int:
    return max(0, int((perf_counter() - start_time) * 1000))


def _enum_value(
    value: object,
    enum_type: type[ShellExecutionMode],
    field_name: str,
) -> ShellExecutionMode:
    if isinstance(value, enum_type):
        return value
    assert isinstance(value, str), f"{field_name} must be a string"
    return enum_type(value)
