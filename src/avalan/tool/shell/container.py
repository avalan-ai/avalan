from ...container import (
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
    ContainerOutputArtifact,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputDiagnosticCode,
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
from ...types import (
    assert_bool as _assert_bool,
)
from .entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
)
from .executor import CommandExecutor, LocalCommandExecutor
from .policy import ExecutionPolicy

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import cast, final

_CONTAINER_OUTPUT_ROOT = "/outputs"
_DEFAULT_CLEANUP_SECONDS = 5.0
_MAX_PROGRESS_STREAM_BYTES = 4096
_MAX_PROGRESS_STREAM_CHUNKS = 16


class ShellExecutionMode(StrEnum):
    LOCAL = "local"
    CONTAINER = "container"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellExecutionPlan:
    mode: ShellExecutionMode | str
    local_spec: ExecutionSpec
    container_plan: ContainerNormalizedRunPlan | None = None
    output_contract: ContainerOutputContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ShellExecutionMode, "mode"),
        )
        assert isinstance(self.local_spec, ExecutionSpec)
        if self.container_plan is not None:
            assert isinstance(self.container_plan, ContainerNormalizedRunPlan)
        if self.output_contract is not None:
            assert isinstance(self.output_contract, ContainerOutputContract)
        if self.mode is ShellExecutionMode.LOCAL:
            assert (
                self.container_plan is None
            ), "local shell plans cannot carry container plans"
        if self.mode is ShellExecutionMode.CONTAINER:
            assert (
                self.container_plan is not None
            ), "container shell plans require a container plan"

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
    container_settings: ContainerEffectiveSettings | None = None,
) -> ShellExecutionPlan:
    assert isinstance(request, ShellCommandRequest)
    assert isinstance(policy, ExecutionPolicy)
    local_spec = await policy.normalize(request)
    return lower_shell_execution_spec(
        local_spec,
        container_settings=container_settings,
    )


def lower_shell_execution_spec(
    spec: ExecutionSpec,
    *,
    container_settings: ContainerEffectiveSettings | None = None,
) -> ShellExecutionPlan:
    assert isinstance(spec, ExecutionSpec)
    if container_settings is None:
        return ShellExecutionPlan(
            mode=ShellExecutionMode.LOCAL,
            local_spec=spec,
        )
    assert isinstance(container_settings, ContainerEffectiveSettings)
    if not container_settings.enabled:
        if container_settings.required:
            raise ShellPolicyDenied(
                ShellExecutionErrorCode.POLICY_DENIED,
                "container execution is required but no profile is enabled",
            )
        return ShellExecutionPlan(
            mode=ShellExecutionMode.LOCAL,
            local_spec=spec,
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
        selection = await _select_backend(
            plan.container_plan,
            self._container_backend,
            opt_in_backends=self._opt_in_backends,
            rootful_authorized=self._rootful_authorized,
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
        result = await run_container_managed_lifecycle(
            self._container_backend,
            plan.container_plan.run_plan,
            output_contract=plan.output_contract,
            deadlines=_deadlines(spec),
            stream_policy=_stream_policy(spec),
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
        await _emit_container_streams(stream, result, stdout, stderr)
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
    status, error_code, error_message = _shell_status(result)
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
        generated_files=_generated_files(result.output),
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


def _generated_files(
    output: ContainerOutputValidationResult | None,
) -> tuple[GeneratedFile, ...]:
    if output is None:
        return ()
    return tuple(_generated_file(artifact) for artifact in output.artifacts)


def _generated_file(artifact: ContainerOutputArtifact) -> GeneratedFile:
    suffix = Path(artifact.path).suffix or ".out"
    return GeneratedFile(
        display_path=artifact.path,
        media_type=artifact.media_type,
        suffix=suffix,
        bytes=artifact.size_bytes,
        sha256=artifact.digest.removeprefix("sha256:"),
        truncated=False,
        metadata={"quarantined": artifact.quarantined},
    )


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
