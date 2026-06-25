from ...entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    ToolValue,
)
from ...isolation import SandboxBackend as IsolationSandboxBackend
from ...isolation import SandboxEffectiveSettings
from ...sandbox import (
    SandboxAsyncBackend,
    SandboxBackendDiagnostic,
    SandboxBackendDiagnosticCode,
    SandboxBackendStream,
    SandboxExecutionResult,
    SandboxOutputArtifact,
    SandboxResultStatus,
    select_sandbox_backend,
)
from .container import (
    ShellExecutionMode,
    ShellExecutionPlan,
    lower_shell_execution_spec,
)
from .entities import (
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
)
from .executor import CommandExecutor

from base64 import b64encode
from collections.abc import Awaitable, Callable, Sequence
from hashlib import sha256
from pathlib import PurePosixPath
from time import perf_counter
from typing import cast, final


@final
class ShellSandboxCommandExecutor(CommandExecutor):
    def __init__(
        self,
        *,
        sandbox_settings: SandboxEffectiveSettings | None,
        sandbox_backend: SandboxAsyncBackend | None = None,
    ) -> None:
        if sandbox_settings is not None:
            assert isinstance(sandbox_settings, SandboxEffectiveSettings)
        if sandbox_backend is not None:
            assert _is_sandbox_backend(sandbox_backend)
        self._sandbox_settings = sandbox_settings
        self._sandbox_backend = sandbox_backend

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
        if spec.executable is None:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.COMMAND_UNAVAILABLE,
                error_code=ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
                error_message="command is unavailable",
                backend="sandbox",
            )
        if self._sandbox_settings is None:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=(
                    "sandbox execution is selected but no settings are "
                    "configured"
                ),
                backend="sandbox",
            )
        try:
            plan = lower_shell_execution_spec(
                spec,
                sandbox_settings=self._sandbox_settings,
            )
        except AssertionError as error:
            status, error_code = _lowering_error_status(str(error))
            return _closed_result(
                spec,
                start_time=start_time,
                status=status,
                error_code=error_code,
                error_message=f"sandbox shell plan is invalid: {error}",
                backend="sandbox",
            )
        assert plan.mode is ShellExecutionMode.SANDBOX
        assert plan.sandbox_plan is not None
        if self._sandbox_backend is None:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=(
                    "sandbox execution is selected but no backend is "
                    "configured"
                ),
                backend="sandbox",
                metadata=_sandbox_metadata(plan, ()),
            )
        try:
            probe = await self._sandbox_backend.probe()
        except Exception as error:
            diagnostics = _backend_exception_diagnostics(error)
            if diagnostics:
                return _closed_result(
                    spec,
                    start_time=start_time,
                    status=ShellExecutionStatus.TOOL_ERROR,
                    error_code=ShellExecutionErrorCode.TOOL_ERROR,
                    error_message=_diagnostic_summary(diagnostics),
                    backend="sandbox",
                    metadata=_sandbox_metadata(plan, diagnostics),
                )
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message="sandbox backend probe failed",
                backend="sandbox",
                metadata=_sandbox_metadata(plan, ()),
            )
        selection = select_sandbox_backend(plan.sandbox_plan, (probe,))
        if not selection.ok:
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message=_diagnostic_summary(selection.diagnostics),
                backend="sandbox",
                metadata=_sandbox_metadata(plan, selection.diagnostics),
            )
        try:
            result = await self._sandbox_backend.execute(plan.sandbox_plan)
        except Exception as error:
            diagnostics = _backend_exception_diagnostics(error)
            if diagnostics:
                return _closed_result(
                    spec,
                    start_time=start_time,
                    status=ShellExecutionStatus.TOOL_ERROR,
                    error_code=ShellExecutionErrorCode.TOOL_ERROR,
                    error_message=_diagnostic_summary(diagnostics),
                    backend="sandbox",
                    metadata=_sandbox_metadata(plan, diagnostics),
                )
            return _closed_result(
                spec,
                start_time=start_time,
                status=ShellExecutionStatus.TOOL_ERROR,
                error_code=ShellExecutionErrorCode.TOOL_ERROR,
                error_message="sandbox backend execution failed",
                backend="sandbox",
                metadata=_sandbox_metadata(plan, ()),
            )
        stdout = _sandbox_stream_capture(
            result.stdout,
            spec.max_stdout_bytes,
            backend_truncated=result.stream_truncated,
        )
        stderr = _sandbox_stream_capture(
            result.stderr,
            spec.max_stderr_bytes,
            backend_truncated=result.stream_truncated,
        )
        await _emit_sandbox_streams(stream, result, stdout, stderr)
        return _sandbox_result_to_shell_result(
            spec,
            plan,
            result,
            stdout=stdout,
            stderr=stderr,
            start_time=start_time,
        )


def _is_sandbox_backend(value: object) -> bool:
    return callable(getattr(value, "probe", None)) and callable(
        getattr(value, "execute", None)
    )


def _backend_exception_diagnostics(
    error: Exception,
) -> tuple[SandboxBackendDiagnostic, ...]:
    diagnostic = getattr(error, "diagnostic", None)
    if diagnostic is None:
        return ()
    if not all(
        hasattr(diagnostic, attribute)
        for attribute in ("code", "operation", "message")
    ):
        return ()
    return (cast(SandboxBackendDiagnostic, diagnostic),)


def _sandbox_result_to_shell_result(
    spec: ExecutionSpec,
    plan: ShellExecutionPlan,
    result: SandboxExecutionResult,
    *,
    stdout: tuple[str, int, bool],
    stderr: tuple[str, int, bool],
    start_time: float,
) -> ExecutionResult:
    generated_files: tuple[GeneratedFile, ...] = ()
    generated_error: _SandboxGeneratedOutputError | None = None
    if spec.output_kind is ShellOutputKind.GENERATED_FILES:
        try:
            generated_files = _generated_files(
                result.output_artifacts,
                spec.output_plan,
            )
        except _SandboxGeneratedOutputError as error:
            generated_error = error
    status, error_code, error_message = _shell_status(
        result,
        generated_error,
    )
    return ExecutionResult(
        backend="sandbox",
        tool_name=spec.tool_name,
        command=spec.command,
        argv=spec.display_argv,
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=result.exit_code,
        stdout=stdout[0],
        stderr=stderr[0],
        stdout_media_type=spec.stdout_media_type,
        output_kind=spec.output_kind,
        generated_files=generated_files,
        stdout_bytes=stdout[1],
        stderr_bytes=stderr[1],
        stdout_truncated=stdout[2],
        stderr_truncated=stderr[2],
        timed_out=status is ShellExecutionStatus.TIMEOUT,
        cancelled=status is ShellExecutionStatus.CANCELLED,
        duration_ms=_duration_ms(start_time),
        error_code=error_code,
        error_message=error_message,
        metadata=_sandbox_metadata(plan, result.diagnostics, result=result),
    )


def _shell_status(
    result: SandboxExecutionResult,
    generated_error: "_SandboxGeneratedOutputError | None",
) -> tuple[ShellExecutionStatus, ShellExecutionErrorCode | None, str | None]:
    result_status = _enum_value(result.status)
    if result_status == SandboxResultStatus.CANCELLED.value:
        return (
            ShellExecutionStatus.CANCELLED,
            ShellExecutionErrorCode.CANCELLED,
            "sandbox execution cancelled",
        )
    if result_status == SandboxResultStatus.TIMED_OUT.value:
        return (
            ShellExecutionStatus.TIMEOUT,
            ShellExecutionErrorCode.TIMEOUT,
            "sandbox execution timed out",
        )
    if generated_error is not None:
        return (
            generated_error.status,
            generated_error.error_code,
            generated_error.message,
        )
    output_rejected = _diagnostics_with_code(
        result.diagnostics,
        SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
    )
    if output_rejected:
        if any(
            "exceed" in diagnostic.message for diagnostic in output_rejected
        ):
            return (
                ShellExecutionStatus.TOO_LARGE,
                ShellExecutionErrorCode.TOO_LARGE,
                "sandbox generated output exceeded limits",
            )
        return (
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "sandbox generated output was rejected",
        )
    if result.cleanup_uncertain or _diagnostics_with_code(
        result.diagnostics,
        SandboxBackendDiagnosticCode.CLEANUP_FAILED,
    ):
        return (
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.TOOL_ERROR,
            "sandbox cleanup failed",
        )
    if result_status == SandboxResultStatus.DENIED.value:
        return (
            ShellExecutionStatus.POLICY_DENIED,
            ShellExecutionErrorCode.POLICY_DENIED,
            _diagnostic_summary(result.diagnostics),
        )
    if result.exit_code is not None and result.exit_code:
        return (
            ShellExecutionStatus.NONZERO_EXIT,
            ShellExecutionErrorCode.NONZERO_EXIT,
            "sandbox command exited non-zero",
        )
    if result_status != SandboxResultStatus.COMPLETED.value:
        return (
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.TOOL_ERROR,
            _diagnostic_summary(result.diagnostics),
        )
    return ShellExecutionStatus.COMPLETED, None, None


def _sandbox_stream_capture(
    raw: bytes,
    byte_cap: int,
    *,
    backend_truncated: bool,
) -> tuple[str, int, bool]:
    content = raw[:byte_cap]
    truncated = len(raw) > byte_cap or (
        backend_truncated and len(raw) >= byte_cap
    )
    return (
        content.decode("utf-8", errors="replace"),
        len(content),
        truncated,
    )


async def _emit_sandbox_streams(
    stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    result: SandboxExecutionResult,
    stdout: tuple[str, int, bool],
    stderr: tuple[str, int, bool],
) -> None:
    if stream is None:
        return
    stdout_emitted = False
    stderr_emitted = False
    for chunk in result.stream_chunks:
        stream_kind = _stream_kind(cast(SandboxBackendStream, chunk.stream))
        if stream_kind is ToolExecutionStreamKind.STDOUT:
            if stdout_emitted or not stdout[0]:
                continue
            stdout_emitted = True
            content = stdout[0]
            metadata: dict[str, ToolValue] = {
                "backend": "sandbox",
                "truncated": stdout[2],
            }
        else:
            if stderr_emitted or not stderr[0]:
                continue
            stderr_emitted = True
            content = stderr[0]
            metadata = {"backend": "sandbox", "truncated": stderr[2]}
        await stream(
            ToolExecutionStreamEvent(
                kind=stream_kind,
                content=content,
                metadata=metadata,
            )
        )


def _stream_kind(stream: SandboxBackendStream) -> ToolExecutionStreamKind:
    if _enum_value(stream) == SandboxBackendStream.STDOUT.value:
        return ToolExecutionStreamKind.STDOUT
    return ToolExecutionStreamKind.STDERR


def _generated_files(
    artifacts: Sequence[SandboxOutputArtifact],
    plan: GeneratedOutputPlan | None,
) -> tuple[GeneratedFile, ...]:
    if plan is None:
        raise _SandboxGeneratedOutputError(
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "sandbox generated output contract is invalid",
        )
    files: list[GeneratedFile] = []
    total_bytes = 0
    for artifact in artifacts:
        if len(files) >= plan.max_files:
            raise _generated_output_too_large()
        path = _safe_artifact_path(artifact.path)
        if not _matches_generated_output_prefix(path.name, plan.prefix_name):
            continue
        suffix = path.suffix or ".out"
        if suffix not in plan.allowed_suffixes:
            raise _generated_output_too_large()
        content = artifact.content
        if len(content) > plan.max_file_bytes:
            raise _generated_output_too_large()
        total_bytes += len(content)
        if total_bytes > plan.max_total_bytes:
            raise _generated_output_too_large()
        display_path = _display_generated_path(path, plan)
        files.append(
            GeneratedFile(
                display_path=display_path,
                media_type=plan.suffix_media_types[suffix],
                suffix=suffix,
                bytes=len(content),
                sha256=sha256(content).hexdigest(),
                content_base64=(
                    b64encode(content).decode("ascii")
                    if len(content) <= plan.max_inline_bytes
                    else None
                ),
                truncated=False,
            )
        )
    return tuple(files)


def _safe_artifact_path(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts:
        raise _SandboxGeneratedOutputError(
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "sandbox generated output path is unsafe",
        )
    if any(part.startswith(".") for part in value.parts):
        raise _SandboxGeneratedOutputError(
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "sandbox generated output path is unsafe",
        )
    return value


def _matches_generated_output_prefix(name: str, prefix_name: str) -> bool:
    if name == prefix_name or name.startswith(f"{prefix_name}."):
        return True
    return name.startswith(f"{prefix_name}-") or name.startswith(
        f"{prefix_name}_"
    )


def _display_generated_path(
    path: PurePosixPath,
    plan: GeneratedOutputPlan,
) -> str:
    name = path.name
    if name == plan.prefix_name:
        display_name = plan.display_prefix
    else:
        display_name = plan.display_prefix + name[len(plan.prefix_name) :]
    if path.parent == PurePosixPath("."):
        return display_name
    return PurePosixPath(path.parent, display_name).as_posix()


def _generated_output_too_large() -> "_SandboxGeneratedOutputError":
    return _SandboxGeneratedOutputError(
        ShellExecutionStatus.TOO_LARGE,
        ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        "sandbox generated output capture failed",
    )


def _diagnostics_with_code(
    diagnostics: Sequence[SandboxBackendDiagnostic],
    code: SandboxBackendDiagnosticCode,
) -> tuple[SandboxBackendDiagnostic, ...]:
    return tuple(
        diagnostic
        for diagnostic in diagnostics
        if _enum_value(diagnostic.code) == code.value
    )


def _sandbox_metadata(
    plan: ShellExecutionPlan,
    diagnostics: Sequence[SandboxBackendDiagnostic],
    *,
    result: SandboxExecutionResult | None = None,
) -> dict[str, object]:
    metadata = dict(plan.local_spec.metadata)
    metadata["execution_backend"] = "sandbox"
    if plan.sandbox_plan is not None:
        sandbox_plan = plan.sandbox_plan
        sandbox_backend = cast(
            IsolationSandboxBackend,
            sandbox_plan.settings.backend,
        )
        metadata["sandbox_backend"] = sandbox_backend.value
        metadata["sandbox_profile"] = sandbox_plan.settings.profile_name
        metadata["sandbox_policy_version"] = (
            sandbox_plan.settings.policy_version
        )
        metadata["sandbox_plan_fingerprint"] = sandbox_plan.plan_fingerprint
    if diagnostics:
        metadata["sandbox_diagnostic_codes"] = tuple(
            _enum_value(diagnostic.code) for diagnostic in diagnostics
        )
    if result is not None and result.cleanup_uncertain:
        metadata["sandbox_cleanup_uncertain"] = True
    return metadata


def _closed_result(
    spec: ExecutionSpec,
    *,
    start_time: float,
    status: ShellExecutionStatus,
    error_code: ShellExecutionErrorCode,
    error_message: str,
    backend: str,
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


def _lowering_error_status(
    message: str,
) -> tuple[ShellExecutionStatus, ShellExecutionErrorCode]:
    if "generated outputs require" in message:
        return (
            ShellExecutionStatus.TOOL_ERROR,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
    return (
        ShellExecutionStatus.POLICY_DENIED,
        ShellExecutionErrorCode.POLICY_DENIED,
    )


def _diagnostic_summary(
    diagnostics: Sequence[SandboxBackendDiagnostic],
) -> str:
    if not diagnostics:
        return "sandbox execution failed"
    codes = sorted(
        {_enum_value(diagnostic.code) for diagnostic in diagnostics}
    )
    return "sandbox execution failed: " + ", ".join(codes)


def _enum_value(value: object) -> str:
    raw_value = getattr(value, "value", value)
    assert isinstance(raw_value, str)
    return raw_value


def _duration_ms(start_time: float) -> int:
    return max(0, int((perf_counter() - start_time) * 1000))


class _SandboxGeneratedOutputError(Exception):
    def __init__(
        self,
        status: ShellExecutionStatus,
        error_code: ShellExecutionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.message = message
