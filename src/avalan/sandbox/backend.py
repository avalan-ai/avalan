from ..isolation import (
    SandboxBackend,
    SandboxChildProcessPolicy,
    SandboxCleanupPolicy,
    SandboxInheritedFdPolicy,
    SandboxNetworkMode,
)
from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from ..types import (
    assert_optional_positive_number as _assert_optional_positive_number,
)
from .planning import (
    SandboxExecutionPlan,
)

from abc import ABC, abstractmethod
from asyncio import CancelledError, Lock, Semaphore, TimeoutError, wait_for
from asyncio import sleep as async_sleep
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from posixpath import normpath as normalize_posix_path
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)


class SandboxBackendOperation(StrEnum):
    PROBE = "probe"
    PREPARE_PROFILE = "prepare_profile"
    START = "start"
    STREAM = "stream"
    WAIT = "wait"
    COLLECT_OUTPUTS = "collect_outputs"
    CLEANUP = "cleanup"


class SandboxBackendDiagnosticCode(StrEnum):
    BACKEND_UNAVAILABLE = "sandbox.backend.unavailable"
    CAPABILITY_MISMATCH = "sandbox.backend.capability_mismatch"
    EXECUTABLE_DENIED = "sandbox.backend.executable_denied"
    PATH_DENIED = "sandbox.backend.path_denied"
    OUTPUT_REJECTED = "sandbox.backend.output_rejected"
    EXECUTION_FAILED = "sandbox.backend.execution_failed"
    CLEANUP_FAILED = "sandbox.backend.cleanup_failed"
    CANCELLED = "sandbox.backend.cancelled"
    TIMEOUT = "sandbox.backend.timeout"
    STREAM_TRUNCATED = "sandbox.backend.stream_truncated"
    CONCURRENCY_LIMIT = "sandbox.backend.concurrency_limit"


class SandboxBackendStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"


class SandboxResultStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxFilesystemControls:
    read_roots: bool
    write_roots: bool
    deny_roots: bool
    path_normalization: bool = True

    def __post_init__(self) -> None:
        _assert_bool(self.read_roots, "read_roots")
        _assert_bool(self.write_roots, "write_roots")
        _assert_bool(self.deny_roots, "deny_roots")
        _assert_bool(self.path_normalization, "path_normalization")

    def to_dict(self) -> dict[str, bool]:
        return {
            "read_roots": self.read_roots,
            "write_roots": self.write_roots,
            "deny_roots": self.deny_roots,
            "path_normalization": self.path_normalization,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxProcessControls:
    process_limits: bool
    child_processes: bool
    inherited_fds: bool

    def __post_init__(self) -> None:
        _assert_bool(self.process_limits, "process_limits")
        _assert_bool(self.child_processes, "child_processes")
        _assert_bool(self.inherited_fds, "inherited_fds")

    def to_dict(self) -> dict[str, bool]:
        return {
            "process_limits": self.process_limits,
            "child_processes": self.child_processes,
            "inherited_fds": self.inherited_fds,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxTempOutputMapping:
    temp_dirs: bool
    output_dirs: bool
    cleanup_budget: bool

    def __post_init__(self) -> None:
        _assert_bool(self.temp_dirs, "temp_dirs")
        _assert_bool(self.output_dirs, "output_dirs")
        _assert_bool(self.cleanup_budget, "cleanup_budget")

    def to_dict(self) -> dict[str, bool]:
        return {
            "temp_dirs": self.temp_dirs,
            "output_dirs": self.output_dirs,
            "cleanup_budget": self.cleanup_budget,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxBackendCapabilities:
    backend: SandboxBackend | str
    host_os: str
    architecture: str
    runtime_name: str
    sandbox_executable: str
    sandbox_executable_available: bool
    filesystem: SandboxFilesystemControls
    network_modes: Sequence[SandboxNetworkMode | str]
    process: SandboxProcessControls
    temp_output: SandboxTempOutputMapping
    unsupported_controls: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, SandboxBackend, "backend"),
        )
        _assert_non_empty_string(self.host_os, "host_os")
        _assert_non_empty_string(self.architecture, "architecture")
        _assert_non_empty_string(self.runtime_name, "runtime_name")
        _assert_non_empty_string(
            self.sandbox_executable,
            "sandbox_executable",
        )
        _assert_bool(
            self.sandbox_executable_available,
            "sandbox_executable_available",
        )
        assert isinstance(self.filesystem, SandboxFilesystemControls)
        network_modes = tuple(
            _enum_value(mode, SandboxNetworkMode, "network_modes")
            for mode in self.network_modes
        )
        assert network_modes, "network_modes must not be empty"
        assert isinstance(self.process, SandboxProcessControls)
        assert isinstance(self.temp_output, SandboxTempOutputMapping)
        object.__setattr__(self, "network_modes", network_modes)
        object.__setattr__(
            self,
            "unsupported_controls",
            _string_tuple(self.unsupported_controls, "unsupported_controls"),
        )

    def to_dict(self) -> dict[str, object]:
        backend = cast(SandboxBackend, self.backend)
        network_modes = cast(
            tuple[SandboxNetworkMode, ...], self.network_modes
        )
        return {
            "backend": backend.value,
            "host_os": self.host_os,
            "architecture": self.architecture,
            "runtime_name": self.runtime_name,
            "sandbox_executable": self.sandbox_executable,
            "sandbox_executable_available": self.sandbox_executable_available,
            "filesystem": self.filesystem.to_dict(),
            "network_modes": [mode.value for mode in network_modes],
            "process": self.process.to_dict(),
            "temp_output": self.temp_output.to_dict(),
            "unsupported_controls": sorted(self.unsupported_controls),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxBackendDiagnostic:
    code: SandboxBackendDiagnosticCode | str
    operation: SandboxBackendOperation | str
    message: str
    backend: SandboxBackend | str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _enum_value(self.code, SandboxBackendDiagnosticCode, "code"),
        )
        object.__setattr__(
            self,
            "operation",
            _enum_value(
                self.operation,
                SandboxBackendOperation,
                "operation",
            ),
        )
        if self.backend is not None:
            object.__setattr__(
                self,
                "backend",
                _enum_value(self.backend, SandboxBackend, "backend"),
            )
        _assert_non_empty_string(self.message, "message")
        _assert_bool(self.retryable, "retryable")

    def to_dict(self) -> dict[str, object]:
        code = cast(SandboxBackendDiagnosticCode, self.code)
        operation = cast(SandboxBackendOperation, self.operation)
        backend = cast(SandboxBackend | None, self.backend)
        return {
            "code": code.value,
            "operation": operation.value,
            "backend": None if backend is None else backend.value,
            "message": self.message,
            "retryable": self.retryable,
        }


class SandboxBackendError(Exception):
    def __init__(self, diagnostic: SandboxBackendDiagnostic) -> None:
        assert isinstance(diagnostic, SandboxBackendDiagnostic)
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxBackendProbeResult:
    backend: SandboxBackend | str
    available: bool
    capabilities: SandboxBackendCapabilities | None = None
    diagnostics: Sequence[SandboxBackendDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, SandboxBackend, "backend"),
        )
        _assert_bool(self.available, "available")
        if self.capabilities is not None:
            assert isinstance(self.capabilities, SandboxBackendCapabilities)
        if self.available:
            assert (
                self.capabilities is not None
            ), "available probes require capabilities"
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, SandboxBackendDiagnostic)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return self.available and not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        backend = cast(SandboxBackend, self.backend)
        return {
            "backend": backend.value,
            "available": self.available,
            "capabilities": (
                None
                if self.capabilities is None
                else self.capabilities.to_dict()
            ),
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxBackendSelection:
    backend: SandboxBackend | str | None
    capabilities: SandboxBackendCapabilities | None = None
    diagnostics: Sequence[SandboxBackendDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        if self.backend is not None:
            object.__setattr__(
                self,
                "backend",
                _enum_value(self.backend, SandboxBackend, "backend"),
            )
        if self.capabilities is not None:
            assert isinstance(self.capabilities, SandboxBackendCapabilities)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, SandboxBackendDiagnostic)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return self.backend is not None and not self.diagnostics


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxStreamChunk:
    stream: SandboxBackendStream | str
    content: bytes
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream",
            _enum_value(self.stream, SandboxBackendStream, "stream"),
        )
        assert isinstance(self.content, bytes), "content must be bytes"
        _assert_non_negative_int(self.sequence, "sequence")

    def to_dict(self) -> dict[str, object]:
        stream = cast(SandboxBackendStream, self.stream)
        return {
            "stream": stream.value,
            "content": self.content.decode("utf-8", errors="replace"),
            "sequence": self.sequence,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxOutputArtifact:
    path: str
    content: bytes

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.path, "path")
        assert isinstance(self.content, bytes), "content must be bytes"

    @property
    def size_bytes(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxExecutionResult:
    status: SandboxResultStatus | str
    exit_code: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    diagnostics: Sequence[SandboxBackendDiagnostic] = field(
        default_factory=tuple,
    )
    stream_chunks: Sequence[SandboxStreamChunk] = field(default_factory=tuple)
    output_artifacts: Sequence[SandboxOutputArtifact] = field(
        default_factory=tuple,
    )
    stream_truncated: bool = False
    cleanup_uncertain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, SandboxResultStatus, "status"),
        )
        if self.exit_code is not None:
            assert isinstance(self.exit_code, int), "exit_code must be integer"
        assert isinstance(self.stdout, bytes), "stdout must be bytes"
        assert isinstance(self.stderr, bytes), "stderr must be bytes"
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, SandboxBackendDiagnostic)
        for chunk in self.stream_chunks:
            assert isinstance(chunk, SandboxStreamChunk)
        for artifact in self.output_artifacts:
            assert isinstance(artifact, SandboxOutputArtifact)
        _assert_bool(self.stream_truncated, "stream_truncated")
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "stream_chunks", tuple(self.stream_chunks))
        object.__setattr__(
            self,
            "output_artifacts",
            tuple(self.output_artifacts),
        )

    @property
    def ok(self) -> bool:
        return self.status is SandboxResultStatus.COMPLETED and (
            not self.diagnostics
        )

    def to_dict(self) -> dict[str, object]:
        status = cast(SandboxResultStatus, self.status)
        return {
            "status": status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout.decode("utf-8", errors="replace"),
            "stderr": self.stderr.decode("utf-8", errors="replace"),
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "stream_chunks": [chunk.to_dict() for chunk in self.stream_chunks],
            "output_artifacts": [
                artifact.to_dict() for artifact in self.output_artifacts
            ],
            "stream_truncated": self.stream_truncated,
            "cleanup_uncertain": self.cleanup_uncertain,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxBackendCapabilityProfile:
    profile_id: str
    capabilities: SandboxBackendCapabilities

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.profile_id, "profile_id")
        assert isinstance(self.capabilities, SandboxBackendCapabilities)

    @property
    def backend(self) -> SandboxBackend:
        return cast(SandboxBackend, self.capabilities.backend)

    def probe(
        self,
        *,
        available: bool = False,
        diagnostics: Sequence[SandboxBackendDiagnostic] = (),
    ) -> SandboxBackendProbeResult:
        _assert_bool(available, "available")
        for diagnostic in diagnostics:
            assert isinstance(diagnostic, SandboxBackendDiagnostic)
        if not available:
            diagnostics = tuple(diagnostics) or (
                _diagnostic(
                    SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    SandboxBackendOperation.PROBE,
                    self.backend,
                    f"{self.profile_id} runtime is unavailable",
                    retryable=True,
                ),
            )
        return SandboxBackendProbeResult(
            backend=self.backend,
            available=available,
            capabilities=self.capabilities if available else None,
            diagnostics=diagnostics,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "capabilities": self.capabilities.to_dict(),
        }


class SandboxAsyncBackend(ABC):
    @abstractmethod
    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> SandboxBackendProbeResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def execute(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxExecutionResult:
        raise NotImplementedError  # pragma: no cover


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxFakeBackendScript:
    capabilities: SandboxBackendCapabilities
    available: bool = True
    probe_delay_seconds: float = 0
    operation_diagnostics: Mapping[
        SandboxBackendOperation | str,
        SandboxBackendDiagnosticCode | str,
    ] = field(default_factory=dict)
    cancel_operations: Sequence[SandboxBackendOperation | str] = field(
        default_factory=tuple,
    )
    timeout_operations: Sequence[SandboxBackendOperation | str] = field(
        default_factory=tuple,
    )
    operation_delay_seconds: Mapping[
        SandboxBackendOperation | str,
        float,
    ] = field(default_factory=dict)
    stream_chunks: Sequence[SandboxStreamChunk] = field(
        default_factory=lambda: (
            SandboxStreamChunk(
                stream=SandboxBackendStream.STDOUT,
                content=b"ok\n",
                sequence=0,
            ),
        ),
    )
    stream_delay_seconds: float = 0
    wait_exit_code: int = 0
    output_files: Mapping[str, bytes] = field(default_factory=dict)
    denied_paths: Sequence[str] = field(default_factory=tuple)
    cleanup_uncertain: bool = False
    max_concurrent_executions: int = 8

    def __post_init__(self) -> None:
        assert isinstance(self.capabilities, SandboxBackendCapabilities)
        _assert_bool(self.available, "available")
        _assert_non_negative_number(
            self.probe_delay_seconds,
            "probe_delay_seconds",
        )
        object.__setattr__(
            self,
            "operation_diagnostics",
            MappingProxyType(
                _operation_diagnostics(self.operation_diagnostics)
            ),
        )
        object.__setattr__(
            self,
            "cancel_operations",
            tuple(
                _enum_value(
                    operation,
                    SandboxBackendOperation,
                    "cancel_operations",
                )
                for operation in self.cancel_operations
            ),
        )
        object.__setattr__(
            self,
            "timeout_operations",
            tuple(
                _enum_value(
                    operation,
                    SandboxBackendOperation,
                    "timeout_operations",
                )
                for operation in self.timeout_operations
            ),
        )
        object.__setattr__(
            self,
            "operation_delay_seconds",
            MappingProxyType(_operation_delays(self.operation_delay_seconds)),
        )
        for chunk in self.stream_chunks:
            assert isinstance(chunk, SandboxStreamChunk)
        _assert_non_negative_number(
            self.stream_delay_seconds,
            "stream_delay_seconds",
        )
        assert isinstance(
            self.wait_exit_code, int
        ), "wait_exit_code must be integer"
        object.__setattr__(
            self,
            "output_files",
            MappingProxyType(_bytes_mapping(self.output_files, "outputs")),
        )
        object.__setattr__(
            self,
            "denied_paths",
            _path_tuple(self.denied_paths, "denied_paths"),
        )
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        assert (
            self.max_concurrent_executions > 0
        ), "max_concurrent_executions must be positive"
        object.__setattr__(self, "stream_chunks", tuple(self.stream_chunks))


@final
class SandboxFakeBackend(SandboxAsyncBackend):
    def __init__(self, script: SandboxFakeBackendScript) -> None:
        assert isinstance(script, SandboxFakeBackendScript)
        self._script = script
        self._operations: list[SandboxBackendOperation] = []
        self._max_concurrent_executions = script.max_concurrent_executions
        self._semaphore = Semaphore(script.max_concurrent_executions)
        self._counter_lock = Lock()
        self._active_executions = 0
        self._max_observed_concurrent_executions = 0

    @property
    def operations(self) -> tuple[SandboxBackendOperation, ...]:
        return tuple(self._operations)

    @property
    def backend(self) -> SandboxBackend:
        return cast(SandboxBackend, self._script.capabilities.backend)

    @property
    def max_observed_concurrent_executions(self) -> int:
        return self._max_observed_concurrent_executions

    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> SandboxBackendProbeResult:
        _assert_optional_positive_number(timeout_seconds, "timeout_seconds")
        self._record(SandboxBackendOperation.PROBE)
        try:
            if timeout_seconds is None:
                return await self._probe_once()
            return await wait_for(self._probe_once(), timeout_seconds)
        except TimeoutError:
            return SandboxBackendProbeResult(
                backend=self.backend,
                available=False,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.TIMEOUT,
                        SandboxBackendOperation.PROBE,
                        self.backend,
                        "sandbox probe exceeded timeout",
                        retryable=True,
                    ),
                ),
            )

    async def execute(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxExecutionResult:
        assert isinstance(plan, SandboxExecutionPlan)
        if not self._script.available:
            return _result(
                SandboxResultStatus.FAILED,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        SandboxBackendOperation.START,
                        self.backend,
                        "sandbox backend is unavailable",
                        retryable=True,
                    ),
                ),
            )
        if plan.settings.backend is not self.backend:
            return _result(
                SandboxResultStatus.DENIED,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
                        SandboxBackendOperation.START,
                        self.backend,
                        "sandbox plan backend does not match fake backend",
                    ),
                ),
            )
        capability_diagnostics = _capability_diagnostics(
            plan,
            self._script.capabilities,
        )
        if capability_diagnostics:
            return _result(
                SandboxResultStatus.DENIED,
                diagnostics=capability_diagnostics,
            )
        if not await self._try_enter_concurrency_slot():
            return _result(
                SandboxResultStatus.DENIED,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.CONCURRENCY_LIMIT,
                        SandboxBackendOperation.START,
                        self.backend,
                        "fake sandbox concurrency limit reached",
                        retryable=True,
                    ),
                ),
            )
        await self._semaphore.acquire()
        try:
            result = await self._execute_with_timeout(plan)
            cleanup_diagnostic = await self._cleanup_with_budget(plan)
            if cleanup_diagnostic is None:
                return result
            return replace(
                result,
                status=SandboxResultStatus.FAILED,
                diagnostics=tuple(result.diagnostics) + (cleanup_diagnostic,),
                cleanup_uncertain=True,
            )
        finally:
            self._semaphore.release()
            await self._leave_concurrency_slot()

    async def _probe_once(self) -> SandboxBackendProbeResult:
        if self._script.probe_delay_seconds:
            await async_sleep(self._script.probe_delay_seconds)
        if not self._script.available:
            return SandboxBackendProbeResult(
                backend=self.backend,
                available=False,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        SandboxBackendOperation.PROBE,
                        self.backend,
                        "sandbox backend is unavailable",
                        retryable=True,
                    ),
                ),
            )
        return SandboxBackendProbeResult(
            backend=self.backend,
            available=True,
            capabilities=self._script.capabilities,
        )

    async def _execute_with_timeout(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxExecutionResult:
        timeout_seconds = plan.settings.profile.resources.timeout_seconds
        try:
            if timeout_seconds is None:
                return await self._execute_body(plan)
            return await wait_for(
                self._execute_body(plan),
                timeout_seconds,
            )
        except SandboxBackendError as exc:
            status = (
                SandboxResultStatus.DENIED
                if exc.diagnostic.code
                in {
                    SandboxBackendDiagnosticCode.EXECUTABLE_DENIED,
                    SandboxBackendDiagnosticCode.PATH_DENIED,
                }
                else SandboxResultStatus.FAILED
            )
            return _result(status, diagnostics=(exc.diagnostic,))
        except TimeoutError:
            return _result(
                SandboxResultStatus.TIMED_OUT,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.TIMEOUT,
                        SandboxBackendOperation.WAIT,
                        self.backend,
                        "sandbox execution exceeded timeout",
                        retryable=True,
                    ),
                ),
            )
        except CancelledError:
            return _result(
                SandboxResultStatus.CANCELLED,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.CANCELLED,
                        SandboxBackendOperation.WAIT,
                        self.backend,
                        "sandbox execution was cancelled",
                        retryable=True,
                    ),
                ),
            )

    async def _execute_body(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxExecutionResult:
        await self._enter(SandboxBackendOperation.PREPARE_PROFILE)
        self._assert_script_paths_allowed(plan)
        await self._enter(SandboxBackendOperation.START)
        stdout, stderr, chunks, stream_diagnostic = await self._stream(plan)
        await self._enter(SandboxBackendOperation.WAIT)
        status = (
            SandboxResultStatus.COMPLETED
            if self._script.wait_exit_code == 0
            else SandboxResultStatus.FAILED
        )
        diagnostics: tuple[SandboxBackendDiagnostic, ...] = ()
        if status is SandboxResultStatus.FAILED:
            diagnostics = (
                _diagnostic(
                    SandboxBackendDiagnosticCode.EXECUTION_FAILED,
                    SandboxBackendOperation.WAIT,
                    self.backend,
                    "sandbox process exited with "
                    f"{self._script.wait_exit_code}",
                ),
            )
        if stream_diagnostic is not None:
            diagnostics = diagnostics + (stream_diagnostic,)
        artifacts, output_diagnostic = await self._collect_outputs(plan)
        if output_diagnostic is not None:
            status = SandboxResultStatus.FAILED
            diagnostics = diagnostics + (output_diagnostic,)
        return SandboxExecutionResult(
            status=status,
            exit_code=self._script.wait_exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostics=diagnostics,
            stream_chunks=chunks,
            output_artifacts=artifacts,
            stream_truncated=stream_diagnostic is not None,
        )

    async def _stream(
        self,
        plan: SandboxExecutionPlan,
    ) -> tuple[
        bytes,
        bytes,
        tuple[SandboxStreamChunk, ...],
        SandboxBackendDiagnostic | None,
    ]:
        await self._enter(SandboxBackendOperation.STREAM)
        stdout = bytearray()
        stderr = bytearray()
        chunks: list[SandboxStreamChunk] = []
        truncated = False
        stdout_limit = min(
            plan.stream_buffer_bytes,
            plan.settings.profile.output.max_stdout_bytes,
        )
        stderr_limit = min(
            plan.stream_buffer_bytes,
            plan.settings.profile.output.max_stderr_bytes,
        )
        for chunk in self._script.stream_chunks:
            if self._script.stream_delay_seconds:
                await async_sleep(self._script.stream_delay_seconds)
            chunks.append(chunk)
            stream = cast(SandboxBackendStream, chunk.stream)
            if stream is SandboxBackendStream.STDOUT:
                truncated = (
                    _append_bounded(
                        stdout,
                        chunk.content,
                        stdout_limit,
                    )
                    or truncated
                )
            else:
                truncated = (
                    _append_bounded(
                        stderr,
                        chunk.content,
                        stderr_limit,
                    )
                    or truncated
                )
        diagnostic = (
            _diagnostic(
                SandboxBackendDiagnosticCode.STREAM_TRUNCATED,
                SandboxBackendOperation.STREAM,
                self.backend,
                "sandbox stream exceeded configured buffer",
            )
            if truncated
            else None
        )
        return bytes(stdout), bytes(stderr), tuple(chunks), diagnostic

    async def _collect_outputs(
        self,
        plan: SandboxExecutionPlan,
    ) -> tuple[
        tuple[SandboxOutputArtifact, ...],
        SandboxBackendDiagnostic | None,
    ]:
        if not plan.collect_outputs and not self._script.output_files:
            return (), None
        await self._enter(SandboxBackendOperation.COLLECT_OUTPUTS)
        output_policy = plan.settings.profile.output
        if not plan.collect_outputs or not output_policy.allow_artifacts:
            return (), _diagnostic(
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
                SandboxBackendOperation.COLLECT_OUTPUTS,
                self.backend,
                "sandbox output collection is disabled",
            )
        artifacts: list[SandboxOutputArtifact] = []
        total_bytes = 0
        for path, content in self._script.output_files.items():
            if _unsafe_output_path(path):
                return (), _diagnostic(
                    SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
                    SandboxBackendOperation.COLLECT_OUTPUTS,
                    self.backend,
                    f"sandbox output path is unsafe: {path}",
                )
            total_bytes += len(content)
            if total_bytes > output_policy.max_artifact_bytes:
                return (), _diagnostic(
                    SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
                    SandboxBackendOperation.COLLECT_OUTPUTS,
                    self.backend,
                    "sandbox outputs exceed artifact byte limit",
                )
            artifacts.append(SandboxOutputArtifact(path=path, content=content))
        return tuple(artifacts), None

    async def _cleanup_with_budget(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxBackendDiagnostic | None:
        try:
            return await wait_for(
                self._cleanup_once(),
                plan.cleanup_budget_seconds,
            )
        except TimeoutError:
            return _diagnostic(
                SandboxBackendDiagnosticCode.CLEANUP_FAILED,
                SandboxBackendOperation.CLEANUP,
                self.backend,
                "sandbox cleanup exceeded cleanup budget",
                retryable=True,
            )
        except CancelledError:
            return _diagnostic(
                SandboxBackendDiagnosticCode.CANCELLED,
                SandboxBackendOperation.CLEANUP,
                self.backend,
                "sandbox cleanup was cancelled",
                retryable=True,
            )

    async def _cleanup_once(self) -> SandboxBackendDiagnostic | None:
        try:
            await self._enter(SandboxBackendOperation.CLEANUP)
        except SandboxBackendError as exc:
            return exc.diagnostic
        except TimeoutError:
            return _diagnostic(
                SandboxBackendDiagnosticCode.CLEANUP_FAILED,
                SandboxBackendOperation.CLEANUP,
                self.backend,
                "sandbox cleanup timed out",
                retryable=True,
            )
        if self._script.cleanup_uncertain:
            return _diagnostic(
                SandboxBackendDiagnosticCode.CLEANUP_FAILED,
                SandboxBackendOperation.CLEANUP,
                self.backend,
                "sandbox cleanup state is uncertain",
                retryable=True,
            )
        return None

    async def _enter(self, operation: SandboxBackendOperation) -> None:
        self._record(operation)
        delay_seconds = self._script.operation_delay_seconds.get(operation)
        if delay_seconds:
            await async_sleep(delay_seconds)
        if operation in self._script.cancel_operations:
            raise CancelledError
        if operation in self._script.timeout_operations:
            raise TimeoutError
        code = self._script.operation_diagnostics.get(operation)
        if code is not None:
            diagnostic_code = cast(SandboxBackendDiagnosticCode, code)
            raise SandboxBackendError(
                _diagnostic(
                    diagnostic_code,
                    operation,
                    self.backend,
                    f"{operation.value} failed with {diagnostic_code.value}",
                    retryable=_retryable(diagnostic_code),
                )
            )

    def _assert_script_paths_allowed(
        self,
        plan: SandboxExecutionPlan,
    ) -> None:
        paths = (plan.request.cwd,) + tuple(plan.request.argv)
        for denied_path in self._script.denied_paths:
            if any(
                _path_inside(candidate, denied_path) for candidate in paths
            ):
                raise SandboxBackendError(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.PATH_DENIED,
                        SandboxBackendOperation.PREPARE_PROFILE,
                        self.backend,
                        f"sandbox path is denied: {denied_path}",
                    )
                )

    async def _try_enter_concurrency_slot(self) -> bool:
        async with self._counter_lock:
            if self._active_executions >= self._max_concurrent_executions:
                return False
            self._active_executions += 1
            if (
                self._active_executions
                > self._max_observed_concurrent_executions
            ):
                self._max_observed_concurrent_executions = (
                    self._active_executions
                )
            return True

    async def _leave_concurrency_slot(self) -> None:
        async with self._counter_lock:
            self._active_executions -= 1

    def _record(self, operation: SandboxBackendOperation) -> None:
        self._operations.append(operation)


def sandbox_backend_capability_profiles(
    backend: SandboxBackend | str | None = None,
) -> tuple[SandboxBackendCapabilityProfile, ...]:
    if backend is None:
        return _SANDBOX_BACKEND_CAPABILITY_PROFILES
    selected = _enum_value(backend, SandboxBackend, "backend")
    return tuple(
        profile
        for profile in _SANDBOX_BACKEND_CAPABILITY_PROFILES
        if profile.backend is selected
    )


def sandbox_backend_capability_profile(
    profile_id: str,
) -> SandboxBackendCapabilityProfile:
    _assert_non_empty_string(profile_id, "profile_id")
    for profile in _SANDBOX_BACKEND_CAPABILITY_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    assert False, "sandbox backend capability profile is unknown"


def sandbox_backend_probe_from_profile(
    profile_id: str,
    *,
    available: bool = False,
    diagnostics: Sequence[SandboxBackendDiagnostic] = (),
) -> SandboxBackendProbeResult:
    return sandbox_backend_capability_profile(profile_id).probe(
        available=available,
        diagnostics=diagnostics,
    )


def select_sandbox_backend(
    plan: SandboxExecutionPlan,
    probes: Sequence[SandboxBackendProbeResult],
) -> SandboxBackendSelection:
    assert isinstance(plan, SandboxExecutionPlan)
    for probe in probes:
        assert isinstance(probe, SandboxBackendProbeResult)
    requested = cast(SandboxBackend, plan.settings.backend)
    candidates = [probe for probe in probes if probe.backend is requested]
    diagnostics: list[SandboxBackendDiagnostic] = []
    if not candidates:
        diagnostics.append(
            _diagnostic(
                SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                SandboxBackendOperation.PROBE,
                requested,
                f"configured sandbox backend {requested.value} is unavailable",
                retryable=True,
            )
        )
    for probe in candidates:
        diagnostics.extend(probe.diagnostics)
        if not probe.ok:
            continue
        assert probe.capabilities is not None
        capability_diagnostics = _capability_diagnostics(
            plan,
            probe.capabilities,
        )
        if capability_diagnostics:
            diagnostics.extend(capability_diagnostics)
            continue
        return SandboxBackendSelection(
            backend=probe.backend,
            capabilities=probe.capabilities,
        )
    return SandboxBackendSelection(backend=None, diagnostics=diagnostics)


def _capability_diagnostics(
    plan: SandboxExecutionPlan,
    capabilities: SandboxBackendCapabilities,
) -> tuple[SandboxBackendDiagnostic, ...]:
    backend = cast(SandboxBackend, capabilities.backend)
    profile = plan.settings.profile
    diagnostics: list[SandboxBackendDiagnostic] = []
    if not capabilities.sandbox_executable_available:
        diagnostics.append(
            _capability_mismatch(backend, "sandbox executable is unavailable")
        )
    filesystem = capabilities.filesystem
    if profile.read_roots and not filesystem.read_roots:
        diagnostics.append(
            _capability_mismatch(backend, "read roots unsupported")
        )
    if profile.write_roots and not filesystem.write_roots:
        diagnostics.append(
            _capability_mismatch(backend, "write roots unsupported")
        )
    if profile.deny_roots and not filesystem.deny_roots:
        diagnostics.append(
            _capability_mismatch(backend, "deny roots unsupported")
        )
    network_mode = cast(SandboxNetworkMode, profile.network.mode)
    if network_mode not in capabilities.network_modes:
        diagnostics.append(
            _capability_mismatch(
                backend,
                f"network mode {network_mode.value} is unsupported",
            )
        )
    process = capabilities.process
    if profile.resources.pids is not None and not process.process_limits:
        diagnostics.append(
            _capability_mismatch(backend, "pid limits unsupported")
        )
    if (
        profile.child_processes is SandboxChildProcessPolicy.ALLOW
        and not process.child_processes
    ):
        diagnostics.append(
            _capability_mismatch(backend, "child process policy unsupported")
        )
    if (
        profile.inherited_fds is not SandboxInheritedFdPolicy.STDIO
        and not process.inherited_fds
    ):
        diagnostics.append(
            _capability_mismatch(backend, "inherited fd policy unsupported")
        )
    temp_output = capabilities.temp_output
    if profile.scratch_roots and not temp_output.temp_dirs:
        diagnostics.append(
            _capability_mismatch(backend, "temp dirs unsupported")
        )
    if profile.output_roots and not temp_output.output_dirs:
        diagnostics.append(
            _capability_mismatch(backend, "output dirs unsupported")
        )
    if (
        profile.cleanup is not SandboxCleanupPolicy.PRESERVE
        and not temp_output.cleanup_budget
    ):
        diagnostics.append(
            _capability_mismatch(backend, "cleanup budgets unsupported")
        )
    return tuple(diagnostics)


def _capability_mismatch(
    backend: SandboxBackend,
    message: str,
) -> SandboxBackendDiagnostic:
    return _diagnostic(
        SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
        SandboxBackendOperation.PROBE,
        backend,
        message,
    )


def _diagnostic(
    code: SandboxBackendDiagnosticCode | str,
    operation: SandboxBackendOperation | str,
    backend: SandboxBackend | str | None,
    message: str,
    *,
    retryable: bool = False,
) -> SandboxBackendDiagnostic:
    return SandboxBackendDiagnostic(
        code=code,
        operation=operation,
        backend=backend,
        message=message,
        retryable=retryable,
    )


def _result(
    status: SandboxResultStatus,
    *,
    diagnostics: Sequence[SandboxBackendDiagnostic] = (),
) -> SandboxExecutionResult:
    return SandboxExecutionResult(status=status, diagnostics=diagnostics)


def _enum_value(
    value: EnumValue | str,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    assert isinstance(value, str), f"{field_name} must be a string"
    try:
        return enum_type(value)
    except ValueError as exc:
        raise AssertionError(
            f"{field_name} contains unsupported value"
        ) from exc


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    normalized: list[str] = []
    for item in value:
        _assert_non_empty_string(item, field_name)
        normalized.append(cast(str, item))
    return tuple(normalized)


def _path_tuple(value: object, field_name: str) -> tuple[str, ...]:
    paths = _string_tuple(value, field_name)
    for path in paths:
        assert "\x00" not in path, f"{field_name} must not contain NUL"
    return paths


def _bytes_mapping(
    value: Mapping[str, bytes], field_name: str
) -> dict[str, bytes]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    normalized: dict[str, bytes] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        assert isinstance(item, bytes), f"{field_name}.{key} must be bytes"
        normalized[key] = item
    return normalized


def _operation_diagnostics(
    value: Mapping[
        SandboxBackendOperation | str, SandboxBackendDiagnosticCode | str
    ],
) -> dict[SandboxBackendOperation, SandboxBackendDiagnosticCode]:
    normalized: dict[SandboxBackendOperation, SandboxBackendDiagnosticCode] = (
        {}
    )
    for operation, code in value.items():
        normalized[
            _enum_value(operation, SandboxBackendOperation, "operation")
        ] = _enum_value(code, SandboxBackendDiagnosticCode, "code")
    return normalized


def _operation_delays(
    value: Mapping[SandboxBackendOperation | str, float],
) -> dict[SandboxBackendOperation, float]:
    normalized: dict[SandboxBackendOperation, float] = {}
    for operation, seconds in value.items():
        _assert_non_negative_number(seconds, "operation delay")
        normalized[
            _enum_value(operation, SandboxBackendOperation, "operation")
        ] = float(seconds)
    return normalized


def _assert_non_negative_number(value: object, field_name: str) -> None:
    assert isinstance(value, int | float), f"{field_name} must be numeric"
    assert value >= 0, f"{field_name} must not be negative"


def _append_bounded(
    target: bytearray,
    content: bytes,
    limit: int,
) -> bool:
    remaining = limit - len(target)
    if remaining <= 0:
        return bool(content)
    target.extend(content[:remaining])
    return len(content) > remaining


def _unsafe_output_path(path: str) -> bool:
    normalized = normalize_posix_path(path)
    return (
        path.startswith("/")
        or path.startswith("\\")
        or normalized == "."
        or normalized == ".."
        or normalized.startswith("../")
        or "\\..\\" in path
    )


def _path_inside(path: str, root: str) -> bool:
    normalized_path = _path_parts(path)
    normalized_root = _path_parts(root)
    return len(normalized_path) >= len(normalized_root) and (
        normalized_path[: len(normalized_root)] == normalized_root
    )


def _path_parts(path: str) -> tuple[str, ...]:
    normalized = normalize_posix_path(path)
    return tuple(part for part in normalized.split("/") if part)


def _retryable(code: SandboxBackendDiagnosticCode) -> bool:
    return code in {
        SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        SandboxBackendDiagnosticCode.CLEANUP_FAILED,
        SandboxBackendDiagnosticCode.CANCELLED,
        SandboxBackendDiagnosticCode.TIMEOUT,
        SandboxBackendDiagnosticCode.CONCURRENCY_LIMIT,
    }


_SANDBOX_BACKEND_CAPABILITY_PROFILES = (
    SandboxBackendCapabilityProfile(
        profile_id="seatbelt-darwin",
        capabilities=SandboxBackendCapabilities(
            backend=SandboxBackend.SEATBELT,
            host_os="darwin",
            architecture="arm64",
            runtime_name="Apple sandbox-exec",
            sandbox_executable="/usr/bin/sandbox-exec",
            sandbox_executable_available=True,
            filesystem=SandboxFilesystemControls(
                read_roots=True,
                write_roots=True,
                deny_roots=True,
            ),
            network_modes=(
                SandboxNetworkMode.NONE,
                SandboxNetworkMode.LOOPBACK,
            ),
            process=SandboxProcessControls(
                process_limits=False,
                child_processes=False,
                inherited_fds=False,
            ),
            temp_output=SandboxTempOutputMapping(
                temp_dirs=True,
                output_dirs=True,
                cleanup_budget=True,
            ),
            unsupported_controls=(
                "network_allowlist",
                "pid_limits",
                "inherited_fds",
            ),
        ),
    ),
    SandboxBackendCapabilityProfile(
        profile_id="bubblewrap-linux",
        capabilities=SandboxBackendCapabilities(
            backend=SandboxBackend.BUBBLEWRAP,
            host_os="linux",
            architecture="amd64",
            runtime_name="bubblewrap",
            sandbox_executable="/usr/bin/bwrap",
            sandbox_executable_available=True,
            filesystem=SandboxFilesystemControls(
                read_roots=True,
                write_roots=True,
                deny_roots=True,
            ),
            network_modes=(
                SandboxNetworkMode.NONE,
                SandboxNetworkMode.LOOPBACK,
                SandboxNetworkMode.FULL,
            ),
            process=SandboxProcessControls(
                process_limits=True,
                child_processes=True,
                inherited_fds=True,
            ),
            temp_output=SandboxTempOutputMapping(
                temp_dirs=True,
                output_dirs=True,
                cleanup_budget=True,
            ),
            unsupported_controls=("network_allowlist",),
        ),
    ),
)
