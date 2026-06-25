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
from asyncio import (
    CancelledError,
    Lock,
    Semaphore,
    StreamReader,
    TimeoutError,
    create_subprocess_exec,
    gather,
    wait_for,
)
from asyncio import sleep as async_sleep
from asyncio.subprocess import PIPE, Process
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from os import X_OK, access
from pathlib import Path
from platform import machine, system
from posixpath import normpath as normalize_posix_path
from shutil import rmtree, which
from tempfile import mkdtemp
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)
ResourceGetLimit = Callable[[int], tuple[int, int]]
ResourceSetLimit = Callable[[int, tuple[int, int]], None]

_RESOURCE_RLIMIT_NPROC: int
_RESOURCE_GETRLIMIT: ResourceGetLimit | None
_RESOURCE_SETRLIMIT: ResourceSetLimit | None

try:
    from resource import RLIMIT_NPROC as _IMPORTED_RLIMIT_NPROC
    from resource import getrlimit as _imported_getrlimit
    from resource import setrlimit as _imported_setrlimit
except ImportError:
    _RESOURCE_RLIMIT_NPROC = -1
    _RESOURCE_GETRLIMIT = None
    _RESOURCE_SETRLIMIT = None
else:
    _RESOURCE_RLIMIT_NPROC = _IMPORTED_RLIMIT_NPROC
    _RESOURCE_GETRLIMIT = _imported_getrlimit
    _RESOURCE_SETRLIMIT = _imported_setrlimit


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
class SandboxSubprocessRequest:
    operation: SandboxBackendOperation | str
    label: str
    argv: Sequence[str]
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_seconds: float | None = None
    stdout_limit_bytes: int = 65536
    stderr_limit_bytes: int = 32768
    process_limit: int | None = None
    close_fds: bool = True
    pass_fds: Sequence[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _enum_value(self.operation, SandboxBackendOperation, "operation"),
        )
        _assert_non_empty_string(self.label, "label")
        object.__setattr__(self, "argv", _string_tuple(self.argv, "argv"))
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(_string_mapping(self.environment, "environment")),
        )
        if self.cwd is not None:
            _assert_non_empty_string(self.cwd, "cwd")
            assert self.cwd.startswith("/"), "cwd must be absolute"
            assert "\x00" not in self.cwd, "cwd must not contain NUL"
            object.__setattr__(
                self,
                "cwd",
                normalize_posix_path(self.cwd),
            )
        _assert_optional_positive_number(
            self.timeout_seconds,
            "timeout_seconds",
        )
        assert (
            self.stdout_limit_bytes > 0
        ), "stdout_limit_bytes must be positive"
        assert (
            self.stderr_limit_bytes > 0
        ), "stderr_limit_bytes must be positive"
        if self.process_limit is not None:
            assert self.process_limit > 0, "process_limit must be positive"
        _assert_bool(self.close_fds, "close_fds")
        pass_fds: list[int] = []
        for fd in self.pass_fds:
            assert isinstance(fd, int), "pass_fds must contain integers"
            assert fd >= 0, "pass_fds must not contain negative integers"
            pass_fds.append(fd)
        object.__setattr__(self, "pass_fds", tuple(pass_fds))

    def to_dict(self) -> dict[str, object]:
        operation = cast(SandboxBackendOperation, self.operation)
        return {
            "operation": operation.value,
            "label": self.label,
            "argv": list(self.argv),
            "environment": dict(self.environment),
            "cwd": self.cwd,
            "timeout_seconds": self.timeout_seconds,
            "stdout_limit_bytes": self.stdout_limit_bytes,
            "stderr_limit_bytes": self.stderr_limit_bytes,
            "process_limit": self.process_limit,
            "close_fds": self.close_fds,
            "pass_fds": list(self.pass_fds),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxSubprocessResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.exit_code, int), "exit_code must be integer"
        assert isinstance(self.stdout, bytes), "stdout must be bytes"
        assert isinstance(self.stderr, bytes), "stderr must be bytes"
        _assert_bool(self.stdout_truncated, "stdout_truncated")
        _assert_bool(self.stderr_truncated, "stderr_truncated")


SandboxCommandRunner = Callable[
    [SandboxSubprocessRequest],
    Awaitable[SandboxSubprocessResult],
]
SandboxCleanupHandler = Callable[
    [Sequence[str]],
    Awaitable[bool],
]


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


@final
class SeatbeltSandboxBackend(SandboxAsyncBackend):
    def __init__(
        self,
        *,
        sandbox_executable: str | None = None,
        host_os: str | None = None,
        architecture: str | None = None,
        executable_available: bool | None = None,
        command_runner: SandboxCommandRunner | None = None,
        cleanup_handler: SandboxCleanupHandler | None = None,
    ) -> None:
        self._sandbox_executable = (
            sandbox_executable
            or which("sandbox-exec")
            or "/usr/bin/sandbox-exec"
        )
        self._host_os = (host_os or system()).lower()
        self._architecture = architecture or machine()
        self._executable_available = executable_available
        self._command_runner = command_runner or _default_command_runner
        self._cleanup_handler = cleanup_handler or _default_cleanup_handler

    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> SandboxBackendProbeResult:
        _assert_optional_positive_number(timeout_seconds, "timeout_seconds")
        try:
            if timeout_seconds is None:
                return await self._probe_once()
            return await wait_for(self._probe_once(), timeout_seconds)
        except TimeoutError:
            return SandboxBackendProbeResult(
                backend=SandboxBackend.SEATBELT,
                available=False,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.TIMEOUT,
                        SandboxBackendOperation.PROBE,
                        SandboxBackend.SEATBELT,
                        "seatbelt probe exceeded timeout",
                        retryable=True,
                    ),
                ),
            )

    async def execute(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxExecutionResult:
        assert isinstance(plan, SandboxExecutionPlan)
        if plan.settings.backend is not SandboxBackend.SEATBELT:
            return _backend_mismatch_result(
                SandboxBackend.SEATBELT,
                "sandbox plan backend does not match seatbelt backend",
            )
        probe = await self.probe()
        selection = select_sandbox_backend(plan, (probe,))
        if not selection.ok:
            return _selection_failure_result(selection)
        assert selection.capabilities is not None
        path_diagnostic = _prepare_and_validate_paths(
            plan,
            SandboxBackend.SEATBELT,
        )
        if path_diagnostic is not None:
            result = _result(
                SandboxResultStatus.DENIED,
                diagnostics=(path_diagnostic,),
            )
            return await _finish_with_cleanup(
                result,
                SandboxBackend.SEATBELT,
                self._cleanup_handler,
                _cleanup_paths(plan),
                plan.cleanup_budget_seconds,
            )
        request = SandboxSubprocessRequest(
            operation=SandboxBackendOperation.START,
            label="seatbelt_execute",
            argv=(
                self._sandbox_executable,
                "-p",
                generate_seatbelt_profile(plan),
                "--",
            )
            + tuple(plan.request.argv),
            environment=_execution_environment(plan),
            cwd=plan.request.cwd,
            timeout_seconds=plan.settings.profile.resources.timeout_seconds,
            stdout_limit_bytes=_stdout_limit(plan),
            stderr_limit_bytes=_stderr_limit(plan),
            close_fds=True,
            pass_fds=(),
        )
        cleanup_paths = _cleanup_paths(plan)
        return await _execute_subprocess_request(
            plan,
            SandboxBackend.SEATBELT,
            request,
            self._command_runner,
            self._cleanup_handler,
            cleanup_paths,
        )

    async def _probe_once(self) -> SandboxBackendProbeResult:
        if self._host_os != "darwin":
            return _unavailable_probe(
                SandboxBackend.SEATBELT,
                "seatbelt is only available on macOS",
            )
        if not self._is_executable_available():
            return _unavailable_probe(
                SandboxBackend.SEATBELT,
                "sandbox-exec executable is unavailable",
            )
        try:
            result = await self._command_runner(
                SandboxSubprocessRequest(
                    operation=SandboxBackendOperation.PROBE,
                    label="seatbelt_basic_profile",
                    argv=(
                        self._sandbox_executable,
                        "-p",
                        "(version 1)\n(allow default)\n",
                        "/usr/bin/true",
                    ),
                    environment=_probe_environment(),
                    timeout_seconds=2,
                )
            )
        except TimeoutError:
            return _unavailable_probe(
                SandboxBackend.SEATBELT,
                "seatbelt probe command timed out",
                code=SandboxBackendDiagnosticCode.TIMEOUT,
                retryable=True,
            )
        except OSError as exc:
            return _unavailable_probe(
                SandboxBackend.SEATBELT,
                f"seatbelt probe command failed: {exc}",
            )
        if result.exit_code != 0:
            return _unavailable_probe(
                SandboxBackend.SEATBELT,
                "seatbelt rejected the basic probe profile",
            )
        return SandboxBackendProbeResult(
            backend=SandboxBackend.SEATBELT,
            available=True,
            capabilities=SandboxBackendCapabilities(
                backend=SandboxBackend.SEATBELT,
                host_os="darwin",
                architecture=self._architecture,
                runtime_name="Apple sandbox-exec",
                sandbox_executable=self._sandbox_executable,
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
                    "cpu_limits",
                    "memory_limits",
                    "network_allowlist",
                    "pid_limits",
                    "inherited_fds",
                ),
            ),
        )

    def _is_executable_available(self) -> bool:
        if self._executable_available is not None:
            return self._executable_available
        path = Path(self._sandbox_executable)
        return path.is_file() and access(path, X_OK)


@final
class BubblewrapSandboxBackend(SandboxAsyncBackend):
    def __init__(
        self,
        *,
        sandbox_executable: str | None = None,
        host_os: str | None = None,
        architecture: str | None = None,
        executable_available: bool | None = None,
        process_limits_supported: bool | None = None,
        command_runner: SandboxCommandRunner | None = None,
        cleanup_handler: SandboxCleanupHandler | None = None,
    ) -> None:
        self._sandbox_executable = (
            sandbox_executable or which("bwrap") or "/usr/bin/bwrap"
        )
        self._host_os = (host_os or system()).lower()
        self._architecture = architecture or machine()
        self._executable_available = executable_available
        self._process_limits_supported = process_limits_supported
        self._command_runner = command_runner or _default_command_runner
        self._cleanup_handler = cleanup_handler or _default_cleanup_handler

    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> SandboxBackendProbeResult:
        _assert_optional_positive_number(timeout_seconds, "timeout_seconds")
        try:
            if timeout_seconds is None:
                return await self._probe_once()
            return await wait_for(self._probe_once(), timeout_seconds)
        except TimeoutError:
            return SandboxBackendProbeResult(
                backend=SandboxBackend.BUBBLEWRAP,
                available=False,
                diagnostics=(
                    _diagnostic(
                        SandboxBackendDiagnosticCode.TIMEOUT,
                        SandboxBackendOperation.PROBE,
                        SandboxBackend.BUBBLEWRAP,
                        "bubblewrap probe exceeded timeout",
                        retryable=True,
                    ),
                ),
            )

    async def execute(
        self,
        plan: SandboxExecutionPlan,
    ) -> SandboxExecutionResult:
        assert isinstance(plan, SandboxExecutionPlan)
        if plan.settings.backend is not SandboxBackend.BUBBLEWRAP:
            return _backend_mismatch_result(
                SandboxBackend.BUBBLEWRAP,
                "sandbox plan backend does not match bubblewrap backend",
            )
        probe = await self.probe()
        selection = select_sandbox_backend(plan, (probe,))
        if not selection.ok:
            return _selection_failure_result(selection)
        assert selection.capabilities is not None
        path_diagnostic = _prepare_and_validate_paths(
            plan,
            SandboxBackend.BUBBLEWRAP,
        )
        if path_diagnostic is not None:
            result = _result(
                SandboxResultStatus.DENIED,
                diagnostics=(path_diagnostic,),
            )
            return await _finish_with_cleanup(
                result,
                SandboxBackend.BUBBLEWRAP,
                self._cleanup_handler,
                _cleanup_paths(plan),
                plan.cleanup_budget_seconds,
            )
        runtime_dir = mkdtemp(prefix="avalan-bwrap-")
        Path(runtime_dir, "deny-empty").mkdir()
        cleanup_paths = (runtime_dir,) + _cleanup_paths(plan)
        request = SandboxSubprocessRequest(
            operation=SandboxBackendOperation.START,
            label="bubblewrap_execute",
            argv=generate_bubblewrap_arguments(
                plan,
                sandbox_executable=self._sandbox_executable,
                private_runtime_dir=runtime_dir,
            ),
            environment=_execution_environment(plan),
            timeout_seconds=plan.settings.profile.resources.timeout_seconds,
            stdout_limit_bytes=_stdout_limit(plan),
            stderr_limit_bytes=_stderr_limit(plan),
            close_fds=True,
            pass_fds=(),
        )
        return await _execute_subprocess_request(
            plan,
            SandboxBackend.BUBBLEWRAP,
            request,
            self._command_runner,
            self._cleanup_handler,
            cleanup_paths,
        )

    async def _probe_once(self) -> SandboxBackendProbeResult:
        if self._host_os != "linux":
            return _unavailable_probe(
                SandboxBackend.BUBBLEWRAP,
                "bubblewrap is only available on Linux",
            )
        if not self._is_executable_available():
            return _unavailable_probe(
                SandboxBackend.BUBBLEWRAP,
                "bwrap executable is unavailable",
            )
        controls = await self._probe_controls()
        if not controls["user_namespace"]:
            return _unavailable_probe(
                SandboxBackend.BUBBLEWRAP,
                "bubblewrap user namespaces are unavailable",
            )
        if not controls["mount"]:
            return _unavailable_probe(
                SandboxBackend.BUBBLEWRAP,
                "bubblewrap bind mounts are unavailable",
            )
        if not controls["proc"]:
            return _unavailable_probe(
                SandboxBackend.BUBBLEWRAP,
                "bubblewrap /proc mounting is unavailable",
            )
        network_modes: tuple[SandboxNetworkMode, ...] = (
            (
                SandboxNetworkMode.NONE,
                SandboxNetworkMode.LOOPBACK,
                SandboxNetworkMode.FULL,
            )
            if controls["network_namespace"]
            else (SandboxNetworkMode.FULL,)
        )
        unsupported_controls = ["cpu_limits", "memory_limits"]
        if not controls["network_namespace"]:
            unsupported_controls.append("network_namespaces")
        unsupported_controls.append("network_allowlist")
        unsupported_controls.append("pid_limits")
        unsupported_controls.append("child_process_denial")
        if self._supports_process_limits():
            unsupported_controls.append("pid_limits_best_effort_rlimit")
        return SandboxBackendProbeResult(
            backend=SandboxBackend.BUBBLEWRAP,
            available=True,
            capabilities=SandboxBackendCapabilities(
                backend=SandboxBackend.BUBBLEWRAP,
                host_os="linux",
                architecture=self._architecture,
                runtime_name="bubblewrap",
                sandbox_executable=self._sandbox_executable,
                sandbox_executable_available=True,
                filesystem=SandboxFilesystemControls(
                    read_roots=True,
                    write_roots=True,
                    deny_roots=True,
                ),
                network_modes=network_modes,
                process=SandboxProcessControls(
                    process_limits=False,
                    child_processes=True,
                    inherited_fds=True,
                ),
                temp_output=SandboxTempOutputMapping(
                    temp_dirs=True,
                    output_dirs=True,
                    cleanup_budget=True,
                ),
                unsupported_controls=tuple(unsupported_controls),
            ),
        )

    async def _probe_controls(self) -> dict[str, bool]:
        labels = {
            "user_namespace": (
                self._sandbox_executable,
                "--unshare-user",
                "--uid",
                "0",
                "--gid",
                "0",
                "--ro-bind",
                "/",
                "/",
                "/bin/true",
            ),
            "mount": (
                self._sandbox_executable,
                "--ro-bind",
                "/",
                "/",
                "/bin/true",
            ),
            "network_namespace": (
                self._sandbox_executable,
                "--unshare-net",
                "--ro-bind",
                "/",
                "/",
                "/bin/true",
            ),
            "proc": (
                self._sandbox_executable,
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "/bin/test",
                "-r",
                "/proc/self/status",
            ),
        }
        results: dict[str, bool] = {}
        for label, argv in labels.items():
            try:
                result = await self._command_runner(
                    SandboxSubprocessRequest(
                        operation=SandboxBackendOperation.PROBE,
                        label=f"bubblewrap_{label}",
                        argv=argv,
                        environment=_probe_environment(),
                        timeout_seconds=2,
                    )
                )
                results[label] = result.exit_code == 0
            except (OSError, TimeoutError):
                results[label] = False
        return results

    def _is_executable_available(self) -> bool:
        if self._executable_available is not None:
            return self._executable_available
        path = Path(self._sandbox_executable)
        return path.is_file() and access(path, X_OK)

    def _supports_process_limits(self) -> bool:
        if self._process_limits_supported is not None:
            return self._process_limits_supported
        return _process_limits_supported()


def generate_seatbelt_profile(plan: SandboxExecutionPlan) -> str:
    assert isinstance(plan, SandboxExecutionPlan)
    assert (
        plan.settings.backend is SandboxBackend.SEATBELT
    ), "seatbelt profiles require a seatbelt plan"
    _assert_policy_roots_safe(plan)
    profile = plan.settings.profile
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow file-read-metadata)",
    ]
    for path in _ordered_unique(
        tuple(profile.executable_search_roots)
        + tuple(profile.trusted_executables)
        + tuple(profile.read_roots)
        + tuple(profile.write_roots)
        + _optional_path_tuple(plan.temp_dir)
        + _optional_path_tuple(plan.output_dir)
    ):
        lines.append(_seatbelt_allow_read(path))
    for path in _ordered_unique(
        tuple(profile.write_roots)
        + _optional_path_tuple(plan.temp_dir)
        + _optional_path_tuple(plan.output_dir)
    ):
        lines.append(_seatbelt_allow_write(path))
    for path in _ordered_unique(profile.deny_roots):
        lines.append(_seatbelt_deny_read(path))
        lines.append(_seatbelt_deny_write(path))
    network_mode = cast(SandboxNetworkMode, profile.network.mode)
    match network_mode:
        case SandboxNetworkMode.NONE:
            lines.append("(deny network*)")
        case SandboxNetworkMode.LOOPBACK:
            lines.append("(deny network*)")
            lines.append('(allow network-outbound (remote ip "127.0.0.1:*"))')
            lines.append('(allow network-inbound (local ip "127.0.0.1:*"))')
        case SandboxNetworkMode.ALLOWLIST:
            raise AssertionError("seatbelt network allowlists are unsupported")
        case SandboxNetworkMode.FULL:
            raise AssertionError("seatbelt full network is unsupported")
    if profile.child_processes is SandboxChildProcessPolicy.DENY:
        lines.append("(deny process-fork*)")
    return "\n".join(lines) + "\n"


def generate_bubblewrap_arguments(
    plan: SandboxExecutionPlan,
    *,
    sandbox_executable: str = "/usr/bin/bwrap",
    private_runtime_dir: str = "/tmp/avalan-bwrap",
) -> tuple[str, ...]:
    assert isinstance(plan, SandboxExecutionPlan)
    assert (
        plan.settings.backend is SandboxBackend.BUBBLEWRAP
    ), "bubblewrap arguments require a bubblewrap plan"
    _assert_non_empty_string(sandbox_executable, "sandbox_executable")
    _assert_non_empty_string(private_runtime_dir, "private_runtime_dir")
    assert sandbox_executable.startswith("/"), "sandbox_executable is absolute"
    assert private_runtime_dir.startswith(
        "/"
    ), "private_runtime_dir is absolute"
    _assert_policy_roots_safe(plan)
    profile = plan.settings.profile
    args: list[str] = [
        sandbox_executable,
        "--die-with-parent",
        "--unshare-user",
        "--uid",
        "0",
        "--gid",
        "0",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--clearenv",
    ]
    for name, value in sorted(_execution_environment(plan).items()):
        args.extend(("--setenv", name, value))
    network_mode = cast(SandboxNetworkMode, profile.network.mode)
    match network_mode:
        case SandboxNetworkMode.NONE | SandboxNetworkMode.LOOPBACK:
            args.append("--unshare-net")
        case SandboxNetworkMode.FULL:
            args.append("--share-net")
        case SandboxNetworkMode.ALLOWLIST:
            raise AssertionError(
                "bubblewrap network allowlists are unsupported"
            )
    for path in _ordered_unique(
        tuple(profile.executable_search_roots)
        + tuple(profile.trusted_executables)
        + tuple(profile.read_roots)
    ):
        args.extend(("--ro-bind", path, path))
    for path in _ordered_unique(
        tuple(profile.write_roots)
        + _optional_path_tuple(plan.temp_dir)
        + _optional_path_tuple(plan.output_dir)
    ):
        args.extend(("--bind", path, path))
    if not _path_inside_any("/proc", profile.deny_roots):
        args.extend(("--proc", "/proc"))
    args.extend(("--dev", "/dev"))
    deny_dir = normalize_posix_path(f"{private_runtime_dir}/deny-empty")
    for path in _ordered_unique(profile.deny_roots):
        args.extend(("--ro-bind", deny_dir, path))
    args.extend(("--chdir", plan.request.cwd, "--"))
    args.extend(plan.request.argv)
    return tuple(args)


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
        backend is SandboxBackend.BUBBLEWRAP
        and profile.child_processes is SandboxChildProcessPolicy.DENY
    ):
        diagnostics.append(
            _capability_mismatch(
                backend,
                "child process denial unsupported",
            )
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


def _path_inside_any(path: str, roots: Sequence[str]) -> bool:
    return any(_path_inside(path, root) for root in roots)


def _path_parts(path: str) -> tuple[str, ...]:
    normalized = normalize_posix_path(path)
    return tuple(part for part in normalized.split("/") if part)


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name}.{key}")
        normalized[key] = item
    return normalized


def _ordered_unique(paths: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        normalized = normalize_posix_path(path)
        if normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return tuple(ordered)


def _optional_path_tuple(path: str | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return (path,)


def _seatbelt_allow_read(path: str) -> str:
    return f"(allow file-read* (subpath {_seatbelt_string(path)}))"


def _seatbelt_allow_write(path: str) -> str:
    return f"(allow file-write* (subpath {_seatbelt_string(path)}))"


def _seatbelt_deny_read(path: str) -> str:
    return f"(deny file-read* (subpath {_seatbelt_string(path)}))"


def _seatbelt_deny_write(path: str) -> str:
    return f"(deny file-write* (subpath {_seatbelt_string(path)}))"


def _seatbelt_string(value: str) -> str:
    _assert_non_empty_string(value, "seatbelt value")
    assert "\x00" not in value, "seatbelt values must not contain NUL"
    assert "\n" not in value, "seatbelt values must not contain newlines"
    assert "\r" not in value, "seatbelt values must not contain newlines"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _execution_environment(plan: SandboxExecutionPlan) -> dict[str, str]:
    profile_environment = plan.settings.profile.environment
    environment = dict(profile_environment.variables)
    environment.update(plan.environment)
    return environment


def _stdout_limit(plan: SandboxExecutionPlan) -> int:
    return min(
        plan.stream_buffer_bytes,
        plan.settings.profile.output.max_stdout_bytes,
    )


def _stderr_limit(plan: SandboxExecutionPlan) -> int:
    return min(
        plan.stream_buffer_bytes,
        plan.settings.profile.output.max_stderr_bytes,
    )


def _probe_environment() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin"}


def _cleanup_paths(plan: SandboxExecutionPlan) -> tuple[str, ...]:
    cleanup = cast(SandboxCleanupPolicy, plan.settings.profile.cleanup)
    if cleanup is not SandboxCleanupPolicy.DELETE:
        return ()
    return _optional_path_tuple(plan.temp_dir)


def _prepare_and_validate_paths(
    plan: SandboxExecutionPlan,
    backend: SandboxBackend,
) -> SandboxBackendDiagnostic | None:
    try:
        _assert_policy_roots_safe(plan)
        _prepare_mapped_directories(plan)
        _assert_backend_paths_allowed(plan)
    except AssertionError as exc:
        return _diagnostic(
            SandboxBackendDiagnosticCode.PATH_DENIED,
            SandboxBackendOperation.PREPARE_PROFILE,
            backend,
            str(exc) or "sandbox path is denied",
        )
    return None


def _assert_policy_roots_safe(plan: SandboxExecutionPlan) -> None:
    profile = plan.settings.profile
    roots: tuple[tuple[str, Sequence[str]], ...] = (
        ("read_roots", profile.read_roots),
        ("write_roots", profile.write_roots),
        ("deny_roots", profile.deny_roots),
        ("scratch_roots", profile.scratch_roots),
        ("output_roots", profile.output_roots),
        ("temp_dir", _optional_path_tuple(plan.temp_dir)),
        ("output_dir", _optional_path_tuple(plan.output_dir)),
    )
    for field_name, paths in roots:
        for path in paths:
            _assert_path_has_no_symlink_escape(path, field_name)


def _assert_path_has_no_symlink_escape(
    path: str,
    field_name: str,
) -> None:
    normalized = normalize_posix_path(path)
    resolved = _real_path(normalized)
    assert normalized == resolved or _system_prefix_resolves_equivalently(
        normalized, resolved
    ), f"{field_name} must not traverse symlink escape: {path}"


def _system_prefix_resolves_equivalently(path: str, resolved: str) -> bool:
    for lexical_prefix, real_prefix in (
        ("/tmp", "/private/tmp"),
        ("/var", "/private/var"),
        ("/etc", "/private/etc"),
    ):
        equivalent = _replace_path_prefix(path, lexical_prefix, real_prefix)
        if equivalent == resolved:
            return True
    return False


def _replace_path_prefix(
    path: str,
    lexical_prefix: str,
    real_prefix: str,
) -> str | None:
    if path == lexical_prefix:
        return real_prefix
    prefix = f"{lexical_prefix}/"
    if path.startswith(prefix):
        return f"{real_prefix}/{path[len(prefix) :]}"
    return None


def _prepare_mapped_directories(plan: SandboxExecutionPlan) -> None:
    for path in (
        tuple(plan.settings.profile.write_roots)
        + _optional_path_tuple(plan.temp_dir)
        + _optional_path_tuple(plan.output_dir)
    ):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)


def _assert_backend_paths_allowed(plan: SandboxExecutionPlan) -> None:
    profile = plan.settings.profile
    allowed_roots = _real_roots(
        tuple(profile.executable_search_roots)
        + tuple(profile.trusted_executables)
        + tuple(profile.read_roots)
        + tuple(profile.write_roots)
        + _optional_path_tuple(plan.temp_dir)
        + _optional_path_tuple(plan.output_dir),
        "allowed roots",
    )
    denied_roots = _real_roots(profile.deny_roots, "deny_roots")
    for root in allowed_roots:
        assert not _path_inside_any(
            root,
            denied_roots,
        ), f"sandbox root is denied: {root}"
    for candidate in _execution_path_candidates(plan):
        assert _real_path_allowed(
            candidate,
            allowed_roots,
            denied_roots,
        ), f"sandbox path escapes allowed roots: {candidate}"


def _execution_path_candidates(
    plan: SandboxExecutionPlan,
) -> tuple[str, ...]:
    candidates = [plan.request.command, plan.request.cwd]
    for argument in plan.request.argv:
        if argument.startswith("/"):
            candidates.append(argument)
    return tuple(candidates)


def _real_roots(roots: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for root in roots:
        assert Path(root).exists(), f"{field_name} must exist: {root}"
        normalized.append(_real_path(root))
    return tuple(_ordered_unique(tuple(normalized)))


def _real_path_allowed(
    path: str,
    allowed_roots: Sequence[str],
    denied_roots: Sequence[str],
) -> bool:
    real_path = _real_path(path)
    if _path_inside_any(real_path, denied_roots):
        return False
    return _path_inside_any(real_path, allowed_roots)


def _real_path(path: str) -> str:
    resolved = Path(path).resolve(strict=False)
    return normalize_posix_path(str(resolved))


async def _execute_subprocess_request(
    plan: SandboxExecutionPlan,
    backend: SandboxBackend,
    request: SandboxSubprocessRequest,
    command_runner: SandboxCommandRunner,
    cleanup_handler: SandboxCleanupHandler,
    cleanup_paths: Sequence[str],
) -> SandboxExecutionResult:
    try:
        completed = await command_runner(request)
    except TimeoutError:
        result = _result(
            SandboxResultStatus.TIMED_OUT,
            diagnostics=(
                _diagnostic(
                    SandboxBackendDiagnosticCode.TIMEOUT,
                    SandboxBackendOperation.WAIT,
                    backend,
                    "sandbox execution exceeded timeout",
                    retryable=True,
                ),
            ),
        )
        return await _finish_with_cleanup(
            result,
            backend,
            cleanup_handler,
            cleanup_paths,
            plan.cleanup_budget_seconds,
        )
    except CancelledError:
        result = _result(
            SandboxResultStatus.CANCELLED,
            diagnostics=(
                _diagnostic(
                    SandboxBackendDiagnosticCode.CANCELLED,
                    SandboxBackendOperation.WAIT,
                    backend,
                    "sandbox execution was cancelled",
                    retryable=True,
                ),
            ),
        )
        return await _finish_with_cleanup(
            result,
            backend,
            cleanup_handler,
            cleanup_paths,
            plan.cleanup_budget_seconds,
        )
    except FileNotFoundError:
        result = _result(
            SandboxResultStatus.FAILED,
            diagnostics=(
                _diagnostic(
                    SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    SandboxBackendOperation.START,
                    backend,
                    "sandbox executable is unavailable",
                    retryable=True,
                ),
            ),
        )
        return await _finish_with_cleanup(
            result,
            backend,
            cleanup_handler,
            cleanup_paths,
            plan.cleanup_budget_seconds,
        )
    except PermissionError as exc:
        result = _result(
            SandboxResultStatus.FAILED,
            diagnostics=(
                _diagnostic(
                    SandboxBackendDiagnosticCode.EXECUTION_FAILED,
                    SandboxBackendOperation.START,
                    backend,
                    f"sandbox executable permission denied: {exc}",
                ),
            ),
        )
        return await _finish_with_cleanup(
            result,
            backend,
            cleanup_handler,
            cleanup_paths,
            plan.cleanup_budget_seconds,
        )
    except OSError as exc:
        result = _result(
            SandboxResultStatus.FAILED,
            diagnostics=(
                _diagnostic(
                    SandboxBackendDiagnosticCode.EXECUTION_FAILED,
                    SandboxBackendOperation.START,
                    backend,
                    f"sandbox process failed to start: {exc}",
                ),
            ),
        )
        return await _finish_with_cleanup(
            result,
            backend,
            cleanup_handler,
            cleanup_paths,
            plan.cleanup_budget_seconds,
        )
    stdout, stderr, chunks, stream_diagnostic = _bounded_completed_streams(
        plan,
        completed,
        backend,
    )
    status = (
        SandboxResultStatus.COMPLETED
        if completed.exit_code == 0
        else SandboxResultStatus.FAILED
    )
    diagnostics: tuple[SandboxBackendDiagnostic, ...] = ()
    if completed.exit_code != 0:
        diagnostics = (
            _diagnostic(
                SandboxBackendDiagnosticCode.EXECUTION_FAILED,
                SandboxBackendOperation.WAIT,
                backend,
                f"sandbox process exited with {completed.exit_code}",
            ),
        )
    if stream_diagnostic is not None:
        diagnostics = diagnostics + (stream_diagnostic,)
    artifacts, output_diagnostic = _collect_real_outputs(plan, backend)
    if output_diagnostic is not None:
        status = SandboxResultStatus.FAILED
        diagnostics = diagnostics + (output_diagnostic,)
    result = SandboxExecutionResult(
        status=status,
        exit_code=completed.exit_code,
        stdout=stdout,
        stderr=stderr,
        diagnostics=diagnostics,
        stream_chunks=chunks,
        output_artifacts=artifacts,
        stream_truncated=stream_diagnostic is not None,
    )
    return await _finish_with_cleanup(
        result,
        backend,
        cleanup_handler,
        cleanup_paths,
        plan.cleanup_budget_seconds,
    )


def _bounded_completed_streams(
    plan: SandboxExecutionPlan,
    completed: SandboxSubprocessResult,
    backend: SandboxBackend,
) -> tuple[
    bytes,
    bytes,
    tuple[SandboxStreamChunk, ...],
    SandboxBackendDiagnostic | None,
]:
    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = (
        _append_bounded(stdout, completed.stdout, _stdout_limit(plan))
        or completed.stdout_truncated
    )
    stderr_truncated = (
        _append_bounded(stderr, completed.stderr, _stderr_limit(plan))
        or completed.stderr_truncated
    )
    chunks: list[SandboxStreamChunk] = []
    if completed.stdout:
        chunks.append(
            SandboxStreamChunk(
                stream=SandboxBackendStream.STDOUT,
                content=bytes(stdout),
                sequence=0,
            )
        )
    if completed.stderr:
        chunks.append(
            SandboxStreamChunk(
                stream=SandboxBackendStream.STDERR,
                content=bytes(stderr),
                sequence=len(chunks),
            )
        )
    diagnostic = (
        _diagnostic(
            SandboxBackendDiagnosticCode.STREAM_TRUNCATED,
            SandboxBackendOperation.STREAM,
            backend,
            "sandbox stream exceeded configured buffer",
        )
        if stdout_truncated or stderr_truncated
        else None
    )
    return bytes(stdout), bytes(stderr), tuple(chunks), diagnostic


def _collect_real_outputs(
    plan: SandboxExecutionPlan,
    backend: SandboxBackend,
) -> tuple[
    tuple[SandboxOutputArtifact, ...],
    SandboxBackendDiagnostic | None,
]:
    if not plan.collect_outputs:
        return (), None
    output_policy = plan.settings.profile.output
    if not output_policy.allow_artifacts:
        return (), _diagnostic(
            SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
            SandboxBackendOperation.COLLECT_OUTPUTS,
            backend,
            "sandbox output collection is disabled",
        )
    if plan.output_dir is None:
        return (), _diagnostic(
            SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
            SandboxBackendOperation.COLLECT_OUTPUTS,
            backend,
            "sandbox output directory is not mapped",
        )
    output_root = Path(plan.output_dir)
    if not output_root.exists():
        return (), None
    artifacts: list[SandboxOutputArtifact] = []
    total_bytes = 0
    for path in output_root.rglob("*"):
        relative_path = path.relative_to(output_root).as_posix()
        if path.is_symlink() or _unsafe_output_path(relative_path):
            return (), _diagnostic(
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
                SandboxBackendOperation.COLLECT_OUTPUTS,
                backend,
                f"sandbox output path is unsafe: {relative_path}",
            )
        if not path.is_file():
            continue
        remaining_bytes = output_policy.max_artifact_bytes - total_bytes
        content = _read_artifact_bounded(path, remaining_bytes)
        if content is None:
            return (), _diagnostic(
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
                SandboxBackendOperation.COLLECT_OUTPUTS,
                backend,
                "sandbox outputs exceed artifact byte limit",
            )
        total_bytes += len(content)
        artifacts.append(
            SandboxOutputArtifact(path=relative_path, content=content)
        )
    return tuple(artifacts), None


def _read_artifact_bounded(path: Path, remaining_bytes: int) -> bytes | None:
    assert remaining_bytes >= 0, "remaining_bytes must not be negative"
    content = bytearray()
    with path.open("rb") as source:
        while True:
            read_size = min(8192, remaining_bytes - len(content) + 1)
            chunk = source.read(read_size)
            if not chunk:
                return bytes(content)
            if len(content) + len(chunk) > remaining_bytes:
                return None
            content.extend(chunk)


async def _finish_with_cleanup(
    result: SandboxExecutionResult,
    backend: SandboxBackend,
    cleanup_handler: SandboxCleanupHandler,
    cleanup_paths: Sequence[str],
    cleanup_budget_seconds: float,
) -> SandboxExecutionResult:
    cleanup_diagnostic = await _cleanup_with_budget(
        cleanup_handler,
        cleanup_paths,
        cleanup_budget_seconds,
        backend,
    )
    if cleanup_diagnostic is None:
        return result
    return replace(
        result,
        status=SandboxResultStatus.FAILED,
        diagnostics=tuple(result.diagnostics) + (cleanup_diagnostic,),
        cleanup_uncertain=True,
    )


async def _cleanup_with_budget(
    cleanup_handler: SandboxCleanupHandler,
    cleanup_paths: Sequence[str],
    cleanup_budget_seconds: float,
    backend: SandboxBackend,
) -> SandboxBackendDiagnostic | None:
    if not cleanup_paths:
        return None
    try:
        cleaned = await wait_for(
            cleanup_handler(tuple(cleanup_paths)),
            cleanup_budget_seconds,
        )
    except TimeoutError:
        return _diagnostic(
            SandboxBackendDiagnosticCode.CLEANUP_FAILED,
            SandboxBackendOperation.CLEANUP,
            backend,
            "sandbox cleanup exceeded cleanup budget",
            retryable=True,
        )
    except CancelledError:
        return _diagnostic(
            SandboxBackendDiagnosticCode.CANCELLED,
            SandboxBackendOperation.CLEANUP,
            backend,
            "sandbox cleanup was cancelled",
            retryable=True,
        )
    if cleaned:
        return None
    return _diagnostic(
        SandboxBackendDiagnosticCode.CLEANUP_FAILED,
        SandboxBackendOperation.CLEANUP,
        backend,
        "sandbox cleanup state is uncertain",
        retryable=True,
    )


async def _default_cleanup_handler(paths: Sequence[str]) -> bool:
    try:
        for path in paths:
            target = Path(path)
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                rmtree(target)
    except OSError:
        return False
    return True


async def _default_command_runner(
    request: SandboxSubprocessRequest,
) -> SandboxSubprocessResult:
    preexec_fn = _process_limit_preexec(request.process_limit)
    process = await create_subprocess_exec(
        *request.argv,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        cwd=request.cwd,
        env=dict(request.environment),
        close_fds=request.close_fds,
        pass_fds=tuple(request.pass_fds),
        preexec_fn=preexec_fn,
    )
    try:
        if process.stdin is not None:
            process.stdin.close()
        if request.timeout_seconds is None:
            (
                stdout,
                stdout_truncated,
                stderr,
                stderr_truncated,
            ) = await _wait_for_bounded_process_output(process, request)
        else:
            (
                stdout,
                stdout_truncated,
                stderr,
                stderr_truncated,
            ) = await wait_for(
                _wait_for_bounded_process_output(process, request),
                request.timeout_seconds,
            )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    except CancelledError:
        process.kill()
        await process.wait()
        raise
    return SandboxSubprocessResult(
        exit_code=process.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


async def _wait_for_bounded_process_output(
    process: Process,
    request: SandboxSubprocessRequest,
) -> tuple[bytes, bool, bytes, bool]:
    stdout_reader = process.stdout
    stderr_reader = process.stderr
    assert stdout_reader is not None
    assert stderr_reader is not None
    stdout_result, stderr_result, _exit_code = await gather(
        _read_stream_bounded(stdout_reader, request.stdout_limit_bytes),
        _read_stream_bounded(stderr_reader, request.stderr_limit_bytes),
        process.wait(),
    )
    stdout, stdout_truncated = stdout_result
    stderr, stderr_truncated = stderr_result
    return stdout, stdout_truncated, stderr, stderr_truncated


async def _read_stream_bounded(
    reader: StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    assert limit > 0, "limit must be positive"
    content = bytearray()
    truncated = False
    while True:
        chunk = await reader.read(8192)
        if not chunk:
            return bytes(content), truncated
        remaining = limit - len(content)
        if remaining > 0:
            content.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True


def _process_limit_preexec(
    process_limit: int | None,
) -> Callable[[], None] | None:
    if process_limit is None:
        return None
    assert _process_limits_supported(), "process limits are unsupported"

    def apply_process_limit() -> None:
        get_limit = _RESOURCE_GETRLIMIT
        set_limit = _RESOURCE_SETRLIMIT
        assert get_limit is not None
        assert set_limit is not None
        _soft_limit, hard_limit = get_limit(_RESOURCE_RLIMIT_NPROC)
        limit = process_limit
        if hard_limit > 0:
            limit = min(limit, hard_limit)
        set_limit(_RESOURCE_RLIMIT_NPROC, (limit, hard_limit))

    return apply_process_limit


def _process_limits_supported() -> bool:
    return (
        _RESOURCE_RLIMIT_NPROC >= 0
        and _RESOURCE_GETRLIMIT is not None
        and _RESOURCE_SETRLIMIT is not None
    )


def _backend_mismatch_result(
    backend: SandboxBackend,
    message: str,
) -> SandboxExecutionResult:
    return _result(
        SandboxResultStatus.DENIED,
        diagnostics=(
            _diagnostic(
                SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
                SandboxBackendOperation.START,
                backend,
                message,
            ),
        ),
    )


def _selection_failure_result(
    selection: SandboxBackendSelection,
) -> SandboxExecutionResult:
    status = (
        SandboxResultStatus.FAILED
        if any(
            diagnostic.code is SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE
            for diagnostic in selection.diagnostics
        )
        else SandboxResultStatus.DENIED
    )
    return _result(status, diagnostics=selection.diagnostics)


def _unavailable_probe(
    backend: SandboxBackend,
    message: str,
    *,
    code: SandboxBackendDiagnosticCode = (
        SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE
    ),
    retryable: bool = True,
) -> SandboxBackendProbeResult:
    return SandboxBackendProbeResult(
        backend=backend,
        available=False,
        diagnostics=(
            _diagnostic(
                code,
                SandboxBackendOperation.PROBE,
                backend,
                message,
                retryable=retryable,
            ),
        ),
    )


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
                process_limits=False,
                child_processes=True,
                inherited_fds=True,
            ),
            temp_output=SandboxTempOutputMapping(
                temp_dirs=True,
                output_dirs=True,
                cleanup_budget=True,
            ),
            unsupported_controls=(
                "cpu_limits",
                "memory_limits",
                "network_allowlist",
                "pid_limits",
                "pid_limits_best_effort_rlimit",
                "child_process_denial",
            ),
        ),
    ),
)
