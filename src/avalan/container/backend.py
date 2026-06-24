from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .conformance import (
    ContainerBackend,
)
from .output import (
    ContainerOutputContract,
    ContainerOutputDecisionType,
    ContainerOutputValidationResult,
)
from .settings import (
    ContainerBackendCapabilities,
    ContainerBuildPolicy,
    ContainerDeviceClass,
    ContainerExecutionResult,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
)

from abc import ABC, abstractmethod
from asyncio import CancelledError, sleep
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)


class ContainerBackendOperation(StrEnum):
    PROBE = "probe"
    IMAGE_RESOLUTION = "image_resolution"
    IMAGE_PULL = "image_pull"
    IMAGE_BUILD = "image_build"
    CREATE = "create"
    START = "start"
    ATTACH = "attach"
    STREAM = "stream"
    WAIT = "wait"
    INSPECT = "inspect"
    STATS = "stats"
    STOP = "stop"
    KILL = "kill"
    REMOVE = "remove"
    COPY_OUTPUTS = "copy_outputs"
    CLEANUP = "cleanup"


class ContainerBackendDiagnosticCode(StrEnum):
    AUTO_NOT_ENABLED = "container.backend.auto_not_enabled"
    BACKEND_UNAVAILABLE = "container.backend.unavailable"
    CAPABILITY_MISMATCH = "container.backend.capability_mismatch"
    ROOTFUL_NOT_AUTHORIZED = "container.backend.rootful_not_authorized"
    IMAGE_DENIED = "container.backend.image_denied"
    PULL_DENIED = "container.backend.pull_denied"
    PULL_FAILED = "container.backend.pull_failed"
    BUILD_DENIED = "container.backend.build_denied"
    BUILD_FAILED = "container.backend.build_failed"
    CREATE_FAILED = "container.backend.create_failed"
    ATTACH_FAILED = "container.backend.attach_failed"
    START_FAILED = "container.backend.start_failed"
    WAIT_FAILED = "container.backend.wait_failed"
    COPY_FAILED = "container.backend.copy_failed"
    CLEANUP_FAILED = "container.backend.cleanup_failed"
    ORPHAN_QUARANTINED = "container.backend.orphan_quarantined"
    CANCELLED = "container.backend.cancelled"
    TIMEOUT = "container.backend.timeout"


class ContainerBackendStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"
    PROGRESS = "progress"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendDiagnostic:
    code: ContainerBackendDiagnosticCode | str
    operation: ContainerBackendOperation | str
    message: str
    backend: ContainerBackend | str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _enum_value(self.code, ContainerBackendDiagnosticCode, "code"),
        )
        object.__setattr__(
            self,
            "operation",
            _enum_value(
                self.operation,
                ContainerBackendOperation,
                "operation",
            ),
        )
        if self.backend is not None:
            object.__setattr__(
                self,
                "backend",
                _enum_value(self.backend, ContainerBackend, "backend"),
            )
        _assert_non_empty_string(self.message, "message")
        _assert_bool(self.retryable, "retryable")

    def to_dict(self) -> dict[str, object]:
        code = cast(ContainerBackendDiagnosticCode, self.code)
        operation = cast(ContainerBackendOperation, self.operation)
        backend = cast(ContainerBackend | None, self.backend)
        return {
            "code": code.value,
            "operation": operation.value,
            "backend": backend.value if backend else None,
            "message": self.message,
            "retryable": self.retryable,
        }


class ContainerBackendError(Exception):
    def __init__(self, diagnostic: ContainerBackendDiagnostic) -> None:
        assert isinstance(diagnostic, ContainerBackendDiagnostic)
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class _ContainerBackendLifecycleFailure(Exception):
    def __init__(
        self,
        status: ContainerResultStatus,
        *,
        kill_container: bool = False,
    ) -> None:
        assert isinstance(status, ContainerResultStatus)
        _assert_bool(kill_container, "kill_container")
        super().__init__(status.value)
        self.status = status
        self.kill_container = kill_container


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendRuntimeRequirements:
    marker: str | None = None
    environment_variables: Sequence[str] = field(default_factory=tuple)
    requires_network: bool = False
    requires_secrets: bool = False

    def __post_init__(self) -> None:
        if self.marker is not None:
            _assert_non_empty_string(self.marker, "marker")
        object.__setattr__(
            self,
            "environment_variables",
            _string_tuple(
                self.environment_variables,
                "environment_variables",
            ),
        )
        _assert_bool(self.requires_network, "requires_network")
        _assert_bool(self.requires_secrets, "requires_secrets")

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "environment_variables": list(self.environment_variables),
            "requires_network": self.requires_network,
            "requires_secrets": self.requires_secrets,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendProbeResult:
    backend: ContainerBackend | str
    available: bool
    capabilities: ContainerBackendCapabilities | None = None
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )
    runtime_requirements: ContainerBackendRuntimeRequirements = field(
        default_factory=ContainerBackendRuntimeRequirements,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, ContainerBackend, "backend"),
        )
        _assert_bool(self.available, "available")
        if self.capabilities is not None:
            assert isinstance(self.capabilities, ContainerBackendCapabilities)
        if self.available:
            assert (
                self.capabilities is not None
            ), "available backend probes require capabilities"
        _assert_diagnostics(self.diagnostics)
        assert isinstance(
            self.runtime_requirements,
            ContainerBackendRuntimeRequirements,
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return self.available and not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        backend = cast(ContainerBackend, self.backend)
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
            "runtime_requirements": self.runtime_requirements.to_dict(),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendSelection:
    backend: ContainerBackend | str | None
    capabilities: ContainerBackendCapabilities | None = None
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        if self.backend is not None:
            object.__setattr__(
                self,
                "backend",
                _enum_value(self.backend, ContainerBackend, "backend"),
            )
        if self.capabilities is not None:
            assert isinstance(self.capabilities, ContainerBackendCapabilities)
        _assert_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return self.backend is not None and not self.diagnostics


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendOperationResult:
    operation: ContainerBackendOperation | str
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _enum_value(
                self.operation,
                ContainerBackendOperation,
                "operation",
            ),
        )
        _assert_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        operation = cast(ContainerBackendOperation, self.operation)
        return {
            "operation": operation.value,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendImageResolution:
    reference: str
    digest: str
    platform: str
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.reference, "reference")
        _assert_non_empty_string(self.digest, "digest")
        _assert_non_empty_string(self.platform, "platform")
        _assert_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "digest": self.digest,
            "platform": self.platform,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendContainer:
    container_id: str
    backend: ContainerBackend | str
    plan_fingerprint: str
    state: str = "created"

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.container_id, "container_id")
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, ContainerBackend, "backend"),
        )
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")
        _assert_non_empty_string(self.state, "state")

    def to_dict(self) -> dict[str, str]:
        backend = cast(ContainerBackend, self.backend)
        return {
            "container_id": self.container_id,
            "backend": backend.value,
            "plan_fingerprint": self.plan_fingerprint,
            "state": self.state,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendStreamChunk:
    stream: ContainerBackendStream | str
    content: bytes
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream",
            _enum_value(self.stream, ContainerBackendStream, "stream"),
        )
        assert isinstance(self.content, bytes), "content must be bytes"
        assert isinstance(self.sequence, int)
        assert self.sequence >= 0, "sequence must not be negative"

    def to_dict(self) -> dict[str, object]:
        stream = cast(ContainerBackendStream, self.stream)
        return {
            "stream": stream.value,
            "content": self.content.decode("utf-8", errors="replace"),
            "sequence": self.sequence,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendStats:
    cpu_nanos: int = 0
    memory_bytes: int = 0
    pids: int = 0

    def __post_init__(self) -> None:
        _assert_non_negative_int(self.cpu_nanos, "cpu_nanos")
        _assert_non_negative_int(self.memory_bytes, "memory_bytes")
        _assert_non_negative_int(self.pids, "pids")

    def to_dict(self) -> dict[str, int]:
        return {
            "cpu_nanos": self.cpu_nanos,
            "memory_bytes": self.memory_bytes,
            "pids": self.pids,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendInspection:
    container_id: str
    status: str
    exit_code: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.container_id, "container_id")
        _assert_non_empty_string(self.status, "status")
        if self.exit_code is not None:
            assert isinstance(self.exit_code, int)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "container_id": self.container_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendWaitResult:
    exit_code: int
    timed_out: bool = False
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.exit_code, int)
        _assert_bool(self.timed_out, "timed_out")
        _assert_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return (
            self.exit_code == 0 and not self.timed_out and not self.diagnostics
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendLifecycleResult:
    execution: ContainerExecutionResult
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )
    stream_chunks: Sequence[ContainerBackendStreamChunk] = field(
        default_factory=tuple,
    )
    stats: Sequence[ContainerBackendStats] = field(default_factory=tuple)
    output: ContainerOutputValidationResult | None = None
    cleanup_uncertain: bool = False
    orphan_quarantined: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.execution, ContainerExecutionResult)
        _assert_diagnostics(self.diagnostics)
        for chunk in self.stream_chunks:
            assert isinstance(chunk, ContainerBackendStreamChunk)
        for stat in self.stats:
            assert isinstance(stat, ContainerBackendStats)
        if self.output is not None:
            assert isinstance(self.output, ContainerOutputValidationResult)
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        _assert_bool(self.orphan_quarantined, "orphan_quarantined")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "stream_chunks", tuple(self.stream_chunks))
        object.__setattr__(self, "stats", tuple(self.stats))

    def to_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution.to_dict(),
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "stream_chunks": [chunk.to_dict() for chunk in self.stream_chunks],
            "stats": [stat.to_dict() for stat in self.stats],
            "output": None if self.output is None else self.output.to_dict(),
            "cleanup_uncertain": self.cleanup_uncertain,
            "orphan_quarantined": self.orphan_quarantined,
        }


class ContainerAsyncBackend(ABC):
    @abstractmethod
    async def probe(self) -> ContainerBackendProbeResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def resolve_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendImageResolution:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def pull_image(
        self,
        plan: ContainerRunPlan,
        image: ContainerBackendImageResolution,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def create(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendContainer:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def start(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def attach(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def stream(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStreamChunk, ...]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def wait(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendWaitResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def inspect(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendInspection:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def stats(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStats, ...]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def stop(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def kill(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def remove(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def cleanup(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        raise NotImplementedError  # pragma: no cover

    async def run(
        self,
        plan: ContainerRunPlan,
        *,
        output_contract: ContainerOutputContract | None = None,
    ) -> ContainerBackendLifecycleResult:
        return await run_container_backend_lifecycle(
            self,
            plan,
            output_contract=output_contract,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerFakeBackendScript:
    capabilities: ContainerBackendCapabilities
    available: bool = True
    resolved_digest: str | None = None
    operation_diagnostics: Mapping[
        ContainerBackendOperation | str,
        ContainerBackendDiagnosticCode | str,
    ] = field(default_factory=dict)
    soft_operation_diagnostics: Mapping[
        ContainerBackendOperation | str,
        ContainerBackendDiagnosticCode | str,
    ] = field(default_factory=dict)
    cancel_operations: Sequence[ContainerBackendOperation | str] = field(
        default_factory=tuple,
    )
    timeout_operations: Sequence[ContainerBackendOperation | str] = field(
        default_factory=tuple,
    )
    stream_chunks: Sequence[ContainerBackendStreamChunk] = field(
        default_factory=lambda: (
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDOUT,
                content=b"ok\n",
                sequence=0,
            ),
        ),
    )
    stream_delay_seconds: float = 0
    stats_samples: Sequence[ContainerBackendStats] = field(
        default_factory=lambda: (ContainerBackendStats(memory_bytes=1024),),
    )
    output_result: ContainerOutputValidationResult | None = None
    wait_exit_code: int = 0
    wait_timed_out: bool = False
    cleanup_uncertain: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.capabilities, ContainerBackendCapabilities)
        _assert_bool(self.available, "available")
        if self.resolved_digest is not None:
            _assert_non_empty_string(self.resolved_digest, "resolved_digest")
        operation_diagnostics: dict[
            ContainerBackendOperation,
            ContainerBackendDiagnosticCode,
        ] = {}
        for operation, code in self.operation_diagnostics.items():
            operation_diagnostics[
                _enum_value(operation, ContainerBackendOperation, "operation")
            ] = _enum_value(code, ContainerBackendDiagnosticCode, "code")
        object.__setattr__(
            self,
            "operation_diagnostics",
            MappingProxyType(operation_diagnostics),
        )
        soft_operation_diagnostics: dict[
            ContainerBackendOperation,
            ContainerBackendDiagnosticCode,
        ] = {}
        for operation, code in self.soft_operation_diagnostics.items():
            soft_operation_diagnostics[
                _enum_value(operation, ContainerBackendOperation, "operation")
            ] = _enum_value(code, ContainerBackendDiagnosticCode, "code")
        object.__setattr__(
            self,
            "soft_operation_diagnostics",
            MappingProxyType(soft_operation_diagnostics),
        )
        object.__setattr__(
            self,
            "cancel_operations",
            tuple(
                _enum_value(
                    operation,
                    ContainerBackendOperation,
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
                    ContainerBackendOperation,
                    "timeout_operations",
                )
                for operation in self.timeout_operations
            ),
        )
        for chunk in self.stream_chunks:
            assert isinstance(chunk, ContainerBackendStreamChunk)
        assert isinstance(self.stream_delay_seconds, int | float)
        assert (
            self.stream_delay_seconds >= 0
        ), "stream_delay_seconds must not be negative"
        for stat in self.stats_samples:
            assert isinstance(stat, ContainerBackendStats)
        if self.output_result is not None:
            assert isinstance(
                self.output_result,
                ContainerOutputValidationResult,
            )
        assert isinstance(self.wait_exit_code, int)
        _assert_bool(self.wait_timed_out, "wait_timed_out")
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        object.__setattr__(self, "stream_chunks", tuple(self.stream_chunks))
        object.__setattr__(self, "stats_samples", tuple(self.stats_samples))


@final
class ContainerFakeBackend(ContainerAsyncBackend):
    def __init__(self, script: ContainerFakeBackendScript) -> None:
        assert isinstance(script, ContainerFakeBackendScript)
        self._script = script
        self._operations: list[ContainerBackendOperation] = []
        self._created_count = 0

    @property
    def operations(self) -> tuple[ContainerBackendOperation, ...]:
        return tuple(self._operations)

    @property
    def backend(self) -> ContainerBackend:
        return cast(ContainerBackend, self._script.capabilities.backend)

    async def probe(self) -> ContainerBackendProbeResult:
        operation = ContainerBackendOperation.PROBE
        self._record(operation)
        if not self._script.available:
            return ContainerBackendProbeResult(
                backend=self.backend,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        operation,
                        self.backend,
                        "backend is unavailable",
                        retryable=True,
                    ),
                ),
            )
        return ContainerBackendProbeResult(
            backend=self.backend,
            available=True,
            capabilities=self._script.capabilities,
        )

    async def resolve_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendImageResolution:
        await self._enter(ContainerBackendOperation.IMAGE_RESOLUTION)
        digest = self._script.resolved_digest or cast(str, plan.image.digest)
        return ContainerBackendImageResolution(
            reference=plan.image.reference,
            digest=digest,
            platform=plan.image.platform,
            diagnostics=self._soft_diagnostics(
                ContainerBackendOperation.IMAGE_RESOLUTION,
            ),
        )

    async def pull_image(
        self,
        plan: ContainerRunPlan,
        image: ContainerBackendImageResolution,
    ) -> ContainerBackendOperationResult:
        assert isinstance(image, ContainerBackendImageResolution)
        await self._enter(ContainerBackendOperation.IMAGE_PULL)
        if plan.image.pull_policy is ContainerPullPolicy.NEVER:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.PULL_DENIED,
                    ContainerBackendOperation.IMAGE_PULL,
                    self.backend,
                    "image pull is disabled by policy",
                )
            )
        return self._operation_result(ContainerBackendOperation.IMAGE_PULL)

    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        await self._enter(ContainerBackendOperation.IMAGE_BUILD)
        if plan.image.build_policy is ContainerBuildPolicy.DISABLED:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BUILD_DENIED,
                    ContainerBackendOperation.IMAGE_BUILD,
                    self.backend,
                    "image builds are disabled by policy",
                )
            )
        return self._operation_result(ContainerBackendOperation.IMAGE_BUILD)

    async def create(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendContainer:
        await self._enter(ContainerBackendOperation.CREATE)
        self._created_count += 1
        return ContainerBackendContainer(
            container_id=f"fake-{self._created_count}",
            backend=self.backend,
            plan_fingerprint=_plan_fingerprint(plan),
        )

    async def start(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.START)
        return self._operation_result(ContainerBackendOperation.START)

    async def attach(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.ATTACH)
        return self._operation_result(ContainerBackendOperation.ATTACH)

    async def stream(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStreamChunk, ...]:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.STREAM)
        chunks: list[ContainerBackendStreamChunk] = []
        for chunk in self._script.stream_chunks:
            if self._script.stream_delay_seconds:
                await sleep(self._script.stream_delay_seconds)
            chunks.append(chunk)
        return tuple(chunks)

    async def wait(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendWaitResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.WAIT)
        return ContainerBackendWaitResult(
            exit_code=self._script.wait_exit_code,
            timed_out=self._script.wait_timed_out,
            diagnostics=self._soft_diagnostics(ContainerBackendOperation.WAIT),
        )

    async def inspect(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendInspection:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.INSPECT)
        return ContainerBackendInspection(
            container_id=container.container_id,
            status="exited",
            exit_code=self._script.wait_exit_code,
            metadata={"backend": self.backend.value},
        )

    async def stats(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStats, ...]:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.STATS)
        return tuple(self._script.stats_samples)

    async def stop(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.STOP)
        return self._operation_result(ContainerBackendOperation.STOP)

    async def kill(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.KILL)
        return self._operation_result(ContainerBackendOperation.KILL)

    async def remove(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.REMOVE)
        return self._operation_result(ContainerBackendOperation.REMOVE)

    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        assert isinstance(container, ContainerBackendContainer)
        assert isinstance(contract, ContainerOutputContract)
        await self._enter(ContainerBackendOperation.COPY_OUTPUTS)
        return self._script.output_result or ContainerOutputValidationResult(
            decision=ContainerOutputDecisionType.ACCEPT,
            contract=contract,
        )

    async def cleanup(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.CLEANUP)
        if self._script.cleanup_uncertain:
            return ContainerBackendOperationResult(
                operation=ContainerBackendOperation.CLEANUP,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED,
                        ContainerBackendOperation.CLEANUP,
                        self.backend,
                        "cleanup state is uncertain; orphan is quarantined",
                        retryable=True,
                    ),
                ),
            )
        return self._operation_result(ContainerBackendOperation.CLEANUP)

    async def _enter(self, operation: ContainerBackendOperation) -> None:
        self._record(operation)
        if operation in self._script.cancel_operations:
            raise CancelledError
        if operation in self._script.timeout_operations:
            raise TimeoutError
        code = self._script.operation_diagnostics.get(operation)
        if code is not None:
            diagnostic_code = cast(ContainerBackendDiagnosticCode, code)
            raise ContainerBackendError(
                _diagnostic(
                    diagnostic_code,
                    operation,
                    self.backend,
                    _diagnostic_message(diagnostic_code, operation),
                    retryable=_is_retryable(diagnostic_code),
                )
            )

    def _operation_result(
        self,
        operation: ContainerBackendOperation,
    ) -> ContainerBackendOperationResult:
        return ContainerBackendOperationResult(
            operation=operation,
            diagnostics=self._soft_diagnostics(operation),
        )

    def _soft_diagnostics(
        self,
        operation: ContainerBackendOperation,
    ) -> tuple[ContainerBackendDiagnostic, ...]:
        code = self._script.soft_operation_diagnostics.get(operation)
        if code is None:
            return ()
        diagnostic_code = cast(ContainerBackendDiagnosticCode, code)
        return (
            _diagnostic(
                diagnostic_code,
                operation,
                self.backend,
                _diagnostic_message(diagnostic_code, operation),
                retryable=_is_retryable(diagnostic_code),
            ),
        )

    def _record(self, operation: ContainerBackendOperation) -> None:
        self._operations.append(operation)


def select_container_backend(
    plan: ContainerRunPlan,
    probes: Sequence[ContainerBackendProbeResult],
    *,
    auto_enabled: bool,
    rootful_authorized: bool = False,
) -> ContainerBackendSelection:
    assert isinstance(plan, ContainerRunPlan)
    for probe in probes:
        assert isinstance(probe, ContainerBackendProbeResult)
    requested = cast(ContainerBackend, plan.backend)
    if requested is ContainerBackend.AUTO and not auto_enabled:
        return ContainerBackendSelection(
            backend=None,
            diagnostics=(
                _diagnostic(
                    ContainerBackendDiagnosticCode.AUTO_NOT_ENABLED,
                    ContainerBackendOperation.PROBE,
                    None,
                    "backend auto-selection is not enabled by operator policy",
                ),
            ),
        )
    candidates = [
        probe
        for probe in probes
        if _candidate_matches_request(probe, requested)
    ]
    diagnostics: list[ContainerBackendDiagnostic] = []
    if not candidates:
        diagnostics.append(
            _backend_unavailable_diagnostic(
                None if requested is ContainerBackend.AUTO else requested,
                (
                    "no container backend probes are available"
                    if requested is ContainerBackend.AUTO
                    else f"configured backend {requested.value} is unavailable"
                ),
            )
        )
    eligible: list[ContainerBackendProbeResult] = []
    for probe in candidates:
        diagnostics.extend(probe.diagnostics)
        if not probe.available or probe.capabilities is None:
            diagnostics.append(
                _backend_unavailable_diagnostic(
                    cast(ContainerBackend, probe.backend),
                    "backend is unavailable",
                )
            )
            continue
        capability_diagnostics = _capability_diagnostics(
            plan,
            probe.capabilities,
            rootful_authorized=rootful_authorized,
        )
        if capability_diagnostics:
            diagnostics.extend(capability_diagnostics)
            continue
        eligible.append(probe)
    if not eligible:
        return ContainerBackendSelection(backend=None, diagnostics=diagnostics)
    selected = sorted(
        eligible,
        key=lambda probe: _selection_score(
            cast(
                ContainerBackendCapabilities,
                probe.capabilities,
            )
        ),
    )[0]
    assert selected.capabilities is not None
    return ContainerBackendSelection(
        backend=selected.backend,
        capabilities=selected.capabilities,
    )


async def run_container_backend_lifecycle(
    backend: ContainerAsyncBackend,
    plan: ContainerRunPlan,
    *,
    output_contract: ContainerOutputContract | None = None,
) -> ContainerBackendLifecycleResult:
    assert isinstance(backend, ContainerAsyncBackend)
    assert isinstance(plan, ContainerRunPlan)
    if output_contract is not None:
        assert isinstance(output_contract, ContainerOutputContract)
    diagnostics: list[ContainerBackendDiagnostic] = []
    stream_chunks: tuple[ContainerBackendStreamChunk, ...] = ()
    stats: tuple[ContainerBackendStats, ...] = ()
    output: ContainerOutputValidationResult | None = None
    container: ContainerBackendContainer | None = None
    exit_code: int | None = None
    cleanup_uncertain = False
    orphan_quarantined = False
    status = ContainerResultStatus.COMPLETED
    try:
        image = await backend.resolve_image(plan)
        diagnostics.extend(image.diagnostics)
        if not image.ok:
            raise _ContainerBackendLifecycleFailure(
                _status_for_diagnostics(image.diagnostics)
            )
        if plan.image.pull_policy is not ContainerPullPolicy.NEVER:
            _append_operation_result(
                diagnostics,
                await backend.pull_image(plan, image),
            )
        if plan.image.build_policy is not ContainerBuildPolicy.DISABLED:
            _append_operation_result(
                diagnostics,
                await backend.build_image(plan),
            )
        container = await backend.create(plan)
        _append_operation_result(
            diagnostics,
            await backend.attach(container),
        )
        _append_operation_result(
            diagnostics,
            await backend.start(container),
        )
        stream_chunks = await backend.stream(container)
        stats = await backend.stats(container)
        wait_result = await backend.wait(container)
        diagnostics.extend(wait_result.diagnostics)
        if wait_result.timed_out:
            diagnostics.append(
                _diagnostic(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    ContainerBackendOperation.WAIT,
                    None,
                    "container execution timed out",
                    retryable=True,
                )
            )
        exit_code = wait_result.exit_code
        if wait_result.timed_out or wait_result.diagnostics:
            raise _ContainerBackendLifecycleFailure(
                _status_for_diagnostics(diagnostics),
                kill_container=True,
            )
        await backend.inspect(container)
        if output_contract is not None:
            output = await backend.copy_outputs(container, output_contract)
            if output.decision is not ContainerOutputDecisionType.ACCEPT:
                status = ContainerResultStatus.FAILED
        if exit_code != 0:
            status = ContainerResultStatus.FAILED
    except CancelledError:
        status = ContainerResultStatus.CANCELLED
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.CANCELLED,
                ContainerBackendOperation.STREAM,
                None,
                "container execution was cancelled",
                retryable=True,
            )
        )
        if container is not None:
            kill_result = await _kill_container(
                backend,
                container,
                diagnostics,
            )
            cleanup_uncertain = cleanup_uncertain or kill_result[0]
            orphan_quarantined = orphan_quarantined or kill_result[1]
    except TimeoutError:
        status = ContainerResultStatus.FAILED
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.TIMEOUT,
                ContainerBackendOperation.WAIT,
                None,
                "container execution timed out",
                retryable=True,
            )
        )
        if container is not None:
            kill_result = await _kill_container(
                backend,
                container,
                diagnostics,
            )
            cleanup_uncertain = cleanup_uncertain or kill_result[0]
            orphan_quarantined = orphan_quarantined or kill_result[1]
    except _ContainerBackendLifecycleFailure as failure:
        status = failure.status
        if failure.kill_container and container is not None:
            kill_result = await _kill_container(
                backend,
                container,
                diagnostics,
            )
            cleanup_uncertain = cleanup_uncertain or kill_result[0]
            orphan_quarantined = orphan_quarantined or kill_result[1]
    except ContainerBackendError as error:
        diagnostics.append(error.diagnostic)
        status = _status_for_diagnostic(error.diagnostic)
    finally:
        if container is not None:
            cleanup = await _cleanup_container(backend, container, diagnostics)
            cleanup_uncertain = cleanup_uncertain or cleanup[0]
            orphan_quarantined = orphan_quarantined or cleanup[1]
            if cleanup_uncertain and status is ContainerResultStatus.COMPLETED:
                status = ContainerResultStatus.FAILED
    return ContainerBackendLifecycleResult(
        execution=ContainerExecutionResult(
            status=status,
            exit_code=exit_code,
            diagnostics=tuple(_diagnostic_text(item) for item in diagnostics),
            metadata={
                "stream_chunks": str(len(stream_chunks)),
                "stats_samples": str(len(stats)),
                "cleanup_uncertain": str(cleanup_uncertain).lower(),
                "orphan_quarantined": str(orphan_quarantined).lower(),
            },
        ),
        diagnostics=diagnostics,
        stream_chunks=stream_chunks,
        stats=stats,
        output=output,
        cleanup_uncertain=cleanup_uncertain,
        orphan_quarantined=orphan_quarantined,
    )


def _append_operation_result(
    diagnostics: list[ContainerBackendDiagnostic],
    result: ContainerBackendOperationResult,
) -> None:
    assert isinstance(result, ContainerBackendOperationResult)
    diagnostics.extend(result.diagnostics)
    if not result.ok:
        raise _ContainerBackendLifecycleFailure(
            _status_for_diagnostics(result.diagnostics)
        )


async def _kill_container(
    backend: ContainerAsyncBackend,
    container: ContainerBackendContainer,
    diagnostics: list[ContainerBackendDiagnostic],
) -> tuple[bool, bool]:
    try:
        kill_result = await backend.kill(container)
        diagnostics.extend(kill_result.diagnostics)
        if kill_result.diagnostics:
            return True, _diagnostics_include_orphan(kill_result.diagnostics)
    except ContainerBackendError as error:
        diagnostics.append(error.diagnostic)
        return (
            True,
            error.diagnostic.code
            is ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED,
        )
    except TimeoutError:
        diagnostics.append(
            _runtime_exception_diagnostic(
                ContainerBackendDiagnosticCode.TIMEOUT,
                ContainerBackendOperation.KILL,
            )
        )
        return True, False
    except CancelledError:
        diagnostics.append(
            _runtime_exception_diagnostic(
                ContainerBackendDiagnosticCode.CANCELLED,
                ContainerBackendOperation.KILL,
            )
        )
        return True, False
    return False, False


async def _cleanup_container(
    backend: ContainerAsyncBackend,
    container: ContainerBackendContainer,
    diagnostics: list[ContainerBackendDiagnostic],
) -> tuple[bool, bool]:
    cleanup_uncertain = False
    orphan_quarantined = False
    try:
        remove_result = await backend.remove(container)
        diagnostics.extend(remove_result.diagnostics)
        if remove_result.diagnostics:
            cleanup_uncertain = True
            orphan_quarantined = _diagnostics_include_orphan(
                remove_result.diagnostics
            )
    except ContainerBackendError as error:
        diagnostics.append(error.diagnostic)
        cleanup_uncertain = True
        orphan_quarantined = (
            error.diagnostic.code
            is ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
        )
    except TimeoutError:
        diagnostics.append(
            _runtime_exception_diagnostic(
                ContainerBackendDiagnosticCode.TIMEOUT,
                ContainerBackendOperation.REMOVE,
            )
        )
        cleanup_uncertain = True
    except CancelledError:
        diagnostics.append(
            _runtime_exception_diagnostic(
                ContainerBackendDiagnosticCode.CANCELLED,
                ContainerBackendOperation.REMOVE,
            )
        )
        cleanup_uncertain = True
    try:
        cleanup_result = await backend.cleanup(container)
        diagnostics.extend(cleanup_result.diagnostics)
        if cleanup_result.diagnostics:
            cleanup_uncertain = True
        orphan_quarantined = orphan_quarantined or _diagnostics_include_orphan(
            cleanup_result.diagnostics
        )
    except ContainerBackendError as error:
        diagnostics.append(error.diagnostic)
        cleanup_uncertain = True
        orphan_quarantined = orphan_quarantined or (
            error.diagnostic.code
            is ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
        )
    except TimeoutError:
        diagnostics.append(
            _runtime_exception_diagnostic(
                ContainerBackendDiagnosticCode.TIMEOUT,
                ContainerBackendOperation.CLEANUP,
            )
        )
        cleanup_uncertain = True
    except CancelledError:
        diagnostics.append(
            _runtime_exception_diagnostic(
                ContainerBackendDiagnosticCode.CANCELLED,
                ContainerBackendOperation.CLEANUP,
            )
        )
        cleanup_uncertain = True
    return cleanup_uncertain, orphan_quarantined


def _diagnostics_include_orphan(
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> bool:
    return any(
        diagnostic.code is ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
        for diagnostic in diagnostics
    )


def _runtime_exception_diagnostic(
    code: ContainerBackendDiagnosticCode,
    operation: ContainerBackendOperation,
) -> ContainerBackendDiagnostic:
    return _diagnostic(
        code,
        operation,
        None,
        _diagnostic_message(code, operation),
        retryable=True,
    )


def _candidate_matches_request(
    probe: ContainerBackendProbeResult,
    requested: ContainerBackend,
) -> bool:
    if requested is ContainerBackend.AUTO:
        return True
    return probe.backend is requested


def _capability_diagnostics(
    plan: ContainerRunPlan,
    capabilities: ContainerBackendCapabilities,
    *,
    rootful_authorized: bool,
) -> tuple[ContainerBackendDiagnostic, ...]:
    backend = cast(ContainerBackend, capabilities.backend)
    diagnostics: list[ContainerBackendDiagnostic] = []
    if (
        not rootful_authorized
        and not capabilities.rootless
        and not capabilities.per_container_vm_isolation
    ):
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.ROOTFUL_NOT_AUTHORIZED,
                ContainerBackendOperation.PROBE,
                backend,
                "rootful backend requires trusted pre-authorization",
            )
        )
    guest_os, architecture = _platform_parts(plan.image.platform)
    if capabilities.guest_os.lower() != guest_os:
        diagnostics.append(
            _capability_mismatch(
                backend,
                f"guest OS {guest_os} is not supported",
            )
        )
    if (
        capabilities.architecture.lower() != architecture
        and not capabilities.platform_emulation
    ):
        diagnostics.append(
            _capability_mismatch(
                backend,
                f"architecture {architecture} requires platform emulation",
            )
        )
    if (
        plan.image.pull_policy is not ContainerPullPolicy.NEVER
        and not capabilities.pull
    ):
        diagnostics.append(
            _capability_mismatch(backend, "image pull is not supported")
        )
    if (
        plan.image.build_policy is not ContainerBuildPolicy.DISABLED
        and not capabilities.build
    ):
        diagnostics.append(
            _capability_mismatch(backend, "image build is not supported")
        )
    network_mode = cast(ContainerNetworkMode, plan.network.mode)
    if network_mode not in capabilities.network_modes:
        diagnostics.append(
            _capability_mismatch(
                backend,
                f"network mode {network_mode.value} is not supported",
            )
        )
    unsupported_mounts = [
        cast(ContainerMountType, mount.mount_type)
        for mount in plan.mounts
        if mount.mount_type not in capabilities.mount_types
    ]
    if unsupported_mounts:
        diagnostics.append(
            _capability_mismatch(
                backend,
                f"mount type {unsupported_mounts[0].value} is not supported",
            )
        )
    unsupported_devices = [
        cast(ContainerDeviceClass, device)
        for device in plan.devices.devices
        if device not in capabilities.device_classes
    ]
    if unsupported_devices:
        diagnostics.append(
            _capability_mismatch(
                backend,
                f"device class {unsupported_devices[0].value} "
                "is not supported",
            )
        )
    resources = plan.resources
    if (
        resources.cpu_count is not None
        or resources.memory_bytes is not None
        or resources.pids is not None
        or resources.timeout_seconds is not None
    ) and not capabilities.resource_limits:
        diagnostics.append(
            _capability_mismatch(backend, "resource limits are not supported")
        )
    if not capabilities.streaming_attach:
        diagnostics.append(
            _capability_mismatch(backend, "streaming attach is not supported")
        )
    return tuple(diagnostics)


def _selection_score(
    capabilities: ContainerBackendCapabilities,
) -> tuple[int, int, str]:
    backend = cast(ContainerBackend, capabilities.backend)
    isolation_score = 0
    if capabilities.rootless:
        isolation_score -= 2
    if capabilities.per_container_vm_isolation:
        isolation_score -= 1
    return (
        isolation_score,
        0 if capabilities.stats else 1,
        backend.value,
    )


def _capability_mismatch(
    backend: ContainerBackend,
    message: str,
) -> ContainerBackendDiagnostic:
    return _diagnostic(
        ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        ContainerBackendOperation.PROBE,
        backend,
        message,
    )


def _backend_unavailable_diagnostic(
    backend: ContainerBackend | None,
    message: str,
) -> ContainerBackendDiagnostic:
    return _diagnostic(
        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        ContainerBackendOperation.PROBE,
        backend,
        message,
        retryable=True,
    )


def _platform_parts(platform: str) -> tuple[str, str]:
    parts = platform.split("/", 1)
    assert len(parts) == 2, "platform must include guest OS and architecture"
    return parts[0].lower(), parts[1].lower()


def _diagnostic(
    code: ContainerBackendDiagnosticCode | str,
    operation: ContainerBackendOperation | str,
    backend: ContainerBackend | str | None,
    message: str,
    *,
    retryable: bool = False,
) -> ContainerBackendDiagnostic:
    return ContainerBackendDiagnostic(
        code=code,
        operation=operation,
        backend=backend,
        message=message,
        retryable=retryable,
    )


def _diagnostic_message(
    code: ContainerBackendDiagnosticCode,
    operation: ContainerBackendOperation,
) -> str:
    return f"{operation.value} failed with {code.value}"


def _diagnostic_text(diagnostic: ContainerBackendDiagnostic) -> str:
    code = cast(ContainerBackendDiagnosticCode, diagnostic.code)
    operation = cast(ContainerBackendOperation, diagnostic.operation)
    return f"{code.value}:{operation.value}:{diagnostic.message}"


def _status_for_diagnostic(
    diagnostic: ContainerBackendDiagnostic,
) -> ContainerResultStatus:
    code = cast(ContainerBackendDiagnosticCode, diagnostic.code)
    if code in {
        ContainerBackendDiagnosticCode.IMAGE_DENIED,
        ContainerBackendDiagnosticCode.PULL_DENIED,
        ContainerBackendDiagnosticCode.BUILD_DENIED,
        ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        ContainerBackendDiagnosticCode.ROOTFUL_NOT_AUTHORIZED,
    }:
        return ContainerResultStatus.DENIED
    return ContainerResultStatus.FAILED


def _status_for_diagnostics(
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> ContainerResultStatus:
    for diagnostic in diagnostics:
        if _status_for_diagnostic(diagnostic) is ContainerResultStatus.DENIED:
            return ContainerResultStatus.DENIED
    return ContainerResultStatus.FAILED


def _is_retryable(code: ContainerBackendDiagnosticCode) -> bool:
    return code in {
        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        ContainerBackendDiagnosticCode.PULL_FAILED,
        ContainerBackendDiagnosticCode.BUILD_FAILED,
        ContainerBackendDiagnosticCode.CREATE_FAILED,
        ContainerBackendDiagnosticCode.ATTACH_FAILED,
        ContainerBackendDiagnosticCode.START_FAILED,
        ContainerBackendDiagnosticCode.WAIT_FAILED,
        ContainerBackendDiagnosticCode.COPY_FAILED,
        ContainerBackendDiagnosticCode.CLEANUP_FAILED,
        ContainerBackendDiagnosticCode.TIMEOUT,
    }


def _plan_fingerprint(plan: ContainerRunPlan) -> str:
    backend = cast(ContainerBackend, plan.backend)
    return (
        f"{backend.value}:"
        f"{plan.profile_name}:"
        f"{plan.image.digest}:"
        f"{plan.command.tool_name}"
    )


def _assert_diagnostics(
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> None:
    for diagnostic in diagnostics:
        assert isinstance(diagnostic, ContainerBackendDiagnostic)


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(
        value, str
    ), f"{field_name} must be a sequence"
    result = tuple(value)
    for item in result:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
    return result


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    result = dict(value)
    for key, item in result.items():
        _assert_non_empty_string(key, field_name)
        _assert_non_empty_string(item, field_name)
        assert isinstance(key, str)
        assert isinstance(item, str)
    return result


def _enum_value(
    value: object,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert value in {
        member.value for member in enum_type
    }, f"{field_name} contains unsupported value"
    return enum_type(value)
