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
    ContainerBackendSupportLevel,
    ContainerBuildPolicy,
    ContainerDeviceClass,
    ContainerExecutionResult,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerPlatformBehavior,
    ContainerPullPolicy,
    ContainerRunPlan,
)

from abc import ABC, abstractmethod
from asyncio import CancelledError, sleep
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
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
    STREAM_TRUNCATED = "container.backend.stream_truncated"
    EVENT_DROPPED = "container.backend.event_dropped"


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
        assert isinstance(
            self.environment_variables, Sequence
        ) and not isinstance(
            self.environment_variables, str | bytes
        ), "environment_variables must be a sequence"
        environment_variables = tuple(self.environment_variables)
        for variable in environment_variables:
            _assert_non_empty_string(variable, "environment_variables")
            assert isinstance(variable, str)
        object.__setattr__(
            self,
            "environment_variables",
            environment_variables,
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
class ContainerBackendCapabilityProfile:
    profile_id: str
    capabilities: ContainerBackendCapabilities
    runtime_requirements: ContainerBackendRuntimeRequirements = field(
        default_factory=ContainerBackendRuntimeRequirements,
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.profile_id, "profile_id")
        assert isinstance(self.capabilities, ContainerBackendCapabilities)
        assert isinstance(
            self.runtime_requirements,
            ContainerBackendRuntimeRequirements,
        )

    @property
    def backend(self) -> ContainerBackend:
        return cast(ContainerBackend, self.capabilities.backend)

    @property
    def support_level(self) -> ContainerBackendSupportLevel:
        return cast(
            ContainerBackendSupportLevel, self.capabilities.support_level
        )

    @property
    def promoted(self) -> bool:
        return self.support_level is ContainerBackendSupportLevel.SUPPORTED

    def probe(
        self,
        *,
        available: bool = False,
        diagnostics: Sequence[ContainerBackendDiagnostic] = (),
    ) -> ContainerBackendProbeResult:
        _assert_bool(available, "available")
        _assert_diagnostics(diagnostics)
        if not available:
            diagnostics = tuple(diagnostics) or (
                _backend_unavailable_diagnostic(
                    self.backend,
                    f"{self.profile_id} runtime is unavailable",
                ),
            )
        return ContainerBackendProbeResult(
            backend=self.backend,
            available=available,
            capabilities=self.capabilities if available else None,
            diagnostics=diagnostics,
            runtime_requirements=self.runtime_requirements,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "promoted": self.promoted,
            "capabilities": self.capabilities.to_dict(),
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
    ) -> (
        Sequence[ContainerBackendStreamChunk]
        | AsyncIterable[ContainerBackendStreamChunk]
    ):
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


def container_backend_capability_profiles(
    backend: ContainerBackend | str | None = None,
) -> tuple[ContainerBackendCapabilityProfile, ...]:
    if backend is None:
        return _CONTAINER_BACKEND_CAPABILITY_PROFILES
    selected = _enum_value(backend, ContainerBackend, "backend")
    return tuple(
        profile
        for profile in _CONTAINER_BACKEND_CAPABILITY_PROFILES
        if profile.backend is selected
    )


def container_backend_capability_profile(
    profile_id: str,
) -> ContainerBackendCapabilityProfile:
    _assert_non_empty_string(profile_id, "profile_id")
    for profile in _CONTAINER_BACKEND_CAPABILITY_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    assert False, "container backend capability profile is unknown"


def container_backend_probe_from_profile(
    profile_id: str,
    *,
    available: bool = False,
    diagnostics: Sequence[ContainerBackendDiagnostic] = (),
) -> ContainerBackendProbeResult:
    return container_backend_capability_profile(profile_id).probe(
        available=available,
        diagnostics=diagnostics,
    )


def _runtime_requirements(
    marker: str,
    environment_variable: str,
) -> ContainerBackendRuntimeRequirements:
    return ContainerBackendRuntimeRequirements(
        marker=marker,
        environment_variables=(environment_variable,),
    )


def _platform_behavior(
    *,
    file_io: str,
    networking: str,
    architecture_emulation: str,
    resources: str,
    signals: str,
    path_syntax: str,
    drive_letters: str,
    case_behavior: str,
) -> ContainerPlatformBehavior:
    return ContainerPlatformBehavior(
        file_io=file_io,
        networking=networking,
        architecture_emulation=architecture_emulation,
        resources=resources,
        signals=signals,
        path_syntax=path_syntax,
        drive_letters=drive_letters,
        case_behavior=case_behavior,
    )


def _capability_profile(
    profile_id: str,
    *,
    backend: ContainerBackend,
    runtime_name: str,
    support_level: ContainerBackendSupportLevel,
    host_os: str,
    guest_os: str,
    architecture: str,
    platform_emulation: bool,
    rootless: bool,
    user_namespace: bool,
    build: bool,
    pull: bool,
    network_modes: Sequence[ContainerNetworkMode],
    mount_types: Sequence[ContainerMountType],
    resource_limits: bool,
    device_classes: Sequence[ContainerDeviceClass],
    platform_behavior: ContainerPlatformBehavior,
    environment_variable: str,
    per_container_vm_isolation: bool = False,
    vm_backed: bool = False,
    remote_engine: bool = False,
    windows_process_isolation: bool = False,
    windows_hyperv_isolation: bool = False,
    streaming_attach: bool = True,
    stats: bool = True,
    lifecycle_normalization: bool = True,
    shared_mount_prefixes: Sequence[str] = (),
    parity_requirements: Sequence[str] = (),
) -> ContainerBackendCapabilityProfile:
    return ContainerBackendCapabilityProfile(
        profile_id=profile_id,
        capabilities=ContainerBackendCapabilities(
            backend=backend,
            host_os=host_os,
            guest_os=guest_os,
            architecture=architecture,
            runtime_name=runtime_name,
            support_level=support_level,
            platform_emulation=platform_emulation,
            rootless=rootless,
            user_namespace=user_namespace,
            build=build,
            pull=pull,
            network_modes=network_modes,
            mount_types=mount_types,
            resource_limits=resource_limits,
            device_classes=device_classes,
            per_container_vm_isolation=per_container_vm_isolation,
            vm_backed=vm_backed,
            remote_engine=remote_engine,
            windows_process_isolation=windows_process_isolation,
            windows_hyperv_isolation=windows_hyperv_isolation,
            streaming_attach=streaming_attach,
            stats=stats,
            lifecycle_normalization=lifecycle_normalization,
            platform_behavior=platform_behavior,
            shared_mount_prefixes=shared_mount_prefixes,
            parity_requirements=parity_requirements,
        ),
        runtime_requirements=_runtime_requirements(
            profile_id,
            environment_variable,
        ),
    )


_LINUX_MOUNT_TYPES = (
    ContainerMountType.INPUT,
    ContainerMountType.WORKSPACE,
    ContainerMountType.SCRATCH,
    ContainerMountType.OUTPUT,
    ContainerMountType.CACHE,
)
_WINDOWS_MOUNT_TYPES = (
    ContainerMountType.INPUT,
    ContainerMountType.WORKSPACE,
    ContainerMountType.SCRATCH,
    ContainerMountType.OUTPUT,
)
_CPU_DEVICE_CLASSES = (ContainerDeviceClass.CPU,)
_DOCKER_NETWORK_MODES = (
    ContainerNetworkMode.NONE,
    ContainerNetworkMode.LOOPBACK,
    ContainerNetworkMode.FULL,
)
_ROOTLESS_NETWORK_MODES = (
    ContainerNetworkMode.NONE,
    ContainerNetworkMode.LOOPBACK,
    ContainerNetworkMode.FULL,
)

_CONTAINER_BACKEND_CAPABILITY_PROFILES = (
    _capability_profile(
        "docker-engine-linux",
        backend=ContainerBackend.DOCKER,
        runtime_name="Docker Engine",
        support_level=ContainerBackendSupportLevel.SUPPORTED,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        platform_emulation=True,
        rootless=False,
        user_namespace=True,
        build=True,
        pull=True,
        network_modes=_DOCKER_NETWORK_MODES,
        mount_types=_LINUX_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        platform_behavior=_platform_behavior(
            file_io="native host filesystem bind mounts",
            networking="engine bridge, host-controlled none, and full modes",
            architecture_emulation="qemu or binfmt when configured by host",
            resources="cgroup-backed CPU, memory, pids, and timeout limits",
            signals="OCI process signals delivered by Docker Engine",
            path_syntax="POSIX host paths",
            drive_letters="not applicable",
            case_behavior="host filesystem dependent, usually sensitive",
        ),
        environment_variable="AVALAN_CONTAINER_DOCKER_E2E",
    ),
    _capability_profile(
        "docker-desktop-macos-linux",
        backend=ContainerBackend.DOCKER,
        runtime_name="Docker Desktop Linux VM",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="darwin",
        guest_os="linux",
        architecture="arm64",
        platform_emulation=True,
        rootless=False,
        user_namespace=False,
        build=True,
        pull=True,
        network_modes=_DOCKER_NETWORK_MODES,
        mount_types=_LINUX_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        vm_backed=True,
        remote_engine=True,
        platform_behavior=_platform_behavior(
            file_io="shared-directory VM file I/O with host sync overhead",
            networking="Docker Desktop VM networking and port forwarding",
            architecture_emulation="Apple silicon arm64 with amd64 emulation",
            resources="Docker Desktop VM resource ceilings apply first",
            signals="signals cross the Docker Desktop VM boundary",
            path_syntax="macOS POSIX paths from shared directories",
            drive_letters="not applicable",
            case_behavior="host volume case behavior may differ from guest",
        ),
        shared_mount_prefixes=("/Users/", "/Volumes/", "/private/", "/tmp/"),
        environment_variable="AVALAN_CONTAINER_DOCKER_E2E",
    ),
    _capability_profile(
        "docker-desktop-wsl2-linux",
        backend=ContainerBackend.DOCKER,
        runtime_name="Docker Desktop WSL2 Linux",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="windows",
        guest_os="linux",
        architecture="amd64",
        platform_emulation=True,
        rootless=False,
        user_namespace=False,
        build=True,
        pull=True,
        network_modes=_DOCKER_NETWORK_MODES,
        mount_types=_LINUX_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        vm_backed=True,
        remote_engine=True,
        platform_behavior=_platform_behavior(
            file_io="WSL2 or Windows shared path file I/O",
            networking="WSL2 VM networking with Docker Desktop forwarding",
            architecture_emulation="amd64 native or host-provided emulation",
            resources="Docker Desktop and WSL2 limits both apply",
            signals="signals cross the WSL2 VM boundary",
            path_syntax="WSL POSIX paths or shared Windows drive paths",
            drive_letters="Windows drive letters must be normalized",
            case_behavior="case behavior depends on WSL or NTFS location",
        ),
        shared_mount_prefixes=("C:\\", "D:\\", "/mnt/"),
        environment_variable="AVALAN_CONTAINER_DOCKER_E2E",
    ),
    _capability_profile(
        "podman-rootless-linux",
        backend=ContainerBackend.PODMAN,
        runtime_name="Podman rootless",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        platform_emulation=False,
        rootless=True,
        user_namespace=True,
        build=True,
        pull=True,
        network_modes=_ROOTLESS_NETWORK_MODES,
        mount_types=_LINUX_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        platform_behavior=_platform_behavior(
            file_io="native rootless bind mounts with user namespace mapping",
            networking="slirp or pasta rootless networking where configured",
            architecture_emulation="host-native unless binfmt is configured",
            resources="cgroup v2 rootless limits when available",
            signals="OCI process signals delivered by Podman",
            path_syntax="POSIX host paths",
            drive_letters="not applicable",
            case_behavior="host filesystem dependent, usually sensitive",
        ),
        environment_variable="AVALAN_CONTAINER_PODMAN_E2E",
    ),
    _capability_profile(
        "podman-machine-macos-linux",
        backend=ContainerBackend.PODMAN,
        runtime_name="Podman machine Linux VM",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="darwin",
        guest_os="linux",
        architecture="arm64",
        platform_emulation=True,
        rootless=True,
        user_namespace=True,
        build=True,
        pull=True,
        network_modes=_ROOTLESS_NETWORK_MODES,
        mount_types=_LINUX_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        vm_backed=True,
        remote_engine=True,
        platform_behavior=_platform_behavior(
            file_io="Podman machine shared-directory VM file I/O",
            networking="Podman machine VM networking and forwarding",
            architecture_emulation="Apple silicon arm64 with VM emulation",
            resources="Podman machine VM resource ceilings apply first",
            signals="signals cross the Podman machine VM boundary",
            path_syntax="macOS POSIX paths from shared directories",
            drive_letters="not applicable",
            case_behavior="host volume case behavior may differ from guest",
        ),
        shared_mount_prefixes=("/Users/", "/Volumes/", "/private/", "/tmp/"),
        environment_variable="AVALAN_CONTAINER_PODMAN_E2E",
    ),
    _capability_profile(
        "nerdctl-containerd-linux",
        backend=ContainerBackend.NERDCTL,
        runtime_name="nerdctl containerd",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        platform_emulation=False,
        rootless=False,
        user_namespace=False,
        build=True,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE, ContainerNetworkMode.FULL),
        mount_types=_LINUX_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        platform_behavior=_platform_behavior(
            file_io="containerd bind mounts through nerdctl",
            networking="CNI-backed none or full modes",
            architecture_emulation="host-native unless binfmt is configured",
            resources="containerd and cgroup resource limits",
            signals="OCI process signals delivered through containerd",
            path_syntax="POSIX host paths",
            drive_letters="not applicable",
            case_behavior="host filesystem dependent, usually sensitive",
        ),
        environment_variable="AVALAN_CONTAINER_NERDCTL_E2E",
    ),
    _capability_profile(
        "apple-container-macos-linux",
        backend=ContainerBackend.APPLE_CONTAINER,
        runtime_name="Apple container",
        support_level=ContainerBackendSupportLevel.OPT_IN,
        host_os="darwin",
        guest_os="linux",
        architecture="arm64",
        platform_emulation=False,
        rootless=False,
        user_namespace=False,
        build=False,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE, ContainerMountType.OUTPUT),
        resource_limits=False,
        device_classes=_CPU_DEVICE_CLASSES,
        per_container_vm_isolation=True,
        vm_backed=True,
        streaming_attach=False,
        stats=False,
        lifecycle_normalization=False,
        platform_behavior=_platform_behavior(
            file_io="Apple Containerization VM file sharing",
            networking="Apple container VM networking, opt-in until proven",
            architecture_emulation="Apple silicon linux/arm64 baseline only",
            resources="resource controls require parity validation",
            signals="signal behavior requires parity validation",
            path_syntax="macOS POSIX paths from shared locations",
            drive_letters="not applicable",
            case_behavior="host volume case behavior may differ from guest",
        ),
        shared_mount_prefixes=("/Users/", "/Volumes/", "/private/", "/tmp/"),
        parity_requirements=(
            "streaming attach parity",
            "mount behavior parity",
            "cleanup certainty",
            "startup and file I/O performance budgets",
        ),
        environment_variable="AVALAN_CONTAINER_APPLE_E2E",
    ),
    _capability_profile(
        "windows-docker-process",
        backend=ContainerBackend.WINDOWS_DOCKER,
        runtime_name="Windows Docker process isolation",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="windows",
        guest_os="windows",
        architecture="amd64",
        platform_emulation=False,
        rootless=False,
        user_namespace=False,
        build=True,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE, ContainerNetworkMode.FULL),
        mount_types=_WINDOWS_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        windows_process_isolation=True,
        platform_behavior=_platform_behavior(
            file_io="Windows bind mounts with NTFS semantics",
            networking="Windows container NAT or none modes",
            architecture_emulation=(
                "Windows process isolation is host-version bound"
            ),
            resources="Windows job object and Docker resource controls",
            signals="Windows stop events differ from POSIX signals",
            path_syntax="Windows drive-letter paths",
            drive_letters="drive letters are required for host paths",
            case_behavior="case-insensitive by default",
        ),
        environment_variable="AVALAN_CONTAINER_WINDOWS_DOCKER_E2E",
    ),
    _capability_profile(
        "windows-docker-hyperv",
        backend=ContainerBackend.WINDOWS_DOCKER,
        runtime_name="Windows Docker Hyper-V isolation",
        support_level=ContainerBackendSupportLevel.OPTIONAL,
        host_os="windows",
        guest_os="windows",
        architecture="amd64",
        platform_emulation=False,
        rootless=False,
        user_namespace=False,
        build=True,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE, ContainerNetworkMode.FULL),
        mount_types=_WINDOWS_MOUNT_TYPES,
        resource_limits=True,
        device_classes=_CPU_DEVICE_CLASSES,
        per_container_vm_isolation=True,
        vm_backed=True,
        windows_hyperv_isolation=True,
        platform_behavior=_platform_behavior(
            file_io="Windows Hyper-V isolated bind mounts",
            networking="Hyper-V isolated Windows container networking",
            architecture_emulation=(
                "Windows image architecture must match host"
            ),
            resources="Hyper-V and Docker resource controls",
            signals="Windows stop events cross the Hyper-V boundary",
            path_syntax="Windows drive-letter paths",
            drive_letters="drive letters are required for host paths",
            case_behavior="case-insensitive by default",
        ),
        environment_variable="AVALAN_CONTAINER_WINDOWS_DOCKER_E2E",
    ),
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
    operation_delay_seconds: Mapping[
        ContainerBackendOperation | str,
        float,
    ] = field(default_factory=dict)
    stream_chunks: Sequence[ContainerBackendStreamChunk] = field(
        default_factory=lambda: (
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDOUT,
                content=b"ok\n",
                sequence=0,
            ),
        ),
    )
    stream_incremental: bool = False
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
        operation_delay_seconds: dict[ContainerBackendOperation, float] = {}
        for operation, seconds in self.operation_delay_seconds.items():
            assert isinstance(seconds, int | float)
            assert seconds >= 0, "operation delay must not be negative"
            operation_delay_seconds[
                _enum_value(operation, ContainerBackendOperation, "operation")
            ] = float(seconds)
        object.__setattr__(
            self,
            "operation_delay_seconds",
            MappingProxyType(operation_delay_seconds),
        )
        for chunk in self.stream_chunks:
            assert isinstance(chunk, ContainerBackendStreamChunk)
        _assert_bool(self.stream_incremental, "stream_incremental")
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
    ) -> (
        Sequence[ContainerBackendStreamChunk]
        | AsyncIterable[ContainerBackendStreamChunk]
    ):
        assert isinstance(container, ContainerBackendContainer)
        await self._enter(ContainerBackendOperation.STREAM)
        if self._script.stream_incremental:
            return self._stream_incrementally()
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
        delay_seconds = self._script.operation_delay_seconds.get(operation)
        if delay_seconds:
            await sleep(delay_seconds)
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

    async def _stream_incrementally(
        self,
    ) -> AsyncIterator[ContainerBackendStreamChunk]:
        for chunk in self._script.stream_chunks:
            if self._script.stream_delay_seconds:
                await sleep(self._script.stream_delay_seconds)
            yield chunk


def select_container_backend(
    plan: ContainerRunPlan,
    probes: Sequence[ContainerBackendProbeResult],
    *,
    auto_enabled: bool,
    rootful_authorized: bool = False,
    opt_in_backends: Sequence[ContainerBackend | str] = (),
) -> ContainerBackendSelection:
    assert isinstance(plan, ContainerRunPlan)
    for probe in probes:
        assert isinstance(probe, ContainerBackendProbeResult)
    requested = cast(ContainerBackend, plan.backend)
    opt_in = frozenset(
        _enum_value(backend, ContainerBackend, "opt_in_backends")
        for backend in opt_in_backends
    )
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
        if not probe.ok:
            if probe.available and probe.capabilities is not None:
                continue
            diagnostics.append(
                _backend_unavailable_diagnostic(
                    cast(ContainerBackend, probe.backend),
                    "backend is unavailable",
                )
            )
            continue
        capabilities = probe.capabilities
        assert capabilities is not None
        capability_diagnostics = _capability_diagnostics(
            plan,
            capabilities,
            rootful_authorized=rootful_authorized,
            opt_in_backends=opt_in,
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
    from .lifecycle import (
        run_container_managed_lifecycle as _run_container_managed_lifecycle,
    )

    return (
        await _run_container_managed_lifecycle(
            backend,
            plan,
            output_contract=output_contract,
        )
    ).to_backend_result()


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
    opt_in_backends: frozenset[ContainerBackend],
) -> tuple[ContainerBackendDiagnostic, ...]:
    backend = cast(ContainerBackend, capabilities.backend)
    diagnostics: list[ContainerBackendDiagnostic] = []
    support_level = cast(
        ContainerBackendSupportLevel,
        capabilities.support_level,
    )
    if support_level is ContainerBackendSupportLevel.CATALOG_ONLY:
        diagnostics.append(
            _capability_mismatch(
                backend,
                "backend support level catalog_only is not selectable",
            )
        )
    if (
        support_level is ContainerBackendSupportLevel.OPT_IN
        and backend not in opt_in_backends
    ):
        diagnostics.append(
            _capability_mismatch(
                backend,
                "backend requires explicit operator opt-in",
            )
        )
    if not capabilities.lifecycle_normalization:
        diagnostics.append(
            _capability_mismatch(
                backend,
                "lifecycle normalization is not supported",
            )
        )
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
    diagnostics.extend(_vm_mount_diagnostics(plan, capabilities, backend))
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


def _vm_mount_diagnostics(
    plan: ContainerRunPlan,
    capabilities: ContainerBackendCapabilities,
    backend: ContainerBackend,
) -> tuple[ContainerBackendDiagnostic, ...]:
    if not capabilities.vm_backed or not capabilities.shared_mount_prefixes:
        return ()
    for mount in plan.mounts:
        if mount.source is None or not _host_path_is_absolute(mount.source):
            continue
        if not _host_path_has_shared_prefix(
            mount.source,
            capabilities.shared_mount_prefixes,
        ):
            return (
                _capability_mismatch(
                    backend,
                    "VM-backed runtime cannot mount source outside "
                    "declared shared prefixes",
                ),
            )
    return ()


def _host_path_is_absolute(path: str) -> bool:
    return (
        path.startswith("/")
        or _host_path_has_windows_drive(path)
        or _host_path_is_unc(path)
    )


def _host_path_has_windows_drive(path: str) -> bool:
    return (
        len(path) >= 3
        and path[1] == ":"
        and path[2] in {"\\", "/"}
        and path[0].isalpha()
    )


def _host_path_is_unc(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")


def _host_path_has_shared_prefix(
    path: str,
    prefixes: Sequence[str],
) -> bool:
    normalized_path = _normalized_host_path(path)
    if normalized_path is None:
        return False
    for prefix in prefixes:
        normalized_prefix = _normalized_host_path(prefix)
        if normalized_prefix is None:
            continue
        if _normalized_path_is_inside_prefix(
            normalized_path,
            normalized_prefix,
        ):
            return True
    return False


def _normalized_host_path(
    path: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    if _host_path_is_unc(path):
        return _normalized_unc_path(path)
    if _host_path_has_windows_drive(path):
        return _normalized_windows_drive_path(path)
    if path.startswith("/"):
        return (
            "posix",
            "/",
            _normalized_path_parts(path.split("/"), case_sensitive=True),
        )
    return None


def _normalized_unc_path(
    path: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    parts = [
        part.casefold() for part in path.replace("\\", "/").split("/") if part
    ]
    if len(parts) < 2:
        return None
    return (
        "unc",
        f"{parts[0]}/{parts[1]}",
        _normalized_path_parts(parts[2:], case_sensitive=False),
    )


def _normalized_windows_drive_path(
    path: str,
) -> tuple[str, str, tuple[str, ...]]:
    normalized_path = path.replace("\\", "/")
    return (
        "windows-drive",
        normalized_path[:2].casefold(),
        _normalized_path_parts(
            normalized_path[3:].split("/"),
            case_sensitive=False,
        ),
    )


def _normalized_path_parts(
    parts: Sequence[str],
    *,
    case_sensitive: bool,
) -> tuple[str, ...]:
    normalized_parts: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if normalized_parts:
                normalized_parts.pop()
            continue
        normalized_parts.append(part if case_sensitive else part.casefold())
    return tuple(normalized_parts)


def _normalized_path_is_inside_prefix(
    path: tuple[str, str, tuple[str, ...]],
    prefix: tuple[str, str, tuple[str, ...]],
) -> bool:
    path_family, path_anchor, path_parts = path
    prefix_family, prefix_anchor, prefix_parts = prefix
    return (
        path_family == prefix_family
        and path_anchor == prefix_anchor
        and len(path_parts) >= len(prefix_parts)
        and path_parts[: len(prefix_parts)] == prefix_parts
    )


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
