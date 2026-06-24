from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .audit import (
    ContainerAuditCorrelation,
    ContainerAuditRecord,
    ContainerMappedDiagnostic,
    ContainerStableDiagnosticCode,
)
from .conformance import (
    ContainerBackend,
    ContainerExecutionScope,
)
from .planning import (
    ContainerDurablePlanMetadata,
    ContainerNormalizedRunPlan,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerPlanRequestKind,
    ContainerRuntimeEnvelopeKind,
)
from .settings import (
    ContainerAuditEventType,
    ContainerDeviceClass,
    ContainerExecutionResult,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerNetworkMode,
    ContainerResultStatus,
)

from abc import ABC, abstractmethod
from asyncio import sleep, wait_for
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from posixpath import normpath as normalize_posix_path
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)
RuntimeOperationResult = TypeVar("RuntimeOperationResult")

_RUNTIME_SCOPE = ContainerExecutionScope.RUNTIME_ENVELOPE


class ContainerRuntimeEnvelopeOperation(StrEnum):
    START = "start"
    READINESS = "readiness"
    SCOPED_EXECUTION = "scoped_execution"
    HEALTH = "health"
    TELEMETRY = "telemetry"
    HANDOFF = "handoff"
    SHUTDOWN = "shutdown"
    CLEANUP = "cleanup"


class ContainerRuntimeEnvelopeState(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    CLEANED = "cleaned"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeHandle:
    envelope_id: str
    envelope_kind: ContainerRuntimeEnvelopeKind | str
    backend: ContainerBackend | str
    plan_fingerprint: str
    state: ContainerRuntimeEnvelopeState | str = (
        ContainerRuntimeEnvelopeState.CREATED
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.envelope_id, "envelope_id")
        object.__setattr__(
            self,
            "envelope_kind",
            _enum_value(
                self.envelope_kind,
                ContainerRuntimeEnvelopeKind,
                "envelope_kind",
            ),
        )
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, ContainerBackend, "backend"),
        )
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")
        object.__setattr__(
            self,
            "state",
            _enum_value(
                self.state,
                ContainerRuntimeEnvelopeState,
                "state",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        envelope_kind = cast(ContainerRuntimeEnvelopeKind, self.envelope_kind)
        backend = cast(ContainerBackend, self.backend)
        state = cast(ContainerRuntimeEnvelopeState, self.state)
        return {
            "envelope_id": self.envelope_id,
            "envelope_kind": envelope_kind.value,
            "backend": backend.value,
            "plan_fingerprint": self.plan_fingerprint,
            "state": state.value,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeReadiness:
    ready: bool
    message: str = "ready"
    diagnostics: Sequence[ContainerMappedDiagnostic] = field(
        default_factory=tuple,
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_bool(self.ready, "ready")
        _assert_non_empty_string(self.message, "message")
        _assert_mapped_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    @property
    def ok(self) -> bool:
        return self.ready and not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "message": self.message,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeHealth:
    healthy: bool
    status: str = "healthy"
    diagnostics: Sequence[ContainerMappedDiagnostic] = field(
        default_factory=tuple,
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_bool(self.healthy, "healthy")
        _assert_non_empty_string(self.status, "status")
        _assert_mapped_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    @property
    def ok(self) -> bool:
        return self.healthy and not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "healthy": self.healthy,
            "status": self.status,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeHandoff:
    metadata: ContainerDurablePlanMetadata
    telemetry: Mapping[str, str] = field(default_factory=dict)
    state: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert isinstance(self.metadata, ContainerDurablePlanMetadata)
        object.__setattr__(
            self,
            "telemetry",
            MappingProxyType(_string_mapping(self.telemetry, "telemetry")),
        )
        object.__setattr__(
            self,
            "state",
            MappingProxyType(_string_mapping(self.state, "state")),
        )
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(_string_mapping(self.artifacts, "artifacts")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_dict(),
            "telemetry": dict(self.telemetry),
            "state": dict(self.state),
            "artifacts": dict(self.artifacts),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeCompositionResult:
    allowed: bool
    diagnostics: Sequence[ContainerMappedDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        _assert_bool(self.allowed, "allowed")
        _assert_mapped_diagnostics(self.diagnostics)
        assert (
            self.allowed or self.diagnostics
        ), "denied composition requires diagnostics"
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return self.allowed and not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeOperationResult:
    execution: ContainerExecutionResult
    audit_records: Sequence[ContainerAuditRecord] = field(
        default_factory=tuple,
    )
    handle: ContainerRuntimeEnvelopeHandle | None = None
    readiness: ContainerRuntimeEnvelopeReadiness | None = None
    health: ContainerRuntimeEnvelopeHealth | None = None
    handoff: ContainerRuntimeEnvelopeHandoff | None = None
    shutdown: "ContainerRuntimeEnvelopeShutdownResult | None" = None
    cleanup: "ContainerRuntimeEnvelopeCleanupResult | None" = None

    def __post_init__(self) -> None:
        assert isinstance(self.execution, ContainerExecutionResult)
        for record in self.audit_records:
            assert isinstance(record, ContainerAuditRecord)
        if self.handle is not None:
            assert isinstance(self.handle, ContainerRuntimeEnvelopeHandle)
        if self.readiness is not None:
            assert isinstance(
                self.readiness, ContainerRuntimeEnvelopeReadiness
            )
        if self.health is not None:
            assert isinstance(self.health, ContainerRuntimeEnvelopeHealth)
        if self.handoff is not None:
            assert isinstance(self.handoff, ContainerRuntimeEnvelopeHandoff)
        if self.shutdown is not None:
            assert isinstance(
                self.shutdown,
                ContainerRuntimeEnvelopeShutdownResult,
            )
        if self.cleanup is not None:
            assert isinstance(
                self.cleanup,
                ContainerRuntimeEnvelopeCleanupResult,
            )
        object.__setattr__(self, "audit_records", tuple(self.audit_records))

    def to_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution.to_dict(),
            "audit_records": [
                record.to_dict() for record in self.audit_records
            ],
            "handle": None if self.handle is None else self.handle.to_dict(),
            "readiness": (
                None if self.readiness is None else self.readiness.to_dict()
            ),
            "health": None if self.health is None else self.health.to_dict(),
            "handoff": (
                None if self.handoff is None else self.handoff.to_dict()
            ),
            "shutdown": (
                None if self.shutdown is None else self.shutdown.to_dict()
            ),
            "cleanup": (
                None if self.cleanup is None else self.cleanup.to_dict()
            ),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeScopedExecutionResult:
    execution: ContainerExecutionResult
    composition: ContainerRuntimeEnvelopeCompositionResult
    telemetry: Mapping[str, str] = field(default_factory=dict)
    health: ContainerRuntimeEnvelopeHealth | None = None
    audit_records: Sequence[ContainerAuditRecord] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.execution, ContainerExecutionResult)
        assert isinstance(
            self.composition,
            ContainerRuntimeEnvelopeCompositionResult,
        )
        if self.health is not None:
            assert isinstance(self.health, ContainerRuntimeEnvelopeHealth)
        for record in self.audit_records:
            assert isinstance(record, ContainerAuditRecord)
        object.__setattr__(
            self,
            "telemetry",
            MappingProxyType(_string_mapping(self.telemetry, "telemetry")),
        )
        object.__setattr__(self, "audit_records", tuple(self.audit_records))

    def to_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution.to_dict(),
            "composition": self.composition.to_dict(),
            "health": None if self.health is None else self.health.to_dict(),
            "telemetry": dict(self.telemetry),
            "audit_records": [
                record.to_dict() for record in self.audit_records
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeShutdownResult:
    graceful: bool
    diagnostics: Sequence[ContainerMappedDiagnostic] = field(
        default_factory=tuple,
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_bool(self.graceful, "graceful")
        _assert_mapped_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    @property
    def ok(self) -> bool:
        return self.graceful and not self.diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "graceful": self.graceful,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeCleanupResult:
    cleanup_completed: bool
    cleanup_uncertain: bool = False
    diagnostics: Sequence[ContainerMappedDiagnostic] = field(
        default_factory=tuple,
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_bool(self.cleanup_completed, "cleanup_completed")
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        _assert_mapped_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    @property
    def ok(self) -> bool:
        return (
            self.cleanup_completed
            and not self.cleanup_uncertain
            and not self.diagnostics
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cleanup_completed": self.cleanup_completed,
            "cleanup_uncertain": self.cleanup_uncertain,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopeCompositionPolicy:
    reject_double_mounts: bool = True
    reject_policy_widening: bool = True

    def __post_init__(self) -> None:
        _assert_bool(self.reject_double_mounts, "reject_double_mounts")
        _assert_bool(self.reject_policy_widening, "reject_policy_widening")

    def validate(
        self,
        envelope_plan: ContainerNormalizedRuntimeEnvelopePlan,
        command_plan: ContainerNormalizedRunPlan,
    ) -> ContainerRuntimeEnvelopeCompositionResult:
        assert isinstance(
            envelope_plan,
            ContainerNormalizedRuntimeEnvelopePlan,
        )
        assert isinstance(command_plan, ContainerNormalizedRunPlan)
        diagnostics: list[ContainerMappedDiagnostic] = []
        if _is_nested_runtime_envelope(command_plan):
            diagnostics.append(
                _policy_diagnostic(
                    "container.runtime_envelope.nested_runtime",
                    "nested runtime envelopes are not allowed",
                )
            )
        if self.reject_double_mounts:
            diagnostics.extend(
                _double_mount_diagnostics(envelope_plan, command_plan)
            )
        if self.reject_policy_widening:
            diagnostics.extend(
                _policy_widening_diagnostics(envelope_plan, command_plan)
            )
        return ContainerRuntimeEnvelopeCompositionResult(
            allowed=not diagnostics,
            diagnostics=diagnostics,
        )


class ContainerRuntimeEnvelopeBackend(ABC):
    @abstractmethod
    async def start(
        self,
        plan: ContainerNormalizedRuntimeEnvelopePlan,
    ) -> ContainerRuntimeEnvelopeHandle:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def readiness(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeReadiness:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def execute(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        plan: ContainerNormalizedRunPlan,
    ) -> ContainerExecutionResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def health(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeHealth:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def telemetry(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> Mapping[str, str]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def handoff(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        metadata: ContainerDurablePlanMetadata,
    ) -> ContainerRuntimeEnvelopeHandoff:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def shutdown(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeShutdownResult:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def cleanup(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeCleanupResult:
        raise NotImplementedError  # pragma: no cover


@final
class ContainerRuntimeEnvelope:
    def __init__(
        self,
        backend: ContainerRuntimeEnvelopeBackend,
        plan: ContainerNormalizedRuntimeEnvelopePlan,
        *,
        correlation: ContainerAuditCorrelation,
        composition_policy: (
            ContainerRuntimeEnvelopeCompositionPolicy | None
        ) = None,
    ) -> None:
        assert isinstance(backend, ContainerRuntimeEnvelopeBackend)
        assert isinstance(plan, ContainerNormalizedRuntimeEnvelopePlan)
        assert isinstance(correlation, ContainerAuditCorrelation)
        _assert_runtime_envelope_plan(plan)
        self._backend = backend
        self._plan = plan
        self._correlation = correlation
        self._composition_policy = (
            composition_policy or ContainerRuntimeEnvelopeCompositionPolicy()
        )
        self._handle: ContainerRuntimeEnvelopeHandle | None = None
        self._state = ContainerRuntimeEnvelopeState.CREATED

    @property
    def handle(self) -> ContainerRuntimeEnvelopeHandle | None:
        return self._handle

    async def start(self) -> ContainerRuntimeEnvelopeOperationResult:
        assert self._state in {
            ContainerRuntimeEnvelopeState.CREATED,
            ContainerRuntimeEnvelopeState.STOPPED,
        }, "runtime envelope cannot be started from current state"
        diagnostics: list[ContainerMappedDiagnostic] = []
        records: list[ContainerAuditRecord] = []
        cleanup: ContainerRuntimeEnvelopeCleanupResult | None = None
        handle: ContainerRuntimeEnvelopeHandle | None = None
        self._state = ContainerRuntimeEnvelopeState.STARTING
        try:
            handle = await _bounded_runtime_operation(
                self._backend.start(self._plan),
                ContainerRuntimeEnvelopeOperation.START,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            diagnostics.append(error.diagnostic)
            self._state = ContainerRuntimeEnvelopeState.DEGRADED
            records.extend(
                _audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.START,
                    (error.diagnostic,),
                    self._correlation,
                )
            )
            return ContainerRuntimeEnvelopeOperationResult(
                execution=_execution_result(
                    diagnostics,
                    self._correlation,
                    status=ContainerResultStatus.FAILED,
                ),
                audit_records=records,
            )
        try:
            self._set_handle(handle, ContainerRuntimeEnvelopeState.STARTING)
            readiness = await wait_for(
                self._backend.readiness(handle),
                timeout=self._plan.envelope_plan.readiness_timeout_seconds,
            )
        except TimeoutError:
            readiness = ContainerRuntimeEnvelopeReadiness(
                ready=False,
                message="runtime envelope readiness timed out",
                diagnostics=(
                    _timeout_diagnostic(
                        "container.runtime_envelope.readiness_timeout",
                        "runtime envelope readiness timed out",
                    ),
                ),
            )
            diagnostics.extend(readiness.diagnostics)
            cleanup = await self._cleanup_after_failed_start(handle)
            if cleanup is not None:
                diagnostics.extend(cleanup.diagnostics)
            records.extend(
                _audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.READINESS,
                    readiness.diagnostics,
                    self._correlation,
                )
            )
            if cleanup is not None:
                records.extend(
                    _audit_records_for_operation(
                        ContainerRuntimeEnvelopeOperation.CLEANUP,
                        cleanup.diagnostics,
                        self._correlation,
                    )
                )
            return ContainerRuntimeEnvelopeOperationResult(
                execution=_execution_result(diagnostics, self._correlation),
                handle=self._handle,
                readiness=readiness,
                cleanup=cleanup,
                audit_records=records,
            )
        diagnostics.extend(readiness.diagnostics)
        if readiness.ok:
            assert self._handle is not None
            self._set_handle(
                self._handle,
                ContainerRuntimeEnvelopeState.READY,
            )
        else:
            cleanup = await self._cleanup_after_failed_start(handle)
            if cleanup is not None:
                diagnostics.extend(cleanup.diagnostics)
        records.extend(
            _audit_records_for_operation(
                ContainerRuntimeEnvelopeOperation.START,
                readiness.diagnostics,
                self._correlation,
            )
        )
        if cleanup is not None:
            records.extend(
                _audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.CLEANUP,
                    cleanup.diagnostics,
                    self._correlation,
                )
            )
        status = (
            ContainerResultStatus.COMPLETED
            if readiness.ok
            else ContainerResultStatus.FAILED
        )
        return ContainerRuntimeEnvelopeOperationResult(
            execution=_execution_result(
                diagnostics,
                self._correlation,
                status=status,
            ),
            handle=self._handle,
            readiness=readiness,
            cleanup=cleanup,
            audit_records=records,
        )

    async def execute(
        self,
        command_plan: ContainerNormalizedRunPlan,
    ) -> ContainerRuntimeEnvelopeScopedExecutionResult:
        assert isinstance(command_plan, ContainerNormalizedRunPlan)
        handle = self._require_ready_handle()
        composition = self._composition_policy.validate(
            self._plan,
            command_plan,
        )
        if not composition.ok:
            return ContainerRuntimeEnvelopeScopedExecutionResult(
                execution=_execution_result(
                    composition.diagnostics,
                    self._correlation,
                ),
                composition=composition,
                audit_records=_audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
                    composition.diagnostics,
                    self._correlation,
                ),
            )
        try:
            health = await _bounded_runtime_operation(
                self._backend.health(handle),
                ContainerRuntimeEnvelopeOperation.HEALTH,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            health = ContainerRuntimeEnvelopeHealth(
                healthy=False,
                status="timeout",
                diagnostics=(error.diagnostic,),
            )
            self._state = ContainerRuntimeEnvelopeState.DEGRADED
            self._set_handle(handle, ContainerRuntimeEnvelopeState.DEGRADED)
            return ContainerRuntimeEnvelopeScopedExecutionResult(
                execution=_execution_result(
                    (error.diagnostic,),
                    self._correlation,
                    status=ContainerResultStatus.FAILED,
                ),
                composition=composition,
                health=health,
                audit_records=_audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.HEALTH,
                    (error.diagnostic,),
                    self._correlation,
                ),
            )
        if not health.ok:
            self._state = ContainerRuntimeEnvelopeState.DEGRADED
            self._set_handle(handle, ContainerRuntimeEnvelopeState.DEGRADED)
            return ContainerRuntimeEnvelopeScopedExecutionResult(
                execution=_execution_result(
                    health.diagnostics,
                    self._correlation,
                    status=ContainerResultStatus.FAILED,
                ),
                composition=composition,
                health=health,
                audit_records=_audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.HEALTH,
                    health.diagnostics,
                    self._correlation,
                ),
            )
        try:
            execution = await _bounded_runtime_operation(
                self._backend.execute(handle, command_plan),
                ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            return ContainerRuntimeEnvelopeScopedExecutionResult(
                execution=_execution_result(
                    (error.diagnostic,),
                    self._correlation,
                    status=ContainerResultStatus.FAILED,
                ),
                composition=composition,
                health=health,
                audit_records=_audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
                    (error.diagnostic,),
                    self._correlation,
                ),
            )
        try:
            telemetry = _string_mapping(
                await _bounded_runtime_operation(
                    self._backend.telemetry(handle),
                    ContainerRuntimeEnvelopeOperation.TELEMETRY,
                    self._operation_timeout_seconds(),
                ),
                "telemetry",
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            return ContainerRuntimeEnvelopeScopedExecutionResult(
                execution=_execution_result(
                    (error.diagnostic,),
                    self._correlation,
                    status=ContainerResultStatus.FAILED,
                ),
                composition=composition,
                health=health,
                audit_records=_audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.TELEMETRY,
                    (error.diagnostic,),
                    self._correlation,
                ),
            )
        diagnostics = tuple(
            _mapped_runtime_diagnostic(
                ContainerStableDiagnosticCode.UNKNOWN,
                message,
                source_code="container.runtime_envelope.execution",
            )
            for message in execution.diagnostics
        )
        return ContainerRuntimeEnvelopeScopedExecutionResult(
            execution=execution,
            composition=composition,
            health=health,
            telemetry=telemetry,
            audit_records=_audit_records_for_operation(
                ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
                diagnostics,
                self._correlation,
            ),
        )

    async def health(self) -> ContainerRuntimeEnvelopeHealth:
        handle = self._require_ready_handle()
        try:
            health = await _bounded_runtime_operation(
                self._backend.health(handle),
                ContainerRuntimeEnvelopeOperation.HEALTH,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            health = ContainerRuntimeEnvelopeHealth(
                healthy=False,
                status="timeout",
                diagnostics=(error.diagnostic,),
            )
        if not health.ok:
            self._state = ContainerRuntimeEnvelopeState.DEGRADED
            assert self._handle is not None
            self._set_handle(
                self._handle,
                ContainerRuntimeEnvelopeState.DEGRADED,
            )
        return health

    async def telemetry(self) -> Mapping[str, str]:
        try:
            telemetry = await _bounded_runtime_operation(
                self._backend.telemetry(self._require_ready_handle()),
                ContainerRuntimeEnvelopeOperation.TELEMETRY,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            return MappingProxyType(
                _diagnostic_telemetry_mapping(
                    error.diagnostic,
                    self._correlation,
                    ContainerRuntimeEnvelopeOperation.TELEMETRY,
                )
            )
        return MappingProxyType(_string_mapping(telemetry, "telemetry"))

    async def handoff(
        self,
        metadata: ContainerDurablePlanMetadata | None = None,
    ) -> ContainerRuntimeEnvelopeOperationResult:
        handle = self._require_ready_handle()
        resolved_metadata = metadata or self._plan.to_metadata()
        resolved_metadata.assert_matches(self._plan)
        try:
            handoff = await _bounded_runtime_operation(
                self._backend.handoff(handle, resolved_metadata),
                ContainerRuntimeEnvelopeOperation.HANDOFF,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            self._state = ContainerRuntimeEnvelopeState.DEGRADED
            self._set_handle(handle, ContainerRuntimeEnvelopeState.DEGRADED)
            return ContainerRuntimeEnvelopeOperationResult(
                execution=_execution_result(
                    (error.diagnostic,),
                    self._correlation,
                    status=ContainerResultStatus.FAILED,
                ),
                handle=self._handle,
                audit_records=_audit_records_for_operation(
                    ContainerRuntimeEnvelopeOperation.HANDOFF,
                    (error.diagnostic,),
                    self._correlation,
                ),
            )
        return ContainerRuntimeEnvelopeOperationResult(
            execution=ContainerExecutionResult(
                status=ContainerResultStatus.COMPLETED,
                metadata=_execution_metadata(self._correlation),
            ),
            handoff=handoff,
            audit_records=_audit_records_for_operation(
                ContainerRuntimeEnvelopeOperation.HANDOFF,
                (),
                self._correlation,
            ),
        )

    async def shutdown(
        self,
        *,
        timeout_seconds: float = 5,
    ) -> ContainerRuntimeEnvelopeOperationResult:
        assert isinstance(timeout_seconds, int | float)
        assert timeout_seconds > 0, "timeout_seconds must be positive"
        handle = self._require_started_handle()
        try:
            shutdown = await wait_for(
                self._backend.shutdown(handle),
                timeout=timeout_seconds,
            )
            diagnostics = tuple(shutdown.diagnostics)
        except TimeoutError:
            diagnostics = (
                _timeout_diagnostic(
                    "container.runtime_envelope.shutdown_timeout",
                    "runtime envelope graceful shutdown timed out",
                ),
            )
            shutdown = ContainerRuntimeEnvelopeShutdownResult(
                graceful=False,
                diagnostics=diagnostics,
            )
        status = (
            ContainerResultStatus.COMPLETED
            if not diagnostics
            else ContainerResultStatus.FAILED
        )
        self._state = (
            ContainerRuntimeEnvelopeState.STOPPED
            if status is ContainerResultStatus.COMPLETED
            else ContainerRuntimeEnvelopeState.DEGRADED
        )
        self._set_handle(handle, self._state)
        return ContainerRuntimeEnvelopeOperationResult(
            execution=_execution_result(
                diagnostics,
                self._correlation,
                status=status,
            ),
            handle=handle,
            shutdown=shutdown,
            audit_records=_audit_records_for_operation(
                ContainerRuntimeEnvelopeOperation.SHUTDOWN,
                diagnostics,
                self._correlation,
            ),
        )

    async def cleanup(self) -> ContainerRuntimeEnvelopeOperationResult:
        handle = self._require_started_handle()
        try:
            cleanup = await _bounded_runtime_operation(
                self._backend.cleanup(handle),
                ContainerRuntimeEnvelopeOperation.CLEANUP,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            cleanup = ContainerRuntimeEnvelopeCleanupResult(
                cleanup_completed=False,
                cleanup_uncertain=True,
                diagnostics=(error.diagnostic,),
            )
        diagnostics = tuple(cleanup.diagnostics)
        status = (
            ContainerResultStatus.COMPLETED
            if cleanup.ok
            else ContainerResultStatus.FAILED
        )
        self._state = (
            ContainerRuntimeEnvelopeState.CLEANED
            if cleanup.ok
            else ContainerRuntimeEnvelopeState.DEGRADED
        )
        self._set_handle(handle, self._state)
        return ContainerRuntimeEnvelopeOperationResult(
            execution=_execution_result(
                diagnostics,
                self._correlation,
                status=status,
            ),
            handle=handle,
            cleanup=cleanup,
            audit_records=_audit_records_for_operation(
                ContainerRuntimeEnvelopeOperation.CLEANUP,
                diagnostics,
                self._correlation,
            ),
        )

    def _require_started_handle(self) -> ContainerRuntimeEnvelopeHandle:
        assert self._handle is not None, "runtime envelope is not started"
        assert (
            self._state is not ContainerRuntimeEnvelopeState.CLEANED
        ), "runtime envelope is already cleaned"
        return self._handle

    def _require_ready_handle(self) -> ContainerRuntimeEnvelopeHandle:
        handle = self._require_started_handle()
        assert (
            self._state is ContainerRuntimeEnvelopeState.READY
        ), "runtime envelope is not ready"
        return handle

    async def _cleanup_after_failed_start(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeCleanupResult:
        try:
            cleanup = await _bounded_runtime_operation(
                self._backend.cleanup(handle),
                ContainerRuntimeEnvelopeOperation.CLEANUP,
                self._operation_timeout_seconds(),
            )
        except _RuntimeEnvelopeOperationTimeout as error:
            cleanup = ContainerRuntimeEnvelopeCleanupResult(
                cleanup_completed=False,
                cleanup_uncertain=True,
                diagnostics=(error.diagnostic,),
            )
        self._state = (
            ContainerRuntimeEnvelopeState.CLEANED
            if cleanup.ok
            else ContainerRuntimeEnvelopeState.DEGRADED
        )
        self._set_handle(handle, self._state)
        return cleanup

    def _set_handle(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        state: ContainerRuntimeEnvelopeState,
    ) -> None:
        self._state = state
        self._handle = ContainerRuntimeEnvelopeHandle(
            envelope_id=handle.envelope_id,
            envelope_kind=handle.envelope_kind,
            backend=handle.backend,
            plan_fingerprint=handle.plan_fingerprint,
            state=state,
        )

    def _operation_timeout_seconds(self) -> float:
        timeout_seconds: int | float | None = (
            self._plan.run_plan.run_plan.resources.timeout_seconds
        )
        if timeout_seconds is None:
            timeout_seconds = (
                self._plan.envelope_plan.readiness_timeout_seconds
            )
        return float(timeout_seconds)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerFakeRuntimeEnvelopeScript:
    ready: bool = True
    readiness_timeout: bool = False
    readiness_delay_seconds: float = 0
    healthy: bool = True
    shutdown_timeout: bool = False
    cleanup_failure: bool = False
    execution_result: ContainerExecutionResult | None = None
    telemetry: Mapping[str, str] = field(default_factory=dict)
    state: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_bool(self.ready, "ready")
        _assert_bool(self.readiness_timeout, "readiness_timeout")
        assert isinstance(self.readiness_delay_seconds, int | float)
        assert (
            self.readiness_delay_seconds >= 0
        ), "readiness_delay_seconds must not be negative"
        _assert_bool(self.healthy, "healthy")
        _assert_bool(self.shutdown_timeout, "shutdown_timeout")
        _assert_bool(self.cleanup_failure, "cleanup_failure")
        if self.execution_result is not None:
            assert isinstance(self.execution_result, ContainerExecutionResult)
        object.__setattr__(
            self,
            "telemetry",
            MappingProxyType(_string_mapping(self.telemetry, "telemetry")),
        )
        object.__setattr__(
            self,
            "state",
            MappingProxyType(_string_mapping(self.state, "state")),
        )
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(_string_mapping(self.artifacts, "artifacts")),
        )


@final
class ContainerFakeRuntimeEnvelopeBackend(ContainerRuntimeEnvelopeBackend):
    def __init__(self, script: ContainerFakeRuntimeEnvelopeScript) -> None:
        assert isinstance(script, ContainerFakeRuntimeEnvelopeScript)
        self._script = script
        self._operations: list[ContainerRuntimeEnvelopeOperation] = []
        self._started_count = 0
        self._executions = 0

    @property
    def operations(self) -> tuple[ContainerRuntimeEnvelopeOperation, ...]:
        return tuple(self._operations)

    async def start(
        self,
        plan: ContainerNormalizedRuntimeEnvelopePlan,
    ) -> ContainerRuntimeEnvelopeHandle:
        self._record(ContainerRuntimeEnvelopeOperation.START)
        self._started_count += 1
        return ContainerRuntimeEnvelopeHandle(
            envelope_id=f"fake-runtime-{self._started_count}",
            envelope_kind=plan.envelope_kind,
            backend=plan.run_plan.run_plan.backend,
            plan_fingerprint=plan.plan_fingerprint,
            state=ContainerRuntimeEnvelopeState.STARTING,
        )

    async def readiness(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeReadiness:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        self._record(ContainerRuntimeEnvelopeOperation.READINESS)
        if self._script.readiness_timeout:
            raise TimeoutError
        if self._script.readiness_delay_seconds:
            await sleep(self._script.readiness_delay_seconds)
        if not self._script.ready:
            return ContainerRuntimeEnvelopeReadiness(
                ready=False,
                message="runtime envelope is not ready",
                diagnostics=(
                    _mapped_runtime_diagnostic(
                        ContainerStableDiagnosticCode.UNKNOWN,
                        "runtime envelope is not ready",
                        source_code="container.runtime_envelope.not_ready",
                    ),
                ),
            )
        return ContainerRuntimeEnvelopeReadiness(
            ready=True,
            metadata={"envelope_id": handle.envelope_id},
        )

    async def execute(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        plan: ContainerNormalizedRunPlan,
    ) -> ContainerExecutionResult:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        assert isinstance(plan, ContainerNormalizedRunPlan)
        self._record(ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION)
        self._executions += 1
        if self._script.execution_result is not None:
            return self._script.execution_result
        return ContainerExecutionResult(
            status=ContainerResultStatus.COMPLETED,
            exit_code=0,
            metadata={"executions": str(self._executions)},
        )

    async def health(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeHealth:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        self._record(ContainerRuntimeEnvelopeOperation.HEALTH)
        if not self._script.healthy:
            return ContainerRuntimeEnvelopeHealth(
                healthy=False,
                status="unhealthy",
                diagnostics=(
                    _mapped_runtime_diagnostic(
                        ContainerStableDiagnosticCode.UNKNOWN,
                        "runtime envelope health check failed",
                        source_code="container.runtime_envelope.health_failed",
                    ),
                ),
            )
        return ContainerRuntimeEnvelopeHealth(
            healthy=True,
            metadata={"envelope_id": handle.envelope_id},
        )

    async def telemetry(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> Mapping[str, str]:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        self._record(ContainerRuntimeEnvelopeOperation.TELEMETRY)
        return dict(self._script.telemetry) | {
            "executions": str(self._executions)
        }

    async def handoff(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        metadata: ContainerDurablePlanMetadata,
    ) -> ContainerRuntimeEnvelopeHandoff:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        assert isinstance(metadata, ContainerDurablePlanMetadata)
        self._record(ContainerRuntimeEnvelopeOperation.HANDOFF)
        return ContainerRuntimeEnvelopeHandoff(
            metadata=metadata,
            telemetry=dict(self._script.telemetry)
            | {"executions": str(self._executions)},
            state=dict(self._script.state)
            | {"envelope_id": handle.envelope_id},
            artifacts=self._script.artifacts,
        )

    async def shutdown(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeShutdownResult:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        self._record(ContainerRuntimeEnvelopeOperation.SHUTDOWN)
        if self._script.shutdown_timeout:
            raise TimeoutError
        return ContainerRuntimeEnvelopeShutdownResult(
            graceful=True,
            metadata={"envelope_id": handle.envelope_id},
        )

    async def cleanup(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeCleanupResult:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        self._record(ContainerRuntimeEnvelopeOperation.CLEANUP)
        if self._script.cleanup_failure:
            return ContainerRuntimeEnvelopeCleanupResult(
                cleanup_completed=False,
                cleanup_uncertain=True,
                diagnostics=(
                    _mapped_runtime_diagnostic(
                        ContainerStableDiagnosticCode.CLEANUP_FAILED,
                        "runtime envelope cleanup failed",
                        source_code=(
                            "container.runtime_envelope.cleanup_failed"
                        ),
                        retryable=True,
                    ),
                ),
            )
        return ContainerRuntimeEnvelopeCleanupResult(
            cleanup_completed=True,
            metadata={"envelope_id": handle.envelope_id},
        )

    def _record(self, operation: ContainerRuntimeEnvelopeOperation) -> None:
        self._operations.append(operation)


def validate_runtime_envelope_composition(
    envelope_plan: ContainerNormalizedRuntimeEnvelopePlan,
    command_plan: ContainerNormalizedRunPlan,
    *,
    policy: ContainerRuntimeEnvelopeCompositionPolicy | None = None,
) -> ContainerRuntimeEnvelopeCompositionResult:
    resolved_policy = policy or ContainerRuntimeEnvelopeCompositionPolicy()
    return resolved_policy.validate(envelope_plan, command_plan)


class _RuntimeEnvelopeOperationTimeout(Exception):
    def __init__(
        self,
        operation: ContainerRuntimeEnvelopeOperation,
        diagnostic: ContainerMappedDiagnostic,
    ) -> None:
        assert isinstance(operation, ContainerRuntimeEnvelopeOperation)
        assert isinstance(diagnostic, ContainerMappedDiagnostic)
        super().__init__(diagnostic.message)
        self.operation = operation
        self.diagnostic = diagnostic


async def _bounded_runtime_operation(
    awaitable: Awaitable[RuntimeOperationResult],
    operation: ContainerRuntimeEnvelopeOperation,
    timeout_seconds: float,
) -> RuntimeOperationResult:
    assert isinstance(operation, ContainerRuntimeEnvelopeOperation)
    assert isinstance(timeout_seconds, int | float)
    assert timeout_seconds > 0, "timeout_seconds must be positive"
    try:
        return await wait_for(awaitable, timeout=timeout_seconds)
    except TimeoutError as error:
        raise _RuntimeEnvelopeOperationTimeout(
            operation,
            _runtime_operation_timeout_diagnostic(operation),
        ) from error


def _runtime_operation_timeout_diagnostic(
    operation: ContainerRuntimeEnvelopeOperation,
) -> ContainerMappedDiagnostic:
    return _timeout_diagnostic(
        f"container.runtime_envelope.{operation.value}_timeout",
        _runtime_operation_timeout_message(operation),
    )


def _runtime_operation_timeout_message(
    operation: ContainerRuntimeEnvelopeOperation,
) -> str:
    return f"runtime envelope {operation.value} timed out"


def _assert_runtime_envelope_plan(
    plan: ContainerNormalizedRuntimeEnvelopePlan,
) -> None:
    assert (
        plan.envelope_plan.scope is _RUNTIME_SCOPE
    ), "runtime envelope plan must use runtime_envelope scope"
    assert (
        plan.run_plan.request.scope is _RUNTIME_SCOPE
    ), "runtime envelope request must use runtime_envelope scope"
    assert (
        plan.run_plan.run_plan.command.scope is _RUNTIME_SCOPE
    ), "runtime envelope command must use runtime_envelope scope"


def _is_nested_runtime_envelope(plan: ContainerNormalizedRunPlan) -> bool:
    return (
        plan.request.request_kind is ContainerPlanRequestKind.RUNTIME_ENVELOPE
        or plan.request.scope is _RUNTIME_SCOPE
        or plan.run_plan.command.scope is _RUNTIME_SCOPE
    )


def _double_mount_diagnostics(
    envelope_plan: ContainerNormalizedRuntimeEnvelopePlan,
    command_plan: ContainerNormalizedRunPlan,
) -> tuple[ContainerMappedDiagnostic, ...]:
    envelope_mounts = envelope_plan.run_plan.run_plan.mounts
    command_mounts = command_plan.run_plan.mounts
    diagnostics: list[ContainerMappedDiagnostic] = []
    envelope_sources = {
        _host_path(mount.source)
        for mount in envelope_mounts
        if mount.source is not None
    }
    for command_mount in command_mounts:
        command_target = _container_path(command_mount.target)
        command_source = _host_path(command_mount.source)
        for envelope_mount in envelope_mounts:
            envelope_target = _container_path(envelope_mount.target)
            if _paths_overlap(command_target, envelope_target):
                diagnostics.append(
                    _policy_diagnostic(
                        "container.runtime_envelope.double_mount",
                        "nested command container would remount "
                        f"{command_target}",
                    )
                )
                break
        if command_source is not None and command_source in envelope_sources:
            diagnostics.append(
                _policy_diagnostic(
                    "container.runtime_envelope.double_mount",
                    "nested command container would reuse a host mount source",
                )
            )
    return tuple(diagnostics)


def _policy_widening_diagnostics(
    envelope_plan: ContainerNormalizedRuntimeEnvelopePlan,
    command_plan: ContainerNormalizedRunPlan,
) -> tuple[ContainerMappedDiagnostic, ...]:
    envelope = envelope_plan.run_plan
    command = command_plan
    envelope_run = envelope.run_plan
    command_run = command.run_plan
    diagnostics: list[ContainerMappedDiagnostic] = []
    if command.profile_registry_id != envelope.profile_registry_id:
        diagnostics.append(_policy_widening("profile registry changed"))
    if command_run.policy_version != envelope_run.policy_version:
        diagnostics.append(_policy_widening("policy version changed"))
    if command_run.backend is not envelope_run.backend:
        diagnostics.append(_policy_widening("backend changed"))
    if command_run.profile_name != envelope_run.profile_name:
        diagnostics.append(_policy_widening("profile changed"))
    if _network_rank(command_run.network.mode) > _network_rank(
        envelope_run.network.mode
    ):
        diagnostics.append(_policy_widening("network mode widened"))
    if not set(command_run.network.egress_allowlist).issubset(
        set(envelope_run.network.egress_allowlist)
    ):
        diagnostics.append(_policy_widening("network egress widened"))
    if not set(command_run.secret_names).issubset(
        set(envelope_run.secret_names)
    ):
        diagnostics.append(_policy_widening("secrets widened"))
    if not set(command_run.environment_names).issubset(
        set(envelope_run.environment_names)
    ):
        diagnostics.append(_policy_widening("environment widened"))
    if not _device_subset(
        command_run.devices.devices,
        envelope_run.devices.devices,
    ):
        diagnostics.append(_policy_widening("devices widened"))
    if _resources_widened(command_run.resources, envelope_run.resources):
        diagnostics.append(_policy_widening("resources widened"))
    diagnostics.extend(
        _mount_policy_widening_diagnostics(
            envelope_run.mounts,
            command_run.mounts,
        )
    )
    return tuple(diagnostics)


def _mount_policy_widening_diagnostics(
    envelope_mounts: Sequence[ContainerMountDeclaration],
    command_mounts: Sequence[ContainerMountDeclaration],
) -> tuple[ContainerMappedDiagnostic, ...]:
    diagnostics: list[ContainerMappedDiagnostic] = []
    envelope_by_target = {
        _container_path(mount.target): mount for mount in envelope_mounts
    }
    for command_mount in command_mounts:
        command_target = _container_path(command_mount.target)
        envelope_mount = envelope_by_target.get(command_target)
        if envelope_mount is None:
            diagnostics.append(_policy_widening("mounts widened"))
            continue
        command_access = cast(ContainerMountAccess, command_mount.access)
        envelope_access = cast(ContainerMountAccess, envelope_mount.access)
        if (
            command_access is ContainerMountAccess.WRITE
            and envelope_access is ContainerMountAccess.READ
        ):
            diagnostics.append(_policy_widening("mount access widened"))
    return tuple(diagnostics)


def _resources_widened(command: object, envelope: object) -> bool:
    for field_name in (
        "cpu_count",
        "memory_bytes",
        "pids",
        "timeout_seconds",
    ):
        command_value = getattr(command, field_name)
        envelope_value = getattr(envelope, field_name)
        if envelope_value is None:
            continue
        if command_value is None or command_value > envelope_value:
            return True
    return False


def _device_subset(
    command_devices: Sequence[ContainerDeviceClass | str],
    envelope_devices: Sequence[ContainerDeviceClass | str],
) -> bool:
    return set(command_devices).issubset(set(envelope_devices))


def _network_rank(value: ContainerNetworkMode | str) -> int:
    mode = _enum_value(value, ContainerNetworkMode, "network mode")
    ranks = {
        ContainerNetworkMode.NONE: 0,
        ContainerNetworkMode.LOOPBACK: 1,
        ContainerNetworkMode.ALLOWLIST: 2,
        ContainerNetworkMode.FULL: 3,
    }
    return ranks[mode]


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _host_path(value: str | None) -> str | None:
    return None if value is None else normalize_posix_path(value)


def _container_path(value: str) -> str:
    normalized = normalize_posix_path(value)
    assert normalized.startswith("/"), "container paths must be absolute"
    return normalized


def _policy_widening(reason: str) -> ContainerMappedDiagnostic:
    return _policy_diagnostic(
        "container.runtime_envelope.policy_widening",
        f"nested command container policy would widen: {reason}",
    )


def _policy_diagnostic(
    source_code: str,
    message: str,
) -> ContainerMappedDiagnostic:
    return _mapped_runtime_diagnostic(
        ContainerStableDiagnosticCode.POLICY_DENIED,
        message,
        source_code=source_code,
    )


def _timeout_diagnostic(
    source_code: str,
    message: str,
) -> ContainerMappedDiagnostic:
    return _mapped_runtime_diagnostic(
        ContainerStableDiagnosticCode.TIMEOUT,
        message,
        source_code=source_code,
        retryable=True,
    )


def _mapped_runtime_diagnostic(
    code: ContainerStableDiagnosticCode,
    message: str,
    *,
    source_code: str,
    retryable: bool = False,
) -> ContainerMappedDiagnostic:
    return ContainerMappedDiagnostic(
        code=code,
        message=message,
        status=_status_for_code(code),
        source_code=source_code,
        retryable=retryable,
        event_type=_event_type_for_code(code),
    )


def _execution_result(
    diagnostics: Sequence[ContainerMappedDiagnostic],
    correlation: ContainerAuditCorrelation,
    *,
    status: ContainerResultStatus | None = None,
) -> ContainerExecutionResult:
    assert isinstance(correlation, ContainerAuditCorrelation)
    resolved_status = status or _status_from_diagnostics(diagnostics)
    return ContainerExecutionResult(
        status=resolved_status,
        diagnostics=tuple(
            diagnostic.result_message() for diagnostic in diagnostics
        ),
        metadata=_execution_metadata(correlation)
        | {
            "diagnostic_count": str(len(diagnostics)),
            "diagnostic_codes": _diagnostic_codes(diagnostics),
        },
    )


def _execution_metadata(
    correlation: ContainerAuditCorrelation,
) -> dict[str, str]:
    return correlation.to_metadata()


def _diagnostic_telemetry_mapping(
    diagnostic: ContainerMappedDiagnostic,
    correlation: ContainerAuditCorrelation,
    operation: ContainerRuntimeEnvelopeOperation,
) -> dict[str, str]:
    assert isinstance(diagnostic, ContainerMappedDiagnostic)
    assert isinstance(correlation, ContainerAuditCorrelation)
    assert isinstance(operation, ContainerRuntimeEnvelopeOperation)
    code = cast(ContainerStableDiagnosticCode, diagnostic.code)
    status = cast(ContainerResultStatus, diagnostic.status)
    event_type = cast(ContainerAuditEventType, diagnostic.event_type)
    return _execution_metadata(correlation) | {
        "operation": operation.value,
        "status": status.value,
        "diagnostic_count": "1",
        "diagnostic_codes": code.value,
        "diagnostic_event_types": event_type.value,
        "diagnostic_messages": diagnostic.result_message(),
        "diagnostic_retryable": str(diagnostic.retryable).lower(),
        "diagnostic_source_codes": diagnostic.source_code or "unknown",
    }


def _diagnostic_codes(
    diagnostics: Sequence[ContainerMappedDiagnostic],
) -> str:
    if not diagnostics:
        return "none"
    return ",".join(
        cast(ContainerStableDiagnosticCode, diagnostic.code).value
        for diagnostic in diagnostics
    )


def _audit_records_for_operation(
    operation: ContainerRuntimeEnvelopeOperation,
    diagnostics: Sequence[ContainerMappedDiagnostic],
    correlation: ContainerAuditCorrelation,
) -> tuple[ContainerAuditRecord, ...]:
    records = [
        ContainerAuditRecord(
            event_type=_event_type_for_operation(operation),
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            correlation=correlation,
            metadata={"operation": operation.value},
            diagnostics=diagnostics,
        )
    ]
    for diagnostic in diagnostics:
        records.append(
            ContainerAuditRecord(
                event_type=cast(
                    ContainerAuditEventType, diagnostic.event_type
                ),
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                correlation=correlation,
                metadata={
                    "operation": operation.value,
                    "source_code": diagnostic.source_code or "unknown",
                },
                diagnostics=(diagnostic,),
            )
        )
    return tuple(records)


def _event_type_for_operation(
    operation: ContainerRuntimeEnvelopeOperation,
) -> ContainerAuditEventType:
    if operation is ContainerRuntimeEnvelopeOperation.START:
        return ContainerAuditEventType.CONTAINER_START
    if operation is ContainerRuntimeEnvelopeOperation.READINESS:
        return ContainerAuditEventType.CONTAINER_START
    if operation is ContainerRuntimeEnvelopeOperation.HEALTH:
        return ContainerAuditEventType.STATS
    if operation is ContainerRuntimeEnvelopeOperation.TELEMETRY:
        return ContainerAuditEventType.STATS
    if operation is ContainerRuntimeEnvelopeOperation.HANDOFF:
        return ContainerAuditEventType.OUTPUT_COPY
    if operation is ContainerRuntimeEnvelopeOperation.CLEANUP:
        return ContainerAuditEventType.CLEANUP
    if operation is ContainerRuntimeEnvelopeOperation.SHUTDOWN:
        return ContainerAuditEventType.EXIT
    return ContainerAuditEventType.RESULT_RECORDED


def _event_type_for_code(
    code: ContainerStableDiagnosticCode,
) -> ContainerAuditEventType:
    if code is ContainerStableDiagnosticCode.TIMEOUT:
        return ContainerAuditEventType.TIMEOUT
    if code is ContainerStableDiagnosticCode.CANCELLED:
        return ContainerAuditEventType.CANCELLATION
    if code is ContainerStableDiagnosticCode.POLICY_DENIED:
        return ContainerAuditEventType.DENIAL
    return ContainerAuditEventType.FAILURE


def _status_from_diagnostics(
    diagnostics: Sequence[ContainerMappedDiagnostic],
) -> ContainerResultStatus:
    if not diagnostics:
        return ContainerResultStatus.COMPLETED
    statuses = {
        cast(ContainerResultStatus, diagnostic.status)
        for diagnostic in diagnostics
    }
    if ContainerResultStatus.DENIED in statuses:
        return ContainerResultStatus.DENIED
    if ContainerResultStatus.CANCELLED in statuses:
        return ContainerResultStatus.CANCELLED
    return ContainerResultStatus.FAILED


def _status_for_code(
    code: ContainerStableDiagnosticCode,
) -> ContainerResultStatus:
    if code is ContainerStableDiagnosticCode.POLICY_DENIED:
        return ContainerResultStatus.DENIED
    if code is ContainerStableDiagnosticCode.CANCELLED:
        return ContainerResultStatus.CANCELLED
    return ContainerResultStatus.FAILED


def _assert_mapped_diagnostics(
    diagnostics: Sequence[ContainerMappedDiagnostic],
) -> None:
    for diagnostic in diagnostics:
        assert isinstance(diagnostic, ContainerMappedDiagnostic)


def _string_mapping(value: object, field_name: str) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    result: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name} value")
        assert isinstance(key, str)
        assert isinstance(item, str)
        result[key] = item
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
