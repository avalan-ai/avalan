from ..container import (
    ContainerAuditMode,
    ContainerAuditPolicy,
    ContainerCleanupMode,
    ContainerCleanupPolicy,
    ContainerDeviceClass,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerOutputPolicy,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerRuntimeEnvelopeKind,
    ContainerSurface,
    normalize_runtime_envelope_plan,
)
from ..entities import (
    Modality,
    TransformerEngineSettings,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from json import dumps
from types import MappingProxyType
from typing import cast, final


class ModelBackendAcceleratorClass(StrEnum):
    CPU = "cpu"
    NVIDIA_CDI = "nvidia_cdi"
    AMD_CDI = "amd_cdi"
    VULKAN_FORWARDED = "vulkan_forwarded"
    METAL_HOST_NATIVE = "metal_host_native"
    UNSUPPORTED = "unsupported"


class ModelBackendEnvelopeLifecyclePhase(StrEnum):
    STARTUP = "startup"
    WARMUP = "warmup"
    REQUEST_STREAMING = "request_streaming"
    CANCELLATION = "cancellation"
    HEALTH = "health"
    SHUTDOWN = "shutdown"
    CLEANUP = "cleanup"


class ModelBackendEnvelopeLifecycleStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelBackendRuntimeDiagnostic:
    code: str
    path: str
    message: str
    hint: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.code, "code")
        _assert_non_empty_string(self.path, "path")
        _assert_non_empty_string(self.message, "message")
        if self.hint is not None:
            _assert_non_empty_string(self.hint, "hint")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "hint": self.hint,
        }


class ModelBackendRuntimePolicyError(RuntimeError):
    def __init__(
        self,
        diagnostics: Sequence[ModelBackendRuntimeDiagnostic],
    ) -> None:
        assert diagnostics, "policy error requires diagnostics"
        self.diagnostics = tuple(diagnostics)
        super().__init__(self.diagnostics[0].message)

    @property
    def diagnostic(self) -> dict[str, str | None]:
        return self.diagnostics[0].to_dict()

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelBackendEnvelopeLifecycleStep:
    phase: ModelBackendEnvelopeLifecyclePhase | str
    status: ModelBackendEnvelopeLifecycleStatus | str
    diagnostic: ModelBackendRuntimeDiagnostic | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase",
            ModelBackendEnvelopeLifecyclePhase(self.phase),
        )
        object.__setattr__(
            self,
            "status",
            ModelBackendEnvelopeLifecycleStatus(self.status),
        )
        if self.diagnostic is not None:
            assert isinstance(self.diagnostic, ModelBackendRuntimeDiagnostic)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    def to_dict(self) -> dict[str, object]:
        phase = cast(ModelBackendEnvelopeLifecyclePhase, self.phase)
        status = cast(ModelBackendEnvelopeLifecycleStatus, self.status)
        return {
            "phase": phase.value,
            "status": status.value,
            "diagnostic": (
                self.diagnostic.to_dict() if self.diagnostic else None
            ),
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelBackendEnvelopeLifecycleResult:
    plan_fingerprint: str
    startup: ModelBackendEnvelopeLifecycleStep
    warmup: ModelBackendEnvelopeLifecycleStep
    request_streaming: ModelBackendEnvelopeLifecycleStep
    cancellation: ModelBackendEnvelopeLifecycleStep
    health: ModelBackendEnvelopeLifecycleStep
    shutdown: ModelBackendEnvelopeLifecycleStep
    cleanup: ModelBackendEnvelopeLifecycleStep
    stream_chunks: Sequence[str] = field(default_factory=tuple)
    result: str | None = None
    diagnostics: Sequence[ModelBackendRuntimeDiagnostic] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")
        _assert_phase(self.startup, ModelBackendEnvelopeLifecyclePhase.STARTUP)
        _assert_phase(self.warmup, ModelBackendEnvelopeLifecyclePhase.WARMUP)
        _assert_phase(
            self.request_streaming,
            ModelBackendEnvelopeLifecyclePhase.REQUEST_STREAMING,
        )
        _assert_phase(
            self.cancellation,
            ModelBackendEnvelopeLifecyclePhase.CANCELLATION,
        )
        _assert_phase(self.health, ModelBackendEnvelopeLifecyclePhase.HEALTH)
        _assert_phase(
            self.shutdown,
            ModelBackendEnvelopeLifecyclePhase.SHUTDOWN,
        )
        _assert_phase(self.cleanup, ModelBackendEnvelopeLifecyclePhase.CLEANUP)
        object.__setattr__(
            self,
            "stream_chunks",
            _string_tuple(self.stream_chunks, "stream_chunks"),
        )
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, ModelBackendRuntimeDiagnostic)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return all(
            step.status is ModelBackendEnvelopeLifecycleStatus.COMPLETED
            or step.status is ModelBackendEnvelopeLifecycleStatus.SKIPPED
            for step in self.steps
        )

    @property
    def steps(self) -> tuple[ModelBackendEnvelopeLifecycleStep, ...]:
        return (
            self.startup,
            self.warmup,
            self.request_streaming,
            self.cancellation,
            self.health,
            self.shutdown,
            self.cleanup,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_fingerprint": self.plan_fingerprint,
            "steps": [step.to_dict() for step in self.steps],
            "stream_chunks": list(self.stream_chunks),
            "result": self.result,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "ok": self.ok,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelBackendEnvelopeLifecycleScript:
    stream_chunks: Sequence[str] = ("token:0", "token:1")
    result: str = "token:0token:1"
    cancel_after_chunks: int | None = None
    startup_timeout: bool = False
    cleanup_failure: bool = False
    health_failure: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream_chunks",
            _string_tuple(self.stream_chunks, "stream_chunks"),
        )
        _assert_non_empty_string(self.result, "result")
        if self.cancel_after_chunks is not None:
            assert (
                self.cancel_after_chunks >= 0
            ), "cancel_after_chunks must be non-negative"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelBackendRuntimePolicy:
    output: ContainerOutputPolicy
    cleanup: ContainerCleanupPolicy
    audit: ContainerAuditPolicy

    def __post_init__(self) -> None:
        assert isinstance(self.output, ContainerOutputPolicy)
        assert isinstance(self.cleanup, ContainerCleanupPolicy)
        assert isinstance(self.audit, ContainerAuditPolicy)

    @property
    def policy_fingerprint(self) -> str:
        serialized = dumps(
            self.canonical_policy_input(),
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def canonical_policy_input(self) -> dict[str, object]:
        return {
            "audit": self.audit.to_dict(),
            "cleanup": self.cleanup.to_dict(),
            "output": self.output.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.canonical_policy_input(),
            "policy_fingerprint": self.policy_fingerprint,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelBackendRuntimeProfileSelection:
    effective_settings: ContainerEffectiveSettings
    plan: ContainerNormalizedRuntimeEnvelopePlan
    runtime_policy: ModelBackendRuntimePolicy
    accelerator: ModelBackendAcceleratorClass | str

    def __post_init__(self) -> None:
        assert isinstance(self.effective_settings, ContainerEffectiveSettings)
        assert isinstance(self.plan, ContainerNormalizedRuntimeEnvelopePlan)
        assert isinstance(self.runtime_policy, ModelBackendRuntimePolicy)
        object.__setattr__(
            self,
            "accelerator",
            ModelBackendAcceleratorClass(self.accelerator),
        )

    @property
    def plan_fingerprint(self) -> str:
        return self.plan.plan_fingerprint

    @property
    def selection_fingerprint(self) -> str:
        serialized = dumps(
            {
                "accelerator": (
                    cast(
                        ModelBackendAcceleratorClass,
                        self.accelerator,
                    ).value
                ),
                "plan_fingerprint": self.plan.plan_fingerprint,
                "runtime_policy_fingerprint": (
                    self.runtime_policy.policy_fingerprint
                ),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        accelerator = cast(ModelBackendAcceleratorClass, self.accelerator)
        return {
            "accelerator": accelerator.value,
            "effective_settings": self.effective_settings.to_dict(),
            "plan": self.plan.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "runtime_policy": self.runtime_policy.to_dict(),
            "selection_fingerprint": self.selection_fingerprint,
        }


def trusted_model_backend_profile_selection(
    effective_settings: ContainerEffectiveSettings,
    *,
    engine_settings: TransformerEngineSettings,
    modality: Modality,
    model_id: str,
) -> ModelBackendRuntimeProfileSelection:
    assert isinstance(effective_settings, ContainerEffectiveSettings)
    assert isinstance(engine_settings, TransformerEngineSettings)
    assert isinstance(modality, Modality)
    _assert_non_empty_string(model_id, "model_id")

    accelerator = model_backend_accelerator_class(engine_settings)
    diagnostics = _effective_settings_diagnostics(
        effective_settings,
        engine_settings,
        accelerator,
    )
    if diagnostics:
        raise ModelBackendRuntimePolicyError(diagnostics)

    assert effective_settings.profile is not None
    runtime_policy = ModelBackendRuntimePolicy(
        output=effective_settings.profile.output,
        cleanup=effective_settings.profile.cleanup,
        audit=effective_settings.profile.audit,
    )
    resources = effective_settings.profile.resources
    assert resources.timeout_seconds is not None
    plan = normalize_runtime_envelope_plan(
        effective_settings,
        ContainerPlanRequest(
            request_kind=ContainerPlanRequestKind.RUNTIME_ENVELOPE,
            logical_name=model_id,
            command="avalan-model",
            argv=("avalan", "model", "run", modality.value, model_id),
            cwd="/workspace",
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            request_id=model_id,
        ),
        envelope_kind=ContainerRuntimeEnvelopeKind.MODEL_BACKEND,
        readiness_timeout_seconds=min(resources.timeout_seconds, 30),
    )
    diagnostics = _plan_preservation_diagnostics(effective_settings, plan)
    diagnostics += _runtime_policy_preservation_diagnostics(
        effective_settings,
        runtime_policy,
    )
    if diagnostics:
        raise ModelBackendRuntimePolicyError(diagnostics)
    return ModelBackendRuntimeProfileSelection(
        effective_settings=effective_settings,
        plan=plan,
        runtime_policy=runtime_policy,
        accelerator=accelerator,
    )


def model_backend_accelerator_class(
    engine_settings: TransformerEngineSettings,
) -> ModelBackendAcceleratorClass:
    assert isinstance(engine_settings, TransformerEngineSettings)
    native_backend = _native_backend(engine_settings)
    device = (engine_settings.device or "").lower()
    requested = (
        device or "cpu"
        if native_backend == "auto"
        else native_backend or device
    )
    if requested in {"", "auto", "cpu"}:
        return ModelBackendAcceleratorClass.CPU
    if requested in {"metal", "mps"} or requested.startswith("mps"):
        return ModelBackendAcceleratorClass.METAL_HOST_NATIVE
    if requested in {"cuda", "nvidia", "nvidia_cdi"} or requested.startswith(
        "cuda"
    ):
        return ModelBackendAcceleratorClass.NVIDIA_CDI
    if requested in {"amd", "amd_cdi", "hip", "rocm"} or requested.startswith(
        (
            "hip",
            "rocm",
        )
    ):
        return ModelBackendAcceleratorClass.AMD_CDI
    if requested in {"vulkan", "vulkan_forwarded"} or requested.startswith(
        "vulkan"
    ):
        return ModelBackendAcceleratorClass.VULKAN_FORWARDED
    return ModelBackendAcceleratorClass.UNSUPPORTED


def simulate_model_backend_envelope_lifecycle(
    plan: ContainerNormalizedRuntimeEnvelopePlan,
    script: ModelBackendEnvelopeLifecycleScript | None = None,
) -> ModelBackendEnvelopeLifecycleResult:
    assert isinstance(plan, ContainerNormalizedRuntimeEnvelopePlan)
    script = script or ModelBackendEnvelopeLifecycleScript()
    diagnostics: list[ModelBackendRuntimeDiagnostic] = []
    if script.startup_timeout:
        diagnostic = _diagnostic(
            "model.runtime.startup_timeout",
            "model.container.lifecycle.startup",
            "Model backend runtime envelope startup timed out.",
            "Increase the trusted profile timeout or use host execution.",
        )
        diagnostics.append(diagnostic)
        return _lifecycle_result(
            plan,
            startup=_step(
                ModelBackendEnvelopeLifecyclePhase.STARTUP,
                ModelBackendEnvelopeLifecycleStatus.FAILED,
                diagnostic=diagnostic,
            ),
            warmup=_skipped(ModelBackendEnvelopeLifecyclePhase.WARMUP),
            request_streaming=_skipped(
                ModelBackendEnvelopeLifecyclePhase.REQUEST_STREAMING,
            ),
            cancellation=_skipped(
                ModelBackendEnvelopeLifecyclePhase.CANCELLATION,
            ),
            health=_skipped(ModelBackendEnvelopeLifecyclePhase.HEALTH),
            shutdown=_skipped(ModelBackendEnvelopeLifecyclePhase.SHUTDOWN),
            cleanup=_step(
                ModelBackendEnvelopeLifecyclePhase.CLEANUP,
                ModelBackendEnvelopeLifecycleStatus.COMPLETED,
            ),
            diagnostics=diagnostics,
        )

    stream_chunks = tuple(script.stream_chunks)
    result: str | None = script.result
    request_status = ModelBackendEnvelopeLifecycleStatus.COMPLETED
    cancellation = _skipped(ModelBackendEnvelopeLifecyclePhase.CANCELLATION)
    if script.cancel_after_chunks is not None:
        stream_chunks = stream_chunks[: script.cancel_after_chunks]
        result = None
        request_status = ModelBackendEnvelopeLifecycleStatus.CANCELLED
        diagnostic = _diagnostic(
            "model.runtime.request_cancelled",
            "model.container.lifecycle.request_streaming",
            "Model backend runtime envelope request was cancelled.",
            "Cancel propagated to the envelope before completing the stream.",
        )
        diagnostics.append(diagnostic)
        cancellation = _step(
            ModelBackendEnvelopeLifecyclePhase.CANCELLATION,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
            diagnostic=diagnostic,
        )

    health = _step(
        ModelBackendEnvelopeLifecyclePhase.HEALTH,
        ModelBackendEnvelopeLifecycleStatus.COMPLETED,
    )
    if script.health_failure:
        diagnostic = _diagnostic(
            "model.runtime.health_failed",
            "model.container.lifecycle.health",
            "Model backend runtime envelope health check failed.",
            "Restart the envelope or use a different trusted profile.",
        )
        diagnostics.append(diagnostic)
        health = _step(
            ModelBackendEnvelopeLifecyclePhase.HEALTH,
            ModelBackendEnvelopeLifecycleStatus.FAILED,
            diagnostic=diagnostic,
        )

    cleanup = _step(
        ModelBackendEnvelopeLifecyclePhase.CLEANUP,
        ModelBackendEnvelopeLifecycleStatus.COMPLETED,
    )
    if script.cleanup_failure:
        diagnostic = _diagnostic(
            "model.runtime.cleanup_failed",
            "model.container.lifecycle.cleanup",
            "Model backend runtime envelope cleanup failed.",
            "Treat the envelope as untrusted until the runtime confirms "
            "cleanup.",
        )
        diagnostics.append(diagnostic)
        cleanup = _step(
            ModelBackendEnvelopeLifecyclePhase.CLEANUP,
            ModelBackendEnvelopeLifecycleStatus.FAILED,
            diagnostic=diagnostic,
        )

    return _lifecycle_result(
        plan,
        startup=_step(
            ModelBackendEnvelopeLifecyclePhase.STARTUP,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
        ),
        warmup=_step(
            ModelBackendEnvelopeLifecyclePhase.WARMUP,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
        ),
        request_streaming=_step(
            ModelBackendEnvelopeLifecyclePhase.REQUEST_STREAMING,
            request_status,
            metadata={"chunks": str(len(stream_chunks))},
        ),
        cancellation=cancellation,
        health=health,
        shutdown=_step(
            ModelBackendEnvelopeLifecyclePhase.SHUTDOWN,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
        ),
        cleanup=cleanup,
        stream_chunks=stream_chunks,
        result=result,
        diagnostics=diagnostics,
    )


def _effective_settings_diagnostics(
    effective_settings: ContainerEffectiveSettings,
    engine_settings: TransformerEngineSettings,
    accelerator: ModelBackendAcceleratorClass,
) -> tuple[ModelBackendRuntimeDiagnostic, ...]:
    diagnostics: list[ModelBackendRuntimeDiagnostic] = []
    source = effective_settings.source
    profile = effective_settings.profile
    profile_name = effective_settings.profile_name

    if not effective_settings.enabled:
        diagnostics.append(
            _diagnostic(
                "model.runtime.disabled",
                "model.container.backend",
                "Model backend runtime envelope settings are disabled.",
                "Enable a trusted model backend runtime profile.",
            )
        )
    if source.surface is not ContainerSurface.MODEL_BACKEND:
        diagnostics.append(
            _diagnostic(
                "model.runtime.surface_invalid",
                "model.container.source.surface",
                "Model backend runtime envelope source surface is invalid.",
                "Use a trusted model_backend container settings source.",
            )
        )
    if not source.can_define_runtime_authority:
        diagnostics.append(
            _diagnostic(
                "model.runtime.profile_untrusted",
                "model.container.source.trust_level",
                "Model backend runtime profile is not trusted.",
                "Only trusted operator or deployment settings may define it.",
            )
        )
    if (
        effective_settings.scope
        is not ContainerExecutionScope.RUNTIME_ENVELOPE
    ):
        diagnostics.append(
            _diagnostic(
                "model.runtime.scope_invalid",
                "model.container.scope",
                "Model backend runtime envelope scope is invalid.",
                "Use runtime_envelope scope for model backend containers.",
            )
        )
    if profile is None or profile_name is None:
        diagnostics.append(
            _diagnostic(
                "model.runtime.profile_missing",
                "model.container.profile",
                "Model backend runtime envelope has no selected profile.",
                "Select a trusted model backend profile.",
            )
        )
        return tuple(diagnostics)
    if profile_name not in effective_settings.allowed_profiles:
        diagnostics.append(
            _diagnostic(
                "model.runtime.profile_mismatch",
                "model.container.profile_name",
                "Model backend runtime profile is not in the allowlist.",
                "Select one of the trusted model backend profiles.",
            )
        )
    if (
        profile.image.digest is None
        or "@sha256:" not in profile.image.reference
    ):
        diagnostics.append(
            _diagnostic(
                "model.runtime.image_untrusted",
                "model.container.profile.image.reference",
                "Model backend runtime image is not digest pinned.",
                "Use a trusted immutable image reference.",
            )
        )

    network_mode = cast(ContainerNetworkMode, profile.network.mode)
    if network_mode is not ContainerNetworkMode.NONE:
        diagnostics.append(
            _diagnostic(
                "model.runtime.network_denied",
                "model.container.profile.network",
                "Model backend runtime profiles cannot enable network.",
                "Use a no-network profile for local model backend envelopes.",
            )
        )
    if profile.secrets or engine_settings.access_token:
        diagnostics.append(
            _diagnostic(
                "model.runtime.secret_leakage_risk",
                "model.container.profile.secrets",
                "Model backend runtime profile can leak secrets.",
                "Do not pass secrets to container-managed model backends.",
            )
        )

    diagnostics.extend(_cache_diagnostics(profile.mounts))
    diagnostics.extend(_resource_diagnostics(profile.resources.to_dict()))
    diagnostics.extend(
        _device_diagnostics(
            cast(Sequence[ContainerDeviceClass], profile.devices.devices),
            accelerator,
        )
    )
    if profile.output.allow_artifacts or profile.output.max_artifact_bytes:
        diagnostics.append(
            _diagnostic(
                "model.runtime.output_policy_unsafe",
                "model.container.profile.output",
                "Model backend runtime output policy permits artifacts.",
                "Use stream-only output for model backend envelopes.",
            )
        )
    cleanup_mode = cast(ContainerCleanupMode, profile.cleanup.mode)
    if cleanup_mode is not ContainerCleanupMode.REMOVE:
        diagnostics.append(
            _diagnostic(
                "model.runtime.cleanup_policy_unsafe",
                "model.container.profile.cleanup",
                "Model backend runtime cleanup policy is unsafe.",
                "Remove model backend runtime envelopes after use.",
            )
        )
    assert isinstance(profile.audit, ContainerAuditPolicy)
    audit_mode = cast(ContainerAuditMode, profile.audit.mode)
    if audit_mode is not ContainerAuditMode.FULL:
        diagnostics.append(
            _diagnostic(
                "model.runtime.audit_insufficient",
                "model.container.profile.audit",
                "Model backend runtime audit policy is insufficient.",
                "Use full audit for container-managed model backends.",
            )
        )
    return tuple(diagnostics)


def _plan_preservation_diagnostics(
    effective_settings: ContainerEffectiveSettings,
    plan: ContainerNormalizedRuntimeEnvelopePlan,
) -> tuple[ModelBackendRuntimeDiagnostic, ...]:
    assert effective_settings.profile is not None
    profile = effective_settings.profile
    run_plan = plan.run_plan.run_plan
    diagnostics: list[ModelBackendRuntimeDiagnostic] = []
    if plan.envelope_kind is not ContainerRuntimeEnvelopeKind.MODEL_BACKEND:
        diagnostics.append(
            _policy_mismatch("envelope_kind", "Envelope kind changed.")
        )
    if (
        plan.envelope_plan.scope
        is not ContainerExecutionScope.RUNTIME_ENVELOPE
    ):
        diagnostics.append(
            _policy_mismatch("scope", "Envelope scope changed.")
        )
    if run_plan.profile_name != profile.name:
        diagnostics.append(
            _policy_mismatch("profile_name", "Selected profile changed.")
        )
    if run_plan.image.to_dict() != profile.image.to_dict():
        diagnostics.append(_policy_mismatch("image", "Image policy changed."))
    if _mount_dicts(run_plan.mounts) != _mount_dicts(profile.mounts):
        diagnostics.append(_policy_mismatch("mounts", "Mount policy changed."))
    if run_plan.secret_names != tuple(
        secret.name for secret in profile.secrets
    ):
        diagnostics.append(
            _policy_mismatch("secrets", "Secret policy changed.")
        )
    if run_plan.network.to_dict() != profile.network.to_dict():
        diagnostics.append(
            _policy_mismatch("network", "Network policy changed.")
        )
    if run_plan.devices.to_dict() != profile.devices.to_dict():
        diagnostics.append(
            _policy_mismatch("devices", "Device policy changed.")
        )
    if run_plan.resources.to_dict() != profile.resources.to_dict():
        diagnostics.append(
            _policy_mismatch("resources", "Resource policy changed.")
        )
    return tuple(diagnostics)


def _runtime_policy_preservation_diagnostics(
    effective_settings: ContainerEffectiveSettings,
    runtime_policy: ModelBackendRuntimePolicy,
) -> tuple[ModelBackendRuntimeDiagnostic, ...]:
    assert effective_settings.profile is not None
    profile = effective_settings.profile
    diagnostics: list[ModelBackendRuntimeDiagnostic] = []
    if runtime_policy.output.to_dict() != profile.output.to_dict():
        diagnostics.append(
            _policy_mismatch("output", "Output policy changed.")
        )
    if runtime_policy.cleanup.to_dict() != profile.cleanup.to_dict():
        diagnostics.append(
            _policy_mismatch("cleanup", "Cleanup policy changed.")
        )
    if runtime_policy.audit.to_dict() != profile.audit.to_dict():
        diagnostics.append(_policy_mismatch("audit", "Audit policy changed."))
    return tuple(diagnostics)


def _native_backend(
    engine_settings: TransformerEngineSettings,
) -> str | None:
    backend_config = engine_settings.backend_config or {}
    native_backend = backend_config.get("native_backend")
    if native_backend is None:
        return None
    if isinstance(native_backend, str):
        return native_backend.lower()
    return "unsupported"


def _cache_diagnostics(
    mounts: Sequence[ContainerMountDeclaration],
) -> tuple[ModelBackendRuntimeDiagnostic, ...]:
    diagnostics: list[ModelBackendRuntimeDiagnostic] = []
    for mount in mounts:
        mount_type = cast(ContainerMountType, mount.mount_type)
        if mount_type is not ContainerMountType.CACHE:
            continue
        if mount.source is not None or not (
            mount.target == "/cache" or mount.target.startswith("/cache/")
        ):
            diagnostics.append(
                _diagnostic(
                    "model.runtime.cache_misuse",
                    "model.container.profile.mounts",
                    "Model backend cache mount is not runtime-managed.",
                    "Use a source-less cache mount under /cache.",
                )
            )
    return tuple(diagnostics)


def _resource_diagnostics(
    resources: Mapping[str, int | None],
) -> tuple[ModelBackendRuntimeDiagnostic, ...]:
    missing = tuple(
        name
        for name in ("cpu_count", "memory_bytes", "timeout_seconds")
        if resources.get(name) is None
    )
    if not missing:
        return ()
    return (
        _diagnostic(
            "model.runtime.resource_mismatch",
            "model.container.profile.resources",
            "Model backend runtime profile has incomplete resource limits.",
            "Set cpu_count, memory_bytes, and timeout_seconds.",
        ),
    )


def _device_diagnostics(
    devices: Sequence[ContainerDeviceClass],
    accelerator: ModelBackendAcceleratorClass,
) -> tuple[ModelBackendRuntimeDiagnostic, ...]:
    if accelerator is ModelBackendAcceleratorClass.METAL_HOST_NATIVE:
        return (
            _diagnostic(
                "model.runtime.metal_container_unsupported",
                "model.container.profile.devices",
                "Host-native Metal is not a container accelerator.",
                "Use a host-native non-container model backend for Metal.",
            ),
        )
    if accelerator is ModelBackendAcceleratorClass.UNSUPPORTED:
        return (
            _diagnostic(
                "model.runtime.device_unsupported",
                "model.container.profile.devices",
                "Model backend requested an unsupported accelerator.",
                "Use cpu, cuda, amd/rocm, or vulkan-forwarded devices.",
            ),
        )
    if accelerator is ModelBackendAcceleratorClass.CPU:
        return ()
    required = ContainerDeviceClass(accelerator.value)
    if required in set(devices):
        return ()
    return (
        _diagnostic(
            "model.runtime.device_denied",
            "model.container.profile.devices",
            "Model backend accelerator is not allowed by the profile.",
            "Select a trusted profile with the requested device class.",
        ),
    )


def _lifecycle_result(
    plan: ContainerNormalizedRuntimeEnvelopePlan,
    *,
    startup: ModelBackendEnvelopeLifecycleStep,
    warmup: ModelBackendEnvelopeLifecycleStep,
    request_streaming: ModelBackendEnvelopeLifecycleStep,
    cancellation: ModelBackendEnvelopeLifecycleStep,
    health: ModelBackendEnvelopeLifecycleStep,
    shutdown: ModelBackendEnvelopeLifecycleStep,
    cleanup: ModelBackendEnvelopeLifecycleStep,
    stream_chunks: Sequence[str] = (),
    result: str | None = None,
    diagnostics: Sequence[ModelBackendRuntimeDiagnostic] = (),
) -> ModelBackendEnvelopeLifecycleResult:
    return ModelBackendEnvelopeLifecycleResult(
        plan_fingerprint=plan.plan_fingerprint,
        startup=startup,
        warmup=warmup,
        request_streaming=request_streaming,
        cancellation=cancellation,
        health=health,
        shutdown=shutdown,
        cleanup=cleanup,
        stream_chunks=stream_chunks,
        result=result,
        diagnostics=diagnostics,
    )


def _skipped(
    phase: ModelBackendEnvelopeLifecyclePhase,
) -> ModelBackendEnvelopeLifecycleStep:
    return _step(phase, ModelBackendEnvelopeLifecycleStatus.SKIPPED)


def _step(
    phase: ModelBackendEnvelopeLifecyclePhase,
    status: ModelBackendEnvelopeLifecycleStatus,
    *,
    diagnostic: ModelBackendRuntimeDiagnostic | None = None,
    metadata: Mapping[str, str] | None = None,
) -> ModelBackendEnvelopeLifecycleStep:
    return ModelBackendEnvelopeLifecycleStep(
        phase=phase,
        status=status,
        diagnostic=diagnostic,
        metadata=metadata or {},
    )


def _assert_phase(
    step: ModelBackendEnvelopeLifecycleStep,
    phase: ModelBackendEnvelopeLifecyclePhase,
) -> None:
    assert isinstance(step, ModelBackendEnvelopeLifecycleStep)
    assert step.phase is phase, f"{phase.value} lifecycle step is required"


def _diagnostic(
    code: str,
    path: str,
    message: str,
    hint: str,
) -> ModelBackendRuntimeDiagnostic:
    return ModelBackendRuntimeDiagnostic(
        code=code,
        path=path,
        message=message,
        hint=hint,
    )


def _policy_mismatch(
    field_name: str,
    message: str,
) -> ModelBackendRuntimeDiagnostic:
    return _diagnostic(
        "model.runtime.policy_mismatch",
        f"model.container.plan.{field_name}",
        f"Model backend runtime envelope policy mismatch: {message}",
        "Recompute the envelope plan from selected effective settings.",
    )


def _mount_dicts(
    mounts: Sequence[ContainerMountDeclaration],
) -> tuple[dict[str, object], ...]:
    return tuple(mount.to_dict() for mount in mounts)


def _string_tuple(
    value: Sequence[str],
    field_name: str,
) -> tuple[str, ...]:
    result = tuple(value)
    for item in result:
        _assert_non_empty_string(item, field_name)
    return result


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    result = dict(value)
    for key, item in result.items():
        _assert_non_empty_string(key, field_name)
        _assert_non_empty_string(item, field_name)
    return result
