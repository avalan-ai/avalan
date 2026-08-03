"""Harden conversation policy, operations, and safe diagnostics."""

from ..types import JsonValue
from .binding import CapabilityEvidenceState, ConversationCapabilityProfile
from .contract import (
    CONFIGURATION_PRECEDENCE,
    AuthorityScope,
    ConfigurationSource,
    ConversationOperation,
    LocalResponseStorage,
    ProviderLaneStorage,
    ResponseOperation,
    RetentionLimits,
)
from .crypto import ConversationDataKey, ConversationKeyStatus
from .envelope import (
    ContinuationEnvelopeKey,
    ContinuationEnvelopeKeyStatus,
)
from .errors import (
    ConversationAuthorizationError,
    ConversationError,
    ConversationErrorCode,
    ConversationKeyCompromisedError,
    ConversationKeyMissingError,
    ConversationKeyPolicyError,
    ConversationKeyRetiredError,
    ConversationLimitError,
    ConversationMigrationRequiredError,
    ConversationValidationError,
)
from .lifecycle import ProviderLifecycleReconciler
from .observability import authority_digest
from .protocols import (
    ConversationOutboxRecoveryWorker,
    ConversationPublisher,
)
from .runtime import CoordinatorAwaitBoundary
from .settings import CompactionOperation, ConversationMode, ReasoningContext
from .store import InMemoryConversationStore
from .stores.pgsql import PgsqlConversationStore
from .value import (
    AuthorityDigest,
    IntegrityDigest,
    SafeAlias,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from asyncio import (
    CancelledError,
    Event,
    Future,
    Lock,
    Task,
    create_task,
    current_task,
    ensure_future,
    get_running_loop,
    wait,
)
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from hmac import digest as hmac_digest
from types import MappingProxyType
from typing import Protocol, TypeVar, cast, final


class ConversationCheckpointBackend(StrEnum):
    """Identify one closed local continuation backend."""

    PROCESS = "process"
    POSTGRESQL = "postgresql"
    CALLER_HELD = "caller_held"


class ConversationOperationalKeyStatus(StrEnum):
    """Identify the complete operational key lifecycle."""

    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"
    COMPROMISED = "compromised"


class ConversationKeyPurpose(StrEnum):
    """Identify a separately activated conversation key purpose."""

    CHECKPOINT = "checkpoint"
    ENVELOPE = "envelope"


class ConversationEventKind(StrEnum):
    """Identify content-safe conversation lifecycle observations."""

    CREATE = "create"
    LOAD = "load"
    COMMIT = "commit"
    BRANCH = "branch"
    CAS_CONFLICT = "cas_conflict"
    MODE = "mode"
    REASONING_CONTEXT = "reasoning_context"
    COMPACTION = "compaction"
    RESTART = "restart"
    EXPIRY = "expiry"
    DELETE = "delete"
    CAPABILITY_REJECTION = "capability_rejection"
    FAILURE_BOUNDARY = "failure_boundary"


class ConversationWorkerState(StrEnum):
    """Identify one background worker lifecycle state."""

    STOPPED = "stopped"
    RUNNING = "running"
    DRAINING = "draining"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class ConversationMaintenanceKind(StrEnum):
    """Identify one bounded maintenance work category."""

    RETENTION = "retention"
    OUTBOX = "outbox"
    RECONCILIATION = "reconciliation"
    PAYLOAD_GC = "payload_gc"
    KEY_ROTATION = "key_rotation"


class ConversationStateSurface(StrEnum):
    """Identify independently versioned durable state surfaces."""

    CHECKPOINT = "checkpoint"
    LANE = "lane"
    PUBLIC_MAPPING = "public_mapping"
    ENVELOPE = "envelope"
    CAPABILITY_PROFILE = "capability_profile"
    EXECUTION_DEFINITION = "execution_definition"
    STRUCTURED_INPUT = "structured_input"


class ConversationRollbackDisposition(StrEnum):
    """Identify the only rollback outcomes for committed state."""

    RESOLVABLE = "resolvable"
    DELETE_ONLY = "delete_only"
    DETERMINISTICALLY_UNAVAILABLE = "deterministically_unavailable"


class ConversationReadinessFailure(StrEnum):
    """Identify content-free operational readiness failures."""

    BACKEND_MIGRATION = "backend_migration"
    ACTIVE_KEYS = "active_keys"
    OUTBOX_LAG = "outbox_lag"
    SWEEPER = "sweeper"
    CAPABILITY_RESOLVER = "capability_resolver"
    ACTIVATION_MANIFEST = "activation_manifest"


class ConversationDeduplicationDisposition(StrEnum):
    """Identify why immutable payload deduplication is enabled or disabled."""

    DISABLED = "disabled"
    AUTHENTICATED_TENANT_SCOPED = "authenticated_tenant_scoped"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCompactionPolicy:
    """Bound enabled compaction operations and inline thresholds."""

    allowed_operations: frozenset[CompactionOperation]
    minimum_inline_threshold: int = 1
    maximum_inline_threshold: int = 2_147_483_647

    def __post_init__(self) -> None:
        if (
            type(self.allowed_operations) is not frozenset
            or not self.allowed_operations
            or any(
                not isinstance(value, CompactionOperation)
                for value in self.allowed_operations
            )
            or CompactionOperation.NONE not in self.allowed_operations
        ):
            raise ConversationValidationError()
        if (
            type(self.minimum_inline_threshold) is not int
            or self.minimum_inline_threshold <= 0
            or type(self.maximum_inline_threshold) is not int
            or self.maximum_inline_threshold < self.minimum_inline_threshold
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationTelemetryPolicy:
    """Configure safe bounded events, metrics, traces, and correlations."""

    enabled: bool = True
    events: bool = True
    metrics: bool = True
    traces: bool = True
    correlation_digests: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.enabled,
            self.events,
            self.metrics,
            self.traces,
            self.correlation_digests,
        ):
            if type(value) is not bool:
                raise ConversationValidationError()
        if not self.enabled and any(
            (
                self.events,
                self.metrics,
                self.traces,
                self.correlation_digests,
            )
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationResourcePolicy:
    """Bound allocation, queues, concurrency, and asynchronous effects."""

    max_items: int = 10_000
    max_checkpoint_bytes: int = 8_388_608
    max_conversation_bytes: int = 67_108_864
    max_depth: int = 32
    max_branches: int = 32
    max_envelope_chars: int = 6_000_000
    max_stream_items: int = 10_000
    max_compact_items: int = 10_000
    max_global_concurrency: int = 128
    max_authority_concurrency: int = 16
    max_conversation_concurrency: int = 2
    max_queue_size: int = 512
    queue_timeout_seconds: float = 30.0
    provider_timeout_seconds: float = 120.0
    store_timeout_seconds: float = 30.0
    key_timeout_seconds: float = 10.0
    cancellation_settlement_seconds: float = 1.0
    readiness_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        for value in (
            self.max_items,
            self.max_checkpoint_bytes,
            self.max_conversation_bytes,
            self.max_depth,
            self.max_branches,
            self.max_envelope_chars,
            self.max_stream_items,
            self.max_compact_items,
            self.max_global_concurrency,
            self.max_authority_concurrency,
            self.max_conversation_concurrency,
            self.max_queue_size,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()
        for timeout_value in (
            self.queue_timeout_seconds,
            self.provider_timeout_seconds,
            self.store_timeout_seconds,
            self.key_timeout_seconds,
            self.cancellation_settlement_seconds,
            self.readiness_timeout_seconds,
        ):
            if (
                not isinstance(timeout_value, int | float)
                or isinstance(timeout_value, bool)
                or timeout_value <= 0
            ):
                raise ConversationValidationError()
        if (
            self.max_checkpoint_bytes > self.max_conversation_bytes
            or self.max_conversation_concurrency
            > self.max_authority_concurrency
            or self.max_authority_concurrency > self.max_global_concurrency
        ):
            raise ConversationValidationError()

    def is_narrower_than(self, policy: "ConversationResourcePolicy") -> bool:
        """Return whether every configured bound is no broader."""
        if type(policy) is not ConversationResourcePolicy:
            raise ConversationValidationError()
        return all(
            mine <= configured
            for mine, configured in zip(
                self._ordered_values(),
                policy._ordered_values(),
                strict=True,
            )
        )

    def _ordered_values(self) -> tuple[float, ...]:
        return (
            self.max_items,
            self.max_checkpoint_bytes,
            self.max_conversation_bytes,
            self.max_depth,
            self.max_branches,
            self.max_envelope_chars,
            self.max_stream_items,
            self.max_compact_items,
            self.max_global_concurrency,
            self.max_authority_concurrency,
            self.max_conversation_concurrency,
            self.max_queue_size,
            self.queue_timeout_seconds,
            self.provider_timeout_seconds,
            self.store_timeout_seconds,
            self.key_timeout_seconds,
            self.cancellation_settlement_seconds,
            self.readiness_timeout_seconds,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationKeyRotationPolicy:
    """Bound rotation windows and incident-only deletion access."""

    max_retiring_keys: int = 2
    minimum_rotation_seconds: int = 60
    maximum_retiring_seconds: int = 86_400
    compromised_deletion_access: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.max_retiring_keys) is not int
            or self.max_retiring_keys <= 0
            or type(self.minimum_rotation_seconds) is not int
            or self.minimum_rotation_seconds <= 0
            or type(self.maximum_retiring_seconds) is not int
            or self.maximum_retiring_seconds < self.minimum_rotation_seconds
            or type(self.compromised_deletion_access) is not bool
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationHardeningPolicy:
    """Define one complete provider-neutral conversation policy ceiling."""

    default_mode: ConversationMode
    allowed_modes: frozenset[ConversationMode]
    allowed_reasoning_contexts: frozenset[ReasoningContext]
    compaction: ConversationCompactionPolicy
    backend: ConversationCheckpointBackend
    retention: RetentionLimits
    resources: ConversationResourcePolicy
    checkpoint_keys: ConversationKeyRotationPolicy
    envelope_keys: ConversationKeyRotationPolicy
    capability_profiles: tuple[SafeAlias, ...]
    telemetry: ConversationTelemetryPolicy

    def __post_init__(self) -> None:
        if (
            not isinstance(self.default_mode, ConversationMode)
            or type(self.allowed_modes) is not frozenset
            or self.default_mode not in self.allowed_modes
            or not self.allowed_modes
            or any(
                not isinstance(value, ConversationMode)
                for value in self.allowed_modes
            )
            or type(self.allowed_reasoning_contexts) is not frozenset
            or not self.allowed_reasoning_contexts
            or any(
                not isinstance(value, ReasoningContext)
                for value in self.allowed_reasoning_contexts
            )
            or type(self.compaction) is not ConversationCompactionPolicy
            or not isinstance(self.backend, ConversationCheckpointBackend)
            or type(self.retention) is not RetentionLimits
            or type(self.resources) is not ConversationResourcePolicy
            or type(self.checkpoint_keys) is not ConversationKeyRotationPolicy
            or type(self.envelope_keys) is not ConversationKeyRotationPolicy
            or type(self.telemetry) is not ConversationTelemetryPolicy
        ):
            raise ConversationValidationError()
        if type(self.capability_profiles) is not tuple or len(
            self.capability_profiles
        ) != len(set(self.capability_profiles)):
            raise ConversationValidationError()
        for value in self.capability_profiles:
            validate_identifier(value, "capability_profile")
        storage = self.retention.storage
        if ConversationMode.STORED in self.allowed_modes and (
            self.backend is not ConversationCheckpointBackend.POSTGRESQL
            or storage.local is not LocalResponseStorage.DURABLE
            or storage.upstream is not ProviderLaneStorage.STORED
        ):
            raise ConversationValidationError()
        if (
            self.backend is ConversationCheckpointBackend.CALLER_HELD
            and storage.local is not LocalResponseStorage.TRANSIENT
        ):
            raise ConversationValidationError()
        _validate_mode_retention_backend(
            self.default_mode,
            self.backend,
            self.retention,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationConfigurationLayer:
    """Carry one optional narrowing layer from a frozen authority source."""

    source: ConfigurationSource
    mode: ConversationMode | None = None
    allowed_modes: frozenset[ConversationMode] | None = None
    reasoning_context: ReasoningContext | None = None
    allowed_reasoning_contexts: frozenset[ReasoningContext] | None = None
    compaction_operation: CompactionOperation | None = None
    inline_threshold: int | None = None
    retention: RetentionLimits | None = None
    resources: ConversationResourcePolicy | None = None
    capability_profiles: tuple[SafeAlias, ...] | None = None
    telemetry_enabled: bool | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, ConfigurationSource)
            or self.source is ConfigurationSource.SERVER_POLICY
        ):
            raise ConversationValidationError()
        if self.mode is not None and not isinstance(
            self.mode, ConversationMode
        ):
            raise ConversationValidationError()
        if self.reasoning_context is not None and not isinstance(
            self.reasoning_context,
            ReasoningContext,
        ):
            raise ConversationValidationError()
        if self.compaction_operation is not None and not isinstance(
            self.compaction_operation,
            CompactionOperation,
        ):
            raise ConversationValidationError()
        _validate_enum_set(self.allowed_modes, ConversationMode)
        _validate_enum_set(
            self.allowed_reasoning_contexts,
            ReasoningContext,
        )
        if self.inline_threshold is not None and (
            type(self.inline_threshold) is not int
            or self.inline_threshold <= 0
        ):
            raise ConversationValidationError()
        if (
            self.inline_threshold is not None
            and self.compaction_operation is not CompactionOperation.INLINE
        ):
            raise ConversationValidationError()
        if (
            self.retention is not None
            and type(self.retention) is not RetentionLimits
        ):
            raise ConversationValidationError()
        if (
            self.resources is not None
            and type(self.resources) is not ConversationResourcePolicy
        ):
            raise ConversationValidationError()
        if self.capability_profiles is not None:
            if type(self.capability_profiles) is not tuple or len(
                self.capability_profiles
            ) != len(set(self.capability_profiles)):
                raise ConversationValidationError()
            for value in self.capability_profiles:
                validate_identifier(value, "capability_profile")
        if (
            self.telemetry_enabled is not None
            and type(self.telemetry_enabled) is not bool
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationPolicySource:
    """Record one content-free effective-policy source decision."""

    field: str
    source: ConfigurationSource

    def __post_init__(self) -> None:
        validate_identifier(self.field, "field")
        if not isinstance(self.source, ConfigurationSource):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveConversationPolicy:
    """Return exact effective policy and safe precedence metadata."""

    mode: ConversationMode
    allowed_modes: frozenset[ConversationMode]
    reasoning_context: ReasoningContext
    allowed_reasoning_contexts: frozenset[ReasoningContext]
    compaction_operation: CompactionOperation
    inline_threshold: int | None
    backend: ConversationCheckpointBackend
    retention: RetentionLimits
    resources: ConversationResourcePolicy
    capability_profiles: tuple[SafeAlias, ...]
    telemetry_enabled: bool
    sources: tuple[ConversationPolicySource, ...]

    def __post_init__(self) -> None:
        if (
            self.mode not in self.allowed_modes
            or self.reasoning_context not in self.allowed_reasoning_contexts
            or (self.compaction_operation is CompactionOperation.INLINE)
            != (self.inline_threshold is not None)
            or not isinstance(self.backend, ConversationCheckpointBackend)
            or type(self.retention) is not RetentionLimits
            or type(self.resources) is not ConversationResourcePolicy
            or type(self.capability_profiles) is not tuple
            or type(self.telemetry_enabled) is not bool
            or type(self.sources) is not tuple
            or any(
                type(value) is not ConversationPolicySource
                for value in self.sources
            )
        ):
            raise ConversationValidationError()

    def diagnostic_metadata(self) -> Mapping[str, JsonValue]:
        """Return only closed content-free effective policy metadata."""
        return MappingProxyType(
            {
                "allowed_modes": tuple(
                    value.value for value in sorted(self.allowed_modes)
                ),
                "allowed_reasoning_contexts": tuple(
                    value.value
                    for value in sorted(self.allowed_reasoning_contexts)
                ),
                "backend": self.backend.value,
                "capability_profile_count": len(self.capability_profiles),
                "compaction": self.compaction_operation.value,
                "effective_ttl_seconds": self.retention.effective_ttl_seconds,
                "envelope_ttl_seconds": self.retention.envelope_ttl_seconds,
                "inline_threshold": self.inline_threshold,
                "local_storage": self.retention.storage.local.value,
                "local_ttl_seconds": self.retention.local_ttl_seconds,
                "mode": self.mode.value,
                "reasoning_context": self.reasoning_context.value,
                "sources": tuple(
                    {"field": value.field, "source": value.source.value}
                    for value in self.sources
                ),
                "telemetry_enabled": self.telemetry_enabled,
                "upstream_storage": self.retention.storage.upstream.value,
            }
        )


def resolve_conversation_policy(
    server_policy: ConversationHardeningPolicy,
    layers: tuple[ConversationConfigurationLayer, ...] = (),
) -> EffectiveConversationPolicy:
    """Resolve strict precedence while rejecting every policy broadening."""
    if (
        type(server_policy) is not ConversationHardeningPolicy
        or type(layers) is not tuple
    ):
        raise ConversationValidationError()
    if any(
        type(layer) is not ConversationConfigurationLayer for layer in layers
    ):
        raise ConversationValidationError()
    by_source = {layer.source: layer for layer in layers}
    if len(by_source) != len(layers):
        raise ConversationValidationError()
    ordered = tuple(
        by_source[source]
        for source in CONFIGURATION_PRECEDENCE
        if source is not ConfigurationSource.SERVER_POLICY
        and source in by_source
    )
    allowed_modes = server_policy.allowed_modes
    allowed_reasoning = server_policy.allowed_reasoning_contexts
    retention = server_policy.retention
    resources = server_policy.resources
    capability_profiles = server_policy.capability_profiles
    telemetry_enabled = server_policy.telemetry.enabled
    source_by_field = {
        "allowed_modes": ConfigurationSource.SERVER_POLICY,
        "allowed_reasoning_contexts": ConfigurationSource.SERVER_POLICY,
        "backend": ConfigurationSource.SERVER_POLICY,
        "capability_profiles": ConfigurationSource.SERVER_POLICY,
        "resources": ConfigurationSource.SERVER_POLICY,
        "retention": ConfigurationSource.SERVER_POLICY,
        "telemetry_enabled": ConfigurationSource.SERVER_POLICY,
    }
    for layer in ordered:
        if layer.allowed_modes is not None:
            if not layer.allowed_modes <= allowed_modes:
                raise ConversationValidationError()
            allowed_modes = layer.allowed_modes
            source_by_field["allowed_modes"] = layer.source
        if layer.allowed_reasoning_contexts is not None:
            if not layer.allowed_reasoning_contexts <= allowed_reasoning:
                raise ConversationValidationError()
            allowed_reasoning = layer.allowed_reasoning_contexts
            source_by_field["allowed_reasoning_contexts"] = layer.source
        if layer.retention is not None:
            if not _retention_is_narrower(layer.retention, retention):
                raise ConversationValidationError()
            retention = layer.retention
            source_by_field["retention"] = layer.source
        if layer.resources is not None:
            if not layer.resources.is_narrower_than(resources):
                raise ConversationValidationError()
            resources = layer.resources
            source_by_field["resources"] = layer.source
        if layer.capability_profiles is not None:
            if not set(layer.capability_profiles) <= set(capability_profiles):
                raise ConversationValidationError()
            capability_profiles = layer.capability_profiles
            source_by_field["capability_profiles"] = layer.source
        if layer.telemetry_enabled is not None:
            if layer.telemetry_enabled and not telemetry_enabled:
                raise ConversationValidationError()
            telemetry_enabled = layer.telemetry_enabled
            source_by_field["telemetry_enabled"] = layer.source
    if not allowed_modes or not allowed_reasoning:
        raise ConversationValidationError()
    mode, mode_source = _select_layer_value(
        ordered,
        "mode",
        server_policy.default_mode,
    )
    if mode not in allowed_modes:
        raise ConversationValidationError()
    reasoning, reasoning_source = _select_layer_value(
        ordered,
        "reasoning_context",
        ReasoningContext.AUTO,
    )
    if reasoning not in allowed_reasoning:
        raise ConversationValidationError()
    compaction, compaction_source = _select_layer_value(
        ordered,
        "compaction_operation",
        CompactionOperation.NONE,
    )
    if compaction not in server_policy.compaction.allowed_operations:
        raise ConversationValidationError()
    threshold: int | None = None
    threshold_source = ConfigurationSource.SERVER_POLICY
    if compaction is CompactionOperation.INLINE:
        threshold_value, threshold_source = _select_layer_value(
            ordered,
            "inline_threshold",
            server_policy.compaction.minimum_inline_threshold,
        )
        assert isinstance(threshold_value, int)
        threshold = threshold_value
        if not (
            server_policy.compaction.minimum_inline_threshold
            <= threshold
            <= server_policy.compaction.maximum_inline_threshold
        ):
            raise ConversationValidationError()
    _validate_mode_retention_backend(
        mode,
        server_policy.backend,
        retention,
    )
    sources = [
        ConversationPolicySource(field=field, source=source)
        for field, source in sorted(source_by_field.items())
    ]
    sources.extend(
        (
            ConversationPolicySource(field="mode", source=mode_source),
            ConversationPolicySource(
                field="reasoning_context",
                source=reasoning_source,
            ),
            ConversationPolicySource(
                field="compaction",
                source=compaction_source,
            ),
            ConversationPolicySource(
                field="inline_threshold",
                source=threshold_source,
            ),
        )
    )
    return EffectiveConversationPolicy(
        mode=mode,
        allowed_modes=allowed_modes,
        reasoning_context=reasoning,
        allowed_reasoning_contexts=allowed_reasoning,
        compaction_operation=compaction,
        inline_threshold=threshold,
        backend=server_policy.backend,
        retention=retention,
        resources=resources,
        capability_profiles=capability_profiles,
        telemetry_enabled=telemetry_enabled,
        sources=tuple(sources),
    )


@final
class ConversationOperationalKey:
    """Carry opaque authority key material and bounded lifecycle metadata."""

    __slots__ = (
        "_key_bytes",
        "activated_at",
        "key_id",
        "purposes",
        "read_until",
        "revision",
        "status",
    )

    def __init__(
        self,
        *,
        key_id: str,
        revision: int,
        status: ConversationOperationalKeyStatus,
        purposes: frozenset[ConversationKeyPurpose],
        key_bytes: bytes,
        activated_at: datetime,
        read_until: datetime | None = None,
    ) -> None:
        validate_identifier(key_id, "key_id")
        if (
            type(revision) is not int
            or revision <= 0
            or not isinstance(status, ConversationOperationalKeyStatus)
            or type(purposes) is not frozenset
            or not purposes
            or any(
                not isinstance(value, ConversationKeyPurpose)
                for value in purposes
            )
            or type(key_bytes) is not bytes
            or len(key_bytes) != 32
            or activated_at.utcoffset() is None
        ):
            raise ConversationKeyPolicyError()
        if read_until is not None and read_until.utcoffset() is None:
            raise ConversationKeyPolicyError()
        if status is ConversationOperationalKeyStatus.RETIRING:
            if read_until is None or read_until <= activated_at:
                raise ConversationKeyPolicyError()
        elif read_until is not None:
            raise ConversationKeyPolicyError()
        self.key_id = key_id
        self.revision = revision
        self.status = status
        self.purposes = purposes
        self._key_bytes = key_bytes
        self.activated_at = activated_at
        self.read_until = read_until

    def _copy_material(self) -> bytes:
        """Return key bytes solely to the typed encryption adapters."""
        return bytes(self._key_bytes)

    def __repr__(self) -> str:
        """Return lifecycle metadata without key material."""
        return (
            "ConversationOperationalKey("
            f"key_id={self.key_id!r}, revision={self.revision}, "
            f"status={self.status.value!r}, "
            "purposes="
            f"{tuple(value.value for value in sorted(self.purposes))!r}, "
            "key_bytes=<redacted>)"
        )

    def __str__(self) -> str:
        """Return the redacted representation."""
        return repr(self)

    def __format__(self, format_spec: str) -> str:
        """Format only the redacted representation."""
        if format_spec:
            raise ConversationValidationError()
        return repr(self)

    def __reduce__(self) -> object:
        """Reject generic serialization of secret key material."""
        raise TypeError("conversation secret serialization is prohibited")

    def __reduce_ex__(self, protocol: int) -> object:
        """Reject protocol-specific serialization of secret key material."""
        del protocol
        raise TypeError("conversation secret serialization is prohibited")


class ConversationSecurityClock(Protocol):
    """Return aware wall time behind an asynchronous boundary."""

    async def now(self) -> datetime:
        """Return one aware current instant."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationKeyHealth:
    """Report content-free key lifecycle readiness."""

    active_checkpoint_keys: int
    active_envelope_keys: int
    retiring_keys: int
    retired_keys: int
    compromised_keys: int
    highest_revision: int

    def __post_init__(self) -> None:
        for value in (
            self.active_checkpoint_keys,
            self.active_envelope_keys,
            self.retiring_keys,
            self.retired_keys,
            self.compromised_keys,
            self.highest_revision,
        ):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()


@final
class AsyncConversationKeyRing:
    """Rotate authority-scoped checkpoint and envelope keys atomically."""

    def __init__(
        self,
        keys: Mapping[AuthorityDigest, tuple[ConversationOperationalKey, ...]],
        *,
        clock: ConversationSecurityClock,
        policy: ConversationKeyRotationPolicy = (
            ConversationKeyRotationPolicy()
        ),
    ) -> None:
        if not isinstance(keys, Mapping) or not keys:
            raise ConversationKeyPolicyError()
        if (
            not callable(getattr(clock, "now", None))
            or type(policy) is not ConversationKeyRotationPolicy
        ):
            raise ConversationKeyPolicyError()
        self._keys: dict[
            AuthorityDigest, tuple[ConversationOperationalKey, ...]
        ] = {}
        for scope, values in keys.items():
            validate_identifier(scope, "authority_digest")
            self._validate_key_set(values, policy)
            self._keys[scope] = values
        self._clock = clock
        self._policy = policy
        self._lock = Lock()

    def checkpoint_resolver(self) -> "HardeningCheckpointKeyResolver":
        """Return a typed checkpoint-key resolver facade."""
        return HardeningCheckpointKeyResolver(key_ring=self)

    def envelope_resolver(self) -> "HardeningEnvelopeKeyResolver":
        """Return a typed caller-envelope key resolver facade."""
        return HardeningEnvelopeKeyResolver(key_ring=self)

    async def replace_keys(
        self,
        scope: AuthorityDigest,
        keys: tuple[ConversationOperationalKey, ...],
    ) -> None:
        """Replace one authority key policy without generation rollback."""
        validate_identifier(scope, "authority_digest")
        self._validate_key_set(keys, self._policy)
        async with self._lock:
            existing = self._keys.get(scope, ())
            previous = max(
                (value.revision for value in existing),
                default=0,
            )
            candidate = max(value.revision for value in keys)
            if candidate < previous:
                raise ConversationKeyPolicyError()
            for value in keys:
                previous_identity = next(
                    (
                        existing_value
                        for existing_value in existing
                        if existing_value.key_id == value.key_id
                        and existing_value.revision == value.revision
                    ),
                    None,
                )
                if previous_identity is not None and not compare_digest(
                    previous_identity._copy_material(),
                    value._copy_material(),
                ):
                    raise ConversationKeyPolicyError()
            for purpose in ConversationKeyPurpose:
                old_active = next(
                    (
                        value
                        for value in existing
                        if value.status
                        is ConversationOperationalKeyStatus.ACTIVE
                        and purpose in value.purposes
                    ),
                    None,
                )
                new_active = next(
                    value
                    for value in keys
                    if value.status is ConversationOperationalKeyStatus.ACTIVE
                    and purpose in value.purposes
                )
                if old_active is not None and (
                    new_active.revision < old_active.revision
                    or new_active.revision == old_active.revision
                    and new_active.key_id != old_active.key_id
                ):
                    raise ConversationKeyPolicyError()
            self._keys[scope] = keys

    async def resolve_active(
        self,
        scope: AuthorityDigest,
        purpose: ConversationKeyPurpose,
    ) -> ConversationOperationalKey:
        """Return the only active key for one exact purpose."""
        validate_identifier(scope, "authority_digest")
        if not isinstance(purpose, ConversationKeyPurpose):
            raise ConversationValidationError()
        async with self._lock:
            values = self._keys.get(scope, ())
            active = tuple(
                value
                for value in values
                if value.status is ConversationOperationalKeyStatus.ACTIVE
                and purpose in value.purposes
            )
        if len(active) != 1:
            raise ConversationKeyMissingError()
        return active[0]

    async def resolve_read(
        self,
        scope: AuthorityDigest,
        *,
        purpose: ConversationKeyPurpose,
        key_id: str,
        revision: int,
    ) -> ConversationOperationalKey:
        """Return one readable key after applying lifecycle and expiry."""
        key = await self._resolve_exact(
            scope,
            purpose=purpose,
            key_id=key_id,
            revision=revision,
        )
        now = await self._clock.now()
        _validate_aware_time(now)
        if key.status is ConversationOperationalKeyStatus.COMPROMISED:
            raise ConversationKeyCompromisedError()
        if key.status is ConversationOperationalKeyStatus.RETIRED or (
            key.status is ConversationOperationalKeyStatus.RETIRING
            and key.read_until is not None
            and now >= key.read_until
        ):
            raise ConversationKeyRetiredError()
        return key

    async def resolve_deletion(
        self,
        scope: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ConversationOperationalKey:
        """Return an incident-policy key solely for deletion reconciliation."""
        key = await self._resolve_exact(
            scope,
            purpose=ConversationKeyPurpose.CHECKPOINT,
            key_id=key_id,
            revision=revision,
        )
        if (
            key.status is ConversationOperationalKeyStatus.COMPROMISED
            and not self._policy.compromised_deletion_access
        ):
            raise ConversationKeyCompromisedError()
        return key

    async def health(self, scope: AuthorityDigest) -> ConversationKeyHealth:
        """Return content-free lifecycle counts for readiness."""
        validate_identifier(scope, "authority_digest")
        async with self._lock:
            keys = self._keys.get(scope, ())
        return ConversationKeyHealth(
            active_checkpoint_keys=sum(
                value.status is ConversationOperationalKeyStatus.ACTIVE
                and ConversationKeyPurpose.CHECKPOINT in value.purposes
                for value in keys
            ),
            active_envelope_keys=sum(
                value.status is ConversationOperationalKeyStatus.ACTIVE
                and ConversationKeyPurpose.ENVELOPE in value.purposes
                for value in keys
            ),
            retiring_keys=sum(
                value.status is ConversationOperationalKeyStatus.RETIRING
                for value in keys
            ),
            retired_keys=sum(
                value.status is ConversationOperationalKeyStatus.RETIRED
                for value in keys
            ),
            compromised_keys=sum(
                value.status is ConversationOperationalKeyStatus.COMPROMISED
                for value in keys
            ),
            highest_revision=max(
                (value.revision for value in keys), default=0
            ),
        )

    async def _resolve_exact(
        self,
        scope: AuthorityDigest,
        *,
        purpose: ConversationKeyPurpose,
        key_id: str,
        revision: int,
    ) -> ConversationOperationalKey:
        validate_identifier(scope, "authority_digest")
        validate_identifier(key_id, "key_id")
        if type(revision) is not int or revision <= 0:
            raise ConversationValidationError()
        async with self._lock:
            key = next(
                (
                    value
                    for value in self._keys.get(scope, ())
                    if value.key_id == key_id
                    and value.revision == revision
                    and purpose in value.purposes
                ),
                None,
            )
        if key is None:
            raise ConversationKeyMissingError()
        return key

    @staticmethod
    def _validate_key_set(
        keys: tuple[ConversationOperationalKey, ...],
        policy: ConversationKeyRotationPolicy,
    ) -> None:
        if (
            type(keys) is not tuple
            or not keys
            or any(
                type(value) is not ConversationOperationalKey for value in keys
            )
        ):
            raise ConversationKeyPolicyError()
        identities = tuple((value.key_id, value.revision) for value in keys)
        if len(identities) != len(set(identities)):
            raise ConversationKeyPolicyError()
        for purpose in ConversationKeyPurpose:
            active = tuple(
                value
                for value in keys
                if value.status is ConversationOperationalKeyStatus.ACTIVE
                and purpose in value.purposes
            )
            if len(active) != 1:
                raise ConversationKeyPolicyError()
            if any(
                value.revision >= active[0].revision
                for value in keys
                if value.status is ConversationOperationalKeyStatus.RETIRING
                and purpose in value.purposes
            ):
                raise ConversationKeyPolicyError()
        retiring = tuple(
            value
            for value in keys
            if value.status is ConversationOperationalKeyStatus.RETIRING
        )
        if len(retiring) > policy.max_retiring_keys:
            raise ConversationKeyPolicyError()
        if any(
            value.read_until is None
            or (value.read_until - value.activated_at).total_seconds()
            > policy.maximum_retiring_seconds
            or (value.read_until - value.activated_at).total_seconds()
            < policy.minimum_rotation_seconds
            for value in retiring
        ):
            raise ConversationKeyPolicyError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class HardeningCheckpointKeyResolver:
    """Adapt the operational key ring to durable checkpoint encryption."""

    key_ring: AsyncConversationKeyRing

    def __post_init__(self) -> None:
        if type(self.key_ring) is not AsyncConversationKeyRing:
            raise ConversationValidationError()

    async def current_write_key(
        self,
        scope: AuthorityDigest,
    ) -> ConversationDataKey:
        """Return the active checkpoint write key."""
        key = await self.key_ring.resolve_active(
            scope,
            ConversationKeyPurpose.CHECKPOINT,
        )
        return _checkpoint_key(key, ConversationKeyStatus.CURRENT)

    async def read_key(
        self,
        scope: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ConversationDataKey:
        """Return one active or retiring checkpoint read key."""
        key = await self.key_ring.resolve_read(
            scope,
            purpose=ConversationKeyPurpose.CHECKPOINT,
            key_id=key_id,
            revision=revision,
        )
        status = (
            ConversationKeyStatus.CURRENT
            if key.status is ConversationOperationalKeyStatus.ACTIVE
            else ConversationKeyStatus.GRACE
        )
        return _checkpoint_key(key, status)

    async def deletion_key(
        self,
        scope: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ConversationDataKey:
        """Return one explicit deletion-reconciliation key."""
        key = await self.key_ring.resolve_deletion(
            scope,
            key_id=key_id,
            revision=revision,
        )
        return _checkpoint_key(key, ConversationKeyStatus.GRACE)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class HardeningEnvelopeKeyResolver:
    """Adapt the operational key ring to caller-held envelope encryption."""

    key_ring: AsyncConversationKeyRing

    def __post_init__(self) -> None:
        if type(self.key_ring) is not AsyncConversationKeyRing:
            raise ConversationValidationError()

    async def active_key(
        self,
        scope: AuthorityDigest,
    ) -> ContinuationEnvelopeKey:
        """Return the active envelope sealing key."""
        key = await self.key_ring.resolve_active(
            scope,
            ConversationKeyPurpose.ENVELOPE,
        )
        return _envelope_key(key)

    async def read_key(
        self,
        scope: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ContinuationEnvelopeKey:
        """Return one active or retiring envelope read key."""
        key = await self.key_ring.resolve_read(
            scope,
            purpose=ConversationKeyPurpose.ENVELOPE,
            key_id=key_id,
            revision=revision,
        )
        return _envelope_key(key)


@final
class ConversationCorrelationKey:
    """Carry a telemetry-only HMAC key without rendering its material."""

    __slots__ = ("_key_bytes", "key_id")

    def __init__(self, *, key_id: str, key_bytes: bytes) -> None:
        validate_identifier(key_id, "key_id")
        if type(key_bytes) is not bytes or len(key_bytes) < 32:
            raise ConversationValidationError()
        self.key_id = key_id
        self._key_bytes = key_bytes

    def _digest(self, payload: bytes) -> bytes:
        """Return one HMAC without exposing the backing key material."""
        return hmac_digest(self._key_bytes, payload, "sha256")

    def __repr__(self) -> str:
        """Return only a redacted correlation-key marker."""
        return "ConversationCorrelationKey(key_bytes=<redacted>)"

    def __str__(self) -> str:
        """Return the redacted representation."""
        return repr(self)

    def __format__(self, format_spec: str) -> str:
        """Format only the redacted representation."""
        if format_spec:
            raise ConversationValidationError()
        return repr(self)

    def __reduce__(self) -> object:
        """Reject generic serialization of correlation key material."""
        raise TypeError("conversation secret serialization is prohibited")

    def __reduce_ex__(self, protocol: int) -> object:
        """Reject protocol-specific serialization of key material."""
        del protocol
        raise TypeError("conversation secret serialization is prohibited")


def conversation_correlation_digest(
    value: str,
    *,
    namespace: str,
    key: ConversationCorrelationKey,
) -> IntegrityDigest:
    """Return an opaque keyed digest suitable for bounded correlation."""
    validate_identifier(value, "correlation_value", max_length=8_192)
    validate_identifier(namespace, "correlation_namespace")
    if type(key) is not ConversationCorrelationKey:
        raise ConversationValidationError()
    payload = (
        b"avalan.conversation.correlation.v1\x00"
        + namespace.encode("utf-8")
        + b"\x00"
        + value.encode("utf-8")
    )
    return IntegrityDigest(key._digest(payload).hex())


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SafeConversationEvent:
    """Carry only closed content-free operational event metadata."""

    kind: ConversationEventKind
    correlation_digest: IntegrityDigest
    parent_digest: IntegrityDigest | None = None
    mode: ConversationMode | None = None
    reasoning_context: ReasoningContext | None = None
    compaction: CompactionOperation | None = None
    item_count: int = 0
    byte_count: int = 0
    revision: int | None = None
    restarted: bool = False
    error_code: ConversationErrorCode | None = None
    failure_boundary: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConversationEventKind):
            raise ConversationValidationError()
        _validate_digest(self.correlation_digest)
        if self.parent_digest is not None:
            _validate_digest(self.parent_digest)
        if self.mode is not None and not isinstance(
            self.mode, ConversationMode
        ):
            raise ConversationValidationError()
        if self.reasoning_context is not None and not isinstance(
            self.reasoning_context,
            ReasoningContext,
        ):
            raise ConversationValidationError()
        if self.compaction is not None and not isinstance(
            self.compaction,
            CompactionOperation,
        ):
            raise ConversationValidationError()
        for value in (self.item_count, self.byte_count):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()
        if self.revision is not None and (
            type(self.revision) is not int or self.revision < 0
        ):
            raise ConversationValidationError()
        if type(self.restarted) is not bool:
            raise ConversationValidationError()
        if self.error_code is not None and not isinstance(
            self.error_code,
            ConversationErrorCode,
        ):
            raise ConversationValidationError()
        if self.failure_boundary is not None:
            validate_identifier(self.failure_boundary, "failure_boundary")

    def to_mapping(self) -> Mapping[str, JsonValue]:
        """Return one immutable safe serialization for logs and metrics."""
        return MappingProxyType(
            {
                "byte_count": self.byte_count,
                "compaction": (
                    self.compaction.value
                    if self.compaction is not None
                    else None
                ),
                "correlation_digest": self.correlation_digest,
                "error_code": (
                    self.error_code.value
                    if self.error_code is not None
                    else None
                ),
                "failure_boundary": self.failure_boundary,
                "item_count": self.item_count,
                "kind": self.kind.value,
                "mode": self.mode.value if self.mode is not None else None,
                "parent_digest": self.parent_digest,
                "reasoning_context": (
                    self.reasoning_context.value
                    if self.reasoning_context is not None
                    else None
                ),
                "restarted": self.restarted,
                "revision": self.revision,
            }
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SafeConversationException:
    """Project one exception without its message, cause, or content."""

    error_code: str
    failure_boundary: str

    def __post_init__(self) -> None:
        validate_identifier(self.error_code, "error_code")
        validate_identifier(self.failure_boundary, "failure_boundary")


def project_conversation_exception(
    error: BaseException,
) -> SafeConversationException:
    """Return the central redaction-safe exception projection."""
    if isinstance(error, ConversationError):
        return SafeConversationException(
            error_code=error.code.value,
            failure_boundary=error.boundary.value,
        )
    return SafeConversationException(
        error_code="conversation_internal_failure",
        failure_boundary="internal",
    )


class ConversationTelemetrySink(Protocol):
    """Publish content-safe telemetry asynchronously."""

    async def emit(self, event: SafeConversationEvent) -> None:
        """Publish one already-redacted event."""
        ...


@final
class BoundedConversationTelemetry:
    """Retain a bounded content-free telemetry snapshot for embedded use."""

    def __init__(self, *, max_events: int) -> None:
        if type(max_events) is not int or max_events <= 0:
            raise ConversationValidationError()
        self._events: deque[SafeConversationEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    async def emit(self, event: SafeConversationEvent) -> None:
        """Publish one already-redacted event."""
        if type(event) is not SafeConversationEvent:
            raise ConversationValidationError()
        async with self._lock:
            self._events.append(event)

    async def snapshot(self) -> tuple[SafeConversationEvent, ...]:
        """Return a stable content-free event snapshot."""
        async with self._lock:
            return tuple(self._events)

    async def clear(self) -> None:
        """Delete retained operational events."""
        async with self._lock:
            self._events.clear()


@final
class ConversationSecurityContext:
    """Bind trusted authority to one served deployment."""

    __slots__ = ("_authority", "_authority_digest", "_deployment_digest")

    def __init__(
        self, *, authority: AuthorityScope, deployment_id: str
    ) -> None:
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        validate_identifier(deployment_id, "deployment_id")
        self._authority = authority
        self._authority_digest = AuthorityDigest(authority_digest(authority))
        self._deployment_digest = sha256(
            deployment_id.encode("utf-8")
        ).digest()

    @property
    def authority(self) -> AuthorityScope:
        """Return trusted authority for typed internal dispatch only."""
        return self._authority

    @property
    def authority_scope_digest(self) -> AuthorityDigest:
        """Return the content-safe authority digest."""
        return self._authority_digest

    def __repr__(self) -> str:
        """Return only digest-bound security context metadata."""
        return (
            "ConversationSecurityContext(authority_scope_digest="
            f"{self._authority_digest!r}, deployment=<redacted>)"
        )

    def __str__(self) -> str:
        """Return the redacted representation."""
        return repr(self)

    def __format__(self, format_spec: str) -> str:
        """Format only the redacted representation."""
        if format_spec:
            raise ConversationValidationError()
        return repr(self)

    def __reduce__(self) -> object:
        """Reject generic serialization of trusted authority state."""
        raise TypeError("conversation authority serialization is prohibited")

    def __reduce_ex__(self, protocol: int) -> object:
        """Reject protocol-specific serialization of trusted authority."""
        del protocol
        raise TypeError("conversation authority serialization is prohibited")


def authorize_conversation_target(
    owner: ConversationSecurityContext,
    caller: ConversationSecurityContext,
    operation: ConversationOperation | ResponseOperation,
) -> None:
    """Authorize exact scope before any target-existence disclosure."""
    if (
        type(owner) is not ConversationSecurityContext
        or type(caller) is not ConversationSecurityContext
        or not isinstance(operation, ConversationOperation | ResponseOperation)
    ):
        raise ConversationValidationError()
    deployment_matches = compare_digest(
        owner._deployment_digest,
        caller._deployment_digest,
    )
    if (
        not compare_digest(
            owner.authority_scope_digest,
            caller.authority_scope_digest,
        )
        or not deployment_matches
    ):
        raise ConversationAuthorizationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationPayloadDeduplicationPolicy:
    """Enable content addressing only with every required safety property."""

    disposition: ConversationDeduplicationDisposition
    minimum_payload_bytes: int
    tenant_isolation: bool
    authenticated_metadata: bool
    durable_refcounts: bool
    deletion_safe: bool
    rotation_safe: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.disposition, ConversationDeduplicationDisposition
            )
            or type(self.minimum_payload_bytes) is not int
            or self.minimum_payload_bytes <= 0
        ):
            raise ConversationValidationError()
        safeguards = (
            self.tenant_isolation,
            self.authenticated_metadata,
            self.durable_refcounts,
            self.deletion_safe,
            self.rotation_safe,
        )
        if any(type(value) is not bool for value in safeguards):
            raise ConversationValidationError()
        if (
            self.disposition
            is ConversationDeduplicationDisposition.AUTHENTICATED_TENANT_SCOPED
            and not all(safeguards)
        ):
            raise ConversationValidationError()

    def address(
        self,
        *,
        authority: AuthorityDigest,
        authenticated_payload_digest: IntegrityDigest,
        authenticated_metadata_digest: IntegrityDigest,
        payload_bytes: int,
    ) -> IntegrityDigest | None:
        """Return a tenant-scoped address only when policy permits sharing."""
        _validate_digest(authority)
        _validate_digest(authenticated_payload_digest)
        _validate_digest(authenticated_metadata_digest)
        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ConversationValidationError()
        if (
            self.disposition is ConversationDeduplicationDisposition.DISABLED
            or payload_bytes < self.minimum_payload_bytes
        ):
            return None
        value = freeze_json_value(
            {
                "authority": authority,
                "metadata": authenticated_metadata_digest,
                "payload": authenticated_payload_digest,
                "version": 1,
            }
        )
        return IntegrityDigest(sha256(canonical_json_bytes(value)).hexdigest())


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationAdmissionKey:
    """Select one authority and conversation fairness bucket by digest."""

    authority_digest: AuthorityDigest
    conversation_digest: IntegrityDigest

    def __post_init__(self) -> None:
        _validate_digest(self.authority_digest)
        _validate_digest(self.conversation_digest)


@final
@dataclass(slots=True)
class _AdmissionWaiter:
    key: ConversationAdmissionKey
    future: Future[None]


@final
class ConversationAdmissionLease:
    """Release one fair admission slot exactly once."""

    def __init__(
        self,
        controller: "FairConversationAdmissionController",
        key: ConversationAdmissionKey,
    ) -> None:
        self._controller = controller
        self._key = key
        self._released = False

    async def __aenter__(self) -> "ConversationAdmissionLease":
        if self._released:
            raise ConversationValidationError()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.release()

    async def release(self) -> None:
        """Release this lease idempotently."""
        if self._released:
            return
        self._released = True
        await self._controller._release(self._key)


@final
class FairConversationAdmissionController:
    """Bound queues and schedule authorities in round-robin order."""

    def __init__(self, policy: ConversationResourcePolicy) -> None:
        if type(policy) is not ConversationResourcePolicy:
            raise ConversationValidationError()
        self._policy = policy
        self._lock = Lock()
        self._queues: dict[AuthorityDigest, deque[_AdmissionWaiter]] = {}
        self._authority_order: deque[AuthorityDigest] = deque()
        self._active_authorities: dict[AuthorityDigest, int] = {}
        self._active_conversations: dict[
            tuple[AuthorityDigest, IntegrityDigest], int
        ] = {}
        self._active_total = 0

    async def acquire(
        self,
        key: ConversationAdmissionKey,
    ) -> ConversationAdmissionLease:
        """Wait outside locks for one bounded fair admission slot."""
        if type(key) is not ConversationAdmissionKey:
            raise ConversationValidationError()
        waiter = _AdmissionWaiter(
            key=key,
            future=get_running_loop().create_future(),
        )
        async with self._lock:
            queued = sum(len(values) for values in self._queues.values())
            if queued >= self._policy.max_queue_size:
                raise ConversationLimitError()
            queue = self._queues.setdefault(key.authority_digest, deque())
            if not queue:
                self._authority_order.append(key.authority_digest)
            queue.append(waiter)
            self._dispatch_locked()
        try:
            done, _ = await wait(
                (waiter.future,),
                timeout=self._policy.queue_timeout_seconds,
            )
        except BaseException:
            acquired = await self._remove_waiter(waiter)
            if acquired:
                await self._release(key)
            raise
        if waiter.future not in done:
            acquired = await self._remove_waiter(waiter)
            if acquired:
                await self._release(key)
            raise ConversationLimitError()
        return ConversationAdmissionLease(self, key)

    async def snapshot(self) -> tuple[int, int]:
        """Return active and queued content-free counts."""
        async with self._lock:
            return self._active_total, sum(
                len(values) for values in self._queues.values()
            )

    async def _remove_waiter(self, waiter: _AdmissionWaiter) -> bool:
        async with self._lock:
            if waiter.future.done() and not waiter.future.cancelled():
                return True
            queue = self._queues.get(waiter.key.authority_digest)
            if queue is not None:
                try:
                    queue.remove(waiter)
                except ValueError:
                    pass
                self._remove_empty_queue(waiter.key.authority_digest)
            if not waiter.future.done():
                waiter.future.cancel()
            self._dispatch_locked()
            return False

    async def _release(self, key: ConversationAdmissionKey) -> None:
        async with self._lock:
            authority_count = self._active_authorities.get(
                key.authority_digest,
                0,
            )
            conversation_key = (
                key.authority_digest,
                key.conversation_digest,
            )
            conversation_count = self._active_conversations.get(
                conversation_key,
                0,
            )
            if authority_count <= 0 or conversation_count <= 0:
                raise ConversationValidationError()
            self._active_total -= 1
            _decrement_count(self._active_authorities, key.authority_digest)
            _decrement_count(self._active_conversations, conversation_key)
            self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while (
            self._active_total < self._policy.max_global_concurrency
            and self._authority_order
        ):
            selected: _AdmissionWaiter | None = None
            rounds = len(self._authority_order)
            for _ in range(rounds):
                authority = self._authority_order.popleft()
                queue = self._queues.get(authority)
                if not queue:
                    self._queues.pop(authority, None)
                    continue
                if self._active_authorities.get(authority, 0) >= (
                    self._policy.max_authority_concurrency
                ):
                    self._authority_order.append(authority)
                    continue
                for _ in range(len(queue)):
                    candidate = queue.popleft()
                    if candidate.future.cancelled():
                        continue
                    conversation_key = (
                        authority,
                        candidate.key.conversation_digest,
                    )
                    if (
                        self._active_conversations.get(
                            conversation_key,
                            0,
                        )
                        < self._policy.max_conversation_concurrency
                    ):
                        selected = candidate
                        break
                    queue.append(candidate)
                if queue:
                    self._authority_order.append(authority)
                else:
                    self._queues.pop(authority, None)
                if selected is not None:
                    break
            if selected is None:
                return
            key = selected.key
            conversation_key = (key.authority_digest, key.conversation_digest)
            self._active_total += 1
            self._active_authorities[key.authority_digest] = (
                self._active_authorities.get(key.authority_digest, 0) + 1
            )
            self._active_conversations[conversation_key] = (
                self._active_conversations.get(conversation_key, 0) + 1
            )
            if not selected.future.done():
                selected.future.set_result(None)

    def _remove_empty_queue(self, authority: AuthorityDigest) -> None:
        queue = self._queues.get(authority)
        if queue:
            return
        self._queues.pop(authority, None)
        try:
            self._authority_order.remove(authority)
        except ValueError:
            pass


_T = TypeVar("_T")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationEffectRunner:
    """Apply typed asynchronous timeouts to every external effect class."""

    policy: ConversationResourcePolicy
    _quarantined_tasks: set[Future[object]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.policy) is not ConversationResourcePolicy:
            raise ConversationValidationError()

    async def provider(self, effect: Awaitable[_T]) -> _T:
        """Run one provider effect under its configured timeout."""
        self._require_settled_quarantine()
        return await self._run(
            effect,
            self.policy.provider_timeout_seconds,
            self.policy.cancellation_settlement_seconds,
        )

    async def store(self, effect: Awaitable[_T]) -> _T:
        """Run one store effect under its configured timeout."""
        self._require_settled_quarantine()
        return await self._run(
            effect,
            self.policy.store_timeout_seconds,
            self.policy.cancellation_settlement_seconds,
        )

    async def key(self, effect: Awaitable[_T]) -> _T:
        """Run one key effect under its configured timeout."""
        self._require_settled_quarantine()
        return await self._run(
            effect,
            self.policy.key_timeout_seconds,
            self.policy.cancellation_settlement_seconds,
        )

    async def _run(
        self,
        effect: Awaitable[_T],
        seconds: float,
        settlement_seconds: float,
    ) -> _T:
        task = ensure_future(effect)
        try:
            done, _ = await wait((task,), timeout=seconds)
        except BaseException:
            try:
                settled = await self._cancel_and_settle(
                    task,
                    settlement_seconds,
                )
            except BaseException:
                if not task.done():
                    self._own_quarantined_task(task)
                raise
            if not settled:
                self._own_quarantined_task(task)
            raise
        if task not in done:
            settled = await self._cancel_and_settle(
                task,
                settlement_seconds,
            )
            if not settled:
                self._own_quarantined_task(task)
            raise TimeoutError()
        return task.result()

    @staticmethod
    async def _cancel_and_settle(
        task: Future[_T],
        settlement_seconds: float,
    ) -> bool:
        """Cancel one task without waiting beyond the settlement bound."""
        task.cancel()
        try:
            done, _ = await wait((task,), timeout=settlement_seconds)
        except BaseException:
            if task.done():
                task.result()
            raise
        if task not in done:
            return False
        try:
            task.result()
        except CancelledError:
            pass
        return True

    @property
    def quarantined_task_count(self) -> int:
        """Return the bounded count of cancellation-resistant effects."""
        self._discard_settled_quarantine()
        return len(self._quarantined_tasks)

    def _require_settled_quarantine(self) -> None:
        """Reject new effects while cancellation-resistant work is live."""
        self._discard_settled_quarantine()
        if self._quarantined_tasks:
            raise ConversationValidationError()

    def _own_quarantined_task(self, task: Future[_T]) -> None:
        """Retain a resistant task until its terminal callback runs."""
        owned = cast(Future[object], task)
        self._quarantined_tasks.add(owned)
        owned.add_done_callback(self._release_quarantined_task)

    def _release_quarantined_task(self, task: Future[object]) -> None:
        """Consume and release only a task that has actually terminated."""
        _consume_background_task(task)
        self._quarantined_tasks.discard(task)

    def _discard_settled_quarantine(self) -> None:
        """Reap terminal cancellation-resistant effects synchronously."""
        for task in tuple(self._quarantined_tasks):
            if task.done():
                self._release_quarantined_task(task)


class ConversationMaintenanceOperation(Protocol):
    """Run one bounded maintenance operation over an existing store."""

    @property
    def kind(self) -> ConversationMaintenanceKind:
        """Return the closed operation category."""
        ...

    async def run(self, *, limit: int) -> int:
        """Run at most one bounded batch and return processed count."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationRetentionMaintenanceOperation:
    """Run retention sweep batches through the existing conversation store."""

    store: InMemoryConversationStore | PgsqlConversationStore
    clock: ConversationSecurityClock

    def __post_init__(self) -> None:
        if type(self.store) not in {
            InMemoryConversationStore,
            PgsqlConversationStore,
        } or not callable(getattr(self.clock, "now", None)):
            raise ConversationValidationError()

    @property
    def kind(self) -> ConversationMaintenanceKind:
        """Return the retention category."""
        return ConversationMaintenanceKind.RETENTION

    async def run(self, *, limit: int) -> int:
        """Run at most one bounded batch and return processed count."""
        if type(limit) is not int or limit <= 0:
            raise ConversationValidationError()
        now = await self.clock.now()
        _validate_aware_time(now)
        receipt = await self.store.sweep(now, limit=limit)
        count = receipt.expired + receipt.deleted
        if count > limit:
            raise ConversationValidationError()
        return count


@final
class ConversationOutboxMaintenanceOperation:
    """Recover leased publication work through the store-owned outbox."""

    def __init__(
        self,
        *,
        store: InMemoryConversationStore | PgsqlConversationStore,
        authority: AuthorityScope,
        publisher: ConversationPublisher,
    ) -> None:
        if (
            type(store)
            not in {InMemoryConversationStore, PgsqlConversationStore}
            or type(authority) is not AuthorityScope
            or not callable(getattr(publisher, "publish", None))
        ):
            raise ConversationValidationError()
        self._worker: ConversationOutboxRecoveryWorker = (
            store.create_outbox_recovery_worker(authority)
        )
        self._publisher = publisher

    @property
    def kind(self) -> ConversationMaintenanceKind:
        """Return the durable outbox category."""
        return ConversationMaintenanceKind.OUTBOX

    async def run(self, *, limit: int) -> int:
        """Publish and settle one leased outbox batch."""
        if type(limit) is not int or limit <= 0:
            raise ConversationValidationError()
        batch = await self._worker.claim(limit=limit)
        settled = 0
        for record in batch.records:
            try:
                await self._publisher.publish(record.intent)
                await self._worker.acknowledge(record)
            except BaseException:
                await self._worker.release(record)
                raise
            settled += 1
        return settled


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationLifecycleMaintenanceOperation:
    """Reconcile provider deletion through the existing durable outbox."""

    reconciler: ProviderLifecycleReconciler

    def __post_init__(self) -> None:
        if type(self.reconciler) is not ProviderLifecycleReconciler:
            raise ConversationValidationError()

    @property
    def kind(self) -> ConversationMaintenanceKind:
        """Return the reconciliation category."""
        return ConversationMaintenanceKind.RECONCILIATION

    async def run(self, *, limit: int) -> int:
        """Settle one bounded provider-lifecycle batch."""
        return await self.reconciler.run_once(limit=limit)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationPayloadGcMaintenanceOperation:
    """Collect unreferenced payloads through PostgreSQL refcounts."""

    store: PgsqlConversationStore

    def __post_init__(self) -> None:
        if type(self.store) is not PgsqlConversationStore:
            raise ConversationValidationError()

    @property
    def kind(self) -> ConversationMaintenanceKind:
        """Return the payload garbage-collection category."""
        return ConversationMaintenanceKind.PAYLOAD_GC

    async def run(self, *, limit: int) -> int:
        """Delete one bounded batch of unreferenced payloads."""
        receipt = await self.store.garbage_collect(limit=limit)
        return receipt.deleted_payloads


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationKeyRotationMaintenanceOperation:
    """Re-encrypt payloads through the store's monotonic key cutover."""

    store: PgsqlConversationStore
    authority: AuthorityScope

    def __post_init__(self) -> None:
        if (
            type(self.store) is not PgsqlConversationStore
            or type(self.authority) is not AuthorityScope
        ):
            raise ConversationValidationError()

    @property
    def kind(self) -> ConversationMaintenanceKind:
        """Return the key-rotation category."""
        return ConversationMaintenanceKind.KEY_ROTATION

    async def run(self, *, limit: int) -> int:
        """Re-encrypt one bounded payload batch."""
        receipt = await self.store.rotate_keys(self.authority, limit=limit)
        return receipt.reencrypted


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationMaintenanceHealth:
    """Report bounded worker lifecycle and pending work."""

    state: ConversationWorkerState
    completed_batches: int
    processed_records: int
    task_active: bool
    failure: SafeConversationException | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ConversationWorkerState):
            raise ConversationValidationError()
        for value in (
            self.completed_batches,
            self.processed_records,
        ):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()
        if type(self.task_active) is not bool:
            raise ConversationValidationError()
        if (
            self.failure is not None
            and type(self.failure) is not SafeConversationException
        ):
            raise ConversationValidationError()


@final
class ConversationMaintenanceWorker:
    """Own start, stop, drain, cancellation, fairness, and health semantics."""

    def __init__(
        self,
        operations: tuple[ConversationMaintenanceOperation, ...],
        *,
        batch_size: int,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
    ) -> None:
        if (
            type(operations) is not tuple
            or not operations
            or any(
                not callable(getattr(value, "run", None))
                for value in operations
            )
            or type(batch_size) is not int
            or batch_size <= 0
        ):
            raise ConversationValidationError()
        for value in (interval_seconds, shutdown_timeout_seconds):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ConversationValidationError()
        kinds = tuple(value.kind for value in operations)
        if len(kinds) != len(set(kinds)):
            raise ConversationValidationError()
        self._operations = operations
        self._batch_size = batch_size
        self._interval_seconds = float(interval_seconds)
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._state = ConversationWorkerState.STOPPED
        self._stop = Event()
        self._task: Task[None] | None = None
        self._lock = Lock()
        self._cycle_lock = Lock()
        self._completed_batches = 0
        self._processed_records = 0
        self._next_operation = 0
        self._failure: SafeConversationException | None = None

    async def start(self) -> None:
        """Start exactly one owned worker task."""
        async with self._lock:
            self._refresh_quarantined_task()
            if self._state is not ConversationWorkerState.STOPPED:
                raise ConversationValidationError()
            self._stop = Event()
            self._state = ConversationWorkerState.RUNNING
            self._failure = None
            self._task = create_task(
                self._run(), name="conversation-maintenance"
            )

    async def run_once(self) -> int:
        """Run every maintenance category once in fair rotated order."""
        async with self._lock:
            if self._state is not ConversationWorkerState.RUNNING:
                raise ConversationValidationError()
        return await self._run_cycle()

    async def _run_cycle(self) -> int:
        """Serialize one already-authorized maintenance cycle."""
        async with self._cycle_lock:
            total = 0
            for _ in range(len(self._operations)):
                operation = self._operations[self._next_operation]
                self._next_operation = (self._next_operation + 1) % len(
                    self._operations
                )
                total += await operation.run(limit=self._batch_size)
            self._completed_batches += 1
            self._processed_records += total
            return total

    async def drain(self) -> None:
        """Stop accepting cycles and await the active bounded batch."""
        async with self._lock:
            self._refresh_quarantined_task()
            if self._state is ConversationWorkerState.STOPPED:
                return
            if self._state is ConversationWorkerState.FAILED:
                task = self._task
            elif self._state is ConversationWorkerState.QUARANTINED:
                task = self._task
            else:
                self._state = ConversationWorkerState.DRAINING
                self._stop.set()
                task = self._task
        settled = True
        if task is not None:
            try:
                done, _ = await wait(
                    (task,),
                    timeout=self._shutdown_timeout_seconds,
                )
                if task not in done:
                    settled = (
                        await ConversationEffectRunner._cancel_and_settle(
                            task,
                            self._shutdown_timeout_seconds,
                        )
                    )
            except BaseException as error:
                await self._record_stop_failure(task, error)
                raise
        async with self._lock:
            if not settled and task is not None and not task.done():
                self._state = ConversationWorkerState.QUARANTINED
                self._task = task
                return
            self._state = ConversationWorkerState.STOPPED
            self._task = None

    async def cancel(self) -> None:
        """Cancel the owned task and settle its lifecycle explicitly."""
        async with self._lock:
            self._refresh_quarantined_task()
            task = self._task
            self._stop.set()
            if task is not None:
                task.cancel()
        settled = True
        if task is not None:
            try:
                settled = await ConversationEffectRunner._cancel_and_settle(
                    task,
                    self._shutdown_timeout_seconds,
                )
            except BaseException as error:
                await self._record_stop_failure(task, error)
                raise
        async with self._lock:
            if not settled and task is not None and not task.done():
                self._state = ConversationWorkerState.QUARANTINED
                self._task = task
                return
            self._state = ConversationWorkerState.STOPPED
            self._task = None

    async def health(self) -> ConversationMaintenanceHealth:
        """Return worker health without provider credentials."""
        async with self._lock:
            self._refresh_quarantined_task()
            task = self._task
            return ConversationMaintenanceHealth(
                state=self._state,
                completed_batches=self._completed_batches,
                processed_records=self._processed_records,
                task_active=task is not None and not task.done(),
                failure=self._failure,
            )

    async def _record_stop_failure(
        self,
        task: Task[None],
        error: BaseException,
    ) -> None:
        """Retain pending ownership or project a terminal stop failure."""
        async with self._lock:
            if task.done():
                self._state = ConversationWorkerState.FAILED
                self._failure = project_conversation_exception(error)
                self._task = None
            else:
                self._state = ConversationWorkerState.QUARANTINED
                self._task = task

    def _refresh_quarantined_task(self) -> None:
        """Release quarantine only after the owned task has terminated."""
        task = self._task
        if (
            self._state is not ConversationWorkerState.QUARANTINED
            or task is None
            or not task.done()
        ):
            return
        failure: BaseException | None = None
        try:
            task.result()
        except CancelledError:
            pass
        except BaseException as error:
            failure = error
        self._task = None
        if failure is None:
            self._state = ConversationWorkerState.STOPPED
        else:
            self._state = ConversationWorkerState.FAILED
            self._failure = project_conversation_exception(failure)

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                await self._run_cycle()
                stop_waiter = create_task(self._stop.wait())
                try:
                    done, _ = await wait(
                        (stop_waiter,), timeout=self._interval_seconds
                    )
                except BaseException:
                    stop_waiter.cancel()
                    try:
                        await stop_waiter
                    except BaseException:
                        if not stop_waiter.cancelled():
                            raise
                    raise
                if stop_waiter not in done:
                    stop_waiter.cancel()
                    try:
                        await stop_waiter
                    except BaseException:
                        if not stop_waiter.cancelled():
                            raise
                    continue
        except Exception as error:
            async with self._lock:
                self._state = ConversationWorkerState.FAILED
                self._failure = project_conversation_exception(error)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationSurfaceRevision:
    """Define the readable and writable window for one state surface."""

    surface: ConversationStateSurface
    minimum_reader: int
    maximum_reader: int
    writer: int

    def __post_init__(self) -> None:
        if not isinstance(self.surface, ConversationStateSurface):
            raise ConversationValidationError()
        for value in (self.minimum_reader, self.maximum_reader, self.writer):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()
        if not self.minimum_reader <= self.writer <= self.maximum_reader:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationMigrationContract:
    """Define N/N+1 reads and explicit rollback behavior for every surface."""

    code_revision: int
    revisions: tuple[ConversationSurfaceRevision, ...]
    rollback_disposition: ConversationRollbackDisposition

    def __post_init__(self) -> None:
        if type(self.code_revision) is not int or self.code_revision <= 0:
            raise ConversationValidationError()
        if (
            type(self.revisions) is not tuple
            or len(self.revisions) != len(ConversationStateSurface)
            or {value.surface for value in self.revisions}
            != set(ConversationStateSurface)
            or any(
                type(value) is not ConversationSurfaceRevision
                for value in self.revisions
            )
            or not isinstance(
                self.rollback_disposition,
                ConversationRollbackDisposition,
            )
        ):
            raise ConversationValidationError()

    def require_readable(
        self,
        surface: ConversationStateSurface,
        revision: int,
    ) -> None:
        """Reject corrupt or unknown future state without partial fallback."""
        if (
            not isinstance(surface, ConversationStateSurface)
            or type(revision) is not int
        ):
            raise ConversationValidationError()
        selected = next(
            value for value in self.revisions if value.surface is surface
        )
        if not selected.minimum_reader <= revision <= selected.maximum_reader:
            raise ConversationMigrationRequiredError()

    def require_operation(self, operation: ResponseOperation) -> None:
        """Enforce rollback without visible-transcript substitution."""
        if not isinstance(operation, ResponseOperation):
            raise ConversationValidationError()
        if operation is ResponseOperation.DELETE:
            return
        if (
            self.rollback_disposition
            is not ConversationRollbackDisposition.RESOLVABLE
        ):
            raise ConversationMigrationRequiredError()

    @property
    def visible_transcript_fallback(self) -> bool:
        """Return false because visible messages never replace opaque state."""
        return False


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationBackendHealth:
    """Report migration and durable outbox health without credentials."""

    migration_ready: bool
    schema_version: int
    application_version: int
    outbox_lag: int
    maximum_outbox_lag: int

    def __post_init__(self) -> None:
        if type(self.migration_ready) is not bool:
            raise ConversationValidationError()
        for value in (
            self.schema_version,
            self.application_version,
            self.outbox_lag,
            self.maximum_outbox_lag,
        ):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCapabilityHealth:
    """Report active and historical resolver profile availability."""

    resolver_available: bool
    active_profiles: int
    resolvable_profiles: int

    def __post_init__(self) -> None:
        if type(self.resolver_available) is not bool:
            raise ConversationValidationError()
        for value in (self.active_profiles, self.resolvable_profiles):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()
        if self.resolvable_profiles < self.active_profiles:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationActivationHealth:
    """Compare loaded and expected activation revisions by safe digest."""

    expected_digest: IntegrityDigest
    loaded_digest: IntegrityDigest

    def __post_init__(self) -> None:
        _validate_digest(self.expected_digest)
        _validate_digest(self.loaded_digest)

    @property
    def consistent(self) -> bool:
        """Return whether the loaded activation manifest is exact."""
        return compare_digest(self.expected_digest, self.loaded_digest)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationReadinessReport:
    """Return one content-free aggregate operational readiness decision."""

    ready: bool
    failures: tuple[ConversationReadinessFailure, ...]

    def __post_init__(self) -> None:
        if (
            type(self.ready) is not bool
            or type(self.failures) is not tuple
            or any(
                not isinstance(value, ConversationReadinessFailure)
                for value in self.failures
            )
            or len(self.failures) != len(set(self.failures))
            or self.ready == bool(self.failures)
        ):
            raise ConversationValidationError()


BackendHealthProbe = Callable[[], Awaitable[ConversationBackendHealth]]
CapabilityHealthProbe = Callable[[], Awaitable[ConversationCapabilityHealth]]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationReadinessChecker:
    """Check local dependencies without probing provider credentials."""

    backend_probe: BackendHealthProbe
    key_ring: AsyncConversationKeyRing
    authority: AuthorityDigest
    workers: tuple[ConversationMaintenanceWorker, ...]
    capability_probe: CapabilityHealthProbe
    activation: ConversationActivationHealth
    probe_timeout_seconds: float = 5.0
    probe_settlement_seconds: float = 1.0
    _effect_runner: ConversationEffectRunner = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not callable(self.backend_probe)
            or type(self.key_ring) is not AsyncConversationKeyRing
            or type(self.workers) is not tuple
            or not self.workers
            or any(
                type(value) is not ConversationMaintenanceWorker
                for value in self.workers
            )
            or not callable(self.capability_probe)
            or type(self.activation) is not ConversationActivationHealth
        ):
            raise ConversationValidationError()
        for timeout in (
            self.probe_timeout_seconds,
            self.probe_settlement_seconds,
        ):
            if (
                not isinstance(timeout, int | float)
                or isinstance(timeout, bool)
                or timeout <= 0
            ):
                raise ConversationValidationError()
        _validate_digest(self.authority)
        object.__setattr__(
            self,
            "_effect_runner",
            ConversationEffectRunner(
                policy=ConversationResourcePolicy(
                    provider_timeout_seconds=self.probe_timeout_seconds,
                    store_timeout_seconds=self.probe_timeout_seconds,
                    key_timeout_seconds=self.probe_timeout_seconds,
                    cancellation_settlement_seconds=(
                        self.probe_settlement_seconds
                    ),
                    readiness_timeout_seconds=self.probe_timeout_seconds,
                )
            ),
        )

    async def check(self) -> ConversationReadinessReport:
        """Return aggregate readiness from local bounded probes only."""
        backend = await self._probe(self.backend_probe())
        keys = await self._probe(self.key_ring.health(self.authority))
        capability = await self._probe(self.capability_probe())
        worker_health = tuple(
            [await self._probe(value.health()) for value in self.workers]
        )
        failures: list[ConversationReadinessFailure] = []
        if not backend.migration_ready:
            failures.append(ConversationReadinessFailure.BACKEND_MIGRATION)
        if keys.active_checkpoint_keys != 1 or keys.active_envelope_keys != 1:
            failures.append(ConversationReadinessFailure.ACTIVE_KEYS)
        if backend.outbox_lag > backend.maximum_outbox_lag:
            failures.append(ConversationReadinessFailure.OUTBOX_LAG)
        if any(
            value.state is not ConversationWorkerState.RUNNING
            or not value.task_active
            or value.failure is not None
            for value in worker_health
        ):
            failures.append(ConversationReadinessFailure.SWEEPER)
        if not capability.resolver_available or (
            capability.resolvable_profiles < capability.active_profiles
        ):
            failures.append(ConversationReadinessFailure.CAPABILITY_RESOLVER)
        if not self.activation.consistent:
            failures.append(ConversationReadinessFailure.ACTIVATION_MANIFEST)
        return ConversationReadinessReport(
            ready=not failures,
            failures=tuple(failures),
        )

    async def _probe(self, effect: Awaitable[_T]) -> _T:
        """Run one readiness probe under both timeout bounds."""
        return await self._effect_runner.provider(effect)


@final
class ConversationHardeningCoordinatorHook:
    """Enforce hardening through the coordinator's real await boundaries."""

    def __init__(
        self,
        *,
        policy: EffectiveConversationPolicy,
        admission: FairConversationAdmissionController,
        admission_key: ConversationAdmissionKey,
        readiness: ConversationReadinessChecker,
        telemetry: ConversationTelemetrySink,
    ) -> None:
        if (
            type(policy) is not EffectiveConversationPolicy
            or type(admission) is not FairConversationAdmissionController
            or type(admission_key) is not ConversationAdmissionKey
            or type(readiness) is not ConversationReadinessChecker
            or not callable(getattr(telemetry, "emit", None))
            or admission_key.authority_digest != readiness.authority
        ):
            raise ConversationValidationError()
        self._policy = policy
        self._admission = admission
        self._admission_key = admission_key
        self._readiness = readiness
        self._telemetry = telemetry
        self._leases: dict[Task[object], ConversationAdmissionLease] = {}
        self._lock = Lock()
        self._started = False

    async def start(self) -> None:
        """Start every expected maintenance worker exactly once."""
        async with self._lock:
            if self._started:
                raise ConversationValidationError()
            self._started = True
        started: list[ConversationMaintenanceWorker] = []
        try:
            for worker in self._readiness.workers:
                await worker.start()
                started.append(worker)
        except BaseException:
            for worker in reversed(started):
                await worker.cancel()
            async with self._lock:
                self._started = False
            raise

    async def reach(self, boundary: CoordinatorAwaitBoundary) -> None:
        """Apply readiness, admission, telemetry, and worker lifecycle."""
        if not isinstance(boundary, CoordinatorAwaitBoundary):
            raise ConversationValidationError()
        task = current_task()
        if task is None:
            raise ConversationValidationError()
        typed_task = task
        if boundary is CoordinatorAwaitBoundary.VALIDATE_PLAN:
            async with self._lock:
                if not self._started or typed_task in self._leases:
                    raise ConversationValidationError()
            readiness = await self._readiness.check()
            if not readiness.ready:
                raise ConversationValidationError()
            lease = await self._admission.acquire(self._admission_key)
            async with self._lock:
                self._leases[typed_task] = lease
            await self._emit(ConversationEventKind.MODE)
            return
        if boundary in {
            CoordinatorAwaitBoundary.OBSERVE,
            CoordinatorAwaitBoundary.ROLLBACK,
        }:
            await self._release(typed_task)
            await self._emit(
                ConversationEventKind.COMMIT
                if boundary is CoordinatorAwaitBoundary.OBSERVE
                else ConversationEventKind.FAILURE_BOUNDARY
            )
            return
        if boundary is CoordinatorAwaitBoundary.CLOSE:
            await self.close()

    async def close(self) -> None:
        """Drain workers and reject leaked dispatch leases."""
        async with self._lock:
            if self._leases:
                raise ConversationValidationError()
            if not self._started:
                return
            self._started = False
        for worker in self._readiness.workers:
            await worker.drain()

    async def _release(self, task: Task[object]) -> None:
        async with self._lock:
            lease = self._leases.pop(task, None)
        if lease is not None:
            await lease.release()

    async def _emit(self, kind: ConversationEventKind) -> None:
        await self._telemetry.emit(
            SafeConversationEvent(
                kind=kind,
                correlation_digest=self._admission_key.conversation_digest,
                mode=self._policy.mode,
                reasoning_context=self._policy.reasoning_context,
                compaction=self._policy.compaction_operation,
            )
        )


def is_trusted_conversation_hardening_hook(value: object) -> bool:
    """Return whether a value is the exact production hardening hook."""
    return type(value) is ConversationHardeningCoordinatorHook


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationThreatControlOwnership:
    """Assign one concrete control to its operational owner."""

    control_id: str
    owner: str

    def __post_init__(self) -> None:
        validate_identifier(self.control_id, "control_id", max_length=1_024)
        validate_identifier(self.owner, "control_owner", max_length=1_024)
        _reject_traceability_placeholder(self.control_id)
        _reject_traceability_placeholder(self.owner)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationThreatControl:
    """Map one threat to controls, tests, detection, response, and risk."""

    threat_id: str
    controls: tuple[str, ...]
    control_owners: tuple[ConversationThreatControlOwnership, ...]
    positive_tests: tuple[str, ...]
    negative_tests: tuple[str, ...]
    operator_detection: str
    incident_response: str
    residual_risk: str

    def __post_init__(self) -> None:
        validate_identifier(self.threat_id, "threat_id")
        for values in (
            self.controls,
            self.positive_tests,
            self.negative_tests,
        ):
            if type(values) is not tuple or not values:
                raise ConversationValidationError()
            for value in values:
                validate_identifier(
                    value, "traceability_value", max_length=1_024
                )
                _reject_traceability_placeholder(value)
        if (
            type(self.control_owners) is not tuple
            or any(
                type(value) is not ConversationThreatControlOwnership
                for value in self.control_owners
            )
            or tuple(value.control_id for value in self.control_owners)
            != self.controls
        ):
            raise ConversationValidationError()
        for value, name in (
            (self.operator_detection, "operator_detection"),
            (self.incident_response, "incident_response"),
            (self.residual_risk, "residual_risk"),
        ):
            validate_identifier(value, name, max_length=2_048)
            _reject_traceability_placeholder(value)


def validate_capability_profile_for_activation(
    profile: ConversationCapabilityProfile,
) -> None:
    """Reject activation profiles without production conformance evidence."""
    if type(profile) is not ConversationCapabilityProfile:
        raise ConversationValidationError()
    if profile.test_only or not any(
        evidence.state is CapabilityEvidenceState.CONFORMANT
        for evidence in profile.capabilities
    ):
        raise ConversationValidationError()


def _validate_enum_set(values: object, enum_type: type[StrEnum]) -> None:
    if values is None:
        return
    if (
        type(values) is not frozenset
        or not values
        or any(not isinstance(value, enum_type) for value in values)
    ):
        raise ConversationValidationError()


def _select_layer_value(
    layers: Sequence[ConversationConfigurationLayer],
    field: str,
    default: _T,
) -> tuple[_T, ConfigurationSource]:
    for layer in layers:
        value = getattr(layer, field)
        if value is not None:
            return value, layer.source
    return default, ConfigurationSource.SERVER_POLICY


def _retention_is_narrower(
    candidate: RetentionLimits,
    configured: RetentionLimits,
) -> bool:
    local_order = {
        LocalResponseStorage.NONE: 0,
        LocalResponseStorage.TRANSIENT: 1,
        LocalResponseStorage.PROCESS_LOCAL: 2,
        LocalResponseStorage.DURABLE: 3,
    }
    upstream_order = {
        ProviderLaneStorage.OFF: 0,
        ProviderLaneStorage.STATELESS: 1,
        ProviderLaneStorage.STORED: 2,
    }
    return (
        local_order[candidate.storage.local]
        <= local_order[configured.storage.local]
        and upstream_order[candidate.storage.upstream]
        <= upstream_order[configured.storage.upstream]
        and _ttl_is_narrower(
            candidate.local_ttl_seconds,
            configured.local_ttl_seconds,
        )
        and _ttl_is_narrower(
            candidate.envelope_ttl_seconds,
            configured.envelope_ttl_seconds,
        )
        and _ttl_is_narrower(
            candidate.known_upstream_ttl_seconds,
            configured.known_upstream_ttl_seconds,
        )
    )


def _validate_mode_retention_backend(
    mode: ConversationMode,
    backend: ConversationCheckpointBackend,
    retention: RetentionLimits,
) -> None:
    """Reject an effective mode, backend, and storage contradiction."""
    storage = retention.storage
    valid = False
    match mode:
        case ConversationMode.OFF:
            valid = (
                storage.local is LocalResponseStorage.NONE
                and storage.upstream is ProviderLaneStorage.OFF
            )
        case ConversationMode.STATELESS:
            valid = storage.upstream is ProviderLaneStorage.STATELESS and (
                backend is ConversationCheckpointBackend.POSTGRESQL
                and storage.local is LocalResponseStorage.DURABLE
                or backend is ConversationCheckpointBackend.PROCESS
                and storage.local is LocalResponseStorage.PROCESS_LOCAL
                or backend is ConversationCheckpointBackend.CALLER_HELD
                and storage.local is LocalResponseStorage.TRANSIENT
            )
        case ConversationMode.STORED:
            valid = (
                backend is ConversationCheckpointBackend.POSTGRESQL
                and storage.local is LocalResponseStorage.DURABLE
                and storage.upstream is ProviderLaneStorage.STORED
            )
    if not valid:
        raise ConversationValidationError()


def _ttl_is_narrower(candidate: int | None, configured: int | None) -> bool:
    if configured is None:
        return True
    return candidate is not None and candidate <= configured


def _checkpoint_key(
    key: ConversationOperationalKey,
    status: ConversationKeyStatus,
) -> ConversationDataKey:
    return ConversationDataKey(
        key_id=key.key_id,
        revision=key.revision,
        status=status,
        key_bytes=key._copy_material(),
    )


def _envelope_key(key: ConversationOperationalKey) -> ContinuationEnvelopeKey:
    status = {
        ConversationOperationalKeyStatus.ACTIVE: (
            ContinuationEnvelopeKeyStatus.ACTIVE
        ),
        ConversationOperationalKeyStatus.RETIRING: (
            ContinuationEnvelopeKeyStatus.RETIRING
        ),
        ConversationOperationalKeyStatus.RETIRED: (
            ContinuationEnvelopeKeyStatus.RETIRED
        ),
        ConversationOperationalKeyStatus.COMPROMISED: (
            ContinuationEnvelopeKeyStatus.COMPROMISED
        ),
    }[key.status]
    return ContinuationEnvelopeKey(
        key_id=key.key_id,
        revision=key.revision,
        status=status,
        key_bytes=key._copy_material(),
    )


def _validate_digest(value: object) -> None:
    validate_identifier(value, "digest")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConversationValidationError()


def _reject_traceability_placeholder(value: str) -> None:
    normalized = value.casefold().replace("_", "-")
    if any(
        marker in normalized
        for marker in (
            "placeholder",
            "positive-evidence",
            "negative-evidence",
            "production-control",
            "todo",
            "tbd",
        )
    ):
        raise ConversationValidationError()


def _validate_aware_time(value: datetime) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ConversationValidationError()


def _decrement_count(mapping: dict[_T, int], key: _T) -> None:
    value = mapping[key] - 1
    if value:
        mapping[key] = value
    else:
        del mapping[key]


def _consume_background_task(task: Future[_T]) -> None:
    """Consume a late task result after bounded cancellation settlement."""
    try:
        task.result()
    except BaseException:
        return
