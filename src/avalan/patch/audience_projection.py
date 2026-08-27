"""Construct closed, unactivated Phase 12 audience projections.

Trusted hosts read authoritative durable truth and create one exact audience
boundary from a sealed plan and its matching durable snapshot.  This module
does not register a server, activate a CLI, write an audit sink, or create
lifecycle events.  The lower delivery codecs remain data-only and hosts hand
only detached bytes to the selected consumer.
"""

from dataclasses import dataclass
from json import dumps
from typing import Never, NoReturn, Protocol, TypeAlias, final

from avalan.patch.coordinator import _sealed_journal_steps
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    Audience,
    Capability,
    CommitStepState,
    LifecyclePhase,
    MutationState,
    PatchArtifactId,
    PatchObserverCorrelationId,
    PatchPublicCorrelationId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    coarsen_error_code,
)
from avalan.patch.durable_store import (
    DurableArtifactJournalEntry,
    DurableArtifactState,
    DurableJournal,
    DurableOutboxRecord,
    DurablePendingRecord,
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestSnapshot,
    DurableReservation,
    DurableStepBinding,
    DurableStepJournalEntry,
    DurableTerminalRecord,
    derive_artifact_state,
)
from avalan.patch.planner import MatchKind, PlannedLineage
from avalan.patch.policy import (
    ApprovalDecisionState,
    PolicyDisclosure,
    SealedPlan,
    _validate_sealed_plan,
)
from avalan.patch.projection_codec import (
    AuditRecordDelivery,
    MetricsRecordDelivery,
    ServerRecordDelivery,
    TelemetryRecordDelivery,
)
from avalan.patch.sandbox_commit import _durable_artifacts

_SCHEMA_VERSION = 1
_MAX_DELIVERY_BYTES = 1_048_576

AudiencePayloadValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["AudiencePayloadValue", ...]
    | dict[str, "AudiencePayloadValue"]
)
AudiencePayload: TypeAlias = dict[str, AudiencePayloadValue]


class AudienceProjectionError(ValueError):
    """Report an invalid audience projection without protected detail."""


class AudienceProjectionStore(Protocol):
    """Read one authenticated canonical durable snapshot."""

    async def inspect(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Return the authoritative snapshot bound to the request access."""


@final
@dataclass(frozen=True, slots=True, repr=False, init=False, eq=False)
class AudienceProjectionAccess:
    """Carry a host-issued exact store/request/correlation witness."""

    _issuer: object
    _store: AudienceProjectionStore
    _access: DurableRequestAccess
    _correlation: PatchObserverCorrelationId

    def __init__(self, token: Never) -> None:
        """Reject public construction of a trusted projection witness."""
        del token
        raise AudienceProjectionError("projection access is host-issued")

    def __repr__(self) -> str:
        """Render an opaque host-issued witness marker."""
        return "AudienceProjectionAccess(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying a host-issued projection witness."""
        raise AudienceProjectionError("projection access cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject copying a host-issued projection witness."""
        del memo
        raise AudienceProjectionError("projection access cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a host-issued projection witness."""
        raise AudienceProjectionError("projection access cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific witness serialization."""
        del protocol
        raise AudienceProjectionError("projection access cannot be serialized")


@final
class AudienceProjectionHost:
    """Issue exact projection witnesses for one authoritative durable store."""

    def __init__(self, store: AudienceProjectionStore) -> None:
        """Bind a single store with no caller-supplied result capability."""
        if not callable(getattr(store, "inspect", None)):
            raise AudienceProjectionError("projection host is invalid")
        self._store = store
        self._issuer = object()

    async def issue_access(
        self,
        access: DurableRequestAccess,
        correlation: PatchObserverCorrelationId,
    ) -> AudienceProjectionAccess:
        """Verify and issue one store-bound projection construction witness."""
        if (
            type(access) is not DurableRequestAccess
            or type(correlation) is not PatchObserverCorrelationId
        ):
            raise AudienceProjectionError("projection access is invalid")
        try:
            snapshot = await self._store.inspect(access)
        except Exception:
            raise AudienceProjectionError(
                "projection access is unavailable"
            ) from None
        _validate_access_snapshot(access, correlation, snapshot)
        return _new_projection_access(
            self._issuer, self._store, access, correlation
        )

    async def source(
        self,
        plan: SealedPlan,
        access: AudienceProjectionAccess,
    ) -> "PatchAudienceProjectionSource":
        """Read canonical store truth and bind it to one sealed plan."""
        if (
            type(plan) is not SealedPlan
            or type(access) is not AudienceProjectionAccess
            or access._issuer is not self._issuer
            or access._store is not self._store
        ):
            raise AudienceProjectionError("projection source is invalid")
        try:
            snapshot = await self._store.inspect(access._access)
        except Exception:
            raise AudienceProjectionError(
                "projection source is unavailable"
            ) from None
        _validate_access_snapshot(
            access._access, access._correlation, snapshot
        )
        _validate_source_truth(plan, snapshot)
        return _new_projection_source(plan, snapshot, self._issuer, access)


@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchAudienceProjectionSource:
    """Hold host-read canonical truth without a public snapshot constructor."""

    plan: SealedPlan
    _snapshot: DurableRequestSnapshot
    _issuer: object
    _access: AudienceProjectionAccess

    def __init__(self, token: Never) -> None:
        """Reject public source construction from caller-built truth."""
        del token
        raise AudienceProjectionError("projection source is host-issued")

    def __repr__(self) -> str:
        """Render a stable marker without request or content disclosure."""
        return "PatchAudienceProjectionSource(<redacted>)"

    def __copy__(self) -> NoReturn:
        """Reject copying a source that retains sealed canonical data."""
        raise AudienceProjectionError("projection source cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copying a source that retains sealed canonical data."""
        del memo
        raise AudienceProjectionError("projection source cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a source that retains sealed canonical data."""
        raise AudienceProjectionError("projection source cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific source serialization."""
        del protocol
        raise AudienceProjectionError("projection source cannot be serialized")


def _new_projection_access(
    issuer: object,
    store: AudienceProjectionStore,
    access: DurableRequestAccess,
    correlation: PatchObserverCorrelationId,
) -> AudienceProjectionAccess:
    """Create a module-private store-issued construction witness."""
    value = object.__new__(AudienceProjectionAccess)
    object.__setattr__(value, "_issuer", issuer)
    object.__setattr__(value, "_store", store)
    object.__setattr__(value, "_access", access)
    object.__setattr__(value, "_correlation", correlation)
    return value


def _new_projection_source(
    plan: SealedPlan,
    snapshot: DurableRequestSnapshot,
    issuer: object,
    access: AudienceProjectionAccess,
) -> PatchAudienceProjectionSource:
    """Create a module-private canonical source after store verification."""
    value = object.__new__(PatchAudienceProjectionSource)
    object.__setattr__(value, "plan", plan)
    object.__setattr__(value, "_snapshot", snapshot)
    object.__setattr__(value, "_issuer", issuer)
    object.__setattr__(value, "_access", access)
    return value


def _validate_access_snapshot(
    access: DurableRequestAccess,
    correlation: PatchObserverCorrelationId,
    snapshot: DurableRequestSnapshot,
) -> None:
    """Require store truth to match the issued request and branch witness."""
    _validate_store_snapshot_shape(snapshot)
    if (
        snapshot.reservation.request_id != access.request_id
        or snapshot.reservation.identity != access.identity
    ):
        raise AudienceProjectionError("projection store truth is invalid")
    terminal = snapshot.terminal
    pending = snapshot.pending
    if terminal is not None:
        if terminal.outbox.correlation_id != correlation:
            raise AudienceProjectionError("projection correlation is invalid")
    elif pending is None or pending.correlation_id != correlation:
        raise AudienceProjectionError("projection correlation is invalid")


def _validate_store_snapshot_shape(snapshot: object) -> None:
    """Reject malformed store truth before accessing semantic fields."""
    try:
        if type(snapshot) is not DurableRequestSnapshot:
            raise AudienceProjectionError("projection store truth is invalid")
        reservation = snapshot.reservation
        plan = snapshot.plan
        journal = snapshot.journal
        terminal = snapshot.terminal
        pending = snapshot.pending
        if (
            type(reservation) is not DurableReservation
            or (plan is not None and type(plan) is not DurablePlanReference)
            or type(journal) is not DurableJournal
            or (
                terminal is not None
                and type(terminal) is not DurableTerminalRecord
            )
            or (
                pending is not None
                and type(pending) is not DurablePendingRecord
            )
        ):
            raise AudienceProjectionError("projection store truth is invalid")
        if plan is not None and (
            type(plan.steps) is not tuple
            or any(type(item) is not DurableStepBinding for item in plan.steps)
        ):
            raise AudienceProjectionError("projection store truth is invalid")
        if (
            type(journal.steps) is not tuple
            or any(
                type(item) is not DurableStepJournalEntry
                for item in journal.steps
            )
            or type(journal.artifacts) is not tuple
            or any(
                type(item) is not DurableArtifactJournalEntry
                for item in journal.artifacts
            )
        ):
            raise AudienceProjectionError("projection store truth is invalid")
        if terminal is not None and (
            type(terminal.result) is not PatchResult
            or type(terminal.outbox) is not DurableOutboxRecord
            or pending is not None
        ):
            raise AudienceProjectionError("projection store truth is invalid")
        if terminal is None and type(pending) is not DurablePendingRecord:
            raise AudienceProjectionError("projection store truth is invalid")
        reservation.identity.__post_init__()
        reservation.__post_init__()
        if plan is not None:
            for binding in plan.steps:
                binding.__post_init__()
            plan.__post_init__()
        for entry in journal.steps:
            entry.__post_init__()
        for artifact in journal.artifacts:
            artifact.__post_init__()
        journal.__post_init__()
        if terminal is not None:
            terminal.result.__post_init__()
            terminal.outbox.__post_init__()
            terminal.__post_init__()
        if pending is not None:
            pending.__post_init__()
        snapshot.__post_init__()
    except Exception:
        raise AudienceProjectionError(
            "projection store truth is invalid"
        ) from None


def _validate_source_truth(
    plan: SealedPlan, snapshot: DurableRequestSnapshot
) -> None:
    """Require one complete sealed plan and matching durable request truth."""
    if type(plan) is not SealedPlan:
        raise AudienceProjectionError("projection source is invalid")
    try:
        _validate_store_snapshot_shape(snapshot)
        if snapshot.plan is None:
            raise AudienceProjectionError("projection source is invalid")
        _validate_sealed_plan(plan)
        snapshot.reservation.__post_init__()
        snapshot.plan.__post_init__()
        snapshot.journal.__post_init__()
        snapshot.__post_init__()
    except Exception:
        raise AudienceProjectionError("projection source is invalid") from None
    durable_plan = snapshot.plan
    request = plan.binding.request
    target = plan.binding.target
    if (
        request.request_id != snapshot.reservation.request_id
        or request.execution_id != snapshot.reservation.identity.execution_id
        or snapshot.reservation.canonical_digest != plan.binding.request_digest
        or durable_plan.plan_id != plan.plan_id
        or durable_plan.canonical_digest
        != snapshot.reservation.canonical_digest
        or durable_plan.fingerprint_digest
        != AlgorithmDigest.from_bytes(plan.fingerprint._value)
        or durable_plan.review_digest != plan.review.diff.digest
        or durable_plan.context_id != target.context_id
        or durable_plan.workspace_id != target.workspace_id
        or durable_plan.domain_id != target.domain_id
        or durable_plan.steps
        != tuple(
            DurableStepBinding(step_id, lineage_id)
            for step_id, lineage_id in _sealed_journal_steps(plan)
        )
    ):
        raise AudienceProjectionError(
            "projection source does not match durable truth"
        )
    terminal = snapshot.terminal
    pending = snapshot.pending
    if terminal is not None and (
        pending is not None
        or type(terminal.result) is not PatchResult
        or type(terminal.outbox) is not DurableOutboxRecord
        or terminal.result.request_id != request.request_id
        or terminal.result.plan_id != plan.plan_id
        or terminal.result.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
        or terminal.outbox.request_id != request.request_id
        or terminal.outbox.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
        or terminal.outbox.sequence != snapshot.event_cursor
        or terminal.result.truth.mutation_state
        is not _complete_journal_mutation_state(snapshot, durable_plan)
        or terminal.result.truth.artifact_state
        is not _complete_journal_artifact_state(plan, snapshot)
        or snapshot.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
    ):
        raise AudienceProjectionError("projection source terminal is invalid")
    if terminal is None:
        assert type(pending) is DurablePendingRecord
        if (
            snapshot.lifecycle is not LifecyclePhase.SETTLEMENT_PENDING
            or pending.request_id != request.request_id
            or pending.execution_id != request.execution_id
        ):
            raise AudienceProjectionError(
                "projection source lifecycle is invalid"
            )


def _journal_mutation_state(
    snapshot: DurableRequestSnapshot,
) -> MutationState:
    """Derive aggregate truth from the authoritative durable journal."""
    states = tuple(
        item.state
        for item in snapshot.journal.steps
        if item.state is not CommitStepState.PLANNED
    )
    if not states or all(
        item is CommitStepState.NOT_COMMITTED for item in states
    ):
        return MutationState.NOT_COMMITTED
    if any(item is CommitStepState.UNKNOWN for item in states):
        return MutationState.INDETERMINATE
    if all(item is CommitStepState.COMMITTED for item in states):
        return MutationState.COMMITTED
    return MutationState.PARTIALLY_COMMITTED


def _complete_journal_mutation_state(
    snapshot: DurableRequestSnapshot,
    plan: DurablePlanReference,
) -> MutationState:
    """Derive terminal truth only from complete exact durable step history."""
    if (
        type(snapshot.journal) is not DurableJournal
        or type(plan) is not DurablePlanReference
    ):
        raise AudienceProjectionError("projection journal is invalid")
    expected = {item.step_id: item.lineage_id for item in plan.steps}
    states: dict[PatchStepId, CommitStepState] = {}
    entries = snapshot.journal.steps
    if len(entries) < len(expected) * 2 or tuple(
        item.cursor.revision.value for item in entries
    ) != tuple(sorted(item.cursor.revision.value for item in entries)):
        raise AudienceProjectionError("projection journal is incomplete")
    for entry in entries:
        if (
            type(entry) is not DurableStepJournalEntry
            or entry.cursor.request_id != snapshot.reservation.request_id
            or expected.get(entry.step_id) != entry.lineage_id
        ):
            raise AudienceProjectionError("projection journal is invalid")
        previous = states.get(entry.step_id)
        if previous is None:
            if entry.state is not CommitStepState.PLANNED:
                raise AudienceProjectionError(
                    "projection journal is incomplete"
                )
        elif previous is CommitStepState.PLANNED:
            if entry.state not in {
                CommitStepState.COMMITTED,
                CommitStepState.NOT_COMMITTED,
                CommitStepState.UNKNOWN,
            }:
                raise AudienceProjectionError("projection journal is invalid")
        else:
            raise AudienceProjectionError("projection journal is invalid")
        states[entry.step_id] = entry.state
    terminal_states = tuple(states[item.step_id] for item in plan.steps)
    if any(item is CommitStepState.UNKNOWN for item in terminal_states):
        return MutationState.INDETERMINATE
    committed = sum(
        item is CommitStepState.COMMITTED for item in terminal_states
    )
    if committed == 0:
        return MutationState.NOT_COMMITTED
    if committed == len(terminal_states):
        return MutationState.COMMITTED
    return MutationState.PARTIALLY_COMMITTED


def _complete_journal_artifact_state(
    plan: SealedPlan,
    snapshot: DurableRequestSnapshot,
) -> ArtifactState:
    """Derive terminal artifact truth from the complete sealed artifact set."""
    entries = snapshot.journal.artifacts
    expected = tuple(item[1] for item in _durable_artifacts(plan))
    histories: dict[PatchArtifactId, list[DurableArtifactState]] = {}
    if len(entries) < len(expected) * 2 or tuple(
        item.cursor.revision.value for item in entries
    ) != tuple(sorted(item.cursor.revision.value for item in entries)):
        raise AudienceProjectionError("projection journal is incomplete")
    for entry in entries:
        histories.setdefault(entry.artifact_id, []).append(entry.state)
    if tuple(histories) != expected or any(
        history[0] is not DurableArtifactState.INTENDED or len(history) < 2
        for history in histories.values()
    ):
        raise AudienceProjectionError("projection journal is incomplete")
    try:
        return derive_artifact_state(entries)
    except Exception:
        raise AudienceProjectionError(
            "projection journal is invalid"
        ) from None


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _AudienceAuthority:
    """Bind one exact detached delivery to its trusted boundary."""

    _issuer: object
    correlation_id: PatchPublicCorrelationId

    def __post_init__(self) -> None:
        """Require the random audience-local correlation witness."""
        if type(self.correlation_id) is not PatchPublicCorrelationId:
            raise AudienceProjectionError("projection authority is invalid")

    def __repr__(self) -> str:
        """Render an opaque authority marker."""
        return f"{type(self).__name__}(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying an exact audience authority."""
        raise AudienceProjectionError("projection authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copying an exact audience authority."""
        del memo
        raise AudienceProjectionError("projection authority cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing an exact audience authority."""
        raise AudienceProjectionError(
            "projection authority cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific authority serialization."""
        del protocol
        raise AudienceProjectionError(
            "projection authority cannot be serialized"
        )


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AuditRecordAuthority(_AudienceAuthority):
    """Authorize only the matching authenticated audit record."""


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class MetricsRecordAuthority(_AudienceAuthority):
    """Authorize only the matching metrics record."""


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class TelemetryRecordAuthority(_AudienceAuthority):
    """Authorize only the matching telemetry record."""


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class ServerRecordAuthority(_AudienceAuthority):
    """Authorize only the matching server-ready data record."""


class _Boundary:
    """Provide copy and serialization fences for audience boundaries."""

    def __copy__(self) -> NoReturn:
        """Reject copying a trusted audience boundary."""
        raise AudienceProjectionError("projection boundary cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copying a trusted audience boundary."""
        del memo
        raise AudienceProjectionError("projection boundary cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a trusted audience boundary."""
        raise AudienceProjectionError(
            "projection boundary cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific boundary serialization."""
        del protocol
        raise AudienceProjectionError(
            "projection boundary cannot be serialized"
        )


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AuditRecordBoundary(_Boundary):
    """Release a detached authenticated audit record once."""

    _issuer: object
    _authority: AuditRecordAuthority
    _delivery: AuditRecordDelivery

    def authority(self) -> AuditRecordAuthority:
        """Return the exact audit-only authority witness."""
        return self._authority

    def project(self, authority: AuditRecordAuthority) -> AuditRecordDelivery:
        """Return the fixed detached audit record for its issuer."""
        _require_authority(authority, self._authority, self._issuer)
        return self._delivery


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class MetricsRecordBoundary(_Boundary):
    """Release a detached metrics record once."""

    _issuer: object
    _authority: MetricsRecordAuthority
    _delivery: MetricsRecordDelivery

    def authority(self) -> MetricsRecordAuthority:
        """Return the exact metrics-only authority witness."""
        return self._authority

    def project(
        self, authority: MetricsRecordAuthority
    ) -> MetricsRecordDelivery:
        """Return the fixed detached metrics record for its issuer."""
        _require_authority(authority, self._authority, self._issuer)
        return self._delivery


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class TelemetryRecordBoundary(_Boundary):
    """Release a detached telemetry record once."""

    _issuer: object
    _authority: TelemetryRecordAuthority
    _delivery: TelemetryRecordDelivery

    def authority(self) -> TelemetryRecordAuthority:
        """Return the exact telemetry-only authority witness."""
        return self._authority

    def project(
        self, authority: TelemetryRecordAuthority
    ) -> TelemetryRecordDelivery:
        """Return the fixed detached telemetry record for its issuer."""
        _require_authority(authority, self._authority, self._issuer)
        return self._delivery


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class ServerRecordBoundary(_Boundary):
    """Release a detached server-ready data record once."""

    _issuer: object
    _authority: ServerRecordAuthority
    _delivery: ServerRecordDelivery

    def authority(self) -> ServerRecordAuthority:
        """Return the exact server-only authority witness."""
        return self._authority

    def project(
        self, authority: ServerRecordAuthority
    ) -> ServerRecordDelivery:
        """Return the fixed detached server data record for its issuer."""
        _require_authority(authority, self._authority, self._issuer)
        return self._delivery


def create_audit_record_boundary(
    source: PatchAudienceProjectionSource,
) -> AuditRecordBoundary:
    """Create a detached authenticated audit projection boundary."""
    return _audit_boundary(source)


def create_metrics_record_boundary(
    source: PatchAudienceProjectionSource,
) -> MetricsRecordBoundary:
    """Create a detached metrics projection boundary."""
    return _metrics_boundary(source)


def create_telemetry_record_boundary(
    source: PatchAudienceProjectionSource,
) -> TelemetryRecordBoundary:
    """Create a detached telemetry projection boundary."""
    return _telemetry_boundary(source)


def create_server_record_boundary(
    source: PatchAudienceProjectionSource,
) -> ServerRecordBoundary:
    """Create a detached server-ready data projection boundary."""
    return _server_boundary(source)


def _audit_boundary(
    source: PatchAudienceProjectionSource,
) -> AuditRecordBoundary:
    """Derive the one authenticated audit delivery without raw content."""
    correlation = PatchPublicCorrelationId.new()
    issuer = object()
    authority = AuditRecordAuthority(issuer, correlation)
    return AuditRecordBoundary(
        issuer,
        authority,
        AuditRecordDelivery(
            _delivery("audit", correlation, _audit_body(source))
        ),
    )


def _metrics_boundary(
    source: PatchAudienceProjectionSource,
) -> MetricsRecordBoundary:
    """Derive the one content-free metrics delivery."""
    correlation = PatchPublicCorrelationId.new()
    issuer = object()
    authority = MetricsRecordAuthority(issuer, correlation)
    return MetricsRecordBoundary(
        issuer,
        authority,
        MetricsRecordDelivery(
            _delivery("metrics", correlation, _metrics_body(source))
        ),
    )


def _telemetry_boundary(
    source: PatchAudienceProjectionSource,
) -> TelemetryRecordBoundary:
    """Derive the one content-free telemetry delivery."""
    correlation = PatchPublicCorrelationId.new()
    issuer = object()
    authority = TelemetryRecordAuthority(issuer, correlation)
    return TelemetryRecordBoundary(
        issuer,
        authority,
        TelemetryRecordDelivery(
            _delivery("telemetry", correlation, _telemetry_body(source))
        ),
    )


def _server_boundary(
    source: PatchAudienceProjectionSource,
) -> ServerRecordBoundary:
    """Derive the one unactivated server-ready data delivery."""
    correlation = PatchPublicCorrelationId.new()
    issuer = object()
    authority = ServerRecordAuthority(issuer, correlation)
    return ServerRecordBoundary(
        issuer,
        authority,
        ServerRecordDelivery(
            _delivery("server", correlation, _server_body(source))
        ),
    )


def _require_authority(
    value: _AudienceAuthority,
    expected: _AudienceAuthority,
    issuer: object,
) -> None:
    """Require the exact type, instance, issuer, and correlation witness."""
    if (
        type(value) is not type(expected)
        or value is not expected
        or value._issuer is not issuer
        or value.correlation_id is not expected.correlation_id
    ):
        raise AudienceProjectionError(
            "projection authority is not issued here"
        )


def _delivery(
    audience: str,
    correlation: PatchPublicCorrelationId,
    payload: AudiencePayload,
) -> bytes:
    """Encode a bounded detached delivery without integrity fingerprints."""
    value = dumps(
        {
            "schema_version": _SCHEMA_VERSION,
            "audience": audience,
            "correlation_id": correlation.value,
            "payload": payload,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(value) > _MAX_DELIVERY_BYTES:
        raise AudienceProjectionError("projection delivery exceeds its bound")
    return value


def _audit_body(source: PatchAudienceProjectionSource) -> AudiencePayload:
    """Return authenticated categories and authorized audit truth."""
    binding = source.plan.binding
    subject = binding.subject
    exact = _exact_truth_authorized(source, PolicyDisclosure.AUDIT_EXACT_TRUTH)
    body = _truth_body(
        source,
        audience=Audience.AUDIT,
        exact=exact,
        include_coarse_error=False,
        include_lineages=exact
        or PolicyDisclosure.AUDIT_PATHS in binding.final.disclosures,
    )
    body["authenticated"] = {
        "tenant": subject.tenant.value,
        "principal": subject.principal.value,
        "run": subject.run.value,
        "session": subject.session.value,
        "task": subject.task.value,
        "agent": subject.agent.value,
    }
    body["context"] = {
        "kind": binding.context_kind.value,
        "context_id": binding.target.context_id.value,
        "workspace_id": binding.target.workspace_id.value,
        "policy_revision": binding.final.revision.value,
    }
    if PolicyDisclosure.AUDIT_PATHS in binding.final.disclosures:
        body["lineages"] = _lineages(
            source,
            include_paths=True,
            include_truth=exact,
        )
        body["paths_redacted"] = False
    else:
        body["paths_redacted"] = True
    return body


def _metrics_body(source: PatchAudienceProjectionSource) -> AudiencePayload:
    """Return aggregate metrics without correlation, path, size, or timing."""
    exact = _exact_truth_authorized(
        source, PolicyDisclosure.METRICS_EXACT_TRUTH
    )
    body = _truth_body(
        source,
        audience=Audience.PUBLIC,
        exact=exact,
        include_coarse_error=True,
        include_lineages=exact
        and PolicyDisclosure.EVENT_METRICS
        in source.plan.binding.final.disclosures,
    )
    body["metric_categories_authorized"] = (
        PolicyDisclosure.EVENT_METRICS in source.plan.binding.final.disclosures
    )
    return body


def _telemetry_body(source: PatchAudienceProjectionSource) -> AudiencePayload:
    """Return coarse completion telemetry without authenticated identifiers."""
    exact = _exact_truth_authorized(
        source, PolicyDisclosure.TELEMETRY_EXACT_TRUTH
    )
    body = _truth_body(
        source,
        audience=Audience.PUBLIC,
        exact=exact,
        include_coarse_error=True,
        include_lineages=exact
        and PolicyDisclosure.EVENT_METRICS
        in source.plan.binding.final.disclosures,
    )
    body["event"] = (
        "request_completed"
        if source._snapshot.terminal
        else "settlement_pending"
    )
    return body


def _server_body(source: PatchAudienceProjectionSource) -> AudiencePayload:
    """Return server-ready structured data without registering a server."""
    exact = _exact_truth_authorized(
        source, PolicyDisclosure.SERVER_EXACT_TRUTH
    )
    body = _truth_body(
        source,
        audience=Audience.PUBLIC,
        exact=exact,
        include_coarse_error=True,
        include_lineages=exact
        and PolicyDisclosure.EVENT_METRICS
        in source.plan.binding.final.disclosures,
    )
    body["server_activation"] = "absent"
    return body


def _truth_body(
    source: PatchAudienceProjectionSource,
    *,
    audience: Audience,
    exact: bool,
    include_coarse_error: bool,
    include_lineages: bool,
) -> AudiencePayload:
    """Return audience-authorized durable truth without content facts."""
    plan = source.plan
    result = (
        None
        if source._snapshot.terminal is None
        else source._snapshot.terminal.result
    )
    body: AudiencePayload = {
        "tool": "patch." + plan.binding.request.operation.value,
        "operation_classes": _operation_classes(plan.candidate.lineages),
        "diagnostic_association": {
            "supported": (
                plan.binding.diagnostic_policy is not None
                and PolicyDisclosure.DIAGNOSTIC_ASSOCIATION
                in plan.binding.final.disclosures
            ),
            "executed": False,
        },
    }
    if result is None:
        body.update(
            {
                "terminal": False,
                "lifecycle": LifecyclePhase.SETTLEMENT_PENDING.value,
                "outcome": "pending",
            }
        )
    else:
        body.update(
            {
                "terminal": True,
                "lifecycle": LifecyclePhase.REQUEST_COMPLETED.value,
                "outcome": "settled",
            }
        )
        if include_coarse_error:
            body["error_category"] = (
                None
                if result.diagnostic is None
                else coarsen_error_code(result.diagnostic.code, audience).value
            )
    if exact:
        body.update(
            {
                "matching_exact": _matching_exact(plan.candidate.lineages),
                "approval": _approval(source),
                "warning_categories": tuple(
                    item.value.value for item in plan.review.warnings
                ),
                "cancellation_requested": (
                    source._snapshot.cancellation_requested
                ),
            }
        )
        if result is not None:
            body.update(
                {
                    "status": result.status.value,
                    "mutation_state": result.truth.mutation_state.value,
                    "requested_effect_occurred": (
                        result.truth.requested_effect_occurred.value
                    ),
                    "artifact_state": result.truth.artifact_state.value,
                    "workspace_change": result.truth.workspace_change.value,
                    "commit_set_exact": result.truth.commit_set_exact,
                    "postcondition": result.truth.postcondition.value,
                    "error_code": (
                        None
                        if result.diagnostic is None
                        else result.diagnostic.code.value
                    ),
                    "error_stage": (
                        None
                        if result.diagnostic is None
                        else result.diagnostic.stage.value
                    ),
                }
            )
    if include_lineages:
        body["lineages"] = _lineages(
            source,
            include_paths=False,
            include_truth=exact,
        )
    return body


def _exact_truth_authorized(
    source: PatchAudienceProjectionSource,
    disclosure: PolicyDisclosure,
) -> bool:
    """Return whether one sealed audience policy allows exact truth."""
    if type(disclosure) is not PolicyDisclosure:
        raise AudienceProjectionError("projection disclosure is invalid")
    return disclosure in source.plan.binding.final.disclosures


def _approval(source: PatchAudienceProjectionSource) -> AudiencePayload:
    """Return closed approval categories without grant or reviewer material."""
    mode = source.plan.binding.final.approval.mode
    result = (
        None
        if source._snapshot.terminal is None
        else source._snapshot.terminal.result
    )
    if mode is ApprovalMode.PREAUTHORIZED:
        outcome = "preauthorized"
    elif mode is ApprovalMode.DENY:
        outcome = "not_available"
    elif result is None:
        outcome = "pending"
    elif result.status is PatchStatus.APPROVAL_DENIED:
        outcome = ApprovalDecisionState.DENIED.value
    elif result.status is PatchStatus.APPROVAL_UNAVAILABLE:
        outcome = ApprovalDecisionState.UNAVAILABLE.value
    elif result.status is PatchStatus.CANCELLED:
        outcome = ApprovalDecisionState.CANCELLED.value
    elif result.status in {
        PatchStatus.COMMITTED,
        PatchStatus.PARTIAL,
        PatchStatus.INDETERMINATE,
    }:
        outcome = ApprovalDecisionState.APPROVED.value
    else:
        outcome = "not_observed"
    return {
        "required": mode is ApprovalMode.REQUIRE_REVIEW,
        "outcome": outcome,
    }


def _lineages(
    source: PatchAudienceProjectionSource,
    *,
    include_paths: bool,
    include_truth: bool,
) -> tuple[AudiencePayload, ...]:
    """Return every authorized lineage once without content facts."""
    states = _lineage_states(source)
    result = (
        None
        if source._snapshot.terminal is None
        else source._snapshot.terminal.result
    )
    values: list[AudiencePayload] = []
    for lineage in source.plan.candidate.lineages:
        item: AudiencePayload = {
            "lineage_id": lineage.lineage_id.value,
            "operation_classes": _lineage_operation_classes(lineage),
        }
        if include_truth:
            item["matching_exact"] = _lineage_matching_exact(lineage)
            item["mutation_state"] = states[lineage.lineage_id.value]
            if result is not None:
                item["aggregate_postcondition"] = (
                    result.truth.postcondition.value
                )
        if include_paths:
            item["source_path"] = (
                None
                if lineage.source_path is None
                else lineage.source_path.value
            )
            item["destination_path"] = (
                None
                if lineage.destination_path is None
                else lineage.destination_path.value
            )
        values.append(item)
    return tuple(values)


def _lineage_states(source: PatchAudienceProjectionSource) -> dict[str, str]:
    """Derive each lineage's requested-effect truth from durable steps."""
    states_by_lineage: dict[str, list[CommitStepState]] = {
        item.lineage_id.value: [] for item in source.plan.candidate.lineages
    }
    for item in source._snapshot.journal.steps:
        if item.state is not CommitStepState.PLANNED:
            states_by_lineage[item.lineage_id.value].append(item.state)
    result: dict[str, str] = {}
    for lineage_id, states in states_by_lineage.items():
        if not states or all(
            item is CommitStepState.NOT_COMMITTED for item in states
        ):
            result[lineage_id] = MutationState.NOT_COMMITTED.value
        elif any(item is CommitStepState.UNKNOWN for item in states):
            result[lineage_id] = MutationState.INDETERMINATE.value
        elif all(item is CommitStepState.COMMITTED for item in states):
            result[lineage_id] = MutationState.COMMITTED.value
        else:
            result[lineage_id] = MutationState.PARTIALLY_COMMITTED.value
    return result


def _matching_exact(lineages: tuple[PlannedLineage, ...]) -> bool | None:
    """Return aggregate matching exactness without conflating commit truth."""
    values = tuple(_lineage_matching_exact(item) for item in lineages)
    matching = tuple(item for item in values if item is not None)
    if not matching:
        return None
    return all(matching)


def _lineage_matching_exact(lineage: PlannedLineage) -> bool | None:
    """Return one lineage's matching exactness or no-match-applicability."""
    if not lineage.matches:
        return None
    return all(item.kind is MatchKind.EXACT_TEXT for item in lineage.matches)


def _operation_classes(
    lineages: tuple[PlannedLineage, ...],
) -> tuple[str, ...]:
    """Return sorted requested-effect categories without paths or sizes."""
    return tuple(
        sorted(
            {
                item
                for lineage in lineages
                for item in _lineage_operation_classes(lineage)
            }
        )
    )


def _lineage_operation_classes(lineage: PlannedLineage) -> tuple[str, ...]:
    """Return one lineage's closed mutation categories."""
    return tuple(
        sorted(
            {
                _OPERATION_CLASS_BY_CAPABILITY[capability]
                for capability in lineage.capabilities
            }
        )
    )


_OPERATION_CLASS_BY_CAPABILITY = {
    Capability.CREATE: "create",
    Capability.UPDATE: "update",
    Capability.UPDATE_EXECUTABLE: "update",
    Capability.DELETE: "delete",
    Capability.MOVE: "move",
}
