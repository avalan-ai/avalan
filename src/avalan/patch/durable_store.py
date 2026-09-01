"""Define durable patch-request storage semantics without a database driver.

This module deliberately contains the semantic contract and deterministic
in-memory reference only.  It does not activate patch mutation, start a
worker, open a database connection, or expose a transport route.
"""

from asyncio import Event, Lock
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Protocol, runtime_checkable

from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    Audience,
    ByteSize,
    CommitStepState,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    LogicalPath,
    MutationState,
    PatchApprovalId,
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchContextId,
    PatchDomainError,
    PatchDomainId,
    PatchEventId,
    PatchExecutionId,
    PatchGrantId,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStepId,
    PatchWorkspaceId,
    SequenceNumber,
)
from avalan.patch.policy import (
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    PolicyBrokerId,
    PolicyReviewerRole,
    PolicyRouteId,
)


class DurableStoreErrorCode(str, Enum):
    """Name closed durable-store failures without protected detail."""

    ACCESS_DENIED = "patch.durable_access_denied"
    APPROVAL_CONSUMED = "patch.durable_approval_consumed"
    APPROVAL_MISMATCH = "patch.durable_approval_mismatch"
    APPROVAL_EXPIRED = "patch.durable_approval_expired"
    FENCED = "patch.durable_fenced"
    IDEMPOTENCY_CONFLICT = "patch.durable_idempotency_conflict"
    INVALID_RESERVATION = "patch.durable_invalid_reservation"
    JOURNAL_CONFLICT = "patch.durable_journal_conflict"
    JOURNAL_INCOMPLETE = "patch.durable_journal_incomplete"
    LEASE_EXPIRED = "patch.durable_lease_expired"
    LIFECYCLE_CONFLICT = "patch.durable_lifecycle_conflict"
    PLAN_MISMATCH = "patch.durable_plan_mismatch"
    RETENTION_CONFLICT = "patch.durable_retention_conflict"
    RETENTION_DENIED = "patch.durable_retention_denied"
    RETENTION_LIMIT = "patch.durable_retention_limit"
    TERMINAL_CONFLICT = "patch.durable_terminal_conflict"


class DurableStoreError(PatchDomainError):
    """Report one stable durable-store semantic failure."""

    def __init__(self, code: DurableStoreErrorCode) -> None:
        """Initialize the closed failure code."""
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class DurableRequestIdentity:
    """Bind the authenticated retransmission tuple before target inspection."""

    tenant_id: PatchTenantId
    principal_id: PatchPrincipalId
    execution_id: PatchExecutionId
    route_id: PolicyRouteId
    retransmission_key: RetransmissionKey

    def __post_init__(self) -> None:
        """Require exact trusted identity components."""
        if (
            type(self.tenant_id) is not PatchTenantId
            or type(self.principal_id) is not PatchPrincipalId
            or type(self.execution_id) is not PatchExecutionId
            or type(self.route_id) is not PolicyRouteId
            or type(self.retransmission_key) is not RetransmissionKey
        ):
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)


@dataclass(frozen=True, slots=True)
class DurableReservation:
    """Return one immutable durable request identity reservation."""

    request_id: PatchRequestId
    identity: DurableRequestIdentity
    canonical_digest: AlgorithmDigest
    replayed: bool

    def __post_init__(self) -> None:
        """Require exact identity and digest witnesses."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.identity) is not DurableRequestIdentity
            or type(self.canonical_digest) is not AlgorithmDigest
            or type(self.replayed) is not bool
        ):
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)


@dataclass(frozen=True, slots=True)
class DurableCoordinationAccess:
    """Bind workspace coordination reads to full originating authority."""

    reservation: DurableReservation
    run_id: PatchRunId
    session_id: PatchSessionId
    task_id: PatchTaskId
    agent_id: PatchAgentId
    context_id: PatchContextId
    workspace_id: PatchWorkspaceId
    domain_id: PatchDomainId

    def __post_init__(self) -> None:
        """Require exact immutable workspace authority components."""
        if (
            type(self.reservation) is not DurableReservation
            or type(self.run_id) is not PatchRunId
            or type(self.session_id) is not PatchSessionId
            or type(self.task_id) is not PatchTaskId
            or type(self.agent_id) is not PatchAgentId
            or type(self.context_id) is not PatchContextId
            or type(self.workspace_id) is not PatchWorkspaceId
            or type(self.domain_id) is not PatchDomainId
        ):
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)


@dataclass(frozen=True, slots=True)
class DurableProtocolOrigin:
    """Bind a durable plan to every originating protocol authority fact."""

    tenant_id: PatchTenantId
    principal_id: PatchPrincipalId
    execution_id: PatchExecutionId
    run_id: PatchRunId
    session_id: PatchSessionId
    task_id: PatchTaskId
    agent_id: PatchAgentId
    route_id: PolicyRouteId
    context_id: PatchContextId
    workspace_id: PatchWorkspaceId

    def __post_init__(self) -> None:
        """Require every exact originating identity component."""
        if (
            type(self.tenant_id) is not PatchTenantId
            or type(self.principal_id) is not PatchPrincipalId
            or type(self.execution_id) is not PatchExecutionId
            or type(self.run_id) is not PatchRunId
            or type(self.session_id) is not PatchSessionId
            or type(self.task_id) is not PatchTaskId
            or type(self.agent_id) is not PatchAgentId
            or type(self.route_id) is not PolicyRouteId
            or type(self.context_id) is not PatchContextId
            or type(self.workspace_id) is not PatchWorkspaceId
        ):
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)

    def matches(self, identity: DurableRequestIdentity) -> bool:
        """Return whether a request tuple is the exact persisted origin."""
        if type(identity) is not DurableRequestIdentity:
            return False
        return (
            self.tenant_id == identity.tenant_id
            and self.principal_id == identity.principal_id
            and self.execution_id == identity.execution_id
            and self.route_id == identity.route_id
        )


@dataclass(frozen=True, slots=True)
class DurableCoordinationAdmission:
    """Bind one workspace mutation to an immutable path footprint."""

    access: DurableCoordinationAccess
    paths: frozenset[LogicalPath]

    def __post_init__(self) -> None:
        """Require exact immutable authority and a nonempty path footprint."""
        if (
            type(self.access) is not DurableCoordinationAccess
            or type(self.paths) is not frozenset
            or not self.paths
            or any(type(path) is not LogicalPath for path in self.paths)
        ):
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)


@dataclass(frozen=True, slots=True)
class DurableStepBinding:
    """Bind one sealed requested-effect step to its lineage."""

    step_id: PatchStepId
    lineage_id: PatchLineageId

    def __post_init__(self) -> None:
        """Require exact typed requested-effect identities."""
        if (
            type(self.step_id) is not PatchStepId
            or type(self.lineage_id) is not PatchLineageId
        ):
            raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)


@dataclass(frozen=True, slots=True)
class DurablePlanReference:
    """Store sealed-plan evidence and opaque restart material."""

    plan_id: PatchPlanId
    canonical_digest: AlgorithmDigest
    fingerprint_digest: AlgorithmDigest
    review_digest: AlgorithmDigest
    context_id: PatchContextId
    workspace_id: PatchWorkspaceId
    domain_id: PatchDomainId
    steps: tuple[DurableStepBinding, ...]
    origin: DurableProtocolOrigin | None = None
    rehydration: bytes = field(repr=False, default=b"")

    def __post_init__(self) -> None:
        """Require a bounded unique sealed requested-effect graph."""
        if (
            type(self.plan_id) is not PatchPlanId
            or type(self.canonical_digest) is not AlgorithmDigest
            or type(self.fingerprint_digest) is not AlgorithmDigest
            or type(self.review_digest) is not AlgorithmDigest
            or type(self.context_id) is not PatchContextId
            or type(self.workspace_id) is not PatchWorkspaceId
            or type(self.domain_id) is not PatchDomainId
            or type(self.steps) is not tuple
            or not self.steps
            or any(type(item) is not DurableStepBinding for item in self.steps)
            or (
                self.origin is not None
                and type(self.origin) is not DurableProtocolOrigin
            )
            or type(self.rehydration) is not bytes
            or len(self.rehydration) > 1_048_576
            or len({item.step_id for item in self.steps}) != len(self.steps)
            or len(self.steps) > 4096
        ):
            raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)


@dataclass(frozen=True, slots=True)
class DurableApproval:
    """Store a broker-attested approval binding consumed at commit claim."""

    grant_id: PatchGrantId
    approval_id: PatchApprovalId
    identity: DurableRequestIdentity
    canonical_digest: AlgorithmDigest
    plan_id: PatchPlanId
    fingerprint_digest: AlgorithmDigest
    review_digest: AlgorithmDigest
    context_id: PatchContextId
    workspace_id: PatchWorkspaceId
    domain_id: PatchDomainId
    policy_revision: str
    broker_id: PolicyBrokerId
    reviewer_role: PolicyReviewerRole
    reviewers: tuple[PatchPrincipalId, ...]
    expires_at: ExpiryTick
    attestation: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Require every approval component to remain exact and typed."""
        if (
            type(self.grant_id) is not PatchGrantId
            or type(self.approval_id) is not PatchApprovalId
            or type(self.identity) is not DurableRequestIdentity
            or type(self.canonical_digest) is not AlgorithmDigest
            or type(self.plan_id) is not PatchPlanId
            or type(self.fingerprint_digest) is not AlgorithmDigest
            or type(self.review_digest) is not AlgorithmDigest
            or type(self.context_id) is not PatchContextId
            or type(self.workspace_id) is not PatchWorkspaceId
            or type(self.domain_id) is not PatchDomainId
            or type(self.policy_revision) is not str
            or not self.policy_revision
            or type(self.broker_id) is not PolicyBrokerId
            or type(self.reviewer_role) is not PolicyReviewerRole
            or type(self.reviewers) is not tuple
            or not self.reviewers
            or any(
                type(item) is not PatchPrincipalId for item in self.reviewers
            )
            or len(set(self.reviewers)) != len(self.reviewers)
            or type(self.expires_at) is not ExpiryTick
            or type(self.attestation) is not bytes
            or len(self.attestation) != 32
        ):
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)


class DurableApprovalVerifier(Protocol):
    """Validate a broker-issued opaque durable approval attestation."""

    def verify(self, approval: DurableApproval) -> None:
        """Reject an approval not issued by the configured broker authority."""


class DenyDurableApprovalVerifier:
    """Fail closed until an authenticated broker verifier is configured."""

    def verify(self, approval: DurableApproval) -> None:
        """Reject every approval without exposing attestation material."""
        if type(approval) is not DurableApproval:
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
        raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)


@dataclass(frozen=True, slots=True)
class DurableCommitLease:
    """Witness one bounded fenced commit-owner epoch."""

    request_id: PatchRequestId
    domain_id: PatchDomainId
    owner_id: PatchCommitOwnerId
    fence: SequenceNumber
    expires_at: ExpiryTick

    def __post_init__(self) -> None:
        """Require a positive nonzero fence and exact typed ownership facts."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.domain_id) is not PatchDomainId
            or type(self.owner_id) is not PatchCommitOwnerId
            or type(self.fence) is not SequenceNumber
            or self.fence.value == 0
            or type(self.expires_at) is not ExpiryTick
        ):
            raise DurableStoreError(DurableStoreErrorCode.FENCED)


@dataclass(frozen=True, slots=True)
class DurableWorkerBinding:
    """Bind one write-capable child lifetime to its durable owner epoch."""

    session_id: str
    channel_id: str
    implementation_id: str
    implementation_digest: AlgorithmDigest
    root_digest: AlgorithmDigest

    def __post_init__(self) -> None:
        """Require complete opaque worker and root identities."""
        if (
            not self.session_id
            or not self.channel_id
            or not self.implementation_id
            or type(self.implementation_digest) is not AlgorithmDigest
            or type(self.root_digest) is not AlgorithmDigest
            or max(
                len(self.session_id),
                len(self.channel_id),
                len(self.implementation_id),
            )
            > 256
        ):
            raise DurableStoreError(DurableStoreErrorCode.FENCED)

    def fingerprint(self) -> str:
        """Return a canonical non-secret identity for durable comparison."""
        return sha256(
            "\x00".join(
                (
                    self.session_id,
                    self.channel_id,
                    self.implementation_id,
                    self.implementation_digest.value,
                    self.root_digest.value,
                )
            ).encode()
        ).hexdigest()


class DurableCommitClaimState(str, Enum):
    """Name whether a caller owns, attaches to, or replays a request."""

    OWNER = "owner"
    ATTACHED = "attached"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class DurableCommitClaim:
    """Return the exact atomic approval-consumption ownership outcome."""

    state: DurableCommitClaimState
    lease: DurableCommitLease | None
    terminal: "DurableTerminalRecord | None"

    def __post_init__(self) -> None:
        """Keep write authority exclusive to the sole newly assigned owner."""
        if (
            type(self.state) is not DurableCommitClaimState
            or (
                self.lease is not None
                and type(self.lease) is not DurableCommitLease
            )
            or (
                self.terminal is not None
                and type(self.terminal) is not DurableTerminalRecord
            )
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if self.state is DurableCommitClaimState.OWNER and (
            self.lease is None or self.terminal is not None
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if self.state is DurableCommitClaimState.ATTACHED and (
            self.lease is not None or self.terminal is not None
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if self.state is DurableCommitClaimState.TERMINAL and (
            self.lease is not None or self.terminal is None
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableJournalCursor:
    """Bind one compare-and-set journal revision to a request."""

    request_id: PatchRequestId
    revision: SequenceNumber

    def __post_init__(self) -> None:
        """Require exact request and monotonic revision types."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.revision) is not SequenceNumber
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableStepJournalEntry:
    """Record one persisted requested-effect state transition."""

    cursor: DurableJournalCursor
    step_id: PatchStepId
    lineage_id: PatchLineageId
    state: CommitStepState

    def __post_init__(self) -> None:
        """Require one exact revision and requested-effect binding."""
        if (
            type(self.cursor) is not DurableJournalCursor
            or type(self.step_id) is not PatchStepId
            or type(self.lineage_id) is not PatchLineageId
            or type(self.state) is not CommitStepState
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


class DurableArtifactState(str, Enum):
    """Name target-owned artifact truth separately from requested effects."""

    INTENDED = "intended"
    NOT_CREATED = "not_created"
    PRESENT = "present"
    REMOVED = "removed"
    LEAKED = "leaked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DurableArtifactJournalEntry:
    """Record one persisted staging-artifact state transition."""

    cursor: DurableJournalCursor
    artifact_id: PatchArtifactId
    state: DurableArtifactState

    def __post_init__(self) -> None:
        """Require one exact revision and target-owned artifact identity."""
        if (
            type(self.cursor) is not DurableJournalCursor
            or type(self.artifact_id) is not PatchArtifactId
            or type(self.state) is not DurableArtifactState
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableJournal:
    """Store immutable monotonic requested-effect and artifact histories."""

    cursor: DurableJournalCursor
    steps: tuple[DurableStepJournalEntry, ...]
    artifacts: tuple[DurableArtifactJournalEntry, ...]

    def __post_init__(self) -> None:
        """Require a unique globally ordered sequence of journal revisions."""
        revisions = tuple(
            item.cursor.revision.value for item in self.steps
        ) + tuple(item.cursor.revision.value for item in self.artifacts)
        if (
            type(self.cursor) is not DurableJournalCursor
            or type(self.steps) is not tuple
            or type(self.artifacts) is not tuple
            or any(
                type(item) is not DurableStepJournalEntry
                for item in self.steps
            )
            or any(
                type(item) is not DurableArtifactJournalEntry
                for item in self.artifacts
            )
            or any(
                item.cursor.request_id != self.cursor.request_id
                for item in self.steps
            )
            or any(
                item.cursor.request_id != self.cursor.request_id
                for item in self.artifacts
            )
            or len(revisions) != len(set(revisions))
            or set(revisions) != set(range(1, self.cursor.revision.value + 1))
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurablePendingRequest:
    """Describe one host-resumable pending operation before store fencing."""

    pending_operation_id: PatchPendingOperationId
    correlation_id: PatchObserverCorrelationId
    next_check_after: DurationTicks

    def __post_init__(self) -> None:
        """Require opaque branch correlation and coarse positive guidance."""
        if (
            type(self.pending_operation_id) is not PatchPendingOperationId
            or type(self.correlation_id) is not PatchObserverCorrelationId
            or type(self.next_check_after) is not DurationTicks
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurablePendingRecord:
    """Store branch, cursor, cancellation, and fence pending facts."""

    request_id: PatchRequestId
    execution_id: PatchExecutionId
    pending_operation_id: PatchPendingOperationId
    correlation_id: PatchObserverCorrelationId
    fence: SequenceNumber
    event_cursor: SequenceNumber
    cancellation_requested: bool
    next_check_after: DurationTicks

    def __post_init__(self) -> None:
        """Require complete bounded pending-resume evidence."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.execution_id) is not PatchExecutionId
            or type(self.pending_operation_id) is not PatchPendingOperationId
            or type(self.correlation_id) is not PatchObserverCorrelationId
            or type(self.fence) is not SequenceNumber
            or self.fence.value == 0
            or type(self.event_cursor) is not SequenceNumber
            or type(self.cancellation_requested) is not bool
            or type(self.next_check_after) is not DurationTicks
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableRequestAccess:
    """Bind authenticated inspection authority to one request identity."""

    request_id: PatchRequestId
    identity: DurableRequestIdentity

    def __post_init__(self) -> None:
        """Require an exact non-bearer request identity binding."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.identity) is not DurableRequestIdentity
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)


@dataclass(frozen=True, slots=True)
class DurablePendingAccess:
    """Bind pending authority to original branch metadata."""

    request: DurableRequestAccess
    pending_operation_id: PatchPendingOperationId
    correlation_id: PatchObserverCorrelationId

    def __post_init__(self) -> None:
        """Require a complete exact pending continuation binding."""
        if (
            type(self.request) is not DurableRequestAccess
            or type(self.pending_operation_id) is not PatchPendingOperationId
            or type(self.correlation_id) is not PatchObserverCorrelationId
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)


@dataclass(frozen=True, slots=True)
class DurableOutboxRecord:
    """Store one content-free at-least-once lifecycle delivery record."""

    event_id: PatchEventId
    request_id: PatchRequestId
    sequence: SequenceNumber
    lifecycle: LifecyclePhase
    correlation_id: PatchObserverCorrelationId

    def __post_init__(self) -> None:
        """Require a stable request-scoped monotonic event identity."""
        if (
            type(self.event_id) is not PatchEventId
            or type(self.request_id) is not PatchRequestId
            or type(self.sequence) is not SequenceNumber
            or self.sequence.value == 0
            or type(self.lifecycle) is not LifecyclePhase
            or type(self.correlation_id) is not PatchObserverCorrelationId
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableTerminalRecord:
    """Store one immutable result and matching terminal outbox record."""

    result: PatchResult
    outbox: DurableOutboxRecord
    pending_operation_id: PatchPendingOperationId | None

    def __post_init__(self) -> None:
        """Require the result and outbox to name one terminal request."""
        if (
            type(self.result) is not PatchResult
            or type(self.outbox) is not DurableOutboxRecord
            or (
                self.pending_operation_id is not None
                and type(self.pending_operation_id)
                is not PatchPendingOperationId
            )
            or self.result.request_id != self.outbox.request_id
            or self.outbox.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
        ):
            raise DurableStoreError(DurableStoreErrorCode.TERMINAL_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableRequestSnapshot:
    """Return one immutable content-free durable request recovery snapshot."""

    reservation: DurableReservation
    plan: DurablePlanReference | None
    lifecycle: LifecyclePhase
    lease: DurableCommitLease | None
    journal: DurableJournal
    pending: DurablePendingRecord | None
    terminal: DurableTerminalRecord | None
    worker_bound: bool
    worker_reaped: bool
    cancellation_requested: bool
    event_cursor: SequenceNumber

    def __post_init__(self) -> None:
        """Keep snapshot fields aligned with terminal lifecycle."""
        if (
            type(self.reservation) is not DurableReservation
            or (
                self.plan is not None
                and type(self.plan) is not DurablePlanReference
            )
            or type(self.lifecycle) is not LifecyclePhase
            or (
                self.lease is not None
                and type(self.lease) is not DurableCommitLease
            )
            or type(self.journal) is not DurableJournal
            or (
                self.pending is not None
                and type(self.pending) is not DurablePendingRecord
            )
            or (
                self.terminal is not None
                and type(self.terminal) is not DurableTerminalRecord
            )
            or type(self.worker_bound) is not bool
            or type(self.worker_reaped) is not bool
            or type(self.cancellation_requested) is not bool
            or type(self.event_cursor) is not SequenceNumber
            or self.journal.cursor.request_id != self.reservation.request_id
            or (
                self.lifecycle is LifecyclePhase.REQUEST_COMPLETED
                and self.terminal is None
            )
            or (
                self.lifecycle is not LifecyclePhase.REQUEST_COMPLETED
                and self.terminal is not None
            )
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


class DurableRetentionKind(str, Enum):
    """Name content-bearing encrypted values permitted for retention."""

    SEALED_PLAN = "sealed_plan"
    REVIEW_ARTIFACT = "review_artifact"
    PRIVATE_STAGING = "private_staging"
    CLI_REVIEW = "cli_review"
    AUDIT_PROJECTION = "audit_projection"
    METRICS_PROJECTION = "metrics_projection"
    TELEMETRY_PROJECTION = "telemetry_projection"
    SERVER_READY_PROJECTION = "server_ready_projection"
    DIAGNOSTIC_ASSOCIATION = "diagnostic_association"


@dataclass(frozen=True, slots=True, repr=False)
class EncryptedRetentionValue:
    """Store opaque encrypted bytes without a plaintext accessor."""

    _ciphertext: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Require an immutable bounded encrypted byte value."""
        if (
            type(self._ciphertext) is not bytes
            or len(self._ciphertext) > 1_048_576
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)

    def __repr__(self) -> str:
        """Render a stable redaction marker without ciphertext."""
        return "EncryptedRetentionValue(<redacted>)"

    def __str__(self) -> str:
        """Return a stable redaction marker without ciphertext."""
        return "<redacted>"

    def digest(self) -> AlgorithmDigest:
        """Return ciphertext integrity evidence without exposing bytes."""
        return AlgorithmDigest("sha256", sha256(self._ciphertext).hexdigest())

    def size(self) -> ByteSize:
        """Return ciphertext length without exposing bytes."""
        return ByteSize(len(self._ciphertext))


@dataclass(frozen=True, slots=True)
class DurableRetentionPolicy:
    """Bind encrypted retention to expiry and terminal deletion only."""

    expires_at: ExpiryTick
    delete_on_terminal: bool

    def __post_init__(self) -> None:
        """Reject caller-selected audiences from the retention record."""
        if (
            type(self.expires_at) is not ExpiryTick
            or type(self.delete_on_terminal) is not bool
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)


@dataclass(frozen=True, slots=True)
class DurableRetentionRecord:
    """Store one bounded encrypted value with a versioned key witness."""

    retention_id: PatchRetentionRecordId
    kind: DurableRetentionKind
    key_id: PatchRetentionKeyId
    value: EncryptedRetentionValue
    policy: DurableRetentionPolicy

    def __post_init__(self) -> None:
        """Reject plaintext-like or untyped retention records."""
        if (
            type(self.retention_id) is not PatchRetentionRecordId
            or type(self.kind) is not DurableRetentionKind
            or type(self.key_id) is not PatchRetentionKeyId
            or type(self.value) is not EncryptedRetentionValue
            or type(self.policy) is not DurableRetentionPolicy
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableRetentionAccess:
    """Bind an authenticated retention read to an exact request identity."""

    request: DurableRequestAccess

    def __post_init__(self) -> None:
        """Reject a caller-supplied retention audience selector."""
        if type(self.request) is not DurableRequestAccess:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)


class DurableRetentionAuthorizer(Protocol):
    """Derive retention audiences from authenticated request authority."""

    async def audiences_for(
        self,
        identity: DurableRequestIdentity,
        kind: DurableRetentionKind,
    ) -> frozenset[Audience]:
        """Return the authoritative audience set for one retained kind."""


class DurableRetentionEnvelopeValidator(Protocol):
    """Authenticate a versioned encrypted retention envelope before use."""

    async def validate(
        self,
        request_id: PatchRequestId,
        record: DurableRetentionRecord,
    ) -> None:
        """Reject ciphertext not bound to the exact retention record."""


class DenyDurableRetentionAuthorizer:
    """Fail closed until a retention authorization source is configured."""

    async def audiences_for(
        self,
        identity: DurableRequestIdentity,
        kind: DurableRetentionKind,
    ) -> frozenset[Audience]:
        """Return no audiences without examining caller-selected state."""
        if (
            type(identity) is not DurableRequestIdentity
            or type(kind) is not DurableRetentionKind
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        return frozenset()


class DenyDurableRetentionEnvelopeValidator:
    """Fail closed until a versioned AEAD envelope validator is configured."""

    async def validate(
        self,
        request_id: PatchRequestId,
        record: DurableRetentionRecord,
    ) -> None:
        """Reject all values that lack a configured authenticated envelope."""
        if (
            type(request_id) is not PatchRequestId
            or type(record) is not DurableRetentionRecord
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)


@dataclass(frozen=True, slots=True)
class DurableRetentionCleanup:
    """Report retention cleanup separately from immutable mutation truth."""

    records_deleted: int
    bytes_deleted: ByteSize

    def __post_init__(self) -> None:
        """Require nonnegative exact cleanup accounting."""
        if (
            type(self.records_deleted) is not int
            or self.records_deleted < 0
            or type(self.bytes_deleted) is not ByteSize
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_CONFLICT)


@dataclass(frozen=True, slots=True)
class DurableStoreLimits:
    """Store finite bounded capacities for durable semantic records."""

    max_journal_entries: int = 8192
    max_artifacts: int = 1024
    max_retention_records: int = 128
    max_retention_bytes: ByteSize = ByteSize(4_194_304)

    def __post_init__(self) -> None:
        """Require positive fixed capacity ceilings."""
        if (
            type(self.max_journal_entries) is not int
            or not 1 <= self.max_journal_entries <= 65_536
            or type(self.max_artifacts) is not int
            or not 1 <= self.max_artifacts <= 4_096
            or type(self.max_retention_records) is not int
            or not 1 <= self.max_retention_records <= 4_096
            or type(self.max_retention_bytes) is not ByteSize
            or self.max_retention_bytes.value == 0
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)


@runtime_checkable
class DurablePatchStore(Protocol):
    """Persist one strict async durable patch semantic request contract."""

    async def reserve(
        self,
        identity: DurableRequestIdentity,
        canonical_digest: AlgorithmDigest,
        request_id: PatchRequestId | None = None,
    ) -> DurableReservation:
        """Reserve an authenticated request identity before inspection."""

    async def admit_coordination(
        self, admission: DurableCoordinationAdmission
    ) -> None:
        """Durably serialize one workspace mutation before planning."""

    async def release_coordination(
        self, access: DurableCoordinationAccess
    ) -> None:
        """Release one exact terminal or unplanned workspace admission."""

    async def release_terminal_coordination(
        self, access: DurableRequestAccess
    ) -> None:
        """Release the matching terminal admission after settlement."""

    async def is_coordination_admitted(
        self, access: DurableCoordinationAccess
    ) -> bool:
        """Return whether one exact workspace admission remains current."""

    async def persist_plan(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
    ) -> DurableRequestSnapshot:
        """Persist one immutable sealed-plan reference through CAS."""

    async def claim_commit(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
        approval: DurableApproval,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
        artifact_ids: tuple[PatchArtifactId, ...],
    ) -> DurableCommitClaim:
        """Consume approval and atomically establish commit start."""

    async def renew_lease(
        self,
        lease: DurableCommitLease,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Renew one still-current fenced lease without changing its epoch."""

    async def bind_worker(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding,
        now: ExpiryTick,
    ) -> None:
        """Persist the exact live write-capable child for this owner epoch."""

    async def mark_worker_reaped(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding,
    ) -> None:
        """Persist child death even after its owner lease has expired."""

    async def mark_worker_absent(self, lease: DurableCommitLease) -> None:
        """Persist that no write-capable child remains for this owner."""

    async def replace_expired_owner(
        self,
        reservation: DurableReservation,
        expired_lease: DurableCommitLease,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Fence an expired owner before assigning a replacement owner."""

    async def is_current_fence(
        self, lease: DurableCommitLease, now: ExpiryTick
    ) -> bool:
        """Return whether a lease remains current and unexpired."""

    async def append_step(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        step_id: PatchStepId,
        state: CommitStepState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Append one compare-and-set requested-effect journal transition."""

    async def append_artifact(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        artifact_id: PatchArtifactId,
        state: DurableArtifactState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Append one compare-and-set target-owned artifact transition."""

    async def suspend(
        self,
        lease: DurableCommitLease,
        pending: DurablePendingRequest,
        now: ExpiryTick,
    ) -> DurablePendingRecord:
        """Persist a fenced host-resumable nonterminal pending operation."""

    async def request_cancellation(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Persist cancellation intent without discarding commit ownership."""

    async def settle(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        result: PatchResult,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
    ) -> DurableTerminalRecord:
        """Atomically persist one terminal result and outbox record."""

    async def inspect(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Read one authenticated content-free durable request snapshot."""

    async def inspect_pending(
        self, access: DurablePendingAccess
    ) -> DurablePendingRecord | DurableTerminalRecord:
        """Read pending or terminal state on exactly the original branch."""

    async def await_terminal(
        self, access: DurablePendingAccess
    ) -> DurableTerminalRecord:
        """Await terminal settlement without holding a transaction or lease."""

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Read stable at-least-once delivery records after one cursor."""

    async def put_retention(
        self,
        reservation: DurableReservation,
        record: DurableRetentionRecord,
    ) -> None:
        """Persist one bounded encrypted value without plaintext access."""

    async def get_retention(
        self,
        access: DurableRetentionAccess,
        retention_id: PatchRetentionRecordId,
        now: ExpiryTick,
    ) -> DurableRetentionRecord:
        """Read one authorized unexpired encrypted retention record."""

    async def get_retention_for_audience(
        self,
        access: DurableRetentionAccess,
        retention_id: PatchRetentionRecordId,
        kind: DurableRetentionKind,
        audience: Audience,
        now: ExpiryTick,
    ) -> DurableRetentionRecord:
        """Read one exact kind only for its configured audience."""

    async def cleanup_retention(
        self, now: ExpiryTick
    ) -> DurableRetentionCleanup:
        """Delete expired or terminal-selected private retained values."""


@dataclass(frozen=True, slots=True)
class DurablePatchStoreBinding:
    """Bind one shared store to the loader-owned async resource lifetime."""

    store: DurablePatchStore
    resource: AbstractAsyncContextManager[object]

    def __post_init__(self) -> None:
        """Require one exact shared async store resource."""
        if (
            not callable(getattr(self.store, "reserve", None))
            or not isinstance(self.resource, AbstractAsyncContextManager)
            or hasattr(self.resource, "__enter__")
            or id(self.resource) != id(self.store)
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


@dataclass(slots=True)
class _DurableRecord:
    """Keep mutable implementation state private to one durable backend."""

    reservation: DurableReservation
    plan: DurablePlanReference | None = None
    lifecycle: LifecyclePhase = LifecyclePhase.RECEIVED
    lease: DurableCommitLease | None = None
    worker: DurableWorkerBinding | None = None
    worker_reaped: bool = False
    steps: dict[PatchStepId, CommitStepState] = field(default_factory=dict)
    step_history: list[DurableStepJournalEntry] = field(default_factory=list)
    artifact_history: list[DurableArtifactJournalEntry] = field(
        default_factory=list
    )
    journal_revision: int = 0
    pending: DurablePendingRecord | None = None
    terminal: DurableTerminalRecord | None = None
    cancellation_requested: bool = False
    event_cursor: int = 0
    outbox: list[DurableOutboxRecord] = field(default_factory=list)
    retention: dict[PatchRetentionRecordId, DurableRetentionRecord] = field(
        default_factory=dict
    )
    terminal_event: Event = field(default_factory=Event)


class InMemoryDurablePatchBackend:
    """Own shared in-memory durable state across store clients."""

    def __init__(
        self,
        limits: DurableStoreLimits = DurableStoreLimits(),
        approval_verifier: DurableApprovalVerifier = (
            DenyDurableApprovalVerifier()
        ),
        retention_authorizer: DurableRetentionAuthorizer = (
            DenyDurableRetentionAuthorizer()
        ),
        retention_validator: DurableRetentionEnvelopeValidator = (
            DenyDurableRetentionEnvelopeValidator()
        ),
    ) -> None:
        """Initialize empty shared state with one short atomic lock."""
        if (
            type(limits) is not DurableStoreLimits
            or not callable(getattr(approval_verifier, "verify", None))
            or not callable(
                getattr(retention_authorizer, "audiences_for", None)
            )
            or not callable(getattr(retention_validator, "validate", None))
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
        self.limits = limits
        self.approval_verifier = approval_verifier
        self.retention_authorizer = retention_authorizer
        self.retention_validator = retention_validator
        self.lock = Lock()
        self.records: dict[DurableRequestIdentity, _DurableRecord] = {}
        self.by_request: dict[PatchRequestId, _DurableRecord] = {}
        self.fences: dict[PatchDomainId, int] = {}
        self.active_leases: dict[PatchDomainId, DurableCommitLease] = {}
        self.coordination: dict[
            PatchWorkspaceId, DurableCoordinationAdmission
        ] = {}
        self.consumed_grants: dict[PatchGrantId, PatchRequestId] = {}
        self.event_ids: set[PatchEventId] = set()


class InMemoryDurablePatchStore:
    """Implement durable semantics with shared in-memory records."""

    def __init__(self, backend: InMemoryDurablePatchBackend) -> None:
        """Bind one store client to one explicit durable-test backend."""
        if type(backend) is not InMemoryDurablePatchBackend:
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)
        self._backend = backend

    async def reserve(
        self,
        identity: DurableRequestIdentity,
        canonical_digest: AlgorithmDigest,
        request_id: PatchRequestId | None = None,
    ) -> DurableReservation:
        """Reserve an authenticated identity or attach only to its digest."""
        _require_exact(identity, DurableRequestIdentity)
        _require_exact(canonical_digest, AlgorithmDigest)
        if request_id is not None:
            _require_exact(request_id, PatchRequestId)
        async with self._backend.lock:
            existing = self._backend.records.get(identity)
            if existing is not None:
                if existing.reservation.canonical_digest != canonical_digest:
                    raise DurableStoreError(
                        DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return DurableReservation(
                    existing.reservation.request_id,
                    identity,
                    canonical_digest,
                    True,
                )
            if (
                request_id is not None
                and request_id in self._backend.by_request
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
                )
            reservation = DurableReservation(
                request_id or PatchRequestId.new(),
                identity,
                canonical_digest,
                False,
            )
            record = _DurableRecord(reservation)
            self._backend.records[identity] = record
            self._backend.by_request[reservation.request_id] = record
            return reservation

    async def admit_coordination(
        self, admission: DurableCoordinationAdmission
    ) -> None:
        """Durably admit only one exact uncompleted workspace mutation."""
        _require_exact(admission, DurableCoordinationAdmission)
        async with self._backend.lock:
            record = self._record_for_reservation(admission.access.reservation)
            if record.terminal is not None:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            existing = self._backend.coordination.get(
                admission.access.workspace_id
            )
            if existing is None:
                self._backend.coordination[admission.access.workspace_id] = (
                    admission
                )
                return
            if existing != admission:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )

    async def release_coordination(
        self, access: DurableCoordinationAccess
    ) -> None:
        """Release only an exact terminal or not-yet-planned admission."""
        _require_exact(access, DurableCoordinationAccess)
        async with self._backend.lock:
            record = self._record_for_reservation(access.reservation)
            existing = self._backend.coordination.get(access.workspace_id)
            if existing is None:
                return
            if existing.access != access:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            if record.terminal is None and record.plan is not None:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            del self._backend.coordination[access.workspace_id]

    async def release_terminal_coordination(
        self, access: DurableRequestAccess
    ) -> None:
        """Release only the terminal admission matching one exact request."""
        _require_exact(access, DurableRequestAccess)
        async with self._backend.lock:
            record = self._record_for_access(access)
            if record.terminal is None:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            for workspace_id, admission in tuple(
                self._backend.coordination.items()
            ):
                if (
                    admission.access.reservation.request_id
                    == access.request_id
                    and admission.access.reservation.identity
                    == access.identity
                ):
                    del self._backend.coordination[workspace_id]
                    return

    async def is_coordination_admitted(
        self, access: DurableCoordinationAccess
    ) -> bool:
        """Return whether this exact admission still owns its workspace."""
        _require_exact(access, DurableCoordinationAccess)
        async with self._backend.lock:
            self._record_for_reservation(access.reservation)
            return (
                self._backend.coordination.get(access.workspace_id) is not None
                and self._backend.coordination[access.workspace_id].access
                == access
            )

    async def persist_plan(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
    ) -> DurableRequestSnapshot:
        """Persist one immutable plan before claiming ownership."""
        _require_exact(plan, DurablePlanReference)
        async with self._backend.lock:
            record = self._record_for_reservation(reservation)
            if plan.canonical_digest != reservation.canonical_digest:
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            if record.terminal is not None:
                if record.plan == plan:
                    return self._snapshot(record)
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            if record.plan is None:
                if record.lifecycle is not LifecyclePhase.RECEIVED:
                    raise DurableStoreError(
                        DurableStoreErrorCode.LIFECYCLE_CONFLICT
                    )
                record.plan = plan
                record.lifecycle = LifecyclePhase.PLANNED
            elif record.plan != plan:
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            return self._snapshot(record)

    async def claim_commit(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
        approval: DurableApproval,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
        artifact_ids: tuple[PatchArtifactId, ...],
    ) -> DurableCommitClaim:
        """Consume approval and establish one fenced commit-start record."""
        _require_exact(plan, DurablePlanReference)
        _require_exact(approval, DurableApproval)
        _require_exact(owner_id, PatchCommitOwnerId)
        _require_exact(now, ExpiryTick)
        _require_exact(lease_duration, DurationTicks)
        _require_artifact_ids(artifact_ids, self._backend.limits)
        self._backend.approval_verifier.verify(approval)
        async with self._backend.lock:
            record = self._record_for_reservation(reservation)
            if record.terminal is not None:
                return DurableCommitClaim(
                    DurableCommitClaimState.TERMINAL, None, record.terminal
                )
            if record.plan != plan:
                raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
            if record.lease is not None:
                return DurableCommitClaim(
                    DurableCommitClaimState.ATTACHED, None, None
                )
            if record.lifecycle is not LifecyclePhase.PLANNED:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            self._validate_approval(reservation, plan, approval, now)
            self._require_unclaimed_domain(plan.domain_id, now)
            consumed = self._backend.consumed_grants.get(approval.grant_id)
            if consumed is not None:
                raise DurableStoreError(
                    DurableStoreErrorCode.APPROVAL_CONSUMED
                )
            fence = self._backend.fences.get(plan.domain_id, 0) + 1
            self._backend.fences[plan.domain_id] = fence
            lease = DurableCommitLease(
                reservation.request_id,
                plan.domain_id,
                owner_id,
                SequenceNumber(fence),
                _lease_expiry(now, lease_duration),
            )
            self._backend.active_leases[plan.domain_id] = lease
            self._backend.consumed_grants[approval.grant_id] = (
                reservation.request_id
            )
            record.lease = lease
            record.lifecycle = LifecyclePhase.COMMIT_STARTED
            for artifact_id in artifact_ids:
                cursor = self._next_cursor(record)
                record.artifact_history.append(
                    DurableArtifactJournalEntry(
                        cursor,
                        artifact_id,
                        DurableArtifactState.INTENDED,
                    )
                )
            return DurableCommitClaim(
                DurableCommitClaimState.OWNER, lease, None
            )

    async def renew_lease(
        self,
        lease: DurableCommitLease,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Renew one current lease without altering owner or fence epoch."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(now, ExpiryTick)
        _require_exact(lease_duration, DurationTicks)
        async with self._backend.lock:
            record = self._current_record(lease, now)
            expires_at = _lease_expiry(now, lease_duration)
            if expires_at.value <= lease.expires_at.value:
                raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)
            renewed = DurableCommitLease(
                lease.request_id,
                lease.domain_id,
                lease.owner_id,
                lease.fence,
                expires_at,
            )
            record.lease = renewed
            self._backend.active_leases[renewed.domain_id] = renewed
            if record.pending is not None:
                record.pending = DurablePendingRecord(
                    record.pending.request_id,
                    record.pending.execution_id,
                    record.pending.pending_operation_id,
                    record.pending.correlation_id,
                    renewed.fence,
                    record.pending.event_cursor,
                    record.pending.cancellation_requested,
                    record.pending.next_check_after,
                )
            return renewed

    async def bind_worker(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding,
        now: ExpiryTick,
    ) -> None:
        """Persist one exact live child before any workspace effect."""
        _require_exact(binding, DurableWorkerBinding)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._current_record(lease, now)
            if record.worker is not None and record.worker != binding:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            record.worker = binding
            record.worker_reaped = False

    async def mark_worker_reaped(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding,
    ) -> None:
        """Persist exact child reaping without requiring an unexpired lease."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(binding, DurableWorkerBinding)
        async with self._backend.lock:
            record = self._record_for_lease(lease)
            if record.lease != lease or record.worker != binding:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            record.worker_reaped = True

    async def mark_worker_absent(self, lease: DurableCommitLease) -> None:
        """Persist exact no-live-child recovery evidence for one owner."""
        _require_exact(lease, DurableCommitLease)
        async with self._backend.lock:
            record = self._record_for_lease(lease)
            if record.lease != lease or record.worker is not None:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            record.worker_reaped = True

    async def replace_expired_owner(
        self,
        reservation: DurableReservation,
        expired_lease: DurableCommitLease,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
    ) -> DurableCommitLease:
        """Advance fencing only after the exact old lease has expired."""
        _require_exact(expired_lease, DurableCommitLease)
        _require_exact(owner_id, PatchCommitOwnerId)
        _require_exact(now, ExpiryTick)
        _require_exact(lease_duration, DurationTicks)
        async with self._backend.lock:
            record = self._record_for_reservation(reservation)
            if (
                record.terminal is not None
                or record.lease != expired_lease
                or record.plan is None
                or record.lifecycle
                not in {
                    LifecyclePhase.COMMIT_STARTED,
                    LifecyclePhase.SETTLEMENT_PENDING,
                }
            ):
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            if now.value < expired_lease.expires_at.value:
                raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)
            if (
                self._backend.active_leases.get(expired_lease.domain_id)
                != expired_lease
                or self._backend.fences.get(expired_lease.domain_id)
                != expired_lease.fence.value
            ):
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            if owner_id == expired_lease.owner_id:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            if record.worker is not None and not record.worker_reaped:
                raise DurableStoreError(DurableStoreErrorCode.FENCED)
            fence = self._backend.fences.get(expired_lease.domain_id, 0) + 1
            self._backend.fences[expired_lease.domain_id] = fence
            lease = DurableCommitLease(
                reservation.request_id,
                expired_lease.domain_id,
                owner_id,
                SequenceNumber(fence),
                _lease_expiry(now, lease_duration),
            )
            record.lease = lease
            record.worker = None
            record.worker_reaped = False
            self._backend.active_leases[lease.domain_id] = lease
            if record.pending is not None:
                record.pending = DurablePendingRecord(
                    record.pending.request_id,
                    record.pending.execution_id,
                    record.pending.pending_operation_id,
                    record.pending.correlation_id,
                    lease.fence,
                    record.pending.event_cursor,
                    record.pending.cancellation_requested,
                    record.pending.next_check_after,
                )
            return lease

    async def is_current_fence(
        self, lease: DurableCommitLease, now: ExpiryTick
    ) -> bool:
        """Return whether the exact owner/fence remains current and live."""
        _require_exact(lease, DurableCommitLease)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._backend.by_request.get(lease.request_id)
            return (
                record is not None
                and record.lease == lease
                and record.terminal is None
                and now.value < lease.expires_at.value
                and self._backend.active_leases.get(lease.domain_id) == lease
                and self._backend.fences.get(lease.domain_id)
                == lease.fence.value
            )

    async def append_step(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        step_id: PatchStepId,
        state: CommitStepState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Append one legal requested-effect state through journal CAS."""
        _require_exact(expected, DurableJournalCursor)
        _require_exact(step_id, PatchStepId)
        _require_exact(state, CommitStepState)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._current_record(lease, now)
            self._require_expected_cursor(record, expected)
            plan = _require_plan(record)
            binding = next(
                (item for item in plan.steps if item.step_id == step_id), None
            )
            if binding is None or not _step_transition(
                record.steps.get(step_id), state
            ):
                raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
            cursor = self._next_cursor(record)
            entry = DurableStepJournalEntry(
                cursor, step_id, binding.lineage_id, state
            )
            record.steps[step_id] = state
            record.step_history.append(entry)
            return self._journal(record)

    async def append_artifact(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        artifact_id: PatchArtifactId,
        state: DurableArtifactState,
        now: ExpiryTick,
    ) -> DurableJournal:
        """Append one legal target-owned artifact state through journal CAS."""
        _require_exact(expected, DurableJournalCursor)
        _require_exact(artifact_id, PatchArtifactId)
        _require_exact(state, DurableArtifactState)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._current_record(lease, now)
            self._require_expected_cursor(record, expected)
            previous = _artifact_current_state(
                tuple(record.artifact_history), artifact_id
            )
            if previous is None or not _artifact_transition(previous, state):
                raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
            cursor = self._next_cursor(record)
            entry = DurableArtifactJournalEntry(cursor, artifact_id, state)
            record.artifact_history.append(entry)
            return self._journal(record)

    async def suspend(
        self,
        lease: DurableCommitLease,
        pending: DurablePendingRequest,
        now: ExpiryTick,
    ) -> DurablePendingRecord:
        """Persist a fenced pending request and one durable pending event."""
        _require_exact(pending, DurablePendingRequest)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._current_record(lease, now)
            if record.pending is not None:
                current = record.pending
                if (
                    current.pending_operation_id
                    == pending.pending_operation_id
                    and current.correlation_id == pending.correlation_id
                    and current.next_check_after == pending.next_check_after
                ):
                    return current
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            if record.lifecycle is not LifecyclePhase.COMMIT_STARTED:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            record.lifecycle = LifecyclePhase.SETTLEMENT_PENDING
            record.event_cursor += 1
            value = DurablePendingRecord(
                record.reservation.request_id,
                record.reservation.identity.execution_id,
                pending.pending_operation_id,
                pending.correlation_id,
                lease.fence,
                SequenceNumber(record.event_cursor),
                record.cancellation_requested,
                pending.next_check_after,
            )
            record.pending = value
            self._append_outbox(
                record,
                LifecyclePhase.SETTLEMENT_PENDING,
                pending.correlation_id,
            )
            return value

    async def request_cancellation(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Persist post-commit cancellation intent without cancelling work."""
        _require_exact(access, DurableRequestAccess)
        async with self._backend.lock:
            record = self._record_for_access(access)
            if record.lifecycle not in {
                LifecyclePhase.COMMIT_STARTED,
                LifecyclePhase.SETTLEMENT_PENDING,
            }:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            record.cancellation_requested = True
            if record.pending is not None:
                record.pending = DurablePendingRecord(
                    record.pending.request_id,
                    record.pending.execution_id,
                    record.pending.pending_operation_id,
                    record.pending.correlation_id,
                    record.pending.fence,
                    record.pending.event_cursor,
                    True,
                    record.pending.next_check_after,
                )
            return self._snapshot(record)

    async def settle(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        result: PatchResult,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
    ) -> DurableTerminalRecord:
        """Persist terminal truth and one request-completed outbox event."""
        _require_exact(expected, DurableJournalCursor)
        _require_exact(result, PatchResult)
        _require_exact(correlation_id, PatchObserverCorrelationId)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._record_for_lease(lease)
            if record.terminal is not None:
                if record.terminal.result == result:
                    return record.terminal
                raise DurableStoreError(
                    DurableStoreErrorCode.TERMINAL_CONFLICT
                )
            self._require_current(record, lease, now)
            self._require_expected_cursor(record, expected)
            plan = _require_plan(record)
            if (
                result.request_id != lease.request_id
                or result.plan_id != plan.plan_id
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.TERMINAL_CONFLICT
                )
            if record.pending is not None and (
                record.pending.correlation_id != correlation_id
            ):
                raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
            mutation = _journal_mutation_state(record, plan)
            artifact = derive_artifact_state(tuple(record.artifact_history))
            if (
                result.truth.mutation_state is not mutation
                or result.truth.artifact_state is not artifact
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.TERMINAL_CONFLICT
                )
            record.event_cursor += 1
            outbox = DurableOutboxRecord(
                PatchEventId.new(),
                lease.request_id,
                SequenceNumber(record.event_cursor),
                LifecyclePhase.REQUEST_COMPLETED,
                correlation_id,
            )
            if outbox.event_id in self._backend.event_ids:
                raise DurableStoreError(
                    DurableStoreErrorCode.TERMINAL_CONFLICT
                )
            terminal = DurableTerminalRecord(
                result,
                outbox,
                (
                    None
                    if record.pending is None
                    else record.pending.pending_operation_id
                ),
            )
            record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
            record.pending = None
            record.terminal = terminal
            record.outbox.append(outbox)
            if self._backend.active_leases.get(lease.domain_id) == lease:
                del self._backend.active_leases[lease.domain_id]
            for workspace_id, admission in tuple(
                self._backend.coordination.items()
            ):
                if (
                    admission.access.reservation.request_id == lease.request_id
                    and admission.access.reservation.identity
                    == record.reservation.identity
                ):
                    del self._backend.coordination[workspace_id]
            self._backend.event_ids.add(outbox.event_id)
            self._delete_terminal_retention(record)
            record.terminal_event.set()
            return terminal

    async def inspect(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Read one exact authenticated durable snapshot without mutation."""
        _require_exact(access, DurableRequestAccess)
        async with self._backend.lock:
            return self._snapshot(self._record_for_access(access))

    async def inspect_pending(
        self, access: DurablePendingAccess
    ) -> DurablePendingRecord | DurableTerminalRecord:
        """Read the original pending branch or terminal continuation."""
        _require_exact(access, DurablePendingAccess)
        async with self._backend.lock:
            record = self._record_for_access(access.request)
            pending = record.pending
            if pending is not None:
                if not _pending_access_matches(access, pending):
                    raise DurableStoreError(
                        DurableStoreErrorCode.ACCESS_DENIED
                    )
                return pending
            terminal = record.terminal
            if terminal is None or not _terminal_access_matches(
                access, terminal
            ):
                raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
            return terminal

    async def await_terminal(
        self, access: DurablePendingAccess
    ) -> DurableTerminalRecord:
        """Await settlement after branch authentication outside locks."""
        _require_exact(access, DurablePendingAccess)
        while True:
            async with self._backend.lock:
                record = self._record_for_access(access.request)
                pending = record.pending
                terminal = record.terminal
                if terminal is not None:
                    if not _terminal_access_matches(access, terminal):
                        raise DurableStoreError(
                            DurableStoreErrorCode.ACCESS_DENIED
                        )
                    return terminal
                if pending is None or not _pending_access_matches(
                    access, pending
                ):
                    raise DurableStoreError(
                        DurableStoreErrorCode.ACCESS_DENIED
                    )
                completed = record.terminal_event
            await completed.wait()

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Read stable outbox records without acknowledging delivery."""
        _require_exact(access, DurableRequestAccess)
        _require_exact(after, SequenceNumber)
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        async with self._backend.lock:
            record = self._record_for_access(access)
            return tuple(
                item
                for item in record.outbox
                if item.sequence.value > after.value
            )[:limit]

    async def put_retention(
        self,
        reservation: DurableReservation,
        record: DurableRetentionRecord,
    ) -> None:
        """Store ciphertext under its immutable versioned-key contract."""
        _require_exact(reservation, DurableReservation)
        _require_exact(record, DurableRetentionRecord)
        async with self._backend.lock:
            self._record_for_reservation(reservation)
        await self._backend.retention_validator.validate(
            reservation.request_id, record
        )
        async with self._backend.lock:
            request = self._record_for_reservation(reservation)
            if request.terminal is not None:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            existing = request.retention.get(record.retention_id)
            if existing is not None:
                if existing != record:
                    raise DurableStoreError(
                        DurableStoreErrorCode.RETENTION_CONFLICT
                    )
                return
            if (
                len(request.retention)
                >= self._backend.limits.max_retention_records
            ):
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
            total = sum(
                item.value.size().value for item in request.retention.values()
            )
            if total + record.value.size().value > (
                self._backend.limits.max_retention_bytes.value
            ):
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
            if request.terminal is not None:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            request.retention[record.retention_id] = record

    async def get_retention(
        self,
        access: DurableRetentionAccess,
        retention_id: PatchRetentionRecordId,
        now: ExpiryTick,
    ) -> DurableRetentionRecord:
        """Return only an authorized unexpired ciphertext record."""
        _require_exact(access, DurableRetentionAccess)
        _require_exact(retention_id, PatchRetentionRecordId)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._record_for_access(access.request)
            retained = record.retention.get(retention_id)
            if retained is None:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            if now.value >= retained.policy.expires_at.value:
                del record.retention[retention_id]
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            identity = record.reservation.identity
        audiences = await self._backend.retention_authorizer.audiences_for(
            identity, retained.kind
        )
        if not isinstance(audiences, frozenset) or any(
            type(item) is not Audience for item in audiences
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        if not audiences:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        await self._backend.retention_validator.validate(
            access.request.request_id, retained
        )
        return retained

    async def get_retention_for_audience(
        self,
        access: DurableRetentionAccess,
        retention_id: PatchRetentionRecordId,
        kind: DurableRetentionKind,
        audience: Audience,
        now: ExpiryTick,
    ) -> DurableRetentionRecord:
        """Return one exact kind only to its authenticated audience."""
        _require_exact(access, DurableRetentionAccess)
        _require_exact(retention_id, PatchRetentionRecordId)
        _require_exact(kind, DurableRetentionKind)
        _require_exact(audience, Audience)
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            record = self._record_for_access(access.request)
            retained = record.retention.get(retention_id)
            if retained is None or retained.kind is not kind:
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            if now.value >= retained.policy.expires_at.value:
                del record.retention[retention_id]
                raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
            identity = record.reservation.identity
        audiences = await self._backend.retention_authorizer.audiences_for(
            identity, kind
        )
        if (
            type(audiences) is not frozenset
            or audience not in audiences
            or any(type(item) is not Audience for item in audiences)
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        await self._backend.retention_validator.validate(
            access.request.request_id, retained
        )
        return retained

    async def cleanup_retention(
        self, now: ExpiryTick
    ) -> DurableRetentionCleanup:
        """Delete records selected by expiry or terminal-retention policy."""
        _require_exact(now, ExpiryTick)
        async with self._backend.lock:
            deleted = 0
            bytes_deleted = 0
            for record in self._backend.by_request.values():
                identifiers = tuple(
                    identifier
                    for identifier, retained in record.retention.items()
                    if now.value >= retained.policy.expires_at.value
                    or (
                        record.terminal is not None
                        and retained.policy.delete_on_terminal
                    )
                )
                for identifier in identifiers:
                    retained = record.retention.pop(identifier)
                    deleted += 1
                    bytes_deleted += retained.value.size().value
            return DurableRetentionCleanup(deleted, ByteSize(bytes_deleted))

    def _record_for_reservation(
        self, reservation: DurableReservation
    ) -> _DurableRecord:
        """Resolve one exact reservation while the backend lock is held."""
        _require_exact(reservation, DurableReservation)
        record = self._backend.records.get(reservation.identity)
        if (
            record is None
            or record.reservation.request_id != reservation.request_id
            or record.reservation.canonical_digest
            != reservation.canonical_digest
        ):
            raise DurableStoreError(DurableStoreErrorCode.INVALID_RESERVATION)
        return record

    def _record_for_access(
        self, access: DurableRequestAccess
    ) -> _DurableRecord:
        """Resolve an authenticated request without leaking existence."""
        record = self._backend.records.get(access.identity)
        if (
            record is None
            or record.reservation.request_id != access.request_id
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
        return record

    def _record_for_lease(self, lease: DurableCommitLease) -> _DurableRecord:
        """Resolve one request for a claimed lease under the backend lock."""
        _require_exact(lease, DurableCommitLease)
        record = self._backend.by_request.get(lease.request_id)
        if record is None:
            raise DurableStoreError(DurableStoreErrorCode.FENCED)
        return record

    def _current_record(
        self, lease: DurableCommitLease, now: ExpiryTick
    ) -> _DurableRecord:
        """Resolve and validate current unexpired fence ownership."""
        _require_exact(lease, DurableCommitLease)
        record = self._record_for_lease(lease)
        self._require_current(record, lease, now)
        return record

    def _require_current(
        self,
        record: _DurableRecord,
        lease: DurableCommitLease,
        now: ExpiryTick,
    ) -> None:
        """Fail closed unless a lease is exact, current, and unexpired."""
        if (
            record.terminal is not None
            or record.lease != lease
            or record.lifecycle
            not in {
                LifecyclePhase.COMMIT_STARTED,
                LifecyclePhase.SETTLEMENT_PENDING,
            }
            or self._backend.fences.get(lease.domain_id) != lease.fence.value
        ):
            raise DurableStoreError(DurableStoreErrorCode.FENCED)
        if now.value >= lease.expires_at.value:
            raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)
        if self._backend.active_leases.get(lease.domain_id) != lease:
            raise DurableStoreError(DurableStoreErrorCode.FENCED)

    def _require_unclaimed_domain(
        self, domain_id: PatchDomainId, now: ExpiryTick
    ) -> None:
        """Reject a different live owner before consuming its approval."""
        del now
        active = self._backend.active_leases.get(domain_id)
        if active is None:
            return
        record = self._backend.by_request.get(active.request_id)
        if (
            record is None
            or record.lease != active
            or record.terminal is not None
            or record.lifecycle
            not in {
                LifecyclePhase.COMMIT_STARTED,
                LifecyclePhase.SETTLEMENT_PENDING,
            }
        ):
            del self._backend.active_leases[domain_id]
            return
        raise DurableStoreError(DurableStoreErrorCode.FENCED)

    def _validate_approval(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
        approval: DurableApproval,
        now: ExpiryTick,
    ) -> None:
        """Require exact unexpired request, plan, and approval bindings."""
        if (
            approval.identity != reservation.identity
            or approval.canonical_digest != reservation.canonical_digest
            or approval.plan_id != plan.plan_id
            or approval.fingerprint_digest != plan.fingerprint_digest
            or approval.review_digest != plan.review_digest
            or approval.context_id != plan.context_id
            or approval.workspace_id != plan.workspace_id
            or approval.domain_id != plan.domain_id
        ):
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
        if now.value >= approval.expires_at.value:
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_EXPIRED)

    def _require_expected_cursor(
        self, record: _DurableRecord, expected: DurableJournalCursor
    ) -> None:
        """Require a journal compare-and-set cursor from the same request."""
        if expected != DurableJournalCursor(
            record.reservation.request_id,
            SequenceNumber(record.journal_revision),
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)

    def _next_cursor(self, record: _DurableRecord) -> DurableJournalCursor:
        """Advance one bounded globally ordered journal revision."""
        if (
            len(record.step_history) + len(record.artifact_history)
            >= self._backend.limits.max_journal_entries
        ):
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
        record.journal_revision += 1
        return DurableJournalCursor(
            record.reservation.request_id,
            SequenceNumber(record.journal_revision),
        )

    def _journal(self, record: _DurableRecord) -> DurableJournal:
        """Return one immutable journal snapshot under the backend lock."""
        return DurableJournal(
            DurableJournalCursor(
                record.reservation.request_id,
                SequenceNumber(record.journal_revision),
            ),
            tuple(record.step_history),
            tuple(record.artifact_history),
        )

    def _snapshot(self, record: _DurableRecord) -> DurableRequestSnapshot:
        """Return one immutable content-free record snapshot under the lock."""
        return DurableRequestSnapshot(
            record.reservation,
            record.plan,
            record.lifecycle,
            record.lease,
            self._journal(record),
            record.pending,
            record.terminal,
            record.worker is not None,
            record.worker_reaped,
            record.cancellation_requested,
            SequenceNumber(record.event_cursor),
        )

    def _append_outbox(
        self,
        record: _DurableRecord,
        lifecycle: LifecyclePhase,
        correlation_id: PatchObserverCorrelationId,
    ) -> None:
        """Append one unique stable lifecycle event while the lock is held."""
        event = DurableOutboxRecord(
            PatchEventId.new(),
            record.reservation.request_id,
            SequenceNumber(record.event_cursor),
            lifecycle,
            correlation_id,
        )
        if event.event_id in self._backend.event_ids:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        record.outbox.append(event)
        self._backend.event_ids.add(event.event_id)

    def _delete_terminal_retention(self, record: _DurableRecord) -> None:
        """Delete values selected for deletion with terminal settlement."""
        identifiers = tuple(
            identifier
            for identifier, retained in record.retention.items()
            if retained.policy.delete_on_terminal
        )
        for identifier in identifiers:
            del record.retention[identifier]


def _require_exact(value: object, expected: type[object]) -> None:
    """Reject subclassed or untyped values at the durable semantic boundary."""
    if type(value) is not expected:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


def _require_artifact_ids(
    artifact_ids: tuple[PatchArtifactId, ...], limits: DurableStoreLimits
) -> None:
    """Require a unique bounded exact set of persisted artifact intents."""
    if (
        type(artifact_ids) is not tuple
        or len(artifact_ids) > limits.max_artifacts
        or any(type(item) is not PatchArtifactId for item in artifact_ids)
        or len(set(artifact_ids)) != len(artifact_ids)
    ):
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)


def _lease_expiry(now: ExpiryTick, duration: DurationTicks) -> ExpiryTick:
    """Return one bounded lease expiry from the trusted monotonic clock."""
    value = now.value + duration.value
    if value > 2**63 - 1:
        raise DurableStoreError(DurableStoreErrorCode.LEASE_EXPIRED)
    return ExpiryTick(value)


def _step_transition(
    previous: CommitStepState | None, next_state: CommitStepState
) -> bool:
    """Return whether one requested-effect journal transition is monotonic."""
    if previous is None:
        return next_state is CommitStepState.PLANNED
    return previous is CommitStepState.PLANNED and next_state in {
        CommitStepState.COMMITTED,
        CommitStepState.NOT_COMMITTED,
        CommitStepState.UNKNOWN,
    }


def _artifact_transition(
    previous: DurableArtifactState, next_state: DurableArtifactState
) -> bool:
    """Return whether one target-owned artifact transition is monotonic."""
    allowed = {
        DurableArtifactState.INTENDED: {
            DurableArtifactState.NOT_CREATED,
            DurableArtifactState.PRESENT,
            DurableArtifactState.UNKNOWN,
        },
        DurableArtifactState.PRESENT: {
            DurableArtifactState.REMOVED,
            DurableArtifactState.LEAKED,
            DurableArtifactState.UNKNOWN,
        },
        DurableArtifactState.NOT_CREATED: set[DurableArtifactState](),
        DurableArtifactState.REMOVED: set[DurableArtifactState](),
        DurableArtifactState.LEAKED: set[DurableArtifactState](),
        DurableArtifactState.UNKNOWN: set[DurableArtifactState](),
    }
    return next_state in allowed[previous]


def _require_plan(record: _DurableRecord) -> DurablePlanReference:
    """Return the persisted plan or fail closed before journal mutation."""
    if record.plan is None:
        raise DurableStoreError(DurableStoreErrorCode.PLAN_MISMATCH)
    return record.plan


def _journal_mutation_state(
    record: _DurableRecord, plan: DurablePlanReference
) -> MutationState:
    """Derive requested-effect truth only from the complete durable journal."""
    states: list[CommitStepState] = []
    for binding in plan.steps:
        state = record.steps.get(binding.step_id)
        if state is None or state is CommitStepState.PLANNED:
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)
        states.append(state)
    if any(item is CommitStepState.UNKNOWN for item in states):
        return MutationState.INDETERMINATE
    committed = sum(item is CommitStepState.COMMITTED for item in states)
    if committed == 0:
        return MutationState.NOT_COMMITTED
    if committed == len(states):
        return MutationState.COMMITTED
    return MutationState.PARTIALLY_COMMITTED


def derive_artifact_state(
    entries: tuple[DurableArtifactJournalEntry, ...],
) -> ArtifactState:
    """Derive terminal artifact truth from the durable artifact journal."""
    if type(entries) is not tuple or any(
        type(entry) is not DurableArtifactJournalEntry for entry in entries
    ):
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_CONFLICT)
    histories: dict[PatchArtifactId, list[DurableArtifactState]] = {}
    for entry in entries:
        histories.setdefault(entry.artifact_id, []).append(entry.state)
    states: list[DurableArtifactState] = []
    for history in histories.values():
        if not history or history[0] is not DurableArtifactState.INTENDED:
            raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)
        current: DurableArtifactState = history[0]
        for state in history[1:]:
            if not _artifact_transition(current, state):
                raise DurableStoreError(
                    DurableStoreErrorCode.JOURNAL_INCOMPLETE
                )
            current = state
        states.append(current)
    if not states or all(
        item is DurableArtifactState.NOT_CREATED for item in states
    ):
        return ArtifactState.ABSENT
    if any(
        item in {DurableArtifactState.INTENDED, DurableArtifactState.PRESENT}
        for item in states
    ):
        raise DurableStoreError(DurableStoreErrorCode.JOURNAL_INCOMPLETE)
    if any(item is DurableArtifactState.UNKNOWN for item in states):
        return ArtifactState.UNKNOWN
    if any(item is DurableArtifactState.LEAKED for item in states):
        return ArtifactState.LEAKED
    return ArtifactState.CLEANED


def _artifact_current_state(
    entries: tuple[DurableArtifactJournalEntry, ...],
    artifact_id: PatchArtifactId,
) -> DurableArtifactState | None:
    """Return one artifact's latest state only from ordered journal history."""
    state: DurableArtifactState | None = None
    for entry in entries:
        if entry.artifact_id == artifact_id:
            state = entry.state
    return state


def _pending_access_matches(
    access: DurablePendingAccess, pending: DurablePendingRecord
) -> bool:
    """Return whether exact pending branch authentication matches."""
    return (
        access.pending_operation_id == pending.pending_operation_id
        and access.correlation_id == pending.correlation_id
    )


def _terminal_access_matches(
    access: DurablePendingAccess, terminal: DurableTerminalRecord
) -> bool:
    """Return whether the terminal event preserves branch correlation."""
    return (
        access.pending_operation_id == terminal.pending_operation_id
        and access.correlation_id == terminal.outbox.correlation_id
    )
