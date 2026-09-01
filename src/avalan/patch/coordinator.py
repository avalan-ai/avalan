"""Coordinate sealed patch settlement against a scripted target only.

The coordinator is deliberately an attached-lifetime, deterministic reference
implementation.  It has no local filesystem implementation, restart handle,
or public tool registration.  Its sole target is ``ScriptedCommitWorker`` so
the first commit protocol can be exhaustively exercised before any real
mutation authority is activated.
"""

from asyncio import CancelledError, Event, Lock
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Protocol, TypeGuard, final, runtime_checkable
from weakref import WeakKeyDictionary

from avalan._patch_authority import _PatchAuthorityValidator
from avalan.patch.domain import (
    AlgorithmDigest,
    ArtifactState,
    CommitStepState,
    CommitTruth,
    ErrorStage,
    LifecyclePhase,
    LineageState,
    MutationState,
    PatchDiagnostic,
    PatchDomainId,
    PatchErrorCode,
    PatchEventId,
    PatchLifecycleEvent,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchObserverId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)
from avalan.patch.policy import (
    ExecutionSubject,
    PlanBoundGrant,
    PolicyError,
    PolicyRouteId,
    SealedPlan,
)


class CoordinatorErrorCode(str, Enum):
    """Name closed runtime coordination failures."""

    IDEMPOTENCY_CONFLICT = "patch.idempotency_conflict"
    INVALID_RESERVATION = "patch.invalid_reservation"
    STALE = "patch.stale"
    FENCED = "patch.fenced"
    PENDING_OWNER = "patch.pending_owner_denied"
    RETRY_BLOCKED = "patch.retry_blocked"
    SCRIPTED_TARGET_ONLY = "patch.scripted_target_required"
    GRANT_CONSUMED = "patch.grant_consumed"
    INVARIANT = "patch.coordinator_invariant"


class CoordinatorError(RuntimeError):
    """Report one stable coordinator outcome without protected data."""

    def __init__(self, code: CoordinatorErrorCode) -> None:
        """Initialize the closed coordinator error code."""
        super().__init__(code.value)
        self.code = code


class CoordinatorBoundary(str, Enum):
    """Name one deterministic coordinator fault or depth boundary."""

    PRIVATE_STAGING = "private_staging"
    LEASE = "lease"
    REVALIDATION = "revalidation"
    OWNER = "owner"
    FENCE = "fence"
    VISIBLE_ARTIFACT = "visible_artifact"
    REQUESTED_EFFECT = "requested_effect"
    JOURNAL = "journal"
    VERIFICATION = "verification"
    CLEANUP = "cleanup"
    SETTLEMENT = "settlement"
    OUTBOX = "outbox"
    TERMINAL_PUBLICATION = "terminal_publication"


class _PrecommitBoundary(str, Enum):
    """Name dormant test-profile checkpoints before commit ownership."""

    RESERVE_REQUEST = "store.reserve_request"
    PERSIST_PLAN = "store.persist_plan"
    LIFECYCLE_PLANNED = "lifecycle.planned"
    LIFECYCLE_AWAITING_APPROVAL = "lifecycle.awaiting_approval"
    CONSUME_GRANT = "store.consume_grant"
    ASSIGN_COMMIT_OWNER = "store.assign_commit_owner"
    LIFECYCLE_COMMIT_OWNER = "lifecycle.commit_owner_assigned"
    INTENT_FENCE = "commit.intent_fence"
    LIFECYCLE_COMPLETED = "lifecycle.request_completed"
    ACQUIRE_LOCK = "target.acquire_lock"
    RELEASE_LOCK = "target.release_lock"
    CANCELLATION = "cancellation.before_commit"
    TIMEOUT = "timeout.before_commit"
    DISCONNECT = "disconnect.before_commit"


class _PrecommitCheckpoint(Protocol):
    """Inject a typed local-test failure before a workspace effect."""

    async def checkpoint(self, boundary: _PrecommitBoundary) -> None:
        """Observe one fixed precommit boundary or fail there."""


@dataclass(frozen=True, slots=True)
class RetransmissionKey:
    """Identify authenticated retransmission metadata outside model input."""

    value: str

    def __post_init__(self) -> None:
        """Require a bounded nonempty route-owned retransmission key."""
        if not self.value or len(self.value) > 128:
            raise CoordinatorError(CoordinatorErrorCode.INVALID_RESERVATION)


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Bind one external retry tuple to a canonical request digest."""

    subject: "ExecutionSubject"
    route: "PolicyRouteId"
    retransmission_key: RetransmissionKey


@dataclass(frozen=True, slots=True)
class Reservation:
    """Return the sole runtime record selected before filesystem inspection."""

    request_id: PatchRequestId
    identity: RuntimeIdentity
    digest: AlgorithmDigest
    replayed: bool


class RevalidationField(str, Enum):
    """Name every closed fact that must survive approval to commit."""

    CONTEXT = "context"
    POLICY = "policy"
    CAPABILITY = "capability"
    GRANT = "grant"
    SOURCE = "source"
    DESTINATION = "destination"
    PARENT = "parent"
    IDENTITY = "identity"
    LINK = "link"
    METADATA = "metadata"
    HASH = "hash"
    SIZE = "size"
    ABSENCE = "absence"
    ALIAS = "alias"
    MOUNT = "mount"
    WORKSPACE = "workspace"
    CWD = "cwd"


@dataclass(frozen=True, slots=True)
class RevalidationFact:
    """Store one sealed, non-disclosing precondition evidence value."""

    field: RevalidationField
    key: str
    value: str

    def __post_init__(self) -> None:
        """Reject empty or oversized opaque revalidation facts."""
        if (
            type(self.field) is not RevalidationField
            or not self.key
            or not self.value
            or len(self.key) > 256
            or len(self.value) > 512
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class RevalidationSnapshot:
    """Store exact sorted facts captured by the selected target context."""

    facts: tuple[RevalidationFact, ...]

    def __post_init__(self) -> None:
        """Require an exact nonempty deterministic fact inventory."""
        if (
            type(self.facts) is not tuple
            or not self.facts
            or any(type(item) is not RevalidationFact for item in self.facts)
            or self.facts
            != tuple(
                sorted(
                    self.facts,
                    key=lambda item: (
                        item.field.value,
                        item.key,
                        item.value,
                    ),
                )
            )
            or len(set(self.facts)) != len(self.facts)
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class RevalidationResult:
    """Return an exact stale witness without revealing target content."""

    matched: bool
    mismatch: RevalidationFact | None

    def __post_init__(self) -> None:
        """Keep mismatch presence equivalent to failed revalidation."""
        if self.matched is (self.mismatch is not None):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class LockFootprint:
    """Represent the sealed conservative workspace lock footprint."""

    domain_id: PatchDomainId
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require deterministic total order including a workspace key."""
        if (
            type(self.keys) is not tuple
            or not self.keys
            or self.keys[1:] != tuple(sorted(self.keys[1:]))
            or len(set(self.keys)) != len(self.keys)
            or self.keys[0] != "workspace"
            or any(not item or len(item) > 2048 for item in self.keys)
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class CommitLease:
    """Identify a sole worker ownership epoch under one domain lock."""

    domain_id: PatchDomainId
    request_id: PatchRequestId
    fence: int


@dataclass(frozen=True, slots=True)
class JournalStep:
    """Record one requested-effect state independent of staging artifacts."""

    identifier: PatchStepId
    lineage: PatchLineageId
    state: CommitStepState

    def __post_init__(self) -> None:
        """Reject unbound or transient terminal journal steps."""
        if (
            type(self.identifier) is not PatchStepId
            or type(self.lineage) is not PatchLineageId
            or type(self.state) is not CommitStepState
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class ArtifactJournal:
    """Record target-private or target-owned staging state separately."""

    identifier: str
    state: ArtifactState

    def __post_init__(self) -> None:
        """Reject unbound artifact state records."""
        if not self.identifier or type(self.state) is not ArtifactState:
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class SettlementJournal:
    """Store exact requested-effect and artifact facts for settlement."""

    steps: tuple[JournalStep, ...]
    artifacts: tuple[ArtifactJournal, ...]
    postcondition: PostconditionState

    def __post_init__(self) -> None:
        """Require a complete unique step vector and stable artifacts."""
        if (
            type(self.steps) is not tuple
            or not self.steps
            or any(type(item) is not JournalStep for item in self.steps)
            or len({item.identifier for item in self.steps}) != len(self.steps)
            or type(self.artifacts) is not tuple
            or any(
                type(item) is not ArtifactJournal for item in self.artifacts
            )
            or len({item.identifier for item in self.artifacts})
            != len(self.artifacts)
            or type(self.postcondition) is not PostconditionState
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


class WorkerState(str, Enum):
    """Name a scripted target response after commit ownership exists."""

    SETTLED = "settled"
    LIVE = "live"
    FENCED = "fenced"


@dataclass(frozen=True, slots=True)
class WorkerReport:
    """Return one scripted state and optional settlement evidence."""

    state: WorkerState
    journal: SettlementJournal | None

    def __post_init__(self) -> None:
        """Require a journal exactly when a worker has settled or fenced."""
        if (self.state is WorkerState.LIVE) is (self.journal is not None):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class SealedCommitCommand:
    """Carry only a trusted sealed plan and owner/fence witness to a worker."""

    plan: "SealedPlan"
    lease: CommitLease
    footprint: LockFootprint

    def __post_init__(self) -> None:
        """Bind the command to the plan's immutable coordination domain."""
        if self.plan.binding.target.domain_id != self.lease.domain_id or (
            self.footprint.domain_id != self.lease.domain_id
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class CoordinatorEvent:
    """Store one content-free, deduplicable canonical runtime event."""

    event: PatchLifecycleEvent
    stable_identity: str


@dataclass(frozen=True, slots=True)
class _AttachedPending:
    """Keep nonterminal settlement private to one attached invocation."""

    lifecycle: LifecyclePhase

    def __post_init__(self) -> None:
        """Reject any terminal state from the private pending algebra."""
        if self.lifecycle is not LifecyclePhase.SETTLEMENT_PENDING:
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    """Report owned resource depths at one deterministic protocol boundary."""

    transaction_depth: int
    lease_depth: int
    worker_depth: int
    target_handle_depth: int
    private_staging_depth: int
    approval_depth: int


class ScriptedFaultController:
    """Record deterministic boundaries and select bounded injected faults."""

    def __init__(
        self,
        failures: frozenset[CoordinatorBoundary] = frozenset(),
    ) -> None:
        """Bind an immutable set of scripted boundary failures."""
        if type(failures) is not frozenset or any(
            type(item) is not CoordinatorBoundary for item in failures
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
        self._failures = failures
        self._observed: list[CoordinatorBoundary] = []
        self._depths: list[RuntimeResources] = []

    @property
    def observed(self) -> tuple[CoordinatorBoundary, ...]:
        """Return the boundary order without a mutable controller view."""
        return tuple(self._observed)

    @property
    def depths(self) -> tuple[RuntimeResources, ...]:
        """Return resource depth witnesses in boundary-observation order."""
        return tuple(self._depths)

    async def checkpoint(
        self,
        boundary: CoordinatorBoundary,
        resources: RuntimeResources,
    ) -> bool:
        """Record one boundary and return whether its script fails closed."""
        if (
            type(boundary) is not CoordinatorBoundary
            or type(resources) is not RuntimeResources
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
        self._observed.append(boundary)
        self._depths.append(resources)
        return boundary in self._failures


class CoordinatorRegistry(Protocol):
    """Reserve and retrieve one idempotent runtime coordination record."""

    async def reserve(
        self, identity: RuntimeIdentity, digest: AlgorithmDigest
    ) -> Reservation:
        """Reserve before target inspection and return replay provenance."""


class LockLeaseManager(Protocol):
    """Acquire one domain-wide lease in deterministic footprint order."""

    async def acquire(
        self, footprint: LockFootprint, reservation: Reservation
    ) -> CommitLease:
        """Acquire a workspace-wide serialized commit lease."""

    async def release(self, lease: CommitLease) -> None:
        """Release one active ownership lease after settlement."""


class IdempotencyStore(Protocol):
    """Atomically bind grant consumption, owner, fence, and commit start."""

    async def begin_commit(
        self,
        reservation: Reservation,
        plan: "SealedPlan",
        grant: "PlanBoundGrant",
        lease: CommitLease,
    ) -> None:
        """Persist sole commit ownership before a possible visible effect."""


class JournalStore(Protocol):
    """Persist a bounded append-only requested-effect settlement journal."""

    async def append(
        self, request_id: PatchRequestId, journal: SettlementJournal
    ) -> None:
        """Append one exact final journal for a commit owner."""


class CommitWorker(Protocol):
    """Execute a sealed command through an asynchronous target boundary."""

    async def commit(self, command: SealedCommitCommand) -> WorkerReport:
        """Return a scripted worker state without filesystem handles."""


class RootedCommitChannel(Protocol):
    """Execute a command in an already-authenticated rooted worker."""

    async def commit_local(self, command: SealedCommitCommand) -> WorkerReport:
        """Return a journal-derived report from the rooted target."""

    async def reconcile_local(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Return retained local settlement truth without another commit."""


@runtime_checkable
class RootedSandboxCommitChannel(Protocol):
    """Execute one command through an authenticated sandbox worker channel."""

    async def commit_sandbox(
        self,
        command: SealedCommitCommand,
        validator: "RootedCommandAuthorityValidator",
    ) -> WorkerReport:
        """Return a journal-derived report from the sandbox target."""

    async def reconcile_sandbox(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Return retained sandbox truth without another commit."""


@dataclass(frozen=True, slots=True, eq=False)
class _RootedWorkerAuthorization:
    """Keep local mutation-worker construction inside target runtime code."""

    token: object = field(default_factory=object)


@final
@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class RootedLocalCommitWorker:
    """Forward a sealed command only to a target-minted local channel."""

    _channel: RootedCommitChannel
    _authorization: _RootedWorkerAuthorization

    async def commit(self, command: SealedCommitCommand) -> WorkerReport:
        """Execute one sealed command without exposing target handles."""
        if await _consume_rooted_command_authority(command) is None:
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        return await self._channel.commit_local(command)

    async def _reconcile_for_owner(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Read a retained local report without replaying its command."""
        return await self._channel.reconcile_local(request_id)


@final
@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class RootedSandboxCommitWorker:
    """Forward commands through a sealed context-owned sandbox endpoint."""

    async def commit(self, command: SealedCommitCommand) -> WorkerReport:
        """Execute one sealed sandbox command without target handles."""
        endpoint = _ROOTED_SANDBOX_WORKERS.get(self)
        validator = await _consume_rooted_command_authority(command)
        if endpoint is None or validator is None:
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        active = _ACTIVE_SANDBOX_WORKER.set(self)
        try:
            return await endpoint.commit_sandbox(command, validator)
        finally:
            _ACTIVE_SANDBOX_WORKER.reset(active)

    async def _reconcile_for_owner(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Read retained sandbox truth without replaying its command."""
        endpoint = _ROOTED_SANDBOX_WORKERS.get(self)
        if endpoint is None:
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        active = _ACTIVE_SANDBOX_WORKER.set(self)
        try:
            return await endpoint.reconcile_sandbox(request_id)
        finally:
            _ACTIVE_SANDBOX_WORKER.reset(active)


@final
@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True, repr=False)
class _RootedSandboxEndpoint:
    """Retain the only channel reference outside a public worker object."""

    _channel: RootedSandboxCommitChannel

    async def commit_sandbox(
        self,
        command: SealedCommitCommand,
        validator: "RootedCommandAuthorityValidator",
    ) -> WorkerReport:
        """Forward one authenticated command to the context endpoint."""
        worker = _ACTIVE_SANDBOX_WORKER.get()
        if worker is None or _ROOTED_SANDBOX_WORKERS.get(worker) is not self:
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        return await self._channel.commit_sandbox(command, validator)

    async def reconcile_sandbox(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Read one retained settlement report from the context endpoint."""
        worker = _ACTIVE_SANDBOX_WORKER.get()
        if worker is None or _ROOTED_SANDBOX_WORKERS.get(worker) is not self:
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        return await self._channel.reconcile_sandbox(request_id)


@runtime_checkable
class RootedCommandAuthorityValidator(Protocol):
    """Validate one worker command against its live ownership record."""

    async def is_rooted_command_current(
        self, command: SealedCommitCommand
    ) -> bool:
        """Return whether this exact command still has live authority."""


@dataclass(frozen=True, slots=True)
class _RootedCommandAuthority:
    """Bind a one-shot rooted command to a live private validator."""

    validator: RootedCommandAuthorityValidator
    nonce: object = field(default_factory=object)


_ROOTED_LOCAL_WORKERS: WeakKeyDictionary[
    RootedLocalCommitWorker, _RootedWorkerAuthorization
] = WeakKeyDictionary()
_ROOTED_SANDBOX_WORKERS: WeakKeyDictionary[
    RootedSandboxCommitWorker, _RootedSandboxEndpoint
] = WeakKeyDictionary()
_ROOTED_SANDBOX_ENDPOINTS: WeakKeyDictionary[
    _RootedSandboxEndpoint, object
] = WeakKeyDictionary()
_ACTIVE_SANDBOX_WORKER: ContextVar[RootedSandboxCommitWorker | None] = (
    ContextVar("active_sandbox_commit_worker", default=None)
)
_ROOTED_COMMAND_AUTHORITIES: WeakKeyDictionary[
    SealedCommitCommand, _RootedCommandAuthority
] = WeakKeyDictionary()


def _rooted_local_worker(
    channel: RootedCommitChannel,
) -> RootedLocalCommitWorker:
    """Construct the sole non-scripted commit worker accepted in this phase."""
    worker = RootedLocalCommitWorker(channel, _RootedWorkerAuthorization())
    _ROOTED_LOCAL_WORKERS[worker] = worker._authorization
    return worker


def _rooted_sandbox_endpoint(
    channel: RootedSandboxCommitChannel,
) -> _RootedSandboxEndpoint:
    """Seal one typed context endpoint before public-worker construction."""
    if not (
        _PatchAuthorityValidator.sandbox_endpoint_is_issued(channel)
        or _PatchAuthorityValidator.container_endpoint_is_issued(channel)
    ):
        raise CoordinatorError(CoordinatorErrorCode.FENCED)
    endpoint = _RootedSandboxEndpoint(channel)
    _ROOTED_SANDBOX_ENDPOINTS[endpoint] = object()
    return endpoint


def _sandbox_worker_for_endpoint(
    endpoint: _RootedSandboxEndpoint,
) -> RootedSandboxCommitWorker:
    """Return a worker only for a coordinator-sealed sandbox endpoint."""
    if (
        type(endpoint) is not _RootedSandboxEndpoint
        or endpoint not in _ROOTED_SANDBOX_ENDPOINTS
    ):
        raise CoordinatorError(CoordinatorErrorCode.FENCED)
    worker = RootedSandboxCommitWorker()
    _ROOTED_SANDBOX_WORKERS[worker] = endpoint
    return worker


def _is_rooted_local_worker(
    worker: CommitWorker,
) -> TypeGuard[RootedLocalCommitWorker]:
    """Accept only a worker minted by the private rooted-worker factory."""
    return (
        type(worker) is RootedLocalCommitWorker
        and _ROOTED_LOCAL_WORKERS.get(worker) is worker._authorization
    )


def _is_rooted_sandbox_worker(
    worker: CommitWorker,
) -> TypeGuard[RootedSandboxCommitWorker]:
    """Accept only a worker minted by the private sandbox-worker factory."""
    return (
        type(worker) is RootedSandboxCommitWorker
        and _ROOTED_SANDBOX_WORKERS.get(worker) is not None
    )


def _is_rooted_target_worker(
    worker: CommitWorker,
) -> TypeGuard[RootedLocalCommitWorker | RootedSandboxCommitWorker]:
    """Return whether one trusted target worker owns this sealed route."""
    return _is_rooted_local_worker(worker) or _is_rooted_sandbox_worker(worker)


async def _issue_rooted_command_authority(
    command: SealedCommitCommand, store: "InMemoryCoordinatorStore"
) -> None:
    """Bind one command identity to a coordinator-owned current-fence check."""
    await _issue_rooted_command_authority_for_validator(
        command, _InMemoryRootedCommandAuthorityValidator(store)
    )


async def _issue_rooted_command_authority_for_validator(
    command: SealedCommitCommand,
    validator: RootedCommandAuthorityValidator,
) -> None:
    """Mint one worker authority after its trusted owner verifies it live."""
    if (
        not isinstance(validator, RootedCommandAuthorityValidator)
        or command in _ROOTED_COMMAND_AUTHORITIES
        or not await validator.is_rooted_command_current(command)
    ):
        raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
    _ROOTED_COMMAND_AUTHORITIES[command] = _RootedCommandAuthority(validator)


async def _consume_rooted_command_authority(
    command: SealedCommitCommand,
) -> RootedCommandAuthorityValidator | None:
    """Consume authority only while its owner/fence remains current."""
    authority = _ROOTED_COMMAND_AUTHORITIES.pop(command, None)
    if authority is None:
        return None
    if (
        authority.nonce is None
        or not await authority.validator.is_rooted_command_current(command)
    ):
        return None
    return authority.validator


class Reconciler(Protocol):
    """Revalidate and reconcile an owned commit without guessing truth."""

    async def revalidate(
        self, expected: RevalidationSnapshot
    ) -> RevalidationResult:
        """Compare every sealed precondition before commit starts."""

    async def reconcile(self, request_id: PatchRequestId) -> WorkerReport:
        """Return a settled or still-live worker state for one request."""


class EventOutbox(Protocol):
    """Append content-free lifecycle events after journal transactions."""

    async def append(self, event: CoordinatorEvent) -> None:
        """Persist one at-least-once transport event identity and sequence."""


class TrustedGrantValidator(Protocol):
    """Validate one issued plan grant at the atomic commit boundary."""

    async def validate_grant(
        self,
        grant: PlanBoundGrant,
        plan: SealedPlan,
        subject: ExecutionSubject,
    ) -> None:
        """Validate issued membership, expiry, and every sealed binding."""


class TerminalPublisher(Protocol):
    """Publish exactly one logical terminal result after settlement."""

    async def publish(self, result: PatchResult) -> PatchResult:
        """Persist and return the one canonical terminal result."""


@dataclass(slots=True)
class _RuntimeRecord:
    """Keep mutable state private to the attached in-memory coordinator."""

    reservation: Reservation
    lifecycle: LifecyclePhase = LifecyclePhase.RECEIVED
    plan: SealedPlan | None = None
    grant: PlanBoundGrant | None = None
    lease: CommitLease | None = None
    journal: SettlementJournal | None = None
    result: PatchResult | None = None
    pending: _AttachedPending | None = None
    pending_owner: str | None = None
    commit_worker: (
        RootedLocalCommitWorker | RootedSandboxCommitWorker | None
    ) = None
    cancellation_requested: bool = False
    sequence: int = 0
    events: list[CoordinatorEvent] = field(default_factory=list)
    completed: Event = field(default_factory=Event)
    state_changed: Event = field(default_factory=Event)
    settlement_lock: Lock = field(default_factory=Lock)
    lease_released: bool = False


class InMemoryCoordinatorStore:
    """Provide deterministic attached records with no durable handle."""

    def __init__(
        self,
        grant_validator: TrustedGrantValidator | None = None,
        *,
        _precommit_checkpoint: _PrecommitCheckpoint | None = None,
    ) -> None:
        """Initialize empty private records and the short in-memory lock."""
        self._lock = Lock()
        self._records: dict[RuntimeIdentity, _RuntimeRecord] = {}
        self._by_request: dict[PatchRequestId, _RuntimeRecord] = {}
        self._fences: dict[PatchDomainId, int] = {}
        self._consumed_grants: set[str] = set()
        self._grant_validator = grant_validator
        self._precommit_checkpoint = _precommit_checkpoint

    async def _checkpoint(self, boundary: _PrecommitBoundary) -> None:
        """Run an unavailable-by-default local test-profile checkpoint."""
        checkpoint = self._precommit_checkpoint
        if checkpoint is not None:
            await checkpoint.checkpoint(boundary)

    async def reserve(
        self, identity: RuntimeIdentity, digest: AlgorithmDigest
    ) -> Reservation:
        """Reserve a request identity or attach only to the same digest."""
        await self._checkpoint(_PrecommitBoundary.RESERVE_REQUEST)
        async with self._lock:
            current = self._records.get(identity)
            if current is not None:
                if current.reservation.digest != digest:
                    raise CoordinatorError(
                        CoordinatorErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return Reservation(
                    current.reservation.request_id,
                    identity,
                    digest,
                    replayed=True,
                )
            reservation = Reservation(
                PatchRequestId.new(), identity, digest, replayed=False
            )
            record = _RuntimeRecord(reservation)
            self._records[identity] = record
            self._by_request[reservation.request_id] = record
            return reservation

    async def record(self, reservation: Reservation) -> _RuntimeRecord:
        """Return the private record only when its reservation is exact."""
        async with self._lock:
            record = self._records.get(reservation.identity)
            if (
                record is None
                or record.reservation.request_id != reservation.request_id
            ):
                raise CoordinatorError(
                    CoordinatorErrorCode.INVALID_RESERVATION
                )
            return record

    async def assign_lease(
        self, reservation: Reservation, domain_id: PatchDomainId
    ) -> CommitLease:
        """Advance a per-domain monotonically increasing fence epoch."""
        async with self._lock:
            record = await self._record_locked(reservation)
            if record.lease is not None:
                return record.lease
            await self._checkpoint(_PrecommitBoundary.INTENT_FENCE)
            fence = self._fences.get(domain_id, 0) + 1
            self._fences[domain_id] = fence
            lease = CommitLease(domain_id, reservation.request_id, fence)
            record.lease = lease
            return lease

    async def begin_commit(
        self,
        reservation: Reservation,
        plan: SealedPlan,
        grant: PlanBoundGrant,
        lease: CommitLease,
    ) -> None:
        """Consume one grant and establish owner before the first effect."""
        async with self._lock:
            record = await self._record_locked(reservation)
            if record.result is not None:
                raise CoordinatorError(CoordinatorErrorCode.RETRY_BLOCKED)
            if record.lease != lease or record.lifecycle not in {
                LifecyclePhase.APPROVED,
                LifecyclePhase.COMMIT_READY,
            }:
                raise CoordinatorError(CoordinatorErrorCode.FENCED)
            validator = self._grant_validator
            if validator is None:
                raise CoordinatorError(CoordinatorErrorCode.STALE)
            try:
                await validator.validate_grant(
                    grant,
                    plan,
                    reservation.identity.subject,
                )
            except PolicyError as exc:
                raise CoordinatorError(CoordinatorErrorCode.STALE) from exc
            await self._checkpoint(_PrecommitBoundary.CONSUME_GRANT)
            if grant.grant_id.value in self._consumed_grants:
                raise CoordinatorError(CoordinatorErrorCode.GRANT_CONSUMED)
            self._consumed_grants.add(grant.grant_id.value)
            await self._checkpoint(_PrecommitBoundary.ASSIGN_COMMIT_OWNER)
            record.plan = plan
            record.grant = grant
            record.lifecycle = LifecyclePhase.COMMIT_STARTED

    async def append(
        self, request_id: PatchRequestId, journal: SettlementJournal
    ) -> None:
        """Append one exact journal while a request remains nonterminal."""
        async with self._lock:
            record = self._by_request.get(request_id)
            if record is None or record.lifecycle not in {
                LifecyclePhase.COMMIT_STARTED,
                LifecyclePhase.SETTLEMENT_PENDING,
            }:
                raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
            if record.journal is not None:
                raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
            record.journal = journal
            record.lifecycle = LifecyclePhase.SETTLED

    async def is_current(self, lease: CommitLease) -> bool:
        """Return whether a lease still owns the newest domain fence."""
        async with self._lock:
            record = self._by_request.get(lease.request_id)
            return (
                record is not None
                and record.lease == lease
                and self._fences.get(lease.domain_id) == lease.fence
            )

    async def authorize(self, command: SealedCommitCommand) -> bool:
        """Return whether a command exactly names a persisted commit owner."""
        async with self._lock:
            record = self._by_request.get(command.lease.request_id)
            return (
                record is not None
                and record.plan is command.plan
                and record.lease == command.lease
                and record.lifecycle is LifecyclePhase.COMMIT_STARTED
                and self._fences.get(command.lease.domain_id)
                == command.lease.fence
            )

    async def terminal(self, request_id: PatchRequestId) -> PatchResult | None:
        """Return a completed result without changing replay state."""
        async with self._lock:
            record = self._by_request.get(request_id)
            return None if record is None else record.result

    async def _record_locked(self, reservation: Reservation) -> _RuntimeRecord:
        """Resolve a reservation while the private lock is already owned."""
        record = self._records.get(reservation.identity)
        if (
            record is None
            or record.reservation.request_id != reservation.request_id
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVALID_RESERVATION)
        return record


@dataclass(frozen=True, slots=True, repr=False)
class _InMemoryRootedCommandAuthorityValidator:
    """Adapt the attached coordinator record to the rooted-worker check."""

    store: InMemoryCoordinatorStore

    async def is_rooted_command_current(
        self, command: SealedCommitCommand
    ) -> bool:
        """Return whether the attached owner still authorizes this command."""
        return await self.store.authorize(command)


class InMemoryLeaseManager:
    """Serialize commits by backing-domain rather than context identity."""

    def __init__(self, store: InMemoryCoordinatorStore) -> None:
        """Bind short locks and ownership bookkeeping to the store."""
        self._store = store
        self._locks: dict[PatchDomainId, Lock] = {}
        self._leases: dict[PatchRequestId, Lock] = {}

    async def acquire(
        self, footprint: LockFootprint, reservation: Reservation
    ) -> CommitLease:
        """Acquire the workspace-wide domain lock before a commit attempt."""
        lock = self._locks.setdefault(footprint.domain_id, Lock())
        await lock.acquire()
        await self._store._checkpoint(_PrecommitBoundary.ACQUIRE_LOCK)
        self._leases[reservation.request_id] = lock
        return await self._store.assign_lease(reservation, footprint.domain_id)

    async def release(self, lease: CommitLease) -> None:
        """Release only the matching request's current domain lease."""
        lock = self._leases.pop(lease.request_id, None)
        if lock is None or not lock.locked():
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        await self._store._checkpoint(_PrecommitBoundary.RELEASE_LOCK)
        lock.release()

    async def is_current(self, lease: CommitLease) -> bool:
        """Return whether a worker lease remains the current fence witness."""
        return await self._store.is_current(lease)


class ScriptedReconciler:
    """Compare target-private facts and scripted worker reports."""

    def __init__(self, current: RevalidationSnapshot) -> None:
        """Initialize current revalidation facts with no target handle."""
        self._current = current
        self._reports: dict[PatchRequestId, WorkerReport] = {}

    def replace_current(self, current: RevalidationSnapshot) -> None:
        """Replace current facts to model a foreign writer or remount."""
        self._current = current

    def set_report(
        self, request_id: PatchRequestId, report: WorkerReport
    ) -> None:
        """Set deterministic reconciliation evidence for a live worker."""
        self._reports[request_id] = report

    async def revalidate(
        self, expected: RevalidationSnapshot
    ) -> RevalidationResult:
        """Return the first deterministic changed fact without target I/O."""
        current = set(self._current.facts)
        for fact in expected.facts:
            if fact not in current:
                return RevalidationResult(False, fact)
        if set(expected.facts) != current:
            return RevalidationResult(False, self._current.facts[0])
        return RevalidationResult(True, None)

    async def reconcile(self, request_id: PatchRequestId) -> WorkerReport:
        """Return scripted live, fenced, or settled request evidence."""
        return self._reports.get(
            request_id, WorkerReport(WorkerState.LIVE, None)
        )


class ScriptedCommitWorker:
    """Execute planned step vectors without filesystem authority."""

    def __init__(
        self,
        report: WorkerReport,
        started: Event | None = None,
        continue_commit: Event | None = None,
    ) -> None:
        """Bind one finite outcome script to this incapable worker."""
        self._report = report
        self._started = started
        self._continue_commit = continue_commit
        self.commands: list[SealedCommitCommand] = []

    async def commit(self, command: SealedCommitCommand) -> WorkerReport:
        """Record one sealed command and return the configured outcome."""
        self.commands.append(command)
        if self._started is not None:
            self._started.set()
        if self._continue_commit is not None:
            await self._continue_commit.wait()
        return self._report


class InMemoryPatchCoordinator:
    """Run an attached, non-retryable scripted commit lifecycle."""

    scheduler_parallel_safe = False

    def __init__(
        self,
        store: InMemoryCoordinatorStore,
        leases: InMemoryLeaseManager,
        reconciler: ScriptedReconciler,
        faults: ScriptedFaultController | None = None,
    ) -> None:
        """Bind stores that never advertise durable continuation."""
        self._store = store
        self._leases = leases
        self._reconciler = reconciler
        self._faults = faults or ScriptedFaultController()
        self._resource = RuntimeResources(0, 0, 0, 0, 0, 0)

    @property
    def resources(self) -> RuntimeResources:
        """Return current in-memory resource depths for tests."""
        return self._resource

    async def reserve(
        self, identity: RuntimeIdentity, digest: AlgorithmDigest
    ) -> Reservation:
        """Reserve before planning and reject conflicts without target use."""
        return await self._store.reserve(identity, digest)

    async def events(
        self, reservation: Reservation
    ) -> tuple[CoordinatorEvent, ...]:
        """Return the attached lifecycle-event sequence for one request."""
        record = await self._store.record(reservation)
        return tuple(record.events)

    async def replay_inert_history(self, records: tuple[bytes, ...]) -> None:
        """Discard provider-looking bytes without reserving a request."""
        if type(records) is not tuple or any(
            type(item) is not bytes for item in records
        ):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)

    async def advance(
        self, reservation: Reservation, next_phase: LifecyclePhase
    ) -> LifecyclePhase:
        """Advance a legal precommit phase without worker authority."""
        record = await self._store.record(reservation)
        allowed = {
            LifecyclePhase.RECEIVED: {
                LifecyclePhase.PARSED,
                LifecyclePhase.REQUEST_COMPLETED,
            },
            LifecyclePhase.PARSED: {
                LifecyclePhase.SCOPE_BOUND,
                LifecyclePhase.REQUEST_COMPLETED,
            },
            LifecyclePhase.SCOPE_BOUND: {
                LifecyclePhase.PREFLIGHT_AUTHORIZED,
                LifecyclePhase.REQUEST_COMPLETED,
            },
            LifecyclePhase.PREFLIGHT_AUTHORIZED: {
                LifecyclePhase.PLANNED,
                LifecyclePhase.REQUEST_COMPLETED,
            },
            LifecyclePhase.PLANNED: {
                LifecyclePhase.APPROVAL_REQUIRED,
                LifecyclePhase.APPROVED,
                LifecyclePhase.REQUEST_COMPLETED,
            },
            LifecyclePhase.APPROVAL_REQUIRED: {
                LifecyclePhase.APPROVED,
                LifecyclePhase.REQUEST_COMPLETED,
            },
            LifecyclePhase.APPROVED: {
                LifecyclePhase.COMMIT_READY,
                LifecyclePhase.REQUEST_COMPLETED,
            },
        }
        if next_phase not in allowed.get(record.lifecycle, set()):
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
        record.lifecycle = next_phase
        await self._emit(record, next_phase)
        return record.lifecycle

    async def prepare(
        self,
        reservation: Reservation,
        plan: SealedPlan,
        *,
        approval_required: bool,
    ) -> LifecyclePhase:
        """Bind a sealed plan and advance only through precommit authority."""
        record = await self._store.record(reservation)
        if reservation.digest != plan.binding.request_digest:
            raise CoordinatorError(CoordinatorErrorCode.STALE)
        if record.plan is None:
            await self._store._checkpoint(_PrecommitBoundary.PERSIST_PLAN)
            record.plan = plan
        elif record.plan is not plan:
            raise CoordinatorError(CoordinatorErrorCode.STALE)
        while record.lifecycle in {
            LifecyclePhase.RECEIVED,
            LifecyclePhase.PARSED,
            LifecyclePhase.SCOPE_BOUND,
            LifecyclePhase.PREFLIGHT_AUTHORIZED,
        }:
            transition = {
                LifecyclePhase.RECEIVED: LifecyclePhase.PARSED,
                LifecyclePhase.PARSED: LifecyclePhase.SCOPE_BOUND,
                LifecyclePhase.SCOPE_BOUND: (
                    LifecyclePhase.PREFLIGHT_AUTHORIZED
                ),
                LifecyclePhase.PREFLIGHT_AUTHORIZED: LifecyclePhase.PLANNED,
            }[record.lifecycle]
            await self.advance(reservation, transition)
            if transition is LifecyclePhase.PLANNED:
                await self._store._checkpoint(
                    _PrecommitBoundary.LIFECYCLE_PLANNED
                )
        if record.lifecycle is LifecyclePhase.PLANNED:
            if approval_required:
                await self._store._checkpoint(
                    _PrecommitBoundary.LIFECYCLE_AWAITING_APPROVAL
                )
            await self.advance(
                reservation,
                (
                    LifecyclePhase.APPROVAL_REQUIRED
                    if approval_required
                    else LifecyclePhase.APPROVED
                ),
            )
        return record.lifecycle

    async def execute(
        self,
        reservation: Reservation,
        plan: SealedPlan,
        grant: PlanBoundGrant,
        expected: RevalidationSnapshot,
        worker: CommitWorker,
        controller: str,
    ) -> PatchResult | _AttachedPending:
        """Commit a sealed plan once or return attached pending state."""
        if type(
            worker
        ) is not ScriptedCommitWorker and not _is_rooted_target_worker(worker):
            raise CoordinatorError(CoordinatorErrorCode.SCRIPTED_TARGET_ONLY)
        record = await self._store.record(reservation)
        if reservation.digest != plan.binding.request_digest:
            raise CoordinatorError(CoordinatorErrorCode.STALE)
        if record.plan is None:
            record.plan = plan
        elif record.plan is not plan:
            raise CoordinatorError(CoordinatorErrorCode.STALE)
        if record.result is not None:
            return record.result
        if record.pending is not None:
            return await self._continue_pending(
                record, reservation, controller, worker
            )
        if record.lifecycle is LifecyclePhase.COMMIT_STARTED:
            await record.state_changed.wait()
            record = await self._store.record(reservation)
            if record.result is not None:
                return record.result
            if record.pending is not None:
                return await self._continue_pending(
                    record, reservation, controller, worker
                )
            raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
        if record.lifecycle is LifecyclePhase.RECEIVED:
            await self.prepare(
                reservation,
                plan,
                approval_required=False,
            )
        if record.lifecycle is LifecyclePhase.APPROVED:
            await self.advance(reservation, LifecyclePhase.COMMIT_READY)
        if record.lifecycle is not LifecyclePhase.COMMIT_READY:
            raise CoordinatorError(CoordinatorErrorCode.RETRY_BLOCKED)
        footprint = footprint_for(plan)
        lease = await self._leases.acquire(footprint, reservation)
        release_lease = True
        self._resource = RuntimeResources(0, 1, 0, 0, 0, 0)
        try:
            record = await self._store.record(reservation)
            if await self._faulted(
                CoordinatorBoundary.PRIVATE_STAGING
            ) or await self._faulted(CoordinatorBoundary.LEASE):
                return await self._complete_precommit(
                    record, plan, PatchStatus.STALE, PatchErrorCode.STALE
                )
            revalidation = await self._reconciler.revalidate(expected)
            if not revalidation.matched or await self._faulted(
                CoordinatorBoundary.REVALIDATION
            ):
                return await self._complete_precommit(
                    record, plan, PatchStatus.STALE, PatchErrorCode.STALE
                )
            await self._store._checkpoint(
                _PrecommitBoundary.LIFECYCLE_COMMIT_OWNER
            )
            await self._store.begin_commit(reservation, plan, grant, lease)
            record = await self._store.record(reservation)
            record.commit_worker = (
                worker if _is_rooted_target_worker(worker) else None
            )
            await self._emit(record, LifecyclePhase.COMMIT_STARTED)
            try:
                if await self._faulted(CoordinatorBoundary.OWNER) or (
                    await self._faulted(CoordinatorBoundary.FENCE)
                ):
                    raise CoordinatorError(CoordinatorErrorCode.FENCED)
                command = SealedCommitCommand(plan, lease, footprint)
                if await self._faulted(
                    CoordinatorBoundary.VISIBLE_ARTIFACT
                ) or await self._faulted(CoordinatorBoundary.REQUESTED_EFFECT):
                    raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
                if not await self._leases.is_current(lease):
                    raise CoordinatorError(CoordinatorErrorCode.FENCED)
                if _is_rooted_target_worker(worker):
                    await _issue_rooted_command_authority(command, self._store)
                self._resource = RuntimeResources(0, 1, 1, 0, 1, 0)
                report = await worker.commit(command)
                self._resource = RuntimeResources(0, 1, 0, 0, 0, 0)
                if report.state is WorkerState.LIVE:
                    raise CoordinatorError(CoordinatorErrorCode.STALE)
                assert report.journal is not None
                for boundary in (
                    CoordinatorBoundary.JOURNAL,
                    CoordinatorBoundary.VERIFICATION,
                    CoordinatorBoundary.CLEANUP,
                    CoordinatorBoundary.SETTLEMENT,
                    CoordinatorBoundary.OUTBOX,
                    CoordinatorBoundary.TERMINAL_PUBLICATION,
                ):
                    if await self._faulted(boundary):
                        raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
                return await self._settle(record, plan, report.journal)
            except (CancelledError, Exception):
                release_lease = False
                return await self._enter_pending(
                    record, reservation, controller
                )
        finally:
            if release_lease:
                await self._release_lease(record, lease)

    async def cancel(
        self, reservation: Reservation, before_commit: bool
    ) -> PatchResult | _AttachedPending:
        """Cancel before commit or request owned settlement."""
        record = await self._store.record(reservation)
        if record.result is not None:
            return record.result
        if (
            before_commit
            and record.lifecycle is not LifecyclePhase.COMMIT_STARTED
        ):
            if record.plan is None:
                raise CoordinatorError(CoordinatorErrorCode.RETRY_BLOCKED)
            await self._store._checkpoint(_PrecommitBoundary.CANCELLATION)
            return await self._complete_precommit(
                record,
                record.plan,
                PatchStatus.CANCELLED,
                PatchErrorCode.CANCELLED,
            )
        record.cancellation_requested = True
        if record.pending is not None:
            return record.pending
        raise CoordinatorError(CoordinatorErrorCode.RETRY_BLOCKED)

    async def _timeout_before_commit(
        self, reservation: Reservation
    ) -> PatchResult | _AttachedPending:
        """Run the timeout cancellation path through its owned checkpoint."""
        await self._store._checkpoint(_PrecommitBoundary.TIMEOUT)
        return await self.cancel(reservation, before_commit=True)

    async def _disconnect_before_commit(
        self, reservation: Reservation
    ) -> PatchResult | _AttachedPending:
        """Run disconnect cancellation through its owned checkpoint."""
        await self._store._checkpoint(_PrecommitBoundary.DISCONNECT)
        return await self.cancel(reservation, before_commit=True)

    async def _continue_pending(
        self,
        record: _RuntimeRecord,
        reservation: Reservation,
        controller: str,
        worker: CommitWorker,
    ) -> PatchResult | _AttachedPending:
        """Reconcile one attached pending owner without a public handle."""
        if record.pending is None or record.pending_owner != controller:
            raise CoordinatorError(CoordinatorErrorCode.PENDING_OWNER)
        if (
            record.commit_worker is not None
            and record.commit_worker is not worker
        ):
            raise CoordinatorError(CoordinatorErrorCode.FENCED)
        async with record.settlement_lock:
            if record.result is not None:
                return record.result
            pending = record.pending
            plan = record.plan
            if pending is None or plan is None:
                raise CoordinatorError(CoordinatorErrorCode.INVARIANT)
            self._resource = RuntimeResources(0, 1, 0, 0, 0, 0)
            try:
                original_worker = record.commit_worker
                report = (
                    await original_worker._reconcile_for_owner(
                        reservation.request_id
                    )
                    if original_worker is not None
                    else await self._reconciler.reconcile(
                        reservation.request_id
                    )
                )
                if report.state is WorkerState.LIVE:
                    return pending
                assert report.journal is not None
                try:
                    return await self._settle(record, plan, report.journal)
                except CoordinatorError:
                    return pending
            except (CancelledError, Exception):
                return pending
            finally:
                if record.result is not None:
                    assert record.lease is not None
                    await self._release_lease(record, record.lease)

    async def _release_lease(
        self, record: _RuntimeRecord, lease: CommitLease
    ) -> None:
        """Release a finished owned lease no more than once."""
        self._resource = RuntimeResources(0, 0, 0, 0, 0, 0)
        if record.lease_released:
            return
        record.lease_released = True
        try:
            await self._leases.release(lease)
        except CoordinatorError:
            return

    async def _faulted(self, boundary: CoordinatorBoundary) -> bool:
        """Return whether one deterministic coordinator boundary failed."""
        return await self._faults.checkpoint(boundary, self._resource)

    async def _enter_pending(
        self,
        record: _RuntimeRecord,
        reservation: Reservation,
        controller: str,
    ) -> _AttachedPending:
        """Retain a fenced attached settlement without a detached handle."""
        if record.pending is not None:
            if record.pending_owner != controller:
                raise CoordinatorError(CoordinatorErrorCode.PENDING_OWNER)
            return record.pending
        pending = _AttachedPending(LifecyclePhase.SETTLEMENT_PENDING)
        record.lifecycle = LifecyclePhase.SETTLEMENT_PENDING
        record.pending = pending
        record.pending_owner = controller
        await self._emit(record, LifecyclePhase.SETTLEMENT_PENDING)
        record.state_changed.set()
        return pending

    async def _complete_precommit(
        self,
        record: _RuntimeRecord,
        plan: SealedPlan,
        status: PatchStatus,
        code: PatchErrorCode,
    ) -> PatchResult:
        """Create a zero-write terminal record prior to commit ownership."""
        truth = _truth((), ArtifactState.ABSENT, PostconditionState.UNKNOWN)
        result = PatchResult(
            1,
            record.reservation.request_id,
            plan.plan_id,
            LifecyclePhase.REQUEST_COMPLETED,
            status,
            truth,
            PatchDiagnostic(
                ErrorStage.REVALIDATION, code, Retryability.NOT_RETRYABLE
            ),
        )
        record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
        record.result = result
        await self._store._checkpoint(_PrecommitBoundary.LIFECYCLE_COMPLETED)
        await self._emit(record, LifecyclePhase.REQUEST_COMPLETED)
        record.completed.set()
        record.state_changed.set()
        return result

    async def _settle(
        self,
        record: _RuntimeRecord,
        plan: SealedPlan,
        journal: SettlementJournal,
    ) -> PatchResult:
        """Append exact journal facts then create the one terminal outcome."""
        _validate_settlement_journal(plan, journal)
        await self._store.append(record.reservation.request_id, journal)
        await self._emit(record, LifecyclePhase.SETTLED)
        truth = _truth(
            tuple(item.state for item in journal.steps),
            _artifact_state(journal.artifacts),
            journal.postcondition,
        )
        status, code = _status(truth.mutation_state)
        diagnostic = (
            None
            if code is None
            else PatchDiagnostic(
                ErrorStage.SETTLEMENT, code, Retryability.NOT_RETRYABLE
            )
        )
        result = PatchResult(
            1,
            record.reservation.request_id,
            plan.plan_id,
            LifecyclePhase.REQUEST_COMPLETED,
            status,
            truth,
            diagnostic,
        )
        record.lifecycle = LifecyclePhase.REQUEST_COMPLETED
        record.pending = None
        record.commit_worker = None
        record.result = result
        await self._emit(record, LifecyclePhase.REQUEST_COMPLETED)
        record.completed.set()
        record.state_changed.set()
        return result

    async def _emit(
        self, record: _RuntimeRecord, lifecycle: LifecyclePhase
    ) -> None:
        """Append one stable event after result settlement."""
        record.sequence += 1
        event = PatchLifecycleEvent(
            1,
            PatchEventId.new(),
            PatchObserverId.new(),
            PatchObserverCorrelationId.new(),
            record.reservation.request_id,
            SequenceNumber(record.sequence),
            lifecycle,
        )
        record.events.append(
            CoordinatorEvent(
                event,
                f"{record.reservation.request_id.value}:{event.sequence.value}",
            )
        )


def footprint_for(plan: SealedPlan) -> LockFootprint:
    """Derive a workspace-wide lock footprint from sealed lineages."""
    keys = {"workspace"}
    for lineage in plan.candidate.lineages:
        for path in (
            lineage.source_path,
            lineage.destination_path,
            lineage.initial.path,
            lineage.final.path,
            *lineage.parent_paths,
            *lineage.lock_footprint,
        ):
            if path is not None:
                keys.add(path.value)
    return LockFootprint(
        plan.binding.target.domain_id,
        ("workspace", *tuple(sorted(keys - {"workspace"}))),
    )


def _sealed_journal_steps(
    plan: SealedPlan,
) -> tuple[tuple[PatchStepId, PatchLineageId], ...]:
    """Derive the exact ordered step identities from one sealed plan."""
    return tuple(
        (
            PatchStepId(
                "step_"
                + sha256(
                    (
                        plan.plan_id.value
                        + ":"
                        + lineage.lineage_id.value
                        + ":"
                        + str(index)
                        + ":"
                        + operation
                    ).encode("utf-8")
                ).hexdigest()[:32]
            ),
            lineage.lineage_id,
        )
        for lineage in plan.candidate.lineages
        for index, operation in enumerate(lineage.step_graph, start=1)
    )


def _sealed_artifact_identifiers(plan: SealedPlan) -> tuple[str, ...]:
    """Derive one target-private artifact record for every sealed lineage."""
    return tuple(
        "artifact:" + lineage.lineage_id.value
        for lineage in plan.candidate.lineages
    )


def _validate_settlement_journal(
    plan: SealedPlan, journal: SettlementJournal
) -> None:
    """Require final facts to match the sealed step and artifact vectors."""
    expected_steps = _sealed_journal_steps(plan)
    observed_steps = tuple(
        (item.identifier, item.lineage) for item in journal.steps
    )
    if (
        observed_steps != expected_steps
        or any(item.state is CommitStepState.PLANNED for item in journal.steps)
        or tuple(item.identifier for item in journal.artifacts)
        != _sealed_artifact_identifiers(plan)
    ):
        raise CoordinatorError(CoordinatorErrorCode.INVARIANT)


def _truth(
    states: tuple[CommitStepState, ...],
    artifact: ArtifactState,
    postcondition: PostconditionState,
) -> CommitTruth:
    """Derive journal truth without inferring filesystem effects."""
    if not states or all(
        item is CommitStepState.NOT_COMMITTED for item in states
    ):
        mutation = MutationState.NOT_COMMITTED
        occurrence = RequestedEffectOccurrence.FALSE
        postcondition = PostconditionState.UNKNOWN
    elif CommitStepState.UNKNOWN in states:
        mutation = MutationState.INDETERMINATE
        occurrence = (
            RequestedEffectOccurrence.TRUE
            if CommitStepState.COMMITTED in states
            else RequestedEffectOccurrence.UNKNOWN
        )
        if occurrence is not RequestedEffectOccurrence.TRUE:
            postcondition = PostconditionState.UNKNOWN
    elif all(item is CommitStepState.COMMITTED for item in states):
        mutation = MutationState.COMMITTED
        occurrence = RequestedEffectOccurrence.TRUE
    else:
        mutation = MutationState.PARTIALLY_COMMITTED
        occurrence = RequestedEffectOccurrence.TRUE
    workspace = (
        WorkspaceChange.CHANGED
        if occurrence is RequestedEffectOccurrence.TRUE
        or artifact in {ArtifactState.STAGED, ArtifactState.LEAKED}
        else (
            WorkspaceChange.UNKNOWN
            if (
                occurrence is RequestedEffectOccurrence.UNKNOWN
                or artifact is ArtifactState.UNKNOWN
            )
            else WorkspaceChange.UNCHANGED
        )
    )
    return CommitTruth(
        mutation,
        LineageState(mutation.value),
        occurrence,
        artifact,
        workspace,
        mutation is not MutationState.INDETERMINATE,
        postcondition,
    )


def _artifact_state(artifacts: tuple[ArtifactJournal, ...]) -> ArtifactState:
    """Aggregate artifact records without changing requested-effect truth."""
    values = frozenset(item.state for item in artifacts)
    if ArtifactState.UNKNOWN in values:
        return ArtifactState.UNKNOWN
    if ArtifactState.LEAKED in values:
        return ArtifactState.LEAKED
    if ArtifactState.STAGED in values:
        return ArtifactState.STAGED
    if ArtifactState.CLEANED in values:
        return ArtifactState.CLEANED
    return ArtifactState.ABSENT


def _status(
    mutation: MutationState,
) -> tuple[PatchStatus, PatchErrorCode | None]:
    """Map exact journal occurrence to the closed terminal status contract."""
    match mutation:
        case MutationState.NOT_COMMITTED:
            return PatchStatus.COMMIT_FAILED, PatchErrorCode.COMMIT_FAILED
        case MutationState.COMMITTED:
            return PatchStatus.COMMITTED, None
        case MutationState.PARTIALLY_COMMITTED:
            return PatchStatus.PARTIAL, PatchErrorCode.PARTIAL_COMMIT
        case MutationState.INDETERMINATE:
            return PatchStatus.INDETERMINATE, PatchErrorCode.INDETERMINATE
