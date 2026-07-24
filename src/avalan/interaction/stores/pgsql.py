"""Persist interactions and portable continuations in PostgreSQL."""

from ...pgsql import (
    PgsqlDatabase,
    PgsqlOperationError,
    PgsqlRow,
    PgsqlUnitOfWork,
    classify_pgsql_error,
)
from ...task.error import TaskError
from ...task.event import TaskInteractionEventType
from ...task.privacy import (
    EncryptedPrivacyValue,
    TaskKeyPurpose,
)
from ...task.queue import (
    TaskDurableExpiredReentryCommit,
    TaskQueueCompletion,
    TaskQueueItemState,
    TaskQueueReentry,
    TaskQueueSuspension,
)
from ...task.settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
    task_durable_resume_settlement_digest,
)
from ...task.state import (
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskRunState,
)
from ...task.store import (
    TaskExecutionResult,
    TaskStoreConflictError,
    TaskStoreError,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)
from ..codec import decode_input_request, encode_input_request
from ..continuation import (
    ContinuationClaim,
    ContinuationClaimOwnerId,
    ContinuationClaimReceipt,
    ContinuationClaimState,
    ContinuationCompletion,
    ContinuationCompletionCommand,
    ContinuationDispatch,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationRejectionCommand,
    ContinuationStoreRevision,
    DurableContinuationRecord,
    PortableContinuation,
    decode_portable_continuation,
    encode_portable_continuation,
)
from ..entities import (
    ActiveControlLeaseNonce,
    AnswerProvenance,
    BranchId,
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ControllerId,
    InputRequestId,
    InteractionStoreGeneration,
    InteractionStoreRevision,
    ModelConfigRevision,
    ModelId,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    RequestState,
    ResolutionIdempotencyKey,
    RunId,
    StateRevision,
)
from ..error import (
    InputContractError,
    InputErrorCode,
    InputValidationError,
    InteractionNotFoundError,
    InteractionStoreClosedError,
)
from ..handler import InputResumer, InputResumptionNotification
from ..policy import (
    InteractionActor,
    InteractionAuthorizer,
    InteractionClock,
    InteractionIdFactory,
    InteractionPolicy,
    TaskInputClassifier,
)
from ..state import InputTransitionError
from ..store import (
    _ADMISSION_CLEANUP_RESOLVER,
    _DEADLINE_RESOLVER,
    _TRUSTED_DEFAULT_RESOLVER,
    AdvisoryWaitState,
    AdvisoryWaitStatus,
    CancelInteractionCommand,
    CancelInteractionResult,
    ControllerActivityResult,
    CreateInteractionApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    CreateInteractionResult,
    DetachInteractionCommand,
    DueInteractionsResult,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionBranchRegistrationResult,
    InteractionBranchRoot,
    InteractionBranchRootLookup,
    InteractionCorrelation,
    InteractionDeadlineSnapshot,
    InteractionDisclosureProjection,
    InteractionExecutionScope,
    InteractionPresentationResult,
    InteractionPresentationState,
    InteractionRecord,
    InteractionResolutionAuthority,
    InteractionResolutionResult,
    ListInteractionsCommand,
    PresentInteractionCommand,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    ResolutionIdempotencyEntry,
    ResolveInteractionCommand,
    ScopeCancellationResult,
    ScopedInteractionLookup,
    ScopeSupersessionResult,
    SupersedeInteractionScopeCommand,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionCommand,
    TerminalizeInteractionResult,
    TerminalizeInteractionScopeCommand,
    TrustedDefaultResolutionCommand,
    TrustedDefaultResolutionResult,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    _InteractionAdmissionCapability,
    _InteractionAdmissionCleanupCommand,
    _InteractionAdmissionCleanupResult,
    _InteractionAdmissionCreateCommand,
    _new_interaction_store_backing,
    _new_partial_interaction_store_backing,
    _new_scoped_interaction_store_backing,
    _snapshot_interaction_store_backing,
    _validate_interaction_admission_cleanup_command,
)
from ..validation import MAX_STATE_REVISION, validate_aware_datetime
from .memory import (
    MemoryInteractionStore,
    _MemoryAdmissionBinding,
    _MemoryInteractionBacking,
    _report_resumption_delivery_failure,
)

from asyncio import (
    CancelledError,
    Future,
    Lock,
    create_task,
    get_running_loop,
    shield,
    sleep,
)
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from importlib.util import find_spec
from inspect import isawaitable
from json import dumps, loads
from re import fullmatch
from typing import Any, NoReturn, Protocol, TypeVar, cast, final
from uuid import uuid4

INTERACTION_PGSQL_HEAD_REVISION = "20260723_0002"
INTERACTION_PGSQL_INSTALL_COMMAND = (
    'python3 -m pip install -U "avalan[task-pgsql]"'
)
INTERACTION_PGSQL_MIGRATION_COMMAND = "avalan task pgsql migrate head"

_MAX_RETENTION_DAYS = 3_650
_SHA256_PATTERN = r"[0-9a-f]{64}"
_T = TypeVar("_T")


class PgsqlInteractionCipher(Protocol):
    """Encrypt and decrypt durable interaction payloads."""

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        """Encrypt one interaction payload."""
        ...

    def decrypt(
        self,
        value: EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        """Decrypt one interaction payload."""
        ...


class DurableContinuationLifecycle(StrEnum):
    """Identify the database-owned continuation readiness state."""

    PENDING = "pending"
    READY = "ready"
    CLAIMED = "claimed"
    DISPATCHING = "dispatching"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"


class ResumptionOutboxStatus(StrEnum):
    """Identify one durable resumption-delivery state."""

    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"
    DEAD = "dead"


class PgsqlInteractionStoreError(InputContractError):
    """Report a content-safe durable-store operation failure."""


class _InteractionCallbackError(Exception):
    """Carry one trusted reducer dependency failure across a transaction."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__("trusted interaction callback failed")


class _PgsqlCreateCapacityError(PgsqlInteractionStoreError):
    """Carry one authorization-safe SQL admission rejection."""

    def __init__(self, command: CreateInteractionCommand) -> None:
        super().__init__(
            InputErrorCode.CAPACITY_EXCEEDED,
            "snapshot_records",
            "process pending-interaction capacity is exhausted",
        )
        self.rejection = CreateInteractionRejected(
            command=command,
            error=InputTransitionError(
                code=self.code,
                path=self.path,
                message=self.safe_message,
            ),
        )


@final
class PgsqlInteractionFeatureUnavailableError(PgsqlInteractionStoreError):
    """Report missing optional dependencies or schema state."""

    install_command = INTERACTION_PGSQL_INSTALL_COMMAND
    migration_command = INTERACTION_PGSQL_MIGRATION_COMMAND

    def __init__(self, *, reason: str) -> None:
        assert isinstance(reason, str) and reason
        self.reason = reason
        super().__init__(
            InputErrorCode.UNAVAILABLE,
            "store.postgresql",
            f"{reason}; install with `{self.install_command}` and migrate "
            f"with `{self.migration_command}`",
        )


@final
class ContinuationStoreConflictError(PgsqlInteractionStoreError):
    """Report a rejected durable continuation compare-and-swap."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(InputErrorCode.STALE_REVISION, path, message)


@final
class ContinuationDispatchAmbiguousError(PgsqlInteractionStoreError):
    """Refuse replay after provider dispatch may have started."""

    def __init__(self) -> None:
        super().__init__(
            InputErrorCode.UNAVAILABLE,
            "continuation.dispatch",
            "provider dispatch is ambiguous and cannot be replayed",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlInteractionStorePolicy:
    """Configure retention, polling, and durable delivery bounds."""

    retention_days: int = 30
    poll_interval_seconds: float = 0.05
    outbox_lease_seconds: int = 30
    outbox_max_attempts: int = 20
    encryption_key_id: str | None = None
    check_schema_on_open: bool = True

    def __post_init__(self) -> None:
        assert isinstance(self.retention_days, int)
        assert not isinstance(self.retention_days, bool)
        assert 1 <= self.retention_days <= _MAX_RETENTION_DAYS
        assert isinstance(self.poll_interval_seconds, int | float)
        assert not isinstance(self.poll_interval_seconds, bool)
        assert self.poll_interval_seconds > 0
        assert isinstance(self.outbox_lease_seconds, int)
        assert not isinstance(self.outbox_lease_seconds, bool)
        assert 1 <= self.outbox_lease_seconds <= 3_600
        assert isinstance(self.outbox_max_attempts, int)
        assert not isinstance(self.outbox_max_attempts, bool)
        assert 1 <= self.outbox_max_attempts <= MAX_STATE_REVISION
        if self.encryption_key_id is not None:
            _assert_opaque(self.encryption_key_id, "encryption_key_id")
        assert isinstance(self.check_schema_on_open, bool)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationSweepResult:
    """Report bounded expiry invalidation and retention deletion."""

    invalidated: tuple[ContinuationId, ...] = ()
    deleted: tuple[ContinuationId, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.invalidated, tuple)
        assert isinstance(self.deleted, tuple)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResumptionOutboxRecord:
    """Carry one content-free resumption delivery claim."""

    outbox_id: str
    continuation_id: ContinuationId
    request_id: InputRequestId
    task_run_id: str | None
    resolution_revision: StateRevision
    status: ResumptionOutboxStatus
    fencing_token: ContinuationFencingToken
    attempts: int

    def __post_init__(self) -> None:
        _assert_opaque(self.outbox_id, "outbox_id")
        _assert_opaque(self.continuation_id, "continuation_id")
        _assert_opaque(self.request_id, "request_id")
        if self.task_run_id is not None:
            _assert_opaque(self.task_run_id, "task_run_id")
        assert isinstance(self.status, ResumptionOutboxStatus)
        assert isinstance(self.resolution_revision, int)
        assert self.resolution_revision > 0
        assert isinstance(self.fencing_token, int)
        assert self.fencing_token >= 0
        assert isinstance(self.attempts, int)
        assert self.attempts >= 0


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskSuspension:
    """Return one atomically persisted interaction and task suspension."""

    interaction: CreateInteractionApplied
    suspension: TaskQueueSuspension

    def __post_init__(self) -> None:
        assert isinstance(self.interaction, CreateInteractionApplied)
        assert isinstance(self.suspension, TaskQueueSuspension)
        request = self.interaction.record.request
        assert str(request.request_id) == self.suspension.request_id
        assert str(request.continuation_id) == self.suspension.continuation_id


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskReentry:
    """Return one atomically resolved interaction and task reentry."""

    resolution: InteractionResolutionResult
    reentry: TaskQueueReentry

    def __post_init__(self) -> None:
        record = getattr(self.resolution, "record", None)
        assert isinstance(record, InteractionRecord)
        assert record.request.state is RequestState.ANSWERED
        assert isinstance(self.reentry, TaskQueueReentry)
        assert str(record.request.origin.run_id) == self.reentry.run.run_id


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskResuspension:
    """Return one atomically completed continuation and new suspension."""

    completed_continuation: PortableContinuation
    interaction: CreateInteractionApplied
    suspension: TaskQueueSuspension

    def __post_init__(self) -> None:
        completed = self.completed_continuation
        assert type(completed) is PortableContinuation
        assert completed.claim.state is ContinuationClaimState.COMPLETED
        assert type(completed.completion) is ContinuationCompletion
        assert isinstance(self.interaction, CreateInteractionApplied)
        assert isinstance(self.suspension, TaskQueueSuspension)
        request = self.interaction.record.request
        assert str(request.request_id) == self.suspension.request_id
        assert str(request.continuation_id) == self.suspension.continuation_id
        assert request.origin.run_id == completed.origin.run_id
        assert request.continuation_id != completed.continuation_id


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskSettlement:
    """Return one atomically completed continuation and terminal task."""

    completed_continuation: PortableContinuation
    completion: TaskQueueCompletion

    def __post_init__(self) -> None:
        continuation = self.completed_continuation
        assert type(continuation) is PortableContinuation
        assert continuation.claim.state is ContinuationClaimState.COMPLETED
        assert type(continuation.completion) is ContinuationCompletion
        assert isinstance(self.completion, TaskQueueCompletion)
        assert str(continuation.origin.run_id) == self.completion.run.run_id


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskAmbiguity:
    """Return one fenced ambiguous continuation and terminal task."""

    ambiguous_continuation: PortableContinuation
    completion: TaskQueueCompletion

    def __post_init__(self) -> None:
        continuation = self.ambiguous_continuation
        assert type(continuation) is PortableContinuation
        assert (
            continuation.claim.state
            is ContinuationClaimState.DISPATCHED_AMBIGUOUS
        )
        assert continuation.completion is None
        assert isinstance(self.completion, TaskQueueCompletion)
        assert (
            self.completion.run.state is TaskRunState.FAILED
            and self.completion.attempt.state is TaskAttemptState.FAILED
        ) or (
            self.completion.run.state is TaskRunState.CANCELLED
            and self.completion.attempt.state is TaskAttemptState.ABANDONED
        )
        assert self.completion.queue_item.state is TaskQueueItemState.DEAD
        assert str(continuation.origin.run_id) == self.completion.run.run_id


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskRejection:
    """Return one invalidated admission and atomically failed task."""

    rejected_continuation: PortableContinuation
    completion: TaskQueueCompletion

    def __post_init__(self) -> None:
        continuation = self.rejected_continuation
        assert type(continuation) is PortableContinuation
        assert (
            continuation.claim.state
            is ContinuationClaimState.FAILED_SAFE_TO_RETRY
        )
        assert continuation.claim.owner_id is None
        assert continuation.completion is None
        assert isinstance(self.completion, TaskQueueCompletion)
        assert self.completion.run.state is TaskRunState.FAILED
        assert self.completion.attempt.state is TaskAttemptState.FAILED
        assert self.completion.queue_item.state is TaskQueueItemState.DEAD
        assert str(continuation.origin.run_id) == self.completion.run.run_id


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlDurableTaskLifecycle:
    """Return one interaction lifecycle result and terminal task replay."""

    interaction: (
        ScopeCancellationResult
        | ScopeSupersessionResult
        | DueInteractionsResult
    )
    completions: tuple[TaskQueueCompletion, ...]

    def __post_init__(self) -> None:
        run_ids: set[str] = set()
        for completion in self.completions:
            assert completion.run.run_id not in run_ids
            run_ids.add(completion.run.run_id)
            assert completion.run.state in {
                TaskRunState.CANCELLED,
                TaskRunState.EXPIRED,
            }
            assert completion.attempt.state in {
                TaskAttemptState.ABANDONED,
                TaskAttemptState.FAILED,
            }
            assert completion.queue_item.state is TaskQueueItemState.DEAD

    def completion_for(
        self,
        task_run_id: str,
    ) -> TaskQueueCompletion | None:
        """Return the terminal completion for one affected task run."""
        _assert_opaque(task_run_id, "task_run_id")
        return next(
            (
                completion
                for completion in self.completions
                if completion.run.run_id == task_run_id
            ),
            None,
        )


class ResumptionOutboxDispatcher(Protocol):
    """Deliver one idempotent durable resumption notification."""

    async def __call__(self, record: ResumptionOutboxRecord) -> None:
        """Deliver one outbox record idempotently."""
        ...


class _PgsqlTaskTransactionStore(Protocol):
    """Apply task suspension mutations inside an existing transaction."""

    @property
    def database(self) -> PgsqlDatabase:
        """Return the shared PostgreSQL database."""
        ...

    async def _suspend_claim_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueSuspension:
        """Suspend one claim inside the supplied transaction."""
        ...

    async def _requeue_suspended_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        run_id: str,
        request_id: str,
        continuation_id: str,
        resolution_revision: int,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueReentry:
        """Requeue one suspension inside the supplied transaction."""
        ...

    async def _terminalize_suspended_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        task_run_id: str,
        correlations: tuple[tuple[str, str], ...],
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        event_type: TaskInteractionEventType,
        reason: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
        replay_only: bool = False,
    ) -> TaskQueueCompletion | None:
        """Terminalize or replay one suspended task lifecycle."""
        ...

    async def _validate_suspended_run_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        task_run_id: str,
    ) -> None:
        """Validate one task-owned interaction attachment boundary."""
        ...

    async def _settle_claim_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        settlement: TaskDurableResumeSettlement,
        observed_at: datetime,
        metadata: Mapping[str, object],
        replay_only: bool = False,
        allow_expired_lease: bool = False,
        terminal_run_state: TaskRunState | None = None,
        terminal_reason: str | None = None,
        interaction_event_type: TaskInteractionEventType | None = None,
        interaction_request_id: str | None = None,
        interaction_continuation_id: str | None = None,
    ) -> TaskQueueCompletion:
        """Settle one resumed task inside the supplied transaction."""
        ...

    async def _terminalize_completed_claim_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        settlement: TaskDurableResumeFailure | TaskDurableResumeCancellation,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        """Terminalize a task whose provider continuation is completed."""
        ...

    async def _release_claimed_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueReentry:
        """Release one claimed reentry inside the supplied transaction."""
        ...

    async def _fail_claimed_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str | None,
        continuation_id: str | None,
        checkpoint_id: str | None,
        result: TaskExecutionResult,
        reason: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
        replay_only: bool = False,
        terminal_run_state: TaskRunState = TaskRunState.FAILED,
        interaction_event_type: TaskInteractionEventType | None = None,
    ) -> TaskQueueCompletion:
        """Fail one claimed reentry inside the supplied transaction."""
        ...

    async def _release_running_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueReentry:
        """Release one running reentry inside the supplied transaction."""
        ...

    async def _cancel_partial_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        active_segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        result: TaskExecutionResult,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        """Cancel one exact partial resumed startup or replay it."""
        ...


@dataclass(slots=True)
class _PgsqlRuntimeState:
    """Retain only process-local callbacks and sealed admission bindings."""

    lock: Lock
    mutation_lock: Lock
    resumers: dict[str, InputResumer]
    admissions: dict[_InteractionAdmissionCapability, _MemoryAdmissionBinding]


@final
class _DeferredAdmissionResumer:
    """Capture one resumption until its PostgreSQL transaction commits."""

    def __init__(self) -> None:
        self.notification: InputResumptionNotification | None = None

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        assert self.notification is None
        self.notification = notification


@dataclass(frozen=True, slots=True)
class _InteractionSnapshotSelection:
    """Select the minimum encrypted interaction snapshot for one operation."""

    correlation: InteractionCorrelation | None = None
    admission_request_id: InputRequestId | None = None
    admission_continuation_id: ContinuationId | None = None
    run_id: RunId | None = None
    principal: PrincipalScope | None = None
    scope_identity_digest: str | None = None
    trusted_task_run_id: str | None = None
    mutation_scope: InteractionExecutionScope | None = None
    include_records: bool = True
    include_branches: bool = True
    tolerate_invalid_records: bool = False
    tolerate_invalid_branches: bool = False
    tolerate_invalid_continuations: bool = False

    def __post_init__(self) -> None:
        if (
            self.admission_request_id is None
            or self.admission_continuation_id is None
        ):
            assert self.admission_request_id is None
            assert self.admission_continuation_id is None
        else:
            assert self.correlation is None
            assert self.run_id is None
            assert self.principal is None
            assert self.scope_identity_digest is None
            assert self.trusted_task_run_id is None
            assert self.mutation_scope is None
        if self.correlation is not None:
            assert self.admission_request_id is None
            assert self.admission_continuation_id is None
            assert self.run_id is None
            assert self.principal is not None
            assert self.scope_identity_digest is None
            assert self.trusted_task_run_id is None
            assert self.mutation_scope is None
        if self.run_id is not None:
            assert self.admission_request_id is None
            assert self.admission_continuation_id is None
            assert (
                self.principal is not None
                or self.scope_identity_digest is not None
                or self.trusted_task_run_id is not None
            )
        if self.principal is not None:
            assert self.correlation is not None or self.run_id is not None
            assert self.scope_identity_digest is None
            assert self.trusted_task_run_id is None
        if self.scope_identity_digest is not None:
            assert self.run_id is not None
            assert self.principal is None
            assert self.correlation is None
            assert fullmatch(_SHA256_PATTERN, self.scope_identity_digest)
        if self.trusted_task_run_id is not None:
            assert self.run_id is not None
            assert str(self.run_id) == self.trusted_task_run_id
            assert self.principal is None
            assert self.correlation is None
            assert self.mutation_scope is None
        if self.mutation_scope is not None:
            assert self.run_id == self.mutation_scope.run_id
            assert self.principal is not None
            assert self.scope_identity_digest is None
            assert self.trusted_task_run_id is None
            assert self.correlation is None


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _ScopeOwnershipPresence:
    """Hold one content-free PostgreSQL scope-ownership result."""

    scope: InteractionExecutionScope
    principal: PrincipalScope
    actor_owned_record_match: bool
    foreign_owned_record_match: bool
    actor_owned_branch_match: bool
    foreign_owned_branch_match: bool


def _correlation_snapshot(
    actor: InteractionActor,
    correlation: InteractionCorrelation,
) -> _InteractionSnapshotSelection:
    """Select one complete actor-bound interaction correlation."""
    return _InteractionSnapshotSelection(
        correlation=correlation,
        principal=actor.principal,
        include_branches=False,
    )


def _scope_snapshot(
    actor: InteractionActor,
    scope: InteractionExecutionScope,
) -> _InteractionSnapshotSelection:
    """Select only one actor-bound persisted run and branch graph."""
    return _InteractionSnapshotSelection(
        run_id=scope.run_id,
        principal=actor.principal,
    )


def _scope_mutation_snapshot(
    actor: InteractionActor,
    scope: InteractionExecutionScope,
) -> _InteractionSnapshotSelection:
    """Select actor rows and attest global ownership for one mutation."""
    return _InteractionSnapshotSelection(
        run_id=scope.run_id,
        principal=actor.principal,
        mutation_scope=scope,
    )


def _create_snapshot(
    command: CreateInteractionCommand,
) -> _InteractionSnapshotSelection:
    """Select only the command actor's request run for creation."""
    return _InteractionSnapshotSelection(
        run_id=command.request.origin.run_id,
        principal=command.actor.principal,
    )


def _branch_snapshot(
    actor: InteractionActor,
    run_id: RunId,
) -> _InteractionSnapshotSelection:
    """Select only one actor-bound persisted branch graph."""
    return _InteractionSnapshotSelection(
        run_id=run_id,
        principal=actor.principal,
        include_records=False,
    )


def _admission_cleanup_snapshot(
    binding: _MemoryAdmissionBinding | None,
) -> _InteractionSnapshotSelection:
    """Select only one process-local admission's exact persisted record."""
    if binding is None:
        return _InteractionSnapshotSelection(
            include_records=False,
            include_branches=False,
        )
    return _InteractionSnapshotSelection(
        admission_request_id=binding.request_id,
        admission_continuation_id=binding.continuation_id,
        include_branches=False,
    )


async def _publish_admission_resumption(
    binding: _MemoryAdmissionBinding,
    notification: InputResumptionNotification,
) -> None:
    """Publish one admission resumption only after durable commit."""

    async def deliver() -> None:
        try:
            await binding.resumer(notification)
        except CancelledError:
            _report_resumption_delivery_failure()
        except Exception:
            _report_resumption_delivery_failure()
        finally:
            if not binding.handoff.done():
                binding.handoff.set_result(None)

    delivery = create_task(deliver())
    try:
        await shield(delivery)
    except CancelledError:
        while not delivery.done():
            try:
                await shield(delivery)
            except CancelledError:
                continue
        delivery.result()
        raise


def _trusted_task_snapshot(
    task_run_id: str,
) -> _InteractionSnapshotSelection:
    """Select one task's exact bound records and required branch closure."""
    return _InteractionSnapshotSelection(
        run_id=RunId(task_run_id),
        trusted_task_run_id=task_run_id,
    )


_DEADLINE_SNAPSHOT_SELECTION = _InteractionSnapshotSelection(
    tolerate_invalid_records=True,
    tolerate_invalid_branches=True,
    tolerate_invalid_continuations=True,
)


@final
class PgsqlInteractionStoreFactory:
    """Open independently closable handles over one PostgreSQL backing."""

    def __init__(
        self,
        database: PgsqlDatabase,
        *,
        policy: InteractionPolicy,
        clock: InteractionClock,
        authorizer: InteractionAuthorizer,
        id_factory: InteractionIdFactory,
        cipher: PgsqlInteractionCipher,
        classifier: TaskInputClassifier | None = None,
        store_policy: PgsqlInteractionStorePolicy | None = None,
    ) -> None:
        assert hasattr(database, "connection")
        assert isinstance(policy, InteractionPolicy)
        assert callable(getattr(clock, "read", None))
        assert callable(getattr(authorizer, "authorize", None))
        assert callable(
            getattr(id_factory, "new_active_control_lease_nonce", None)
        )
        assert hasattr(cipher, "encrypt")
        assert hasattr(cipher, "decrypt")
        assert classifier is None or callable(
            getattr(classifier, "classify_task_input", None)
        )
        self._database = database
        self._policy = policy
        self._clock = clock
        self._authorizer = authorizer
        self._id_factory = id_factory
        self._cipher = cipher
        self._classifier = classifier
        self._store_policy = store_policy or PgsqlInteractionStorePolicy()
        self._runtime = _PgsqlRuntimeState(
            lock=Lock(),
            mutation_lock=Lock(),
            resumers={},
            admissions={},
        )
        self._opened = False

    async def open(self) -> "PgsqlInteractionStore":
        """Return one independently closable PostgreSQL store handle."""
        if not self._opened:
            opener = getattr(self._database, "open", None)
            if opener is not None:
                pending = opener()
                if isawaitable(pending):
                    await pending
            if self._store_policy.check_schema_on_open:
                await _check_schema(self._database)
            self._opened = True
        return PgsqlInteractionStore(
            database=self._database,
            policy=self._policy,
            clock=self._clock,
            authorizer=self._authorizer,
            id_factory=self._id_factory,
            cipher=self._cipher,
            classifier=self._classifier,
            store_policy=self._store_policy,
            runtime=self._runtime,
        )


@final
class PgsqlInteractionStore:
    """Implement strict interaction and durable continuation transactions."""

    def __init__(
        self,
        *,
        database: PgsqlDatabase,
        policy: InteractionPolicy,
        clock: InteractionClock,
        authorizer: InteractionAuthorizer,
        id_factory: InteractionIdFactory,
        cipher: PgsqlInteractionCipher,
        classifier: TaskInputClassifier | None,
        store_policy: PgsqlInteractionStorePolicy,
        runtime: _PgsqlRuntimeState,
    ) -> None:
        self._database = database
        self._policy = policy
        self._clock = clock
        self._authorizer = authorizer
        self._id_factory = id_factory
        self._cipher = cipher
        self._classifier = classifier
        self._store_policy = store_policy
        self._runtime = runtime
        self._closed = False

    async def create(
        self,
        command: CreateInteractionCommand,
    ) -> CreateInteractionResult:
        """Create one interaction through the shared strict reducer."""
        if (
            isinstance(command, CreateInteractionCommand)
            and command.request.origin.task_id is not None
        ):
            _task_coordinator_required("create.request.origin.task_id")

        async def create(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> CreateInteractionResult:
            result = await store.create(command)
            if isinstance(command, CreateInteractionCommand):
                await self._enforce_create_process_capacity(
                    unit,
                    command,
                    result,
                )
            return result

        try:
            return await self._run_memory_with_unit(
                "interaction_create",
                create,
                mutate=True,
                selection=(
                    _create_snapshot(command)
                    if isinstance(command, CreateInteractionCommand)
                    else _InteractionSnapshotSelection(
                        include_records=False,
                        include_branches=False,
                    )
                ),
            )
        except _PgsqlCreateCapacityError as error:
            return error.rejection

    async def create_durable(
        self,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        *,
        task_run_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> CreateInteractionResult:
        """Atomically create one request and encrypted continuation."""
        if command.request.origin.task_id is not None:
            _task_coordinator_required("create.request.origin.task_id")
        if task_run_id is not None:
            _task_coordinator_required("task_run_id")
        if command.request.request_id != continuation.request_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation.request_id",
                "continuation does not match the request",
            )
        if command.request.continuation_id != continuation.continuation_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation.continuation_id",
                "continuation does not match the request",
            )
        if command.request.origin != continuation.origin:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation.origin",
                "continuation does not match the request origin",
            )
        _validate_request_continuation_expiry(command, continuation)
        if (task_run_id is None) != (checkpoint_id is None):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "checkpoint_id",
                "task-bound continuations require a checkpoint identifier",
            )

        async def operation(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> CreateInteractionResult:
            result = await store.create(command)
            await self._enforce_create_process_capacity(
                unit,
                command,
                result,
            )
            return result

        async def persist_continuation(
            unit: PgsqlUnitOfWork,
            result: CreateInteractionResult,
        ) -> None:
            record = getattr(result, "record", None)
            if isinstance(record, InteractionRecord):
                await self._insert_continuation(
                    unit,
                    continuation,
                    task_run_id=task_run_id,
                    checkpoint_id=checkpoint_id,
                )

        try:
            return await self._run_memory_with_unit(
                "interaction_create_durable",
                operation,
                mutate=True,
                after_persist=persist_continuation,
                selection=_create_snapshot(command),
            )
        except _PgsqlCreateCapacityError as error:
            return error.rejection

    async def create_admission(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Create one sealed broker admission in the shared transaction."""
        selection = (
            _create_snapshot(command._command)
            if type(command) is _InteractionAdmissionCreateCommand
            else _InteractionSnapshotSelection(
                include_records=False,
                include_branches=False,
            )
        )

        async def create(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> CreateInteractionResult:
            result = await store.create_admission(command)
            if type(command) is _InteractionAdmissionCreateCommand:
                await self._enforce_create_process_capacity(
                    unit,
                    command._command,
                    result,
                )
            return result

        try:
            return await self._run_memory_with_unit(
                "interaction_create_admission",
                create,
                mutate=True,
                selection=selection,
            )
        except _PgsqlCreateCapacityError as error:
            return error.rejection

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Settle one process-local sealed admission binding."""
        command = _validate_interaction_admission_cleanup_command(command)
        return await self._run_memory(
            "interaction_cleanup_admission",
            lambda store: store.cleanup_admission(command),
            mutate=True,
            admission_capability=command._capability,
        )

    async def lookup_scoped(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionDisclosureProjection | None:
        """Return one authorized persisted interaction projection."""
        return await self._run_memory(
            "interaction_lookup",
            lambda store: store.lookup_scoped(query),
            mutate=False,
            selection=_correlation_snapshot(
                query.actor,
                query.correlation,
            ),
        )

    async def list_scoped(
        self,
        command: ListInteractionsCommand,
    ) -> tuple[InteractionDisclosureProjection, ...]:
        """Return authorized persisted projections in one scope."""
        return await self._run_memory(
            "interaction_list",
            lambda store: store.list_scoped(command),
            mutate=False,
            selection=_scope_snapshot(command.actor, command.scope),
        )

    async def lookup_branch_root(
        self,
        query: InteractionBranchRootLookup,
    ) -> InteractionBranchRoot | None:
        """Return one authorized persisted branch-root mapping."""
        return await self._run_memory(
            "interaction_branch_lookup",
            lambda store: store.lookup_branch_root(query),
            mutate=False,
            selection=_branch_snapshot(query.actor, query.run_id),
        )

    async def mark_presented(
        self,
        command: PresentInteractionCommand,
    ) -> InteractionPresentationResult:
        """Persist one attached presentation mutation."""
        return await self._run_memory(
            "interaction_mark_presented",
            lambda store: store.mark_presented(command),
            mutate=True,
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def mark_detached(
        self,
        command: DetachInteractionCommand,
    ) -> InteractionPresentationResult:
        """Persist one detached presentation mutation."""
        return await self._run_memory(
            "interaction_mark_detached",
            lambda store: store.mark_detached(command),
            mutate=True,
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def resolve(
        self,
        command: ResolveInteractionCommand,
    ) -> InteractionResolutionResult:
        """Resolve and expose one continuation-ready outbox atomically."""

        async def operation(
            store: MemoryInteractionStore,
            _unit: PgsqlUnitOfWork,
        ) -> InteractionResolutionResult:
            return await store.resolve(command)

        async def persist(
            unit: PgsqlUnitOfWork,
            result: InteractionResolutionResult,
        ) -> None:
            record = getattr(result, "record", None)
            if (
                isinstance(record, InteractionRecord)
                and record.request.resolution is not None
            ):
                await self._reject_standalone_task_lifecycle(
                    unit,
                    record,
                )
                if record.request.state is RequestState.ANSWERED:
                    await self._ready_continuation(unit, record)
                else:
                    await self._invalidate_for_record(unit, record)
                await self._sync_resolution_keys(unit, record)

        return await self._run_memory_with_unit(
            "interaction_resolve",
            operation,
            mutate=True,
            after_persist=persist,
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def resolve_trusted_default(
        self,
        command: TrustedDefaultResolutionCommand,
    ) -> TrustedDefaultResolutionResult:
        """Resolve trusted defaults and expose durable resumption."""
        return await self._terminal_memory_operation(
            "interaction_resolve_default",
            lambda store: store.resolve_trusted_default(command),
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def terminalize(
        self,
        command: TerminalizeInteractionCommand,
    ) -> TerminalizeInteractionResult:
        """Terminalize and invalidate any retained continuation."""
        return await self._terminal_memory_operation(
            "interaction_terminalize",
            lambda store: store.terminalize(command),
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def cancel(
        self,
        command: CancelInteractionCommand,
    ) -> CancelInteractionResult:
        """Cancel and invalidate any retained continuation."""
        return await self._terminal_memory_operation(
            "interaction_cancel",
            lambda store: store.cancel(command),
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def terminalize_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> ScopeCancellationResult:
        """Persist one scope cancellation and cascade invalidation."""
        return await self._scope_memory_operation(
            "interaction_cancel_scope",
            lambda store: store.terminalize_scope(command),
            selection=_scope_mutation_snapshot(
                command.actor,
                command.scope,
            ),
        )

    async def supersede_scope(
        self,
        command: SupersedeInteractionScopeCommand,
    ) -> ScopeSupersessionResult:
        """Persist one scope supersession and cascade invalidation."""
        return await self._scope_memory_operation(
            "interaction_supersede_scope",
            lambda store: store.supersede_scope(command),
            selection=_scope_mutation_snapshot(
                command.actor,
                command.scope,
            ),
        )

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> InteractionBranchRegistrationResult:
        """Persist one authorized branch edge."""
        return await self._run_memory(
            "interaction_register_branch",
            lambda store: store.register_branch(command),
            mutate=True,
            selection=_branch_snapshot(
                command.actor,
                command.registration.run_id,
            ),
        )

    async def record_activity(
        self,
        command: RecordControllerActivityCommand,
    ) -> ControllerActivityResult:
        """Persist one active-controller lease mutation."""
        return await self._run_memory(
            "interaction_record_activity",
            lambda store: store.record_activity(command),
            mutate=True,
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> InteractionDisclosureProjection:
        """Poll durably for a newer authorized interaction revision."""

        async def read(
            store: MemoryInteractionStore,
            records: tuple[InteractionRecord, ...],
        ) -> tuple[
            InteractionDisclosureProjection | None,
            InteractionStoreRevision | None,
        ]:
            projection = await store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=command.actor,
                    correlation=command.correlation,
                )
            )
            revision = next(
                (
                    record.store_revision
                    for record in records
                    if record.request.request_id
                    == command.correlation.request_id
                ),
                None,
            )
            return projection, revision

        while True:
            self._ensure_open()
            projection, revision = await self._run_memory_read_snapshot(
                "interaction_wait",
                read,
                selection=_correlation_snapshot(
                    command.actor,
                    command.correlation,
                ),
            )
            if projection is None or revision is None:
                raise InteractionNotFoundError()
            if revision > command.after_store_revision:
                return projection
            await sleep(self._store_policy.poll_interval_seconds)

    async def next_deadline(self) -> InteractionDeadlineSnapshot:
        """Return the persisted store's earliest deadline."""
        return await self._run_memory(
            "interaction_next_deadline",
            lambda store: store.next_deadline(),
            mutate=False,
            selection=_DEADLINE_SNAPSHOT_SELECTION,
        )

    async def wait_for_deadline_change(
        self,
        command: WaitForDeadlineChangeCommand,
    ) -> InteractionDeadlineSnapshot:
        """Poll durably for a newer deadline-schedule revision."""
        while True:
            self._ensure_open()
            snapshot = await self.next_deadline()
            if snapshot.schedule_revision > command.after_schedule_revision:
                return snapshot
            await sleep(self._store_policy.poll_interval_seconds)

    async def terminalize_due(
        self,
        command: TerminalizeDueInteractionsCommand,
    ) -> DueInteractionsResult:
        """Settle one persisted bounded due-interaction batch."""
        return await self._scope_memory_operation(
            "interaction_terminalize_due",
            lambda store: store.terminalize_due(command),
            selection=_DEADLINE_SNAPSHOT_SELECTION,
        )

    async def claim(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        lease_expires_at: datetime,
        dispatch_id: ContinuationDispatchId,
        provider_idempotency_key: ProviderIdempotencyKey,
        now: datetime,
    ) -> ContinuationClaimReceipt:
        """Claim one ready continuation with a fresh fencing token."""
        _assert_opaque(continuation_id, "continuation_id")
        _assert_opaque(owner_id, "owner_id")
        _assert_opaque(dispatch_id, "dispatch_id")
        _assert_opaque(
            provider_idempotency_key,
            "provider_idempotency_key",
        )
        now = validate_aware_datetime(now, "now")
        lease_expires_at = validate_aware_datetime(
            lease_expires_at,
            "lease_expires_at",
        )
        if lease_expires_at <= now:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "lease_expires_at",
                "claim lease must expire after the claim time",
            )

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            continuation = self._continuation_from_row(row)
            lifecycle = _lifecycle(row)
            request_expires_at = _row_datetime(
                row,
                "request_absolute_expires_at",
            )
            if continuation.store_revision != expected_store_revision:
                _stale("expected_store_revision")
            if request_expires_at != continuation.expires_at:
                raise PgsqlInteractionStoreError(
                    InputErrorCode.CORRELATION_MISMATCH,
                    "continuation.expires_at",
                    "continuation expiry does not match the "
                    "interaction deadline",
                )
            if min(request_expires_at, continuation.expires_at) <= now:
                raise PgsqlInteractionStoreError(
                    InputErrorCode.EXPIRED,
                    "continuation",
                    "continuation has expired",
                )
            if lease_expires_at > continuation.expires_at:
                raise InputValidationError(
                    InputErrorCode.OUT_OF_BOUNDS,
                    "lease_expires_at",
                    "claim lease cannot outlive the continuation",
                )
            if lifecycle is DurableContinuationLifecycle.DISPATCHING:
                raise ContinuationDispatchAmbiguousError()
            claimable = lifecycle is DurableContinuationLifecycle.READY
            expired_claim = (
                lifecycle is DurableContinuationLifecycle.CLAIMED
                and continuation.claim.lease_expires_at is not None
                and continuation.claim.lease_expires_at <= now
            )
            retryable = (
                continuation.claim.state
                is ContinuationClaimState.FAILED_SAFE_TO_RETRY
            )
            if not (claimable or expired_claim or retryable):
                raise ContinuationStoreConflictError(
                    "continuation.claim",
                    "continuation is not claimable",
                )
            fencing_token = ContinuationFencingToken(
                int(continuation.fencing_token) + 1
            )
            updated = replace(
                continuation,
                claim=ContinuationClaim(
                    state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    attempt=continuation.claim.attempt + 1,
                ),
                fencing_token=fencing_token,
                dispatch=ContinuationDispatch(
                    dispatch_id=dispatch_id,
                    provider_idempotency_key=provider_idempotency_key,
                    marked_at=now,
                ),
                completion=None,
                store_revision=_next_continuation_revision(continuation),
                updated_at=now,
            )
            await self._update_continuation(
                unit,
                updated,
                lifecycle=DurableContinuationLifecycle.CLAIMED,
                expected_revision=expected_store_revision,
            )
            return ContinuationClaimReceipt(
                continuation=updated,
                fencing_token=fencing_token,
            )

        return await self._transaction("continuation_claim", execute)

    async def mark_dispatching(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        """Fence replay immediately before the provider side effect."""
        now = validate_aware_datetime(now, "now")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            continuation = self._claimed_continuation(
                row,
                expected_store_revision=expected_store_revision,
                owner_id=owner_id,
                fencing_token=fencing_token,
                now=now,
            )
            assert continuation.dispatch is not None
            updated = replace(
                continuation,
                claim=ContinuationClaim(
                    state=ContinuationClaimState.DISPATCHED_AMBIGUOUS,
                    owner_id=owner_id,
                    attempt=continuation.claim.attempt,
                ),
                store_revision=_next_continuation_revision(continuation),
                updated_at=now,
            )
            await self._update_continuation(
                unit,
                updated,
                lifecycle=DurableContinuationLifecycle.DISPATCHING,
                expected_revision=expected_store_revision,
            )
            return updated

        return await self._transaction(
            "continuation_mark_dispatching",
            execute,
        )

    async def renew_claim(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        lease_expires_at: datetime,
        now: datetime,
    ) -> bool:
        """Renew one exact pre-dispatch continuation claim."""
        _assert_opaque(continuation_id, "continuation_id")
        _assert_opaque(owner_id, "owner_id")
        now = validate_aware_datetime(now, "now")
        lease_expires_at = validate_aware_datetime(
            lease_expires_at,
            "lease_expires_at",
        )

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            continuation = self._continuation_from_row(row)
            lifecycle = _lifecycle(row)
            if lifecycle is DurableContinuationLifecycle.DISPATCHING:
                current_revision = int(continuation.store_revision)
                claimed_revision = int(expected_store_revision)
                dispatch_revision_delta = (
                    2 if row.get("dispatch_completed_at") is not None else 1
                )
                exact_dispatch_claim = (
                    continuation.claim.state
                    is ContinuationClaimState.DISPATCHED_AMBIGUOUS
                    and continuation.claim.owner_id == owner_id
                    and continuation.fencing_token == fencing_token
                    and current_revision
                    == claimed_revision + dispatch_revision_delta
                )
                if exact_dispatch_claim:
                    return False
                _stale("continuation.claim")
            if lifecycle is not DurableContinuationLifecycle.CLAIMED:
                settled_revision_delta = {
                    ContinuationClaimState.FAILED_SAFE_TO_RETRY: 1,
                    ContinuationClaimState.COMPLETED: 3,
                }.get(continuation.claim.state)
                if (
                    settled_revision_delta is not None
                    and _row_optional_str(row, "claim_owner_id") == owner_id
                    and continuation.fencing_token == fencing_token
                    and int(continuation.store_revision)
                    == int(expected_store_revision) + settled_revision_delta
                ):
                    return False
                raise ContinuationStoreConflictError(
                    "continuation.claim",
                    "continuation claim cannot be renewed",
                )
            claimed = self._claimed_continuation(
                row,
                expected_store_revision=expected_store_revision,
                owner_id=owner_id,
                fencing_token=fencing_token,
                now=now,
            )
            if lease_expires_at <= now:
                raise InputValidationError(
                    InputErrorCode.OUT_OF_BOUNDS,
                    "lease_expires_at",
                    "claim lease must expire after the renewal time",
                )
            if lease_expires_at > claimed.expires_at:
                raise InputValidationError(
                    InputErrorCode.OUT_OF_BOUNDS,
                    "lease_expires_at",
                    "claim lease cannot outlive the continuation",
                )
            current_expiry = claimed.claim.lease_expires_at
            assert current_expiry is not None
            if lease_expires_at <= current_expiry:
                return True
            updated = replace(
                claimed,
                claim=replace(
                    claimed.claim,
                    lease_expires_at=lease_expires_at,
                ),
                updated_at=now,
            )
            await self._update_continuation(
                unit,
                updated,
                lifecycle=DurableContinuationLifecycle.CLAIMED,
                expected_revision=expected_store_revision,
            )
            return True

        renewed: object = await self._transaction(
            "continuation_renew_claim",
            execute,
        )
        if type(renewed) is not bool:
            raise PgsqlInteractionStoreError(
                InputErrorCode.UNAVAILABLE,
                "continuation.claim",
                "continuation renewal returned invalid state",
            )
        return renewed

    async def mark_dispatched(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        """Confirm the fenced dispatch marker without enabling replay."""
        now = validate_aware_datetime(now, "now")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            continuation = self._ambiguous_continuation(
                row,
                expected_store_revision=expected_store_revision,
                owner_id=owner_id,
                fencing_token=fencing_token,
            )
            updated = replace(
                continuation,
                store_revision=_next_continuation_revision(continuation),
                updated_at=now,
            )
            await self._update_continuation(
                unit,
                updated,
                lifecycle=DurableContinuationLifecycle.DISPATCHING,
                expected_revision=expected_store_revision,
                dispatch_completed_at=now,
            )
            return updated

        return await self._transaction("continuation_mark_dispatched", execute)

    async def complete(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        result_digest: str,
        now: datetime,
    ) -> PortableContinuation:
        """Complete one exact dispatch under its active fencing token."""
        _assert_digest(result_digest, "result_digest")
        now = validate_aware_datetime(now, "now")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            if _row_optional_str(row, "task_run_id") is not None:
                _task_coordinator_required("continuation.task_run_id")
            return await self._complete_continuation_in_unit(
                unit,
                continuation_id=continuation_id,
                expected_store_revision=expected_store_revision,
                owner_id=owner_id,
                fencing_token=fencing_token,
                result_digest=result_digest,
                now=now,
            )

        return await self._transaction("continuation_complete", execute)

    async def _complete_continuation_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        continuation_id: ContinuationId,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        result_digest: str,
        now: datetime,
        expected_task_run_id: str | None = None,
        expected_task_segment_id: str | None = None,
    ) -> PortableContinuation:
        """Complete one dispatched continuation inside an existing unit."""
        if expected_task_run_id is None:
            assert expected_task_segment_id is None
            row = await _lock_continuation(unit, continuation_id)
        elif expected_task_segment_id is None:
            row = await _lock_task_run_continuation(
                unit,
                continuation_id=continuation_id,
                task_run_id=expected_task_run_id,
            )
        else:
            row = await _lock_resumed_task_continuation(
                unit,
                continuation_id=continuation_id,
                task_run_id=expected_task_run_id,
                segment_id=expected_task_segment_id,
            )
        continuation = self._ambiguous_continuation(
            row,
            expected_store_revision=expected_store_revision,
            owner_id=owner_id,
            fencing_token=fencing_token,
        )
        updated = replace(
            continuation,
            claim=ContinuationClaim(
                state=ContinuationClaimState.COMPLETED,
                attempt=continuation.claim.attempt,
            ),
            completion=ContinuationCompletion(
                completed_at=now,
                result_digest=result_digest,
            ),
            store_revision=_next_continuation_revision(continuation),
            updated_at=now,
        )
        await self._update_continuation(
            unit,
            updated,
            lifecycle=DurableContinuationLifecycle.COMPLETED,
            expected_revision=expected_store_revision,
            dispatch_completed_at=now,
            settled_claim_owner_id=owner_id,
        )
        return updated

    async def invalidate(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        reason: str,
        now: datetime,
    ) -> PortableContinuation:
        """Invalidate one non-completed continuation by revision CAS."""
        _assert_opaque(reason, "reason")
        now = validate_aware_datetime(now, "now")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            if _row_optional_str(row, "task_run_id") is not None:
                _task_coordinator_required("continuation.task_run_id")
            continuation = self._continuation_from_row(row)
            if continuation.store_revision != expected_store_revision:
                _stale("expected_store_revision")
            lifecycle = _lifecycle(row)
            if lifecycle is DurableContinuationLifecycle.COMPLETED:
                raise ContinuationStoreConflictError(
                    "continuation",
                    "completed continuation cannot be invalidated",
                )
            if lifecycle is DurableContinuationLifecycle.DISPATCHING:
                raise ContinuationDispatchAmbiguousError()
            updated = replace(
                continuation,
                claim=_invalidated_claim(continuation),
                store_revision=_next_continuation_revision(continuation),
                updated_at=now,
            )
            await self._update_continuation(
                unit,
                updated,
                lifecycle=DurableContinuationLifecycle.INVALIDATED,
                expected_revision=expected_store_revision,
                invalid_reason=reason,
            )
            await unit.cursor.execute(
                _DEAD_OUTBOX_SQL,
                (now, continuation_id),
            )
            return updated

        return await self._transaction("continuation_invalidate", execute)

    async def release(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        """Release only a provably pre-dispatch claim for safe retry."""
        now = validate_aware_datetime(now, "now")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_continuation(unit, continuation_id)
            continuation = self._claimed_continuation(
                row,
                expected_store_revision=expected_store_revision,
                owner_id=owner_id,
                fencing_token=fencing_token,
                now=now,
                allow_expired=True,
            )
            updated = replace(
                continuation,
                claim=ContinuationClaim(
                    state=ContinuationClaimState.FAILED_SAFE_TO_RETRY,
                    attempt=continuation.claim.attempt,
                ),
                store_revision=_next_continuation_revision(continuation),
                updated_at=now,
            )
            await self._update_continuation(
                unit,
                updated,
                lifecycle=DurableContinuationLifecycle.READY,
                expected_revision=expected_store_revision,
                settled_claim_owner_id=owner_id,
            )
            return updated

        return await self._transaction("continuation_release", execute)

    async def get_continuation(
        self,
        continuation_id: ContinuationId,
    ) -> PortableContinuation:
        """Load and decrypt one portable continuation."""
        return (
            await self.get_continuation_record(continuation_id)
        ).continuation

    async def get_continuation_record(
        self,
        continuation_id: ContinuationId,
    ) -> DurableContinuationRecord:
        """Load one continuation with its durable task binding."""
        _assert_opaque(continuation_id, "continuation_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _SELECT_CONTINUATION_SQL,
                (continuation_id,),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise InteractionNotFoundError()
            return self._durable_continuation_record(row)

        return await self._transaction("continuation_record_get", execute)

    async def get_task_continuation_record(
        self,
        task_run_id: str,
    ) -> DurableContinuationRecord:
        """Load the unique active continuation bound to one task run."""
        _assert_opaque(task_run_id, "task_run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _SELECT_ACTIVE_CONTINUATIONS_BY_TASK_SQL,
                (task_run_id,),
            )
            rows = await unit.cursor.fetchall()
            if not rows:
                raise InteractionNotFoundError()
            if len(rows) != 1:
                raise ContinuationStoreConflictError(
                    "continuation.task_run_id",
                    "task run has ambiguous active continuations",
                )
            return self._durable_continuation_record(rows[0])

        return await self._transaction(
            "continuation_task_record_get",
            execute,
        )

    async def sweep(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> ContinuationSweepResult:
        """Invalidate expired work and delete retention-expired rows."""
        return await self._sweep(
            now=now,
            limit=limit,
            allow_task_bound_retention=False,
        )

    async def _sweep(
        self,
        *,
        now: datetime,
        limit: int,
        allow_task_bound_retention: bool,
    ) -> ContinuationSweepResult:
        """Sweep standalone rows or coordinator-approved task retention."""
        now = validate_aware_datetime(now, "now")
        _assert_limit(limit)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(_SELECT_SWEEP_SQL, (now, now, limit))
            rows = await unit.cursor.fetchall()
            for row in rows:
                if _row_optional_str(row, "task_run_id") is None:
                    continue
                lifecycle = _lifecycle(row)
                retention_deadline = _row_datetime(
                    row,
                    "interaction_retention_deadline_at",
                )
                if (
                    not allow_task_bound_retention
                    or retention_deadline > now
                    or lifecycle
                    not in {
                        DurableContinuationLifecycle.COMPLETED,
                        DurableContinuationLifecycle.INVALIDATED,
                    }
                ):
                    _task_coordinator_required("continuation.task_run_id")
            invalidated: list[ContinuationId] = []
            deleted: list[ContinuationId] = []
            for row in rows:
                retention_deadline = _row_datetime(
                    row,
                    "interaction_retention_deadline_at",
                )
                continuation_value = _row_optional_str(
                    row,
                    "continuation_id",
                )
                if retention_deadline <= now:
                    run_id = _row_str(row, "interaction_run_id")
                    scope_identity_digest = _row_str(
                        row,
                        "interaction_scope_identity_digest",
                    )
                    await unit.cursor.execute(
                        _LOCK_RETENTION_SCOPE_SQL,
                        (run_id, scope_identity_digest),
                    )
                    await unit.cursor.execute(
                        _DELETE_RECORD_SQL,
                        (_row_str(row, "interaction_request_id"),),
                    )
                    await unit.cursor.execute(
                        _DELETE_ORPHANED_BRANCHES_SQL,
                        (
                            run_id,
                            scope_identity_digest,
                            run_id,
                            scope_identity_digest,
                        ),
                    )
                    if continuation_value is not None:
                        deleted.append(ContinuationId(continuation_value))
                    continue
                if continuation_value is None:
                    continue
                continuation_id = ContinuationId(continuation_value)
                lifecycle = _lifecycle(row)
                if lifecycle is DurableContinuationLifecycle.DISPATCHING:
                    continue
                try:
                    continuation = self._continuation_from_row(row)
                except InputContractError:
                    continue
                if continuation.expires_at > now or lifecycle in {
                    DurableContinuationLifecycle.COMPLETED,
                    DurableContinuationLifecycle.INVALIDATED,
                }:
                    continue
                updated = replace(
                    continuation,
                    claim=_invalidated_claim(continuation),
                    store_revision=_next_continuation_revision(continuation),
                    updated_at=now,
                )
                await self._update_continuation(
                    unit,
                    updated,
                    lifecycle=DurableContinuationLifecycle.INVALIDATED,
                    expected_revision=continuation.store_revision,
                    invalid_reason="expired",
                )
                invalidated.append(continuation_id)
            return ContinuationSweepResult(
                invalidated=tuple(invalidated),
                deleted=tuple(deleted),
            )

        return await self._transaction("continuation_sweep", execute)

    async def claim_outbox(
        self,
        *,
        owner_id: ContinuationClaimOwnerId,
        now: datetime,
        limit: int = 10,
    ) -> tuple[ResumptionOutboxRecord, ...]:
        """Claim a bounded outbox batch with lease recovery."""
        _assert_opaque(owner_id, "owner_id")
        now = validate_aware_datetime(now, "now")
        _assert_limit(limit)
        lease_expires_at = now + timedelta(
            seconds=self._store_policy.outbox_lease_seconds
        )

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _CLAIM_OUTBOX_SQL,
                (
                    now,
                    now,
                    limit,
                    owner_id,
                    lease_expires_at,
                    now,
                ),
            )
            return tuple(
                _outbox_record(row) for row in await unit.cursor.fetchall()
            )

        return await self._transaction("interaction_outbox_claim", execute)

    async def complete_outbox(
        self,
        record: ResumptionOutboxRecord,
        *,
        owner_id: ContinuationClaimOwnerId,
        now: datetime,
    ) -> None:
        """Mark one exact fenced outbox delivery complete."""
        now = validate_aware_datetime(now, "now")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _COMPLETE_OUTBOX_SQL,
                (
                    now,
                    now,
                    record.outbox_id,
                    owner_id,
                    record.fencing_token,
                ),
            )
            if await unit.cursor.fetchone() is None:
                _stale("outbox.fencing_token")
            return None

        await self._transaction("interaction_outbox_complete", execute)

    async def release_outbox(
        self,
        record: ResumptionOutboxRecord,
        *,
        owner_id: ContinuationClaimOwnerId,
        error_code: str,
        now: datetime,
    ) -> None:
        """Release or dead-letter one exact outbox delivery claim."""
        _assert_opaque(error_code, "error_code")
        now = validate_aware_datetime(now, "now")
        dead = record.attempts >= self._store_policy.outbox_max_attempts

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _RELEASE_OUTBOX_SQL,
                (
                    (
                        ResumptionOutboxStatus.DEAD.value
                        if dead
                        else ResumptionOutboxStatus.PENDING.value
                    ),
                    error_code,
                    now,
                    record.outbox_id,
                    owner_id,
                    record.fencing_token,
                ),
            )
            if await unit.cursor.fetchone() is None:
                _stale("outbox.fencing_token")
            return None

        await self._transaction("interaction_outbox_release", execute)

    async def aclose(self) -> None:
        """Idempotently close only this handle."""
        self._closed = True

    async def _terminal_memory_operation(
        self,
        operation_name: str,
        callback: Callable[
            [MemoryInteractionStore],
            Awaitable[_T],
        ],
        *,
        selection: _InteractionSnapshotSelection | None = None,
    ) -> _T:
        async def operation(
            store: MemoryInteractionStore,
            _unit: PgsqlUnitOfWork,
        ) -> _T:
            return await callback(store)

        async def persist(unit: PgsqlUnitOfWork, result: _T) -> None:
            record = getattr(result, "record", None)
            if (
                isinstance(record, InteractionRecord)
                and record.request.resolution is not None
            ):
                await self._reject_standalone_task_lifecycle(
                    unit,
                    record,
                )
                if record.request.state is RequestState.ANSWERED:
                    await self._ready_continuation(unit, record)
                else:
                    await self._invalidate_for_record(unit, record)
                await self._sync_resolution_keys(unit, record)

        return await self._run_memory_with_unit(
            operation_name,
            operation,
            mutate=True,
            after_persist=persist,
            selection=selection,
        )

    async def _scope_memory_operation(
        self,
        operation_name: str,
        callback: Callable[
            [MemoryInteractionStore],
            Awaitable[_T],
        ],
        *,
        selection: _InteractionSnapshotSelection | None = None,
    ) -> _T:
        async def operation(
            store: MemoryInteractionStore,
            _unit: PgsqlUnitOfWork,
        ) -> _T:
            return await callback(store)

        async def persist(unit: PgsqlUnitOfWork, result: _T) -> None:
            records = getattr(result, "records", ())
            if isinstance(records, tuple):
                for record in records:
                    if (
                        isinstance(record, InteractionRecord)
                        and record.request.resolution is not None
                    ):
                        await self._reject_standalone_task_lifecycle(
                            unit,
                            record,
                        )
                for record in records:
                    if (
                        isinstance(record, InteractionRecord)
                        and record.request.resolution is not None
                    ):
                        await self._invalidate_for_record(unit, record)
                        await self._sync_resolution_keys(unit, record)

        return await self._run_memory_with_unit(
            operation_name,
            operation,
            mutate=True,
            after_persist=persist,
            selection=selection,
        )

    async def _run_memory(
        self,
        operation: str,
        callback: Callable[[MemoryInteractionStore], Awaitable[_T]],
        *,
        mutate: bool,
        selection: _InteractionSnapshotSelection | None = None,
        admission_capability: _InteractionAdmissionCapability | None = None,
    ) -> _T:
        assert mutate or admission_capability is None
        if not mutate:
            return await self._run_memory_read(
                operation,
                callback,
                selection=selection,
            )

        async def with_unit(
            store: MemoryInteractionStore,
            _unit: PgsqlUnitOfWork,
        ) -> _T:
            return await callback(store)

        return await self._run_memory_with_unit(
            operation,
            with_unit,
            mutate=mutate,
            selection=selection,
            admission_capability=admission_capability,
        )

    async def _run_memory_read(
        self,
        operation: str,
        callback: Callable[[MemoryInteractionStore], Awaitable[_T]],
        *,
        selection: _InteractionSnapshotSelection | None = None,
    ) -> _T:
        """Authorize and project one coherent read-only snapshot."""

        async def read(
            store: MemoryInteractionStore,
            _records: tuple[InteractionRecord, ...],
        ) -> _T:
            return await callback(store)

        return await self._run_memory_read_snapshot(
            operation,
            read,
            selection=selection,
        )

    async def _run_memory_read_snapshot(
        self,
        operation: str,
        callback: Callable[
            [
                MemoryInteractionStore,
                tuple[InteractionRecord, ...],
            ],
            Awaitable[_T],
        ],
        *,
        selection: _InteractionSnapshotSelection | None = None,
    ) -> _T:
        """Project one coherent snapshot after its short transaction closes."""
        self._ensure_open()

        async def load(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(_SELECT_STORE_METADATA_SQL)
            metadata = await unit.cursor.fetchone()
            if metadata is None:
                raise PgsqlInteractionFeatureUnavailableError(
                    reason=(
                        "durable interaction schema is not at the "
                        "required head"
                    )
                )
            resolved_selection = await self._resolve_snapshot_selection(
                unit,
                selection,
            )
            records = await self._load_records(
                unit,
                selection=resolved_selection,
            )
            branches = await self._load_branches(
                unit,
                selection=resolved_selection,
            )
            if (
                resolved_selection is not None
                and resolved_selection.tolerate_invalid_branches
            ):
                records = _records_with_valid_branch_edges(
                    records,
                    branches,
                )
            if (
                resolved_selection is not None
                and resolved_selection.tolerate_invalid_continuations
            ):
                records = await self._records_with_valid_continuations(
                    unit,
                    records,
                    for_update=False,
                )
            return metadata, records, branches

        raw: object = await self._transaction(
            f"{operation}_snapshot",
            load,
            repeatable_read=True,
        )
        metadata, records, branches = cast(
            tuple[
                PgsqlRow,
                tuple[InteractionRecord, ...],
                tuple[InteractionBranchRecord, ...],
            ],
            raw,
        )
        backing = self._memory_backing(
            metadata=metadata,
            records=records,
            branches=branches,
        )
        async with self._runtime.lock:
            backing.resumers = dict(self._runtime.resumers)
            backing.admissions = dict(self._runtime.admissions)
        store = MemoryInteractionStore(backing)
        backing.handles.add(store)
        return await callback(store, records)

    async def _run_memory_with_unit(
        self,
        operation: str,
        callback: Callable[
            [MemoryInteractionStore, PgsqlUnitOfWork],
            Awaitable[_T],
        ],
        *,
        mutate: bool,
        after_persist: (
            Callable[
                [PgsqlUnitOfWork, _T],
                Awaitable[None],
            ]
            | None
        ) = None,
        selection: _InteractionSnapshotSelection | None = None,
        admission_capability: _InteractionAdmissionCapability | None = None,
    ) -> _T:
        self._ensure_open()
        assert admission_capability is None or selection is None
        initial_resumers: dict[str, InputResumer] = {}
        initial_admissions: dict[
            _InteractionAdmissionCapability,
            _MemoryAdmissionBinding,
        ] = {}
        committed_resumers: dict[str, InputResumer] | None = None
        committed_admissions: (
            dict[
                _InteractionAdmissionCapability,
                _MemoryAdmissionBinding,
            ]
            | None
        ) = None
        operation_selection = selection
        admission_binding: _MemoryAdmissionBinding | None = None
        deferred_resumer: _DeferredAdmissionResumer | None = None
        temporary_binding: _MemoryAdmissionBinding | None = None

        async def execute(unit: PgsqlUnitOfWork) -> object:
            nonlocal committed_admissions, committed_resumers
            metadata = await _lock_store_metadata(unit)
            resolved_selection = await self._resolve_snapshot_selection(
                unit,
                operation_selection,
            )
            scope_ownership_presence = (
                await self._load_scope_ownership_presence(
                    unit,
                    resolved_selection,
                )
            )
            records = await self._load_records(
                unit,
                selection=resolved_selection,
            )
            branches = await self._load_branches(
                unit,
                selection=resolved_selection,
            )
            if (
                resolved_selection is not None
                and resolved_selection.tolerate_invalid_branches
            ):
                records = _records_with_valid_branch_edges(
                    records,
                    branches,
                )
            if (
                resolved_selection is not None
                and resolved_selection.tolerate_invalid_continuations
            ):
                records = await self._records_with_valid_continuations(
                    unit,
                    records,
                    for_update=True,
                )
            backing = self._memory_backing(
                metadata=metadata,
                records=records,
                branches=branches,
                trusted_task_branch_closure=(
                    resolved_selection is not None
                    and resolved_selection.trusted_task_run_id is not None
                ),
                scope_ownership_presence=scope_ownership_presence,
            )
            backing.resumers = dict(initial_resumers)
            backing.admissions = dict(initial_admissions)
            if (
                admission_capability is not None
                and admission_binding is not None
                and deferred_resumer is not None
                and temporary_binding is not None
            ):
                continuation_key = str(admission_binding.continuation_id)
                if (
                    backing.resumers.get(continuation_key)
                    is admission_binding.resumer
                ):
                    backing.resumers[continuation_key] = deferred_resumer
                    backing.admissions[admission_capability] = (
                        temporary_binding
                    )
            store = MemoryInteractionStore(backing)
            backing.handles.add(store)
            try:
                result = await callback(store, unit)
            except (KeyboardInterrupt, SystemExit, CancelledError):
                raise
            except InputContractError:
                raise
            except BaseException as error:
                raise _InteractionCallbackError(error) from error
            if (
                admission_capability is not None
                and admission_binding is not None
                and temporary_binding is not None
                and backing.admissions.get(admission_capability)
                is temporary_binding
            ):
                backing.admissions[admission_capability] = admission_binding
            if mutate:
                snapshot = _snapshot_interaction_store_backing(backing.backing)
                await self._save_records(
                    unit,
                    _changed_records(records, snapshot.records),
                )
                await self._save_branches(
                    unit,
                    _changed_branches(branches, snapshot.branch_records),
                    roots=_branch_roots(snapshot.branch_records),
                )
                await unit.cursor.execute(
                    _UPDATE_STORE_METADATA_SQL,
                    (
                        snapshot.store_generation,
                        backing.schedule_revision,
                    ),
                )
            if after_persist is not None:
                await after_persist(unit, result)
            committed_resumers = dict(backing.resumers)
            committed_admissions = dict(backing.admissions)
            return result

        async with self._runtime.mutation_lock:
            async with self._runtime.lock:
                initial_resumers = dict(self._runtime.resumers)
                initial_admissions = dict(self._runtime.admissions)
            if admission_capability is not None:
                admission_binding = initial_admissions.get(
                    admission_capability
                )
                operation_selection = _admission_cleanup_snapshot(
                    admission_binding
                )
                if admission_binding is not None:
                    continuation_key = str(admission_binding.continuation_id)
                    if (
                        initial_resumers.get(continuation_key)
                        is admission_binding.resumer
                    ):
                        deferred_resumer = _DeferredAdmissionResumer()
                        temporary_handoff: Future[None] = (
                            get_running_loop().create_future()
                        )
                        temporary_binding = _MemoryAdmissionBinding(
                            request_id=admission_binding.request_id,
                            continuation_id=(
                                admission_binding.continuation_id
                            ),
                            resumer=deferred_resumer,
                            handoff=temporary_handoff,
                        )
            result: _T = await self._transaction(operation, execute)
            assert committed_resumers is not None
            assert committed_admissions is not None
            async with self._runtime.lock:
                self._runtime.resumers = committed_resumers
                self._runtime.admissions = committed_admissions
            if (
                admission_binding is not None
                and deferred_resumer is not None
                and deferred_resumer.notification is not None
            ):
                await _publish_admission_resumption(
                    admission_binding,
                    deferred_resumer.notification,
                )
            return result

    def _memory_backing(
        self,
        *,
        metadata: PgsqlRow,
        records: tuple[InteractionRecord, ...],
        branches: tuple[InteractionBranchRecord, ...],
        trusted_task_branch_closure: bool = False,
        scope_ownership_presence: _ScopeOwnershipPresence | None = None,
    ) -> _MemoryInteractionBacking:
        backing = _MemoryInteractionBacking(
            policy=self._policy,
            clock=self._clock,
            authorizer=self._authorizer,
            id_factory=self._id_factory,
            classifier=self._classifier,
        )
        store_generation = InteractionStoreGeneration(
            _row_int(metadata, "store_generation")
        )
        if trusted_task_branch_closure:
            assert scope_ownership_presence is None
            backing.backing = _new_partial_interaction_store_backing(
                records=records,
                branch_records=branches,
                store_generation=store_generation,
            )
        elif scope_ownership_presence is not None:
            backing.backing = _new_scoped_interaction_store_backing(
                records=records,
                branch_records=branches,
                store_generation=store_generation,
                scope=scope_ownership_presence.scope,
                principal=scope_ownership_presence.principal,
                actor_owned_record_match=(
                    scope_ownership_presence.actor_owned_record_match
                ),
                foreign_owned_record_match=(
                    scope_ownership_presence.foreign_owned_record_match
                ),
                actor_owned_branch_match=(
                    scope_ownership_presence.actor_owned_branch_match
                ),
                foreign_owned_branch_match=(
                    scope_ownership_presence.foreign_owned_branch_match
                ),
            )
        else:
            backing.backing = _new_interaction_store_backing(
                records=records,
                branch_records=branches,
                store_generation=store_generation,
            )
        backing.schedule_revision = cast(
            Any,
            _row_int(metadata, "schedule_revision"),
        )
        return backing

    async def _resolve_snapshot_selection(
        self,
        unit: PgsqlUnitOfWork,
        selection: _InteractionSnapshotSelection | None,
    ) -> _InteractionSnapshotSelection | None:
        """Resolve trusted task ownership without decrypting another scope."""
        if selection is None or selection.trusted_task_run_id is None:
            return selection
        task_run_id = selection.trusted_task_run_id
        await unit.cursor.execute(
            _SELECT_TASK_SCOPE_IDENTITIES_FOR_UPDATE_SQL,
            (task_run_id,),
        )
        rows = await unit.cursor.fetchall()
        if not rows:
            _task_coordinator_required("task_run_id")
        run_ids = {_row_str(row, "run_id") for row in rows}
        if run_ids != {task_run_id}:
            raise PgsqlInteractionStoreError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "task-bound interaction origin does not match its run",
            )
        digests = {_row_str(row, "scope_identity_digest") for row in rows}
        if len(digests) != 1:
            raise PgsqlInteractionStoreError(
                InputErrorCode.FORBIDDEN,
                "task_run_id",
                "task-bound interactions do not share one trusted owner",
            )
        return replace(
            selection,
            scope_identity_digest=digests.pop(),
        )

    async def _load_scope_ownership_presence(
        self,
        unit: PgsqlUnitOfWork,
        selection: _InteractionSnapshotSelection | None,
    ) -> _ScopeOwnershipPresence | None:
        """Load only aggregate ownership presence before actor decryption."""
        if selection is None or selection.mutation_scope is None:
            return None
        scope = selection.mutation_scope
        principal = selection.principal
        assert principal is not None
        actor_scope_identity = _scope_identity_digest(
            scope.run_id,
            principal,
        )
        await unit.cursor.execute(
            _SELECT_SCOPE_OWNERSHIP_PRESENCE_SQL,
            (
                scope.run_id,
                scope.run_id,
                scope.branch_id,
                scope.branch_id,
                scope.run_id,
                scope.include_descendants,
                scope.run_id,
                scope.turn_id,
                scope.turn_id,
                scope.task_id,
                scope.task_id,
                scope.agent_id,
                scope.agent_id,
                scope.branch_id,
                scope.run_id,
                scope.branch_id,
                actor_scope_identity,
                actor_scope_identity,
                actor_scope_identity,
                actor_scope_identity,
            ),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            raise PgsqlInteractionStoreError(
                InputErrorCode.UNAVAILABLE,
                "scope",
                "scope ownership query returned no result",
            )
        return _ScopeOwnershipPresence(
            scope=scope,
            principal=principal,
            actor_owned_record_match=_row_bool(
                row,
                "actor_owned_record_match",
            ),
            foreign_owned_record_match=_row_bool(
                row,
                "foreign_owned_record_match",
            ),
            actor_owned_branch_match=_row_bool(
                row,
                "actor_owned_branch_match",
            ),
            foreign_owned_branch_match=_row_bool(
                row,
                "foreign_owned_branch_match",
            ),
        )

    async def _records_with_valid_continuations(
        self,
        unit: PgsqlUnitOfWork,
        records: tuple[InteractionRecord, ...],
        *,
        for_update: bool,
    ) -> tuple[InteractionRecord, ...]:
        """Quarantine invalid continuation pairs from global maintenance."""
        query = (
            _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL
            if for_update
            else _SELECT_CONTINUATION_BY_REQUEST_SQL
        )
        selected: list[InteractionRecord] = []
        for record in records:
            await unit.cursor.execute(
                query,
                (record.request.request_id,),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                selected.append(record)
                continue
            try:
                continuation = self._continuation_from_row(row)
                if (
                    continuation.request_id != record.request.request_id
                    or continuation.continuation_id
                    != record.request.continuation_id
                    or continuation.origin != record.request.origin
                ):
                    _invalid_payload("continuation.binding")
            except InputContractError:
                continue
            selected.append(record)
        return tuple(selected)

    async def _enforce_create_process_capacity(
        self,
        unit: PgsqlUnitOfWork,
        command: CreateInteractionCommand,
        result: CreateInteractionResult,
    ) -> None:
        """Enforce global capacity without decrypting unrelated records."""
        if not isinstance(result, CreateInteractionApplied):
            return
        await unit.cursor.execute(_SELECT_PENDING_RECORD_COUNT_SQL)
        row = await unit.cursor.fetchone()
        if row is None:
            raise PgsqlInteractionFeatureUnavailableError(
                reason="interaction capacity query returned no result"
            )
        if _row_int(row, "pending_count") >= (
            self._policy.maximum_pending_interactions_per_process
        ):
            raise _PgsqlCreateCapacityError(command)

    async def _transaction(
        self,
        operation: str,
        callback: Callable[[PgsqlUnitOfWork], Awaitable[object]],
        *,
        repeatable_read: bool = False,
    ) -> _T:
        self._ensure_open()
        try:
            async with self._database.connection() as connection:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        if repeatable_read:
                            await cursor.execute(_SET_REPEATABLE_READ_ONLY_SQL)
                        result = await callback(
                            PgsqlUnitOfWork(
                                connection=connection,
                                cursor=cursor,
                            )
                        )
            return cast(_T, result)
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except InputContractError:
            raise
        except TaskStoreError:
            raise
        except _InteractionCallbackError as error:
            raise error.error
        except BaseException as error:
            failure = classify_pgsql_error(error, operation=operation)
            raise PgsqlOperationError(failure) from None

    async def _insert_continuation(
        self,
        unit: PgsqlUnitOfWork,
        continuation: PortableContinuation,
        *,
        task_run_id: str | None,
        checkpoint_id: str | None,
    ) -> None:
        await unit.cursor.execute(
            _SELECT_RECORD_DEADLINE_FOR_UPDATE_SQL,
            (continuation.request_id,),
        )
        deadline_row = await unit.cursor.fetchone()
        if deadline_row is None:
            raise InteractionNotFoundError()
        request_expires_at = _row_datetime(
            deadline_row,
            "absolute_expires_at",
        )
        if request_expires_at != continuation.expires_at:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation.expires_at",
                "continuation expiry does not match the interaction deadline",
            )
        encrypted = self._encrypt(
            encode_portable_continuation(continuation).encode("utf-8"),
            kind="continuation",
            identifier=str(continuation.continuation_id),
        )
        retention_deadline = continuation.expires_at + timedelta(
            days=self._store_policy.retention_days
        )
        await unit.cursor.execute(
            _INSERT_CONTINUATION_SQL,
            (
                continuation.continuation_id,
                checkpoint_id,
                continuation.request_id,
                task_run_id,
                DurableContinuationLifecycle.PENDING.value,
                continuation.state_revision,
                continuation.store_revision,
                continuation.fencing_token,
                encrypted.ciphertext,
                encrypted.key_id,
                encrypted.algorithm,
                _json(encrypted.metadata or {}),
                continuation.expires_at,
                retention_deadline,
                continuation.created_at,
                continuation.updated_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise ContinuationStoreConflictError(
                "continuation.continuation_id",
                "continuation already exists",
            )

    async def _ready_continuation(
        self,
        unit: PgsqlUnitOfWork,
        record: InteractionRecord,
    ) -> None:
        await unit.cursor.execute(
            _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
            (record.request.request_id,),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            return
        continuation = self._continuation_from_row(row)
        lifecycle = _lifecycle(row)
        if lifecycle in {
            DurableContinuationLifecycle.READY,
            DurableContinuationLifecycle.CLAIMED,
            DurableContinuationLifecycle.DISPATCHING,
            DurableContinuationLifecycle.COMPLETED,
        }:
            return
        if lifecycle is DurableContinuationLifecycle.INVALIDATED:
            raise ContinuationStoreConflictError(
                "continuation",
                "invalidated continuation cannot become ready",
            )
        resolution = record.request.resolution
        assert resolution is not None
        updated = replace(
            continuation,
            state_revision=record.request.state_revision,
            store_revision=_next_continuation_revision(continuation),
            updated_at=resolution.resolved_at,
        )
        await self._update_continuation(
            unit,
            updated,
            lifecycle=DurableContinuationLifecycle.READY,
            expected_revision=continuation.store_revision,
        )
        await unit.cursor.execute(
            _INSERT_OUTBOX_SQL,
            (
                _uuid_id(),
                continuation.continuation_id,
                continuation.request_id,
                _row_optional_str(row, "task_run_id"),
                record.request.state_revision,
                resolution.resolved_at,
                resolution.resolved_at,
                resolution.resolved_at,
            ),
        )

    async def _invalidate_for_record(
        self,
        unit: PgsqlUnitOfWork,
        record: InteractionRecord,
    ) -> None:
        await unit.cursor.execute(
            _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
            (record.request.request_id,),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            return
        lifecycle = _lifecycle(row)
        if lifecycle in {
            DurableContinuationLifecycle.COMPLETED,
            DurableContinuationLifecycle.INVALIDATED,
        }:
            return
        if lifecycle is DurableContinuationLifecycle.DISPATCHING:
            raise ContinuationDispatchAmbiguousError()
        continuation = self._continuation_from_row(row)
        resolved_at = cast(Any, record.request.resolution).resolved_at
        updated = replace(
            continuation,
            claim=_invalidated_claim(continuation),
            state_revision=record.request.state_revision,
            store_revision=_next_continuation_revision(continuation),
            updated_at=resolved_at,
        )
        await self._update_continuation(
            unit,
            updated,
            lifecycle=DurableContinuationLifecycle.INVALIDATED,
            expected_revision=continuation.store_revision,
            invalid_reason=record.request.state.value,
        )

    async def _reject_standalone_task_lifecycle(
        self,
        unit: PgsqlUnitOfWork,
        record: InteractionRecord,
    ) -> None:
        """Reject lifecycle mutation that requires task-store convergence."""
        await unit.cursor.execute(
            _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
            (record.request.request_id,),
        )
        row = await unit.cursor.fetchone()
        if (
            row is not None
            and _row_optional_str(row, "task_run_id") is not None
        ):
            raise PgsqlInteractionStoreError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "interaction.task_lifecycle",
                "task-bound interaction requires the durable task coordinator",
            )

    async def _sync_resolution_keys(
        self,
        unit: PgsqlUnitOfWork,
        record: InteractionRecord,
    ) -> None:
        for entry in record.idempotency_ledger:
            await unit.cursor.execute(
                _INSERT_RESOLUTION_KEY_SQL,
                (
                    record.request.request_id,
                    entry.key,
                    entry.resolution_digest,
                    record.request.state_revision,
                ),
            )

    async def _update_continuation(
        self,
        unit: PgsqlUnitOfWork,
        continuation: PortableContinuation,
        *,
        lifecycle: DurableContinuationLifecycle,
        expected_revision: ContinuationStoreRevision,
        invalid_reason: str | None = None,
        dispatch_completed_at: datetime | None = None,
        settled_claim_owner_id: ContinuationClaimOwnerId | None = None,
    ) -> None:
        encrypted = self._encrypt(
            encode_portable_continuation(continuation).encode("utf-8"),
            kind="continuation",
            identifier=str(continuation.continuation_id),
        )
        claim_owner = continuation.claim.owner_id
        if settled_claim_owner_id is not None:
            assert claim_owner is None
            assert lifecycle in {
                DurableContinuationLifecycle.READY,
                DurableContinuationLifecycle.COMPLETED,
            }
            assert continuation.claim.state in {
                ContinuationClaimState.FAILED_SAFE_TO_RETRY,
                ContinuationClaimState.COMPLETED,
            }
            claim_owner = settled_claim_owner_id
        claim_expiry = continuation.claim.lease_expires_at
        dispatch = continuation.dispatch
        await unit.cursor.execute(
            _UPDATE_CONTINUATION_SQL,
            (
                lifecycle.value,
                continuation.state_revision,
                continuation.store_revision,
                claim_owner,
                claim_expiry,
                continuation.fencing_token,
                dispatch.dispatch_id if dispatch is not None else None,
                dispatch.marked_at if dispatch is not None else None,
                dispatch_completed_at,
                lifecycle is DurableContinuationLifecycle.DISPATCHING,
                invalid_reason,
                encrypted.ciphertext,
                encrypted.key_id,
                encrypted.algorithm,
                _json(encrypted.metadata or {}),
                continuation.updated_at,
                continuation.continuation_id,
                expected_revision,
            ),
        )
        if await unit.cursor.fetchone() is None:
            _stale("continuation.store_revision")

    def _claimed_continuation(
        self,
        row: PgsqlRow,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
        allow_expired: bool = False,
    ) -> PortableContinuation:
        continuation = self._continuation_from_row(row)
        if (
            continuation.store_revision != expected_store_revision
            or ContinuationStoreRevision(_row_int(row, "store_revision"))
            != expected_store_revision
        ):
            _stale("expected_store_revision")
        if _lifecycle(row) is DurableContinuationLifecycle.DISPATCHING:
            raise ContinuationDispatchAmbiguousError()
        if _lifecycle(row) is not DurableContinuationLifecycle.CLAIMED:
            raise ContinuationStoreConflictError(
                "continuation.claim",
                "continuation does not have a pre-dispatch claim",
            )
        if (
            continuation.claim.owner_id != owner_id
            or _row_optional_str(row, "claim_owner_id") != owner_id
            or continuation.fencing_token != fencing_token
            or ContinuationFencingToken(_row_int(row, "fencing_token"))
            != fencing_token
        ):
            _stale("continuation.fencing_token")
        lease_expires_at = continuation.claim.lease_expires_at
        assert lease_expires_at is not None
        if continuation.expires_at <= now:
            raise PgsqlInteractionStoreError(
                InputErrorCode.EXPIRED,
                "continuation",
                "continuation has expired",
            )
        if not allow_expired and lease_expires_at <= now:
            raise ContinuationStoreConflictError(
                "continuation.claim.lease_expires_at",
                "continuation claim lease has expired",
            )
        return continuation

    def _ambiguous_continuation(
        self,
        row: PgsqlRow,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
    ) -> PortableContinuation:
        continuation = self._continuation_from_row(row)
        if continuation.store_revision != expected_store_revision:
            _stale("expected_store_revision")
        if (
            _lifecycle(row) is not DurableContinuationLifecycle.DISPATCHING
            or continuation.claim.state
            is not ContinuationClaimState.DISPATCHED_AMBIGUOUS
        ):
            raise ContinuationStoreConflictError(
                "continuation.dispatch",
                "continuation is not in the dispatching state",
            )
        if (
            continuation.claim.owner_id != owner_id
            or continuation.fencing_token != fencing_token
        ):
            _stale("continuation.fencing_token")
        return continuation

    async def _load_records(
        self,
        unit: PgsqlUnitOfWork,
        *,
        selection: _InteractionSnapshotSelection | None = None,
    ) -> tuple[InteractionRecord, ...]:
        if selection is not None and not selection.include_records:
            return ()
        trusted_task_run_id = (
            None if selection is None else selection.trusted_task_run_id
        )
        if (
            selection is not None
            and selection.admission_request_id is not None
        ):
            assert selection.admission_continuation_id is not None
            await unit.cursor.execute(
                _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL,
                (
                    selection.admission_request_id,
                    selection.admission_continuation_id,
                ),
            )
        elif selection is not None and selection.correlation is not None:
            assert selection.principal is not None
            await unit.cursor.execute(
                _SELECT_SCOPED_RECORD_SQL,
                _correlation_parameters(
                    selection.correlation,
                    selection.principal,
                ),
            )
        elif trusted_task_run_id is not None:
            assert selection is not None
            assert selection.scope_identity_digest is not None
            await unit.cursor.execute(
                _SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL,
                (trusted_task_run_id,),
            )
        elif selection is not None and selection.run_id is not None:
            if selection.principal is None:
                assert selection.scope_identity_digest is not None
                await unit.cursor.execute(
                    _SELECT_SCOPE_RECORDS_SQL,
                    (
                        selection.run_id,
                        selection.scope_identity_digest,
                    ),
                )
            else:
                await unit.cursor.execute(
                    _SELECT_SCOPE_RECORDS_SQL,
                    (
                        selection.run_id,
                        _scope_identity_digest(
                            selection.run_id,
                            selection.principal,
                        ),
                    ),
                )
        else:
            await unit.cursor.execute(_SELECT_RECORDS_SQL)
        records: list[InteractionRecord] = []
        for row in await unit.cursor.fetchall():
            try:
                records.append(self._record_from_row(row))
            except InputContractError:
                if selection is None or not selection.tolerate_invalid_records:
                    raise
        return tuple(records)

    async def _load_branches(
        self,
        unit: PgsqlUnitOfWork,
        *,
        selection: _InteractionSnapshotSelection | None = None,
    ) -> tuple[InteractionBranchRecord, ...]:
        if selection is not None and not selection.include_branches:
            return ()
        trusted_task_run_id = (
            None if selection is None else selection.trusted_task_run_id
        )
        if trusted_task_run_id is not None:
            assert selection is not None
            assert selection.scope_identity_digest is not None
            await unit.cursor.execute(
                _SELECT_TASK_BRANCH_CLOSURE_SQL,
                (
                    trusted_task_run_id,
                    selection.run_id,
                    selection.scope_identity_digest,
                ),
            )
        elif selection is not None and selection.run_id is not None:
            if selection.principal is None:
                assert selection.scope_identity_digest is not None
                await unit.cursor.execute(
                    _SELECT_SCOPE_BRANCHES_SQL,
                    (
                        selection.run_id,
                        selection.scope_identity_digest,
                    ),
                )
            else:
                await unit.cursor.execute(
                    _SELECT_SCOPE_BRANCHES_SQL,
                    (
                        selection.run_id,
                        _scope_identity_digest(
                            selection.run_id,
                            selection.principal,
                        ),
                    ),
                )
        else:
            await unit.cursor.execute(_SELECT_BRANCHES_SQL)
        decoded: list[tuple[PgsqlRow, InteractionBranchRecord]] = []
        for row in await unit.cursor.fetchall():
            try:
                decoded.append((row, self._branch_from_row(row)))
            except InputContractError:
                if (
                    selection is None
                    or not selection.tolerate_invalid_branches
                ):
                    raise
        return _branches_with_valid_row_roots(
            tuple(decoded),
            tolerate_invalid=(
                selection is not None and selection.tolerate_invalid_branches
            ),
        )

    async def _save_records(
        self,
        unit: PgsqlUnitOfWork,
        records: tuple[InteractionRecord, ...],
    ) -> None:
        for record in records:
            encrypted = self._encrypt(
                _encode_record(record),
                kind="record",
                identifier=str(record.request.request_id),
            )
            retention_deadline = record.absolute_expires_at + timedelta(
                days=self._store_policy.retention_days
            )
            origin = record.request.origin
            await unit.cursor.execute(
                _UPSERT_RECORD_SQL,
                (
                    record.request.request_id,
                    record.request.continuation_id,
                    origin.run_id,
                    origin.turn_id,
                    origin.task_id,
                    origin.agent_id,
                    origin.branch_id,
                    origin.model_call_id,
                    _scope_identity_digest(
                        origin.run_id,
                        origin.principal,
                    ),
                    record.request.state.value,
                    record.request.state_revision,
                    record.store_revision,
                    record.absolute_expires_at,
                    retention_deadline,
                    encrypted.ciphertext,
                    encrypted.key_id,
                    encrypted.algorithm,
                    _json(encrypted.metadata or {}),
                    record.request.created_at,
                    _record_updated_at(record),
                ),
            )
            if await unit.cursor.fetchone() is None:
                _scope_identity_conflict("record")

    async def _save_branches(
        self,
        unit: PgsqlUnitOfWork,
        branches: tuple[InteractionBranchRecord, ...],
        *,
        roots: Mapping[tuple[str, str, str], str],
    ) -> None:
        for branch in branches:
            registration = branch.registration
            scope_identity_digest = _scope_identity_digest(
                registration.run_id,
                registration.principal,
            )
            encrypted = self._encrypt(
                _encode_branch(branch),
                kind="branch",
                identifier=(
                    f"{registration.run_id}:{scope_identity_digest}:"
                    f"{registration.branch_id}"
                ),
            )
            await unit.cursor.execute(
                _UPSERT_BRANCH_SQL,
                (
                    registration.run_id,
                    registration.branch_id,
                    registration.parent_branch_id,
                    roots[
                        (
                            str(registration.run_id),
                            scope_identity_digest,
                            str(registration.branch_id),
                        )
                    ],
                    branch.store_revision,
                    scope_identity_digest,
                    encrypted.ciphertext,
                    encrypted.key_id,
                    encrypted.algorithm,
                    _json(encrypted.metadata or {}),
                ),
            )
            if await unit.cursor.fetchone() is None:
                _scope_identity_conflict("branch")

    def _record_from_row(self, row: PgsqlRow) -> InteractionRecord:
        plaintext = self._decrypt(
            row,
            kind="record",
            identifier=_row_str(row, "request_id"),
        )
        record = _decode_record(plaintext)
        _validate_record_row_identity(row, record)
        return record

    def _branch_from_row(self, row: PgsqlRow) -> InteractionBranchRecord:
        plaintext = self._decrypt(
            row,
            kind="branch",
            identifier=(
                f"{_row_str(row, 'run_id')}:"
                f"{_row_str(row, 'scope_identity_digest')}:"
                f"{_row_str(row, 'branch_id')}"
            ),
        )
        branch = _decode_branch(plaintext)
        _validate_branch_row_identity(row, branch)
        return branch

    def _continuation_from_row(
        self,
        row: PgsqlRow,
    ) -> PortableContinuation:
        plaintext = self._decrypt(
            row,
            kind="continuation",
            identifier=_row_str(row, "continuation_id"),
        )
        try:
            text = plaintext.decode("utf-8")
            payload = loads(text)
        except (UnicodeError, ValueError) as error:
            raise PgsqlInteractionStoreError(
                InputErrorCode.SNAPSHOT_INVALID,
                "continuation",
                "encrypted continuation payload is invalid",
            ) from error
        if not isinstance(payload, Mapping):
            raise PgsqlInteractionStoreError(
                InputErrorCode.SNAPSHOT_INVALID,
                "continuation",
                "encrypted continuation payload is invalid",
            )
        binding = _continuation_binding(payload)
        return decode_portable_continuation(
            text,
            expected_binding=binding,
        )

    def _durable_continuation_record(
        self,
        row: PgsqlRow,
    ) -> DurableContinuationRecord:
        return DurableContinuationRecord(
            continuation=self._continuation_from_row(row),
            task_run_id=_row_optional_str(row, "task_run_id"),
            checkpoint_id=_row_optional_str(row, "checkpoint_id"),
        )

    def _encrypt(
        self,
        plaintext: bytes,
        *,
        kind: str,
        identifier: str,
    ) -> EncryptedPrivacyValue:
        try:
            encrypted = self._cipher.encrypt(
                plaintext,
                purpose=TaskKeyPurpose.RAW_VALUE,
                key_id=self._store_policy.encryption_key_id,
                context={"kind": kind, "identifier": identifier},
            )
        except Exception as error:
            raise PgsqlInteractionFeatureUnavailableError(
                reason="interaction encryption key is unavailable"
            ) from error
        if not isinstance(encrypted, EncryptedPrivacyValue):
            raise PgsqlInteractionFeatureUnavailableError(
                reason="interaction encryption provider returned invalid data"
            )
        return encrypted

    def _decrypt(
        self,
        row: PgsqlRow,
        *,
        kind: str,
        identifier: str,
    ) -> bytes:
        encrypted = EncryptedPrivacyValue(
            ciphertext=_row_bytes(row, "ciphertext"),
            key_id=_row_str(row, "encryption_key_id"),
            algorithm=_row_str(row, "encryption_algorithm"),
            metadata=_row_string_mapping(row, "encryption_metadata"),
        )
        try:
            plaintext = self._cipher.decrypt(
                encrypted,
                purpose=TaskKeyPurpose.RAW_VALUE,
                context={"kind": kind, "identifier": identifier},
            )
        except Exception as error:
            raise PgsqlInteractionFeatureUnavailableError(
                reason="interaction decryption key is unavailable"
            ) from error
        if not isinstance(plaintext, bytes) or not plaintext:
            raise PgsqlInteractionStoreError(
                InputErrorCode.SNAPSHOT_INVALID,
                f"{kind}.ciphertext",
                "decrypted interaction payload is invalid",
            )
        return plaintext

    def _ensure_open(self) -> None:
        if self._closed:
            raise InteractionStoreClosedError()


@final
class PgsqlDurableTaskCoordinator:
    """Commit durable interaction and task-queue boundaries together."""

    def __init__(
        self,
        interaction_store: PgsqlInteractionStore,
        task_store: _PgsqlTaskTransactionStore,
    ) -> None:
        assert isinstance(interaction_store, PgsqlInteractionStore)
        assert hasattr(task_store, "_suspend_claim_in_unit")
        assert hasattr(task_store, "_requeue_suspended_in_unit")
        assert hasattr(task_store, "_terminalize_suspended_in_unit")
        assert hasattr(task_store, "_validate_suspended_run_in_unit")
        assert hasattr(task_store, "_settle_claim_in_unit")
        assert hasattr(
            task_store,
            "_terminalize_completed_claim_in_unit",
        )
        assert hasattr(task_store, "_release_claimed_reentry_in_unit")
        assert hasattr(task_store, "_fail_claimed_reentry_in_unit")
        assert hasattr(task_store, "_release_running_reentry_in_unit")
        assert hasattr(task_store, "_cancel_partial_reentry_in_unit")
        if task_store.database is not interaction_store._database:
            raise ValueError(
                "durable interaction and task stores must share one database"
            )
        self._interaction_store = interaction_store
        self._task_store = task_store

    async def create_and_suspend(
        self,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskSuspension:
        """Create a checkpoint and release its worker claim atomically."""
        if not isinstance(command, CreateInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be a create interaction command",
            )
        if type(continuation) is not PortableContinuation:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "continuation",
                "value must be a portable continuation",
            )
        if command.resumer is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command.resumer",
                "durable task suspension cannot retain a live resumer",
            )
        self._validate_continuation(command, continuation)
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(segment_id, "segment_id")
        _assert_opaque(task_run_id, "task_run_id")
        if str(command.request.origin.run_id) != task_run_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "task run does not match the interaction origin",
            )
        if checkpoint_id is not None:
            _assert_opaque(checkpoint_id, "checkpoint_id")
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)
        suspension: TaskQueueSuspension | None = None

        async def create(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> CreateInteractionResult:
            result = await store.create(command)
            await self._interaction_store._enforce_create_process_capacity(
                unit,
                command,
                result,
            )
            return result

        async def persist(
            unit: PgsqlUnitOfWork,
            result: CreateInteractionResult,
        ) -> None:
            nonlocal suspension
            if not isinstance(result, CreateInteractionApplied):
                raise PgsqlInteractionStoreError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "interaction.create",
                    "durable task interaction could not be created",
                )
            await self._interaction_store._insert_continuation(
                unit,
                continuation,
                task_run_id=task_run_id,
                checkpoint_id=checkpoint_id,
            )
            suspension = await self._task_store._suspend_claim_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                request_id=str(continuation.request_id),
                continuation_id=str(continuation.continuation_id),
                checkpoint_id=checkpoint_id,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        result = await self._interaction_store._run_memory_with_unit(
            "durable_task_create_and_suspend",
            create,
            mutate=True,
            after_persist=persist,
            selection=_create_snapshot(command),
        )
        assert isinstance(result, CreateInteractionApplied)
        assert suspension is not None
        return PgsqlDurableTaskSuspension(
            interaction=result,
            suspension=suspension,
        )

    async def create_pending_interaction(
        self,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        *,
        task_run_id: str,
        checkpoint_id: str,
    ) -> CreateInteractionApplied:
        """Attach one additional branch to an already suspended task."""
        if not isinstance(command, CreateInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be a create interaction command",
            )
        if type(continuation) is not PortableContinuation:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "continuation",
                "value must be a portable continuation",
            )
        if command.resumer is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command.resumer",
                "durable task suspension cannot retain a live resumer",
            )
        self._validate_continuation(command, continuation)
        _assert_opaque(task_run_id, "task_run_id")
        _assert_opaque(checkpoint_id, "checkpoint_id")
        if str(command.request.origin.run_id) != task_run_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "task run does not match the interaction origin",
            )

        async def create(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> CreateInteractionResult:
            result = await store.create(command)
            await self._interaction_store._enforce_create_process_capacity(
                unit,
                command,
                result,
            )
            return result

        async def persist(
            unit: PgsqlUnitOfWork,
            result: CreateInteractionResult,
        ) -> None:
            if not isinstance(result, CreateInteractionApplied):
                raise PgsqlInteractionStoreError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "interaction.create",
                    "durable task interaction could not be created",
                )
            await self._task_store._validate_suspended_run_in_unit(
                unit,
                task_run_id=task_run_id,
            )
            await self._interaction_store._insert_continuation(
                unit,
                continuation,
                task_run_id=task_run_id,
                checkpoint_id=checkpoint_id,
            )

        result = await self._interaction_store._run_memory_with_unit(
            "durable_task_create_pending_interaction",
            create,
            mutate=True,
            after_persist=persist,
            selection=_create_snapshot(command),
        )
        assert isinstance(result, CreateInteractionApplied)
        return result

    async def complete_and_resuspend(
        self,
        completion: ContinuationCompletionCommand,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskResuspension:
        """Complete one dispatch and persist its successor suspension."""
        if type(completion) is not ContinuationCompletionCommand:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "completion",
                "value must be a continuation completion command",
            )
        if not isinstance(command, CreateInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be a create interaction command",
            )
        if type(continuation) is not PortableContinuation:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "continuation",
                "value must be a portable continuation",
            )
        if command.resumer is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command.resumer",
                "durable task suspension cannot retain a live resumer",
            )
        self._validate_continuation(command, continuation)
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(segment_id, "segment_id")
        _assert_opaque(task_run_id, "task_run_id")
        _assert_opaque(checkpoint_id, "checkpoint_id")
        self._validate_successor(
            completion,
            command=command,
            continuation=continuation,
            task_run_id=task_run_id,
        )
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)
        completed: PortableContinuation | None = None
        suspension: TaskQueueSuspension | None = None

        async def create(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> CreateInteractionResult:
            result = await store.create(command)
            await self._interaction_store._enforce_create_process_capacity(
                unit,
                command,
                result,
            )
            return result

        async def persist(
            unit: PgsqlUnitOfWork,
            result: CreateInteractionResult,
        ) -> None:
            nonlocal completed, suspension
            if not isinstance(result, CreateInteractionApplied):
                raise PgsqlInteractionStoreError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "interaction.create",
                    "successor task interaction could not be created",
                )
            completed = (
                await self._interaction_store._complete_continuation_in_unit(
                    unit,
                    continuation_id=completion.continuation_id,
                    expected_store_revision=(
                        completion.expected_store_revision
                    ),
                    owner_id=completion.owner_id,
                    fencing_token=completion.fencing_token,
                    result_digest=completion.result_digest,
                    now=observed_at,
                    expected_task_run_id=task_run_id,
                    expected_task_segment_id=segment_id,
                )
            )
            self._validate_completed_successor(
                completed,
                continuation,
            )
            await self._interaction_store._insert_continuation(
                unit,
                continuation,
                task_run_id=task_run_id,
                checkpoint_id=checkpoint_id,
            )
            suspension = await self._task_store._suspend_claim_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                request_id=str(continuation.request_id),
                continuation_id=str(continuation.continuation_id),
                checkpoint_id=checkpoint_id,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        result = await self._interaction_store._run_memory_with_unit(
            "durable_task_complete_and_resuspend",
            create,
            mutate=True,
            after_persist=persist,
            selection=_create_snapshot(command),
        )
        assert isinstance(result, CreateInteractionApplied)
        assert completed is not None
        assert suspension is not None
        return PgsqlDurableTaskResuspension(
            completed_continuation=completed,
            interaction=result,
            suspension=suspension,
        )

    async def settle_resume(
        self,
        completion: ContinuationCompletionCommand,
        settlement: TaskDurableResumeSettlement,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskSettlement:
        """Settle one resumed continuation and terminal task atomically."""
        if type(completion) is not ContinuationCompletionCommand:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "completion",
                "value must be a continuation completion command",
            )
        expected_digest = task_durable_resume_settlement_digest(settlement)
        if completion.result_digest != expected_digest:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "completion.result_digest",
                "completion digest does not match the task settlement",
            )
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(segment_id, "segment_id")
        _assert_opaque(task_run_id, "task_run_id")
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            (
                completed,
                replayed,
            ) = await self._complete_settlement_continuation_in_unit(
                unit,
                completion=completion,
                task_run_id=task_run_id,
                segment_id=segment_id,
                now=observed_at,
            )
            task_completion = await self._task_store._settle_claim_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                task_run_id=task_run_id,
                settlement=settlement,
                observed_at=observed_at,
                metadata=safe_metadata,
                replay_only=replayed,
            )
            await unit.cursor.execute(
                _DEAD_OUTBOX_SQL,
                (observed_at, completion.continuation_id),
            )
            return PgsqlDurableTaskSettlement(
                completed_continuation=completed,
                completion=task_completion,
            )

        return await self._interaction_store._transaction(
            "durable_task_settle_resume",
            execute,
        )

    async def terminalize_completed_resume(
        self,
        completion: ContinuationCompletionCommand,
        settlement: TaskDurableResumeFailure | TaskDurableResumeCancellation,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskSettlement:
        """Terminalize only task state after provider completion."""
        if type(completion) is not ContinuationCompletionCommand:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "completion",
                "value must be a continuation completion command",
            )
        if type(settlement) not in {
            TaskDurableResumeFailure,
            TaskDurableResumeCancellation,
        }:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "settlement",
                "completed provider settlement must fail or cancel the task",
            )
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(segment_id, "segment_id")
        _assert_opaque(task_run_id, "task_run_id")
        _assert_opaque(request_id, "request_id")
        _assert_opaque(checkpoint_id, "checkpoint_id")
        if completion.owner_id != claim_token:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "completion.owner_id",
                "provider completion owner does not match the task claim",
            )
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            completed = await self._completed_task_continuation_in_unit(
                unit,
                completion=completion,
                task_run_id=task_run_id,
                request_id=request_id,
                checkpoint_id=checkpoint_id,
            )
            task_completion = (
                await self._task_store._terminalize_completed_claim_in_unit(
                    unit,
                    queue_item_id=queue_item_id,
                    claim_token=claim_token,
                    segment_id=segment_id,
                    task_run_id=task_run_id,
                    request_id=request_id,
                    continuation_id=str(completion.continuation_id),
                    checkpoint_id=checkpoint_id,
                    settlement=settlement,
                    observed_at=observed_at,
                    metadata=safe_metadata,
                )
            )
            await unit.cursor.execute(
                _DEAD_OUTBOX_SQL,
                (observed_at, completion.continuation_id),
            )
            return PgsqlDurableTaskSettlement(
                completed_continuation=completed,
                completion=task_completion,
            )

        return await self._interaction_store._transaction(
            "durable_task_terminalize_completed_resume",
            execute,
        )

    async def mark_resume_ambiguous(
        self,
        completion: ContinuationCompletionCommand,
        failure: TaskDurableResumeFailure,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskAmbiguity:
        """Fence ambiguous provider dispatch and fail its task atomically."""
        if type(completion) is not ContinuationCompletionCommand:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "completion",
                "value must be a continuation completion command",
            )
        if type(failure) is not TaskDurableResumeFailure:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "failure",
                "value must be a durable resume failure",
            )
        expected_digest = task_durable_resume_settlement_digest(failure)
        if completion.result_digest != expected_digest:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "completion.result_digest",
                "completion digest does not match the ambiguity failure",
            )
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(segment_id, "segment_id")
        _assert_opaque(task_run_id, "task_run_id")
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _lock_resumed_task_continuation(
                unit,
                continuation_id=completion.continuation_id,
                task_run_id=task_run_id,
                segment_id=segment_id,
            )
            ambiguous = self._interaction_store._ambiguous_continuation(
                row,
                expected_store_revision=(completion.expected_store_revision),
                owner_id=completion.owner_id,
                fencing_token=completion.fencing_token,
            )
            task_completion = await self._task_store._settle_claim_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                task_run_id=task_run_id,
                settlement=failure,
                observed_at=observed_at,
                metadata=safe_metadata,
            )
            await unit.cursor.execute(
                _DEAD_OUTBOX_SQL,
                (observed_at, completion.continuation_id),
            )
            return PgsqlDurableTaskAmbiguity(
                ambiguous_continuation=ambiguous,
                completion=task_completion,
            )

        return await self._interaction_store._transaction(
            "durable_task_mark_resume_ambiguous",
            execute,
        )

    async def release_claimed_reentry(
        self,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueReentry:
        """Release one pre-dispatch claimed task reentry."""
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(task_run_id, "task_run_id")
        _assert_opaque(request_id, "request_id")
        _assert_opaque(continuation_id, "continuation_id")
        _assert_opaque(checkpoint_id, "checkpoint_id")
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await self._task_store._release_claimed_reentry_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                task_run_id=task_run_id,
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        return await self._interaction_store._transaction(
            "durable_task_release_claimed_reentry",
            execute,
        )

    async def fail_claimed_reentry(
        self,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str | None,
        continuation_id: str | None,
        checkpoint_id: str | None,
        result: TaskExecutionResult,
        reason: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion:
        """Fail one claimed task reentry without ordinary retry."""
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(task_run_id, "task_run_id")
        correlation = (request_id, continuation_id, checkpoint_id)
        if any(value is None for value in correlation):
            if any(value is not None for value in correlation):
                raise InputValidationError(
                    InputErrorCode.CORRELATION_MISMATCH,
                    "reentry.provenance",
                    "reentry provenance must be complete or entirely absent",
                )
        else:
            _assert_opaque(cast(str, request_id), "request_id")
            _assert_opaque(
                cast(str, continuation_id),
                "continuation_id",
            )
            _assert_opaque(cast(str, checkpoint_id), "checkpoint_id")
        if not isinstance(result, TaskExecutionResult):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result",
                "value must be a task execution result",
            )
        _assert_opaque(reason, "reason")
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await self._task_store._fail_claimed_reentry_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                task_run_id=task_run_id,
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
                result=result,
                reason=reason,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        return await self._interaction_store._transaction(
            "durable_task_fail_claimed_reentry",
            execute,
        )

    async def fail_admitted_reentry(
        self,
        rejection: ContinuationRejectionCommand,
        failure: TaskDurableResumeFailure,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskRejection:
        """Reject one pre-dispatch admission and fail its task atomically."""
        if type(rejection) is not ContinuationRejectionCommand:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "rejection",
                "value must be a continuation rejection command",
            )
        if type(failure) is not TaskDurableResumeFailure:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "failure",
                "value must be a durable resume failure",
            )
        expected_digest = task_durable_resume_settlement_digest(failure)
        if rejection.result_digest != expected_digest:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "rejection.result_digest",
                "rejection digest does not match the task failure",
            )
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(task_run_id, "task_run_id")
        _assert_opaque(request_id, "request_id")
        _assert_opaque(continuation_id, "continuation_id")
        _assert_opaque(checkpoint_id, "checkpoint_id")
        if (
            str(rejection.owner_id) != claim_token
            or str(rejection.continuation_id) != continuation_id
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "rejection",
                "rejection fence does not match the claimed task",
            )
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            (
                rejected,
                replayed,
            ) = await self._reject_admitted_continuation_in_unit(
                unit,
                rejection=rejection,
                task_run_id=task_run_id,
                request_id=request_id,
                checkpoint_id=checkpoint_id,
                now=observed_at,
            )
            completion = await self._task_store._fail_claimed_reentry_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                task_run_id=task_run_id,
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
                result=failure.result,
                reason="task_resume_rejected",
                observed_at=observed_at,
                metadata=safe_metadata,
                replay_only=replayed,
            )
            return PgsqlDurableTaskRejection(
                rejected_continuation=rejected,
                completion=completion,
            )

        return await self._interaction_store._transaction(
            "durable_task_fail_admitted_reentry",
            execute,
        )

    async def release_running_reentry(
        self,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueReentry:
        """Requeue one safely released running continuation."""
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(claim_token, "claim_token")
        _assert_opaque(segment_id, "segment_id")
        _assert_opaque(task_run_id, "task_run_id")
        _assert_opaque(request_id, "request_id")
        _assert_opaque(continuation_id, "continuation_id")
        _assert_opaque(checkpoint_id, "checkpoint_id")
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await self._task_store._release_running_reentry_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                task_run_id=task_run_id,
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        return await self._interaction_store._transaction(
            "durable_task_release_running_reentry",
            execute,
        )

    async def reconcile_expired_reentry(
        self,
        *,
        queue_item_id: str,
        expected_claim_token: str,
        task_run_id: str,
        result: TaskExecutionResult,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskDurableExpiredReentryCommit:
        """Restore pre-dispatch work or terminalize ambiguous expired work."""
        _assert_opaque(queue_item_id, "queue_item_id")
        _assert_opaque(expected_claim_token, "expected_claim_token")
        _assert_opaque(task_run_id, "task_run_id")
        if not isinstance(result, TaskExecutionResult):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result",
                "value must be a task execution result",
            )
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _SELECT_EXPIRED_TASK_REENTRY_SQL,
                (
                    queue_item_id,
                    task_run_id,
                    expected_claim_token,
                    expected_claim_token,
                    expected_claim_token,
                ),
            )
            provenance = await unit.cursor.fetchone()
            if provenance is None:
                raise TaskStoreConflictError(
                    "expired durable task reentry claim did not match"
                )
            request_id = _row_str(provenance, "previous_request_id")
            continuation_id = _row_str(
                provenance,
                "previous_continuation_id",
            )
            checkpoint_id = _row_str(
                provenance,
                "previous_checkpoint_id",
            )
            active_segment_id = _row_str(
                provenance,
                "active_segment_id",
            )
            run_state = TaskRunState(_row_str(provenance, "durable_run_state"))
            attempt_state = TaskAttemptState(
                _row_str(provenance, "durable_attempt_state")
            )
            active_segment_state = _row_str(
                provenance,
                "active_segment_state",
            )
            continuation_row = await _lock_task_continuation(
                unit,
                continuation_id=ContinuationId(continuation_id),
                task_run_id=task_run_id,
                request_id=request_id,
                checkpoint_id=checkpoint_id,
            )
            lifecycle = _lifecycle(continuation_row)
            continuation = self._interaction_store._continuation_from_row(
                continuation_row
            )
            request_expires_at = _row_datetime(
                continuation_row,
                "request_absolute_expires_at",
            )
            deadline_expired = (
                min(request_expires_at, continuation.expires_at) <= observed_at
            )
            task_lease_expires_at = provenance.get("lease_expires_at")
            task_lease_expired = (
                isinstance(task_lease_expires_at, datetime)
                and task_lease_expires_at <= observed_at
            )
            if (
                run_state is TaskRunState.CANCELLED
                and attempt_state is TaskAttemptState.ABANDONED
                and active_segment_state
                in {
                    TaskAttemptSegmentState.SUSPENDED.value,
                    TaskAttemptSegmentState.ABANDONED.value,
                }
            ):
                _validate_cancelled_continuation_replay(
                    continuation_row,
                    continuation,
                )
                completion = (
                    await self._task_store._cancel_partial_reentry_in_unit(
                        unit,
                        queue_item_id=queue_item_id,
                        claim_token=expected_claim_token,
                        active_segment_id=active_segment_id,
                        task_run_id=task_run_id,
                        request_id=request_id,
                        continuation_id=continuation_id,
                        checkpoint_id=checkpoint_id,
                        result=result,
                        observed_at=observed_at,
                        metadata=safe_metadata,
                    )
                )
                return TaskDurableExpiredReentryCommit(completion=completion)
            if not deadline_expired and not task_lease_expired:
                raise TaskStoreConflictError(
                    "durable task reentry has not expired"
                )
            if run_state is TaskRunState.CANCEL_REQUESTED:
                partial_startup = (
                    attempt_state is TaskAttemptState.SUSPENDED
                    and active_segment_state
                    == TaskAttemptSegmentState.SUSPENDED.value
                ) or (
                    attempt_state is TaskAttemptState.RUNNING
                    and active_segment_state
                    in {
                        TaskAttemptSegmentState.SUSPENDED.value,
                        TaskAttemptSegmentState.CREATED.value,
                    }
                )
                pre_dispatch_claim = lifecycle in {
                    DurableContinuationLifecycle.READY,
                    DurableContinuationLifecycle.CLAIMED,
                }
                if partial_startup or pre_dispatch_claim:
                    continuation = (
                        _validate_cancel_requested_pre_dispatch_claim(
                            self._interaction_store,
                            task_row=provenance,
                            continuation_row=continuation_row,
                            continuation=continuation,
                            task_run_id=task_run_id,
                            request_id=request_id,
                            continuation_id=continuation_id,
                            checkpoint_id=checkpoint_id,
                            expected_claim_token=expected_claim_token,
                        )
                    )
                if partial_startup:
                    invalidated = replace(
                        continuation,
                        claim=_invalidated_claim(continuation),
                        store_revision=_next_continuation_revision(
                            continuation
                        ),
                        updated_at=observed_at,
                    )
                    await self._interaction_store._update_continuation(
                        unit,
                        invalidated,
                        lifecycle=(DurableContinuationLifecycle.INVALIDATED),
                        expected_revision=continuation.store_revision,
                        invalid_reason="task_cancelled",
                    )
                    await unit.cursor.execute(
                        _DEAD_OUTBOX_SQL,
                        (observed_at, continuation.continuation_id),
                    )
                    completion = (
                        await (
                            self._task_store._cancel_partial_reentry_in_unit(
                                unit,
                                queue_item_id=queue_item_id,
                                claim_token=expected_claim_token,
                                active_segment_id=active_segment_id,
                                task_run_id=task_run_id,
                                request_id=request_id,
                                continuation_id=continuation_id,
                                checkpoint_id=checkpoint_id,
                                result=result,
                                observed_at=observed_at,
                                metadata=safe_metadata,
                            )
                        )
                    )
                    return TaskDurableExpiredReentryCommit(
                        completion=completion
                    )
                if (
                    attempt_state is not TaskAttemptState.RUNNING
                    or active_segment_state
                    != TaskAttemptSegmentState.RUNNING.value
                ):
                    raise TaskStoreConflictError(
                        "cancel-requested task reentry provenance did not "
                        "match"
                    )
                if pre_dispatch_claim:
                    invalidated = replace(
                        continuation,
                        claim=_invalidated_claim(continuation),
                        store_revision=_next_continuation_revision(
                            continuation
                        ),
                        updated_at=observed_at,
                    )
                    await self._interaction_store._update_continuation(
                        unit,
                        invalidated,
                        lifecycle=DurableContinuationLifecycle.INVALIDATED,
                        expected_revision=continuation.store_revision,
                        invalid_reason="task_cancelled",
                    )
                await unit.cursor.execute(
                    _DEAD_OUTBOX_SQL,
                    (observed_at, continuation.continuation_id),
                )
                completion = await self._task_store._settle_claim_in_unit(
                    unit,
                    queue_item_id=queue_item_id,
                    claim_token=expected_claim_token,
                    segment_id=active_segment_id,
                    task_run_id=task_run_id,
                    settlement=TaskDurableResumeFailure(result=result),
                    observed_at=observed_at,
                    metadata=safe_metadata,
                    allow_expired_lease=True,
                )
                return TaskDurableExpiredReentryCommit(completion=completion)
            timeout_result = TaskExecutionResult(
                error=freeze_snapshot_value(TaskError.timeout().as_dict()),
                metadata={
                    "interaction_event_type": (
                        TaskInteractionEventType.INPUT_EXPIRED.value
                    )
                },
            )
            if (
                deadline_expired
                and lifecycle is DurableContinuationLifecycle.INVALIDATED
                and _row_optional_str(continuation_row, "invalid_reason")
                == "expired"
                and run_state is TaskRunState.EXPIRED
                and attempt_state is TaskAttemptState.FAILED
            ):
                completion = (
                    await self._task_store._fail_claimed_reentry_in_unit(
                        unit,
                        queue_item_id=queue_item_id,
                        claim_token=expected_claim_token,
                        task_run_id=task_run_id,
                        request_id=request_id,
                        continuation_id=continuation_id,
                        checkpoint_id=checkpoint_id,
                        result=timeout_result,
                        reason="input_expired",
                        observed_at=observed_at,
                        metadata=safe_metadata,
                        replay_only=True,
                        terminal_run_state=TaskRunState.EXPIRED,
                        interaction_event_type=(
                            TaskInteractionEventType.INPUT_EXPIRED
                        ),
                    )
                )
                return TaskDurableExpiredReentryCommit(completion=completion)
            if deadline_expired and lifecycle in {
                DurableContinuationLifecycle.READY,
                DurableContinuationLifecycle.CLAIMED,
            }:
                invalidated = replace(
                    continuation,
                    claim=_invalidated_claim(continuation),
                    store_revision=_next_continuation_revision(continuation),
                    updated_at=observed_at,
                )
                await self._interaction_store._update_continuation(
                    unit,
                    invalidated,
                    lifecycle=DurableContinuationLifecycle.INVALIDATED,
                    expected_revision=continuation.store_revision,
                    invalid_reason="expired",
                )
                await unit.cursor.execute(
                    _DEAD_OUTBOX_SQL,
                    (observed_at, continuation.continuation_id),
                )
                if (
                    run_state is TaskRunState.RUNNING
                    and attempt_state is TaskAttemptState.RUNNING
                    and active_segment_state
                    == TaskAttemptSegmentState.RUNNING.value
                ):
                    completion = await self._task_store._settle_claim_in_unit(
                        unit,
                        queue_item_id=queue_item_id,
                        claim_token=expected_claim_token,
                        segment_id=active_segment_id,
                        task_run_id=task_run_id,
                        settlement=TaskDurableResumeFailure(
                            result=timeout_result
                        ),
                        observed_at=observed_at,
                        metadata=safe_metadata,
                        allow_expired_lease=True,
                        terminal_run_state=TaskRunState.EXPIRED,
                        terminal_reason="input_expired",
                        interaction_event_type=(
                            TaskInteractionEventType.INPUT_EXPIRED
                        ),
                        interaction_request_id=request_id,
                        interaction_continuation_id=continuation_id,
                    )
                else:
                    completion = (
                        await (
                            self._task_store._fail_claimed_reentry_in_unit(
                                unit,
                                queue_item_id=queue_item_id,
                                claim_token=expected_claim_token,
                                task_run_id=task_run_id,
                                request_id=request_id,
                                continuation_id=continuation_id,
                                checkpoint_id=checkpoint_id,
                                result=timeout_result,
                                reason="input_expired",
                                observed_at=observed_at,
                                metadata=safe_metadata,
                                terminal_run_state=TaskRunState.EXPIRED,
                                interaction_event_type=(
                                    TaskInteractionEventType.INPUT_EXPIRED
                                ),
                            )
                        )
                    )
                return TaskDurableExpiredReentryCommit(completion=completion)
            safe_to_release = lifecycle in {
                DurableContinuationLifecycle.READY,
                DurableContinuationLifecycle.CLAIMED,
            }
            if lifecycle is DurableContinuationLifecycle.CLAIMED:
                continuation = self._interaction_store._claimed_continuation(
                    continuation_row,
                    expected_store_revision=ContinuationStoreRevision(
                        _row_int(continuation_row, "store_revision")
                    ),
                    owner_id=ContinuationClaimOwnerId(expected_claim_token),
                    fencing_token=ContinuationFencingToken(
                        _row_int(continuation_row, "fencing_token")
                    ),
                    now=observed_at,
                    allow_expired=True,
                )
                released = replace(
                    continuation,
                    claim=ContinuationClaim(
                        state=(ContinuationClaimState.FAILED_SAFE_TO_RETRY),
                        attempt=continuation.claim.attempt,
                    ),
                    store_revision=_next_continuation_revision(continuation),
                    updated_at=observed_at,
                )
                await self._interaction_store._update_continuation(
                    unit,
                    released,
                    lifecycle=DurableContinuationLifecycle.READY,
                    expected_revision=continuation.store_revision,
                    settled_claim_owner_id=ContinuationClaimOwnerId(
                        expected_claim_token
                    ),
                )
            if safe_to_release and (
                run_state is TaskRunState.CLAIMED
                and attempt_state is TaskAttemptState.SUSPENDED
                and active_segment_state
                == TaskAttemptSegmentState.SUSPENDED.value
            ):
                reentry = (
                    await self._task_store._release_claimed_reentry_in_unit(
                        unit,
                        queue_item_id=queue_item_id,
                        claim_token=expected_claim_token,
                        task_run_id=task_run_id,
                        request_id=request_id,
                        continuation_id=continuation_id,
                        checkpoint_id=checkpoint_id,
                        observed_at=observed_at,
                        metadata=safe_metadata,
                    )
                )
                return TaskDurableExpiredReentryCommit(reentry=reentry)
            if safe_to_release and (
                run_state is TaskRunState.RUNNING
                and attempt_state is TaskAttemptState.RUNNING
                and active_segment_state
                == TaskAttemptSegmentState.RUNNING.value
            ):
                reentry = (
                    await self._task_store._release_running_reentry_in_unit(
                        unit,
                        queue_item_id=queue_item_id,
                        claim_token=expected_claim_token,
                        segment_id=active_segment_id,
                        task_run_id=task_run_id,
                        request_id=request_id,
                        continuation_id=continuation_id,
                        checkpoint_id=checkpoint_id,
                        observed_at=observed_at,
                        metadata=safe_metadata,
                    )
                )
                return TaskDurableExpiredReentryCommit(reentry=reentry)
            failure = TaskDurableResumeFailure(result=result)
            if (
                run_state is TaskRunState.RUNNING
                and attempt_state is TaskAttemptState.RUNNING
                and active_segment_state
                == TaskAttemptSegmentState.RUNNING.value
            ):
                completion = await self._task_store._settle_claim_in_unit(
                    unit,
                    queue_item_id=queue_item_id,
                    claim_token=expected_claim_token,
                    segment_id=active_segment_id,
                    task_run_id=task_run_id,
                    settlement=failure,
                    observed_at=observed_at,
                    metadata=safe_metadata,
                    allow_expired_lease=True,
                )
            else:
                completion = (
                    await self._task_store._fail_claimed_reentry_in_unit(
                        unit,
                        queue_item_id=queue_item_id,
                        claim_token=expected_claim_token,
                        task_run_id=task_run_id,
                        request_id=request_id,
                        continuation_id=continuation_id,
                        checkpoint_id=checkpoint_id,
                        result=result,
                        reason="resume_dispatch_ambiguous",
                        observed_at=observed_at,
                        metadata=safe_metadata,
                    )
                )
            return TaskDurableExpiredReentryCommit(completion=completion)

        return await self._interaction_store._transaction(
            "durable_task_reconcile_expired_reentry",
            execute,
        )

    async def resolve_and_requeue(
        self,
        command: ResolveInteractionCommand,
        *,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskReentry:
        """Resolve input and expose the same task for work atomically."""
        if not isinstance(command, ResolveInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be a resolve interaction command",
            )
        _assert_opaque(task_run_id, "task_run_id")
        if str(command.correlation.run_id) != task_run_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "task run does not match the interaction correlation",
            )
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)
        reentry: TaskQueueReentry | None = None

        async def resolve(
            store: MemoryInteractionStore,
            _unit: PgsqlUnitOfWork,
        ) -> InteractionResolutionResult:
            return await store.resolve(command)

        async def persist(
            unit: PgsqlUnitOfWork,
            result: InteractionResolutionResult,
        ) -> None:
            nonlocal reentry
            record = getattr(result, "record", None)
            if (
                not isinstance(record, InteractionRecord)
                or record.request.state is not RequestState.ANSWERED
                or record.request.resolution is None
            ):
                raise PgsqlInteractionStoreError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "interaction.resolve",
                    "durable task reentry requires an accepted answer",
                )
            await self._interaction_store._ready_continuation(unit, record)
            await self._interaction_store._sync_resolution_keys(unit, record)
            reentry = await self._task_store._requeue_suspended_in_unit(
                unit,
                run_id=task_run_id,
                request_id=str(record.request.request_id),
                continuation_id=str(record.request.continuation_id),
                resolution_revision=record.request.state_revision,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        resolution = await self._interaction_store._run_memory_with_unit(
            "durable_task_resolve_and_requeue",
            resolve,
            mutate=True,
            after_persist=persist,
            selection=_correlation_snapshot(
                command.actor,
                command.correlation,
            ),
        )
        assert reentry is not None
        return PgsqlDurableTaskReentry(
            resolution=resolution,
            reentry=reentry,
        )

    async def cancel_suspended_task(
        self,
        command: TerminalizeInteractionScopeCommand,
        *,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskLifecycle:
        """Cancel one pending interaction and containing task atomically."""
        if not isinstance(command, TerminalizeInteractionScopeCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be a containing-run cancellation command",
            )
        _assert_opaque(task_run_id, "task_run_id")
        if str(command.scope.run_id) != task_run_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "task run does not match the cancellation scope",
            )
        return await self._terminalize_task_lifecycle(
            "durable_task_cancel_suspended",
            lambda store, _unit: store.terminalize_scope(command),
            task_run_id=task_run_id,
            request_state=RequestState.CANCELLED,
            run_state=TaskRunState.CANCELLED,
            attempt_state=TaskAttemptState.ABANDONED,
            event_type=TaskInteractionEventType.INPUT_CANCELLED,
            reason="input_cancelled",
            now=now,
            metadata=metadata,
            selection=_scope_snapshot(command.actor, command.scope),
        )

    async def cancel_input_required_task(
        self,
        *,
        task_run_id: str,
        now: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        """Cancel one task using its trusted persisted interaction owner."""
        _assert_opaque(task_run_id, "task_run_id")
        now = validate_aware_datetime(now, "now")
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def cancel_owned_run(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> ScopeCancellationResult:
            await unit.cursor.execute(
                _SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL,
                (task_run_id,),
            )
            rows = await unit.cursor.fetchall()
            if not rows:
                _task_coordinator_required("task_run_id")
            records = tuple(
                self._interaction_store._record_from_row(row) for row in rows
            )
            if any(
                record.request.origin.run_id != RunId(task_run_id)
                for record in records
            ):
                raise PgsqlInteractionStoreError(
                    InputErrorCode.CORRELATION_MISMATCH,
                    "task_run_id",
                    "task-bound interaction origin does not match its run",
                )
            principals = {
                record.request.origin.principal for record in records
            }
            if len(principals) != 1:
                raise PgsqlInteractionStoreError(
                    InputErrorCode.FORBIDDEN,
                    "task_run_id",
                    "task-bound interactions do not share one trusted owner",
                )
            return await store.terminalize_scope(
                TerminalizeInteractionScopeCommand(
                    actor=InteractionActor(principal=principals.pop()),
                    scope=InteractionExecutionScope(
                        run_id=RunId(task_run_id),
                    ),
                    provenance=AnswerProvenance.HUMAN,
                )
            )

        lifecycle = await self._terminalize_task_lifecycle(
            "durable_task_cancel_input_required",
            cancel_owned_run,
            task_run_id=task_run_id,
            request_state=RequestState.CANCELLED,
            run_state=TaskRunState.CANCELLED,
            attempt_state=TaskAttemptState.ABANDONED,
            event_type=TaskInteractionEventType.INPUT_CANCELLED,
            reason="input_cancelled",
            now=now,
            metadata=safe_metadata,
            selection=_trusted_task_snapshot(task_run_id),
        )
        completion = lifecycle.completion_for(task_run_id)
        if completion is None:
            raise TaskStoreError(
                "task cancellation lost the lifecycle resolution race"
            )
        return completion

    async def supersede_suspended_task(
        self,
        command: SupersedeInteractionScopeCommand,
        *,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskLifecycle:
        """Supersede one pending interaction and containing task atomically."""
        if not isinstance(command, SupersedeInteractionScopeCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be an interaction supersession command",
            )
        _assert_opaque(task_run_id, "task_run_id")
        if str(command.scope.run_id) != task_run_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "task run does not match the supersession scope",
            )
        return await self._terminalize_task_lifecycle(
            "durable_task_supersede_suspended",
            lambda store, _unit: store.supersede_scope(command),
            task_run_id=task_run_id,
            request_state=RequestState.SUPERSEDED,
            run_state=TaskRunState.CANCELLED,
            attempt_state=TaskAttemptState.ABANDONED,
            event_type=TaskInteractionEventType.INPUT_SUPERSEDED,
            reason="input_superseded",
            now=now,
            metadata=metadata,
            selection=_scope_snapshot(command.actor, command.scope),
        )

    async def expire_suspended_task(
        self,
        command: TerminalizeDueInteractionsCommand,
        *,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PgsqlDurableTaskLifecycle:
        """Expire one due interaction and containing task atomically."""
        if not isinstance(command, TerminalizeDueInteractionsCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "command",
                "value must be a due-interaction command",
            )
        _assert_opaque(task_run_id, "task_run_id")
        return await self._terminalize_task_lifecycle(
            "durable_task_expire_suspended",
            lambda store, _unit: store.terminalize_due(command),
            task_run_id=task_run_id,
            request_state=RequestState.EXPIRED,
            run_state=TaskRunState.EXPIRED,
            attempt_state=TaskAttemptState.FAILED,
            event_type=TaskInteractionEventType.INPUT_EXPIRED,
            reason="input_expired",
            now=now,
            metadata=metadata,
            selection=_DEADLINE_SNAPSHOT_SELECTION,
        )

    async def sweep_retention(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> ContinuationSweepResult:
        """Delete only terminal task-bound rows after retention expires."""
        return await self._interaction_store._sweep(
            now=now,
            limit=limit,
            allow_task_bound_retention=True,
        )

    async def _terminalize_task_lifecycle(
        self,
        operation_name: str,
        callback: Callable[
            [MemoryInteractionStore, PgsqlUnitOfWork],
            Awaitable[
                ScopeCancellationResult
                | ScopeSupersessionResult
                | DueInteractionsResult
            ],
        ],
        *,
        task_run_id: str,
        request_state: RequestState,
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        event_type: TaskInteractionEventType,
        reason: str,
        now: datetime | None,
        metadata: Mapping[str, object] | None,
        selection: _InteractionSnapshotSelection,
    ) -> PgsqlDurableTaskLifecycle:
        """Converge one interaction lifecycle with its suspended task."""
        observed_at = await self._observed_at(now)
        safe_metadata = freeze_snapshot_metadata(metadata)
        completions: list[TaskQueueCompletion] = []

        async def operation(
            store: MemoryInteractionStore,
            unit: PgsqlUnitOfWork,
        ) -> (
            ScopeCancellationResult
            | ScopeSupersessionResult
            | DueInteractionsResult
        ):
            return await callback(store, unit)

        async def persist(
            unit: PgsqlUnitOfWork,
            result: (
                ScopeCancellationResult
                | ScopeSupersessionResult
                | DueInteractionsResult
            ),
        ) -> None:
            records = getattr(result, "records", None)
            if not isinstance(records, tuple):
                return
            task_correlations: dict[str, list[tuple[str, str]]] = {}
            for record in records:
                if not isinstance(record, InteractionRecord):
                    raise PgsqlInteractionStoreError(
                        InputErrorCode.SNAPSHOT_INVALID,
                        "interaction.lifecycle",
                        "lifecycle result contains an invalid record",
                    )
                if record.request.resolution is None:
                    raise PgsqlInteractionStoreError(
                        InputErrorCode.ILLEGAL_TRANSITION,
                        "interaction.lifecycle",
                        "lifecycle result is not terminal",
                    )
                await unit.cursor.execute(
                    _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
                    (record.request.request_id,),
                )
                continuation_row = await unit.cursor.fetchone()
                bound_task_run_id = (
                    _row_optional_str(
                        continuation_row,
                        "task_run_id",
                    )
                    if continuation_row is not None
                    else None
                )
                if bound_task_run_id is not None and bound_task_run_id != str(
                    record.request.origin.run_id
                ):
                    raise PgsqlInteractionStoreError(
                        InputErrorCode.CORRELATION_MISMATCH,
                        "interaction.lifecycle",
                        "task continuation origin does not match its run",
                    )
                if (
                    bound_task_run_id is not None
                    and record.request.state is not request_state
                ):
                    raise PgsqlInteractionStoreError(
                        InputErrorCode.ILLEGAL_TRANSITION,
                        "interaction.lifecycle",
                        "task interaction settled with an unexpected state",
                    )
                await self._interaction_store._invalidate_for_record(
                    unit,
                    record,
                )
                await self._interaction_store._sync_resolution_keys(
                    unit,
                    record,
                )
                if bound_task_run_id is not None:
                    task_correlations.setdefault(
                        bound_task_run_id,
                        [],
                    ).append(
                        (
                            str(record.request.request_id),
                            str(record.request.continuation_id),
                        )
                    )
            for affected_run_id, correlations in task_correlations.items():
                completion = (
                    await self._task_store._terminalize_suspended_in_unit(
                        unit,
                        task_run_id=affected_run_id,
                        correlations=tuple(correlations),
                        run_state=run_state,
                        attempt_state=attempt_state,
                        event_type=event_type,
                        reason=reason,
                        observed_at=observed_at,
                        metadata=safe_metadata,
                    )
                )
                assert completion is not None
                completions.append(completion)
            if task_run_id not in task_correlations:
                replay = await self._task_store._terminalize_suspended_in_unit(
                    unit,
                    task_run_id=task_run_id,
                    correlations=(),
                    run_state=run_state,
                    attempt_state=attempt_state,
                    event_type=event_type,
                    reason=reason,
                    observed_at=observed_at,
                    metadata=safe_metadata,
                    replay_only=True,
                )
                if replay is not None:
                    completions.append(replay)

        result = await self._interaction_store._run_memory_with_unit(
            operation_name,
            operation,
            mutate=True,
            after_persist=persist,
            selection=selection,
        )
        return PgsqlDurableTaskLifecycle(
            interaction=result,
            completions=tuple(completions),
        )

    async def _complete_settlement_continuation_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        completion: ContinuationCompletionCommand,
        task_run_id: str,
        segment_id: str,
        now: datetime,
    ) -> tuple[PortableContinuation, bool]:
        row = await _lock_resumed_task_continuation(
            unit,
            continuation_id=completion.continuation_id,
            task_run_id=task_run_id,
            segment_id=segment_id,
        )
        continuation = self._interaction_store._continuation_from_row(row)
        if _lifecycle(row) is DurableContinuationLifecycle.COMPLETED:
            recorded = continuation.completion
            if (
                int(continuation.store_revision)
                != int(completion.expected_store_revision) + 1
                or continuation.fencing_token != completion.fencing_token
                or _row_optional_str(row, "claim_owner_id")
                != completion.owner_id
                or recorded is None
                or recorded.result_digest != completion.result_digest
            ):
                raise ContinuationStoreConflictError(
                    "continuation.completion",
                    "completed continuation does not match the replay",
                )
            return continuation, True
        completed = (
            await self._interaction_store._complete_continuation_in_unit(
                unit,
                continuation_id=completion.continuation_id,
                expected_store_revision=(completion.expected_store_revision),
                owner_id=completion.owner_id,
                fencing_token=completion.fencing_token,
                result_digest=completion.result_digest,
                now=now,
                expected_task_run_id=task_run_id,
                expected_task_segment_id=segment_id,
            )
        )
        return completed, False

    async def _completed_task_continuation_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        completion: ContinuationCompletionCommand,
        task_run_id: str,
        request_id: str,
        checkpoint_id: str,
    ) -> PortableContinuation:
        """Validate exact immutable provider completion evidence."""
        row = await _lock_task_continuation(
            unit,
            continuation_id=completion.continuation_id,
            task_run_id=task_run_id,
            request_id=request_id,
            checkpoint_id=checkpoint_id,
        )
        continuation = self._interaction_store._continuation_from_row(row)
        recorded = continuation.completion
        if (
            str(continuation.origin.run_id) != task_run_id
            or str(continuation.request_id) != request_id
            or _lifecycle(row) is not DurableContinuationLifecycle.COMPLETED
            or continuation.claim.state is not ContinuationClaimState.COMPLETED
            or int(continuation.store_revision)
            != int(completion.expected_store_revision) + 1
            or continuation.fencing_token != completion.fencing_token
            or _row_optional_str(row, "claim_owner_id") != completion.owner_id
            or recorded is None
            or recorded.result_digest != completion.result_digest
        ):
            raise ContinuationStoreConflictError(
                "continuation.completion",
                "provider completion evidence does not match",
            )
        return continuation

    async def _reject_admitted_continuation_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        rejection: ContinuationRejectionCommand,
        task_run_id: str,
        request_id: str,
        checkpoint_id: str,
        now: datetime,
    ) -> tuple[PortableContinuation, bool]:
        row = await _lock_task_continuation(
            unit,
            continuation_id=rejection.continuation_id,
            task_run_id=task_run_id,
            request_id=request_id,
            checkpoint_id=checkpoint_id,
        )
        continuation = self._interaction_store._continuation_from_row(row)
        if (
            str(continuation.origin.run_id) != task_run_id
            or str(continuation.request_id) != request_id
        ):
            raise PgsqlInteractionStoreError(
                InputErrorCode.SNAPSHOT_INVALID,
                "continuation",
                "task continuation payload does not match its binding",
            )
        if _lifecycle(row) is DurableContinuationLifecycle.INVALIDATED:
            if (
                int(continuation.store_revision)
                != int(rejection.expected_store_revision) + 1
                or continuation.fencing_token != rejection.fencing_token
                or _row_optional_str(row, "invalid_reason")
                != "task_resume_rejected"
                or continuation.claim.state
                is not ContinuationClaimState.FAILED_SAFE_TO_RETRY
            ):
                raise ContinuationStoreConflictError(
                    "continuation.rejection",
                    "invalidated continuation does not match the replay",
                )
            return continuation, True
        claimed = self._interaction_store._claimed_continuation(
            row,
            expected_store_revision=rejection.expected_store_revision,
            owner_id=rejection.owner_id,
            fencing_token=rejection.fencing_token,
            now=now,
            allow_expired=True,
        )
        if (
            claimed.claim.state
            is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        ):
            raise ContinuationStoreConflictError(
                "continuation.rejection",
                "continuation is not safely rejectable before dispatch",
            )
        rejected = replace(
            claimed,
            claim=_invalidated_claim(claimed),
            store_revision=_next_continuation_revision(claimed),
            updated_at=now,
        )
        await self._interaction_store._update_continuation(
            unit,
            rejected,
            lifecycle=DurableContinuationLifecycle.INVALIDATED,
            expected_revision=rejection.expected_store_revision,
            invalid_reason="task_resume_rejected",
        )
        await unit.cursor.execute(
            _DEAD_OUTBOX_SQL,
            (now, rejection.continuation_id),
        )
        return rejected, False

    async def _observed_at(self, now: datetime | None) -> datetime:
        if now is not None:
            return validate_aware_datetime(now, "now")
        observation = await self._interaction_store._clock.read()
        return validate_aware_datetime(
            observation.wall_time, "clock.wall_time"
        )

    @staticmethod
    def _validate_continuation(
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
    ) -> None:
        _validate_request_continuation_expiry(command, continuation)
        request = command.request
        if (
            request.request_id != continuation.request_id
            or request.continuation_id != continuation.continuation_id
            or request.origin != continuation.origin
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation",
                "continuation does not match the staged task interaction",
            )

    @staticmethod
    def _validate_successor(
        completion: ContinuationCompletionCommand,
        *,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        task_run_id: str,
    ) -> None:
        request = command.request
        if (
            str(request.origin.run_id) != task_run_id
            or str(continuation.origin.run_id) != task_run_id
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "task_run_id",
                "successor interaction does not belong to the resumed task",
            )
        if (
            completion.continuation_id == continuation.continuation_id
            or continuation.interaction_count == 0
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation",
                "successor continuation must be a fresh interaction",
            )

    @staticmethod
    def _validate_completed_successor(
        previous: PortableContinuation,
        successor: PortableContinuation,
    ) -> None:
        pristine = (
            successor.claim.state is ContinuationClaimState.UNCLAIMED
            and successor.claim.owner_id is None
            and successor.claim.lease_expires_at is None
            and successor.claim.attempt == 0
            and int(successor.store_revision) == 0
            and int(successor.fencing_token) == 0
            and successor.dispatch is None
            and successor.completion is None
        )
        if (
            successor.origin != previous.origin
            or successor.definition != previous.definition
            or successor.revision_binding != previous.revision_binding
            or successor.request_id == previous.request_id
            or successor.continuation_id == previous.continuation_id
            or successor.provider_call_correlation_id
            == previous.provider_call_correlation_id
            or successor.operation_cursor < previous.operation_cursor
            or successor.interaction_count != previous.interaction_count + 1
            or successor.tool_loop_count < previous.tool_loop_count
            or successor.stream_sequence < previous.stream_sequence
            or not pristine
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "continuation.successor",
                "successor continuation does not extend the dispatched state",
            )


@final
class PgsqlResumptionReconciler:
    """Deliver durable outbox records with idempotent lease recovery."""

    def __init__(
        self,
        store: PgsqlInteractionStore,
        *,
        owner_id: ContinuationClaimOwnerId,
        dispatcher: ResumptionOutboxDispatcher,
        clock: Callable[[], datetime],
    ) -> None:
        assert isinstance(store, PgsqlInteractionStore)
        _assert_opaque(owner_id, "owner_id")
        assert callable(dispatcher)
        assert callable(clock)
        self._store = store
        self._owner_id = owner_id
        self._dispatcher = dispatcher
        self._clock = clock

    async def run_once(self, *, limit: int = 10) -> int:
        """Deliver one bounded batch and converge every claimed item."""
        _assert_limit(limit)
        records = await self._store.claim_outbox(
            owner_id=self._owner_id,
            now=self._now(),
            limit=limit,
        )
        for record in records:
            try:
                await self._dispatcher(record)
            except CancelledError:
                await self._store.release_outbox(
                    record,
                    owner_id=self._owner_id,
                    error_code="cancelled",
                    now=self._now(),
                )
                raise
            except Exception:
                await self._store.release_outbox(
                    record,
                    owner_id=self._owner_id,
                    error_code="delivery_failed",
                    now=self._now(),
                )
            else:
                await self._store.complete_outbox(
                    record,
                    owner_id=self._owner_id,
                    now=self._now(),
                )
        return len(records)

    def _now(self) -> datetime:
        return validate_aware_datetime(self._clock(), "reconciler.clock")


def require_interaction_pgsql_dependencies(
    *,
    module_finder: Callable[[str], object | None] = find_spec,
) -> None:
    """Fail closed with exact operator commands when PostgreSQL is absent."""
    assert callable(module_finder)
    missing = tuple(
        module
        for module in ("psycopg", "psycopg_pool")
        if module_finder(module) is None
    )
    if missing:
        raise PgsqlInteractionFeatureUnavailableError(
            reason=(
                "durable interaction persistence requires optional modules "
                + ", ".join(missing)
            )
        )


async def _check_schema(database: PgsqlDatabase) -> None:
    async def execute(unit: PgsqlUnitOfWork) -> object:
        await unit.cursor.execute(_CHECK_SCHEMA_SQL)
        row = await unit.cursor.fetchone()
        if row is None or _row_str(row, "version_num") != (
            INTERACTION_PGSQL_HEAD_REVISION
        ):
            raise PgsqlInteractionFeatureUnavailableError(
                reason="durable interaction schema is not current"
            )
        return None

    try:
        async with database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await execute(
                        PgsqlUnitOfWork(
                            connection=connection,
                            cursor=cursor,
                        )
                    )
    except PgsqlInteractionFeatureUnavailableError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit, CancelledError)):
            raise
        raise PgsqlInteractionFeatureUnavailableError(
            reason="durable interaction schema check failed"
        ) from error


async def _lock_store_metadata(unit: PgsqlUnitOfWork) -> PgsqlRow:
    await unit.cursor.execute(_LOCK_STORE_METADATA_SQL)
    row = await unit.cursor.fetchone()
    if row is None:
        raise PgsqlInteractionFeatureUnavailableError(
            reason="durable interaction schema metadata is missing"
        )
    return row


async def _lock_continuation(
    unit: PgsqlUnitOfWork,
    continuation_id: ContinuationId,
) -> PgsqlRow:
    _assert_opaque(continuation_id, "continuation_id")
    await unit.cursor.execute(
        _SELECT_CONTINUATION_FOR_UPDATE_SQL,
        (continuation_id,),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise InteractionNotFoundError()
    return row


async def _lock_task_run_continuation(
    unit: PgsqlUnitOfWork,
    *,
    continuation_id: ContinuationId,
    task_run_id: str,
) -> PgsqlRow:
    """Lock a continuation only after its task ownership matches."""
    _assert_opaque(continuation_id, "continuation_id")
    _assert_opaque(task_run_id, "task_run_id")
    await unit.cursor.execute(
        _SELECT_TASK_RUN_CONTINUATION_FOR_UPDATE_SQL,
        (continuation_id, task_run_id),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise InteractionNotFoundError()
    return row


async def _lock_task_continuation(
    unit: PgsqlUnitOfWork,
    *,
    continuation_id: ContinuationId,
    task_run_id: str,
    request_id: str,
    checkpoint_id: str,
) -> PgsqlRow:
    """Lock a continuation only after its full task tuple matches."""
    _assert_opaque(continuation_id, "continuation_id")
    _assert_opaque(task_run_id, "task_run_id")
    _assert_opaque(request_id, "request_id")
    _assert_opaque(checkpoint_id, "checkpoint_id")
    await unit.cursor.execute(
        _SELECT_TASK_CONTINUATION_FOR_UPDATE_SQL,
        (
            continuation_id,
            task_run_id,
            request_id,
            checkpoint_id,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise InteractionNotFoundError()
    return row


async def _lock_resumed_task_continuation(
    unit: PgsqlUnitOfWork,
    *,
    continuation_id: ContinuationId,
    task_run_id: str,
    segment_id: str,
) -> PgsqlRow:
    """Lock a continuation through exact persisted resume provenance."""
    _assert_opaque(continuation_id, "continuation_id")
    _assert_opaque(task_run_id, "task_run_id")
    _assert_opaque(segment_id, "segment_id")
    await unit.cursor.execute(
        _SELECT_RESUMED_TASK_CONTINUATION_FOR_UPDATE_SQL,
        (continuation_id, task_run_id, segment_id),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise InteractionNotFoundError()
    return row


def _encode_record(record: InteractionRecord) -> bytes:
    payload: dict[str, object] = {
        "request": encode_input_request(record.request),
        "semantic_fingerprint": record.semantic_fingerprint,
        "absolute_expires_at": record.absolute_expires_at.isoformat(),
        "presentation": record.presentation.value,
        "store_revision": int(record.store_revision),
        "advisory_wait": _encode_advisory(record.advisory_wait),
        "resolution_digest": record.resolution_digest,
        "idempotency_ledger": [
            {
                "key": str(entry.key),
                "resolution_digest": entry.resolution_digest,
            }
            for entry in record.idempotency_ledger
        ],
        "resolved_by": _encode_resolver(record.resolved_by),
    }
    return _canonical_bytes(payload)


def _decode_record(value: bytes) -> InteractionRecord:
    payload = _object_payload(value, "record")
    ledger_value = payload.get("idempotency_ledger")
    if not isinstance(ledger_value, list):
        _invalid_payload("record.idempotency_ledger")
    ledger: list[ResolutionIdempotencyEntry] = []
    for raw_entry in ledger_value:
        if not isinstance(raw_entry, Mapping):
            _invalid_payload("record.idempotency_ledger")
        ledger.append(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey(_mapping_str(raw_entry, "key")),
                resolution_digest=_mapping_str(
                    raw_entry,
                    "resolution_digest",
                ),
            )
        )
    request = decode_input_request(payload.get("request"))
    return InteractionRecord(
        request=request,
        semantic_fingerprint=_mapping_str(
            payload,
            "semantic_fingerprint",
        ),
        absolute_expires_at=_payload_datetime(
            payload,
            "absolute_expires_at",
        ),
        presentation=InteractionPresentationState(
            _mapping_str(payload, "presentation")
        ),
        store_revision=InteractionStoreRevision(
            _mapping_int(payload, "store_revision")
        ),
        advisory_wait=_decode_advisory(payload.get("advisory_wait")),
        resolution_digest=_mapping_optional_str(
            payload,
            "resolution_digest",
        ),
        idempotency_ledger=tuple(ledger),
        resolved_by=_decode_resolver(payload.get("resolved_by")),
    )


def _encode_branch(record: InteractionBranchRecord) -> bytes:
    registration = record.registration
    return _canonical_bytes(
        {
            "run_id": str(registration.run_id),
            "branch_id": str(registration.branch_id),
            "parent_branch_id": str(registration.parent_branch_id),
            "principal": _principal_payload(registration.principal),
            "store_revision": int(record.store_revision),
        }
    )


def _decode_branch(value: bytes) -> InteractionBranchRecord:
    payload = _object_payload(value, "branch")
    return InteractionBranchRecord(
        registration=InteractionBranchRegistration(
            run_id=RunId(_mapping_str(payload, "run_id")),
            branch_id=BranchId(_mapping_str(payload, "branch_id")),
            parent_branch_id=BranchId(
                _mapping_str(payload, "parent_branch_id")
            ),
            principal=_principal_from_payload(payload.get("principal")),
        ),
        store_revision=InteractionStoreRevision(
            _mapping_int(payload, "store_revision")
        ),
    )


def _encode_advisory(value: AdvisoryWaitState | None) -> object:
    if value is None:
        return None
    return {
        "status": value.status.value,
        "budget_seconds": value.budget_seconds,
        "remaining_seconds": value.remaining_seconds,
        "presented_at": (
            value.presented_at.isoformat()
            if value.presented_at is not None
            else None
        ),
        "running_since_monotonic": value.running_since_monotonic,
        "controller_id": (
            str(value.controller_id)
            if value.controller_id is not None
            else None
        ),
        "lease_nonce": (
            str(value.lease_nonce) if value.lease_nonce is not None else None
        ),
        "activity_sequence": value.activity_sequence,
        "lease_expires_at_monotonic": value.lease_expires_at_monotonic,
    }


def _decode_advisory(value: object) -> AdvisoryWaitState | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _invalid_payload("record.advisory_wait")
    payload = cast(Mapping[str, object], value)
    return AdvisoryWaitState(
        status=AdvisoryWaitStatus(_mapping_str(payload, "status")),
        budget_seconds=_mapping_int(payload, "budget_seconds"),
        remaining_seconds=_mapping_number(payload, "remaining_seconds"),
        presented_at=_mapping_optional_datetime(payload, "presented_at"),
        running_since_monotonic=_mapping_optional_number(
            payload,
            "running_since_monotonic",
        ),
        controller_id=cast(
            ControllerId | None,
            _mapping_optional_str(payload, "controller_id"),
        ),
        lease_nonce=cast(
            ActiveControlLeaseNonce | None,
            _mapping_optional_str(payload, "lease_nonce"),
        ),
        activity_sequence=_mapping_optional_int(
            payload,
            "activity_sequence",
        ),
        lease_expires_at_monotonic=_mapping_optional_number(
            payload,
            "lease_expires_at_monotonic",
        ),
    )


def _encode_resolver(value: object | None) -> object:
    if value is None:
        return None
    if isinstance(value, PrincipalScope):
        return {"kind": "principal", "value": _principal_payload(value)}
    for resolver, name in (
        (_DEADLINE_RESOLVER, "deadline"),
        (_TRUSTED_DEFAULT_RESOLVER, "trusted_default"),
        (_ADMISSION_CLEANUP_RESOLVER, "admission_cleanup"),
    ):
        if value is resolver:
            return {"kind": name}
    _invalid_payload("record.resolved_by")


def _decode_resolver(
    value: object,
) -> InteractionResolutionAuthority | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _invalid_payload("record.resolved_by")
    payload = cast(Mapping[str, object], value)
    kind = _mapping_str(payload, "kind")
    if kind == "principal":
        return _principal_from_payload(payload.get("value"))
    resolvers = {
        "deadline": _DEADLINE_RESOLVER,
        "trusted_default": _TRUSTED_DEFAULT_RESOLVER,
        "admission_cleanup": _ADMISSION_CLEANUP_RESOLVER,
    }
    resolver = resolvers.get(kind)
    if resolver is None:
        _invalid_payload("record.resolved_by.kind")
    return resolver


def _principal_payload(value: PrincipalScope) -> dict[str, str | None]:
    return {
        "user_id": value.user_id,
        "tenant_id": value.tenant_id,
        "participant_id": value.participant_id,
        "session_id": value.session_id,
    }


def _principal_from_payload(value: object) -> PrincipalScope:
    if not isinstance(value, Mapping):
        _invalid_payload("principal")
    payload = cast(Mapping[str, object], value)
    return PrincipalScope(
        user_id=cast(Any, _mapping_optional_str(payload, "user_id")),
        tenant_id=cast(Any, _mapping_optional_str(payload, "tenant_id")),
        participant_id=cast(
            Any,
            _mapping_optional_str(payload, "participant_id"),
        ),
        session_id=cast(Any, _mapping_optional_str(payload, "session_id")),
    )


def _branch_roots(
    records: tuple[InteractionBranchRecord, ...],
) -> dict[tuple[str, str, str], str]:
    parents = {
        (
            str(record.registration.run_id),
            _scope_identity_digest(
                record.registration.run_id,
                record.registration.principal,
            ),
            str(record.registration.branch_id),
        ): str(record.registration.parent_branch_id)
        for record in records
    }
    roots: dict[tuple[str, str, str], str] = {}
    for key in parents:
        run_id, scope_identity_digest, branch_id = key
        seen: set[str] = set()
        current = branch_id
        while (run_id, scope_identity_digest, current) in parents:
            if current in seen:
                _invalid_payload("branch.cycle")
            seen.add(current)
            current = parents[(run_id, scope_identity_digest, current)]
        roots[key] = current
    return roots


def _branches_with_valid_row_roots(
    decoded: tuple[tuple[PgsqlRow, InteractionBranchRecord], ...],
    *,
    tolerate_invalid: bool,
) -> tuple[InteractionBranchRecord, ...]:
    """Validate searchable roots and isolate invalid maintenance rows."""
    remaining = decoded
    while remaining:
        branches = tuple(branch for _, branch in remaining)
        roots = _branch_roots(branches)
        invalid: set[tuple[str, str, str]] = set()
        for row, branch in remaining:
            try:
                _validate_branch_row_root(row, branch, roots)
            except InputContractError:
                if not tolerate_invalid:
                    raise
                registration = branch.registration
                invalid.add(
                    (
                        str(registration.run_id),
                        _scope_identity_digest(
                            registration.run_id,
                            registration.principal,
                        ),
                        str(registration.branch_id),
                    )
                )
        if not invalid:
            return branches
        remaining = tuple(
            (row, branch)
            for row, branch in remaining
            if (
                str(branch.registration.run_id),
                _scope_identity_digest(
                    branch.registration.run_id,
                    branch.registration.principal,
                ),
                str(branch.registration.branch_id),
            )
            not in invalid
        )
    return ()


def _record_updated_at(record: InteractionRecord) -> datetime:
    resolution = record.request.resolution
    if resolution is not None:
        return resolution.resolved_at
    if record.advisory_wait is not None:
        presented_at = record.advisory_wait.presented_at
        if presented_at is not None:
            return presented_at
    return record.request.created_at


def _continuation_binding(
    payload: Mapping[object, object],
) -> ContinuationRevisionBinding:
    raw = payload.get("revision_binding")
    if not isinstance(raw, Mapping):
        _invalid_payload("continuation.revision_binding")
    binding = cast(Mapping[str, object], raw)
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName(
            _mapping_str(binding, "provider_family")
        ),
        model_id=ModelId(_mapping_str(binding, "model_id")),
        provider_config_revision=ProviderConfigRevision(
            _mapping_str(
                binding,
                "provider_config_revision",
            )
        ),
        model_config_revision=ModelConfigRevision(
            _mapping_str(
                binding,
                "model_config_revision",
            )
        ),
        capability_revision=CapabilityRevision(
            _mapping_str(
                binding,
                "capability_revision",
            )
        ),
    )


def _outbox_record(row: PgsqlRow) -> ResumptionOutboxRecord:
    return ResumptionOutboxRecord(
        outbox_id=_row_str(row, "outbox_id"),
        continuation_id=ContinuationId(_row_str(row, "continuation_id")),
        request_id=InputRequestId(_row_str(row, "request_id")),
        task_run_id=_row_optional_str(row, "task_run_id"),
        resolution_revision=StateRevision(
            _row_int(row, "resolution_revision")
        ),
        status=ResumptionOutboxStatus(_row_str(row, "status")),
        fencing_token=ContinuationFencingToken(_row_int(row, "fencing_token")),
        attempts=_row_int(row, "attempts"),
    )


def _lifecycle(row: Mapping[str, object]) -> DurableContinuationLifecycle:
    return DurableContinuationLifecycle(_row_str(row, "lifecycle_state"))


def _next_continuation_revision(
    continuation: PortableContinuation,
) -> ContinuationStoreRevision:
    if continuation.store_revision >= MAX_STATE_REVISION:
        raise PgsqlInteractionStoreError(
            InputErrorCode.STATE_REVISION_EXHAUSTED,
            "continuation.store_revision",
            "continuation store revision is exhausted",
        )
    return ContinuationStoreRevision(continuation.store_revision + 1)


def _invalidated_claim(
    continuation: PortableContinuation,
) -> ContinuationClaim:
    state = (
        ContinuationClaimState.UNCLAIMED
        if continuation.dispatch is None
        else ContinuationClaimState.FAILED_SAFE_TO_RETRY
    )
    return ContinuationClaim(
        state=state,
        attempt=continuation.claim.attempt,
    )


def _validate_cancel_requested_pre_dispatch_claim(
    store: PgsqlInteractionStore,
    *,
    task_row: PgsqlRow,
    continuation_row: PgsqlRow,
    continuation: PortableContinuation,
    task_run_id: str,
    request_id: str,
    continuation_id: str,
    checkpoint_id: str,
    expected_claim_token: str,
) -> PortableContinuation:
    """Return one exact task-bound pre-dispatch continuation claim."""
    previous_segment_id = _row_str(task_row, "previous_segment_id")
    active_segment_id = _row_str(task_row, "active_segment_id")
    attempt_state = TaskAttemptState(
        _row_str(task_row, "durable_attempt_state")
    )
    active_segment_state = TaskAttemptSegmentState(
        _row_str(task_row, "active_segment_state")
    )
    common_provenance_matches = (
        _row_str(task_row, "state") == TaskQueueItemState.CLAIMED.value
        and _row_str(task_row, "run_id") == task_run_id
        and _row_str(task_row, "claim_token") == expected_claim_token
        and _row_optional_str(task_row, "request_id") == request_id
        and _row_optional_str(task_row, "continuation_id") == continuation_id
        and _row_str(task_row, "durable_run_state")
        == TaskRunState.CANCEL_REQUESTED.value
        and _row_optional_str(task_row, "durable_run_claim_token")
        == expected_claim_token
        and _row_str(task_row, "durable_attempt_id")
        == _row_str(task_row, "attempt_id")
        and _row_optional_str(task_row, "durable_attempt_claim_token")
        == expected_claim_token
        and _row_str(task_row, "segment_id") == previous_segment_id
        and _row_str(task_row, "previous_attempt_id")
        == _row_str(task_row, "durable_attempt_id")
        and _row_str(task_row, "previous_run_id") == task_run_id
        and TaskAttemptSegmentState(
            _row_str(task_row, "previous_segment_state")
        )
        is TaskAttemptSegmentState.SUSPENDED
        and _row_optional_str(task_row, "previous_segment_claim_token") is None
        and _row_optional_str(task_row, "previous_request_id") == request_id
        and _row_optional_str(task_row, "previous_continuation_id")
        == continuation_id
        and _row_optional_str(task_row, "previous_checkpoint_id")
        == checkpoint_id
        and _row_str(task_row, "active_attempt_id")
        == _row_str(task_row, "durable_attempt_id")
        and _row_str(task_row, "active_run_id") == task_run_id
        and _row_optional_str(continuation_row, "task_run_id") == task_run_id
        and _row_str(continuation_row, "request_id") == request_id
        and _row_str(continuation_row, "continuation_id") == continuation_id
        and _row_optional_str(continuation_row, "checkpoint_id")
        == checkpoint_id
        and str(continuation.origin.run_id) == task_run_id
        and str(continuation.request_id) == request_id
        and str(continuation.continuation_id) == continuation_id
    )
    if active_segment_id == previous_segment_id:
        startup_provenance_matches = (
            (
                attempt_state,
                active_segment_state,
            )
            in {
                (
                    TaskAttemptState.SUSPENDED,
                    TaskAttemptSegmentState.SUSPENDED,
                ),
                (
                    TaskAttemptState.RUNNING,
                    TaskAttemptSegmentState.SUSPENDED,
                ),
            }
            and _row_optional_str(task_row, "active_segment_claim_token")
            is None
            and _row_optional_str(
                task_row,
                "active_resumed_from_segment_id",
            )
            is None
            and _row_optional_str(task_row, "active_request_id") == request_id
            and _row_optional_str(task_row, "active_continuation_id")
            == continuation_id
            and _row_optional_str(task_row, "active_checkpoint_id")
            == checkpoint_id
        )
    else:
        startup_provenance_matches = (
            attempt_state is TaskAttemptState.RUNNING
            and active_segment_state
            in {
                TaskAttemptSegmentState.CREATED,
                TaskAttemptSegmentState.RUNNING,
            }
            and _row_optional_str(task_row, "active_segment_claim_token")
            == expected_claim_token
            and _row_optional_str(
                task_row,
                "active_resumed_from_segment_id",
            )
            == previous_segment_id
            and _row_optional_str(task_row, "active_request_id") is None
            and _row_optional_str(task_row, "active_continuation_id") is None
            and _row_optional_str(task_row, "active_checkpoint_id") is None
            and _row_int(task_row, "active_segment_number")
            == _row_int(task_row, "previous_segment_number") + 1
        )
    if not common_provenance_matches or not startup_provenance_matches:
        raise TaskStoreConflictError(
            "cancel-requested task reentry provenance did not match"
        )
    claimed = store._claimed_continuation(
        continuation_row,
        expected_store_revision=continuation.store_revision,
        owner_id=ContinuationClaimOwnerId(expected_claim_token),
        fencing_token=continuation.fencing_token,
        now=continuation.updated_at,
        allow_expired=True,
    )
    if (
        claimed.claim.state is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        or claimed.dispatch is None
    ):
        raise ContinuationStoreConflictError(
            "continuation.claim",
            "continuation does not have an exact pre-dispatch claim",
        )
    return claimed


def _validate_cancelled_continuation_replay(
    row: Mapping[str, object],
    continuation: PortableContinuation,
) -> None:
    lifecycle = _lifecycle(row)
    if continuation.store_revision != ContinuationStoreRevision(
        _row_int(row, "store_revision")
    ) or continuation.fencing_token != ContinuationFencingToken(
        _row_int(row, "fencing_token")
    ):
        raise TaskStoreConflictError(
            "cancelled durable continuation does not match the replay"
        )
    if lifecycle is DurableContinuationLifecycle.INVALIDATED:
        if (
            _row_optional_str(row, "invalid_reason") != "task_cancelled"
            or continuation.claim.state
            is not ContinuationClaimState.FAILED_SAFE_TO_RETRY
            or continuation.claim.owner_id is not None
            or continuation.claim.lease_expires_at is not None
            or _row_optional_str(row, "claim_owner_id") is not None
        ):
            raise TaskStoreConflictError(
                "cancelled durable continuation does not match the replay"
            )
        return
    if lifecycle not in {
        DurableContinuationLifecycle.DISPATCHING,
        DurableContinuationLifecycle.COMPLETED,
    }:
        raise TaskStoreConflictError(
            "cancelled durable continuation does not match the replay"
        )


def _object_payload(value: bytes, path: str) -> Mapping[str, object]:
    try:
        payload = loads(value)
    except (UnicodeError, ValueError) as error:
        raise PgsqlInteractionStoreError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "encrypted payload is invalid",
        ) from error
    if not isinstance(payload, Mapping):
        _invalid_payload(path)
    return cast(Mapping[str, object], payload)


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json(value: object) -> str:
    return dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _row_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        _invalid_payload(key)
    return value


def _row_optional_str(
    row: Mapping[str, object],
    key: str,
) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _invalid_payload(key)
    return value


def _row_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _invalid_payload(key)
    return value


def _row_bool(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        _invalid_payload(key)
    return value


def _row_bytes(row: Mapping[str, object], key: str) -> bytes:
    value = row.get(key)
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, bytes) or not value:
        _invalid_payload(key)
    return value


def _row_datetime(row: Mapping[str, object], key: str) -> datetime:
    return validate_aware_datetime(row.get(key), key)


def _row_string_mapping(
    row: Mapping[str, object],
    key: str,
) -> Mapping[str, str]:
    value = row.get(key)
    if isinstance(value, str):
        value = loads(value)
    if not isinstance(value, Mapping):
        _invalid_payload(key)
    result: dict[str, str] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, str):
            _invalid_payload(key)
        result[item_key] = item_value
    return result


def _mapping_str(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        _invalid_payload(key)
    return item


def _mapping_optional_str(
    value: Mapping[str, object],
    key: str,
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        _invalid_payload(key)
    return item


def _mapping_int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        _invalid_payload(key)
    return item


def _mapping_optional_int(
    value: Mapping[str, object],
    key: str,
) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool):
        _invalid_payload(key)
    return item


def _mapping_number(value: Mapping[str, object], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, int | float) or isinstance(item, bool):
        _invalid_payload(key)
    return float(item)


def _mapping_optional_number(
    value: Mapping[str, object],
    key: str,
) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, int | float) or isinstance(item, bool):
        _invalid_payload(key)
    return float(item)


def _payload_datetime(
    value: Mapping[str, object],
    key: str,
) -> datetime:
    item = _mapping_str(value, key)
    try:
        parsed = datetime.fromisoformat(item)
    except ValueError:
        _invalid_payload(key)
    return validate_aware_datetime(parsed, key)


def _mapping_optional_datetime(
    value: Mapping[str, object],
    key: str,
) -> datetime | None:
    item = _mapping_optional_str(value, key)
    if item is None:
        return None
    try:
        parsed = datetime.fromisoformat(item)
    except ValueError:
        _invalid_payload(key)
    return validate_aware_datetime(parsed, key)


def _assert_opaque(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_048:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be a bounded non-empty opaque identifier",
        )


def _scope_identity_digest(
    run_id: RunId,
    principal: PrincipalScope,
) -> str:
    """Return one domain-separated run-bound principal identity digest."""
    payload = {
        "principal": {
            "participant_id": (
                None
                if principal.participant_id is None
                else str(principal.participant_id)
            ),
            "session_id": (
                None
                if principal.session_id is None
                else str(principal.session_id)
            ),
            "tenant_id": (
                None
                if principal.tenant_id is None
                else str(principal.tenant_id)
            ),
            "user_id": (
                None if principal.user_id is None else str(principal.user_id)
            ),
        },
        "run_id": str(run_id),
    }
    encoded = dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(b"avalan.interaction.scope.v1\0" + encoded).hexdigest()


def _correlation_parameters(
    correlation: InteractionCorrelation,
    principal: PrincipalScope,
) -> tuple[object, ...]:
    """Return exact searchable parameters for one scoped correlation."""
    return (
        correlation.request_id,
        correlation.continuation_id,
        correlation.run_id,
        correlation.turn_id,
        correlation.task_id,
        correlation.agent_id,
        correlation.branch_id,
        correlation.model_call_id,
        _scope_identity_digest(correlation.run_id, principal),
    )


def _changed_records(
    before: tuple[InteractionRecord, ...],
    after: tuple[InteractionRecord, ...],
) -> tuple[InteractionRecord, ...]:
    """Return only inserted or semantically changed interaction records."""
    previous = {record.request.request_id: record for record in before}
    return tuple(
        record
        for record in after
        if previous.get(record.request.request_id) != record
    )


def _changed_branches(
    before: tuple[InteractionBranchRecord, ...],
    after: tuple[InteractionBranchRecord, ...],
) -> tuple[InteractionBranchRecord, ...]:
    """Return only inserted or semantically changed branch records."""
    previous = {
        (
            record.registration.run_id,
            record.registration.principal,
            record.registration.branch_id,
        ): record
        for record in before
    }
    return tuple(
        record
        for record in after
        if previous.get(
            (
                record.registration.run_id,
                record.registration.principal,
                record.registration.branch_id,
            )
        )
        != record
    )


def _records_with_valid_branch_edges(
    records: tuple[InteractionRecord, ...],
    branches: tuple[InteractionBranchRecord, ...],
) -> tuple[InteractionRecord, ...]:
    """Exclude global-maintenance rows whose required edge is unavailable."""
    registrations = {
        (
            branch.registration.run_id,
            branch.registration.principal,
            branch.registration.branch_id,
        ): branch.registration
        for branch in branches
    }
    selected: list[InteractionRecord] = []
    for record in records:
        origin = record.request.origin
        registration = registrations.get(
            (
                origin.run_id,
                origin.principal,
                origin.branch_id,
            )
        )
        if origin.parent_branch_id is None:
            if registration is None:
                selected.append(record)
            continue
        if (
            registration is not None
            and registration.parent_branch_id == origin.parent_branch_id
            and registration.principal == origin.principal
        ):
            selected.append(record)
    return tuple(selected)


def _validate_record_row_identity(
    row: PgsqlRow,
    record: InteractionRecord,
) -> None:
    """Fail closed when searchable record identity differs from ciphertext."""
    request = record.request
    origin = request.origin
    identity = (
        (_row_str(row, "request_id"), str(request.request_id)),
        (_row_str(row, "continuation_id"), str(request.continuation_id)),
        (_row_str(row, "run_id"), str(origin.run_id)),
        (_row_str(row, "turn_id"), str(origin.turn_id)),
        (_row_optional_str(row, "task_id"), _optional_text(origin.task_id)),
        (_row_str(row, "agent_id"), str(origin.agent_id)),
        (_row_str(row, "branch_id"), str(origin.branch_id)),
        (_row_str(row, "model_call_id"), str(origin.model_call_id)),
        (
            _row_str(row, "scope_identity_digest"),
            _scope_identity_digest(origin.run_id, origin.principal),
        ),
        (_row_str(row, "request_state"), request.state.value),
        (_row_int(row, "state_revision"), int(request.state_revision)),
        (_row_int(row, "store_revision"), int(record.store_revision)),
        (
            _row_datetime(row, "absolute_expires_at"),
            record.absolute_expires_at,
        ),
        (_row_datetime(row, "created_at"), request.created_at),
        (_row_datetime(row, "updated_at"), _record_updated_at(record)),
    )
    if any(persisted != encrypted for persisted, encrypted in identity):
        _invalid_payload("record.identity")


def _validate_branch_row_identity(
    row: PgsqlRow,
    record: InteractionBranchRecord,
) -> None:
    """Fail closed when searchable branch identity differs from ciphertext."""
    registration = record.registration
    identity = (
        (_row_str(row, "run_id"), str(registration.run_id)),
        (_row_str(row, "branch_id"), str(registration.branch_id)),
        (
            _row_str(row, "parent_branch_id"),
            str(registration.parent_branch_id),
        ),
        (_row_int(row, "store_revision"), int(record.store_revision)),
        (
            _row_str(row, "scope_identity_digest"),
            _scope_identity_digest(
                registration.run_id,
                registration.principal,
            ),
        ),
    )
    if any(persisted != encrypted for persisted, encrypted in identity):
        _invalid_payload("branch.identity")


def _validate_branch_row_root(
    row: PgsqlRow,
    record: InteractionBranchRecord,
    roots: Mapping[tuple[str, str, str], str],
) -> None:
    """Fail closed when the searchable root differs from the branch graph."""
    registration = record.registration
    key = (
        str(registration.run_id),
        _scope_identity_digest(
            registration.run_id,
            registration.principal,
        ),
        str(registration.branch_id),
    )
    if _row_str(row, "root_branch_id") != roots[key]:
        _invalid_payload("branch.root_branch_id")


def _optional_text(value: object | None) -> str | None:
    """Return the string representation of one optional opaque identifier."""
    return None if value is None else str(value)


def _scope_identity_conflict(kind: str) -> NoReturn:
    """Reject a write colliding with another authorization scope."""
    raise PgsqlInteractionStoreError(
        InputErrorCode.FORBIDDEN,
        f"{kind}.identity",
        "persisted interaction identity is unavailable",
    )


def _validate_request_continuation_expiry(
    command: CreateInteractionCommand,
    continuation: PortableContinuation,
) -> None:
    request = command.request
    request_expires_at = request.created_at + timedelta(
        seconds=request.continuation_ttl_seconds
    )
    if request_expires_at != continuation.expires_at:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "continuation.expires_at",
            "continuation expiry does not match the interaction deadline",
        )


def _task_coordinator_required(path: str) -> NoReturn:
    raise PgsqlInteractionStoreError(
        InputErrorCode.FORBIDDEN,
        path,
        "task-bound interaction requires the durable task coordinator",
    )


def _assert_digest(value: object, path: str) -> None:
    if not isinstance(value, str) or fullmatch(_SHA256_PATTERN, value) is None:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be a lowercase SHA-256 digest",
        )


def _assert_limit(value: object) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > 1_000
    ):
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            "limit",
            "limit must be between 1 and 1000",
        )


def _invalid_payload(path: str) -> NoReturn:
    raise PgsqlInteractionStoreError(
        InputErrorCode.SNAPSHOT_INVALID,
        path,
        "persisted interaction payload is invalid",
    )


def _stale(path: str) -> NoReturn:
    raise ContinuationStoreConflictError(
        path,
        "durable compare-and-swap token is stale",
    )


def _uuid_id() -> str:
    return str(uuid4())


_CHECK_SCHEMA_SQL = f"""
SELECT "version_num"
FROM "avalan_task_alembic_version"
WHERE "version_num" = '{INTERACTION_PGSQL_HEAD_REVISION}'
"""

_LOCK_STORE_METADATA_SQL = """
SELECT "store_generation", "schedule_revision"
FROM "interaction_store_metadata"
WHERE "singleton_id" = 1
FOR UPDATE
"""

_SELECT_STORE_METADATA_SQL = """
SELECT "store_generation", "schedule_revision"
FROM "interaction_store_metadata"
WHERE "singleton_id" = 1
"""

_SET_REPEATABLE_READ_ONLY_SQL = """
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY
"""

_UPDATE_STORE_METADATA_SQL = """
UPDATE "interaction_store_metadata"
SET
    "store_generation" = %s,
    "schedule_revision" = %s,
    "updated_at" = CURRENT_TIMESTAMP
WHERE "singleton_id" = 1
"""

_SELECT_RECORDS_SQL = """
SELECT *
FROM "interaction_records"
ORDER BY "request_id"
"""

_SELECT_PENDING_RECORD_COUNT_SQL = """
SELECT COUNT(*) AS "pending_count"
FROM "interaction_records"
WHERE "request_state" = 'pending'
"""

_SELECT_SCOPED_RECORD_SQL = """
SELECT *
FROM "interaction_records"
WHERE "request_id" = %s
  AND "continuation_id" = %s
  AND "run_id" = %s
  AND "turn_id" = %s
  AND "task_id" IS NOT DISTINCT FROM %s
  AND "agent_id" = %s
  AND "branch_id" = %s
  AND "model_call_id" = %s
  AND "scope_identity_digest" = %s
ORDER BY "request_id"
"""

_SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL = """
SELECT *
FROM "interaction_records"
WHERE "request_id" = %s
  AND "continuation_id" = %s
ORDER BY "request_id"
FOR UPDATE
"""

_SELECT_SCOPE_RECORDS_SQL = """
SELECT *
FROM "interaction_records"
WHERE "run_id" = %s
  AND "scope_identity_digest" = %s
ORDER BY "request_id"
"""

_SELECT_SCOPE_OWNERSHIP_PRESENCE_SQL = """
WITH RECURSIVE "scope_owners" AS (
    SELECT "scope_identity_digest"
    FROM "interaction_records"
    WHERE "run_id" = %s
    UNION
    SELECT "scope_identity_digest"
    FROM "interaction_branches"
    WHERE "run_id" = %s
),
"scope_branches" ("scope_identity_digest", "branch_id") AS (
    SELECT
        owners."scope_identity_digest",
        %s::text
    FROM "scope_owners" AS owners
    WHERE %s::text IS NOT NULL
    UNION
    SELECT
        child."scope_identity_digest",
        child."branch_id"
    FROM "interaction_branches" AS child
    JOIN "scope_branches" AS parent
      ON parent."scope_identity_digest" = child."scope_identity_digest"
     AND parent."branch_id" = child."parent_branch_id"
    WHERE child."run_id" = %s
      AND %s
),
"matching_records" AS (
    SELECT records."scope_identity_digest"
    FROM "interaction_records" AS records
    WHERE records."run_id" = %s
      AND (%s::text IS NULL OR records."turn_id" = %s)
      AND (%s::text IS NULL OR records."task_id" = %s)
      AND (%s::text IS NULL OR records."agent_id" = %s)
      AND (
          %s::text IS NULL
          OR EXISTS (
              SELECT 1
              FROM "scope_branches" AS branches
              WHERE branches."scope_identity_digest"
                    = records."scope_identity_digest"
                AND branches."branch_id" = records."branch_id"
          )
      )
),
"matching_branches" AS (
    SELECT registered."scope_identity_digest"
    FROM "interaction_branches" AS registered
    JOIN "scope_branches" AS selected
      ON selected."scope_identity_digest"
         = registered."scope_identity_digest"
     AND selected."branch_id" = registered."branch_id"
    WHERE registered."run_id" = %s
      AND %s::text IS NOT NULL
)
SELECT
    COALESCE(
        bool_or("scope_identity_digest" = %s),
        FALSE
    ) AS "actor_owned_record_match",
    COALESCE(
        bool_or("scope_identity_digest" <> %s),
        FALSE
    ) AS "foreign_owned_record_match",
    COALESCE(
        (
            SELECT bool_or("scope_identity_digest" = %s)
            FROM "matching_branches"
        ),
        FALSE
    ) AS "actor_owned_branch_match",
    COALESCE(
        (
            SELECT bool_or("scope_identity_digest" <> %s)
            FROM "matching_branches"
        ),
        FALSE
    ) AS "foreign_owned_branch_match"
FROM "matching_records"
"""

_SELECT_RECORD_DEADLINE_FOR_UPDATE_SQL = """
SELECT "absolute_expires_at"
FROM "interaction_records"
WHERE "request_id" = %s
FOR UPDATE
"""

_SELECT_TASK_SCOPE_IDENTITIES_FOR_UPDATE_SQL = """
SELECT
    records."run_id",
    records."scope_identity_digest"
FROM "interaction_continuations" AS continuation
JOIN "interaction_records" AS records
  ON continuation."request_id" = records."request_id"
WHERE continuation."task_run_id" = %s
ORDER BY records."request_id"
FOR UPDATE OF records, continuation
"""

_SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL = """
SELECT records.*
FROM "interaction_continuations" AS continuation
JOIN "interaction_records" AS records
  ON records."request_id" = continuation."request_id"
WHERE continuation."task_run_id" = %s
ORDER BY records."request_id"
FOR UPDATE OF records, continuation
"""

_SELECT_EXPIRED_TASK_REENTRY_SQL = """
SELECT
    queue.*,
    run."state" AS "durable_run_state",
    (run."claim"->>'claim_token') AS "durable_run_claim_token",
    attempt."attempt_id" AS "durable_attempt_id",
    attempt."state" AS "durable_attempt_state",
    (attempt."context"->'claim'->>'claim_token')
        AS "durable_attempt_claim_token",
    previous."segment_id" AS "previous_segment_id",
    previous."attempt_id" AS "previous_attempt_id",
    previous."run_id" AS "previous_run_id",
    previous."segment_number" AS "previous_segment_number",
    previous."state" AS "previous_segment_state",
    (previous."claim"->>'claim_token')
        AS "previous_segment_claim_token",
    previous."request_id" AS "previous_request_id",
    previous."continuation_id" AS "previous_continuation_id",
    previous."checkpoint_id" AS "previous_checkpoint_id",
    active."segment_id" AS "active_segment_id",
    active."attempt_id" AS "active_attempt_id",
    active."run_id" AS "active_run_id",
    active."segment_number" AS "active_segment_number",
    active."state" AS "active_segment_state",
    (active."claim"->>'claim_token') AS "active_segment_claim_token",
    active."resumed_from_segment_id"
        AS "active_resumed_from_segment_id",
    active."request_id" AS "active_request_id",
    active."continuation_id" AS "active_continuation_id",
    active."checkpoint_id" AS "active_checkpoint_id"
FROM "task_queue_items" AS queue
JOIN "task_runs" AS run
  ON run."run_id" = queue."run_id"
JOIN "task_attempts" AS attempt
  ON attempt."attempt_id" = run."last_attempt_id"
JOIN "task_attempt_segments" AS previous
  ON previous."segment_id" = queue."segment_id"
JOIN "task_attempt_segments" AS active
  ON active."attempt_id" = attempt."attempt_id"
 AND active."segment_number" = (
     SELECT MAX(candidate."segment_number")
     FROM "task_attempt_segments" AS candidate
     WHERE candidate."attempt_id" = attempt."attempt_id"
 )
WHERE queue."queue_item_id" = %s
  AND queue."run_id" = %s
  AND (
      (
          queue."state" = 'claimed'
          AND queue."claim_token" = %s
          AND (run."claim"->>'claim_token') = queue."claim_token"
          AND run."state" IN ('claimed', 'running', 'cancel_requested')
      )
      OR (
          queue."state" = 'dead'
          AND run."state" = 'expired'
          AND attempt."state" = 'failed'
      )
      OR (
          queue."state" = 'dead'
          AND run."state" = 'cancelled'
          AND attempt."state" = 'abandoned'
          AND (
              (
                  active."state" = 'abandoned'
                  AND (active."claim"->>'claim_token') = %s
              )
              OR (
                  active."segment_id" = previous."segment_id"
                  AND active."state" = 'suspended'
                  AND active."claim" IS NULL
                  AND (attempt."context"->'claim'->>'claim_token') = %s
              )
          )
      )
  )
FOR UPDATE OF queue, run, attempt, previous, active
"""

_SELECT_BRANCHES_SQL = """
SELECT *
FROM "interaction_branches"
ORDER BY "run_id", "scope_identity_digest", "branch_id"
"""

_SELECT_SCOPE_BRANCHES_SQL = """
SELECT *
FROM "interaction_branches"
WHERE "run_id" = %s
  AND "scope_identity_digest" = %s
ORDER BY "run_id", "scope_identity_digest", "branch_id"
"""

_SELECT_TASK_BRANCH_CLOSURE_SQL = """
WITH RECURSIVE "task_branch_closure" AS (
    SELECT
        branch."run_id",
        branch."scope_identity_digest",
        branch."branch_id",
        branch."parent_branch_id"
    FROM "interaction_branches" AS branch
    JOIN (
        SELECT DISTINCT
            records."run_id",
            records."scope_identity_digest",
            records."branch_id"
        FROM "interaction_continuations" AS continuation
        JOIN "interaction_records" AS records
          ON records."request_id" = continuation."request_id"
        WHERE continuation."task_run_id" = %s
          AND records."run_id" = %s
          AND records."scope_identity_digest" = %s
    ) AS task_branch
      ON task_branch."run_id" = branch."run_id"
     AND task_branch."scope_identity_digest"
        = branch."scope_identity_digest"
     AND task_branch."branch_id" = branch."branch_id"
    UNION
    SELECT
        parent."run_id",
        parent."scope_identity_digest",
        parent."branch_id",
        parent."parent_branch_id"
    FROM "interaction_branches" AS parent
    JOIN "task_branch_closure" AS child
      ON parent."run_id" = child."run_id"
     AND parent."scope_identity_digest" = child."scope_identity_digest"
     AND parent."branch_id" = child."parent_branch_id"
)
SELECT branch.*
FROM "interaction_branches" AS branch
JOIN "task_branch_closure" AS closure
  ON closure."run_id" = branch."run_id"
 AND closure."scope_identity_digest" = branch."scope_identity_digest"
 AND closure."branch_id" = branch."branch_id"
ORDER BY branch."run_id", branch."scope_identity_digest", branch."branch_id"
FOR UPDATE OF branch
"""

_UPSERT_RECORD_SQL = """
INSERT INTO "interaction_records" (
    "request_id", "continuation_id", "run_id", "turn_id", "task_id",
    "agent_id", "branch_id", "model_call_id", "scope_identity_digest",
    "request_state", "state_revision", "store_revision",
    "absolute_expires_at", "retention_deadline_at", "ciphertext",
    "encryption_key_id", "encryption_algorithm", "encryption_metadata",
    "created_at", "updated_at"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s::jsonb, %s, %s
)
ON CONFLICT ("request_id") DO UPDATE SET
    "request_state" = EXCLUDED."request_state",
    "state_revision" = EXCLUDED."state_revision",
    "store_revision" = EXCLUDED."store_revision",
    "absolute_expires_at" = EXCLUDED."absolute_expires_at",
    "retention_deadline_at" = EXCLUDED."retention_deadline_at",
    "ciphertext" = EXCLUDED."ciphertext",
    "encryption_key_id" = EXCLUDED."encryption_key_id",
    "encryption_algorithm" = EXCLUDED."encryption_algorithm",
    "encryption_metadata" = EXCLUDED."encryption_metadata",
    "updated_at" = EXCLUDED."updated_at"
WHERE "interaction_records"."scope_identity_digest"
    = EXCLUDED."scope_identity_digest"
RETURNING "request_id"
"""

_UPSERT_BRANCH_SQL = """
INSERT INTO "interaction_branches" (
    "run_id", "branch_id", "parent_branch_id", "root_branch_id",
    "store_revision", "scope_identity_digest", "ciphertext",
    "encryption_key_id", "encryption_algorithm", "encryption_metadata"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT ("run_id", "branch_id", "scope_identity_digest") DO UPDATE SET
    "parent_branch_id" = EXCLUDED."parent_branch_id",
    "root_branch_id" = EXCLUDED."root_branch_id",
    "store_revision" = EXCLUDED."store_revision",
    "ciphertext" = EXCLUDED."ciphertext",
    "encryption_key_id" = EXCLUDED."encryption_key_id",
    "encryption_algorithm" = EXCLUDED."encryption_algorithm",
    "encryption_metadata" = EXCLUDED."encryption_metadata",
    "updated_at" = CURRENT_TIMESTAMP
RETURNING "run_id"
"""

_INSERT_CONTINUATION_SQL = """
INSERT INTO "interaction_continuations" (
    "continuation_id", "checkpoint_id", "request_id", "task_run_id",
    "lifecycle_state",
    "state_revision", "store_revision", "fencing_token", "ciphertext",
    "encryption_key_id", "encryption_algorithm", "encryption_metadata",
    "expires_at", "retention_deadline_at", "created_at", "updated_at"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,
    %s
)
ON CONFLICT DO NOTHING
RETURNING *
"""

_SELECT_CONTINUATION_SQL = """
SELECT *
FROM "interaction_continuations"
WHERE "continuation_id" = %s
"""

_SELECT_ACTIVE_CONTINUATIONS_BY_TASK_SQL = """
SELECT *
FROM "interaction_continuations"
WHERE "task_run_id" = %s
  AND "lifecycle_state" IN ('pending', 'ready', 'claimed', 'dispatching')
ORDER BY "created_at", "continuation_id"
LIMIT 2
"""

_SELECT_CONTINUATION_FOR_UPDATE_SQL = """
SELECT
    continuation.*,
    records."absolute_expires_at" AS "request_absolute_expires_at"
FROM "interaction_continuations" AS continuation
JOIN "interaction_records" AS records
  ON records."request_id" = continuation."request_id"
WHERE continuation."continuation_id" = %s
FOR UPDATE OF continuation, records
"""

_SELECT_TASK_RUN_CONTINUATION_FOR_UPDATE_SQL = """
SELECT
    continuation.*,
    records."absolute_expires_at" AS "request_absolute_expires_at"
FROM "interaction_continuations" AS continuation
JOIN "interaction_records" AS records
  ON records."request_id" = continuation."request_id"
WHERE continuation."continuation_id" = %s
  AND continuation."task_run_id" = %s
FOR UPDATE OF continuation, records
"""

_SELECT_TASK_CONTINUATION_FOR_UPDATE_SQL = """
SELECT
    continuation.*,
    records."absolute_expires_at" AS "request_absolute_expires_at"
FROM "interaction_continuations" AS continuation
JOIN "interaction_records" AS records
  ON records."request_id" = continuation."request_id"
WHERE continuation."continuation_id" = %s
  AND continuation."task_run_id" = %s
  AND continuation."request_id" = %s
  AND continuation."checkpoint_id" = %s
FOR UPDATE OF continuation, records
"""

_SELECT_RESUMED_TASK_CONTINUATION_FOR_UPDATE_SQL = """
SELECT
    continuation.*,
    records."absolute_expires_at" AS "request_absolute_expires_at"
FROM "interaction_continuations" AS continuation
JOIN "interaction_records" AS records
  ON records."request_id" = continuation."request_id"
JOIN "task_attempt_segments" AS previous
  ON previous."run_id" = continuation."task_run_id"
 AND previous."request_id" = continuation."request_id"
 AND previous."continuation_id" = continuation."continuation_id"
 AND previous."checkpoint_id" = continuation."checkpoint_id"
JOIN "task_attempt_segments" AS active
  ON active."resumed_from_segment_id" = previous."segment_id"
 AND active."run_id" = previous."run_id"
WHERE continuation."continuation_id" = %s
  AND continuation."task_run_id" = %s
  AND active."segment_id" = %s
FOR UPDATE OF continuation, records, previous, active
"""

_SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL = """
SELECT *
FROM "interaction_continuations"
WHERE "request_id" = %s
FOR UPDATE
"""

_SELECT_CONTINUATION_BY_REQUEST_SQL = """
SELECT *
FROM "interaction_continuations"
WHERE "request_id" = %s
"""

_UPDATE_CONTINUATION_SQL = """
UPDATE "interaction_continuations"
SET
    "lifecycle_state" = %s,
    "state_revision" = %s,
    "store_revision" = %s,
    "claim_owner_id" = %s,
    "claim_lease_expires_at" = %s,
    "fencing_token" = %s,
    "dispatch_id" = %s,
    "dispatch_started_at" = %s,
    "dispatch_completed_at" = %s,
    "dispatch_ambiguous" = %s,
    "invalid_reason" = %s,
    "ciphertext" = %s,
    "encryption_key_id" = %s,
    "encryption_algorithm" = %s,
    "encryption_metadata" = %s::jsonb,
    "updated_at" = %s
WHERE "continuation_id" = %s
  AND "store_revision" = %s
RETURNING *
"""

_INSERT_RESOLUTION_KEY_SQL = """
INSERT INTO "interaction_resolution_keys" (
    "request_id", "idempotency_key", "resolution_digest", "state_revision"
) VALUES (%s, %s, %s, %s)
ON CONFLICT ("request_id", "idempotency_key") DO NOTHING
"""

_INSERT_OUTBOX_SQL = """
INSERT INTO "interaction_resumption_outbox" (
    "outbox_id", "continuation_id", "request_id", "task_run_id",
    "resolution_revision", "available_at", "created_at", "updated_at"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ("continuation_id", "resolution_revision") DO NOTHING
"""

_DEAD_OUTBOX_SQL = """
UPDATE "interaction_resumption_outbox"
SET
    "status" = 'dead',
    "claim_owner_id" = NULL,
    "claim_lease_expires_at" = NULL,
    "updated_at" = %s
WHERE "continuation_id" = %s
  AND "status" IN ('pending', 'claimed')
"""

_SELECT_SWEEP_SQL = """
SELECT
    continuation.*,
    records."request_id" AS "interaction_request_id",
    records."run_id" AS "interaction_run_id",
    records."scope_identity_digest"
        AS "interaction_scope_identity_digest",
    records."retention_deadline_at"
        AS "interaction_retention_deadline_at"
FROM "interaction_records" AS records
LEFT JOIN "interaction_continuations" AS continuation
  ON continuation."request_id" = records."request_id"
WHERE
    (
        continuation."continuation_id" IS NOT NULL
        AND continuation."expires_at" <= %s
    )
    OR records."retention_deadline_at" <= %s
ORDER BY
    records."retention_deadline_at",
    COALESCE(continuation."continuation_id", records."request_id")
FOR UPDATE OF records SKIP LOCKED
LIMIT %s
"""

_DELETE_RECORD_SQL = """
DELETE FROM "interaction_records"
WHERE "request_id" = %s
"""

_LOCK_RETENTION_SCOPE_SQL = """
SELECT pg_advisory_xact_lock(
    hashtextextended(
        jsonb_build_array(
            'avalan.interaction.retention.v1',
            %s::text,
            %s::text
        )::text,
        0
    )
)
"""

_DELETE_ORPHANED_BRANCHES_SQL = """
DELETE FROM "interaction_branches" AS branches
WHERE branches."run_id" = %s
  AND branches."scope_identity_digest" = %s
  AND NOT EXISTS (
      SELECT 1
      FROM "interaction_records" AS records
      WHERE records."run_id" = %s
        AND records."scope_identity_digest" = %s
  )
"""

_CLAIM_OUTBOX_SQL = """
WITH candidates AS (
    SELECT "outbox_id"
    FROM "interaction_resumption_outbox"
    WHERE
        (
            "status" = 'pending'
            AND "available_at" <= %s
        )
        OR (
            "status" = 'claimed'
            AND "claim_lease_expires_at" <= %s
        )
    ORDER BY "available_at", "outbox_id"
    FOR UPDATE SKIP LOCKED
    LIMIT %s
)
UPDATE "interaction_resumption_outbox" AS outbox
SET
    "status" = 'claimed',
    "claim_owner_id" = %s,
    "claim_lease_expires_at" = %s,
    "fencing_token" = outbox."fencing_token" + 1,
    "attempts" = outbox."attempts" + 1,
    "updated_at" = %s
FROM candidates
WHERE outbox."outbox_id" = candidates."outbox_id"
RETURNING outbox.*
"""

_COMPLETE_OUTBOX_SQL = """
UPDATE "interaction_resumption_outbox"
SET
    "status" = 'delivered',
    "claim_owner_id" = NULL,
    "claim_lease_expires_at" = NULL,
    "delivered_at" = %s,
    "updated_at" = %s
WHERE "outbox_id" = %s
  AND "status" = 'claimed'
  AND "claim_owner_id" = %s
  AND "fencing_token" = %s
RETURNING *
"""

_RELEASE_OUTBOX_SQL = """
UPDATE "interaction_resumption_outbox"
SET
    "status" = %s,
    "claim_owner_id" = NULL,
    "claim_lease_expires_at" = NULL,
    "last_error_code" = %s,
    "updated_at" = %s
WHERE "outbox_id" = %s
  AND "status" = 'claimed'
  AND "claim_owner_id" = %s
  AND "fencing_token" = %s
RETURNING *
"""
