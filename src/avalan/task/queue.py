from ..interaction import (
    ContinuationCompletionCommand,
    ContinuationRejectionCommand,
    CreateInteractionCommand,
    PortableContinuation,
)
from ..types import (
    assert_int as _assert_int,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from .artifact import (
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
)
from .idempotency import (
    TaskIdempotencyIdentity,
    TaskIdempotencyReservationResult,
)
from .settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
)
from .state import TaskAttemptSegmentState, TaskAttemptState, TaskRunState
from .store import (
    TaskAttempt,
    TaskAttemptSegment,
    TaskClaim,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskSnapshotMetadata,
    empty_snapshot_metadata,
    freeze_snapshot_metadata,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class TaskQueueError(RuntimeError):
    pass


class TaskQueueConflictError(TaskQueueError):
    pass


class TaskQueueNotFoundError(TaskQueueError):
    pass


class TaskQueueItemState(StrEnum):
    AVAILABLE = "available"
    CLAIMED = "claimed"
    SUSPENDED = "suspended"
    DONE = "done"
    DEAD = "dead"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueArtifact:
    ref: TaskArtifactRef
    purpose: TaskArtifactPurpose = TaskArtifactPurpose.INPUT
    state: TaskArtifactState = TaskArtifactState.READY
    provenance: TaskArtifactProvenance = field(
        default_factory=TaskArtifactProvenance
    )
    retention: TaskArtifactRetention = field(
        default_factory=TaskArtifactRetention
    )
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        assert isinstance(self.ref, TaskArtifactRef)
        assert isinstance(self.purpose, TaskArtifactPurpose)
        assert self.state == TaskArtifactState.READY
        assert isinstance(self.provenance, TaskArtifactProvenance)
        assert isinstance(self.retention, TaskArtifactRetention)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueItem:
    queue_item_id: str
    run_id: str
    queue_name: str
    state: TaskQueueItemState
    priority: int
    available_at: datetime
    attempts: int
    created_at: datetime
    updated_at: datetime
    run_state: TaskRunState
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    worker_id: str | None = None
    claim_token: str | None = None
    heartbeat_at: datetime | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.queue_item_id, "queue_item_id")
        _assert_non_empty_string(self.run_id, "run_id")
        _assert_non_empty_string(self.queue_name, "queue_name")
        assert isinstance(self.state, TaskQueueItemState)
        _assert_int(self.priority, "priority")
        _assert_datetime(self.available_at, "available_at")
        _assert_non_negative_int(self.attempts, "attempts")
        _assert_datetime(self.created_at, "created_at")
        _assert_datetime(self.updated_at, "updated_at")
        assert self.updated_at >= self.created_at
        assert isinstance(self.run_state, TaskRunState)
        if self.claimed_at is not None:
            _assert_datetime(self.claimed_at, "claimed_at")
        if self.lease_expires_at is not None:
            _assert_datetime(self.lease_expires_at, "lease_expires_at")
            assert (
                self.claimed_at is not None
            ), "lease_expires_at requires claimed_at"
            assert (
                self.lease_expires_at > self.claimed_at
            ), "lease_expires_at must be after claimed_at"
        if self.worker_id is not None:
            _assert_non_empty_string(self.worker_id, "worker_id")
        if self.claim_token is not None:
            _assert_non_empty_string(self.claim_token, "claim_token")
        if self.heartbeat_at is not None:
            _assert_datetime(self.heartbeat_at, "heartbeat_at")
            assert self.claimed_at is not None, "heartbeat_at requires claim"
            assert (
                self.heartbeat_at >= self.claimed_at
            ), "heartbeat_at must not be before claimed_at"
        if self.state == TaskQueueItemState.CLAIMED:
            assert self.claimed_at is not None, "claimed jobs need claim time"
            assert (
                self.lease_expires_at is not None
            ), "claimed jobs need lease expiry"
            assert self.worker_id is not None, "claimed jobs need worker id"
            assert self.claim_token is not None, "claimed jobs need token"
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    @property
    def cancel_requested(self) -> bool:
        return self.run_state == TaskRunState.CANCEL_REQUESTED


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueDepth:
    queue_name: str
    available: int
    scheduled: int
    claimed: int
    dead: int
    cancel_requested: int

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.queue_name, "queue_name")
        _assert_non_negative_int(self.available, "available")
        _assert_non_negative_int(self.scheduled, "scheduled")
        _assert_non_negative_int(self.claimed, "claimed")
        _assert_non_negative_int(self.dead, "dead")
        _assert_non_negative_int(
            self.cancel_requested,
            "cancel_requested",
        )

    @property
    def active(self) -> int:
        return self.available + self.scheduled + self.claimed


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueHealth:
    queue_name: str
    depth: TaskQueueDepth
    checked_at: datetime
    oldest_available_at: datetime | None = None
    expired_claims: int = 0

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.queue_name, "queue_name")
        assert isinstance(self.depth, TaskQueueDepth)
        assert self.depth.queue_name == self.queue_name
        _assert_datetime(self.checked_at, "checked_at")
        if self.oldest_available_at is not None:
            _assert_datetime(self.oldest_available_at, "oldest_available_at")
        _assert_non_negative_int(self.expired_claims, "expired_claims")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueSubmission:
    run: TaskRun
    created: bool
    queue_item: TaskQueueItem | None = None
    idempotency: TaskIdempotencyReservationResult | None = None
    artifacts: tuple[TaskArtifactRecord, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.created, bool)
        if self.queue_item is not None:
            assert isinstance(self.queue_item, TaskQueueItem)
            assert self.queue_item.run_id == self.run.run_id
        if self.idempotency is not None:
            assert isinstance(
                self.idempotency, TaskIdempotencyReservationResult
            )
            assert self.idempotency.reservation.run_id == self.run.run_id
        assert isinstance(self.artifacts, tuple)
        for artifact in self.artifacts:
            assert isinstance(artifact, TaskArtifactRecord)
            assert artifact.run_id == self.run.run_id


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueClaim:
    queue_item: TaskQueueItem
    run: TaskRun
    attempt: TaskAttempt

    def __post_init__(self) -> None:
        assert isinstance(self.queue_item, TaskQueueItem)
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        assert self.queue_item.state == TaskQueueItemState.CLAIMED
        assert self.run.state == TaskRunState.CLAIMED
        assert self.run.claim is not None
        assert isinstance(self.run.claim, TaskClaim)
        assert self.attempt.state in {
            TaskAttemptState.CREATED,
            TaskAttemptState.SUSPENDED,
        }
        assert self.queue_item.run_id == self.run.run_id
        assert self.attempt.run_id == self.run.run_id
        assert self.queue_item.claim_token == self.run.claim.claim_token
        assert self.queue_item.worker_id == self.run.claim.worker_id
        assert self.queue_item.claimed_at == self.run.claim.claimed_at
        assert (
            self.queue_item.lease_expires_at == self.run.claim.lease_expires_at
        )
        assert self.attempt.context.claim == self.run.claim


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueSuspension:
    queue_item: TaskQueueItem
    run: TaskRun
    attempt: TaskAttempt
    segment: TaskAttemptSegment
    request_id: str
    continuation_id: str
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.queue_item, TaskQueueItem)
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        assert isinstance(self.segment, TaskAttemptSegment)
        _assert_non_empty_string(self.request_id, "request_id")
        _assert_non_empty_string(self.continuation_id, "continuation_id")
        if self.checkpoint_id is not None:
            _assert_non_empty_string(self.checkpoint_id, "checkpoint_id")
        assert self.queue_item.run_id == self.run.run_id
        assert self.attempt.run_id == self.run.run_id
        assert self.segment.run_id == self.run.run_id
        assert self.segment.attempt_id == self.attempt.attempt_id
        assert self.queue_item.state == TaskQueueItemState.SUSPENDED
        assert self.run.state == TaskRunState.INPUT_REQUIRED
        assert self.attempt.state == TaskAttemptState.SUSPENDED
        assert self.segment.state == TaskAttemptSegmentState.SUSPENDED
        assert self.segment.request_id == self.request_id
        assert self.segment.continuation_id == self.continuation_id
        assert self.segment.checkpoint_id == self.checkpoint_id
        assert self.queue_item.claimed_at is None
        assert self.queue_item.lease_expires_at is None
        assert self.queue_item.worker_id is None
        assert self.queue_item.claim_token is None
        assert self.run.claim is None


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueReentry:
    queue_item: TaskQueueItem
    run: TaskRun
    attempt: TaskAttempt
    previous_segment: TaskAttemptSegment

    def __post_init__(self) -> None:
        assert isinstance(self.queue_item, TaskQueueItem)
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        assert isinstance(self.previous_segment, TaskAttemptSegment)
        assert self.queue_item.run_id == self.run.run_id
        assert self.attempt.run_id == self.run.run_id
        assert self.previous_segment.run_id == self.run.run_id
        assert self.previous_segment.attempt_id == self.attempt.attempt_id
        assert self.queue_item.state == TaskQueueItemState.AVAILABLE
        assert self.run.state == TaskRunState.QUEUED
        assert self.attempt.state == TaskAttemptState.SUSPENDED
        assert self.previous_segment.state == TaskAttemptSegmentState.SUSPENDED
        assert self.queue_item.claimed_at is None
        assert self.queue_item.lease_expires_at is None
        assert self.queue_item.worker_id is None
        assert self.queue_item.claim_token is None
        assert self.run.claim is None


class TaskDurableSuspensionCommit(Protocol):
    @property
    def suspension(self) -> TaskQueueSuspension: ...


class TaskDurableResuspensionCommit(Protocol):
    @property
    def suspension(self) -> TaskQueueSuspension: ...


class TaskDurableSettlementCommit(Protocol):
    @property
    def completion(self) -> "TaskQueueCompletion": ...


class TaskDurableAmbiguityCommit(Protocol):
    @property
    def completion(self) -> "TaskQueueCompletion": ...


class TaskDurableRejectionCommit(Protocol):
    @property
    def completion(self) -> "TaskQueueCompletion": ...


class TaskDurableSuspensionCoordinator(Protocol):
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
    ) -> TaskDurableSuspensionCommit: ...

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
    ) -> TaskDurableResuspensionCommit: ...

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
    ) -> TaskDurableSettlementCommit: ...

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
    ) -> TaskDurableSettlementCommit: ...

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
    ) -> TaskQueueReentry: ...

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
    ) -> TaskDurableAmbiguityCommit: ...

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
    ) -> TaskQueueReentry: ...

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
    ) -> "TaskQueueCompletion": ...

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
    ) -> TaskDurableRejectionCommit: ...

    async def reconcile_expired_reentry(
        self,
        *,
        queue_item_id: str,
        expected_claim_token: str,
        task_run_id: str,
        result: TaskExecutionResult,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskDurableExpiredReentryCommit": ...


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueCompletion:
    queue_item: TaskQueueItem
    run: TaskRun
    attempt: TaskAttempt

    def __post_init__(self) -> None:
        assert isinstance(self.queue_item, TaskQueueItem)
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        assert self.queue_item.run_id == self.run.run_id
        assert self.attempt.run_id == self.run.run_id
        assert self.queue_item.state in {
            TaskQueueItemState.DONE,
            TaskQueueItemState.DEAD,
        }
        assert self.run.state in {
            TaskRunState.SUCCEEDED,
            TaskRunState.FAILED,
            TaskRunState.CANCELLED,
            TaskRunState.EXPIRED,
        }
        assert self.attempt.state in {
            TaskAttemptState.SUCCEEDED,
            TaskAttemptState.FAILED,
            TaskAttemptState.ABANDONED,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDurableExpiredReentryCommit:
    """Carry one atomic expired durable-reentry reconciliation."""

    reentry: TaskQueueReentry | None = None
    completion: TaskQueueCompletion | None = None

    def __post_init__(self) -> None:
        if (self.reentry is None) == (self.completion is None):
            raise AssertionError(
                "expired reentry must be restored or terminalized"
            )
        if self.reentry is not None:
            assert isinstance(self.reentry, TaskQueueReentry)
        if self.completion is not None:
            assert isinstance(self.completion, TaskQueueCompletion)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueRetry:
    queue_item: TaskQueueItem
    run: TaskRun
    attempt: TaskAttempt

    def __post_init__(self) -> None:
        assert isinstance(self.queue_item, TaskQueueItem)
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        assert self.queue_item.run_id == self.run.run_id
        assert self.attempt.run_id == self.run.run_id
        assert (
            self.queue_item.state == TaskQueueItemState.AVAILABLE
            and self.run.state == TaskRunState.QUEUED
        ) or (
            self.queue_item.state == TaskQueueItemState.DEAD
            and self.run.state == TaskRunState.FAILED
        )
        assert self.attempt.state == TaskAttemptState.FAILED

    @property
    def retryable(self) -> bool:
        return self.queue_item.state == TaskQueueItemState.AVAILABLE

    @property
    def exhausted(self) -> bool:
        return self.queue_item.state == TaskQueueItemState.DEAD


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskQueueAbandonment:
    queue_item: TaskQueueItem
    run: TaskRun
    attempt: TaskAttempt

    def __post_init__(self) -> None:
        assert isinstance(self.queue_item, TaskQueueItem)
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        assert self.queue_item.run_id == self.run.run_id
        assert self.attempt.run_id == self.run.run_id
        assert self.queue_item.state in {
            TaskQueueItemState.AVAILABLE,
            TaskQueueItemState.DEAD,
        }
        assert self.run.state in {
            TaskRunState.QUEUED,
            TaskRunState.FAILED,
            TaskRunState.CANCELLED,
        }
        assert self.attempt.state == TaskAttemptState.ABANDONED

    @property
    def retryable(self) -> bool:
        return self.queue_item.state == TaskQueueItemState.AVAILABLE


class TaskQueue(Protocol):
    async def enqueue_run(
        self,
        request: TaskExecutionRequest,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency: TaskIdempotencyIdentity | None = None,
        idempotency_expires_at: datetime | None = None,
        artifacts: tuple[TaskQueueArtifact, ...] = (),
        run_metadata: Mapping[str, object] | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission: ...

    async def enqueue(
        self,
        run_id: str,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueItem: ...

    async def claim(
        self,
        queue_name: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueClaim | None: ...

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
    ) -> TaskQueueItem: ...

    async def suspend_claim(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        segment_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSuspension: ...

    async def requeue_suspended(
        self,
        run_id: str,
        *,
        request_id: str,
        continuation_id: str,
        resolution_revision: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueReentry: ...

    async def complete(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        result: TaskExecutionResult | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion: ...

    async def retry(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        result: TaskExecutionResult,
        available_at: datetime,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueRetry: ...

    async def abandon(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueAbandonment: ...

    async def abandon_expired(
        self,
        queue_name: str,
        *,
        max_attempts: int,
        limit: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[TaskQueueAbandonment, ...]: ...

    async def drain(
        self,
        queue_name: str,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> tuple[TaskQueueItem, ...]: ...

    async def depth(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueDepth: ...

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth: ...


def _assert_datetime(value: datetime, field_name: str) -> None:
    assert isinstance(value, datetime), f"{field_name} must be a datetime"
