"""Describe validated submission writes and acknowledged outcomes."""

from ..types import assert_int, assert_non_empty_string
from .artifact import (
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
)
from .context import TaskInputFile
from .idempotency import (
    TaskIdempotencyIdentity,
    TaskIdempotencyReservationResult,
)
from .queue import TaskQueueItem
from .store import (
    TaskExecutionRequest,
    TaskRun,
    TaskSnapshotMetadata,
    empty_snapshot_metadata,
    freeze_snapshot_metadata,
)

from asyncio import CancelledError
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import cast


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSubmissionArtifact:
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
class TaskSubmissionRequest:
    """Provide task input and queue options without owner authority."""

    input_value: object = field(default=None, repr=False)
    files: tuple[TaskInputFile, ...] = field(default=(), repr=False)
    metadata: Mapping[str, object] = field(
        default_factory=empty_snapshot_metadata, repr=False
    )
    available_at: datetime | None = None
    queue_name: str | None = None
    queue_metadata: Mapping[str, object] = field(
        default_factory=empty_snapshot_metadata, repr=False
    )
    idempotency_key: str | None = field(default=None, repr=False)
    idempotency_window: object = field(default=None, repr=False)
    idempotency_expires_at: datetime | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.files, tuple)
        for file in self.files:
            assert isinstance(file, TaskInputFile)
        if self.queue_name is not None:
            assert_non_empty_string(self.queue_name, "queue_name")
        if self.idempotency_key is not None:
            assert_non_empty_string(self.idempotency_key, "idempotency_key")
        for name in ("available_at", "idempotency_expires_at"):
            value = getattr(self, name)
            if value is not None:
                _assert_aware_datetime(value)
        object.__setattr__(
            self, "metadata", freeze_snapshot_metadata(self.metadata)
        )
        assert isinstance(self.queue_metadata, Mapping)
        object.__setattr__(
            self,
            "queue_metadata",
            cast(
                Mapping[str, object],
                _freeze_untrusted_metadata(self.queue_metadata),
            ),
        )


_PREPARATION_AUTHORITY = object()


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedTaskSubmission:
    """Carry validated persistence values and temporary artifact ownership.

    Allocate submission and run identities before persistence. Keep this
    object for reconciliation after an uncertain commit acknowledgment.
    Only preparation may create it; persistence must not accept a raw
    execution request as evidence that validation succeeded.
    """

    submission_id: str
    run_id: str
    owner_scope: str = field(repr=False)
    execution: TaskExecutionRequest = field(repr=False)
    priority: int
    available_at: datetime | None
    idempotency: TaskIdempotencyIdentity | None = field(repr=False)
    idempotency_expires_at: datetime | None
    artifacts: tuple[TaskSubmissionArtifact, ...] = field(repr=False)
    temporary_artifacts: tuple[TaskArtifactRef, ...] = field(repr=False)
    run_metadata: TaskSnapshotMetadata = field(repr=False)
    queue_metadata: TaskSnapshotMetadata = field(repr=False)
    _authority: object = field(repr=False, compare=False)
    _participant: object = field(default=None, repr=False, compare=False)
    occurrence_id: str | None = None
    execution_deployment_id: str | None = None

    def __post_init__(self) -> None:
        assert (
            self._authority is _PREPARATION_AUTHORITY
        ), "task submission requires validated preparation"
        for name in ("submission_id", "run_id", "owner_scope"):
            assert_non_empty_string(getattr(self, name), name)
        for name in ("occurrence_id", "execution_deployment_id"):
            value = getattr(self, name)
            if value is not None:
                assert_non_empty_string(value, name)
        assert isinstance(self.execution, TaskExecutionRequest)
        assert self.execution.queue is not None
        assert_int(self.priority, "priority")
        if self.idempotency is not None:
            assert isinstance(self.idempotency, TaskIdempotencyIdentity)
            assert (
                self.execution.idempotency_key == self.idempotency.identity_key
            )
        else:
            assert self.execution.idempotency_key is None
        for value in (self.available_at, self.idempotency_expires_at):
            if value is not None:
                _assert_aware_datetime(value)
        assert isinstance(self.artifacts, tuple)
        for artifact in self.artifacts:
            assert isinstance(artifact, TaskSubmissionArtifact)
        assert isinstance(self.temporary_artifacts, tuple)
        for ref in self.temporary_artifacts:
            assert isinstance(ref, TaskArtifactRef)
            assert any(artifact.ref == ref for artifact in self.artifacts)
        assert len(
            {
                _artifact_storage_identity(ref)
                for ref in self.temporary_artifacts
            }
        ) == len(self.temporary_artifacts)
        for name in ("run_metadata", "queue_metadata"):
            object.__setattr__(
                self, name, freeze_snapshot_metadata(getattr(self, name))
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSubmissionWrite:
    """Describe provisional SQL writes without acknowledging their commit."""

    submission_id: str
    run: TaskRun
    created: bool
    queue_item: TaskQueueItem | None = None
    idempotency: TaskIdempotencyReservationResult | None = None
    artifacts: tuple[TaskArtifactRecord, ...] = ()

    def __post_init__(self) -> None:
        assert_non_empty_string(self.submission_id, "submission_id")
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


class TaskSubmissionOutcome(StrEnum):
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSubmissionResult:
    """Expose only acknowledged or reconciled committed run associations.

    Unknown outcomes retain the stable submission identity but cannot
    authorize cleanup or retry. A provisional write is not commit evidence.
    """

    submission_id: str
    outcome: TaskSubmissionOutcome
    write: TaskSubmissionWrite | None = None
    prepared: PreparedTaskSubmission | None = field(default=None, repr=False)
    cleanup_pending: tuple[TaskArtifactRef, ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        assert_non_empty_string(self.submission_id, "submission_id")
        assert isinstance(self.outcome, TaskSubmissionOutcome)
        if self.prepared is not None:
            assert isinstance(self.prepared, PreparedTaskSubmission)
            assert self.prepared.submission_id == self.submission_id
        if self.outcome == TaskSubmissionOutcome.COMMITTED:
            assert isinstance(self.write, TaskSubmissionWrite)
            assert self.write.submission_id == self.submission_id
        else:
            assert self.write is None

    @property
    def run(self) -> TaskRun:
        return self.committed_write().run

    @property
    def created(self) -> bool:
        return self.committed_write().created

    @property
    def queue_item(self) -> TaskQueueItem | None:
        return self.committed_write().queue_item

    @property
    def idempotency(self) -> TaskIdempotencyReservationResult | None:
        return self.committed_write().idempotency

    @property
    def artifacts(self) -> tuple[TaskArtifactRecord, ...]:
        return self.committed_write().artifacts

    def committed_write(self) -> TaskSubmissionWrite:
        """Require settled commit evidence before exposing a run."""
        if self.outcome != TaskSubmissionOutcome.COMMITTED:
            raise TaskSubmissionUnsettledError(self)
        assert self.write is not None
        return self.write

    def unused_artifacts(
        self, prepared: PreparedTaskSubmission
    ) -> tuple[TaskArtifactRef, ...]:
        """Return temporary refs safe to release after a settled outcome."""
        assert isinstance(prepared, PreparedTaskSubmission)
        assert prepared.submission_id == self.submission_id
        if self.outcome == TaskSubmissionOutcome.UNKNOWN:
            return ()
        if self.outcome == TaskSubmissionOutcome.NOT_COMMITTED:
            return prepared.temporary_artifacts
        assert self.write is not None
        owned = {
            _artifact_storage_identity(artifact.ref)
            for artifact in self.write.artifacts
        }
        return tuple(
            ref
            for ref in prepared.temporary_artifacts
            if _artifact_storage_identity(ref) not in owned
        )


def _assert_aware_datetime(value: datetime) -> None:
    assert isinstance(value, datetime)
    assert value.tzinfo is not None and value.utcoffset() is not None


def _artifact_storage_identity(ref: TaskArtifactRef) -> tuple[str, str]:
    return ref.store, ref.storage_key


class TaskSubmissionUnsettledError(RuntimeError):
    """Retain an explicit outcome when a caller requires a committed run."""

    def __init__(self, result: TaskSubmissionResult) -> None:
        self.result = result
        super().__init__(f"task submission outcome: {result.outcome.value}")


def _freeze_untrusted_metadata(value: object) -> object:
    """Snapshot queue metadata before its privacy sanitizer handles it."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_untrusted_metadata(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_untrusted_metadata(item) for item in value)
    return value


class TaskSubmissionInterruption(BaseException):
    """Retain the recovery handle and latest outcome after interruption."""

    def __init__(self, result: TaskSubmissionResult) -> None:
        assert result.prepared is not None
        self.result = result
        super().__init__("Task submission interrupted; recovery is available.")


class TaskSubmissionCancelledError(TaskSubmissionInterruption, CancelledError):
    """Preserve asyncio cancellation with an explicit recovery result."""


class TaskSubmissionKeyboardInterrupt(
    TaskSubmissionInterruption, KeyboardInterrupt
):
    """Preserve keyboard interruption with an explicit recovery result."""


class TaskSubmissionSystemExit(TaskSubmissionInterruption, SystemExit):
    """Preserve termination with an explicit recovery result and exit code."""
