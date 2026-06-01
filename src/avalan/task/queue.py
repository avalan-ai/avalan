from .state import TaskRunState
from .store import (
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
    DONE = "done"
    DEAD = "dead"


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


class TaskQueue(Protocol):
    async def enqueue(
        self,
        run_id: str,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueItem: ...

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


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _assert_datetime(value: datetime, field_name: str) -> None:
    assert isinstance(value, datetime), f"{field_name} must be a datetime"


def _assert_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"


def _assert_non_negative_int(value: int, field_name: str) -> None:
    _assert_int(value, field_name)
    assert value >= 0, f"{field_name} must not be negative"
