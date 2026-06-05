from ..types import assert_non_empty_string as _assert_non_empty_string
from .artifact import (
    ArtifactStore,
    ArtifactStoreNotFoundError,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactState,
    artifact_retention_expired,
)
from .store import (
    TaskSnapshotMetadata,
    TaskStore,
    TaskStoreConflictError,
    freeze_snapshot_metadata,
)

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class TaskRetentionAction(StrEnum):
    DELETED = "deleted"
    LOST = "lost"


class TaskRetentionError(RuntimeError):
    pass


class TaskRetentionStoreNotFoundError(TaskRetentionError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRetentionResult:
    artifact_id: str
    run_id: str
    purpose: TaskArtifactPurpose
    action: TaskRetentionAction
    record: TaskArtifactRecord
    metadata: TaskSnapshotMetadata

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.artifact_id, "artifact_id")
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.purpose, TaskArtifactPurpose)
        assert isinstance(self.action, TaskRetentionAction)
        assert isinstance(self.record, TaskArtifactRecord)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRetentionSweep:
    run_id: str
    enforced_at: datetime
    results: tuple[TaskRetentionResult, ...]

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        _assert_datetime(self.enforced_at, "enforced_at")
        assert isinstance(self.results, tuple)
        for result in self.results:
            assert isinstance(result, TaskRetentionResult)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRetentionBatchSweep:
    enforced_at: datetime
    limit: int
    results: tuple[TaskRetentionResult, ...]

    def __post_init__(self) -> None:
        _assert_datetime(self.enforced_at, "enforced_at")
        _assert_positive_limit(self.limit)
        assert isinstance(self.results, tuple)
        for result in self.results:
            assert isinstance(result, TaskRetentionResult)


class TaskRetentionService:
    def __init__(
        self,
        store: TaskStore,
        artifact_stores: Mapping[str, ArtifactStore],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        assert isinstance(artifact_stores, Mapping)
        for name in artifact_stores:
            _assert_non_empty_string(name, "artifact store name")
        self._store = store
        self._artifact_stores = dict(artifact_stores)
        self._clock = clock or _utc_now

    async def enforce_run(
        self,
        run_id: str,
        *,
        purposes: Collection[TaskArtifactPurpose] | None = None,
        now: datetime | None = None,
    ) -> TaskRetentionSweep:
        _assert_non_empty_string(run_id, "run_id")
        selected_purposes = _purpose_filter(purposes)
        enforced_at = now or self._now()
        _assert_datetime(enforced_at, "now")
        records = await self._store.list_artifacts(
            run_id,
            state=TaskArtifactState.READY,
        )
        results: list[TaskRetentionResult] = []
        for record in records:
            if (
                selected_purposes is not None
                and record.purpose not in selected_purposes
            ):
                continue
            if not _retention_expired(record, enforced_at):
                continue
            result = await self._enforce_record(
                record,
                enforced_at=enforced_at,
            )
            if result is not None:
                results.append(result)
        return TaskRetentionSweep(
            run_id=run_id,
            enforced_at=enforced_at,
            results=tuple(results),
        )

    async def sweep_expired(
        self,
        *,
        purposes: Collection[TaskArtifactPurpose] | None = None,
        now: datetime | None = None,
        limit: int = 100,
    ) -> TaskRetentionBatchSweep:
        selected_purposes = _purpose_filter(purposes)
        enforced_at = now or self._now()
        _assert_datetime(enforced_at, "now")
        _assert_positive_limit(limit)
        results: list[TaskRetentionResult] = []
        for purpose in _sweep_purposes(selected_purposes):
            remaining = limit - len(results)
            if remaining <= 0:
                break
            records = await self._store.list_retention_artifacts(
                expired_at=enforced_at,
                purpose=purpose,
                limit=remaining,
            )
            for record in records:
                if not _retention_expired(record, enforced_at):
                    continue
                result = await self._enforce_record(
                    record,
                    enforced_at=enforced_at,
                )
                if result is not None:
                    results.append(result)
        return TaskRetentionBatchSweep(
            enforced_at=enforced_at,
            limit=limit,
            results=tuple(results),
        )

    async def _enforce_record(
        self,
        record: TaskArtifactRecord,
        *,
        enforced_at: datetime,
    ) -> TaskRetentionResult | None:
        artifact_store = self._artifact_stores.get(record.ref.store)
        if artifact_store is None:
            raise TaskRetentionStoreNotFoundError(
                "artifact store is not configured for retention"
            )
        action = TaskRetentionAction.DELETED
        reason = "retention_expired"
        try:
            await artifact_store.stat(record.ref)
        except ArtifactStoreNotFoundError:
            action = TaskRetentionAction.LOST
            reason = "artifact_bytes_missing"
        metadata = _retention_metadata(
            record,
            action=action,
            enforced_at=enforced_at,
            reason=reason,
        )
        try:
            updated = await self._store.transition_artifact(
                record.artifact_id,
                from_states={TaskArtifactState.READY},
                to_state=_state_for_action(action),
                reason=reason,
                metadata=metadata,
            )
        except TaskStoreConflictError:
            return None
        if action == TaskRetentionAction.DELETED:
            try:
                await artifact_store.delete(record.ref)
            except ArtifactStoreNotFoundError:
                pass
        return TaskRetentionResult(
            artifact_id=record.artifact_id,
            run_id=record.run_id,
            purpose=record.purpose,
            action=action,
            record=updated,
            metadata=metadata,
        )

    def _now(self) -> datetime:
        value = self._clock()
        _assert_datetime(value, "clock result")
        return value


def _retention_expired(
    record: TaskArtifactRecord,
    enforced_at: datetime,
) -> bool:
    return artifact_retention_expired(record, enforced_at)


def _retention_metadata(
    record: TaskArtifactRecord,
    *,
    action: TaskRetentionAction,
    enforced_at: datetime,
    reason: str,
) -> TaskSnapshotMetadata:
    metadata: dict[str, object] = (
        dict(record.metadata) if record.retention.retain_metadata else {}
    )
    audit: dict[str, object] = {
        "action": action.value,
        "reason": reason,
        "deleted_at": enforced_at.isoformat(),
        "purpose": record.purpose.value,
        "previous_state": record.state.value,
        "retain_metadata": record.retention.retain_metadata,
    }
    if record.retention.expires_at is not None:
        audit["expires_at"] = record.retention.expires_at.isoformat()
    if record.retention.delete_after_days is not None:
        audit["delete_after_days"] = record.retention.delete_after_days
    if record.ref.size_bytes is not None:
        audit["size_bytes"] = record.ref.size_bytes
    metadata["retention"] = audit
    return freeze_snapshot_metadata(metadata)


def _state_for_action(action: TaskRetentionAction) -> TaskArtifactState:
    assert isinstance(action, TaskRetentionAction)
    match action:
        case TaskRetentionAction.DELETED:
            return TaskArtifactState.DELETED
        case TaskRetentionAction.LOST:
            return TaskArtifactState.LOST


def _purpose_filter(
    purposes: Collection[TaskArtifactPurpose] | None,
) -> frozenset[TaskArtifactPurpose] | None:
    if purposes is None:
        return None
    assert isinstance(purposes, Collection), "purposes must be a collection"
    assert purposes, "purposes must not be empty"
    selected: set[TaskArtifactPurpose] = set()
    for purpose in purposes:
        assert isinstance(
            purpose,
            TaskArtifactPurpose,
        ), "purposes must contain TaskArtifactPurpose values"
        selected.add(purpose)
    return frozenset(selected)


def _sweep_purposes(
    purposes: frozenset[TaskArtifactPurpose] | None,
) -> tuple[TaskArtifactPurpose | None, ...]:
    if purposes is None:
        return (None,)
    return tuple(sorted(purposes, key=lambda purpose: purpose.value))


def _assert_datetime(value: datetime, field_name: str) -> None:
    assert isinstance(value, datetime), f"{field_name} must be a datetime"


def _assert_positive_limit(value: int) -> None:
    assert isinstance(value, int), "limit must be an integer"
    assert not isinstance(value, bool), "limit must be an integer"
    assert value > 0, "limit must be positive"


def _utc_now() -> datetime:
    return datetime.now(UTC)
