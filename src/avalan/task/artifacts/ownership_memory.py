"""Provide rollback-capable physical ownership for combined memory stores."""

from ..artifact import (
    ArtifactStoreConflictError,
    TaskArtifactRef,
    TaskArtifactState,
)
from ..artifact_ownership import (
    ArtifactDeletion,
    ArtifactObject,
    ArtifactObjectState,
    ArtifactRevisionOwner,
    ArtifactStagingOwner,
)
from ..submission import TaskSubmissionOutcome

from asyncio import Lock
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from ..stores.memory import InMemoryTaskStore


@dataclass(slots=True)
class MemoryArtifactUnit:
    """Hold a private transaction snapshot while its owner lock is held."""

    store: "MemoryArtifactOwnership"
    objects: dict[str, ArtifactObject]
    staging: dict[tuple[str, str], ArtifactStagingOwner]
    revisions: set[tuple[ArtifactRevisionOwner, str]]
    runs: dict[str, str]
    active: bool = True

    def require_active(self) -> None:
        assert self.active

    def stage(
        self, value: ArtifactObject, owner: ArtifactStagingOwner
    ) -> None:
        self.require_active()
        assert value.state == ArtifactObjectState.LIVE
        existing = self.objects.get(value.object_id)
        if existing is not None:
            existing.require_same_bytes(value)
            if existing.state != ArtifactObjectState.LIVE:
                raise ArtifactStoreConflictError("artifact deletion is fenced")
        key = (owner.staging_id, value.object_id)
        previous = self.staging.get(key)
        if previous is not None and previous != owner:
            raise ArtifactStoreConflictError("staging identity changed")
        self.objects[value.object_id] = (
            replace(
                existing or value, staged_at=datetime.now(UTC), was_staged=True
            )
            if previous is None
            else existing or value
        )
        self.staging[key] = owner
        tasks = self.store._task_store
        if tasks is not None:
            for record in tasks._artifacts.values():
                if (
                    record.state == TaskArtifactState.READY
                    and record.ref.store == value.ref.store
                    and record.ref.storage_key == value.ref.storage_key
                ):
                    self.attach_submitted_run(record.ref)

    def _require_staging(
        self, object_id: str, staging: ArtifactStagingOwner
    ) -> None:
        self.require_active()
        if self.staging.get((staging.staging_id, object_id)) != staging:
            raise ArtifactStoreConflictError(
                "artifact staging owner is absent"
            )
        if self.objects[object_id].state != ArtifactObjectState.LIVE:
            raise ArtifactStoreConflictError("artifact deletion is fenced")

    def attach_revision(
        self,
        object_id: str,
        staging: ArtifactStagingOwner,
        revision: ArtifactRevisionOwner,
    ) -> None:
        self._require_staging(object_id, staging)
        assert revision.owner_scope == staging.owner_scope
        self.revisions.add((revision, object_id))

    def attach_run(
        self, object_id: str, staging: ArtifactStagingOwner, artifact_id: str
    ) -> None:
        self._require_staging(object_id, staging)
        assert artifact_id.strip()
        previous = self.runs.get(artifact_id)
        if previous is not None and previous != object_id:
            raise ArtifactStoreConflictError("run artifact identity changed")
        self.runs[artifact_id] = object_id

    def release_staging(
        self, owner: ArtifactStagingOwner, *, outcome: TaskSubmissionOutcome
    ) -> None:
        """Release only after the caller's completion-fenced reconciliation."""
        self.require_active()
        assert isinstance(outcome, TaskSubmissionOutcome)
        if outcome == TaskSubmissionOutcome.UNKNOWN:
            raise ArtifactStoreConflictError("staging outcome remains unknown")
        self.staging = {
            key: value for key, value in self.staging.items() if value != owner
        }

    def attach_submitted_run(self, ref: TaskArtifactRef) -> None:
        """Acquire a run reference in the combined submission snapshot."""
        self.require_active()
        value = next(
            (
                item
                for item in self.objects.values()
                if item.ref.store == ref.store
                and item.ref.storage_key == ref.storage_key
            ),
            None,
        )
        if value is None:
            if ref.sha256 is None or ref.size_bytes is None:
                return
            value = ArtifactObject.from_ref(ref)
            self.objects[value.object_id] = value
        if (
            value.state != ArtifactObjectState.LIVE
            or (ref.sha256 is not None and ref.sha256 != value.ref.sha256)
            or (
                ref.size_bytes is not None
                and ref.size_bytes != value.ref.size_bytes
            )
        ):
            raise ArtifactStoreConflictError(
                "physical artifact changed or deletion is fenced"
            )
        previous = self.runs.get(ref.artifact_id)
        if previous is not None and previous != value.object_id:
            raise ArtifactStoreConflictError("run artifact ownership changed")
        self.runs[ref.artifact_id] = value.object_id

    def release_revision(self, owner: ArtifactRevisionOwner) -> None:
        self.require_active()
        self.revisions = {pair for pair in self.revisions if pair[0] != owner}

    def release_run(self, artifact_id: str) -> None:
        self.require_active()
        self.runs.pop(artifact_id, None)

    def claim_delete(
        self,
        object_id: str,
        *,
        grace_seconds: int = 86400,
        now: datetime | None = None,
    ) -> ArtifactDeletion | None:
        assert type(grace_seconds) is int and 0 <= grace_seconds <= 2592000
        decision = now or datetime.now(UTC)
        assert decision.utcoffset() is not None
        self.require_active()
        value = self.objects.get(object_id)
        if value is None or value.state == ArtifactObjectState.DELETED:
            return None
        if value.state == ArtifactObjectState.DELETING:
            return ArtifactDeletion(object=value)
        if decision < value.staged_at + timedelta(seconds=grace_seconds):
            return None
        if (
            any(key[1] == object_id for key in self.staging)
            or any(pair[1] == object_id for pair in self.revisions)
            or object_id in self.runs.values()
        ):
            return None
        value = replace(
            value,
            state=ArtifactObjectState.DELETING,
            cleanup_token=str(uuid4()),
        )
        self.objects[object_id] = value
        return ArtifactDeletion(object=value)


class MemoryArtifactOwnership:
    """Commit immutable ownership snapshots under one transaction lock."""

    def __init__(self, task_store: "InMemoryTaskStore | None" = None) -> None:
        self._task_store = task_store
        self._lock = Lock()
        self._objects: dict[str, ArtifactObject] = {}
        self._staging: dict[tuple[str, str], ArtifactStagingOwner] = {}
        self._revisions: set[tuple[ArtifactRevisionOwner, str]] = set()
        self._runs: dict[str, str] = {}

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[MemoryArtifactUnit]:
        async with self._lock:
            unit = MemoryArtifactUnit(
                store=self,
                objects=dict(self._objects),
                staging=dict(self._staging),
                revisions=set(self._revisions),
                runs=dict(self._runs),
            )
            try:
                yield unit
                self._objects = unit.objects
                self._staging = unit.staging
                self._revisions = unit.revisions
                self._runs = unit.runs
            finally:
                unit.active = False

    async def claim_unowned(
        self, *, limit: int = 50, grace_seconds: int = 86400
    ) -> tuple[ArtifactDeletion, ...]:
        """Fence a bounded batch of aged objects with no live reference."""
        assert type(limit) is int and 1 <= limit <= 200
        assert type(grace_seconds) is int and 0 <= grace_seconds <= 2592000
        claimed = []
        async with self.transaction() as unit:
            for key in sorted(unit.objects):
                if unit.objects[key].state != ArtifactObjectState.LIVE:
                    continue
                value = unit.claim_delete(key, grace_seconds=grace_seconds)
                if value is not None:
                    claimed.append(value)
                    if len(claimed) == limit:
                        break
        return tuple(claimed)

    async def pending_deletions(
        self, *, limit: int = 50
    ) -> tuple[ArtifactDeletion, ...]:
        """Read a bounded set of committed fences for backend retry."""
        assert type(limit) is int and 1 <= limit <= 200
        async with self._lock:
            return tuple(
                ArtifactDeletion(object=value)
                for value in sorted(
                    self._objects.values(), key=lambda item: item.object_id
                )
                if value.state == ArtifactObjectState.DELETING
            )[:limit]

    async def finish_delete(self, deletion: ArtifactDeletion) -> None:
        async with self.transaction() as unit:
            value = unit.objects.get(deletion.object.object_id)
            if value is None or value.cleanup_token != (
                deletion.object.cleanup_token
            ):
                raise ArtifactStoreConflictError(
                    "artifact deletion fence changed"
                )
            unit.objects[value.object_id] = replace(
                value, state=ArtifactObjectState.DELETED
            )
