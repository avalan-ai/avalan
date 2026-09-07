"""Participate in physical ownership using the caller's PostgreSQL UoW."""

from ...pgsql import PgsqlDatabase, PgsqlUnitOfWork
from ..artifact import ArtifactStoreConflictError, TaskArtifactRef
from ..artifact_codec import _artifact_ref_to_payload
from ..artifact_ownership import (
    ArtifactDeletion,
    ArtifactObject,
    ArtifactObjectState,
    ArtifactRevisionOwner,
    ArtifactStagingOwner,
    physical_artifact_identity,
)
from ..submission import TaskSubmissionOutcome

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from json import dumps
from uuid import UUID, uuid4

ARTIFACT_OWNERSHIP_SCHEMA = "20260907_0002_triggers"


def _object(row: Mapping[str, object]) -> ArtifactObject:
    for key in ("object_id", "store", "storage_key", "sha256", "status"):
        if not isinstance(row[key], str):
            raise ArtifactStoreConflictError("invalid artifact object record")
    was_staged = row["was_staged"]
    if type(was_staged) is not bool:
        raise ArtifactStoreConflictError("invalid artifact object record")
    staged_at = row["staged_at"]
    if not isinstance(staged_at, datetime) or staged_at.utcoffset() is None:
        raise ArtifactStoreConflictError("invalid artifact object record")
    size = row["size_bytes"]
    token = row["cleanup_token"]
    if type(size) is not int or (
        token is not None and not isinstance(token, str)
    ):
        raise ArtifactStoreConflictError("invalid artifact object record")
    assert isinstance(size, int)
    assert token is None or isinstance(token, str)
    # Scalars have been checked before constructing the closed record.
    return ArtifactObject(
        object_id=str(row["object_id"]),
        staged_at=staged_at,
        was_staged=was_staged,
        ref=TaskArtifactRef(
            artifact_id=str(row["object_id"]),
            store=str(row["store"]),
            storage_key=str(row["storage_key"]),
            sha256=str(row["sha256"]),
            size_bytes=size,
        ),
        state=ArtifactObjectState(str(row["status"])),
        cleanup_token=token,
    )


class PgsqlArtifactOwnership:
    """Bind every ownership write to the exact task database instance."""

    def __init__(self, database: PgsqlDatabase) -> None:
        self.database = database

    async def validate_unit(self, unit: PgsqlUnitOfWork) -> None:
        if unit.database is not self.database:
            raise ArtifactStoreConflictError("artifact database differs")
        await unit.cursor.execute(
            "SELECT to_regclass('avalan_task_alembic_version') AS present"
        )
        row = await unit.cursor.fetchone()
        if row is None or row["present"] is None:
            raise ArtifactStoreConflictError(
                "artifact ownership schema missing"
            )
        await unit.cursor.execute(
            "SELECT version_num FROM avalan_task_alembic_version"
        )
        rows = await unit.cursor.fetchall()
        if (
            len(rows) != 1
            or rows[0]["version_num"] != ARTIFACT_OWNERSHIP_SCHEMA
        ):
            raise ArtifactStoreConflictError(
                "artifact ownership schema differs"
            )
        await unit.cursor.execute(
            "SELECT current_setting('transaction_isolation') AS isolation"
        )
        row = await unit.cursor.fetchone()
        if row is None or row["isolation"] != "read committed":
            raise ArtifactStoreConflictError("artifact isolation differs")

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[PgsqlUnitOfWork]:
        async with self.database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    unit = PgsqlUnitOfWork(
                        database=self.database,
                        connection=connection,
                        cursor=cursor,
                    )
                    await self.validate_unit(unit)
                    yield unit

    async def _read(
        self, unit: PgsqlUnitOfWork, object_id: str
    ) -> ArtifactObject | None:
        if unit.database is not self.database:
            raise ArtifactStoreConflictError("artifact database differs")
        await unit.cursor.execute(
            """
SELECT object_id::text, store, storage_key, sha256, size_bytes, status,
       cleanup_token::text, staged_at, was_staged FROM task_artifact_objects
WHERE object_id = %s::uuid FOR UPDATE
""",
            (object_id,),
        )
        row = await unit.cursor.fetchone()
        return _object(row) if row is not None else None

    async def _lock_physical(
        self, unit: PgsqlUnitOfWork, ref: TaskArtifactRef
    ) -> None:
        """Serialize first acquisition even before the object row exists."""
        identity = physical_artifact_identity(ref)
        key = int.from_bytes(UUID(identity).bytes[:8], "big", signed=True)
        await unit.cursor.execute("SELECT pg_advisory_xact_lock(%s)", (key,))

    async def stage(
        self,
        value: ArtifactObject,
        owner: ArtifactStagingOwner,
        *,
        unit_of_work: PgsqlUnitOfWork,
    ) -> None:
        unit = unit_of_work
        await self.validate_unit(unit)
        assert value.state == ArtifactObjectState.LIVE
        await self._lock_physical(unit, value.ref)
        await unit.cursor.execute(
            """
INSERT INTO task_artifact_objects
    (object_id, store, storage_key, sha256, size_bytes, status, created_at)
VALUES (%s::uuid, %s, %s, %s, %s, 'live', clock_timestamp())
ON CONFLICT (store, storage_key) DO NOTHING
""",
            (
                value.object_id,
                value.ref.store,
                value.ref.storage_key,
                value.ref.sha256,
                value.ref.size_bytes,
            ),
        )
        existing = await self._read(unit, value.object_id)
        if existing is None or existing.state != ArtifactObjectState.LIVE:
            raise ArtifactStoreConflictError("artifact deletion is fenced")
        existing.require_same_bytes(value)
        await unit.cursor.execute(
            """
INSERT INTO task_artifact_staging_owners
    (staging_id, object_id, owner_scope_id, recovery_id, outcome, created_at)
VALUES (%s::uuid, %s::uuid, %s, %s, 'unknown', clock_timestamp())
ON CONFLICT (staging_id, object_id) DO NOTHING
""",
            (
                owner.staging_id,
                value.object_id,
                owner.owner_scope,
                owner.recovery_id,
            ),
        )
        await unit.cursor.execute(
            """
INSERT INTO task_artifact_run_owners (artifact_id, object_id)
SELECT artifact_id, %s::uuid FROM task_artifacts
WHERE state = 'ready' AND ref->>'store' = %s AND ref->>'storage_key' = %s
ON CONFLICT (artifact_id) DO NOTHING
""",
            (value.object_id, value.ref.store, value.ref.storage_key),
        )
        await self._require_staging(unit, value.object_id, owner)
        await unit.cursor.execute(
            "UPDATE task_artifact_objects SET staged_at = clock_timestamp(),"
            " was_staged = TRUE WHERE object_id = %s::uuid",
            (value.object_id,),
        )

    async def _require_staging(
        self,
        unit: PgsqlUnitOfWork,
        object_id: str,
        owner: ArtifactStagingOwner,
    ) -> None:
        value = await self._read(unit, object_id)
        if value is None or value.state != ArtifactObjectState.LIVE:
            raise ArtifactStoreConflictError("artifact deletion is fenced")
        await unit.cursor.execute(
            """
SELECT owner_scope_id, recovery_id FROM task_artifact_staging_owners
WHERE staging_id = %s::uuid AND object_id = %s::uuid
""",
            (owner.staging_id, object_id),
        )
        row = await unit.cursor.fetchone()
        if (
            row is None
            or row["owner_scope_id"] != owner.owner_scope
            or (row["recovery_id"] != owner.recovery_id)
        ):
            raise ArtifactStoreConflictError(
                "artifact staging owner is absent"
            )

    async def attach_revision(
        self,
        value: ArtifactObject,
        staging: ArtifactStagingOwner,
        revision: ArtifactRevisionOwner,
        *,
        unit_of_work: PgsqlUnitOfWork,
    ) -> None:
        assert revision.owner_scope == staging.owner_scope
        unit = unit_of_work
        await self._require_staging(unit, value.object_id, staging)
        await unit.cursor.execute(
            """
INSERT INTO trigger_input_artifacts
    (owner_scope_id, trigger_id, revision, object_id, ref)
VALUES (%s, %s, %s, %s::uuid, %s::jsonb)
ON CONFLICT DO NOTHING
""",
            (
                revision.owner_scope,
                revision.resource_id,
                revision.revision,
                value.object_id,
                dumps(_artifact_ref_to_payload(value.ref)),
            ),
        )
        await unit.cursor.execute(
            """
SELECT released_at FROM trigger_input_artifacts
WHERE owner_scope_id = %s AND trigger_id = %s AND revision = %s
    AND object_id = %s::uuid
""",
            (
                revision.owner_scope,
                revision.resource_id,
                revision.revision,
                value.object_id,
            ),
        )
        row = await unit.cursor.fetchone()
        if row is None or row["released_at"] is not None:
            raise ArtifactStoreConflictError(
                "revision artifact owner was released"
            )

    async def attach_run(
        self,
        object_id: str,
        staging: ArtifactStagingOwner,
        artifact_id: str,
        *,
        unit_of_work: PgsqlUnitOfWork,
    ) -> None:
        unit = unit_of_work
        await self._require_staging(unit, object_id, staging)
        await unit.cursor.execute(
            """
INSERT INTO task_artifact_run_owners (artifact_id, object_id)
VALUES (%s, %s::uuid) ON CONFLICT DO NOTHING
""",
            (artifact_id, object_id),
        )
        await unit.cursor.execute(
            """
SELECT object_id::text, released_at FROM task_artifact_run_owners
WHERE artifact_id = %s
""",
            (artifact_id,),
        )
        row = await unit.cursor.fetchone()
        if (
            row is None
            or row["object_id"] != object_id
            or (row["released_at"] is not None)
        ):
            raise ArtifactStoreConflictError("run artifact owner changed")

    async def release_staging(
        self,
        owner: ArtifactStagingOwner,
        *,
        outcome: TaskSubmissionOutcome,
        unit_of_work: PgsqlUnitOfWork,
    ) -> None:
        assert isinstance(outcome, TaskSubmissionOutcome)
        if outcome == TaskSubmissionOutcome.UNKNOWN:
            raise ArtifactStoreConflictError("staging outcome remains unknown")
        await self.validate_unit(unit_of_work)
        await unit_of_work.cursor.execute(
            """
DELETE FROM task_artifact_staging_owners
WHERE staging_id = %s::uuid AND owner_scope_id = %s AND recovery_id = %s
""",
            (owner.staging_id, owner.owner_scope, owner.recovery_id),
        )

    async def attach_submitted_run(
        self, ref: TaskArtifactRef, *, unit_of_work: PgsqlUnitOfWork
    ) -> None:
        """Acquire a persisted run reference in its submission transaction."""
        unit = unit_of_work
        await self.validate_unit(unit)
        await self._lock_physical(unit, ref)
        if ref.sha256 is not None and ref.size_bytes is not None:
            value = ArtifactObject.from_ref(ref)
            await unit.cursor.execute(
                """
INSERT INTO task_artifact_objects
    (object_id, store, storage_key, sha256, size_bytes, status, created_at)
VALUES (%s::uuid, %s, %s, %s, %s, 'live', clock_timestamp())
ON CONFLICT (store, storage_key) DO NOTHING
""",
                (
                    value.object_id,
                    ref.store,
                    ref.storage_key,
                    ref.sha256,
                    ref.size_bytes,
                ),
            )
        await unit.cursor.execute(
            """
SELECT object_id::text, store, storage_key, sha256, size_bytes, status,
       cleanup_token::text, staged_at, was_staged FROM task_artifact_objects
WHERE store = %s AND storage_key = %s FOR UPDATE
""",
            (ref.store, ref.storage_key),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            return
        value = _object(row)
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
        await unit.cursor.execute(
            """
INSERT INTO task_artifact_run_owners (artifact_id, object_id)
VALUES (%s, %s::uuid) ON CONFLICT (artifact_id) DO NOTHING
""",
            (ref.artifact_id, value.object_id),
        )
        await unit.cursor.execute(
            "SELECT object_id::text, released_at FROM task_artifact_run_owners"
            " WHERE artifact_id = %s",
            (ref.artifact_id,),
        )
        row = await unit.cursor.fetchone()
        if (
            row is None
            or row["object_id"] != value.object_id
            or row["released_at"] is not None
        ):
            raise ArtifactStoreConflictError("run artifact ownership changed")

    async def release_run(
        self, artifact_id: str, *, unit_of_work: PgsqlUnitOfWork
    ) -> None:
        await self.validate_unit(unit_of_work)
        await unit_of_work.cursor.execute(
            """
UPDATE task_artifact_run_owners SET released_at = clock_timestamp()
WHERE artifact_id = %s AND released_at IS NULL
""",
            (artifact_id,),
        )

    async def claim_delete(
        self,
        object_id: str,
        *,
        unit_of_work: PgsqlUnitOfWork,
        grace_seconds: int = 86400,
    ) -> ArtifactDeletion | None:
        assert type(grace_seconds) is int and 0 <= grace_seconds <= 2592000
        unit = unit_of_work
        value = await self._read(unit, object_id)
        if value is None or value.state == ArtifactObjectState.DELETED:
            return None
        if value.state == ArtifactObjectState.DELETING:
            return ArtifactDeletion(object=value)
        # A separate READ COMMITTED statement after the row lock sees every
        # earlier acquisition that held the same object lock until commit.
        await unit.cursor.execute(
            """
SELECT EXISTS (
    SELECT 1 FROM task_artifact_staging_owners WHERE object_id = %s::uuid
    UNION ALL SELECT 1 FROM trigger_input_artifacts
        WHERE object_id = %s::uuid AND released_at IS NULL
    UNION ALL SELECT 1 FROM task_artifact_run_owners
        WHERE object_id = %s::uuid AND released_at IS NULL
) AS owned, clock_timestamp() >= %s + %s * interval '1 second' AS aged
""",
            (object_id, object_id, object_id, value.staged_at, grace_seconds),
        )
        row = await unit.cursor.fetchone()
        if row is None or type(row["owned"]) is not bool:
            raise ArtifactStoreConflictError(
                "invalid artifact ownership record"
            )
        if type(row["aged"]) is not bool:
            raise ArtifactStoreConflictError(
                "invalid artifact ownership record"
            )
        if row["owned"] or not row["aged"]:
            return None
        await unit.cursor.execute(
            """
UPDATE task_artifact_objects SET status = 'deleting', cleanup_token = %s::uuid
WHERE object_id = %s::uuid
RETURNING object_id::text, store, storage_key, sha256, size_bytes, status,
          cleanup_token::text, staged_at, was_staged
""",
            (str(uuid4()), object_id),
        )
        row = await unit.cursor.fetchone()
        if row is None:
            raise ArtifactStoreConflictError("artifact deletion fence missing")
        return ArtifactDeletion(object=_object(row))

    async def claim_unowned(
        self, *, limit: int = 50, grace_seconds: int = 86400
    ) -> tuple[ArtifactDeletion, ...]:
        """Fence aged unowned objects under acquisition-compatible locks."""
        assert type(limit) is int and 1 <= limit <= 200
        assert type(grace_seconds) is int and 0 <= grace_seconds <= 2592000
        claimed = []
        async with self.transaction() as unit:
            await unit.cursor.execute(
                """
SELECT object_id::text FROM task_artifact_objects AS object
WHERE status = 'live'
  AND staged_at + %s * interval '1 second' <= clock_timestamp()
  AND NOT EXISTS (SELECT 1 FROM task_artifact_staging_owners AS owner
                  WHERE owner.object_id = object.object_id)
  AND NOT EXISTS (SELECT 1 FROM trigger_input_artifacts AS owner
                  WHERE owner.object_id = object.object_id
                    AND released_at IS NULL)
  AND NOT EXISTS (SELECT 1 FROM task_artifact_run_owners AS owner
                  WHERE owner.object_id = object.object_id
                    AND released_at IS NULL)
ORDER BY object_id LIMIT %s FOR UPDATE SKIP LOCKED
""",
                (grace_seconds, limit),
            )
            rows = await unit.cursor.fetchall()
            for row in rows:
                identity = row["object_id"]
                if not isinstance(identity, str):
                    raise ArtifactStoreConflictError(
                        "invalid artifact object record"
                    )
                value = await self.claim_delete(
                    identity, unit_of_work=unit, grace_seconds=grace_seconds
                )
                if value is not None:
                    claimed.append(value)
        return tuple(claimed)

    async def pending_deletions(
        self, *, limit: int = 50
    ) -> tuple[ArtifactDeletion, ...]:
        """Read committed deletion fences without holding locks during I/O."""
        assert type(limit) is int and 1 <= limit <= 200
        async with self.transaction() as unit:
            await unit.cursor.execute(
                """
SELECT object_id::text, store, storage_key, sha256, size_bytes, status,
       cleanup_token::text, staged_at, was_staged
FROM task_artifact_objects WHERE status = 'deleting'
ORDER BY object_id LIMIT %s
""",
                (limit,),
            )
            return tuple(
                ArtifactDeletion(object=_object(row))
                for row in await unit.cursor.fetchall()
            )

    async def finish_delete(self, deletion: ArtifactDeletion) -> None:
        async with self.transaction() as unit:
            value = await self._read(unit, deletion.object.object_id)
            if value is None or value.cleanup_token != (
                deletion.object.cleanup_token
            ):
                raise ArtifactStoreConflictError(
                    "artifact deletion fence changed"
                )
            await unit.cursor.execute(
                """
UPDATE task_artifact_objects SET status = 'deleted'
WHERE object_id = %s::uuid AND cleanup_token = %s::uuid
""",
                (value.object_id, value.cleanup_token),
            )
