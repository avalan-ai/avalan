"""Validate malformed ownership rows without presenting mocks as SQL proof."""

from dataclasses import replace
from datetime import UTC, datetime
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pytest import mark, raises
from trigger.pgsql_store_test import Database

from avalan.pgsql import PgsqlRow, PgsqlUnitOfWork
from avalan.task.artifact import ArtifactStoreConflictError, TaskArtifactRef
from avalan.task.artifact_ownership import (
    ArtifactDeletion,
    ArtifactObject,
    ArtifactObjectState,
    ArtifactRevisionOwner,
    ArtifactStagingOwner,
)
from avalan.task.artifacts.ownership_pgsql import (
    PgsqlArtifactOwnership,
    _object,
)
from avalan.task.submission import TaskSubmissionOutcome


def row() -> dict[str, object]:
    value = ArtifactObject.from_ref(
        TaskArtifactRef(
            artifact_id="ref",
            store="object",
            storage_key="objects/ph/physical",
            sha256="a" * 64,
            size_bytes=4,
        )
    )
    return {
        "object_id": value.object_id,
        "store": value.ref.store,
        "storage_key": value.ref.storage_key,
        "sha256": value.ref.sha256,
        "size_bytes": 4,
        "status": "live",
        "cleanup_token": None,
        "staged_at": datetime.now(UTC),
        "was_staged": True,
    }


@mark.parametrize(
    "field,value",
    [
        ("object_id", 1),
        ("was_staged", 1),
        ("staged_at", datetime(2026, 1, 1)),
        ("size_bytes", True),
        ("cleanup_token", 1),
        ("status", "unknown"),
    ],
)
def test_invalid_physical_rows_are_rejected(field: str, value: object) -> None:
    invalid = row()
    invalid[field] = value
    with raises((ArtifactStoreConflictError, ValueError)):
        _object(invalid)


class OwnershipCapabilityTest(IsolatedAsyncioTestCase):
    async def test_pairing_missing_schema_and_isolation_fail_closed(
        self,
    ) -> None:
        database = Database()
        ownership = PgsqlArtifactOwnership(database)
        unit = PgsqlUnitOfWork(
            database=database,
            connection=database.handle,
            cursor=database.cursor,
        )
        wrong = PgsqlUnitOfWork(
            database=Database(),
            connection=database.handle,
            cursor=database.cursor,
        )
        with raises(ArtifactStoreConflictError):
            await ownership.validate_unit(wrong)
        with raises(ArtifactStoreConflictError):
            await ownership._read(wrong, str(row()["object_id"]))
        # Use the cursor's explicit response slots for this protocol boundary.
        database.cursor.present = False
        # Ownership expects its own query alias; a narrow cursor override
        # returns typed evidence for each requested capability.
        original = database.cursor.fetchone

        async def evidence() -> PgsqlRow | None:
            if "to_regclass" in database.cursor.query:
                return {
                    "present": "version" if database.cursor.present else None
                }
            return await original()

        with patch.object(database.cursor, "fetchone", evidence):
            with raises(ArtifactStoreConflictError):
                await ownership.validate_unit(unit)
            database.cursor.present = True
            database.cursor.version = None
            with raises(ArtifactStoreConflictError):
                await ownership.validate_unit(unit)
            database.cursor.version = "20260907_0002_triggers"
            database.cursor.isolation = "repeatable read"
            with raises(ArtifactStoreConflictError):
                await ownership.validate_unit(unit)

    async def test_invalid_cleanup_evidence_and_missing_fences(self) -> None:
        database = Database()
        ownership = PgsqlArtifactOwnership(database)
        unit = PgsqlUnitOfWork(
            database=database,
            connection=database.handle,
            cursor=database.cursor,
        )
        value = _object(row())
        owner = ArtifactStagingOwner(
            staging_id=str(uuid4()), owner_scope="owner", recovery_id="request"
        )
        with raises(ArtifactStoreConflictError):
            await ownership.release_staging(
                owner, outcome=TaskSubmissionOutcome.UNKNOWN, unit_of_work=unit
            )
        with patch.object(ownership, "_read", AsyncMock(return_value=None)):
            assert (
                await ownership.claim_delete(
                    value.object_id, unit_of_work=unit
                )
                is None
            )
            with raises(ArtifactStoreConflictError):
                await ownership._require_staging(unit, value.object_id, owner)
        deleting = replace(
            value,
            state=ArtifactObjectState.DELETING,
            cleanup_token=str(uuid4()),
        )
        with patch.object(
            ownership, "_read", AsyncMock(return_value=deleting)
        ):
            assert await ownership.claim_delete(
                value.object_id, unit_of_work=unit
            ) == ArtifactDeletion(object=deleting)
        with patch.object(ownership, "_read", AsyncMock(return_value=value)):
            for evidence in (
                None,
                {"owned": 1, "aged": True},
                {"owned": False, "aged": 1},
            ):
                with patch.object(
                    database.cursor,
                    "fetchone",
                    AsyncMock(return_value=evidence),
                ):
                    with raises(ArtifactStoreConflictError):
                        await ownership.claim_delete(
                            value.object_id, unit_of_work=unit
                        )
            with patch.object(
                database.cursor,
                "fetchone",
                AsyncMock(side_effect=[{"owned": False, "aged": True}, None]),
            ):
                with raises(ArtifactStoreConflictError):
                    await ownership.claim_delete(
                        value.object_id, unit_of_work=unit
                    )
            with patch.object(
                database.cursor, "fetchone", AsyncMock(return_value=None)
            ):
                with raises(ArtifactStoreConflictError):
                    await ownership._require_staging(
                        unit, value.object_id, owner
                    )

    async def test_released_or_corrupted_acquisitions_fail_closed(
        self,
    ) -> None:
        database = Database()
        ownership = PgsqlArtifactOwnership(database)
        unit = PgsqlUnitOfWork(
            database=database,
            connection=database.handle,
            cursor=database.cursor,
        )
        value = _object(row())
        owner = ArtifactStagingOwner(
            staging_id=str(uuid4()), owner_scope="owner", recovery_id="request"
        )
        revision = ArtifactRevisionOwner(
            owner_scope="owner", resource_id="trigger", revision=1
        )
        with patch.object(ownership, "validate_unit", AsyncMock()):
            with patch.object(
                ownership, "_read", AsyncMock(return_value=None)
            ):
                with raises(ArtifactStoreConflictError):
                    await ownership.stage(value, owner, unit_of_work=unit)
            with patch.object(ownership, "_require_staging", AsyncMock()):
                with patch.object(
                    database.cursor, "fetchone", AsyncMock(return_value=None)
                ):
                    with raises(ArtifactStoreConflictError):
                        await ownership.attach_revision(
                            value, owner, revision, unit_of_work=unit
                        )
                    with raises(ArtifactStoreConflictError):
                        await ownership.attach_run(
                            value.object_id,
                            owner,
                            "artifact",
                            unit_of_work=unit,
                        )
            with patch.object(
                ownership, "_read", AsyncMock(return_value=value)
            ):
                with patch.object(
                    database.cursor,
                    "fetchone",
                    AsyncMock(side_effect=[row(), None]),
                ):
                    with raises(ArtifactStoreConflictError):
                        await ownership.attach_submitted_run(
                            value.ref, unit_of_work=unit
                        )
            with patch.object(
                database.cursor,
                "fetchall",
                AsyncMock(return_value=({"object_id": 1},)),
            ):
                with raises(ArtifactStoreConflictError):
                    await ownership.claim_unowned()
            with patch.object(
                ownership, "_read", AsyncMock(return_value=None)
            ):
                deletion = ArtifactDeletion(
                    object=replace(
                        value,
                        state=ArtifactObjectState.DELETING,
                        cleanup_token=str(uuid4()),
                    )
                )
                with raises(ArtifactStoreConflictError):
                    await ownership.finish_delete(deletion)
