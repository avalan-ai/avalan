from asyncio import CancelledError
from dataclasses import replace
from datetime import UTC, datetime
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from uuid import uuid4

from pytest import raises

from avalan.task.artifact import (
    ArtifactStoreConflictError,
    ArtifactStoreNotFoundError,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactState,
)
from avalan.task.artifact_ownership import (
    ArtifactDeletion,
    ArtifactObject,
    ArtifactObjectState,
    ArtifactRevisionOwner,
    ArtifactStagingOwner,
    delete_owned_bytes,
)
from avalan.task.artifact_retention import (
    retire_artifact_bytes,
    retry_artifact_cleanup,
)
from avalan.task.artifacts.local import LocalArtifactStore
from avalan.task.artifacts.ownership_memory import MemoryArtifactOwnership
from avalan.task.submission import TaskSubmissionOutcome


def physical(key: str = "bytes") -> ArtifactObject:
    return replace(
        ArtifactObject.from_ref(
            TaskArtifactRef(
                artifact_id=str(uuid4()),
                store="test",
                storage_key=key,
                sha256="a" * 64,
                size_bytes=4,
            )
        ),
        staged_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def staging() -> ArtifactStagingOwner:
    return ArtifactStagingOwner(
        staging_id=str(uuid4()), owner_scope="owner", recovery_id="apply-1"
    )


class ArtifactOwnershipTest(IsolatedAsyncioTestCase):
    async def test_distinct_reference_owners_and_unknown_protection(
        self,
    ) -> None:
        store = MemoryArtifactOwnership()
        value, pending = physical(), staging()
        revision = ArtifactRevisionOwner(
            owner_scope="owner", resource_id="resource", revision=1
        )
        first, second = value.new_reference(), value.new_reference()
        assert first.artifact_id != second.artifact_id != value.ref.artifact_id
        assert first.storage_key == second.storage_key == value.ref.storage_key
        async with store.transaction() as unit:
            unit.stage(value, pending)
            unit.stage(physical(), pending)
            unit.attach_revision(value.object_id, pending, revision)
            unit.attach_run(value.object_id, pending, first.artifact_id)
            unit.attach_run(value.object_id, pending, second.artifact_id)
            unit.attach_run(value.object_id, pending, second.artifact_id)
            with raises(ArtifactStoreConflictError):
                unit.release_staging(
                    pending, outcome=TaskSubmissionOutcome.UNKNOWN
                )
            assert unit.claim_delete(value.object_id, grace_seconds=0) is None
        async with store.transaction() as unit:
            unit.release_staging(
                pending, outcome=TaskSubmissionOutcome.COMMITTED
            )
            assert unit.claim_delete(value.object_id, grace_seconds=0) is None
            unit.release_run(first.artifact_id)
            unit.release_run(second.artifact_id)
            assert unit.claim_delete(value.object_id, grace_seconds=0) is None
            unit.release_revision(revision)
            deletion = unit.claim_delete(value.object_id, grace_seconds=0)
            assert deletion is not None
            assert (
                unit.claim_delete(value.object_id, grace_seconds=0) == deletion
            )
        with raises(AssertionError):
            unit.release_run(first.artifact_id)
        async with store.transaction() as unit:
            with raises(ArtifactStoreConflictError):
                unit.stage(value, staging())
        backend = AsyncMock()
        backend.delete.side_effect = OSError("lost backend acknowledgment")
        with raises(OSError):
            await delete_owned_bytes(deletion, {"test": backend}, store)
        async with store.transaction() as unit:
            assert (
                unit.claim_delete(value.object_id, grace_seconds=0) == deletion
            )
        backend.delete.side_effect = ArtifactStoreNotFoundError("gone")
        await delete_owned_bytes(deletion, {"test": backend}, store)
        await store.finish_delete(deletion)
        async with store.transaction() as unit:
            assert unit.claim_delete(value.object_id, grace_seconds=0) is None
            assert unit.claim_delete(str(uuid4()), grace_seconds=0) is None
            with raises(ArtifactStoreConflictError):
                unit.stage(value, staging())

    async def test_rollback_preserves_all_owner_kinds(self) -> None:
        store = MemoryArtifactOwnership()
        value, pending = physical(), staging()
        revision = ArtifactRevisionOwner(
            owner_scope="owner", resource_id="resource", revision=1
        )
        with raises(CancelledError):
            async with store.transaction() as unit:
                unit.stage(value, pending)
                unit.attach_revision(value.object_id, pending, revision)
                unit.attach_run(value.object_id, pending, "run-artifact")
                raise CancelledError
        async with store.transaction() as unit:
            assert not unit.objects and not unit.staging
            assert not unit.revisions and not unit.runs
            unit.stage(value, pending)
            unit.attach_revision(value.object_id, pending, revision)
            unit.attach_run(value.object_id, pending, "run-artifact")
        with raises(RuntimeError):
            async with store.transaction() as unit:
                unit.release_staging(
                    pending, outcome=TaskSubmissionOutcome.COMMITTED
                )
                unit.release_revision(revision)
                unit.release_run("run-artifact")
                assert (
                    unit.claim_delete(value.object_id, grace_seconds=0)
                    is not None
                )
                raise RuntimeError
        async with store.transaction() as unit:
            assert unit.staging and unit.revisions and unit.runs
            assert (
                unit.objects[value.object_id].state == ArtifactObjectState.LIVE
            )

    async def test_cleanup_grace_unknown_owners_and_retry_batch(self) -> None:
        store = MemoryArtifactOwnership()
        first, second, unknown = (
            physical("first"),
            physical("second"),
            physical("unknown"),
        )
        pending = staging()
        async with store.transaction() as unit:
            for value in (first, second, unknown):
                unit.stage(value, pending)
            unit.release_staging(
                pending, outcome=TaskSubmissionOutcome.NOT_COMMITTED
            )
            unit.stage(unknown, pending)
            assert unit.claim_delete(first.object_id) is None
        assert not await store.claim_unowned()
        backend = AsyncMock()
        backend.delete.side_effect = OSError("lost deletion acknowledgement")
        with raises(OSError):
            await retry_artifact_cleanup(
                store, {"test": backend}, limit=1, orphan_grace_seconds=0
            )
        assert len(await store.pending_deletions()) == 1
        backend.delete.side_effect = None
        assert (
            await retry_artifact_cleanup(
                store, {"test": backend}, limit=1, orphan_grace_seconds=0
            )
            == 1
        )
        assert (
            await retry_artifact_cleanup(
                store, {"test": backend}, orphan_grace_seconds=0
            )
            == 1
        )
        assert not await store.pending_deletions()
        assert not await store.claim_unowned(grace_seconds=0)
        async with store.transaction() as unit:
            assert (
                unit.claim_delete(unknown.object_id, grace_seconds=0) is None
            )
        with raises(AssertionError):
            await store.pending_deletions(limit=0)
        with raises(AssertionError):
            await store.claim_unowned(grace_seconds=2592001)
        with raises(AssertionError):
            replace(first, object_id=str(uuid4()))

    async def test_conflicts_missing_backends_and_tokens(self) -> None:
        store = MemoryArtifactOwnership()
        value, pending = physical(), staging()
        async with store.transaction() as unit:
            with raises(ArtifactStoreConflictError):
                unit.attach_run(value.object_id, pending, "artifact")
            unit.stage(value, pending)
            with raises(ArtifactStoreConflictError):
                unit.stage(value, replace(pending, recovery_id="other"))
            with raises(ArtifactStoreConflictError):
                unit.stage(
                    replace(value, ref=replace(value.ref, sha256="b" * 64)),
                    pending,
                )
            other = physical("other")
            unit.stage(other, pending)
            unit.attach_run(other.object_id, pending, "artifact")
            with raises(ArtifactStoreConflictError):
                unit.attach_run(value.object_id, pending, "artifact")
            unit.release_staging(
                pending, outcome=TaskSubmissionOutcome.COMMITTED
            )
            deletion = unit.claim_delete(value.object_id, grace_seconds=0)
            assert deletion is not None
            # Reject a corrupted/recovered owner pointing at a tombstone.
            unit.staging[(pending.staging_id, value.object_id)] = pending
            with raises(ArtifactStoreConflictError):
                unit.attach_run(value.object_id, pending, "other-artifact")
        with raises(ArtifactStoreNotFoundError):
            await delete_owned_bytes(deletion, {}, store)
        with raises(ArtifactStoreConflictError):
            await store.finish_delete(
                ArtifactDeletion(
                    object=replace(deletion.object, cleanup_token=str(uuid4()))
                )
            )
        backend = AsyncMock()
        await delete_owned_bytes(deletion, {"test": backend}, store)
        backend.delete.assert_awaited_once_with(value.ref)

    async def test_sparse_and_conflicting_direct_references(self) -> None:
        store = MemoryArtifactOwnership()
        first, second = physical("first"), physical("second")
        async with store.transaction() as unit:
            unit.attach_submitted_run(
                replace(first.ref, sha256=None, size_bytes=None)
            )
            assert not unit.objects and not unit.runs
            unit.attach_submitted_run(first.ref)
            with raises(ArtifactStoreConflictError):
                unit.attach_submitted_run(replace(first.ref, sha256="b" * 64))
            with raises(ArtifactStoreConflictError):
                unit.attach_submitted_run(
                    replace(second.ref, artifact_id=first.ref.artifact_id)
                )

    async def test_preexisting_untracked_reference_can_retire(self) -> None:
        with TemporaryDirectory() as root:
            backend = LocalArtifactStore(root, raw_storage_allowed=True)
            ref = await backend.put(b"expired", artifact_id="expired")
            stat = await backend.stat(ref)
            now = datetime.now(UTC)
            record = TaskArtifactRecord(
                artifact_id=ref.artifact_id,
                run_id="run",
                purpose=TaskArtifactPurpose.OUTPUT,
                state=TaskArtifactState.DELETED,
                ref=ref,
                created_at=now,
                updated_at=now,
            )
            ownership = MemoryArtifactOwnership()
            await retire_artifact_bytes(record, stat, backend, ownership)
            with raises(ArtifactStoreNotFoundError):
                await backend.open(ref)
