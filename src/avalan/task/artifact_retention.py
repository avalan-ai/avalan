"""Fence physical cleanup while retaining independent run/revision owners."""

from .artifact import ArtifactStore, TaskArtifactRecord, TaskArtifactStat
from .artifact_ownership import (
    ArtifactObject,
    delete_owned_bytes,
)
from .artifacts.ownership_memory import MemoryArtifactOwnership
from .artifacts.ownership_pgsql import PgsqlArtifactOwnership
from .state import is_terminal_run_state
from .store import TaskStore

from collections.abc import Mapping
from dataclasses import replace


async def can_release_run_artifact(
    record: TaskArtifactRecord,
    store: TaskStore,
    ownership: PgsqlArtifactOwnership | MemoryArtifactOwnership,
) -> bool:
    """Retain a managed reference until its authoritative run is terminal."""
    if isinstance(ownership, PgsqlArtifactOwnership):
        async with ownership.transaction() as unit:
            await unit.cursor.execute(
                "SELECT owner.artifact_id FROM task_artifact_run_owners AS"
                " owner JOIN task_artifact_objects AS object USING (object_id)"
                " WHERE artifact_id = %s AND released_at IS NULL AND"
                " object.was_staged",
                (record.artifact_id,),
            )
            managed = await unit.cursor.fetchone() is not None
    else:
        async with ownership.transaction() as memory:
            identity = memory.runs.get(record.artifact_id)
            managed = (
                identity is not None and memory.objects[identity].was_staged
            )
    if not managed:
        return True
    run = await store.get_run(record.run_id)
    return is_terminal_run_state(run.state)


async def retire_artifact_bytes(
    record: TaskArtifactRecord,
    stat: TaskArtifactStat,
    backend: ArtifactStore,
    ownership: PgsqlArtifactOwnership | MemoryArtifactOwnership,
) -> None:
    """Retire one reference and delete only after committing an owner fence."""
    value = ArtifactObject.from_ref(
        replace(stat.ref, sha256=stat.sha256, size_bytes=stat.size_bytes)
    )
    # Retirement is not a staging acquisition. Preserve the age and every
    # existing owner; only previously unmanaged ordinary references follow
    # their already-satisfied retention deadline without orphan grace.
    if isinstance(ownership, PgsqlArtifactOwnership):
        async with ownership.transaction() as unit:
            existing = await ownership._read(unit, value.object_id)
            if existing is None:
                await ownership.attach_submitted_run(
                    value.ref, unit_of_work=unit
                )
            else:
                existing.require_same_bytes(value)
            await ownership.release_run(record.artifact_id, unit_of_work=unit)
            deletion = await ownership.claim_delete(
                value.object_id,
                unit_of_work=unit,
                grace_seconds=(
                    86400
                    if existing is not None and existing.was_staged
                    else 0
                ),
            )
    else:
        async with ownership.transaction() as memory:
            existing = memory.objects.get(value.object_id)
            if existing is None:
                memory.attach_submitted_run(value.ref)
            else:
                existing.require_same_bytes(value)
            memory.release_run(record.artifact_id)
            deletion = memory.claim_delete(
                value.object_id,
                grace_seconds=(
                    86400
                    if existing is not None and existing.was_staged
                    else 0
                ),
            )
    if deletion is not None:
        await delete_owned_bytes(
            deletion, {record.ref.store: backend}, ownership
        )


async def retry_artifact_cleanup(
    ownership: PgsqlArtifactOwnership | MemoryArtifactOwnership,
    stores: Mapping[str, ArtifactStore],
    *,
    limit: int = 50,
    orphan_grace_seconds: int = 86400,
) -> int:
    """Retry committed fences and claim settled unowned bytes in a batch.

    Live staging owners remain ineligible regardless of age. A failed
    backend deletion retains its committed fence for the next invocation.
    """
    assert type(limit) is int and 1 <= limit <= 200
    assert (
        type(orphan_grace_seconds) is int
        and 0 <= orphan_grace_seconds <= 2592000
    )
    pending = await ownership.pending_deletions(limit=limit)
    remaining = limit - len(pending)
    claimed = (
        await ownership.claim_unowned(
            limit=remaining, grace_seconds=orphan_grace_seconds
        )
        if remaining
        else ()
    )
    for deletion in pending + claimed:
        await delete_owned_bytes(deletion, stores, ownership)
    return len(pending) + len(claimed)
