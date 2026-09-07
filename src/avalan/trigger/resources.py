"""Release closed revision ownership under admission-compatible fences."""

from ..task.artifact_ownership import ArtifactRevisionOwner
from ..task.artifacts.ownership_memory import MemoryArtifactOwnership
from ..task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from .definition import integer
from .error import TriggerError, TriggerErrorCode
from .stores.memory import InMemoryTriggerStore
from .stores.pgsql import PgsqlTriggerStore, lock_identity


class TriggerResourceService:
    """Retain active, paused and uncertain inputs until closure is proven."""

    def __init__(
        self,
        store: InMemoryTriggerStore | PgsqlTriggerStore,
        ownership: MemoryArtifactOwnership | PgsqlArtifactOwnership,
    ) -> None:
        if isinstance(store, PgsqlTriggerStore):
            assert isinstance(ownership, PgsqlArtifactOwnership)
            assert store.database is ownership.database
        else:
            assert isinstance(ownership, MemoryArtifactOwnership)
            assert store._artifact_ownership is ownership
        self.store = store
        self.ownership = ownership

    async def release_closed_revision(self, name: str, revision: int) -> bool:
        """Release only closed input with no unresolved staged acquisitions.

        Run references remain independent. This method does not delete bytes;
        the ordinary ownership cleanup still enforces grace and tombstones.
        Retrying an uncertain acknowledgement is idempotent.
        """
        integer(revision, 1, 9223372036854775807, "revision")
        store, ownership = self.store, self.ownership
        if isinstance(store, PgsqlTriggerStore):
            assert isinstance(ownership, PgsqlArtifactOwnership)
            async with store.transaction() as unit:
                await lock_identity(unit, store.owner, name)
                current = await store._read(unit, name, lock=True)
                if current is None:
                    raise TriggerError(TriggerErrorCode.CONFLICT, "name")
                await unit.cursor.execute(
                    "SELECT closed_at FROM trigger_definitions WHERE"
                    " owner_scope_id = %s AND trigger_id = %s AND revision"
                    " = %s",
                    (store.owner.value, current.state.trigger_id, revision),
                )
                row = await unit.cursor.fetchone()
                if row is None or row["closed_at"] is None:
                    return False
                await unit.cursor.execute(
                    "SELECT object_id::text FROM trigger_input_artifacts WHERE"
                    " owner_scope_id = %s AND trigger_id = %s AND revision ="
                    " %s AND released_at IS NULL ORDER BY object_id",
                    (store.owner.value, current.state.trigger_id, revision),
                )
                rows = await unit.cursor.fetchall()
                for row in rows:
                    identity = row["object_id"]
                    if not isinstance(identity, str):
                        raise TriggerError(
                            TriggerErrorCode.UNSUPPORTED_VERSION,
                            "store.artifact",
                        )
                    await ownership._read(unit, identity)
                    await unit.cursor.execute(
                        "SELECT 1 AS pending FROM task_artifact_staging_owners"
                        " WHERE object_id = %s::uuid LIMIT 1",
                        (identity,),
                    )
                    if await unit.cursor.fetchone() is not None:
                        return False
                await unit.cursor.execute(
                    "UPDATE trigger_input_artifacts SET released_at ="
                    " clock_timestamp() WHERE owner_scope_id = %s AND"
                    " trigger_id = %s AND revision = %s AND released_at IS"
                    " NULL",
                    (store.owner.value, current.state.trigger_id, revision),
                )
            return True
        assert isinstance(ownership, MemoryArtifactOwnership)
        async with store._lock:
            current = store._require(name)
            key = (current.state.trigger_id, revision)
            if key not in store._closed:
                return False
            owner = ArtifactRevisionOwner(
                owner_scope=store.owner.value,
                resource_id=key[0],
                revision=revision,
            )
            async with ownership.transaction() as unit:
                identities = {
                    identity
                    for candidate, identity in unit.revisions
                    if candidate == owner
                }
                if any(key[1] in identities for key in unit.staging):
                    return False
                unit.release_revision(owner)
            return True
