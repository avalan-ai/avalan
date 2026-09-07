"""Reserve recoverable physical identities before external artifact writes."""

from .artifact import (
    ArtifactStoreConflictError,
    TaskArtifactRef,
    TaskArtifactStat,
    read_artifact_stream_bytes,
)
from .artifact_ownership import ArtifactObject, ArtifactStagingOwner
from .artifacts.object_store import (
    ObjectArtifactStore,
)
from .artifacts.object_store import (
    _storage_key as object_storage_key,
)
from .artifacts.ownership_memory import MemoryArtifactOwnership
from .artifacts.ownership_pgsql import PgsqlArtifactOwnership
from .artifacts.pgsql import (
    PgsqlArtifactStore,
)
from .artifacts.pgsql import (
    _storage_key as pgsql_storage_key,
)

from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from typing import BinaryIO
from uuid import uuid4


class StagedArtifactStore:
    """Reserve ownership before writes without claiming backend completion.

    On any interruption, objects contains every attempted physical key.
    Unknown external acknowledgments retain those owners. No automatic
    cleanup infers rollback from a backend's temporary absence.
    """

    def __init__(
        self,
        backend: PgsqlArtifactStore | ObjectArtifactStore,
        ownership: PgsqlArtifactOwnership | MemoryArtifactOwnership,
        owner: ArtifactStagingOwner,
    ) -> None:
        assert isinstance(backend, PgsqlArtifactStore | ObjectArtifactStore)
        self.backend = backend
        self.ownership = ownership
        self.owner = owner
        self.objects: list[ArtifactObject] = []

    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        return await self.put_stream(
            BytesIO(content),
            artifact_id=artifact_id,
            media_type=media_type,
            metadata=metadata,
            expected_size_bytes=len(content),
        )

    async def put_stream(
        self,
        stream: BinaryIO,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
        max_bytes: int | None = None,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> TaskArtifactRef:
        self.backend._require_policy()
        bounds = tuple(
            value
            for value in (
                max_bytes,
                expected_size_bytes,
                self.backend._policy.max_bytes,
            )
            if value is not None
        )
        content = read_artifact_stream_bytes(
            stream,
            max_bytes=min(bounds),
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
        identity = artifact_id or str(uuid4())
        key = (
            pgsql_storage_key(identity)
            if isinstance(self.backend, PgsqlArtifactStore)
            else object_storage_key(identity)
        )
        ref = TaskArtifactRef(
            artifact_id=identity,
            store=self.backend._store_name,
            storage_key=key,
            media_type=media_type,
            size_bytes=len(content),
            sha256=sha256(content).hexdigest(),
        )
        value = ArtifactObject.from_ref(ref)
        self.objects.append(value)
        if isinstance(self.ownership, PgsqlArtifactOwnership):
            async with self.ownership.transaction() as unit:
                await self.ownership.stage(
                    value, self.owner, unit_of_work=unit
                )
        else:
            async with self.ownership.transaction() as memory:
                memory.stage(value, self.owner)
        written = await self.backend.put_stream(
            BytesIO(content),
            artifact_id=identity,
            media_type=media_type,
            metadata=metadata,
            max_bytes=len(content),
            expected_size_bytes=len(content),
            expected_sha256=ref.sha256,
        )
        value.require_same_bytes(ArtifactObject.from_ref(written))
        stat = await self.backend.stat(written)
        if stat.sha256 != ref.sha256 or stat.size_bytes != ref.size_bytes:
            raise ArtifactStoreConflictError(
                "staged artifact readback differs"
            )
        return written

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        return await self.backend.open(ref)

    async def open_stream(
        self, ref: TaskArtifactRef, *, max_bytes: int | None = None
    ) -> BinaryIO:
        return await self.backend.open_stream(ref, max_bytes=max_bytes)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return await self.backend.stat(ref)

    async def delete(self, ref: TaskArtifactRef) -> None:
        raise ArtifactStoreConflictError(
            "staged artifact requires settled ownership cleanup"
        )
