from datetime import UTC, datetime
from io import BytesIO
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pytest import raises
from task.artifacts.object_store_test import FakeObjectClient, XorObjectCipher

from avalan.task.artifact import (
    ArtifactStoreConflictError,
    ArtifactStorePolicyError,
    TaskArtifactStat,
)
from avalan.task.artifact_codec import _plain
from avalan.task.artifact_ownership import ArtifactStagingOwner
from avalan.task.artifact_staging import StagedArtifactStore
from avalan.task.artifacts.object_store import (
    ObjectArtifactStore,
    ObjectArtifactStorePolicy,
)
from avalan.task.artifacts.ownership_memory import MemoryArtifactOwnership


class ArtifactStagingTest(IsolatedAsyncioTestCase):
    async def test_bounded_writes_delegate_reads_but_reject_direct_cleanup(
        self,
    ) -> None:
        backend = ObjectArtifactStore(
            FakeObjectClient(),
            cipher=XorObjectCipher(),
            policy=ObjectArtifactStorePolicy(
                raw_storage_allowed=True, retention_days=1
            ),
        )
        ownership = MemoryArtifactOwnership()
        token = str(uuid4())
        wrapper = StagedArtifactStore(
            backend,
            ownership,
            ArtifactStagingOwner(
                staging_id=token, owner_scope="owner", recovery_id=token
            ),
        )
        with raises(ArtifactStorePolicyError):
            await wrapper.put_stream(BytesIO(b"input"))
        backend = ObjectArtifactStore(
            FakeObjectClient(),
            cipher=XorObjectCipher(),
            policy=ObjectArtifactStorePolicy(
                raw_storage_allowed=True, retention_days=1, max_bytes=4096
            ),
        )
        wrapper = StagedArtifactStore(backend, ownership, wrapper.owner)
        ref = await wrapper.put(b"input", artifact_id="physical")
        with await wrapper.open(ref) as reader:
            assert reader.read() == b"input"
        with await wrapper.open_stream(ref, max_bytes=5) as reader:
            assert reader.read() == b"input"
        assert (await wrapper.stat(ref)).size_bytes == 5
        with raises(ArtifactStoreConflictError):
            await wrapper.delete(ref)
        with patch.object(
            backend,
            "stat",
            AsyncMock(
                return_value=TaskArtifactStat(
                    ref=ref,
                    sha256="b" * 64,
                    size_bytes=5,
                )
            ),
        ):
            with raises(ArtifactStoreConflictError):
                await wrapper.put(b"input", artifact_id="other-physical")
        assert len(wrapper.objects) == 2 and ownership._staging


def test_artifact_metadata_lists_keep_json_shape() -> None:
    assert (
        _plain(datetime(2026, 1, 1, tzinfo=UTC)) == "2026-01-01T00:00:00+00:00"
    )
    assert _plain({"items": [1, {"nested": (2, 3)}]}) == {
        "items": [1, {"nested": [2, 3]}]
    }
