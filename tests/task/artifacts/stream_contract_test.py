from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from json import loads
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    ArtifactStore,
    ArtifactStoreConflictError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    EncryptedPrivacyValue,
    TaskArtifactRef,
    TaskArtifactStat,
    TaskFeature,
    TaskKeyPurpose,
    bounded_artifact_reader,
    freeze_snapshot_metadata,
    read_artifact_stream_bytes,
)
from avalan.task.artifacts import (
    LocalArtifactStore,
    PgsqlArtifactByteStoragePolicy,
    PgsqlArtifactStore,
)


class FakeObjectArtifactStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        assert isinstance(content, bytes)
        return await self.put_stream(
            BytesIO(content),
            artifact_id=artifact_id,
            media_type=media_type,
            metadata=metadata,
            max_bytes=len(content),
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
        if max_bytes is None:
            raise ArtifactStorePolicyError(
                "artifact stream requires maximum bytes"
            )
        new_artifact_id = artifact_id or f"artifact-{len(self.objects) + 1}"
        storage_key = f"objects/{new_artifact_id}"
        if storage_key in self.objects:
            raise ArtifactStoreConflictError("artifact already exists")
        content = read_artifact_stream_bytes(
            stream,
            max_bytes=max_bytes,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
        self.objects[storage_key] = content
        return TaskArtifactRef(
            artifact_id=new_artifact_id,
            store="object",
            storage_key=storage_key,
            media_type=media_type,
            size_bytes=len(content),
            sha256=sha256(content).hexdigest(),
            metadata=freeze_snapshot_metadata(metadata),
        )

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        return await self.open_stream(ref)

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BinaryIO:
        content = self._content(ref)
        return bounded_artifact_reader(BytesIO(content), max_bytes=max_bytes)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        self._content(ref)
        assert ref.size_bytes is not None
        assert ref.sha256 is not None
        return TaskArtifactStat(
            ref=ref,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        self._content(ref)
        del self.objects[ref.storage_key]

    def _content(self, ref: TaskArtifactRef) -> bytes:
        if ref.store != "object" or ref.storage_key not in self.objects:
            raise ArtifactStoreNotFoundError("artifact was not found")
        return self.objects[ref.storage_key]


class ContractCipher:
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        _ = purpose
        return EncryptedPrivacyValue(
            ciphertext=b"encrypted:" + value,
            key_id=key_id or "enc-v1",
            algorithm="test-aead",
            metadata=context,
        )

    def decrypt(
        self,
        value: EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        _ = purpose, context
        return value.ciphertext.removeprefix(b"encrypted:")


class ContractDatabase:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    def connection(self) -> "ContractConnectionContext":
        return ContractConnectionContext(self)


class ContractConnectionContext:
    def __init__(self, database: ContractDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "ContractConnection":
        return ContractConnection(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ContractConnection:
    def __init__(self, database: ContractDatabase) -> None:
        self.database = database

    def cursor(self) -> "ContractCursorContext":
        return ContractCursorContext(self.database)

    def transaction(self) -> "ContractTransactionContext":
        return ContractTransactionContext(self.database)


class ContractTransactionContext:
    def __init__(self, database: ContractDatabase) -> None:
        self.database = database
        self.snapshot: dict[str, dict[str, object]] = {}

    async def __aenter__(self) -> "ContractTransactionContext":
        self.snapshot = {
            key: dict(value) for key, value in self.database.rows.items()
        }
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        if exc_type is not None:
            self.database.rows = self.snapshot
        return False


class ContractCursorContext:
    def __init__(self, database: ContractDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "ContractCursor":
        return ContractCursor(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ContractCursor:
    def __init__(self, database: ContractDatabase) -> None:
        self.database = database
        self.row: Mapping[str, object] | None = None

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        assert parameters is not None
        if "INSERT INTO" in query:
            self._insert(parameters)
        elif "SELECT" in query:
            self.row = self.database.rows.get(cast(str, parameters[0]))
        elif "DELETE FROM" in query:
            storage_key = cast(str, parameters[0])
            self.row = (
                {"storage_key": storage_key}
                if self.database.rows.pop(storage_key, None) is not None
                else None
            )
        else:
            raise AssertionError("unexpected query")

    async def fetchone(self) -> Mapping[str, object] | None:
        return self.row

    def _insert(self, parameters: tuple[object, ...]) -> None:
        storage_key = cast(str, parameters[1])
        if storage_key in self.database.rows:
            self.row = None
            return
        self.database.rows[storage_key] = {
            "artifact_id": parameters[0],
            "storage_key": storage_key,
            "media_type": parameters[2],
            "size_bytes": parameters[3],
            "sha256": parameters[4],
            "ciphertext": parameters[5],
            "encryption_key_id": parameters[6],
            "encryption_algorithm": parameters[7],
            "encryption_metadata": loads(cast(str, parameters[8])),
            "retention_days": parameters[9],
            "retention_deadline_at": parameters[10],
            "metadata": loads(cast(str, parameters[11])),
        }
        self.row = {"storage_key": storage_key}


class ArtifactStoreStreamContractTest(IsolatedAsyncioTestCase):
    async def test_local_store_satisfies_stream_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            await self._assert_stream_contract(
                LocalArtifactStore(tmp, raw_storage_allowed=True)
            )

    async def test_pgsql_store_satisfies_stream_contract(self) -> None:
        store = PgsqlArtifactStore(
            ContractDatabase(),
            cipher=ContractCipher(),
            policy=PgsqlArtifactByteStoragePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                max_bytes=64,
                enabled_features=(
                    TaskFeature.POSTGRESQL,
                    TaskFeature.RAW_STORAGE,
                ),
            ),
        )

        await self._assert_stream_contract(store)

    async def test_object_store_satisfies_stream_contract(self) -> None:
        await self._assert_stream_contract(FakeObjectArtifactStore())

    async def _assert_stream_contract(self, store: ArtifactStore) -> None:
        content = b"private contract bytes"
        expected_sha256 = sha256(content).hexdigest()
        ref = await store.put_stream(
            BytesIO(content),
            artifact_id="artifact-1",
            media_type="text/plain",
            metadata={"label": "sanitized"},
            max_bytes=64,
            expected_size_bytes=len(content),
            expected_sha256=expected_sha256,
        )
        stat = await store.stat(ref)
        reader = await store.open_stream(ref, max_bytes=64)
        try:
            read_content = reader.read()
        finally:
            reader.close()
        bounded = await store.open_stream(ref, max_bytes=3)
        try:
            with self.assertRaises(ArtifactStorePolicyError):
                bounded.read()
        finally:
            bounded.close()

        self.assertEqual(read_content, content)
        self.assertEqual(stat.size_bytes, len(content))
        self.assertEqual(stat.sha256, expected_sha256)
        self.assertEqual(ref.metadata["label"], "sanitized")
        self.assertNotIn("private", str(ref.summary()))
        with self.assertRaises(ArtifactStoreConflictError):
            await store.put_stream(
                BytesIO(b"replacement"),
                artifact_id=ref.artifact_id,
                max_bytes=64,
            )
        await store.delete(ref)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.stat(ref)


if __name__ == "__main__":
    main()
