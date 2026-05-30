from collections.abc import Mapping
from json import loads
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    ENCRYPTED_MARKER,
    ArtifactStoreConflictError,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    EncryptedPrivacyValue,
    TaskArtifactRef,
    TaskFeature,
    TaskKeyPurpose,
)
from avalan.task.artifacts import (
    PgsqlArtifactByteStoragePolicy,
    PgsqlArtifactStore,
)


class ReversibleCipher:
    def __init__(self) -> None:
        self.encrypt_context: Mapping[str, str] | None = None
        self.decrypt_context: Mapping[str, str] | None = None
        self.decrypted = False

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        self.encrypt_context = context
        self.encrypted_purpose = purpose
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
        self.decrypt_context = context
        self.decrypted = True
        self.decrypted_purpose = purpose
        if not value.ciphertext.startswith(b"encrypted:"):
            raise AssertionError("ciphertext was not encrypted")
        return value.ciphertext.removeprefix(b"encrypted:")


class TamperingCipher(ReversibleCipher):
    def decrypt(
        self,
        value: EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        return b"tampered"


class FakeDatabase:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.invalid_row = False
        self.last_parameters: tuple[object, ...] | None = None

    def connection(self) -> "FakeConnectionContext":
        return FakeConnectionContext(self)


class FakeConnectionContext:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeConnection":
        return FakeConnection(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def cursor(self) -> "FakeCursorContext":
        return FakeCursorContext(self.database)


class FakeCursorContext:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeCursor":
        return FakeCursor(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.row: object | None = None

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        assert parameters is not None
        self.database.last_parameters = parameters
        if "INSERT INTO" in query:
            self._insert(parameters)
        elif "SELECT" in query:
            self.row = (
                ["invalid-row"]
                if self.database.invalid_row
                else self.database.rows.get(cast(str, parameters[0]))
            )
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
        return cast(Mapping[str, object] | None, self.row)

    def _insert(self, parameters: tuple[object, ...]) -> None:
        storage_key = cast(str, parameters[1])
        if storage_key in self.database.rows:
            self.row = None
            return
        row = {
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
            "metadata": loads(cast(str, parameters[10])),
        }
        self.database.rows[storage_key] = row
        self.row = {"storage_key": storage_key}


class PgsqlArtifactStoreTest(IsolatedAsyncioTestCase):
    async def test_put_open_stat_and_delete_encrypted_bytes(self) -> None:
        database = FakeDatabase()
        cipher = ReversibleCipher()
        store = PgsqlArtifactStore(
            database,
            cipher=cipher,
            policy=self._enabled_policy(),
            id_factory=lambda: "artifact-1",
        )

        ref = await store.put(
            b"private bytes",
            media_type="text/plain",
            metadata={"label": "sanitized"},
        )
        row = database.rows[ref.storage_key]
        row["encryption_metadata"] = (
            '{"artifact_id":"artifact-1",'
            '"purpose":"artifact_content","store":"pgsql"}'
        )
        reader = await store.open(ref)
        try:
            content = reader.read()
        finally:
            reader.close()
        stat = await store.stat(ref)
        await store.delete(ref)

        self.assertEqual(content, b"private bytes")
        self.assertEqual(ref.artifact_id, "artifact-1")
        self.assertEqual(ref.store, "pgsql")
        self.assertEqual(ref.storage_key, "artifact-bytes/artifact-1")
        self.assertEqual(ref.media_type, "text/plain")
        self.assertEqual(ref.size_bytes, 13)
        self.assertEqual(stat.size_bytes, 13)
        self.assertEqual(ref.sha256, stat.sha256)
        self.assertEqual(ref.metadata["retention_days"], 7)
        encryption = cast(Mapping[str, object], ref.metadata["encryption"])
        self.assertEqual(encryption["privacy"], ENCRYPTED_MARKER)
        self.assertEqual(encryption["key_id"], "enc-v1")
        self.assertNotIn("ciphertext", encryption)
        self.assertEqual(
            database.rows.get("artifact-bytes/artifact-1"),
            None,
        )
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.stat(ref)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.delete(ref)
        self.assertTrue(cipher.decrypted)
        self.assertEqual(
            cipher.encrypt_context,
            {
                "artifact_id": "artifact-1",
                "purpose": "artifact_content",
                "store": "pgsql",
            },
        )

    async def test_raw_storage_requires_dependency_gate_and_enablement(
        self,
    ) -> None:
        missing_dependency_store = PgsqlArtifactStore(
            FakeDatabase(),
            cipher=ReversibleCipher(),
            policy=PgsqlArtifactByteStoragePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                enabled_features=(TaskFeature.RAW_STORAGE,),
                module_finder=self._missing_module,
            ),
        )
        disabled_store = PgsqlArtifactStore(
            FakeDatabase(),
            cipher=ReversibleCipher(),
            policy=PgsqlArtifactByteStoragePolicy(
                retention_days=7,
                enabled_features=(
                    TaskFeature.POSTGRESQL,
                    TaskFeature.RAW_STORAGE,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ArtifactStorePolicyError,
            "dependency.task_pgsql_missing",
        ):
            await missing_dependency_store.put(b"private bytes")
        with self.assertRaisesRegex(
            ArtifactStorePolicyError,
            "artifact byte storage is disabled",
        ):
            await disabled_store.put(b"private bytes")

    async def test_retention_is_required_before_bytes_are_written(
        self,
    ) -> None:
        database = FakeDatabase()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=PgsqlArtifactByteStoragePolicy(
                raw_storage_allowed=True,
                enabled_features=(
                    TaskFeature.POSTGRESQL,
                    TaskFeature.RAW_STORAGE,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ArtifactStorePolicyError,
            "requires retention",
        ):
            await store.put(b"private bytes")

        self.assertEqual(database.rows, {})

    async def test_duplicate_artifact_id_is_rejected_without_overwrite(
        self,
    ) -> None:
        database = FakeDatabase()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )
        ref = await store.put(b"first", artifact_id="artifact-1")

        with self.assertRaises(ArtifactStoreConflictError):
            await store.put(b"second", artifact_id=ref.artifact_id)

        reader = await store.open(ref)
        try:
            self.assertEqual(reader.read(), b"first")
        finally:
            reader.close()

    async def test_references_cannot_cross_store_or_escape(self) -> None:
        store = PgsqlArtifactStore(
            FakeDatabase(),
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )
        wrong_store = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="artifact-bytes/artifact-1",
        )
        escaped = TaskArtifactRef(
            artifact_id="artifact-1",
            store="pgsql",
            storage_key="artifact-bytes/../artifact-1",
        )
        mismatched = TaskArtifactRef(
            artifact_id="artifact-2",
            store="pgsql",
            storage_key="artifact-bytes/artifact-1",
        )

        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.open(wrong_store)
        with self.assertRaises(AssertionError):
            await store.open(escaped)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.open(mismatched)

    async def test_digest_mismatch_fails_closed(self) -> None:
        store = PgsqlArtifactStore(
            FakeDatabase(),
            cipher=TamperingCipher(),
            policy=self._enabled_policy(),
            id_factory=lambda: "artifact-1",
        )
        ref = await store.put(b"private bytes")

        with self.assertRaises(ArtifactStoreError):
            await store.open(ref)

    async def test_default_id_and_empty_encryption_metadata_round_trip(
        self,
    ) -> None:
        database = FakeDatabase()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )
        ref = await store.put(b"private bytes")
        database.rows[ref.storage_key]["encryption_metadata"] = None

        reader = await store.open(ref)
        try:
            content = reader.read()
        finally:
            reader.close()

        self.assertEqual(content, b"private bytes")
        self.assertEqual(len(ref.artifact_id), 32)

    async def test_invalid_query_row_fails_safely(self) -> None:
        database = FakeDatabase()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
            id_factory=lambda: "artifact-1",
        )
        ref = await store.put(b"private bytes")
        database.invalid_row = True

        with self.assertRaises(ArtifactStoreError):
            await store.stat(ref)

    async def test_invalid_backend_configuration_and_inputs_fail_fast(
        self,
    ) -> None:
        database = FakeDatabase()
        with self.assertRaises(AssertionError):
            PgsqlArtifactByteStoragePolicy(retention_days=0)
        with self.assertRaises(AssertionError):
            PgsqlArtifactStore(
                database,
                cipher=object(),  # type: ignore[arg-type]
                policy=self._enabled_policy(),
            )
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )
        with self.assertRaises(AssertionError):
            await store.put("not bytes")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await store.put(b"private bytes", artifact_id="../bad")
        with self.assertRaises(AssertionError):
            await store.put(b"private bytes", media_type="")

    @staticmethod
    def _enabled_policy() -> PgsqlArtifactByteStoragePolicy:
        return PgsqlArtifactByteStoragePolicy(
            raw_storage_allowed=True,
            retention_days=7,
            enabled_features=(
                TaskFeature.POSTGRESQL,
                TaskFeature.RAW_STORAGE,
            ),
        )

    @staticmethod
    def _missing_module(module: str) -> object | None:
        return None


if __name__ == "__main__":
    main()
