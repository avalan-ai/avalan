from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from json import loads
from os import environ
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch
from uuid import UUID, uuid4

from pytest import importorskip

from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
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
from avalan.task.artifacts import pgsql as pgsql_artifacts
from avalan.task.stores import PgsqlTaskMigrationSettings, task_pgsql_upgrade


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


class Uuid7Module:
    def uuid7(self) -> UUID:
        return UUID("018f8a92-6400-7000-8000-000000000001")


class UuidWithoutUuid7Module:
    pass


class PgsqlArtifactIdTest(IsolatedAsyncioTestCase):
    def test_uuid_id_prefers_uuid7_when_available(self) -> None:
        with patch.object(pgsql_artifacts, "_UUID_MODULE", Uuid7Module()):
            self.assertEqual(
                pgsql_artifacts._uuid_id(),
                "018f8a92640070008000000000000001",
            )

    def test_uuid_id_falls_back_to_uuid4(self) -> None:
        with (
            patch.object(
                pgsql_artifacts,
                "_UUID_MODULE",
                UuidWithoutUuid7Module(),
            ),
            patch.object(
                pgsql_artifacts,
                "uuid4",
                return_value=UUID("00000000-0000-4000-8000-000000000001"),
            ),
        ):
            self.assertEqual(
                pgsql_artifacts._uuid_id(),
                "00000000000040008000000000000001",
            )


class RecordingReader:
    def __init__(self) -> None:
        self.read_called = False

    def read(self, _size: int = -1) -> bytes:
        self.read_called = True
        return b"private bytes"


class FakeDatabase:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.invalid_row = False
        self.fail_after_insert = False
        self.last_parameters: tuple[object, ...] | None = None
        self.selects: list[str] = []
        self.open_count = 0
        self.close_count = 0

    def connection(self) -> "FakeConnectionContext":
        return FakeConnectionContext(self)

    async def open(self) -> None:
        self.open_count += 1

    async def aclose(self) -> None:
        self.close_count += 1


class FakeCloseOnlyDatabase:
    def __init__(self) -> None:
        self.close_count = 0

    def connection(self) -> "FakeConnectionContext":
        raise AssertionError("connection should not be used")

    async def close(self) -> None:
        self.close_count += 1


class FakeConnectionOnlyDatabase:
    def connection(self) -> "FakeConnectionContext":
        raise AssertionError("connection should not be used")


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

    def transaction(self) -> "FakeTransactionContext":
        return FakeTransactionContext(self.database)


class FakeTransactionContext:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.snapshot: dict[str, dict[str, object]] = {}

    async def __aenter__(self) -> "FakeTransactionContext":
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
            self._select(query, parameters)
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
            "retention_deadline_at": parameters[10],
            "metadata": loads(cast(str, parameters[11])),
        }
        self.database.rows[storage_key] = row
        if self.database.fail_after_insert:
            raise RuntimeError("raw database payload with secret")
        self.row = {"storage_key": storage_key}

    def _select(self, query: str, parameters: tuple[object, ...]) -> None:
        self.database.selects.append(query)
        if self.database.invalid_row:
            self.row = ["invalid-row"]
            return
        row = self.database.rows.get(cast(str, parameters[0]))
        if row is None or '"ciphertext"' in query:
            self.row = row
            return
        self.row = {
            "artifact_id": row["artifact_id"],
            "storage_key": row["storage_key"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }


class PgsqlArtifactStoreTest(IsolatedAsyncioTestCase):
    async def test_context_manager_opens_and_closes_database(self) -> None:
        database = FakeDatabase()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )

        async with store as opened:
            self.assertIs(opened, store)

        self.assertEqual(database.open_count, 1)
        self.assertEqual(database.close_count, 1)

    async def test_lifecycle_methods_tolerate_minimal_databases(self) -> None:
        connection_only_store = PgsqlArtifactStore(
            FakeConnectionOnlyDatabase(),
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )
        close_only_database = FakeCloseOnlyDatabase()
        close_only_store = PgsqlArtifactStore(
            close_only_database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
        )

        await connection_only_store.open_store()
        await close_only_store.aclose()

        self.assertEqual(close_only_database.close_count, 1)

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
        self.assertFalse(
            any('"ciphertext"' in query for query in database.selects[-1:]),
        )
        self.assertEqual(ref.metadata["retention_days"], 7)
        self.assertIsInstance(row["retention_deadline_at"], datetime)
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
            await store.open(ref)
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

    async def test_put_stream_buffers_under_explicit_cap(self) -> None:
        database = FakeDatabase()
        cipher = ReversibleCipher()
        store = PgsqlArtifactStore(
            database,
            cipher=cipher,
            policy=PgsqlArtifactByteStoragePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                max_bytes=20,
                enabled_features=(
                    TaskFeature.POSTGRESQL,
                    TaskFeature.RAW_STORAGE,
                ),
            ),
            id_factory=lambda: "artifact-1",
        )
        content = b"private bytes"
        expected_sha256 = sha256(content).hexdigest()

        ref = await store.put_stream(
            BytesIO(content),
            media_type="text/plain",
            metadata={"label": "sanitized"},
            expected_size_bytes=len(content),
            expected_sha256=expected_sha256,
        )
        reader = await store.open_stream(ref, max_bytes=len(content))
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
        self.assertEqual(ref.size_bytes, len(content))
        self.assertEqual(ref.sha256, expected_sha256)
        self.assertEqual(ref.metadata["retention_days"], 7)
        self.assertEqual(cipher.encrypt_context["artifact_id"], "artifact-1")

    async def test_put_stream_enforces_configured_cap_and_skips_invalid_writes(
        self,
    ) -> None:
        database = FakeDatabase()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=PgsqlArtifactByteStoragePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                max_bytes=8,
                enabled_features=(
                    TaskFeature.POSTGRESQL,
                    TaskFeature.RAW_STORAGE,
                ),
            ),
        )

        with self.assertRaises(ArtifactStorePolicyError):
            await store.put_stream(
                BytesIO(b"private bytes"),
                artifact_id="artifact-1",
                max_bytes=3,
            )
        with self.assertRaises(ArtifactStoreError):
            await store.put_stream(
                BytesIO(b"private bytes"),
                artifact_id="artifact-2",
                max_bytes=20,
                expected_size_bytes=99,
            )
        with self.assertRaises(ArtifactStorePolicyError):
            await store.put_stream(
                BytesIO(b"private bytes"),
                artifact_id="artifact-3",
                max_bytes=99,
                expected_size_bytes=13,
            )
        with self.assertRaises(ArtifactStorePolicyError):
            await store.put(
                b"private bytes",
                artifact_id="artifact-4",
            )

        self.assertEqual(database.rows, {})

    async def test_configured_cap_is_required_before_stream_is_read(
        self,
    ) -> None:
        database = FakeDatabase()
        reader = RecordingReader()
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=PgsqlArtifactByteStoragePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                enabled_features=(
                    TaskFeature.POSTGRESQL,
                    TaskFeature.RAW_STORAGE,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ArtifactStorePolicyError,
            "requires maximum bytes",
        ):
            await store.put_stream(reader, max_bytes=20)  # type: ignore[arg-type]

        self.assertFalse(reader.read_called)
        self.assertEqual(database.rows, {})

    async def test_real_postgresql_byte_backend_when_configured(
        self,
    ) -> None:
        dsn = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("psycopg")
        importorskip("psycopg_pool")
        importorskip("sqlalchemy")
        schema = f"avalan_task_artifact_test_{uuid4().hex}"
        task_pgsql_upgrade(PgsqlTaskMigrationSettings(url=dsn, schema=schema))
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(dsn=_psycopg_dsn(dsn), schema=schema)
        )
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
            id_factory=lambda: "artifact-real-1",
        )

        try:
            async with store:
                ref = await store.put(
                    b"private bytes",
                    media_type="text/plain",
                )
                reader = await store.open(ref)
                try:
                    content = reader.read()
                finally:
                    reader.close()
                stat = await store.stat(ref)
                await store.delete(ref)

                self.assertEqual(content, b"private bytes")
                self.assertEqual(stat.size_bytes, len(content))
                with self.assertRaises(ArtifactStoreNotFoundError):
                    await store.open(ref)
        finally:
            await _drop_schema(dsn, schema)

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

    async def test_failed_write_rolls_back_and_reports_safe_error(
        self,
    ) -> None:
        database = FakeDatabase()
        database.fail_after_insert = True
        store = PgsqlArtifactStore(
            database,
            cipher=ReversibleCipher(),
            policy=self._enabled_policy(),
            id_factory=lambda: "artifact-1",
        )

        with self.assertRaises(ArtifactStoreError) as caught:
            await store.put(b"private bytes")

        self.assertEqual(database.rows, {})
        self.assertNotIn("raw database payload", str(caught.exception))
        self.assertNotIn("private bytes", str(caught.exception))

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
            PgsqlArtifactByteStoragePolicy(max_bytes=-1)
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
        with self.assertRaises(AssertionError):
            await store.put_stream(object(), max_bytes=1)  # type: ignore[arg-type]
        reader = RecordingReader()
        with self.assertRaises(AssertionError):
            await store.put_stream(
                reader,  # type: ignore[arg-type]
                metadata={"bad": object()},
                max_bytes=20,
            )
        self.assertFalse(reader.read_called)

    @staticmethod
    def _enabled_policy() -> PgsqlArtifactByteStoragePolicy:
        return PgsqlArtifactByteStoragePolicy(
            raw_storage_allowed=True,
            retention_days=7,
            max_bytes=64,
            enabled_features=(
                TaskFeature.POSTGRESQL,
                TaskFeature.RAW_STORAGE,
            ),
        )

    @staticmethod
    def _missing_module(module: str) -> object | None:
        return None


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=_psycopg_dsn(dsn)))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


def _psycopg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return "postgresql://" + dsn.removeprefix("postgresql+psycopg://")
    return dsn


if __name__ == "__main__":
    main()
