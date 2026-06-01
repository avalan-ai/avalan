from ...pgsql import (
    PgsqlDatabase,
    PgsqlOperationError,
    PgsqlUnitOfWork,
    run_pgsql_transaction,
)
from ...types import assert_non_empty_string as _assert_non_empty_string
from ..artifact import (
    ArtifactStoreConflictError,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    TaskArtifactRef,
    TaskArtifactStat,
)
from ..feature_gate import ModuleFinder, TaskFeature, require_features
from ..privacy import ENCRYPTED_MARKER, EncryptedPrivacyValue, TaskKeyPurpose
from ..store import freeze_snapshot_metadata

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from importlib.util import find_spec
from inspect import isawaitable
from io import BytesIO
from json import dumps, loads
from re import fullmatch
from typing import BinaryIO, Protocol, cast
from uuid import uuid4


class PgsqlArtifactCipher(Protocol):
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue: ...

    def decrypt(
        self,
        value: EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        context: Mapping[str, str] | None = None,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlArtifactByteStoragePolicy:
    raw_storage_allowed: bool = False
    retention_days: int | None = None
    enabled_features: tuple[TaskFeature, ...] = ()
    module_finder: ModuleFinder = find_spec

    def __post_init__(self) -> None:
        assert isinstance(self.raw_storage_allowed, bool)
        if self.retention_days is not None:
            assert isinstance(self.retention_days, int)
            assert not isinstance(self.retention_days, bool)
            assert self.retention_days > 0
        assert isinstance(self.enabled_features, tuple)
        for feature in self.enabled_features:
            assert isinstance(feature, TaskFeature)
        assert callable(self.module_finder)


class PgsqlArtifactStore:
    def __init__(
        self,
        database: PgsqlDatabase,
        *,
        cipher: PgsqlArtifactCipher,
        policy: PgsqlArtifactByteStoragePolicy,
        store_name: str = "pgsql",
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        assert hasattr(database, "connection")
        assert hasattr(cipher, "encrypt")
        assert hasattr(cipher, "decrypt")
        assert isinstance(policy, PgsqlArtifactByteStoragePolicy)
        _assert_non_empty_string(store_name, "store_name")
        self._database = database
        self._cipher = cipher
        self._policy = policy
        self._store_name = store_name
        self._id_factory = id_factory or _uuid_id

    async def open_store(self) -> None:
        open_database = getattr(self._database, "open", None)
        if open_database is None:
            return
        result = open_database()
        if isawaitable(result):
            await result

    async def aclose(self) -> None:
        aclose = getattr(self._database, "aclose", None)
        if aclose is not None:
            result = aclose()
        else:
            close = getattr(self._database, "close", None)
            result = close() if close is not None else None
        if isawaitable(result):
            await result

    async def __aenter__(self) -> "PgsqlArtifactStore":
        await self.open_store()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        await self.aclose()
        return None

    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        assert isinstance(content, bytes), "content must be bytes"
        self._require_policy()
        if media_type is not None:
            _assert_non_empty_string(media_type, "media_type")
        new_artifact_id = artifact_id or self._new_id()
        _assert_artifact_id(new_artifact_id)
        storage_key = _storage_key(new_artifact_id)
        digest = sha256(content).hexdigest()
        context = _encryption_context(
            artifact_id=new_artifact_id,
            store=self._store_name,
        )
        encrypted = self._cipher.encrypt(
            content,
            purpose=TaskKeyPurpose.ARTIFACT_CONTENT,
            context=context,
        )
        retention_deadline_at = _retention_deadline_at(
            self._policy.retention_days
        )
        ref_metadata = freeze_snapshot_metadata(
            {
                **dict(metadata or {}),
                "encryption": {
                    "algorithm": encrypted.algorithm,
                    "key_id": encrypted.key_id,
                    "privacy": ENCRYPTED_MARKER,
                },
                "retention_days": self._policy.retention_days,
            }
        )
        ref = TaskArtifactRef(
            artifact_id=new_artifact_id,
            store=self._store_name,
            storage_key=storage_key,
            media_type=media_type,
            size_bytes=len(content),
            sha256=digest,
            metadata=ref_metadata,
        )
        row = await self._execute_returning(
            _INSERT_ARTIFACT_BYTES_SQL,
            (
                new_artifact_id,
                storage_key,
                media_type,
                len(content),
                digest,
                encrypted.ciphertext,
                encrypted.key_id,
                encrypted.algorithm,
                dumps(dict(encrypted.metadata or {}), sort_keys=True),
                self._policy.retention_days,
                retention_deadline_at,
                dumps(dict(metadata or {}), sort_keys=True),
            ),
        )
        if row is None:
            raise ArtifactStoreConflictError("artifact already exists")
        return ref

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        record = await self._fetch_record(ref)
        encrypted = EncryptedPrivacyValue(
            ciphertext=record.ciphertext,
            key_id=record.encryption_key_id,
            algorithm=record.encryption_algorithm,
            metadata=record.encryption_metadata,
        )
        content = self._cipher.decrypt(
            encrypted,
            purpose=TaskKeyPurpose.ARTIFACT_CONTENT,
            context=_encryption_context(
                artifact_id=ref.artifact_id,
                store=self._store_name,
            ),
        )
        if sha256(content).hexdigest() != record.sha256:
            raise ArtifactStoreError("artifact content digest mismatch")
        return BytesIO(content)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        record = await self._fetch_record(ref)
        return TaskArtifactStat(
            ref=ref,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        self._assert_ref(ref)
        row = await self._execute_returning(
            _DELETE_ARTIFACT_BYTES_SQL,
            (ref.storage_key,),
        )
        if row is None:
            raise ArtifactStoreNotFoundError("artifact was not found")

    async def _fetch_record(
        self,
        ref: TaskArtifactRef,
    ) -> "_PgsqlArtifactByteRecord":
        self._assert_ref(ref)
        row = await self._execute_returning(
            _SELECT_ARTIFACT_BYTES_SQL,
            (ref.storage_key,),
        )
        if row is None:
            raise ArtifactStoreNotFoundError("artifact was not found")
        return _record_from_row(row)

    async def _execute_returning(
        self,
        query: str,
        parameters: tuple[object, ...],
    ) -> Mapping[str, object] | None:
        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(query, parameters)
            return await unit.cursor.fetchone()

        try:
            row = await run_pgsql_transaction(
                self._database,
                operation="task_artifact_bytes",
                callback=execute,
            )
        except PgsqlOperationError as error:
            raise ArtifactStoreError(str(error)) from None
        if row is not None and not isinstance(row, Mapping):
            raise ArtifactStoreError("artifact query returned invalid row")
        return row

    def _assert_ref(self, ref: TaskArtifactRef) -> None:
        assert isinstance(ref, TaskArtifactRef)
        if ref.store != self._store_name:
            raise ArtifactStoreNotFoundError("artifact store does not match")
        _assert_storage_key(ref.storage_key)
        if ref.storage_key != _storage_key(ref.artifact_id):
            raise ArtifactStoreNotFoundError("artifact storage key mismatch")

    def _require_policy(self) -> None:
        diagnostics = require_features(
            (TaskFeature.POSTGRESQL, TaskFeature.RAW_STORAGE),
            enabled_features=self._policy.enabled_features,
            module_finder=self._policy.module_finder,
        )
        if diagnostics:
            diagnostic = diagnostics[0]
            raise ArtifactStorePolicyError(
                f"{diagnostic.code}: {diagnostic.message}"
            )
        if not self._policy.raw_storage_allowed:
            raise ArtifactStorePolicyError("artifact byte storage is disabled")
        if self._policy.retention_days is None:
            raise ArtifactStorePolicyError(
                "artifact byte storage requires retention"
            )

    def _new_id(self) -> str:
        value = self._id_factory()
        _assert_artifact_id(value)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class _PgsqlArtifactByteRecord:
    artifact_id: str
    storage_key: str
    media_type: str | None
    size_bytes: int
    sha256: str
    ciphertext: bytes
    encryption_key_id: str
    encryption_algorithm: str
    encryption_metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        _assert_artifact_id(self.artifact_id)
        _assert_storage_key(self.storage_key)
        if self.media_type is not None:
            _assert_non_empty_string(self.media_type, "media_type")
        assert isinstance(self.size_bytes, int)
        assert not isinstance(self.size_bytes, bool)
        assert self.size_bytes >= 0
        _assert_sha256(self.sha256)
        assert isinstance(self.ciphertext, bytes) and self.ciphertext
        _assert_non_empty_string(self.encryption_key_id, "encryption_key_id")
        _assert_non_empty_string(
            self.encryption_algorithm,
            "encryption_algorithm",
        )
        assert isinstance(self.encryption_metadata, Mapping)
        for key, value in self.encryption_metadata.items():
            _assert_non_empty_string(key, "encryption_metadata key")
            assert isinstance(value, str)


def _record_from_row(row: Mapping[str, object]) -> _PgsqlArtifactByteRecord:
    return _PgsqlArtifactByteRecord(
        artifact_id=_row_string(row, "artifact_id"),
        storage_key=_row_string(row, "storage_key"),
        media_type=_row_optional_string(row, "media_type"),
        size_bytes=_row_int(row, "size_bytes"),
        sha256=_row_string(row, "sha256"),
        ciphertext=_row_bytes(row, "ciphertext"),
        encryption_key_id=_row_string(row, "encryption_key_id"),
        encryption_algorithm=_row_string(row, "encryption_algorithm"),
        encryption_metadata=_row_metadata(row, "encryption_metadata"),
    )


def _row_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    _assert_non_empty_string(value, key)
    return cast(str, value)


def _row_optional_string(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    _assert_non_empty_string(value, key)
    return cast(str, value)


def _row_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    assert isinstance(value, int)
    assert not isinstance(value, bool)
    assert value >= 0
    return value


def _row_bytes(row: Mapping[str, object], key: str) -> bytes:
    value = row.get(key)
    assert isinstance(value, bytes) and value
    return value


def _row_metadata(row: Mapping[str, object], key: str) -> Mapping[str, str]:
    value = row.get(key)
    if value is None:
        return {}
    if isinstance(value, str):
        decoded = loads(value)
        assert isinstance(decoded, Mapping)
        value = decoded
    assert isinstance(value, Mapping)
    metadata: dict[str, str] = {}
    for metadata_key, metadata_value in value.items():
        _assert_non_empty_string(metadata_key, "metadata key")
        assert isinstance(metadata_value, str)
        metadata[metadata_key] = metadata_value
    return metadata


def _storage_key(artifact_id: str) -> str:
    return f"artifact-bytes/{artifact_id}"


def _assert_storage_key(value: str) -> None:
    _assert_non_empty_string(value, "storage_key")
    assert value.startswith("artifact-bytes/")
    assert ".." not in value.split("/")


def _assert_artifact_id(value: str) -> None:
    _assert_non_empty_string(value, "artifact_id")
    assert fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        value,
    ), "artifact_id must be a stable token"


def _assert_sha256(value: str) -> None:
    _assert_non_empty_string(value, "sha256")
    assert fullmatch(r"[0-9a-f]{64}", value)


def _encryption_context(
    *,
    artifact_id: str,
    store: str,
) -> Mapping[str, str]:
    return {
        "artifact_id": artifact_id,
        "purpose": TaskKeyPurpose.ARTIFACT_CONTENT.value,
        "store": store,
    }


def _retention_deadline_at(retention_days: int | None) -> datetime:
    assert retention_days is not None
    return datetime.now(UTC) + timedelta(days=retention_days)


def _uuid_id() -> str:
    return uuid4().hex


_INSERT_ARTIFACT_BYTES_SQL = """
INSERT INTO "task_artifact_bytes"(
    "artifact_id",
    "storage_key",
    "media_type",
    "size_bytes",
    "sha256",
    "ciphertext",
    "encryption_key_id",
    "encryption_algorithm",
    "encryption_metadata",
    "retention_days",
    "retention_deadline_at",
    "metadata"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb
)
ON CONFLICT ("storage_key") DO NOTHING
RETURNING "storage_key"
"""

_SELECT_ARTIFACT_BYTES_SQL = """
SELECT
    "artifact_id",
    "storage_key",
    "media_type",
    "size_bytes",
    "sha256",
    "ciphertext",
    "encryption_key_id",
    "encryption_algorithm",
    "encryption_metadata"
FROM "task_artifact_bytes"
WHERE "storage_key" = %s
    AND "deleted_at" IS NULL
"""

_DELETE_ARTIFACT_BYTES_SQL = """
DELETE FROM "task_artifact_bytes"
WHERE "storage_key" = %s
RETURNING "storage_key"
"""
