from ...types import assert_non_empty_string as _assert_non_empty_string
from ..artifact import (
    DEFAULT_ARTIFACT_STREAM_CHUNK_SIZE,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    TaskArtifactRef,
    TaskArtifactStat,
    bounded_artifact_reader,
)
from ..privacy import ENCRYPTED_MARKER, STORED_MARKER, TaskKeyPurpose
from ..store import TaskSnapshotMetadata, freeze_snapshot_metadata

from asyncio import CancelledError
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from re import fullmatch
from typing import BinaryIO, Protocol, cast
from uuid import uuid4


class ObjectArtifactClientTransientError(ArtifactStoreError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectArtifactEncryption:
    key_id: str
    algorithm: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.key_id, "key_id")
        _assert_non_empty_string(self.algorithm, "algorithm")
        assert isinstance(self.metadata, Mapping)
        for key, value in self.metadata.items():
            _assert_non_empty_string(key, "encryption metadata key")
            _assert_non_empty_string(value, "encryption metadata value")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectArtifactMultipartUpload:
    key: str
    upload_id: str

    def __post_init__(self) -> None:
        _assert_storage_key(self.key)
        _assert_non_empty_string(self.upload_id, "upload_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectArtifactPart:
    part_number: int
    etag: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _assert_positive_int(self.part_number, "part_number")
        _assert_non_empty_string(self.etag, "etag")
        _assert_non_negative_int(self.size_bytes, "size_bytes")
        _assert_sha256(self.sha256)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectArtifactHead:
    key: str
    size_bytes: int
    sha256: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_storage_key(self.key)
        _assert_non_negative_int(self.size_bytes, "size_bytes")
        _assert_sha256(self.sha256)
        assert isinstance(self.metadata, Mapping)
        for key, value in self.metadata.items():
            _assert_non_empty_string(key, "object metadata key")
            _assert_non_empty_string(value, "object metadata value")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectArtifactStorePolicy:
    raw_storage_allowed: bool = False
    retention_days: int | None = None
    max_bytes: int | None = None
    part_size: int = DEFAULT_ARTIFACT_STREAM_CHUNK_SIZE
    max_part_retries: int = 2
    encryption_required: bool = True
    signed_url_max_seconds: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.raw_storage_allowed, bool)
        _assert_optional_positive_int(
            self.retention_days,
            "retention_days",
        )
        _assert_optional_non_negative_int(self.max_bytes, "max_bytes")
        _assert_positive_int(self.part_size, "part_size")
        _assert_non_negative_int(self.max_part_retries, "max_part_retries")
        assert isinstance(self.encryption_required, bool)
        _assert_optional_positive_int(
            self.signed_url_max_seconds,
            "signed_url_max_seconds",
        )


class ObjectArtifactCipher(Protocol):
    def start_encryption(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> ObjectArtifactEncryption: ...

    def encrypt_chunk(
        self,
        value: bytes,
        *,
        encryption: ObjectArtifactEncryption,
        context: Mapping[str, str] | None = None,
    ) -> bytes: ...

    def decrypt_chunk(
        self,
        value: bytes,
        *,
        encryption: ObjectArtifactEncryption,
        context: Mapping[str, str] | None = None,
    ) -> bytes: ...


class ObjectArtifactClient(Protocol):
    async def create_multipart_upload(
        self,
        *,
        key: str,
        media_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectArtifactMultipartUpload: ...

    async def upload_part(
        self,
        upload: ObjectArtifactMultipartUpload,
        *,
        part_number: int,
        content: bytes,
    ) -> ObjectArtifactPart: ...

    async def complete_multipart_upload(
        self,
        upload: ObjectArtifactMultipartUpload,
        *,
        parts: Sequence[ObjectArtifactPart],
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectArtifactHead: ...

    async def abort_multipart_upload(
        self,
        upload: ObjectArtifactMultipartUpload,
    ) -> None: ...

    async def open_object(self, key: str) -> BinaryIO: ...

    async def head_object(self, key: str) -> ObjectArtifactHead: ...

    async def delete_object(self, key: str) -> None: ...

    async def signed_url(
        self,
        key: str,
        *,
        expires_in_seconds: int,
        method: str = "GET",
    ) -> str: ...


class ObjectArtifactStore:
    def __init__(
        self,
        client: ObjectArtifactClient,
        *,
        cipher: ObjectArtifactCipher | None,
        policy: ObjectArtifactStorePolicy,
        store_name: str = "object",
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        assert hasattr(client, "create_multipart_upload")
        assert hasattr(client, "upload_part")
        assert hasattr(client, "complete_multipart_upload")
        assert hasattr(client, "abort_multipart_upload")
        assert hasattr(client, "open_object")
        assert hasattr(client, "head_object")
        assert hasattr(client, "delete_object")
        assert hasattr(client, "signed_url")
        if cipher is not None:
            assert hasattr(cipher, "start_encryption")
            assert hasattr(cipher, "encrypt_chunk")
            assert hasattr(cipher, "decrypt_chunk")
        assert isinstance(policy, ObjectArtifactStorePolicy)
        _assert_non_empty_string(store_name, "store_name")
        self._client = client
        self._cipher = cipher
        self._policy = policy
        self._store_name = store_name
        self._id_factory = id_factory or _uuid_id

    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        assert isinstance(content, bytes), "content must be bytes"
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
        assert hasattr(stream, "read"), "stream must be readable"
        self._require_policy()
        if media_type is not None:
            _assert_non_empty_string(media_type, "media_type")
        new_artifact_id = artifact_id or self._new_id()
        _assert_artifact_id(new_artifact_id)
        ref_metadata = freeze_snapshot_metadata(metadata)
        effective_max_bytes = _effective_max_bytes(
            requested_max_bytes=max_bytes,
            configured_max_bytes=self._policy.max_bytes,
        )
        _assert_stream_expectations(
            max_bytes=effective_max_bytes,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
        storage_key = _storage_key(new_artifact_id)
        context = _encryption_context(
            artifact_id=new_artifact_id,
            store=self._store_name,
        )
        encryption = self._start_encryption(context)
        upload: ObjectArtifactMultipartUpload | None = None
        try:
            upload = await self._client.create_multipart_upload(
                key=storage_key,
                media_type=media_type,
                metadata=_initial_object_metadata(
                    artifact_id=new_artifact_id,
                    store=self._store_name,
                    encrypted=encryption is not None,
                ),
            )
            digest = sha256()
            object_digest = sha256()
            size_bytes = 0
            object_size_bytes = 0
            parts: list[ObjectArtifactPart] = []
            part_number = 1
            while True:
                chunk = stream.read(self._policy.part_size)
                assert isinstance(
                    chunk,
                    bytes,
                ), "artifact stream must return bytes"
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > effective_max_bytes:
                    raise ArtifactStorePolicyError(
                        "artifact stream exceeds maximum bytes"
                    )
                digest.update(chunk)
                object_chunk = self._encrypt_chunk(
                    chunk,
                    encryption=encryption,
                    context=context,
                )
                object_digest.update(object_chunk)
                object_size_bytes += len(object_chunk)
                parts.append(
                    await self._upload_part_with_retries(
                        upload,
                        part_number=part_number,
                        content=object_chunk,
                    )
                )
                part_number += 1
            plaintext_sha256 = digest.hexdigest()
            _validate_stream_digest(
                size_bytes=size_bytes,
                expected_size_bytes=expected_size_bytes,
                sha256=plaintext_sha256,
                expected_sha256=expected_sha256,
            )
            object_sha256 = object_digest.hexdigest()
            final_metadata = _final_object_metadata(
                artifact_id=new_artifact_id,
                store=self._store_name,
                size_bytes=size_bytes,
                sha256=plaintext_sha256,
                object_size_bytes=object_size_bytes,
                object_sha256=object_sha256,
                retention_days=self._policy.retention_days,
                encryption=encryption,
            )
            head = await self._client.complete_multipart_upload(
                upload,
                parts=tuple(parts),
                metadata=final_metadata,
            )
            try:
                _validate_object_head(
                    head,
                    storage_key=storage_key,
                    size_bytes=object_size_bytes,
                    sha256=object_sha256,
                )
            except ArtifactStoreError:
                await self._delete_completed_object(storage_key)
                raise
            return TaskArtifactRef(
                artifact_id=new_artifact_id,
                store=self._store_name,
                storage_key=storage_key,
                media_type=media_type,
                size_bytes=size_bytes,
                sha256=plaintext_sha256,
                metadata=_public_ref_metadata(
                    user_metadata=ref_metadata,
                    storage_key=storage_key,
                    signed_urls_supported=True,
                    retention_days=self._policy.retention_days,
                    encryption=encryption,
                ),
            )
        except (
            ArtifactStoreError,
            ArtifactStorePolicyError,
            CancelledError,
            KeyboardInterrupt,
            SystemExit,
        ):
            if upload is not None:
                await self._abort_upload(upload)
            raise
        except Exception as exc:
            if upload is not None:
                await self._abort_upload(upload)
            raise ArtifactStoreError("object artifact write failed") from exc

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        return await self.open_stream(ref)

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BinaryIO:
        self._assert_ref(ref)
        _assert_optional_non_negative_int(max_bytes, "max_bytes")
        try:
            head = await self._client.head_object(ref.storage_key)
            encryption = _encryption_from_head(head)
            reader = await self._client.open_object(ref.storage_key)
        except ArtifactStoreError:
            raise
        except Exception as exc:
            raise ArtifactStoreError("object artifact read failed") from exc
        plaintext_reader = _ObjectArtifactReader(
            reader,
            cipher=self._cipher,
            encryption=encryption,
            context=_encryption_context(
                artifact_id=ref.artifact_id,
                store=self._store_name,
            ),
            expected_size_bytes=_ref_size_or_head(ref, head),
            expected_sha256=_ref_sha256_or_head(ref, head),
        )
        return bounded_artifact_reader(
            cast(BinaryIO, plaintext_reader),
            max_bytes=max_bytes,
        )

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        self._assert_ref(ref)
        try:
            head = await self._client.head_object(ref.storage_key)
        except ArtifactStoreError:
            raise
        except Exception as exc:
            raise ArtifactStoreError("object artifact stat failed") from exc
        return TaskArtifactStat(
            ref=ref,
            size_bytes=_ref_size_or_head(ref, head),
            sha256=_ref_sha256_or_head(ref, head),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        self._assert_ref(ref)
        try:
            await self._client.delete_object(ref.storage_key)
        except ArtifactStoreError:
            raise
        except Exception as exc:
            raise ArtifactStoreError("object artifact delete failed") from exc

    async def signed_url(
        self,
        ref: TaskArtifactRef,
        *,
        expires_in_seconds: int,
        method: str = "GET",
    ) -> str:
        self._assert_ref(ref)
        _assert_positive_int(expires_in_seconds, "expires_in_seconds")
        _assert_non_empty_string(method, "method")
        if (
            self._policy.signed_url_max_seconds is not None
            and expires_in_seconds > self._policy.signed_url_max_seconds
        ):
            raise ArtifactStorePolicyError(
                "signed URL expiration exceeds maximum seconds"
            )
        try:
            value = await self._client.signed_url(
                ref.storage_key,
                expires_in_seconds=expires_in_seconds,
                method=method,
            )
        except ArtifactStoreError:
            raise
        except Exception as exc:
            raise ArtifactStoreError(
                "object artifact signed URL failed"
            ) from exc
        _assert_non_empty_string(value, "signed URL")
        return value

    async def _upload_part_with_retries(
        self,
        upload: ObjectArtifactMultipartUpload,
        *,
        part_number: int,
        content: bytes,
    ) -> ObjectArtifactPart:
        attempts = 0
        while True:
            try:
                part = await self._client.upload_part(
                    upload,
                    part_number=part_number,
                    content=content,
                )
                _validate_part(part, part_number=part_number, content=content)
                return part
            except ObjectArtifactClientTransientError as exc:
                attempts += 1
                if attempts > self._policy.max_part_retries:
                    raise ArtifactStoreError(
                        "object artifact part upload failed"
                    ) from exc

    async def _abort_upload(
        self,
        upload: ObjectArtifactMultipartUpload,
    ) -> None:
        with suppress(Exception):
            await self._client.abort_multipart_upload(upload)

    async def _delete_completed_object(self, storage_key: str) -> None:
        with suppress(Exception):
            await self._client.delete_object(storage_key)

    def _start_encryption(
        self,
        context: Mapping[str, str],
    ) -> ObjectArtifactEncryption | None:
        if self._cipher is None:
            return None
        encryption = self._cipher.start_encryption(
            purpose=TaskKeyPurpose.ARTIFACT_CONTENT,
            context=context,
        )
        assert isinstance(encryption, ObjectArtifactEncryption)
        return encryption

    def _encrypt_chunk(
        self,
        chunk: bytes,
        *,
        encryption: ObjectArtifactEncryption | None,
        context: Mapping[str, str],
    ) -> bytes:
        if encryption is None:
            return chunk
        assert self._cipher is not None
        encrypted = self._cipher.encrypt_chunk(
            chunk,
            encryption=encryption,
            context=context,
        )
        assert isinstance(encrypted, bytes), "encrypted chunk must be bytes"
        return encrypted

    def _assert_ref(self, ref: TaskArtifactRef) -> None:
        assert isinstance(ref, TaskArtifactRef)
        if ref.store != self._store_name:
            raise ArtifactStoreNotFoundError("artifact store does not match")
        _assert_storage_key(ref.storage_key)
        if ref.storage_key != _storage_key(ref.artifact_id):
            raise ArtifactStoreNotFoundError("artifact storage key mismatch")

    def _require_policy(self) -> None:
        if not self._policy.raw_storage_allowed:
            raise ArtifactStorePolicyError("artifact byte storage is disabled")
        if self._policy.retention_days is None:
            raise ArtifactStorePolicyError(
                "artifact byte storage requires retention"
            )
        if self._policy.max_bytes is None:
            raise ArtifactStorePolicyError(
                "artifact byte storage requires maximum bytes"
            )
        if self._policy.encryption_required and self._cipher is None:
            raise ArtifactStorePolicyError(
                "object artifact storage requires encryption"
            )

    def _new_id(self) -> str:
        value = self._id_factory()
        _assert_artifact_id(value)
        return value


class _ObjectArtifactReader:
    def __init__(
        self,
        reader: BinaryIO,
        *,
        cipher: ObjectArtifactCipher | None,
        encryption: ObjectArtifactEncryption | None,
        context: Mapping[str, str],
        expected_size_bytes: int,
        expected_sha256: str,
    ) -> None:
        self._reader = reader
        self._cipher = cipher
        self._encryption = encryption
        self._context = context
        self._expected_size_bytes = expected_size_bytes
        self._expected_sha256 = expected_sha256
        self._digest = sha256()
        self._size_bytes = 0
        self._validated = False

    def read(self, size: int = -1) -> bytes:
        assert isinstance(size, int)
        assert not isinstance(size, bool)
        chunk = self._reader.read(size)
        assert isinstance(chunk, bytes), "artifact reader must return bytes"
        if chunk:
            plaintext = self._decrypt_chunk(chunk)
            self._digest.update(plaintext)
            self._size_bytes += len(plaintext)
            if size < 0:
                self._validate()
            return plaintext
        self._validate()
        return b""

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> "_ObjectArtifactReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._reader, name)

    def _decrypt_chunk(self, chunk: bytes) -> bytes:
        if self._encryption is None:
            return chunk
        if self._cipher is None:
            raise ArtifactStorePolicyError(
                "object artifact storage requires encryption"
            )
        plaintext = self._cipher.decrypt_chunk(
            chunk,
            encryption=self._encryption,
            context=self._context,
        )
        assert isinstance(plaintext, bytes), "decrypted chunk must be bytes"
        return plaintext

    def _validate(self) -> None:
        if self._validated:
            return
        self._validated = True
        if self._size_bytes != self._expected_size_bytes:
            raise ArtifactStoreError("artifact stream size mismatch")
        if self._digest.hexdigest() != self._expected_sha256:
            raise ArtifactStoreError("artifact stream digest mismatch")


def _storage_key(artifact_id: str) -> str:
    return f"objects/{artifact_id[:2].lower()}/{artifact_id}"


def _assert_storage_key(value: str) -> None:
    _assert_non_empty_string(value, "storage_key")
    if not fullmatch(
        r"objects/[A-Za-z0-9]{1,2}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        value,
    ):
        raise ArtifactStoreError("storage_key must be a stable object token")


def _assert_artifact_id(value: str) -> None:
    _assert_non_empty_string(value, "artifact_id")
    assert fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        value,
    ), "artifact_id must be a stable token"


def _assert_positive_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"


def _assert_optional_positive_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    _assert_positive_int(value, field_name)


def _assert_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    _assert_non_negative_int(value, field_name)


def _assert_sha256(value: str) -> None:
    _assert_non_empty_string(value, "sha256")
    assert fullmatch(
        r"[0-9a-f]{64}", value
    ), "sha256 must be a lowercase SHA-256 hex digest"


def _assert_stream_expectations(
    *,
    max_bytes: int,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
) -> None:
    _assert_non_negative_int(max_bytes, "max_bytes")
    _assert_optional_non_negative_int(
        expected_size_bytes,
        "expected_size_bytes",
    )
    if expected_sha256 is not None:
        _assert_sha256(expected_sha256)
    if expected_size_bytes is not None and expected_size_bytes > max_bytes:
        raise ArtifactStorePolicyError("artifact stream exceeds maximum bytes")


def _validate_stream_digest(
    *,
    size_bytes: int,
    expected_size_bytes: int | None,
    sha256: str,
    expected_sha256: str | None,
) -> None:
    if expected_size_bytes is not None and size_bytes != expected_size_bytes:
        raise ArtifactStoreError("artifact stream size mismatch")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ArtifactStoreError("artifact stream digest mismatch")


def _effective_max_bytes(
    *,
    requested_max_bytes: int | None,
    configured_max_bytes: int | None,
) -> int:
    assert configured_max_bytes is not None
    _assert_optional_non_negative_int(requested_max_bytes, "max_bytes")
    if requested_max_bytes is None:
        return configured_max_bytes
    return min(requested_max_bytes, configured_max_bytes)


def _validate_part(
    part: ObjectArtifactPart,
    *,
    part_number: int,
    content: bytes,
) -> None:
    assert isinstance(part, ObjectArtifactPart)
    if part.part_number != part_number:
        raise ArtifactStoreError("object artifact part number mismatch")
    if part.size_bytes != len(content):
        raise ArtifactStoreError("object artifact part size mismatch")
    if part.sha256 != sha256(content).hexdigest():
        raise ArtifactStoreError("object artifact part digest mismatch")


def _validate_object_head(
    head: ObjectArtifactHead,
    *,
    storage_key: str,
    size_bytes: int,
    sha256: str,
) -> None:
    assert isinstance(head, ObjectArtifactHead)
    if head.key != storage_key:
        raise ArtifactStoreError("object artifact storage key mismatch")
    if head.size_bytes != size_bytes:
        raise ArtifactStoreError("object artifact size mismatch")
    if head.sha256 != sha256:
        raise ArtifactStoreError("object artifact digest mismatch")


def _initial_object_metadata(
    *,
    artifact_id: str,
    store: str,
    encrypted: bool,
) -> Mapping[str, str]:
    return {
        "artifact_id": artifact_id,
        "encrypted": "true" if encrypted else "false",
        "store": store,
    }


def _final_object_metadata(
    *,
    artifact_id: str,
    store: str,
    size_bytes: int,
    sha256: str,
    object_size_bytes: int,
    object_sha256: str,
    retention_days: int | None,
    encryption: ObjectArtifactEncryption | None,
) -> Mapping[str, str]:
    metadata = {
        "artifact_id": artifact_id,
        "object_sha256": object_sha256,
        "object_size_bytes": str(object_size_bytes),
        "plaintext_sha256": sha256,
        "plaintext_size_bytes": str(size_bytes),
        "retention_days": str(retention_days) if retention_days else "",
        "store": store,
    }
    if encryption is not None:
        metadata.update(
            {
                "encryption_algorithm": encryption.algorithm,
                "encryption_key_id": encryption.key_id,
                "encrypted": "true",
            }
        )
        for key, value in encryption.metadata.items():
            metadata[f"encryption_metadata_{key}"] = value
    else:
        metadata["encrypted"] = "false"
    return metadata


def _public_ref_metadata(
    *,
    user_metadata: Mapping[str, object],
    storage_key: str,
    signed_urls_supported: bool,
    retention_days: int | None,
    encryption: ObjectArtifactEncryption | None,
) -> TaskSnapshotMetadata:
    metadata: dict[str, object] = dict(user_metadata)
    metadata["object"] = {
        "key_sha256": sha256(storage_key.encode("utf-8")).hexdigest(),
        "privacy": STORED_MARKER,
        "reference": "object_store",
        "signed_urls": signed_urls_supported,
    }
    metadata["retention_days"] = retention_days
    if encryption is not None:
        encryption_metadata: dict[str, object] = {
            "algorithm": encryption.algorithm,
            "key_id": encryption.key_id,
            "privacy": ENCRYPTED_MARKER,
        }
        metadata["encryption"] = encryption_metadata
    return freeze_snapshot_metadata(metadata)


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


def _encryption_from_head(
    head: ObjectArtifactHead,
) -> ObjectArtifactEncryption | None:
    if head.metadata.get("encrypted") != "true":
        return None
    key_id = head.metadata.get("encryption_key_id")
    algorithm = head.metadata.get("encryption_algorithm")
    if key_id is None or algorithm is None:
        raise ArtifactStoreError("object artifact encryption metadata missing")
    metadata: dict[str, str] = {}
    prefix = "encryption_metadata_"
    for key, value in head.metadata.items():
        if key.startswith(prefix):
            metadata[key.removeprefix(prefix)] = value
    return ObjectArtifactEncryption(
        key_id=key_id,
        algorithm=algorithm,
        metadata=metadata,
    )


def _ref_size_or_head(ref: TaskArtifactRef, head: ObjectArtifactHead) -> int:
    if ref.size_bytes is not None:
        return ref.size_bytes
    size_value = head.metadata.get("plaintext_size_bytes")
    if size_value is None:
        raise ArtifactStoreError("object artifact size metadata missing")
    return _metadata_int(size_value, "plaintext_size_bytes")


def _ref_sha256_or_head(
    ref: TaskArtifactRef,
    head: ObjectArtifactHead,
) -> str:
    if ref.sha256 is not None:
        return ref.sha256
    value = head.metadata.get("plaintext_sha256")
    if value is None:
        raise ArtifactStoreError("object artifact digest metadata missing")
    _assert_sha256(value)
    return value


def _metadata_int(value: str, field_name: str) -> int:
    assert isinstance(value, str)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArtifactStoreError(
            f"object artifact {field_name} metadata is invalid"
        ) from exc
    _assert_non_negative_int(parsed, field_name)
    return parsed


def _uuid_id() -> str:
    return uuid4().hex
