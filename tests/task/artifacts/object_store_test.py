from asyncio import CancelledError
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.task import (
    ArtifactStoreConflictError,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskRetentionAction,
    TaskRetentionService,
)
from avalan.task.artifacts import (
    ObjectArtifactClientTransientError,
    ObjectArtifactEncryption,
    ObjectArtifactHead,
    ObjectArtifactMultipartUpload,
    ObjectArtifactPart,
    ObjectArtifactStore,
    ObjectArtifactStorePolicy,
)
from avalan.task.stores import InMemoryTaskStore


class _DefaultCipher:
    pass


_DEFAULT_CIPHER = _DefaultCipher()


@dataclass(slots=True)
class FakeObjectRecord:
    content: bytes
    media_type: str | None
    metadata: Mapping[str, str]


@dataclass(slots=True)
class FakeUploadState:
    key: str
    media_type: str | None
    metadata: Mapping[str, str]
    parts: dict[int, bytes] = field(default_factory=dict)


class XorObjectCipher:
    def __init__(self) -> None:
        self.start_context: Mapping[str, str] | None = None
        self.encrypt_contexts: list[Mapping[str, str] | None] = []
        self.decrypt_contexts: list[Mapping[str, str] | None] = []

    def start_encryption(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> ObjectArtifactEncryption:
        self.start_context = context
        assert purpose == TaskKeyPurpose.ARTIFACT_CONTENT
        return ObjectArtifactEncryption(
            key_id=key_id or "object-key-v1",
            algorithm="xor-test",
            metadata={"cipher_scope": "safe"},
        )

    def encrypt_chunk(
        self,
        value: bytes,
        *,
        encryption: ObjectArtifactEncryption,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        _ = encryption
        self.encrypt_contexts.append(context)
        return _xor(value)

    def decrypt_chunk(
        self,
        value: bytes,
        *,
        encryption: ObjectArtifactEncryption,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        _ = encryption
        self.decrypt_contexts.append(context)
        return _xor(value)


class CancellingReader:
    def read(self, _size: int = -1) -> bytes:
        raise CancelledError


class FakeObjectClient:
    def __init__(self) -> None:
        self.objects: dict[str, FakeObjectRecord] = {}
        self.uploads: dict[str, FakeUploadState] = {}
        self.aborted_upload_ids: list[str] = []
        self.deleted_keys: list[str] = []
        self.signed_requests: list[tuple[str, int, str]] = []
        self.part_attempts: dict[int, int] = {}
        self.fail_part_once: set[int] = set()
        self.fail_part_always: set[int] = set()
        self.bad_part_digest = False
        self.bad_part_number = False
        self.bad_part_size = False
        self.complete_error = False
        self.create_error = False
        self.delete_error = False
        self.head_error = False
        self.open_error = False
        self.signed_url_error = False
        self.wrong_complete_digest = False
        self.wrong_complete_key = False
        self.wrong_complete_size = False
        self.upload_counter = 0

    async def create_multipart_upload(
        self,
        *,
        key: str,
        media_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectArtifactMultipartUpload:
        if self.create_error:
            raise RuntimeError("private bucket failure")
        if key in self.objects:
            raise ArtifactStoreConflictError("artifact already exists")
        self.upload_counter += 1
        upload = ObjectArtifactMultipartUpload(
            key=key,
            upload_id=f"upload-{self.upload_counter}",
        )
        self.uploads[upload.upload_id] = FakeUploadState(
            key=key,
            media_type=media_type,
            metadata=dict(metadata or {}),
        )
        return upload

    async def upload_part(
        self,
        upload: ObjectArtifactMultipartUpload,
        *,
        part_number: int,
        content: bytes,
    ) -> ObjectArtifactPart:
        self.part_attempts[part_number] = (
            self.part_attempts.get(part_number, 0) + 1
        )
        if part_number in self.fail_part_once:
            self.fail_part_once.remove(part_number)
            raise ObjectArtifactClientTransientError(
                "transient object upload failure"
            )
        if part_number in self.fail_part_always:
            raise ObjectArtifactClientTransientError(
                "transient object upload failure"
            )
        state = self.uploads[upload.upload_id]
        state.parts[part_number] = content
        return ObjectArtifactPart(
            part_number=(
                part_number + 1 if self.bad_part_number else (part_number)
            ),
            etag=f"etag-{part_number}",
            size_bytes=(
                len(content) + 1 if self.bad_part_size else (len(content))
            ),
            sha256=(
                "0" * 64
                if self.bad_part_digest
                else (sha256(content).hexdigest())
            ),
        )

    async def complete_multipart_upload(
        self,
        upload: ObjectArtifactMultipartUpload,
        *,
        parts: Sequence[ObjectArtifactPart],
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectArtifactHead:
        if self.complete_error:
            raise RuntimeError("private completion failure")
        if upload.key in self.objects:
            raise ArtifactStoreConflictError("artifact already exists")
        state = self.uploads[upload.upload_id]
        content = b"".join(state.parts[part.part_number] for part in parts)
        self.objects[upload.key] = FakeObjectRecord(
            content=content,
            media_type=state.media_type,
            metadata=dict(metadata or {}),
        )
        return ObjectArtifactHead(
            key="objects/xx/wrong" if self.wrong_complete_key else upload.key,
            size_bytes=(
                len(content) + 1
                if self.wrong_complete_size
                else (len(content))
            ),
            sha256=(
                "0" * 64
                if self.wrong_complete_digest
                else (sha256(content).hexdigest())
            ),
            metadata=dict(metadata or {}),
        )

    async def abort_multipart_upload(
        self,
        upload: ObjectArtifactMultipartUpload,
    ) -> None:
        self.aborted_upload_ids.append(upload.upload_id)
        self.uploads.pop(upload.upload_id, None)

    async def open_object(self, key: str) -> BinaryIO:
        if self.open_error:
            raise RuntimeError("private read failure")
        record = self.objects.get(key)
        if record is None:
            raise ArtifactStoreNotFoundError("artifact was not found")
        return BytesIO(record.content)

    async def head_object(self, key: str) -> ObjectArtifactHead:
        if self.head_error:
            raise RuntimeError("private stat failure")
        record = self.objects.get(key)
        if record is None:
            raise ArtifactStoreNotFoundError("artifact was not found")
        return ObjectArtifactHead(
            key=key,
            size_bytes=len(record.content),
            sha256=sha256(record.content).hexdigest(),
            metadata=record.metadata,
        )

    async def delete_object(self, key: str) -> None:
        if self.delete_error:
            raise RuntimeError("private delete failure")
        if key not in self.objects:
            raise ArtifactStoreNotFoundError("artifact was not found")
        self.deleted_keys.append(key)
        del self.objects[key]

    async def signed_url(
        self,
        key: str,
        *,
        expires_in_seconds: int,
        method: str = "GET",
    ) -> str:
        if self.signed_url_error:
            raise RuntimeError("private signing failure")
        if key not in self.objects:
            raise ArtifactStoreNotFoundError("artifact was not found")
        self.signed_requests.append((key, expires_in_seconds, method))
        return (
            "https://signed.example.test/object"
            f"?method={method}&expires={expires_in_seconds}"
        )


class ObjectArtifactStoreTest(IsolatedAsyncioTestCase):
    async def test_put_open_stat_delete_and_signed_url_round_trip(
        self,
    ) -> None:
        client = FakeObjectClient()
        cipher = XorObjectCipher()
        store = _store(client, cipher=cipher, id_factory=lambda: "artifact-1")
        content = b"private bytes for object storage"
        expected_sha256 = sha256(content).hexdigest()

        ref = await store.put_stream(
            BytesIO(content),
            media_type="text/plain",
            metadata={"label": "sanitized"},
            max_bytes=128,
            expected_size_bytes=len(content),
            expected_sha256=expected_sha256,
        )
        stored = client.objects[ref.storage_key]
        stat = await store.stat(ref)
        reader = await store.open_stream(ref, max_bytes=128)
        try:
            read_content = reader.read()
        finally:
            reader.close()
        signed_url = await store.signed_url(
            ref,
            expires_in_seconds=60,
            method="GET",
        )
        await store.delete(ref)

        self.assertEqual(ref.artifact_id, "artifact-1")
        self.assertEqual(ref.store, "object")
        self.assertEqual(ref.storage_key, "objects/ar/artifact-1")
        self.assertEqual(ref.size_bytes, len(content))
        self.assertEqual(ref.sha256, expected_sha256)
        self.assertEqual(ref.media_type, "text/plain")
        self.assertEqual(stat.size_bytes, len(content))
        self.assertEqual(stat.sha256, expected_sha256)
        self.assertEqual(read_content, content)
        self.assertNotEqual(stored.content, content)
        self.assertEqual(stored.media_type, "text/plain")
        self.assertEqual(stored.metadata["plaintext_sha256"], expected_sha256)
        self.assertEqual(stored.metadata["encrypted"], "true")
        self.assertEqual(
            cipher.start_context,
            {
                "artifact_id": "artifact-1",
                "purpose": "artifact_content",
                "store": "object",
            },
        )
        self.assertTrue(cipher.encrypt_contexts)
        self.assertTrue(cipher.decrypt_contexts)
        self.assertEqual(
            client.signed_requests,
            [(ref.storage_key, 60, "GET")],
        )
        self.assertIn("https://signed.example.test/object", signed_url)
        object_metadata = cast(Mapping[str, object], ref.metadata["object"])
        encryption = cast(Mapping[str, object], ref.metadata["encryption"])
        self.assertEqual(ref.metadata["label"], "sanitized")
        self.assertEqual(object_metadata["privacy"], "<stored>")
        self.assertEqual(object_metadata["reference"], "object_store")
        self.assertEqual(encryption["privacy"], "<encrypted>")
        self.assertEqual(encryption["key_id"], "object-key-v1")
        self.assertNotIn(ref.storage_key, str(ref.summary()))
        self.assertNotIn(ref.storage_key, str(ref.metadata))
        self.assertNotIn("private", str(ref.summary()))
        self.assertNotIn(ref.storage_key, str(stored.metadata.values()))
        self.assertNotIn(ref.storage_key, client.objects)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.stat(ref)

    async def test_put_rejects_unconfigured_storage_policy(self) -> None:
        client = FakeObjectClient()
        cipher = XorObjectCipher()

        with self.assertRaises(ArtifactStorePolicyError):
            await _store(
                client,
                cipher=cipher,
                policy=ObjectArtifactStorePolicy(
                    raw_storage_allowed=False,
                    retention_days=7,
                    max_bytes=64,
                ),
            ).put(b"private bytes")
        with self.assertRaises(ArtifactStorePolicyError):
            await _store(
                client,
                cipher=cipher,
                policy=ObjectArtifactStorePolicy(
                    raw_storage_allowed=True,
                    max_bytes=64,
                ),
            ).put(b"private bytes")
        with self.assertRaises(ArtifactStorePolicyError):
            await _store(
                client,
                cipher=cipher,
                policy=ObjectArtifactStorePolicy(
                    raw_storage_allowed=True,
                    retention_days=7,
                ),
            ).put(b"private bytes")
        with self.assertRaises(ArtifactStorePolicyError):
            await _store(
                client,
                cipher=None,
                policy=ObjectArtifactStorePolicy(
                    raw_storage_allowed=True,
                    retention_days=7,
                    max_bytes=64,
                    encryption_required=True,
                ),
            ).put(b"private bytes")

        self.assertEqual(client.objects, {})

    async def test_unencrypted_mode_requires_explicit_policy(self) -> None:
        client = FakeObjectClient()
        store = _store(
            client,
            cipher=None,
            policy=ObjectArtifactStorePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                max_bytes=64,
                encryption_required=False,
            ),
            id_factory=lambda: "artifact-1",
        )

        ref = await store.put(b"plain object bytes")
        reader = await store.open(ref)
        try:
            content = reader.read()
        finally:
            reader.close()

        self.assertEqual(content, b"plain object bytes")
        self.assertNotIn("encryption", ref.metadata)
        self.assertEqual(
            client.objects[ref.storage_key].content,
            b"plain object bytes",
        )

    async def test_stream_validation_failures_abort_without_object(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client, id_factory=lambda: "artifact-1")

        with self.assertRaises(ArtifactStoreError):
            await store.put_stream(
                BytesIO(b"private bytes"),
                max_bytes=None,
                expected_size_bytes=5,
            )
        with self.assertRaises(ArtifactStorePolicyError):
            await store.put_stream(BytesIO(b"private bytes"), max_bytes=3)
        with self.assertRaises(ArtifactStorePolicyError):
            await store.put_stream(
                BytesIO(b"private bytes"),
                artifact_id="artifact-2",
                max_bytes=10,
                expected_size_bytes=11,
            )
        with self.assertRaises(ArtifactStoreError):
            await store.put_stream(
                BytesIO(b"private bytes"),
                artifact_id="artifact-3",
                max_bytes=20,
                expected_sha256="0" * 64,
            )
        with self.assertRaises(CancelledError):
            await store.put_stream(
                CancellingReader(),  # type: ignore[arg-type]
                artifact_id="artifact-4",
                max_bytes=20,
            )
        with self.assertRaises(AssertionError):
            await store.put(
                b"private bytes",
                artifact_id="artifact-5",
                metadata={"bad": object()},
            )

        self.assertEqual(client.objects, {})
        self.assertEqual(
            client.aborted_upload_ids,
            ["upload-1", "upload-2", "upload-3", "upload-4"],
        )

    async def test_unexpected_write_failure_is_sanitized_and_aborted(
        self,
    ) -> None:
        client = FakeObjectClient()
        client.complete_error = True
        store = _store(client, id_factory=lambda: "artifact-1")

        with self.assertRaises(ArtifactStoreError) as error:
            await store.put(b"private bytes")

        self.assertEqual(str(error.exception), "object artifact write failed")
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(client.objects, {})
        self.assertEqual(client.aborted_upload_ids, ["upload-1"])

    async def test_default_artifact_id_factory_creates_safe_object_key(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)

        ref = await store.put_stream(BytesIO(b"bytes"), max_bytes=None)

        self.assertTrue(ref.artifact_id)
        self.assertTrue(ref.storage_key.startswith("objects/"))
        self.assertIn(ref.storage_key, client.objects)

    async def test_multipart_upload_retries_transient_part_failure(
        self,
    ) -> None:
        client = FakeObjectClient()
        client.fail_part_once.add(2)
        store = _store(
            client,
            policy=_policy(part_size=4, max_part_retries=2),
            id_factory=lambda: "artifact-1",
        )

        ref = await store.put(b"abcdefghij")

        self.assertIn(ref.storage_key, client.objects)
        self.assertEqual(client.part_attempts[2], 2)
        self.assertEqual(client.aborted_upload_ids, [])

    async def test_multipart_upload_aborts_after_retry_exhaustion(
        self,
    ) -> None:
        client = FakeObjectClient()
        client.fail_part_always.add(1)
        store = _store(
            client,
            policy=_policy(part_size=4, max_part_retries=1),
            id_factory=lambda: "artifact-1",
        )

        with self.assertRaises(ArtifactStoreError) as error:
            await store.put(b"abcdefghij")

        self.assertEqual(
            str(error.exception), "object artifact part upload failed"
        )
        self.assertEqual(client.part_attempts[1], 2)
        self.assertEqual(client.objects, {})
        self.assertEqual(client.aborted_upload_ids, ["upload-1"])

    async def test_duplicate_artifact_id_keeps_existing_object(self) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"first", artifact_id="artifact-1")

        with self.assertRaises(ArtifactStoreConflictError):
            await store.put(b"second", artifact_id=ref.artifact_id)
        reader = await store.open(ref)
        try:
            content = reader.read()
        finally:
            reader.close()

        self.assertEqual(content, b"first")
        self.assertEqual(len(client.objects), 1)

    async def test_references_are_store_scoped_and_key_checked(self) -> None:
        store = _store(FakeObjectClient())
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        wrong_store = TaskArtifactRef(
            artifact_id=ref.artifact_id,
            store="other",
            storage_key=ref.storage_key,
        )
        mismatched = TaskArtifactRef(
            artifact_id="artifact-2",
            store=ref.store,
            storage_key=ref.storage_key,
        )
        escaped = TaskArtifactRef(
            artifact_id="artifact-3",
            store=ref.store,
            storage_key="../artifact-3",
        )

        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.open(wrong_store)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.open(mismatched)
        with self.assertRaises(ArtifactStoreError):
            await store.open(escaped)

    async def test_content_address_failures_cleanup_uploaded_object(
        self,
    ) -> None:
        client = FakeObjectClient()
        client.wrong_complete_digest = True
        store = _store(client, id_factory=lambda: "artifact-1")

        with self.assertRaises(ArtifactStoreError) as error:
            await store.put(b"private bytes")

        self.assertEqual(
            str(error.exception), "object artifact digest mismatch"
        )
        self.assertEqual(client.objects, {})
        self.assertEqual(client.deleted_keys, ["objects/ar/artifact-1"])
        self.assertEqual(client.aborted_upload_ids, ["upload-1"])

    async def test_part_validation_failures_abort_upload(self) -> None:
        for attribute, message in (
            ("bad_part_digest", "object artifact part digest mismatch"),
            ("bad_part_number", "object artifact part number mismatch"),
            ("bad_part_size", "object artifact part size mismatch"),
        ):
            with self.subTest(attribute=attribute):
                client = FakeObjectClient()
                setattr(client, attribute, True)
                store = _store(client, id_factory=lambda: "artifact-1")

                with self.assertRaises(ArtifactStoreError) as error:
                    await store.put(b"private bytes")

                self.assertEqual(str(error.exception), message)
                self.assertEqual(client.objects, {})
                self.assertEqual(client.aborted_upload_ids, ["upload-1"])

    async def test_complete_head_key_and_size_are_verified(self) -> None:
        for attribute, message in (
            ("wrong_complete_key", "object artifact storage key mismatch"),
            ("wrong_complete_size", "object artifact size mismatch"),
        ):
            with self.subTest(attribute=attribute):
                client = FakeObjectClient()
                setattr(client, attribute, True)
                store = _store(client, id_factory=lambda: "artifact-1")

                with self.assertRaises(ArtifactStoreError) as error:
                    await store.put(b"private bytes")

                self.assertEqual(str(error.exception), message)
                self.assertEqual(client.objects, {})

    async def test_legacy_ref_uses_head_metadata_without_reading_bytes(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        legacy_ref = TaskArtifactRef(
            artifact_id=ref.artifact_id,
            store=ref.store,
            storage_key=ref.storage_key,
            media_type=ref.media_type,
        )

        stat = await store.stat(legacy_ref)

        self.assertEqual(stat.size_bytes, len(b"private bytes"))
        self.assertEqual(stat.sha256, sha256(b"private bytes").hexdigest())

    async def test_missing_or_invalid_head_metadata_is_rejected(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        legacy_ref = TaskArtifactRef(
            artifact_id=ref.artifact_id,
            store=ref.store,
            storage_key=ref.storage_key,
        )
        record = client.objects[ref.storage_key]
        client.objects[ref.storage_key] = FakeObjectRecord(
            content=record.content,
            media_type=record.media_type,
            metadata={"plaintext_sha256": ref.sha256 or ""},
        )
        with self.assertRaises(ArtifactStoreError):
            await store.stat(legacy_ref)
        client.objects[ref.storage_key] = FakeObjectRecord(
            content=record.content,
            media_type=record.media_type,
            metadata={
                "plaintext_size_bytes": str(ref.size_bytes),
            },
        )
        with self.assertRaises(ArtifactStoreError):
            await store.stat(legacy_ref)
        client.objects[ref.storage_key] = FakeObjectRecord(
            content=record.content,
            media_type=record.media_type,
            metadata={
                "plaintext_size_bytes": "not-an-int",
                "plaintext_sha256": ref.sha256 or "",
            },
        )
        with self.assertRaises(ArtifactStoreError):
            await store.stat(legacy_ref)
        client.objects[ref.storage_key] = FakeObjectRecord(
            content=record.content,
            media_type=record.media_type,
            metadata={
                "encrypted": "true",
                "plaintext_size_bytes": str(ref.size_bytes),
                "plaintext_sha256": ref.sha256 or "",
            },
        )
        with self.assertRaises(ArtifactStoreError):
            await store.open(legacy_ref)

    async def test_reader_validates_eof_and_supports_context_protocol(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        reader = await store.open(ref)

        with reader as entered:
            self.assertIs(entered, reader)
            self.assertTrue(reader.read(3))
            self.assertTrue(callable(reader.seekable))
            while reader.read(3):
                pass
            self.assertEqual(reader.read(), b"")

    async def test_encrypted_object_requires_matching_reader_cipher(
        self,
    ) -> None:
        client = FakeObjectClient()
        encrypted_store = _store(client)
        ref = await encrypted_store.put(
            b"private bytes",
            artifact_id="artifact-1",
        )
        unencrypted_store = _store(
            client,
            cipher=None,
            policy=ObjectArtifactStorePolicy(
                raw_storage_allowed=True,
                retention_days=7,
                max_bytes=128,
                encryption_required=False,
            ),
        )
        reader = await unencrypted_store.open(ref)
        try:
            with self.assertRaises(ArtifactStorePolicyError):
                reader.read()
        finally:
            reader.close()

    async def test_corrupt_object_content_is_detected_on_read(self) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        record = client.objects[ref.storage_key]
        client.objects[ref.storage_key] = FakeObjectRecord(
            content=b"corrupt",
            media_type=record.media_type,
            metadata=record.metadata,
        )
        reader = await store.open(ref)
        try:
            with self.assertRaises(ArtifactStoreError):
                reader.read()
        finally:
            reader.close()

    async def test_same_size_corrupt_object_digest_is_detected_on_read(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        record = client.objects[ref.storage_key]
        client.objects[ref.storage_key] = FakeObjectRecord(
            content=b"0" * len(record.content),
            media_type=record.media_type,
            metadata=record.metadata,
        )
        reader = await store.open(ref)
        try:
            with self.assertRaises(ArtifactStoreError) as error:
                reader.read()
        finally:
            reader.close()

        self.assertEqual(
            str(error.exception), "artifact stream digest mismatch"
        )

    async def test_open_stat_delete_and_signed_url_wrap_client_failures(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        client.head_error = True
        with self.assertRaises(ArtifactStoreError) as stat_error:
            await store.stat(ref)
        with self.assertRaises(ArtifactStoreError) as open_error:
            await store.open(ref)
        client.head_error = False
        client.open_error = True
        with self.assertRaises(ArtifactStoreError) as read_error:
            await store.open(ref)
        client.open_error = False
        client.delete_error = True
        with self.assertRaises(ArtifactStoreError) as delete_error:
            await store.delete(ref)
        client.delete_error = False
        client.signed_url_error = True
        with self.assertRaises(ArtifactStoreError) as signed_error:
            await store.signed_url(ref, expires_in_seconds=60)
        with self.assertRaises(ArtifactStorePolicyError):
            await _store(
                client,
                policy=_policy(signed_url_max_seconds=30),
            ).signed_url(ref, expires_in_seconds=31)

        self.assertEqual(
            str(stat_error.exception), "object artifact stat failed"
        )
        self.assertEqual(
            str(open_error.exception), "object artifact read failed"
        )
        self.assertEqual(
            str(read_error.exception), "object artifact read failed"
        )
        self.assertEqual(
            str(delete_error.exception),
            "object artifact delete failed",
        )
        self.assertEqual(
            str(signed_error.exception),
            "object artifact signed URL failed",
        )
        for error in (
            stat_error.exception,
            open_error.exception,
            read_error.exception,
            delete_error.exception,
            signed_error.exception,
        ):
            self.assertNotIn("private", str(error))

    async def test_missing_object_operations_raise_not_found(self) -> None:
        store = _store(FakeObjectClient())
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="object",
            storage_key="objects/ar/artifact-1",
        )

        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.stat(ref)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.open(ref)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.delete(ref)
        with self.assertRaises(ArtifactStoreNotFoundError):
            await store.signed_url(ref, expires_in_seconds=60)

    async def test_retention_service_deletes_object_artifacts(self) -> None:
        client = FakeObjectClient()
        store = _store(client)
        task_store = await _task_store()
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        await task_store.append_artifact(
            run.run_id,
            ref=ref,
            purpose=TaskArtifactPurpose.INPUT,
            retention=TaskArtifactRetention(
                expires_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )
        service = TaskRetentionService(task_store, {"object": store})

        sweep = await service.sweep_expired(
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )

        self.assertEqual(len(sweep.results), 1)
        self.assertEqual(sweep.results[0].action, TaskRetentionAction.DELETED)
        self.assertNotIn(ref.storage_key, client.objects)
        self.assertEqual(
            (await task_store.get_artifact(ref.artifact_id)).state,
            TaskArtifactState.DELETED,
        )

    async def test_retention_service_marks_lifecycle_deleted_object_lost(
        self,
    ) -> None:
        client = FakeObjectClient()
        store = _store(client)
        task_store = await _task_store()
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        ref = await store.put(b"private bytes", artifact_id="artifact-1")
        del client.objects[ref.storage_key]
        await task_store.append_artifact(
            run.run_id,
            ref=ref,
            purpose=TaskArtifactPurpose.INPUT,
            retention=TaskArtifactRetention(
                expires_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )
        service = TaskRetentionService(task_store, {"object": store})

        sweep = await service.sweep_expired(
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )

        self.assertEqual(len(sweep.results), 1)
        self.assertEqual(sweep.results[0].action, TaskRetentionAction.LOST)
        self.assertEqual(
            (await task_store.get_artifact(ref.artifact_id)).state,
            TaskArtifactState.LOST,
        )


class ObjectArtifactDataclassValidationTest(TestCase):
    def test_policy_and_value_objects_validate_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            ObjectArtifactStorePolicy(retention_days=0)
        with self.assertRaises(AssertionError):
            ObjectArtifactStorePolicy(max_bytes=-1)
        with self.assertRaises(AssertionError):
            ObjectArtifactStorePolicy(part_size=0)
        with self.assertRaises(AssertionError):
            ObjectArtifactStorePolicy(max_part_retries=-1)
        with self.assertRaises(AssertionError):
            ObjectArtifactStorePolicy(signed_url_max_seconds=0)
        with self.assertRaises(AssertionError):
            ObjectArtifactEncryption(
                key_id="key",
                algorithm="algo",
                metadata={"bad": ""},
            )
        with self.assertRaises(ArtifactStoreError):
            ObjectArtifactMultipartUpload(key="../bad", upload_id="u1")
        with self.assertRaises(AssertionError):
            ObjectArtifactPart(
                part_number=0,
                etag="etag",
                size_bytes=1,
                sha256="0" * 64,
            )
        with self.assertRaises(AssertionError):
            ObjectArtifactHead(
                key="objects/ar/artifact-1",
                size_bytes=1,
                sha256="bad",
            )


def _store(
    client: FakeObjectClient,
    *,
    cipher: XorObjectCipher | None | _DefaultCipher = _DEFAULT_CIPHER,
    policy: ObjectArtifactStorePolicy | None = None,
    id_factory: Callable[[], str] | None = None,
) -> ObjectArtifactStore:
    actual_cipher = (
        XorObjectCipher()
        if isinstance(
            cipher,
            _DefaultCipher,
        )
        else cipher
    )
    return ObjectArtifactStore(
        client,
        cipher=actual_cipher,
        policy=policy or _policy(),
        id_factory=id_factory,
    )


def _policy(
    *,
    part_size: int = 8,
    max_bytes: int = 128,
    max_part_retries: int = 2,
    signed_url_max_seconds: int | None = None,
) -> ObjectArtifactStorePolicy:
    return ObjectArtifactStorePolicy(
        raw_storage_allowed=True,
        retention_days=7,
        max_bytes=max_bytes,
        part_size=part_size,
        max_part_retries=max_part_retries,
        signed_url_max_seconds=signed_url_max_seconds,
    )


async def _task_store() -> InMemoryTaskStore:
    store = InMemoryTaskStore()
    await store.register_definition(_definition(), definition_hash="hash-a")
    return store


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


def _xor(content: bytes) -> bytes:
    return bytes(byte ^ 0xA5 for byte in content)


if __name__ == "__main__":
    main()
