from collections.abc import Callable, ItemsView, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.task import (
    ArtifactStoreError,
    ArtifactStorePolicyError,
    HmacProvider,
    PrivacySanitizationError,
    TaskArtifactPolicy,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactStat,
    TaskArtifactState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskFileMaterializationError,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskLimitsPolicy,
    TaskMaterializedFile,
    TaskMetadata,
    TaskOutputContract,
    TaskProviderReferenceKind,
    TaskRemoteUrlPolicy,
    materialize_task_input_files,
    read_artifact_stream_bytes,
    task_file_descriptors_from_input,
)
from avalan.task import materialization as task_materialization
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.materialization import (
    task_provider_reference_input_files_from_input,
)
from avalan.task.stores import InMemoryTaskStore


class StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"test-secret",
        )


class BrokenHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        raise PrivacySanitizationError("test failure")


class DisappearingDescriptor(Mapping[str, object]):
    def __init__(self) -> None:
        self._data = {
            "source_kind": "local_path",
            "reference": "input.txt",
            "mime_type": "text/plain",
        }

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


class VolatileStructuredInput(Mapping[str, object]):
    def __init__(self) -> None:
        self._remaining_safe_items_calls = 2
        self._data: dict[str, object] = {
            "document": {
                "source_kind": "local_path",
                "reference": "private.txt",
            }
        }

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)

    def items(self) -> ItemsView[str, object]:
        if self._remaining_safe_items_calls <= 0:
            raise RuntimeError("private descriptor traversal failure")
        self._remaining_safe_items_calls -= 1
        return self._data.items()


class VolatileDescriptorLookup(Mapping[str, object]):
    def __init__(self) -> None:
        self._data: dict[str, object] = {
            "source_kind": "local_path",
            "reference": "private.txt",
        }

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: object = None) -> object:
        raise RuntimeError("private descriptor lookup failure")

    def items(self) -> ItemsView[str, object]:
        return self._data.items()


class VolatileKeyInput(Mapping[object, object]):
    def __init__(self) -> None:
        self._remaining_safe_items_calls = 2
        self._safe_data: dict[object, object] = {"safe": "ok"}
        self._unsafe_data: dict[object, object] = {1: "private"}

    def __getitem__(self, key: object) -> object:
        if self._remaining_safe_items_calls > 0:
            return self._safe_data[key]
        return self._unsafe_data[key]

    def __iter__(self) -> Iterator[object]:
        if self._remaining_safe_items_calls > 0:
            return iter(self._safe_data)
        return iter(self._unsafe_data)

    def __len__(self) -> int:
        if self._remaining_safe_items_calls > 0:
            return len(self._safe_data)
        return len(self._unsafe_data)

    def get(self, key: object, default: object = None) -> object:
        return default

    def items(self) -> ItemsView[object, object]:
        if self._remaining_safe_items_calls > 0:
            self._remaining_safe_items_calls -= 1
            return self._safe_data.items()
        return self._unsafe_data.items()


class StatFailingPath:
    def stat(self) -> object:
        raise OSError("private stat failure")


class DirectoryStat:
    st_mode = 0o040000


class FailingReadStream:
    def read(self, size: int = -1) -> bytes:
        _ = size
        raise OSError("private path failure")

    def __enter__(self) -> "FailingReadStream":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = exc_type, exc, traceback


class StreamingOnlyArtifactStore:
    def __init__(self) -> None:
        self.puts = 0
        self.streams: list[tuple[int | None, int | None, str | None]] = []
        self.deleted: list[TaskArtifactRef] = []

    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        _ = content, artifact_id, media_type, metadata
        self.puts += 1
        raise AssertionError("put bytes should not be used")

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
        _ = metadata
        self.streams.append((max_bytes, expected_size_bytes, expected_sha256))
        content = read_artifact_stream_bytes(
            stream,
            max_bytes=max_bytes,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
        artifact_name = artifact_id or f"artifact-{len(self.streams)}"
        return TaskArtifactRef(
            artifact_id=artifact_name,
            store="memory",
            storage_key=f"artifacts/{artifact_name}",
            media_type=media_type,
            size_bytes=len(content),
            sha256=sha256(content).hexdigest(),
        )

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        _ = ref
        return BytesIO()

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BinaryIO:
        _ = max_bytes
        return await self.open(ref)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return TaskArtifactStat(
            ref=ref,
            size_bytes=ref.size_bytes or 0,
            sha256=ref.sha256 or ("0" * 64),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        self.deleted.append(ref)


class PolicyFailingArtifactStore(StreamingOnlyArtifactStore):
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
        _ = (
            artifact_id,
            media_type,
            metadata,
            max_bytes,
            expected_size_bytes,
            expected_sha256,
        )
        while stream.read(2):
            pass
        raise ArtifactStorePolicyError("artifact backend policy failure")


class BackendFailingArtifactStore(StreamingOnlyArtifactStore):
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
        _ = (
            stream,
            artifact_id,
            media_type,
            metadata,
            max_bytes,
            expected_size_bytes,
            expected_sha256,
        )
        raise ArtifactStoreError("artifact backend failed")


class FailingArtifactAppendStore(InMemoryTaskStore):
    async def append_artifact(
        self,
        run_id: str,
        *,
        ref: TaskArtifactRef,
        purpose: TaskArtifactPurpose,
        state: TaskArtifactState | None = None,
        attempt_id: str | None = None,
        provenance: TaskArtifactProvenance | None = None,
        retention: TaskArtifactRetention | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRecord:
        _ = (
            run_id,
            ref,
            purpose,
            state,
            attempt_id,
            provenance,
            retention,
            metadata,
        )
        raise RuntimeError("private artifact metadata failure")


class TaskFileMaterializationTest(IsolatedAsyncioTestCase):
    async def test_non_file_input_returns_no_materialized_files(self) -> None:
        files = await materialize_task_input_files(
            _definition(input_contract=TaskInputContract.string()),
            "plain input",
            roots=(),
            artifact_store=None,
        )

        self.assertEqual(files, ())

    async def test_missing_non_file_input_raises_validation_error(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(input_contract=TaskInputContract.string()),
                None,
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["input.invalid_type"],
        )

    async def test_invalid_file_array_shape_returns_validation_error(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(input_contract=TaskInputContract.file_array()),
                "private scalar path",
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["input.invalid_type"],
        )
        self.assertNotIn("private scalar path", str(error.exception))

    async def test_provider_reference_input_skips_byte_materialization(
        self,
    ) -> None:
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-openai",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            mime_type="application/pdf",
            owner_scope="tenant-a",
            identity_hmac="hmac-value",
        )

        materialized = await materialize_task_input_files(
            _definition(input_contract=TaskInputContract.file()),
            descriptor,
            roots=(),
            artifact_store=None,
        )
        files = task_provider_reference_input_files_from_input(
            _definition(input_contract=TaskInputContract.file()),
            descriptor,
        )

        self.assertEqual(materialized, ())
        self.assertEqual(len(files), 1)
        self.assertEqual(
            files[0].logical_path, "provider:openai:provider_file_id"
        )
        self.assertIsNotNone(files[0].provider_reference)
        assert files[0].provider_reference is not None
        self.assertEqual(files[0].provider_reference.reference, "file-openai")

    async def test_provider_reference_only_input_preserves_count_errors(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(
                    input_contract=TaskInputContract.file_array(),
                    limits=TaskLimitsPolicy(file_count=1),
                ),
                [
                    TaskFileDescriptor.provider_reference_descriptor(
                        "file-a",
                        kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                        provider="openai",
                    ),
                    TaskFileDescriptor.provider_reference_descriptor(
                        "file-b",
                        kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                        provider="openai",
                    ),
                ],
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input")
        self.assertNotIn("file-a", str(error.exception))

    def test_provider_reference_extraction_rejects_expired_values(
        self,
    ) -> None:
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "https://example.test/private",
            kind=TaskProviderReferenceKind.EXPIRING_PROVIDER_HANDLE,
            provider="openai",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
            durable=False,
        )

        with self.assertRaises(TaskFileMaterializationError) as error:
            task_provider_reference_input_files_from_input(
                _definition(input_contract=TaskInputContract.file()),
                descriptor,
                now=datetime.now(UTC),
            )

        self.assertEqual(
            error.exception.issues[0].path,
            "input.provider_reference.expires_at",
        )
        self.assertNotIn("example.test", str(error.exception))

    def test_provider_reference_extraction_rejects_missing_reference(
        self,
    ) -> None:
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-openai",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
        )
        object.__setattr__(descriptor, "provider_reference", None)

        with self.assertRaises(TaskFileMaterializationError) as error:
            task_provider_reference_input_files_from_input(
                _definition(input_contract=TaskInputContract.file()),
                descriptor,
            )

        self.assertEqual(
            error.exception.issues[0].path, "input.provider_reference"
        )

    def test_provider_reference_mapping_extracts_datetime_shapes(self) -> None:
        expires_at = datetime.now(UTC) + timedelta(minutes=5)
        string_file = task_provider_reference_input_files_from_input(
            _definition(input_contract=TaskInputContract.file()),
            {
                "source_kind": "provider_reference",
                "reference": "https://example.test/private",
                "provider_reference": {
                    "kind": "hosted_url",
                    "provider": "openai",
                    "reference": "https://example.test/private",
                    "expires_at": expires_at.isoformat(),
                    "durable": False,
                },
            },
        )[0]
        datetime_file = task_provider_reference_input_files_from_input(
            _definition(input_contract=TaskInputContract.file()),
            {
                "source_kind": "provider_reference",
                "reference": "file-openai",
                "provider_reference": {
                    "kind": TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    "provider": "openai",
                    "reference": "file-openai",
                    "expires_at": expires_at,
                    "durable": False,
                },
            },
        )[0]
        object_reference = TaskFileDescriptor.provider_reference_descriptor(
            "file-object",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
        ).provider_reference
        object_file = task_provider_reference_input_files_from_input(
            _definition(input_contract=TaskInputContract.file()),
            {
                "source_kind": "provider_reference",
                "reference": "file-object",
                "provider_reference": object_reference,
            },
        )[0]

        self.assertIsNotNone(string_file.provider_reference)
        self.assertIsNotNone(datetime_file.provider_reference)
        self.assertIsNotNone(object_file.provider_reference)
        assert string_file.provider_reference is not None
        assert datetime_file.provider_reference is not None
        assert object_file.provider_reference is not None
        self.assertFalse(string_file.provider_reference.durable_for_queue)
        self.assertFalse(datetime_file.provider_reference.durable_for_queue)
        self.assertTrue(object_file.provider_reference.durable_for_queue)

    async def test_local_file_materializes_to_artifact_backend(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            input_path = Path(root, "uploads", "input.txt")
            input_path.parent.mkdir()
            input_path.write_bytes(b"private text")
            store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-1",
            )

            files = await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.local_path(
                    "uploads/input.txt",
                    mime_type="text/plain",
                    metadata={"filename": "input.txt"},
                ),
                roots=(root,),
                artifact_store=store,
                hmac_provider=cast(HmacProvider, StaticHmacProvider()),
            )

            self.assertEqual(len(files), 1)
            materialized = files[0]
            self.assertIsInstance(materialized, TaskMaterializedFile)
            self.assertEqual(materialized.ref.artifact_id, "artifact-1")
            self.assertEqual(materialized.ref.media_type, "text/plain")
            self.assertEqual(materialized.ref.size_bytes, 12)
            self.assertEqual(materialized.identity["privacy"], "<hmac-sha256>")
            target_file = materialized.as_input_file()
            self.assertEqual(target_file.logical_path, "artifact:artifact-1")
            self.assertNotIn("input.txt", str(materialized.identity))
            reader = await store.open(materialized.ref)
            try:
                self.assertEqual(reader.read(), b"private text")
            finally:
                reader.close()

    async def test_local_file_materialization_uses_streaming_store_contract(
        self,
    ) -> None:
        with TemporaryDirectory() as root:
            Path(root, "input.txt").write_bytes(b"private text")
            store = StreamingOnlyArtifactStore()

            files = await materialize_task_input_files(
                _definition(limits=TaskLimitsPolicy(file_bytes=20)),
                TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                    size_bytes=12,
                    sha256=sha256(b"private text").hexdigest(),
                ),
                roots=(root,),
                artifact_store=store,
            )

        self.assertEqual(store.puts, 0)
        self.assertEqual(
            store.streams,
            [(20, 12, sha256(b"private text").hexdigest())],
        )
        self.assertEqual(files[0].ref.artifact_id, "artifact-1")
        self.assertEqual(files[0].ref.size_bytes, 12)
        self.assertEqual(
            files[0].ref.sha256, sha256(b"private text").hexdigest()
        )

    async def test_streaming_policy_failure_returns_size_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as root:
            Path(root, "input.txt").write_bytes(b"private text")

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    _definition(limits=TaskLimitsPolicy(file_bytes=20)),
                    TaskFileDescriptor.local_path(
                        "input.txt",
                        mime_type="text/plain",
                        size_bytes=12,
                    ),
                    roots=(root,),
                    artifact_store=PolicyFailingArtifactStore(),
                )

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.size_bytes"],
        )
        self.assertNotIn("private text", str(error.exception))

    async def test_backend_write_failure_is_not_recast_as_input_mismatch(
        self,
    ) -> None:
        with TemporaryDirectory() as root:
            Path(root, "input.txt").write_bytes(b"private text")

            with self.assertRaises(ArtifactStoreError) as error:
                await materialize_task_input_files(
                    _definition(limits=TaskLimitsPolicy(file_bytes=20)),
                    TaskFileDescriptor.local_path(
                        "input.txt",
                        mime_type="text/plain",
                        size_bytes=12,
                    ),
                    roots=(root,),
                    artifact_store=BackendFailingArtifactStore(),
                )

        self.assertNotIn("input.txt", str(error.exception))
        self.assertNotIn("private text", str(error.exception))

    async def test_mapping_descriptor_materializes_with_redacted_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")
            store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-1",
            )

            files = await materialize_task_input_files(
                _definition(
                    input_contract=TaskInputContract.file(
                        conversions=("markdown",),
                        mime_types=("text/plain",),
                    )
                ),
                {
                    "source_kind": "local_path",
                    "reference": "input.txt",
                    "role": "source",
                    "mime_type": "text/plain",
                    "size_bytes": 12,
                    "sha256": sha256(b"private text").hexdigest(),
                    "conversions": [
                        "markdown",
                        TaskFileConversionRequest(name="markdown"),
                        {"name": "markdown", "options": {"strict": True}},
                    ],
                    "metadata": {"caller": "private"},
                },
                roots=(root,),
                artifact_store=store,
            )

            self.assertEqual(files[0].identity, {"privacy": "<redacted>"})

    async def test_structured_input_materializes_nested_file_descriptors(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "report-a.txt").write_bytes(b"private report a")
            Path(root, "report-b.txt").write_bytes(b"private report b")
            artifact_ids = iter(("artifact-a", "artifact-b"))
            store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            definition = _definition(input_contract=_object_contract())

            files = await materialize_task_input_files(
                definition,
                {
                    "question": "Summarize the reports.",
                    "document": {
                        "source_kind": "local_path",
                        "reference": "report-a.txt",
                        "mime_type": "text/plain",
                    },
                    "attachments": [
                        {
                            "source_kind": "local_path",
                            "reference": "report-b.txt",
                            "mime_type": "text/plain",
                        }
                    ],
                },
                roots=(root,),
                artifact_store=store,
                hmac_provider=cast(HmacProvider, StaticHmacProvider()),
            )

            self.assertEqual(
                [file.descriptor_path for file in files],
                ["input.document", "input.attachments[0]"],
            )
            self.assertEqual(
                [file.ref.artifact_id for file in files],
                ["artifact-a", "artifact-b"],
            )
            self.assertNotIn("report-a.txt", str(files[0].identity))
            self.assertNotIn("private report", str(files[1].identity))

    async def test_structured_input_rejects_invalid_descriptors_safely(
        self,
    ) -> None:
        cases = (
            (
                {"source_kind": 1, "reference": "private.txt"},
                "input.document.source_kind",
            ),
            (
                {"source_kind": "unknown", "reference": "private.txt"},
                "input.document.source_kind",
            ),
            (
                {"source_kind": "local_path", "reference": ""},
                "input.document.reference",
            ),
            (
                {
                    "source_kind": "local_path",
                    "reference": "private.txt",
                    "conversions": "text",
                },
                "input.document.conversions",
            ),
            (
                {
                    "source_kind": "local_path",
                    "reference": "private.txt",
                    "role": 1,
                },
                "input.document",
            ),
        )

        for descriptor, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(input_contract=_object_contract()),
                        {"document": descriptor},
                        roots=(),
                        artifact_store=None,
                    )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                [path],
            )
            self.assertNotIn("private.txt", str(error.exception))

    async def test_hmac_failure_falls_back_to_redacted_identity(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")

            files = await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                ),
                roots=(root,),
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-1",
                ),
                hmac_provider=cast(HmacProvider, BrokenHmacProvider()),
            )

            self.assertEqual(files[0].identity, {"privacy": "<redacted>"})

    async def test_materialization_can_persist_sanitized_artifact_record(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-1",
            )
            task_store = InMemoryTaskStore(id_factory=_id_factory())
            definition = _definition()
            await task_store.register_definition(
                definition,
                definition_hash="definition-1",
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="definition-1")
            )

            await materialize_task_input_files(
                definition,
                TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                ),
                roots=(root,),
                artifact_store=artifact_store,
                hmac_provider=cast(HmacProvider, StaticHmacProvider()),
                task_store=task_store,
                run_id=run.run_id,
            )

            records = await task_store.list_artifacts(
                run.run_id,
                purpose=TaskArtifactPurpose.INPUT,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].artifact_id, "artifact-1")
            self.assertNotIn("input.txt", str(records[0].summary()))

    async def test_metadata_append_failure_deletes_uploaded_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as root:
            Path(root, "input.txt").write_bytes(b"private text")
            artifact_store = StreamingOnlyArtifactStore()
            task_store = FailingArtifactAppendStore(id_factory=_id_factory())
            definition = _definition()
            await task_store.register_definition(
                definition,
                definition_hash="definition-1",
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="definition-1")
            )

            with self.assertRaises(RuntimeError) as error:
                await materialize_task_input_files(
                    definition,
                    TaskFileDescriptor.local_path(
                        "input.txt",
                        mime_type="text/plain",
                    ),
                    roots=(root,),
                    artifact_store=artifact_store,
                    hmac_provider=cast(HmacProvider, StaticHmacProvider()),
                    task_store=task_store,
                    run_id=run.run_id,
                )

        self.assertEqual(len(artifact_store.deleted), 1)
        self.assertEqual(artifact_store.deleted[0].artifact_id, "artifact-1")
        self.assertNotIn("input.txt", str(error.exception))

    async def test_malformed_descriptor_returns_existing_validation_issues(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(),
                {"reference": "input.txt"},
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.source_kind", "input.mime_type"],
        )

    async def test_descriptor_coercion_failure_returns_safe_diagnostic(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(),
                DisappearingDescriptor(),
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [issue.path for issue in error.exception.issues], ["input"]
        )
        self.assertNotIn("input.txt", str(error.exception))

    async def test_structured_descriptor_traversal_failure_is_safe(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(input_contract=_object_contract()),
                VolatileStructuredInput(),
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [(issue.code, issue.path) for issue in error.exception.issues],
            [("input.invalid_file", "input")],
        )
        self.assertNotIn("private", str(error.exception))

    async def test_structured_descriptor_lookup_failure_is_safe(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(input_contract=_object_contract()),
                {"document": VolatileDescriptorLookup()},
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [(issue.code, issue.path) for issue in error.exception.issues],
            [("input.invalid_file", "input.document")],
        )
        self.assertNotIn("private", str(error.exception))

    async def test_structured_descriptor_non_string_key_is_safe(
        self,
    ) -> None:
        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(input_contract=_object_contract()),
                VolatileKeyInput(),
                roots=(),
                artifact_store=None,
            )

        self.assertEqual(
            [(issue.code, issue.path) for issue in error.exception.issues],
            [("input.invalid_file", "input")],
        )
        self.assertNotIn("private", str(error.exception))

    async def test_missing_artifact_backend_fails_before_path_access(
        self,
    ) -> None:
        with TemporaryDirectory() as root:
            descriptor = TaskFileDescriptor.local_path(
                "missing-secret.txt",
                mime_type="text/plain",
            )

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    _definition(),
                    descriptor,
                    roots=(root,),
                    artifact_store=None,
                )

            self.assertEqual(
                [issue.code for issue in error.exception.issues],
                ["artifact.bytes_unsupported"],
            )
            self.assertNotIn("missing-secret", str(error.exception))

    async def test_remote_source_is_rejected_by_default_before_fetch(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            store = LocalArtifactStore(artifacts, raw_storage_allowed=True)

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    _definition(),
                    TaskFileDescriptor.remote_url(
                        "https://example.test/private.txt",
                        mime_type="text/plain",
                    ),
                    roots=(),
                    artifact_store=store,
                )

            self.assertEqual(
                [issue.code for issue in error.exception.issues],
                ["feature.remote_url_file_inputs_disabled"],
            )
            self.assertNotIn("example.test", str(error.exception))
            self.assertNotIn("private.txt", str(error.exception))

    async def test_enabled_remote_source_is_deferred_without_fetcher(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            store = LocalArtifactStore(artifacts, raw_storage_allowed=True)

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    _definition(),
                    TaskFileDescriptor.remote_url(
                        "https://example.test/private.txt",
                        mime_type="text/plain",
                    ),
                    roots=(),
                    artifact_store=store,
                    remote_url_policy=TaskRemoteUrlPolicy(
                        enabled=True,
                        max_bytes=100,
                    ),
                )

            self.assertEqual(
                [issue.code for issue in error.exception.issues],
                ["input.invalid_file"],
            )
            self.assertNotIn("example.test", str(error.exception))

    async def test_unsupported_source_and_missing_roots_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            store = LocalArtifactStore(artifacts, raw_storage_allowed=True)

            for descriptor in (
                TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                ),
            ):
                with self.subTest(source_kind=descriptor.source_kind.value):
                    with self.assertRaises(
                        TaskFileMaterializationError
                    ) as error:
                        await materialize_task_input_files(
                            _definition(),
                            descriptor,
                            roots=(),
                            artifact_store=store,
                        )

                    self.assertEqual(
                        [issue.code for issue in error.exception.issues],
                        ["input.invalid_file"],
                    )

    async def test_traversal_and_symlink_escape_fail_closed(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as outside:
            Path(outside, "secret.txt").write_bytes(b"private text")
            Path(root, "link").symlink_to(outside, target_is_directory=True)
            store = LocalArtifactStore(root, raw_storage_allowed=True)

            for descriptor in (
                TaskFileDescriptor.local_path(
                    "../outside/secret.txt",
                    mime_type="text/plain",
                ),
                TaskFileDescriptor.local_path(
                    "..\\outside\\secret.txt",
                    mime_type="text/plain",
                ),
                TaskFileDescriptor.local_path(
                    str(Path(outside, "secret.txt")),
                    mime_type="text/plain",
                ),
                TaskFileDescriptor.local_path(
                    "safe\x00secret.txt",
                    mime_type="text/plain",
                ),
                TaskFileDescriptor.local_path(
                    "link/secret.txt",
                    mime_type="text/plain",
                ),
            ):
                with self.subTest(reference=descriptor.reference):
                    with self.assertRaises(
                        TaskFileMaterializationError
                    ) as error:
                        await materialize_task_input_files(
                            _definition(),
                            descriptor,
                            roots=(root,),
                            artifact_store=store,
                        )

                    self.assertEqual(
                        [issue.code for issue in error.exception.issues],
                        ["input.invalid_file"],
                    )
                    self.assertNotIn("secret.txt", str(error.exception))

    async def test_symlink_swap_after_validation_is_rejected(self) -> None:
        with (
            TemporaryDirectory() as root,
            TemporaryDirectory() as outside,
            TemporaryDirectory() as artifacts,
        ):
            input_path = Path(root, "secret.txt")
            outside_path = Path(outside, "secret.txt")
            input_path.write_bytes(b"safe text")
            outside_path.write_bytes(b"private text")
            open_stream = task_materialization._open_regular_file_stream

            def mutate_before_open(path: Path) -> BinaryIO:
                input_path.unlink()
                input_path.symlink_to(outside_path)
                return open_stream(path)

            with patch(
                "avalan.task.materialization._open_regular_file_stream",
                side_effect=mutate_before_open,
            ):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.local_path(
                            "secret.txt",
                            mime_type="text/plain",
                        ),
                        roots=(root,),
                        artifact_store=LocalArtifactStore(
                            artifacts,
                            raw_storage_allowed=True,
                            id_factory=lambda: "artifact-1",
                        ),
                    )

            self.assertEqual(
                [(issue.code, issue.path) for issue in error.exception.issues],
                [("input.invalid_file", "input.reference")],
            )
            self.assertFalse(
                any(path.is_file() for path in Path(artifacts).rglob("*"))
            )
            self.assertNotIn("secret.txt", str(error.exception))
            self.assertNotIn("private text", str(error.exception))

    async def test_structured_input_traversal_fails_closed(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as outside:
            Path(outside, "secret.txt").write_bytes(b"private text")
            store = LocalArtifactStore(root, raw_storage_allowed=True)

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    _definition(input_contract=_object_contract()),
                    {
                        "documents": [
                            {
                                "source_kind": "local_path",
                                "reference": "../outside/secret.txt",
                            }
                        ]
                    },
                    roots=(root,),
                    artifact_store=store,
                )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input.documents[0].reference"],
            )
            self.assertNotIn("secret.txt", str(error.exception))

    async def test_missing_local_path_fails_without_leaking_reference(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    _definition(),
                    TaskFileDescriptor.local_path(
                        "missing-secret.txt",
                        mime_type="text/plain",
                    ),
                    roots=(root,),
                    artifact_store=LocalArtifactStore(
                        artifacts,
                        raw_storage_allowed=True,
                    ),
                )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input.reference"],
            )
            self.assertNotIn("missing-secret", str(error.exception))

    async def test_declared_size_and_digest_are_enforced(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "secret.txt").write_bytes(b"private text")
            store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
            )

            files = await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.local_path(
                    "secret.txt",
                    mime_type="text/plain",
                    size_bytes=12,
                    sha256=sha256(b"private text").hexdigest(),
                ),
                roots=(root,),
                artifact_store=store,
            )

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].ref.size_bytes, 12)

            cases = (
                (
                    TaskFileDescriptor.local_path(
                        "secret.txt",
                        mime_type="text/plain",
                        size_bytes=11,
                    ),
                    "input.size_bytes",
                ),
                (
                    TaskFileDescriptor.local_path(
                        "secret.txt",
                        mime_type="text/plain",
                        sha256="0" * 64,
                    ),
                    "input.sha256",
                ),
            )

            for descriptor, path in cases:
                with self.subTest(path=path):
                    with self.assertRaises(
                        TaskFileMaterializationError
                    ) as error:
                        await materialize_task_input_files(
                            _definition(),
                            descriptor,
                            roots=(root,),
                            artifact_store=store,
                        )

                    self.assertEqual(
                        [issue.path for issue in error.exception.issues],
                        [path],
                    )
                    self.assertNotIn("secret.txt", str(error.exception))
                    self.assertNotIn("private text", str(error.exception))

    async def test_file_growth_after_validation_is_rejected(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            input_path = Path(root, "secret.txt")
            input_path.write_bytes(b"1234")
            open_stream = task_materialization._open_regular_file_stream

            def mutate_before_open(path: Path) -> BinaryIO:
                input_path.write_bytes(b"12345-private")
                return open_stream(path)

            with patch(
                "avalan.task.materialization._open_regular_file_stream",
                side_effect=mutate_before_open,
            ):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(
                            limits=TaskLimitsPolicy(file_bytes=4),
                        ),
                        TaskFileDescriptor.local_path(
                            "secret.txt",
                            mime_type="text/plain",
                        ),
                        roots=(root,),
                        artifact_store=LocalArtifactStore(
                            artifacts,
                            raw_storage_allowed=True,
                            id_factory=lambda: "artifact-1",
                        ),
                    )

            self.assertEqual(
                [(issue.code, issue.path) for issue in error.exception.issues],
                [("input.invalid_file", "input.size_bytes")],
            )
            self.assertFalse(
                any(path.is_file() for path in Path(artifacts).rglob("*"))
            )
            self.assertNotIn("secret.txt", str(error.exception))
            self.assertNotIn("12345-private", str(error.exception))

    async def test_declared_size_is_rechecked_after_validation(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            input_path = Path(root, "secret.txt")
            input_path.write_bytes(b"1234")
            open_stream = task_materialization._open_regular_file_stream

            def mutate_before_open(path: Path) -> BinaryIO:
                input_path.write_bytes(b"12345")
                return open_stream(path)

            with patch(
                "avalan.task.materialization._open_regular_file_stream",
                side_effect=mutate_before_open,
            ):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.local_path(
                            "secret.txt",
                            mime_type="text/plain",
                            size_bytes=4,
                        ),
                        roots=(root,),
                        artifact_store=LocalArtifactStore(
                            artifacts,
                            raw_storage_allowed=True,
                            id_factory=lambda: "artifact-1",
                        ),
                    )

            self.assertEqual(
                [(issue.code, issue.path) for issue in error.exception.issues],
                [("input.invalid_file", "input.size_bytes")],
            )
            self.assertFalse(
                any(path.is_file() for path in Path(artifacts).rglob("*"))
            )
            self.assertNotIn("secret.txt", str(error.exception))
            self.assertNotIn("12345", str(error.exception))

    async def test_read_failure_returns_safe_diagnostic(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "secret.txt").write_bytes(b"private text")

            with patch(
                "avalan.task.materialization._open_regular_file_stream",
                return_value=FailingReadStream(),
            ):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.local_path(
                            "secret.txt",
                            mime_type="text/plain",
                        ),
                        roots=(root,),
                        artifact_store=LocalArtifactStore(
                            artifacts,
                            raw_storage_allowed=True,
                        ),
                    )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input.reference"],
            )
            self.assertNotIn("secret.txt", str(error.exception))
            self.assertNotIn("private path failure", str(error.exception))

    async def test_non_regular_file_read_returns_safe_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "secret.txt").write_bytes(b"private text")

            with patch(
                "avalan.task.materialization.fstat",
                return_value=DirectoryStat(),
            ):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.local_path(
                            "secret.txt",
                            mime_type="text/plain",
                        ),
                        roots=(root,),
                        artifact_store=LocalArtifactStore(
                            artifacts,
                            raw_storage_allowed=True,
                        ),
                    )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input.reference"],
            )
            self.assertNotIn("secret.txt", str(error.exception))
            self.assertNotIn("private text", str(error.exception))

    async def test_stat_failure_returns_safe_diagnostic(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "secret.txt").write_bytes(b"private text")

            with patch(
                "avalan.task.materialization._resolve_within_root",
                return_value=StatFailingPath(),
            ):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.local_path(
                            "secret.txt",
                            mime_type="text/plain",
                        ),
                        roots=(root,),
                        artifact_store=LocalArtifactStore(
                            artifacts,
                            raw_storage_allowed=True,
                        ),
                    )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input.reference"],
            )
            self.assertNotIn("secret.txt", str(error.exception))
            self.assertNotIn("private stat failure", str(error.exception))

    async def test_actual_count_and_size_limits_are_enforced(self) -> None:
        with TemporaryDirectory() as root:
            Path(root, "a.txt").write_bytes(b"12345")
            Path(root, "b.txt").write_bytes(b"67890")
            store = LocalArtifactStore(root, raw_storage_allowed=True)
            definition = _definition(
                input_contract=TaskInputContract.file_array(),
                limits=TaskLimitsPolicy(file_count=1, file_bytes=4),
            )

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    definition,
                    [
                        TaskFileDescriptor.local_path("a.txt"),
                        TaskFileDescriptor.local_path("b.txt"),
                    ],
                    roots=(root,),
                    artifact_store=store,
                )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input", "input[0].size_bytes", "input[1].size_bytes"],
            )

    async def test_artifact_count_limit_is_enforced(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "a.txt").write_bytes(b"123")
            Path(root, "b.txt").write_bytes(b"456")
            definition = _definition(
                input_contract=TaskInputContract.file_array(),
                artifact=TaskArtifactPolicy(max_count=1),
            )

            with self.assertRaises(TaskFileMaterializationError) as error:
                await materialize_task_input_files(
                    definition,
                    [
                        TaskFileDescriptor.local_path("a.txt"),
                        TaskFileDescriptor.local_path("b.txt"),
                    ],
                    roots=(root,),
                    artifact_store=LocalArtifactStore(
                        artifacts,
                        raw_storage_allowed=True,
                    ),
                )

            self.assertEqual(
                [issue.path for issue in error.exception.issues],
                ["input"],
            )

    def test_descriptor_extraction_matches_contract_shape(self) -> None:
        descriptors = task_file_descriptors_from_input(
            _definition(input_contract=TaskInputContract.file_array()),
            [
                {"source_kind": "local_path", "reference": "a.txt"},
                TaskFileDescriptor.local_path("b.txt"),
            ],
        )

        self.assertEqual(
            [descriptor.reference for descriptor in descriptors],
            ["a.txt", "b.txt"],
        )

    def test_descriptor_extraction_rejects_invalid_descriptor(self) -> None:
        with self.assertRaises(ValueError):
            task_file_descriptors_from_input(
                _definition(),
                {"source_kind": "local_path", "reference": ""},
            )

    def test_descriptor_extraction_collects_structured_files(self) -> None:
        descriptors = task_file_descriptors_from_input(
            _definition(input_contract=_object_contract()),
            {
                "prompt": "Compare inputs.",
                "primary": {
                    "source_kind": "local_path",
                    "reference": "a.txt",
                },
                "nested": {
                    "attachments": [
                        {
                            "source_kind": "local_path",
                            "reference": "b.txt",
                        }
                    ]
                },
            },
        )

        self.assertEqual(
            [descriptor.reference for descriptor in descriptors],
            ["a.txt", "b.txt"],
        )


def _definition(
    *,
    input_contract: TaskInputContract | None = None,
    artifact: TaskArtifactPolicy | None = None,
    limits: TaskLimitsPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="file_task", version="1"),
        input=input_contract
        or TaskInputContract.file(mime_types=("text/plain",)),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/file_task.toml"),
        artifact=artifact or TaskArtifactPolicy(),
        limits=limits or TaskLimitsPolicy(),
    )


def _object_contract() -> TaskInputContract:
    return TaskInputContract.object(schema={"type": "object"})


def _id_factory() -> Callable[[], str]:
    values = iter(
        (
            "run-1",
            "transition-1",
            "attempt-1",
            "artifact-record-1",
        )
    )
    return lambda: next(values)


if __name__ == "__main__":
    main()
