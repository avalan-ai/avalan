from collections.abc import Callable, ItemsView, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch
from urllib.error import HTTPError, URLError

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
    TaskValidationCategory,
    TaskValidationIssue,
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


class TimeoutReadStream:
    def __init__(self) -> None:
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        _ = size
        raise TimeoutError("private timeout detail")

    def close(self) -> None:
        self.closed = True


class FakeRemoteResolver:
    def __init__(
        self,
        addresses: Mapping[str, tuple[str, ...] | OSError] | None = None,
    ) -> None:
        self.addresses = dict(addresses or {})
        self.calls: list[str] = []

    def resolve(self, hostname: str) -> tuple[str, ...]:
        self.calls.append(hostname)
        value = self.addresses.get(hostname, ("8.8.8.8",))
        if isinstance(value, OSError):
            raise value
        return value


class FakeRemoteClient:
    def __init__(
        self,
        responses: Mapping[
            str,
            task_materialization.TaskRemoteUrlResponse | Exception,
        ],
    ) -> None:
        self.responses = dict(responses)
        self.calls: list[tuple[str, float]] = []

    def open(
        self,
        url: str,
        *,
        timeout_seconds: float,
    ) -> task_materialization.TaskRemoteUrlResponse:
        self.calls.append((url, timeout_seconds))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


class FakeHttpResponse(BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        status: int,
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(content)
        self.status = status
        self.headers = dict(headers)


class FakeOpener:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[object, float]] = []

    def open(self, request: object, *, timeout: float) -> object:
        self.calls.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _remote_response(
    content: bytes,
    *,
    status_code: int = 200,
    headers: Mapping[str, str] | None = None,
) -> task_materialization.TaskRemoteUrlResponse:
    return task_materialization.TaskRemoteUrlResponse(
        status_code=status_code,
        headers=headers
        or {
            "Content-Length": str(len(content)),
            "Content-Type": "text/plain",
        },
        stream=BytesIO(content),
    )


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
            metadata={
                "filename": "private.pdf",
                "url": "https://private.example.test/raw",
            },
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
        self.assertEqual(
            files[0].metadata["metadata"], {"privacy": "<redacted>"}
        )
        self.assertNotIn("private.pdf", str(files[0].summary()))
        self.assertNotIn("private.example", str(files[0].summary()))

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

    async def test_mixed_file_array_preserves_descriptor_order(self) -> None:
        content = b"remote text"
        client = FakeRemoteClient(
            {
                "https://example.test/private.txt": _remote_response(content),
            }
        )
        store = StreamingOnlyArtifactStore()
        with TemporaryDirectory() as root:
            Path(root, "local.txt").write_bytes(b"local text")

            files = await materialize_task_input_files(
                _definition(input_contract=TaskInputContract.file_array()),
                [
                    TaskFileDescriptor.remote_url(
                        "https://example.test/private.txt",
                        mime_type="text/plain",
                    ),
                    TaskFileDescriptor.local_path(
                        "local.txt",
                        mime_type="text/plain",
                    ),
                ],
                roots=(root,),
                artifact_store=store,
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    max_bytes=100,
                ),
                remote_url_http_client=client,
                remote_url_resolver=FakeRemoteResolver(),
            )

        self.assertEqual(
            [file.descriptor_path for file in files],
            ["input[0]", "input[1]"],
        )
        self.assertEqual(
            [file.ref.artifact_id for file in files],
            ["artifact-1", "artifact-2"],
        )
        self.assertEqual(
            [file.ref.metadata["source_kind"] for file in files],
            ["remote_url", "local_path"],
        )
        self.assertEqual(
            client.calls,
            [("https://example.test/private.txt", 10.0)],
        )
        self.assertNotIn("private.txt", str(files[0].ref.metadata))
        self.assertNotIn("local.txt", str(files[1].ref.metadata))

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
                "input.document.role",
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

    def test_duplicate_materialization_issues_are_collapsed(self) -> None:
        issue = TaskValidationIssue(
            code="input.invalid_file",
            path="input.document.source_kind",
            message="Task file source kind is invalid.",
            hint="Use a supported file source kind.",
            category=TaskValidationCategory.VALUE,
        )
        other = TaskValidationIssue(
            code="input.invalid_file",
            path="input.document.reference",
            message="Task file reference is invalid.",
            hint="Pass a non-empty file reference.",
            category=TaskValidationCategory.VALUE,
        )

        deduplicated = task_materialization._deduplicate_issues(
            (
                issue,
                issue,
                other,
            )
        )

        self.assertEqual(deduplicated, [issue, other])

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

    async def test_enabled_remote_source_rejects_unresolved_host_before_fetch(
        self,
    ) -> None:
        store = StreamingOnlyArtifactStore()
        client = FakeRemoteClient(
            {"https://example.test/private.txt": _remote_response(b"private")}
        )

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
                remote_url_http_client=client,
                remote_url_resolver=FakeRemoteResolver(
                    {"example.test": OSError("private dns failure")}
                ),
            )

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.reference"],
        )
        self.assertEqual(client.calls, [])
        self.assertNotIn("example.test", str(error.exception))

    async def test_remote_source_materializes_with_sanitized_metadata(
        self,
    ) -> None:
        content = b"remote text"
        store = StreamingOnlyArtifactStore()
        client = FakeRemoteClient(
            {
                "https://example.test/private.txt": _remote_response(
                    content,
                    headers={
                        "Content-Length": str(len(content)),
                        "Content-Type": "text/plain; charset=utf-8",
                    },
                )
            }
        )

        files = await materialize_task_input_files(
            _definition(),
            TaskFileDescriptor.remote_url(
                "https://example.test/private.txt",
                mime_type="text/plain",
            ),
            roots=(),
            artifact_store=store,
            hmac_provider=cast(HmacProvider, StaticHmacProvider()),
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                max_bytes=100,
                timeout_seconds=3.5,
            ),
            remote_url_http_client=client,
            remote_url_resolver=FakeRemoteResolver(),
        )

        self.assertEqual(len(files), 1)
        self.assertEqual(
            client.calls, [("https://example.test/private.txt", 3.5)]
        )
        self.assertEqual(store.streams, [(100, len(content), None)])
        self.assertEqual(files[0].ref.media_type, "text/plain")
        self.assertEqual(
            files[0].ref.metadata["remote_delivery"],
            "avalan_fetched_url",
        )
        self.assertEqual(files[0].ref.metadata["source_kind"], "remote_url")
        self.assertNotIn("example.test", str(files[0].ref.metadata))
        self.assertNotIn("private.txt", str(files[0].ref.metadata))

    async def test_remote_source_materializes_without_response_mime_hint(
        self,
    ) -> None:
        store = StreamingOnlyArtifactStore()

        files = await materialize_task_input_files(
            _definition(input_contract=TaskInputContract.file(mime_types=())),
            TaskFileDescriptor.remote_url(
                "https://example.test/input.bin",
            ),
            roots=(),
            artifact_store=store,
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                max_bytes=100,
            ),
            remote_url_http_client=FakeRemoteClient(
                {
                    "https://example.test/input.bin": _remote_response(
                        b"bin",
                        headers={"Content-Length": "3"},
                    )
                }
            ),
            remote_url_resolver=FakeRemoteResolver(),
        )

        self.assertEqual(len(files), 1)
        self.assertIsNone(files[0].ref.media_type)

    async def test_remote_source_appends_artifact_metadata(self) -> None:
        store = StreamingOnlyArtifactStore()
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
            TaskFileDescriptor.remote_url(
                "https://example.test/private.txt",
                mime_type="text/plain",
            ),
            roots=(),
            artifact_store=store,
            hmac_provider=cast(HmacProvider, StaticHmacProvider()),
            remote_url_policy=TaskRemoteUrlPolicy(
                enabled=True,
                max_bytes=100,
            ),
            remote_url_http_client=FakeRemoteClient(
                {"https://example.test/private.txt": _remote_response(b"text")}
            ),
            remote_url_resolver=FakeRemoteResolver(),
            task_store=task_store,
            run_id=run.run_id,
        )

        records = await task_store.list_artifacts(
            run.run_id,
            purpose=TaskArtifactPurpose.INPUT,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].provenance.metadata["remote_delivery"],
            "avalan_fetched_url",
        )
        self.assertNotIn("example.test", str(records[0].summary()))

    async def test_remote_source_validation_rejects_before_fetch(self) -> None:
        store = StreamingOnlyArtifactStore()
        client = FakeRemoteClient({})

        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.remote_url(
                    "ftp://example.test/private.txt",
                    mime_type="text/plain",
                ),
                roots=(),
                artifact_store=store,
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    max_bytes=100,
                ),
                remote_url_http_client=client,
                remote_url_resolver=FakeRemoteResolver(),
            )

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.reference"],
        )
        self.assertEqual(client.calls, [])
        self.assertNotIn("private.txt", str(error.exception))

    async def test_remote_source_rejects_private_dns_before_fetch(
        self,
    ) -> None:
        store = StreamingOnlyArtifactStore()
        client = FakeRemoteClient(
            {"https://example.test/private.txt": _remote_response(b"private")}
        )

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
                remote_url_http_client=client,
                remote_url_resolver=FakeRemoteResolver(
                    {"example.test": ("10.0.0.1",)}
                ),
            )

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.reference"],
        )
        self.assertEqual(client.calls, [])
        self.assertNotIn("10.0.0.1", str(error.exception))

    async def test_remote_source_rejects_empty_dns_answers_before_fetch(
        self,
    ) -> None:
        client = FakeRemoteClient(
            {"https://example.test/private.txt": _remote_response(b"private")}
        )

        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/private.txt",
                    mime_type="text/plain",
                ),
                roots=(),
                artifact_store=StreamingOnlyArtifactStore(),
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    max_bytes=100,
                ),
                remote_url_http_client=client,
                remote_url_resolver=FakeRemoteResolver({"example.test": ()}),
            )

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.reference"],
        )
        self.assertEqual(client.calls, [])

    async def test_remote_source_reports_client_failures_safely(self) -> None:
        cases = (
            (
                "timeout",
                TimeoutError("private timeout"),
                "input.reference",
            ),
            (
                "fetch_error",
                OSError("private network detail"),
                "input.reference",
            ),
            (
                "http_status",
                _remote_response(
                    b"",
                    status_code=404,
                    headers={"Content-Length": "0"},
                ),
                "input.reference",
            ),
        )

        for name, response, expected_path in cases:
            with self.subTest(name=name):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.remote_url(
                            "https://example.test/input.txt",
                            mime_type="text/plain",
                        ),
                        roots=(),
                        artifact_store=StreamingOnlyArtifactStore(),
                        remote_url_policy=TaskRemoteUrlPolicy(
                            enabled=True,
                            max_bytes=100,
                        ),
                        remote_url_http_client=FakeRemoteClient(
                            {"https://example.test/input.txt": response}
                        ),
                        remote_url_resolver=FakeRemoteResolver(),
                    )

                self.assertEqual(
                    [issue.path for issue in error.exception.issues],
                    [expected_path],
                )
                self.assertNotIn("private", str(error.exception))

    async def test_remote_source_handles_redirect_failures(self) -> None:
        cases = (
            (
                "disabled",
                TaskRemoteUrlPolicy(enabled=True, max_bytes=100),
                {
                    "https://example.test/input.txt": _remote_response(
                        b"",
                        status_code=302,
                        headers={"Location": "https://cdn.example.test/a.txt"},
                    )
                },
                ["https://example.test/input.txt"],
            ),
            (
                "missing_location",
                TaskRemoteUrlPolicy(
                    enabled=True,
                    allow_redirects=True,
                    max_redirects=2,
                    max_bytes=100,
                ),
                {
                    "https://example.test/input.txt": _remote_response(
                        b"",
                        status_code=302,
                        headers={},
                    )
                },
                ["https://example.test/input.txt"],
            ),
            (
                "loop",
                TaskRemoteUrlPolicy(
                    enabled=True,
                    allow_redirects=True,
                    max_redirects=2,
                    max_bytes=100,
                ),
                {
                    "https://example.test/input.txt": _remote_response(
                        b"",
                        status_code=302,
                        headers={"Location": "https://example.test/input.txt"},
                    )
                },
                ["https://example.test/input.txt"],
            ),
            (
                "limit",
                TaskRemoteUrlPolicy(
                    enabled=True,
                    allow_redirects=True,
                    max_redirects=1,
                    max_bytes=100,
                ),
                {
                    "https://example.test/input.txt": _remote_response(
                        b"",
                        status_code=302,
                        headers={"Location": "https://cdn.example.test/a.txt"},
                    ),
                    "https://cdn.example.test/a.txt": _remote_response(
                        b"",
                        status_code=302,
                        headers={"Location": "https://cdn.example.test/b.txt"},
                    ),
                },
                [
                    "https://example.test/input.txt",
                    "https://cdn.example.test/a.txt",
                ],
            ),
        )

        for name, policy, responses, expected_urls in cases:
            with self.subTest(name=name):
                client = FakeRemoteClient(responses)
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.remote_url(
                            "https://example.test/input.txt",
                            mime_type="text/plain",
                        ),
                        roots=(),
                        artifact_store=StreamingOnlyArtifactStore(),
                        remote_url_policy=policy,
                        remote_url_http_client=client,
                        remote_url_resolver=FakeRemoteResolver(
                            {
                                "example.test": ("8.8.8.8",),
                                "cdn.example.test": ("1.1.1.1",),
                            }
                        ),
                    )

                self.assertEqual(
                    [url for url, _timeout in client.calls],
                    expected_urls,
                )
                self.assertEqual(
                    [issue.path for issue in error.exception.issues],
                    ["input.redirects"],
                )

    async def test_remote_source_rejects_redirect_to_private_host(
        self,
    ) -> None:
        client = FakeRemoteClient(
            {
                "https://example.test/input.txt": _remote_response(
                    b"",
                    status_code=302,
                    headers={"Location": "https://127.0.0.1/private.txt"},
                )
            }
        )

        with self.assertRaises(TaskFileMaterializationError) as error:
            await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                roots=(),
                artifact_store=StreamingOnlyArtifactStore(),
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    allow_redirects=True,
                    max_redirects=2,
                    max_bytes=100,
                ),
                remote_url_http_client=client,
                remote_url_resolver=FakeRemoteResolver(
                    {"example.test": ("8.8.8.8",)}
                ),
            )

        self.assertEqual(
            [url for url, _timeout in client.calls],
            ["https://example.test/input.txt"],
        )
        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["input.reference"],
        )
        self.assertNotIn("127.0.0.1", str(error.exception))

    async def test_remote_source_revalidates_redirect_targets(self) -> None:
        cases = (
            ("invalid_port", "https://example.test:bad/private.txt"),
            ("unsupported_scheme", "ftp://cdn.example.test/private.txt"),
            ("credentials", "https://user:pass@cdn.example.test/private.txt"),
            ("missing_host", "https://:443/private.txt"),
            ("local_hostname", "https://worker.localhost/private.txt"),
        )

        for name, location in cases:
            with self.subTest(name=name):
                client = FakeRemoteClient(
                    {
                        "https://example.test/input.txt": _remote_response(
                            b"",
                            status_code=302,
                            headers={"Location": location},
                        )
                    }
                )
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        TaskFileDescriptor.remote_url(
                            "https://example.test/input.txt",
                            mime_type="text/plain",
                        ),
                        roots=(),
                        artifact_store=StreamingOnlyArtifactStore(),
                        remote_url_policy=TaskRemoteUrlPolicy(
                            enabled=True,
                            allow_redirects=True,
                            max_redirects=2,
                            max_bytes=100,
                        ),
                        remote_url_http_client=client,
                        remote_url_resolver=FakeRemoteResolver(
                            {"example.test": ("8.8.8.8",)}
                        ),
                    )

                self.assertEqual(
                    [issue.path for issue in error.exception.issues],
                    ["input.reference"],
                )
                self.assertEqual(
                    [url for url, _timeout in client.calls],
                    ["https://example.test/input.txt"],
                )
                self.assertNotIn("private.txt", str(error.exception))

    async def test_remote_source_rejects_response_metadata_failures(
        self,
    ) -> None:
        cases = (
            (
                "unknown_size",
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(
                    b"text",
                    headers={"Content-Type": "text/plain"},
                ),
                "input.size_bytes",
            ),
            (
                "invalid_size",
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(
                    b"text",
                    headers={
                        "Content-Length": "bad",
                        "Content-Type": "text/plain",
                    },
                ),
                "input.size_bytes",
            ),
            (
                "negative_size",
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(
                    b"text",
                    headers={
                        "Content-Length": "-1",
                        "Content-Type": "text/plain",
                    },
                ),
                "input.size_bytes",
            ),
            (
                "over_limit",
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(
                    b"private text",
                    headers={
                        "Content-Length": "12",
                        "Content-Type": "text/plain",
                    },
                ),
                "input.size_bytes",
            ),
            (
                "descriptor_size_mismatch",
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                    size_bytes=5,
                ),
                _remote_response(b"text"),
                "input.size_bytes",
            ),
            (
                "descriptor_mime_mismatch",
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(
                    b"text",
                    headers={
                        "Content-Length": "4",
                        "Content-Type": "application/pdf",
                    },
                ),
                "input.mime_type",
            ),
        )

        for name, definition, descriptor, response, expected_path in cases:
            with self.subTest(name=name):
                store = StreamingOnlyArtifactStore()
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        definition,
                        descriptor,
                        roots=(),
                        artifact_store=store,
                        remote_url_policy=TaskRemoteUrlPolicy(
                            enabled=True,
                            max_bytes=10,
                        ),
                        remote_url_http_client=FakeRemoteClient(
                            {"https://example.test/input.txt": response}
                        ),
                        remote_url_resolver=FakeRemoteResolver(),
                    )

                self.assertEqual(
                    [issue.path for issue in error.exception.issues],
                    [expected_path],
                )
                self.assertEqual(store.streams, [])
                self.assertTrue(response.stream.closed)
                self.assertNotIn("example.test", str(error.exception))

    async def test_remote_source_reports_stream_failures_safely(self) -> None:
        timeout_stream = TimeoutReadStream()
        cases = (
            (
                "timeout",
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                task_materialization.TaskRemoteUrlResponse(
                    status_code=200,
                    headers={
                        "Content-Length": "4",
                        "Content-Type": "text/plain",
                    },
                    stream=cast(BinaryIO, timeout_stream),
                ),
                StreamingOnlyArtifactStore(),
                "input.reference",
            ),
            (
                "read_error",
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                task_materialization.TaskRemoteUrlResponse(
                    status_code=200,
                    headers={
                        "Content-Length": "4",
                        "Content-Type": "text/plain",
                    },
                    stream=cast(BinaryIO, FailingReadStream()),
                ),
                StreamingOnlyArtifactStore(),
                "input.reference",
            ),
            (
                "size_mismatch",
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(
                    b"abc",
                    headers={
                        "Content-Length": "4",
                        "Content-Type": "text/plain",
                    },
                ),
                StreamingOnlyArtifactStore(),
                "input.size_bytes",
            ),
            (
                "digest_mismatch",
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                    sha256="0" * 64,
                ),
                _remote_response(b"abc"),
                StreamingOnlyArtifactStore(),
                "input.sha256",
            ),
            (
                "store_policy",
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                _remote_response(b"abc"),
                PolicyFailingArtifactStore(),
                "input.size_bytes",
            ),
        )

        for name, descriptor, response, store, expected_path in cases:
            with self.subTest(name=name):
                with self.assertRaises(TaskFileMaterializationError) as error:
                    await materialize_task_input_files(
                        _definition(),
                        descriptor,
                        roots=(),
                        artifact_store=store,
                        remote_url_policy=TaskRemoteUrlPolicy(
                            enabled=True,
                            max_bytes=10,
                        ),
                        remote_url_http_client=FakeRemoteClient(
                            {"https://example.test/input.txt": response}
                        ),
                        remote_url_resolver=FakeRemoteResolver(),
                    )

                self.assertEqual(
                    [issue.path for issue in error.exception.issues],
                    [expected_path],
                )
                self.assertNotIn("example.test", str(error.exception))

        self.assertTrue(timeout_stream.closed)

    async def test_remote_backend_error_without_stream_issue_is_preserved(
        self,
    ) -> None:
        with self.assertRaises(ArtifactStoreError) as error:
            await materialize_task_input_files(
                _definition(),
                TaskFileDescriptor.remote_url(
                    "https://example.test/input.txt",
                    mime_type="text/plain",
                ),
                roots=(),
                artifact_store=BackendFailingArtifactStore(),
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    max_bytes=10,
                ),
                remote_url_http_client=FakeRemoteClient(
                    {
                        "https://example.test/input.txt": _remote_response(
                            b"abc"
                        )
                    }
                ),
                remote_url_resolver=FakeRemoteResolver(),
            )

        self.assertNotIn("example.test", str(error.exception))

    async def test_remote_source_metadata_append_failure_deletes_artifact(
        self,
    ) -> None:
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
                TaskFileDescriptor.remote_url(
                    "https://example.test/private.txt",
                    mime_type="text/plain",
                ),
                roots=(),
                artifact_store=artifact_store,
                hmac_provider=cast(HmacProvider, StaticHmacProvider()),
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    max_bytes=100,
                ),
                remote_url_http_client=FakeRemoteClient(
                    {
                        "https://example.test/private.txt": _remote_response(
                            b"text"
                        )
                    }
                ),
                remote_url_resolver=FakeRemoteResolver(),
                task_store=task_store,
                run_id=run.run_id,
            )

        self.assertEqual(len(artifact_store.deleted), 1)
        self.assertEqual(artifact_store.deleted[0].artifact_id, "artifact-1")
        self.assertNotIn("example.test", str(error.exception))

    def test_default_remote_resolver_uses_unique_socket_addresses(
        self,
    ) -> None:
        resolver = task_materialization.DefaultTaskRemoteUrlResolver()

        with patch.object(
            task_materialization,
            "getaddrinfo",
            return_value=[
                (0, 0, 0, "", ("203.0.113.10", 443)),
                (0, 0, 0, "", ("203.0.113.10", 443)),
                (0, 0, 0, "", ("2001:db8::1%en0", 443)),
            ],
        ):
            addresses = resolver.resolve("example.test")

        self.assertEqual(addresses, ("2001:db8::1", "203.0.113.10"))

    def test_default_remote_resolver_reports_dns_failures(self) -> None:
        resolver = task_materialization.DefaultTaskRemoteUrlResolver()

        with patch.object(
            task_materialization,
            "getaddrinfo",
            side_effect=task_materialization.gaierror(),
        ):
            with self.assertRaises(OSError):
                resolver.resolve("example.test")
        with patch.object(
            task_materialization, "getaddrinfo", return_value=[]
        ):
            with self.assertRaises(OSError):
                resolver.resolve("example.test")

    def test_urllib_remote_client_maps_responses_and_errors(self) -> None:
        client = task_materialization.UrllibTaskRemoteUrlHttpClient()
        success_opener = FakeOpener(
            FakeHttpResponse(
                b"text",
                status=200,
                headers={"Content-Length": "4"},
            )
        )
        setattr(client, "_opener", success_opener)

        response = client.open(
            "https://example.test/input.txt",
            timeout_seconds=1.5,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Length"], "4")
        self.assertEqual(success_opener.calls[0][1], 1.5)

        redirect_headers = Message()
        redirect_headers["Location"] = "https://cdn.example.test/input.txt"
        redirect_client = task_materialization.UrllibTaskRemoteUrlHttpClient()
        setattr(
            redirect_client,
            "_opener",
            FakeOpener(
                HTTPError(
                    "https://example.test/input.txt",
                    302,
                    "found",
                    redirect_headers,
                    BytesIO(),
                )
            ),
        )

        redirect = redirect_client.open(
            "https://example.test/input.txt",
            timeout_seconds=2.0,
        )

        self.assertEqual(redirect.status_code, 302)
        self.assertEqual(
            redirect.headers["Location"],
            "https://cdn.example.test/input.txt",
        )

        for error in (
            HTTPError(
                "https://example.test/input.txt",
                404,
                "not found",
                Message(),
                BytesIO(),
            ),
            URLError("private failure"),
        ):
            failing_client = (
                task_materialization.UrllibTaskRemoteUrlHttpClient()
            )
            setattr(failing_client, "_opener", FakeOpener(error))
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(OSError):
                    failing_client.open(
                        "https://example.test/input.txt",
                        timeout_seconds=2.0,
                    )

    def test_remote_redirect_handler_disables_builtin_redirects(self) -> None:
        handler = task_materialization._NoRedirectHandler()

        request = task_materialization.Request(
            "https://example.test/input.txt"
        )

        self.assertIsNone(
            handler.redirect_request(
                request,
                object(),
                302,
                "found",
                object(),
                "https://cdn.example.test/input.txt",
            )
        )

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
                TaskFileDescriptor.inline_bytes(
                    "inline-1",
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
