from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    HmacProvider,
    PrivacySanitizationError,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
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
    TaskRemoteUrlPolicy,
    materialize_task_input_files,
    task_file_descriptors_from_input,
)
from avalan.task.artifacts import LocalArtifactStore
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
                    "sha256": "a" * 64,
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

    async def test_descriptor_coercion_failure_without_issues_propagates(
        self,
    ) -> None:
        with self.assertRaises(KeyError):
            await materialize_task_input_files(
                _definition(),
                DisappearingDescriptor(),
                roots=(),
                artifact_store=None,
            )

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
