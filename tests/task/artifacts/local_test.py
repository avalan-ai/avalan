from asyncio import CancelledError
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.task import (
    ArtifactStoreConflictError,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    TaskArtifactRef,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.artifacts import local as local_artifacts


class BrokenWriter:
    def __enter__(self) -> "BrokenWriter":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def write(self, _content: bytes) -> int:
        raise OSError("private write failure")


class CancellingReader:
    def read(self, _size: int = -1) -> bytes:
        raise CancelledError


class LocalArtifactStoreTest(IsolatedAsyncioTestCase):
    async def test_put_open_stat_and_delete_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-1",
            )

            ref = await store.put(
                b"private bytes",
                media_type="text/plain",
                metadata={"source": "sanitized"},
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
            self.assertEqual(ref.store, "local")
            self.assertEqual(ref.storage_key, "ar/artifact-1")
            self.assertEqual(ref.size_bytes, 13)
            self.assertEqual(ref.sha256, stat.sha256)
            self.assertEqual(stat.size_bytes, 13)
            self.assertFalse(Path(tmp, ref.storage_key).exists())
            with self.assertRaises(ArtifactStoreNotFoundError):
                await store.stat(ref)

    async def test_raw_storage_must_be_explicitly_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)

            with self.assertRaises(ArtifactStorePolicyError):
                await store.put(b"private bytes")

            self.assertEqual(list(Path(tmp).glob("**/*")), [])

    async def test_duplicate_artifact_id_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)
            ref = await store.put(b"first", artifact_id="artifact-1")

            with self.assertRaises(ArtifactStoreConflictError):
                await store.put(b"second", artifact_id=ref.artifact_id)

            reader = await store.open(ref)
            try:
                self.assertEqual(reader.read(), b"first")
            finally:
                reader.close()

    async def test_put_stream_open_stream_and_stat_use_recorded_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-1",
            )
            content = b"private stream bytes"
            expected_sha256 = sha256(content).hexdigest()

            ref = await store.put_stream(
                BytesIO(content),
                media_type="text/plain",
                metadata={"source": "sanitized"},
                max_bytes=len(content),
                expected_size_bytes=len(content),
                expected_sha256=expected_sha256,
            )
            reader = await store.open_stream(ref, max_bytes=len(content))
            try:
                read_content = reader.read()
            finally:
                reader.close()
            with patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("raw bytes were read"),
            ):
                stat = await store.stat(ref)
            legacy_stat = await store.stat(
                TaskArtifactRef(
                    artifact_id=ref.artifact_id,
                    store=ref.store,
                    storage_key=ref.storage_key,
                    media_type=ref.media_type,
                )
            )

            self.assertEqual(read_content, content)
            self.assertEqual(ref.size_bytes, len(content))
            self.assertEqual(ref.sha256, expected_sha256)
            self.assertEqual(ref.metadata["source"], "sanitized")
            self.assertEqual(stat.size_bytes, len(content))
            self.assertEqual(stat.sha256, expected_sha256)
            self.assertEqual(legacy_stat.size_bytes, len(content))
            self.assertEqual(legacy_stat.sha256, expected_sha256)
            bounded = await store.open_stream(ref, max_bytes=3)
            try:
                with self.assertRaises(ArtifactStorePolicyError):
                    bounded.read()
            finally:
                bounded.close()

    async def test_put_stream_validation_failures_leave_no_bytes(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)

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
                    expected_sha256="0" * 64,
                )
            with self.assertRaises(CancelledError):
                await store.put_stream(
                    CancellingReader(),  # type: ignore[arg-type]
                    artifact_id="artifact-3",
                    max_bytes=20,
                )

            self.assertEqual(_files_under(tmp), ())

    async def test_put_stream_duplicate_keeps_existing_bytes_and_cleans_temp(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)
            ref = await store.put(b"first", artifact_id="artifact-1")

            with self.assertRaises(ArtifactStoreConflictError):
                await store.put_stream(
                    BytesIO(b"second"),
                    artifact_id=ref.artifact_id,
                    max_bytes=6,
                )
            reader = await store.open(ref)
            try:
                content = reader.read()
            finally:
                reader.close()

            self.assertEqual(content, b"first")
            self.assertEqual(_files_under(tmp), (Path(tmp, ref.storage_key),))

    async def test_parent_directory_swap_does_not_write_outside_root(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)
            original_path_for_storage_key = store._path_for_storage_key
            checked_path = Path(tmp, "ar", "artifact-1")
            calls = 0

            def swap_after_directory_check(storage_key: str) -> Path:
                nonlocal calls
                calls += 1
                path = original_path_for_storage_key(storage_key)
                if calls == 2:
                    checked_path.parent.rmdir()
                    checked_path.parent.symlink_to(
                        outside,
                        target_is_directory=True,
                    )
                    return checked_path
                return path

            with patch.object(
                store,
                "_path_for_storage_key",
                side_effect=swap_after_directory_check,
            ):
                with self.assertRaises(ArtifactStoreError) as error:
                    await store.put(
                        b"private bytes",
                        artifact_id="artifact-1",
                    )

            self.assertEqual(
                str(error.exception),
                "artifact storage path is unsafe",
            )
            self.assertFalse(Path(outside, "artifact-1").exists())

    def test_unsafe_final_path_returns_sanitized_error(self) -> None:
        with TemporaryDirectory() as tmp:
            parent = Path(tmp, "ar")
            parent.mkdir()

            with self.assertRaises(ArtifactStoreError) as error:
                local_artifacts._write_new_file(
                    parent / ("x" * 10_000),
                    b"private bytes",
                )

            self.assertEqual(
                str(error.exception),
                "artifact storage path is unsafe",
            )
            self.assertNotIn("private", str(error.exception))

    def test_failed_write_removes_partial_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "artifact-1")

            with patch.object(
                local_artifacts,
                "fdopen",
                return_value=BrokenWriter(),
            ):
                with self.assertRaises(ArtifactStoreError) as error:
                    local_artifacts._write_new_file(path, b"private bytes")

            self.assertEqual(str(error.exception), "artifact write failed")
            self.assertFalse(path.exists())
            self.assertEqual(_files_under(tmp), ())

    async def test_failed_finalize_removes_temporary_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)

            with patch.object(
                local_artifacts,
                "link",
                side_effect=OSError("private finalize failure"),
            ):
                with self.assertRaises(ArtifactStoreError) as error:
                    await store.put_stream(
                        BytesIO(b"private bytes"),
                        artifact_id="artifact-1",
                        max_bytes=20,
                    )

            self.assertEqual(
                str(error.exception),
                "artifact storage path is unsafe",
            )
            self.assertEqual(_files_under(tmp), ())

    async def test_references_cannot_escape_or_cross_store(self) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)
            ref = await store.put(b"private bytes", artifact_id="artifact-1")
            wrong_store = TaskArtifactRef(
                artifact_id=ref.artifact_id,
                store="other",
                storage_key=ref.storage_key,
            )
            escaped = TaskArtifactRef(
                artifact_id="artifact-2",
                store="local",
                storage_key="../artifact-2",
            )
            mismatched = TaskArtifactRef(
                artifact_id="artifact-2",
                store="local",
                storage_key=ref.storage_key,
            )
            symlink = Path(tmp, "ln")
            symlink.symlink_to(outside, target_is_directory=True)
            symlink_escape = TaskArtifactRef(
                artifact_id="artifact-3",
                store="local",
                storage_key="ln/artifact-3",
            )
            current_directory = TaskArtifactRef(
                artifact_id="artifact-4",
                store="local",
                storage_key=".",
            )

            with self.assertRaises(ArtifactStoreNotFoundError):
                await store.open(wrong_store)
            with self.assertRaises(ArtifactStoreError):
                await store.open(escaped)
            with self.assertRaises(ArtifactStoreNotFoundError):
                await store.open(mismatched)
            with self.assertRaises(ArtifactStoreError):
                await store.open(symlink_escape)
            with self.assertRaises(ArtifactStoreError):
                await store.open(current_directory)

    async def test_default_artifact_id_factory_creates_safe_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)

            ref = await store.put(b"private bytes")

            self.assertTrue(ref.artifact_id)
            self.assertTrue(Path(tmp, ref.storage_key).exists())

    async def test_invalid_metadata_does_not_write_bytes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)

            with self.assertRaises(AssertionError):
                await store.put(
                    b"private bytes",
                    artifact_id="artifact-1",
                    metadata={"bad": object()},
                )

            self.assertFalse(Path(tmp, "ar", "artifact-1").exists())

    async def test_invalid_backend_configuration_and_inputs_fail_fast(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            LocalArtifactStore("", store_name="local")
        with TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp, raw_storage_allowed=True)
            with self.assertRaises(AssertionError):
                await store.put("not bytes")  # type: ignore[arg-type]
            with self.assertRaises(AssertionError):
                await store.put(b"private bytes", artifact_id="../bad")
            with self.assertRaises(AssertionError):
                await store.put(b"private bytes", media_type="")
            with self.assertRaises(AssertionError):
                await store.put_stream(
                    object(),  # type: ignore[arg-type]
                    max_bytes=1,
                )
            ref = await store.put(b"private bytes", artifact_id="artifact-1")
            with self.assertRaises(AssertionError):
                await store.open_stream(
                    ref,
                    max_bytes=-1,
                )


def _files_under(root: str) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in Path(root).glob("**/*") if path.is_file())
    )


if __name__ == "__main__":
    main()
