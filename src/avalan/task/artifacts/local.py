from ...types import assert_non_empty_string as _assert_non_empty_string
from ..artifact import (
    ArtifactStoreConflictError,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    TaskArtifactRef,
    TaskArtifactStat,
    TaskArtifactStreamDigest,
    bounded_artifact_reader,
    copy_artifact_stream,
)
from ..store import freeze_snapshot_metadata

from asyncio import CancelledError
from collections.abc import Callable, Mapping
from contextlib import suppress
from hashlib import sha256
from io import BytesIO
from os import (
    O_CREAT,
    O_DIRECTORY,
    O_EXCL,
    O_NOFOLLOW,
    O_RDONLY,
    O_WRONLY,
    close,
    fdopen,
    link,
    unlink,
)
from os import open as open_file_descriptor
from pathlib import Path, PurePosixPath
from re import fullmatch
from typing import BinaryIO
from uuid import uuid4


class LocalArtifactStore:
    def __init__(
        self,
        root: str | Path,
        *,
        store_name: str = "local",
        raw_storage_allowed: bool = False,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        assert isinstance(root, str | Path), "root must be a path"
        if isinstance(root, str):
            _assert_non_empty_string(root, "root")
        _assert_non_empty_string(store_name, "store_name")
        assert isinstance(raw_storage_allowed, bool)
        self._root = Path(root)
        self._store_name = store_name
        self._raw_storage_allowed = raw_storage_allowed
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
        if not self._raw_storage_allowed:
            raise ArtifactStorePolicyError("artifact byte storage is disabled")
        if media_type is not None:
            _assert_non_empty_string(media_type, "media_type")
        new_artifact_id = artifact_id or self._new_id()
        _assert_artifact_id(new_artifact_id)
        ref_metadata = freeze_snapshot_metadata(metadata)
        storage_key = _storage_key(new_artifact_id)
        path = self._path_for_storage_key(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._path_for_storage_key(storage_key)
        digest = _write_new_file_stream(
            path,
            stream,
            max_bytes=max_bytes,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )
        ref = TaskArtifactRef(
            artifact_id=new_artifact_id,
            store=self._store_name,
            storage_key=storage_key,
            media_type=media_type,
            size_bytes=digest.size_bytes,
            sha256=digest.sha256,
            metadata=ref_metadata,
        )
        return ref

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        return await self.open_stream(ref)

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BinaryIO:
        path = self._existing_path_for_ref(ref)
        return bounded_artifact_reader(path.open("rb"), max_bytes=max_bytes)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        path = self._existing_path_for_ref(ref)
        if ref.size_bytes is not None and ref.sha256 is not None:
            return TaskArtifactStat(
                ref=ref,
                size_bytes=ref.size_bytes,
                sha256=ref.sha256,
            )
        content = path.read_bytes()
        return TaskArtifactStat(
            ref=ref,
            size_bytes=len(content),
            sha256=sha256(content).hexdigest(),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        path = self._existing_path_for_ref(ref)
        path.unlink()

    def _existing_path_for_ref(self, ref: TaskArtifactRef) -> Path:
        assert isinstance(ref, TaskArtifactRef)
        if ref.store != self._store_name:
            raise ArtifactStoreNotFoundError("artifact store does not match")
        path = self._path_for_storage_key(ref.storage_key)
        if ref.storage_key != _storage_key(ref.artifact_id):
            raise ArtifactStoreNotFoundError("artifact storage key mismatch")
        if not path.is_file():
            raise ArtifactStoreNotFoundError("artifact was not found")
        return path

    def _path_for_storage_key(self, storage_key: str) -> Path:
        _assert_storage_key(storage_key)
        root = self._root.resolve(strict=False)
        path = (self._root / Path(*PurePosixPath(storage_key).parts)).resolve(
            strict=False
        )
        if not _is_relative_to(path, root):
            raise ArtifactStoreError("artifact storage key escapes root")
        return path

    def _new_id(self) -> str:
        value = self._id_factory()
        _assert_artifact_id(value)
        return value


def _storage_key(artifact_id: str) -> str:
    prefix = artifact_id[:2].lower()
    return f"{prefix}/{artifact_id}"


def _assert_storage_key(value: str) -> None:
    _assert_non_empty_string(value, "storage_key")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ArtifactStoreError("artifact storage key escapes root")


def _assert_artifact_id(value: str) -> None:
    _assert_non_empty_string(value, "artifact_id")
    assert fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        value,
    ), "artifact_id must be a stable token"


def _write_new_file(path: Path, content: bytes) -> None:
    _write_new_file_stream(
        path,
        BytesIO(content),
        max_bytes=len(content),
        expected_size_bytes=len(content),
    )


def _write_new_file_stream(
    path: Path,
    stream: BinaryIO,
    *,
    max_bytes: int | None = None,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> TaskArtifactStreamDigest:
    temp_name = f".{path.name}.{uuid4().hex}.tmp"
    try:
        parent_descriptor = open_file_descriptor(
            path.parent,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW,
        )
    except OSError as exc:
        raise ArtifactStoreError("artifact storage path is unsafe") from exc
    try:
        try:
            file_descriptor = open_file_descriptor(
                temp_name,
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ArtifactStoreError(
                "artifact storage path is unsafe"
            ) from exc
        digest: TaskArtifactStreamDigest
        try:
            with fdopen(file_descriptor, "wb") as file:
                digest = copy_artifact_stream(
                    stream,
                    file.write,
                    max_bytes=max_bytes,
                    expected_size_bytes=expected_size_bytes,
                    expected_sha256=expected_sha256,
                )
            try:
                link(
                    temp_name,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ArtifactStoreConflictError(
                    "artifact already exists"
                ) from exc
            except OSError as exc:
                raise ArtifactStoreError(
                    "artifact storage path is unsafe"
                ) from exc
            return digest
        except (
            ArtifactStoreError,
            ArtifactStorePolicyError,
            CancelledError,
            KeyboardInterrupt,
            SystemExit,
        ):
            with suppress(OSError):
                unlink(temp_name, dir_fd=parent_descriptor)
            raise
        except Exception as exc:
            with suppress(OSError):
                close(file_descriptor)
            with suppress(OSError):
                unlink(temp_name, dir_fd=parent_descriptor)
            raise ArtifactStoreError("artifact write failed") from exc
        finally:
            with suppress(OSError):
                unlink(temp_name, dir_fd=parent_descriptor)
    finally:
        close(parent_descriptor)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _uuid_id() -> str:
    return uuid4().hex
