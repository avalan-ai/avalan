from ..artifact import (
    ArtifactStoreConflictError,
    ArtifactStoreError,
    ArtifactStoreNotFoundError,
    ArtifactStorePolicyError,
    TaskArtifactRef,
    TaskArtifactStat,
)
from ..store import freeze_snapshot_metadata

from collections.abc import Callable, Mapping
from hashlib import sha256
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
        if not self._raw_storage_allowed:
            raise ArtifactStorePolicyError("artifact byte storage is disabled")
        if media_type is not None:
            _assert_non_empty_string(media_type, "media_type")
        new_artifact_id = artifact_id or self._new_id()
        _assert_artifact_id(new_artifact_id)
        storage_key = _storage_key(new_artifact_id)
        path = self._path_for_storage_key(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._path_for_storage_key(storage_key)
        if path.exists():
            raise ArtifactStoreConflictError("artifact already exists")
        digest = sha256(content).hexdigest()
        ref = TaskArtifactRef(
            artifact_id=new_artifact_id,
            store=self._store_name,
            storage_key=storage_key,
            media_type=media_type,
            size_bytes=len(content),
            sha256=digest,
            metadata=freeze_snapshot_metadata(metadata),
        )
        path.write_bytes(content)
        return ref

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        path = self._existing_path_for_ref(ref)
        return path.open("rb")

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        path = self._existing_path_for_ref(ref)
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


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _uuid_id() -> str:
    return uuid4().hex
