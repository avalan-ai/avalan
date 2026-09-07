"""Serialize immutable artifact references through one canonical helper."""

from .artifact import TaskArtifactRef

from collections.abc import Mapping
from datetime import datetime


def _artifact_ref_to_payload(ref: TaskArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": ref.artifact_id,
        "media_type": ref.media_type,
        "metadata": _plain(ref.metadata),
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
        "storage_key": ref.storage_key,
        "store": ref.store,
    }


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
