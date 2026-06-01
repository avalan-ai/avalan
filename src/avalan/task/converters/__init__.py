from ...types import assert_non_empty_string as _assert_non_empty_string
from ..artifact import (
    ArtifactStore,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
)
from ..input import TaskFileConversionRequest
from ..store import TaskSnapshotMetadata, TaskStore, freeze_snapshot_metadata

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class TaskFileConversionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskFileConversionResult:
    content: bytes
    media_type: str
    metadata: TaskSnapshotMetadata

    def __post_init__(self) -> None:
        assert isinstance(self.content, bytes), "content must be bytes"
        _assert_non_empty_string(self.media_type, "media_type")
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskConvertedArtifact:
    source_ref: TaskArtifactRef
    ref: TaskArtifactRef
    request: TaskFileConversionRequest
    converter_name: str
    converter_version: str
    result_metadata: TaskSnapshotMetadata
    record: TaskArtifactRecord | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.source_ref, TaskArtifactRef)
        assert isinstance(self.ref, TaskArtifactRef)
        assert isinstance(self.request, TaskFileConversionRequest)
        _assert_non_empty_string(self.converter_name, "converter_name")
        _assert_non_empty_string(self.converter_version, "converter_version")
        object.__setattr__(
            self,
            "result_metadata",
            freeze_snapshot_metadata(self.result_metadata),
        )
        if self.record is not None:
            assert isinstance(self.record, TaskArtifactRecord)


class FileConverter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult: ...


async def convert_task_artifact(
    source_ref: TaskArtifactRef,
    request: TaskFileConversionRequest,
    *,
    converter: FileConverter,
    artifact_store: ArtifactStore,
    task_store: TaskStore | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    retention: TaskArtifactRetention | None = None,
) -> TaskConvertedArtifact:
    assert isinstance(source_ref, TaskArtifactRef)
    assert isinstance(request, TaskFileConversionRequest)
    _assert_converter(converter)
    if task_store is not None:
        _assert_non_empty_string(run_id, "run_id")
    if attempt_id is not None:
        _assert_non_empty_string(attempt_id, "attempt_id")
    if retention is not None:
        assert isinstance(retention, TaskArtifactRetention)

    reader = await artifact_store.open(source_ref)
    try:
        content = reader.read()
    finally:
        reader.close()
    assert isinstance(content, bytes), "artifact readers must return bytes"

    result = await converter.convert(
        content,
        source_media_type=source_ref.media_type,
        options=request.options,
    )
    assert isinstance(result, TaskFileConversionResult)
    conversion_metadata = _conversion_metadata(
        source_ref,
        request,
        converter=converter,
        result=result,
    )
    ref = await artifact_store.put(
        result.content,
        media_type=result.media_type,
        metadata=conversion_metadata,
    )
    identity = _content_identity(ref)
    options = freeze_snapshot_metadata(request.options)
    record: TaskArtifactRecord | None = None
    record_metadata = freeze_snapshot_metadata(
        {
            "converter": _converter_identity(converter),
            "identity": identity,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
        }
    )
    if task_store is not None:
        record = await task_store.append_artifact(
            run_id or "",
            ref=ref,
            purpose=TaskArtifactPurpose.CONVERTED,
            attempt_id=attempt_id,
            provenance=TaskArtifactProvenance(
                source_artifact_id=source_ref.artifact_id,
                operation="conversion",
                converter=converter.name,
                metadata={
                    "converter": _converter_identity(converter),
                    "options": options,
                    "source_media_type": source_ref.media_type,
                    "target_media_type": ref.media_type,
                    "size_bytes": ref.size_bytes,
                    "identity": identity,
                    "result": result.metadata,
                },
            ),
            retention=retention or TaskArtifactRetention(),
            metadata=record_metadata,
        )
    return TaskConvertedArtifact(
        source_ref=source_ref,
        ref=ref,
        request=request,
        converter_name=converter.name,
        converter_version=converter.version,
        result_metadata=result.metadata,
        record=record,
    )


def _conversion_metadata(
    source_ref: TaskArtifactRef,
    request: TaskFileConversionRequest,
    *,
    converter: FileConverter,
    result: TaskFileConversionResult,
) -> TaskSnapshotMetadata:
    options = freeze_snapshot_metadata(request.options)
    return freeze_snapshot_metadata(
        {
            "source_artifact_id": source_ref.artifact_id,
            "converter": _converter_identity(converter),
            "options": options,
            "source_media_type": source_ref.media_type,
            "target_media_type": result.media_type,
            "result": result.metadata,
        }
    )


def _converter_identity(converter: FileConverter) -> TaskSnapshotMetadata:
    return freeze_snapshot_metadata(
        {
            "name": converter.name,
            "version": converter.version,
        }
    )


def _content_identity(ref: TaskArtifactRef) -> TaskSnapshotMetadata:
    value: dict[str, object] = {}
    if ref.sha256 is not None:
        value["sha256"] = ref.sha256
    if ref.size_bytes is not None:
        value["size_bytes"] = ref.size_bytes
    return freeze_snapshot_metadata(value)


def _assert_converter(converter: FileConverter) -> None:
    _assert_non_empty_string(converter.name, "converter.name")
    _assert_non_empty_string(converter.version, "converter.version")
