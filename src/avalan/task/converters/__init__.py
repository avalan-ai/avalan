from ...types import assert_non_empty_string as _assert_non_empty_string
from ..artifact import (
    ArtifactStore,
    ArtifactStorePolicyError,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    read_artifact_stream_bytes,
)
from ..feature_gate import TaskFeature, feature_available
from ..input import TaskFileConversionRequest
from ..store import TaskSnapshotMetadata, TaskStore, freeze_snapshot_metadata

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

_TEXT_CONVERTER_MAX_BYTES = 1024 * 1024
_TEXT_CONVERTER_OPTIONS_SCHEMA = MappingProxyType(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "encoding": {"type": "string", "minLength": 1},
            "errors": {"enum": ("strict", "replace", "ignore")},
            "newline": {"enum": ("preserve", "lf")},
        },
    }
)
_MARKDOWN_CONVERTER_OPTIONS_SCHEMA = MappingProxyType(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "encoding": {"type": "string", "minLength": 1},
            "errors": {"enum": ("strict", "replace", "ignore")},
            "html_heading_style": {"enum": ("ATX", "ATX_CLOSED", "SETEXT")},
        },
    }
)


class TaskFileConversionError(RuntimeError):
    pass


class TaskFileConversionDependencyError(TaskFileConversionError):
    feature: TaskFeature

    def __init__(self, feature: TaskFeature) -> None:
        assert isinstance(feature, TaskFeature)
        self.feature = feature
        super().__init__("file conversion dependency is unavailable")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskFileConverterCapability:
    source_mime_types: tuple[str, ...]
    output_mime_types: tuple[str, ...]
    supports_streaming: bool
    max_input_bytes: int | None
    max_output_bytes: int | None
    max_pages: int | None = None
    min_dpi: int | None = None
    max_dpi: int | None = None
    min_quality: int | None = None
    max_quality: int | None = None
    max_pixels: int | None = None
    estimated_memory_bytes: int | None = None
    timeout_seconds: int | None = None
    options_schema: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    dependency_gates: tuple[TaskFeature, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(
            self.source_mime_types, tuple
        ), "source_mime_types must be a tuple"
        assert self.source_mime_types, "source_mime_types must not be empty"
        for mime_type in self.source_mime_types:
            _assert_non_empty_string(mime_type, "source_mime_types")
        assert isinstance(
            self.output_mime_types, tuple
        ), "output_mime_types must be a tuple"
        assert self.output_mime_types, "output_mime_types must not be empty"
        for mime_type in self.output_mime_types:
            _assert_non_empty_string(mime_type, "output_mime_types")
        assert isinstance(self.supports_streaming, bool)
        for field_name in ("max_input_bytes", "max_output_bytes"):
            value = getattr(self, field_name)
            if value is not None:
                assert isinstance(value, int), f"{field_name} must be an int"
                assert not isinstance(
                    value, bool
                ), f"{field_name} must be an int"
                assert value > 0, f"{field_name} must be positive"
        for field_name in (
            "max_pages",
            "min_dpi",
            "max_dpi",
            "min_quality",
            "max_quality",
            "max_pixels",
            "estimated_memory_bytes",
            "timeout_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None:
                assert isinstance(value, int), f"{field_name} must be an int"
                assert not isinstance(
                    value, bool
                ), f"{field_name} must be an int"
                assert value > 0, f"{field_name} must be positive"
        if self.min_dpi is not None and self.max_dpi is not None:
            assert self.min_dpi <= self.max_dpi
        if self.min_quality is not None and self.max_quality is not None:
            assert self.min_quality <= self.max_quality
        assert isinstance(self.options_schema, Mapping)
        object.__setattr__(
            self,
            "options_schema",
            freeze_snapshot_metadata(self.options_schema),
        )
        assert isinstance(self.dependency_gates, tuple)
        for feature in self.dependency_gates:
            assert isinstance(feature, TaskFeature)


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

    @property
    def capability(self) -> TaskFileConverterCapability: ...

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult: ...


class UnavailableFileConverter:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        capability: TaskFileConverterCapability,
    ) -> None:
        _assert_non_empty_string(name, "name")
        _assert_non_empty_string(version, "version")
        assert isinstance(capability, TaskFileConverterCapability)
        assert capability.dependency_gates
        self._name = name
        self._version = version
        self._capability = capability

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        raise TaskFileConversionDependencyError(
            self._capability.dependency_gates[0]
        )


def text_converter_capability() -> TaskFileConverterCapability:
    return TaskFileConverterCapability(
        source_mime_types=(
            "text/*",
            "application/json",
            "application/*+json",
            "application/xml",
            "application/*+xml",
        ),
        output_mime_types=("text/plain",),
        supports_streaming=False,
        max_input_bytes=_TEXT_CONVERTER_MAX_BYTES,
        max_output_bytes=_TEXT_CONVERTER_MAX_BYTES,
        options_schema=_TEXT_CONVERTER_OPTIONS_SCHEMA,
    )


def markdown_converter_capability() -> TaskFileConverterCapability:
    return TaskFileConverterCapability(
        source_mime_types=(
            "text/markdown",
            "text/x-markdown",
            "text/plain",
            "text/html",
            "application/xhtml+xml",
        ),
        output_mime_types=("text/markdown",),
        supports_streaming=False,
        max_input_bytes=_TEXT_CONVERTER_MAX_BYTES,
        max_output_bytes=_TEXT_CONVERTER_MAX_BYTES,
        options_schema=_MARKDOWN_CONVERTER_OPTIONS_SCHEMA,
        dependency_gates=(TaskFeature.DOCUMENT_CONVERSION,),
    )


def validate_conversion_request(
    converter: FileConverter,
    request: TaskFileConversionRequest,
    *,
    source_media_type: str | None,
    source_size_bytes: int | None,
) -> None:
    assert isinstance(request, TaskFileConversionRequest)
    _assert_converter(converter)
    capability = converter.capability
    for feature in capability.dependency_gates:
        if not feature_available(feature):
            raise TaskFileConversionDependencyError(feature)
    if (
        not capability.supports_streaming
        and capability.max_input_bytes is None
    ):
        raise TaskFileConversionError(
            "file converter requires an explicit input byte limit"
        )
    if source_media_type is not None and not _mime_type_supported(
        source_media_type,
        capability.source_mime_types,
    ):
        raise TaskFileConversionError(
            "file media type is not supported by the converter"
        )
    if (
        source_size_bytes is not None
        and capability.max_input_bytes is not None
        and source_size_bytes > capability.max_input_bytes
    ):
        raise TaskFileConversionError(
            "file input exceeds the converter byte limit"
        )
    validate_options = getattr(converter, "validate_options", None)
    if callable(validate_options):
        validate_options(request.options)


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

    stat = await artifact_store.stat(source_ref)
    validate_conversion_request(
        converter,
        request,
        source_media_type=source_ref.media_type,
        source_size_bytes=stat.size_bytes,
    )
    capability = converter.capability
    reader = await artifact_store.open_stream(
        source_ref,
        max_bytes=capability.max_input_bytes,
    )
    try:
        content = read_artifact_stream_bytes(
            reader,
            max_bytes=capability.max_input_bytes,
            expected_size_bytes=stat.size_bytes,
            expected_sha256=stat.sha256,
        )
    except ArtifactStorePolicyError as error:
        raise TaskFileConversionError(
            "file input exceeds the converter byte limit"
        ) from error
    finally:
        reader.close()
    assert isinstance(content, bytes), "artifact readers must return bytes"

    result = await converter.convert(
        content,
        source_media_type=source_ref.media_type,
        options=request.options,
    )
    assert isinstance(result, TaskFileConversionResult)
    if (
        capability.max_output_bytes is not None
        and len(result.content) > capability.max_output_bytes
    ):
        raise TaskFileConversionError(
            "file output exceeds the converter byte limit"
        )
    if not _mime_type_supported(
        result.media_type, capability.output_mime_types
    ):
        raise TaskFileConversionError(
            "file output media type is not supported by the converter"
        )
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
    assert isinstance(converter.capability, TaskFileConverterCapability)


def _mime_type_supported(
    media_type: str,
    supported: tuple[str, ...],
) -> bool:
    lowered = media_type.lower()
    for pattern in supported:
        candidate = pattern.lower()
        if candidate == "*/*" or candidate == lowered:
            return True
        if candidate.endswith("/*") and lowered.startswith(candidate[:-1]):
            return True
        if "*" in candidate:
            prefix, suffix = candidate.split("*", 1)
            if lowered.startswith(prefix) and lowered.endswith(suffix):
                return True
    return False
