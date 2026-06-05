from ..model.file_delivery import (
    FileDeliveryDecision,
    FileDeliveryDiagnostic,
    FileDeliveryMode,
    FileDeliveryProfile,
    FileDeliveryRequest,
)
from .artifact import ArtifactStore, ArtifactStoreError, TaskArtifactStat
from .context import TaskInputFile
from .definition import TaskDefinition

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskFileDeliveryPlan:
    decision: FileDeliveryDecision
    size_bytes: int | None = None
    size_bucket: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.decision, FileDeliveryDecision)
        if self.size_bytes is not None:
            assert isinstance(self.size_bytes, int)
            assert not isinstance(self.size_bytes, bool)
            assert self.size_bytes >= 0
        if self.size_bucket is not None:
            _assert_non_empty_string(self.size_bucket, "size_bucket")


async def plan_task_file_delivery(
    definition: TaskDefinition,
    file: TaskInputFile,
    *,
    profile: FileDeliveryProfile,
    artifact_store: ArtifactStore | None = None,
) -> TaskFileDeliveryPlan:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(file, TaskInputFile)
    assert isinstance(profile, FileDeliveryProfile)
    if artifact_store is not None:
        assert callable(getattr(artifact_store, "stat", None))
    metadata = _delivery_metadata(file)
    mime_type = _file_mime_type(file)
    stat = await _resolve_stat_if_needed(
        definition,
        file,
        profile=profile,
        artifact_store=artifact_store,
        metadata=metadata,
    )
    size_bytes = _resolved_size(file, stat)
    size_bucket = _size_bucket(size_bytes)
    if not profile.supports_file_delivery:
        return _plan(
            _reject(
                code="task.file_delivery.unsupported",
                message="Task file delivery is not supported.",
                hint="Use a target with file support or declare conversion.",
            ),
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    limit_decision = _task_limit_decision(
        definition,
        has_artifact=file.artifact_ref is not None,
        size_bytes=size_bytes,
        size_bucket=size_bucket,
    )
    if limit_decision is not None:
        return _plan(
            limit_decision,
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    reference_decision = _reference_delivery_decision(
        profile,
        metadata=metadata,
        mime_type=mime_type,
        size_bytes=size_bytes,
    )
    if reference_decision is not None:
        return _plan(
            reference_decision,
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    if _requires_mime_rejection(
        definition,
        file,
        profile=profile,
        mime_type=mime_type,
    ):
        return _plan(
            _reject(
                code="task.file_delivery.unsupported_mime",
                message="Task file MIME type is not supported.",
                hint="Use a supported MIME type or declare conversion.",
            ),
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    if file.artifact_ref is not None and artifact_store is None:
        return _plan(
            _reject(
                code="task.file_delivery.missing_artifact_store",
                message="Task file delivery requires an artifact store.",
                hint="Configure an artifact store before dispatch.",
            ),
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    rejected_inline_decision: FileDeliveryDecision | None = None
    if _can_inline_image(profile, file, mime_type=mime_type):
        decision = profile.plan_delivery(
            FileDeliveryRequest(
                mime_type=mime_type,
                size_bytes=size_bytes,
                has_artifact=True,
                metadata=metadata,
            )
        )
        if decision.mode != FileDeliveryMode.REJECT:
            return _plan(
                decision,
                size_bytes=size_bytes,
                size_bucket=size_bucket,
            )
        rejected_inline_decision = decision
    if _can_inline_bytes(profile, file, mime_type=mime_type):
        decision = _inline_bytes_decision(
            profile,
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
        if decision.mode != FileDeliveryMode.REJECT:
            return _plan(
                decision,
                size_bytes=size_bytes,
                size_bucket=size_bucket,
            )
        rejected_inline_decision = decision
    if _can_inline_text(profile, file, mime_type=mime_type):
        decision = _inline_text_decision(
            profile,
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
        if decision.mode != FileDeliveryMode.REJECT:
            return _plan(
                decision,
                size_bytes=size_bytes,
                size_bucket=size_bucket,
            )
        rejected_inline_decision = decision
    if _can_convert(definition, profile):
        return _plan(
            FileDeliveryDecision(mode=FileDeliveryMode.CONVERTED_ARTIFACT),
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    if _can_retrieve(definition, profile):
        return _plan(
            FileDeliveryDecision(mode=FileDeliveryMode.RETRIEVAL_CONTEXT),
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    if profile.supports_delivery_mode(FileDeliveryMode.MAP_REDUCE_CONTEXT):
        return _plan(
            FileDeliveryDecision(mode=FileDeliveryMode.MAP_REDUCE_CONTEXT),
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    if rejected_inline_decision is not None:
        return _plan(
            rejected_inline_decision,
            size_bytes=size_bytes,
            size_bucket=size_bucket,
        )
    return _plan(
        _reject(
            code="task.file_delivery.rejected",
            message="Task file has no supported delivery mode.",
            hint="Use a supported provider reference, conversion, or limit.",
        ),
        size_bytes=size_bytes,
        size_bucket=size_bucket,
    )


def _delivery_metadata(file: TaskInputFile) -> Mapping[str, object]:
    metadata = _metadata_with_flat_dimensions(file.metadata)
    if file.provider_reference is None:
        return _metadata_without_provider_reference(metadata)
    return MappingProxyType(
        {
            **metadata,
            "provider_reference": file.provider_reference.execution_metadata(),
        }
    )


def _metadata_without_provider_reference(
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    if "provider_reference" not in metadata:
        return metadata
    return MappingProxyType(
        {
            key: value
            for key, value in metadata.items()
            if key != "provider_reference"
        }
    )


def _metadata_with_flat_dimensions(
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    dimensions = metadata.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return metadata
    flattened = dict(metadata)
    for key in ("height_pixels", "width_pixels"):
        value = dimensions.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            flattened.setdefault(key, value)
    return MappingProxyType(flattened)


def _file_mime_type(file: TaskInputFile) -> str | None:
    if file.media_type is not None:
        return file.media_type
    if file.provider_reference is not None:
        return file.provider_reference.mime_type
    return None


async def _resolve_stat_if_needed(
    definition: TaskDefinition,
    file: TaskInputFile,
    *,
    profile: FileDeliveryProfile,
    artifact_store: ArtifactStore | None,
    metadata: Mapping[str, object],
) -> TaskArtifactStat | None:
    if _known_size(file) is not None:
        return None
    if file.artifact_ref is None:
        return None
    if not _has_byte_limited_candidate(definition, profile, metadata):
        return None
    if artifact_store is None:
        return None
    try:
        return await artifact_store.stat(file.artifact_ref)
    except ArtifactStoreError:
        return None


def _known_size(file: TaskInputFile) -> int | None:
    if file.size_bytes is not None:
        return file.size_bytes
    if file.artifact_ref is not None:
        return file.artifact_ref.size_bytes
    return None


def _resolved_size(
    file: TaskInputFile,
    stat: TaskArtifactStat | None,
) -> int | None:
    if stat is not None:
        return stat.size_bytes
    return _known_size(file)


def _has_byte_limited_candidate(
    definition: TaskDefinition,
    profile: FileDeliveryProfile,
    metadata: Mapping[str, object],
) -> bool:
    if definition.limits.file_bytes is not None:
        return True
    if definition.artifact.max_bytes is not None:
        return True
    if _has_reference_metadata(metadata):
        return False
    return any(
        limit is not None and limit.max_bytes is not None
        for limit in (
            profile.inline_byte_limit,
            profile.inline_image_limit,
            profile.inline_text_limit,
        )
    )


def _task_limit_decision(
    definition: TaskDefinition,
    *,
    has_artifact: bool,
    size_bytes: int | None,
    size_bucket: str | None,
) -> FileDeliveryDecision | None:
    assert isinstance(has_artifact, bool)
    limits = (
        ("task_file_bytes", definition.limits.file_bytes),
        (
            "artifact_bytes",
            definition.artifact.max_bytes if has_artifact else None,
        ),
    )
    for limit_name, max_bytes in limits:
        if max_bytes is None:
            continue
        if size_bytes is None:
            return _reject(
                code="task.file_delivery.unknown_size",
                message="Task file size is required for this delivery mode.",
                hint=f"Provide file stat metadata for {limit_name}.",
            )
        if size_bytes > max_bytes:
            return _limit_reject(limit_name, size_bucket)
    return None


def _reference_delivery_decision(
    profile: FileDeliveryProfile,
    *,
    metadata: Mapping[str, object],
    mime_type: str | None,
    size_bytes: int | None,
) -> FileDeliveryDecision | None:
    if not _has_reference_metadata(metadata):
        return None
    decision = profile.plan_delivery(
        FileDeliveryRequest(
            mime_type=mime_type,
            size_bytes=size_bytes,
            has_artifact=False,
            metadata=metadata,
        )
    )
    return decision


def _has_reference_metadata(metadata: Mapping[str, object]) -> bool:
    return "provider_reference" in metadata


def _requires_mime_rejection(
    definition: TaskDefinition,
    file: TaskInputFile,
    *,
    profile: FileDeliveryProfile,
    mime_type: str | None,
) -> bool:
    if _can_convert(definition, profile):
        return False
    if file.artifact_ref is None:
        return False
    if mime_type is None:
        return True
    return not profile.accepts_mime_type(mime_type)


def _can_inline_bytes(
    profile: FileDeliveryProfile,
    file: TaskInputFile,
    *,
    mime_type: str | None,
) -> bool:
    return (
        file.artifact_ref is not None
        and profile.supports_delivery_mode(FileDeliveryMode.INLINE_BYTES)
        and not (
            _is_image_mime_type(mime_type) and profile.supports_image_blocks()
        )
        and (mime_type is None or profile.accepts_mime_type(mime_type))
    )


def _can_inline_image(
    profile: FileDeliveryProfile,
    file: TaskInputFile,
    *,
    mime_type: str | None,
) -> bool:
    return (
        file.artifact_ref is not None
        and profile.supports_image_blocks()
        and _is_image_mime_type(mime_type)
    )


def _inline_bytes_decision(
    profile: FileDeliveryProfile,
    *,
    size_bytes: int | None,
    size_bucket: str | None,
) -> FileDeliveryDecision:
    limit = profile.inline_byte_limit
    if limit is not None and limit.max_bytes is not None:
        if size_bytes is None:
            return _unknown_limit_reject(limit.name)
        if _base64_size(size_bytes) > limit.max_bytes:
            return _limit_reject(limit.name, size_bucket)
    return FileDeliveryDecision(mode=FileDeliveryMode.INLINE_BYTES)


def _can_inline_text(
    profile: FileDeliveryProfile,
    file: TaskInputFile,
    *,
    mime_type: str | None,
) -> bool:
    return (
        file.artifact_ref is not None
        and profile.supports_delivery_mode(FileDeliveryMode.INLINE_TEXT)
        and not (
            _is_image_mime_type(mime_type) and profile.supports_image_blocks()
        )
        and profile.accepts_mime_type(mime_type)
    )


def _inline_text_decision(
    profile: FileDeliveryProfile,
    *,
    size_bytes: int | None,
    size_bucket: str | None,
) -> FileDeliveryDecision:
    limit = profile.inline_text_limit
    if limit is not None and limit.max_bytes is not None:
        if size_bytes is None:
            return _unknown_limit_reject(limit.name)
        if size_bytes > limit.max_bytes:
            return _limit_reject(limit.name, size_bucket)
    return FileDeliveryDecision(mode=FileDeliveryMode.INLINE_TEXT)


def _is_image_mime_type(mime_type: str | None) -> bool:
    return isinstance(mime_type, str) and _mime_type_matches(
        mime_type,
        "image/*",
    )


def _can_convert(
    definition: TaskDefinition,
    profile: FileDeliveryProfile,
) -> bool:
    return bool(
        definition.input.file_conversions
    ) and profile.supports_delivery_mode(
        FileDeliveryMode.CONVERTED_ARTIFACT,
    )


def _can_retrieve(
    definition: TaskDefinition,
    profile: FileDeliveryProfile,
) -> bool:
    return (
        definition.limits.total_tokens is not None
        and profile.supports_delivery_mode(FileDeliveryMode.RETRIEVAL_CONTEXT)
    )


def _unknown_limit_reject(limit_name: str) -> FileDeliveryDecision:
    return _reject(
        code="task.file_delivery.unknown_size",
        message="Task file size is required for this delivery mode.",
        hint=f"Provide file stat metadata for {limit_name}.",
    )


def _limit_reject(
    limit_name: str,
    size_bucket: str | None,
) -> FileDeliveryDecision:
    bucket = size_bucket or "unknown"
    return _reject(
        code="task.file_delivery.limit_exceeded",
        message="Task file exceeds a delivery limit.",
        hint=f"Use a delivery mode within {limit_name}; size_bucket={bucket}.",
    )


def _reject(
    *,
    code: str,
    message: str,
    hint: str,
) -> FileDeliveryDecision:
    return FileDeliveryDecision(
        mode=FileDeliveryMode.REJECT,
        diagnostic=FileDeliveryDiagnostic(
            code=code,
            message=message,
            hint=hint,
        ),
    )


def _plan(
    decision: FileDeliveryDecision,
    *,
    size_bytes: int | None,
    size_bucket: str | None,
) -> TaskFileDeliveryPlan:
    return TaskFileDeliveryPlan(
        decision=decision,
        size_bytes=size_bytes,
        size_bucket=size_bucket,
    )


def _base64_size(size_bytes: int) -> int:
    assert isinstance(size_bytes, int)
    assert not isinstance(size_bytes, bool)
    assert size_bytes >= 0
    return ((size_bytes + 2) // 3) * 4


def _mime_type_matches(mime_type: str, pattern: str) -> bool:
    lowered = mime_type.lower()
    candidate = pattern.lower()
    if candidate == "*/*" or candidate == lowered:
        return True
    if candidate.endswith("/*"):
        return lowered.startswith(candidate[:-1])
    return False


def _size_bucket(size_bytes: int | None) -> str | None:
    if size_bytes is None:
        return None
    if size_bytes == 0:
        return "0B"
    for label, upper_bound in (
        ("1B-1KB", 1024),
        ("1KB-1MB", 1024 * 1024),
        ("1MB-10MB", 10 * 1024 * 1024),
        ("10MB-100MB", 100 * 1024 * 1024),
    ):
        if size_bytes <= upper_bound:
            return label
    return "100MB+"


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value, f"{field_name} must not be empty"
