from .artifact import (
    ArtifactStore,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
)
from .context import TaskInputFile
from .definition import (
    PrivacyAction,
    TaskDefinition,
    TaskInputType,
    TaskPrivacyPolicy,
)
from .input import (
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskFileSourceKind,
    TaskRemoteUrlPolicy,
)
from .privacy import (
    HmacProvider,
    PrivacyField,
    PrivacySanitizationError,
    PrivacySanitizer,
)
from .store import (
    TaskSnapshotMetadata,
    TaskStore,
    freeze_snapshot_metadata,
)
from .validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    validate_task_input,
)

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePath, PureWindowsPath

_REMOTE_URL_DISABLED_CODE = "feature.remote_url_file_inputs_disabled"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskMaterializedFile:
    descriptor: TaskFileDescriptor
    descriptor_path: str
    ref: TaskArtifactRef
    identity: TaskSnapshotMetadata

    def as_input_file(self) -> TaskInputFile:
        return TaskInputFile(
            logical_path=f"artifact:{self.ref.artifact_id}",
            artifact_ref=self.ref,
            media_type=self.ref.media_type,
            size_bytes=self.ref.size_bytes,
            metadata=self.identity,
        )


class TaskFileMaterializationError(TaskValidationError):
    pass


async def materialize_task_input_files(
    definition: TaskDefinition,
    value: object,
    *,
    roots: Iterable[str | Path],
    artifact_store: ArtifactStore | None,
    hmac_provider: HmacProvider | None = None,
    remote_url_policy: TaskRemoteUrlPolicy | None = None,
    task_store: TaskStore | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> tuple[TaskMaterializedFile, ...]:
    assert isinstance(definition, TaskDefinition)
    root_paths = tuple(Path(root) for root in roots)
    issues = list(
        validate_task_input(
            definition,
            value,
            remote_url_policy=remote_url_policy,
        )
    )
    descriptor_entries, descriptor_issues = (
        _task_file_descriptor_entries_from_input(definition, value)
    )
    issues.extend(descriptor_issues)
    if not descriptor_entries:
        if issues:
            raise TaskFileMaterializationError(tuple(issues))
        return ()
    if _has_issue_code(issues, _REMOTE_URL_DISABLED_CODE):
        raise TaskFileMaterializationError(tuple(issues))
    if artifact_store is None:
        issues.append(
            _issue(
                code="artifact.bytes_unsupported",
                path="input",
                message=(
                    "Task file materialization requires an artifact backend."
                ),
                hint=(
                    "Configure an artifact backend before materializing files."
                ),
                category=TaskValidationCategory.UNSUPPORTED,
            )
        )
        raise TaskFileMaterializationError(tuple(issues))
    if task_store is not None:
        assert isinstance(run_id, str) and run_id.strip()
    descriptors = tuple(entry.descriptor for entry in descriptor_entries)
    issues.extend(_validate_count_limits(definition, descriptors))
    resolved = tuple(
        _resolve_descriptor_path(entry, root_paths)
        for entry in descriptor_entries
    )
    for result in resolved:
        if isinstance(result, TaskValidationIssue):
            issues.append(result)
        else:
            issues.extend(
                _validate_resolved_file(
                    definition,
                    result.path,
                    result.descriptor_path,
                )
            )
    if issues:
        raise TaskFileMaterializationError(tuple(issues))

    materialized: list[TaskMaterializedFile] = []
    for result in resolved:
        assert isinstance(result, _ResolvedInputFile)
        content = result.path.read_bytes()
        digest = sha256(content).hexdigest()
        identity = _safe_file_identity(
            result.descriptor,
            digest,
            hmac_provider=hmac_provider,
        )
        ref = await artifact_store.put(
            content,
            media_type=result.descriptor.mime_type,
            metadata={
                "identity": identity,
                "source_kind": result.descriptor.source_kind.value,
            },
        )
        if task_store is not None:
            await task_store.append_artifact(
                run_id or "",
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
                attempt_id=attempt_id,
                provenance=TaskArtifactProvenance(
                    operation="materialization",
                    metadata={
                        "identity": identity,
                        "source_kind": result.descriptor.source_kind.value,
                    },
                ),
                retention=TaskArtifactRetention(
                    delete_after_days=definition.artifact.retention_days,
                ),
                metadata={"identity": identity},
            )
        materialized.append(
            TaskMaterializedFile(
                descriptor=result.descriptor,
                descriptor_path=result.descriptor_path,
                ref=ref,
                identity=identity,
            )
        )
    return tuple(materialized)


def task_file_descriptors_from_input(
    definition: TaskDefinition,
    value: object,
) -> tuple[TaskFileDescriptor, ...]:
    entries, issues = _task_file_descriptor_entries_from_input(
        definition,
        value,
    )
    if issues:
        raise ValueError("task input contains invalid file descriptors")
    return tuple(entry.descriptor for entry in entries)


@dataclass(frozen=True, slots=True, kw_only=True)
class _InputFileDescriptor:
    descriptor: TaskFileDescriptor
    path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _InputFileDescriptorExtraction:
    entries: tuple[_InputFileDescriptor, ...]
    issues: tuple[TaskValidationIssue, ...]


def _task_file_descriptor_entries_from_input(
    definition: TaskDefinition,
    value: object,
) -> tuple[tuple[_InputFileDescriptor, ...], tuple[TaskValidationIssue, ...]]:
    assert isinstance(definition, TaskDefinition)
    if value is None:
        return (), ()
    match definition.input.type:
        case TaskInputType.FILE:
            return _file_descriptor_entries_from_values(
                ((value, "input"),),
            )
        case TaskInputType.FILE_ARRAY:
            if not isinstance(value, list | tuple):
                return (), ()
            return _file_descriptor_entries_from_values(
                (item, f"input[{index}]") for index, item in enumerate(value)
            )
        case TaskInputType.OBJECT | TaskInputType.ARRAY:
            extraction = _collect_file_descriptor_entries(value, "input")
            return extraction.entries, extraction.issues
        case _:
            return (), ()


def _file_descriptor_entries_from_values(
    values: Iterable[tuple[object, str]],
) -> tuple[tuple[_InputFileDescriptor, ...], tuple[TaskValidationIssue, ...]]:
    entries: list[_InputFileDescriptor] = []
    issues: list[TaskValidationIssue] = []
    for value, path in values:
        entry = _file_descriptor_entry(value, path)
        if isinstance(entry, TaskValidationIssue):
            issues.append(entry)
        elif entry is not None:
            entries.append(entry)
    return tuple(entries), tuple(issues)


def _collect_file_descriptor_entries(
    value: object,
    path: str,
) -> _InputFileDescriptorExtraction:
    entries: list[_InputFileDescriptor] = []
    issues: list[TaskValidationIssue] = []
    _collect_file_descriptor_entries_into(value, path, entries, issues)
    return _InputFileDescriptorExtraction(
        entries=tuple(entries),
        issues=tuple(issues),
    )


def _collect_file_descriptor_entries_into(
    value: object,
    path: str,
    entries: list[_InputFileDescriptor],
    issues: list[TaskValidationIssue],
) -> None:
    entry = _file_descriptor_entry(value, path)
    if isinstance(entry, TaskValidationIssue):
        issues.append(entry)
        return
    if entry is not None:
        entries.append(entry)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _collect_file_descriptor_entries_into(
                    item,
                    f"{path}.{key}",
                    entries,
                    issues,
                )
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _collect_file_descriptor_entries_into(
                item,
                f"{path}[{index}]",
                entries,
                issues,
            )


def _file_descriptor_entry(
    value: object,
    path: str,
) -> _InputFileDescriptor | TaskValidationIssue | None:
    if isinstance(value, TaskFileDescriptor):
        return _InputFileDescriptor(descriptor=value, path=path)
    if not isinstance(value, Mapping):
        return None
    source_kind = value.get("source_kind")
    if source_kind is None:
        return None
    if not isinstance(source_kind, str | TaskFileSourceKind):
        return _descriptor_issue(path=f"{path}.source_kind")
    try:
        TaskFileSourceKind(source_kind)
    except ValueError:
        return _descriptor_issue(path=f"{path}.source_kind")
    reference = value.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        return _descriptor_issue(path=f"{path}.reference")
    conversions = value.get("conversions", ())
    if not isinstance(conversions, list | tuple):
        return _descriptor_issue(path=f"{path}.conversions")
    try:
        descriptor = _coerce_file_descriptor(value)
    except (AssertionError, KeyError, ValueError):
        return _descriptor_issue(path=path)
    return _InputFileDescriptor(descriptor=descriptor, path=path)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ResolvedInputFile:
    descriptor: TaskFileDescriptor
    descriptor_path: str
    path: Path


def _coerce_file_descriptor(value: object) -> TaskFileDescriptor:
    assert isinstance(value, Mapping)
    source_kind = value.get("source_kind")
    assert isinstance(source_kind, str | TaskFileSourceKind)
    conversions = value.get("conversions", ())
    assert isinstance(conversions, list | tuple)
    return TaskFileDescriptor(
        source_kind=(
            source_kind
            if isinstance(source_kind, TaskFileSourceKind)
            else TaskFileSourceKind(source_kind)
        ),
        reference=str(value["reference"]),
        role=_optional_string(value.get("role")),
        mime_type=_optional_string(value.get("mime_type")),
        size_bytes=_optional_int(value.get("size_bytes")),
        sha256=_optional_string(value.get("sha256")),
        conversions=tuple(
            _coerce_conversion(conversion) for conversion in conversions
        ),
        metadata=_mapping_or_empty(value.get("metadata")),
    )


def _coerce_conversion(value: object) -> TaskFileConversionRequest:
    if isinstance(value, TaskFileConversionRequest):
        return value
    if isinstance(value, str):
        return TaskFileConversionRequest(name=value)
    assert isinstance(value, Mapping)
    return TaskFileConversionRequest(
        name=str(value["name"]),
        options=_mapping_or_empty(value.get("options")),
    )


def _resolve_descriptor_path(
    entry: _InputFileDescriptor,
    roots: tuple[Path, ...],
) -> _ResolvedInputFile | TaskValidationIssue:
    descriptor = entry.descriptor
    path = entry.path
    if descriptor.source_kind != TaskFileSourceKind.LOCAL_PATH:
        return _issue(
            code="input.invalid_file",
            path=f"{path}.source_kind",
            message="Task file source kind cannot be materialized locally.",
            hint=(
                "Use a local path descriptor or a source with a supported"
                " materializer."
            ),
            category=TaskValidationCategory.UNSUPPORTED,
        )
    if not roots:
        return _issue(
            code="input.invalid_file",
            path=path,
            message="Task file materialization has no allowlisted roots.",
            hint="Configure at least one input file root.",
            category=TaskValidationCategory.UNSUPPORTED,
        )
    if _has_traversal(descriptor.reference):
        return _issue(
            code="input.invalid_file",
            path=f"{path}.reference",
            message="Task file reference is not allowed.",
            hint="Use a path that does not contain traversal segments.",
            category=TaskValidationCategory.VALUE,
        )
    for root in roots:
        resolved = _resolve_within_root(descriptor.reference, root)
        if resolved is not None:
            return _ResolvedInputFile(
                descriptor=descriptor,
                descriptor_path=path,
                path=resolved,
            )
    return _issue(
        code="input.invalid_file",
        path=f"{path}.reference",
        message="Task file reference is outside the allowed roots.",
        hint="Use a file within an allowlisted root.",
        category=TaskValidationCategory.VALUE,
    )


def _resolve_within_root(reference: str, root: Path) -> Path | None:
    try:
        resolved_root = root.resolve(strict=True)
        candidate = Path(reference)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or not _is_relative_to(resolved, resolved_root):
        return None
    return resolved


def _validate_count_limits(
    definition: TaskDefinition,
    descriptors: tuple[TaskFileDescriptor, ...],
) -> tuple[TaskValidationIssue, ...]:
    count = len(descriptors)
    limit = definition.artifact.max_count
    if limit is None or count <= limit:
        return ()
    return (
        _issue(
            code="input.invalid_file",
            path="input",
            message="Task input exceeds the file count limit.",
            hint="Pass fewer file descriptors.",
            category=TaskValidationCategory.VALUE,
        ),
    )


def _validate_resolved_file(
    definition: TaskDefinition,
    path: Path,
    descriptor_path: str,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    size = path.stat().st_size
    limits = (
        definition.limits.file_bytes,
        definition.artifact.max_bytes,
    )
    if any(limit is not None and size > limit for limit in limits):
        issues.append(
            _issue(
                code="input.invalid_file",
                path=f"{descriptor_path}.size_bytes",
                message="Task file exceeds the byte limit.",
                hint="Pass a smaller file.",
                category=TaskValidationCategory.VALUE,
            )
        )
    return tuple(issues)


def _safe_file_identity(
    descriptor: TaskFileDescriptor,
    digest: str,
    *,
    hmac_provider: HmacProvider | None,
) -> TaskSnapshotMetadata:
    value = {
        "content_sha256": digest,
        "reference": descriptor.reference,
        "source_kind": descriptor.source_kind.value,
    }
    identity: Mapping[str, object]
    if hmac_provider is None:
        identity = {"privacy": "<redacted>"}
    else:
        try:
            sanitized = PrivacySanitizer(
                TaskPrivacyPolicy(files=PrivacyAction.HASH),
                hmac_provider=hmac_provider,
            ).sanitize(PrivacyField.FILES, value)
            identity = (
                sanitized
                if isinstance(sanitized, Mapping)
                else {"privacy": "<redacted>"}
            )
        except PrivacySanitizationError:
            identity = {"privacy": "<redacted>"}
    return freeze_snapshot_metadata(identity)


def _issue(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    category: TaskValidationCategory,
) -> TaskValidationIssue:
    return TaskValidationIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=category,
    )


def _descriptor_issue(*, path: str) -> TaskValidationIssue:
    return _issue(
        code="input.invalid_file",
        path=path,
        message="Task file descriptor is invalid.",
        hint="Pass a file descriptor with a valid source kind and reference.",
        category=TaskValidationCategory.VALUE,
    )


def _has_traversal(reference: str) -> bool:
    return (
        ".." in PurePath(reference).parts
        or ".." in PureWindowsPath(reference).parts
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_issue_code(issues: Iterable[TaskValidationIssue], code: str) -> bool:
    return any(issue.code == code for issue in issues)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    assert isinstance(value, str)
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    assert isinstance(value, int)
    return value


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    assert isinstance(value, Mapping)
    return value
