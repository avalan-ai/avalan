from .artifact import (
    ArtifactStore,
    ArtifactStoreError,
    ArtifactStorePolicyError,
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
    TaskProviderReference,
    TaskProviderReferenceKind,
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
from datetime import datetime
from hashlib import sha256
from ipaddress import ip_address
from os import O_NOFOLLOW, O_RDONLY, close, fdopen, fstat
from os import open as open_file_descriptor
from pathlib import Path, PurePath, PureWindowsPath
from socket import SOCK_STREAM, gaierror, getaddrinfo
from stat import S_ISREG
from typing import BinaryIO, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_REMOTE_URL_DISABLED_CODE = "feature.remote_url_file_inputs_disabled"
_REMOTE_URL_DELIVERY = "avalan_fetched_url"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRemoteUrlResponse:
    status_code: int
    headers: Mapping[str, str]
    stream: BinaryIO

    def __post_init__(self) -> None:
        assert isinstance(self.status_code, int)
        assert not isinstance(self.status_code, bool)
        assert 100 <= self.status_code <= 599
        assert isinstance(self.headers, Mapping)
        assert hasattr(self.stream, "read"), "stream must be readable"


class TaskRemoteUrlResolver(Protocol):
    def resolve(self, hostname: str) -> tuple[str, ...]: ...


class TaskRemoteUrlHttpClient(Protocol):
    def open(
        self,
        url: str,
        *,
        timeout_seconds: float,
    ) -> TaskRemoteUrlResponse: ...


class DefaultTaskRemoteUrlResolver:
    def resolve(self, hostname: str) -> tuple[str, ...]:
        assert isinstance(hostname, str) and hostname.strip()
        try:
            infos = getaddrinfo(hostname, None, type=SOCK_STREAM)
        except gaierror as error:
            raise OSError("remote host resolution failed") from error
        addresses = tuple(
            sorted(
                {str(info[4][0]).split("%", 1)[0] for info in infos if info[4]}
            )
        )
        if not addresses:
            raise OSError("remote host resolution failed")
        return addresses


class UrllibTaskRemoteUrlHttpClient:
    def __init__(self) -> None:
        self._opener = build_opener(_NoRedirectHandler)

    def open(
        self,
        url: str,
        *,
        timeout_seconds: float,
    ) -> TaskRemoteUrlResponse:
        assert isinstance(url, str) and url.strip()
        assert isinstance(timeout_seconds, int | float)
        request = Request(url, method="GET")
        try:
            response = self._opener.open(
                request,
                timeout=float(timeout_seconds),
            )
        except HTTPError as error:
            if 300 <= error.code <= 399:
                return TaskRemoteUrlResponse(
                    status_code=error.code,
                    headers=dict(error.headers.items()),
                    stream=cast(BinaryIO, error),
                )
            raise OSError("remote URL request failed") from error
        except URLError as error:
            raise OSError("remote URL request failed") from error
        return TaskRemoteUrlResponse(
            status_code=response.status,
            headers=dict(response.headers.items()),
            stream=cast(BinaryIO, response),
        )


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        _ = req, fp, code, msg, headers, newurl
        return None


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
    remote_url_http_client: TaskRemoteUrlHttpClient | None = None,
    remote_url_resolver: TaskRemoteUrlResolver | None = None,
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
    _extend_missing_issue_paths(issues, descriptor_issues)
    if not descriptor_entries:
        if issues:
            raise TaskFileMaterializationError(tuple(issues))
        return ()
    descriptors = tuple(entry.descriptor for entry in descriptor_entries)
    issues.extend(_validate_count_limits(definition, descriptors))
    issues = _deduplicate_issues(issues)
    local_entries = tuple(
        entry
        for entry in descriptor_entries
        if entry.descriptor.source_kind == TaskFileSourceKind.LOCAL_PATH
    )
    remote_entries = tuple(
        entry
        for entry in descriptor_entries
        if entry.descriptor.source_kind == TaskFileSourceKind.REMOTE_URL
    )
    unsupported_entries = tuple(
        entry
        for entry in descriptor_entries
        if entry.descriptor.source_kind
        not in {
            TaskFileSourceKind.LOCAL_PATH,
            TaskFileSourceKind.PROVIDER_REFERENCE,
            TaskFileSourceKind.REMOTE_URL,
        }
    )
    if _has_issue_code(issues, _REMOTE_URL_DISABLED_CODE):
        raise TaskFileMaterializationError(tuple(issues))
    for entry in unsupported_entries:
        issues.append(_unsupported_source_issue(path=entry.path))
    issues = _deduplicate_issues(issues)
    materializable_count = len(local_entries) + len(remote_entries)
    if not materializable_count:
        if issues:
            raise TaskFileMaterializationError(tuple(issues))
        return ()
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
    resolved = tuple(
        _resolve_descriptor_path(entry, root_paths) for entry in local_entries
    )
    validated: list[_ValidatedInputFile] = []
    for result in resolved:
        if isinstance(result, TaskValidationIssue):
            issues.append(result)
        else:
            validated_result = _validate_resolved_file(definition, result)
            if isinstance(validated_result, tuple):
                issues.extend(validated_result)
            else:
                validated.append(validated_result)
    issues = _deduplicate_issues(issues)
    if issues:
        raise TaskFileMaterializationError(tuple(issues))

    validated_by_path = {
        file.resolved.descriptor_path: file for file in validated
    }
    materializable_entries = tuple(
        entry
        for entry in descriptor_entries
        if entry.descriptor.source_kind
        in {TaskFileSourceKind.LOCAL_PATH, TaskFileSourceKind.REMOTE_URL}
    )
    materialized: list[TaskMaterializedFile] = []
    for entry in materializable_entries:
        match entry.descriptor.source_kind:
            case TaskFileSourceKind.LOCAL_PATH:
                materialized.append(
                    await _materialize_validated_file(
                        definition,
                        validated_by_path[entry.path],
                        artifact_store=artifact_store,
                        hmac_provider=hmac_provider,
                        task_store=task_store,
                        run_id=run_id,
                        attempt_id=attempt_id,
                    )
                )
            case TaskFileSourceKind.REMOTE_URL:
                materialized.append(
                    await _materialize_remote_url_file(
                        definition,
                        entry,
                        artifact_store=artifact_store,
                        hmac_provider=hmac_provider,
                        remote_url_policy=remote_url_policy,
                        remote_url_http_client=remote_url_http_client,
                        remote_url_resolver=remote_url_resolver,
                        task_store=task_store,
                        run_id=run_id,
                        attempt_id=attempt_id,
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


def task_provider_reference_input_files_from_input(
    definition: TaskDefinition,
    value: object,
    *,
    now: datetime | None = None,
) -> tuple[TaskInputFile, ...]:
    entries, issues = _task_file_descriptor_entries_from_input(
        definition,
        value,
    )
    provider_entries = tuple(
        entry
        for entry in entries
        if entry.descriptor.source_kind
        == TaskFileSourceKind.PROVIDER_REFERENCE
    )
    for entry in provider_entries:
        provider_reference = entry.descriptor.provider_reference
        if provider_reference is not None and provider_reference.is_expired(
            now
        ):
            issues = (
                *issues,
                _issue(
                    code="input.invalid_file",
                    path=f"{entry.path}.provider_reference.expires_at",
                    message="Task file provider reference has expired.",
                    hint="Refresh the provider reference before execution.",
                    category=TaskValidationCategory.VALUE,
                ),
            )
    if issues:
        raise TaskFileMaterializationError(tuple(issues))
    files: list[TaskInputFile] = []
    for entry in provider_entries:
        descriptor = entry.descriptor
        provider_reference = descriptor.provider_reference
        if provider_reference is None:
            raise TaskFileMaterializationError(
                (
                    _descriptor_issue(
                        path=f"{entry.path}.provider_reference",
                    ),
                )
            )
        files.append(
            TaskInputFile(
                logical_path=(
                    "provider:"
                    f"{provider_reference.provider}:"
                    f"{provider_reference.kind.value}"
                ),
                provider_reference=provider_reference,
                media_type=descriptor.mime_type
                or provider_reference.mime_type,
                size_bytes=descriptor.size_bytes,
                metadata=_provider_reference_file_metadata(descriptor),
            )
        )
    return tuple(files)


def _provider_reference_file_metadata(
    descriptor: TaskFileDescriptor,
) -> TaskSnapshotMetadata:
    value: dict[str, object] = {
        "source_kind": descriptor.source_kind.value,
    }
    if descriptor.role is not None:
        value["role"] = descriptor.role
    if descriptor.sha256 is not None:
        value["sha256"] = descriptor.sha256
    if descriptor.metadata:
        value["metadata"] = {"privacy": "<redacted>"}
    return freeze_snapshot_metadata(value)


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
    try:
        entry = _file_descriptor_entry(value, path)
    except Exception:
        issues.append(_descriptor_issue(path=path))
        return
    if isinstance(entry, TaskValidationIssue):
        issues.append(entry)
        return
    if entry is not None:
        entries.append(entry)
        return
    if isinstance(value, Mapping):
        try:
            items = tuple(value.items())
        except Exception:
            issues.append(_descriptor_issue(path=path))
            return
        for key, item in items:
            if not isinstance(key, str):
                issues.append(_descriptor_issue(path=path))
                continue
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _ValidatedInputFile:
    resolved: _ResolvedInputFile
    size_bytes: int


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
        provider_reference=_coerce_provider_reference(
            value.get("provider_reference"),
            reference=str(value["reference"]),
        ),
        metadata=_mapping_or_empty(value.get("metadata")),
    )


def _coerce_provider_reference(
    value: object,
    *,
    reference: str,
) -> TaskProviderReference | None:
    if value is None:
        return None
    if isinstance(value, TaskProviderReference):
        return value
    assert isinstance(value, Mapping)
    raw_kind = value["kind"]
    assert isinstance(raw_kind, str | TaskProviderReferenceKind)
    raw_reference = value["reference"]
    assert isinstance(raw_reference, str)
    assert raw_reference == reference
    raw_expires_at = value.get("expires_at")
    expires_at = (
        _coerce_datetime(raw_expires_at)
        if raw_expires_at is not None
        else None
    )
    durable = value.get("durable", True)
    assert isinstance(durable, bool)
    return TaskProviderReference(
        kind=(
            raw_kind
            if isinstance(raw_kind, TaskProviderReferenceKind)
            else TaskProviderReferenceKind(raw_kind)
        ),
        provider=str(value["provider"]),
        reference=raw_reference,
        owner_scope=_optional_string(value.get("owner_scope")),
        expires_at=expires_at,
        mime_type=_optional_string(value.get("mime_type")),
        size_bucket=_optional_string(value.get("size_bucket")),
        identity_hmac=_optional_string(value.get("identity_hmac")),
        durable=durable,
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
    assert descriptor.source_kind == TaskFileSourceKind.LOCAL_PATH
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
    except (OSError, ValueError):
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
    resolved: _ResolvedInputFile,
) -> _ValidatedInputFile | tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    descriptor = resolved.descriptor
    descriptor_path = resolved.descriptor_path
    try:
        size = resolved.path.stat().st_size
    except OSError:
        return (
            _issue(
                code="input.invalid_file",
                path=f"{descriptor_path}.reference",
                message="Task file cannot be read.",
                hint="Pass an available file within an allowlisted root.",
                category=TaskValidationCategory.VALUE,
            ),
        )
    if descriptor.size_bytes is not None and size != descriptor.size_bytes:
        issues.append(
            _issue(
                code="input.invalid_file",
                path=f"{descriptor_path}.size_bytes",
                message="Task file size does not match the descriptor.",
                hint="Pass a descriptor whose size matches the file bytes.",
                category=TaskValidationCategory.VALUE,
            )
        )
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
    if issues:
        return tuple(issues)
    return _ValidatedInputFile(resolved=resolved, size_bytes=size)


async def _materialize_validated_file(
    definition: TaskDefinition,
    validated: _ValidatedInputFile,
    *,
    artifact_store: ArtifactStore,
    hmac_provider: HmacProvider | None,
    task_store: TaskStore | None,
    run_id: str | None,
    attempt_id: str | None,
) -> TaskMaterializedFile:
    resolved = validated.resolved
    descriptor = resolved.descriptor
    stream_result = _open_verified_file_stream(
        resolved.path,
        descriptor_path=resolved.descriptor_path,
    )
    if isinstance(stream_result, TaskValidationIssue):
        raise TaskFileMaterializationError((stream_result,))
    with stream_result as file:
        stream = _HashingInputStream(file)
        try:
            ref = await artifact_store.put_stream(
                cast(BinaryIO, stream),
                media_type=descriptor.mime_type,
                metadata={"source_kind": descriptor.source_kind.value},
                max_bytes=_stream_size_limit(definition),
                expected_size_bytes=validated.size_bytes,
                expected_sha256=descriptor.sha256,
            )
        except ArtifactStorePolicyError as error:
            issue = _validated_stream_issue(
                resolved,
                stream,
                expected_size_bytes=validated.size_bytes,
                default_to_limit=True,
            )
            assert issue is not None
            raise TaskFileMaterializationError((issue,)) from error
        except ArtifactStoreError as error:
            issue = _validated_stream_issue(
                resolved,
                stream,
                expected_size_bytes=validated.size_bytes,
                default_to_limit=False,
            )
            if issue is None:
                raise
            raise TaskFileMaterializationError((issue,)) from error
    digest = ref.sha256 or stream.sha256
    identity = _safe_file_identity(
        descriptor,
        digest,
        hmac_provider=hmac_provider,
    )
    ref = TaskArtifactRef(
        artifact_id=ref.artifact_id,
        store=ref.store,
        storage_key=ref.storage_key,
        media_type=ref.media_type,
        size_bytes=(
            ref.size_bytes if ref.size_bytes is not None else stream.size_bytes
        ),
        sha256=digest,
        metadata=freeze_snapshot_metadata(
            {
                **dict(ref.metadata),
                "identity": identity,
                "source_kind": descriptor.source_kind.value,
            }
        ),
    )
    if task_store is not None:
        try:
            await task_store.append_artifact(
                run_id or "",
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
                attempt_id=attempt_id,
                provenance=TaskArtifactProvenance(
                    operation="materialization",
                    metadata={
                        "identity": identity,
                        "source_kind": descriptor.source_kind.value,
                    },
                ),
                retention=TaskArtifactRetention(
                    delete_after_days=definition.artifact.retention_days,
                ),
                metadata={"identity": identity},
            )
        except BaseException:
            await artifact_store.delete(ref)
            raise
    return TaskMaterializedFile(
        descriptor=descriptor,
        descriptor_path=resolved.descriptor_path,
        ref=ref,
        identity=identity,
    )


async def _materialize_remote_url_file(
    definition: TaskDefinition,
    entry: _InputFileDescriptor,
    *,
    artifact_store: ArtifactStore,
    hmac_provider: HmacProvider | None,
    remote_url_policy: TaskRemoteUrlPolicy | None,
    remote_url_http_client: TaskRemoteUrlHttpClient | None,
    remote_url_resolver: TaskRemoteUrlResolver | None,
    task_store: TaskStore | None,
    run_id: str | None,
    attempt_id: str | None,
) -> TaskMaterializedFile:
    policy = remote_url_policy
    assert policy is not None and policy.enabled
    client = remote_url_http_client or UrllibTaskRemoteUrlHttpClient()
    resolver = remote_url_resolver or DefaultTaskRemoteUrlResolver()
    response = _open_remote_url_response(
        entry.descriptor.reference,
        descriptor_path=entry.path,
        policy=policy,
        client=client,
        resolver=resolver,
    )
    if isinstance(response, TaskValidationIssue):
        raise TaskFileMaterializationError((response,))
    try:
        validation = _validate_remote_url_response(
            definition,
            entry,
            response=response,
            policy=policy,
        )
        if isinstance(validation, tuple):
            raise TaskFileMaterializationError(validation)
        return await _store_remote_url_response(
            definition,
            entry,
            response=response,
            validation=validation,
            artifact_store=artifact_store,
            hmac_provider=hmac_provider,
            task_store=task_store,
            run_id=run_id,
            attempt_id=attempt_id,
        )
    finally:
        _close_stream(response.stream)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ValidatedRemoteUrlResponse:
    media_type: str | None
    size_bytes: int
    max_bytes: int


async def _store_remote_url_response(
    definition: TaskDefinition,
    entry: _InputFileDescriptor,
    *,
    response: TaskRemoteUrlResponse,
    validation: _ValidatedRemoteUrlResponse,
    artifact_store: ArtifactStore,
    hmac_provider: HmacProvider | None,
    task_store: TaskStore | None,
    run_id: str | None,
    attempt_id: str | None,
) -> TaskMaterializedFile:
    descriptor = entry.descriptor
    stream = _HashingInputStream(response.stream)
    try:
        ref = await artifact_store.put_stream(
            cast(BinaryIO, stream),
            media_type=descriptor.mime_type or validation.media_type,
            metadata={
                "remote_delivery": _REMOTE_URL_DELIVERY,
                "source_kind": descriptor.source_kind.value,
            },
            max_bytes=validation.max_bytes,
            expected_size_bytes=validation.size_bytes,
            expected_sha256=descriptor.sha256,
        )
    except ArtifactStorePolicyError as error:
        issue = _validated_remote_stream_issue(
            entry,
            stream,
            expected_size_bytes=validation.size_bytes,
            default_to_limit=True,
        )
        assert issue is not None
        raise TaskFileMaterializationError((issue,)) from error
    except (ArtifactStoreError, OSError, TimeoutError) as error:
        issue = _validated_remote_stream_issue(
            entry,
            stream,
            expected_size_bytes=validation.size_bytes,
            default_to_limit=False,
        )
        if issue is None:
            raise
        raise TaskFileMaterializationError((issue,)) from error
    digest = ref.sha256 or stream.sha256
    identity = _safe_file_identity(
        descriptor,
        digest,
        hmac_provider=hmac_provider,
    )
    ref = TaskArtifactRef(
        artifact_id=ref.artifact_id,
        store=ref.store,
        storage_key=ref.storage_key,
        media_type=ref.media_type,
        size_bytes=(
            ref.size_bytes if ref.size_bytes is not None else stream.size_bytes
        ),
        sha256=digest,
        metadata=freeze_snapshot_metadata(
            {
                **dict(ref.metadata),
                "identity": identity,
                "remote_delivery": _REMOTE_URL_DELIVERY,
                "source_kind": descriptor.source_kind.value,
            }
        ),
    )
    if task_store is not None:
        try:
            await task_store.append_artifact(
                run_id or "",
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
                attempt_id=attempt_id,
                provenance=TaskArtifactProvenance(
                    operation="remote_materialization",
                    metadata={
                        "identity": identity,
                        "remote_delivery": _REMOTE_URL_DELIVERY,
                        "source_kind": descriptor.source_kind.value,
                    },
                ),
                retention=TaskArtifactRetention(
                    delete_after_days=definition.artifact.retention_days,
                ),
                metadata={"identity": identity},
            )
        except BaseException:
            await artifact_store.delete(ref)
            raise
    return TaskMaterializedFile(
        descriptor=descriptor,
        descriptor_path=entry.path,
        ref=ref,
        identity=identity,
    )


def _open_remote_url_response(
    url: str,
    *,
    descriptor_path: str,
    policy: TaskRemoteUrlPolicy,
    client: TaskRemoteUrlHttpClient,
    resolver: TaskRemoteUrlResolver,
) -> TaskRemoteUrlResponse | TaskValidationIssue:
    current_url = url
    seen = {current_url}
    redirects = 0
    while True:
        access_issue = _validate_remote_url_access(
            current_url,
            descriptor_path=descriptor_path,
            policy=policy,
            resolver=resolver,
        )
        if access_issue is not None:
            return access_issue
        try:
            response = client.open(
                current_url,
                timeout_seconds=float(policy.timeout_seconds),
            )
        except TimeoutError:
            return _remote_issue(
                path=f"{descriptor_path}.reference",
                message="Remote URL file request timed out.",
                hint="Use a faster source or increase the remote URL timeout.",
                category=TaskValidationCategory.VALUE,
            )
        except Exception:
            return _remote_issue(
                path=f"{descriptor_path}.reference",
                message="Remote URL file could not be fetched.",
                hint="Use an available remote file source.",
                category=TaskValidationCategory.VALUE,
            )
        if not _is_redirect_response(response.status_code):
            if 200 <= response.status_code <= 299:
                return response
            _close_stream(response.stream)
            return _remote_issue(
                path=f"{descriptor_path}.reference",
                message="Remote URL file returned an unsupported status.",
                hint="Use a URL that returns a successful response.",
                category=TaskValidationCategory.VALUE,
            )
        redirect_issue_or_url = _redirect_target_url(
            current_url,
            response,
            descriptor_path=descriptor_path,
            policy=policy,
            seen=seen,
            redirects=redirects,
        )
        _close_stream(response.stream)
        if isinstance(redirect_issue_or_url, TaskValidationIssue):
            return redirect_issue_or_url
        current_url = redirect_issue_or_url
        seen.add(current_url)
        redirects += 1


def _validate_remote_url_access(
    url: str,
    *,
    descriptor_path: str,
    policy: TaskRemoteUrlPolicy,
    resolver: TaskRemoteUrlResolver,
) -> TaskValidationIssue | None:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _port = parsed.port
    except ValueError:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file reference is invalid.",
            hint="Use a valid absolute URL.",
            category=TaskValidationCategory.VALUE,
        )
    scheme = parsed.scheme.lower()
    if scheme not in policy.allowed_schemes:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file scheme is not allowed.",
            hint="Use an allowed remote URL scheme.",
            category=TaskValidationCategory.VALUE,
        )
    if not hostname:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file host is invalid.",
            hint="Use an absolute URL with a host.",
            category=TaskValidationCategory.VALUE,
        )
    if parsed.username is not None or parsed.password is not None:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file credentials are not allowed.",
            hint="Pass credentials through a configured secret provider.",
            category=TaskValidationCategory.VALUE,
        )
    if not policy.allow_private_networks and _is_private_network_host(
        hostname
    ):
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file host is not allowed.",
            hint="Use a public remote host or disable remote URL inputs.",
            category=TaskValidationCategory.VALUE,
        )
    try:
        addresses = resolver.resolve(hostname)
    except OSError:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file host could not be resolved.",
            hint="Use a resolvable public remote host.",
            category=TaskValidationCategory.VALUE,
        )
    if not addresses:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file host could not be resolved.",
            hint="Use a resolvable public remote host.",
            category=TaskValidationCategory.VALUE,
        )
    if not policy.allow_private_networks and any(
        _is_private_network_address(address) for address in addresses
    ):
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file host is not allowed.",
            hint="Use a public remote host or disable remote URL inputs.",
            category=TaskValidationCategory.VALUE,
        )
    return None


def _redirect_target_url(
    current_url: str,
    response: TaskRemoteUrlResponse,
    *,
    descriptor_path: str,
    policy: TaskRemoteUrlPolicy,
    seen: set[str],
    redirects: int,
) -> str | TaskValidationIssue:
    if not policy.allow_redirects:
        return _remote_issue(
            path=f"{descriptor_path}.redirects",
            message="Remote URL redirects are disabled.",
            hint="Use a URL that does not redirect or enable redirects.",
            category=TaskValidationCategory.VALUE,
        )
    if redirects >= policy.max_redirects:
        return _remote_issue(
            path=f"{descriptor_path}.redirects",
            message="Remote URL redirect limit was exceeded.",
            hint="Use a direct URL or increase the redirect limit.",
            category=TaskValidationCategory.VALUE,
        )
    location = _header_value(response.headers, "location")
    if location is None or not location.strip():
        return _remote_issue(
            path=f"{descriptor_path}.redirects",
            message="Remote URL redirect target is invalid.",
            hint="Use a redirect with a valid Location header.",
            category=TaskValidationCategory.VALUE,
        )
    next_url = urljoin(current_url, location)
    if next_url in seen:
        return _remote_issue(
            path=f"{descriptor_path}.redirects",
            message="Remote URL redirect loop was detected.",
            hint="Use a URL with a finite redirect chain.",
            category=TaskValidationCategory.VALUE,
        )
    return next_url


def _validate_remote_url_response(
    definition: TaskDefinition,
    entry: _InputFileDescriptor,
    *,
    response: TaskRemoteUrlResponse,
    policy: TaskRemoteUrlPolicy,
) -> _ValidatedRemoteUrlResponse | tuple[TaskValidationIssue, ...]:
    descriptor = entry.descriptor
    media_type = _response_media_type(response)
    size = _response_size(response)
    issues: list[TaskValidationIssue] = []
    if size is None:
        issues.append(
            _remote_issue(
                path=f"{entry.path}.size_bytes",
                message="Remote URL file size is unknown.",
                hint="Use a source that sends a valid Content-Length header.",
                category=TaskValidationCategory.VALUE,
            )
        )
    elif size < 0:
        issues.append(
            _remote_issue(
                path=f"{entry.path}.size_bytes",
                message="Remote URL file size is invalid.",
                hint="Use a source that sends a valid Content-Length header.",
                category=TaskValidationCategory.VALUE,
            )
        )
    elif descriptor.size_bytes is not None and size != descriptor.size_bytes:
        issues.append(
            _remote_issue(
                path=f"{entry.path}.size_bytes",
                message="Remote URL file size does not match the descriptor.",
                hint="Pass a descriptor whose size matches the remote file.",
                category=TaskValidationCategory.VALUE,
            )
        )
    max_bytes = _remote_url_size_limit(definition, policy)
    if size is not None and size > max_bytes:
        issues.append(
            _remote_issue(
                path=f"{entry.path}.size_bytes",
                message="Remote URL file exceeds the byte limit.",
                hint="Use a smaller remote file.",
                category=TaskValidationCategory.VALUE,
            )
        )
    if descriptor.mime_type is not None and media_type != descriptor.mime_type:
        issues.append(
            _remote_issue(
                path=f"{entry.path}.mime_type",
                message="Remote URL file MIME type does not match.",
                hint="Pass a descriptor whose MIME type matches the response.",
                category=TaskValidationCategory.VALUE,
            )
        )
    if issues:
        return tuple(issues)
    assert size is not None
    assert max_bytes is not None
    return _ValidatedRemoteUrlResponse(
        media_type=media_type,
        size_bytes=size,
        max_bytes=max_bytes,
    )


def _open_verified_file_stream(
    path: Path,
    *,
    descriptor_path: str,
) -> BinaryIO | TaskValidationIssue:
    try:
        return _open_regular_file_stream(path)
    except OSError:
        return _issue(
            code="input.invalid_file",
            path=f"{descriptor_path}.reference",
            message="Task file cannot be read.",
            hint="Pass an available file within an allowlisted root.",
            category=TaskValidationCategory.VALUE,
        )


def _open_regular_file_stream(path: Path) -> BinaryIO:
    file_descriptor = open_file_descriptor(path, O_RDONLY | O_NOFOLLOW)
    try:
        if not S_ISREG(fstat(file_descriptor).st_mode):
            raise OSError
    except Exception:
        close(file_descriptor)
        raise
    return fdopen(file_descriptor, "rb")


def _stream_size_limit(
    definition: TaskDefinition,
) -> int | None:
    limits = tuple(
        limit
        for limit in (
            definition.limits.file_bytes,
            definition.artifact.max_bytes,
        )
        if limit is not None
    )
    if not limits:
        return None
    return min(limits)


def _validated_stream_issue(
    resolved: _ResolvedInputFile,
    stream: "_HashingInputStream",
    *,
    expected_size_bytes: int,
    default_to_limit: bool,
) -> TaskValidationIssue | None:
    descriptor = resolved.descriptor
    descriptor_path = resolved.descriptor_path
    if stream.read_failed:
        return _issue(
            code="input.invalid_file",
            path=f"{descriptor_path}.reference",
            message="Task file cannot be read.",
            hint="Pass an available file within an allowlisted root.",
            category=TaskValidationCategory.VALUE,
        )
    if stream.read_complete and stream.size_bytes != expected_size_bytes:
        return _issue(
            code="input.invalid_file",
            path=f"{descriptor_path}.size_bytes",
            message="Task file size does not match the descriptor.",
            hint="Pass a descriptor whose size matches the file bytes.",
            category=TaskValidationCategory.VALUE,
        )
    if (
        stream.read_complete
        and descriptor.sha256 is not None
        and stream.sha256 != descriptor.sha256
    ):
        return _issue(
            code="input.invalid_file",
            path=f"{descriptor_path}.sha256",
            message="Task file digest does not match the descriptor.",
            hint="Pass a descriptor whose digest matches the file bytes.",
            category=TaskValidationCategory.VALUE,
        )
    if default_to_limit:
        return _issue(
            code="input.invalid_file",
            path=f"{descriptor_path}.size_bytes",
            message="Task file exceeds the byte limit.",
            hint="Pass a smaller file.",
            category=TaskValidationCategory.VALUE,
        )
    return None


def _validated_remote_stream_issue(
    entry: _InputFileDescriptor,
    stream: "_HashingInputStream",
    *,
    expected_size_bytes: int,
    default_to_limit: bool,
) -> TaskValidationIssue | None:
    descriptor_path = entry.path
    descriptor = entry.descriptor
    if stream.timed_out:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file read timed out.",
            hint="Use a faster source or increase the remote URL timeout.",
            category=TaskValidationCategory.VALUE,
        )
    if stream.read_failed:
        return _remote_issue(
            path=f"{descriptor_path}.reference",
            message="Remote URL file cannot be read.",
            hint="Use an available remote file source.",
            category=TaskValidationCategory.VALUE,
        )
    if stream.read_complete and stream.size_bytes != expected_size_bytes:
        return _remote_issue(
            path=f"{descriptor_path}.size_bytes",
            message="Remote URL file size does not match the descriptor.",
            hint="Use a source with a stable Content-Length value.",
            category=TaskValidationCategory.VALUE,
        )
    if (
        stream.read_complete
        and descriptor.sha256 is not None
        and stream.sha256 != descriptor.sha256
    ):
        return _remote_issue(
            path=f"{descriptor_path}.sha256",
            message="Remote URL file digest does not match the descriptor.",
            hint="Pass a descriptor whose digest matches the remote file.",
            category=TaskValidationCategory.VALUE,
        )
    if default_to_limit:
        return _remote_issue(
            path=f"{descriptor_path}.size_bytes",
            message="Remote URL file exceeds the byte limit.",
            hint="Use a smaller remote file.",
            category=TaskValidationCategory.VALUE,
        )
    return None


def _remote_url_size_limit(
    definition: TaskDefinition,
    policy: TaskRemoteUrlPolicy,
) -> int:
    limits = tuple(
        limit
        for limit in (
            policy.max_bytes,
            definition.limits.file_bytes,
            definition.artifact.max_bytes,
        )
        if limit is not None
    )
    assert limits, "remote URL materialization requires a byte limit"
    return min(limits)


def _response_media_type(response: TaskRemoteUrlResponse) -> str | None:
    value = _header_value(response.headers, "content-type")
    if value is None:
        return None
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type or None


def _response_size(response: TaskRemoteUrlResponse) -> int | None:
    value = _header_value(response.headers, "content-length")
    if value is None:
        return None
    try:
        size = int(value)
    except ValueError:
        return -1
    if size < 0:
        return -1
    return size


def _header_value(
    headers: Mapping[str, str],
    name: str,
) -> str | None:
    normalized = name.lower()
    for key, value in headers.items():
        if key.lower() == normalized:
            return value
    return None


def _is_redirect_response(status_code: int) -> bool:
    return status_code in {301, 302, 303, 307, 308}


def _is_private_network_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if (
        normalized == "localhost"
        or normalized.endswith(".localhost")
        or normalized.endswith(".local")
    ):
        return True
    return _is_private_network_address(normalized)


def _is_private_network_address(value: str) -> bool:
    try:
        address = ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


class _HashingInputStream:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = sha256()
        self.size_bytes = 0
        self.read_failed = False
        self.timed_out = False
        self.read_complete = False

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def read(self, size: int = -1) -> bytes:
        try:
            chunk = self._stream.read(size)
        except TimeoutError:
            self.timed_out = True
            raise
        except OSError:
            self.read_failed = True
            raise
        assert isinstance(chunk, bytes), "input file stream must return bytes"
        if not chunk:
            self.read_complete = True
        self.size_bytes += len(chunk)
        self._digest.update(chunk)
        return chunk


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


def _unsupported_source_issue(*, path: str) -> TaskValidationIssue:
    return _issue(
        code="input.invalid_file",
        path=f"{path}.source_kind",
        message="Task file source kind cannot be materialized.",
        hint="Use a local path, remote URL, or provider reference descriptor.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _remote_issue(
    *,
    path: str,
    message: str,
    hint: str,
    category: TaskValidationCategory,
) -> TaskValidationIssue:
    return _issue(
        code="input.invalid_file",
        path=path,
        message=message,
        hint=hint,
        category=category,
    )


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


def _deduplicate_issues(
    issues: Iterable[TaskValidationIssue],
) -> list[TaskValidationIssue]:
    seen: set[tuple[str, str, str, str]] = set()
    deduplicated: list[TaskValidationIssue] = []
    for issue in issues:
        key = (issue.code, issue.path, issue.message, issue.hint)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(issue)
    return deduplicated


def _extend_missing_issue_paths(
    issues: list[TaskValidationIssue],
    new_issues: Iterable[TaskValidationIssue],
) -> None:
    existing = {(issue.code, issue.path) for issue in issues}
    for issue in new_issues:
        key = (issue.code, issue.path)
        if key in existing or any(
            issue.code == existing_issue.code
            and _issue_paths_overlap(existing_issue.path, issue.path)
            for existing_issue in issues
        ):
            continue
        existing.add(key)
        issues.append(issue)


def _issue_paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(f"{right}.")
        or left.startswith(f"{right}[")
        or right.startswith(f"{left}.")
        or right.startswith(f"{left}[")
    )


def _close_stream(stream: BinaryIO) -> None:
    close_stream = getattr(stream, "close", None)
    if callable(close_stream):
        close_stream()


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


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    assert isinstance(value, str)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
