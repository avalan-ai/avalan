from .contract import (
    SKILL_MAIN_RESOURCE_FILENAME,
    SKILL_MAIN_RESOURCE_ID,
    SkillDiagnosticCode,
    SkillFailureMode,
    SkillManifestField,
    SkillStatus,
)
from .entities import (
    SkillDiagnosticInfo,
    SkillMetadata,
    SkillModelValue,
    SkillResourceHandle,
    diagnostic_from_failure,
    model_dict,
)
from .normalizer import (
    normalize_skill_description,
    normalize_skill_name,
    normalize_skill_resource_id,
    normalize_skill_source_label,
    normalize_skill_tags,
    skill_name_denial_reason,
    skill_resource_denial_reason,
)
from .path_policy import skill_model_handle_denial_reason
from .resolver import (
    SkillAsyncFileSystem,
    SkillAuthorizedResource,
    SkillAuthorizedSourceRoot,
    SkillSourceFileSystem,
)
from .settings import SkillIndexLimits, SkillReadLimits

from ast import literal_eval
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from re import fullmatch
from typing import TypeAlias, cast

SKILL_ID_CONVENTION = "manifest_name_slug"
_SKILL_DASH_MANIFEST_BASENAME_PATTERN = (
    r"SKILL-([a-z][a-z0-9]*(?:[._-][a-z0-9]+)*)\.md"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillDeclaredResource:
    source_label: str
    skill_id: str
    resource_id: str
    source_resource_id: str
    media_type: str = "text/markdown"
    size_bytes: int | None = None
    recursive: bool = False
    status: SkillStatus = SkillStatus.OK

    def __post_init__(self) -> None:
        assert self.source_label == normalize_skill_source_label(
            self.source_label
        )
        assert normalize_skill_name(self.skill_id) == self.skill_id
        assert (
            self.resource_id == SKILL_MAIN_RESOURCE_ID
            or normalize_skill_resource_id(self.resource_id)
            == self.resource_id
        )
        assert (
            normalize_skill_resource_id(self.source_resource_id)
            == self.source_resource_id
        )
        assert isinstance(self.media_type, str)
        assert self.media_type == "text/markdown"
        if self.size_bytes is not None:
            assert isinstance(self.size_bytes, int)
            assert not isinstance(self.size_bytes, bool)
            assert self.size_bytes >= 0
        assert isinstance(self.recursive, bool)
        assert not self.recursive, "Phase 3 resources are non-recursive"
        assert isinstance(self.status, SkillStatus)

    def as_handle(self) -> SkillResourceHandle:
        return SkillResourceHandle(
            source_label=self.source_label,
            skill_id=self.skill_id,
            resource_id=self.resource_id,
            media_type=self.media_type,
            size_bytes=self.size_bytes,
            status=self.status,
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "source_label": self.source_label,
            "skill_id": self.skill_id,
            "resource_id": self.resource_id,
            "media_type": self.media_type,
            "recursive": self.recursive,
            "status": self.status.value,
        }
        if self.size_bytes is not None:
            value["size_bytes"] = self.size_bytes
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillManifestDocument:
    source_label: str
    manifest_resource_id: str
    package_resource_id: str
    skill_id: str | None = None
    metadata: SkillMetadata | None = None
    declared_resources: tuple[SkillDeclaredResource, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert self.source_label == normalize_skill_source_label(
            self.source_label
        )
        assert normalize_skill_resource_id(self.manifest_resource_id) in {
            self.manifest_resource_id,
            None,
        }
        assert self.package_resource_id == "." or (
            normalize_skill_resource_id(self.package_resource_id)
            == self.package_resource_id
        )
        if self.skill_id is not None:
            assert normalize_skill_name(self.skill_id) == self.skill_id
        if self.metadata is not None:
            assert isinstance(self.metadata, SkillMetadata)
            assert self.metadata.skill_id == self.skill_id
            assert self.metadata.source_label == self.source_label
        assert isinstance(self.declared_resources, tuple)
        for resource in self.declared_resources:
            assert isinstance(resource, SkillDeclaredResource)
            assert resource.source_label == self.source_label
            if self.skill_id is not None:
                assert resource.skill_id == self.skill_id
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, SkillDiagnosticInfo)

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        if self.metadata is not None:
            return self.metadata.status
        return SkillStatus.MALFORMED

    @property
    def readable(self) -> bool:
        return self.usable

    @property
    def usable(self) -> bool:
        return (
            self.metadata is not None
            and self.metadata.enabled
            and self.metadata.status == SkillStatus.OK
            and not self.diagnostics
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "source_label": self.source_label,
            "manifest_resource_id": self.manifest_resource_id,
            "package_resource_id": self.package_resource_id,
            "status": self.status.value,
            "readable": self.readable,
            "usable": self.usable,
            "declared_resources": tuple(
                resource.as_model_dict()
                for resource in self.declared_resources
            ),
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
        }
        if self.skill_id is not None:
            value["skill_id"] = self.skill_id
        if self.metadata is not None:
            value["metadata"] = self.metadata.as_model_dict()
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillManifestLoadResult:
    manifests: tuple[SkillManifestDocument, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.manifests, tuple)
        for manifest in self.manifests:
            assert isinstance(manifest, SkillManifestDocument)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, SkillDiagnosticInfo)

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        for manifest in self.manifests:
            if manifest.diagnostics:
                return manifest.diagnostics[0].status
        if self.manifests:
            return SkillStatus.OK
        return SkillStatus.EMPTY

    @property
    def usable_manifests(self) -> tuple[SkillManifestDocument, ...]:
        return tuple(
            manifest for manifest in self.manifests if manifest.usable
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "status": self.status.value,
                "manifests": tuple(
                    manifest.as_model_dict() for manifest in self.manifests
                ),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


async def parse_skill_manifests(
    sources: tuple[SkillAuthorizedSourceRoot, ...],
    *,
    file_system: SkillSourceFileSystem | None = None,
    read_limits: SkillReadLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
) -> SkillManifestLoadResult:
    """Parse manifests from authorized Phase 2 source roots."""
    assert isinstance(sources, tuple), "sources must be a tuple"
    for source in sources:
        assert isinstance(source, SkillAuthorizedSourceRoot)
    if file_system is None:
        file_system = SkillAsyncFileSystem()
    if read_limits is None:
        read_limits = SkillReadLimits()
    if index_limits is None:
        index_limits = SkillIndexLimits()

    manifests: list[SkillManifestDocument] = []
    diagnostics: list[SkillDiagnosticInfo] = []
    manifest_resources = tuple(
        resource
        for source in sources
        for resource in source.resources
        if _is_manifest_resource_id(resource.resource_id)
    )
    projected_documents = _manifest_diagnostic_documents(sources)
    if (
        len(manifest_resources) + len(projected_documents)
        > index_limits.max_skills
    ):
        return SkillManifestLoadResult(
            diagnostics=(
                _resource_oversized_diagnostic(
                    path="manifest.count",
                    reason="max_skills",
                ),
            )
        )

    source_by_label = {source.label: source for source in sources}
    for resource in manifest_resources:
        source = source_by_label[resource.source_label]
        document = await _parse_manifest_resource(
            source=source,
            resource=resource,
            file_system=file_system,
            read_limits=read_limits,
            index_limits=index_limits,
        )
        manifests.append(document)
    manifests.extend(projected_documents)

    normalized_manifests, duplicate_diagnostics = normalize_manifest_documents(
        tuple(manifests)
    )
    diagnostics.extend(duplicate_diagnostics)
    return SkillManifestLoadResult(
        manifests=normalized_manifests,
        diagnostics=tuple(diagnostics),
    )


def parse_skill_manifest_markdown(
    content: str,
    *,
    source_label: str,
    manifest_resource_id: str = SKILL_MAIN_RESOURCE_FILENAME,
    manifest_size_bytes: int | None = None,
    authorized_resources: tuple[SkillAuthorizedResource, ...] = (),
    index_limits: SkillIndexLimits | None = None,
) -> SkillManifestDocument:
    """Parse one Markdown skill manifest from trusted text."""
    assert isinstance(content, str), "content must be a string"
    assert isinstance(source_label, str), "source_label must be a string"
    assert isinstance(manifest_resource_id, str)
    assert isinstance(authorized_resources, tuple)
    for resource in authorized_resources:
        assert isinstance(resource, SkillAuthorizedResource)
    if index_limits is None:
        index_limits = SkillIndexLimits()

    safe_source_label = normalize_skill_source_label(source_label)
    safe_manifest_resource_id = _safe_manifest_resource_id(
        manifest_resource_id
    )
    package_resource_id = _package_resource_id(safe_manifest_resource_id)
    parsed, diagnostic = _parse_front_matter(content)
    if diagnostic is not None:
        return SkillManifestDocument(
            source_label=safe_source_label,
            manifest_resource_id=safe_manifest_resource_id,
            package_resource_id=package_resource_id,
            diagnostics=(diagnostic,),
        )

    normalized = _normalize_front_matter(
        parsed,
        source_label=safe_source_label,
        manifest_resource_id=safe_manifest_resource_id,
        manifest_size_bytes=manifest_size_bytes,
        package_resource_id=package_resource_id,
        authorized_resources=authorized_resources,
        index_limits=index_limits,
    )
    return normalized


def normalize_manifest_documents(
    manifests: tuple[SkillManifestDocument, ...],
) -> tuple[tuple[SkillManifestDocument, ...], tuple[SkillDiagnosticInfo, ...]]:
    """Apply cross-manifest normalization checks."""
    assert isinstance(manifests, tuple), "manifests must be a tuple"
    for manifest in manifests:
        assert isinstance(manifest, SkillManifestDocument)
    counts = Counter(
        manifest.skill_id
        for manifest in manifests
        if manifest.skill_id is not None
    )
    duplicates = tuple(
        sorted(
            skill_id
            for skill_id, count in counts.items()
            if skill_id is not None and count > 1
        )
    )
    if not duplicates:
        return manifests, ()

    diagnostic = diagnostic_from_failure(
        SkillFailureMode.DUPLICATE_SKILL_IDS,
        path="manifest.name",
        candidates=duplicates,
    )
    duplicate_set = set(duplicates)
    blocked = tuple(
        (
            _with_duplicate_diagnostic(manifest, diagnostic)
            if manifest.skill_id in duplicate_set
            else manifest
        )
        for manifest in manifests
    )
    return blocked, (diagnostic,)


SkillManifest: TypeAlias = SkillManifestDocument
SkillManifestNormalization: TypeAlias = SkillManifestLoadResult
SkillManifestNormalizationResult: TypeAlias = SkillManifestLoadResult
SkillManifestParseResult: TypeAlias = SkillManifestDocument


def normalize_skill_manifest_resource(value: str) -> str | None:
    """Return the normalized declared skill resource ID."""
    return normalize_skill_resource_id(value)


def normalize_skill_manifests(
    manifests: tuple[SkillManifestDocument, ...],
) -> SkillManifestLoadResult:
    """Return a normalized manifest load result for parsed documents."""
    normalized, diagnostics = normalize_manifest_documents(manifests)
    return SkillManifestLoadResult(
        manifests=normalized,
        diagnostics=diagnostics,
    )


async def _parse_manifest_resource(
    *,
    source: SkillAuthorizedSourceRoot,
    resource: SkillAuthorizedResource,
    file_system: SkillSourceFileSystem,
    read_limits: SkillReadLimits,
    index_limits: SkillIndexLimits,
) -> SkillManifestDocument:
    try:
        content = await file_system.read_bytes(
            resource.path,
            read_limits.max_bytes_per_read + 1,
        )
    except OSError:
        return _resource_diagnostic_document(
            source_label=source.label,
            manifest_resource_id=resource.resource_id,
            diagnostic=diagnostic_from_failure(
                SkillFailureMode.SOURCE_UNAVAILABLE,
                path="source",
            ),
        )
    if len(content) > read_limits.max_bytes_per_read:
        return _resource_diagnostic_document(
            source_label=source.label,
            manifest_resource_id=resource.resource_id,
            diagnostic=_resource_oversized_diagnostic(
                path="resource.main",
                reason="max_bytes_per_read",
            ),
        )
    if b"\x00" in content:
        return _resource_diagnostic_document(
            source_label=source.label,
            manifest_resource_id=resource.resource_id,
            diagnostic=_binary_resource_diagnostic(reason="nul_byte"),
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return _resource_diagnostic_document(
            source_label=source.label,
            manifest_resource_id=resource.resource_id,
            diagnostic=_binary_resource_diagnostic(reason="non_utf8"),
        )
    return parse_skill_manifest_markdown(
        text,
        source_label=source.label,
        manifest_resource_id=resource.resource_id,
        manifest_size_bytes=resource.size_bytes,
        authorized_resources=source.resources,
        index_limits=index_limits,
    )


def _normalize_front_matter(
    manifest: Mapping[str, object],
    *,
    source_label: str,
    manifest_resource_id: str,
    manifest_size_bytes: int | None,
    package_resource_id: str,
    authorized_resources: tuple[SkillAuthorizedResource, ...],
    index_limits: SkillIndexLimits,
) -> SkillManifestDocument:
    diagnostics: list[SkillDiagnosticInfo] = []
    for field in (
        SkillManifestField.NAME,
        SkillManifestField.DESCRIPTION,
    ):
        if field.value not in manifest:
            diagnostics.append(
                _malformed_manifest_diagnostic(
                    path=f"manifest.{field.value}",
                    reason="missing_field",
                )
            )

    raw_name = manifest.get(SkillManifestField.NAME.value)
    skill_id: str | None = None
    if isinstance(raw_name, str):
        skill_id = normalize_skill_name(raw_name)
        if skill_id is None:
            diagnostics.append(_name_diagnostic(raw_name))

    raw_description = manifest.get(SkillManifestField.DESCRIPTION.value)
    description: str | None = None
    if isinstance(raw_description, str):
        description = normalize_skill_description(raw_description)
        assert description is not None
        if not _is_model_safe_metadata_text(description):
            diagnostics.append(
                _malformed_manifest_diagnostic(
                    path="manifest.description",
                    reason="unsafe_metadata",
                )
            )

    enabled = cast(
        bool,
        manifest.get(SkillManifestField.ENABLED.value, True),
    )
    raw_tags = cast(
        tuple[str, ...],
        manifest.get(SkillManifestField.TAGS.value, ()),
    )
    tags = normalize_skill_tags(raw_tags)
    if tags is None:
        diagnostics.append(
            _malformed_manifest_diagnostic(
                path="manifest.tags",
                reason="bad_tags",
            )
        )

    version = _normalize_optional_version(
        manifest.get(SkillManifestField.VERSION.value),
    )
    if version is not None and not _is_model_safe_metadata_text(version):
        diagnostics.append(
            _malformed_manifest_diagnostic(
                path="manifest.version",
                reason="unsafe_metadata",
            )
        )

    if skill_id is None or description is None or tags is None:
        return SkillManifestDocument(
            source_label=source_label,
            manifest_resource_id=manifest_resource_id,
            package_resource_id=package_resource_id,
            skill_id=skill_id,
            diagnostics=tuple(diagnostics),
        )

    declared_resources, resource_diagnostics = _declared_resources(
        raw_resources=cast(
            tuple[str, ...],
            manifest.get(SkillManifestField.RESOURCES.value, ()),
        ),
        source_label=source_label,
        skill_id=skill_id,
        manifest_resource_id=manifest_resource_id,
        manifest_size_bytes=manifest_size_bytes,
        package_resource_id=package_resource_id,
        authorized_resources=authorized_resources,
        index_limits=index_limits,
    )
    diagnostics.extend(resource_diagnostics)
    if diagnostics:
        return SkillManifestDocument(
            source_label=source_label,
            manifest_resource_id=manifest_resource_id,
            package_resource_id=package_resource_id,
            skill_id=skill_id,
            declared_resources=declared_resources,
            diagnostics=tuple(diagnostics),
        )

    status = SkillStatus.OK if enabled else SkillStatus.DISABLED
    metadata = SkillMetadata(
        skill_id=skill_id,
        name=skill_id,
        description=description,
        source_label=source_label,
        enabled=enabled,
        status=status,
        tags=tags,
        version=version,
        resources=tuple(
            resource.as_handle() for resource in declared_resources
        ),
    )
    diagnostics_tuple: tuple[SkillDiagnosticInfo, ...] = ()
    if not enabled:
        diagnostics_tuple = (
            diagnostic_from_failure(
                SkillFailureMode.DISABLED_SKILL,
                path="manifest.enabled",
                candidates=(skill_id,),
            ),
        )
    return SkillManifestDocument(
        source_label=source_label,
        manifest_resource_id=manifest_resource_id,
        package_resource_id=package_resource_id,
        skill_id=skill_id,
        metadata=metadata,
        declared_resources=declared_resources,
        diagnostics=diagnostics_tuple,
    )


def _parse_front_matter(
    content: str,
) -> tuple[dict[str, object], SkillDiagnosticInfo | None]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, _malformed_manifest_diagnostic(
            path="manifest",
            reason="missing_front_matter",
        )

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, _malformed_manifest_diagnostic(
            path="manifest",
            reason="unclosed_front_matter",
        )

    manifest: dict[str, object] = {}
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            return {}, _malformed_manifest_diagnostic(
                path=f"manifest.line.{line_number}",
                reason="bad_line",
            )
        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        if key not in _SUPPORTED_FIELD_NAMES:
            return {}, _malformed_manifest_diagnostic(
                path=_field_diagnostic_path(key),
                reason="unsupported_field",
            )
        if key in manifest:
            return {}, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                reason="duplicate_field",
            )
        value, diagnostic = _parse_front_matter_value(
            key=key,
            raw_value=raw_value.strip(),
        )
        if diagnostic is not None:
            return {}, diagnostic
        manifest[key] = value
    return manifest, None


def _parse_front_matter_value(
    *, key: str, raw_value: str
) -> tuple[object | None, SkillDiagnosticInfo | None]:
    if key == SkillManifestField.ENABLED.value:
        if raw_value == "true":
            return True, None
        if raw_value == "false":
            return False, None
        return None, _malformed_manifest_diagnostic(
            path="manifest.enabled",
            reason="bad_enabled",
        )

    if key in {
        SkillManifestField.TAGS.value,
        SkillManifestField.RESOURCES.value,
    }:
        try:
            parsed = literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return None, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                reason="bad_list",
            )
        if not isinstance(parsed, list) or any(
            not isinstance(item, str) for item in parsed
        ):
            return None, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                reason="bad_list",
            )
        return tuple(parsed), None

    return _parse_string_metadata_value(key=key, raw_value=raw_value)


def _parse_string_metadata_value(
    *,
    key: str,
    raw_value: str,
) -> tuple[object | None, SkillDiagnosticInfo | None]:
    if not raw_value:
        return None, _malformed_manifest_diagnostic(
            path=f"manifest.{key}",
            reason="empty_string",
        )
    if _looks_like_container_scalar(raw_value):
        return None, _malformed_manifest_diagnostic(
            path=f"manifest.{key}",
            reason="bad_string",
        )
    if raw_value[0] in {"'", '"'}:
        try:
            parsed = literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return None, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                reason="bad_string",
            )
        if not isinstance(parsed, str) or not parsed.strip():
            return None, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                reason="bad_string",
            )
        return parsed, None
    return raw_value, None


def _looks_like_container_scalar(raw_value: str) -> bool:
    return raw_value[0] in {"[", "{"}


def _declared_resources(
    *,
    raw_resources: tuple[str, ...],
    source_label: str,
    skill_id: str,
    manifest_resource_id: str,
    manifest_size_bytes: int | None,
    package_resource_id: str,
    authorized_resources: tuple[SkillAuthorizedResource, ...],
    index_limits: SkillIndexLimits,
) -> tuple[
    tuple[SkillDeclaredResource, ...],
    tuple[SkillDiagnosticInfo, ...],
]:
    diagnostics: list[SkillDiagnosticInfo] = []
    resources: list[SkillDeclaredResource] = [
        SkillDeclaredResource(
            source_label=source_label,
            skill_id=skill_id,
            resource_id=SKILL_MAIN_RESOURCE_ID,
            source_resource_id=manifest_resource_id,
            size_bytes=manifest_size_bytes,
        )
    ]
    if len(raw_resources) + 1 > index_limits.max_resources_per_skill:
        return tuple(resources), (
            _resource_oversized_diagnostic(
                path="manifest.resources",
                reason="max_resources_per_skill",
            ),
        )

    normalized_resource_ids = tuple(
        resource_id
        for resource_id in (
            normalize_skill_resource_id(raw_resource)
            for raw_resource in raw_resources
        )
        if resource_id is not None
    )
    if _has_duplicates(normalized_resource_ids):
        return tuple(resources), (
            diagnostic_from_failure(
                SkillFailureMode.DUPLICATE_SKILL_IDS,
                path="manifest.resources",
                candidates=(skill_id,),
                details={"reason": "duplicate_resource_id"},
            ),
        )

    authorized_by_id = {
        resource.resource_id: resource
        for resource in authorized_resources
        if resource.source_label == source_label
    }
    for raw_resource in raw_resources:
        resource_id = normalize_skill_resource_id(raw_resource)
        if resource_id is None:
            reason = skill_resource_denial_reason(raw_resource) or "bad_id"
            diagnostics.append(_resource_declaration_diagnostic(reason=reason))
            continue
        source_resource_id = _source_resource_id(
            package_resource_id,
            resource_id,
        )
        authorized = authorized_by_id.get(source_resource_id)
        if authorized is None:
            diagnostics.append(
                diagnostic_from_failure(
                    SkillFailureMode.RESOURCE_MISSING,
                    path="manifest.resources",
                    details={"resource_id": resource_id},
                )
            )
            continue
        resources.append(
            SkillDeclaredResource(
                source_label=source_label,
                skill_id=skill_id,
                resource_id=resource_id,
                source_resource_id=source_resource_id,
                size_bytes=(
                    authorized.size_bytes if authorized is not None else None
                ),
            )
        )
    return tuple(resources), tuple(diagnostics)


def _has_duplicates(values: tuple[str, ...]) -> bool:
    return len(set(values)) != len(values)


def _resource_declaration_diagnostic(*, reason: str) -> SkillDiagnosticInfo:
    if reason in {
        "absolute_handle",
        "backslash",
        "environment_expansion",
        "hidden_path",
        "home_expansion",
        "nul_byte",
        "sensitive_path",
        "traversal",
        "url_handle",
    }:
        return diagnostic_from_failure(
            SkillFailureMode.RESOURCE_OUTSIDE_AUTHORIZED_ROOT,
            path="manifest.resources",
            details={"reason": reason},
        )
    return _malformed_manifest_diagnostic(
        path="manifest.resources",
        reason=reason,
    )


def _name_diagnostic(raw_name: str) -> SkillDiagnosticInfo:
    reason = skill_name_denial_reason(raw_name) or "bad_name"
    if reason == "path_like_name":
        return diagnostic_from_failure(
            SkillFailureMode.AMBIGUOUS_SKILL_NAME,
            path="manifest.name",
            details={"reason": reason},
        )
    return _malformed_manifest_diagnostic(
        path="manifest.name",
        reason=reason,
    )


def _malformed_manifest_diagnostic(
    *,
    path: str,
    reason: str,
) -> SkillDiagnosticInfo:
    return diagnostic_from_failure(
        SkillFailureMode.MALFORMED_MANIFEST,
        path=path,
        details={"reason": reason},
    )


def _resource_oversized_diagnostic(
    *,
    path: str,
    reason: str,
) -> SkillDiagnosticInfo:
    return diagnostic_from_failure(
        SkillFailureMode.RESOURCE_REQUIRES_CURSOR,
        path=path,
        details={"reason": reason},
    )


def _binary_resource_diagnostic(*, reason: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.BINARY_RESOURCE,
        status=SkillStatus.UNAVAILABLE,
        message="The skill resource is binary or non-UTF-8.",
        path="resource.main",
        hint="Expose only UTF-8 Markdown skill resources.",
        details={"reason": reason},
    )


def _resource_diagnostic_document(
    *,
    source_label: str,
    manifest_resource_id: str,
    diagnostic: SkillDiagnosticInfo,
) -> SkillManifestDocument:
    safe_manifest_resource_id = _safe_manifest_resource_id(
        manifest_resource_id
    )
    return SkillManifestDocument(
        source_label=source_label,
        manifest_resource_id=safe_manifest_resource_id,
        package_resource_id=_package_resource_id(safe_manifest_resource_id),
        diagnostics=(diagnostic,),
    )


def _manifest_diagnostic_documents(
    sources: tuple[SkillAuthorizedSourceRoot, ...],
) -> tuple[SkillManifestDocument, ...]:
    documents: list[SkillManifestDocument] = []
    for source in sources:
        for diagnostic in source.diagnostics:
            manifest_resource_id = _diagnostic_manifest_resource_id(diagnostic)
            if manifest_resource_id is not None:
                documents.append(
                    _resource_diagnostic_document(
                        source_label=source.label,
                        manifest_resource_id=manifest_resource_id,
                        diagnostic=diagnostic,
                    )
                )
    return tuple(documents)


def _diagnostic_manifest_resource_id(
    diagnostic: SkillDiagnosticInfo,
) -> str | None:
    resource_id = diagnostic.details.get("resource_id")
    if isinstance(resource_id, str) and _is_manifest_resource_id(resource_id):
        return resource_id
    return None


def _with_duplicate_diagnostic(
    manifest: SkillManifestDocument,
    diagnostic: SkillDiagnosticInfo,
) -> SkillManifestDocument:
    metadata = manifest.metadata
    if metadata is not None and metadata.enabled:
        metadata = replace(metadata, status=SkillStatus.BLOCKED)
    return replace(
        manifest,
        metadata=metadata,
        diagnostics=(diagnostic, *manifest.diagnostics),
    )


def _normalize_optional_version(value: object) -> str | None:
    if value is None:
        return None
    assert isinstance(value, str)
    return normalize_skill_description(value)


def _is_model_safe_metadata_text(value: str) -> bool:
    try:
        model_dict({"value": value})
    except AssertionError:
        return False
    return True


def _is_manifest_resource_id(resource_id: str) -> bool:
    name = PurePosixPath(resource_id).name
    if name == SKILL_MAIN_RESOURCE_FILENAME:
        return True
    matched = fullmatch(_SKILL_DASH_MANIFEST_BASENAME_PATTERN, name)
    if matched is None:
        return False
    return skill_model_handle_denial_reason(matched.group(1)) is None


def _field_diagnostic_path(key: str) -> str:
    if key and key.replace("_", "").replace("-", "").isalnum():
        return f"manifest.{key.lower().replace('_', '-')}"
    return "manifest.field"


def _safe_manifest_resource_id(value: str) -> str:
    normalized = normalize_skill_resource_id(value)
    if normalized is not None:
        return normalized
    return SKILL_MAIN_RESOURCE_FILENAME


def _package_resource_id(manifest_resource_id: str) -> str:
    parent = PurePosixPath(manifest_resource_id).parent.as_posix()
    return "." if parent == "." else parent


def _source_resource_id(
    package_resource_id: str,
    resource_id: str,
) -> str:
    if package_resource_id == ".":
        return resource_id
    return f"{package_resource_id}/{resource_id}"


_SUPPORTED_FIELD_NAMES = frozenset(field.value for field in SkillManifestField)
