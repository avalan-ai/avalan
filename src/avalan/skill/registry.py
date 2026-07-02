from ..event import EventType
from .contract import SkillDiagnosticCode, SkillFailureMode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillMetadata,
    SkillModelValue,
    SkillRegistryVersion,
    SkillResourceHandle,
    SkillSourceAuthority,
    diagnostic_from_failure,
    model_dict,
)
from .manifest import (
    SkillDeclaredResource,
    SkillManifestDocument,
    parse_skill_manifests,
)
from .observability import (
    SkillAuditDeliveryError,
    SkillEventPublisher,
    assert_skill_event_publisher,
    emit_skill_audit_event,
    skill_audit_authority_value,
    skill_audit_correlation_id,
    skill_audit_diagnostic_fields,
    skill_audit_hash_prefix,
    skill_audit_registry_fields,
)
from .resolver import (
    SkillAsyncFileSystem,
    SkillAuthorizedResource,
    SkillAuthorizedSourceRoot,
    SkillSourceFileSystem,
    SkillSourceResolutionResult,
)
from .settings import (
    SkillIndexLimits,
    SkillReadLimits,
    TrustedSkillSettings,
    skill_source_identity_dict,
)

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps
from pathlib import Path
from re import fullmatch
from types import MappingProxyType
from typing import TypeAlias

_REGISTRY_SCHEMA = "skills.registry.v1"
_RESOURCE_KEY_SEPARATOR = "\x1f"

_SkillRegistryResourceKey: TypeAlias = tuple[str, str, str]


def _empty_source_identity() -> Mapping[str, SkillModelValue]:
    return MappingProxyType({})


def _empty_skill_mapping() -> Mapping[str, "SkillRegistrySkill"]:
    return MappingProxyType({})


def _empty_resource_mapping() -> (
    Mapping[_SkillRegistryResourceKey, "SkillRegisteredResource"]
):
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResourceFingerprint:
    source_label: str
    resource_id: str
    size_bytes: int
    line_count: int
    content_sha256: str | None = None
    status: SkillStatus = SkillStatus.OK

    def __post_init__(self) -> None:
        _assert_source_label(self.source_label)
        _assert_resource_id(self.resource_id)
        _assert_non_negative_int(self.size_bytes, "size_bytes")
        _assert_non_negative_int(self.line_count, "line_count")
        if self.content_sha256 is not None:
            assert (
                fullmatch(r"[a-f0-9]{64}", self.content_sha256) is not None
            ), "content_sha256 must be a SHA-256 hex digest"
        assert isinstance(self.status, SkillStatus)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "source_label": self.source_label,
                "resource_id": self.resource_id,
                "size_bytes": self.size_bytes,
                "line_count": self.line_count,
                "status": self.status.value,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRegisteredResource:
    handle: SkillResourceHandle
    source_resource_id: str
    path: Path
    fingerprint: SkillResourceFingerprint
    declaration: SkillDeclaredResource
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.handle, SkillResourceHandle)
        _assert_resource_id(self.source_resource_id)
        assert isinstance(self.path, Path)
        assert isinstance(self.fingerprint, SkillResourceFingerprint)
        assert isinstance(self.declaration, SkillDeclaredResource)
        _assert_diagnostic_tuple(self.diagnostics)
        assert self.handle.source_label == self.declaration.source_label
        assert self.handle.skill_id == self.declaration.skill_id
        assert self.handle.resource_id == self.declaration.resource_id
        assert self.source_resource_id == self.declaration.source_resource_id
        assert self.fingerprint.source_label == self.handle.source_label
        assert self.fingerprint.resource_id == self.source_resource_id

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "handle": self.handle.as_model_dict(),
                "source_resource_id": self.source_resource_id,
                "fingerprint": self.fingerprint.as_model_dict(),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRegistrySource:
    label: str
    authority: SkillSourceAuthority
    status: SkillStatus = SkillStatus.OK
    resource_count: int = 0
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()
    source_identity: Mapping[str, SkillModelValue] = field(
        default_factory=_empty_source_identity
    )

    def __post_init__(self) -> None:
        _assert_source_label(self.label)
        assert isinstance(self.authority, SkillSourceAuthority)
        assert isinstance(self.status, SkillStatus)
        _assert_non_negative_int(self.resource_count, "resource_count")
        _assert_diagnostic_tuple(self.diagnostics)
        assert isinstance(self.source_identity, Mapping)
        object.__setattr__(
            self,
            "source_identity",
            MappingProxyType(model_dict(self.source_identity)),
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "label": self.label,
                "authority": self.authority.as_model_dict(),
                "status": self.status.value,
                "resource_count": self.resource_count,
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRegistrySkill:
    source_label: str
    manifest_resource_id: str
    package_resource_id: str
    status: SkillStatus
    readable: bool
    usable: bool
    skill_id: str | None = None
    metadata: SkillMetadata | None = None
    resources: tuple[SkillRegisteredResource, ...] = ()
    manifest_fingerprint: SkillResourceFingerprint | None = None
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        _assert_source_label(self.source_label)
        _assert_resource_id(self.manifest_resource_id)
        assert self.package_resource_id == "." or self.package_resource_id
        if self.skill_id is not None:
            _assert_logical_id(self.skill_id, "skill_id")
        if self.metadata is not None:
            assert isinstance(self.metadata, SkillMetadata)
            assert self.metadata.skill_id == self.skill_id
            assert self.metadata.source_label == self.source_label
        _assert_registered_resource_tuple(self.resources)
        if self.manifest_fingerprint is not None:
            assert isinstance(
                self.manifest_fingerprint, SkillResourceFingerprint
            )
        _assert_diagnostic_tuple(self.diagnostics)
        assert isinstance(self.status, SkillStatus)
        assert isinstance(self.readable, bool)
        assert isinstance(self.usable, bool)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "source_label": self.source_label,
            "manifest_resource_id": self.manifest_resource_id,
            "package_resource_id": self.package_resource_id,
            "status": self.status.value,
            "readable": self.readable,
            "usable": self.usable,
            "resources": tuple(
                resource.as_model_dict() for resource in self.resources
            ),
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
        }
        if self.skill_id is not None:
            value["skill_id"] = self.skill_id
        if self.metadata is not None:
            value["metadata"] = self.metadata.as_model_dict()
        if self.manifest_fingerprint is not None:
            value["manifest_fingerprint"] = (
                self.manifest_fingerprint.as_model_dict()
            )
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRegistryResourceCheck:
    handle: SkillResourceHandle
    stored_fingerprint: SkillResourceFingerprint | None = None
    current_fingerprint: SkillResourceFingerprint | None = None
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.handle, SkillResourceHandle)
        if self.stored_fingerprint is not None:
            assert isinstance(
                self.stored_fingerprint, SkillResourceFingerprint
            )
        if self.current_fingerprint is not None:
            assert isinstance(
                self.current_fingerprint, SkillResourceFingerprint
            )
        _assert_diagnostic_tuple(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        return self.handle.status

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "status": self.status.value,
            "handle": self.handle.as_model_dict(),
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
        }
        if self.stored_fingerprint is not None:
            value["stored_fingerprint"] = (
                self.stored_fingerprint.as_model_dict()
            )
        if self.current_fingerprint is not None:
            value["current_fingerprint"] = (
                self.current_fingerprint.as_model_dict()
            )
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRegistry:
    registry_version: SkillRegistryVersion
    read_limits: SkillReadLimits
    index_limits: SkillIndexLimits
    sources: tuple[SkillRegistrySource, ...] = ()
    skills: tuple[SkillRegistrySkill, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()
    settings: TrustedSkillSettings | None = None
    skills_by_id: Mapping[str, SkillRegistrySkill] = field(
        default_factory=_empty_skill_mapping
    )
    resources_by_key: Mapping[
        _SkillRegistryResourceKey, SkillRegisteredResource
    ] = field(default_factory=_empty_resource_mapping)

    def __post_init__(self) -> None:
        assert isinstance(self.registry_version, SkillRegistryVersion)
        assert isinstance(self.read_limits, SkillReadLimits)
        assert isinstance(self.index_limits, SkillIndexLimits)
        _assert_registry_source_tuple(self.sources)
        _assert_registry_skill_tuple(self.skills)
        _assert_diagnostic_tuple(self.diagnostics)
        if self.settings is not None:
            assert isinstance(self.settings, TrustedSkillSettings)

        skills_by_id = _skills_by_id(self.skills)
        resources_by_key = _resources_by_key(self.skills)
        object.__setattr__(
            self, "skills_by_id", MappingProxyType(skills_by_id)
        )
        object.__setattr__(
            self,
            "resources_by_key",
            MappingProxyType(resources_by_key),
        )

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        if any(skill.usable for skill in self.skills):
            return SkillStatus.OK
        if self.skills:
            return self.skills[0].status
        return SkillStatus.EMPTY

    @property
    def metadata(self) -> tuple[SkillMetadata, ...]:
        return tuple(
            skill.metadata
            for skill in self.skills
            if skill.metadata is not None
        )

    @property
    def usable_metadata(self) -> tuple[SkillMetadata, ...]:
        return tuple(
            skill.metadata
            for skill in self.skills
            if skill.metadata is not None and skill.usable
        )

    @property
    def source_diagnostics(self) -> tuple[SkillDiagnosticInfo, ...]:
        return tuple(
            diagnostic
            for source in self.sources
            for diagnostic in source.diagnostics
        )

    @property
    def resource_handles(self) -> tuple[SkillResourceHandle, ...]:
        return tuple(
            resource.handle
            for skill in self.skills
            for resource in skill.resources
        )

    def resource_for_handle(
        self,
        handle: SkillResourceHandle,
    ) -> SkillRegisteredResource | None:
        assert isinstance(handle, SkillResourceHandle)
        return self.resources_by_key.get(_resource_key(handle))

    async def check_resource(
        self,
        handle: SkillResourceHandle,
        *,
        file_system: SkillSourceFileSystem | None = None,
        read_limits: SkillReadLimits | None = None,
    ) -> SkillRegistryResourceCheck:
        return await check_skill_registry_resource(
            self,
            handle,
            file_system=file_system,
            read_limits=read_limits,
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "status": self.status.value,
            "registry_version": self.registry_version.as_model_value(),
            "read_limits": self.read_limits.as_model_dict(),
            "index_limits": self.index_limits.as_model_dict(),
            "sources": tuple(
                source.as_model_dict() for source in self.sources
            ),
            "skills": tuple(skill.as_model_dict() for skill in self.skills),
            "metadata": tuple(
                metadata.as_model_dict() for metadata in self.metadata
            ),
            "resource_handles": tuple(
                handle.as_model_dict() for handle in self.resource_handles
            ),
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
        }
        if self.settings is not None:
            value["settings"] = self.settings.as_model_dict()
        return model_dict(value)


async def build_skill_registry(
    resolution: (
        SkillSourceResolutionResult | tuple[SkillAuthorizedSourceRoot, ...]
    ),
    *,
    settings: TrustedSkillSettings | None = None,
    read_limits: SkillReadLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
    file_system: SkillSourceFileSystem | None = None,
    event_manager: SkillEventPublisher | None = None,
    audit_operation_id: str | None = None,
) -> SkillRegistry:
    assert isinstance(resolution, SkillSourceResolutionResult | tuple)
    if settings is not None:
        assert isinstance(settings, TrustedSkillSettings)
    assert_skill_event_publisher(event_manager)
    assert audit_operation_id is None or isinstance(audit_operation_id, str)
    if event_manager is not None and audit_operation_id is None:
        audit_operation_id = skill_audit_correlation_id("skill-registry-build")
    if read_limits is None:
        read_limits = (
            settings.read_limits
            if settings is not None
            else (SkillReadLimits())
        )
    if index_limits is None:
        index_limits = (
            settings.index_limits
            if settings is not None
            else (SkillIndexLimits())
        )
    if file_system is None:
        file_system = SkillAsyncFileSystem()
    assert isinstance(read_limits, SkillReadLimits)
    assert isinstance(index_limits, SkillIndexLimits)

    await emit_skill_audit_event(
        event_manager,
        settings,
        EventType.SKILL_REGISTRY_BUILD_STARTED,
        {
            "operation_id": audit_operation_id,
            "status": "started",
        },
    )
    try:
        sources, resolution_diagnostics = _resolution_parts(resolution)
        manifests = await parse_skill_manifests(
            sources,
            file_system=file_system,
            read_limits=read_limits,
            index_limits=index_limits,
        )
        source_by_label = {source.label: source for source in sources}
        source_resources = _source_resources_by_key(sources)
        registry_sources = tuple(
            _registry_source(source) for source in sources
        )
        skills, fingerprint_diagnostics = await _registry_skills(
            manifests.manifests,
            source_by_label=source_by_label,
            source_resources=source_resources,
            file_system=file_system,
            read_limits=read_limits,
        )
        diagnostics = (
            *resolution_diagnostics,
            *manifests.diagnostics,
            *fingerprint_diagnostics,
        )
        if not skills and not diagnostics:
            diagnostics = (
                diagnostic_from_failure(
                    SkillFailureMode.EMPTY_REGISTRY,
                    path="skills",
                ),
            )
        registry_version = _registry_version(
            sources=registry_sources,
            skills=skills,
            diagnostics=diagnostics,
            settings=settings,
            read_limits=read_limits,
            index_limits=index_limits,
        )
        registry = SkillRegistry(
            registry_version=registry_version,
            read_limits=read_limits,
            index_limits=index_limits,
            sources=registry_sources,
            skills=skills,
            diagnostics=diagnostics,
            settings=settings,
        )
        await _emit_registry_skill_events(
            event_manager,
            settings,
            registry,
            operation_id=audit_operation_id,
        )
        await emit_skill_audit_event(
            event_manager,
            settings,
            EventType.SKILL_REGISTRY_BUILD_COMPLETED,
            {
                "operation_id": audit_operation_id,
                **skill_audit_registry_fields(
                    registry.registry_version,
                    status=registry.status,
                ),
                "source_count": len(registry.sources),
                "skill_count": len(registry.skills),
                "diagnostic_count": len(registry.diagnostics),
            },
        )
        return registry
    except SkillAuditDeliveryError:
        raise
    except Exception:
        await emit_skill_audit_event(
            event_manager,
            settings,
            EventType.SKILL_REGISTRY_BUILD_FAILED,
            {
                "operation_id": audit_operation_id,
                "status": SkillStatus.BLOCKED.value,
            },
        )
        raise


async def check_skill_registry_resource(
    registry: SkillRegistry,
    handle: SkillResourceHandle,
    *,
    file_system: SkillSourceFileSystem | None = None,
    read_limits: SkillReadLimits | None = None,
) -> SkillRegistryResourceCheck:
    assert isinstance(registry, SkillRegistry)
    assert isinstance(handle, SkillResourceHandle)
    if file_system is None:
        file_system = SkillAsyncFileSystem()
    if read_limits is None:
        read_limits = registry.read_limits
    assert isinstance(read_limits, SkillReadLimits)

    registered = registry.resource_for_handle(handle)
    if registered is None:
        return SkillRegistryResourceCheck(
            handle=handle,
            diagnostics=(
                diagnostic_from_failure(
                    SkillFailureMode.UNKNOWN_SKILL_ID,
                    path="resource.lookup",
                    details={"resource_id": handle.resource_id},
                ),
            ),
        )

    canonical_handle = registered.handle
    owner = _owner_for_registered_resource(registry, registered)
    stored_diagnostic = _stored_resource_diagnostic(owner, registered)
    if stored_diagnostic is not None:
        return SkillRegistryResourceCheck(
            handle=_diagnostic_handle(
                canonical_handle,
                stored_diagnostic.status,
            ),
            stored_fingerprint=registered.fingerprint,
            diagnostics=(stored_diagnostic,),
        )

    current, diagnostic = await _runtime_fingerprint(
        registered,
        file_system=file_system,
        read_limits=read_limits,
    )
    if diagnostic is not None:
        return SkillRegistryResourceCheck(
            handle=_diagnostic_handle(canonical_handle, diagnostic.status),
            stored_fingerprint=registered.fingerprint,
            current_fingerprint=current,
            diagnostics=(diagnostic,),
        )
    assert current is not None
    if current != registered.fingerprint:
        return SkillRegistryResourceCheck(
            handle=_diagnostic_handle(canonical_handle, SkillStatus.STALE),
            stored_fingerprint=registered.fingerprint,
            current_fingerprint=current,
            diagnostics=(
                _stale_resource_diagnostic(
                    reason="content_changed",
                    resource_id=canonical_handle.resource_id,
                ),
            ),
        )
    return SkillRegistryResourceCheck(
        handle=canonical_handle,
        stored_fingerprint=registered.fingerprint,
        current_fingerprint=current,
    )


def _resolution_parts(
    resolution: (
        SkillSourceResolutionResult | tuple[SkillAuthorizedSourceRoot, ...]
    ),
) -> tuple[
    tuple[SkillAuthorizedSourceRoot, ...],
    tuple[SkillDiagnosticInfo, ...],
]:
    if isinstance(resolution, SkillSourceResolutionResult):
        return resolution.sources, resolution.diagnostics
    assert isinstance(resolution, tuple)
    for source in resolution:
        assert isinstance(source, SkillAuthorizedSourceRoot)
    return resolution, ()


async def _emit_registry_skill_events(
    event_manager: SkillEventPublisher | None,
    settings: TrustedSkillSettings | None,
    registry: SkillRegistry,
    *,
    operation_id: str | None,
) -> None:
    counts = Counter(
        skill.skill_id for skill in registry.skills if skill.skill_id
    )
    duplicate_ids = {
        skill_id for skill_id, count in counts.items() if count > 1
    }
    seen_duplicates: set[str] = set()
    source_by_label = {source.label: source for source in registry.sources}
    for skill in registry.skills:
        source_authority = _registry_source_authority_value(
            source_by_label,
            skill.source_label,
        )
        if (
            skill.skill_id is not None
            and skill.skill_id in duplicate_ids
            and skill.skill_id in seen_duplicates
        ):
            await _emit_registry_skill_event(
                event_manager,
                settings,
                EventType.SKILL_SHADOWED,
                registry,
                skill,
                operation_id=operation_id,
                source_authority=source_authority,
            )
        event_type = _registry_skill_event_type(skill, duplicate_ids)
        if event_type is None:
            continue
        await _emit_registry_skill_event(
            event_manager,
            settings,
            event_type,
            registry,
            skill,
            operation_id=operation_id,
            source_authority=source_authority,
        )
        if skill.skill_id is not None and skill.skill_id in duplicate_ids:
            seen_duplicates.add(skill.skill_id)


def _registry_skill_event_type(
    skill: SkillRegistrySkill,
    duplicate_ids: set[str],
) -> EventType | None:
    if skill.skill_id is not None and skill.skill_id in duplicate_ids:
        return EventType.SKILL_DUPLICATE
    if skill.status is SkillStatus.DISABLED:
        return EventType.SKILL_DISABLED
    if skill.status is SkillStatus.MALFORMED or skill.metadata is None:
        return EventType.SKILL_MALFORMED
    if skill.usable and skill.status is SkillStatus.OK:
        return EventType.SKILL_REGISTERED
    return None


async def _emit_registry_skill_event(
    event_manager: SkillEventPublisher | None,
    settings: TrustedSkillSettings | None,
    event_type: EventType,
    registry: SkillRegistry,
    skill: SkillRegistrySkill,
    *,
    operation_id: str | None,
    source_authority: str | None,
) -> None:
    diagnostic = skill.diagnostics[0] if skill.diagnostics else None
    fields: dict[str, object] = {
        "operation_id": operation_id,
        **skill_audit_registry_fields(
            registry.registry_version,
            status=skill.status,
        ),
        "source_label": skill.source_label,
        "source_authority": source_authority,
        "skill_id": skill.skill_id,
        "resource_count": len(skill.resources),
        "hash_prefix": skill_audit_hash_prefix(
            skill.manifest_fingerprint.content_sha256
            if skill.manifest_fingerprint is not None
            else None
        ),
    }
    fields.update(skill_audit_diagnostic_fields(diagnostic))
    await emit_skill_audit_event(event_manager, settings, event_type, fields)


def _registry_source_authority_value(
    source_by_label: Mapping[str, SkillRegistrySource],
    source_label: str,
) -> str | None:
    source = source_by_label.get(source_label)
    if source is None:
        return None
    return skill_audit_authority_value(source.authority)


def _registry_source(source: SkillAuthorizedSourceRoot) -> SkillRegistrySource:
    identity_root = (
        source.identity_root
        if source.identity_root is not None
        else source.root
    )
    return SkillRegistrySource(
        label=source.label,
        authority=source.authority,
        status=_status_for_source(source),
        resource_count=len(source.resources),
        diagnostics=source.diagnostics,
        source_identity=skill_source_identity_dict(
            label=source.label,
            authority=source.authority,
            root_path=identity_root,
            allow_hidden_paths=source.allow_hidden_paths,
            status=_status_for_source(source),
        ),
    )


def _status_for_source(source: SkillAuthorizedSourceRoot) -> SkillStatus:
    if source.diagnostics:
        return source.diagnostics[0].status
    return source.status


async def _registry_skills(
    manifests: tuple[SkillManifestDocument, ...],
    *,
    source_by_label: Mapping[str, SkillAuthorizedSourceRoot],
    source_resources: Mapping[tuple[str, str], SkillAuthorizedResource],
    file_system: SkillSourceFileSystem,
    read_limits: SkillReadLimits,
) -> tuple[
    tuple[SkillRegistrySkill, ...],
    tuple[SkillDiagnosticInfo, ...],
]:
    skills: list[SkillRegistrySkill] = []
    diagnostics: list[SkillDiagnosticInfo] = []
    for manifest in manifests:
        source = source_by_label.get(manifest.source_label)
        resources: list[SkillRegisteredResource] = []
        manifest_fingerprint = await _manifest_fingerprint(
            manifest,
            source_resources=source_resources,
            file_system=file_system,
            read_limits=read_limits,
        )
        skill_diagnostics: list[SkillDiagnosticInfo] = list(
            manifest.diagnostics
        )
        for declaration in manifest.declared_resources:
            source_resource = source_resources.get(
                (
                    declaration.source_label,
                    declaration.source_resource_id,
                )
            )
            assert source_resource is not None
            fingerprint, diagnostic = await _build_fingerprint(
                source_resource,
                file_system=file_system,
                read_limits=read_limits,
            )
            if diagnostic is not None:
                skill_diagnostics.append(diagnostic)
                diagnostics.append(diagnostic)
            resource_diagnostics = (diagnostic,) if diagnostic else ()
            resources.append(
                SkillRegisteredResource(
                    handle=declaration.as_handle(),
                    source_resource_id=declaration.source_resource_id,
                    path=source_resource.path,
                    fingerprint=fingerprint,
                    declaration=declaration,
                    diagnostics=resource_diagnostics,
                )
            )
        skills.append(
            SkillRegistrySkill(
                source_label=manifest.source_label,
                manifest_resource_id=manifest.manifest_resource_id,
                package_resource_id=manifest.package_resource_id,
                skill_id=manifest.skill_id,
                metadata=manifest.metadata,
                status=manifest.status,
                readable=manifest.readable,
                usable=manifest.usable and not skill_diagnostics,
                resources=tuple(resources),
                manifest_fingerprint=manifest_fingerprint,
                diagnostics=tuple(skill_diagnostics),
            )
        )
        assert source is None or source.label == manifest.source_label
    return tuple(skills), tuple(diagnostics)


async def _manifest_fingerprint(
    manifest: SkillManifestDocument,
    *,
    source_resources: Mapping[tuple[str, str], SkillAuthorizedResource],
    file_system: SkillSourceFileSystem,
    read_limits: SkillReadLimits,
) -> SkillResourceFingerprint | None:
    source_resource = source_resources.get(
        (
            manifest.source_label,
            manifest.manifest_resource_id,
        )
    )
    if source_resource is None:
        return None
    fingerprint, _ = await _build_fingerprint(
        source_resource,
        file_system=file_system,
        read_limits=read_limits,
    )
    return fingerprint


async def _build_fingerprint(
    resource: SkillAuthorizedResource,
    *,
    file_system: SkillSourceFileSystem,
    read_limits: SkillReadLimits,
) -> tuple[SkillResourceFingerprint, SkillDiagnosticInfo | None]:
    try:
        content = await file_system.read_bytes(
            resource.path,
            read_limits.max_bytes_per_read + 1,
        )
    except OSError:
        fingerprint = SkillResourceFingerprint(
            source_label=resource.source_label,
            resource_id=resource.resource_id,
            size_bytes=resource.size_bytes,
            line_count=resource.line_count,
            status=SkillStatus.UNAVAILABLE,
        )
        return fingerprint, diagnostic_from_failure(
            SkillFailureMode.SOURCE_UNAVAILABLE,
            path="source.availability",
            details={"resource_id": resource.resource_id},
        )

    current = _fingerprint_from_content(
        source_label=resource.source_label,
        resource_id=resource.resource_id,
        content=content[: read_limits.max_bytes_per_read],
        status=SkillStatus.OK,
    )
    if (
        len(content) > read_limits.max_bytes_per_read
        or current.size_bytes != resource.size_bytes
        or current.line_count != resource.line_count
    ):
        stale = SkillResourceFingerprint(
            source_label=current.source_label,
            resource_id=current.resource_id,
            size_bytes=current.size_bytes,
            line_count=current.line_count,
            content_sha256=current.content_sha256,
            status=SkillStatus.STALE,
        )
        return stale, _stale_resource_diagnostic(
            reason="resolved_metadata_changed",
            resource_id=resource.resource_id,
        )
    return current, None


async def _runtime_fingerprint(
    registered: SkillRegisteredResource,
    *,
    file_system: SkillSourceFileSystem,
    read_limits: SkillReadLimits,
) -> tuple[SkillResourceFingerprint | None, SkillDiagnosticInfo | None]:
    try:
        stat = await file_system.stat_path(registered.path)
    except FileNotFoundError:
        return None, diagnostic_from_failure(
            SkillFailureMode.RESOURCE_MISSING,
            path="resource.lookup",
            details={"resource_id": registered.handle.resource_id},
        )
    except OSError:
        return None, _runtime_unavailable_diagnostic(
            resource_id=registered.handle.resource_id
        )

    try:
        content = await file_system.read_bytes(
            registered.path,
            read_limits.max_bytes_per_read + 1,
        )
    except FileNotFoundError:
        return None, diagnostic_from_failure(
            SkillFailureMode.RESOURCE_MISSING,
            path="resource.lookup",
            details={"resource_id": registered.handle.resource_id},
        )
    except OSError:
        return None, _runtime_unavailable_diagnostic(
            resource_id=registered.handle.resource_id
        )

    current = _fingerprint_from_content(
        source_label=registered.fingerprint.source_label,
        resource_id=registered.fingerprint.resource_id,
        content=content[: read_limits.max_bytes_per_read],
        status=SkillStatus.OK,
    )
    if len(content) > read_limits.max_bytes_per_read:
        return current, _stale_resource_diagnostic(
            reason="read_limit_exceeded",
            resource_id=registered.handle.resource_id,
        )
    if stat.st_size != current.size_bytes:
        return current, _stale_resource_diagnostic(
            reason="stat_content_mismatch",
            resource_id=registered.handle.resource_id,
        )
    return current, None


def _fingerprint_from_content(
    *,
    source_label: str,
    resource_id: str,
    content: bytes,
    status: SkillStatus,
) -> SkillResourceFingerprint:
    return SkillResourceFingerprint(
        source_label=source_label,
        resource_id=resource_id,
        size_bytes=len(content),
        line_count=_line_count(content),
        content_sha256=sha256(content).hexdigest(),
        status=status,
    )


def _skill_version_dict(
    skill: SkillRegistrySkill,
) -> dict[str, SkillModelValue]:
    value: dict[str, object] = {
        "source_label": skill.source_label,
        "manifest_resource_id": skill.manifest_resource_id,
        "package_resource_id": skill.package_resource_id,
        "status": skill.status.value,
        "readable": skill.readable,
        "usable": skill.usable,
        "resources": tuple(
            _registered_resource_version_dict(resource)
            for resource in skill.resources
        ),
        "diagnostics": tuple(
            diagnostic.as_model_dict() for diagnostic in skill.diagnostics
        ),
    }
    if skill.skill_id is not None:
        value["skill_id"] = skill.skill_id
    if skill.metadata is not None:
        value["metadata"] = skill.metadata.as_model_dict()
    if skill.manifest_fingerprint is not None:
        value["manifest_fingerprint"] = _fingerprint_version_dict(
            skill.manifest_fingerprint
        )
    return model_dict(value)


def _registered_resource_version_dict(
    resource: SkillRegisteredResource,
) -> dict[str, SkillModelValue]:
    return model_dict(
        {
            "handle": resource.handle.as_model_dict(),
            "source_resource_id": resource.source_resource_id,
            "fingerprint": _fingerprint_version_dict(resource.fingerprint),
            "diagnostics": tuple(
                diagnostic.as_model_dict()
                for diagnostic in resource.diagnostics
            ),
        }
    )


def _fingerprint_version_dict(
    fingerprint: SkillResourceFingerprint,
) -> dict[str, SkillModelValue]:
    value: dict[str, object] = {
        "source_label": fingerprint.source_label,
        "resource_id": fingerprint.resource_id,
        "size_bytes": fingerprint.size_bytes,
        "line_count": fingerprint.line_count,
        "status": fingerprint.status.value,
    }
    if fingerprint.content_sha256 is not None:
        value["content_sha256"] = fingerprint.content_sha256
    return model_dict(value)


def _registry_version(
    *,
    sources: tuple[SkillRegistrySource, ...],
    skills: tuple[SkillRegistrySkill, ...],
    diagnostics: tuple[SkillDiagnosticInfo, ...],
    settings: TrustedSkillSettings | None,
    read_limits: SkillReadLimits,
    index_limits: SkillIndexLimits,
) -> SkillRegistryVersion:
    payload: dict[str, object] = {
        "schema": _REGISTRY_SCHEMA,
        "read_limits": read_limits.as_model_dict(),
        "index_limits": index_limits.as_model_dict(),
        "sources": tuple(source.as_model_dict() for source in sources),
        "source_identities": tuple(
            source.source_identity for source in sources
        ),
        "skills": tuple(_skill_version_dict(skill) for skill in skills),
        "diagnostics": tuple(
            diagnostic.as_model_dict() for diagnostic in diagnostics
        ),
    }
    if settings is not None:
        payload["settings"] = settings.as_model_dict()
    canonical = dumps(
        model_dict(payload),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return SkillRegistryVersion(value=f"skills-registry:{digest}")


def _source_resources_by_key(
    sources: tuple[SkillAuthorizedSourceRoot, ...],
) -> Mapping[tuple[str, str], SkillAuthorizedResource]:
    return MappingProxyType(
        {
            (resource.source_label, resource.resource_id): resource
            for source in sources
            for resource in source.resources
        }
    )


def _owner_for_registered_resource(
    registry: SkillRegistry,
    registered: SkillRegisteredResource,
) -> SkillRegistrySkill:
    return next(
        skill
        for skill in registry.skills
        if any(resource is registered for resource in skill.resources)
    )


def _stored_resource_diagnostic(
    owner: SkillRegistrySkill,
    registered: SkillRegisteredResource,
) -> SkillDiagnosticInfo | None:
    if registered.fingerprint.status is not SkillStatus.OK:
        assert registered.diagnostics
        return registered.diagnostics[0]
    if owner.usable:
        return None
    assert owner.diagnostics
    return owner.diagnostics[0]


def _skills_by_id(
    skills: tuple[SkillRegistrySkill, ...],
) -> dict[str, SkillRegistrySkill]:
    counts = Counter(
        skill.skill_id for skill in skills if skill.skill_id is not None
    )
    return {
        skill.skill_id: skill
        for skill in skills
        if skill.skill_id is not None and counts[skill.skill_id] == 1
    }


def _resources_by_key(
    skills: tuple[SkillRegistrySkill, ...],
) -> dict[_SkillRegistryResourceKey, SkillRegisteredResource]:
    resources: dict[_SkillRegistryResourceKey, SkillRegisteredResource] = {}
    for skill in skills:
        for resource in skill.resources:
            key = _resource_key(resource.handle)
            if key not in resources:
                resources[key] = resource
    return resources


def _resource_key(
    handle: SkillResourceHandle,
) -> _SkillRegistryResourceKey:
    return (handle.source_label, handle.skill_id, handle.resource_id)


def _diagnostic_handle(
    handle: SkillResourceHandle,
    status: SkillStatus,
) -> SkillResourceHandle:
    if status is SkillStatus.STALE:
        return SkillResourceHandle(
            source_label=handle.source_label,
            skill_id=handle.skill_id,
            resource_id=handle.resource_id,
            media_type=handle.media_type,
            size_bytes=handle.size_bytes,
            status=SkillStatus.STALE,
            stale=True,
        )
    return SkillResourceHandle(
        source_label=handle.source_label,
        skill_id=handle.skill_id,
        resource_id=handle.resource_id,
        media_type=handle.media_type,
        size_bytes=handle.size_bytes,
        status=status,
    )


def _stale_resource_diagnostic(
    *,
    reason: str,
    resource_id: str,
) -> SkillDiagnosticInfo:
    return diagnostic_from_failure(
        SkillFailureMode.RESOURCE_STALE,
        path="resource.fingerprint",
        details={"reason": reason, "resource_id": resource_id},
    )


def _runtime_unavailable_diagnostic(
    *, resource_id: str
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
        status=SkillStatus.UNAVAILABLE,
        message=(
            "The sandbox or container runtime cannot access the configured "
            "source."
        ),
        path="source.availability",
        hint="Keep the registry unavailable instead of widening access.",
        details={"resource_id": resource_id},
    )


def _line_count(content: bytes) -> int:
    if not content:
        return 0
    line_count = content.count(b"\n")
    if not content.endswith(b"\n"):
        line_count += 1
    return line_count


def _assert_registry_source_tuple(
    values: tuple[SkillRegistrySource, ...],
) -> None:
    assert isinstance(values, tuple), "sources must be a tuple"
    for value in values:
        assert isinstance(value, SkillRegistrySource)


def _assert_registry_skill_tuple(
    values: tuple[SkillRegistrySkill, ...],
) -> None:
    assert isinstance(values, tuple), "skills must be a tuple"
    for value in values:
        assert isinstance(value, SkillRegistrySkill)


def _assert_registered_resource_tuple(
    values: tuple[SkillRegisteredResource, ...],
) -> None:
    assert isinstance(values, tuple), "resources must be a tuple"
    for value in values:
        assert isinstance(value, SkillRegisteredResource)


def _assert_diagnostic_tuple(
    values: tuple[SkillDiagnosticInfo, ...],
) -> None:
    assert isinstance(values, tuple), "diagnostics must be a tuple"
    for value in values:
        assert isinstance(value, SkillDiagnosticInfo)


def _assert_source_label(value: str) -> None:
    _assert_logical_id(value, "source_label")


def _assert_logical_id(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert (
        fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", value) is not None
    ), f"{field_name} must be a logical ID"


def _assert_resource_id(value: str) -> None:
    assert isinstance(value, str), "resource_id must be a string"
    assert value.strip(), "resource_id must be non-empty"
    assert "\x00" not in value
    assert "\\" not in value
    assert not value.startswith(("/", "~", "$"))
    assert _RESOURCE_KEY_SEPARATOR not in value


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must be non-negative"
