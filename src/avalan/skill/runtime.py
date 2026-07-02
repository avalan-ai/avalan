from ._async import skill_cancellation_checkpoint
from .contract import SkillDiagnosticCode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillSourceAuthority,
    model_dict,
)
from .path_policy import sanitize_skill_source_label
from .registry import SkillRegistry, build_skill_registry
from .resolver import (
    SkillAsyncFileSystem,
    SkillAuthorizedSourceRoot,
    SkillConfiguredSource,
    SkillResolverSourceConfig,
    SkillSourceFileSystem,
    SkillSourceResolutionResult,
    SkillSourceRootConfig,
    resolve_skill_sources,
)
from .settings import (
    SkillIndexLimits,
    SkillReadLimits,
    SkillSourceLimits,
    TrustedSkillSettings,
)

from dataclasses import dataclass, replace
from enum import StrEnum
from os import stat_result
from pathlib import Path
from stat import S_ISLNK
from typing import Protocol


class SkillRuntimeMode(StrEnum):
    LOCAL = "local"
    SANDBOX = "sandbox"
    CONTAINER = "container"


class SkillRuntimeMappingKind(StrEnum):
    DIRECT = "direct"
    MATERIALIZED_COPY = "materialized_copy"


class SkillRuntimeBackend(Protocol):
    @property
    def mode(self) -> SkillRuntimeMode:
        pass

    async def map_sources(
        self,
        sources: tuple[SkillAuthorizedSourceRoot, ...],
    ) -> "SkillRuntimeMappingResult":
        pass


class SkillRuntimeMappedFileSystem:
    def __init__(
        self,
        *,
        roots: tuple[Path, ...],
        file_system: SkillSourceFileSystem | None = None,
    ) -> None:
        assert isinstance(roots, tuple), "roots must be a tuple"
        for root in roots:
            assert isinstance(root, Path)
        if file_system is None:
            file_system = SkillAsyncFileSystem()
        self._roots = roots
        self._file_system = file_system

    async def resolve_path(self, path: Path) -> Path:
        await skill_cancellation_checkpoint()
        root = self._authorized_root(path)
        if root is None:
            raise PermissionError("skill runtime path outside mapped roots")
        await self._assert_authorized_root_is_not_symlink(root)
        resolved = await self._file_system.resolve_path(path)
        if not self._path_authorized(resolved):
            self._assert_path_authorized(resolved)
        return resolved

    async def stat_path(self, path: Path) -> stat_result:
        await skill_cancellation_checkpoint()
        resolved = await self.resolve_path(path)
        return await self._file_system.stat_path(resolved)

    async def lstat_path(self, path: Path) -> stat_result:
        await skill_cancellation_checkpoint()
        root = self._authorized_root(path)
        if root is None:
            raise PermissionError("skill runtime path outside mapped roots")
        await self._assert_authorized_root_is_not_symlink(root)
        return await self._file_system.lstat_path(path)

    async def _assert_authorized_root_is_not_symlink(
        self,
        root: Path,
    ) -> None:
        assert isinstance(root, Path)
        try:
            root_stat = await self._file_system.lstat_path(root)
        except OSError as exc:
            raise PermissionError(
                "skill runtime path outside mapped roots"
            ) from exc
        if S_ISLNK(root_stat.st_mode):
            raise PermissionError("skill runtime path outside mapped roots")

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        await skill_cancellation_checkpoint()
        resolved = await self.resolve_path(path)
        entries = await self._file_system.list_directory(resolved, limit)
        for entry in entries:
            self._assert_path_authorized(entry)
        return entries

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        await skill_cancellation_checkpoint()
        resolved = await self.resolve_path(path)
        return await self._file_system.read_bytes(resolved, limit)

    def _assert_path_authorized(self, path: Path) -> None:
        assert isinstance(path, Path)
        if self._path_authorized(path):
            return
        raise PermissionError("skill runtime path outside mapped roots")

    def _path_authorized(self, path: Path) -> bool:
        assert isinstance(path, Path)
        return self._authorized_root(path) is not None

    def _authorized_root(self, path: Path) -> Path | None:
        assert isinstance(path, Path)
        for root in self._roots:
            if _path_is_relative_to(path, root):
                return root
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRuntimeSourceMapping:
    source_label: str
    authority: SkillSourceAuthority
    host_root: Path
    runtime_root: Path
    kind: SkillRuntimeMappingKind = SkillRuntimeMappingKind.DIRECT
    source_root_proven: bool = True
    path_sharing_proven: bool = True
    read_only_proven: bool = True
    available: bool = True
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.source_label, str)
        assert self.source_label == sanitize_skill_source_label(
            self.source_label
        )
        assert isinstance(self.authority, SkillSourceAuthority)
        assert isinstance(self.host_root, Path)
        assert isinstance(self.runtime_root, Path)
        assert isinstance(self.kind, SkillRuntimeMappingKind)
        assert isinstance(self.source_root_proven, bool)
        assert isinstance(self.path_sharing_proven, bool)
        assert isinstance(self.read_only_proven, bool)
        assert isinstance(self.available, bool)
        _assert_diagnostics(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        if not self.available:
            return SkillStatus.UNAVAILABLE
        return SkillStatus.OK

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "source_label": self.source_label,
                "authority": self.authority.as_model_dict(),
                "status": self.status.value,
                "kind": self.kind.value,
                "source_root_proven": self.source_root_proven,
                "path_sharing_proven": self.path_sharing_proven,
                "read_only_proven": self.read_only_proven,
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRuntimeMappingResult:
    mode: SkillRuntimeMode
    file_system: SkillSourceFileSystem
    mappings: tuple[SkillRuntimeSourceMapping, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.mode, SkillRuntimeMode)
        assert isinstance(self.mappings, tuple)
        for mapping in self.mappings:
            assert isinstance(mapping, SkillRuntimeSourceMapping)
        _assert_diagnostics(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        for mapping in self.mappings:
            if mapping.status is not SkillStatus.OK:
                return mapping.status
        if self.mappings:
            return SkillStatus.OK
        return SkillStatus.EMPTY

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "mode": self.mode.value,
                "status": self.status.value,
                "mappings": tuple(
                    mapping.as_model_dict() for mapping in self.mappings
                ),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRuntimeResolution:
    mode: SkillRuntimeMode
    mapping: SkillRuntimeMappingResult
    resolution: SkillSourceResolutionResult
    file_system: SkillSourceFileSystem

    def __post_init__(self) -> None:
        assert isinstance(self.mode, SkillRuntimeMode)
        assert isinstance(self.mapping, SkillRuntimeMappingResult)
        assert isinstance(self.resolution, SkillSourceResolutionResult)

    @property
    def status(self) -> SkillStatus:
        if self.resolution.status is not SkillStatus.EMPTY:
            return self.resolution.status
        return self.mapping.status

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "mode": self.mode.value,
                "status": self.status.value,
                "mapping": self.mapping.as_model_dict(),
                "resolution": self.resolution.as_model_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRuntimeRegistry:
    mode: SkillRuntimeMode
    mapping: SkillRuntimeMappingResult
    registry: SkillRegistry
    file_system: SkillSourceFileSystem

    def __post_init__(self) -> None:
        assert isinstance(self.mode, SkillRuntimeMode)
        assert isinstance(self.mapping, SkillRuntimeMappingResult)
        assert isinstance(self.registry, SkillRegistry)

    @property
    def status(self) -> SkillStatus:
        return self.registry.status

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "mode": self.mode.value,
                "status": self.status.value,
                "mapping": self.mapping.as_model_dict(),
                "registry": self.registry.as_model_dict(),
            }
        )


class SkillLocalRuntimeBackend:
    def __init__(
        self,
        *,
        file_system: SkillSourceFileSystem | None = None,
    ) -> None:
        if file_system is None:
            file_system = SkillAsyncFileSystem()
        self._file_system = file_system

    @property
    def mode(self) -> SkillRuntimeMode:
        return SkillRuntimeMode.LOCAL

    async def map_sources(
        self,
        sources: tuple[SkillAuthorizedSourceRoot, ...],
    ) -> SkillRuntimeMappingResult:
        _assert_sources(sources)
        await skill_cancellation_checkpoint()
        return SkillRuntimeMappingResult(
            mode=self.mode,
            file_system=self._file_system,
            mappings=tuple(
                SkillRuntimeSourceMapping(
                    source_label=source.label,
                    authority=source.authority,
                    host_root=source.root,
                    runtime_root=source.root,
                    kind=SkillRuntimeMappingKind.DIRECT,
                    source_root_proven=True,
                    path_sharing_proven=True,
                    read_only_proven=True,
                )
                for source in sources
            ),
        )


async def resolve_skill_runtime_sources(
    configs: tuple[SkillResolverSourceConfig, ...],
    *,
    backend: SkillRuntimeBackend | None = None,
    settings: TrustedSkillSettings | None = None,
    source_limits: SkillSourceLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
    read_limits: SkillReadLimits | None = None,
    file_system: SkillSourceFileSystem | None = None,
) -> SkillRuntimeResolution:
    assert isinstance(configs, tuple), "configs must be a tuple"
    for config in configs:
        assert isinstance(
            config,
            SkillConfiguredSource | SkillSourceRootConfig,
        )
    await skill_cancellation_checkpoint()
    if file_system is None:
        file_system = SkillAsyncFileSystem()
    if backend is None:
        backend = SkillLocalRuntimeBackend(file_system=file_system)
    effective_source_limits = (
        source_limits
        if source_limits is not None
        else (settings.source_limits if settings is not None else None)
    )
    effective_index_limits = (
        index_limits
        if index_limits is not None
        else (settings.index_limits if settings is not None else None)
    )
    effective_read_limits = (
        read_limits
        if read_limits is not None
        else (settings.read_limits if settings is not None else None)
    )

    host_resolution = await resolve_skill_sources(
        configs,
        settings=settings,
        source_limits=effective_source_limits,
        index_limits=effective_index_limits,
        read_limits=effective_read_limits,
        file_system=file_system,
    )
    if not host_resolution.sources:
        runtime_file_system = _empty_runtime_file_system(file_system)
        mapping = SkillRuntimeMappingResult(
            mode=backend.mode,
            file_system=runtime_file_system,
        )
        return SkillRuntimeResolution(
            mode=backend.mode,
            mapping=mapping,
            resolution=host_resolution,
            file_system=runtime_file_system,
        )

    try:
        await skill_cancellation_checkpoint()
        mapping = await backend.map_sources(host_resolution.sources)
    except (OSError, RuntimeError, ValueError):
        mapping = SkillRuntimeMappingResult(
            mode=backend.mode,
            file_system=file_system,
            diagnostics=(
                _runtime_unavailable_diagnostic(reason="backend_unavailable"),
            ),
        )
        return _failed_runtime_resolution(
            mapping,
            _empty_runtime_file_system(file_system),
        )

    diagnostics = _mapping_diagnostics(host_resolution.sources, mapping)
    if diagnostics:
        mapping = replace(mapping, diagnostics=diagnostics)
        return _failed_runtime_resolution(
            mapping,
            _empty_runtime_file_system(mapping.file_system),
        )

    validated_mappings = _validated_mappings(
        host_resolution.sources,
        mapping,
    )
    await skill_cancellation_checkpoint()
    mapping = replace(mapping, mappings=validated_mappings)
    runtime_file_system = SkillRuntimeMappedFileSystem(
        roots=tuple(source.runtime_root for source in validated_mappings),
        file_system=mapping.file_system,
    )
    mapping = replace(mapping, file_system=runtime_file_system)
    mapped_resolution = await resolve_skill_sources(
        _mapped_configs(host_resolution.sources, mapping),
        source_limits=effective_source_limits,
        index_limits=effective_index_limits,
        read_limits=effective_read_limits,
        file_system=runtime_file_system,
    )
    parity_diagnostics = _resource_parity_diagnostics(
        host_resolution.sources,
        mapped_resolution.sources,
    )
    await skill_cancellation_checkpoint()
    if parity_diagnostics:
        mapping = replace(mapping, diagnostics=parity_diagnostics)
        return _failed_runtime_resolution(
            mapping,
            _empty_runtime_file_system(runtime_file_system),
        )

    return SkillRuntimeResolution(
        mode=mapping.mode,
        mapping=mapping,
        resolution=SkillSourceResolutionResult(
            sources=_identity_sources(
                host_resolution.sources,
                mapped_resolution.sources,
            ),
            diagnostics=(
                *host_resolution.diagnostics,
                *mapped_resolution.diagnostics,
            ),
        ),
        file_system=runtime_file_system,
    )


async def build_skill_runtime_registry(
    configs: tuple[SkillResolverSourceConfig, ...],
    *,
    backend: SkillRuntimeBackend | None = None,
    settings: TrustedSkillSettings | None = None,
    source_limits: SkillSourceLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
    read_limits: SkillReadLimits | None = None,
    file_system: SkillSourceFileSystem | None = None,
) -> SkillRuntimeRegistry:
    runtime_resolution = await resolve_skill_runtime_sources(
        configs,
        backend=backend,
        settings=settings,
        source_limits=source_limits,
        index_limits=index_limits,
        read_limits=read_limits,
        file_system=file_system,
    )
    await skill_cancellation_checkpoint()
    registry = await build_skill_registry(
        runtime_resolution.resolution,
        settings=settings,
        read_limits=read_limits,
        index_limits=index_limits,
        file_system=runtime_resolution.file_system,
    )
    return SkillRuntimeRegistry(
        mode=runtime_resolution.mode,
        mapping=runtime_resolution.mapping,
        registry=registry,
        file_system=runtime_resolution.file_system,
    )


def _mapped_configs(
    host_sources: tuple[SkillAuthorizedSourceRoot, ...],
    mapping: SkillRuntimeMappingResult,
) -> tuple[SkillSourceRootConfig, ...]:
    mapping_by_label = _mapping_by_label(mapping.mappings)
    return tuple(
        SkillSourceRootConfig(
            label=source.label,
            authority=source.authority,
            root=mapping_by_label[source.label].runtime_root,
            allow_hidden_paths=source.allow_hidden_paths,
        )
        for source in host_sources
    )


def _identity_sources(
    host_sources: tuple[SkillAuthorizedSourceRoot, ...],
    runtime_sources: tuple[SkillAuthorizedSourceRoot, ...],
) -> tuple[SkillAuthorizedSourceRoot, ...]:
    host_by_label = {source.label: source for source in host_sources}
    return tuple(
        replace(
            source,
            identity_root=host_by_label[source.label].root,
        )
        for source in runtime_sources
    )


def _mapping_diagnostics(
    host_sources: tuple[SkillAuthorizedSourceRoot, ...],
    mapping: SkillRuntimeMappingResult,
) -> tuple[SkillDiagnosticInfo, ...]:
    if mapping.diagnostics:
        return mapping.diagnostics
    expected_labels = tuple(source.label for source in host_sources)
    duplicate_labels = _duplicate_mapping_labels(mapping.mappings)
    if duplicate_labels:
        return (
            _runtime_policy_diagnostic(
                reason="duplicate_mapping_label",
                source_label=duplicate_labels[0],
            ),
        )
    mapping_by_label = _mapping_by_label(mapping.mappings)
    if set(mapping_by_label) != set(expected_labels):
        return (
            _runtime_policy_diagnostic(
                reason="mapping_source_set_changed",
            ),
        )
    diagnostics: list[SkillDiagnosticInfo] = []
    source_by_label = {source.label: source for source in host_sources}
    for label in expected_labels:
        source = source_by_label[label]
        source_mapping = mapping_by_label[label]
        if source_mapping.diagnostics:
            diagnostics.extend(source_mapping.diagnostics)
            continue
        if not source_mapping.available:
            diagnostics.append(
                _runtime_unavailable_diagnostic(
                    reason="mapping_unavailable",
                    source_label=label,
                )
            )
            continue
        if source_mapping.authority != source.authority:
            diagnostics.append(
                _runtime_policy_diagnostic(
                    reason="mapping_authority_changed",
                    source_label=label,
                )
            )
            continue
        if source_mapping.host_root != source.root:
            diagnostics.append(
                _runtime_policy_diagnostic(
                    reason="source_root_widened",
                    source_label=label,
                )
            )
            continue
        if not source_mapping.source_root_proven:
            diagnostics.append(
                _runtime_unavailable_diagnostic(
                    reason="source_root_unproven",
                    source_label=label,
                )
            )
            continue
        if mapping.mode in {
            SkillRuntimeMode.SANDBOX,
            SkillRuntimeMode.CONTAINER,
        }:
            diagnostics.extend(
                _isolated_mapping_diagnostics(source_mapping, label)
            )
    return tuple(diagnostics)


def _isolated_mapping_diagnostics(
    mapping: SkillRuntimeSourceMapping,
    source_label: str,
) -> tuple[SkillDiagnosticInfo, ...]:
    diagnostics: list[SkillDiagnosticInfo] = []
    if not mapping.read_only_proven:
        diagnostics.append(
            _runtime_unavailable_diagnostic(
                reason="read_only_unproven",
                source_label=source_label,
            )
        )
    if (
        mapping.kind is SkillRuntimeMappingKind.DIRECT
        and not mapping.path_sharing_proven
    ):
        diagnostics.append(
            _runtime_unavailable_diagnostic(
                reason="path_sharing_unproven",
                source_label=source_label,
            )
        )
    return tuple(diagnostics)


def _resource_parity_diagnostics(
    host_sources: tuple[SkillAuthorizedSourceRoot, ...],
    runtime_sources: tuple[SkillAuthorizedSourceRoot, ...],
) -> tuple[SkillDiagnosticInfo, ...]:
    host_resources = _resource_identity_by_label(host_sources)
    runtime_resources = _resource_identity_by_label(runtime_sources)
    if host_resources == runtime_resources:
        return ()
    return (
        _runtime_policy_diagnostic(
            reason="runtime_resource_set_changed",
        ),
    )


def _resource_identity_by_label(
    sources: tuple[SkillAuthorizedSourceRoot, ...],
) -> dict[str, tuple[tuple[str, int, int, str | None], ...]]:
    return {
        source.label: tuple(
            sorted(
                (
                    resource.resource_id,
                    resource.size_bytes,
                    resource.line_count,
                    resource.content_sha256,
                )
                for resource in source.resources
            )
        )
        for source in sources
    }


def _validated_mappings(
    host_sources: tuple[SkillAuthorizedSourceRoot, ...],
    mapping: SkillRuntimeMappingResult,
) -> tuple[SkillRuntimeSourceMapping, ...]:
    mapping_by_label = _mapping_by_label(mapping.mappings)
    return tuple(mapping_by_label[source.label] for source in host_sources)


def _duplicate_mapping_labels(
    mappings: tuple[SkillRuntimeSourceMapping, ...],
) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for mapping in mappings:
        if mapping.source_label in seen:
            duplicates.add(mapping.source_label)
        seen.add(mapping.source_label)
    return tuple(sorted(duplicates))


def _mapping_by_label(
    mappings: tuple[SkillRuntimeSourceMapping, ...],
) -> dict[str, SkillRuntimeSourceMapping]:
    mapping_by_label: dict[str, SkillRuntimeSourceMapping] = {}
    for mapping in mappings:
        assert mapping.source_label not in mapping_by_label
        mapping_by_label[mapping.source_label] = mapping
    return mapping_by_label


def _failed_runtime_resolution(
    mapping: SkillRuntimeMappingResult,
    file_system: SkillSourceFileSystem,
) -> SkillRuntimeResolution:
    mapping = replace(mapping, file_system=file_system)
    return SkillRuntimeResolution(
        mode=mapping.mode,
        mapping=mapping,
        resolution=SkillSourceResolutionResult(
            diagnostics=mapping.diagnostics
        ),
        file_system=file_system,
    )


def _empty_runtime_file_system(
    file_system: SkillSourceFileSystem,
) -> SkillRuntimeMappedFileSystem:
    return SkillRuntimeMappedFileSystem(roots=(), file_system=file_system)


def _runtime_unavailable_diagnostic(
    *,
    reason: str,
    source_label: str | None = None,
) -> SkillDiagnosticInfo:
    details: dict[str, SkillModelValue] = {"reason": reason}
    if source_label is not None:
        details["source_label"] = source_label
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
        status=SkillStatus.UNAVAILABLE,
        message=(
            "The sandbox or container runtime cannot access the configured "
            "source."
        ),
        path="source.availability",
        hint="Keep the registry unavailable instead of widening access.",
        details=details,
    )


def _runtime_policy_diagnostic(
    *,
    reason: str,
    source_label: str | None = None,
) -> SkillDiagnosticInfo:
    details: dict[str, SkillModelValue] = {"reason": reason}
    if source_label is not None:
        details["source_label"] = source_label
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="The runtime skill source mapping is not authorized.",
        path="source.runtime",
        hint="Expose only the exact authorized read-only skill source root.",
        details=details,
    )


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_sources(
    values: tuple[SkillAuthorizedSourceRoot, ...],
) -> None:
    assert isinstance(values, tuple), "sources must be a tuple"
    for value in values:
        assert isinstance(value, SkillAuthorizedSourceRoot)


def _assert_diagnostics(
    values: tuple[SkillDiagnosticInfo, ...],
) -> None:
    assert isinstance(values, tuple), "diagnostics must be a tuple"
    for value in values:
        assert isinstance(value, SkillDiagnosticInfo)
