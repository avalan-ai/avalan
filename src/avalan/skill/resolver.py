from .contract import SkillDiagnosticCode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillSourceAuthority,
    model_dict,
)
from .path_policy import (
    redact_host_path,
    sanitize_skill_resource_id,
    sanitize_skill_source_label,
    skill_model_handle_denial_reason,
    skill_source_root_denial_reason,
)
from .settings import (
    SkillIndexLimits,
    SkillReadLimits,
    SkillSourceLimits,
    TrustedSkillSettings,
)

from asyncio import Semaphore, to_thread
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from os import stat_result
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import Protocol, TypeAlias, TypeVar

_T = TypeVar("_T")


class SkillSourceFileSystem(Protocol):
    async def resolve_path(self, path: Path) -> Path:
        raise NotImplementedError  # pragma: no cover

    async def stat_path(self, path: Path) -> stat_result:
        raise NotImplementedError  # pragma: no cover

    async def lstat_path(self, path: Path) -> stat_result:
        raise NotImplementedError  # pragma: no cover

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        raise NotImplementedError  # pragma: no cover

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise NotImplementedError  # pragma: no cover


class SkillAsyncFileSystem:
    def __init__(self, *, max_concurrency: int = 8) -> None:
        assert isinstance(max_concurrency, int) and not isinstance(
            max_concurrency, bool
        ), "max_concurrency must be an integer"
        assert max_concurrency > 0, "max_concurrency must be positive"
        self._semaphore = Semaphore(max_concurrency)

    async def resolve_path(self, path: Path) -> Path:
        assert isinstance(path, Path)
        return await self._run(lambda: path.resolve(strict=True))

    async def stat_path(self, path: Path) -> stat_result:
        assert isinstance(path, Path)
        return await self._run(path.stat)

    async def lstat_path(self, path: Path) -> stat_result:
        assert isinstance(path, Path)
        return await self._run(path.lstat)

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        assert isinstance(path, Path)
        assert isinstance(limit, int) and not isinstance(limit, bool)
        assert limit > 0, "limit must be positive"
        return await self._run(lambda: _bounded_list_directory(path, limit))

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        assert isinstance(path, Path)
        assert isinstance(limit, int) and not isinstance(limit, bool)
        assert limit > 0, "limit must be positive"
        return await self._run(lambda: _read_bytes(path, limit))

    async def _run(self, call: Callable[[], _T]) -> _T:
        async with self._semaphore:
            return await to_thread(call)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillConfiguredSource:
    label: str
    authority: SkillSourceAuthority
    root_path: str | Path
    package_path: str | None = None
    enabled: bool = True
    allow_hidden_paths: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.label, str), "label must be a string"
        assert self.label.strip(), "label must be non-empty"
        assert isinstance(self.authority, SkillSourceAuthority)
        assert isinstance(self.root_path, str | Path)
        if self.package_path is not None:
            assert isinstance(self.package_path, str)
            assert self.package_path.strip(), "package_path must be non-empty"
        assert isinstance(self.enabled, bool)
        assert isinstance(self.allow_hidden_paths, bool)

    @property
    def source_label(self) -> str:
        return sanitize_skill_source_label(self.label)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "label": self.source_label,
            "authority": self.authority.as_model_dict(),
            "enabled": self.enabled,
            "allow_hidden_paths": self.allow_hidden_paths,
        }
        if self.package_path is not None:
            value["package"] = _model_package_path(self.package_path)
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSourceRootConfig:
    label: str
    authority: SkillSourceAuthority
    root: str | Path
    package_path: str = "."
    enabled: bool = True
    allow_hidden_paths: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.label, str), "label must be a string"
        assert self.label.strip(), "label must be non-empty"
        assert isinstance(self.authority, SkillSourceAuthority)
        assert isinstance(self.root, str | Path), "root must be a path"
        assert isinstance(
            self.package_path, str
        ), "package_path must be a string"
        assert self.package_path.strip(), "package_path must be non-empty"
        assert isinstance(self.enabled, bool)
        assert isinstance(self.allow_hidden_paths, bool)

    @property
    def source_label(self) -> str:
        return sanitize_skill_source_label(self.label)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "label": self.source_label,
                "authority": self.authority.as_model_dict(),
                "enabled": self.enabled,
                "allow_hidden_paths": self.allow_hidden_paths,
                "package_path": _model_package_path(self.package_path),
            }
        )


SkillResolverSourceConfig: TypeAlias = (
    SkillConfiguredSource | SkillSourceRootConfig
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillAuthorizedResource:
    source_label: str
    resource_id: str
    path: Path
    size_bytes: int
    line_count: int

    def __post_init__(self) -> None:
        assert isinstance(self.source_label, str)
        assert self.source_label == sanitize_skill_source_label(
            self.source_label
        )
        assert isinstance(self.resource_id, str)
        assert self.resource_id == sanitize_skill_resource_id(self.resource_id)
        assert isinstance(self.path, Path)
        _assert_non_negative_int(self.size_bytes, "size_bytes")
        _assert_non_negative_int(self.line_count, "line_count")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "source_label": self.source_label,
                "resource_id": self.resource_id,
                "size_bytes": self.size_bytes,
                "line_count": self.line_count,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillAuthorizedSourceRoot:
    label: str
    authority: SkillSourceAuthority
    root: Path
    allow_hidden_paths: bool = False
    resources: tuple[SkillAuthorizedResource, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.label, str)
        assert self.label == sanitize_skill_source_label(self.label)
        assert isinstance(self.authority, SkillSourceAuthority)
        assert isinstance(self.root, Path)
        assert isinstance(self.allow_hidden_paths, bool)
        _assert_resource_tuple(self.resources)
        for resource in self.resources:
            assert resource.source_label == self.label
        _assert_diagnostic_tuple(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        return SkillStatus.OK

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "label": self.label,
                "source_id": f"source:{self.label}",
                "authority": self.authority.as_model_dict(),
                "status": self.status.value,
                "allow_hidden_paths": self.allow_hidden_paths,
                "resource_count": len(self.resources),
                "resources": tuple(
                    resource.as_model_dict() for resource in self.resources
                ),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResourceAuthorizationResult:
    source_label: str
    resource: SkillAuthorizedResource | None = None
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.source_label, str)
        assert self.source_label == sanitize_skill_source_label(
            self.source_label
        )
        if self.resource is not None:
            assert isinstance(self.resource, SkillAuthorizedResource)
            assert self.resource.source_label == self.source_label
        _assert_diagnostic_tuple(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        if self.resource is not None:
            return SkillStatus.OK
        if self.diagnostics:
            return self.diagnostics[0].status
        return SkillStatus.NOT_FOUND

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "status": self.status.value,
            "source_label": self.source_label,
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
        }
        if self.resource is not None:
            value["resource"] = self.resource.as_model_dict()
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSourceResolutionResult:
    sources: tuple[SkillAuthorizedSourceRoot, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        _assert_source_tuple(self.sources)
        _assert_diagnostic_tuple(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        if self.sources:
            return SkillStatus.OK
        if self.diagnostics:
            return self.diagnostics[0].status
        return SkillStatus.EMPTY

    @property
    def resources(self) -> tuple[SkillAuthorizedResource, ...]:
        return tuple(
            resource
            for source in self.sources
            for resource in source.resources
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "status": self.status.value,
                "sources": tuple(
                    source.as_model_dict() for source in self.sources
                ),
                "resources": tuple(
                    resource.as_model_dict() for resource in self.resources
                ),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


class SkillSourceResolver:
    def __init__(
        self,
        *,
        source_limits: SkillSourceLimits | None = None,
        index_limits: SkillIndexLimits | None = None,
        read_limits: SkillReadLimits | None = None,
        file_system: SkillSourceFileSystem | None = None,
    ) -> None:
        if source_limits is None:
            source_limits = SkillSourceLimits()
        if index_limits is None:
            index_limits = SkillIndexLimits()
        if read_limits is None:
            read_limits = SkillReadLimits()
        if file_system is None:
            file_system = SkillAsyncFileSystem()
        assert isinstance(source_limits, SkillSourceLimits)
        assert isinstance(index_limits, SkillIndexLimits)
        assert isinstance(read_limits, SkillReadLimits)
        self._source_limits = source_limits
        self._index_limits = index_limits
        self._read_limits = read_limits
        self._file_system = file_system

    async def resolve(
        self,
        configs: tuple[SkillResolverSourceConfig, ...],
        *,
        settings: TrustedSkillSettings | None = None,
    ) -> SkillSourceResolutionResult:
        _assert_config_tuple(configs)
        diagnostics: list[SkillDiagnosticInfo] = []
        if settings is not None and not settings.enabled:
            diagnostics.append(
                _policy_diagnostic(
                    path="settings.enabled",
                    message="Trusted skills settings are disabled.",
                    hint="Do not resolve skill sources when skills are off.",
                    reason="settings_disabled",
                    code=SkillDiagnosticCode.DISABLED,
                    status=SkillStatus.DISABLED,
                )
            )
            return SkillSourceResolutionResult(diagnostics=tuple(diagnostics))
        if len(configs) > self._source_limits.max_sources:
            diagnostics.append(
                _policy_diagnostic(
                    path="source.count",
                    message="Too many skill sources are configured.",
                    hint="Reduce trusted skills source configuration.",
                    reason="source_count",
                    status=SkillStatus.BLOCKED,
                )
            )
            return SkillSourceResolutionResult(diagnostics=tuple(diagnostics))
        configured: list[SkillSourceRootConfig] = []
        for config in configs[: self._source_limits.max_sources]:
            normalized, diagnostic = _normalize_config(config)
            assert diagnostic is None
            assert normalized is not None
            trust_diagnostic = _trusted_settings_diagnostic(
                normalized,
                settings,
            )
            if trust_diagnostic is not None:
                diagnostics.append(trust_diagnostic)
                continue
            configured.append(normalized)
        configured_tuple = tuple(configured)
        duplicate_labels = _duplicate_labels(configured_tuple)
        if duplicate_labels:
            diagnostics.append(
                _policy_diagnostic(
                    path="source.label",
                    message="Configured skill sources use duplicate labels.",
                    hint="Assign a unique logical label to each source.",
                    reason="duplicate_source_label",
                    candidates=duplicate_labels,
                    code=SkillDiagnosticCode.DUPLICATE_ID,
                    status=SkillStatus.BLOCKED,
                )
            )

        sources: list[SkillAuthorizedSourceRoot] = []
        for config in configured_tuple:
            source_label = config.source_label
            if source_label in duplicate_labels:
                continue
            if not config.enabled:
                continue
            root, diagnostic = await self._resolve_root(config)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
                continue
            assert root is not None
            resources, source_diagnostics = await self._scan_source(
                source_label=source_label,
                root=root,
                allow_hidden_paths=config.allow_hidden_paths,
            )
            sources.append(
                SkillAuthorizedSourceRoot(
                    label=source_label,
                    authority=config.authority,
                    root=root,
                    allow_hidden_paths=config.allow_hidden_paths,
                    resources=resources,
                    diagnostics=source_diagnostics,
                )
            )
        return SkillSourceResolutionResult(
            sources=tuple(sources),
            diagnostics=tuple(diagnostics),
        )

    async def authorize_resource(
        self,
        source: SkillAuthorizedSourceRoot,
        model_handle: str,
    ) -> SkillResourceAuthorizationResult:
        assert isinstance(source, SkillAuthorizedSourceRoot)
        assert isinstance(model_handle, str)
        reason = skill_model_handle_denial_reason(
            model_handle,
            allow_hidden_paths=source.allow_hidden_paths,
        )
        resource_id = sanitize_skill_resource_id(model_handle)
        if reason is not None:
            return SkillResourceAuthorizationResult(
                source_label=source.label,
                diagnostics=(
                    _resource_policy_diagnostic(
                        reason=reason,
                        resource_id=resource_id,
                    ),
                ),
            )
        candidate = source.root.joinpath(*PurePosixPath(model_handle).parts)
        resource, diagnostic = await self._authorize_path(
            source_label=source.label,
            root=source.root,
            resource_path=candidate,
            resource_id=resource_id,
            allow_hidden_paths=source.allow_hidden_paths,
        )
        if diagnostic is not None:
            return SkillResourceAuthorizationResult(
                source_label=source.label,
                diagnostics=(diagnostic,),
            )
        assert resource is not None
        return SkillResourceAuthorizationResult(
            source_label=source.label,
            resource=resource,
        )

    async def _resolve_root(
        self, config: SkillSourceRootConfig
    ) -> tuple[Path | None, SkillDiagnosticInfo | None]:
        root_text = str(config.root)
        if "\x00" in config.label:
            return None, _policy_diagnostic(
                path="source.label",
                message="The configured skill source label is unsafe.",
                hint="Use logical source labels without control bytes.",
                reason="nul_byte",
            )
        reason = skill_source_root_denial_reason(
            root_text,
            allow_hidden_paths=config.allow_hidden_paths,
        )
        if reason is not None:
            return None, _source_unavailable_diagnostic(
                reason=reason,
                root_path=root_text,
            )
        package_reason = _package_path_denial_reason(config)
        if package_reason is not None:
            return None, _source_unavailable_diagnostic(
                reason=package_reason,
                root_path=root_text,
            )
        try:
            root = await self._file_system.resolve_path(Path(root_text))
            stat = await self._file_system.stat_path(root)
        except (OSError, RuntimeError, ValueError):
            return None, _source_unavailable_diagnostic(
                reason="unavailable",
                root_path=root_text,
            )
        if not S_ISDIR(stat.st_mode):
            return None, _source_unavailable_diagnostic(
                reason="not_directory",
                root_path=root_text,
            )
        if config.package_path == ".":
            return root, None
        package_root = root.joinpath(*PurePosixPath(config.package_path).parts)
        try:
            resolved_package = await self._file_system.resolve_path(
                package_root
            )
            package_stat = await self._file_system.stat_path(resolved_package)
        except (OSError, RuntimeError, ValueError):
            return None, _source_unavailable_diagnostic(
                reason="unavailable_package",
                root_path=str(package_root),
            )
        if not _path_is_relative_to(resolved_package, root):
            return None, _outside_root_diagnostic(reason="package_escape")
        if not S_ISDIR(package_stat.st_mode):
            return None, _source_unavailable_diagnostic(
                reason="package_not_directory",
                root_path=str(package_root),
            )
        return resolved_package, None

    async def _scan_source(
        self,
        *,
        source_label: str,
        root: Path,
        allow_hidden_paths: bool,
    ) -> tuple[
        tuple[SkillAuthorizedResource, ...],
        tuple[SkillDiagnosticInfo, ...],
    ]:
        resources: list[SkillAuthorizedResource] = []
        diagnostics: list[SkillDiagnosticInfo] = []
        stack: list[tuple[Path, int]] = [(root, 0)]
        file_count = 0
        indexed_bytes = 0
        work_units = 0

        while stack:
            directory, depth = stack.pop()
            remaining_work = (
                self._source_limits.max_directory_entries_per_source
                - work_units
            )
            try:
                entries = await self._file_system.list_directory(
                    directory,
                    remaining_work + 1,
                )
            except OSError:
                diagnostics.append(
                    _source_unavailable_diagnostic(reason="unavailable")
                )
                break
            if len(entries) > remaining_work:
                diagnostics.append(_bound_diagnostic("directory_traversal"))
                return tuple(resources), tuple(diagnostics)
            work_units += len(entries)
            for entry in entries:
                relative = _relative_resource_id(root, entry)
                reason = skill_model_handle_denial_reason(
                    relative,
                    allow_hidden_paths=allow_hidden_paths,
                )
                if reason is not None:
                    diagnostics.append(
                        _resource_policy_diagnostic(
                            reason=reason,
                            resource_id=sanitize_skill_resource_id(relative),
                        )
                    )
                    continue
                try:
                    lstat = await self._file_system.lstat_path(entry)
                except OSError:
                    diagnostics.append(_missing_resource_diagnostic())
                    continue
                entry_depth = len(PurePosixPath(relative).parts)
                if S_ISDIR(lstat.st_mode):
                    if entry_depth > self._source_limits.max_source_depth:
                        diagnostics.append(_bound_diagnostic("source_depth"))
                    else:
                        stack.append((entry, entry_depth))
                    continue
                file_count += 1
                if file_count > self._source_limits.max_files_per_source:
                    diagnostics.append(_bound_diagnostic("file_count"))
                    return tuple(resources), tuple(diagnostics)
                if (
                    len(resources)
                    >= self._source_limits.max_resources_per_source
                ):
                    diagnostics.append(_bound_diagnostic("resource_count"))
                    return tuple(resources), tuple(diagnostics)
                if S_ISLNK(lstat.st_mode):
                    resource, diagnostic = await self._authorize_symlink(
                        source_label=source_label,
                        root=root,
                        path=entry,
                        relative=relative,
                        allow_hidden_paths=allow_hidden_paths,
                    )
                else:
                    if entry_depth > self._source_limits.max_source_depth:
                        diagnostics.append(_bound_diagnostic("source_depth"))
                        continue
                    resource, diagnostic = await self._authorize_path(
                        source_label=source_label,
                        root=root,
                        resource_path=entry,
                        resource_id=sanitize_skill_resource_id(relative),
                        allow_hidden_paths=allow_hidden_paths,
                    )
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                    continue
                assert resource is not None
                indexed_bytes += resource.size_bytes
                if indexed_bytes > self._index_limits.max_indexed_bytes:
                    diagnostics.append(_bound_diagnostic("indexed_bytes"))
                    return tuple(resources), tuple(diagnostics)
                resources.append(resource)
        return tuple(resources), tuple(diagnostics)

    async def _authorize_symlink(
        self,
        *,
        source_label: str,
        root: Path,
        path: Path,
        relative: str,
        allow_hidden_paths: bool,
    ) -> tuple[SkillAuthorizedResource | None, SkillDiagnosticInfo | None]:
        try:
            resolved = await self._file_system.resolve_path(path)
        except (OSError, RuntimeError, ValueError):
            return None, _source_unavailable_diagnostic(
                reason="dangling_symlink"
            )
        if not _path_is_relative_to(resolved, root):
            return None, _outside_root_diagnostic(reason="symlink_escape")
        target_relative = _relative_resource_id(root, resolved)
        target_depth = len(PurePosixPath(target_relative).parts)
        if target_depth > self._source_limits.max_source_depth:
            return None, _bound_diagnostic("source_depth")
        reason = skill_model_handle_denial_reason(
            target_relative,
            allow_hidden_paths=allow_hidden_paths,
        )
        if reason is not None:
            return None, _resource_policy_diagnostic(
                reason=reason,
                resource_id=sanitize_skill_resource_id(relative),
            )
        return await self._authorize_path(
            source_label=source_label,
            root=root,
            resource_path=resolved,
            resource_id=sanitize_skill_resource_id(relative),
            allow_hidden_paths=allow_hidden_paths,
        )

    async def _authorize_path(
        self,
        *,
        source_label: str,
        root: Path,
        resource_path: Path,
        resource_id: str,
        allow_hidden_paths: bool,
    ) -> tuple[SkillAuthorizedResource | None, SkillDiagnosticInfo | None]:
        try:
            resolved = await self._file_system.resolve_path(resource_path)
        except FileNotFoundError:
            return None, _missing_resource_diagnostic()
        except (OSError, RuntimeError, ValueError):
            return None, _source_unavailable_diagnostic(reason="unavailable")
        if not _path_is_relative_to(resolved, root):
            return None, _outside_root_diagnostic(reason="path_escape")
        target_relative = _relative_resource_id(root, resolved)
        target_depth = len(PurePosixPath(target_relative).parts)
        if target_depth > self._source_limits.max_source_depth:
            return None, _bound_diagnostic("source_depth")
        reason = skill_model_handle_denial_reason(
            target_relative,
            allow_hidden_paths=allow_hidden_paths,
        )
        if reason is not None:
            return None, _resource_policy_diagnostic(
                reason=reason,
                resource_id=resource_id,
            )
        try:
            stat = await self._file_system.stat_path(resolved)
        except OSError:
            return None, _source_unavailable_diagnostic(reason="unavailable")
        if not S_ISREG(stat.st_mode):
            return None, _resource_policy_diagnostic(
                reason="special_file",
                resource_id=resource_id,
            )
        size_bytes = stat.st_size
        if size_bytes > self._read_limits.max_bytes_per_read:
            return None, _oversized_resource_diagnostic(
                reason="per_resource_bytes",
                resource_id=resource_id,
            )
        try:
            content = await self._file_system.read_bytes(
                resolved,
                self._read_limits.max_bytes_per_read + 1,
            )
        except OSError:
            return None, _source_unavailable_diagnostic(reason="unavailable")
        if len(content) > self._read_limits.max_bytes_per_read:
            return None, _oversized_resource_diagnostic(
                reason="per_resource_bytes",
                resource_id=resource_id,
            )
        if b"\x00" in content:
            return None, _resource_policy_diagnostic(
                reason="nul_byte",
                resource_id=resource_id,
            )
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return None, _resource_policy_diagnostic(
                reason="non_utf8_resource",
                resource_id=resource_id,
            )
        line_count = _line_count(content)
        if line_count > self._read_limits.max_lines_per_read:
            return None, _oversized_resource_diagnostic(
                reason="line_count",
                resource_id=resource_id,
            )
        return (
            SkillAuthorizedResource(
                source_label=source_label,
                resource_id=resource_id,
                path=resolved,
                size_bytes=size_bytes,
                line_count=line_count,
            ),
            None,
        )


async def resolve_skill_sources(
    configs: tuple[SkillResolverSourceConfig, ...],
    *,
    settings: TrustedSkillSettings | None = None,
    source_limits: SkillSourceLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
    read_limits: SkillReadLimits | None = None,
    file_system: SkillSourceFileSystem | None = None,
) -> SkillSourceResolutionResult:
    if settings is not None:
        assert isinstance(settings, TrustedSkillSettings)
        if source_limits is None:
            source_limits = settings.source_limits
        if index_limits is None:
            index_limits = settings.index_limits
        if read_limits is None:
            read_limits = settings.read_limits
    resolver = SkillSourceResolver(
        source_limits=source_limits,
        index_limits=index_limits,
        read_limits=read_limits,
        file_system=file_system,
    )
    return await resolver.resolve(configs, settings=settings)


async def authorize_skill_resource(
    source: SkillAuthorizedSourceRoot,
    model_handle: str,
    *,
    source_limits: SkillSourceLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
    read_limits: SkillReadLimits | None = None,
    file_system: SkillSourceFileSystem | None = None,
) -> SkillResourceAuthorizationResult:
    resolver = SkillSourceResolver(
        source_limits=source_limits,
        index_limits=index_limits,
        read_limits=read_limits,
        file_system=file_system,
    )
    return await resolver.authorize_resource(source, model_handle)


AuthorizedSkillResource: TypeAlias = SkillAuthorizedResource
AuthorizedSkillSource: TypeAlias = SkillAuthorizedSourceRoot
LocalSkillSourceFilesystem: TypeAlias = SkillAsyncFileSystem
SkillSourceFilesystem: TypeAlias = SkillSourceFileSystem
SkillSourceResolution: TypeAlias = SkillSourceResolutionResult


def _read_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as input_file:
        return input_file.read(limit)


def _bounded_list_directory(path: Path, limit: int) -> tuple[Path, ...]:
    entries: list[Path] = []
    for index, entry in enumerate(path.iterdir()):
        if index >= limit:
            break
        entries.append(entry)
    return tuple(sorted(entries, key=lambda item: item.name))


def _relative_resource_id(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return sanitize_skill_resource_id(path.name)


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _model_package_path(value: str) -> str:
    if value == ".":
        return "."
    return sanitize_skill_resource_id(value)


def _package_path_denial_reason(
    config: SkillSourceRootConfig,
) -> str | None:
    if config.package_path == ".":
        return None
    return skill_model_handle_denial_reason(
        config.package_path,
        allow_hidden_paths=config.allow_hidden_paths,
    )


def _line_count(content: bytes) -> int:
    if not content:
        return 0
    line_count = content.count(b"\n")
    if not content.endswith(b"\n"):
        line_count += 1
    return line_count


def _duplicate_labels(
    configs: tuple[SkillSourceRootConfig, ...],
) -> tuple[str, ...]:
    counts = Counter(config.source_label for config in configs)
    return tuple(sorted(label for label, count in counts.items() if count > 1))


def _normalize_config(
    config: SkillResolverSourceConfig,
) -> tuple[SkillSourceRootConfig | None, SkillDiagnosticInfo | None]:
    if isinstance(config, SkillSourceRootConfig):
        return config, None
    assert isinstance(config, SkillConfiguredSource)
    return (
        SkillSourceRootConfig(
            label=config.label,
            authority=config.authority,
            root=config.root_path,
            package_path=config.package_path or ".",
            enabled=config.enabled,
            allow_hidden_paths=config.allow_hidden_paths,
        ),
        None,
    )


def _trusted_settings_diagnostic(
    config: SkillSourceRootConfig,
    settings: TrustedSkillSettings | None,
) -> SkillDiagnosticInfo | None:
    if settings is None:
        return None
    if config.authority.kind not in settings.authority_kinds:
        return _policy_diagnostic(
            path="source.authority",
            message="The configured skill source authority is not trusted.",
            hint="Use only authorities allowed by trusted settings.",
            reason="untrusted_authority",
        )
    if not settings.sources:
        return None
    source_by_label = {source.label: source for source in settings.sources}
    trusted_source = source_by_label.get(config.source_label)
    if trusted_source is None:
        return _policy_diagnostic(
            path="source.label",
            message="The configured skill source label is not trusted.",
            hint="Declare source labels in trusted operator settings.",
            reason="untrusted_label",
        )
    if trusted_source.authority != config.authority:
        return _policy_diagnostic(
            path="source.authority",
            message="The configured skill source authority is not trusted.",
            hint="Use the authority assigned to the trusted source label.",
            reason="untrusted_authority",
        )
    if (
        not trusted_source.enabled
        or trusted_source.status is SkillStatus.DISABLED
    ):
        return _policy_diagnostic(
            path="source.enabled",
            message="The trusted skill source is disabled.",
            hint="Do not resolve disabled skill sources.",
            reason="disabled_source",
            code=SkillDiagnosticCode.DISABLED,
            status=SkillStatus.DISABLED,
        )
    if trusted_source.status is not SkillStatus.OK:
        if trusted_source.diagnostics:
            return trusted_source.diagnostics[0]
        return _source_unavailable_diagnostic(reason="trusted_source_status")
    return None


def _policy_diagnostic(
    *,
    path: str,
    message: str,
    hint: str,
    reason: str,
    candidates: tuple[str, ...] = (),
    code: SkillDiagnosticCode = SkillDiagnosticCode.POLICY_DENIED,
    status: SkillStatus = SkillStatus.POLICY_DENIED,
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=code,
        status=status,
        message=message,
        path=path,
        hint=hint,
        candidates=candidates,
        details={"reason": reason},
    )


def _resource_policy_diagnostic(
    *,
    reason: str,
    resource_id: str,
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="The skill resource path is not authorized.",
        path="resource.policy",
        hint="Use logical resource handles inside the authorized source root.",
        details={"reason": reason, "resource_id": resource_id},
    )


def _outside_root_diagnostic(*, reason: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
        status=SkillStatus.POLICY_DENIED,
        message="The skill resource is outside its authorized root.",
        path="resource.policy",
        hint="Reject traversal and use logical resource IDs only.",
        details={"reason": reason},
    )


def _source_unavailable_diagnostic(
    *,
    reason: str,
    root_path: str | None = None,
) -> SkillDiagnosticInfo:
    details: dict[str, SkillModelValue] = {"reason": reason}
    if root_path is not None:
        details["path"] = redact_host_path(root_path)
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.SOURCE_UNAVAILABLE,
        status=SkillStatus.UNAVAILABLE,
        message="The configured skill source is unavailable.",
        path="source.availability",
        hint="Do not substitute another source during this run.",
        details=details,
    )


def _missing_resource_diagnostic() -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_MISSING,
        status=SkillStatus.NOT_FOUND,
        message="The requested skill resource is missing.",
        path="resource.lookup",
        hint="Read only declared resources from the current registry.",
    )


def _oversized_resource_diagnostic(
    *,
    reason: str,
    resource_id: str,
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OVERSIZED,
        status=SkillStatus.TRUNCATED,
        message="The skill resource exceeds configured bounds.",
        path="resource.bounds",
        hint="Use only bounded skill resources.",
        details={"reason": reason, "resource_id": resource_id},
    )


def _bound_diagnostic(reason: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OVERSIZED,
        status=SkillStatus.TRUNCATED,
        message="The skill source exceeds configured traversal bounds.",
        path="source.bounds",
        hint="Reduce source size or raise trusted operator limits.",
        details={"reason": reason},
    )


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must be non-negative"


def _assert_config_tuple(
    values: tuple[SkillResolverSourceConfig, ...],
) -> None:
    assert isinstance(values, tuple), "configs must be a tuple"
    for value in values:
        assert isinstance(value, SkillSourceRootConfig | SkillConfiguredSource)


def _assert_source_tuple(
    values: tuple[SkillAuthorizedSourceRoot, ...],
) -> None:
    assert isinstance(values, tuple), "sources must be a tuple"
    for value in values:
        assert isinstance(value, SkillAuthorizedSourceRoot)


def _assert_resource_tuple(
    values: tuple[SkillAuthorizedResource, ...],
) -> None:
    assert isinstance(values, tuple), "resources must be a tuple"
    for value in values:
        assert isinstance(value, SkillAuthorizedResource)


def _assert_diagnostic_tuple(values: tuple[SkillDiagnosticInfo, ...]) -> None:
    assert isinstance(values, tuple), "diagnostics must be a tuple"
    for value in values:
        assert isinstance(value, SkillDiagnosticInfo)
