from .contract import SkillDiagnosticCode, SkillFailureMode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillProvenance,
    SkillReadCursor,
    SkillResourceContent,
    diagnostic_from_failure,
    model_dict,
)
from .envelope import SkillResponseEnvelope
from .normalizer import normalize_skill_name, skill_name_denial_reason
from .path_policy import skill_model_handle_denial_reason
from .registry import (
    SkillRegisteredResource,
    SkillRegistry,
    SkillRegistryResourceCheck,
    SkillRegistrySkill,
)
from .resolver import SkillAsyncFileSystem, SkillSourceFileSystem
from .settings import SkillCursorLimits, SkillReadLimits

from asyncio import Lock
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from re import fullmatch
from secrets import token_hex
from stat import S_ISLNK, S_ISREG
from time import monotonic

_HASH_PREFIX_LENGTH = 16
_MAIN_RESOURCE_ID = "main"


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReadTarget:
    skill: SkillRegistrySkill
    resource: SkillRegisteredResource


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReadWindow:
    text: str
    start_byte: int
    end_byte: int
    truncated: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _StoredCursor:
    cursor: SkillReadCursor
    created_at: float
    expires_at: float
    limit_lines: int
    sequence: int


class SkillResourceReader:
    """Read registry-authorized skill resources with bounded cursors."""

    def __init__(
        self,
        *,
        read_limits: SkillReadLimits | None = None,
        cursor_limits: SkillCursorLimits | None = None,
        file_system: SkillSourceFileSystem | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if read_limits is not None:
            assert isinstance(read_limits, SkillReadLimits)
        if cursor_limits is not None:
            assert isinstance(cursor_limits, SkillCursorLimits)
        if file_system is None:
            file_system = SkillAsyncFileSystem()
        if clock is None:
            clock = monotonic
        self._read_limits = read_limits
        self._cursor_limits = cursor_limits
        self._file_system = file_system
        self._clock = clock
        self._cursors: dict[str, _StoredCursor] = {}
        self._lock = Lock()
        self._sequence = 0

    @property
    def active_cursor_count(self) -> int:
        return len(self._cursors)

    async def read(
        self,
        registry: SkillRegistry,
        skill: str | None = None,
        *,
        resource_id: str = _MAIN_RESOURCE_ID,
        source_label: str | None = None,
        cursor_id: str | None = None,
        allow_cursor: bool = True,
        read_limits: SkillReadLimits | None = None,
        cursor_limits: SkillCursorLimits | None = None,
        file_system: SkillSourceFileSystem | None = None,
    ) -> SkillResponseEnvelope:
        assert isinstance(registry, SkillRegistry)
        assert isinstance(allow_cursor, bool)
        file_system = file_system or self._file_system
        if cursor_id is not None:
            return await self._read_cursor(
                registry,
                cursor_id=cursor_id,
                requested_cursor_limits=cursor_limits,
                allow_cursor=allow_cursor,
                file_system=file_system,
            )

        limits, limits_diagnostic = _effective_read_limits(
            registry,
            read_limits or self._read_limits or registry.read_limits,
        )
        if limits_diagnostic is not None:
            return _diagnostic_envelope(registry, limits_diagnostic)
        assert limits is not None
        target, diagnostics = _target_for_request(
            registry,
            skill,
            resource_id=resource_id,
            source_label=source_label,
        )
        if diagnostics:
            return _diagnostic_envelope(registry, diagnostics[0])
        assert target is not None
        return await self._read_target(
            registry,
            target=target,
            offset_bytes=0,
            read_limits=limits,
            cursor_limits=(
                cursor_limits
                or self._cursor_limits
                or _registry_cursor_limits(registry)
            ),
            allow_cursor=allow_cursor,
            file_system=file_system,
        )

    async def check(
        self,
        registry: SkillRegistry,
        skill: str,
        *,
        resource_id: str = _MAIN_RESOURCE_ID,
        source_label: str | None = None,
        file_system: SkillSourceFileSystem | None = None,
    ) -> SkillResponseEnvelope:
        assert isinstance(registry, SkillRegistry)
        file_system = file_system or self._file_system
        target, diagnostics = _target_for_request(
            registry,
            skill,
            resource_id=resource_id,
            source_label=source_label,
        )
        if diagnostics:
            return _diagnostic_envelope(registry, diagnostics[0])
        assert target is not None
        check = await _checked_resource(
            registry,
            target.resource,
            file_system=file_system,
        )
        if check.status is not SkillStatus.OK:
            return SkillResponseEnvelope(
                status=check.status,
                registry_version=registry.registry_version,
                diagnostics=check.diagnostics,
            )
        content_sha256 = target.resource.fingerprint.content_sha256
        assert content_sha256 is not None
        return SkillResponseEnvelope(
            status=SkillStatus.OK,
            registry_version=registry.registry_version,
            provenance=(
                _provenance(
                    registry,
                    target=target,
                    content_sha256_prefix=content_sha256[:_HASH_PREFIX_LENGTH],
                    truncated=False,
                ),
            ),
        )

    async def _read_cursor(
        self,
        registry: SkillRegistry,
        *,
        cursor_id: str,
        requested_cursor_limits: SkillCursorLimits | None,
        allow_cursor: bool,
        file_system: SkillSourceFileSystem,
    ) -> SkillResponseEnvelope:
        stored, diagnostic = await self._stored_cursor(
            registry,
            cursor_id,
        )
        if diagnostic is not None:
            return _diagnostic_envelope(registry, diagnostic)
        assert stored is not None
        target, diagnostics = _target_for_cursor(registry, stored.cursor)
        if diagnostics:
            await self._drop_cursor(cursor_id)
            return _diagnostic_envelope(registry, diagnostics[0])
        assert target is not None
        read_limits = SkillReadLimits(
            max_bytes_per_read=stored.cursor.limit_bytes,
            max_lines_per_read=stored.limit_lines,
        )
        limits, limits_diagnostic = _effective_read_limits(
            registry,
            read_limits,
        )
        if limits_diagnostic is not None:
            return _diagnostic_envelope(registry, limits_diagnostic)
        assert limits is not None
        return await self._read_target(
            registry,
            target=target,
            offset_bytes=stored.cursor.offset_bytes,
            read_limits=limits,
            cursor_limits=(
                requested_cursor_limits
                or self._cursor_limits
                or _registry_cursor_limits(registry)
            ),
            allow_cursor=allow_cursor,
            file_system=file_system,
        )

    async def _read_target(
        self,
        registry: SkillRegistry,
        *,
        target: _ReadTarget,
        offset_bytes: int,
        read_limits: SkillReadLimits,
        cursor_limits: SkillCursorLimits,
        allow_cursor: bool,
        file_system: SkillSourceFileSystem,
    ) -> SkillResponseEnvelope:
        check = await _checked_resource(
            registry,
            target.resource,
            file_system=file_system,
        )
        if check.status is not SkillStatus.OK:
            return SkillResponseEnvelope(
                status=check.status,
                registry_version=registry.registry_version,
                diagnostics=check.diagnostics,
            )

        content_bytes, diagnostic = await _verified_content(
            target.resource,
            file_system=file_system,
        )
        if diagnostic is not None:
            return _diagnostic_envelope(registry, diagnostic)
        assert content_bytes is not None

        window, diagnostic = _read_window(
            content_bytes,
            offset_bytes=offset_bytes,
            read_limits=read_limits,
            resource_id=target.resource.handle.resource_id,
        )
        if diagnostic is not None:
            return _diagnostic_envelope(registry, diagnostic)
        assert window is not None
        if window.truncated and not allow_cursor:
            return _diagnostic_envelope(
                registry,
                _oversized_resource_diagnostic(
                    reason=window.reason or "cursor_required",
                    resource_id=target.resource.handle.resource_id,
                    read_limits=read_limits,
                    cursor_available=False,
                ),
            )

        try:
            content = SkillResourceContent(
                handle=target.resource.handle,
                text=window.text,
                start_byte=window.start_byte,
                end_byte=window.end_byte,
                truncated=window.truncated,
            )
        except AssertionError:
            return _diagnostic_envelope(
                registry,
                _policy_diagnostic(
                    reason="unsafe_model_text",
                    resource_id=target.resource.handle.resource_id,
                ),
            )

        cursor: SkillReadCursor | None = None
        if window.truncated:
            effective_cursor_limits, cursor_diagnostic = (
                _effective_cursor_limits(
                    registry,
                    cursor_limits,
                )
            )
            if cursor_diagnostic is not None:
                return _diagnostic_envelope(registry, cursor_diagnostic)
            assert effective_cursor_limits is not None
            cursor = await self._create_cursor(
                registry,
                target=target,
                offset_bytes=window.end_byte,
                read_limits=read_limits,
                cursor_limits=effective_cursor_limits,
            )

        diagnostics: tuple[SkillDiagnosticInfo, ...] = ()
        status = SkillStatus.OK
        if window.truncated:
            diagnostics = (
                _oversized_resource_diagnostic(
                    reason=window.reason or "cursor_required",
                    resource_id=target.resource.handle.resource_id,
                    read_limits=read_limits,
                ),
            )
            status = SkillStatus.TRUNCATED

        return SkillResponseEnvelope(
            status=status,
            registry_version=registry.registry_version,
            content=content,
            diagnostics=diagnostics,
            next_cursor=cursor,
            provenance=(
                _provenance(
                    registry,
                    target=target,
                    content_sha256_prefix=sha256(content_bytes).hexdigest()[
                        :_HASH_PREFIX_LENGTH
                    ],
                    truncated=window.truncated,
                ),
            ),
        )

    async def _stored_cursor(
        self,
        registry: SkillRegistry,
        cursor_id: str,
    ) -> tuple[_StoredCursor | None, SkillDiagnosticInfo | None]:
        if not _is_opaque_id(cursor_id):
            return None, _cursor_diagnostic(reason="invalid_cursor")
        now = self._clock()
        async with self._lock:
            stored = self._cursors.get(cursor_id)
            if stored is None:
                self._prune_expired_locked(now)
                return None, _cursor_diagnostic(reason="unknown_cursor")
            if stored.cursor.registry_version != registry.registry_version:
                del self._cursors[cursor_id]
                return None, _stale_cursor_diagnostic(
                    reason="registry_version"
                )
            if now > stored.expires_at:
                del self._cursors[cursor_id]
                self._prune_expired_locked(now)
                return None, _stale_cursor_diagnostic(reason="cursor_expired")
            del self._cursors[cursor_id]
            self._prune_expired_locked(now)
            return stored, None

    async def _create_cursor(
        self,
        registry: SkillRegistry,
        *,
        target: _ReadTarget,
        offset_bytes: int,
        read_limits: SkillReadLimits,
        cursor_limits: SkillCursorLimits,
    ) -> SkillReadCursor:
        now = self._clock()
        async with self._lock:
            self._prune_expired_locked(now)
            while len(self._cursors) >= cursor_limits.max_active_cursors:
                oldest = min(
                    self._cursors.values(),
                    key=lambda value: (value.created_at, value.sequence),
                )
                del self._cursors[oldest.cursor.cursor_id]
            self._sequence += 1
            cursor_id = _cursor_id()
            while cursor_id in self._cursors:
                cursor_id = _cursor_id()
            cursor = SkillReadCursor(
                cursor_id=cursor_id,
                registry_version=registry.registry_version,
                source_label=target.resource.handle.source_label,
                skill_id=target.resource.handle.skill_id,
                resource_id=target.resource.handle.resource_id,
                offset_bytes=offset_bytes,
                limit_bytes=read_limits.max_bytes_per_read,
            )
            self._cursors[cursor_id] = _StoredCursor(
                cursor=cursor,
                created_at=now,
                expires_at=now + cursor_limits.max_cursor_age_seconds,
                limit_lines=read_limits.max_lines_per_read,
                sequence=self._sequence,
            )
            return cursor

    async def _drop_cursor(self, cursor_id: str) -> None:
        async with self._lock:
            self._cursors.pop(cursor_id, None)

    def _prune_expired_locked(
        self,
        now: float,
    ) -> None:
        expired = tuple(
            cursor_id
            for cursor_id, stored in self._cursors.items()
            if now > stored.expires_at
        )
        for cursor_id in expired:
            del self._cursors[cursor_id]


async def read_skill_registry_resource(
    registry: SkillRegistry,
    skill: str | None = None,
    *,
    resource_id: str = _MAIN_RESOURCE_ID,
    source_label: str | None = None,
    cursor_id: str | None = None,
    allow_cursor: bool = True,
    reader: SkillResourceReader | None = None,
    read_limits: SkillReadLimits | None = None,
    cursor_limits: SkillCursorLimits | None = None,
    file_system: SkillSourceFileSystem | None = None,
) -> SkillResponseEnvelope:
    assert isinstance(registry, SkillRegistry)
    owns_reader = reader is None
    if reader is None:
        reader = SkillResourceReader(file_system=file_system)
    return await reader.read(
        registry,
        skill,
        resource_id=resource_id,
        source_label=source_label,
        cursor_id=cursor_id,
        allow_cursor=allow_cursor and not owns_reader,
        read_limits=read_limits,
        cursor_limits=cursor_limits,
        file_system=file_system,
    )


async def check_skill_registry_read(
    registry: SkillRegistry,
    skill: str,
    *,
    resource_id: str = _MAIN_RESOURCE_ID,
    source_label: str | None = None,
    reader: SkillResourceReader | None = None,
    file_system: SkillSourceFileSystem | None = None,
) -> SkillResponseEnvelope:
    assert isinstance(registry, SkillRegistry)
    if reader is None:
        reader = SkillResourceReader(file_system=file_system)
    return await reader.check(
        registry,
        skill,
        resource_id=resource_id,
        source_label=source_label,
        file_system=file_system,
    )


def _target_for_request(
    registry: SkillRegistry,
    skill_ref: str | None,
    *,
    resource_id: str,
    source_label: str | None,
) -> tuple[_ReadTarget | None, tuple[SkillDiagnosticInfo, ...]]:
    if skill_ref is None or not skill_ref.strip():
        return None, (
            diagnostic_from_failure(
                SkillFailureMode.UNKNOWN_SKILL_ID,
                path="skills.read",
                details={"reason": "missing_skill"},
            ),
        )
    skill, diagnostics = _skill_for_reference(
        registry,
        skill_ref,
        source_label=source_label,
    )
    if diagnostics:
        return None, diagnostics
    assert skill is not None
    diagnostics = _skill_usability_diagnostics(registry, skill)
    if diagnostics:
        return None, diagnostics
    resource, diagnostic = _resource_for_skill(skill, resource_id)
    if diagnostic is not None:
        return None, (diagnostic,)
    assert resource is not None
    return _ReadTarget(skill=skill, resource=resource), ()


def _target_for_cursor(
    registry: SkillRegistry,
    cursor: SkillReadCursor,
) -> tuple[_ReadTarget | None, tuple[SkillDiagnosticInfo, ...]]:
    settings_diagnostic = _settings_enabled_diagnostic(registry)
    if settings_diagnostic is not None:
        return None, (settings_diagnostic,)
    matching_skills = tuple(
        skill
        for skill in registry.skills
        if skill.source_label == cursor.source_label
        and skill.skill_id == cursor.skill_id
    )
    if len(matching_skills) != 1:
        return None, (_stale_cursor_diagnostic(reason="skill_removed"),)
    skill = matching_skills[0]
    policy_diagnostic = _skill_policy_diagnostic(registry, skill)
    if policy_diagnostic is not None:
        return None, (policy_diagnostic,)
    diagnostics = _skill_usability_diagnostics(registry, skill)
    if diagnostics:
        return None, diagnostics
    resource = next(
        (
            resource
            for resource in skill.resources
            if resource.handle.resource_id == cursor.resource_id
        ),
        None,
    )
    if resource is None:
        return None, (_stale_cursor_diagnostic(reason="resource_removed"),)
    return _ReadTarget(skill=skill, resource=resource), ()


def _skill_for_reference(
    registry: SkillRegistry,
    skill_ref: str,
    *,
    source_label: str | None,
) -> tuple[SkillRegistrySkill | None, tuple[SkillDiagnosticInfo, ...]]:
    settings_diagnostic = _settings_enabled_diagnostic(registry)
    if settings_diagnostic is not None:
        return None, (settings_diagnostic,)
    source_diagnostic = _source_label_diagnostic(source_label)
    if source_diagnostic is not None:
        return None, (source_diagnostic,)
    skill_ref_diagnostic = _skill_ref_diagnostic(skill_ref)
    if skill_ref_diagnostic is not None:
        return None, (skill_ref_diagnostic,)
    normalized = normalize_skill_name(skill_ref)
    candidates = tuple(
        skill
        for skill in registry.skills
        if skill.skill_id is not None
        and (
            skill.skill_id == skill_ref
            or (normalized is not None and skill.skill_id == normalized)
        )
        and (source_label is None or skill.source_label == source_label)
    )
    if not candidates:
        return None, (
            diagnostic_from_failure(
                SkillFailureMode.UNKNOWN_SKILL_ID,
                path="skills.read",
                details={"reason": "unknown_skill"},
            ),
        )
    if len(candidates) > 1:
        return None, (
            diagnostic_from_failure(
                SkillFailureMode.AMBIGUOUS_SKILL_NAME,
                path="skills.read",
                candidates=_skill_candidates(candidates),
                details={"reason": "ambiguous_skill"},
            ),
        )
    skill = candidates[0]
    assert skill.skill_id is not None
    policy_diagnostic = _skill_policy_diagnostic(registry, skill)
    if policy_diagnostic is not None:
        return None, (policy_diagnostic,)
    return skill, ()


def _settings_enabled_diagnostic(
    registry: SkillRegistry,
) -> SkillDiagnosticInfo | None:
    if registry.settings is None or registry.settings.enabled:
        return None
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.DISABLED,
        status=SkillStatus.DISABLED,
        message="Trusted skills settings are disabled.",
        path="settings.enabled",
        hint="Do not read skills when trusted skills are off.",
        details={"reason": "settings_disabled"},
    )


def _skill_policy_diagnostic(
    registry: SkillRegistry,
    skill: SkillRegistrySkill,
) -> SkillDiagnosticInfo | None:
    settings = registry.settings
    if settings is None or skill.skill_id is None:
        return None
    if (
        settings.allowed_skill_ids
        and skill.skill_id not in settings.allowed_skill_ids
    ):
        return SkillDiagnosticInfo(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="The requested skill is not allowed by policy.",
            path="settings.skill_ids",
            hint="Read only skills allowed by trusted settings.",
            candidates=(skill.skill_id,),
            details={"reason": "skill_not_allowed"},
        )
    source_labels = {source.label for source in settings.sources}
    if source_labels and skill.source_label not in source_labels:
        return SkillDiagnosticInfo(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="The requested skill source is not allowed by policy.",
            path="settings.source_labels",
            hint="Read only skills from trusted sources.",
            candidates=(skill.source_label,),
            details={"reason": "source_not_allowed"},
        )
    source = next(
        (
            source
            for source in registry.sources
            if source.label == skill.source_label
        ),
        None,
    )
    if source is not None and source.authority.kind not in (
        settings.authority_kinds
    ):
        return SkillDiagnosticInfo(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="The requested skill authority is not allowed by policy.",
            path="settings.authority_kinds",
            hint="Read only skills from trusted authorities.",
            candidates=(source.authority.kind.value,),
            details={"reason": "authority_not_allowed"},
        )
    return None


def _skill_usability_diagnostics(
    registry: SkillRegistry,
    skill: SkillRegistrySkill,
) -> tuple[SkillDiagnosticInfo, ...]:
    if skill.usable and skill.status is SkillStatus.OK:
        return ()
    if skill.diagnostics:
        return skill.diagnostics
    if skill.metadata is not None and not skill.metadata.enabled:
        return (
            diagnostic_from_failure(
                SkillFailureMode.DISABLED_SKILL,
                path="skills.read",
                candidates=(skill.metadata.skill_id,),
            ),
        )
    if skill.status is SkillStatus.DISABLED:
        return (
            diagnostic_from_failure(
                SkillFailureMode.DISABLED_SKILL,
                path="skills.read",
                candidates=_skill_candidates((skill,)),
            ),
        )
    if skill.status is SkillStatus.MALFORMED or skill.metadata is None:
        return (
            diagnostic_from_failure(
                SkillFailureMode.MALFORMED_MANIFEST,
                path="skills.read",
                candidates=_skill_candidates((skill,)),
            ),
        )
    if registry.diagnostics:
        return registry.diagnostics
    return (
        SkillDiagnosticInfo(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="The requested skill is not usable.",
            path="skills.read",
            hint="Read only usable skills from the current registry.",
            candidates=_skill_candidates((skill,)),
            details={"reason": "skill_not_usable"},
        ),
    )


def _resource_for_skill(
    skill: SkillRegistrySkill,
    resource_id: str,
) -> tuple[SkillRegisteredResource | None, SkillDiagnosticInfo | None]:
    diagnostic = _resource_id_diagnostic(resource_id)
    if diagnostic is not None:
        return None, diagnostic
    for resource in skill.resources:
        if resource.handle.resource_id == resource_id:
            return resource, None
    return None, diagnostic_from_failure(
        SkillFailureMode.RESOURCE_MISSING,
        path="resource.lookup",
        details={
            "reason": "undeclared_resource",
            "resource_id": _safe_resource_detail(resource_id),
        },
    )


async def _checked_resource(
    registry: SkillRegistry,
    resource: SkillRegisteredResource,
    *,
    file_system: SkillSourceFileSystem,
) -> SkillRegistryResourceCheck:
    path_diagnostic = await _runtime_path_diagnostic(
        resource,
        file_system=file_system,
    )
    if path_diagnostic is not None:
        return SkillRegistryResourceCheck(
            handle=resource.handle,
            stored_fingerprint=resource.fingerprint,
            diagnostics=(path_diagnostic,),
        )
    return await registry.check_resource(
        resource.handle,
        file_system=file_system,
        read_limits=registry.read_limits,
    )


async def _runtime_path_diagnostic(
    resource: SkillRegisteredResource,
    *,
    file_system: SkillSourceFileSystem,
) -> SkillDiagnosticInfo | None:
    resource_id = resource.handle.resource_id
    try:
        lstat = await file_system.lstat_path(resource.path)
    except FileNotFoundError:
        return diagnostic_from_failure(
            SkillFailureMode.RESOURCE_MISSING,
            path="resource.lookup",
            details={"resource_id": resource_id},
        )
    except (AttributeError, OSError):
        return _runtime_unavailable_diagnostic(resource_id=resource_id)
    if S_ISLNK(lstat.st_mode):
        return _outside_root_diagnostic(
            reason="symlink_escape",
            resource_id=resource_id,
        )
    try:
        resolved = await file_system.resolve_path(resource.path)
    except FileNotFoundError:
        return diagnostic_from_failure(
            SkillFailureMode.RESOURCE_MISSING,
            path="resource.lookup",
            details={"resource_id": resource_id},
        )
    except (AttributeError, OSError, RuntimeError, ValueError):
        return _runtime_unavailable_diagnostic(resource_id=resource_id)
    if resolved != resource.path:
        return _outside_root_diagnostic(
            reason="path_escape",
            resource_id=resource_id,
        )
    try:
        stat = await file_system.stat_path(resource.path)
    except FileNotFoundError:
        return diagnostic_from_failure(
            SkillFailureMode.RESOURCE_MISSING,
            path="resource.lookup",
            details={"resource_id": resource_id},
        )
    except OSError:
        return _runtime_unavailable_diagnostic(resource_id=resource_id)
    if not S_ISREG(stat.st_mode):
        return _policy_diagnostic(
            reason="special_file",
            resource_id=resource_id,
        )
    return None


async def _verified_content(
    resource: SkillRegisteredResource,
    *,
    file_system: SkillSourceFileSystem,
) -> tuple[bytes | None, SkillDiagnosticInfo | None]:
    resource_id = resource.handle.resource_id
    limit = resource.fingerprint.size_bytes + 1
    try:
        content = await file_system.read_bytes(resource.path, limit)
    except FileNotFoundError:
        return None, diagnostic_from_failure(
            SkillFailureMode.RESOURCE_MISSING,
            path="resource.lookup",
            details={"resource_id": resource_id},
        )
    except OSError:
        return None, _runtime_unavailable_diagnostic(resource_id=resource_id)
    if b"\x00" in content:
        return None, _binary_resource_diagnostic(
            reason="nul_byte",
            resource_id=resource_id,
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None, _binary_resource_diagnostic(
            reason="non_utf8",
            resource_id=resource_id,
        )
    if len(content) != resource.fingerprint.size_bytes:
        return None, _stale_resource_diagnostic(
            reason="size_changed",
            resource_id=resource_id,
        )
    stored_hash = resource.fingerprint.content_sha256
    if stored_hash is None or sha256(content).hexdigest() != stored_hash:
        return None, _stale_resource_diagnostic(
            reason="content_changed",
            resource_id=resource_id,
        )
    try:
        model_dict({"text": text})
    except AssertionError:
        return None, _policy_diagnostic(
            reason="unsafe_model_text",
            resource_id=resource_id,
        )
    return content, None


def _read_window(
    content: bytes,
    *,
    offset_bytes: int,
    read_limits: SkillReadLimits,
    resource_id: str,
) -> tuple[_ReadWindow | None, SkillDiagnosticInfo | None]:
    if offset_bytes > len(content):
        return None, _cursor_diagnostic(
            reason="offset_out_of_bounds",
            resource_id=resource_id,
        )
    if offset_bytes == len(content):
        return (
            _ReadWindow(
                text="",
                start_byte=offset_bytes,
                end_byte=offset_bytes,
                truncated=False,
            ),
            None,
        )

    end_byte = min(
        len(content),
        offset_bytes + read_limits.max_bytes_per_read,
    )
    text, safe_end = _decode_prefix(content, offset_bytes, end_byte)
    if text is None:
        return None, _oversized_resource_diagnostic(
            reason="utf8_character_exceeds_byte_limit",
            resource_id=resource_id,
            read_limits=read_limits,
            cursor_available=False,
        )
    reason = "max_bytes_per_read" if safe_end < len(content) else None
    limited_text, line_truncated = _line_limited_text(
        text,
        read_limits.max_lines_per_read,
    )
    if line_truncated:
        text = limited_text
        safe_end = offset_bytes + len(text.encode("utf-8"))
        reason = "max_lines_per_read"
    truncated = safe_end < len(content)
    return (
        _ReadWindow(
            text=text,
            start_byte=offset_bytes,
            end_byte=safe_end,
            truncated=truncated,
            reason=reason,
        ),
        None,
    )


def _decode_prefix(
    content: bytes,
    offset_bytes: int,
    end_byte: int,
) -> tuple[str | None, int]:
    safe_end = end_byte
    while safe_end > offset_bytes:
        try:
            return content[offset_bytes:safe_end].decode("utf-8"), safe_end
        except UnicodeDecodeError:
            safe_end -= 1
    return None, offset_bytes


def _line_limited_text(text: str, limit: int) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    if len(lines) <= limit:
        return text, False
    return "".join(lines[:limit]), True


def _effective_read_limits(
    registry: SkillRegistry,
    requested: SkillReadLimits,
) -> tuple[SkillReadLimits | None, SkillDiagnosticInfo | None]:
    assert isinstance(requested, SkillReadLimits)
    if registry.read_limits.allows(requested):
        return requested, None
    return None, SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="Requested skill read limits exceed trusted limits.",
        path="settings.read_limits",
        hint="Use read limits no wider than the current registry allows.",
        details={"reason": "read_limits_exceeded"},
    )


def _effective_cursor_limits(
    registry: SkillRegistry,
    requested: SkillCursorLimits,
) -> tuple[SkillCursorLimits | None, SkillDiagnosticInfo | None]:
    assert isinstance(requested, SkillCursorLimits)
    trusted = _registry_cursor_limits(registry)
    if trusted.allows(requested):
        return requested, None
    return None, SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="Requested skill cursor limits exceed trusted limits.",
        path="settings.cursor_limits",
        hint="Use cursor limits no wider than the current registry allows.",
        details={"reason": "cursor_limits_exceeded"},
    )


def _registry_cursor_limits(registry: SkillRegistry) -> SkillCursorLimits:
    if registry.settings is not None:
        return registry.settings.cursor_limits
    return SkillCursorLimits()


def _provenance(
    registry: SkillRegistry,
    *,
    target: _ReadTarget,
    content_sha256_prefix: str | None,
    truncated: bool,
) -> SkillProvenance:
    source = next(
        source
        for source in registry.sources
        if source.label == target.resource.handle.source_label
    )
    return SkillProvenance(
        registry_version=registry.registry_version,
        source_label=target.resource.handle.source_label,
        skill_id=target.resource.handle.skill_id,
        resource_id=target.resource.handle.resource_id,
        authority=source.authority.kind,
        content_sha256_prefix=content_sha256_prefix,
        truncated=truncated,
        declared_follow_up_resources=tuple(
            resource.handle.resource_id
            for resource in target.skill.resources
            if resource.handle.resource_id
            != target.resource.handle.resource_id
        ),
    )


def _cursor_id() -> str:
    return f"skill-cursor:{token_hex(16)}"


def _source_label_diagnostic(
    source_label: str | None,
) -> SkillDiagnosticInfo | None:
    if source_label is None:
        return None
    if fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", source_label):
        return None
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.NOT_FOUND,
        status=SkillStatus.NOT_FOUND,
        message="The requested skill source label is unknown.",
        path="skills.read",
        hint="Use source labels returned by skills.list or skills.match.",
        details={"reason": "invalid_source_label"},
    )


def _skill_ref_diagnostic(skill_ref: str) -> SkillDiagnosticInfo | None:
    reason = skill_name_denial_reason(skill_ref)
    if reason is None:
        return None
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="The requested skill reference is not authorized.",
        path="skills.read",
        hint="Use a logical skill ID or normalized skill name.",
        details={"reason": reason},
    )


def _resource_id_diagnostic(resource_id: str) -> SkillDiagnosticInfo | None:
    if resource_id == _MAIN_RESOURCE_ID:
        return None
    reason = skill_model_handle_denial_reason(resource_id)
    if reason is None:
        return None
    if reason == "traversal":
        return _outside_root_diagnostic(
            reason=reason,
            resource_id="resource/unsafe",
        )
    return _policy_diagnostic(
        reason=reason,
        resource_id="resource/unsafe",
    )


def _safe_resource_detail(resource_id: str) -> str:
    try:
        model_dict({"resource_id": resource_id})
    except AssertionError:
        return "resource/unsafe"
    return resource_id


def _diagnostic_envelope(
    registry: SkillRegistry,
    diagnostic: SkillDiagnosticInfo,
) -> SkillResponseEnvelope:
    return SkillResponseEnvelope(
        status=diagnostic.status,
        registry_version=registry.registry_version,
        diagnostics=(diagnostic,),
    )


def _skill_candidates(
    skills: tuple[SkillRegistrySkill, ...],
) -> tuple[str, ...]:
    skill_ids = tuple(
        skill.skill_id for skill in skills if skill.skill_id is not None
    )
    if len(set(skill_ids)) == len(skill_ids):
        return tuple(sorted(skill_ids))
    return tuple(
        f"{skill.source_label}.{skill.skill_id}.{index}"
        for index, skill in enumerate(
            sorted(
                (skill for skill in skills if skill.skill_id is not None),
                key=lambda value: (
                    value.source_label,
                    value.skill_id or "",
                    value.manifest_resource_id,
                ),
            ),
            start=1,
        )
    )


def _is_opaque_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and fullmatch(r"[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*", value) is not None
    )


def _cursor_diagnostic(
    *,
    reason: str,
    resource_id: str | None = None,
) -> SkillDiagnosticInfo:
    details: dict[str, SkillModelValue] = {"reason": reason}
    if resource_id is not None:
        details["resource_id"] = resource_id
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.NOT_FOUND,
        status=SkillStatus.NOT_FOUND,
        message="The requested skill read cursor is invalid.",
        path="resource.cursor",
        hint="Restart the resource read from the registry.",
        details=details,
    )


def _stale_cursor_diagnostic(*, reason: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_STALE,
        status=SkillStatus.STALE,
        message="The skill read cursor is stale.",
        path="resource.cursor",
        hint="Restart the resource read from the current registry.",
        details={"reason": reason},
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
    *,
    resource_id: str,
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


def _outside_root_diagnostic(
    *,
    reason: str,
    resource_id: str,
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
        status=SkillStatus.POLICY_DENIED,
        message="The skill resource is outside its authorized root.",
        path="resource.policy",
        hint="Reject traversal and use logical resource IDs only.",
        details={"reason": reason, "resource_id": resource_id},
    )


def _policy_diagnostic(
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


def _binary_resource_diagnostic(
    *,
    reason: str,
    resource_id: str,
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.BINARY_RESOURCE,
        status=SkillStatus.UNAVAILABLE,
        message="The skill resource is binary or non-UTF-8.",
        path="resource.content",
        hint="Expose only UTF-8 Markdown skill resources.",
        details={"reason": reason, "resource_id": resource_id},
    )


def _oversized_resource_diagnostic(
    *,
    reason: str,
    resource_id: str,
    read_limits: SkillReadLimits,
    cursor_available: bool = True,
) -> SkillDiagnosticInfo:
    hint = (
        "Continue with the returned bounded read cursor."
        if cursor_available
        else "Restart the read with cursor support or wider trusted limits."
    )
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OVERSIZED,
        status=SkillStatus.TRUNCATED,
        message="The skill resource is too large for one read.",
        path="resource.bounds",
        hint=hint,
        details={
            "reason": reason,
            "resource_id": resource_id,
            "max_bytes_per_read": read_limits.max_bytes_per_read,
            "max_lines_per_read": read_limits.max_lines_per_read,
        },
    )
