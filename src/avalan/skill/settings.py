from .contract import SkillDiagnosticCode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    model_dict,
)

from dataclasses import dataclass, field, replace
from enum import StrEnum
from re import fullmatch


class SkillSettingsSurface(StrEnum):
    OPERATOR = "operator"
    AGENT = "agent"
    FLOW = "flow"
    TASK = "task"
    REQUEST = "request"
    WORKER_ENVELOPE = "worker_envelope"


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillReadLimits:
    max_bytes_per_read: int = 65_536
    max_lines_per_read: int = 2_000

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_bytes_per_read, "max_bytes_per_read")
        _assert_positive_int(self.max_lines_per_read, "max_lines_per_read")

    def allows(self, requested: "SkillReadLimits") -> bool:
        assert isinstance(requested, SkillReadLimits)
        return (
            requested.max_bytes_per_read <= self.max_bytes_per_read
            and requested.max_lines_per_read <= self.max_lines_per_read
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "max_bytes_per_read": self.max_bytes_per_read,
                "max_lines_per_read": self.max_lines_per_read,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillIndexLimits:
    max_skills: int = 256
    max_resources_per_skill: int = 32
    max_indexed_bytes: int = 4_194_304

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_skills, "max_skills")
        _assert_positive_int(
            self.max_resources_per_skill, "max_resources_per_skill"
        )
        _assert_positive_int(self.max_indexed_bytes, "max_indexed_bytes")

    def allows(self, requested: "SkillIndexLimits") -> bool:
        assert isinstance(requested, SkillIndexLimits)
        return (
            requested.max_skills <= self.max_skills
            and requested.max_resources_per_skill
            <= self.max_resources_per_skill
            and requested.max_indexed_bytes <= self.max_indexed_bytes
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "max_skills": self.max_skills,
                "max_resources_per_skill": self.max_resources_per_skill,
                "max_indexed_bytes": self.max_indexed_bytes,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSourceLimits:
    max_sources: int = 16
    max_resources_per_source: int = 512
    max_source_depth: int = 8

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_sources, "max_sources")
        _assert_positive_int(
            self.max_resources_per_source, "max_resources_per_source"
        )
        _assert_positive_int(self.max_source_depth, "max_source_depth")

    def allows(self, requested: "SkillSourceLimits") -> bool:
        assert isinstance(requested, SkillSourceLimits)
        return (
            requested.max_sources <= self.max_sources
            and requested.max_resources_per_source
            <= self.max_resources_per_source
            and requested.max_source_depth <= self.max_source_depth
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "max_sources": self.max_sources,
                "max_resources_per_source": self.max_resources_per_source,
                "max_source_depth": self.max_source_depth,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillCursorLimits:
    max_active_cursors: int = 64
    max_cursor_age_seconds: int = 900

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_active_cursors, "max_active_cursors")
        _assert_positive_int(
            self.max_cursor_age_seconds, "max_cursor_age_seconds"
        )

    def allows(self, requested: "SkillCursorLimits") -> bool:
        assert isinstance(requested, SkillCursorLimits)
        return (
            requested.max_active_cursors <= self.max_active_cursors
            and requested.max_cursor_age_seconds <= self.max_cursor_age_seconds
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "max_active_cursors": self.max_active_cursors,
                "max_cursor_age_seconds": self.max_cursor_age_seconds,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillPrivacySettings:
    include_source_labels: bool = True
    include_authority: bool = True
    include_diagnostic_paths: bool = True
    redact_host_paths: bool = True

    def __post_init__(self) -> None:
        _assert_bool(self.include_source_labels, "include_source_labels")
        _assert_bool(self.include_authority, "include_authority")
        _assert_bool(self.include_diagnostic_paths, "include_diagnostic_paths")
        _assert_bool(self.redact_host_paths, "redact_host_paths")

    def allows(self, requested: "SkillPrivacySettings") -> bool:
        assert isinstance(requested, SkillPrivacySettings)
        return (
            _allows_exposure(
                self.include_source_labels, requested.include_source_labels
            )
            and _allows_exposure(
                self.include_authority, requested.include_authority
            )
            and _allows_exposure(
                self.include_diagnostic_paths,
                requested.include_diagnostic_paths,
            )
            and _allows_redaction(
                self.redact_host_paths, requested.redact_host_paths
            )
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "include_source_labels": self.include_source_labels,
                "include_authority": self.include_authority,
                "include_diagnostic_paths": self.include_diagnostic_paths,
                "redact_host_paths": self.redact_host_paths,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillObservabilitySettings:
    enabled: bool = True
    emit_events: bool = True
    include_diagnostics: bool = True
    include_byte_counts: bool = False

    def __post_init__(self) -> None:
        _assert_bool(self.enabled, "enabled")
        _assert_bool(self.emit_events, "emit_events")
        _assert_bool(self.include_diagnostics, "include_diagnostics")
        _assert_bool(self.include_byte_counts, "include_byte_counts")

    def allows(self, requested: "SkillObservabilitySettings") -> bool:
        assert isinstance(requested, SkillObservabilitySettings)
        return (
            _allows_exposure(self.enabled, requested.enabled)
            and _allows_exposure(self.emit_events, requested.emit_events)
            and _allows_exposure(
                self.include_diagnostics, requested.include_diagnostics
            )
            and _allows_exposure(
                self.include_byte_counts, requested.include_byte_counts
            )
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "enabled": self.enabled,
                "emit_events": self.emit_events,
                "include_diagnostics": self.include_diagnostics,
                "include_byte_counts": self.include_byte_counts,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrustedSkillSettings:
    enabled: bool = True
    authority_kinds: tuple[SkillSourceAuthorityKind, ...] = (
        SkillSourceAuthorityKind.BUNDLED,
        SkillSourceAuthorityKind.WORKSPACE,
        SkillSourceAuthorityKind.USER_LOCAL,
        SkillSourceAuthorityKind.PLUGIN_PROVIDED,
        SkillSourceAuthorityKind.PREINSTALLED_REMOTE,
    )
    sources: tuple[SkillSourceConfig, ...] = ()
    allowed_skill_ids: tuple[str, ...] = ()
    read_limits: SkillReadLimits = field(default_factory=SkillReadLimits)
    index_limits: SkillIndexLimits = field(default_factory=SkillIndexLimits)
    source_limits: SkillSourceLimits = field(default_factory=SkillSourceLimits)
    cursor_limits: SkillCursorLimits = field(default_factory=SkillCursorLimits)
    privacy: SkillPrivacySettings = field(default_factory=SkillPrivacySettings)
    observability: SkillObservabilitySettings = field(
        default_factory=SkillObservabilitySettings
    )

    def __post_init__(self) -> None:
        _assert_bool(self.enabled, "enabled")
        _assert_authority_kind_tuple(self.authority_kinds)
        _assert_source_tuple(self.sources)
        _assert_unique_source_labels(self.sources)
        for source in self.sources:
            assert (
                source.authority.kind in self.authority_kinds
            ), "source authority must be trusted"
        _assert_logical_id_tuple(self.allowed_skill_ids, "allowed_skill_ids")
        assert isinstance(self.read_limits, SkillReadLimits)
        assert isinstance(self.index_limits, SkillIndexLimits)
        assert isinstance(self.source_limits, SkillSourceLimits)
        assert len(self.sources) <= self.source_limits.max_sources
        assert isinstance(self.cursor_limits, SkillCursorLimits)
        assert isinstance(self.privacy, SkillPrivacySettings)
        assert isinstance(self.observability, SkillObservabilitySettings)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "enabled": self.enabled,
            "authority_kinds": tuple(
                authority_kind.value for authority_kind in self.authority_kinds
            ),
            "sources": tuple(
                source.as_model_dict() for source in self.sources
            ),
            "read_limits": self.read_limits.as_model_dict(),
            "index_limits": self.index_limits.as_model_dict(),
            "source_limits": self.source_limits.as_model_dict(),
            "cursor_limits": self.cursor_limits.as_model_dict(),
            "privacy": self.privacy.as_model_dict(),
            "observability": self.observability.as_model_dict(),
        }
        if self.allowed_skill_ids:
            value["allowed_skill_ids"] = self.allowed_skill_ids
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class UntrustedSkillSettings:
    surface: SkillSettingsSurface
    enabled: bool | None = None
    authority_kinds: tuple[SkillSourceAuthorityKind, ...] = ()
    source_labels: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    read_limits: SkillReadLimits | None = None
    index_limits: SkillIndexLimits | None = None
    source_limits: SkillSourceLimits | None = None
    cursor_limits: SkillCursorLimits | None = None
    privacy: SkillPrivacySettings | None = None
    observability: SkillObservabilitySettings | None = None
    sources: tuple[SkillSourceConfig, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.surface, SkillSettingsSurface)
        assert self.surface is not SkillSettingsSurface.OPERATOR
        if self.enabled is not None:
            _assert_bool(self.enabled, "enabled")
        _assert_authority_kind_tuple(self.authority_kinds, allow_empty=True)
        _assert_source_label_tuple(self.source_labels, "source_labels")
        _assert_logical_id_tuple(self.skill_ids, "skill_ids")
        _assert_optional_limit(self.read_limits, SkillReadLimits)
        _assert_optional_limit(self.index_limits, SkillIndexLimits)
        _assert_optional_limit(self.source_limits, SkillSourceLimits)
        _assert_optional_limit(self.cursor_limits, SkillCursorLimits)
        if self.privacy is not None:
            assert isinstance(self.privacy, SkillPrivacySettings)
        if self.observability is not None:
            assert isinstance(self.observability, SkillObservabilitySettings)
        _assert_source_tuple(self.sources)
        _assert_unique_source_labels(self.sources)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSettingsMergeResult:
    settings: TrustedSkillSettings
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.settings, TrustedSkillSettings)
        _assert_diagnostic_tuple(self.diagnostics)

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        return SkillStatus.OK

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "status": self.status.value,
                "settings": self.settings.as_model_dict(),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


def merge_skill_settings(
    trusted: TrustedSkillSettings,
    override: UntrustedSkillSettings | None = None,
) -> SkillSettingsMergeResult:
    assert isinstance(trusted, TrustedSkillSettings)
    if override is None:
        return SkillSettingsMergeResult(settings=trusted)
    assert isinstance(override, UntrustedSkillSettings)

    settings = trusted
    diagnostics: list[SkillDiagnosticInfo] = []

    if override.enabled is False:
        settings = replace(settings, enabled=False)
    elif override.enabled is True and not trusted.enabled:
        diagnostics.append(_policy_diagnostic("settings.enabled"))

    if override.sources:
        diagnostics.append(_policy_diagnostic("settings.sources"))

    if override.authority_kinds:
        if _authority_kinds_allowed(
            settings.authority_kinds, override.authority_kinds
        ):
            settings = replace(
                settings,
                authority_kinds=override.authority_kinds,
                sources=tuple(
                    source
                    for source in settings.sources
                    if source.authority.kind in override.authority_kinds
                ),
            )
        else:
            diagnostics.append(_policy_diagnostic("settings.authority_kinds"))

    if override.source_labels:
        filtered_sources, diagnostic = _filter_sources(
            settings.sources, override.source_labels
        )
        if diagnostic is None:
            settings = replace(settings, sources=filtered_sources)
        else:
            diagnostics.append(diagnostic)

    if override.skill_ids:
        if _skill_ids_allowed(settings.allowed_skill_ids, override.skill_ids):
            settings = replace(settings, allowed_skill_ids=override.skill_ids)
        else:
            diagnostics.append(_policy_diagnostic("settings.skill_ids"))

    if override.read_limits is not None:
        if settings.read_limits.allows(override.read_limits):
            settings = replace(settings, read_limits=override.read_limits)
        else:
            diagnostics.append(_policy_diagnostic("settings.read_limits"))

    if override.index_limits is not None:
        if settings.index_limits.allows(override.index_limits):
            settings = replace(settings, index_limits=override.index_limits)
        else:
            diagnostics.append(_policy_diagnostic("settings.index_limits"))

    if override.source_limits is not None:
        if (
            settings.source_limits.allows(override.source_limits)
            and len(settings.sources) <= override.source_limits.max_sources
        ):
            settings = replace(settings, source_limits=override.source_limits)
        else:
            diagnostics.append(_policy_diagnostic("settings.source_limits"))

    if override.cursor_limits is not None:
        if settings.cursor_limits.allows(override.cursor_limits):
            settings = replace(settings, cursor_limits=override.cursor_limits)
        else:
            diagnostics.append(_policy_diagnostic("settings.cursor_limits"))

    if override.privacy is not None:
        if settings.privacy.allows(override.privacy):
            settings = replace(settings, privacy=override.privacy)
        else:
            diagnostics.append(_policy_diagnostic("settings.privacy"))

    if override.observability is not None:
        if settings.observability.allows(override.observability):
            settings = replace(settings, observability=override.observability)
        else:
            diagnostics.append(_policy_diagnostic("settings.observability"))

    return SkillSettingsMergeResult(
        settings=settings,
        diagnostics=tuple(diagnostics),
    )


def _filter_sources(
    sources: tuple[SkillSourceConfig, ...],
    labels: tuple[str, ...],
) -> tuple[tuple[SkillSourceConfig, ...], SkillDiagnosticInfo | None]:
    source_by_label = {source.label: source for source in sources}
    missing = tuple(label for label in labels if label not in source_by_label)
    if missing:
        return (), _not_found_diagnostic(
            "settings.source_labels", candidates=missing
        )
    return tuple(source_by_label[label] for label in labels), None


def _skill_ids_allowed(
    trusted_skill_ids: tuple[str, ...], requested_skill_ids: tuple[str, ...]
) -> bool:
    if not trusted_skill_ids:
        return True
    return set(requested_skill_ids).issubset(set(trusted_skill_ids))


def _authority_kinds_allowed(
    trusted_authority_kinds: tuple[SkillSourceAuthorityKind, ...],
    requested_authority_kinds: tuple[SkillSourceAuthorityKind, ...],
) -> bool:
    return set(requested_authority_kinds).issubset(
        set(trusted_authority_kinds)
    )


def _policy_diagnostic(path: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="Untrusted skills settings cannot widen trusted settings.",
        path=path,
        hint=(
            "Define sources, authorities, limits, privacy, and observability "
            "in trusted operator configuration."
        ),
    )


def _not_found_diagnostic(
    path: str, *, candidates: tuple[str, ...]
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.NOT_FOUND,
        status=SkillStatus.NOT_FOUND,
        message="The requested skills setting references unknown values.",
        path=path,
        hint="Use only source labels present in trusted settings.",
        candidates=tuple(sorted(candidates)),
    )


def _assert_positive_int(value: int, field_name: str) -> None:
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_bool(value: bool, field_name: str) -> None:
    assert isinstance(value, bool), f"{field_name} must be a bool"


def _assert_source_tuple(values: tuple[SkillSourceConfig, ...]) -> None:
    assert isinstance(values, tuple), "sources must be a tuple"
    for value in values:
        assert isinstance(value, SkillSourceConfig)


def _assert_authority_kind_tuple(
    values: tuple[SkillSourceAuthorityKind, ...],
    *,
    allow_empty: bool = False,
) -> None:
    assert isinstance(values, tuple), "authority_kinds must be a tuple"
    if not allow_empty:
        assert values, "authority_kinds must be non-empty"
    for value in values:
        assert isinstance(value, SkillSourceAuthorityKind)
    assert len(set(values)) == len(values), "authority_kinds must be unique"


def _assert_unique_source_labels(
    values: tuple[SkillSourceConfig, ...],
) -> None:
    labels = tuple(value.label for value in values)
    assert len(set(labels)) == len(labels), "source labels must be unique"


def _assert_logical_id_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_logical_id(value, field_name)
    assert len(set(values)) == len(values), f"{field_name} must be unique"


def _assert_source_label_tuple(
    values: tuple[str, ...], field_name: str
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_source_label(value, field_name)
    assert len(set(values)) == len(values), f"{field_name} must be unique"


def _assert_source_label(value: str, field_name: str) -> None:
    _assert_logical_id(value, field_name)


def _assert_logical_id(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value, f"{field_name} must be non-empty"
    assert (
        fullmatch(
            r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*",
            value,
        )
        is not None
    ), f"{field_name} must be a logical ID"


def _assert_optional_limit(value: object, expected_type: type[object]) -> None:
    if value is not None:
        assert isinstance(value, expected_type)


def _assert_diagnostic_tuple(values: tuple[SkillDiagnosticInfo, ...]) -> None:
    assert isinstance(values, tuple), "diagnostics must be a tuple"
    for value in values:
        assert isinstance(value, SkillDiagnosticInfo)


def _allows_exposure(trusted: bool, requested: bool) -> bool:
    return trusted or not requested


def _allows_redaction(trusted: bool, requested: bool) -> bool:
    return requested or not trusted
