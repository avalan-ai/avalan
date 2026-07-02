from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_model_safe_text_tuple,
    assert_tuple_items,
    assert_unique_sequence,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .contract import SkillDiagnosticCode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillSourceAuthority,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    model_dict,
)

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from hashlib import sha256
from json import dumps
from pathlib import Path, PurePosixPath
from re import fullmatch
from types import MappingProxyType
from typing import cast

SKILL_SETTINGS_POLICY_VERSION = "skills.settings.phase9"
_ROOT_EXPLICIT_FIELDS = "__root__"


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
    max_files_per_source: int = 1024
    max_directory_entries_per_source: int = 4096

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_sources, "max_sources")
        _assert_positive_int(
            self.max_resources_per_source, "max_resources_per_source"
        )
        _assert_positive_int(self.max_source_depth, "max_source_depth")
        _assert_positive_int(self.max_files_per_source, "max_files_per_source")
        _assert_positive_int(
            self.max_directory_entries_per_source,
            "max_directory_entries_per_source",
        )

    def allows(self, requested: "SkillSourceLimits") -> bool:
        assert isinstance(requested, SkillSourceLimits)
        return (
            requested.max_sources <= self.max_sources
            and requested.max_resources_per_source
            <= self.max_resources_per_source
            and requested.max_source_depth <= self.max_source_depth
            and requested.max_files_per_source <= self.max_files_per_source
            and requested.max_directory_entries_per_source
            <= self.max_directory_entries_per_source
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "max_sources": self.max_sources,
                "max_resources_per_source": self.max_resources_per_source,
                "max_source_depth": self.max_source_depth,
                "max_files_per_source": self.max_files_per_source,
                "max_directory_entries_per_source": (
                    self.max_directory_entries_per_source
                ),
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
    audit_fail_closed: bool = False

    def __post_init__(self) -> None:
        _assert_bool(self.enabled, "enabled")
        _assert_bool(self.emit_events, "emit_events")
        _assert_bool(self.include_diagnostics, "include_diagnostics")
        _assert_bool(self.include_byte_counts, "include_byte_counts")
        _assert_bool(self.audit_fail_closed, "audit_fail_closed")

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
            and _allows_enforcement(
                self.audit_fail_closed, requested.audit_fail_closed
            )
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "enabled": self.enabled,
                "emit_events": self.emit_events,
                "include_diagnostics": self.include_diagnostics,
                "include_byte_counts": self.include_byte_counts,
                "audit_fail_closed": self.audit_fail_closed,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillBootstrapPromptSettings:
    include_tool_summary: bool = True
    include_discovery_guidance: bool = True
    include_read_guidance: bool = True
    include_check_guidance: bool = True
    include_behavior_guidance: bool = True
    additional_instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_bool(self.include_tool_summary, "include_tool_summary")
        _assert_bool(
            self.include_discovery_guidance,
            "include_discovery_guidance",
        )
        _assert_bool(self.include_read_guidance, "include_read_guidance")
        _assert_bool(self.include_check_guidance, "include_check_guidance")
        _assert_bool(
            self.include_behavior_guidance,
            "include_behavior_guidance",
        )
        assert_non_empty_model_safe_text_tuple(
            self.additional_instructions,
            "additional_instructions",
            unique=True,
        )

    def allows(self, requested: "SkillBootstrapPromptSettings") -> bool:
        assert isinstance(requested, SkillBootstrapPromptSettings)
        if not (
            _allows_exposure(
                self.include_tool_summary,
                requested.include_tool_summary,
            )
            and _allows_exposure(
                self.include_discovery_guidance,
                requested.include_discovery_guidance,
            )
            and _allows_exposure(
                self.include_read_guidance,
                requested.include_read_guidance,
            )
            and _allows_exposure(
                self.include_check_guidance,
                requested.include_check_guidance,
            )
            and _allows_exposure(
                self.include_behavior_guidance,
                requested.include_behavior_guidance,
            )
        ):
            return False
        return _sequence_contains_ordered_subset(
            self.additional_instructions,
            requested.additional_instructions,
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "include_tool_summary": self.include_tool_summary,
            "include_discovery_guidance": self.include_discovery_guidance,
            "include_read_guidance": self.include_read_guidance,
            "include_check_guidance": self.include_check_guidance,
            "include_behavior_guidance": self.include_behavior_guidance,
        }
        if self.additional_instructions:
            value["additional_instruction_count"] = len(
                self.additional_instructions
            )
            value["additional_instructions_sha256"] = _settings_fingerprint(
                {"additional_instructions": self.additional_instructions}
            )
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrustedSkillSettings:
    enabled: bool = True
    bootstrap_enabled: bool = True
    bootstrap_prompt: SkillBootstrapPromptSettings = field(
        default_factory=SkillBootstrapPromptSettings
    )
    authority_kinds: tuple[SkillSourceAuthorityKind, ...] = (
        SkillSourceAuthorityKind.BUNDLED,
        SkillSourceAuthorityKind.WORKSPACE,
        SkillSourceAuthorityKind.USER_LOCAL,
        SkillSourceAuthorityKind.PLUGIN_PROVIDED,
        SkillSourceAuthorityKind.PREINSTALLED_REMOTE,
    )
    sources: tuple[SkillSourceConfig, ...] = ()
    sources_explicit: bool = False
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
        _assert_bool(self.bootstrap_enabled, "bootstrap_enabled")
        assert isinstance(self.bootstrap_prompt, SkillBootstrapPromptSettings)
        _assert_authority_kind_tuple(self.authority_kinds)
        _assert_source_tuple(self.sources)
        _assert_bool(self.sources_explicit, "sources_explicit")
        object.__setattr__(
            self,
            "sources_explicit",
            bool(self.sources_explicit or self.sources),
        )
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
            "bootstrap_enabled": self.bootstrap_enabled,
            "bootstrap_prompt": self.bootstrap_prompt.as_model_dict(),
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
    bootstrap_enabled: bool | None = None
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
    explicit_fields: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        assert isinstance(self.surface, SkillSettingsSurface)
        assert self.surface is not SkillSettingsSurface.OPERATOR
        if self.enabled is not None:
            _assert_bool(self.enabled, "enabled")
        if self.bootstrap_enabled is not None:
            _assert_bool(self.bootstrap_enabled, "bootstrap_enabled")
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
        _assert_explicit_fields(self.explicit_fields)
        object.__setattr__(
            self,
            "explicit_fields",
            MappingProxyType(dict(self.explicit_fields)),
        )


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

    if override.bootstrap_enabled is False:
        settings = replace(settings, bootstrap_enabled=False)
    elif override.bootstrap_enabled is True and not trusted.bootstrap_enabled:
        diagnostics.append(_policy_diagnostic("settings.bootstrap_enabled"))

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


def parse_untrusted_skill_settings_config(
    skills_config: Mapping[str, object],
    *,
    trusted: TrustedSkillSettings,
    surface: SkillSettingsSurface,
    section: str,
) -> UntrustedSkillSettings:
    assert isinstance(
        skills_config, Mapping
    ), f"{section} section must be a mapping"
    assert isinstance(trusted, TrustedSkillSettings)
    assert isinstance(surface, SkillSettingsSurface)
    assert surface is not SkillSettingsSurface.OPERATOR
    assert isinstance(section, str) and section.strip()
    supported_keys = {
        "enabled",
        "bootstrap",
        "authority_kinds",
        "source_labels",
        "skill_ids",
        "read_limits",
        "index_limits",
        "source_limits",
        "cursor_limits",
        "privacy",
        "observability",
    }
    assert "source" not in skills_config, (
        f"{section} cannot define sources; configure trusted sources through "
        "operator settings"
    )
    assert "sources" not in skills_config, (
        f"{section} cannot define sources; configure trusted sources through "
        "operator settings"
    )
    unknown_keys = sorted(set(skills_config) - supported_keys)
    assert not unknown_keys, f"{section} has unknown keys"

    enabled_value = skills_config.get("enabled")
    assert enabled_value is None or isinstance(
        enabled_value, bool
    ), f"{section}.enabled must be a boolean"
    bootstrap_enabled = _settings_bootstrap_enabled(
        skills_config.get("bootstrap"),
        section=section,
    )
    explicit_fields = _settings_explicit_fields(skills_config)

    return UntrustedSkillSettings(
        surface=surface,
        enabled=enabled_value,
        bootstrap_enabled=bootstrap_enabled,
        authority_kinds=_settings_authority_kinds(
            skills_config.get("authority_kinds"),
            section=section,
        ),
        source_labels=_settings_string_tuple(
            skills_config.get("source_labels"),
            f"{section}.source_labels",
        ),
        skill_ids=_settings_string_tuple(
            skills_config.get("skill_ids"),
            f"{section}.skill_ids",
        ),
        read_limits=cast(
            SkillReadLimits | None,
            _settings_limit(
                skills_config.get("read_limits"),
                settings_cls=SkillReadLimits,
                supported_keys={
                    "max_bytes_per_read",
                    "max_lines_per_read",
                },
                section=f"{section}.read_limits",
                base=trusted.read_limits,
            ),
        ),
        index_limits=cast(
            SkillIndexLimits | None,
            _settings_limit(
                skills_config.get("index_limits"),
                settings_cls=SkillIndexLimits,
                supported_keys={
                    "max_skills",
                    "max_resources_per_skill",
                    "max_indexed_bytes",
                },
                section=f"{section}.index_limits",
                base=trusted.index_limits,
            ),
        ),
        source_limits=cast(
            SkillSourceLimits | None,
            _settings_limit(
                skills_config.get("source_limits"),
                settings_cls=SkillSourceLimits,
                supported_keys={
                    "max_sources",
                    "max_resources_per_source",
                    "max_source_depth",
                    "max_files_per_source",
                    "max_directory_entries_per_source",
                },
                section=f"{section}.source_limits",
                base=trusted.source_limits,
            ),
        ),
        cursor_limits=cast(
            SkillCursorLimits | None,
            _settings_limit(
                skills_config.get("cursor_limits"),
                settings_cls=SkillCursorLimits,
                supported_keys={
                    "max_active_cursors",
                    "max_cursor_age_seconds",
                },
                section=f"{section}.cursor_limits",
                base=trusted.cursor_limits,
            ),
        ),
        privacy=cast(
            SkillPrivacySettings | None,
            _settings_limit(
                skills_config.get("privacy"),
                settings_cls=SkillPrivacySettings,
                supported_keys={
                    "include_source_labels",
                    "include_authority",
                    "include_diagnostic_paths",
                    "redact_host_paths",
                },
                section=f"{section}.privacy",
                base=trusted.privacy,
            ),
        ),
        observability=cast(
            SkillObservabilitySettings | None,
            _settings_limit(
                skills_config.get("observability"),
                settings_cls=SkillObservabilitySettings,
                supported_keys={
                    "enabled",
                    "emit_events",
                    "include_diagnostics",
                    "include_byte_counts",
                    "audit_fail_closed",
                },
                section=f"{section}.observability",
                base=trusted.observability,
            ),
        ),
        explicit_fields=explicit_fields,
    )


def untrusted_skill_settings_config_dict(
    settings: UntrustedSkillSettings,
) -> dict[str, SkillModelValue]:
    assert isinstance(settings, UntrustedSkillSettings)
    value: dict[str, object] = {}
    root_fields = settings.explicit_fields.get(_ROOT_EXPLICIT_FIELDS)
    if settings.enabled is not None and (
        settings.enabled is False or _field_explicit(root_fields, "enabled")
    ):
        value["enabled"] = settings.enabled
    if settings.bootstrap_enabled is not None and (
        settings.bootstrap_enabled is False
        or _field_explicit(root_fields, "bootstrap")
    ):
        value["bootstrap"] = "auto" if settings.bootstrap_enabled else "off"
    if settings.authority_kinds or _field_explicit(
        root_fields, "authority_kinds"
    ):
        value["authority_kinds"] = tuple(
            authority.value for authority in settings.authority_kinds
        )
    if settings.source_labels or _field_explicit(root_fields, "source_labels"):
        value["source_labels"] = settings.source_labels
    if settings.skill_ids or _field_explicit(root_fields, "skill_ids"):
        value["skill_ids"] = settings.skill_ids
    if settings.read_limits is not None:
        value["read_limits"] = _explicit_model_dict(
            settings.read_limits.as_model_dict(),
            settings.explicit_fields.get("read_limits"),
        )
    if settings.index_limits is not None:
        value["index_limits"] = _explicit_model_dict(
            settings.index_limits.as_model_dict(),
            settings.explicit_fields.get("index_limits"),
        )
    if settings.source_limits is not None:
        value["source_limits"] = _explicit_model_dict(
            settings.source_limits.as_model_dict(),
            settings.explicit_fields.get("source_limits"),
        )
    if settings.cursor_limits is not None:
        value["cursor_limits"] = _explicit_model_dict(
            settings.cursor_limits.as_model_dict(),
            settings.explicit_fields.get("cursor_limits"),
        )
    if settings.privacy is not None:
        value["privacy"] = _explicit_model_dict(
            settings.privacy.as_model_dict(),
            settings.explicit_fields.get("privacy"),
        )
    if settings.observability is not None:
        value["observability"] = _explicit_model_dict(
            settings.observability.as_model_dict(),
            settings.explicit_fields.get("observability"),
        )
    return model_dict(value)


def trusted_skill_settings_identity_dict(
    settings: TrustedSkillSettings,
) -> dict[str, SkillModelValue]:
    assert isinstance(settings, TrustedSkillSettings)
    return model_dict(
        {
            "enabled": settings.enabled,
            "bootstrap_enabled": settings.bootstrap_enabled,
            "bootstrap_prompt": settings.bootstrap_prompt.as_model_dict(),
            "authority_kinds": tuple(
                authority.value for authority in settings.authority_kinds
            ),
            "source_labels": tuple(
                source.label for source in settings.sources
            ),
            "sources": tuple(
                trusted_skill_source_identity_dict(source)
                for source in settings.sources
            ),
            "sources_explicit": settings.sources_explicit,
            "allowed_skill_ids": settings.allowed_skill_ids,
            "read_limits": settings.read_limits.as_model_dict(),
            "index_limits": settings.index_limits.as_model_dict(),
            "source_limits": settings.source_limits.as_model_dict(),
            "cursor_limits": settings.cursor_limits.as_model_dict(),
            "privacy": settings.privacy.as_model_dict(),
            "observability": settings.observability.as_model_dict(),
        }
    )


def trusted_skill_settings_fingerprint(
    settings: TrustedSkillSettings,
) -> str:
    assert isinstance(settings, TrustedSkillSettings)
    return _settings_fingerprint(
        trusted_skill_settings_identity_dict(settings)
    )


def trusted_skill_source_fingerprint(
    settings: TrustedSkillSettings,
) -> str:
    assert isinstance(settings, TrustedSkillSettings)
    return _settings_fingerprint(
        model_dict(
            {
                "authority_kinds": tuple(
                    authority.value for authority in settings.authority_kinds
                ),
                "sources": tuple(
                    trusted_skill_source_identity_dict(source)
                    for source in settings.sources
                ),
                "sources_explicit": settings.sources_explicit,
                "allowed_skill_ids": settings.allowed_skill_ids,
                "source_limits": settings.source_limits.as_model_dict(),
            }
        )
    )


def trusted_skill_source_identity_dict(
    source: SkillSourceConfig,
) -> dict[str, SkillModelValue]:
    assert isinstance(source, SkillSourceConfig)
    return skill_source_identity_dict(
        label=source.label,
        authority=source.authority,
        root_path=source.root_path,
        package_path=source.package_path,
        enabled=source.enabled,
        allow_hidden_paths=source.allow_hidden_paths,
        status=source.status,
    )


def skill_source_identity_dict(
    *,
    label: str,
    authority: SkillSourceAuthority,
    root_path: str | Path | None = None,
    package_path: str | None = None,
    enabled: bool = True,
    allow_hidden_paths: bool = False,
    status: SkillStatus = SkillStatus.OK,
) -> dict[str, SkillModelValue]:
    assert isinstance(label, str) and label.strip()
    assert isinstance(authority, SkillSourceAuthority)
    assert isinstance(enabled, bool)
    assert isinstance(allow_hidden_paths, bool)
    assert isinstance(status, SkillStatus)
    value: dict[str, object] = {
        "label": label,
        "authority": authority.as_model_dict(),
        "enabled": enabled,
        "allow_hidden_paths": allow_hidden_paths,
        "status": status.value,
    }
    if root_path is not None:
        value["effective_root_sha256"] = _identity_digest(
            _normalized_effective_root_path(root_path, package_path)
        )
    elif package_path is not None:
        value["package_path_sha256"] = _identity_digest(
            _normalized_package_path(package_path)
        )
    return model_dict(value)


def _settings_bootstrap_enabled(
    value: object,
    *,
    section: str,
) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    assert isinstance(value, str), f"{section}.bootstrap must be a string"
    assert value in {
        "auto",
        "off",
    }, f"{section}.bootstrap must be auto or off"
    return value == "auto"


def _settings_authority_kinds(
    value: object,
    *,
    section: str,
) -> tuple[SkillSourceAuthorityKind, ...]:
    if value is None:
        return ()
    assert isinstance(
        value,
        list,
    ), f"{section}.authority_kinds must be a list"
    kinds: list[SkillSourceAuthorityKind] = []
    for item in value:
        assert isinstance(
            item,
            str,
        ), f"{section}.authority_kinds entries must be strings"
        assert item in {
            authority.value for authority in SkillSourceAuthorityKind
        }, f"{section}.authority_kinds entries must be valid authorities"
        kinds.append(SkillSourceAuthorityKind(item))
    return tuple(kinds)


def _settings_string_tuple(
    value: object,
    section: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    assert isinstance(value, list), f"{section} must be a list"
    values: list[str] = []
    for item in value:
        assert isinstance(item, str), f"{section} entries must be strings"
        values.append(item)
    return tuple(values)


def _settings_limit(
    value: object,
    *,
    settings_cls: (
        type[SkillReadLimits]
        | type[SkillIndexLimits]
        | type[SkillSourceLimits]
        | type[SkillCursorLimits]
        | type[SkillPrivacySettings]
        | type[SkillObservabilitySettings]
    ),
    supported_keys: set[str],
    section: str,
    base: (
        SkillReadLimits
        | SkillIndexLimits
        | SkillSourceLimits
        | SkillCursorLimits
        | SkillPrivacySettings
        | SkillObservabilitySettings
        | None
    ) = None,
) -> (
    SkillReadLimits
    | SkillIndexLimits
    | SkillSourceLimits
    | SkillCursorLimits
    | SkillPrivacySettings
    | SkillObservabilitySettings
    | None
):
    if value is None:
        return None
    assert isinstance(value, Mapping), f"{section} must be a mapping"
    unknown_keys = sorted(set(value) - supported_keys)
    assert not unknown_keys, f"{section} has unknown keys"
    values = dict(value)
    if base is not None:
        assert isinstance(base, settings_cls)
        inherited = {
            field.name: getattr(base, field.name) for field in fields(base)
        }
        inherited.update(values)
        values = inherited
    return settings_cls(**values)


def _settings_explicit_fields(
    skills_config: Mapping[str, object],
) -> Mapping[str, tuple[str, ...]]:
    nested_sections = (
        "read_limits",
        "index_limits",
        "source_limits",
        "cursor_limits",
        "privacy",
        "observability",
    )
    fields_by_section: dict[str, tuple[str, ...]] = {
        _ROOT_EXPLICIT_FIELDS: tuple(
            str(key) for key in skills_config if key not in nested_sections
        )
    }
    for section in nested_sections:
        value = skills_config.get(section)
        if isinstance(value, Mapping):
            fields_by_section[section] = tuple(str(key) for key in value)
    return MappingProxyType(fields_by_section)


def _explicit_model_dict(
    value: dict[str, SkillModelValue],
    fields: tuple[str, ...] | None,
) -> dict[str, SkillModelValue]:
    if fields is None:
        return value
    return model_dict(
        {field: value[field] for field in fields if field in value}
    )


def _field_explicit(fields: tuple[str, ...] | None, field_name: str) -> bool:
    return fields is not None and field_name in fields


def _settings_fingerprint(value: Mapping[str, object]) -> str:
    canonical = dumps(
        model_dict(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _identity_digest(value: str) -> str:
    assert isinstance(value, str)
    return sha256(value.encode("utf-8")).hexdigest()


def _normalized_root_path(value: str | Path) -> str:
    assert isinstance(value, str | Path)
    return str(Path(value).expanduser().resolve(strict=False))


def _normalized_effective_root_path(
    root_path: str | Path,
    package_path: str | None,
) -> str:
    if package_path is None or package_path == ".":
        return _normalized_root_path(root_path)
    root = Path(root_path).expanduser()
    root = root.joinpath(*PurePosixPath(package_path).parts)
    return str(root.resolve(strict=False))


def _normalized_package_path(value: str) -> str:
    assert isinstance(value, str) and value.strip()
    return value.strip().replace("\\", "/")


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


def _assert_source_tuple(values: tuple[SkillSourceConfig, ...]) -> None:
    assert_tuple_items(values, "sources", SkillSourceConfig)


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
    assert_unique_sequence(labels, "source labels")


def _assert_logical_id_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_logical_id(value, field_name)
    assert_unique_sequence(values, field_name)


def _assert_source_label_tuple(
    values: tuple[str, ...], field_name: str
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_source_label(value, field_name)
    assert_unique_sequence(values, field_name)


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


def _assert_explicit_fields(
    value: Mapping[str, tuple[str, ...]],
) -> None:
    assert isinstance(value, Mapping)
    for section, section_fields in value.items():
        assert isinstance(section, str) and section.strip()
        assert isinstance(section_fields, tuple)
        for field_name in section_fields:
            assert isinstance(field_name, str) and field_name.strip()


def _assert_diagnostic_tuple(values: tuple[SkillDiagnosticInfo, ...]) -> None:
    assert_tuple_items(values, "diagnostics", SkillDiagnosticInfo)


def _allows_exposure(trusted: bool, requested: bool) -> bool:
    return trusted or not requested


def _allows_redaction(trusted: bool, requested: bool) -> bool:
    return requested or not trusted


def _allows_enforcement(trusted: bool, requested: bool) -> bool:
    return requested or not trusted


def _sequence_contains_ordered_subset(
    trusted: tuple[str, ...],
    requested: tuple[str, ...],
) -> bool:
    assert isinstance(trusted, tuple)
    assert isinstance(requested, tuple)
    trusted_index = 0
    for item in requested:
        while trusted_index < len(trusted) and trusted[trusted_index] != item:
            trusted_index += 1
        if trusted_index == len(trusted):
            return False
        trusted_index += 1
    return True
