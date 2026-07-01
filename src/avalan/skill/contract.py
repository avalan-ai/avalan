from ast import literal_eval
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from json import dumps
from pathlib import PurePosixPath
from re import fullmatch, search
from types import MappingProxyType
from typing import Any, cast


class SkillReleaseTarget(StrEnum):
    FILESYSTEM_TRUSTED_SOURCES = "filesystem_trusted_sources"
    READ_QUERY_ONLY_TOOLS = "read_query_only_tools"
    PLUGIN_PROVIDED_SOURCES = "plugin_provided_sources"
    BUNDLED_SOURCES = "bundled_sources"
    USER_LOCAL_SOURCES = "user_local_sources"
    REMOTE_PREINSTALLED_SOURCES = "remote_preinstalled_sources"
    RICHER_MATCHING = "richer_matching"
    BACKEND_MATERIALIZED_SOURCES = "backend_materialized_sources"


class SkillStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"
    POLICY_DENIED = "policy_denied"
    TRUNCATED = "truncated"
    STALE = "stale"
    BLOCKED = "blocked"


class SkillDiagnosticCode(StrEnum):
    EMPTY_REGISTRY = "skills.empty_registry"
    NO_MATCH = "skills.no_match"
    NOT_FOUND = "skills.not_found"
    AMBIGUOUS_NAME = "skills.ambiguous_name"
    DISABLED = "skills.disabled"
    SOURCE_UNAVAILABLE = "skills.source_unavailable"
    MANIFEST_MALFORMED = "skills.manifest_malformed"
    RESOURCE_MISSING = "skills.resource_missing"
    RESOURCE_OUTSIDE_ROOT = "skills.resource_outside_root"
    RESOURCE_OVERSIZED = "skills.resource_oversized"
    RESOURCE_STALE = "skills.resource_stale"
    DUPLICATE_ID = "skills.duplicate_id"
    RUNTIME_SOURCE_UNAVAILABLE = "skills.runtime_source_unavailable"
    BINARY_RESOURCE = "skills.resource_binary"
    POLICY_DENIED = "skills.policy_denied"
    SKILLS_SYNTAX_UNSUPPORTED = "skills.syntax_unsupported"


class SkillFailureMode(StrEnum):
    EMPTY_REGISTRY = "empty_registry"
    NO_MATCH = "no_match"
    UNKNOWN_SKILL_ID = "unknown_skill_id"
    AMBIGUOUS_SKILL_NAME = "ambiguous_skill_name"
    DISABLED_SKILL = "disabled_skill"
    SOURCE_UNAVAILABLE = "source_unavailable"
    MALFORMED_MANIFEST = "malformed_manifest"
    RESOURCE_MISSING = "resource_missing"
    RESOURCE_OUTSIDE_AUTHORIZED_ROOT = "resource_outside_authorized_root"
    RESOURCE_REQUIRES_CURSOR = "resource_requires_cursor"
    RESOURCE_STALE = "resource_stale"
    DUPLICATE_SKILL_IDS = "duplicate_skill_ids"
    RUNTIME_SOURCE_INACCESSIBLE = "runtime_source_inaccessible"


class SkillVocabularyTerm(StrEnum):
    SOURCE = "source"
    SOURCE_AUTHORITY = "source_authority"
    SOURCE_LABEL = "source_label"
    SKILL_ID = "skill_id"
    RESOURCE_ID = "resource_id"
    MAIN_RESOURCE = "main_resource"
    REGISTRY_VERSION = "registry_version"
    READ_CURSOR = "read_cursor"
    DIAGNOSTIC = "diagnostic"
    STATUS = "status"
    PROVENANCE = "provenance"


class SkillManifestField(StrEnum):
    NAME = "name"
    DESCRIPTION = "description"
    TAGS = "tags"
    VERSION = "version"
    ENABLED = "enabled"
    RESOURCES = "resources"


class SkillSyntaxSurface(StrEnum):
    SDK_SETTINGS = "sdk_settings"
    CLI_SETTINGS = "cli_settings"
    AGENT_TOML = "agent_toml"
    FLOW_DEFINITION = "flow_definition"
    TASK_DEFINITION = "task_definition"
    SERVER_REQUEST = "server_request"
    WORKER_ENVELOPE = "worker_envelope"


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillVocabularyEntry:
    term: SkillVocabularyTerm
    definition: str

    def __post_init__(self) -> None:
        assert isinstance(self.term, SkillVocabularyTerm)
        assert self.definition.strip(), "definition must be non-empty"


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillFailureDiagnosticContract:
    failure_mode: SkillFailureMode
    status: SkillStatus
    code: SkillDiagnosticCode
    message: str
    hint: str

    def __post_init__(self) -> None:
        assert isinstance(self.failure_mode, SkillFailureMode)
        assert isinstance(self.status, SkillStatus)
        assert isinstance(self.code, SkillDiagnosticCode)
        assert self.message.strip(), "message must be non-empty"
        assert self.hint.strip(), "hint must be non-empty"


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillDiagnostic:
    code: SkillDiagnosticCode
    status: SkillStatus
    message: str
    path: str
    hint: str
    candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.code, SkillDiagnosticCode)
        assert isinstance(self.status, SkillStatus)
        assert self.message.strip(), "message must be non-empty"
        assert self.path.strip(), "path must be non-empty"
        assert self.hint.strip(), "hint must be non-empty"
        assert isinstance(self.candidates, tuple)
        for candidate in self.candidates:
            _assert_logical_id(candidate, "candidate")

    def as_model_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "code": self.code.value,
            "status": self.status.value,
            "message": self.message,
            "hint": self.hint,
        }
        path = _model_safe_diagnostic_path(self.path)
        if path is not None:
            value["path"] = path
        if self.candidates:
            value["candidates"] = self.candidates
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillContractMetadata:
    skill_id: str
    name: str
    description: str
    source_label: str
    main_resource_id: str = "main"
    enabled: bool = True
    status: SkillStatus = SkillStatus.OK
    tags: tuple[str, ...] = ()
    version: str | None = None
    resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_skill_id(self.skill_id, "skill_id")
        assert self.name == self.skill_id
        assert self.description.strip(), "description must be non-empty"
        assert _is_model_safe_metadata_text(
            self.description
        ), "description must not expose host paths"
        _assert_source_label(self.source_label)
        _assert_resource_id(self.main_resource_id, "main_resource_id")
        assert isinstance(self.enabled, bool)
        assert isinstance(self.status, SkillStatus)
        assert isinstance(self.tags, tuple)
        for tag in self.tags:
            _assert_logical_id(tag, "tag")
        if self.version is not None:
            assert self.version.strip(), "version must be non-empty"
            assert _is_model_safe_metadata_text(
                self.version
            ), "version must not expose host paths"
        assert isinstance(self.resources, tuple)
        for resource in self.resources:
            _assert_resource_id(resource, "resource")

    def as_model_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "source_label": self.source_label,
            "main_resource_id": self.main_resource_id,
            "enabled": self.enabled,
            "status": self.status.value,
        }
        if self.tags:
            value["tags"] = self.tags
        if self.version is not None:
            value["version"] = self.version
        if self.resources:
            value["resources"] = self.resources
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillContractFixture:
    source_label: str
    content: str | None = None
    content_bytes: bytes | None = None
    host_path: str | None = None
    source_available: bool = True
    runtime_available: bool = True
    policy_allowed: bool = True
    stale: bool = False
    requested_name: str | None = None
    ambiguous_candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_source_label(self.source_label)
        assert (
            self.content is None or self.content_bytes is None
        ), "content and content_bytes are mutually exclusive"
        if self.content is not None:
            assert isinstance(self.content, str)
        if self.content_bytes is not None:
            assert isinstance(self.content_bytes, bytes)
        if self.host_path is not None:
            assert self.host_path.strip(), "host_path must be non-empty"
        assert isinstance(self.source_available, bool)
        assert isinstance(self.runtime_available, bool)
        assert isinstance(self.policy_allowed, bool)
        assert isinstance(self.stale, bool)
        if self.requested_name is not None:
            assert self.requested_name.strip(), "requested_name is required"
        assert isinstance(self.ambiguous_candidates, tuple)
        if self.ambiguous_candidates:
            assert (
                len(self.ambiguous_candidates) >= 2
            ), "ambiguous fixtures need at least two candidates"
        for candidate in self.ambiguous_candidates:
            _assert_skill_id(candidate, "ambiguous candidate")


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillContractCompilation:
    status: SkillStatus
    registry_version: str
    items: tuple[SkillContractMetadata, ...] = ()
    diagnostics: tuple[SkillDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.status, SkillStatus)
        assert self.registry_version.strip(), "registry_version is required"
        assert isinstance(self.items, tuple)
        for item in self.items:
            assert isinstance(item, SkillContractMetadata)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, SkillDiagnostic)

    def as_model_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "registry_version": self.registry_version,
            "items": tuple(item.as_model_dict() for item in self.items),
            "diagnostics": tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            ),
            "provenance": tuple(
                {
                    "source_label": item.source_label,
                    "skill_id": item.skill_id,
                    "resource_id": item.main_resource_id,
                }
                for item in self.items
            ),
        }


FIRST_RELEASE_SKILL_TARGETS: tuple[SkillReleaseTarget, ...] = (
    SkillReleaseTarget.FILESYSTEM_TRUSTED_SOURCES,
    SkillReleaseTarget.READ_QUERY_ONLY_TOOLS,
)
LATER_RELEASE_SKILL_TARGETS: tuple[SkillReleaseTarget, ...] = (
    SkillReleaseTarget.PLUGIN_PROVIDED_SOURCES,
    SkillReleaseTarget.BUNDLED_SOURCES,
    SkillReleaseTarget.USER_LOCAL_SOURCES,
    SkillReleaseTarget.REMOTE_PREINSTALLED_SOURCES,
    SkillReleaseTarget.RICHER_MATCHING,
    SkillReleaseTarget.BACKEND_MATERIALIZED_SOURCES,
)
SKILL_BACKWARD_COMPATIBILITY_REQUIRED = False
SKILL_MANIFEST_FORMAT = "markdown_front_matter"
SKILL_MAIN_RESOURCE_ID = "main"
SKILL_MAIN_RESOURCE_FILENAME = "SKILL.md"
CANONICAL_SKILLS_TOOL_NAMES: tuple[str, ...] = (
    "skills.list",
    "skills.match",
    "skills.read",
    "skills.check",
)
DISALLOWED_MODEL_FACING_SKILLS_TOOL_NAMES: tuple[str, ...] = ("skills.load",)
REQUIRED_SKILL_MANIFEST_FIELDS: tuple[SkillManifestField, ...] = (
    SkillManifestField.NAME,
    SkillManifestField.DESCRIPTION,
)
SUPPORTED_SKILL_MANIFEST_FIELDS: tuple[SkillManifestField, ...] = (
    SkillManifestField.NAME,
    SkillManifestField.DESCRIPTION,
    SkillManifestField.TAGS,
    SkillManifestField.VERSION,
    SkillManifestField.ENABLED,
    SkillManifestField.RESOURCES,
)
SKILLS_SYNTAX_REJECTING_SURFACES: tuple[SkillSyntaxSurface, ...] = (
    SkillSyntaxSurface.SDK_SETTINGS,
    SkillSyntaxSurface.CLI_SETTINGS,
    SkillSyntaxSurface.AGENT_TOML,
    SkillSyntaxSurface.FLOW_DEFINITION,
    SkillSyntaxSurface.TASK_DEFINITION,
    SkillSyntaxSurface.SERVER_REQUEST,
    SkillSyntaxSurface.WORKER_ENVELOPE,
)

SKILL_VOCABULARY: Mapping[SkillVocabularyTerm, SkillVocabularyEntry] = (
    MappingProxyType(
        {
            SkillVocabularyTerm.SOURCE: SkillVocabularyEntry(
                term=SkillVocabularyTerm.SOURCE,
                definition=(
                    "A trusted configured filesystem location that "
                    "contributes skills to a registry."
                ),
            ),
            SkillVocabularyTerm.SOURCE_AUTHORITY: SkillVocabularyEntry(
                term=SkillVocabularyTerm.SOURCE_AUTHORITY,
                definition=(
                    "The trust category assigned by runtime configuration "
                    "before source resolution."
                ),
            ),
            SkillVocabularyTerm.SOURCE_LABEL: SkillVocabularyEntry(
                term=SkillVocabularyTerm.SOURCE_LABEL,
                definition=(
                    "A logical, model-facing source name that never reveals "
                    "the host path."
                ),
            ),
            SkillVocabularyTerm.SKILL_ID: SkillVocabularyEntry(
                term=SkillVocabularyTerm.SKILL_ID,
                definition=(
                    "The stable logical identifier for a skill; in the first "
                    "release it is the safe manifest name."
                ),
            ),
            SkillVocabularyTerm.RESOURCE_ID: SkillVocabularyEntry(
                term=SkillVocabularyTerm.RESOURCE_ID,
                definition=(
                    "A logical readable artifact identifier owned by a "
                    "registered skill."
                ),
            ),
            SkillVocabularyTerm.MAIN_RESOURCE: SkillVocabularyEntry(
                term=SkillVocabularyTerm.MAIN_RESOURCE,
                definition=(
                    "The primary instruction resource for a skill, exposed as "
                    "the logical resource ID 'main'."
                ),
            ),
            SkillVocabularyTerm.REGISTRY_VERSION: SkillVocabularyEntry(
                term=SkillVocabularyTerm.REGISTRY_VERSION,
                definition=(
                    "A short deterministic identifier for the registry state "
                    "visible to a model run."
                ),
            ),
            SkillVocabularyTerm.READ_CURSOR: SkillVocabularyEntry(
                term=SkillVocabularyTerm.READ_CURSOR,
                definition=(
                    "An opaque logical continuation handle for bounded "
                    "resource reads."
                ),
            ),
            SkillVocabularyTerm.DIAGNOSTIC: SkillVocabularyEntry(
                term=SkillVocabularyTerm.DIAGNOSTIC,
                definition=(
                    "A structured finding with code, status, message, hint, "
                    "and optional candidate skill IDs."
                ),
            ),
            SkillVocabularyTerm.STATUS: SkillVocabularyEntry(
                term=SkillVocabularyTerm.STATUS,
                definition=(
                    "A compact response classification such as ok, empty, "
                    "not_found, malformed, or policy_denied."
                ),
            ),
            SkillVocabularyTerm.PROVENANCE: SkillVocabularyEntry(
                term=SkillVocabularyTerm.PROVENANCE,
                definition=(
                    "Model-facing source, skill, and resource identifiers "
                    "that omit host-absolute paths."
                ),
            ),
        }
    )
)

_RUNTIME_SOURCE_MODE = SkillFailureMode.RUNTIME_SOURCE_INACCESSIBLE
_RUNTIME_SOURCE_DIAGNOSTIC = SkillFailureDiagnosticContract(
    failure_mode=_RUNTIME_SOURCE_MODE,
    status=SkillStatus.UNAVAILABLE,
    code=SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
    message=(
        "The sandbox or container runtime cannot access the configured source."
    ),
    hint="Keep the registry unavailable instead of widening access.",
)

SECTION_14_FAILURE_DIAGNOSTICS: Mapping[
    SkillFailureMode, SkillFailureDiagnosticContract
] = MappingProxyType(
    {
        SkillFailureMode.EMPTY_REGISTRY: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.EMPTY_REGISTRY,
            status=SkillStatus.EMPTY,
            code=SkillDiagnosticCode.EMPTY_REGISTRY,
            message="No skills are configured.",
            hint="Continue without skills or ask for runtime configuration.",
        ),
        SkillFailureMode.NO_MATCH: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.NO_MATCH,
            status=SkillStatus.EMPTY,
            code=SkillDiagnosticCode.NO_MATCH,
            message="No skills match the task.",
            hint="Continue without a skill or refine the query.",
        ),
        SkillFailureMode.UNKNOWN_SKILL_ID: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.UNKNOWN_SKILL_ID,
            status=SkillStatus.NOT_FOUND,
            code=SkillDiagnosticCode.NOT_FOUND,
            message="The requested skill ID is unknown.",
            hint="Call skills.list or skills.match for available skills.",
        ),
        SkillFailureMode.AMBIGUOUS_SKILL_NAME: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.AMBIGUOUS_SKILL_NAME,
            status=SkillStatus.AMBIGUOUS,
            code=SkillDiagnosticCode.AMBIGUOUS_NAME,
            message="The requested skill name is ambiguous.",
            hint="Choose one of the candidate skill IDs explicitly.",
        ),
        SkillFailureMode.DISABLED_SKILL: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.DISABLED_SKILL,
            status=SkillStatus.DISABLED,
            code=SkillDiagnosticCode.DISABLED,
            message="The requested skill exists but is disabled.",
            hint=(
                "Continue without the skill or ask the operator to enable it."
            ),
        ),
        SkillFailureMode.SOURCE_UNAVAILABLE: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.SOURCE_UNAVAILABLE,
            status=SkillStatus.UNAVAILABLE,
            code=SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            message="The skill source is unavailable.",
            hint="Do not substitute another source during this run.",
        ),
        SkillFailureMode.MALFORMED_MANIFEST: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.MALFORMED_MANIFEST,
            status=SkillStatus.MALFORMED,
            code=SkillDiagnosticCode.MANIFEST_MALFORMED,
            message="The skill manifest is malformed.",
            hint="Skip the skill until its front matter is corrected.",
        ),
        SkillFailureMode.RESOURCE_MISSING: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.RESOURCE_MISSING,
            status=SkillStatus.NOT_FOUND,
            code=SkillDiagnosticCode.RESOURCE_MISSING,
            message="The requested skill resource is missing.",
            hint="Read only declared resources from the current registry.",
        ),
        SkillFailureMode.RESOURCE_OUTSIDE_AUTHORIZED_ROOT: (
            SkillFailureDiagnosticContract(
                failure_mode=(
                    SkillFailureMode.RESOURCE_OUTSIDE_AUTHORIZED_ROOT
                ),
                status=SkillStatus.POLICY_DENIED,
                code=SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
                message="The skill resource is outside its authorized root.",
                hint="Reject traversal and use logical resource IDs only.",
            )
        ),
        SkillFailureMode.RESOURCE_REQUIRES_CURSOR: (
            SkillFailureDiagnosticContract(
                failure_mode=SkillFailureMode.RESOURCE_REQUIRES_CURSOR,
                status=SkillStatus.TRUNCATED,
                code=SkillDiagnosticCode.RESOURCE_OVERSIZED,
                message="The skill resource is too large for one read.",
                hint="Use a bounded read cursor when the reader supports it.",
            )
        ),
        SkillFailureMode.RESOURCE_STALE: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.RESOURCE_STALE,
            status=SkillStatus.STALE,
            code=SkillDiagnosticCode.RESOURCE_STALE,
            message=(
                "The skill changed after the registry version was established."
            ),
            hint="Rebuild the registry before trusting the resource.",
        ),
        SkillFailureMode.DUPLICATE_SKILL_IDS: SkillFailureDiagnosticContract(
            failure_mode=SkillFailureMode.DUPLICATE_SKILL_IDS,
            status=SkillStatus.BLOCKED,
            code=SkillDiagnosticCode.DUPLICATE_ID,
            message="The source contains duplicate skill IDs.",
            hint=(
                "Reject duplicates until an explicit precedence policy exists."
            ),
        ),
        _RUNTIME_SOURCE_MODE: _RUNTIME_SOURCE_DIAGNOSTIC,
    }
)

_INVALID_REGISTRY_VERSION = "skills-contract:invalid"
_SKILL_ID_PATTERN = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
_SOURCE_LABEL_PATTERN = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
_UNSAFE_METADATA_TEXT_FRAGMENTS = (
    "$home",
    "${home}",
    "~/",
    ".aws/",
    ".codex/",
    ".config/",
    ".env",
    ".ssh/",
    "/.aws",
    "/.codex",
    "/.config",
    "/.env",
    "/.ssh",
    "/home/",
    "/private/",
    "/root/",
    "/secrets/",
    "/tmp/",
    "/users/",
    "/var/folders/",
    "c:/users/",
    "private/",
    "secrets/",
)
_ABSOLUTE_PATH_TEXT_PATTERN = r"(^|[\s(])/(?:[^/\s]+/)+[^\s]*"
_WINDOWS_PATH_TEXT_PATTERN = r"(^|[\s(])[a-z]:/"
_MODEL_SAFE_DIAGNOSTIC_PATH_ROOTS = frozenset(
    {
        "skills",
        "manifest",
        "source",
        "resource",
    }
)
_SKILLS_TOOL_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "skills.list": "Return compact metadata for available skills.",
        "skills.match": "Return ranked skill metadata candidates for a task.",
        "skills.read": (
            "Return bounded content for an authorized skill resource."
        ),
        "skills.check": (
            "Return skill diagnostics without reading resource bodies."
        ),
    }
)
_SUPPORTED_FIELD_NAMES = frozenset(
    field.value for field in SUPPORTED_SKILL_MANIFEST_FIELDS
)


def all_failure_diagnostic_contracts() -> (
    Mapping[SkillFailureMode, SkillFailureDiagnosticContract]
):
    return SECTION_14_FAILURE_DIAGNOSTICS


def diagnostic_contract_for_failure(
    failure_mode: SkillFailureMode,
) -> SkillFailureDiagnosticContract:
    assert isinstance(failure_mode, SkillFailureMode)
    return SECTION_14_FAILURE_DIAGNOSTICS[failure_mode]


def canonical_skills_tool_schemas() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _tool_description(name),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
        for name in CANONICAL_SKILLS_TOOL_NAMES
    )


def rejects_skills_syntax(surface: SkillSyntaxSurface) -> bool:
    assert isinstance(surface, SkillSyntaxSurface)
    return surface in SKILLS_SYNTAX_REJECTING_SURFACES


def reject_skills_syntax(
    surface: SkillSyntaxSurface, *, path: str = "skills"
) -> SkillDiagnostic:
    assert isinstance(surface, SkillSyntaxSurface)
    assert path.strip(), "path must be non-empty"
    return SkillDiagnostic(
        code=SkillDiagnosticCode.SKILLS_SYNTAX_UNSUPPORTED,
        status=SkillStatus.BLOCKED,
        message=(
            f"Skills syntax is not supported on {surface.value} in this phase."
        ),
        path=path,
        hint=(
            "Remove skills configuration from this surface until its "
            "integration phase implements it."
        ),
    )


def compile_skill_contract_fixtures(
    fixtures: Sequence[SkillContractFixture],
    *,
    maximum_manifest_bytes: int = 65_536,
) -> SkillContractCompilation:
    """Compile Phase 0 contract fixtures, not production skill sources."""
    assert isinstance(fixtures, Sequence)
    assert not isinstance(fixtures, str)
    assert (
        maximum_manifest_bytes > 0
    ), "maximum_manifest_bytes must be positive"

    if not fixtures:
        return _diagnostic_compilation(
            _diagnostic_from_failure(
                SkillFailureMode.EMPTY_REGISTRY,
                path="skills",
            )
        )

    items: list[SkillContractMetadata] = []
    diagnostics: list[SkillDiagnostic] = []
    for fixture in fixtures:
        assert isinstance(fixture, SkillContractFixture)
        diagnostic = _preflight_fixture_diagnostic(fixture)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
            continue

        if fixture.ambiguous_candidates:
            diagnostics.append(
                _diagnostic_from_failure(
                    SkillFailureMode.AMBIGUOUS_SKILL_NAME,
                    path="skills.request.name",
                    candidates=fixture.ambiguous_candidates,
                )
            )
            continue

        content_bytes = _fixture_content_bytes(fixture)
        if len(content_bytes) > maximum_manifest_bytes:
            diagnostics.append(
                _diagnostic_from_failure(
                    SkillFailureMode.RESOURCE_REQUIRES_CURSOR,
                    path="resource.main",
                )
            )
            continue

        try:
            content = _decode_fixture_content(content_bytes)
        except UnicodeDecodeError:
            diagnostics.append(
                SkillDiagnostic(
                    code=SkillDiagnosticCode.BINARY_RESOURCE,
                    status=SkillStatus.UNAVAILABLE,
                    message="The skill resource is binary or non-UTF-8.",
                    path="resource.main",
                    hint="Expose only UTF-8 Markdown skill resources.",
                )
            )
            continue

        item, manifest_diagnostics = _compile_manifest(
            content,
            source_label=fixture.source_label,
        )
        diagnostics.extend(manifest_diagnostics)
        if item is not None:
            items.append(item)

    if diagnostics:
        return SkillContractCompilation(
            status=diagnostics[0].status,
            registry_version=_INVALID_REGISTRY_VERSION,
            diagnostics=tuple(diagnostics),
        )

    duplicate_ids = _duplicate_skill_ids(items)
    if duplicate_ids:
        return _diagnostic_compilation(
            _diagnostic_from_failure(
                SkillFailureMode.DUPLICATE_SKILL_IDS,
                path="manifest.name",
                candidates=duplicate_ids,
            )
        )

    assert items, "non-empty fixture compilation produced no items"
    return SkillContractCompilation(
        status=SkillStatus.OK,
        registry_version=_registry_version(items),
        items=tuple(items),
    )


def _preflight_fixture_diagnostic(
    fixture: SkillContractFixture,
) -> SkillDiagnostic | None:
    if not fixture.source_available:
        return _diagnostic_from_failure(
            SkillFailureMode.SOURCE_UNAVAILABLE,
            path="source",
        )
    if not fixture.runtime_available:
        return _diagnostic_from_failure(
            SkillFailureMode.RUNTIME_SOURCE_INACCESSIBLE,
            path="source",
        )
    if not fixture.policy_allowed:
        return SkillDiagnostic(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="Skill access is denied by runtime policy.",
            path="source.policy",
            hint="Do not treat skill configuration as a permission grant.",
        )
    if fixture.stale:
        return _diagnostic_from_failure(
            SkillFailureMode.RESOURCE_STALE,
            path="resource.main",
        )
    return None


def _fixture_content_bytes(fixture: SkillContractFixture) -> bytes:
    if fixture.content_bytes is not None:
        return fixture.content_bytes
    if fixture.content is not None:
        return fixture.content.encode("utf-8")
    return b""


def _decode_fixture_content(content_bytes: bytes) -> str:
    if b"\x00" in content_bytes:
        raise UnicodeDecodeError("utf-8", content_bytes, 0, 1, "NUL byte")
    return content_bytes.decode("utf-8")


def _compile_manifest(
    content: str, *, source_label: str
) -> tuple[SkillContractMetadata | None, tuple[SkillDiagnostic, ...]]:
    manifest, front_matter_diagnostic = _parse_front_matter(content)
    if front_matter_diagnostic is not None:
        return None, (front_matter_diagnostic,)

    diagnostics = _validate_manifest_fields(manifest)
    if diagnostics:
        return None, diagnostics

    name = cast(str, manifest[SkillManifestField.NAME.value])
    description = cast(str, manifest[SkillManifestField.DESCRIPTION.value])
    enabled = cast(bool, manifest.get(SkillManifestField.ENABLED.value, True))
    tags_value = manifest.get(SkillManifestField.TAGS.value, ())
    resources_value = manifest.get(SkillManifestField.RESOURCES.value, ())
    assert isinstance(tags_value, tuple)
    assert isinstance(resources_value, tuple)
    tags = cast(tuple[str, ...], tags_value)
    resources = cast(tuple[str, ...], resources_value)
    version = cast(str | None, manifest.get(SkillManifestField.VERSION.value))

    if not enabled:
        return None, (
            _diagnostic_from_failure(
                SkillFailureMode.DISABLED_SKILL,
                path="manifest.enabled",
                candidates=(name,),
            ),
        )

    return (
        SkillContractMetadata(
            skill_id=name,
            name=name,
            description=description,
            source_label=source_label,
            tags=tags,
            version=version,
            resources=resources,
        ),
        (),
    )


def _parse_front_matter(
    content: str,
) -> tuple[dict[str, object], SkillDiagnostic | None]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, _malformed_manifest_diagnostic(
            path="manifest",
            hint="Start the skill Markdown file with YAML-style front matter.",
        )

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, _malformed_manifest_diagnostic(
            path="manifest",
            hint="Close the front matter block with a standalone '---' line.",
        )

    manifest: dict[str, object] = {}
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            return {}, _malformed_manifest_diagnostic(
                path=f"manifest.line.{line_number}",
                hint="Use 'field: value' entries in front matter.",
            )
        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        if key not in _SUPPORTED_FIELD_NAMES:
            return {}, _malformed_manifest_diagnostic(
                path=f"manifest.{key or 'field'}",
                hint="Use only the supported first-release manifest fields.",
            )
        if key in manifest:
            return {}, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                hint="Declare each manifest field at most once.",
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
) -> tuple[object | None, SkillDiagnostic | None]:
    if key == SkillManifestField.ENABLED.value:
        if raw_value == "true":
            return True, None
        if raw_value == "false":
            return False, None
        return None, _malformed_manifest_diagnostic(
            path="manifest.enabled",
            hint="Use true or false for enabled.",
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
                hint="Use an inline list of strings.",
            )
        if not isinstance(parsed, list) or any(
            not isinstance(item, str) for item in parsed
        ):
            return None, _malformed_manifest_diagnostic(
                path=f"manifest.{key}",
                hint="Use an inline list of strings.",
            )
        return tuple(parsed), None

    return _parse_string_metadata_value(key=key, raw_value=raw_value)


def _parse_string_scalar(raw_value: str) -> str | None:
    if raw_value[0] in {"'", '"'}:
        try:
            parsed = literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, str):
            return None
        return parsed if parsed.strip() else None
    return raw_value if raw_value.strip() else None


def _parse_string_metadata_value(
    *, key: str, raw_value: str
) -> tuple[object | None, SkillDiagnostic | None]:
    if not raw_value:
        return None, _malformed_manifest_diagnostic(
            path=f"manifest.{key}",
            hint="Use a non-empty string value.",
        )
    parsed_scalar = _parse_string_scalar(raw_value)
    if parsed_scalar is None:
        return None, _malformed_manifest_diagnostic(
            path=f"manifest.{key}",
            hint="Use a string value.",
        )
    return parsed_scalar, None


def _validate_manifest_fields(
    manifest: Mapping[str, object],
) -> tuple[SkillDiagnostic, ...]:
    diagnostics: list[SkillDiagnostic] = []
    for field_name in REQUIRED_SKILL_MANIFEST_FIELDS:
        if field_name.value not in manifest:
            diagnostics.append(
                _malformed_manifest_diagnostic(
                    path=f"manifest.{field_name.value}",
                    hint=(
                        "Declare required first-release fields: name and "
                        "description."
                    ),
                )
            )

    name = manifest.get(SkillManifestField.NAME.value)
    if isinstance(name, str) and not _is_skill_id(name):
        diagnostics.append(
            _malformed_manifest_diagnostic(
                path="manifest.name",
                hint="Use a lowercase logical skill ID as the name.",
            )
        )

    for field_name in (
        SkillManifestField.DESCRIPTION,
        SkillManifestField.VERSION,
    ):
        value = manifest.get(field_name.value)
        if isinstance(value, str) and not _is_model_safe_metadata_text(value):
            diagnostics.append(
                _malformed_manifest_diagnostic(
                    path=f"manifest.{field_name.value}",
                    hint=(
                        "Do not include host paths, user-local paths, hidden "
                        "paths, or sensitive path fragments in metadata."
                    ),
                )
            )

    enabled = manifest.get(SkillManifestField.ENABLED.value)
    if enabled is not None:
        assert isinstance(enabled, bool)

    diagnostics.extend(
        _validate_string_tuple_field(
            manifest,
            field=SkillManifestField.TAGS,
            validator=_is_logical_id,
            hint="Use lowercase logical IDs for tags.",
        )
    )
    diagnostics.extend(_validate_resources(manifest))
    return tuple(diagnostics)


def _validate_string_tuple_field(
    manifest: Mapping[str, object],
    *,
    field: SkillManifestField,
    validator: Any,
    hint: str,
) -> tuple[SkillDiagnostic, ...]:
    value = manifest.get(field.value)
    if value is None:
        return ()
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not validator(item) for item in value
    ):
        return (
            _malformed_manifest_diagnostic(
                path=f"manifest.{field.value}",
                hint=hint,
            ),
        )
    return ()


def _validate_resources(
    manifest: Mapping[str, object],
) -> tuple[SkillDiagnostic, ...]:
    value = manifest.get(SkillManifestField.RESOURCES.value)
    if value is None:
        return ()
    assert isinstance(value, tuple)
    diagnostics: list[SkillDiagnostic] = []
    for resource in value:
        assert isinstance(resource, str)
        if _resource_has_traversal(resource):
            diagnostics.append(
                _diagnostic_from_failure(
                    SkillFailureMode.RESOURCE_OUTSIDE_AUTHORIZED_ROOT,
                    path="manifest.resources",
                    candidates=(),
                )
            )
        elif not _is_resource_id(resource):
            diagnostics.append(
                _malformed_manifest_diagnostic(
                    path="manifest.resources",
                    hint="Use safe logical relative resource IDs.",
                )
            )
    return tuple(diagnostics)


def _diagnostic_from_failure(
    failure_mode: SkillFailureMode,
    *,
    path: str,
    candidates: tuple[str, ...] = (),
) -> SkillDiagnostic:
    contract = diagnostic_contract_for_failure(failure_mode)
    return SkillDiagnostic(
        code=contract.code,
        status=contract.status,
        message=contract.message,
        path=path,
        hint=contract.hint,
        candidates=tuple(sorted(candidates)),
    )


def _malformed_manifest_diagnostic(*, path: str, hint: str) -> SkillDiagnostic:
    contract = diagnostic_contract_for_failure(
        SkillFailureMode.MALFORMED_MANIFEST
    )
    return SkillDiagnostic(
        code=contract.code,
        status=contract.status,
        message=contract.message,
        path=path,
        hint=hint,
    )


def _diagnostic_compilation(
    diagnostic: SkillDiagnostic,
) -> SkillContractCompilation:
    return SkillContractCompilation(
        status=diagnostic.status,
        registry_version=_INVALID_REGISTRY_VERSION,
        diagnostics=(diagnostic,),
    )


def _duplicate_skill_ids(
    items: Sequence[SkillContractMetadata],
) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item.skill_id in seen:
            duplicates.add(item.skill_id)
        seen.add(item.skill_id)
    return tuple(sorted(duplicates))


def _registry_version(items: Sequence[SkillContractMetadata]) -> str:
    public_items = [item.as_model_dict() for item in items]
    payload = dumps(public_items, sort_keys=True, separators=(",", ":"))
    digest = sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"skills-contract:{digest}"


def _tool_description(name: str) -> str:
    return _SKILLS_TOOL_DESCRIPTIONS[name]


def _model_safe_diagnostic_path(path: str) -> str | None:
    if not path or "\x00" in path or "\\" in path:
        return None
    if "/" in path or path.startswith(("/", "~", "$")):
        return None
    parts = path.split(".")
    if parts[0] not in _MODEL_SAFE_DIAGNOSTIC_PATH_ROOTS:
        return None
    if not all(
        part and fullmatch(r"[a-z0-9][a-z0-9_-]*", part) for part in parts
    ):
        return None
    return path


def _is_model_safe_metadata_text(value: str) -> bool:
    assert isinstance(value, str)
    if not value.strip() or "\x00" in value:
        return False

    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    stripped = lowered.strip()
    if stripped.startswith(("/", "~", "$")):
        return False
    if "../" in lowered:
        return False
    if search(_ABSOLUTE_PATH_TEXT_PATTERN, lowered) is not None:
        return False
    if search(_WINDOWS_PATH_TEXT_PATTERN, lowered) is not None:
        return False
    return not any(
        fragment in lowered for fragment in _UNSAFE_METADATA_TEXT_FRAGMENTS
    )


def _resource_has_traversal(resource_id: str) -> bool:
    if "\x00" in resource_id:
        return True
    if resource_id.startswith(("/", "~", "$")):
        return True
    path = PurePosixPath(resource_id)
    return path.is_absolute() or ".." in path.parts


def _is_skill_id(value: str) -> bool:
    return bool(fullmatch(_SKILL_ID_PATTERN, value))


def _is_logical_id(value: str) -> bool:
    return bool(fullmatch(_SKILL_ID_PATTERN, value))


def _is_resource_id(value: str) -> bool:
    if not value or "\x00" in value or "\\" in value:
        return False
    if value.startswith(("/", "~", "$")):
        return False
    if any(part in {"", "."} for part in value.split("/")):
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return False
    return True


def _assert_skill_id(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert _is_skill_id(value), f"{field_name} must be a logical skill ID"


def _assert_logical_id(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert _is_logical_id(value), f"{field_name} must be a logical ID"


def _assert_source_label(value: str) -> None:
    assert isinstance(value, str), "source_label must be a string"
    assert fullmatch(
        _SOURCE_LABEL_PATTERN, value
    ), "source_label must be a logical label"


def _assert_resource_id(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert _is_resource_id(value), f"{field_name} must be a resource ID"
