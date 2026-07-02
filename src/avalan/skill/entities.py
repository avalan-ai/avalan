from ..types import (
    assert_model_safe_text as _assert_model_text,
)
from ..types import (
    assert_non_empty_model_safe_text as _assert_non_empty_model_text,
)
from ..types import (
    assert_non_empty_model_safe_text_tuple,
    assert_tuple_items,
    assert_unique_sequence,
)
from ..types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from ..types import (
    assert_relative_resource_id as _assert_resource_id,
)
from ..types import (
    assert_relative_resource_id_tuple as _assert_resource_id_tuple,
)
from ..types import (
    is_relative_resource_id as _is_resource_id,
)
from .contract import (
    SkillDiagnosticCode,
    SkillFailureMode,
    SkillStatus,
    diagnostic_contract_for_failure,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from json import dumps
from math import isfinite
from pathlib import Path
from re import fullmatch
from types import MappingProxyType
from typing import TypeAlias

SkillModelValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["SkillModelValue", ...]
    | Mapping[str, "SkillModelValue"]
)
SkillModelMapping: TypeAlias = Mapping[str, SkillModelValue]


class SkillSourceAuthorityKind(StrEnum):
    BUNDLED = "bundled"
    WORKSPACE = "workspace"
    USER_LOCAL = "user_local"
    PLUGIN_PROVIDED = "plugin_provided"
    PREINSTALLED_REMOTE = "preinstalled_remote"


def _empty_model_mapping() -> SkillModelMapping:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillRegistryVersion:
    value: str

    def __post_init__(self) -> None:
        _assert_registry_version(self.value)

    def as_model_value(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSourceAuthority:
    kind: SkillSourceAuthorityKind

    def __post_init__(self) -> None:
        assert isinstance(self.kind, SkillSourceAuthorityKind)
        if type(self) is SkillSourceAuthority:
            assert self.kind not in {
                SkillSourceAuthorityKind.PLUGIN_PROVIDED,
                SkillSourceAuthorityKind.PREINSTALLED_REMOTE,
            }, "identity-bearing authorities require concrete types"

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return {"kind": self.kind.value}


@dataclass(frozen=True, slots=True, kw_only=True)
class BundledSkillSourceAuthority(SkillSourceAuthority):
    kind: SkillSourceAuthorityKind = SkillSourceAuthorityKind.BUNDLED
    bundle_id: str = "avalan"

    def __post_init__(self) -> None:
        SkillSourceAuthority.__post_init__(self)
        _assert_authority_kind(self.kind, SkillSourceAuthorityKind.BUNDLED)
        _assert_logical_id(self.bundle_id, "bundle_id")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value = SkillSourceAuthority.as_model_dict(self)
        value["bundle_id"] = self.bundle_id
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkspaceSkillSourceAuthority(SkillSourceAuthority):
    kind: SkillSourceAuthorityKind = SkillSourceAuthorityKind.WORKSPACE
    workspace_id: str = "workspace"

    def __post_init__(self) -> None:
        SkillSourceAuthority.__post_init__(self)
        _assert_authority_kind(self.kind, SkillSourceAuthorityKind.WORKSPACE)
        _assert_logical_id(self.workspace_id, "workspace_id")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value = SkillSourceAuthority.as_model_dict(self)
        value["workspace_id"] = self.workspace_id
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class UserLocalSkillSourceAuthority(SkillSourceAuthority):
    kind: SkillSourceAuthorityKind = SkillSourceAuthorityKind.USER_LOCAL
    profile_id: str = "user-local"

    def __post_init__(self) -> None:
        SkillSourceAuthority.__post_init__(self)
        _assert_authority_kind(self.kind, SkillSourceAuthorityKind.USER_LOCAL)
        _assert_logical_id(self.profile_id, "profile_id")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value = SkillSourceAuthority.as_model_dict(self)
        value["profile_id"] = self.profile_id
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginProvidedSkillSourceAuthority(SkillSourceAuthority):
    plugin_id: str
    kind: SkillSourceAuthorityKind = SkillSourceAuthorityKind.PLUGIN_PROVIDED

    def __post_init__(self) -> None:
        SkillSourceAuthority.__post_init__(self)
        _assert_authority_kind(
            self.kind, SkillSourceAuthorityKind.PLUGIN_PROVIDED
        )
        _assert_logical_id(self.plugin_id, "plugin_id")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value = SkillSourceAuthority.as_model_dict(self)
        value["plugin_id"] = self.plugin_id
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PreinstalledRemoteSkillSourceAuthority(SkillSourceAuthority):
    registry_id: str
    kind: SkillSourceAuthorityKind = (
        SkillSourceAuthorityKind.PREINSTALLED_REMOTE
    )

    def __post_init__(self) -> None:
        SkillSourceAuthority.__post_init__(self)
        _assert_authority_kind(
            self.kind, SkillSourceAuthorityKind.PREINSTALLED_REMOTE
        )
        _assert_logical_id(self.registry_id, "registry_id")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value = SkillSourceAuthority.as_model_dict(self)
        value["registry_id"] = self.registry_id
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillDiagnosticInfo:
    code: SkillDiagnosticCode
    status: SkillStatus
    message: str
    path: str
    hint: str
    candidates: tuple[str, ...] = ()
    details: SkillModelMapping = field(default_factory=_empty_model_mapping)

    def __post_init__(self) -> None:
        assert isinstance(self.code, SkillDiagnosticCode)
        assert isinstance(self.status, SkillStatus)
        _assert_non_empty_model_text(self.message, "message")
        _assert_diagnostic_path(self.path)
        _assert_non_empty_model_text(self.hint, "hint")
        _assert_logical_id_tuple(self.candidates, "candidates")
        object.__setattr__(
            self, "details", _freeze_model_mapping(self.details)
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "code": self.code.value,
            "status": self.status.value,
            "message": self.message,
            "path": self.path,
            "hint": self.hint,
        }
        if self.candidates:
            value["candidates"] = self.candidates
        if self.details:
            value["details"] = self.details
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillSourceConfig:
    label: str
    authority: SkillSourceAuthority
    root_path: str | Path | None = None
    package_path: str | None = None
    enabled: bool = True
    allow_hidden_paths: bool = False
    status: SkillStatus = SkillStatus.OK
    tags: tuple[str, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        _assert_source_label(self.label)
        assert isinstance(self.authority, SkillSourceAuthority)
        if self.root_path is not None:
            assert isinstance(
                self.root_path, str | Path
            ), "root_path must be a path"
        if self.package_path is not None:
            assert isinstance(
                self.package_path, str
            ), "package_path must be a string"
            assert self.package_path.strip(), "package_path must be non-empty"
        assert isinstance(self.enabled, bool)
        assert isinstance(self.allow_hidden_paths, bool)
        assert isinstance(self.status, SkillStatus)
        if self.enabled:
            assert self.status != SkillStatus.DISABLED
        else:
            assert self.status == SkillStatus.DISABLED
        _assert_logical_id_tuple(self.tags, "tags")
        _assert_diagnostic_tuple(self.diagnostics)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "label": self.label,
            "authority": self.authority.as_model_dict(),
            "enabled": self.enabled,
            "allow_hidden_paths": self.allow_hidden_paths,
            "status": self.status.value,
        }
        if self.package_path is not None:
            value["package_path"] = _model_safe_package_path(self.package_path)
        if self.tags:
            value["tags"] = self.tags
        if self.diagnostics:
            value["diagnostics"] = tuple(
                diagnostic.as_model_dict() for diagnostic in self.diagnostics
            )
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResourceHandle:
    source_label: str
    skill_id: str
    resource_id: str
    media_type: str = "text/markdown"
    size_bytes: int | None = None
    status: SkillStatus = SkillStatus.OK
    stale: bool = False

    def __post_init__(self) -> None:
        _assert_source_label(self.source_label)
        _assert_skill_id(self.skill_id, "skill_id")
        _assert_resource_id(self.resource_id, "resource_id")
        _assert_media_type(self.media_type)
        if self.size_bytes is not None:
            _assert_non_negative_int(self.size_bytes, "size_bytes")
        assert isinstance(self.status, SkillStatus)
        assert isinstance(self.stale, bool)
        if self.stale:
            assert self.status == SkillStatus.STALE
        if self.status == SkillStatus.STALE:
            assert self.stale, "stale status requires stale=True"

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "source_label": self.source_label,
            "skill_id": self.skill_id,
            "resource_id": self.resource_id,
            "media_type": self.media_type,
            "status": self.status.value,
            "stale": self.stale,
        }
        if self.size_bytes is not None:
            value["size_bytes"] = self.size_bytes
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMetadata:
    skill_id: str
    name: str
    description: str
    source_label: str
    main_resource_id: str = "main"
    enabled: bool = True
    status: SkillStatus = SkillStatus.OK
    tags: tuple[str, ...] = ()
    version: str | None = None
    resources: tuple[SkillResourceHandle, ...] = ()

    def __post_init__(self) -> None:
        _assert_skill_id(self.skill_id, "skill_id")
        assert self.name == self.skill_id
        _assert_non_empty_model_text(self.description, "description")
        _assert_source_label(self.source_label)
        _assert_resource_id(self.main_resource_id, "main_resource_id")
        assert isinstance(self.enabled, bool)
        assert isinstance(self.status, SkillStatus)
        if self.enabled:
            assert self.status != SkillStatus.DISABLED
        else:
            assert self.status == SkillStatus.DISABLED
        _assert_logical_id_tuple(self.tags, "tags")
        if self.version is not None:
            _assert_non_empty_model_text(self.version, "version")
        _assert_resource_tuple(self.resources)
        for resource in self.resources:
            assert resource.skill_id == self.skill_id
            assert resource.source_label == self.source_label
        if self.resources:
            assert any(
                resource.resource_id == self.main_resource_id
                for resource in self.resources
            ), "resources must include main_resource_id"

    def as_model_dict(self) -> dict[str, SkillModelValue]:
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
            value["resources"] = tuple(
                resource.as_model_dict() for resource in self.resources
            )
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResourceContent:
    handle: SkillResourceHandle
    text: str
    start_byte: int = 0
    end_byte: int = 0
    truncated: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.handle, SkillResourceHandle)
        _assert_model_text(self.text, "text")
        _assert_non_negative_int(self.start_byte, "start_byte")
        _assert_non_negative_int(self.end_byte, "end_byte")
        assert self.end_byte >= self.start_byte
        assert isinstance(self.truncated, bool)
        content_bytes = self.text.encode("utf-8")
        span_bytes = self.end_byte - self.start_byte
        assert span_bytes == len(
            content_bytes
        ), "content byte range must match text bytes"
        if self.truncated:
            assert self.text, "truncated content must be non-empty"

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "handle": self.handle.as_model_dict(),
                "text": self.text,
                "start_byte": self.start_byte,
                "end_byte": self.end_byte,
                "truncated": self.truncated,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillReadCursor:
    cursor_id: str
    registry_version: SkillRegistryVersion
    source_label: str
    skill_id: str
    resource_id: str
    offset_bytes: int
    limit_bytes: int

    def __post_init__(self) -> None:
        _assert_opaque_id(self.cursor_id, "cursor_id")
        assert isinstance(self.registry_version, SkillRegistryVersion)
        _assert_source_label(self.source_label)
        _assert_skill_id(self.skill_id, "skill_id")
        _assert_resource_id(self.resource_id, "resource_id")
        _assert_non_negative_int(self.offset_bytes, "offset_bytes")
        _assert_positive_int(self.limit_bytes, "limit_bytes")

    def as_model_value(self) -> str:
        return self.cursor_id

    def as_internal_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "cursor_id": self.cursor_id,
                "registry_version": self.registry_version.as_model_value(),
                "source_label": self.source_label,
                "skill_id": self.skill_id,
                "resource_id": self.resource_id,
                "offset_bytes": self.offset_bytes,
                "limit_bytes": self.limit_bytes,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMatchResult:
    metadata: SkillMetadata
    score: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.metadata, SkillMetadata)
        assert isinstance(self.score, int | float) and not isinstance(
            self.score, bool
        )
        assert isfinite(self.score)
        assert 0 <= self.score <= 1
        object.__setattr__(self, "score", float(self.score))
        _assert_model_text_tuple(self.reasons, "reasons")

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "metadata": self.metadata.as_model_dict(),
            "score": self.score,
        }
        if self.reasons:
            value["reasons"] = self.reasons
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillProvenance:
    registry_version: SkillRegistryVersion
    source_label: str
    skill_id: str
    resource_id: str
    authority: SkillSourceAuthorityKind
    content_sha256_prefix: str | None = None
    truncated: bool = False
    declared_follow_up_resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.registry_version, SkillRegistryVersion)
        _assert_source_label(self.source_label)
        _assert_skill_id(self.skill_id, "skill_id")
        _assert_resource_id(self.resource_id, "resource_id")
        assert isinstance(self.authority, SkillSourceAuthorityKind)
        if self.content_sha256_prefix is not None:
            assert (
                fullmatch(r"[a-f0-9]{8,64}", self.content_sha256_prefix)
                is not None
            ), "content_sha256_prefix must be a SHA-256 hex prefix"
        assert isinstance(self.truncated, bool)
        _assert_resource_id_tuple(
            self.declared_follow_up_resources,
            "declared_follow_up_resources",
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {
            "registry_version": self.registry_version.as_model_value(),
            "source_label": self.source_label,
            "skill_id": self.skill_id,
            "resource_id": self.resource_id,
            "authority": self.authority.value,
            "truncated": self.truncated,
        }
        if self.content_sha256_prefix is not None:
            value["content_sha256_prefix"] = self.content_sha256_prefix
        if self.declared_follow_up_resources:
            value["declared_follow_up_resources"] = (
                self.declared_follow_up_resources
            )
        return model_dict(value)


def diagnostic_from_failure(
    failure_mode: SkillFailureMode,
    *,
    path: str,
    candidates: tuple[str, ...] = (),
    details: SkillModelMapping = _empty_model_mapping(),
) -> SkillDiagnosticInfo:
    assert isinstance(failure_mode, SkillFailureMode)
    contract = diagnostic_contract_for_failure(failure_mode)
    return SkillDiagnosticInfo(
        code=contract.code,
        status=contract.status,
        message=contract.message,
        path=path,
        hint=contract.hint,
        candidates=tuple(sorted(candidates)),
        details=details,
    )


def model_dict(value: Mapping[str, object]) -> dict[str, SkillModelValue]:
    assert isinstance(value, Mapping), "value must be a mapping"
    model_value = to_model_value(value)
    assert isinstance(model_value, dict)
    dumps(model_value, allow_nan=False, sort_keys=True)
    return model_value


def to_model_value(value: object) -> SkillModelValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        _assert_model_text(value, "value")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        assert isfinite(value), "float values must be finite"
        return value
    if isinstance(value, list | tuple):
        return tuple(to_model_value(item) for item in value)
    if isinstance(value, set | frozenset):
        items = tuple(to_model_value(item) for item in value)
        return tuple(sorted(items, key=_model_sort_key))
    if isinstance(value, Mapping):
        result: dict[str, SkillModelValue] = {}
        for key, item in value.items():
            _assert_model_key(key)
            result[key] = to_model_value(item)
        return result
    raise AssertionError("value must be JSON-safe")


def _freeze_model_mapping(value: Mapping[str, object]) -> SkillModelMapping:
    assert isinstance(value, Mapping), "details must be a mapping"
    frozen: dict[str, SkillModelValue] = {}
    for key, item in value.items():
        _assert_model_key(key)
        frozen[key] = _freeze_model_value(item)
    return MappingProxyType(frozen)


def _freeze_model_value(value: object) -> SkillModelValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        _assert_model_text(value, "value")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        assert isfinite(value), "float values must be finite"
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze_model_value(item) for item in value)
    if isinstance(value, set | frozenset):
        items = tuple(_freeze_model_value(item) for item in value)
        return tuple(sorted(items, key=_model_sort_key))
    if isinstance(value, Mapping):
        frozen: dict[str, SkillModelValue] = {}
        for key, item in value.items():
            _assert_model_key(key)
            frozen[key] = _freeze_model_value(item)
        return MappingProxyType(frozen)
    raise AssertionError("value must be JSON-safe")


def _model_sort_key(value: SkillModelValue) -> str:
    return dumps(to_model_value(value), allow_nan=False, sort_keys=True)


def _assert_authority_kind(
    value: SkillSourceAuthorityKind, expected: SkillSourceAuthorityKind
) -> None:
    assert value is expected, f"authority kind must be {expected.value}"


def _assert_registry_version(value: str) -> None:
    assert isinstance(value, str), "registry_version must be a string"
    assert fullmatch(
        r"skills(?:-[a-z0-9]+)*:[a-z0-9][a-z0-9._:-]{5,95}",
        value,
    ), "registry_version must be a logical skills version"


def _assert_model_key(value: object) -> None:
    assert isinstance(value, str), "model mapping keys must be strings"
    assert fullmatch(
        r"[a-z][a-z0-9_]*", value
    ), "model mapping keys must be logical names"


def _assert_diagnostic_path(value: str) -> None:
    _assert_non_empty_string(value, "path")
    assert "\x00" not in value
    assert "\\" not in value
    assert "/" not in value
    assert not value.startswith(("~", "$"))
    parts = value.split(".")
    assert parts[0] in {
        "skills",
        "settings",
        "source",
        "manifest",
        "resource",
        "registry",
    }, "path must start with a public skills root"
    assert all(
        fullmatch(r"[a-z0-9][a-z0-9_-]*", part) is not None for part in parts
    ), "path must be a public diagnostic path"


def _assert_source_label(value: str) -> None:
    _assert_non_empty_string(value, "source_label")
    assert _is_logical_id(value), "source_label must be a logical label"


def _assert_skill_id(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert _is_logical_id(value), f"{field_name} must be a logical skill ID"


def _assert_logical_id(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert _is_logical_id(value), f"{field_name} must be a logical ID"


def _assert_opaque_id(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert fullmatch(
        r"[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*", value
    ), f"{field_name} must be an opaque logical ID"


def _assert_media_type(value: str) -> None:
    _assert_non_empty_string(value, "media_type")
    assert fullmatch(
        r"[a-z][a-z0-9.+-]*/[a-z][a-z0-9.+-]*", value
    ), "media_type must be a safe MIME type"


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must be non-empty"


def _assert_logical_id_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_logical_id(value, field_name)
    _assert_unique(values, field_name)


def _assert_model_text_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert_non_empty_model_safe_text_tuple(values, field_name)


def _assert_diagnostic_tuple(values: tuple[SkillDiagnosticInfo, ...]) -> None:
    assert_tuple_items(values, "diagnostics", SkillDiagnosticInfo)


def _assert_resource_tuple(values: tuple[SkillResourceHandle, ...]) -> None:
    assert_tuple_items(values, "resources", SkillResourceHandle)


def _assert_unique(values: tuple[str, ...], field_name: str) -> None:
    assert_unique_sequence(values, field_name)


def _is_logical_id(value: str) -> bool:
    return fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", value) is not None


def _model_safe_package_path(value: str) -> str:
    if value == ".":
        return "."
    if not _is_resource_id(value):
        return "redacted"
    return value
