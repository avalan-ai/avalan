from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .backend import ContainerAsyncBackend
from .conformance import (
    ContainerBackend,
    ContainerExecutionScope,
    ContainerSurface,
)
from .settings import (
    ContainerAuditPolicy,
    ContainerCleanupPolicy,
    ContainerDevicePolicy,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerEscalationPolicy,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkPolicy,
    ContainerOutputPolicy,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerPullPolicy,
    ContainerResourceLimits,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerTrustLevel,
    ContainerWorkspaceMapping,
)

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast, final


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerToolRuntimeSettings:
    effective_settings: ContainerEffectiveSettings | None = None
    backend: ContainerAsyncBackend | None = None
    opt_in_backends: Sequence[ContainerBackend | str] = field(
        default_factory=tuple,
    )
    rootful_authorized: bool = False
    authorization_provider: Callable[[object], object] | None = None
    secret_resolver: Callable[[str], object] | None = None
    audit_listeners: Sequence[Callable[[object], object]] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        if self.effective_settings is not None:
            assert isinstance(
                self.effective_settings,
                ContainerEffectiveSettings,
            )
        if self.backend is not None:
            assert isinstance(self.backend, ContainerAsyncBackend)
        object.__setattr__(
            self,
            "opt_in_backends",
            tuple(
                ContainerBackend(backend) for backend in self.opt_in_backends
            ),
        )
        _assert_bool(self.rootful_authorized, "rootful_authorized")
        if self.authorization_provider is not None:
            assert callable(
                self.authorization_provider
            ), "authorization_provider must be callable"
        if self.secret_resolver is not None:
            assert callable(
                self.secret_resolver
            ), "secret_resolver must be callable"
        for listener in self.audit_listeners:
            assert callable(listener), "audit listeners must be callable"
        object.__setattr__(
            self,
            "audit_listeners",
            tuple(self.audit_listeners),
        )


def disabled_required_container_settings(
    surface: ContainerSurface | str = ContainerSurface.SDK,
) -> ContainerEffectiveSettings:
    source = ContainerSettingsSource(
        surface=surface,
        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
    )
    return ContainerSettings(source=source).select_profile(
        ContainerProfileSelection(required=True)
    )


def trusted_container_settings_from_mapping(
    raw: Mapping[str, object],
    *,
    source: ContainerSettingsSource,
) -> ContainerSettings:
    assert isinstance(source, ContainerSettingsSource)
    normalized = _normalize_container_settings_mapping(raw)
    return ContainerSettings.from_dict(normalized, source=source)


def container_selection_from_mapping(
    raw: Mapping[str, object],
    *,
    source: ContainerSettingsSource,
    scope: ContainerExecutionScope | str = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ),
) -> ContainerProfileSelection:
    assert isinstance(source, ContainerSettingsSource)
    normalized = dict(raw)
    normalized.setdefault("scope", _scope_value(scope))
    return ContainerProfileSelection.from_dict(normalized, source=source)


def trusted_container_runtime_from_mapping(
    raw: Mapping[str, object],
    *,
    source: ContainerSettingsSource,
    selection: ContainerProfileSelection | None = None,
) -> ContainerToolRuntimeSettings:
    settings = trusted_container_settings_from_mapping(raw, source=source)
    effective = settings.select_profile(
        selection or ContainerProfileSelection()
    )
    return ContainerToolRuntimeSettings(
        effective_settings=effective,
        rootful_authorized=source.can_define_runtime_authority,
    )


def trusted_container_source(
    surface: ContainerSurface | str,
) -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=surface,
        trust_level=ContainerTrustLevel.TRUSTED_DEPLOYMENT,
    )


_SETTINGS_KEYS = {
    "backend",
    "default_profile",
    "allowed_profiles",
    "profiles",
    "profile_registry_id",
    "policy_version",
}
_PROFILE_KEYS = {
    "name",
    "image",
    "digest",
    "pull_policy",
    "build_policy",
    "platform",
    "workspace",
    "workspace_root",
    "container_workspace",
    "working_directory",
    "mounts",
    "environment",
    "secrets",
    "network",
    "devices",
    "resources",
    "output",
    "cleanup",
    "pooling",
    "audit",
    "escalation",
    "review_mode",
    "command_mode",
    "read_only_rootfs",
    "user",
}
_MOUNT_KEYS = {
    "source",
    "target",
    "access",
    "mount_type",
    "type",
}


def _normalize_container_settings_mapping(
    raw: Mapping[str, object],
) -> dict[str, object]:
    _assert_mapping(raw, "container")
    _assert_known_keys(raw, _SETTINGS_KEYS, "container")
    profiles = {
        name: _normalize_profile_mapping(name, value).to_dict()
        for name, value in _mapping(
            raw.get("profiles", {}),
            "container.profiles",
        ).items()
    }
    return {
        "backend": _optional_string(raw, "backend", ContainerBackend.NONE),
        "default_profile": raw.get("default_profile"),
        "allowed_profiles": raw.get("allowed_profiles", ()),
        "profiles": profiles,
        "profile_registry_id": raw.get("profile_registry_id", "default"),
        "policy_version": raw.get("policy_version", "phase11"),
    }


def _normalize_profile_mapping(
    name: object,
    value: object,
) -> ContainerProfile:
    _assert_non_empty_string(name, "profile name")
    profile_name = cast(str, name)
    raw = _mapping(value, f"container.profiles.{name}")
    _assert_known_keys(raw, _PROFILE_KEYS, f"container.profiles.{name}")
    if "name" in raw:
        assert (
            raw["name"] == profile_name
        ), "profile table name must match name"
    workspace = _workspace_mapping(raw)
    return ContainerProfile(
        name=profile_name,
        image=_image_policy(raw),
        workspace=workspace,
        mounts=_profile_mounts(
            raw,
            workspace,
        ),
        environment=ContainerEnvironmentPolicy.from_dict(
            _mapping(raw.get("environment", {}), "environment")
        ),
        secrets=tuple(
            _secret_mapping(item)
            for item in _sequence(raw.get("secrets", ()), "secrets")
        ),
        network=_network_policy(raw.get("network", {})),
        devices=ContainerDevicePolicy.from_dict(
            _mapping(raw.get("devices", {}), "devices")
        ),
        resources=ContainerResourceLimits.from_dict(
            _mapping(raw.get("resources", {}), "resources")
        ),
        output=ContainerOutputPolicy.from_dict(
            _mapping(raw.get("output", {}), "output")
        ),
        cleanup=ContainerCleanupPolicy.from_dict(
            _mapping(raw.get("cleanup", {}), "cleanup")
        ),
        audit=ContainerAuditPolicy.from_dict(
            _mapping(raw.get("audit", {}), "audit")
        ),
        escalation=_escalation_policy(raw),
        command_mode=_optional_string(
            raw,
            "command_mode",
            "fixed_executable",
        ),
        read_only_rootfs=_optional_bool(raw, "read_only_rootfs", True),
        user=_optional_string(raw, "user", "1000:1000"),
    )


def _image_policy(raw: Mapping[str, object]) -> ContainerImagePolicy:
    image = raw.get("image")
    if isinstance(image, Mapping):
        return ContainerImagePolicy.from_dict(image)
    assert isinstance(image, str), "profile image must be a string or mapping"
    return ContainerImagePolicy(
        reference=image,
        digest=_optional_raw_string(raw, "digest"),
        pull_policy=_optional_string(
            raw,
            "pull_policy",
            ContainerPullPolicy.NEVER,
        ),
        build_policy=_optional_string(raw, "build_policy", "disabled"),
        platform=_optional_string(raw, "platform", "linux/amd64"),
    )


def _workspace_mapping(
    raw: Mapping[str, object],
) -> ContainerWorkspaceMapping:
    workspace = raw.get("workspace")
    if isinstance(workspace, Mapping):
        return ContainerWorkspaceMapping.from_dict(workspace)
    if workspace is not None:
        assert isinstance(
            workspace, str
        ), "workspace must be string or mapping"
    container_path = workspace or raw.get("container_workspace", "/workspace")
    assert isinstance(
        container_path, str
    ), "container workspace must be string"
    return ContainerWorkspaceMapping(
        host_root=_optional_string(raw, "workspace_root", "."),
        container_path=container_path,
        working_directory=_optional_string(
            raw,
            "working_directory",
            container_path,
        ),
    )


def _mount_declaration(
    value: object,
    workspace_path: str,
) -> ContainerMountDeclaration:
    raw = _mapping(value, "mount")
    _assert_known_keys(raw, _MOUNT_KEYS, "mount")
    target = _required_string(raw, "target")
    mount_type = raw.get("mount_type", raw.get("type"))
    if mount_type is None:
        mount_type = (
            ContainerMountType.WORKSPACE.value
            if target == workspace_path
            else ContainerMountType.INPUT.value
        )
    assert isinstance(mount_type, str), "mount type must be a string"
    return ContainerMountDeclaration(
        source=_optional_raw_string(raw, "source"),
        target=target,
        mount_type=mount_type,
        access=_optional_string(raw, "access", ContainerMountAccess.READ),
    )


def _profile_mounts(
    raw: Mapping[str, object],
    workspace: ContainerWorkspaceMapping,
) -> tuple[ContainerMountDeclaration, ...]:
    explicit_mounts = tuple(
        _mount_declaration(item, workspace.container_path)
        for item in _sequence(raw.get("mounts", ()), "mounts")
    )
    if any(
        mount.target == workspace.container_path for mount in explicit_mounts
    ):
        return explicit_mounts
    return (
        ContainerMountDeclaration(
            source=workspace.host_root,
            target=workspace.container_path,
            mount_type=ContainerMountType.WORKSPACE,
            access=ContainerMountAccess.READ,
        ),
        *explicit_mounts,
    )


def _secret_mapping(value: object) -> ContainerSecretReference:
    return ContainerSecretReference.from_dict(_mapping(value, "secret"))


def _network_policy(value: object) -> ContainerNetworkPolicy:
    if isinstance(value, str):
        return ContainerNetworkPolicy(mode=value)
    return ContainerNetworkPolicy.from_dict(_mapping(value, "network"))


def _escalation_policy(raw: Mapping[str, object]) -> ContainerEscalationPolicy:
    if "escalation" in raw:
        return ContainerEscalationPolicy.from_dict(
            _mapping(raw["escalation"], "escalation")
        )
    if "review_mode" in raw:
        return ContainerEscalationPolicy(
            mode=_required_string(raw, "review_mode")
        )
    return ContainerEscalationPolicy()


def _assert_mapping(value: object, field_name: str) -> None:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"


def _assert_known_keys(
    raw: Mapping[str, object],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    assert not unknown, f"Unknown container fields in {field_name}: {unknown}"


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    _assert_mapping(value, field_name)
    return cast(Mapping[str, object], value)


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if value is None:
        return ()
    assert not isinstance(value, str), f"{field_name} must be a sequence"
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    return value


def _required_string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    assert isinstance(value, str) and value, f"{key} must be a string"
    return value


def _optional_string(
    raw: Mapping[str, object],
    key: str,
    default: object,
) -> str:
    value = raw.get(key, default)
    if isinstance(
        value,
        (
            ContainerBackend,
            ContainerMountAccess,
            ContainerPullPolicy,
        ),
    ):
        return value.value
    assert isinstance(value, str), f"{key} must be a string"
    return value


def _optional_raw_string(
    raw: Mapping[str, object],
    key: str,
) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    assert isinstance(value, str), f"{key} must be a string"
    return value


def _optional_bool(
    raw: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = raw.get(key, default)
    _assert_bool(value, key)
    return cast(bool, value)


def _scope_value(scope: ContainerExecutionScope | str) -> str:
    if isinstance(scope, ContainerExecutionScope):
        return scope.value
    _assert_non_empty_string(scope, "scope")
    return scope
