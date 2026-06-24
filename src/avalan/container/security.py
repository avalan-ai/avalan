from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_env_name as _assert_env_name,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .settings import (
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerEnvironmentPolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerProfile,
    ContainerResourceLimits,
    ContainerSecretReference,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path
from posixpath import normpath as normalize_posix_path
from stat import S_ISDIR, S_ISREG
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

_CREDENTIAL_COMPONENTS = {
    ".aws",
    ".azure",
    ".config",
    ".docker",
    ".gnupg",
    ".kube",
    ".ssh",
}
_CREDENTIAL_FILENAMES = {
    ".env",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
}
_PSEUDO_FILESYSTEMS = {
    "/dev",
    "/proc",
    "/sys",
}
_SENSITIVE_PATHS = {
    "/etc",
    "/private/etc",
    "/private/var/db",
    "/private/var/root",
    "/root",
    "/var/db",
    "/var/root",
}
_RUNTIME_SOCKET_SOURCES = {
    "/private/var/run/docker.sock",
    "/private/var/run/podman/podman.sock",
    "/run/docker.sock",
    "/run/podman/podman.sock",
    "/var/run/docker.sock",
    "/var/run/podman/podman.sock",
}
_SECRET_ENV_MARKERS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PRIVATE_KEY",
)
_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata",
}


class ContainerHostPathKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    MISSING = "missing"


class ContainerResourceControl(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    PIDS = "pids"
    TIMEOUT = "timeout"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerHostPathPolicy:
    allowed_roots: Sequence[str]
    runtime_shared_roots: Sequence[str] = field(default_factory=tuple)
    scratch_root: str | None = None
    output_root: str | None = None
    cache_root: str | None = None
    allow_hidden_paths: bool = False
    allow_vcs_internals: bool = False
    allow_runtime_sockets: bool = False

    def __post_init__(self) -> None:
        allowed_roots = _root_tuple(self.allowed_roots, "allowed_roots")
        runtime_shared_roots = _root_tuple(
            self.runtime_shared_roots or allowed_roots,
            "runtime_shared_roots",
        )
        _assert_bool(self.allow_hidden_paths, "allow_hidden_paths")
        _assert_bool(self.allow_vcs_internals, "allow_vcs_internals")
        _assert_bool(self.allow_runtime_sockets, "allow_runtime_sockets")
        object.__setattr__(self, "allowed_roots", allowed_roots)
        object.__setattr__(
            self,
            "runtime_shared_roots",
            runtime_shared_roots,
        )
        object.__setattr__(
            self,
            "scratch_root",
            _optional_policy_root(
                self.scratch_root,
                f"{allowed_roots[0]}/avalan-scratch",
                "scratch_root",
            ),
        )
        object.__setattr__(
            self,
            "output_root",
            _optional_policy_root(
                self.output_root,
                f"{allowed_roots[0]}/avalan-output",
                "output_root",
            ),
        )
        object.__setattr__(
            self,
            "cache_root",
            _optional_policy_root(
                self.cache_root,
                f"{allowed_roots[0]}/avalan-cache",
                "cache_root",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerValidatedHostPath:
    original_path: str
    normalized_path: str
    allowed_root: str
    path_kind: ContainerHostPathKind | str

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.original_path, "original_path")
        _assert_non_empty_string(self.normalized_path, "normalized_path")
        _assert_non_empty_string(self.allowed_root, "allowed_root")
        object.__setattr__(
            self,
            "path_kind",
            _enum_value(self.path_kind, ContainerHostPathKind, "path_kind"),
        )

    def to_dict(self) -> dict[str, str]:
        path_kind = cast(ContainerHostPathKind, self.path_kind)
        return {
            "original_path": self.original_path,
            "normalized_path": self.normalized_path,
            "allowed_root": self.allowed_root,
            "path_kind": path_kind.value,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPlannedMount:
    source: ContainerValidatedHostPath
    target: str
    mount_type: ContainerMountType | str
    access: ContainerMountAccess | str

    def __post_init__(self) -> None:
        assert isinstance(self.source, ContainerValidatedHostPath)
        _assert_container_path(self.target, "target")
        object.__setattr__(
            self,
            "mount_type",
            _enum_value(self.mount_type, ContainerMountType, "mount_type"),
        )
        object.__setattr__(
            self,
            "access",
            _enum_value(self.access, ContainerMountAccess, "access"),
        )

    @property
    def read_only(self) -> bool:
        access = cast(ContainerMountAccess, self.access)
        return access is ContainerMountAccess.READ

    def to_dict(self) -> dict[str, object]:
        mount_type = cast(ContainerMountType, self.mount_type)
        access = cast(ContainerMountAccess, self.access)
        return {
            "source": self.source.to_dict(),
            "target": self.target,
            "mount_type": mount_type.value,
            "access": access.value,
            "read_only": self.read_only,
            "redacted_source": redact_host_path(self.source.normalized_path),
        }

    def to_redacted_dict(self) -> dict[str, object]:
        mount_type = cast(ContainerMountType, self.mount_type)
        access = cast(ContainerMountAccess, self.access)
        path_kind = cast(ContainerHostPathKind, self.source.path_kind)
        return {
            "source": redact_host_path(self.source.normalized_path),
            "path_kind": path_kind.value,
            "target": self.target,
            "mount_type": mount_type.value,
            "access": access.value,
            "read_only": self.read_only,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerMountPlan:
    mounts: Sequence[ContainerPlannedMount]

    def __post_init__(self) -> None:
        mounts = tuple(self.mounts)
        for mount in mounts:
            assert isinstance(mount, ContainerPlannedMount)
        object.__setattr__(
            self,
            "mounts",
            tuple(sorted(mounts, key=lambda mount: mount.target)),
        )

    def to_dict(self) -> dict[str, object]:
        return {"mounts": [mount.to_dict() for mount in self.mounts]}

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "mounts": [mount.to_redacted_dict() for mount in self.mounts],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerEnvironmentPlan:
    variables: Mapping[str, str]

    def __post_init__(self) -> None:
        variables = _string_mapping(self.variables, "variables")
        object.__setattr__(
            self,
            "variables",
            MappingProxyType(dict(sorted(variables.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        return {"variables": dict(self.variables)}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSecretPolicy:
    allowed_secret_names: Sequence[str] = field(default_factory=tuple)
    allow_env_delivery: bool = True
    allow_mount_delivery: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_secret_names",
            _string_tuple(self.allowed_secret_names, "allowed_secret_names"),
        )
        _assert_bool(self.allow_env_delivery, "allow_env_delivery")
        _assert_bool(self.allow_mount_delivery, "allow_mount_delivery")


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSecretDelivery:
    name: str
    env_name: str | None = None
    mount_path: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        if self.env_name is not None:
            _assert_env_name(self.env_name, "env_name")
        if self.mount_path is not None:
            _assert_container_path(self.mount_path, "mount_path")
        assert (
            self.env_name is not None or self.mount_path is not None
        ), "secret delivery requires env or mount target"

    def to_redacted_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "env_name": self.env_name,
            "mount_path": self.mount_path,
            "value": "<redacted>",
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSecretPlan:
    deliveries: Sequence[ContainerSecretDelivery]

    def __post_init__(self) -> None:
        deliveries = tuple(self.deliveries)
        for delivery in deliveries:
            assert isinstance(delivery, ContainerSecretDelivery)
        object.__setattr__(
            self,
            "deliveries",
            tuple(sorted(deliveries, key=lambda item: item.name)),
        )

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "deliveries": [
                delivery.to_redacted_dict() for delivery in self.deliveries
            ]
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerNetworkPolicyLimits:
    allowed_modes: Sequence[ContainerNetworkMode | str] = field(
        default_factory=lambda: (ContainerNetworkMode.NONE,),
    )
    allowed_egress_hosts: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_modes",
            tuple(
                _enum_value(mode, ContainerNetworkMode, "allowed_modes")
                for mode in self.allowed_modes
            ),
        )
        object.__setattr__(
            self,
            "allowed_egress_hosts",
            _string_tuple(self.allowed_egress_hosts, "allowed_egress_hosts"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerNetworkPlan:
    mode: ContainerNetworkMode | str
    egress_allowlist: Sequence[str] = field(default_factory=tuple)
    escalation_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ContainerNetworkMode, "mode"),
        )
        object.__setattr__(
            self,
            "egress_allowlist",
            tuple(sorted(_string_tuple(self.egress_allowlist, "egress"))),
        )
        _assert_bool(self.escalation_required, "escalation_required")

    def to_dict(self) -> dict[str, object]:
        mode = cast(ContainerNetworkMode, self.mode)
        return {
            "mode": mode.value,
            "egress_allowlist": list(self.egress_allowlist),
            "escalation_required": self.escalation_required,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerDevicePolicyLimits:
    allowed_devices: Sequence[ContainerDeviceClass | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_devices",
            tuple(
                _enum_value(device, ContainerDeviceClass, "allowed_devices")
                for device in self.allowed_devices
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerDevicePlan:
    devices: Sequence[ContainerDeviceClass | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "devices",
            tuple(
                sorted(
                    (
                        _enum_value(device, ContainerDeviceClass, "devices")
                        for device in self.devices
                    ),
                    key=lambda device: device.value,
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        devices = cast(tuple[ContainerDeviceClass, ...], self.devices)
        return {"devices": [device.value for device in devices]}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerResourcePolicy:
    supported_controls: Sequence[ContainerResourceControl | str] = field(
        default_factory=tuple,
    )
    required_controls: Sequence[ContainerResourceControl | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        supported = tuple(
            _enum_value(control, ContainerResourceControl, "supported")
            for control in self.supported_controls
        )
        required = tuple(
            _enum_value(control, ContainerResourceControl, "required")
            for control in self.required_controls
        )
        object.__setattr__(self, "supported_controls", supported)
        object.__setattr__(self, "required_controls", required)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerResourcePlan:
    limits: ContainerResourceLimits
    best_effort_unsupported: Sequence[ContainerResourceControl | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.limits, ContainerResourceLimits)
        object.__setattr__(
            self,
            "best_effort_unsupported",
            tuple(
                _enum_value(control, ContainerResourceControl, "unsupported")
                for control in self.best_effort_unsupported
            ),
        )

    def to_dict(self) -> dict[str, object]:
        unsupported = cast(
            tuple[ContainerResourceControl, ...],
            self.best_effort_unsupported,
        )
        return {
            "limits": self.limits.to_dict(),
            "best_effort_unsupported": [
                control.value for control in unsupported
            ],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProcessSecurityPlan:
    user: str
    privileged: bool = False
    capabilities: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.user, "user")
        _assert_bool(self.privileged, "privileged")
        object.__setattr__(
            self,
            "capabilities",
            _string_tuple(self.capabilities, "capabilities"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "user": self.user,
            "privileged": self.privileged,
            "capabilities": list(self.capabilities),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSecurityPlan:
    mounts: ContainerMountPlan
    environment: ContainerEnvironmentPlan
    secrets: ContainerSecretPlan
    network: ContainerNetworkPlan
    devices: ContainerDevicePlan
    resources: ContainerResourcePlan
    process: ContainerProcessSecurityPlan
    risk_tags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assert isinstance(self.mounts, ContainerMountPlan)
        assert isinstance(self.environment, ContainerEnvironmentPlan)
        assert isinstance(self.secrets, ContainerSecretPlan)
        assert isinstance(self.network, ContainerNetworkPlan)
        assert isinstance(self.devices, ContainerDevicePlan)
        assert isinstance(self.resources, ContainerResourcePlan)
        assert isinstance(self.process, ContainerProcessSecurityPlan)
        object.__setattr__(
            self,
            "risk_tags",
            tuple(sorted(_string_tuple(self.risk_tags, "risk_tags"))),
        )

    def to_redacted_dict(self) -> dict[str, object]:
        return {
            "mounts": self.mounts.to_redacted_dict(),
            "environment": self.environment.to_dict(),
            "secrets": self.secrets.to_redacted_dict(),
            "network": self.network.to_dict(),
            "devices": self.devices.to_dict(),
            "resources": self.resources.to_dict(),
            "process": self.process.to_dict(),
            "risk_tags": list(self.risk_tags),
        }


def validate_host_path(
    path: str,
    policy: ContainerHostPathPolicy,
    *,
    require_exists: bool = True,
) -> ContainerValidatedHostPath:
    assert isinstance(policy, ContainerHostPathPolicy)
    _assert_host_path_text(path, "path")
    normalized = normalize_posix_path(path)
    _assert_not_pseudo_filesystem(normalized)
    _assert_not_sensitive_path(normalized)
    assert (
        not _is_runtime_socket_path(normalized) or policy.allow_runtime_sockets
    ), "runtime socket mounts are denied"
    allowed_root = _matching_root(normalized, policy.allowed_roots)
    assert allowed_root is not None, "path is outside allowed roots"
    shared_root = _matching_root(normalized, policy.runtime_shared_roots)
    assert shared_root is not None, "path is not shared with runtime"
    relative_parts = _relative_parts(normalized, allowed_root)
    _assert_safe_relative_parts(relative_parts, policy)
    assert not _has_symlink_component(normalized), "symlink path denied"
    path_obj = Path(normalized)
    if require_exists:
        assert path_obj.exists(), "path must exist"
    path_kind = _path_kind(path_obj)
    if path_obj.exists():
        assert path_kind in {
            ContainerHostPathKind.FILE,
            ContainerHostPathKind.DIRECTORY,
        }, "special files are denied"
    return ContainerValidatedHostPath(
        original_path=path,
        normalized_path=normalized,
        allowed_root=allowed_root,
        path_kind=path_kind,
    )


def plan_container_mounts(
    profile: ContainerProfile,
    policy: ContainerHostPathPolicy,
) -> ContainerMountPlan:
    assert isinstance(profile, ContainerProfile)
    mounts: list[ContainerPlannedMount] = []
    for mount in profile.mounts:
        source = mount.source or _generated_mount_source(mount, policy)
        mount_type = cast(ContainerMountType, mount.mount_type)
        access = cast(ContainerMountAccess, mount.access)
        assert not (
            access is ContainerMountAccess.WRITE
            and mount_type
            not in {ContainerMountType.SCRATCH, ContainerMountType.OUTPUT}
        ), "write mounts must be scratch or output"
        require_exists = mount_type in {
            ContainerMountType.INPUT,
            ContainerMountType.WORKSPACE,
            ContainerMountType.SECRET,
        }
        mounts.append(
            ContainerPlannedMount(
                source=validate_host_path(
                    source,
                    policy,
                    require_exists=require_exists,
                ),
                target=mount.target,
                mount_type=mount_type,
                access=access,
            )
        )
    return ContainerMountPlan(mounts=mounts)


def plan_container_environment(
    environment: ContainerEnvironmentPolicy,
    host_environment: Mapping[str, str],
) -> ContainerEnvironmentPlan:
    assert isinstance(environment, ContainerEnvironmentPolicy)
    assert isinstance(host_environment, Mapping)
    assert not environment.inherit_host, "host environment inheritance denied"
    variables: dict[str, str] = {}
    for name, value in environment.variables.items():
        _assert_env_name(name, "environment variable")
        _assert_not_secret_env_name(name)
        _assert_non_empty_string(value, name)
        variables[name] = value
    for name in environment.allowlist:
        _assert_env_name(name, "environment allowlist")
        _assert_not_secret_env_name(name)
        if name in host_environment:
            variables[name] = host_environment[name]
    return ContainerEnvironmentPlan(variables=variables)


def plan_container_secrets(
    secrets: Sequence[ContainerSecretReference],
    policy: ContainerSecretPolicy,
) -> ContainerSecretPlan:
    assert isinstance(policy, ContainerSecretPolicy)
    allowed = set(policy.allowed_secret_names)
    deliveries: list[ContainerSecretDelivery] = []
    for secret in secrets:
        assert isinstance(secret, ContainerSecretReference)
        assert secret.name in allowed, "secret is not authorized"
        assert not (
            secret.env_name and not policy.allow_env_delivery
        ), "secret env delivery is denied"
        assert not (
            secret.mount_path and not policy.allow_mount_delivery
        ), "secret mount delivery is denied"
        deliveries.append(
            ContainerSecretDelivery(
                name=secret.name,
                env_name=secret.env_name,
                mount_path=secret.mount_path,
            )
        )
    return ContainerSecretPlan(deliveries=deliveries)


def validate_container_network(
    network: ContainerNetworkPolicy,
    policy: ContainerNetworkPolicyLimits,
) -> ContainerNetworkPlan:
    assert isinstance(network, ContainerNetworkPolicy)
    assert isinstance(policy, ContainerNetworkPolicyLimits)
    mode = cast(ContainerNetworkMode, network.mode)
    assert mode in set(policy.allowed_modes), "network mode is denied"
    allowed_hosts = set(policy.allowed_egress_hosts)
    for host in network.egress_allowlist:
        _assert_safe_egress_host(host)
        assert host in allowed_hosts, "network host is not authorized"
    return ContainerNetworkPlan(
        mode=mode,
        egress_allowlist=network.egress_allowlist,
        escalation_required=mode is not ContainerNetworkMode.NONE,
    )


def validate_container_devices(
    devices: ContainerDevicePolicy,
    policy: ContainerDevicePolicyLimits,
) -> ContainerDevicePlan:
    assert isinstance(devices, ContainerDevicePolicy)
    assert isinstance(policy, ContainerDevicePolicyLimits)
    allowed = set(policy.allowed_devices)
    requested = cast(tuple[ContainerDeviceClass, ...], devices.devices)
    for device in requested:
        assert device in allowed, "device class is denied"
    return ContainerDevicePlan(devices=requested)


def validate_container_resources(
    resources: ContainerResourceLimits,
    policy: ContainerResourcePolicy,
) -> ContainerResourcePlan:
    assert isinstance(resources, ContainerResourceLimits)
    assert isinstance(policy, ContainerResourcePolicy)
    unsupported: list[ContainerResourceControl] = []
    requested = _requested_resource_controls(resources)
    supported = set(policy.supported_controls)
    required = set(policy.required_controls)
    for control in requested:
        if control not in supported:
            assert (
                control not in required
            ), "required resource control is unsupported"
            unsupported.append(control)
    return ContainerResourcePlan(
        limits=resources,
        best_effort_unsupported=unsupported,
    )


def validate_container_process_security(
    plan: ContainerProcessSecurityPlan,
) -> ContainerProcessSecurityPlan:
    assert isinstance(plan, ContainerProcessSecurityPlan)
    assert plan.user != "0" and not plan.user.startswith(
        "0:"
    ), "root user is denied"
    assert not plan.privileged, "privileged containers are denied"
    assert not plan.capabilities, "extra capabilities are denied"
    return plan


def validate_container_security_profile(
    profile: ContainerProfile,
    *,
    path_policy: ContainerHostPathPolicy,
    secret_policy: ContainerSecretPolicy,
    network_policy: ContainerNetworkPolicyLimits,
    device_policy: ContainerDevicePolicyLimits,
    resource_policy: ContainerResourcePolicy,
    host_environment: Mapping[str, str],
) -> ContainerSecurityPlan:
    assert isinstance(profile, ContainerProfile)
    network = validate_container_network(profile.network, network_policy)
    secrets = plan_container_secrets(profile.secrets, secret_policy)
    risk_tags = _risk_tags(secrets, network)
    return ContainerSecurityPlan(
        mounts=plan_container_mounts(profile, path_policy),
        environment=plan_container_environment(
            profile.environment,
            host_environment,
        ),
        secrets=secrets,
        network=network,
        devices=validate_container_devices(profile.devices, device_policy),
        resources=validate_container_resources(
            profile.resources,
            resource_policy,
        ),
        process=validate_container_process_security(
            ContainerProcessSecurityPlan(user=profile.user)
        ),
        risk_tags=risk_tags,
    )


def redact_host_path(path: str) -> str:
    _assert_non_empty_string(path, "path")
    name = Path(path).name
    return f"<host-path>/{name}" if name else "<host-path>"


def _risk_tags(
    secrets: ContainerSecretPlan,
    network: ContainerNetworkPlan,
) -> tuple[str, ...]:
    tags: set[str] = set()
    if secrets.deliveries:
        tags.add("secret")
    if network.mode is not ContainerNetworkMode.NONE:
        tags.add("network")
    if len(tags) > 1:
        tags.add("combined_risk")
    return tuple(sorted(tags))


def _generated_mount_source(
    mount: ContainerMountDeclaration,
    policy: ContainerHostPathPolicy,
) -> str:
    mount_type = cast(ContainerMountType, mount.mount_type)
    target_name = mount.target.strip("/").replace("/", "_") or "root"
    if mount_type is ContainerMountType.SCRATCH:
        return f"{policy.scratch_root}/{target_name}"
    if mount_type is ContainerMountType.OUTPUT:
        return f"{policy.output_root}/{target_name}"
    if mount_type is ContainerMountType.CACHE:
        return f"{policy.cache_root}/{target_name}"
    assert False, "mount source is required"


def _requested_resource_controls(
    resources: ContainerResourceLimits,
) -> tuple[ContainerResourceControl, ...]:
    controls: list[ContainerResourceControl] = []
    if resources.cpu_count is not None:
        controls.append(ContainerResourceControl.CPU)
    if resources.memory_bytes is not None:
        controls.append(ContainerResourceControl.MEMORY)
    if resources.pids is not None:
        controls.append(ContainerResourceControl.PIDS)
    if resources.timeout_seconds is not None:
        controls.append(ContainerResourceControl.TIMEOUT)
    return tuple(controls)


def _path_kind(path: Path) -> ContainerHostPathKind:
    if not path.exists():
        return ContainerHostPathKind.MISSING
    mode = path.lstat().st_mode
    if S_ISDIR(mode):
        return ContainerHostPathKind.DIRECTORY
    if S_ISREG(mode):
        return ContainerHostPathKind.FILE
    return ContainerHostPathKind.MISSING


def _root_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    roots = _string_tuple(value, field_name)
    assert roots, f"{field_name} must not be empty"
    return tuple(_normalize_root(root, field_name) for root in roots)


def _normalize_root(path: str, field_name: str) -> str:
    _assert_host_path_text(path, field_name)
    return normalize_posix_path(path)


def _optional_policy_root(
    path: str | None,
    default: str,
    field_name: str,
) -> str:
    return _normalize_root(path or default, field_name)


def _assert_host_path_text(path: str, field_name: str) -> None:
    _assert_non_empty_string(path, field_name)
    assert "\x00" not in path, f"{field_name} must not contain NUL"
    assert not path.startswith("~"), f"{field_name} must not expand home"
    assert "$" not in path, f"{field_name} must not expand variables"
    assert "://" not in path, f"{field_name} must not be remote"
    assert not path.startswith("//"), f"{field_name} must not be remote"
    assert Path(path).is_absolute(), f"{field_name} must be absolute"
    assert ".." not in Path(path).parts, f"{field_name} must not traverse"


def _assert_not_pseudo_filesystem(path: str) -> None:
    for root in _PSEUDO_FILESYSTEMS:
        assert path != root and not path.startswith(
            f"{root}/"
        ), "pseudo-filesystem paths are denied"


def _assert_not_sensitive_path(path: str) -> None:
    for root in _SENSITIVE_PATHS:
        assert path != root and not path.startswith(
            f"{root}/"
        ), "sensitive host paths are denied"


def _matching_root(path: str, roots: Sequence[str]) -> str | None:
    path_obj = Path(path)
    for root in roots:
        root_obj = Path(root)
        if path_obj == root_obj or root_obj in path_obj.parents:
            return root
    return None


def _relative_parts(path: str, root: str) -> tuple[str, ...]:
    path_parts = Path(path).parts
    root_parts = Path(root).parts
    return tuple(path_parts[len(root_parts) :])


def _assert_safe_relative_parts(
    parts: Sequence[str],
    policy: ContainerHostPathPolicy,
) -> None:
    for part in parts:
        assert policy.allow_hidden_paths or not part.startswith(
            "."
        ), "hidden paths are denied"
        assert policy.allow_vcs_internals or part not in {
            ".git",
            ".hg",
            ".svn",
        }, "VCS internals are denied"
        assert (
            part not in _CREDENTIAL_COMPONENTS
        ), "credential locations are denied"
        assert part not in _CREDENTIAL_FILENAMES, "credential files are denied"


def _has_symlink_component(path: str) -> bool:
    candidate = Path("/")
    for part in Path(path).parts[1:]:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def _is_runtime_socket_path(path: str) -> bool:
    return normalize_posix_path(path) in _RUNTIME_SOCKET_SOURCES


def _assert_container_path(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert value.startswith("/"), f"{field_name} must be absolute"
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    assert ".." not in Path(value).parts, f"{field_name} must not traverse"


def _assert_not_secret_env_name(name: str) -> None:
    upper = name.upper()
    assert not any(
        marker in upper for marker in _SECRET_ENV_MARKERS
    ), "raw secret environment variables are denied"


def _assert_safe_egress_host(host: str) -> None:
    _assert_non_empty_string(host, "egress host")
    assert (
        "://" not in host and "/" not in host
    ), "redirect-style egress entries are denied"
    assert "*" not in host, "wildcard egress entries are denied"
    lowered = host.lower().rstrip(".")
    assert lowered not in _METADATA_HOSTS, "metadata endpoints are denied"
    assert lowered != "localhost" and not lowered.endswith(
        ".localhost"
    ), "localhost egress is denied"
    try:
        address = ip_address(lowered)
    except ValueError:
        assert "." in lowered, "single-label DNS egress is denied"
        return
    assert not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_unspecified
    ), "private or metadata IP egress is denied"


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    result: dict[str, str] = {}
    for key, item in value.items():
        _assert_env_name(key, field_name)
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
        result[key] = item
    return result


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(
        value, str
    ), f"{field_name} must be a sequence"
    result = tuple(value)
    for item in result:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
    return result


def _enum_value(
    value: object,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert value in {
        member.value for member in enum_type
    }, f"{field_name} contains unsupported value"
    return enum_type(value)
