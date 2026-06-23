from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_env_name as _assert_env_name,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_optional_positive_int as _assert_optional_positive_int,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .conformance import (
    ContainerBackend,
    ContainerExecutionScope,
    ContainerSurface,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from re import compile as compile_pattern
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

_PROFILE_NAME_PATTERN = compile_pattern(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DIGEST_PATTERN = compile_pattern(r"^sha256:[0-9a-f]{64}$")
_PLATFORM_PATTERN = compile_pattern(r"^[A-Za-z0-9_+.-]+/[A-Za-z0-9_+.-]+$")


class ContainerTrustLevel(StrEnum):
    TRUSTED_OPERATOR = "trusted_operator"
    TRUSTED_DEPLOYMENT = "trusted_deployment"
    UNTRUSTED_AGENT = "untrusted_agent"
    UNTRUSTED_FLOW = "untrusted_flow"
    UNTRUSTED_TASK = "untrusted_task"
    UNTRUSTED_REQUEST = "untrusted_request"
    MODEL = "model"


class ContainerPullPolicy(StrEnum):
    NEVER = "never"
    IF_MISSING = "if_missing"
    ALWAYS = "always"


class ContainerBuildPolicy(StrEnum):
    DISABLED = "disabled"
    TRUSTED_ONLY = "trusted_only"


class ContainerCommandMode(StrEnum):
    FIXED_ENTRYPOINT = "fixed_entrypoint"
    FIXED_EXECUTABLE = "fixed_executable"
    SERVICE_COMMAND = "service_command"


class ContainerMountType(StrEnum):
    INPUT = "input"
    WORKSPACE = "workspace"
    SCRATCH = "scratch"
    OUTPUT = "output"
    CACHE = "cache"
    SECRET = "secret"


class ContainerMountAccess(StrEnum):
    READ = "read"
    WRITE = "write"


class ContainerNetworkMode(StrEnum):
    NONE = "none"
    LOOPBACK = "loopback"
    ALLOWLIST = "allowlist"


class ContainerDeviceClass(StrEnum):
    CPU = "cpu"
    NVIDIA_CDI = "nvidia_cdi"
    AMD_CDI = "amd_cdi"
    VULKAN_FORWARDED = "vulkan_forwarded"


class ContainerCleanupMode(StrEnum):
    REMOVE = "remove"
    QUARANTINE = "quarantine"


class ContainerPoolingMode(StrEnum):
    DISABLED = "disabled"
    SHORT_LIVED = "short_lived"


class ContainerAuditMode(StrEnum):
    MINIMAL = "minimal"
    FULL = "full"


class ContainerEscalationMode(StrEnum):
    DENY = "deny"
    REQUIRE_REVIEW = "require_review"
    PREAUTHORIZED = "preauthorized"


class ContainerAuthorizationDecisionType(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRES_REVIEW = "requires_review"


class ContainerResultStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"


class ContainerAuditEventType(StrEnum):
    POLICY_EVALUATION = "policy_evaluation"
    REVIEW_REQUEST = "review_request"
    REVIEW_DECISION = "review_decision"
    BACKEND_SELECTION = "backend_selection"
    IMAGE_RESOLUTION = "image_resolution"
    IMAGE_PULL = "image_pull"
    BUILD_PROGRESS = "build_progress"
    MOUNT_PREPARATION = "mount_preparation"
    CONTAINER_CREATE = "container_create"
    CONTAINER_START = "container_start"
    STDOUT_CHUNK = "stdout_chunk"
    STDERR_CHUNK = "stderr_chunk"
    STREAM_CHUNK = "stream_chunk"
    PROGRESS = "progress"
    STATS = "stats"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    EXIT = "exit"
    OUTPUT_COPY = "output_copy"
    CLEANUP = "cleanup"
    DENIAL = "denial"
    FAILURE = "failure"
    SETTINGS_VALIDATED = "settings_validated"
    PLAN_CREATED = "plan_created"
    DECISION_RECORDED = "decision_recorded"
    RESULT_RECORDED = "result_recorded"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSettingsSource:
    surface: ContainerSurface | str
    trust_level: ContainerTrustLevel | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface",
            _enum_value(self.surface, ContainerSurface, "surface"),
        )
        object.__setattr__(
            self,
            "trust_level",
            _enum_value(
                self.trust_level,
                ContainerTrustLevel,
                "trust_level",
            ),
        )

    @property
    def can_define_runtime_authority(self) -> bool:
        return self.trust_level in {
            ContainerTrustLevel.TRUSTED_OPERATOR,
            ContainerTrustLevel.TRUSTED_DEPLOYMENT,
        }

    def to_dict(self) -> dict[str, str]:
        surface = cast(ContainerSurface, self.surface)
        trust_level = cast(ContainerTrustLevel, self.trust_level)
        return {
            "surface": surface.value,
            "trust_level": trust_level.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerSettingsSource":
        _assert_fields(raw, {"surface", "trust_level"}, "source")
        return cls(
            surface=_required_str(raw, "surface", "source"),
            trust_level=_required_str(raw, "trust_level", "source"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerImagePolicy:
    reference: str
    digest: str | None = None
    pull_policy: ContainerPullPolicy | str = ContainerPullPolicy.NEVER
    build_policy: ContainerBuildPolicy | str = ContainerBuildPolicy.DISABLED
    platform: str = "linux/amd64"

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.reference, "reference")
        digest = self.digest or _digest_from_reference(self.reference)
        assert digest is not None, "image reference must be digest pinned"
        _assert_digest(digest, "digest")
        _assert_platform(self.platform)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(
            self,
            "pull_policy",
            _enum_value(
                self.pull_policy,
                ContainerPullPolicy,
                "pull_policy",
            ),
        )
        object.__setattr__(
            self,
            "build_policy",
            _enum_value(
                self.build_policy,
                ContainerBuildPolicy,
                "build_policy",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        assert self.digest is not None
        pull_policy = cast(ContainerPullPolicy, self.pull_policy)
        build_policy = cast(ContainerBuildPolicy, self.build_policy)
        return {
            "reference": self.reference,
            "digest": self.digest,
            "pull_policy": pull_policy.value,
            "build_policy": build_policy.value,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerImagePolicy":
        _assert_fields(
            raw,
            {"reference", "digest", "pull_policy", "build_policy", "platform"},
            "image",
        )
        return cls(
            reference=_required_str(raw, "reference", "image"),
            digest=_optional_str(raw, "digest"),
            pull_policy=_optional_str_or_default(
                raw,
                "pull_policy",
                ContainerPullPolicy.NEVER.value,
            ),
            build_policy=_optional_str_or_default(
                raw,
                "build_policy",
                ContainerBuildPolicy.DISABLED.value,
            ),
            platform=_optional_str_or_default(
                raw,
                "platform",
                "linux/amd64",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerWorkspaceMapping:
    host_root: str = "."
    container_path: str = "/workspace"
    working_directory: str = "/workspace"

    def __post_init__(self) -> None:
        _assert_host_path(self.host_root, "host_root")
        _assert_container_path(self.container_path, "container_path")
        _assert_container_path(self.working_directory, "working_directory")

    def to_dict(self) -> dict[str, str]:
        return {
            "host_root": self.host_root,
            "container_path": self.container_path,
            "working_directory": self.working_directory,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerWorkspaceMapping":
        _assert_fields(
            raw,
            {"host_root", "container_path", "working_directory"},
            "workspace",
        )
        return cls(
            host_root=_optional_str_or_default(raw, "host_root", "."),
            container_path=_optional_str_or_default(
                raw,
                "container_path",
                "/workspace",
            ),
            working_directory=_optional_str_or_default(
                raw,
                "working_directory",
                "/workspace",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerMountDeclaration:
    target: str
    mount_type: ContainerMountType | str
    access: ContainerMountAccess | str = ContainerMountAccess.READ
    source: str | None = None

    def __post_init__(self) -> None:
        mount_type = _enum_value(
            self.mount_type,
            ContainerMountType,
            "mount_type",
        )
        access = _enum_value(self.access, ContainerMountAccess, "access")
        _assert_container_path(self.target, "target")
        if self.source is not None:
            _assert_host_path(self.source, "source")
        assert not (
            access is ContainerMountAccess.WRITE
            and mount_type
            not in {ContainerMountType.SCRATCH, ContainerMountType.OUTPUT}
        ), "only scratch and output mounts may be writable"
        object.__setattr__(self, "mount_type", mount_type)
        object.__setattr__(self, "access", access)

    def to_dict(self) -> dict[str, object]:
        mount_type = cast(ContainerMountType, self.mount_type)
        access = cast(ContainerMountAccess, self.access)
        return {
            "target": self.target,
            "mount_type": mount_type.value,
            "access": access.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerMountDeclaration":
        _assert_fields(
            raw,
            {"target", "mount_type", "access", "source"},
            "mount",
        )
        return cls(
            target=_required_str(raw, "target", "mount"),
            mount_type=_required_str(raw, "mount_type", "mount"),
            access=_optional_str_or_default(
                raw,
                "access",
                ContainerMountAccess.READ.value,
            ),
            source=_optional_str(raw, "source"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerEnvironmentPolicy:
    variables: Mapping[str, str] = field(default_factory=dict)
    allowlist: Sequence[str] = field(default_factory=tuple)
    inherit_host: bool = False

    def __post_init__(self) -> None:
        variables = _string_mapping(self.variables, "variables")
        allowlist = _string_tuple(self.allowlist, "allowlist")
        for name in variables:
            _assert_env_name(name, "variables")
        for name in allowlist:
            _assert_env_name(name, "allowlist")
        _assert_bool(self.inherit_host, "inherit_host")
        assert not self.inherit_host, "host environment inheritance is unsafe"
        object.__setattr__(self, "variables", MappingProxyType(variables))
        object.__setattr__(self, "allowlist", allowlist)

    def to_dict(self) -> dict[str, object]:
        return {
            "variables": dict(self.variables),
            "allowlist": list(self.allowlist),
            "inherit_host": self.inherit_host,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerEnvironmentPolicy":
        _assert_fields(raw, {"variables", "allowlist", "inherit_host"}, "env")
        return cls(
            variables=_string_mapping(raw.get("variables", {}), "variables"),
            allowlist=_string_tuple(raw.get("allowlist", ()), "allowlist"),
            inherit_host=_optional_bool(raw, "inherit_host") or False,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSecretReference:
    name: str
    env_name: str | None = None
    mount_path: str | None = None

    def __post_init__(self) -> None:
        _assert_profile_name(self.name, "name")
        if self.env_name is not None:
            _assert_env_name(self.env_name, "env_name")
        if self.mount_path is not None:
            _assert_container_path(self.mount_path, "mount_path")
        assert (
            self.env_name is not None or self.mount_path is not None
        ), "secret reference must target env or mount"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "env_name": self.env_name,
            "mount_path": self.mount_path,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerSecretReference":
        _assert_fields(raw, {"name", "env_name", "mount_path"}, "secret")
        return cls(
            name=_required_str(raw, "name", "secret"),
            env_name=_optional_str(raw, "env_name"),
            mount_path=_optional_str(raw, "mount_path"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerNetworkPolicy:
    mode: ContainerNetworkMode | str = ContainerNetworkMode.NONE
    egress_allowlist: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        mode = _enum_value(self.mode, ContainerNetworkMode, "mode")
        egress_allowlist = _string_tuple(
            self.egress_allowlist,
            "egress_allowlist",
        )
        assert not (
            mode is ContainerNetworkMode.NONE and egress_allowlist
        ), "network none cannot define egress"
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "egress_allowlist", egress_allowlist)

    def to_dict(self) -> dict[str, object]:
        mode = cast(ContainerNetworkMode, self.mode)
        return {
            "mode": mode.value,
            "egress_allowlist": list(self.egress_allowlist),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerNetworkPolicy":
        _assert_fields(raw, {"mode", "egress_allowlist"}, "network")
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerNetworkMode.NONE.value,
            ),
            egress_allowlist=_string_tuple(
                raw.get("egress_allowlist", ()),
                "egress_allowlist",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerDevicePolicy:
    devices: Sequence[ContainerDeviceClass | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "devices",
            tuple(
                _enum_value(device, ContainerDeviceClass, "devices")
                for device in self.devices
            ),
        )

    def to_dict(self) -> dict[str, object]:
        devices = cast(tuple[ContainerDeviceClass, ...], self.devices)
        return {"devices": [device.value for device in devices]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerDevicePolicy":
        _assert_fields(raw, {"devices"}, "devices")
        return cls(devices=_string_tuple(raw.get("devices", ()), "devices"))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerResourceLimits:
    cpu_count: int | None = None
    memory_bytes: int | None = None
    pids: int | None = None
    timeout_seconds: int | None = None

    def __post_init__(self) -> None:
        _assert_optional_positive_int(self.cpu_count, "cpu_count")
        _assert_optional_positive_int(self.memory_bytes, "memory_bytes")
        _assert_optional_positive_int(self.pids, "pids")
        _assert_optional_positive_int(self.timeout_seconds, "timeout_seconds")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "cpu_count": self.cpu_count,
            "memory_bytes": self.memory_bytes,
            "pids": self.pids,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerResourceLimits":
        _assert_fields(
            raw,
            {"cpu_count", "memory_bytes", "pids", "timeout_seconds"},
            "resources",
        )
        return cls(
            cpu_count=_optional_int(raw, "cpu_count"),
            memory_bytes=_optional_int(raw, "memory_bytes"),
            pids=_optional_int(raw, "pids"),
            timeout_seconds=_optional_int(raw, "timeout_seconds"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerOutputPolicy:
    max_stdout_bytes: int = 65536
    max_stderr_bytes: int = 32768
    max_artifact_bytes: int = 0
    allow_artifacts: bool = False

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_stdout_bytes, "max_stdout_bytes")
        _assert_positive_int(self.max_stderr_bytes, "max_stderr_bytes")
        _assert_positive_int(self.max_artifact_bytes + 1, "max_artifact_bytes")
        _assert_bool(self.allow_artifacts, "allow_artifacts")
        assert (
            self.allow_artifacts or self.max_artifact_bytes == 0
        ), "artifact bytes require artifact output"

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
            "allow_artifacts": self.allow_artifacts,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerOutputPolicy":
        _assert_fields(
            raw,
            {
                "max_stdout_bytes",
                "max_stderr_bytes",
                "max_artifact_bytes",
                "allow_artifacts",
            },
            "output",
        )
        return cls(
            max_stdout_bytes=_optional_int_or_default(
                raw,
                "max_stdout_bytes",
                65536,
            ),
            max_stderr_bytes=_optional_int_or_default(
                raw,
                "max_stderr_bytes",
                32768,
            ),
            max_artifact_bytes=_optional_int_or_default(
                raw,
                "max_artifact_bytes",
                0,
            ),
            allow_artifacts=_optional_bool_or_default(
                raw,
                "allow_artifacts",
                False,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerCleanupPolicy:
    mode: ContainerCleanupMode | str = ContainerCleanupMode.REMOVE
    grace_seconds: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ContainerCleanupMode, "mode"),
        )
        _assert_positive_int(self.grace_seconds, "grace_seconds")

    def to_dict(self) -> dict[str, object]:
        mode = cast(ContainerCleanupMode, self.mode)
        return {"mode": mode.value, "grace_seconds": self.grace_seconds}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerCleanupPolicy":
        _assert_fields(raw, {"mode", "grace_seconds"}, "cleanup")
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerCleanupMode.REMOVE.value,
            ),
            grace_seconds=_optional_int_or_default(
                raw,
                "grace_seconds",
                5,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPoolingPolicy:
    mode: ContainerPoolingMode | str = ContainerPoolingMode.DISABLED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ContainerPoolingMode, "mode"),
        )

    def to_dict(self) -> dict[str, str]:
        mode = cast(ContainerPoolingMode, self.mode)
        return {"mode": mode.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerPoolingPolicy":
        _assert_fields(raw, {"mode"}, "pooling")
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerPoolingMode.DISABLED.value,
            )
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerAuditPolicy:
    mode: ContainerAuditMode | str = ContainerAuditMode.MINIMAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ContainerAuditMode, "mode"),
        )

    def to_dict(self) -> dict[str, str]:
        mode = cast(ContainerAuditMode, self.mode)
        return {"mode": mode.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerAuditPolicy":
        _assert_fields(raw, {"mode"}, "audit")
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerAuditMode.MINIMAL.value,
            )
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerEscalationPolicy:
    mode: ContainerEscalationMode | str = ContainerEscalationMode.DENY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ContainerEscalationMode, "mode"),
        )

    def to_dict(self) -> dict[str, str]:
        mode = cast(ContainerEscalationMode, self.mode)
        return {"mode": mode.value}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerEscalationPolicy":
        _assert_fields(raw, {"mode"}, "escalation")
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerEscalationMode.DENY.value,
            )
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProfile:
    name: str
    image: ContainerImagePolicy
    workspace: ContainerWorkspaceMapping = field(
        default_factory=ContainerWorkspaceMapping,
    )
    mounts: Sequence[ContainerMountDeclaration] = field(default_factory=tuple)
    environment: ContainerEnvironmentPolicy = field(
        default_factory=ContainerEnvironmentPolicy,
    )
    secrets: Sequence[ContainerSecretReference] = field(default_factory=tuple)
    network: ContainerNetworkPolicy = field(
        default_factory=ContainerNetworkPolicy,
    )
    devices: ContainerDevicePolicy = field(
        default_factory=ContainerDevicePolicy,
    )
    resources: ContainerResourceLimits = field(
        default_factory=ContainerResourceLimits,
    )
    output: ContainerOutputPolicy = field(
        default_factory=ContainerOutputPolicy,
    )
    cleanup: ContainerCleanupPolicy = field(
        default_factory=ContainerCleanupPolicy,
    )
    pooling: ContainerPoolingPolicy = field(
        default_factory=ContainerPoolingPolicy,
    )
    audit: ContainerAuditPolicy = field(default_factory=ContainerAuditPolicy)
    escalation: ContainerEscalationPolicy = field(
        default_factory=ContainerEscalationPolicy,
    )
    command_mode: ContainerCommandMode | str = (
        ContainerCommandMode.FIXED_EXECUTABLE
    )
    read_only_rootfs: bool = True
    user: str = "1000:1000"

    def __post_init__(self) -> None:
        _assert_profile_name(self.name, "name")
        assert isinstance(self.image, ContainerImagePolicy)
        assert isinstance(self.workspace, ContainerWorkspaceMapping)
        assert isinstance(self.environment, ContainerEnvironmentPolicy)
        assert isinstance(self.network, ContainerNetworkPolicy)
        assert isinstance(self.devices, ContainerDevicePolicy)
        assert isinstance(self.resources, ContainerResourceLimits)
        assert isinstance(self.output, ContainerOutputPolicy)
        assert isinstance(self.cleanup, ContainerCleanupPolicy)
        assert isinstance(self.pooling, ContainerPoolingPolicy)
        assert isinstance(self.audit, ContainerAuditPolicy)
        assert isinstance(self.escalation, ContainerEscalationPolicy)
        mounts = tuple(self.mounts)
        secrets = tuple(self.secrets)
        for mount in mounts:
            assert isinstance(mount, ContainerMountDeclaration)
        for secret in secrets:
            assert isinstance(secret, ContainerSecretReference)
        _assert_bool(self.read_only_rootfs, "read_only_rootfs")
        assert (
            self.read_only_rootfs
        ), "profiles must default to read-only rootfs"
        _assert_non_empty_string(self.user, "user")
        assert self.user != "0" and not self.user.startswith(
            "0:"
        ), "root user is unsafe"
        object.__setattr__(
            self,
            "command_mode",
            _enum_value(
                self.command_mode,
                ContainerCommandMode,
                "command_mode",
            ),
        )
        object.__setattr__(self, "mounts", mounts)
        object.__setattr__(self, "secrets", secrets)

    @classmethod
    def minimal_readonly(
        cls,
        *,
        name: str,
        image_reference: str,
    ) -> "ContainerProfile":
        return cls(
            name=name,
            image=ContainerImagePolicy(reference=image_reference),
            mounts=(
                ContainerMountDeclaration(
                    source=".",
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                ),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        command_mode = cast(ContainerCommandMode, self.command_mode)
        return {
            "name": self.name,
            "image": self.image.to_dict(),
            "workspace": self.workspace.to_dict(),
            "mounts": [mount.to_dict() for mount in self.mounts],
            "environment": self.environment.to_dict(),
            "secrets": [secret.to_dict() for secret in self.secrets],
            "network": self.network.to_dict(),
            "devices": self.devices.to_dict(),
            "resources": self.resources.to_dict(),
            "output": self.output.to_dict(),
            "cleanup": self.cleanup.to_dict(),
            "pooling": self.pooling.to_dict(),
            "audit": self.audit.to_dict(),
            "escalation": self.escalation.to_dict(),
            "command_mode": command_mode.value,
            "read_only_rootfs": self.read_only_rootfs,
            "user": self.user,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerProfile":
        _assert_fields(raw, _PROFILE_FIELDS, "profile")
        return cls(
            name=_required_str(raw, "name", "profile"),
            image=ContainerImagePolicy.from_dict(
                _mapping(_required(raw, "image", "profile"), "image")
            ),
            workspace=ContainerWorkspaceMapping.from_dict(
                _mapping(raw.get("workspace", {}), "workspace")
            ),
            mounts=tuple(
                ContainerMountDeclaration.from_dict(_mapping(item, "mount"))
                for item in _sequence(raw.get("mounts", ()), "mounts")
            ),
            environment=ContainerEnvironmentPolicy.from_dict(
                _mapping(raw.get("environment", {}), "environment")
            ),
            secrets=tuple(
                ContainerSecretReference.from_dict(_mapping(item, "secret"))
                for item in _sequence(raw.get("secrets", ()), "secrets")
            ),
            network=ContainerNetworkPolicy.from_dict(
                _mapping(raw.get("network", {}), "network")
            ),
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
            pooling=ContainerPoolingPolicy.from_dict(
                _mapping(raw.get("pooling", {}), "pooling")
            ),
            audit=ContainerAuditPolicy.from_dict(
                _mapping(raw.get("audit", {}), "audit")
            ),
            escalation=ContainerEscalationPolicy.from_dict(
                _mapping(raw.get("escalation", {}), "escalation")
            ),
            command_mode=_optional_str_or_default(
                raw,
                "command_mode",
                ContainerCommandMode.FIXED_EXECUTABLE.value,
            ),
            read_only_rootfs=(
                True
                if "read_only_rootfs" not in raw
                else _required_bool(raw, "read_only_rootfs", "profile")
            ),
            user=_optional_str_or_default(raw, "user", "1000:1000"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProfileSelection:
    profile: str | None = None
    required: bool = False
    scope: ContainerExecutionScope | str = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    )

    def __post_init__(self) -> None:
        if self.profile is not None:
            _assert_profile_name(self.profile, "profile")
        _assert_bool(self.required, "required")
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, ContainerExecutionScope, "scope"),
        )

    def to_dict(self) -> dict[str, object]:
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "profile": self.profile,
            "required": self.required,
            "scope": scope.value,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: ContainerSettingsSource,
    ) -> "ContainerProfileSelection":
        assert isinstance(source, ContainerSettingsSource)
        assert (
            source.trust_level is not ContainerTrustLevel.MODEL
        ), "model output cannot select container runtime profiles"
        _assert_fields(raw, {"profile", "required", "scope"}, "selection")
        return cls(
            profile=_optional_str(raw, "profile"),
            required=_optional_bool_or_default(raw, "required", False),
            scope=_optional_str_or_default(
                raw,
                "scope",
                ContainerExecutionScope.SHELL_CONTAINER_EXECUTION.value,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSettings:
    source: ContainerSettingsSource
    backend: ContainerBackend | str = ContainerBackend.NONE
    default_profile: str | None = None
    allowed_profiles: Sequence[str] = field(default_factory=tuple)
    profiles: Mapping[str, ContainerProfile] = field(default_factory=dict)
    profile_registry_id: str = "default"
    policy_version: str = "phase1"

    def __post_init__(self) -> None:
        assert isinstance(self.source, ContainerSettingsSource)
        assert (
            self.source.can_define_runtime_authority
        ), "untrusted sources cannot define container runtime authority"
        backend = _enum_value(self.backend, ContainerBackend, "backend")
        profiles = _profile_mapping(self.profiles)
        allowed_profiles = _string_tuple(
            self.allowed_profiles or tuple(profiles),
            "allowed_profiles",
        )
        if self.default_profile is not None:
            _assert_profile_name(self.default_profile, "default_profile")
            assert (
                self.default_profile in profiles
            ), "default profile must be defined"
            assert (
                self.default_profile in allowed_profiles
            ), "default profile must be allowed"
        for profile_name in allowed_profiles:
            _assert_profile_name(profile_name, "allowed_profiles")
            assert profile_name in profiles, "allowed profiles must be defined"
        if backend is ContainerBackend.NONE:
            assert (
                not profiles
            ), "disabled container settings cannot define profiles"
            assert (
                not allowed_profiles
            ), "disabled container settings cannot allow profiles"
            assert (
                self.default_profile is None
            ), "disabled container settings cannot select a profile"
        _assert_profile_name(self.profile_registry_id, "profile_registry_id")
        _assert_profile_name(self.policy_version, "policy_version")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "allowed_profiles", allowed_profiles)
        object.__setattr__(self, "profiles", MappingProxyType(profiles))

    @property
    def enabled(self) -> bool:
        return self.backend is not ContainerBackend.NONE

    def select_profile(
        self,
        selection: ContainerProfileSelection,
    ) -> "ContainerEffectiveSettings":
        assert isinstance(selection, ContainerProfileSelection)
        selected_profile = selection.profile or self.default_profile
        if not self.enabled:
            assert (
                selected_profile is None
            ), "disabled container settings cannot select a profile"
            return ContainerEffectiveSettings(
                backend=self.backend,
                required=selection.required,
                scope=selection.scope,
                source=self.source,
                policy_version=self.policy_version,
                profile_registry_id=self.profile_registry_id,
            )
        assert selected_profile is not None, "container profile is required"
        assert (
            selected_profile in self.allowed_profiles
        ), "selected profile must be allowed"
        return ContainerEffectiveSettings(
            backend=self.backend,
            required=selection.required,
            scope=selection.scope,
            source=self.source,
            policy_version=self.policy_version,
            profile_registry_id=self.profile_registry_id,
            profile_name=selected_profile,
            profile=self.profiles[selected_profile],
            allowed_profiles=self.allowed_profiles,
        )

    def to_dict(self) -> dict[str, object]:
        backend = cast(ContainerBackend, self.backend)
        return {
            "source": self.source.to_dict(),
            "backend": backend.value,
            "default_profile": self.default_profile,
            "allowed_profiles": list(self.allowed_profiles),
            "profiles": {
                name: profile.to_dict()
                for name, profile in sorted(self.profiles.items())
            },
            "profile_registry_id": self.profile_registry_id,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: ContainerSettingsSource | None = None,
    ) -> "ContainerSettings":
        assert (
            source is not None
        ), "container settings source must be supplied by trusted loader"
        assert isinstance(source, ContainerSettingsSource)
        _assert_fields(raw, _SETTINGS_FIELDS, "settings")
        profiles = _profile_mapping_from_dict(
            _mapping(raw.get("profiles", {}), "profiles")
        )
        return cls(
            source=source,
            backend=_optional_str_or_default(
                raw,
                "backend",
                ContainerBackend.NONE.value,
            ),
            default_profile=_optional_str(raw, "default_profile"),
            allowed_profiles=_string_tuple(
                raw.get("allowed_profiles", ()),
                "allowed_profiles",
            ),
            profiles=profiles,
            profile_registry_id=_optional_str_or_default(
                raw,
                "profile_registry_id",
                "default",
            ),
            policy_version=_optional_str_or_default(
                raw,
                "policy_version",
                "phase1",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerEffectiveSettings:
    backend: ContainerBackend | str
    required: bool
    scope: ContainerExecutionScope | str
    source: ContainerSettingsSource
    policy_version: str
    profile_registry_id: str
    profile_name: str | None = None
    profile: ContainerProfile | None = None
    allowed_profiles: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        backend = _enum_value(self.backend, ContainerBackend, "backend")
        scope = _enum_value(self.scope, ContainerExecutionScope, "scope")
        assert isinstance(self.source, ContainerSettingsSource)
        _assert_bool(self.required, "required")
        allowed_profiles = _string_tuple(
            self.allowed_profiles,
            "allowed_profiles",
        )
        if self.profile_name is not None:
            _assert_profile_name(self.profile_name, "profile_name")
            assert self.profile is not None, "profile_name requires profile"
        if self.profile is not None:
            assert isinstance(self.profile, ContainerProfile)
            assert (
                self.profile_name == self.profile.name
            ), "profile_name must match profile"
        assert (
            backend is not ContainerBackend.NONE or self.profile is None
        ), "disabled settings cannot carry a profile"
        _assert_profile_name(self.profile_registry_id, "profile_registry_id")
        _assert_profile_name(self.policy_version, "policy_version")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "allowed_profiles", allowed_profiles)

    @property
    def enabled(self) -> bool:
        return self.backend is not ContainerBackend.NONE

    def to_dict(self) -> dict[str, object]:
        backend = cast(ContainerBackend, self.backend)
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "backend": backend.value,
            "required": self.required,
            "scope": scope.value,
            "source": self.source.to_dict(),
            "policy_version": self.policy_version,
            "profile_registry_id": self.profile_registry_id,
            "profile_name": self.profile_name,
            "profile": self.profile.to_dict() if self.profile else None,
            "allowed_profiles": list(self.allowed_profiles),
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerEffectiveSettings":
        _assert_fields(raw, _EFFECTIVE_SETTINGS_FIELDS, "effective")
        profile_raw = raw.get("profile")
        return cls(
            backend=_required_str(raw, "backend", "effective"),
            required=_required_bool(raw, "required", "effective"),
            scope=_required_str(raw, "scope", "effective"),
            source=ContainerSettingsSource.from_dict(
                _mapping(_required(raw, "source", "effective"), "source")
            ),
            policy_version=_required_str(
                raw,
                "policy_version",
                "effective",
            ),
            profile_registry_id=_required_str(
                raw,
                "profile_registry_id",
                "effective",
            ),
            profile_name=_optional_str(raw, "profile_name"),
            profile=(
                None
                if profile_raw is None
                else ContainerProfile.from_dict(
                    _mapping(profile_raw, "profile")
                )
            ),
            allowed_profiles=_string_tuple(
                raw.get("allowed_profiles", ()),
                "allowed_profiles",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendCapabilities:
    backend: ContainerBackend | str
    host_os: str
    guest_os: str
    architecture: str
    platform_emulation: bool = False
    rootless: bool = False
    user_namespace: bool = False
    build: bool = False
    pull: bool = False
    network_modes: Sequence[ContainerNetworkMode | str] = field(
        default_factory=lambda: (ContainerNetworkMode.NONE,),
    )
    mount_types: Sequence[ContainerMountType | str] = field(
        default_factory=tuple,
    )
    resource_limits: bool = False
    device_classes: Sequence[ContainerDeviceClass | str] = field(
        default_factory=tuple,
    )
    per_container_vm_isolation: bool = False
    windows_process_isolation: bool = False
    windows_hyperv_isolation: bool = False
    streaming_attach: bool = False
    stats: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, ContainerBackend, "backend"),
        )
        _assert_non_empty_string(self.host_os, "host_os")
        _assert_non_empty_string(self.guest_os, "guest_os")
        _assert_non_empty_string(self.architecture, "architecture")
        for field_name in (
            "rootless",
            "user_namespace",
            "build",
            "pull",
            "platform_emulation",
            "resource_limits",
            "per_container_vm_isolation",
            "windows_process_isolation",
            "windows_hyperv_isolation",
            "streaming_attach",
            "stats",
        ):
            _assert_bool(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "network_modes",
            tuple(
                _enum_value(mode, ContainerNetworkMode, "network_modes")
                for mode in self.network_modes
            ),
        )
        object.__setattr__(
            self,
            "mount_types",
            tuple(
                _enum_value(mount_type, ContainerMountType, "mount_types")
                for mount_type in self.mount_types
            ),
        )
        object.__setattr__(
            self,
            "device_classes",
            tuple(
                _enum_value(device, ContainerDeviceClass, "device_classes")
                for device in self.device_classes
            ),
        )

    def to_dict(self) -> dict[str, object]:
        backend = cast(ContainerBackend, self.backend)
        network_modes = cast(
            tuple[ContainerNetworkMode, ...],
            self.network_modes,
        )
        mount_types = cast(tuple[ContainerMountType, ...], self.mount_types)
        device_classes = cast(
            tuple[ContainerDeviceClass, ...],
            self.device_classes,
        )
        return {
            "backend": backend.value,
            "host_os": self.host_os,
            "guest_os": self.guest_os,
            "architecture": self.architecture,
            "platform_emulation": self.platform_emulation,
            "rootless": self.rootless,
            "user_namespace": self.user_namespace,
            "build": self.build,
            "pull": self.pull,
            "network_modes": [mode.value for mode in network_modes],
            "mount_types": [mount_type.value for mount_type in mount_types],
            "resource_limits": self.resource_limits,
            "device_classes": [device.value for device in device_classes],
            "per_container_vm_isolation": self.per_container_vm_isolation,
            "windows_process_isolation": self.windows_process_isolation,
            "windows_hyperv_isolation": self.windows_hyperv_isolation,
            "streaming_attach": self.streaming_attach,
            "stats": self.stats,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerCommandPlan:
    tool_name: str
    command: str
    argv: Sequence[str]
    cwd: str
    scope: ContainerExecutionScope | str

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.tool_name, "tool_name")
        _assert_non_empty_string(self.command, "command")
        _assert_container_path(self.cwd, "cwd")
        object.__setattr__(self, "argv", _string_tuple(self.argv, "argv"))
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, ContainerExecutionScope, "scope"),
        )

    def to_dict(self) -> dict[str, object]:
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "tool_name": self.tool_name,
            "command": self.command,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "scope": scope.value,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRunPlan:
    backend: ContainerBackend | str
    profile_name: str
    image: ContainerImagePolicy
    command: ContainerCommandPlan
    mounts: Sequence[ContainerMountDeclaration] = field(default_factory=tuple)
    environment_names: Sequence[str] = field(default_factory=tuple)
    secret_names: Sequence[str] = field(default_factory=tuple)
    network: ContainerNetworkPolicy = field(
        default_factory=ContainerNetworkPolicy,
    )
    devices: ContainerDevicePolicy = field(
        default_factory=ContainerDevicePolicy,
    )
    resources: ContainerResourceLimits = field(
        default_factory=ContainerResourceLimits,
    )
    policy_version: str = "phase1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, ContainerBackend, "backend"),
        )
        _assert_profile_name(self.profile_name, "profile_name")
        assert isinstance(self.image, ContainerImagePolicy)
        assert isinstance(self.command, ContainerCommandPlan)
        mounts = tuple(self.mounts)
        for mount in mounts:
            assert isinstance(mount, ContainerMountDeclaration)
        for name in self.environment_names:
            _assert_env_name(name, "environment_names")
        secret_names = _string_tuple(self.secret_names, "secret_names")
        assert isinstance(self.network, ContainerNetworkPolicy)
        assert isinstance(self.devices, ContainerDevicePolicy)
        assert isinstance(self.resources, ContainerResourceLimits)
        _assert_profile_name(self.policy_version, "policy_version")
        object.__setattr__(self, "mounts", mounts)
        object.__setattr__(
            self,
            "environment_names",
            tuple(self.environment_names),
        )
        object.__setattr__(self, "secret_names", secret_names)

    def to_dict(self) -> dict[str, object]:
        backend = cast(ContainerBackend, self.backend)
        return {
            "backend": backend.value,
            "profile_name": self.profile_name,
            "image": self.image.to_dict(),
            "command": self.command.to_dict(),
            "mounts": [mount.to_dict() for mount in self.mounts],
            "environment_names": list(self.environment_names),
            "secret_names": list(self.secret_names),
            "network": self.network.to_dict(),
            "devices": self.devices.to_dict(),
            "resources": self.resources.to_dict(),
            "policy_version": self.policy_version,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerRuntimeEnvelopePlan:
    scope: ContainerExecutionScope | str
    profile_name: str
    command: ContainerCommandPlan
    readiness_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, ContainerExecutionScope, "scope"),
        )
        _assert_profile_name(self.profile_name, "profile_name")
        assert isinstance(self.command, ContainerCommandPlan)
        _assert_positive_int(
            self.readiness_timeout_seconds,
            "readiness_timeout_seconds",
        )

    def to_dict(self) -> dict[str, object]:
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "scope": scope.value,
            "profile_name": self.profile_name,
            "command": self.command.to_dict(),
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerAuthorizationDecision:
    decision: ContainerAuthorizationDecisionType | str
    code: str
    explanation: str
    policy_version: str
    profile_name: str | None = None
    retryable: bool = False
    cacheable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision",
            _enum_value(
                self.decision,
                ContainerAuthorizationDecisionType,
                "decision",
            ),
        )
        _assert_profile_name(self.code, "code")
        _assert_non_empty_string(self.explanation, "explanation")
        _assert_profile_name(self.policy_version, "policy_version")
        if self.profile_name is not None:
            _assert_profile_name(self.profile_name, "profile_name")
        _assert_bool(self.retryable, "retryable")
        _assert_bool(self.cacheable, "cacheable")

    def to_dict(self) -> dict[str, object]:
        decision = cast(ContainerAuthorizationDecisionType, self.decision)
        return {
            "decision": decision.value,
            "code": self.code,
            "explanation": self.explanation,
            "policy_version": self.policy_version,
            "profile_name": self.profile_name,
            "retryable": self.retryable,
            "cacheable": self.cacheable,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerExecutionResult:
    status: ContainerResultStatus | str
    exit_code: int | None = None
    diagnostics: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, ContainerResultStatus, "status"),
        )
        if self.exit_code is not None:
            assert isinstance(self.exit_code, int)
        object.__setattr__(
            self,
            "diagnostics",
            _string_tuple(self.diagnostics, "diagnostics"),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    def to_dict(self) -> dict[str, object]:
        status = cast(ContainerResultStatus, self.status)
        return {
            "status": status.value,
            "exit_code": self.exit_code,
            "diagnostics": list(self.diagnostics),
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerAuditEvent:
    event_type: ContainerAuditEventType | str
    scope: ContainerExecutionScope | str
    profile_name: str | None
    policy_version: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_type",
            _enum_value(
                self.event_type,
                ContainerAuditEventType,
                "event_type",
            ),
        )
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, ContainerExecutionScope, "scope"),
        )
        if self.profile_name is not None:
            _assert_profile_name(self.profile_name, "profile_name")
        _assert_profile_name(self.policy_version, "policy_version")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    def to_dict(self) -> dict[str, object]:
        event_type = cast(ContainerAuditEventType, self.event_type)
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "event_type": event_type.value,
            "scope": scope.value,
            "profile_name": self.profile_name,
            "policy_version": self.policy_version,
            "metadata": dict(self.metadata),
        }


_PROFILE_FIELDS = {
    "name",
    "image",
    "workspace",
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
    "command_mode",
    "read_only_rootfs",
    "user",
}
_SETTINGS_FIELDS = {
    "source",
    "backend",
    "default_profile",
    "allowed_profiles",
    "profiles",
    "profile_registry_id",
    "policy_version",
}
_EFFECTIVE_SETTINGS_FIELDS = {
    "backend",
    "required",
    "scope",
    "source",
    "policy_version",
    "profile_registry_id",
    "profile_name",
    "profile",
    "allowed_profiles",
}


def _assert_fields(
    raw: Mapping[str, object],
    allowed: set[str],
    path: str,
) -> None:
    assert isinstance(raw, Mapping), f"{path} must be a mapping"
    unknown = sorted(str(key) for key in raw if key not in allowed)
    assert not unknown, f"{path} contains unknown fields: {', '.join(unknown)}"


def _required(raw: Mapping[str, object], key: str, path: str) -> object:
    assert key in raw, f"{path}.{key} is required"
    return raw[key]


def _required_str(raw: Mapping[str, object], key: str, path: str) -> str:
    value = _required(raw, key, path)
    _assert_non_empty_string(value, f"{path}.{key}")
    assert isinstance(value, str)
    return value


def _required_bool(raw: Mapping[str, object], key: str, path: str) -> bool:
    value = _required(raw, key, path)
    _assert_bool(value, f"{path}.{key}")
    assert isinstance(value, bool)
    return value


def _optional_str(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    _assert_non_empty_string(value, key)
    assert isinstance(value, str)
    return value


def _optional_str_or_default(
    raw: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = _optional_str(raw, key)
    if value is None:
        return default
    return value


def _optional_bool(raw: Mapping[str, object], key: str) -> bool | None:
    value = raw.get(key)
    if value is None:
        return None
    _assert_bool(value, key)
    assert isinstance(value, bool)
    return value


def _optional_bool_or_default(
    raw: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = _optional_bool(raw, key)
    if value is None:
        return default
    return value


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _optional_int_or_default(
    raw: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = _optional_int(raw, key)
    if value is None:
        return default
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    sequence = _sequence(value, field_name)
    result: list[str] = []
    for item in sequence:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
        result.append(item)
    return tuple(result)


def _string_mapping(value: object, field_name: str) -> dict[str, str]:
    mapping = _mapping(value, field_name)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name}.{key}")
        assert isinstance(key, str)
        assert isinstance(item, str)
        result[key] = item
    return result


def _profile_mapping(
    value: Mapping[str, ContainerProfile],
) -> dict[str, ContainerProfile]:
    assert isinstance(value, Mapping), "profiles must be a mapping"
    profiles: dict[str, ContainerProfile] = {}
    for name, profile in value.items():
        _assert_profile_name(name, "profiles key")
        assert isinstance(profile, ContainerProfile)
        assert profile.name == name, "profile mapping key must match name"
        profiles[name] = profile
    return profiles


def _profile_mapping_from_dict(
    value: Mapping[str, object],
) -> dict[str, ContainerProfile]:
    profiles: dict[str, ContainerProfile] = {}
    for name, profile in value.items():
        _assert_profile_name(name, "profiles key")
        assert isinstance(name, str)
        profiles[name] = ContainerProfile.from_dict(
            _mapping(profile, "profile")
        )
    return profiles


def _enum_value(
    value: EnumValue | str,
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


def _assert_profile_name(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert _PROFILE_NAME_PATTERN.match(
        value
    ), f"{field_name} must be a container identifier"


def _assert_host_path(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    assert not value.startswith("~"), f"{field_name} must not expand home"


def _assert_container_path(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    assert value.startswith("/"), f"{field_name} must be absolute"


def _digest_from_reference(reference: str) -> str | None:
    marker = "@sha256:"
    if marker not in reference:
        return None
    return "sha256:" + reference.rsplit(marker, maxsplit=1)[1]


def _assert_digest(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert _DIGEST_PATTERN.match(value), f"{field_name} must be sha256 digest"


def _assert_platform(value: object) -> None:
    _assert_non_empty_string(value, "platform")
    assert isinstance(value, str)
    assert _PLATFORM_PATTERN.match(value), "platform must be os/architecture"
