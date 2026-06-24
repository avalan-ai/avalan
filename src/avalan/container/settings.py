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


class ContainerSettingsPrecedence(StrEnum):
    SERVER_OPERATOR = "server_operator"
    WORKER = "worker"
    SDK = "sdk"
    CLI = "cli"
    AGENT_TOML = "agent_toml"
    FLOW_TOML = "flow_toml"
    TASK_TOML = "task_toml"
    REQUEST = "request"


CONTAINER_SETTINGS_PRECEDENCE = (
    ContainerSettingsPrecedence.SERVER_OPERATOR,
    ContainerSettingsPrecedence.WORKER,
    ContainerSettingsPrecedence.SDK,
    ContainerSettingsPrecedence.CLI,
    ContainerSettingsPrecedence.AGENT_TOML,
    ContainerSettingsPrecedence.FLOW_TOML,
    ContainerSettingsPrecedence.TASK_TOML,
    ContainerSettingsPrecedence.REQUEST,
)


class ContainerPullPolicy(StrEnum):
    NEVER = "never"
    IF_MISSING = "if_missing"
    ALWAYS = "always"


class ContainerBuildPolicy(StrEnum):
    DISABLED = "disabled"
    TRUSTED_ONLY = "trusted_only"


class ContainerCacheMode(StrEnum):
    DISABLED = "disabled"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


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
    FULL = "full"


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
    SERVICE = "service"


class ContainerPoolTeardownMode(StrEnum):
    REMOVE = "remove"
    RESET = "reset"
    QUARANTINE = "quarantine"


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


class ContainerBackendSupportLevel(StrEnum):
    SUPPORTED = "supported"
    OPTIONAL = "optional"
    OPT_IN = "opt_in"
    CATALOG_ONLY = "catalog_only"


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
class ContainerImageCachePolicy:
    mode: ContainerCacheMode | str = ContainerCacheMode.DISABLED
    ttl_seconds: int = 0
    require_digest_key: bool = True
    allow_stale: bool = False

    def __post_init__(self) -> None:
        mode = _enum_value(self.mode, ContainerCacheMode, "mode")
        _assert_optional_non_negative_int(self.ttl_seconds, "ttl_seconds")
        _assert_bool(self.require_digest_key, "require_digest_key")
        _assert_bool(self.allow_stale, "allow_stale")
        if mode is ContainerCacheMode.DISABLED:
            assert self.ttl_seconds == 0, "disabled cache cannot set ttl"
        else:
            _assert_positive_int(self.ttl_seconds, "ttl_seconds")
            assert self.require_digest_key, "image cache must key by digest"
            assert not self.allow_stale, "image cache cannot allow stale hits"
        object.__setattr__(self, "mode", mode)

    @property
    def enabled(self) -> bool:
        return self.mode is not ContainerCacheMode.DISABLED

    def to_dict(self) -> dict[str, int | str | bool]:
        mode = cast(ContainerCacheMode, self.mode)
        return {
            "mode": mode.value,
            "ttl_seconds": self.ttl_seconds,
            "require_digest_key": self.require_digest_key,
            "allow_stale": self.allow_stale,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerImageCachePolicy":
        _assert_fields(
            raw,
            {
                "mode",
                "ttl_seconds",
                "require_digest_key",
                "allow_stale",
            },
            "image_cache",
        )
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerCacheMode.DISABLED.value,
            ),
            ttl_seconds=_optional_int_or_default(raw, "ttl_seconds", 0),
            require_digest_key=_optional_bool_or_default(
                raw,
                "require_digest_key",
                True,
            ),
            allow_stale=_optional_bool_or_default(
                raw,
                "allow_stale",
                False,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBuildCachePolicy:
    mode: ContainerCacheMode | str = ContainerCacheMode.DISABLED
    ttl_seconds: int = 0
    require_context_digest_key: bool = True
    allow_stale: bool = False

    def __post_init__(self) -> None:
        mode = _enum_value(self.mode, ContainerCacheMode, "mode")
        _assert_optional_non_negative_int(self.ttl_seconds, "ttl_seconds")
        _assert_bool(
            self.require_context_digest_key,
            "require_context_digest_key",
        )
        _assert_bool(self.allow_stale, "allow_stale")
        if mode is ContainerCacheMode.DISABLED:
            assert self.ttl_seconds == 0, "disabled cache cannot set ttl"
        else:
            _assert_positive_int(self.ttl_seconds, "ttl_seconds")
            assert (
                self.require_context_digest_key
            ), "build cache must key by context digest"
            assert not self.allow_stale, "build cache cannot allow stale hits"
        object.__setattr__(self, "mode", mode)

    @property
    def enabled(self) -> bool:
        return self.mode is not ContainerCacheMode.DISABLED

    def to_dict(self) -> dict[str, int | str | bool]:
        mode = cast(ContainerCacheMode, self.mode)
        return {
            "mode": mode.value,
            "ttl_seconds": self.ttl_seconds,
            "require_context_digest_key": self.require_context_digest_key,
            "allow_stale": self.allow_stale,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerBuildCachePolicy":
        _assert_fields(
            raw,
            {
                "mode",
                "ttl_seconds",
                "require_context_digest_key",
                "allow_stale",
            },
            "build_cache",
        )
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerCacheMode.DISABLED.value,
            ),
            ttl_seconds=_optional_int_or_default(raw, "ttl_seconds", 0),
            require_context_digest_key=_optional_bool_or_default(
                raw,
                "require_context_digest_key",
                True,
            ),
            allow_stale=_optional_bool_or_default(
                raw,
                "allow_stale",
                False,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBuildContextPolicy:
    context_path: str
    context_digest: str
    context_size_bytes: int
    dockerfile_path: str = "Dockerfile"
    dockerignore_path: str = ".dockerignore"
    max_context_bytes: int = 10485760
    secret_names: Sequence[str] = field(default_factory=tuple)
    network: ContainerNetworkMode | str = ContainerNetworkMode.NONE
    allow_remote_context: bool = False
    allow_build_secrets: bool = False
    allow_build_network: bool = False

    def __post_init__(self) -> None:
        _assert_bool(self.allow_remote_context, "allow_remote_context")
        _assert_bool(self.allow_build_secrets, "allow_build_secrets")
        _assert_bool(self.allow_build_network, "allow_build_network")
        _assert_bounded_relative_path(
            self.context_path,
            "context_path",
            allow_remote=self.allow_remote_context,
        )
        _assert_bounded_relative_path(self.dockerfile_path, "dockerfile_path")
        _assert_bounded_relative_path(
            self.dockerignore_path,
            "dockerignore_path",
        )
        _assert_digest(self.context_digest, "context_digest")
        _assert_positive_int(self.context_size_bytes, "context_size_bytes")
        _assert_positive_int(self.max_context_bytes, "max_context_bytes")
        assert (
            self.context_size_bytes <= self.max_context_bytes
        ), "build context exceeds configured bound"
        secret_names = _string_tuple(self.secret_names, "secret_names")
        assert (
            not secret_names or self.allow_build_secrets
        ), "build secrets require explicit trusted authorization"
        network = _enum_value(self.network, ContainerNetworkMode, "network")
        assert (
            network is ContainerNetworkMode.NONE or self.allow_build_network
        ), "build network requires explicit trusted authorization"
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "secret_names", secret_names)

    def to_dict(self) -> dict[str, object]:
        network = cast(ContainerNetworkMode, self.network)
        return {
            "context_path": self.context_path,
            "context_digest": self.context_digest,
            "context_size_bytes": self.context_size_bytes,
            "dockerfile_path": self.dockerfile_path,
            "dockerignore_path": self.dockerignore_path,
            "max_context_bytes": self.max_context_bytes,
            "secret_names": list(self.secret_names),
            "network": network.value,
            "allow_remote_context": self.allow_remote_context,
            "allow_build_secrets": self.allow_build_secrets,
            "allow_build_network": self.allow_build_network,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerBuildContextPolicy":
        _assert_fields(
            raw,
            {
                "context_path",
                "context_digest",
                "context_size_bytes",
                "dockerfile_path",
                "dockerignore_path",
                "max_context_bytes",
                "secret_names",
                "network",
                "allow_remote_context",
                "allow_build_secrets",
                "allow_build_network",
            },
            "build_context",
        )
        return cls(
            context_path=_required_str(raw, "context_path", "build_context"),
            context_digest=_required_str(
                raw,
                "context_digest",
                "build_context",
            ),
            context_size_bytes=_required_int(
                raw,
                "context_size_bytes",
                "build_context",
            ),
            dockerfile_path=_optional_str_or_default(
                raw,
                "dockerfile_path",
                "Dockerfile",
            ),
            dockerignore_path=_optional_str_or_default(
                raw,
                "dockerignore_path",
                ".dockerignore",
            ),
            max_context_bytes=_optional_int_or_default(
                raw,
                "max_context_bytes",
                10485760,
            ),
            secret_names=_string_tuple(
                raw.get("secret_names", ()),
                "secret_names",
            ),
            network=_optional_str_or_default(
                raw,
                "network",
                ContainerNetworkMode.NONE.value,
            ),
            allow_remote_context=_optional_bool_or_default(
                raw,
                "allow_remote_context",
                False,
            ),
            allow_build_secrets=_optional_bool_or_default(
                raw,
                "allow_build_secrets",
                False,
            ),
            allow_build_network=_optional_bool_or_default(
                raw,
                "allow_build_network",
                False,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerImagePolicy:
    reference: str
    digest: str | None = None
    pull_policy: ContainerPullPolicy | str = ContainerPullPolicy.NEVER
    build_policy: ContainerBuildPolicy | str = ContainerBuildPolicy.DISABLED
    platform: str = "linux/amd64"
    build_context: ContainerBuildContextPolicy | None = None
    image_cache: ContainerImageCachePolicy = field(
        default_factory=ContainerImageCachePolicy,
    )
    build_cache: ContainerBuildCachePolicy = field(
        default_factory=ContainerBuildCachePolicy,
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.reference, "reference")
        reference_digest = _digest_from_reference(self.reference)
        if self.digest is not None:
            _assert_digest(self.digest, "digest")
        if reference_digest is not None:
            _assert_digest(reference_digest, "reference digest")
        assert not (
            reference_digest is not None
            and self.digest is not None
            and reference_digest != self.digest
        ), "image reference digest must match explicit digest"
        digest = self.digest or reference_digest
        assert digest is not None, "image reference must be digest pinned"
        _assert_digest(digest, "digest")
        _assert_platform(self.platform)
        pull_policy = _enum_value(
            self.pull_policy,
            ContainerPullPolicy,
            "pull_policy",
        )
        build_policy = _enum_value(
            self.build_policy,
            ContainerBuildPolicy,
            "build_policy",
        )
        if self.build_context is not None:
            assert isinstance(
                self.build_context,
                ContainerBuildContextPolicy,
            )
            assert (
                build_policy is not ContainerBuildPolicy.DISABLED
            ), "build context requires enabled build policy"
        assert isinstance(self.image_cache, ContainerImageCachePolicy)
        assert isinstance(self.build_cache, ContainerBuildCachePolicy)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "pull_policy", pull_policy)
        object.__setattr__(self, "build_policy", build_policy)

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
            "build_context": (
                None
                if self.build_context is None
                else self.build_context.to_dict()
            ),
            "image_cache": self.image_cache.to_dict(),
            "build_cache": self.build_cache.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerImagePolicy":
        _assert_fields(
            raw,
            {
                "reference",
                "digest",
                "pull_policy",
                "build_policy",
                "platform",
                "build_context",
                "image_cache",
                "build_cache",
            },
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
            build_context=_optional_build_context(raw),
            image_cache=ContainerImageCachePolicy.from_dict(
                _mapping(raw.get("image_cache", {}), "image_cache")
            ),
            build_cache=ContainerBuildCachePolicy.from_dict(
                _mapping(raw.get("build_cache", {}), "build_cache")
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
    max_age_seconds: int = 60
    max_uses: int = 1
    idle_ttl_seconds: int = 30
    require_clean_scratch: bool = True
    require_no_leftover_processes: bool = True
    health_check_command: Sequence[str] = field(default_factory=tuple)
    teardown: ContainerPoolTeardownMode | str = (
        ContainerPoolTeardownMode.REMOVE
    )
    audit_labels: Mapping[str, str] = field(default_factory=dict)
    allow_secret_reuse: bool = False

    def __post_init__(self) -> None:
        mode = _enum_value(self.mode, ContainerPoolingMode, "mode")
        teardown = _enum_value(
            self.teardown,
            ContainerPoolTeardownMode,
            "teardown",
        )
        _assert_positive_int(self.max_age_seconds, "max_age_seconds")
        _assert_positive_int(self.max_uses, "max_uses")
        _assert_positive_int(self.idle_ttl_seconds, "idle_ttl_seconds")
        _assert_bool(self.require_clean_scratch, "require_clean_scratch")
        _assert_bool(
            self.require_no_leftover_processes,
            "require_no_leftover_processes",
        )
        _assert_bool(self.allow_secret_reuse, "allow_secret_reuse")
        health_check_command = _string_tuple(
            self.health_check_command,
            "health_check_command",
        )
        audit_labels = _string_mapping(self.audit_labels, "audit_labels")
        if mode is not ContainerPoolingMode.DISABLED:
            assert (
                self.require_clean_scratch
            ), "container pooling requires clean scratch checks"
            assert (
                self.require_no_leftover_processes
            ), "container pooling requires process cleanup checks"
        if mode is ContainerPoolingMode.SERVICE:
            assert health_check_command, "service pools require health checks"
            assert audit_labels, "service pools require audit labels"
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "teardown", teardown)
        object.__setattr__(
            self,
            "health_check_command",
            health_check_command,
        )
        object.__setattr__(
            self,
            "audit_labels",
            MappingProxyType(audit_labels),
        )

    def to_dict(self) -> dict[str, object]:
        mode = cast(ContainerPoolingMode, self.mode)
        teardown = cast(ContainerPoolTeardownMode, self.teardown)
        return {
            "mode": mode.value,
            "max_age_seconds": self.max_age_seconds,
            "max_uses": self.max_uses,
            "idle_ttl_seconds": self.idle_ttl_seconds,
            "require_clean_scratch": self.require_clean_scratch,
            "require_no_leftover_processes": (
                self.require_no_leftover_processes
            ),
            "health_check_command": list(self.health_check_command),
            "teardown": teardown.value,
            "audit_labels": dict(self.audit_labels),
            "allow_secret_reuse": self.allow_secret_reuse,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerPoolingPolicy":
        _assert_fields(
            raw,
            {
                "mode",
                "max_age_seconds",
                "max_uses",
                "idle_ttl_seconds",
                "require_clean_scratch",
                "require_no_leftover_processes",
                "health_check_command",
                "teardown",
                "audit_labels",
                "allow_secret_reuse",
            },
            "pooling",
        )
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                ContainerPoolingMode.DISABLED.value,
            ),
            max_age_seconds=_optional_int_or_default(
                raw,
                "max_age_seconds",
                60,
            ),
            max_uses=_optional_int_or_default(raw, "max_uses", 1),
            idle_ttl_seconds=_optional_int_or_default(
                raw,
                "idle_ttl_seconds",
                30,
            ),
            require_clean_scratch=_optional_bool_or_default(
                raw,
                "require_clean_scratch",
                True,
            ),
            require_no_leftover_processes=_optional_bool_or_default(
                raw,
                "require_no_leftover_processes",
                True,
            ),
            health_check_command=_string_tuple(
                raw.get("health_check_command", ()),
                "health_check_command",
            ),
            teardown=_optional_str_or_default(
                raw,
                "teardown",
                ContainerPoolTeardownMode.REMOVE.value,
            ),
            audit_labels=_string_mapping(
                raw.get("audit_labels", {}),
                "audit_labels",
            ),
            allow_secret_reuse=_optional_bool_or_default(
                raw,
                "allow_secret_reuse",
                False,
            ),
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
        scope = _optional_str_or_default(
            raw,
            "scope",
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION.value,
        )
        _assert_scope_allowed_for_source(
            source,
            _enum_value(scope, ContainerExecutionScope, "scope"),
        )
        return cls(
            profile=_optional_str(raw, "profile"),
            required=_optional_bool_or_default(raw, "required", False),
            scope=scope,
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

    def canonical_policy_input(self) -> dict[str, object]:
        serialized = self.to_dict()
        serialized.pop("source")
        serialized["allowed_profiles"] = sorted(self.allowed_profiles)
        profile = serialized["profile"]
        if profile is not None:
            serialized["profile"] = _canonical_profile_policy(
                _mapping(profile, "profile")
            )
        return serialized

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
class ContainerOutputPolicyOverride:
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    max_artifact_bytes: int | None = None
    allow_artifacts: bool | None = None

    def __post_init__(self) -> None:
        _assert_optional_positive_int(
            self.max_stdout_bytes,
            "max_stdout_bytes",
        )
        _assert_optional_positive_int(
            self.max_stderr_bytes,
            "max_stderr_bytes",
        )
        _assert_optional_non_negative_int(
            self.max_artifact_bytes,
            "max_artifact_bytes",
        )
        if self.allow_artifacts is not None:
            _assert_bool(self.allow_artifacts, "allow_artifacts")

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
            "allow_artifacts": self.allow_artifacts,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerOutputPolicyOverride":
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
            max_stdout_bytes=_optional_int(raw, "max_stdout_bytes"),
            max_stderr_bytes=_optional_int(raw, "max_stderr_bytes"),
            max_artifact_bytes=_optional_int(raw, "max_artifact_bytes"),
            allow_artifacts=_optional_bool(raw, "allow_artifacts"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerCleanupPolicyOverride:
    mode: ContainerCleanupMode | str | None = None
    grace_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.mode is not None:
            object.__setattr__(
                self,
                "mode",
                _enum_value(self.mode, ContainerCleanupMode, "mode"),
            )
        _assert_optional_positive_int(self.grace_seconds, "grace_seconds")

    def to_dict(self) -> dict[str, int | str | None]:
        mode = cast(ContainerCleanupMode | None, self.mode)
        return {
            "mode": mode.value if mode else None,
            "grace_seconds": self.grace_seconds,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerCleanupPolicyOverride":
        _assert_fields(raw, {"mode", "grace_seconds"}, "cleanup")
        return cls(
            mode=_optional_str(raw, "mode"),
            grace_seconds=_optional_int(raw, "grace_seconds"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerSettingsOverride:
    source: ContainerSettingsSource
    layer: ContainerSettingsPrecedence | str
    profile: str | None = None
    required: bool | None = None
    scope: ContainerExecutionScope | str | None = None
    backend: ContainerBackend | str | None = None
    image: ContainerImagePolicy | None = None
    workspace: ContainerWorkspaceMapping | None = None
    mounts: Sequence[ContainerMountDeclaration] | None = None
    environment: ContainerEnvironmentPolicy | None = None
    secrets: Sequence[ContainerSecretReference] | None = None
    network: ContainerNetworkPolicy | None = None
    devices: ContainerDevicePolicy | None = None
    resources: ContainerResourceLimits | None = None
    output: ContainerOutputPolicyOverride | None = None
    cleanup: ContainerCleanupPolicyOverride | None = None
    audit: ContainerAuditPolicy | None = None
    escalation: ContainerEscalationPolicy | None = None
    command_mode: ContainerCommandMode | str | None = None
    read_only_rootfs: bool | None = None
    user: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.source, ContainerSettingsSource)
        assert (
            self.source.trust_level is not ContainerTrustLevel.MODEL
        ), "model output cannot provide container settings overrides"
        layer = _enum_value(
            self.layer,
            ContainerSettingsPrecedence,
            "layer",
        )
        if self.profile is not None:
            _assert_profile_name(self.profile, "profile")
        if self.required is not None:
            _assert_bool(self.required, "required")
        if self.scope is not None:
            scope = _enum_value(
                self.scope,
                ContainerExecutionScope,
                "scope",
            )
            _assert_scope_allowed_for_source(self.source, scope)
            object.__setattr__(self, "scope", scope)
        if self.backend is not None:
            object.__setattr__(
                self,
                "backend",
                _enum_value(self.backend, ContainerBackend, "backend"),
            )
        if self.command_mode is not None:
            object.__setattr__(
                self,
                "command_mode",
                _enum_value(
                    self.command_mode,
                    ContainerCommandMode,
                    "command_mode",
                ),
            )
        if self.read_only_rootfs is not None:
            _assert_bool(self.read_only_rootfs, "read_only_rootfs")
        if self.user is not None:
            _assert_non_empty_string(self.user, "user")
            assert self.user != "0" and not self.user.startswith(
                "0:"
            ), "root user is unsafe"
        _assert_override_types(self)
        if not self.source.can_define_runtime_authority:
            assert layer is _settings_precedence_for_source(
                self.source
            ), "untrusted override layer must match its source"
            assert not _override_has_trusted_only_fields(
                self
            ), "untrusted sources can only select or narrow profiles"
        object.__setattr__(self, "layer", layer)
        if self.mounts is not None:
            object.__setattr__(self, "mounts", tuple(self.mounts))
        if self.secrets is not None:
            object.__setattr__(self, "secrets", tuple(self.secrets))

    def to_dict(self) -> dict[str, object]:
        layer = cast(ContainerSettingsPrecedence, self.layer)
        backend = cast(ContainerBackend | None, self.backend)
        scope = cast(ContainerExecutionScope | None, self.scope)
        command_mode = cast(ContainerCommandMode | None, self.command_mode)
        return {
            "layer": layer.value,
            "profile": self.profile,
            "required": self.required,
            "scope": scope.value if scope else None,
            "backend": backend.value if backend else None,
            "image": self.image.to_dict() if self.image else None,
            "workspace": self.workspace.to_dict() if self.workspace else None,
            "mounts": (
                None
                if self.mounts is None
                else [mount.to_dict() for mount in self.mounts]
            ),
            "environment": (
                self.environment.to_dict() if self.environment else None
            ),
            "secrets": (
                None
                if self.secrets is None
                else [secret.to_dict() for secret in self.secrets]
            ),
            "network": self.network.to_dict() if self.network else None,
            "devices": self.devices.to_dict() if self.devices else None,
            "resources": self.resources.to_dict() if self.resources else None,
            "output": self.output.to_dict() if self.output else None,
            "cleanup": self.cleanup.to_dict() if self.cleanup else None,
            "audit": self.audit.to_dict() if self.audit else None,
            "escalation": (
                self.escalation.to_dict() if self.escalation else None
            ),
            "command_mode": command_mode.value if command_mode else None,
            "read_only_rootfs": self.read_only_rootfs,
            "user": self.user,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: ContainerSettingsSource,
        layer: ContainerSettingsPrecedence | str | None = None,
    ) -> "ContainerSettingsOverride":
        assert isinstance(source, ContainerSettingsSource)
        _assert_fields(raw, _OVERRIDE_FIELDS, "override")
        inferred_layer = (
            layer
            if layer is not None
            else _settings_precedence_for_source(source)
        )
        raw_layer = _optional_str(raw, "layer")
        if raw_layer is not None:
            assert _enum_value(
                raw_layer,
                ContainerSettingsPrecedence,
                "layer",
            ) is _enum_value(
                inferred_layer,
                ContainerSettingsPrecedence,
                "layer",
            ), "override layer must be provided by the trusted loader"
        return cls(
            source=source,
            layer=inferred_layer,
            profile=_optional_str(raw, "profile"),
            required=_optional_bool(raw, "required"),
            scope=_optional_str(raw, "scope"),
            backend=_optional_str(raw, "backend"),
            image=_optional_image(raw),
            workspace=_optional_workspace(raw),
            mounts=_optional_mounts(raw),
            environment=_optional_environment(raw),
            secrets=_optional_secrets(raw),
            network=_optional_network(raw),
            devices=_optional_devices(raw),
            resources=_optional_resources(raw),
            output=_optional_output_override(raw),
            cleanup=_optional_cleanup_override(raw),
            audit=_optional_audit(raw),
            escalation=_optional_escalation(raw),
            command_mode=_optional_str(raw, "command_mode"),
            read_only_rootfs=_optional_bool(raw, "read_only_rootfs"),
            user=_optional_str(raw, "user"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerAuthorityCaps:
    settings: ContainerSettings
    layer: ContainerSettingsPrecedence | str = (
        ContainerSettingsPrecedence.SERVER_OPERATOR
    )

    def __post_init__(self) -> None:
        assert isinstance(self.settings, ContainerSettings)
        object.__setattr__(
            self,
            "layer",
            _enum_value(
                self.layer,
                ContainerSettingsPrecedence,
                "layer",
            ),
        )

    def merge(
        self,
        overrides: Sequence[ContainerSettingsOverride] = (),
    ) -> ContainerEffectiveSettings:
        ordered = _ordered_overrides(overrides)
        backend = cast(ContainerBackend, self.settings.backend)
        profile_name = self.settings.default_profile
        required = False
        scope = ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
        for override in ordered:
            if override.backend is not None:
                backend = _narrow_backend(backend, override)
            if override.profile is not None:
                assert (
                    override.profile in self.settings.allowed_profiles
                ), "selected profile must be allowed"
                if not override.source.can_define_runtime_authority:
                    assert (
                        profile_name is not None
                    ), "untrusted profile selection requires a trusted cap"
                    _assert_profile_no_wider(
                        self.settings.profiles[profile_name],
                        self.settings.profiles[override.profile],
                    )
                profile_name = override.profile
            if override.required is not None:
                required = required or override.required
            if override.scope is not None:
                scope = cast(ContainerExecutionScope, override.scope)
        if backend is ContainerBackend.NONE:
            assert (
                profile_name is None
            ), "disabled container settings cannot select a profile"
            return ContainerEffectiveSettings(
                backend=backend,
                required=required,
                scope=scope,
                source=self.settings.source,
                policy_version=self.settings.policy_version,
                profile_registry_id=self.settings.profile_registry_id,
            )
        assert profile_name is not None, "container profile is required"
        assert (
            profile_name in self.settings.allowed_profiles
        ), "selected profile must be allowed"
        profile = self.settings.profiles[profile_name]
        for override in ordered:
            profile = _narrow_profile(profile, override)
        return ContainerEffectiveSettings(
            backend=backend,
            required=required,
            scope=scope,
            source=self.settings.source,
            policy_version=self.settings.policy_version,
            profile_registry_id=self.settings.profile_registry_id,
            profile_name=profile.name,
            profile=profile,
            allowed_profiles=self.settings.allowed_profiles,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPlatformBehavior:
    file_io: str
    networking: str
    architecture_emulation: str
    resources: str
    signals: str
    path_syntax: str
    drive_letters: str
    case_behavior: str

    def __post_init__(self) -> None:
        for field_name in (
            "file_io",
            "networking",
            "architecture_emulation",
            "resources",
            "signals",
            "path_syntax",
            "drive_letters",
            "case_behavior",
        ):
            _assert_non_empty_string(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "file_io": self.file_io,
            "networking": self.networking,
            "architecture_emulation": self.architecture_emulation,
            "resources": self.resources,
            "signals": self.signals,
            "path_syntax": self.path_syntax,
            "drive_letters": self.drive_letters,
            "case_behavior": self.case_behavior,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerPlatformBehavior":
        _assert_fields(
            raw,
            {
                "file_io",
                "networking",
                "architecture_emulation",
                "resources",
                "signals",
                "path_syntax",
                "drive_letters",
                "case_behavior",
            },
            "platform_behavior",
        )
        return cls(
            file_io=_required_str(raw, "file_io", "platform_behavior"),
            networking=_required_str(raw, "networking", "platform_behavior"),
            architecture_emulation=_required_str(
                raw,
                "architecture_emulation",
                "platform_behavior",
            ),
            resources=_required_str(raw, "resources", "platform_behavior"),
            signals=_required_str(raw, "signals", "platform_behavior"),
            path_syntax=_required_str(
                raw,
                "path_syntax",
                "platform_behavior",
            ),
            drive_letters=_required_str(
                raw,
                "drive_letters",
                "platform_behavior",
            ),
            case_behavior=_required_str(
                raw,
                "case_behavior",
                "platform_behavior",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerBackendCapabilities:
    backend: ContainerBackend | str
    host_os: str
    guest_os: str
    architecture: str
    runtime_name: str | None = None
    support_level: ContainerBackendSupportLevel | str = (
        ContainerBackendSupportLevel.SUPPORTED
    )
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
    vm_backed: bool = False
    remote_engine: bool = False
    windows_process_isolation: bool = False
    windows_hyperv_isolation: bool = False
    streaming_attach: bool = False
    stats: bool = False
    lifecycle_normalization: bool = True
    platform_behavior: ContainerPlatformBehavior | None = None
    shared_mount_prefixes: Sequence[str] = field(default_factory=tuple)
    parity_requirements: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _enum_value(self.backend, ContainerBackend, "backend"),
        )
        _assert_non_empty_string(self.host_os, "host_os")
        _assert_non_empty_string(self.guest_os, "guest_os")
        _assert_non_empty_string(self.architecture, "architecture")
        if self.runtime_name is not None:
            _assert_non_empty_string(self.runtime_name, "runtime_name")
        object.__setattr__(
            self,
            "support_level",
            _enum_value(
                self.support_level,
                ContainerBackendSupportLevel,
                "support_level",
            ),
        )
        for field_name in (
            "rootless",
            "user_namespace",
            "build",
            "pull",
            "platform_emulation",
            "resource_limits",
            "per_container_vm_isolation",
            "vm_backed",
            "remote_engine",
            "windows_process_isolation",
            "windows_hyperv_isolation",
            "streaming_attach",
            "stats",
            "lifecycle_normalization",
        ):
            _assert_bool(getattr(self, field_name), field_name)
        if self.platform_behavior is not None:
            assert isinstance(
                self.platform_behavior,
                ContainerPlatformBehavior,
            )
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
        object.__setattr__(
            self,
            "shared_mount_prefixes",
            _string_tuple(
                self.shared_mount_prefixes,
                "shared_mount_prefixes",
            ),
        )
        object.__setattr__(
            self,
            "parity_requirements",
            _string_tuple(self.parity_requirements, "parity_requirements"),
        )

    def to_dict(self) -> dict[str, object]:
        backend = cast(ContainerBackend, self.backend)
        support_level = cast(
            ContainerBackendSupportLevel,
            self.support_level,
        )
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
            "runtime_name": self.runtime_name,
            "support_level": support_level.value,
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
            "vm_backed": self.vm_backed,
            "remote_engine": self.remote_engine,
            "windows_process_isolation": self.windows_process_isolation,
            "windows_hyperv_isolation": self.windows_hyperv_isolation,
            "streaming_attach": self.streaming_attach,
            "stats": self.stats,
            "lifecycle_normalization": self.lifecycle_normalization,
            "platform_behavior": (
                None
                if self.platform_behavior is None
                else self.platform_behavior.to_dict()
            ),
            "shared_mount_prefixes": list(self.shared_mount_prefixes),
            "parity_requirements": list(self.parity_requirements),
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
    pooling: ContainerPoolingPolicy = field(
        default_factory=ContainerPoolingPolicy,
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
        assert isinstance(self.pooling, ContainerPoolingPolicy)
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
            "pooling": self.pooling.to_dict(),
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
_OVERRIDE_FIELDS = {
    "layer",
    "profile",
    "required",
    "scope",
    "backend",
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
    "audit",
    "escalation",
    "command_mode",
    "read_only_rootfs",
    "user",
}
_TRUSTED_ONLY_OVERRIDE_FIELDS = (
    "backend",
    "image",
    "workspace",
    "command_mode",
    "read_only_rootfs",
    "user",
)
_PRECEDENCE_RANK = {
    layer: index for index, layer in enumerate(CONTAINER_SETTINGS_PRECEDENCE)
}
_ESCALATION_RANK = {
    ContainerEscalationMode.DENY: 0,
    ContainerEscalationMode.REQUIRE_REVIEW: 1,
    ContainerEscalationMode.PREAUTHORIZED: 2,
}


def _canonical_profile_policy(
    profile: Mapping[str, object],
) -> dict[str, object]:
    canonical = dict(profile)
    canonical["mounts"] = sorted(
        (
            dict(_mapping(mount, "mount"))
            for mount in _sequence(canonical.get("mounts", ()), "mounts")
        ),
        key=lambda mount: (
            mount.get("target"),
            mount.get("source"),
            mount.get("mount_type"),
            mount.get("access"),
        ),
    )
    canonical["secrets"] = sorted(
        (
            dict(_mapping(secret, "secret"))
            for secret in _sequence(canonical.get("secrets", ()), "secrets")
        ),
        key=lambda secret: (
            secret.get("name"),
            secret.get("env_name"),
            secret.get("mount_path"),
        ),
    )
    canonical["environment"] = _canonical_environment_policy(
        _mapping(canonical.get("environment", {}), "environment")
    )
    canonical["network"] = _canonical_network_policy(
        _mapping(canonical.get("network", {}), "network")
    )
    canonical["devices"] = _canonical_device_policy(
        _mapping(canonical.get("devices", {}), "devices")
    )
    canonical["pooling"] = _canonical_pooling_policy(
        _mapping(canonical.get("pooling", {}), "pooling")
    )
    return canonical


def _canonical_environment_policy(
    environment: Mapping[str, object],
) -> dict[str, object]:
    canonical = dict(environment)
    canonical["variables"] = dict(
        sorted(
            _string_mapping(
                canonical.get("variables", {}),
                "variables",
            ).items()
        )
    )
    canonical["allowlist"] = sorted(
        _string_tuple(canonical.get("allowlist", ()), "allowlist")
    )
    return canonical


def _canonical_network_policy(
    network: Mapping[str, object],
) -> dict[str, object]:
    canonical = dict(network)
    canonical["egress_allowlist"] = sorted(
        _string_tuple(
            canonical.get("egress_allowlist", ()),
            "egress_allowlist",
        )
    )
    return canonical


def _canonical_device_policy(
    devices: Mapping[str, object],
) -> dict[str, object]:
    canonical = dict(devices)
    canonical["devices"] = sorted(
        _string_tuple(canonical.get("devices", ()), "devices")
    )
    return canonical


def _canonical_pooling_policy(
    pooling: Mapping[str, object],
) -> dict[str, object]:
    canonical = dict(pooling)
    canonical["health_check_command"] = _string_tuple(
        canonical.get("health_check_command", ()),
        "health_check_command",
    )
    canonical["audit_labels"] = dict(
        sorted(
            _string_mapping(
                canonical.get("audit_labels", {}),
                "audit_labels",
            ).items()
        )
    )
    return canonical


def _settings_precedence_for_source(
    source: ContainerSettingsSource,
) -> ContainerSettingsPrecedence:
    surface = cast(ContainerSurface, source.surface)
    trust_level = cast(ContainerTrustLevel, source.trust_level)
    if trust_level is ContainerTrustLevel.UNTRUSTED_REQUEST:
        return ContainerSettingsPrecedence.REQUEST
    if surface is ContainerSurface.SDK:
        return ContainerSettingsPrecedence.SDK
    if surface is ContainerSurface.CLI:
        return ContainerSettingsPrecedence.CLI
    if surface is ContainerSurface.AGENT_TOML:
        return ContainerSettingsPrecedence.AGENT_TOML
    if surface is ContainerSurface.FLOW_TOML:
        return ContainerSettingsPrecedence.FLOW_TOML
    if surface is ContainerSurface.TASK_TOML:
        return ContainerSettingsPrecedence.TASK_TOML
    return ContainerSettingsPrecedence.SERVER_OPERATOR


def _assert_scope_allowed_for_source(
    source: ContainerSettingsSource,
    scope: ContainerExecutionScope,
) -> None:
    if source.can_define_runtime_authority:
        return
    assert (
        scope is ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ), "untrusted sources cannot raise container execution scope"


def _ordered_overrides(
    overrides: Sequence[ContainerSettingsOverride],
) -> tuple[ContainerSettingsOverride, ...]:
    for override in overrides:
        assert isinstance(override, ContainerSettingsOverride)
    return tuple(
        sorted(
            overrides,
            key=lambda override: _PRECEDENCE_RANK[
                cast(ContainerSettingsPrecedence, override.layer)
            ],
        )
    )


def _narrow_backend(
    backend: ContainerBackend,
    override: ContainerSettingsOverride,
) -> ContainerBackend:
    requested = cast(ContainerBackend, override.backend)
    assert (
        override.source.can_define_runtime_authority
    ), "backend selection requires trusted authority"
    assert requested in {
        backend,
        ContainerBackend.NONE,
    }, "backend override cannot widen authority caps"
    return requested


def _assert_profile_no_wider(
    caps: ContainerProfile,
    requested: ContainerProfile,
) -> None:
    _narrow_image(caps.image, requested.image)
    _narrow_workspace(caps.workspace, requested.workspace)
    _narrow_mounts(caps.mounts, requested.mounts)
    _narrow_environment(caps.environment, requested.environment)
    _narrow_secrets(caps.secrets, requested.secrets)
    _narrow_network(caps.network, requested.network)
    _narrow_devices(caps.devices, requested.devices)
    _assert_resources_no_wider(caps.resources, requested.resources)
    _narrow_output(
        caps.output,
        ContainerOutputPolicyOverride(
            max_stdout_bytes=requested.output.max_stdout_bytes,
            max_stderr_bytes=requested.output.max_stderr_bytes,
            max_artifact_bytes=requested.output.max_artifact_bytes,
            allow_artifacts=requested.output.allow_artifacts,
        ),
    )
    _narrow_cleanup(
        caps.cleanup,
        ContainerCleanupPolicyOverride(
            mode=requested.cleanup.mode,
            grace_seconds=requested.cleanup.grace_seconds,
        ),
    )
    _narrow_audit(caps.audit, requested.audit)
    _narrow_escalation(caps.escalation, requested.escalation)
    _narrow_command_mode(caps.command_mode, requested.command_mode)
    _narrow_read_only_rootfs(caps.read_only_rootfs, requested.read_only_rootfs)
    _narrow_user(caps.user, requested.user)
    assert (
        requested.pooling.to_dict() == caps.pooling.to_dict()
    ), "pooling policy cannot widen"


def _narrow_profile(
    profile: ContainerProfile,
    override: ContainerSettingsOverride,
) -> ContainerProfile:
    image = _narrow_image(profile.image, override.image)
    workspace = _narrow_workspace(profile.workspace, override.workspace)
    mounts = _narrow_mounts(profile.mounts, override.mounts)
    environment = _narrow_environment(
        profile.environment,
        override.environment,
    )
    secrets = _narrow_secrets(profile.secrets, override.secrets)
    network = _narrow_network(profile.network, override.network)
    devices = _narrow_devices(profile.devices, override.devices)
    resources = _narrow_resources(profile.resources, override.resources)
    output = _narrow_output(profile.output, override.output)
    cleanup = _narrow_cleanup(profile.cleanup, override.cleanup)
    audit = _narrow_audit(profile.audit, override.audit)
    escalation = _narrow_escalation(
        profile.escalation,
        override.escalation,
    )
    command_mode = _narrow_command_mode(
        profile.command_mode,
        override.command_mode,
    )
    read_only_rootfs = _narrow_read_only_rootfs(
        profile.read_only_rootfs,
        override.read_only_rootfs,
    )
    user = _narrow_user(profile.user, override.user)
    return ContainerProfile(
        name=profile.name,
        image=image,
        workspace=workspace,
        mounts=mounts,
        environment=environment,
        secrets=secrets,
        network=network,
        devices=devices,
        resources=resources,
        output=output,
        cleanup=cleanup,
        pooling=profile.pooling,
        audit=audit,
        escalation=escalation,
        command_mode=command_mode,
        read_only_rootfs=read_only_rootfs,
        user=user,
    )


def _narrow_image(
    caps: ContainerImagePolicy,
    requested: ContainerImagePolicy | None,
) -> ContainerImagePolicy:
    if requested is None:
        return caps
    assert (
        requested.to_dict() == caps.to_dict()
    ), "image override cannot change trusted image policy"
    return caps


def _narrow_workspace(
    caps: ContainerWorkspaceMapping,
    requested: ContainerWorkspaceMapping | None,
) -> ContainerWorkspaceMapping:
    if requested is None:
        return caps
    assert (
        requested.to_dict() == caps.to_dict()
    ), "workspace override cannot change trusted path mapping"
    return caps


def _narrow_mounts(
    caps: Sequence[ContainerMountDeclaration],
    requested: Sequence[ContainerMountDeclaration] | None,
) -> tuple[ContainerMountDeclaration, ...]:
    if requested is None:
        return tuple(caps)
    caps_by_target = _mounts_by_target(caps)
    narrowed: list[ContainerMountDeclaration] = []
    seen: set[str] = set()
    for mount in requested:
        assert mount.target not in seen, "mount overrides must be unique"
        seen.add(mount.target)
        assert mount.target in caps_by_target, "mount override must be allowed"
        narrowed.append(_narrow_mount(caps_by_target[mount.target], mount))
    return tuple(narrowed)


def _narrow_mount(
    caps: ContainerMountDeclaration,
    requested: ContainerMountDeclaration,
) -> ContainerMountDeclaration:
    caps_type = cast(ContainerMountType, caps.mount_type)
    requested_type = cast(ContainerMountType, requested.mount_type)
    caps_access = cast(ContainerMountAccess, caps.access)
    requested_access = cast(ContainerMountAccess, requested.access)
    assert requested_type is caps_type, "mount type cannot change"
    assert requested.source == caps.source, "mount source cannot change"
    assert not (
        caps_access is ContainerMountAccess.READ
        and requested_access is ContainerMountAccess.WRITE
    ), "mount access cannot widen"
    return requested


def _mounts_by_target(
    mounts: Sequence[ContainerMountDeclaration],
) -> dict[str, ContainerMountDeclaration]:
    result: dict[str, ContainerMountDeclaration] = {}
    for mount in mounts:
        assert mount.target not in result, "mount caps must be unique"
        result[mount.target] = mount
    return result


def _narrow_environment(
    caps: ContainerEnvironmentPolicy,
    requested: ContainerEnvironmentPolicy | None,
) -> ContainerEnvironmentPolicy:
    if requested is None:
        return caps
    for name, value in requested.variables.items():
        assert name in caps.variables, "environment variable must be allowed"
        assert (
            value == caps.variables[name]
        ), "environment variable value cannot change"
    for name in requested.allowlist:
        assert name in caps.allowlist, "environment allowlist cannot widen"
    return requested


def _narrow_secrets(
    caps: Sequence[ContainerSecretReference],
    requested: Sequence[ContainerSecretReference] | None,
) -> tuple[ContainerSecretReference, ...]:
    if requested is None:
        return tuple(caps)
    caps_by_name = _secrets_by_name(caps)
    narrowed: list[ContainerSecretReference] = []
    seen: set[str] = set()
    for secret in requested:
        assert secret.name not in seen, "secret overrides must be unique"
        seen.add(secret.name)
        assert secret.name in caps_by_name, "secret override must be allowed"
        assert (
            secret.to_dict() == caps_by_name[secret.name].to_dict()
        ), "secret delivery cannot change"
        narrowed.append(secret)
    return tuple(narrowed)


def _secrets_by_name(
    secrets: Sequence[ContainerSecretReference],
) -> dict[str, ContainerSecretReference]:
    result: dict[str, ContainerSecretReference] = {}
    for secret in secrets:
        assert secret.name not in result, "secret caps must be unique"
        result[secret.name] = secret
    return result


def _narrow_network(
    caps: ContainerNetworkPolicy,
    requested: ContainerNetworkPolicy | None,
) -> ContainerNetworkPolicy:
    if requested is None:
        return caps
    caps_mode = cast(ContainerNetworkMode, caps.mode)
    requested_mode = cast(ContainerNetworkMode, requested.mode)
    if requested_mode is ContainerNetworkMode.NONE:
        return requested
    if caps_mode is ContainerNetworkMode.NONE:
        assert False, "network override cannot enable network"
    if requested_mode is ContainerNetworkMode.LOOPBACK:
        assert caps_mode in {
            ContainerNetworkMode.LOOPBACK,
            ContainerNetworkMode.ALLOWLIST,
            ContainerNetworkMode.FULL,
        }, "network mode cannot widen"
        return requested
    if requested_mode is ContainerNetworkMode.FULL:
        assert (
            caps_mode is ContainerNetworkMode.FULL
        ), "network full cannot widen mode"
        assert not requested.egress_allowlist, "network full cannot use egress"
        return requested
    assert (
        requested_mode is ContainerNetworkMode.ALLOWLIST
    ), "network mode is unsupported"
    assert caps_mode in {
        ContainerNetworkMode.ALLOWLIST,
        ContainerNetworkMode.FULL,
    }, "network allowlist cannot widen mode"
    if caps_mode is ContainerNetworkMode.ALLOWLIST:
        for host in requested.egress_allowlist:
            assert (
                host in caps.egress_allowlist
            ), "network allowlist cannot widen"
    return requested


def _narrow_devices(
    caps: ContainerDevicePolicy,
    requested: ContainerDevicePolicy | None,
) -> ContainerDevicePolicy:
    if requested is None:
        return caps
    caps_devices = set(cast(tuple[ContainerDeviceClass, ...], caps.devices))
    for device in requested.devices:
        assert device in caps_devices, "device override must be allowed"
    return requested


def _narrow_resources(
    caps: ContainerResourceLimits,
    requested: ContainerResourceLimits | None,
) -> ContainerResourceLimits:
    if requested is None:
        return caps
    return ContainerResourceLimits(
        cpu_count=_narrow_optional_limit(caps.cpu_count, requested.cpu_count),
        memory_bytes=_narrow_optional_limit(
            caps.memory_bytes,
            requested.memory_bytes,
        ),
        pids=_narrow_optional_limit(caps.pids, requested.pids),
        timeout_seconds=_narrow_optional_limit(
            caps.timeout_seconds,
            requested.timeout_seconds,
        ),
    )


def _assert_resources_no_wider(
    caps: ContainerResourceLimits,
    requested: ContainerResourceLimits,
) -> None:
    _assert_optional_limit_no_wider(caps.cpu_count, requested.cpu_count)
    _assert_optional_limit_no_wider(
        caps.memory_bytes,
        requested.memory_bytes,
    )
    _assert_optional_limit_no_wider(caps.pids, requested.pids)
    _assert_optional_limit_no_wider(
        caps.timeout_seconds,
        requested.timeout_seconds,
    )


def _narrow_optional_limit(
    caps: int | None,
    requested: int | None,
) -> int | None:
    if requested is None:
        return caps
    assert caps is None or requested <= caps, "resource limit cannot widen"
    return requested


def _assert_optional_limit_no_wider(
    caps: int | None,
    requested: int | None,
) -> None:
    if caps is None:
        return
    assert (
        requested is not None and requested <= caps
    ), "resource limit cannot widen"


def _narrow_output(
    caps: ContainerOutputPolicy,
    requested: ContainerOutputPolicyOverride | None,
) -> ContainerOutputPolicy:
    if requested is None:
        return caps
    max_stdout_bytes = _narrow_required_limit(
        caps.max_stdout_bytes,
        requested.max_stdout_bytes,
        "max_stdout_bytes",
    )
    max_stderr_bytes = _narrow_required_limit(
        caps.max_stderr_bytes,
        requested.max_stderr_bytes,
        "max_stderr_bytes",
    )
    allow_artifacts = (
        caps.allow_artifacts
        if requested.allow_artifacts is None
        else requested.allow_artifacts
    )
    assert (
        not allow_artifacts or caps.allow_artifacts
    ), "artifact output cannot be enabled by override"
    max_artifact_bytes = _narrow_artifact_limit(caps, requested)
    if not allow_artifacts:
        max_artifact_bytes = 0
    return ContainerOutputPolicy(
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        max_artifact_bytes=max_artifact_bytes,
        allow_artifacts=allow_artifacts,
    )


def _narrow_required_limit(
    caps: int,
    requested: int | None,
    field_name: str,
) -> int:
    if requested is None:
        return caps
    assert requested <= caps, f"{field_name} cannot widen"
    return requested


def _narrow_artifact_limit(
    caps: ContainerOutputPolicy,
    requested: ContainerOutputPolicyOverride,
) -> int:
    if requested.max_artifact_bytes is None:
        return caps.max_artifact_bytes
    assert (
        requested.max_artifact_bytes <= caps.max_artifact_bytes
    ), "max_artifact_bytes cannot widen"
    return requested.max_artifact_bytes


def _narrow_cleanup(
    caps: ContainerCleanupPolicy,
    requested: ContainerCleanupPolicyOverride | None,
) -> ContainerCleanupPolicy:
    if requested is None:
        return caps
    caps_mode = cast(ContainerCleanupMode, caps.mode)
    requested_mode = (
        caps_mode
        if requested.mode is None
        else cast(ContainerCleanupMode, requested.mode)
    )
    assert requested_mode is caps_mode, "cleanup mode cannot change"
    grace_seconds = _narrow_required_limit(
        caps.grace_seconds,
        requested.grace_seconds,
        "grace_seconds",
    )
    return ContainerCleanupPolicy(
        mode=requested_mode,
        grace_seconds=grace_seconds,
    )


def _narrow_audit(
    caps: ContainerAuditPolicy,
    requested: ContainerAuditPolicy | None,
) -> ContainerAuditPolicy:
    if requested is None:
        return caps
    assert requested.mode is caps.mode, "audit policy cannot be reduced"
    return requested


def _narrow_escalation(
    caps: ContainerEscalationPolicy,
    requested: ContainerEscalationPolicy | None,
) -> ContainerEscalationPolicy:
    if requested is None:
        return caps
    caps_mode = cast(ContainerEscalationMode, caps.mode)
    requested_mode = cast(ContainerEscalationMode, requested.mode)
    assert (
        _ESCALATION_RANK[requested_mode] <= _ESCALATION_RANK[caps_mode]
    ), "escalation policy cannot widen"
    return requested


def _narrow_command_mode(
    caps: ContainerCommandMode | str,
    requested: ContainerCommandMode | str | None,
) -> ContainerCommandMode:
    caps_mode = cast(ContainerCommandMode, caps)
    if requested is None:
        return caps_mode
    requested_mode = cast(ContainerCommandMode, requested)
    assert requested_mode is caps_mode, "command mode cannot change"
    return caps_mode


def _narrow_read_only_rootfs(caps: bool, requested: bool | None) -> bool:
    if requested is None:
        return caps
    assert requested and caps, "root filesystem cannot be made writable"
    return caps


def _narrow_user(caps: str, requested: str | None) -> str:
    if requested is None:
        return caps
    assert requested == caps, "user cannot change"
    return caps


def _assert_override_types(override: ContainerSettingsOverride) -> None:
    if override.image is not None:
        assert isinstance(override.image, ContainerImagePolicy)
    if override.workspace is not None:
        assert isinstance(override.workspace, ContainerWorkspaceMapping)
    if override.mounts is not None:
        for mount in override.mounts:
            assert isinstance(mount, ContainerMountDeclaration)
    if override.environment is not None:
        assert isinstance(override.environment, ContainerEnvironmentPolicy)
    if override.secrets is not None:
        for secret in override.secrets:
            assert isinstance(secret, ContainerSecretReference)
    if override.network is not None:
        assert isinstance(override.network, ContainerNetworkPolicy)
    if override.devices is not None:
        assert isinstance(override.devices, ContainerDevicePolicy)
    if override.resources is not None:
        assert isinstance(override.resources, ContainerResourceLimits)
    if override.output is not None:
        assert isinstance(override.output, ContainerOutputPolicyOverride)
    if override.cleanup is not None:
        assert isinstance(override.cleanup, ContainerCleanupPolicyOverride)
    if override.audit is not None:
        assert isinstance(override.audit, ContainerAuditPolicy)
    if override.escalation is not None:
        assert isinstance(override.escalation, ContainerEscalationPolicy)


def _override_has_trusted_only_fields(
    override: ContainerSettingsOverride,
) -> bool:
    return any(
        getattr(override, field_name) is not None
        for field_name in _TRUSTED_ONLY_OVERRIDE_FIELDS
    )


def _optional_image(
    raw: Mapping[str, object],
) -> ContainerImagePolicy | None:
    value = raw.get("image")
    if value is None:
        return None
    return ContainerImagePolicy.from_dict(_mapping(value, "image"))


def _optional_workspace(
    raw: Mapping[str, object],
) -> ContainerWorkspaceMapping | None:
    value = raw.get("workspace")
    if value is None:
        return None
    return ContainerWorkspaceMapping.from_dict(_mapping(value, "workspace"))


def _optional_mounts(
    raw: Mapping[str, object],
) -> tuple[ContainerMountDeclaration, ...] | None:
    value = raw.get("mounts")
    if value is None:
        return None
    return tuple(
        ContainerMountDeclaration.from_dict(_mapping(item, "mount"))
        for item in _sequence(value, "mounts")
    )


def _optional_environment(
    raw: Mapping[str, object],
) -> ContainerEnvironmentPolicy | None:
    value = raw.get("environment")
    if value is None:
        return None
    return ContainerEnvironmentPolicy.from_dict(_mapping(value, "environment"))


def _optional_secrets(
    raw: Mapping[str, object],
) -> tuple[ContainerSecretReference, ...] | None:
    value = raw.get("secrets")
    if value is None:
        return None
    return tuple(
        ContainerSecretReference.from_dict(_mapping(item, "secret"))
        for item in _sequence(value, "secrets")
    )


def _optional_network(
    raw: Mapping[str, object],
) -> ContainerNetworkPolicy | None:
    value = raw.get("network")
    if value is None:
        return None
    return ContainerNetworkPolicy.from_dict(_mapping(value, "network"))


def _optional_devices(
    raw: Mapping[str, object],
) -> ContainerDevicePolicy | None:
    value = raw.get("devices")
    if value is None:
        return None
    return ContainerDevicePolicy.from_dict(_mapping(value, "devices"))


def _optional_resources(
    raw: Mapping[str, object],
) -> ContainerResourceLimits | None:
    value = raw.get("resources")
    if value is None:
        return None
    return ContainerResourceLimits.from_dict(_mapping(value, "resources"))


def _optional_output_override(
    raw: Mapping[str, object],
) -> ContainerOutputPolicyOverride | None:
    value = raw.get("output")
    if value is None:
        return None
    return ContainerOutputPolicyOverride.from_dict(_mapping(value, "output"))


def _optional_cleanup_override(
    raw: Mapping[str, object],
) -> ContainerCleanupPolicyOverride | None:
    value = raw.get("cleanup")
    if value is None:
        return None
    return ContainerCleanupPolicyOverride.from_dict(_mapping(value, "cleanup"))


def _optional_audit(
    raw: Mapping[str, object],
) -> ContainerAuditPolicy | None:
    value = raw.get("audit")
    if value is None:
        return None
    return ContainerAuditPolicy.from_dict(_mapping(value, "audit"))


def _optional_escalation(
    raw: Mapping[str, object],
) -> ContainerEscalationPolicy | None:
    value = raw.get("escalation")
    if value is None:
        return None
    return ContainerEscalationPolicy.from_dict(_mapping(value, "escalation"))


def _optional_build_context(
    raw: Mapping[str, object],
) -> ContainerBuildContextPolicy | None:
    value = raw.get("build_context")
    if value is None:
        return None
    return ContainerBuildContextPolicy.from_dict(
        _mapping(value, "build_context")
    )


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


def _required_int(raw: Mapping[str, object], key: str, path: str) -> int:
    value = _required(raw, key, path)
    assert isinstance(value, int) and not isinstance(value, bool)
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


def _assert_optional_non_negative_int(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= 0, f"{field_name} must not be negative"


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


def _assert_bounded_relative_path(
    value: object,
    field_name: str,
    *,
    allow_remote: bool = False,
) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    is_remote = "://" in value
    assert allow_remote or not is_remote, f"{field_name} cannot be remote"
    if is_remote:
        return
    normalized = value.replace("\\", "/")
    assert not normalized.startswith("/"), f"{field_name} must be relative"
    assert not normalized.startswith("~"), f"{field_name} must not expand home"
    assert ":" not in normalized, f"{field_name} must not include drive"
    assert normalized != "", f"{field_name} must name a path"
    assert ".." not in normalized.split(
        "/"
    ), f"{field_name} must stay inside build context"


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
