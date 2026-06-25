from ..container import (
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerProfileSelection,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
)
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from json import dumps as json_dumps
from json import loads as json_loads
from re import compile as compile_pattern
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

_PROFILE_NAME_PATTERN = compile_pattern(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class IsolationMode(StrEnum):
    LOCAL = "local"
    SANDBOX = "sandbox"
    CONTAINER = "container"


class SandboxBackend(StrEnum):
    SEATBELT = "seatbelt"
    BUBBLEWRAP = "bubblewrap"


class IsolationSettingsSurface(StrEnum):
    SDK = "sdk"
    CLI = "cli"
    AGENT_TOML = "agent_toml"
    FLOW_TOML = "flow_toml"
    TASK_TOML = "task_toml"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"
    RUNTIME_ENVELOPE = "runtime_envelope"
    MODEL_BACKEND = "model_backend"
    WORKER = "worker"


class IsolationTrustLevel(StrEnum):
    TRUSTED_OPERATOR = "trusted_operator"
    TRUSTED_DEPLOYMENT = "trusted_deployment"
    TRUSTED_SDK = "trusted_sdk"
    TRUSTED_CLI = "trusted_cli"
    UNTRUSTED_AGENT = "untrusted_agent"
    UNTRUSTED_FLOW = "untrusted_flow"
    UNTRUSTED_TASK = "untrusted_task"
    UNTRUSTED_REQUEST = "untrusted_request"
    MODEL = "model"


class IsolationDiagnosticCategory(StrEnum):
    UNSUPPORTED = "unsupported"
    VALUE = "value"
    SECURITY = "security"


class IsolationDiagnosticCode(StrEnum):
    MODE_CONFLICT = "isolation.mode_conflict"
    UNSUPPORTED_MODE = "isolation.unsupported_mode"
    UNSUPPORTED_BACKEND = "isolation.unsupported_backend"
    POLICY_WIDENING = "isolation.policy_widening"
    UNSUPPORTED_SYNTAX = "isolation.unsupported_syntax"


class SandboxNetworkMode(StrEnum):
    NONE = "none"
    LOOPBACK = "loopback"
    ALLOWLIST = "allowlist"
    FULL = "full"


class SandboxChildProcessPolicy(StrEnum):
    DENY = "deny"
    ALLOW = "allow"


class SandboxInheritedFdPolicy(StrEnum):
    DENY = "deny"
    STDIO = "stdio"
    EXPLICIT = "explicit"


class SandboxCleanupPolicy(StrEnum):
    DELETE = "delete"
    QUARANTINE = "quarantine"
    PRESERVE = "preserve"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationDiagnostic:
    code: IsolationDiagnosticCode | str
    path: str
    message: str
    hint: str
    category: IsolationDiagnosticCategory | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code",
            _enum_value(self.code, IsolationDiagnosticCode, "code"),
        )
        object.__setattr__(
            self,
            "category",
            _enum_value(
                self.category,
                IsolationDiagnosticCategory,
                "category",
            ),
        )
        _assert_non_empty_string(self.path, "path")
        _assert_non_empty_string(self.message, "message")
        _assert_non_empty_string(self.hint, "hint")

    def to_dict(self) -> dict[str, str]:
        code = cast(IsolationDiagnosticCode, self.code)
        category = cast(IsolationDiagnosticCategory, self.category)
        return {
            "code": code.value,
            "path": self.path,
            "category": category.value,
            "message": self.message,
            "hint": self.hint,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationSettingsSource:
    surface: IsolationSettingsSurface | str
    trust_level: IsolationTrustLevel | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface",
            _enum_value(
                self.surface,
                IsolationSettingsSurface,
                "surface",
            ),
        )
        object.__setattr__(
            self,
            "trust_level",
            _enum_value(
                self.trust_level,
                IsolationTrustLevel,
                "trust_level",
            ),
        )

    @property
    def can_define_runtime_authority(self) -> bool:
        return self.trust_level in {
            IsolationTrustLevel.TRUSTED_OPERATOR,
            IsolationTrustLevel.TRUSTED_DEPLOYMENT,
            IsolationTrustLevel.TRUSTED_SDK,
            IsolationTrustLevel.TRUSTED_CLI,
        }

    def to_container_source(self) -> ContainerSettingsSource:
        surface = cast(IsolationSettingsSurface, self.surface)
        trust_level = cast(IsolationTrustLevel, self.trust_level)
        return ContainerSettingsSource(
            surface=_CONTAINER_SURFACES[surface],
            trust_level=_CONTAINER_TRUST_LEVELS[trust_level],
        )

    def to_dict(self) -> dict[str, str]:
        surface = cast(IsolationSettingsSurface, self.surface)
        trust_level = cast(IsolationTrustLevel, self.trust_level)
        return {
            "surface": surface.value,
            "trust_level": trust_level.value,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "IsolationSettingsSource":
        _assert_fields(raw, {"surface", "trust_level"}, "source")
        return cls(
            surface=_required_str(raw, "surface", "source"),
            trust_level=_required_str(raw, "trust_level", "source"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class LocalIsolationPolicy:
    approval_required: bool = True
    allowed_roots: Sequence[str] = field(default_factory=tuple)
    executable_allowlist: Sequence[str] = field(default_factory=tuple)
    timeout_seconds: int | None = None
    max_stdout_bytes: int = 65536
    max_stderr_bytes: int = 32768
    isolated: bool = False

    def __post_init__(self) -> None:
        _assert_bool(self.approval_required, "approval_required")
        _assert_bool(self.isolated, "isolated")
        assert not self.isolated, "local mode cannot be marked isolated"
        _assert_optional_positive_int(
            self.timeout_seconds,
            "timeout_seconds",
        )
        _assert_positive_int(self.max_stdout_bytes, "max_stdout_bytes")
        _assert_positive_int(self.max_stderr_bytes, "max_stderr_bytes")
        object.__setattr__(
            self,
            "allowed_roots",
            _path_tuple(self.allowed_roots, "allowed_roots"),
        )
        object.__setattr__(
            self,
            "executable_allowlist",
            _path_tuple(
                self.executable_allowlist,
                "executable_allowlist",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "approval_required": self.approval_required,
            "allowed_roots": list(self.allowed_roots),
            "executable_allowlist": list(self.executable_allowlist),
            "timeout_seconds": self.timeout_seconds,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "isolated": self.isolated,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "LocalIsolationPolicy":
        _assert_fields(raw, _LOCAL_FIELDS, "local")
        return cls(
            approval_required=_optional_bool_or_default(
                raw,
                "approval_required",
                True,
            ),
            allowed_roots=_path_tuple(
                raw.get("allowed_roots", ()),
                "allowed_roots",
            ),
            executable_allowlist=_path_tuple(
                raw.get("executable_allowlist", ()),
                "executable_allowlist",
            ),
            timeout_seconds=_optional_int(raw, "timeout_seconds"),
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
            isolated=_optional_bool_or_default(raw, "isolated", False),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxEnvironmentPolicy:
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
        assert (
            not self.inherit_host
        ), "sandbox profiles cannot inherit the host environment"
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
    ) -> "SandboxEnvironmentPolicy":
        _assert_fields(raw, _SANDBOX_ENVIRONMENT_FIELDS, "environment")
        return cls(
            variables=_string_mapping(raw.get("variables", {}), "variables"),
            allowlist=_string_tuple(raw.get("allowlist", ()), "allowlist"),
            inherit_host=_optional_bool_or_default(
                raw,
                "inherit_host",
                False,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxNetworkPolicy:
    mode: SandboxNetworkMode | str = SandboxNetworkMode.NONE
    egress_allowlist: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        mode = _enum_value(self.mode, SandboxNetworkMode, "mode")
        egress_allowlist = _string_tuple(
            self.egress_allowlist,
            "egress_allowlist",
        )
        assert not (
            mode is SandboxNetworkMode.NONE and egress_allowlist
        ), "network none cannot define egress"
        assert not (
            mode is not SandboxNetworkMode.ALLOWLIST and egress_allowlist
        ), "network egress allowlist requires allowlist mode"
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "egress_allowlist", egress_allowlist)

    def to_dict(self) -> dict[str, object]:
        mode = cast(SandboxNetworkMode, self.mode)
        return {
            "mode": mode.value,
            "egress_allowlist": list(self.egress_allowlist),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SandboxNetworkPolicy":
        _assert_fields(raw, _SANDBOX_NETWORK_FIELDS, "network")
        return cls(
            mode=_optional_str_or_default(
                raw,
                "mode",
                SandboxNetworkMode.NONE.value,
            ),
            egress_allowlist=_string_tuple(
                raw.get("egress_allowlist", ()),
                "egress_allowlist",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxResourceLimits:
    timeout_seconds: int | None = None
    pids: int | None = None

    def __post_init__(self) -> None:
        _assert_optional_positive_int(
            self.timeout_seconds,
            "timeout_seconds",
        )
        _assert_optional_positive_int(self.pids, "pids")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "pids": self.pids,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SandboxResourceLimits":
        _assert_fields(raw, _SANDBOX_RESOURCE_FIELDS, "resources")
        return cls(
            timeout_seconds=_optional_int(raw, "timeout_seconds"),
            pids=_optional_int(raw, "pids"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxOutputPolicy:
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
    def from_dict(cls, raw: Mapping[str, object]) -> "SandboxOutputPolicy":
        _assert_fields(raw, _SANDBOX_OUTPUT_FIELDS, "output")
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
class SandboxProfile:
    name: str
    trusted_executables: Sequence[str] = field(default_factory=tuple)
    executable_search_roots: Sequence[str] = field(default_factory=tuple)
    read_roots: Sequence[str] = field(default_factory=tuple)
    write_roots: Sequence[str] = field(default_factory=tuple)
    deny_roots: Sequence[str] = field(default_factory=tuple)
    scratch_roots: Sequence[str] = field(default_factory=tuple)
    output_roots: Sequence[str] = field(default_factory=tuple)
    environment: SandboxEnvironmentPolicy = field(
        default_factory=SandboxEnvironmentPolicy,
    )
    network: SandboxNetworkPolicy = field(default_factory=SandboxNetworkPolicy)
    resources: SandboxResourceLimits = field(
        default_factory=SandboxResourceLimits,
    )
    output: SandboxOutputPolicy = field(default_factory=SandboxOutputPolicy)
    child_processes: SandboxChildProcessPolicy | str = (
        SandboxChildProcessPolicy.DENY
    )
    inherited_fds: SandboxInheritedFdPolicy | str = (
        SandboxInheritedFdPolicy.STDIO
    )
    cleanup: SandboxCleanupPolicy | str = SandboxCleanupPolicy.DELETE

    def __post_init__(self) -> None:
        _assert_profile_name(self.name, "name")
        assert isinstance(self.environment, SandboxEnvironmentPolicy)
        assert isinstance(self.network, SandboxNetworkPolicy)
        assert isinstance(self.resources, SandboxResourceLimits)
        assert isinstance(self.output, SandboxOutputPolicy)
        object.__setattr__(
            self,
            "trusted_executables",
            _absolute_path_tuple(
                self.trusted_executables,
                "trusted_executables",
            ),
        )
        object.__setattr__(
            self,
            "executable_search_roots",
            _absolute_path_tuple(
                self.executable_search_roots,
                "executable_search_roots",
            ),
        )
        object.__setattr__(
            self,
            "read_roots",
            _absolute_path_tuple(self.read_roots, "read_roots"),
        )
        object.__setattr__(
            self,
            "write_roots",
            _absolute_path_tuple(self.write_roots, "write_roots"),
        )
        object.__setattr__(
            self,
            "deny_roots",
            _absolute_path_tuple(self.deny_roots, "deny_roots"),
        )
        object.__setattr__(
            self,
            "scratch_roots",
            _absolute_path_tuple(self.scratch_roots, "scratch_roots"),
        )
        object.__setattr__(
            self,
            "output_roots",
            _absolute_path_tuple(self.output_roots, "output_roots"),
        )
        object.__setattr__(
            self,
            "child_processes",
            _enum_value(
                self.child_processes,
                SandboxChildProcessPolicy,
                "child_processes",
            ),
        )
        object.__setattr__(
            self,
            "inherited_fds",
            _enum_value(
                self.inherited_fds,
                SandboxInheritedFdPolicy,
                "inherited_fds",
            ),
        )
        object.__setattr__(
            self,
            "cleanup",
            _enum_value(
                self.cleanup,
                SandboxCleanupPolicy,
                "cleanup",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        child_processes = cast(
            SandboxChildProcessPolicy,
            self.child_processes,
        )
        inherited_fds = cast(SandboxInheritedFdPolicy, self.inherited_fds)
        cleanup = cast(SandboxCleanupPolicy, self.cleanup)
        return {
            "name": self.name,
            "trusted_executables": list(self.trusted_executables),
            "executable_search_roots": list(self.executable_search_roots),
            "read_roots": list(self.read_roots),
            "write_roots": list(self.write_roots),
            "deny_roots": list(self.deny_roots),
            "scratch_roots": list(self.scratch_roots),
            "output_roots": list(self.output_roots),
            "environment": self.environment.to_dict(),
            "network": self.network.to_dict(),
            "resources": self.resources.to_dict(),
            "output": self.output.to_dict(),
            "child_processes": child_processes.value,
            "inherited_fds": inherited_fds.value,
            "cleanup": cleanup.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SandboxProfile":
        _assert_fields(raw, _SANDBOX_PROFILE_FIELDS, "sandbox profile")
        return cls(
            name=_required_str(raw, "name", "sandbox profile"),
            trusted_executables=_path_tuple(
                raw.get("trusted_executables", ()),
                "trusted_executables",
            ),
            executable_search_roots=_path_tuple(
                raw.get("executable_search_roots", ()),
                "executable_search_roots",
            ),
            read_roots=_path_tuple(raw.get("read_roots", ()), "read_roots"),
            write_roots=_path_tuple(
                raw.get("write_roots", ()),
                "write_roots",
            ),
            deny_roots=_path_tuple(raw.get("deny_roots", ()), "deny_roots"),
            scratch_roots=_path_tuple(
                raw.get("scratch_roots", ()),
                "scratch_roots",
            ),
            output_roots=_path_tuple(
                raw.get("output_roots", ()),
                "output_roots",
            ),
            environment=SandboxEnvironmentPolicy.from_dict(
                _mapping(raw.get("environment", {}), "environment")
            ),
            network=SandboxNetworkPolicy.from_dict(
                _mapping(raw.get("network", {}), "network")
            ),
            resources=SandboxResourceLimits.from_dict(
                _mapping(raw.get("resources", {}), "resources")
            ),
            output=SandboxOutputPolicy.from_dict(
                _mapping(raw.get("output", {}), "output")
            ),
            child_processes=_optional_str_or_default(
                raw,
                "child_processes",
                SandboxChildProcessPolicy.DENY.value,
            ),
            inherited_fds=_optional_str_or_default(
                raw,
                "inherited_fds",
                SandboxInheritedFdPolicy.STDIO.value,
            ),
            cleanup=_optional_str_or_default(
                raw,
                "cleanup",
                SandboxCleanupPolicy.DELETE.value,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxProfileSelection:
    profile: str | None = None
    required: bool = False

    def __post_init__(self) -> None:
        if self.profile is not None:
            _assert_profile_name(self.profile, "profile")
        _assert_bool(self.required, "required")

    def to_dict(self) -> dict[str, object]:
        return {"profile": self.profile, "required": self.required}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: IsolationSettingsSource,
    ) -> "SandboxProfileSelection":
        assert isinstance(source, IsolationSettingsSource)
        assert (
            source.trust_level is not IsolationTrustLevel.MODEL
        ), "model output cannot select sandbox runtime profiles"
        _assert_fields(raw, _SANDBOX_SELECTION_FIELDS, "sandbox selection")
        return cls(
            profile=_optional_str(raw, "profile"),
            required=_optional_bool_or_default(raw, "required", False),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxSettings:
    source: IsolationSettingsSource
    backend: SandboxBackend | str
    default_profile: str | None = None
    allowed_profiles: Sequence[str] = field(default_factory=tuple)
    profiles: Mapping[str, SandboxProfile] = field(default_factory=dict)
    profile_registry_id: str = "default"
    policy_version: str = "phase2"

    def __post_init__(self) -> None:
        assert isinstance(self.source, IsolationSettingsSource)
        assert (
            self.source.can_define_runtime_authority
        ), "untrusted sources cannot define sandbox runtime authority"
        backend = _enum_value(self.backend, SandboxBackend, "backend")
        profiles = _sandbox_profile_mapping(self.profiles)
        assert profiles, "sandbox settings require profiles"
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
        _assert_profile_name(self.profile_registry_id, "profile_registry_id")
        _assert_profile_name(self.policy_version, "policy_version")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        object.__setattr__(self, "allowed_profiles", allowed_profiles)

    def select_profile(
        self,
        selection: SandboxProfileSelection,
    ) -> "SandboxEffectiveSettings":
        assert isinstance(selection, SandboxProfileSelection)
        selected_profile = selection.profile or self.default_profile
        assert selected_profile is not None, "sandbox profile is required"
        assert (
            selected_profile in self.allowed_profiles
        ), "selected profile must be allowed"
        return SandboxEffectiveSettings(
            backend=self.backend,
            required=selection.required,
            source=self.source,
            policy_version=self.policy_version,
            profile_registry_id=self.profile_registry_id,
            profile_name=selected_profile,
            profile=self.profiles[selected_profile],
            allowed_profiles=self.allowed_profiles,
        )

    def to_dict(self) -> dict[str, object]:
        backend = cast(SandboxBackend, self.backend)
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
        source: IsolationSettingsSource,
    ) -> "SandboxSettings":
        assert isinstance(source, IsolationSettingsSource)
        _assert_fields(raw, _SANDBOX_SETTINGS_FIELDS, "sandbox")
        return cls(
            source=source,
            backend=_required_str(raw, "backend", "sandbox"),
            default_profile=_optional_str(raw, "default_profile"),
            allowed_profiles=_string_tuple(
                raw.get("allowed_profiles", ()),
                "allowed_profiles",
            ),
            profiles=_sandbox_profile_mapping_from_dict(
                _mapping(raw.get("profiles", {}), "profiles")
            ),
            profile_registry_id=_optional_str_or_default(
                raw,
                "profile_registry_id",
                "default",
            ),
            policy_version=_optional_str_or_default(
                raw,
                "policy_version",
                "phase2",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxEffectiveSettings:
    backend: SandboxBackend | str
    required: bool
    source: IsolationSettingsSource
    policy_version: str
    profile_registry_id: str
    profile_name: str
    profile: SandboxProfile
    allowed_profiles: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        backend = _enum_value(self.backend, SandboxBackend, "backend")
        _assert_bool(self.required, "required")
        assert isinstance(self.source, IsolationSettingsSource)
        _assert_profile_name(self.profile_registry_id, "profile_registry_id")
        _assert_profile_name(self.policy_version, "policy_version")
        _assert_profile_name(self.profile_name, "profile_name")
        assert isinstance(self.profile, SandboxProfile)
        assert (
            self.profile_name == self.profile.name
        ), "profile_name must match profile"
        allowed_profiles = _string_tuple(
            self.allowed_profiles,
            "allowed_profiles",
        )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "allowed_profiles", allowed_profiles)

    def to_dict(self) -> dict[str, object]:
        backend = cast(SandboxBackend, self.backend)
        return {
            "backend": backend.value,
            "required": self.required,
            "source": self.source.to_dict(),
            "policy_version": self.policy_version,
            "profile_registry_id": self.profile_registry_id,
            "profile_name": self.profile_name,
            "profile": self.profile.to_dict(),
            "allowed_profiles": list(self.allowed_profiles),
        }

    def canonical_policy_input(self) -> dict[str, object]:
        serialized = self.to_dict()
        serialized.pop("source")
        serialized["allowed_profiles"] = sorted(self.allowed_profiles)
        serialized["profile"] = _canonical_sandbox_profile(self.profile)
        return serialized

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "SandboxEffectiveSettings":
        _assert_fields(raw, _SANDBOX_EFFECTIVE_FIELDS, "sandbox effective")
        return cls(
            backend=_required_str(raw, "backend", "sandbox effective"),
            required=_required_bool(raw, "required", "sandbox effective"),
            source=IsolationSettingsSource.from_dict(
                _mapping(
                    _required(raw, "source", "sandbox effective"),
                    "source",
                )
            ),
            policy_version=_required_str(
                raw,
                "policy_version",
                "sandbox effective",
            ),
            profile_registry_id=_required_str(
                raw,
                "profile_registry_id",
                "sandbox effective",
            ),
            profile_name=_required_str(
                raw,
                "profile_name",
                "sandbox effective",
            ),
            profile=SandboxProfile.from_dict(
                _mapping(
                    _required(raw, "profile", "sandbox effective"),
                    "profile",
                )
            ),
            allowed_profiles=_string_tuple(
                raw.get("allowed_profiles", ()),
                "allowed_profiles",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationProfileSelection:
    mode: IsolationMode | str | None = None
    profile: str | None = None
    required: bool = False
    scope: ContainerExecutionScope | str = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    )

    def __post_init__(self) -> None:
        if self.mode is not None:
            object.__setattr__(
                self,
                "mode",
                _enum_value(self.mode, IsolationMode, "mode"),
            )
        if self.profile is not None:
            _assert_profile_name(self.profile, "profile")
        _assert_bool(self.required, "required")
        object.__setattr__(
            self,
            "scope",
            _enum_value(self.scope, ContainerExecutionScope, "scope"),
        )

    def to_dict(self) -> dict[str, object]:
        mode = cast(IsolationMode | None, self.mode)
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "mode": None if mode is None else mode.value,
            "profile": self.profile,
            "required": self.required,
            "scope": scope.value,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: IsolationSettingsSource,
    ) -> "IsolationProfileSelection":
        assert isinstance(source, IsolationSettingsSource)
        assert (
            source.trust_level is not IsolationTrustLevel.MODEL
        ), "model output cannot select isolation runtime profiles"
        _assert_fields(raw, _ISOLATION_SELECTION_FIELDS, "selection")
        scope = _optional_str_or_default(
            raw,
            "scope",
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION.value,
        )
        ContainerProfileSelection.from_dict(
            {"scope": scope},
            source=source.to_container_source(),
        )
        return cls(
            mode=_optional_str(raw, "mode"),
            profile=_optional_str(raw, "profile"),
            required=_optional_bool_or_default(raw, "required", False),
            scope=scope,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationSettings:
    source: IsolationSettingsSource
    mode: IsolationMode | str
    local: LocalIsolationPolicy | None = None
    sandbox: SandboxSettings | None = None
    container: ContainerSettings | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.source, IsolationSettingsSource)
        assert (
            self.source.can_define_runtime_authority
        ), "untrusted sources cannot define isolation runtime authority"
        mode = _enum_value(self.mode, IsolationMode, "mode")
        _assert_tagged_union(
            mode,
            self.local,
            self.sandbox,
            self.container,
            "settings",
        )
        if self.local is not None:
            assert isinstance(self.local, LocalIsolationPolicy)
        if self.sandbox is not None:
            assert isinstance(self.sandbox, SandboxSettings)
        if self.container is not None:
            assert isinstance(self.container, ContainerSettings)
            assert self.container.enabled, (
                "container isolation settings require an enabled container "
                "policy"
            )
        object.__setattr__(self, "mode", mode)

    def select_profile(
        self,
        selection: IsolationProfileSelection,
    ) -> "IsolationEffectiveSettings":
        assert isinstance(selection, IsolationProfileSelection)
        selected_mode = selection.mode or self.mode
        assert (
            selected_mode == self.mode
        ), "profile selection cannot change isolation mode"
        mode = cast(IsolationMode, self.mode)
        match mode:
            case IsolationMode.LOCAL:
                assert (
                    selection.profile is None
                ), "local isolation cannot select profiles"
                assert self.local is not None
                return IsolationEffectiveSettings(
                    mode=mode,
                    source=self.source,
                    local=self.local,
                )
            case IsolationMode.SANDBOX:
                assert self.sandbox is not None
                return IsolationEffectiveSettings(
                    mode=mode,
                    source=self.source,
                    sandbox=self.sandbox.select_profile(
                        SandboxProfileSelection(
                            profile=selection.profile,
                            required=selection.required,
                        )
                    ),
                )
            case IsolationMode.CONTAINER:
                assert self.container is not None
                return IsolationEffectiveSettings(
                    mode=mode,
                    source=self.source,
                    container=self.container.select_profile(
                        ContainerProfileSelection(
                            profile=selection.profile,
                            required=selection.required,
                            scope=selection.scope,
                        )
                    ),
                )

    def to_dict(self) -> dict[str, object]:
        mode = cast(IsolationMode, self.mode)
        return {
            "source": self.source.to_dict(),
            "mode": mode.value,
            "local": self.local.to_dict() if self.local else None,
            "sandbox": self.sandbox.to_dict() if self.sandbox else None,
            "container": self.container.to_dict() if self.container else None,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: IsolationSettingsSource,
    ) -> "IsolationSettings":
        assert isinstance(source, IsolationSettingsSource)
        _assert_fields(raw, _ISOLATION_SETTINGS_FIELDS, "isolation")
        mode = _enum_value(
            _optional_str_or_default(raw, "mode", IsolationMode.LOCAL.value),
            IsolationMode,
            "mode",
        )
        return cls(
            source=source,
            mode=mode,
            local=_local_policy_from_raw(raw, mode),
            sandbox=_sandbox_settings_from_raw(raw, source),
            container=_container_settings_from_raw(raw, source),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationEffectiveSettings:
    mode: IsolationMode | str
    source: IsolationSettingsSource
    local: LocalIsolationPolicy | None = None
    sandbox: SandboxEffectiveSettings | None = None
    container: ContainerEffectiveSettings | None = None

    def __post_init__(self) -> None:
        mode = _enum_value(self.mode, IsolationMode, "mode")
        assert isinstance(self.source, IsolationSettingsSource)
        _assert_tagged_union(
            mode,
            self.local,
            self.sandbox,
            self.container,
            "effective settings",
        )
        if self.local is not None:
            assert isinstance(self.local, LocalIsolationPolicy)
        if self.sandbox is not None:
            assert isinstance(self.sandbox, SandboxEffectiveSettings)
        if self.container is not None:
            assert isinstance(self.container, ContainerEffectiveSettings)
            assert (
                self.container.enabled
            ), "container isolation requires an enabled container policy"
        object.__setattr__(self, "mode", mode)

    def to_dict(self) -> dict[str, object]:
        mode = cast(IsolationMode, self.mode)
        return {
            "mode": mode.value,
            "source": self.source.to_dict(),
            "local": self.local.to_dict() if self.local else None,
            "sandbox": self.sandbox.to_dict() if self.sandbox else None,
            "container": self.container.to_dict() if self.container else None,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    def canonical_policy_input(self) -> dict[str, object]:
        mode = cast(IsolationMode, self.mode)
        match mode:
            case IsolationMode.LOCAL:
                assert self.local is not None
                branch: object = self.local.to_dict()
            case IsolationMode.SANDBOX:
                assert self.sandbox is not None
                branch = self.sandbox.canonical_policy_input()
            case IsolationMode.CONTAINER:
                assert self.container is not None
                branch = self.container.canonical_policy_input()
        return {"mode": mode.value, mode.value: branch}

    def canonical_json(self) -> str:
        return _stable_json(self.canonical_policy_input())

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "IsolationEffectiveSettings":
        _assert_fields(raw, _ISOLATION_EFFECTIVE_FIELDS, "effective")
        mode = _enum_value(
            _required_str(raw, "mode", "effective"),
            IsolationMode,
            "mode",
        )
        return cls(
            mode=mode,
            source=IsolationSettingsSource.from_dict(
                _mapping(_required(raw, "source", "effective"), "source")
            ),
            local=_local_effective_from_raw(raw, mode),
            sandbox=_sandbox_effective_from_raw(raw),
            container=_container_effective_from_raw(raw),
        )

    @classmethod
    def from_json(cls, serialized: str) -> "IsolationEffectiveSettings":
        loaded = json_loads(serialized)
        return cls.from_dict(_mapping(loaded, "effective"))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationToolRuntimeSettings:
    effective_settings: IsolationEffectiveSettings
    authorization_provider: Callable[[object], object] | None = None
    audit_listeners: Sequence[Callable[[object], object]] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.effective_settings, IsolationEffectiveSettings)
        if self.authorization_provider is not None:
            assert callable(
                self.authorization_provider
            ), "authorization_provider must be callable"
        for listener in self.audit_listeners:
            assert callable(listener), "audit listeners must be callable"
        object.__setattr__(
            self,
            "audit_listeners",
            tuple(self.audit_listeners),
        )

    @property
    def mode(self) -> IsolationMode:
        return cast(IsolationMode, self.effective_settings.mode)

    @property
    def local(self) -> LocalIsolationPolicy | None:
        return self.effective_settings.local

    @property
    def sandbox(self) -> SandboxEffectiveSettings | None:
        return self.effective_settings.sandbox

    @property
    def container(self) -> ContainerEffectiveSettings | None:
        return self.effective_settings.container

    def to_dict(self) -> dict[str, object]:
        return {"effective_settings": self.effective_settings.to_dict()}


def trusted_isolation_source(
    surface: IsolationSettingsSurface | str,
) -> IsolationSettingsSource:
    normalized = _enum_value(surface, IsolationSettingsSurface, "surface")
    trust_level = (
        IsolationTrustLevel.TRUSTED_SDK
        if normalized is IsolationSettingsSurface.SDK
        else (
            IsolationTrustLevel.TRUSTED_CLI
            if normalized is IsolationSettingsSurface.CLI
            else IsolationTrustLevel.TRUSTED_DEPLOYMENT
        )
    )
    return IsolationSettingsSource(
        surface=normalized,
        trust_level=trust_level,
    )


def trusted_isolation_settings_from_mapping(
    raw: Mapping[str, object],
    *,
    source: IsolationSettingsSource,
) -> IsolationSettings:
    assert isinstance(source, IsolationSettingsSource)
    return IsolationSettings.from_dict(raw, source=source)


def isolation_selection_from_mapping(
    raw: Mapping[str, object],
    *,
    source: IsolationSettingsSource,
) -> IsolationProfileSelection:
    assert isinstance(source, IsolationSettingsSource)
    return IsolationProfileSelection.from_dict(raw, source=source)


def trusted_isolation_runtime_from_mapping(
    raw: Mapping[str, object],
    *,
    source: IsolationSettingsSource,
    selection: IsolationProfileSelection | None = None,
) -> IsolationToolRuntimeSettings:
    settings = trusted_isolation_settings_from_mapping(raw, source=source)
    effective = settings.select_profile(
        selection
        or IsolationProfileSelection(
            mode=settings.mode,
        )
    )
    return IsolationToolRuntimeSettings(effective_settings=effective)


def serialize_isolation_effective_settings(
    settings: IsolationEffectiveSettings,
) -> str:
    assert isinstance(settings, IsolationEffectiveSettings)
    return settings.to_json()


def deserialize_isolation_effective_settings(
    serialized: str,
) -> IsolationEffectiveSettings:
    _assert_non_empty_string(serialized, "serialized")
    return IsolationEffectiveSettings.from_json(serialized)


def isolation_diagnostic(
    code: IsolationDiagnosticCode | str,
    *,
    path: str,
    message: str,
    hint: str,
    category: IsolationDiagnosticCategory | str,
) -> IsolationDiagnostic:
    return IsolationDiagnostic(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=category,
    )


_CONTAINER_SURFACES = {
    IsolationSettingsSurface.SDK: ContainerSurface.SDK,
    IsolationSettingsSurface.CLI: ContainerSurface.CLI,
    IsolationSettingsSurface.AGENT_TOML: ContainerSurface.AGENT_TOML,
    IsolationSettingsSurface.FLOW_TOML: ContainerSurface.FLOW_TOML,
    IsolationSettingsSurface.TASK_TOML: ContainerSurface.TASK_TOML,
    IsolationSettingsSurface.SERVER: ContainerSurface.SERVER,
    IsolationSettingsSurface.MCP: ContainerSurface.MCP,
    IsolationSettingsSurface.A2A: ContainerSurface.A2A,
    IsolationSettingsSurface.RUNTIME_ENVELOPE: (
        ContainerSurface.RUNTIME_ENVELOPE
    ),
    IsolationSettingsSurface.MODEL_BACKEND: ContainerSurface.MODEL_BACKEND,
    IsolationSettingsSurface.WORKER: ContainerSurface.RUNTIME_ENVELOPE,
}
_CONTAINER_TRUST_LEVELS = {
    IsolationTrustLevel.TRUSTED_OPERATOR: ContainerTrustLevel.TRUSTED_OPERATOR,
    IsolationTrustLevel.TRUSTED_DEPLOYMENT: (
        ContainerTrustLevel.TRUSTED_DEPLOYMENT
    ),
    IsolationTrustLevel.TRUSTED_SDK: ContainerTrustLevel.TRUSTED_DEPLOYMENT,
    IsolationTrustLevel.TRUSTED_CLI: ContainerTrustLevel.TRUSTED_DEPLOYMENT,
    IsolationTrustLevel.UNTRUSTED_AGENT: ContainerTrustLevel.UNTRUSTED_AGENT,
    IsolationTrustLevel.UNTRUSTED_FLOW: ContainerTrustLevel.UNTRUSTED_FLOW,
    IsolationTrustLevel.UNTRUSTED_TASK: ContainerTrustLevel.UNTRUSTED_TASK,
    IsolationTrustLevel.UNTRUSTED_REQUEST: (
        ContainerTrustLevel.UNTRUSTED_REQUEST
    ),
    IsolationTrustLevel.MODEL: ContainerTrustLevel.MODEL,
}
_LOCAL_FIELDS = {
    "approval_required",
    "allowed_roots",
    "executable_allowlist",
    "timeout_seconds",
    "max_stdout_bytes",
    "max_stderr_bytes",
    "isolated",
}
_SANDBOX_ENVIRONMENT_FIELDS = {"variables", "allowlist", "inherit_host"}
_SANDBOX_NETWORK_FIELDS = {"mode", "egress_allowlist"}
_SANDBOX_RESOURCE_FIELDS = {"timeout_seconds", "pids"}
_SANDBOX_OUTPUT_FIELDS = {
    "max_stdout_bytes",
    "max_stderr_bytes",
    "max_artifact_bytes",
    "allow_artifacts",
}
_SANDBOX_PROFILE_FIELDS = {
    "name",
    "trusted_executables",
    "executable_search_roots",
    "read_roots",
    "write_roots",
    "deny_roots",
    "scratch_roots",
    "output_roots",
    "environment",
    "network",
    "resources",
    "output",
    "child_processes",
    "inherited_fds",
    "cleanup",
}
_SANDBOX_SELECTION_FIELDS = {"profile", "required"}
_SANDBOX_SETTINGS_FIELDS = {
    "source",
    "backend",
    "default_profile",
    "allowed_profiles",
    "profiles",
    "profile_registry_id",
    "policy_version",
}
_SANDBOX_EFFECTIVE_FIELDS = {
    "backend",
    "required",
    "source",
    "policy_version",
    "profile_registry_id",
    "profile_name",
    "profile",
    "allowed_profiles",
}
_ISOLATION_SELECTION_FIELDS = {"mode", "profile", "required", "scope"}
_ISOLATION_SETTINGS_FIELDS = {
    "source",
    "mode",
    "local",
    "sandbox",
    "container",
}
_ISOLATION_EFFECTIVE_FIELDS = {
    "mode",
    "source",
    "local",
    "sandbox",
    "container",
}


def _enum_value(
    value: EnumValue | str,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    assert isinstance(value, str), f"{field_name} must be a string"
    try:
        return enum_type(value)
    except ValueError as exc:
        raise AssertionError(
            f"{field_name} contains unsupported value"
        ) from exc


def _assert_tagged_union(
    mode: IsolationMode,
    local: object,
    sandbox: object,
    container: object,
    field_name: str,
) -> None:
    active = (
        int(local is not None)
        + int(sandbox is not None)
        + int(container is not None)
    )
    assert active == 1, f"{field_name} must carry exactly one policy"
    assert not (
        mode is IsolationMode.LOCAL and (sandbox is not None or container)
    ), "local mode cannot carry sandbox or container policy"
    assert not (
        mode is IsolationMode.SANDBOX and (local is not None or container)
    ), "sandbox mode can carry only sandbox policy"
    assert not (
        mode is IsolationMode.CONTAINER and (local is not None or sandbox)
    ), "container mode can carry only container policy"
    assert not (
        mode is IsolationMode.LOCAL and local is None
    ), "local mode requires local policy"
    assert not (
        mode is IsolationMode.SANDBOX and sandbox is None
    ), "sandbox mode requires sandbox policy"
    assert not (
        mode is IsolationMode.CONTAINER and container is None
    ), "container mode requires container policy"


def _assert_fields(
    raw: Mapping[str, object],
    allowed: set[str],
    field_name: str,
) -> None:
    _assert_mapping(raw, field_name)
    unknown = sorted(str(key) for key in raw if key not in allowed)
    assert not unknown, f"Unknown isolation fields in {field_name}: {unknown}"


def _assert_mapping(value: object, field_name: str) -> None:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"


def _assert_profile_name(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert _PROFILE_NAME_PATTERN.match(
        value
    ), f"{field_name} must be a profile name"


def _assert_path(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert "\x00" not in value, f"{field_name} must not contain NUL"


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    _assert_mapping(value, field_name)
    return cast(Mapping[str, object], value)


def _required(
    raw: Mapping[str, object],
    key: str,
    field_name: str,
) -> object:
    assert key in raw, f"{field_name}.{key} is required"
    return raw[key]


def _required_str(
    raw: Mapping[str, object],
    key: str,
    field_name: str,
) -> str:
    value = _required(raw, key, field_name)
    _assert_non_empty_string(value, f"{field_name}.{key}")
    return cast(str, value)


def _required_bool(
    raw: Mapping[str, object],
    key: str,
    field_name: str,
) -> bool:
    value = _required(raw, key, field_name)
    _assert_bool(value, f"{field_name}.{key}")
    return cast(bool, value)


def _optional_str(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    _assert_non_empty_string(value, key)
    return cast(str, value)


def _optional_str_or_default(
    raw: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = raw.get(key, default)
    _assert_non_empty_string(value, key)
    return cast(str, value)


def _optional_bool_or_default(
    raw: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = raw.get(key, default)
    _assert_bool(value, key)
    return cast(bool, value)


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    assert isinstance(value, int), f"{key} must be an integer"
    return value


def _optional_int_or_default(
    raw: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = raw.get(key, default)
    assert isinstance(value, int), f"{key} must be an integer"
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if value is None:
        return ()
    assert not isinstance(value, str), f"{field_name} must be a sequence"
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    items = _sequence(value, field_name)
    normalized: list[str] = []
    for item in items:
        _assert_non_empty_string(item, field_name)
        normalized.append(cast(str, item))
    return tuple(normalized)


def _path_tuple(value: object, field_name: str) -> tuple[str, ...]:
    items = _sequence(value, field_name)
    normalized: list[str] = []
    for item in items:
        _assert_path(item, field_name)
        normalized.append(cast(str, item))
    return tuple(normalized)


def _absolute_path_tuple(value: object, field_name: str) -> tuple[str, ...]:
    paths = _path_tuple(value, field_name)
    for path in paths:
        assert path.startswith(
            "/"
        ), f"{field_name} must contain absolute paths"
    return paths


def _string_mapping(
    value: object,
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name}.{key}")
        normalized[cast(str, key)] = cast(str, item)
    return normalized


def _sandbox_profile_mapping(
    value: Mapping[str, SandboxProfile],
) -> dict[str, SandboxProfile]:
    assert isinstance(value, Mapping), "profiles must be a mapping"
    normalized: dict[str, SandboxProfile] = {}
    for name, profile in value.items():
        _assert_profile_name(name, "profile name")
        assert isinstance(profile, SandboxProfile)
        assert profile.name == name, "profile table name must match name"
        normalized[name] = profile
    return normalized


def _sandbox_profile_mapping_from_dict(
    value: Mapping[str, object],
) -> dict[str, SandboxProfile]:
    profiles: dict[str, SandboxProfile] = {}
    for name, profile_raw in value.items():
        _assert_profile_name(name, "profile name")
        raw = dict(_mapping(profile_raw, "profile"))
        raw.setdefault("name", name)
        assert raw["name"] == name, "profile table name must match name"
        profiles[name] = SandboxProfile.from_dict(raw)
    return profiles


def _local_policy_from_raw(
    raw: Mapping[str, object],
    mode: IsolationMode,
) -> LocalIsolationPolicy | None:
    value = raw.get("local")
    if value is not None:
        return LocalIsolationPolicy.from_dict(_mapping(value, "local"))
    if mode is IsolationMode.LOCAL:
        return LocalIsolationPolicy()
    return None


def _sandbox_settings_from_raw(
    raw: Mapping[str, object],
    source: IsolationSettingsSource,
) -> SandboxSettings | None:
    value = raw.get("sandbox")
    if value is None:
        return None
    return SandboxSettings.from_dict(_mapping(value, "sandbox"), source=source)


def _container_settings_from_raw(
    raw: Mapping[str, object],
    source: IsolationSettingsSource,
) -> ContainerSettings | None:
    value = raw.get("container")
    if value is None:
        return None
    return ContainerSettings.from_dict(
        _mapping(value, "container"),
        source=source.to_container_source(),
    )


def _local_effective_from_raw(
    raw: Mapping[str, object],
    mode: IsolationMode,
) -> LocalIsolationPolicy | None:
    if "local" in raw and raw["local"] is not None:
        return LocalIsolationPolicy.from_dict(_mapping(raw["local"], "local"))
    if mode is IsolationMode.LOCAL:
        return None
    return None


def _sandbox_effective_from_raw(
    raw: Mapping[str, object],
) -> SandboxEffectiveSettings | None:
    value = raw.get("sandbox")
    if value is None:
        return None
    return SandboxEffectiveSettings.from_dict(_mapping(value, "sandbox"))


def _container_effective_from_raw(
    raw: Mapping[str, object],
) -> ContainerEffectiveSettings | None:
    value = raw.get("container")
    if value is None:
        return None
    return ContainerEffectiveSettings.from_dict(_mapping(value, "container"))


def _canonical_sandbox_profile(profile: SandboxProfile) -> dict[str, object]:
    serialized = profile.to_dict()
    for key in (
        "trusted_executables",
        "executable_search_roots",
        "read_roots",
        "write_roots",
        "deny_roots",
        "scratch_roots",
        "output_roots",
    ):
        serialized[key] = sorted(cast(list[str], serialized[key]))
    environment = cast(dict[str, object], serialized["environment"])
    environment["allowlist"] = sorted(
        cast(list[str], environment["allowlist"])
    )
    network = cast(dict[str, object], serialized["network"])
    network["egress_allowlist"] = sorted(
        cast(list[str], network["egress_allowlist"])
    )
    return serialized


def _stable_json(value: Mapping[str, object]) -> str:
    return json_dumps(value, sort_keys=True, separators=(",", ":"))
