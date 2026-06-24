from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .conformance import (
    ContainerBackend,
    ContainerExecutionScope,
)
from .settings import (
    ContainerAuditMode,
    ContainerAuditPolicy,
    ContainerCleanupMode,
    ContainerCleanupPolicy,
    ContainerCommandMode,
    ContainerDevicePolicy,
    ContainerEscalationMode,
    ContainerEscalationPolicy,
    ContainerImagePolicy,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputPolicy,
    ContainerPoolingMode,
    ContainerPoolingPolicy,
    ContainerPoolTeardownMode,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerPullPolicy,
    ContainerResourceLimits,
    ContainerSettings,
    ContainerSettingsSource,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from re import compile as compile_pattern
from re import sub as substitute_pattern
from types import MappingProxyType
from typing import TypeAlias, TypeVar, cast, final

ContainerProfilePayloadValue: TypeAlias = bool | int | str
EnumValue = TypeVar("EnumValue", bound=StrEnum)

_PROFILE_NAME_PATTERN = compile_pattern(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PAYLOAD_NAME_PATTERN = compile_pattern(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_RUNTIME_AUTHORITY_KEYS = frozenset(
    {
        "backend",
        "backendflag",
        "backendflags",
        "backendoption",
        "backendoptions",
        "buildpolicy",
        "capabilities",
        "capability",
        "commandmode",
        "container",
        "containerflags",
        "containerpolicy",
        "containerprofile",
        "containerprofiles",
        "containersettings",
        "containerruntime",
        "containerruntimepolicy",
        "containerworkspace",
        "device",
        "devicerequest",
        "devicerequests",
        "devices",
        "digest",
        "egress",
        "egressallowlist",
        "env",
        "environment",
        "environmentvariable",
        "environmentvariables",
        "envvar",
        "envvars",
        "gid",
        "image",
        "imageref",
        "imagereference",
        "images",
        "memorybytes",
        "mount",
        "mountpath",
        "mountpaths",
        "mounts",
        "network",
        "networkmode",
        "networkpolicy",
        "networks",
        "pids",
        "platform",
        "policyversion",
        "privileged",
        "pullpolicy",
        "readonlyrootfs",
        "resource",
        "resourcelimit",
        "resourcelimits",
        "resources",
        "runtime",
        "runtimecontainer",
        "runtimeenvelope",
        "runtimeflags",
        "runtimelimits",
        "runtimepolicy",
        "runtimeprofile",
        "runtimeprofiles",
        "secret",
        "secretdeliveries",
        "secretdelivery",
        "secrets",
        "timeoutseconds",
        "uid",
        "user",
        "workdir",
        "workingdirectory",
        "workspace",
        "workspaceroot",
    }
)
_RUNTIME_AUTHORITY_PREFIXES = ("container", "runtime")


class ContainerProfilePayloadType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProfilePayloadField:
    name: str
    payload_type: ContainerProfilePayloadType | str
    required: bool = True
    max_length: int | None = None
    min_value: int | None = None
    max_value: int | None = None

    def __post_init__(self) -> None:
        _assert_payload_name(self.name, "name")
        assert not _payload_runtime_authority_key(
            self.name
        ), "payload field cannot shadow runtime authority"
        payload_type = _enum_value(
            self.payload_type,
            ContainerProfilePayloadType,
            "payload_type",
        )
        _assert_bool(self.required, "required")
        if self.max_length is not None:
            _assert_positive_int(self.max_length, "max_length")
        _assert_optional_int(self.min_value, "min_value")
        _assert_optional_int(self.max_value, "max_value")
        if self.min_value is not None and self.max_value is not None:
            assert (
                self.min_value <= self.max_value
            ), "min_value cannot exceed max_value"
        if payload_type is ContainerProfilePayloadType.STRING:
            assert (
                self.min_value is None
            ), "string payload cannot set min_value"
            assert (
                self.max_value is None
            ), "string payload cannot set max_value"
        elif payload_type is ContainerProfilePayloadType.INTEGER:
            assert (
                self.max_length is None
            ), "integer payload cannot set max_length"
        else:
            assert (
                self.max_length is None
            ), "boolean payload cannot set max_length"
            assert (
                self.min_value is None
            ), "boolean payload cannot set min_value"
            assert (
                self.max_value is None
            ), "boolean payload cannot set max_value"
        object.__setattr__(self, "payload_type", payload_type)

    def validate(self, value: object) -> ContainerProfilePayloadValue:
        payload_type = cast(ContainerProfilePayloadType, self.payload_type)
        if payload_type is ContainerProfilePayloadType.STRING:
            _assert_non_empty_string(value, self.name)
            assert isinstance(value, str)
            if self.max_length is not None:
                assert (
                    len(value) <= self.max_length
                ), f"{self.name} exceeds max_length"
            return value
        if payload_type is ContainerProfilePayloadType.INTEGER:
            _assert_int(value, self.name)
            assert isinstance(value, int)
            if self.min_value is not None:
                assert (
                    value >= self.min_value
                ), f"{self.name} is below min_value"
            if self.max_value is not None:
                assert (
                    value <= self.max_value
                ), f"{self.name} exceeds max_value"
            return value
        _assert_bool(value, self.name)
        assert isinstance(value, bool)
        return value

    def to_dict(self) -> dict[str, int | str | bool | None]:
        payload_type = cast(ContainerProfilePayloadType, self.payload_type)
        return {
            "name": self.name,
            "payload_type": payload_type.value,
            "required": self.required,
            "max_length": self.max_length,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerProfilePayloadField":
        _assert_fields(
            raw,
            {
                "name",
                "payload_type",
                "required",
                "max_length",
                "min_value",
                "max_value",
            },
            "payload_field",
        )
        return cls(
            name=_required_str(raw, "name", "payload_field"),
            payload_type=_required_str(raw, "payload_type", "payload_field"),
            required=_optional_bool_or_default(raw, "required", True),
            max_length=_optional_int(raw, "max_length"),
            min_value=_optional_int(raw, "min_value"),
            max_value=_optional_int(raw, "max_value"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProfilePayloadContract:
    fields: Mapping[str, ContainerProfilePayloadField] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        fields: dict[str, ContainerProfilePayloadField] = {}
        for name, payload_field in self.fields.items():
            _assert_payload_name(name, "fields")
            assert not _payload_runtime_authority_key(
                name
            ), "payload field cannot shadow runtime authority"
            assert isinstance(payload_field, ContainerProfilePayloadField)
            assert (
                payload_field.name == name
            ), "payload field name must match mapping key"
            assert name not in fields, "payload fields must be unique"
            fields[name] = payload_field
        object.__setattr__(self, "fields", MappingProxyType(fields))

    def validate(
        self,
        payload: Mapping[str, object],
    ) -> Mapping[str, ContainerProfilePayloadValue]:
        _assert_mapping(payload, "payload")
        _assert_fields(payload, set(self.fields), "payload")
        result: dict[str, ContainerProfilePayloadValue] = {}
        for name, payload_field in self.fields.items():
            if name not in payload:
                assert (
                    not payload_field.required
                ), "required payload is missing"
                continue
            result[name] = payload_field.validate(payload[name])
        return MappingProxyType(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "fields": {
                name: payload_field.to_dict()
                for name, payload_field in sorted(self.fields.items())
            },
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerProfilePayloadContract":
        _assert_fields(raw, {"fields"}, "payload_contract")
        fields: dict[str, ContainerProfilePayloadField] = {}
        for name, value in _mapping(
            raw.get("fields", {}),
            "payload_contract.fields",
        ).items():
            _assert_payload_name(name, "payload field")
            field_raw = dict(_mapping(value, "payload_field"))
            field_raw.setdefault("name", name)
            assert field_raw["name"] == name, "payload field name mismatch"
            fields[name] = ContainerProfilePayloadField.from_dict(field_raw)
        return cls(fields=fields)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProfileBackedToolInvocation:
    profile: str
    payload: Mapping[str, ContainerProfilePayloadValue] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        _assert_profile_name(self.profile, "profile")
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(_payload_value_mapping(self.payload, "payload")),
        )

    def to_selection(
        self,
        *,
        required: bool = True,
        scope: ContainerExecutionScope | str = (
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
        ),
    ) -> ContainerProfileSelection:
        return ContainerProfileSelection(
            profile=self.profile,
            required=required,
            scope=scope,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "payload": dict(self.payload),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerProfileBackedToolContract:
    name: str
    allowed_profiles: Sequence[str]
    payload_contract: ContainerProfilePayloadContract = field(
        default_factory=ContainerProfilePayloadContract,
    )

    def __post_init__(self) -> None:
        _assert_profile_name(self.name, "name")
        allowed_profiles = _string_tuple(
            self.allowed_profiles,
            "allowed_profiles",
        )
        assert allowed_profiles, "allowed_profiles must not be empty"
        assert len(set(allowed_profiles)) == len(
            allowed_profiles
        ), "allowed_profiles must be unique"
        for profile in allowed_profiles:
            _assert_profile_name(profile, "allowed_profiles")
        assert isinstance(
            self.payload_contract,
            ContainerProfilePayloadContract,
        )
        object.__setattr__(self, "allowed_profiles", allowed_profiles)

    def invocation_from_mapping(
        self,
        raw: Mapping[str, object],
        *,
        source: ContainerSettingsSource | None = None,
    ) -> ContainerProfileBackedToolInvocation:
        if source is not None:
            assert isinstance(source, ContainerSettingsSource)
        _assert_mapping(raw, "profile_invocation")
        _assert_fields(raw, {"profile", "payload"}, "profile_invocation")
        profile = _required_str(raw, "profile", "profile_invocation")
        assert profile in self.allowed_profiles, "profile is not allowed"
        payload = self.payload_contract.validate(
            _mapping(raw.get("payload", {}), "payload")
        )
        return ContainerProfileBackedToolInvocation(
            profile=profile,
            payload=payload,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "allowed_profiles": list(self.allowed_profiles),
            "payload_contract": self.payload_contract.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerProfileBackedToolContract":
        _assert_fields(
            raw,
            {"name", "allowed_profiles", "payload_contract"},
            "profile_contract",
        )
        return cls(
            name=_required_str(raw, "name", "profile_contract"),
            allowed_profiles=_string_tuple(
                raw.get("allowed_profiles", ()),
                "allowed_profiles",
            ),
            payload_contract=ContainerProfilePayloadContract.from_dict(
                _mapping(raw.get("payload_contract", {}), "payload_contract")
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerServiceProfileSpec:
    name: str
    image_reference: str
    health_check_command: Sequence[str]
    audit_labels: Mapping[str, str]
    max_age_seconds: int
    max_uses: int
    idle_ttl_seconds: int
    cleanup_grace_seconds: int = 5
    teardown: ContainerPoolTeardownMode | str = (
        ContainerPoolTeardownMode.REMOVE
    )
    network: ContainerNetworkMode | str = ContainerNetworkMode.LOOPBACK
    resources: ContainerResourceLimits = field(
        default_factory=ContainerResourceLimits,
    )
    output: ContainerOutputPolicy = field(
        default_factory=ContainerOutputPolicy,
    )
    pull_policy: ContainerPullPolicy | str = ContainerPullPolicy.NEVER
    platform: str = "linux/amd64"

    def __post_init__(self) -> None:
        _assert_profile_name(self.name, "name")
        _assert_non_empty_string(self.image_reference, "image_reference")
        health_check_command = _string_tuple(
            self.health_check_command,
            "health_check_command",
        )
        assert health_check_command, "service profiles require health checks"
        audit_labels = _string_mapping(self.audit_labels, "audit_labels")
        assert audit_labels, "service profiles require audit labels"
        _assert_positive_int(self.max_age_seconds, "max_age_seconds")
        _assert_positive_int(self.max_uses, "max_uses")
        _assert_positive_int(self.idle_ttl_seconds, "idle_ttl_seconds")
        _assert_positive_int(
            self.cleanup_grace_seconds,
            "cleanup_grace_seconds",
        )
        teardown = _enum_value(
            self.teardown,
            ContainerPoolTeardownMode,
            "teardown",
        )
        assert (
            teardown is not ContainerPoolTeardownMode.RESET
        ), "short-lived service teardown cannot reset containers"
        network = _enum_value(self.network, ContainerNetworkMode, "network")
        assert network in {
            ContainerNetworkMode.NONE,
            ContainerNetworkMode.LOOPBACK,
        }, "service profiles cannot widen network access"
        assert isinstance(self.resources, ContainerResourceLimits)
        assert (
            self.resources.cpu_count is not None
        ), "service profiles require cpu_count"
        assert (
            self.resources.memory_bytes is not None
        ), "service profiles require memory_bytes"
        assert self.resources.pids is not None, "service profiles require pids"
        assert (
            self.resources.timeout_seconds is not None
        ), "service profiles require timeout_seconds"
        assert isinstance(self.output, ContainerOutputPolicy)
        pull_policy = _enum_value(
            self.pull_policy,
            ContainerPullPolicy,
            "pull_policy",
        )
        _assert_non_empty_string(self.platform, "platform")
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
        object.__setattr__(self, "teardown", teardown)
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "pull_policy", pull_policy)

    def to_profile(self) -> ContainerProfile:
        teardown = cast(ContainerPoolTeardownMode, self.teardown)
        network = cast(ContainerNetworkMode, self.network)
        return ContainerProfile(
            name=self.name,
            image=ContainerImagePolicy(
                reference=self.image_reference,
                pull_policy=self.pull_policy,
                platform=self.platform,
            ),
            network=ContainerNetworkPolicy(mode=network),
            devices=ContainerDevicePolicy(),
            resources=self.resources,
            output=self.output,
            cleanup=ContainerCleanupPolicy(
                mode=ContainerCleanupMode.REMOVE,
                grace_seconds=self.cleanup_grace_seconds,
            ),
            pooling=ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SERVICE,
                max_age_seconds=self.max_age_seconds,
                max_uses=self.max_uses,
                idle_ttl_seconds=self.idle_ttl_seconds,
                health_check_command=self.health_check_command,
                teardown=teardown,
                audit_labels=self.audit_labels,
                allow_secret_reuse=False,
            ),
            audit=ContainerAuditPolicy(mode=ContainerAuditMode.FULL),
            escalation=ContainerEscalationPolicy(
                mode=ContainerEscalationMode.DENY,
            ),
            command_mode=ContainerCommandMode.SERVICE_COMMAND,
        )

    def to_dict(self) -> dict[str, object]:
        teardown = cast(ContainerPoolTeardownMode, self.teardown)
        network = cast(ContainerNetworkMode, self.network)
        pull_policy = cast(ContainerPullPolicy, self.pull_policy)
        return {
            "name": self.name,
            "image_reference": self.image_reference,
            "health_check_command": list(self.health_check_command),
            "audit_labels": dict(self.audit_labels),
            "max_age_seconds": self.max_age_seconds,
            "max_uses": self.max_uses,
            "idle_ttl_seconds": self.idle_ttl_seconds,
            "cleanup_grace_seconds": self.cleanup_grace_seconds,
            "teardown": teardown.value,
            "network": network.value,
            "resources": self.resources.to_dict(),
            "output": self.output.to_dict(),
            "pull_policy": pull_policy.value,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
        *,
        source: ContainerSettingsSource,
    ) -> "ContainerServiceProfileSpec":
        assert isinstance(source, ContainerSettingsSource)
        assert (
            source.can_define_runtime_authority
        ), "service profile specs require trusted runtime authority"
        _assert_fields(raw, _SERVICE_PROFILE_SPEC_FIELDS, "service_profile")
        return cls(
            name=_required_str(raw, "name", "service_profile"),
            image_reference=_required_str(
                raw,
                "image_reference",
                "service_profile",
            ),
            health_check_command=_string_tuple(
                raw.get("health_check_command", ()),
                "health_check_command",
            ),
            audit_labels=_string_mapping(
                raw.get("audit_labels", {}),
                "audit_labels",
            ),
            max_age_seconds=_required_int(
                raw,
                "max_age_seconds",
                "service_profile",
            ),
            max_uses=_required_int(raw, "max_uses", "service_profile"),
            idle_ttl_seconds=_required_int(
                raw,
                "idle_ttl_seconds",
                "service_profile",
            ),
            cleanup_grace_seconds=_optional_int_or_default(
                raw,
                "cleanup_grace_seconds",
                5,
            ),
            teardown=_optional_str_or_default(
                raw,
                "teardown",
                ContainerPoolTeardownMode.REMOVE.value,
            ),
            network=_optional_str_or_default(
                raw,
                "network",
                ContainerNetworkMode.LOOPBACK.value,
            ),
            resources=ContainerResourceLimits.from_dict(
                _mapping(raw.get("resources", {}), "resources")
            ),
            output=ContainerOutputPolicy.from_dict(
                _mapping(raw.get("output", {}), "output")
            ),
            pull_policy=_optional_str_or_default(
                raw,
                "pull_policy",
                ContainerPullPolicy.NEVER.value,
            ),
            platform=_optional_str_or_default(raw, "platform", "linux/amd64"),
        )


def container_profile_settings(
    *,
    source: ContainerSettingsSource,
    backend: ContainerBackend | str,
    profiles: Sequence[ContainerProfile],
    default_profile: str | None = None,
    allowed_profiles: Sequence[str] | None = None,
    profile_registry_id: str = "profile-backed",
    policy_version: str = "phase19",
) -> ContainerSettings:
    assert isinstance(source, ContainerSettingsSource)
    profile_map = {profile.name: profile for profile in profiles}
    assert len(profile_map) == len(profiles), "profile names must be unique"
    selected_profiles = (
        tuple(profile_map) if allowed_profiles is None else allowed_profiles
    )
    return ContainerSettings(
        source=source,
        backend=backend,
        default_profile=default_profile,
        allowed_profiles=selected_profiles,
        profiles=profile_map,
        profile_registry_id=profile_registry_id,
        policy_version=policy_version,
    )


def container_service_profile(
    *,
    name: str,
    image_reference: str,
    health_check_command: Sequence[str],
    audit_labels: Mapping[str, str],
    max_age_seconds: int,
    max_uses: int,
    idle_ttl_seconds: int,
    resources: ContainerResourceLimits,
) -> ContainerProfile:
    return ContainerServiceProfileSpec(
        name=name,
        image_reference=image_reference,
        health_check_command=health_check_command,
        audit_labels=audit_labels,
        max_age_seconds=max_age_seconds,
        max_uses=max_uses,
        idle_ttl_seconds=idle_ttl_seconds,
        resources=resources,
    ).to_profile()


_SERVICE_PROFILE_SPEC_FIELDS = {
    "name",
    "image_reference",
    "health_check_command",
    "audit_labels",
    "max_age_seconds",
    "max_uses",
    "idle_ttl_seconds",
    "cleanup_grace_seconds",
    "teardown",
    "network",
    "resources",
    "output",
    "pull_policy",
    "platform",
}


def _enum_value(
    value: EnumValue | str,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    try:
        return enum_type(value)
    except ValueError as error:
        raise AssertionError(f"{field_name} has unsupported value") from error


def _assert_fields(
    raw: Mapping[str, object],
    fields: set[str],
    context: str,
) -> None:
    _assert_mapping(raw, context)
    for key in raw:
        assert isinstance(key, str), f"{context} field names must be strings"
    unexpected = set(raw) - fields
    assert (
        not unexpected
    ), f"{context} has unknown fields: {sorted(unexpected)}"


def _assert_mapping(value: object, field_name: str) -> None:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"


def _assert_profile_name(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert (
        _PROFILE_NAME_PATTERN.match(value) is not None
    ), f"{field_name} must be a safe profile name"


def _assert_payload_name(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert (
        _PAYLOAD_NAME_PATTERN.match(value) is not None
    ), f"{field_name} must be a safe payload name"


def _payload_runtime_authority_key(key: object) -> bool:
    normalized = substitute_pattern(r"[^a-z0-9]", "", str(key).lower())
    if not normalized:
        return False
    if normalized in _RUNTIME_AUTHORITY_KEYS:
        return True
    if normalized.startswith(_RUNTIME_AUTHORITY_PREFIXES):
        return True
    if normalized.endswith("backend"):
        return True
    return "secret" in normalized


def _assert_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"


def _assert_optional_int(value: object | None, field_name: str) -> None:
    if value is None:
        return
    _assert_int(value, field_name)


def _required(
    raw: Mapping[str, object],
    key: str,
    context: str,
) -> object:
    assert key in raw, f"{context}.{key} is required"
    return raw[key]


def _required_str(
    raw: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    value = _required(raw, key, context)
    _assert_non_empty_string(value, f"{context}.{key}")
    assert isinstance(value, str)
    return value


def _required_int(
    raw: Mapping[str, object],
    key: str,
    context: str,
) -> int:
    value = _required(raw, key, context)
    _assert_int(value, f"{context}.{key}")
    assert isinstance(value, int)
    return value


def _optional_str_or_default(
    raw: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = raw.get(key, default)
    _assert_non_empty_string(value, key)
    assert isinstance(value, str)
    return value


def _optional_bool_or_default(
    raw: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = raw.get(key, default)
    _assert_bool(value, key)
    assert isinstance(value, bool)
    return value


def _optional_int(
    raw: Mapping[str, object],
    key: str,
) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    _assert_int(value, key)
    assert isinstance(value, int)
    return value


def _optional_int_or_default(
    raw: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = raw.get(key, default)
    _assert_int(value, key)
    assert isinstance(value, int)
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    _assert_mapping(value, field_name)
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), f"{field_name} must be a sequence"
    result: list[str] = []
    for item in value:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
        result.append(item)
    return tuple(result)


def _string_mapping(value: object, field_name: str) -> dict[str, str]:
    _assert_mapping(value, field_name)
    assert isinstance(value, Mapping)
    result: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name}.{key}")
        assert isinstance(key, str)
        assert isinstance(item, str)
        result[key] = item
    return result


def _payload_value_mapping(
    value: object,
    field_name: str,
) -> dict[str, ContainerProfilePayloadValue]:
    _assert_mapping(value, field_name)
    assert isinstance(value, Mapping)
    result: dict[str, ContainerProfilePayloadValue] = {}
    for key, item in value.items():
        _assert_payload_name(key, f"{field_name} key")
        assert not _payload_runtime_authority_key(
            key
        ), "payload field cannot shadow runtime authority"
        assert isinstance(
            item,
            bool | int | str,
        ), f"{field_name}.{key} has unsupported payload type"
        result[key] = item
    return result
