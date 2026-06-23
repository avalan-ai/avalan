from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .conformance import (
    ContainerExecutionScope,
)
from .settings import (
    ContainerCommandMode,
    ContainerCommandPlan,
    ContainerDevicePolicy,
    ContainerEffectiveSettings,
    ContainerImagePolicy,
    ContainerMountDeclaration,
    ContainerNetworkPolicy,
    ContainerResourceLimits,
    ContainerRunPlan,
    ContainerRuntimeEnvelopePlan,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from json import dumps
from posixpath import normpath as normalize_posix_path
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)


class ContainerPlanRequestKind(StrEnum):
    TYPED_TOOL = "typed_tool"
    FLOW_NODE = "flow_node"
    TASK_ATTEMPT = "task_attempt"
    AGENT_SESSION = "agent_session"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"
    RUNTIME_ENVELOPE = "runtime_envelope"


class ContainerRuntimeEnvelopeKind(StrEnum):
    WHOLE_AGENT = "whole_agent"
    FLOW_RUNTIME = "flow_runtime"
    TASK_WORKER = "task_worker"
    SERVER = "server"
    MODEL_BACKEND = "model_backend"


class ContainerDurablePlanKind(StrEnum):
    RUN = "run"
    RUNTIME_ENVELOPE = "runtime_envelope"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPlanRequest:
    request_kind: ContainerPlanRequestKind | str
    logical_name: str
    command: str
    argv: Sequence[str]
    cwd: str = "/workspace"
    scope: ContainerExecutionScope | str = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    )
    request_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        request_kind = _enum_value(
            self.request_kind,
            ContainerPlanRequestKind,
            "request_kind",
        )
        scope = _enum_value(self.scope, ContainerExecutionScope, "scope")
        _assert_non_empty_string(self.logical_name, "logical_name")
        _assert_non_empty_string(self.command, "command")
        cwd = _container_path(self.cwd, "cwd")
        argv = _string_tuple(self.argv, "argv")
        assert argv, "argv must not be empty"
        if self.request_id is not None:
            _assert_non_empty_string(self.request_id, "request_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if request_kind is ContainerPlanRequestKind.TASK_ATTEMPT:
            assert (
                self.attempt_id is not None
            ), "task attempt requests require attempt_id"
        object.__setattr__(self, "request_kind", request_kind)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "argv", argv)

    def to_command_plan(self) -> ContainerCommandPlan:
        request_kind = cast(ContainerPlanRequestKind, self.request_kind)
        return ContainerCommandPlan(
            tool_name=f"{request_kind.value}:{self.logical_name}",
            command=self.command,
            argv=self.argv,
            cwd=self.cwd,
            scope=cast(ContainerExecutionScope, self.scope),
        )

    def to_dict(self) -> dict[str, object]:
        request_kind = cast(ContainerPlanRequestKind, self.request_kind)
        scope = cast(ContainerExecutionScope, self.scope)
        return {
            "request_kind": request_kind.value,
            "logical_name": self.logical_name,
            "command": self.command,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "scope": scope.value,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ContainerPlanRequest":
        _assert_fields(
            raw,
            {
                "request_kind",
                "logical_name",
                "command",
                "argv",
                "cwd",
                "scope",
                "request_id",
                "attempt_id",
            },
            "request",
        )
        return cls(
            request_kind=_required_str(raw, "request_kind", "request"),
            logical_name=_required_str(raw, "logical_name", "request"),
            command=_required_str(raw, "command", "request"),
            argv=_string_tuple(_required(raw, "argv", "request"), "argv"),
            cwd=_optional_str_or_default(raw, "cwd", "/workspace"),
            scope=_optional_str_or_default(
                raw,
                "scope",
                ContainerExecutionScope.SHELL_CONTAINER_EXECUTION.value,
            ),
            request_id=_optional_str(raw, "request_id"),
            attempt_id=_optional_str(raw, "attempt_id"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerNormalizedRunPlan:
    request: ContainerPlanRequest
    run_plan: ContainerRunPlan
    command_mode: ContainerCommandMode | str
    profile_registry_id: str

    def __post_init__(self) -> None:
        assert isinstance(self.request, ContainerPlanRequest)
        assert isinstance(self.run_plan, ContainerRunPlan)
        command_mode = _enum_value(
            self.command_mode,
            ContainerCommandMode,
            "command_mode",
        )
        _assert_non_empty_string(
            self.profile_registry_id,
            "profile_registry_id",
        )
        object.__setattr__(self, "command_mode", command_mode)

    @property
    def plan_fingerprint(self) -> str:
        return _fingerprint(self.canonical_policy_input())

    def canonical_policy_input(self) -> dict[str, object]:
        request_kind = cast(
            ContainerPlanRequestKind,
            self.request.request_kind,
        )
        command_mode = cast(ContainerCommandMode, self.command_mode)
        run_plan = self.run_plan
        backend = run_plan.to_dict()["backend"]
        command = _canonical_command_plan(run_plan.command)
        return {
            "backend": backend,
            "command": command,
            "command_mode": command_mode.value,
            "devices": _canonical_device_policy(run_plan.devices),
            "environment_names": sorted(run_plan.environment_names),
            "image": _canonical_image_policy(run_plan.image),
            "mounts": _canonical_mounts(run_plan.mounts),
            "network": _canonical_network_policy(run_plan.network),
            "policy_version": run_plan.policy_version,
            "profile_name": run_plan.profile_name,
            "profile_registry_id": self.profile_registry_id,
            "request_kind": request_kind.value,
            "resources": run_plan.resources.to_dict(),
            "secret_names": sorted(run_plan.secret_names),
        }

    def to_dict(self) -> dict[str, object]:
        command_mode = cast(ContainerCommandMode, self.command_mode)
        return {
            "request": self.request.to_dict(),
            "run_plan": self.run_plan.to_dict(),
            "command_mode": command_mode.value,
            "profile_registry_id": self.profile_registry_id,
            "plan_fingerprint": self.plan_fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerNormalizedRunPlan":
        _assert_fields(
            raw,
            {
                "request",
                "run_plan",
                "command_mode",
                "profile_registry_id",
                "plan_fingerprint",
            },
            "normalized_run_plan",
        )
        plan = cls(
            request=ContainerPlanRequest.from_dict(
                _mapping(_required(raw, "request", "normalized"), "request")
            ),
            run_plan=_run_plan_from_dict(
                _mapping(_required(raw, "run_plan", "normalized"), "run_plan")
            ),
            command_mode=_required_str(
                raw,
                "command_mode",
                "normalized",
            ),
            profile_registry_id=_required_str(
                raw,
                "profile_registry_id",
                "normalized",
            ),
        )
        assert (
            _required_str(raw, "plan_fingerprint", "normalized")
            == plan.plan_fingerprint
        ), "normalized run plan fingerprint mismatch"
        return plan

    def to_metadata(self) -> "ContainerDurablePlanMetadata":
        request_kind = cast(
            ContainerPlanRequestKind,
            self.request.request_kind,
        )
        return ContainerDurablePlanMetadata(
            plan_kind=ContainerDurablePlanKind.RUN,
            request_kind=request_kind,
            request_id=self.request.request_id,
            attempt_id=self.request.attempt_id,
            profile_name=self.run_plan.profile_name,
            profile_registry_id=self.profile_registry_id,
            policy_version=self.run_plan.policy_version,
            plan_fingerprint=self.plan_fingerprint,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerNormalizedRuntimeEnvelopePlan:
    envelope_kind: ContainerRuntimeEnvelopeKind | str
    run_plan: ContainerNormalizedRunPlan
    envelope_plan: ContainerRuntimeEnvelopePlan

    def __post_init__(self) -> None:
        envelope_kind = _enum_value(
            self.envelope_kind,
            ContainerRuntimeEnvelopeKind,
            "envelope_kind",
        )
        assert isinstance(self.run_plan, ContainerNormalizedRunPlan)
        assert isinstance(self.envelope_plan, ContainerRuntimeEnvelopePlan)
        assert (
            self.envelope_plan.profile_name
            == self.run_plan.run_plan.profile_name
        ), "runtime envelope profile must match run plan"
        assert (
            self.envelope_plan.command.to_dict()
            == self.run_plan.run_plan.command.to_dict()
        ), "runtime envelope command must match run plan"
        object.__setattr__(self, "envelope_kind", envelope_kind)

    @property
    def plan_fingerprint(self) -> str:
        return _fingerprint(self.canonical_policy_input())

    def canonical_policy_input(self) -> dict[str, object]:
        envelope_kind = cast(ContainerRuntimeEnvelopeKind, self.envelope_kind)
        scope = cast(ContainerExecutionScope, self.envelope_plan.scope)
        return {
            "envelope_kind": envelope_kind.value,
            "envelope_scope": scope.value,
            "readiness_timeout_seconds": (
                self.envelope_plan.readiness_timeout_seconds
            ),
            "run_plan": self.run_plan.canonical_policy_input(),
        }

    def to_dict(self) -> dict[str, object]:
        envelope_kind = cast(ContainerRuntimeEnvelopeKind, self.envelope_kind)
        return {
            "envelope_kind": envelope_kind.value,
            "run_plan": self.run_plan.to_dict(),
            "envelope_plan": self.envelope_plan.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerNormalizedRuntimeEnvelopePlan":
        _assert_fields(
            raw,
            {
                "envelope_kind",
                "run_plan",
                "envelope_plan",
                "plan_fingerprint",
            },
            "runtime_envelope_plan",
        )
        plan = cls(
            envelope_kind=_required_str(
                raw,
                "envelope_kind",
                "runtime_envelope_plan",
            ),
            run_plan=ContainerNormalizedRunPlan.from_dict(
                _mapping(_required(raw, "run_plan", "envelope"), "run_plan")
            ),
            envelope_plan=_runtime_envelope_plan_from_dict(
                _mapping(
                    _required(raw, "envelope_plan", "envelope"),
                    "envelope_plan",
                )
            ),
        )
        assert (
            _required_str(raw, "plan_fingerprint", "envelope")
            == plan.plan_fingerprint
        ), "runtime envelope fingerprint mismatch"
        return plan

    def to_metadata(self) -> "ContainerDurablePlanMetadata":
        metadata = self.run_plan.to_metadata()
        return ContainerDurablePlanMetadata(
            plan_kind=ContainerDurablePlanKind.RUNTIME_ENVELOPE,
            request_kind=metadata.request_kind,
            request_id=metadata.request_id,
            attempt_id=metadata.attempt_id,
            profile_name=metadata.profile_name,
            profile_registry_id=metadata.profile_registry_id,
            policy_version=metadata.policy_version,
            plan_fingerprint=self.plan_fingerprint,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerDurablePlanMetadata:
    plan_kind: ContainerDurablePlanKind | str
    request_kind: ContainerPlanRequestKind | str
    request_id: str | None
    attempt_id: str | None
    profile_name: str
    profile_registry_id: str
    policy_version: str
    plan_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_kind",
            _enum_value(self.plan_kind, ContainerDurablePlanKind, "plan_kind"),
        )
        object.__setattr__(
            self,
            "request_kind",
            _enum_value(
                self.request_kind,
                ContainerPlanRequestKind,
                "request_kind",
            ),
        )
        if self.request_id is not None:
            _assert_non_empty_string(self.request_id, "request_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_non_empty_string(self.profile_name, "profile_name")
        _assert_non_empty_string(
            self.profile_registry_id,
            "profile_registry_id",
        )
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")

    def assert_matches(
        self,
        plan: (
            ContainerNormalizedRunPlan | ContainerNormalizedRuntimeEnvelopePlan
        ),
    ) -> None:
        if isinstance(plan, ContainerNormalizedRuntimeEnvelopePlan):
            expected_kind = ContainerDurablePlanKind.RUNTIME_ENVELOPE
            run_plan = plan.run_plan
            plan_fingerprint = plan.plan_fingerprint
        else:
            assert isinstance(plan, ContainerNormalizedRunPlan)
            expected_kind = ContainerDurablePlanKind.RUN
            run_plan = plan
            plan_fingerprint = plan.plan_fingerprint
        assert self.plan_kind is expected_kind, "durable plan kind changed"
        assert (
            self.request_kind is run_plan.request.request_kind
        ), "durable request kind changed"
        assert (
            self.request_id == run_plan.request.request_id
        ), "durable request id changed"
        assert (
            self.attempt_id == run_plan.request.attempt_id
        ), "durable attempt id changed"
        assert (
            self.profile_name == run_plan.run_plan.profile_name
        ), "durable profile changed"
        assert (
            self.profile_registry_id == run_plan.profile_registry_id
        ), "durable profile registry changed"
        assert (
            self.policy_version == run_plan.run_plan.policy_version
        ), "durable policy version changed"
        assert (
            self.plan_fingerprint == plan_fingerprint
        ), "durable plan fingerprint changed"

    def to_dict(self) -> dict[str, object]:
        plan_kind = cast(ContainerDurablePlanKind, self.plan_kind)
        request_kind = cast(ContainerPlanRequestKind, self.request_kind)
        return {
            "plan_kind": plan_kind.value,
            "request_kind": request_kind.value,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "profile_name": self.profile_name,
            "profile_registry_id": self.profile_registry_id,
            "policy_version": self.policy_version,
            "plan_fingerprint": self.plan_fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerDurablePlanMetadata":
        _assert_fields(
            raw,
            {
                "plan_kind",
                "request_kind",
                "request_id",
                "attempt_id",
                "profile_name",
                "profile_registry_id",
                "policy_version",
                "plan_fingerprint",
            },
            "durable_metadata",
        )
        return cls(
            plan_kind=_required_str(raw, "plan_kind", "metadata"),
            request_kind=_required_str(raw, "request_kind", "metadata"),
            request_id=_optional_str(raw, "request_id"),
            attempt_id=_optional_str(raw, "attempt_id"),
            profile_name=_required_str(raw, "profile_name", "metadata"),
            profile_registry_id=_required_str(
                raw,
                "profile_registry_id",
                "metadata",
            ),
            policy_version=_required_str(raw, "policy_version", "metadata"),
            plan_fingerprint=_required_str(
                raw,
                "plan_fingerprint",
                "metadata",
            ),
        )


def normalize_container_run_plan(
    settings: ContainerEffectiveSettings,
    request: ContainerPlanRequest,
    *,
    resolved_image_digest: str | None = None,
) -> ContainerNormalizedRunPlan:
    assert isinstance(settings, ContainerEffectiveSettings)
    assert isinstance(request, ContainerPlanRequest)
    assert settings.enabled, "container run planning requires enabled settings"
    assert (
        settings.profile is not None
    ), "container run planning requires profile"
    profile = settings.profile
    command = request.to_command_plan()
    image = _resolve_image_for_plan(profile.image, resolved_image_digest)
    run_plan = ContainerRunPlan(
        backend=settings.backend,
        profile_name=profile.name,
        image=image,
        command=command,
        mounts=profile.mounts,
        environment_names=_environment_names(profile.environment.to_dict()),
        secret_names=tuple(secret.name for secret in profile.secrets),
        network=profile.network,
        devices=profile.devices,
        resources=profile.resources,
        policy_version=settings.policy_version,
    )
    return ContainerNormalizedRunPlan(
        request=request,
        run_plan=run_plan,
        command_mode=profile.command_mode,
        profile_registry_id=settings.profile_registry_id,
    )


def normalize_runtime_envelope_plan(
    settings: ContainerEffectiveSettings,
    request: ContainerPlanRequest,
    *,
    envelope_kind: ContainerRuntimeEnvelopeKind | str,
    resolved_image_digest: str | None = None,
    readiness_timeout_seconds: int = 30,
) -> ContainerNormalizedRuntimeEnvelopePlan:
    normalized_run = normalize_container_run_plan(
        settings,
        request,
        resolved_image_digest=resolved_image_digest,
    )
    envelope_plan = ContainerRuntimeEnvelopePlan(
        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        profile_name=normalized_run.run_plan.profile_name,
        command=normalized_run.run_plan.command,
        readiness_timeout_seconds=readiness_timeout_seconds,
    )
    return ContainerNormalizedRuntimeEnvelopePlan(
        envelope_kind=envelope_kind,
        run_plan=normalized_run,
        envelope_plan=envelope_plan,
    )


def _resolve_image_for_plan(
    image: ContainerImagePolicy,
    resolved_image_digest: str | None,
) -> ContainerImagePolicy:
    assert isinstance(image, ContainerImagePolicy)
    if "@sha256:" not in image.reference:
        _assert_non_empty_string(
            resolved_image_digest,
            "resolved_image_digest",
        )
        assert isinstance(resolved_image_digest, str)
        return ContainerImagePolicy(
            reference=image.reference,
            digest=resolved_image_digest,
            pull_policy=image.pull_policy,
            build_policy=image.build_policy,
            platform=image.platform,
        )
    return image


def _canonical_command_plan(
    command: ContainerCommandPlan,
) -> dict[str, object]:
    serialized = command.to_dict()
    serialized["cwd"] = _container_path(serialized["cwd"], "cwd")
    return serialized


def _canonical_image_policy(image: ContainerImagePolicy) -> dict[str, object]:
    serialized = image.to_dict()
    return {
        "build_policy": serialized["build_policy"],
        "digest": serialized["digest"],
        "platform": serialized["platform"],
        "pull_policy": serialized["pull_policy"],
        "reference": serialized["reference"],
        "reference_is_mutable": "@sha256:" not in image.reference,
    }


def _canonical_mounts(
    mounts: Sequence[ContainerMountDeclaration],
) -> list[dict[str, object]]:
    return sorted(
        (mount.to_dict() for mount in mounts),
        key=lambda mount: (
            mount["target"],
            mount["source"] or "",
            mount["mount_type"],
            mount["access"],
        ),
    )


def _canonical_network_policy(
    network: ContainerNetworkPolicy,
) -> dict[str, object]:
    serialized = network.to_dict()
    serialized["egress_allowlist"] = sorted(
        _string_tuple(serialized["egress_allowlist"], "egress_allowlist")
    )
    return serialized


def _canonical_device_policy(
    devices: ContainerDevicePolicy,
) -> dict[str, object]:
    serialized = devices.to_dict()
    serialized["devices"] = sorted(
        _string_tuple(serialized["devices"], "devices")
    )
    return serialized


def _run_plan_from_dict(raw: Mapping[str, object]) -> ContainerRunPlan:
    _assert_fields(
        raw,
        {
            "backend",
            "profile_name",
            "image",
            "command",
            "mounts",
            "environment_names",
            "secret_names",
            "network",
            "devices",
            "resources",
            "policy_version",
        },
        "run_plan",
    )
    return ContainerRunPlan(
        backend=_required_str(raw, "backend", "run_plan"),
        profile_name=_required_str(raw, "profile_name", "run_plan"),
        image=ContainerImagePolicy.from_dict(
            _mapping(_required(raw, "image", "run_plan"), "image")
        ),
        command=_command_plan_from_dict(
            _mapping(_required(raw, "command", "run_plan"), "command")
        ),
        mounts=tuple(
            ContainerMountDeclaration.from_dict(_mapping(mount, "mount"))
            for mount in _sequence(raw.get("mounts", ()), "mounts")
        ),
        environment_names=_string_tuple(
            raw.get("environment_names", ()),
            "environment_names",
        ),
        secret_names=_string_tuple(
            raw.get("secret_names", ()),
            "secret_names",
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
        policy_version=_required_str(raw, "policy_version", "run_plan"),
    )


def _runtime_envelope_plan_from_dict(
    raw: Mapping[str, object],
) -> ContainerRuntimeEnvelopePlan:
    _assert_fields(
        raw,
        {
            "scope",
            "profile_name",
            "command",
            "readiness_timeout_seconds",
        },
        "runtime_envelope_plan",
    )
    return ContainerRuntimeEnvelopePlan(
        scope=_required_str(raw, "scope", "runtime_envelope_plan"),
        profile_name=_required_str(
            raw,
            "profile_name",
            "runtime_envelope_plan",
        ),
        command=_command_plan_from_dict(
            _mapping(
                _required(raw, "command", "runtime_envelope_plan"),
                "command",
            )
        ),
        readiness_timeout_seconds=_required_int(
            raw,
            "readiness_timeout_seconds",
            "runtime_envelope_plan",
        ),
    )


def _command_plan_from_dict(raw: Mapping[str, object]) -> ContainerCommandPlan:
    _assert_fields(
        raw,
        {"tool_name", "command", "argv", "cwd", "scope"},
        "command_plan",
    )
    return ContainerCommandPlan(
        tool_name=_required_str(raw, "tool_name", "command_plan"),
        command=_required_str(raw, "command", "command_plan"),
        argv=_string_tuple(_required(raw, "argv", "command_plan"), "argv"),
        cwd=_required_str(raw, "cwd", "command_plan"),
        scope=_required_str(raw, "scope", "command_plan"),
    )


def _environment_names(environment: Mapping[str, object]) -> tuple[str, ...]:
    variables = _mapping(environment.get("variables", {}), "variables")
    allowlist = _string_tuple(environment.get("allowlist", ()), "allowlist")
    names = set(allowlist) | {str(name) for name in variables}
    return tuple(sorted(names))


def _fingerprint(value: Mapping[str, object]) -> str:
    payload = dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _assert_fields(
    raw: Mapping[str, object],
    allowed: set[str],
    field_name: str,
) -> None:
    assert isinstance(raw, Mapping), f"{field_name} must be a mapping"
    unknown = set(raw) - allowed
    assert not unknown, f"{field_name} contains unknown fields"


def _required(raw: Mapping[str, object], key: str, field_name: str) -> object:
    assert key in raw, f"{field_name}.{key} is required"
    return raw[key]


def _required_str(
    raw: Mapping[str, object],
    key: str,
    field_name: str,
) -> str:
    value = _required(raw, key, field_name)
    _assert_non_empty_string(value, key)
    assert isinstance(value, str)
    return value


def _required_int(
    raw: Mapping[str, object],
    key: str,
    field_name: str,
) -> int:
    value = _required(raw, key, field_name)
    assert isinstance(value, int), f"{field_name}.{key} must be an integer"
    _assert_positive_int(value, key)
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
    value = raw.get(key)
    if value is None:
        return default
    _assert_non_empty_string(value, key)
    assert isinstance(value, str)
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(
        value, str
    ), f"{field_name} must be a sequence"
    result = tuple(value)
    for item in result:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str)
    return result


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    assert isinstance(value, Sequence) and not isinstance(
        value, str
    ), f"{field_name} must be a sequence"
    return value


def _container_path(value: object, field_name: str) -> str:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert value.startswith("/"), f"{field_name} must be absolute"
    normalized = normalize_posix_path(value)
    assert normalized.startswith("/"), f"{field_name} must remain absolute"
    return normalized


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
