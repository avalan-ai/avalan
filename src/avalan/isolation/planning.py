from ..container import (
    ContainerBackend,
    ContainerEffectiveSettings,
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
    assert_positive_int as _assert_positive_int,
)
from ..types import (
    assert_positive_number as _assert_positive_number,
)
from .settings import (
    IsolationEffectiveSettings,
    IsolationMode,
    LocalIsolationPolicy,
    SandboxBackend,
    SandboxEffectiveSettings,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from json import dumps, loads
from posixpath import normpath as normalize_posix_path
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)


class IsolationPlanRequestKind(StrEnum):
    TYPED_TOOL = "typed_tool"
    FLOW_NODE = "flow_node"
    TASK_ATTEMPT = "task_attempt"
    AGENT_SESSION = "agent_session"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"
    RUNTIME_ENVELOPE = "runtime_envelope"


class IsolationReviewSurface(StrEnum):
    INTERACTIVE_CLI = "interactive_cli"
    STRICT_FLOW = "strict_flow"
    DIRECT_TASK = "direct_task"
    QUEUED_TASK = "queued_task"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"


class IsolationReviewMode(StrEnum):
    INTERACTIVE = "interactive"
    DURABLE = "durable"
    FAIL_CLOSED = "fail_closed"


class IsolationDecisionType(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRES_REVIEW = "requires_review"


class IsolationElevationRung(StrEnum):
    CONTAINER = "container"
    SANDBOX = "sandbox"
    LOCAL = "local"


class IsolationCleanupStatus(StrEnum):
    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationShellRequest:
    request_kind: IsolationPlanRequestKind | str
    logical_name: str
    command: str
    argv: Sequence[str]
    cwd: str
    environment: Mapping[str, str] = field(default_factory=dict)
    request_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        request_kind = _enum_value(
            self.request_kind,
            IsolationPlanRequestKind,
            "request_kind",
        )
        _assert_non_empty_string(self.logical_name, "logical_name")
        command = _absolute_path(self.command, "command")
        cwd = _absolute_path(self.cwd, "cwd")
        argv = _string_tuple(self.argv, "argv")
        assert argv, "argv must not be empty"
        environment = _string_mapping(self.environment, "environment")
        for name in environment:
            _assert_env_name(name, "environment")
        if self.request_id is not None:
            _assert_non_empty_string(self.request_id, "request_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if request_kind is IsolationPlanRequestKind.TASK_ATTEMPT:
            assert (
                self.attempt_id is not None
            ), "task attempt requests require attempt_id"
        object.__setattr__(self, "request_kind", request_kind)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(environment),
        )

    def to_command_plan(self) -> "IsolationCommandPlan":
        return IsolationCommandPlan(
            request_kind=self.request_kind,
            logical_name=self.logical_name,
            executable_path=self.command,
            argv=self.argv,
            cwd=self.cwd,
            environment_names=tuple(sorted(self.environment)),
            request_id=self.request_id,
            attempt_id=self.attempt_id,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationCommandPlan:
    request_kind: IsolationPlanRequestKind | str
    logical_name: str
    executable_path: str
    argv: Sequence[str]
    cwd: str
    environment_names: Sequence[str] = field(default_factory=tuple)
    request_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        request_kind = _enum_value(
            self.request_kind,
            IsolationPlanRequestKind,
            "request_kind",
        )
        _assert_non_empty_string(self.logical_name, "logical_name")
        executable_path = _absolute_path(
            self.executable_path,
            "executable_path",
        )
        cwd = _absolute_path(self.cwd, "cwd")
        argv = _string_tuple(self.argv, "argv")
        environment_names = _environment_name_tuple(
            self.environment_names,
            "environment_names",
        )
        assert argv, "argv must not be empty"
        if self.request_id is not None:
            _assert_non_empty_string(self.request_id, "request_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if request_kind is IsolationPlanRequestKind.TASK_ATTEMPT:
            assert (
                self.attempt_id is not None
            ), "task attempt requests require attempt_id"
        object.__setattr__(self, "request_kind", request_kind)
        object.__setattr__(self, "executable_path", executable_path)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "environment_names", environment_names)

    def canonical_policy_input(self) -> dict[str, object]:
        request_kind = cast(IsolationPlanRequestKind, self.request_kind)
        return {
            "request_kind": request_kind.value,
            "logical_name": self.logical_name,
            "executable_path": self.executable_path,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment_names": list(self.environment_names),
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
        }

    def to_dict(self) -> dict[str, object]:
        return self.canonical_policy_input()

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "IsolationCommandPlan":
        _assert_fields(raw, _COMMAND_FIELDS, "command")
        return cls(
            request_kind=_required_str(raw, "request_kind", "command"),
            logical_name=_required_str(raw, "logical_name", "command"),
            executable_path=_required_str(
                raw,
                "executable_path",
                "command",
            ),
            argv=_string_tuple(_required(raw, "argv", "command"), "argv"),
            cwd=_required_str(raw, "cwd", "command"),
            environment_names=_environment_name_tuple(
                raw.get("environment_names", ()),
                "environment_names",
            ),
            request_id=_optional_str(raw, "request_id"),
            attempt_id=_optional_str(raw, "attempt_id"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class LocalIsolationSubplan:
    command: IsolationCommandPlan
    policy: LocalIsolationPolicy
    policy_version: str = "phase4"
    profile_name: str = "local"
    approval_identity: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.command, IsolationCommandPlan)
        assert isinstance(self.policy, LocalIsolationPolicy)
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_non_empty_string(self.profile_name, "profile_name")
        if self.approval_identity is not None:
            _assert_non_empty_string(
                self.approval_identity,
                "approval_identity",
            )
        assert _path_inside_any(
            self.command.cwd,
            self.policy.allowed_roots,
        ), "local cwd must be inside allowed roots"
        if self.policy.executable_allowlist:
            assert (
                self.command.executable_path
                in self.policy.executable_allowlist
            ), "local executable must be explicitly allowed"

    def with_approval_identity(
        self,
        reviewer_identity: str,
    ) -> "LocalIsolationSubplan":
        return LocalIsolationSubplan(
            command=self.command,
            policy=self.policy,
            policy_version=self.policy_version,
            profile_name=self.profile_name,
            approval_identity=reviewer_identity,
        )

    def canonical_policy_input(self) -> dict[str, object]:
        return {
            "mode": IsolationMode.LOCAL.value,
            "command": self.command.canonical_policy_input(),
            "executable_path": self.command.executable_path,
            "cwd": self.command.cwd,
            "policy": _canonical_local_policy(self.policy),
            "allowed_roots": sorted(self.policy.allowed_roots),
            "executable_allowlist": sorted(self.policy.executable_allowlist),
            "environment_names": list(self.command.environment_names),
            "output_policy": {
                "max_stdout_bytes": self.policy.max_stdout_bytes,
                "max_stderr_bytes": self.policy.max_stderr_bytes,
            },
            "timeout_seconds": self.policy.timeout_seconds,
            "approval_required": self.policy.approval_required,
            "approval_identity": self.approval_identity,
            "isolated": self.policy.isolated,
            "policy_version": self.policy_version,
            "profile_name": self.profile_name,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command.to_dict(),
            "policy": self.policy.to_dict(),
            "policy_version": self.policy_version,
            "profile_name": self.profile_name,
            "approval_identity": self.approval_identity,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "LocalIsolationSubplan":
        _assert_fields(raw, _LOCAL_SUBPLAN_FIELDS, "local plan")
        return cls(
            command=IsolationCommandPlan.from_dict(
                _mapping(_required(raw, "command", "local plan"), "command")
            ),
            policy=LocalIsolationPolicy.from_dict(
                _mapping(_required(raw, "policy", "local plan"), "policy")
            ),
            policy_version=_optional_str_or_default(
                raw,
                "policy_version",
                "phase4",
            ),
            profile_name=_optional_str_or_default(
                raw,
                "profile_name",
                "local",
            ),
            approval_identity=_optional_str(raw, "approval_identity"),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxIsolationSubplan:
    command: IsolationCommandPlan
    settings: SandboxEffectiveSettings
    temp_dir: str | None = None
    output_dir: str | None = None
    collect_outputs: bool = False
    cleanup_budget_seconds: float = 2.0
    stream_buffer_bytes: int = 65536

    def __post_init__(self) -> None:
        assert isinstance(self.command, IsolationCommandPlan)
        assert isinstance(self.settings, SandboxEffectiveSettings)
        profile = self.settings.profile
        assert (
            self.command.executable_path in profile.trusted_executables
        ), "sandbox executable must be trusted"
        assert _path_allowed_for_read(
            self.command.cwd,
            profile.read_roots,
            profile.write_roots,
            profile.deny_roots,
        ), "sandbox cwd must be inside read/write roots"
        _assert_environment_names_allowed(
            self.command.environment_names,
            tuple(profile.environment.allowlist)
            + tuple(profile.environment.variables),
            "sandbox environment",
        )
        temp_dir = _optional_absolute_path(self.temp_dir, "temp_dir")
        output_dir = _optional_absolute_path(self.output_dir, "output_dir")
        if temp_dir is not None:
            assert _path_inside_any(
                temp_dir,
                profile.scratch_roots,
            ), "temp_dir must be inside sandbox scratch roots"
        if output_dir is not None:
            assert _path_inside_any(
                output_dir,
                profile.output_roots,
            ), "output_dir must be inside sandbox output roots"
        _assert_bool(self.collect_outputs, "collect_outputs")
        _assert_positive_number(
            self.cleanup_budget_seconds,
            "cleanup_budget_seconds",
        )
        _assert_positive_int(self.stream_buffer_bytes, "stream_buffer_bytes")
        object.__setattr__(self, "temp_dir", temp_dir)
        object.__setattr__(self, "output_dir", output_dir)

    @property
    def backend(self) -> SandboxBackend:
        return cast(SandboxBackend, self.settings.backend)

    def canonical_policy_input(self) -> dict[str, object]:
        return {
            "mode": IsolationMode.SANDBOX.value,
            "backend": self.backend.value,
            "command": self.command.canonical_policy_input(),
            "settings": self.settings.canonical_policy_input(),
            "temp_mapping": {
                "temp_dir": self.temp_dir,
            },
            "output_mapping": {
                "output_dir": self.output_dir,
                "collect_outputs": self.collect_outputs,
            },
            "cleanup_budget_seconds": self.cleanup_budget_seconds,
            "stream_buffer_bytes": self.stream_buffer_bytes,
            "policy_version": self.settings.policy_version,
            "profile_registry_id": self.settings.profile_registry_id,
            "profile_name": self.settings.profile_name,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command.to_dict(),
            "settings": self.settings.to_dict(),
            "temp_dir": self.temp_dir,
            "output_dir": self.output_dir,
            "collect_outputs": self.collect_outputs,
            "cleanup_budget_seconds": self.cleanup_budget_seconds,
            "stream_buffer_bytes": self.stream_buffer_bytes,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "SandboxIsolationSubplan":
        _assert_fields(raw, _SANDBOX_SUBPLAN_FIELDS, "sandbox plan")
        return cls(
            command=IsolationCommandPlan.from_dict(
                _mapping(_required(raw, "command", "sandbox plan"), "command")
            ),
            settings=SandboxEffectiveSettings.from_dict(
                _mapping(
                    _required(raw, "settings", "sandbox plan"),
                    "settings",
                )
            ),
            temp_dir=_optional_str(raw, "temp_dir"),
            output_dir=_optional_str(raw, "output_dir"),
            collect_outputs=_optional_bool_or_default(
                raw,
                "collect_outputs",
                False,
            ),
            cleanup_budget_seconds=_optional_number_or_default(
                raw,
                "cleanup_budget_seconds",
                2.0,
            ),
            stream_buffer_bytes=_optional_int_or_default(
                raw,
                "stream_buffer_bytes",
                65536,
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerIsolationSubplan:
    command: IsolationCommandPlan
    settings: ContainerEffectiveSettings

    def __post_init__(self) -> None:
        assert isinstance(self.command, IsolationCommandPlan)
        assert isinstance(self.settings, ContainerEffectiveSettings)
        assert self.settings.enabled, "container plan requires enabled backend"
        assert (
            self.settings.profile is not None
        ), "container plan requires profile"

    @property
    def backend(self) -> ContainerBackend:
        return cast(ContainerBackend, self.settings.backend)

    def canonical_policy_input(self) -> dict[str, object]:
        return {
            "mode": IsolationMode.CONTAINER.value,
            "backend": self.backend.value,
            "command": self.command.canonical_policy_input(),
            "settings": self.settings.canonical_policy_input(),
            "policy_version": self.settings.policy_version,
            "profile_registry_id": self.settings.profile_registry_id,
            "profile_name": self.settings.profile_name,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command.to_dict(),
            "settings": self.settings.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ContainerIsolationSubplan":
        _assert_fields(raw, _CONTAINER_SUBPLAN_FIELDS, "container plan")
        return cls(
            command=IsolationCommandPlan.from_dict(
                _mapping(
                    _required(raw, "command", "container plan"),
                    "command",
                )
            ),
            settings=ContainerEffectiveSettings.from_dict(
                _mapping(
                    _required(raw, "settings", "container plan"),
                    "settings",
                )
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationPlan:
    requested_mode: IsolationMode | str
    local: LocalIsolationSubplan | None = None
    sandbox: SandboxIsolationSubplan | None = None
    container: ContainerIsolationSubplan | None = None

    def __post_init__(self) -> None:
        requested_mode = _enum_value(
            self.requested_mode,
            IsolationMode,
            "requested_mode",
        )
        _assert_plan_union(self.local, self.sandbox, self.container)
        effective_mode = _active_mode(self.local, self.sandbox, self.container)
        assert _rung_index(effective_mode) >= _rung_index(
            requested_mode,
        ), "isolation plan cannot move to a stronger mode implicitly"
        assert (
            _rung_index(effective_mode)
            <= _rung_index(
                requested_mode,
            )
            + 1
        ), "isolation elevation must advance exactly one rung at a time"
        object.__setattr__(self, "requested_mode", requested_mode)

    @property
    def effective_mode(self) -> IsolationMode:
        return _active_mode(self.local, self.sandbox, self.container)

    @property
    def command(self) -> IsolationCommandPlan:
        if self.local is not None:
            return self.local.command
        if self.sandbox is not None:
            return self.sandbox.command
        assert self.container is not None
        return self.container.command

    @property
    def backend(self) -> str | None:
        if self.container is not None:
            return self.container.backend.value
        if self.sandbox is not None:
            return self.sandbox.backend.value
        return None

    @property
    def profile_name(self) -> str:
        if self.local is not None:
            return self.local.profile_name
        if self.sandbox is not None:
            return self.sandbox.settings.profile_name
        assert self.container is not None
        assert self.container.settings.profile_name is not None
        return self.container.settings.profile_name

    @property
    def policy_version(self) -> str:
        if self.local is not None:
            return self.local.policy_version
        if self.sandbox is not None:
            return self.sandbox.settings.policy_version
        assert self.container is not None
        return self.container.settings.policy_version

    @property
    def elevation_rung(self) -> IsolationElevationRung:
        return IsolationElevationRung(self.effective_mode.value)

    @property
    def lost_controls(self) -> tuple[str, ...]:
        if self.requested_mode == self.effective_mode:
            return ()
        if self.effective_mode is IsolationMode.SANDBOX:
            return _CONTAINER_TO_SANDBOX_LOST_CONTROLS
        return _SANDBOX_TO_LOCAL_LOST_CONTROLS

    @property
    def plan_fingerprint(self) -> str:
        return _fingerprint(self.canonical_policy_input())

    def canonical_policy_input(self) -> dict[str, object]:
        requested_mode = cast(IsolationMode, self.requested_mode)
        return {
            "requested_mode": requested_mode.value,
            "effective_mode": self.effective_mode.value,
            "elevation_rung": self.elevation_rung.value,
            "plan": _active_subplan(self).canonical_policy_input(),
        }

    def to_metadata(self) -> "IsolationDurablePlanMetadata":
        request_kind = cast(
            IsolationPlanRequestKind,
            self.command.request_kind,
        )
        return IsolationDurablePlanMetadata(
            request_kind=request_kind,
            request_id=self.command.request_id,
            attempt_id=self.command.attempt_id,
            requested_mode=cast(IsolationMode, self.requested_mode),
            effective_mode=self.effective_mode,
            backend=self.backend,
            profile_name=self.profile_name,
            policy_version=self.policy_version,
            plan_fingerprint=self.plan_fingerprint,
            elevation_rung=self.elevation_rung,
        )

    def to_dict(self) -> dict[str, object]:
        requested_mode = cast(IsolationMode, self.requested_mode)
        return {
            "requested_mode": requested_mode.value,
            "local": self.local.to_dict() if self.local else None,
            "sandbox": self.sandbox.to_dict() if self.sandbox else None,
            "container": self.container.to_dict() if self.container else None,
            "plan_fingerprint": self.plan_fingerprint,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "IsolationPlan":
        _assert_fields(raw, _ISOLATION_PLAN_FIELDS, "isolation plan")
        plan = cls(
            requested_mode=_required_str(
                raw,
                "requested_mode",
                "isolation plan",
            ),
            local=_local_subplan_from_raw(raw),
            sandbox=_sandbox_subplan_from_raw(raw),
            container=_container_subplan_from_raw(raw),
        )
        assert (
            _required_str(raw, "plan_fingerprint", "isolation plan")
            == plan.plan_fingerprint
        ), "isolation plan fingerprint changed"
        return plan

    @classmethod
    def from_json(cls, serialized: str) -> "IsolationPlan":
        _assert_non_empty_string(serialized, "serialized")
        return cls.from_dict(_mapping(loads(serialized), "isolation plan"))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationDurablePlanMetadata:
    request_kind: IsolationPlanRequestKind | str
    request_id: str | None
    attempt_id: str | None
    requested_mode: IsolationMode | str
    effective_mode: IsolationMode | str
    backend: str | None
    profile_name: str
    policy_version: str
    plan_fingerprint: str
    elevation_rung: IsolationElevationRung | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_kind",
            _enum_value(
                self.request_kind,
                IsolationPlanRequestKind,
                "request_kind",
            ),
        )
        object.__setattr__(
            self,
            "requested_mode",
            _enum_value(self.requested_mode, IsolationMode, "requested_mode"),
        )
        object.__setattr__(
            self,
            "effective_mode",
            _enum_value(self.effective_mode, IsolationMode, "effective_mode"),
        )
        object.__setattr__(
            self,
            "elevation_rung",
            _enum_value(
                self.elevation_rung,
                IsolationElevationRung,
                "elevation_rung",
            ),
        )
        if self.backend is not None:
            _assert_non_empty_string(self.backend, "backend")
        if self.request_id is not None:
            _assert_non_empty_string(self.request_id, "request_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_non_empty_string(self.profile_name, "profile_name")
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")

    def assert_matches(self, plan: IsolationPlan) -> None:
        assert isinstance(plan, IsolationPlan)
        assert (
            self.request_kind is plan.command.request_kind
        ), "durable request kind changed"
        assert (
            self.request_id == plan.command.request_id
        ), "durable request id changed"
        assert (
            self.attempt_id == plan.command.attempt_id
        ), "durable attempt id changed"
        assert (
            self.requested_mode is plan.requested_mode
        ), "durable requested mode changed"
        assert (
            self.effective_mode is plan.effective_mode
        ), "durable effective mode changed"
        assert self.backend == plan.backend, "durable backend changed"
        assert (
            self.profile_name == plan.profile_name
        ), "durable profile changed"
        assert (
            self.policy_version == plan.policy_version
        ), "durable policy version changed"
        assert (
            self.plan_fingerprint == plan.plan_fingerprint
        ), "durable plan fingerprint changed"
        assert (
            self.elevation_rung is plan.elevation_rung
        ), "durable elevation rung changed"

    def to_dict(self) -> dict[str, object]:
        request_kind = cast(IsolationPlanRequestKind, self.request_kind)
        requested_mode = cast(IsolationMode, self.requested_mode)
        effective_mode = cast(IsolationMode, self.effective_mode)
        elevation_rung = cast(IsolationElevationRung, self.elevation_rung)
        return {
            "request_kind": request_kind.value,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "requested_mode": requested_mode.value,
            "effective_mode": effective_mode.value,
            "backend": self.backend,
            "profile_name": self.profile_name,
            "policy_version": self.policy_version,
            "plan_fingerprint": self.plan_fingerprint,
            "elevation_rung": elevation_rung.value,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "IsolationDurablePlanMetadata":
        _assert_fields(raw, _METADATA_FIELDS, "metadata")
        return cls(
            request_kind=_required_str(raw, "request_kind", "metadata"),
            request_id=_optional_str(raw, "request_id"),
            attempt_id=_optional_str(raw, "attempt_id"),
            requested_mode=_required_str(raw, "requested_mode", "metadata"),
            effective_mode=_required_str(raw, "effective_mode", "metadata"),
            backend=_optional_str(raw, "backend"),
            profile_name=_required_str(raw, "profile_name", "metadata"),
            policy_version=_required_str(
                raw,
                "policy_version",
                "metadata",
            ),
            plan_fingerprint=_required_str(
                raw,
                "plan_fingerprint",
                "metadata",
            ),
            elevation_rung=_required_str(
                raw,
                "elevation_rung",
                "metadata",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationPolicyContext:
    surface: IsolationReviewSurface | str
    scope_id: str
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        surface = _enum_value(
            self.surface,
            IsolationReviewSurface,
            "surface",
        )
        _assert_non_empty_string(self.scope_id, "scope_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if surface in _ATTEMPT_BOUND_SURFACES:
            assert (
                self.attempt_id is not None
            ), "attempt-bound review surfaces require attempt_id"
        object.__setattr__(self, "surface", surface)

    @property
    def review_mode(self) -> IsolationReviewMode:
        surface = cast(IsolationReviewSurface, self.surface)
        return _REVIEW_MODES[surface]

    def to_dict(self) -> dict[str, object]:
        surface = cast(IsolationReviewSurface, self.surface)
        return {
            "surface": surface.value,
            "scope_id": self.scope_id,
            "attempt_id": self.attempt_id,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationApprovalRecord:
    reviewer_identity: str
    review_mode: IsolationReviewMode | str
    mode: IsolationMode | str
    plan_fingerprint: str
    scope_id: str
    attempt_id: str | None
    policy_version: str
    elevation_rung: IsolationElevationRung | str
    expires_at_seconds: int

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.reviewer_identity, "reviewer_identity")
        object.__setattr__(
            self,
            "review_mode",
            _enum_value(self.review_mode, IsolationReviewMode, "review_mode"),
        )
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, IsolationMode, "mode"),
        )
        object.__setattr__(
            self,
            "elevation_rung",
            _enum_value(
                self.elevation_rung,
                IsolationElevationRung,
                "elevation_rung",
            ),
        )
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")
        _assert_non_empty_string(self.scope_id, "scope_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_positive_int(self.expires_at_seconds, "expires_at_seconds")

    @classmethod
    def for_plan(
        cls,
        plan: IsolationPlan,
        context: IsolationPolicyContext,
        *,
        reviewer_identity: str,
        expires_at_seconds: int,
    ) -> "IsolationApprovalRecord":
        assert isinstance(plan, IsolationPlan)
        assert isinstance(context, IsolationPolicyContext)
        assert (
            context.review_mode is IsolationReviewMode.DURABLE
        ), "approval records require durable review surfaces"
        return cls(
            reviewer_identity=reviewer_identity,
            review_mode=context.review_mode,
            mode=plan.effective_mode,
            plan_fingerprint=plan.plan_fingerprint,
            scope_id=context.scope_id,
            attempt_id=context.attempt_id,
            policy_version=plan.policy_version,
            elevation_rung=plan.elevation_rung,
            expires_at_seconds=expires_at_seconds,
        )

    def assert_applies_to(
        self,
        plan: IsolationPlan,
        context: IsolationPolicyContext,
        *,
        now_seconds: int,
    ) -> None:
        assert isinstance(plan, IsolationPlan)
        assert isinstance(context, IsolationPolicyContext)
        assert isinstance(now_seconds, int), "now_seconds must be an integer"
        assert now_seconds >= 0, "now_seconds must not be negative"
        assert now_seconds < self.expires_at_seconds, "approval is stale"
        assert (
            self.review_mode is context.review_mode
        ), "approval review mode does not match"
        assert self.mode is plan.effective_mode, "approval mode changed"
        assert (
            self.plan_fingerprint == plan.plan_fingerprint
        ), "approval plan fingerprint changed"
        assert self.scope_id == context.scope_id, "approval scope changed"
        assert (
            self.attempt_id == context.attempt_id
        ), "approval attempt changed"
        assert (
            self.policy_version == plan.policy_version
        ), "approval policy version changed"
        assert (
            self.elevation_rung is plan.elevation_rung
        ), "approval elevation rung changed"

    def to_dict(self) -> dict[str, object]:
        review_mode = cast(IsolationReviewMode, self.review_mode)
        mode = cast(IsolationMode, self.mode)
        elevation_rung = cast(IsolationElevationRung, self.elevation_rung)
        return {
            "reviewer_identity": self.reviewer_identity,
            "review_mode": review_mode.value,
            "mode": mode.value,
            "plan_fingerprint": self.plan_fingerprint,
            "scope_id": self.scope_id,
            "attempt_id": self.attempt_id,
            "policy_version": self.policy_version,
            "elevation_rung": elevation_rung.value,
            "expires_at_seconds": self.expires_at_seconds,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "IsolationApprovalRecord":
        _assert_fields(raw, _APPROVAL_FIELDS, "approval")
        return cls(
            reviewer_identity=_required_str(
                raw,
                "reviewer_identity",
                "approval",
            ),
            review_mode=_required_str(raw, "review_mode", "approval"),
            mode=_required_str(raw, "mode", "approval"),
            plan_fingerprint=_required_str(
                raw,
                "plan_fingerprint",
                "approval",
            ),
            scope_id=_required_str(raw, "scope_id", "approval"),
            attempt_id=_optional_str(raw, "attempt_id"),
            policy_version=_required_str(raw, "policy_version", "approval"),
            elevation_rung=_required_str(
                raw,
                "elevation_rung",
                "approval",
            ),
            expires_at_seconds=_required_int(
                raw,
                "expires_at_seconds",
                "approval",
            ),
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationPolicyDecision:
    decision: IsolationDecisionType | str
    code: str
    explanation: str
    policy_version: str
    requested_mode: IsolationMode | str
    effective_mode: IsolationMode | str
    plan_fingerprint: str
    elevation_rung: IsolationElevationRung | str
    lost_controls: Sequence[str] = field(default_factory=tuple)
    reviewer_identity: str | None = None
    retryable: bool = False
    cacheable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision",
            _enum_value(self.decision, IsolationDecisionType, "decision"),
        )
        object.__setattr__(
            self,
            "requested_mode",
            _enum_value(self.requested_mode, IsolationMode, "requested_mode"),
        )
        object.__setattr__(
            self,
            "effective_mode",
            _enum_value(self.effective_mode, IsolationMode, "effective_mode"),
        )
        object.__setattr__(
            self,
            "elevation_rung",
            _enum_value(
                self.elevation_rung,
                IsolationElevationRung,
                "elevation_rung",
            ),
        )
        _assert_non_empty_string(self.code, "code")
        _assert_non_empty_string(self.explanation, "explanation")
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")
        lost_controls = _string_tuple(self.lost_controls, "lost_controls")
        if self.reviewer_identity is not None:
            _assert_non_empty_string(
                self.reviewer_identity,
                "reviewer_identity",
            )
        _assert_bool(self.retryable, "retryable")
        _assert_bool(self.cacheable, "cacheable")
        object.__setattr__(self, "lost_controls", lost_controls)

    def to_dict(self) -> dict[str, object]:
        decision = cast(IsolationDecisionType, self.decision)
        requested_mode = cast(IsolationMode, self.requested_mode)
        effective_mode = cast(IsolationMode, self.effective_mode)
        elevation_rung = cast(IsolationElevationRung, self.elevation_rung)
        return {
            "decision": decision.value,
            "code": self.code,
            "explanation": self.explanation,
            "policy_version": self.policy_version,
            "requested_mode": requested_mode.value,
            "effective_mode": effective_mode.value,
            "plan_fingerprint": self.plan_fingerprint,
            "elevation_rung": elevation_rung.value,
            "lost_controls": list(self.lost_controls),
            "reviewer_identity": self.reviewer_identity,
            "retryable": self.retryable,
            "cacheable": self.cacheable,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationPolicy:
    policy_version: str
    preapproved_modes: Sequence[IsolationMode | str] = field(
        default_factory=lambda: (
            IsolationMode.CONTAINER,
            IsolationMode.SANDBOX,
        ),
    )
    denied_modes: Sequence[IsolationMode | str] = field(default_factory=tuple)
    explanations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.policy_version, "policy_version")
        object.__setattr__(
            self,
            "preapproved_modes",
            _mode_tuple(self.preapproved_modes, "preapproved_modes"),
        )
        object.__setattr__(
            self,
            "denied_modes",
            _mode_tuple(self.denied_modes, "denied_modes"),
        )
        object.__setattr__(
            self,
            "explanations",
            MappingProxyType(
                _string_mapping(self.explanations, "explanations")
            ),
        )

    def authorize(
        self,
        plan: IsolationPlan,
        context: IsolationPolicyContext,
        *,
        approval: IsolationApprovalRecord | None = None,
        now_seconds: int = 0,
    ) -> IsolationPolicyDecision:
        assert isinstance(plan, IsolationPlan)
        assert isinstance(context, IsolationPolicyContext)
        assert (
            plan.policy_version == self.policy_version
        ), "plan policy version must match policy version"
        if plan.effective_mode in self.denied_modes:
            return self._decision(
                IsolationDecisionType.DENY,
                code=f"isolation.deny.{plan.effective_mode.value}",
                explanation="Isolation mode is denied by policy.",
                plan=plan,
            )
        if approval is not None:
            assert isinstance(approval, IsolationApprovalRecord)
            assert (
                context.review_mode is IsolationReviewMode.DURABLE
            ), "approval records require durable review surfaces"
            approval.assert_applies_to(
                plan,
                context,
                now_seconds=now_seconds,
            )
            return self._decision(
                IsolationDecisionType.ALLOW,
                code="isolation.allow.cached_approval",
                explanation="Isolation plan matches cached review.",
                plan=plan,
                reviewer_identity=approval.reviewer_identity,
                cacheable=True,
            )
        if _requires_review(plan):
            return self._review_decision(plan, context)
        if plan.effective_mode in self.preapproved_modes:
            return self._decision(
                IsolationDecisionType.ALLOW,
                code="isolation.allow.preapproved",
                explanation="Trusted policy pre-authorizes this plan.",
                plan=plan,
                cacheable=True,
            )
        return self._review_decision(plan, context)

    def cache_key(
        self,
        plan: IsolationPlan,
        context: IsolationPolicyContext,
    ) -> str:
        assert isinstance(plan, IsolationPlan)
        assert isinstance(context, IsolationPolicyContext)
        return _fingerprint(
            {
                "policy_fingerprint": self.policy_fingerprint,
                "plan_fingerprint": plan.plan_fingerprint,
                "context": context.to_dict(),
            }
        )

    @property
    def policy_fingerprint(self) -> str:
        return _fingerprint(self.canonical_policy_input())

    def canonical_policy_input(self) -> dict[str, object]:
        preapproved_modes = cast(
            tuple[IsolationMode, ...],
            self.preapproved_modes,
        )
        denied_modes = cast(tuple[IsolationMode, ...], self.denied_modes)
        return {
            "policy_version": self.policy_version,
            "preapproved_modes": sorted(
                mode.value for mode in preapproved_modes
            ),
            "denied_modes": sorted(mode.value for mode in denied_modes),
            "explanations": dict(sorted(self.explanations.items())),
        }

    def _review_decision(
        self,
        plan: IsolationPlan,
        context: IsolationPolicyContext,
    ) -> IsolationPolicyDecision:
        if context.review_mode is IsolationReviewMode.FAIL_CLOSED:
            return self._decision(
                IsolationDecisionType.DENY,
                code="isolation.deny.review_unavailable",
                explanation="Execution surface cannot perform review.",
                plan=plan,
            )
        return self._decision(
            IsolationDecisionType.REQUIRES_REVIEW,
            code=f"isolation.review.{context.review_mode.value}",
            explanation="Isolation plan requires operator review.",
            plan=plan,
            retryable=True,
        )

    def _decision(
        self,
        decision: IsolationDecisionType,
        *,
        code: str,
        explanation: str,
        plan: IsolationPlan,
        reviewer_identity: str | None = None,
        retryable: bool = False,
        cacheable: bool = False,
    ) -> IsolationPolicyDecision:
        return IsolationPolicyDecision(
            decision=decision,
            code=code,
            explanation=self.explanations.get(code, explanation),
            policy_version=self.policy_version,
            requested_mode=plan.requested_mode,
            effective_mode=plan.effective_mode,
            plan_fingerprint=plan.plan_fingerprint,
            elevation_rung=plan.elevation_rung,
            lost_controls=plan.lost_controls,
            reviewer_identity=reviewer_identity,
            retryable=retryable,
            cacheable=cacheable,
        )


@final
@dataclass(kw_only=True, slots=True)
class IsolationPolicyEvaluationCache:
    max_entries: int = 128
    _entries: dict[str, IsolationPolicyDecision] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_entries, "max_entries")

    def authorize(
        self,
        policy: IsolationPolicy,
        plan: IsolationPlan,
        context: IsolationPolicyContext,
    ) -> IsolationPolicyDecision:
        key = policy.cache_key(plan, context)
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        decision = policy.authorize(plan, context)
        if decision.cacheable:
            if len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = decision
        return decision

    @property
    def size(self) -> int:
        return len(self._entries)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class IsolationAuditRecord:
    requested_mode: IsolationMode | str
    effective_mode: IsolationMode | str
    backend: str | None
    profile: str
    policy_version: str
    fingerprint: str
    reviewer_identity: str | None
    elevation_rung: IsolationElevationRung | str
    lost_controls: Sequence[str]
    diagnostic_code: str
    cleanup_status: IsolationCleanupStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_mode",
            _enum_value(self.requested_mode, IsolationMode, "requested_mode"),
        )
        object.__setattr__(
            self,
            "effective_mode",
            _enum_value(self.effective_mode, IsolationMode, "effective_mode"),
        )
        object.__setattr__(
            self,
            "elevation_rung",
            _enum_value(
                self.elevation_rung,
                IsolationElevationRung,
                "elevation_rung",
            ),
        )
        object.__setattr__(
            self,
            "cleanup_status",
            _enum_value(
                self.cleanup_status,
                IsolationCleanupStatus,
                "cleanup_status",
            ),
        )
        if self.backend is not None:
            _assert_non_empty_string(self.backend, "backend")
        _assert_non_empty_string(self.profile, "profile")
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_non_empty_string(self.fingerprint, "fingerprint")
        if self.reviewer_identity is not None:
            _assert_non_empty_string(
                self.reviewer_identity,
                "reviewer_identity",
            )
        object.__setattr__(
            self,
            "lost_controls",
            _string_tuple(self.lost_controls, "lost_controls"),
        )
        _assert_non_empty_string(self.diagnostic_code, "diagnostic_code")

    @classmethod
    def from_plan_decision(
        cls,
        plan: IsolationPlan,
        decision: IsolationPolicyDecision,
        *,
        cleanup_status: IsolationCleanupStatus | str = (
            IsolationCleanupStatus.NOT_STARTED
        ),
    ) -> "IsolationAuditRecord":
        assert isinstance(plan, IsolationPlan)
        assert isinstance(decision, IsolationPolicyDecision)
        assert (
            decision.plan_fingerprint == plan.plan_fingerprint
        ), "audit decision fingerprint must match plan"
        return cls(
            requested_mode=plan.requested_mode,
            effective_mode=plan.effective_mode,
            backend=plan.backend,
            profile=plan.profile_name,
            policy_version=decision.policy_version,
            fingerprint=plan.plan_fingerprint,
            reviewer_identity=decision.reviewer_identity,
            elevation_rung=plan.elevation_rung,
            lost_controls=decision.lost_controls,
            diagnostic_code=decision.code,
            cleanup_status=cleanup_status,
        )

    def to_dict(self) -> dict[str, object]:
        requested_mode = cast(IsolationMode, self.requested_mode)
        effective_mode = cast(IsolationMode, self.effective_mode)
        elevation_rung = cast(IsolationElevationRung, self.elevation_rung)
        cleanup_status = cast(IsolationCleanupStatus, self.cleanup_status)
        return {
            "requested_mode": requested_mode.value,
            "effective_mode": effective_mode.value,
            "backend": self.backend,
            "profile": self.profile,
            "policy_version": self.policy_version,
            "fingerprint": self.fingerprint,
            "reviewer_identity": self.reviewer_identity,
            "elevation_rung": elevation_rung.value,
            "lost_controls": list(self.lost_controls),
            "diagnostic_code": self.diagnostic_code,
            "cleanup_status": cleanup_status.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "IsolationAuditRecord":
        _assert_fields(raw, _AUDIT_FIELDS, "audit")
        return cls(
            requested_mode=_required_str(raw, "requested_mode", "audit"),
            effective_mode=_required_str(raw, "effective_mode", "audit"),
            backend=_optional_str(raw, "backend"),
            profile=_required_str(raw, "profile", "audit"),
            policy_version=_required_str(raw, "policy_version", "audit"),
            fingerprint=_required_str(raw, "fingerprint", "audit"),
            reviewer_identity=_optional_str(raw, "reviewer_identity"),
            elevation_rung=_required_str(raw, "elevation_rung", "audit"),
            lost_controls=_string_tuple(
                raw.get("lost_controls", ()),
                "lost_controls",
            ),
            diagnostic_code=_required_str(raw, "diagnostic_code", "audit"),
            cleanup_status=_required_str(raw, "cleanup_status", "audit"),
        )


def normalize_shell_request(
    request: IsolationShellRequest,
) -> IsolationCommandPlan:
    assert isinstance(request, IsolationShellRequest)
    return request.to_command_plan()


def lower_isolation_plan(
    settings: object,
    request: IsolationShellRequest | IsolationCommandPlan,
    *,
    requested_mode: IsolationMode | str | None = None,
    approval_identity: str | None = None,
    temp_dir: str | None = None,
    output_dir: str | None = None,
    collect_outputs: bool = False,
) -> IsolationPlan:
    command = (
        request.to_command_plan()
        if isinstance(request, IsolationShellRequest)
        else request
    )
    assert isinstance(command, IsolationCommandPlan)
    if isinstance(settings, IsolationEffectiveSettings):
        if settings.container is not None:
            return lower_isolation_plan(
                settings.container,
                command,
                requested_mode=requested_mode,
            )
        if settings.sandbox is not None:
            return lower_isolation_plan(
                settings.sandbox,
                command,
                requested_mode=requested_mode,
                temp_dir=temp_dir,
                output_dir=output_dir,
                collect_outputs=collect_outputs,
            )
        assert settings.local is not None
        return lower_isolation_plan(
            settings.local,
            command,
            requested_mode=requested_mode,
            approval_identity=approval_identity,
        )
    if isinstance(settings, ContainerEffectiveSettings):
        mode = IsolationMode.CONTAINER
        container_subplan = ContainerIsolationSubplan(
            command=command,
            settings=settings,
        )
        return IsolationPlan(
            requested_mode=requested_mode or mode,
            container=container_subplan,
        )
    if isinstance(settings, SandboxEffectiveSettings):
        mode = IsolationMode.SANDBOX
        sandbox_subplan = SandboxIsolationSubplan(
            command=command,
            settings=settings,
            temp_dir=temp_dir,
            output_dir=output_dir,
            collect_outputs=collect_outputs,
        )
        return IsolationPlan(
            requested_mode=requested_mode or mode,
            sandbox=sandbox_subplan,
        )
    assert isinstance(settings, LocalIsolationPolicy)
    mode = IsolationMode.LOCAL
    local_subplan = LocalIsolationSubplan(
        command=command,
        policy=settings,
        approval_identity=approval_identity,
    )
    return IsolationPlan(
        requested_mode=requested_mode or mode,
        local=local_subplan,
    )


def elevate_isolation_plan(
    plan: IsolationPlan,
    settings: SandboxEffectiveSettings | LocalIsolationPolicy,
    *,
    approval_identity: str | None = None,
    temp_dir: str | None = None,
    output_dir: str | None = None,
    collect_outputs: bool = False,
) -> IsolationPlan:
    assert isinstance(plan, IsolationPlan)
    next_mode = _next_mode(plan.effective_mode)
    if isinstance(settings, SandboxEffectiveSettings):
        assert (
            next_mode is IsolationMode.SANDBOX
        ), "sandbox settings can only elevate a container plan"
    else:
        assert isinstance(settings, LocalIsolationPolicy)
        assert (
            next_mode is IsolationMode.LOCAL
        ), "local settings can only elevate a sandbox plan"
    elevated = lower_isolation_plan(
        settings,
        plan.command,
        requested_mode=plan.effective_mode,
        approval_identity=approval_identity,
        temp_dir=temp_dir,
        output_dir=output_dir,
        collect_outputs=collect_outputs,
    )
    assert (
        elevated.plan_fingerprint != plan.plan_fingerprint
    ), "elevation requires a fresh plan fingerprint"
    return elevated


_COMMAND_FIELDS = {
    "request_kind",
    "logical_name",
    "executable_path",
    "argv",
    "cwd",
    "environment_names",
    "request_id",
    "attempt_id",
}
_LOCAL_SUBPLAN_FIELDS = {
    "command",
    "policy",
    "policy_version",
    "profile_name",
    "approval_identity",
}
_SANDBOX_SUBPLAN_FIELDS = {
    "command",
    "settings",
    "temp_dir",
    "output_dir",
    "collect_outputs",
    "cleanup_budget_seconds",
    "stream_buffer_bytes",
}
_CONTAINER_SUBPLAN_FIELDS = {"command", "settings"}
_ISOLATION_PLAN_FIELDS = {
    "requested_mode",
    "local",
    "sandbox",
    "container",
    "plan_fingerprint",
}
_METADATA_FIELDS = {
    "request_kind",
    "request_id",
    "attempt_id",
    "requested_mode",
    "effective_mode",
    "backend",
    "profile_name",
    "policy_version",
    "plan_fingerprint",
    "elevation_rung",
}
_APPROVAL_FIELDS = {
    "reviewer_identity",
    "review_mode",
    "mode",
    "plan_fingerprint",
    "scope_id",
    "attempt_id",
    "policy_version",
    "elevation_rung",
    "expires_at_seconds",
}
_AUDIT_FIELDS = {
    "requested_mode",
    "effective_mode",
    "backend",
    "profile",
    "policy_version",
    "fingerprint",
    "reviewer_identity",
    "elevation_rung",
    "lost_controls",
    "diagnostic_code",
    "cleanup_status",
}
_REVIEW_MODES = {
    IsolationReviewSurface.INTERACTIVE_CLI: IsolationReviewMode.INTERACTIVE,
    IsolationReviewSurface.STRICT_FLOW: IsolationReviewMode.DURABLE,
    IsolationReviewSurface.DIRECT_TASK: IsolationReviewMode.FAIL_CLOSED,
    IsolationReviewSurface.QUEUED_TASK: IsolationReviewMode.DURABLE,
    IsolationReviewSurface.SERVER: IsolationReviewMode.FAIL_CLOSED,
    IsolationReviewSurface.MCP: IsolationReviewMode.FAIL_CLOSED,
    IsolationReviewSurface.A2A: IsolationReviewMode.FAIL_CLOSED,
}
_ATTEMPT_BOUND_SURFACES = {IsolationReviewSurface.QUEUED_TASK}
_CONTAINER_TO_SANDBOX_LOST_CONTROLS = (
    "container_image_isolation",
    "container_mount_namespace",
    "container_resource_limits",
)
_SANDBOX_TO_LOCAL_LOST_CONTROLS = (
    "sandbox_filesystem_policy",
    "sandbox_process_policy",
    "sandbox_network_policy",
)


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


def _mode_tuple(
    value: Sequence[IsolationMode | str],
    field_name: str,
) -> tuple[IsolationMode, ...]:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), f"{field_name} must be a sequence"
    modes = tuple(
        _enum_value(item, IsolationMode, field_name) for item in value
    )
    assert len(set(modes)) == len(modes), f"{field_name} must be unique"
    return modes


def _absolute_path(value: object, field_name: str) -> str:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    assert value.startswith("/"), f"{field_name} must be absolute"
    return normalize_posix_path(value)


def _optional_absolute_path(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _absolute_path(value, field_name)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value,
        str | bytes,
    ), f"{field_name} must be a sequence"
    normalized: list[str] = []
    for item in value:
        _assert_non_empty_string(item, field_name)
        assert isinstance(item, str), f"{field_name} must contain strings"
        assert "\x00" not in item, f"{field_name} must not contain NUL"
        normalized.append(item)
    return tuple(normalized)


def _environment_name_tuple(
    value: object,
    field_name: str,
) -> tuple[str, ...]:
    names = _string_tuple(value, field_name)
    for name in names:
        _assert_env_name(name, field_name)
    assert len(set(names)) == len(names), f"{field_name} must be unique"
    return tuple(sorted(names))


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    normalized: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_non_empty_string(item, f"{field_name}.{key}")
        normalized[key] = item
    return normalized


def _assert_environment_names_allowed(
    names: Sequence[str],
    allowed: Sequence[str],
    field_name: str,
) -> None:
    allowed_names = set(allowed)
    for name in names:
        assert name in allowed_names, f"{field_name} variable {name} is denied"


def _path_allowed_for_read(
    path: str,
    read_roots: Sequence[str],
    write_roots: Sequence[str],
    deny_roots: Sequence[str],
) -> bool:
    if _path_inside_any(path, deny_roots):
        return False
    return _path_inside_any(path, tuple(read_roots) + tuple(write_roots))


def _path_inside_any(path: str, roots: Sequence[str]) -> bool:
    if not roots:
        return True
    normalized_path = _normalized_parts(path)
    for root in roots:
        normalized_root = _normalized_parts(root)
        if len(normalized_path) >= len(normalized_root) and (
            normalized_path[: len(normalized_root)] == normalized_root
        ):
            return True
    return False


def _normalized_parts(path: str) -> tuple[str, ...]:
    normalized = normalize_posix_path(path)
    return tuple(part for part in normalized.split("/") if part)


def _canonical_local_policy(
    policy: LocalIsolationPolicy,
) -> dict[str, object]:
    serialized = policy.to_dict()
    serialized["allowed_roots"] = sorted(policy.allowed_roots)
    serialized["executable_allowlist"] = sorted(policy.executable_allowlist)
    return serialized


def _assert_plan_union(
    local: LocalIsolationSubplan | None,
    sandbox: SandboxIsolationSubplan | None,
    container: ContainerIsolationSubplan | None,
) -> None:
    branches = tuple(
        branch for branch in (local, sandbox, container) if branch is not None
    )
    assert len(branches) == 1, "isolation plan requires exactly one subplan"
    if local is not None:
        assert isinstance(local, LocalIsolationSubplan)
    if sandbox is not None:
        assert isinstance(sandbox, SandboxIsolationSubplan)
    if container is not None:
        assert isinstance(container, ContainerIsolationSubplan)


def _active_mode(
    local: LocalIsolationSubplan | None,
    sandbox: SandboxIsolationSubplan | None,
    container: ContainerIsolationSubplan | None,
) -> IsolationMode:
    if local is not None:
        return IsolationMode.LOCAL
    if sandbox is not None:
        return IsolationMode.SANDBOX
    assert container is not None
    return IsolationMode.CONTAINER


def _active_subplan(
    plan: IsolationPlan,
) -> (
    LocalIsolationSubplan | SandboxIsolationSubplan | ContainerIsolationSubplan
):
    if plan.local is not None:
        return plan.local
    if plan.sandbox is not None:
        return plan.sandbox
    assert plan.container is not None
    return plan.container


def _rung_index(mode: IsolationMode) -> int:
    return {
        IsolationMode.CONTAINER: 0,
        IsolationMode.SANDBOX: 1,
        IsolationMode.LOCAL: 2,
    }[mode]


def _next_mode(mode: IsolationMode) -> IsolationMode:
    if mode is IsolationMode.CONTAINER:
        return IsolationMode.SANDBOX
    if mode is IsolationMode.SANDBOX:
        return IsolationMode.LOCAL
    raise AssertionError("local mode has no elevation rung")


def _requires_review(plan: IsolationPlan) -> bool:
    if plan.requested_mode != plan.effective_mode:
        return True
    if plan.local is not None:
        return True
    return False


def _local_subplan_from_raw(
    raw: Mapping[str, object],
) -> LocalIsolationSubplan | None:
    branch = raw.get("local")
    if branch is None:
        return None
    return LocalIsolationSubplan.from_dict(_mapping(branch, "local"))


def _sandbox_subplan_from_raw(
    raw: Mapping[str, object],
) -> SandboxIsolationSubplan | None:
    branch = raw.get("sandbox")
    if branch is None:
        return None
    return SandboxIsolationSubplan.from_dict(_mapping(branch, "sandbox"))


def _container_subplan_from_raw(
    raw: Mapping[str, object],
) -> ContainerIsolationSubplan | None:
    branch = raw.get("container")
    if branch is None:
        return None
    return ContainerIsolationSubplan.from_dict(_mapping(branch, "container"))


def _fingerprint(value: Mapping[str, object]) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Mapping[str, object]) -> str:
    return dumps(value, sort_keys=True, separators=(",", ":"))


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
    assert not isinstance(
        value,
        bool,
    ), f"{field_name}.{key} must be an integer"
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


def _optional_int_or_default(
    raw: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = raw.get(key)
    if value is None:
        return default
    assert isinstance(value, int), f"{key} must be an integer"
    assert not isinstance(value, bool), f"{key} must be an integer"
    return value


def _optional_number_or_default(
    raw: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    value = raw.get(key)
    if value is None:
        return default
    assert isinstance(value, int | float), f"{key} must be numeric"
    assert not isinstance(value, bool), f"{key} must be numeric"
    return float(value)


def _optional_bool_or_default(
    raw: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = raw.get(key)
    if value is None:
        return default
    _assert_bool(value, key)
    return cast(bool, value)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    return cast(Mapping[str, object], value)
