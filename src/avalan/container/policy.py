from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .settings import (
    ContainerAuthorizationDecision,
    ContainerAuthorizationDecisionType,
    ContainerBuildPolicy,
    ContainerEffectiveSettings,
    ContainerEscalationMode,
    ContainerMountAccess,
    ContainerMountType,
    ContainerNetworkMode,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from json import dumps
from posixpath import normpath as normalize_posix_path
from types import MappingProxyType
from typing import cast, final


class ContainerReviewSurface(StrEnum):
    INTERACTIVE_CLI = "interactive_cli"
    STRICT_FLOW = "strict_flow"
    DIRECT_TASK = "direct_task"
    QUEUED_TASK = "queued_task"
    SERVER = "server"
    MCP = "mcp"
    A2A = "a2a"


class ContainerReviewMode(StrEnum):
    INTERACTIVE = "interactive"
    DURABLE = "durable"
    FAIL_CLOSED = "fail_closed"


class ContainerEscalationTrigger(StrEnum):
    WRITE_MOUNT = "write_mount"
    SECRET = "secret"
    NETWORK = "network"
    DEVICE = "device"
    ROOT_USER = "root_user"
    PRIVILEGED = "privileged"
    CAPABILITY = "capability"
    RUNTIME_SOCKET = "runtime_socket"
    HOST_NAMESPACE = "host_namespace"
    MUTABLE_IMAGE = "mutable_image"
    BUILD = "build"
    RESOURCE_INCREASE = "resource_increase"
    IMAGE_CHANGE = "image_change"
    PROFILE_CHANGE = "profile_change"
    COMBINED_RISK = "combined_risk"


_REVIEW_MODES = {
    ContainerReviewSurface.INTERACTIVE_CLI: ContainerReviewMode.INTERACTIVE,
    ContainerReviewSurface.STRICT_FLOW: ContainerReviewMode.DURABLE,
    ContainerReviewSurface.DIRECT_TASK: ContainerReviewMode.FAIL_CLOSED,
    ContainerReviewSurface.QUEUED_TASK: ContainerReviewMode.DURABLE,
    ContainerReviewSurface.SERVER: ContainerReviewMode.FAIL_CLOSED,
    ContainerReviewSurface.MCP: ContainerReviewMode.FAIL_CLOSED,
    ContainerReviewSurface.A2A: ContainerReviewMode.FAIL_CLOSED,
}

_ATTEMPT_BOUND_SURFACES = {
    ContainerReviewSurface.QUEUED_TASK,
}

_RUNTIME_SOCKET_SOURCES = {
    "/private/var/run/docker.sock",
    "/private/var/run/podman/podman.sock",
    "/run/docker.sock",
    "/run/podman/podman.sock",
    "/var/run/docker.sock",
    "/var/run/podman/podman.sock",
}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPolicyContext:
    surface: ContainerReviewSurface | str
    scope_id: str
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        surface = _enum_value(
            self.surface,
            ContainerReviewSurface,
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
    def review_mode(self) -> ContainerReviewMode:
        surface = cast(ContainerReviewSurface, self.surface)
        return _REVIEW_MODES[surface]

    def to_dict(self) -> dict[str, str | None]:
        surface = cast(ContainerReviewSurface, self.surface)
        return {
            "surface": surface.value,
            "scope_id": self.scope_id,
            "attempt_id": self.attempt_id,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPolicyPlan:
    effective_settings: ContainerEffectiveSettings
    context: ContainerPolicyContext
    command_fingerprint: str
    escalation_triggers: Sequence[ContainerEscalationTrigger | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.effective_settings, ContainerEffectiveSettings)
        assert isinstance(self.context, ContainerPolicyContext)
        _assert_non_empty_string(
            self.command_fingerprint,
            "command_fingerprint",
        )
        triggers = tuple(
            cast(
                ContainerEscalationTrigger,
                _enum_value(
                    trigger,
                    ContainerEscalationTrigger,
                    "escalation_triggers",
                ),
            )
            for trigger in self.escalation_triggers
        )
        assert len(set(triggers)) == len(
            triggers
        ), "escalation triggers must be unique"
        triggers = _normalized_triggers(
            set(triggers)
            | set(
                _required_escalation_triggers(
                    self.effective_settings,
                    self.context,
                )
            )
        )
        object.__setattr__(
            self,
            "escalation_triggers",
            triggers,
        )

    @property
    def plan_fingerprint(self) -> str:
        return _fingerprint(self.canonical_policy_input())

    @property
    def profile_name(self) -> str:
        assert (
            self.effective_settings.profile_name is not None
        ), "container policy plan requires a profile"
        return self.effective_settings.profile_name

    @property
    def policy_version(self) -> str:
        return self.effective_settings.policy_version

    def canonical_policy_input(self) -> dict[str, object]:
        surface = cast(ContainerReviewSurface, self.context.surface)
        return {
            "command_fingerprint": self.command_fingerprint,
            "escalation_triggers": [
                trigger.value
                for trigger in cast(
                    tuple[ContainerEscalationTrigger, ...],
                    self.escalation_triggers,
                )
            ],
            "review_mode": self.context.review_mode.value,
            "review_surface": surface.value,
            "settings": self.effective_settings.canonical_policy_input(),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerApprovalRecord:
    reviewer_identity: str
    review_mode: ContainerReviewMode | str
    plan_fingerprint: str
    scope_id: str
    attempt_id: str | None
    profile_name: str
    policy_version: str
    expires_at_seconds: int
    approved_triggers: Sequence[ContainerEscalationTrigger | str] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.reviewer_identity, "reviewer_identity")
        object.__setattr__(
            self,
            "review_mode",
            _enum_value(self.review_mode, ContainerReviewMode, "review_mode"),
        )
        _assert_non_empty_string(self.plan_fingerprint, "plan_fingerprint")
        _assert_non_empty_string(self.scope_id, "scope_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_non_empty_string(self.profile_name, "profile_name")
        _assert_non_empty_string(self.policy_version, "policy_version")
        _assert_positive_int(self.expires_at_seconds, "expires_at_seconds")
        triggers = tuple(
            _enum_value(
                trigger,
                ContainerEscalationTrigger,
                "approved_triggers",
            )
            for trigger in self.approved_triggers
        )
        assert len(set(triggers)) == len(
            triggers
        ), "approved triggers must be unique"
        object.__setattr__(
            self,
            "approved_triggers",
            tuple(sorted(triggers, key=lambda trigger: trigger.value)),
        )

    @classmethod
    def for_plan(
        cls,
        plan: ContainerPolicyPlan,
        *,
        reviewer_identity: str,
        expires_at_seconds: int,
    ) -> "ContainerApprovalRecord":
        assert (
            plan.context.review_mode is ContainerReviewMode.DURABLE
        ), "approval records require durable review surfaces"
        return cls(
            reviewer_identity=reviewer_identity,
            review_mode=plan.context.review_mode,
            plan_fingerprint=plan.plan_fingerprint,
            scope_id=plan.context.scope_id,
            attempt_id=plan.context.attempt_id,
            profile_name=plan.profile_name,
            policy_version=plan.policy_version,
            expires_at_seconds=expires_at_seconds,
            approved_triggers=plan.escalation_triggers,
        )

    def assert_applies_to(
        self,
        plan: ContainerPolicyPlan,
        *,
        now_seconds: int,
    ) -> None:
        _assert_positive_int(now_seconds + 1, "now_seconds")
        assert now_seconds < self.expires_at_seconds, "approval is stale"
        assert (
            self.review_mode is plan.context.review_mode
        ), "approval review mode does not match execution surface"
        assert (
            self.plan_fingerprint == plan.plan_fingerprint
        ), "approval plan fingerprint does not match"
        assert (
            self.scope_id == plan.context.scope_id
        ), "approval scope does not match"
        assert (
            self.attempt_id == plan.context.attempt_id
        ), "approval attempt does not match"
        assert (
            self.profile_name == plan.profile_name
        ), "approval profile does not match"
        assert (
            self.policy_version == plan.policy_version
        ), "approval policy version does not match"
        approved_triggers = set(self.approved_triggers)
        for trigger in plan.escalation_triggers:
            assert (
                trigger in approved_triggers
            ), "approval does not cover all escalation triggers"

    def to_dict(self) -> dict[str, object]:
        review_mode = cast(ContainerReviewMode, self.review_mode)
        triggers = cast(
            tuple[ContainerEscalationTrigger, ...],
            self.approved_triggers,
        )
        return {
            "reviewer_identity": self.reviewer_identity,
            "review_mode": review_mode.value,
            "plan_fingerprint": self.plan_fingerprint,
            "scope_id": self.scope_id,
            "attempt_id": self.attempt_id,
            "profile_name": self.profile_name,
            "policy_version": self.policy_version,
            "expires_at_seconds": self.expires_at_seconds,
            "approved_triggers": [trigger.value for trigger in triggers],
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerPolicy:
    policy_version: str
    explanations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.policy_version, "policy_version")
        object.__setattr__(
            self,
            "explanations",
            MappingProxyType(_string_mapping(self.explanations)),
        )

    def authorize(
        self,
        plan: ContainerPolicyPlan,
        *,
        approval: ContainerApprovalRecord | None = None,
        now_seconds: int = 0,
    ) -> ContainerAuthorizationDecision:
        assert isinstance(plan, ContainerPolicyPlan)
        assert (
            plan.policy_version == self.policy_version
        ), "plan policy version must match policy version"
        if approval is not None:
            assert isinstance(approval, ContainerApprovalRecord)
            assert (
                plan.context.review_mode is ContainerReviewMode.DURABLE
            ), "approval records require durable review surfaces"
            approval.assert_applies_to(plan, now_seconds=now_seconds)
            return self._decision(
                ContainerAuthorizationDecisionType.ALLOW,
                code="container.allow.cached_approval",
                explanation="Approved container plan matches cached review.",
                plan=plan,
                cacheable=True,
            )
        if not plan.escalation_triggers:
            return self._decision(
                ContainerAuthorizationDecisionType.ALLOW,
                code="container.allow.preauthorized_profile",
                explanation=(
                    "Trusted profile pre-authorizes this container plan."
                ),
                plan=plan,
                cacheable=True,
            )
        escalation_mode = _plan_escalation_mode(plan)
        if (
            escalation_mode is ContainerEscalationMode.PREAUTHORIZED
            and ContainerEscalationTrigger.COMBINED_RISK
            not in plan.escalation_triggers
        ):
            return self._decision(
                ContainerAuthorizationDecisionType.ALLOW,
                code="container.allow.preauthorized_escalation",
                explanation=(
                    "Trusted profile pre-authorizes escalation triggers."
                ),
                plan=plan,
                cacheable=True,
            )
        if escalation_mode is ContainerEscalationMode.DENY:
            return self._decision(
                ContainerAuthorizationDecisionType.DENY,
                code="container.deny.escalation",
                explanation="Container plan requires denied escalation.",
                plan=plan,
            )
        review_mode = plan.context.review_mode
        if review_mode is ContainerReviewMode.FAIL_CLOSED:
            return self._decision(
                ContainerAuthorizationDecisionType.DENY,
                code="container.deny.review_unavailable",
                explanation=(
                    "Execution surface cannot perform required review."
                ),
                plan=plan,
            )
        return self._decision(
            ContainerAuthorizationDecisionType.REQUIRES_REVIEW,
            code=f"container.review.{review_mode.value}",
            explanation="Container plan requires operator review.",
            plan=plan,
            retryable=True,
        )

    def _decision(
        self,
        decision: ContainerAuthorizationDecisionType,
        *,
        code: str,
        explanation: str,
        plan: ContainerPolicyPlan,
        retryable: bool = False,
        cacheable: bool = False,
    ) -> ContainerAuthorizationDecision:
        return ContainerAuthorizationDecision(
            decision=decision,
            code=code,
            explanation=self.explanations.get(code, explanation),
            policy_version=self.policy_version,
            profile_name=plan.profile_name,
            retryable=retryable,
            cacheable=cacheable,
        )


def _plan_escalation_mode(
    plan: ContainerPolicyPlan,
) -> ContainerEscalationMode:
    profile = plan.effective_settings.profile
    assert profile is not None, "container policy plan requires a profile"
    return cast(ContainerEscalationMode, profile.escalation.mode)


def _required_escalation_triggers(
    settings: ContainerEffectiveSettings,
    context: ContainerPolicyContext,
) -> tuple[ContainerEscalationTrigger, ...]:
    profile = settings.profile
    assert profile is not None, "container policy plan requires a profile"
    triggers: set[ContainerEscalationTrigger] = set()
    for mount in profile.mounts:
        access = cast(ContainerMountAccess, mount.access)
        mount_type = cast(ContainerMountType, mount.mount_type)
        if access is ContainerMountAccess.WRITE:
            triggers.add(ContainerEscalationTrigger.WRITE_MOUNT)
        if mount_type is ContainerMountType.SECRET:
            triggers.add(ContainerEscalationTrigger.SECRET)
        if _is_runtime_socket_source(mount.source):
            triggers.add(ContainerEscalationTrigger.RUNTIME_SOCKET)
    if profile.secrets:
        triggers.add(ContainerEscalationTrigger.SECRET)
    network_mode = cast(ContainerNetworkMode, profile.network.mode)
    if (
        network_mode is not ContainerNetworkMode.NONE
        or profile.network.egress_allowlist
    ):
        triggers.add(ContainerEscalationTrigger.NETWORK)
    if profile.devices.devices:
        triggers.add(ContainerEscalationTrigger.DEVICE)
    if profile.user == "0" or profile.user.startswith("0:"):
        triggers.add(ContainerEscalationTrigger.ROOT_USER)
    build_policy = cast(ContainerBuildPolicy, profile.image.build_policy)
    if build_policy is not ContainerBuildPolicy.DISABLED:
        triggers.add(ContainerEscalationTrigger.BUILD)
    if "@sha256:" not in profile.image.reference:
        triggers.add(ContainerEscalationTrigger.MUTABLE_IMAGE)
        if context.review_mode is not ContainerReviewMode.INTERACTIVE:
            triggers.add(ContainerEscalationTrigger.COMBINED_RISK)
    if _resource_limits_requested(settings):
        triggers.add(ContainerEscalationTrigger.RESOURCE_INCREASE)
    return _normalized_triggers(triggers)


def _normalized_triggers(
    triggers: set[ContainerEscalationTrigger],
) -> tuple[ContainerEscalationTrigger, ...]:
    base_triggers = triggers - {ContainerEscalationTrigger.COMBINED_RISK}
    if len(base_triggers) > 1:
        triggers.add(ContainerEscalationTrigger.COMBINED_RISK)
    return tuple(sorted(triggers, key=lambda trigger: trigger.value))


def _resource_limits_requested(settings: ContainerEffectiveSettings) -> bool:
    assert settings.profile is not None
    resources = settings.profile.resources
    return any(
        limit is not None
        for limit in (
            resources.cpu_count,
            resources.memory_bytes,
            resources.pids,
            resources.timeout_seconds,
        )
    )


def _is_runtime_socket_source(source: str | None) -> bool:
    if source is None:
        return False
    return normalize_posix_path(source) in _RUNTIME_SOCKET_SOURCES


def _fingerprint(value: Mapping[str, object]) -> str:
    payload = dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _string_mapping(value: object) -> dict[str, str]:
    assert isinstance(value, Mapping), "explanations must be a mapping"
    result: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, "explanations key")
        _assert_non_empty_string(item, "explanations value")
        assert isinstance(key, str)
        assert isinstance(item, str)
        result[key] = item
    return result


def _enum_value(
    value: object,
    enum_type: type[StrEnum],
    field_name: str,
) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert value in {
        member.value for member in enum_type
    }, f"{field_name} contains unsupported value"
    return enum_type(value)
