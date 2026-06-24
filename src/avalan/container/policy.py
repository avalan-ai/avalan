from ..types import (
    assert_bool as _assert_bool,
)
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
    ContainerPoolingMode,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from json import dumps
from posixpath import normpath as normalize_posix_path
from re import compile as compile_pattern
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
    POOLING = "pooling"


class ContainerVulnerabilityPolicy(StrEnum):
    ALLOW = "allow"
    DENY_CRITICAL = "deny_critical"
    DENY_HIGH_OR_CRITICAL = "deny_high_or_critical"


class ContainerVulnerabilitySeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_DIGEST_PATTERN = compile_pattern(r"^sha256:[0-9a-f]{64}$")
_PLATFORM_PATTERN = compile_pattern(r"^[A-Za-z0-9_+.-]+/[A-Za-z0-9_+.-]+$")
_VULNERABILITY_RANK = {
    ContainerVulnerabilitySeverity.NONE: 0,
    ContainerVulnerabilitySeverity.LOW: 1,
    ContainerVulnerabilitySeverity.MEDIUM: 2,
    ContainerVulnerabilitySeverity.HIGH: 3,
    ContainerVulnerabilitySeverity.CRITICAL: 4,
}


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
class ContainerImageTrustPolicy:
    allowed_registries: Sequence[str] = field(default_factory=tuple)
    digest_pins: Mapping[str, str] = field(default_factory=dict)
    require_signature_verification: bool = False
    verified_digests: Sequence[str] = field(default_factory=tuple)
    require_attestations: bool = False
    attestation_references: Mapping[str, str] = field(default_factory=dict)
    require_sbom: bool = False
    sbom_references: Mapping[str, str] = field(default_factory=dict)
    vulnerability_policy: ContainerVulnerabilityPolicy | str = (
        ContainerVulnerabilityPolicy.ALLOW
    )
    vulnerability_findings: Mapping[
        str,
        ContainerVulnerabilitySeverity | str,
    ] = field(default_factory=dict)
    allowed_platforms: Sequence[str] = field(default_factory=tuple)
    production: bool = False
    require_production_digest_pinning: bool = True
    require_noninteractive_digest_pinning: bool = True

    def __post_init__(self) -> None:
        allowed_registries = _string_tuple(
            self.allowed_registries,
            "allowed_registries",
        )
        digest_pins = _digest_mapping(self.digest_pins, "digest_pins")
        verified_digests = _digest_tuple(
            self.verified_digests,
            "verified_digests",
        )
        attestation_references = _digest_string_mapping(
            self.attestation_references,
            "attestation_references",
        )
        sbom_references = _digest_string_mapping(
            self.sbom_references,
            "sbom_references",
        )
        vulnerability_findings = _vulnerability_mapping(
            self.vulnerability_findings,
        )
        allowed_platforms = _platform_tuple(
            self.allowed_platforms,
            "allowed_platforms",
        )
        object.__setattr__(
            self,
            "vulnerability_policy",
            _enum_value(
                self.vulnerability_policy,
                ContainerVulnerabilityPolicy,
                "vulnerability_policy",
            ),
        )
        for field_name in (
            "require_signature_verification",
            "require_attestations",
            "require_sbom",
            "production",
            "require_production_digest_pinning",
            "require_noninteractive_digest_pinning",
        ):
            _assert_bool(getattr(self, field_name), field_name)
        object.__setattr__(self, "allowed_registries", allowed_registries)
        object.__setattr__(
            self,
            "digest_pins",
            MappingProxyType(digest_pins),
        )
        object.__setattr__(self, "verified_digests", verified_digests)
        object.__setattr__(
            self,
            "attestation_references",
            MappingProxyType(attestation_references),
        )
        object.__setattr__(
            self,
            "sbom_references",
            MappingProxyType(sbom_references),
        )
        object.__setattr__(
            self,
            "vulnerability_findings",
            MappingProxyType(vulnerability_findings),
        )
        object.__setattr__(self, "allowed_platforms", allowed_platforms)

    def to_dict(self) -> dict[str, object]:
        vulnerability_policy = cast(
            ContainerVulnerabilityPolicy,
            self.vulnerability_policy,
        )
        vulnerability_findings = cast(
            Mapping[str, ContainerVulnerabilitySeverity],
            self.vulnerability_findings,
        )
        return {
            "allowed_registries": list(self.allowed_registries),
            "digest_pins": dict(self.digest_pins),
            "require_signature_verification": (
                self.require_signature_verification
            ),
            "verified_digests": list(self.verified_digests),
            "require_attestations": self.require_attestations,
            "attestation_references": dict(self.attestation_references),
            "require_sbom": self.require_sbom,
            "sbom_references": dict(self.sbom_references),
            "vulnerability_policy": vulnerability_policy.value,
            "vulnerability_findings": {
                digest: severity.value
                for digest, severity in vulnerability_findings.items()
            },
            "allowed_platforms": list(self.allowed_platforms),
            "production": self.production,
            "require_production_digest_pinning": (
                self.require_production_digest_pinning
            ),
            "require_noninteractive_digest_pinning": (
                self.require_noninteractive_digest_pinning
            ),
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
    image_trust: ContainerImageTrustPolicy = field(
        default_factory=ContainerImageTrustPolicy,
    )
    explanations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.policy_version, "policy_version")
        assert isinstance(self.image_trust, ContainerImageTrustPolicy)
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
        image_trust_denial = _image_trust_denial(self.image_trust, plan)
        if image_trust_denial is not None:
            code, explanation = image_trust_denial
            return self._decision(
                ContainerAuthorizationDecisionType.DENY,
                code=code,
                explanation=explanation,
                plan=plan,
            )
        if _secret_pool_reuse_denied(plan):
            return self._decision(
                ContainerAuthorizationDecisionType.DENY,
                code="container.deny.secret_reuse",
                explanation=(
                    "Container pool reuse with secrets is not authorized."
                ),
                plan=plan,
            )
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
        if escalation_mode is ContainerEscalationMode.PREAUTHORIZED and (
            ContainerEscalationTrigger.COMBINED_RISK
            not in plan.escalation_triggers
            or _secret_pool_combined_risk_authorized(plan)
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


def _image_trust_denial(
    trust: ContainerImageTrustPolicy,
    plan: ContainerPolicyPlan,
) -> tuple[str, str] | None:
    assert isinstance(trust, ContainerImageTrustPolicy)
    profile = plan.effective_settings.profile
    assert profile is not None, "container policy plan requires a profile"
    image = profile.image
    assert image.digest is not None
    reference_is_mutable = "@sha256:" not in image.reference
    if (
        trust.production
        and trust.require_production_digest_pinning
        and reference_is_mutable
    ):
        return (
            "container.deny.image_trust",
            "Production container images must use digest-pinned references.",
        )
    if (
        trust.require_noninteractive_digest_pinning
        and plan.context.review_mode is not ContainerReviewMode.INTERACTIVE
        and reference_is_mutable
    ):
        return (
            "container.deny.image_trust",
            (
                "Noninteractive container images must use digest-pinned"
                " references."
            ),
        )
    registry = _image_registry(image.reference)
    if trust.allowed_registries and registry not in trust.allowed_registries:
        return (
            "container.deny.image_trust",
            "Container image registry is not trusted by policy.",
        )
    pinned_digest = _pinned_digest_for_reference(
        trust.digest_pins,
        image.reference,
    )
    if pinned_digest is not None and image.digest != pinned_digest:
        return (
            "container.deny.image_trust",
            "Container image digest does not match trusted pin.",
        )
    if (
        trust.allowed_platforms
        and image.platform not in trust.allowed_platforms
    ):
        return (
            "container.deny.image_trust",
            "Container image platform is not trusted by policy.",
        )
    if (
        trust.require_signature_verification
        and image.digest not in trust.verified_digests
    ):
        return (
            "container.deny.image_trust",
            "Container image signature verification failed.",
        )
    if (
        trust.require_attestations
        and image.digest not in trust.attestation_references
    ):
        return (
            "container.deny.image_trust",
            "Container image attestation is missing.",
        )
    if trust.require_sbom and image.digest not in trust.sbom_references:
        return (
            "container.deny.image_trust",
            "Container image SBOM reference is missing.",
        )
    vulnerability_denial = _vulnerability_denial(trust, image.digest)
    if vulnerability_denial is not None:
        return (
            "container.deny.image_trust",
            vulnerability_denial,
        )
    return None


def _secret_pool_reuse_denied(plan: ContainerPolicyPlan) -> bool:
    profile = plan.effective_settings.profile
    assert profile is not None, "container policy plan requires a profile"
    pool_mode = cast(ContainerPoolingMode, profile.pooling.mode)
    if pool_mode is ContainerPoolingMode.DISABLED:
        return False
    if profile.pooling.allow_secret_reuse:
        return False
    return ContainerEscalationTrigger.SECRET in plan.escalation_triggers


def _secret_pool_combined_risk_authorized(
    plan: ContainerPolicyPlan,
) -> bool:
    profile = plan.effective_settings.profile
    assert profile is not None, "container policy plan requires a profile"
    if not profile.pooling.allow_secret_reuse:
        return False
    base_triggers = set(plan.escalation_triggers) - {
        ContainerEscalationTrigger.COMBINED_RISK,
    }
    return base_triggers == {
        ContainerEscalationTrigger.POOLING,
        ContainerEscalationTrigger.SECRET,
    }


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
    pool_mode = cast(ContainerPoolingMode, profile.pooling.mode)
    if pool_mode is not ContainerPoolingMode.DISABLED:
        triggers.add(ContainerEscalationTrigger.POOLING)
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


def _image_registry(reference: str) -> str:
    name = reference.split("@", maxsplit=1)[0]
    first_component = name.split("/", maxsplit=1)[0]
    if (
        "." in first_component
        or ":" in first_component
        or first_component == "localhost"
    ):
        return first_component
    return "docker.io"


def _pinned_digest_for_reference(
    digest_pins: Mapping[str, str],
    reference: str,
) -> str | None:
    for key in _reference_pin_keys(reference):
        pinned_digest = digest_pins.get(key)
        if pinned_digest is not None:
            return pinned_digest
    return None


def _reference_pin_keys(reference: str) -> tuple[str, ...]:
    without_digest = reference.split("@", maxsplit=1)[0]
    without_tag = without_digest.rsplit(":", maxsplit=1)[0]
    if without_tag == without_digest:
        return (reference, without_digest)
    return (reference, without_digest, without_tag)


def _vulnerability_denial(
    trust: ContainerImageTrustPolicy,
    digest: str,
) -> str | None:
    policy = cast(
        ContainerVulnerabilityPolicy,
        trust.vulnerability_policy,
    )
    if policy is ContainerVulnerabilityPolicy.ALLOW:
        return None
    vulnerability_findings = cast(
        Mapping[str, ContainerVulnerabilitySeverity],
        trust.vulnerability_findings,
    )
    severity = vulnerability_findings.get(digest)
    if severity is None:
        return "Container image vulnerability scan is missing."
    threshold = (
        ContainerVulnerabilitySeverity.CRITICAL
        if policy is ContainerVulnerabilityPolicy.DENY_CRITICAL
        else ContainerVulnerabilitySeverity.HIGH
    )
    if _VULNERABILITY_RANK[severity] >= _VULNERABILITY_RANK[threshold]:
        return "Container image vulnerability policy denied execution."
    return None


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


def _digest_mapping(value: object, field_name: str) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    result: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        _assert_digest(item, field_name)
        assert isinstance(key, str)
        assert isinstance(item, str)
        result[key] = item
    return result


def _digest_string_mapping(value: object, field_name: str) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    result: dict[str, str] = {}
    for key, item in value.items():
        _assert_digest(key, f"{field_name} key")
        _assert_non_empty_string(item, field_name)
        assert isinstance(key, str)
        assert isinstance(item, str)
        result[key] = item
    return result


def _vulnerability_mapping(
    value: object,
) -> dict[str, ContainerVulnerabilitySeverity]:
    assert isinstance(
        value, Mapping
    ), "vulnerability_findings must be a mapping"
    result: dict[str, ContainerVulnerabilitySeverity] = {}
    for key, item in value.items():
        _assert_digest(key, "vulnerability_findings key")
        assert isinstance(key, str)
        result[key] = cast(
            ContainerVulnerabilitySeverity,
            _enum_value(
                item,
                ContainerVulnerabilitySeverity,
                "vulnerability_findings",
            ),
        )
    return result


def _digest_tuple(value: object, field_name: str) -> tuple[str, ...]:
    sequence = _string_tuple(value, field_name)
    for item in sequence:
        _assert_digest(item, field_name)
    return sequence


def _platform_tuple(value: object, field_name: str) -> tuple[str, ...]:
    sequence = _string_tuple(value, field_name)
    for item in sequence:
        assert _PLATFORM_PATTERN.match(
            item
        ), f"{field_name} must contain os/architecture platforms"
    return sequence


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


def _assert_digest(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert _DIGEST_PATTERN.match(value), f"{field_name} must be sha256 digest"


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
