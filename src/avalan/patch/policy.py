"""Authorize, seal, review, and approve dormant patch plans.

This module is deliberately isolated from workspace configuration and target
mutation.  It turns trusted immutable policy, target observations, and a
planner candidate into a sealed plan that Phase 6 may later revalidate and
commit.  No value in this module grants a write by itself.
"""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from hmac import compare_digest
from secrets import token_bytes
from typing import Protocol

from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ByteSize,
    Capability,
    ContextKind,
    DiffBytes,
    DurationTicks,
    ExpiryTick,
    LogicalPath,
    MetadataProfile,
    OperationType,
    PatchApprovalId,
    PatchFingerprint,
    PatchGrantId,
    PatchLimits,
    PatchPlanId,
    PatchRequest,
    PatchValidationError,
    ReviewArtifact,
)
from avalan.patch.planner import (
    Match,
    PlannedFile,
    PlannedLineage,
    PlannerCandidate,
)
from avalan.patch.target import TargetHandshake, TargetIdentity

_FINGERPRINT_DOMAIN = b"avalan.patch.sealed-plan.v1\0"
_PLAN_SCHEMA_VERSION = 1
_GRAMMAR_VERSION = 1
_SEMANTIC_MODEL_VERSION = 1


class PolicyErrorCode(str, Enum):
    """Name stable fail-closed policy outcomes."""

    DENIED = "patch.capability_required"
    PATH_DENIED = "patch.path_denied"
    LIMIT_EXCEEDED = "patch.limit_exceeded"
    APPROVAL_UNAVAILABLE = "patch.approval_unavailable"
    APPROVAL_DENIED = "patch.approval_denied"
    APPROVAL_MISMATCH = "patch.approval_mismatch"
    APPROVAL_EXPIRED = "patch.approval_expired"
    INVALID_PLAN = "patch.invalid_request"


class PolicyError(PatchValidationError):
    """Report a closed policy or approval outcome."""

    def __init__(self, code: PolicyErrorCode) -> None:
        """Initialize the stable policy error code."""
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class PolicyRevision:
    """Identify one trusted immutable policy revision."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized revision labels."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PolicyRouteId:
    """Identify one trusted approval route."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized route identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PolicyBrokerId:
    """Identify one trusted asynchronous approval broker."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized broker identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PolicyReviewerRole:
    """Identify one bounded reviewer role selected by policy."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized role identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PreauthorizationClass:
    """Identify one bounded trusted preauthorization class."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized class identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class DiagnosticPolicyId:
    """Identify one separately authorized diagnostic-association policy."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized diagnostic policy identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PatchPrincipalId:
    """Identify one authenticated requesting or reviewing principal."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized principal identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PatchTenantId:
    """Identify one authenticated tenant."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized tenant identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PatchRunId:
    """Identify one authenticated runtime run."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized run identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PatchSessionId:
    """Identify one authenticated runtime session."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized session identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PatchTaskId:
    """Identify one authenticated runtime task."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized task identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PatchAgentId:
    """Identify one authenticated requesting agent."""

    value: str

    def __post_init__(self) -> None:
        """Reject blank or oversized agent identifiers."""
        if not self.value or len(self.value) > 128:
            raise PolicyError(PolicyErrorCode.DENIED)


class PolicyDisclosure(str, Enum):
    """Name independently controlled patch disclosure surfaces."""

    MODEL_DIFF = "model_diff"
    MODEL_METADATA = "model_metadata"
    MODEL_MATCH_DETAILS = "model_match_details"
    COMPLETE_REVIEW = "complete_review"
    AUDIT_PATHS = "audit_paths"
    EVENT_METRICS = "event_metrics"
    DIAGNOSTIC_ASSOCIATION = "diagnostic_association"
    SDK_HOST = "sdk_host"


class PathVisibility(str, Enum):
    """Name the closed path classes evaluated before inspection."""

    ORDINARY = "ordinary"
    HIDDEN = "hidden"
    VERSION_CONTROL = "version_control"


@dataclass(frozen=True, slots=True)
class PolicyPathSelector:
    """Select one policy-owned subtree without path-pattern execution."""

    prefix: LogicalPath | None
    include_hidden: bool = False

    def matches(self, path: LogicalPath) -> bool:
        """Return whether one logical path is in the selected subtree."""
        if _path_visibility(path) is PathVisibility.VERSION_CONTROL:
            return False
        if (
            _path_visibility(path) is PathVisibility.HIDDEN
            and not self.include_hidden
        ):
            return False
        if self.prefix is None:
            return True
        return path.value == self.prefix.value or path.value.startswith(
            self.prefix.value + "/"
        )

    def specificity(self) -> tuple[int, int]:
        """Return a deterministic selector precedence without path access."""
        return (
            0 if self.prefix is None else len(self.prefix.value),
            1 if self.include_hidden else 0,
        )


@dataclass(frozen=True, slots=True)
class CapabilityMode:
    """Bind one independent effect or inspection capability to a mode."""

    value: Capability
    mode: ApprovalMode
    preauthorization: PreauthorizationClass | None = None

    def __post_init__(self) -> None:
        """Require an exact bounded class for preauthorized effects."""
        if (
            self.mode is ApprovalMode.PREAUTHORIZED
            and self.preauthorization is None
        ) or (
            self.mode is not ApprovalMode.PREAUTHORIZED
            and self.preauthorization is not None
        ):
            raise PolicyError(PolicyErrorCode.DENIED)
        if (
            self.value
            in {
                Capability.READ_FOR_MUTATION,
                Capability.OBSERVE_MUTATION_PRECONDITIONS,
            }
            and self.mode is ApprovalMode.REQUIRE_REVIEW
        ):
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """Store one trusted bounded path rule without configuration loading."""

    selector: PolicyPathSelector
    modes: tuple[CapabilityMode, ...]
    disclosures: frozenset[PolicyDisclosure] = frozenset()
    atomicity_classes: frozenset[str] = frozenset(("single_step",))
    staging_classes: frozenset[str] = frozenset(("target_private",))

    def __post_init__(self) -> None:
        """Require immutable unique capability and disclosure declarations."""
        if (
            type(self.modes) is not tuple
            or type(self.disclosures) is not frozenset
            or type(self.atomicity_classes) is not frozenset
            or type(self.staging_classes) is not frozenset
            or any(type(item) is not CapabilityMode for item in self.modes)
            or any(
                type(item) is not PolicyDisclosure for item in self.disclosures
            )
            or any(
                type(item) is not str or not item
                for item in self.atomicity_classes
            )
            or any(
                type(item) is not str or not item
                for item in self.staging_classes
            )
            or not self.atomicity_classes
            or not self.staging_classes
            or len({item.value for item in self.modes}) != len(self.modes)
        ):
            raise PolicyError(PolicyErrorCode.DENIED)

    def mode_for(self, value: Capability) -> CapabilityMode | None:
        """Return the explicitly configured mode for one capability."""
        for item in self.modes:
            if item.value is value:
                return item
        return None


_DEFAULT_LIMITS = PatchLimits(
    input_bytes=ByteSize(1_048_576),
    path_count=ByteSize(1_024),
    path_length=ByteSize(1_024),
    file_count=ByteSize(1_024),
    operation_count=ByteSize(1_024),
    snapshot_bytes=ByteSize(4_194_304),
    proposed_bytes=ByteSize(4_194_304),
    review_diff_bytes=ByteSize(1_048_576),
    planning_duration=DurationTicks(60_000),
    approval_duration=DurationTicks(3_600_000),
    commit_duration=DurationTicks(60_000),
)


@dataclass(frozen=True, slots=True)
class ApprovalRequirements:
    """Bind the policy-owned approval route to one sealed plan."""

    mode: ApprovalMode
    route: PolicyRouteId
    broker: PolicyBrokerId
    reviewer_role: PolicyReviewerRole
    quorum: int
    preauthorization: PreauthorizationClass | None = None

    def __post_init__(self) -> None:
        """Require a positive quorum and mode-consistent bounded class."""
        if (
            type(self.mode) is not ApprovalMode
            or type(self.route) is not PolicyRouteId
            or type(self.broker) is not PolicyBrokerId
            or type(self.reviewer_role) is not PolicyReviewerRole
            or type(self.quorum) is not int
            or not 1 <= self.quorum <= 64
            or (
                self.mode is ApprovalMode.PREAUTHORIZED
                and self.preauthorization is None
            )
            or (
                self.mode is not ApprovalMode.PREAUTHORIZED
                and self.preauthorization is not None
            )
        ):
            raise PolicyError(PolicyErrorCode.DENIED)


def _default_approval_requirements() -> ApprovalRequirements:
    """Return the policy-owned default that cannot authorize a plan."""
    return ApprovalRequirements(
        ApprovalMode.DENY,
        PolicyRouteId("default-deny-route"),
        PolicyBrokerId("default-deny-broker"),
        PolicyReviewerRole("default-deny-reviewer"),
        1,
    )


@dataclass(frozen=True, slots=True)
class TrustedPatchPolicy:
    """Store default-deny values detached from workspace configuration."""

    revision: PolicyRevision
    enabled_operations: frozenset[OperationType] = frozenset()
    rules: tuple[PolicyRule, ...] = ()
    limits: PatchLimits = _DEFAULT_LIMITS
    approval: ApprovalRequirements = field(
        default_factory=_default_approval_requirements
    )

    def __post_init__(self) -> None:
        """Reject mutable, duplicate, or untyped trusted policy values."""
        selectors = tuple(item.selector for item in self.rules)
        if (
            type(self.enabled_operations) is not frozenset
            or type(self.rules) is not tuple
            or any(
                type(item) is not OperationType
                for item in self.enabled_operations
            )
            or any(type(item) is not PolicyRule for item in self.rules)
            or len(set(selectors)) != len(selectors)
            or not isinstance(self.limits, PatchLimits)
            or type(self.approval) is not ApprovalRequirements
        ):
            raise PolicyError(PolicyErrorCode.DENIED)

    @classmethod
    def empty(cls) -> "TrustedPatchPolicy":
        """Return the immutable default policy with no patch authority."""
        return cls(PolicyRevision("default-deny"))

    def rule_for(self, path: LogicalPath) -> PolicyRule | None:
        """Return the most-specific configured rule for one safe path."""
        if _path_visibility(path) is PathVisibility.VERSION_CONTROL:
            return None
        candidates = tuple(
            item for item in self.rules if item.selector.matches(path)
        )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.selector.specificity())


@dataclass(frozen=True, slots=True)
class EffectiveLimits:
    """Bind the strictest limits selected by every trusted boundary."""

    value: PatchLimits


def compose_limits(
    provider: PatchLimits,
    manager: PatchLimits,
    policy: PatchLimits,
    context: PatchLimits,
    target: PatchLimits,
) -> EffectiveLimits:
    """Return the component-wise strictest trusted finite limits."""
    values = (provider, manager, policy, context, target)
    return EffectiveLimits(
        PatchLimits(
            input_bytes=ByteSize(
                min(item.input_bytes.value for item in values)
            ),
            path_count=ByteSize(min(item.path_count.value for item in values)),
            path_length=ByteSize(
                min(item.path_length.value for item in values)
            ),
            file_count=ByteSize(min(item.file_count.value for item in values)),
            operation_count=ByteSize(
                min(item.operation_count.value for item in values)
            ),
            snapshot_bytes=ByteSize(
                min(item.snapshot_bytes.value for item in values)
            ),
            proposed_bytes=ByteSize(
                min(item.proposed_bytes.value for item in values)
            ),
            review_diff_bytes=ByteSize(
                min(item.review_diff_bytes.value for item in values)
            ),
            planning_duration=DurationTicks(
                min(item.planning_duration.value for item in values)
            ),
            approval_duration=DurationTicks(
                min(item.approval_duration.value for item in values)
            ),
            commit_duration=DurationTicks(
                min(item.commit_duration.value for item in values)
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class PreflightRequest:
    """Describe the conservative request upper bound before target calls."""

    operation: OperationType
    paths: tuple[LogicalPath, ...]
    external_effects: frozenset[Capability]
    external_read_paths: frozenset[LogicalPath]
    effective_limits: EffectiveLimits

    def __post_init__(self) -> None:
        """Reject untyped, duplicate, or unbounded preflight declarations."""
        if (
            type(self.paths) is not tuple
            or type(self.external_effects) is not frozenset
            or type(self.external_read_paths) is not frozenset
            or any(type(item) is not LogicalPath for item in self.paths)
            or any(
                type(item) is not Capability for item in self.external_effects
            )
            or any(
                type(item) is not LogicalPath
                for item in self.external_read_paths
            )
            or len(set(self.paths)) != len(self.paths)
            or not self.paths
            or not self.external_read_paths.issubset(frozenset(self.paths))
            or len(self.paths) > self.effective_limits.value.path_count.value
        ):
            raise PolicyError(PolicyErrorCode.DENIED)


@dataclass(frozen=True, slots=True)
class PreflightAuthorization:
    """Record an inspection-safe authority result without target handles."""

    revision: PolicyRevision
    paths: tuple[LogicalPath, ...]
    effects: frozenset[Capability]
    effective_limits: EffectiveLimits


@dataclass(frozen=True, slots=True)
class FinalAuthorization:
    """Record exact final-effect policy without a commit capability."""

    revision: PolicyRevision
    handshake: TargetHandshake
    effects: frozenset[Capability]
    disclosures: frozenset[PolicyDisclosure]
    effective_limits: EffectiveLimits
    approval: ApprovalRequirements


@dataclass(frozen=True, slots=True)
class ExecutionSubject:
    """Bind authenticated caller identities required by review and grants."""

    principal: PatchPrincipalId
    tenant: PatchTenantId
    run: PatchRunId
    session: PatchSessionId
    task: PatchTaskId
    agent: PatchAgentId


@dataclass(frozen=True, slots=True)
class PlanBinding:
    """Bind non-content plan inputs required by the fingerprint contract."""

    request: PatchRequest
    request_digest: AlgorithmDigest
    subject: ExecutionSubject
    context_kind: ContextKind
    target: TargetIdentity
    cwd: LogicalPath | None
    preflight: PreflightAuthorization
    final: FinalAuthorization
    diagnostic_policy: DiagnosticPolicyId | None = None

    def __post_init__(self) -> None:
        """Reject mismatched scope identities before a plan can be sealed."""
        if (
            self.preflight.revision != self.final.revision
            or self.target.policy_revision != self.final.revision.value
            or not _target_matches_handshake(
                self.target, self.final.handshake.identity
            )
        ):
            raise PolicyError(PolicyErrorCode.INVALID_PLAN)


@dataclass(frozen=True, slots=True)
class ReviewRegion:
    """Describe one exact selected source region without source content."""

    logical_start: int
    logical_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class CapabilityWarning:
    """Describe an elevated final effect for a trusted reviewer artifact."""

    value: Capability


@dataclass(frozen=True, slots=True)
class ReviewLineage:
    """Project one complete resolved lineage into the reviewer artifact."""

    lineage_id: str
    source_path: LogicalPath | None
    destination_path: LogicalPath | None
    effects: frozenset[Capability]
    regions: tuple[ReviewRegion, ...]
    atomicity: str
    staging: str


@dataclass(frozen=True, slots=True)
class CompleteReviewArtifact:
    """Store a full detached review artifact with no truncation state."""

    lineages: tuple[ReviewLineage, ...]
    warnings: tuple[CapabilityWarning, ...]
    diff: ReviewArtifact
    expiry: ExpiryTick
    fingerprint: PatchFingerprint

    def __post_init__(self) -> None:
        """Require exactly one complete lineage record per sealed candidate."""
        if (
            type(self.lineages) is not tuple
            or type(self.warnings) is not tuple
            or any(type(item) is not ReviewLineage for item in self.lineages)
            or any(
                type(item) is not CapabilityWarning for item in self.warnings
            )
            or not self.lineages
        ):
            raise PolicyError(PolicyErrorCode.INVALID_PLAN)


@dataclass(frozen=True, slots=True)
class SealedPlan:
    """Store a fingerprinted immutable candidate owned by trusted runtime."""

    plan_id: PatchPlanId
    binding: PlanBinding
    candidate: PlannerCandidate
    review: CompleteReviewArtifact
    fingerprint: PatchFingerprint

    def __post_init__(self) -> None:
        """Require review, candidate, and fingerprint integrity agreement."""
        if (
            self.review.fingerprint != self.fingerprint
            or self.review.diff.digest != self.candidate.diff.digest
            or self.review.diff.size.value != len(self.candidate.diff.rendered)
        ):
            raise PolicyError(PolicyErrorCode.INVALID_PLAN)


@dataclass(frozen=True, slots=True)
class PrivateArtifactRetention:
    """Bound private Phase 5 authority records before commit ownership."""

    max_records: int

    def __post_init__(self) -> None:
        """Require one finite positive private-record retention limit."""
        if (
            type(self.max_records) is not int
            or not 1 <= self.max_records <= 4096
        ):
            raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)


_DEFAULT_PRIVATE_ARTIFACT_RETENTION = PrivateArtifactRetention(256)


@dataclass(frozen=True, slots=True)
class _SealedAuthority:
    """Keep runtime-owned seal facts outside caller-controlled plan objects."""

    plan: SealedPlan
    plan_id: PatchPlanId
    canonical: bytes
    fingerprint: bytes


_SEALED_AUTHORITIES: dict[int, _SealedAuthority] = {}


def cleanup_sealed_authorities(now: ExpiryTick) -> int:
    """Remove expired seal records in deterministic expiry and plan order."""
    expired = tuple(
        identity
        for identity, authority in sorted(
            _SEALED_AUTHORITIES.items(),
            key=lambda item: (
                item[1].plan.review.expiry.value,
                item[1].plan.plan_id.value,
                item[0],
            ),
        )
        if authority.plan.review.expiry.value <= now.value
    )
    for identity in expired:
        del _SEALED_AUTHORITIES[identity]
    return len(expired)


def seal_plan(
    plan_id: PatchPlanId,
    binding: PlanBinding,
    candidate: PlannerCandidate,
    expiry: ExpiryTick,
    retention: PrivateArtifactRetention = _DEFAULT_PRIVATE_ARTIFACT_RETENTION,
) -> SealedPlan:
    """Seal one fully planned candidate with a domain-separated fingerprint."""
    if type(retention) is not PrivateArtifactRetention:
        raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)
    _validate_candidate_integrity(candidate)
    _validate_candidate_limits(candidate, binding.final.effective_limits.value)
    if len(_SEALED_AUTHORITIES) >= retention.max_records:
        raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)
    review_value = ReviewArtifact(
        diff=DiffBytes(candidate.diff.rendered),
        digest=candidate.diff.digest,
        size=ByteSize(len(candidate.diff.rendered)),
    )
    fingerprint = PatchFingerprint(
        sha256(
            _canonical_fingerprint_bytes(binding, candidate, expiry)
        ).digest()
    )
    sealed = SealedPlan(
        plan_id=plan_id,
        binding=binding,
        candidate=candidate,
        review=CompleteReviewArtifact(
            lineages=tuple(
                _review_lineage(item) for item in candidate.lineages
            ),
            warnings=tuple(
                CapabilityWarning(item)
                for item in sorted(
                    binding.final.effects,
                    key=lambda value: value.value,
                )
                if item
                in {
                    Capability.DELETE,
                    Capability.MOVE,
                    Capability.UPDATE_EXECUTABLE,
                }
            ),
            diff=review_value,
            expiry=expiry,
            fingerprint=fingerprint,
        ),
        fingerprint=fingerprint,
    )
    _SEALED_AUTHORITIES[id(sealed)] = _SealedAuthority(
        sealed,
        plan_id,
        _canonical_fingerprint_bytes(binding, candidate, expiry),
        fingerprint._value,
    )
    return sealed


def _validate_sealed_plan(plan: SealedPlan) -> None:
    """Reject caller-side mutation of any fact covered by a runtime seal."""
    authority = _SEALED_AUTHORITIES.get(id(plan))
    if (
        authority is None
        or authority.plan is not plan
        or authority.plan_id != plan.plan_id
    ):
        raise PolicyError(PolicyErrorCode.INVALID_PLAN)
    _validate_candidate_integrity(plan.candidate)
    canonical = _canonical_fingerprint_bytes(
        plan.binding,
        plan.candidate,
        plan.review.expiry,
    )
    expected_lineages = tuple(
        _review_lineage(item) for item in plan.candidate.lineages
    )
    expected_warnings = tuple(
        CapabilityWarning(item)
        for item in sorted(
            plan.binding.final.effects,
            key=lambda item: item.value,
        )
        if item
        in {
            Capability.DELETE,
            Capability.MOVE,
            Capability.UPDATE_EXECUTABLE,
        }
    )
    if (
        canonical != authority.canonical
        or not compare_digest(
            sha256(canonical).digest(), authority.fingerprint
        )
        or not compare_digest(plan.fingerprint._value, authority.fingerprint)
        or plan.review.fingerprint != plan.fingerprint
        or plan.review.lineages != expected_lineages
        or plan.review.warnings != expected_warnings
        or plan.review.diff.digest != plan.candidate.diff.digest
        or plan.review.diff.size.value != len(plan.candidate.diff.rendered)
    ):
        raise PolicyError(PolicyErrorCode.INVALID_PLAN)


class PatchPolicyService(Protocol):
    """Load a trusted immutable policy through an asynchronous boundary."""

    async def load(self, revision: PolicyRevision) -> TrustedPatchPolicy:
        """Return one exact trusted policy revision."""


class PolicyAuthorizer:
    """Apply policy gates without target inspection, locking, or mutation."""

    def __init__(self, policy: TrustedPatchPolicy) -> None:
        """Bind an already trusted immutable policy value."""
        self._policy = policy

    async def authorize_preinspection(
        self, request: PreflightRequest
    ) -> PreflightAuthorization:
        """Authorize conservative observations before any target operation."""
        if request.operation not in self._policy.enabled_operations:
            raise PolicyError(PolicyErrorCode.DENIED)
        required = request.external_effects | frozenset(
            (Capability.OBSERVE_MUTATION_PRECONDITIONS,)
        )
        if request.external_read_paths:
            required = required | frozenset((Capability.READ_FOR_MUTATION,))
        for path in request.paths:
            path_effects = required
            if (
                path not in request.external_read_paths
                and Capability.READ_FOR_MUTATION in path_effects
            ):
                path_effects = path_effects - frozenset(
                    (Capability.READ_FOR_MUTATION,)
                )
            self._authorize_path(path, path_effects, preinspection=True)
        return PreflightAuthorization(
            revision=self._policy.revision,
            paths=request.paths,
            effects=required,
            effective_limits=request.effective_limits,
        )

    async def authorize_final(
        self,
        preflight: PreflightAuthorization,
        candidate: PlannerCandidate,
        handshake: TargetHandshake,
    ) -> FinalAuthorization:
        """Authorize final effects against policy and handshake witnesses."""
        if (
            preflight.revision != self._policy.revision
            or (
                handshake.identity.policy_revision
                != self._policy.revision.value
            )
            or not handshake.supports_inspection()
        ):
            raise PolicyError(PolicyErrorCode.DENIED)
        preflight_paths = frozenset(preflight.paths)
        candidate_paths = frozenset(
            path
            for lineage in candidate.lineages
            for path in (
                lineage.initial.path,
                lineage.final.path,
                lineage.source_path,
                lineage.destination_path,
            )
            if path is not None
        ) | frozenset(item.path for item in candidate.final_files)
        if not candidate_paths.issubset(preflight_paths):
            raise PolicyError(PolicyErrorCode.DENIED)
        final_effects = frozenset(
            item
            for lineage in candidate.lineages
            for item in lineage.capabilities
        )
        modes: list[CapabilityMode] = []
        disclosure_sets: list[frozenset[PolicyDisclosure]] = []
        for lineage in candidate.lineages:
            paths = tuple(
                item
                for item in (lineage.source_path, lineage.destination_path)
                if item is not None
            )
            for path in paths:
                rule = self._rule(path)
                if (
                    lineage.atomicity_class not in rule.atomicity_classes
                    or lineage.staging_class not in rule.staging_classes
                ):
                    raise PolicyError(PolicyErrorCode.DENIED)
                for value in lineage.capabilities:
                    mode = rule.mode_for(value)
                    if mode is None:
                        raise PolicyError(PolicyErrorCode.DENIED)
                    modes.append(mode)
                if (
                    lineage.final.metadata is not None
                    and lineage.final.metadata.mode.value & 0o111
                    and Capability.UPDATE in lineage.capabilities
                ):
                    mode = rule.mode_for(Capability.UPDATE_EXECUTABLE)
                    if mode is None:
                        raise PolicyError(PolicyErrorCode.DENIED)
                    modes.append(mode)
                    final_effects = final_effects | frozenset(
                        (Capability.UPDATE_EXECUTABLE,)
                    )
                disclosure_sets.append(rule.disclosures)
        if not final_effects.issubset(preflight.effects):
            raise PolicyError(PolicyErrorCode.DENIED)
        if not final_effects.issubset(handshake.advertised_operations()):
            raise PolicyError(PolicyErrorCode.DENIED)
        selected = _most_restrictive_mode(tuple(modes))
        if selected.mode is ApprovalMode.DENY:
            raise PolicyError(PolicyErrorCode.DENIED)
        if selected.mode is not self._policy.approval.mode or (
            selected.preauthorization != self._policy.approval.preauthorization
        ):
            raise PolicyError(PolicyErrorCode.DENIED)
        disclosures = (
            frozenset.intersection(*disclosure_sets)
            if disclosure_sets
            else frozenset()
        )
        _validate_candidate_limits(candidate, preflight.effective_limits.value)
        return FinalAuthorization(
            revision=self._policy.revision,
            handshake=handshake,
            effects=final_effects,
            disclosures=disclosures,
            effective_limits=preflight.effective_limits,
            approval=self._policy.approval,
        )

    def _authorize_path(
        self,
        path: LogicalPath,
        values: frozenset[Capability],
        *,
        preinspection: bool,
    ) -> None:
        """Require every independent path capability before inspection."""
        rule = self._rule(path)
        for value in values:
            mode = rule.mode_for(value)
            if mode is None or mode.mode is ApprovalMode.DENY:
                raise PolicyError(PolicyErrorCode.DENIED)
            if (
                preinspection
                and value
                in {
                    Capability.READ_FOR_MUTATION,
                    Capability.OBSERVE_MUTATION_PRECONDITIONS,
                }
                and mode.mode is not ApprovalMode.PREAUTHORIZED
            ):
                raise PolicyError(PolicyErrorCode.DENIED)

    def _rule(self, path: LogicalPath) -> PolicyRule:
        """Return one configured safe path rule or a coarse denial."""
        rule = self._policy.rule_for(path)
        if rule is None:
            raise PolicyError(PolicyErrorCode.PATH_DENIED)
        return rule


class ApprovalDecisionState(str, Enum):
    """Name trusted broker outcomes without generic confirmation values."""

    APPROVED = "approved"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PlanReviewRequest:
    """Carry one complete immutable plan-review request to a trusted broker."""

    plan: SealedPlan
    subject: ExecutionSubject
    requirements: ApprovalRequirements

    def __post_init__(self) -> None:
        """Require exact identity and approval bindings for review requests."""
        _validate_sealed_plan(self.plan)
        if (
            type(self.subject) is not ExecutionSubject
            or type(self.requirements) is not ApprovalRequirements
            or self.plan.binding.subject != self.subject
            or self.plan.binding.final.approval != self.requirements
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)


class Phase5IngressSurface(str, Enum):
    """Name untrusted inputs that cannot widen Phase 5 authority."""

    TOOL_ARGUMENTS = "tool_arguments"
    TOML = "toml"
    PROVIDER_FRAME = "provider_frame"
    REMOTE_FIELDS = "remote_fields"
    WORKSPACE_CONFIGURATION = "workspace_configuration"
    FILTER_DATA = "filter_data"
    APPROVAL_INPUT = "approval_input"


@dataclass(frozen=True, slots=True)
class Phase5ControlIngress:
    """Carry one rejected untrusted control-plane widening attempt."""

    surface: Phase5IngressSurface
    payload: object

    def __post_init__(self) -> None:
        """Reject non-enum surface labels before control validation."""
        if type(self.surface) is not Phase5IngressSurface:
            raise PolicyError(PolicyErrorCode.DENIED)


class Phase5IngressBoundary:
    """Reject untrusted policy controls and admit exact complete reviews."""

    def reject_control_widening(self, ingress: Phase5ControlIngress) -> None:
        """Fail closed for every caller-controlled policy widening attempt."""
        if type(ingress) is not Phase5ControlIngress:
            raise PolicyError(PolicyErrorCode.DENIED)
        raise PolicyError(PolicyErrorCode.DENIED)

    def review_request(
        self,
        plan: SealedPlan,
        subject: ExecutionSubject,
        artifact: object,
    ) -> PlanReviewRequest:
        """Create a review request only for the exact sealed full artifact."""
        if (
            type(plan) is not SealedPlan
            or type(subject) is not ExecutionSubject
            or type(artifact) is not CompleteReviewArtifact
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)
        _validate_sealed_plan(plan)
        if artifact != plan.review:
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)
        return PlanReviewRequest(plan, subject, plan.binding.final.approval)


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    """Record one broker-authenticated reviewer decision."""

    reviewer: PatchPrincipalId
    tenant: PatchTenantId
    reviewer_role: PolicyReviewerRole
    state: ApprovalDecisionState

    def __post_init__(self) -> None:
        """Reject raw approvals without an exact typed reviewer identity."""
        if (
            type(self.reviewer) is not PatchPrincipalId
            or type(self.tenant) is not PatchTenantId
            or type(self.reviewer_role) is not PolicyReviewerRole
            or type(self.state) is not ApprovalDecisionState
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)


@dataclass(frozen=True, slots=True)
class BrokerDecision:
    """Store a closed asynchronous broker result for exact review requests."""

    broker: PolicyBrokerId
    decisions: tuple[ReviewerDecision, ...]

    def __post_init__(self) -> None:
        """Reject duplicate reviewers or empty broker decision records."""
        if (
            type(self.decisions) is not tuple
            or not self.decisions
            or any(
                type(item) is not ReviewerDecision for item in self.decisions
            )
            or len({item.reviewer for item in self.decisions})
            != len(self.decisions)
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)


class PlanApprovalBroker(Protocol):
    """Obtain typed plan decisions through an asynchronous broker."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Return only a closed decision for the complete supplied review."""


class ApprovalClock(Protocol):
    """Read a trusted monotonic approval clock asynchronously."""

    async def now(self) -> ExpiryTick:
        """Return the current trusted monotonic expiry tick."""


@dataclass(frozen=True, slots=True, repr=False)
class PlanBoundGrant:
    """Store an opaque unconsumed grant bound to one exact sealed plan."""

    grant_id: PatchGrantId
    approval_id: PatchApprovalId
    plan_id: PatchPlanId
    expiry: ExpiryTick
    binding: PlanBinding
    fingerprint: PatchFingerprint
    diff_digest: AlgorithmDigest
    reviewers: tuple[PatchPrincipalId, ...]
    _secret: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject malformed grant records before Phase 6 consumption exists."""
        if (
            type(self.reviewers) is not tuple
            or len(self.reviewers) < self.binding.final.approval.quorum
            or len(set(self.reviewers)) != len(self.reviewers)
            or type(self._secret) is not bytes
            or len(self._secret) != 32
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    """Return a validated but deliberately unconsumed review outcome."""

    state: ApprovalDecisionState
    grant: PlanBoundGrant | None = None

    def __post_init__(self) -> None:
        """Keep a grant exclusive to an approved broker outcome."""
        if (self.state is ApprovalDecisionState.APPROVED) != (
            self.grant is not None
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)


class GrantStore(Protocol):
    """Keep plan-bound grants opaque in runtime-owned asynchronous storage."""

    async def put(self, grant: PlanBoundGrant) -> None:
        """Store one issued but deliberately unconsumed approval grant."""

    async def get(self, grant_id: PatchGrantId) -> PlanBoundGrant | None:
        """Return one private grant record to a trusted runtime caller."""


class RuntimeGrantStore:
    """Store opaque grants without exposing consumption before Phase 6."""

    def __init__(
        self,
        retention: PrivateArtifactRetention = (
            _DEFAULT_PRIVATE_ARTIFACT_RETENTION
        ),
    ) -> None:
        """Initialize an empty runtime-owned grant store."""
        if type(retention) is not PrivateArtifactRetention:
            raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)
        self._retention = retention
        self._grants: dict[PatchGrantId, PlanBoundGrant] = {}

    async def put(self, grant: PlanBoundGrant) -> None:
        """Store exactly one immutable grant record for its identifier."""
        current = self._grants.get(grant.grant_id)
        if current is not None and current != grant:
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)
        if (
            current is None
            and len(self._grants) >= self._retention.max_records
        ):
            raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)
        self._grants[grant.grant_id] = grant

    async def get(self, grant_id: PatchGrantId) -> PlanBoundGrant | None:
        """Return a private unconsumed record without changing its state."""
        return self._grants.get(grant_id)

    async def cleanup_expired(self, now: ExpiryTick) -> int:
        """Remove expired grants in deterministic expiry and ID order."""
        expired = tuple(
            identifier
            for identifier, grant in sorted(
                self._grants.items(),
                key=lambda item: (item[1].expiry.value, item[0].value),
            )
            if grant.expiry.value <= now.value
        )
        for identifier in expired:
            del self._grants[identifier]
        return len(expired)


class ApprovalService:
    """Await typed approval without locks, handles, or grant consumption."""

    def __init__(
        self,
        broker: PlanApprovalBroker,
        clock: ApprovalClock,
        grants: GrantStore,
    ) -> None:
        """Bind the trusted asynchronous broker and monotonic clock."""
        self._broker = broker
        self._clock = clock
        self._grants = grants

    async def await_review(self, request: PlanReviewRequest) -> ApprovalResult:
        """Resolve one complete review without a commit transaction."""
        _validate_sealed_plan(request.plan)
        now = await self._clock.now()
        if now.value >= request.plan.review.expiry.value:
            return ApprovalResult(ApprovalDecisionState.DENIED)
        try:
            result = await self._broker.decide(request)
        except PolicyError:
            return ApprovalResult(ApprovalDecisionState.UNAVAILABLE)
        if result.broker != request.requirements.broker:
            return ApprovalResult(ApprovalDecisionState.DENIED)
        if any(
            item.state is ApprovalDecisionState.UNAVAILABLE
            for item in result.decisions
        ):
            return ApprovalResult(ApprovalDecisionState.UNAVAILABLE)
        approved = tuple(
            item
            for item in result.decisions
            if item.state is ApprovalDecisionState.APPROVED
            and item.tenant == request.subject.tenant
            and item.reviewer_role == request.requirements.reviewer_role
        )
        if any(
            item.state
            in {
                ApprovalDecisionState.DENIED,
                ApprovalDecisionState.CANCELLED,
            }
            for item in result.decisions
        ):
            return ApprovalResult(ApprovalDecisionState.DENIED)
        if len(approved) < request.requirements.quorum:
            return ApprovalResult(ApprovalDecisionState.DENIED)
        after = await self._clock.now()
        if after.value >= request.plan.review.expiry.value:
            return ApprovalResult(ApprovalDecisionState.DENIED)
        grant = PlanBoundGrant(
            grant_id=PatchGrantId.new(),
            approval_id=PatchApprovalId.new(),
            plan_id=request.plan.plan_id,
            expiry=request.plan.review.expiry,
            binding=request.plan.binding,
            fingerprint=request.plan.fingerprint,
            diff_digest=request.plan.review.diff.digest,
            reviewers=tuple(item.reviewer for item in approved),
            _secret=token_bytes(32),
        )
        await self._grants.put(grant)
        return ApprovalResult(ApprovalDecisionState.APPROVED, grant)

    async def validate_grant(
        self,
        grant: PlanBoundGrant,
        plan: SealedPlan,
        subject: ExecutionSubject,
    ) -> None:
        """Validate an unconsumed grant while Phase 6 ownership is absent."""
        _validate_sealed_plan(plan)
        now = await self._clock.now()
        issued = await self._grants.get(grant.grant_id)
        if now.value >= grant.expiry.value:
            raise PolicyError(PolicyErrorCode.APPROVAL_EXPIRED)
        if (
            issued is None
            or not compare_digest(issued._secret, grant._secret)
            or grant != issued
            or grant.plan_id != plan.plan_id
            or grant.binding != plan.binding
            or grant.binding.subject != subject
            or not compare_digest(
                grant.fingerprint._value, plan.fingerprint._value
            )
            or grant.diff_digest != plan.review.diff.digest
            or len(grant.reviewers) < plan.binding.final.approval.quorum
        ):
            raise PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)


@dataclass(frozen=True, slots=True)
class BoundedPlanSummary:
    """Project a sealed plan without diff, hashes, sizes, or grant material."""

    observer_id: str
    operation: OperationType
    lineage_count: int
    approval_mode: ApprovalMode


class PlanStore(Protocol):
    """Persist authoritative plans behind an asynchronous opaque boundary."""

    async def put(self, plan: SealedPlan) -> BoundedPlanSummary:
        """Store one sealed plan and return its bounded observer projection."""

    async def get(self, plan_id: PatchPlanId) -> SealedPlan | None:
        """Return an authoritative plan only to the trusted runtime caller."""


class RuntimePlanStore:
    """Keep authoritative plans private with random observer tokens."""

    def __init__(
        self,
        retention: PrivateArtifactRetention = (
            _DEFAULT_PRIVATE_ARTIFACT_RETENTION
        ),
    ) -> None:
        """Initialize an empty runtime-owned plan store."""
        if type(retention) is not PrivateArtifactRetention:
            raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)
        self._retention = retention
        self._plans: dict[PatchPlanId, SealedPlan] = {}
        self._observers: dict[PatchPlanId, str] = {}

    async def put(self, plan: SealedPlan) -> BoundedPlanSummary:
        """Store one immutable plan without exposing its authority material."""
        _validate_sealed_plan(plan)
        existing = self._plans.get(plan.plan_id)
        if existing is not None and existing != plan:
            raise PolicyError(PolicyErrorCode.INVALID_PLAN)
        if (
            existing is None
            and len(self._plans) >= self._retention.max_records
        ):
            raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)
        self._plans[plan.plan_id] = plan
        observer = self._observers.setdefault(
            plan.plan_id,
            sha256(token_bytes(32)).hexdigest(),
        )
        return BoundedPlanSummary(
            observer_id=observer,
            operation=plan.binding.request.operation,
            lineage_count=len(plan.candidate.lineages),
            approval_mode=plan.binding.final.approval.mode,
        )

    async def get(self, plan_id: PatchPlanId) -> SealedPlan | None:
        """Return a private authoritative plan without public projection."""
        return self._plans.get(plan_id)

    async def cleanup_expired(self, now: ExpiryTick) -> int:
        """Remove expired plans and private observer values in stable order."""
        expired = tuple(
            identifier
            for identifier, plan in sorted(
                self._plans.items(),
                key=lambda item: (
                    item[1].review.expiry.value,
                    item[0].value,
                ),
            )
            if plan.review.expiry.value <= now.value
        )
        for identifier in expired:
            del self._plans[identifier]
            del self._observers[identifier]
        return len(expired)


@dataclass(frozen=True, slots=True)
class ModelProjection:
    """Project only policy-authorized model-facing plan facts."""

    paths: tuple[LogicalPath, ...]
    diff: DiffBytes | None
    hashes_and_sizes: bool
    detailed_matches: bool


@dataclass(frozen=True, slots=True)
class ReviewerProjection:
    """Project a complete artifact only to an authorized reviewer channel."""

    artifact: CompleteReviewArtifact


@dataclass(frozen=True, slots=True)
class SdkHostProjection:
    """Project bounded host status without plan contents or grant material."""

    observer_id: str
    approval_mode: ApprovalMode


@dataclass(frozen=True, slots=True)
class CoarseDenialProjection:
    """Project a stable private-free denial suitable for public audiences."""

    code: PolicyErrorCode


def project_model(plan: SealedPlan) -> ModelProjection:
    """Project only explicitly authorized model disclosure fields."""
    _validate_sealed_plan(plan)
    disclosures = plan.binding.final.disclosures
    paths = (
        tuple(
            path
            for lineage in plan.candidate.lineages
            for path in (lineage.source_path, lineage.destination_path)
            if path is not None
        )
        if PolicyDisclosure.AUDIT_PATHS in disclosures
        else ()
    )
    return ModelProjection(
        paths=paths,
        diff=(
            plan.review.diff.diff
            if PolicyDisclosure.MODEL_DIFF in disclosures
            else None
        ),
        hashes_and_sizes=PolicyDisclosure.MODEL_METADATA in disclosures,
        detailed_matches=PolicyDisclosure.MODEL_MATCH_DETAILS in disclosures,
    )


def project_reviewer(plan: SealedPlan) -> ReviewerProjection:
    """Project a complete untruncated artifact only when policy permits it."""
    _validate_sealed_plan(plan)
    if PolicyDisclosure.COMPLETE_REVIEW not in plan.binding.final.disclosures:
        raise PolicyError(PolicyErrorCode.PATH_DENIED)
    return ReviewerProjection(plan.review)


def project_sdk_host(
    summary: BoundedPlanSummary,
) -> SdkHostProjection:
    """Project a bounded SDK-host summary without widening disclosure."""
    return SdkHostProjection(summary.observer_id, summary.approval_mode)


def project_denial(error: PolicyError) -> CoarseDenialProjection:
    """Coarsen a policy error without paths, contents, or rule detail."""
    return CoarseDenialProjection(
        PolicyErrorCode.PATH_DENIED
        if error.code is PolicyErrorCode.PATH_DENIED
        else PolicyErrorCode.DENIED
    )


def _path_visibility(path: LogicalPath) -> PathVisibility:
    """Classify a canonical path before any target observation occurs."""
    parts = tuple(path.value.split("/"))
    if any(item in {".git", ".hg", ".svn"} for item in parts):
        return PathVisibility.VERSION_CONTROL
    if any(item.startswith(".") for item in parts):
        return PathVisibility.HIDDEN
    return PathVisibility.ORDINARY


def _target_matches_handshake(
    target: TargetIdentity, handshake: TargetIdentity
) -> bool:
    """Compare durable target facts while deliberately excluding leases."""
    return (
        target.context_id == handshake.context_id
        and target.workspace_id == handshake.workspace_id
        and target.domain_id == handshake.domain_id
        and target.target_id == handshake.target_id
        and target.protocol_id == handshake.protocol_id
        and target.filesystem_id == handshake.filesystem_id
        and target.mount_id == handshake.mount_id
        and target.policy_revision == handshake.policy_revision
        and target.approval_channel_id == handshake.approval_channel_id
    )


def _most_restrictive_mode(
    values: tuple[CapabilityMode, ...],
) -> CapabilityMode:
    """Return deny, review, or bounded preauthorization in safe precedence."""
    if not values:
        raise PolicyError(PolicyErrorCode.DENIED)
    rank = {
        ApprovalMode.DENY: 0,
        ApprovalMode.REQUIRE_REVIEW: 1,
        ApprovalMode.PREAUTHORIZED: 2,
    }
    return min(values, key=lambda item: rank[item.mode])


def _validate_candidate_limits(
    candidate: PlannerCandidate, limits: PatchLimits
) -> None:
    """Reject post-planning resource facts beyond sealed effective limits."""
    if (
        len(candidate.lineages) > limits.file_count.value
        or len(candidate.lineages) > limits.operation_count.value
        or len(candidate.diff.rendered) > limits.review_diff_bytes.value
        or sum(
            item.final.size.value
            for item in candidate.lineages
            if item.final.present
        )
        > limits.proposed_bytes.value
    ):
        raise PolicyError(PolicyErrorCode.LIMIT_EXCEEDED)


def _validate_candidate_integrity(candidate: PlannerCandidate) -> None:
    """Reject altered private bytes or diff facts before authority use."""
    files = (
        *(item.initial for item in candidate.lineages),
        *(item.final for item in candidate.lineages),
        *candidate.final_files,
    )
    if candidate.diff.digest != AlgorithmDigest.from_bytes(
        candidate.diff.rendered
    ):
        raise PolicyError(PolicyErrorCode.INVALID_PLAN)
    for item in files:
        if item.present and (
            item.bytes_value is None
            or item.digest != item.bytes_value.digest()
            or item.size != item.bytes_value.size()
        ):
            raise PolicyError(PolicyErrorCode.INVALID_PLAN)


def _review_lineage(lineage: PlannedLineage) -> ReviewLineage:
    """Project one planner lineage into complete detached review fields."""
    return ReviewLineage(
        lineage_id=lineage.lineage_id.value,
        source_path=lineage.source_path,
        destination_path=lineage.destination_path,
        effects=lineage.capabilities,
        regions=tuple(_review_region(item) for item in lineage.matches),
        atomicity=lineage.atomicity_class,
        staging=lineage.staging_class,
    )


def _review_region(value: Match) -> ReviewRegion:
    """Project exact resolved match coordinates without source text."""
    return ReviewRegion(
        logical_start=value.span.logical_start,
        logical_end=value.span.logical_end,
        byte_start=value.span.byte_start,
        byte_end=value.span.byte_end,
    )


def _canonical_fingerprint_bytes(
    binding: PlanBinding,
    candidate: PlannerCandidate,
    expiry: ExpiryTick,
) -> bytes:
    """Serialize durable plan facts in a versioned domain-separated form."""
    values = (
        _FINGERPRINT_DOMAIN,
        _integer(_PLAN_SCHEMA_VERSION),
        _integer(_GRAMMAR_VERSION),
        _integer(_SEMANTIC_MODEL_VERSION),
        _text(binding.request.operation.value),
        _integer(binding.request.schema_version),
        _text(binding.request.request_id.value),
        _text(binding.request.execution_id.value),
        _digest(binding.request_digest),
        _digest(binding.request.input_bytes.digest()),
        _paths(binding.request.logical_paths),
        _text(binding.subject.principal.value),
        _text(binding.subject.tenant.value),
        _text(binding.subject.run.value),
        _text(binding.subject.session.value),
        _text(binding.subject.task.value),
        _text(binding.subject.agent.value),
        _text(binding.context_kind.value),
        _text(binding.target.context_id.value),
        _text(binding.target.workspace_id.value),
        _text(binding.target.domain_id.value),
        _text(binding.target.target_id.value),
        _text(binding.target.protocol_id.value),
        _text(binding.target.filesystem_id),
        _text(binding.target.mount_id),
        _text(binding.target.policy_revision),
        _text(binding.target.approval_channel_id.value),
        _handshake(binding.final.handshake),
        _path(binding.cwd),
        _text(binding.preflight.revision.value),
        _paths(binding.preflight.paths),
        _capabilities(binding.preflight.effects),
        _limits(binding.preflight.effective_limits.value),
        _text(binding.final.revision.value),
        _capabilities(binding.final.effects),
        _disclosures(binding.final.disclosures),
        _limits(binding.final.effective_limits.value),
        _text(binding.final.approval.mode.value),
        _text(binding.final.approval.route.value),
        _text(binding.final.approval.broker.value),
        _text(binding.final.approval.reviewer_role.value),
        _integer(binding.final.approval.quorum),
        _optional_text(binding.final.approval.preauthorization),
        _optional_diagnostic(binding.diagnostic_policy),
        _integer(expiry.value),
        _digest(candidate.request_digest),
        _integer(len(candidate.lineages)),
        *tuple(_lineage_bytes(item) for item in candidate.lineages),
        _integer(len(candidate.final_files)),
        *tuple(_planned_file_bytes(item) for item in candidate.final_files),
        _byte_values(candidate.diff.entries),
        _digest(candidate.diff.digest),
        _integer(len(candidate.diff.rendered)),
    )
    return b"".join(_length_prefix(item) for item in values)


def _lineage_bytes(value: PlannedLineage) -> bytes:
    """Serialize one canonical lineage including before and after facts."""
    pieces = (
        _text(value.lineage_id.value),
        _planned_file_bytes(value.initial),
        _planned_file_bytes(value.final),
        _path(value.source_path),
        _path(value.destination_path),
        _capabilities(value.capabilities),
        _matches(value.matches),
        _paths(value.parent_paths),
        _texts(value.mount_ids),
        _paths(value.lock_footprint),
        _text(value.atomicity_class),
        _texts(value.step_graph),
        _text(value.staging_class),
        value.diff_contribution,
    )
    return b"".join(_length_prefix(item) for item in pieces)


def _planned_file_bytes(value: PlannedFile) -> bytes:
    """Serialize a present or expected-absent terminal file fact."""
    return b"".join(
        _length_prefix(item)
        for item in (
            _path(value.path),
            _boolean(value.present),
            _integer(value.size.value),
            _optional_digest(value.digest),
            _optional_metadata(value.metadata),
        )
    )


def _matches(values: tuple[Match, ...]) -> bytes:
    """Serialize ordered exact selected match regions and strategies."""
    return b"".join(
        _length_prefix(
            b"".join(
                _length_prefix(item)
                for item in (
                    _text(value.kind.value),
                    _integer(value.span.logical_start),
                    _integer(value.span.logical_end),
                    _integer(value.span.byte_start),
                    _integer(value.span.byte_end),
                )
            )
        )
        for value in values
    )


def _limits(value: PatchLimits) -> bytes:
    """Serialize every effective finite limit in explicit stable order."""
    return b"".join(
        _length_prefix(_integer(item))
        for item in (
            value.input_bytes.value,
            value.path_count.value,
            value.path_length.value,
            value.file_count.value,
            value.operation_count.value,
            value.snapshot_bytes.value,
            value.proposed_bytes.value,
            value.review_diff_bytes.value,
            value.planning_duration.value,
            value.approval_duration.value,
            value.commit_duration.value,
        )
    )


def _capabilities(values: frozenset[Capability]) -> bytes:
    """Serialize capabilities in enum-value order independent of insertion."""
    return _texts(
        tuple(
            item.value for item in sorted(values, key=lambda item: item.value)
        )
    )


def _handshake(value: TargetHandshake) -> bytes:
    """Serialize the plan-bound, worker-free target capability witness."""
    primitives = tuple(
        item.value
        for item in sorted(value.primitives, key=lambda item: item.value)
    )
    probes = tuple(
        item.primitive.value + ":" + item.state.value
        for item in sorted(value.probes, key=lambda item: item.primitive.value)
    )
    return b"".join(
        _length_prefix(item)
        for item in (
            _target_identity(value.identity),
            _texts(primitives),
            _texts(probes),
            _texts(
                tuple(
                    item.value
                    for item in sorted(
                        value.incapable_reasons,
                        key=lambda item: item.value,
                    )
                )
            ),
            _text(value.platform.value),
            _text(value.foreign_writer_guarantee.value),
        )
    )


def _target_identity(value: TargetIdentity) -> bytes:
    """Serialize durable target identity while excluding the lease epoch."""
    return b"".join(
        _length_prefix(item)
        for item in (
            _text(value.context_id.value),
            _text(value.workspace_id.value),
            _text(value.domain_id.value),
            _text(value.target_id.value),
            _text(value.protocol_id.value),
            _text(value.filesystem_id),
            _text(value.mount_id),
            _text(value.policy_revision),
            _text(value.approval_channel_id.value),
        )
    )


def _disclosures(values: frozenset[PolicyDisclosure]) -> bytes:
    """Serialize policy disclosure controls in enum-value order."""
    return _texts(
        tuple(
            item.value for item in sorted(values, key=lambda item: item.value)
        )
    )


def _paths(values: tuple[LogicalPath, ...]) -> bytes:
    """Serialize ordered canonical logical paths."""
    return _texts(tuple(item.value for item in values))


def _texts(values: tuple[str, ...]) -> bytes:
    """Serialize an ordered sequence of text values without delimiters."""
    return b"".join(_length_prefix(_text(item)) for item in values)


def _byte_values(values: tuple[bytes, ...]) -> bytes:
    """Serialize ordered private diff entries without text decoding."""
    return b"".join(_length_prefix(item) for item in values)


def _path(value: LogicalPath | None) -> bytes:
    """Serialize an optional canonical logical path."""
    return b"" if value is None else _text(value.value)


def _digest(value: AlgorithmDigest) -> bytes:
    """Serialize an algorithm-qualified digest rather than a bare hash."""
    return _text(value.algorithm) + b":" + _text(value.value)


def _optional_digest(value: AlgorithmDigest | None) -> bytes:
    """Serialize an optional algorithm-qualified digest."""
    return b"" if value is None else _digest(value)


def _optional_metadata(value: MetadataProfile | None) -> bytes:
    """Serialize a validated metadata profile without dynamic projection."""
    if value is None:
        return b""
    mode = value.mode
    bom = value.has_utf8_bom
    newline = value.newline
    return b"".join(
        _length_prefix(item)
        for item in (_integer(mode.value), _boolean(bom), _text(newline))
    )


def _optional_text(value: PreauthorizationClass | None) -> bytes:
    """Serialize an optional bounded preauthorization identifier."""
    return b"" if value is None else _text(value.value)


def _optional_diagnostic(value: DiagnosticPolicyId | None) -> bytes:
    """Serialize an optional independently authorized diagnostic policy."""
    return b"" if value is None else _text(value.value)


def _integer(value: int) -> bytes:
    """Serialize one nonnegative bounded integer canonically."""
    if type(value) is not int or value < 0:
        raise PolicyError(PolicyErrorCode.INVALID_PLAN)
    return str(value).encode("ascii")


def _boolean(value: bool) -> bytes:
    """Serialize one exact Boolean fact canonically."""
    return b"1" if value else b"0"


def _text(value: str) -> bytes:
    """Serialize one UTF-8 text fact with no implicit normalization."""
    return value.encode("utf-8", "strict")


def _length_prefix(value: bytes) -> bytes:
    """Length-delimit canonical fields to prevent concatenation ambiguity."""
    return str(len(value)).encode("ascii") + b":" + value
