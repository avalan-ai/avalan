"""Issue and validate broker-attested durable approval claim bindings."""

from dataclasses import dataclass
from hmac import compare_digest, digest
from json import dumps
from secrets import token_bytes

from avalan.patch.durable_store import (
    DurableApproval,
    DurableApprovalVerifier,
    DurablePlanReference,
    DurableRequestIdentity,
    DurableStoreError,
    DurableStoreErrorCode,
)
from avalan.patch.policy import (
    ApprovalService,
    ExecutionSubject,
    PlanBoundGrant,
    SealedPlan,
)


@dataclass(frozen=True, slots=True, repr=False)
class DurableApprovalSigningKey:
    """Carry one private HMAC key without exposing it in diagnostics."""

    key_bytes: bytes

    def __post_init__(self) -> None:
        """Require one exact bounded broker-owned signing key."""
        if type(self.key_bytes) is not bytes or len(self.key_bytes) != 32:
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)

    def __repr__(self) -> str:
        """Render only a redacted durable-attestation key marker."""
        return "DurableApprovalSigningKey(<redacted>)"


class HmacDurableApprovalAuthority(DurableApprovalVerifier):
    """Seal and verify complete durable approval claim bindings."""

    def __init__(self, key: DurableApprovalSigningKey) -> None:
        """Bind one broker-owned key without performing approval effects."""
        if type(key) is not DurableApprovalSigningKey:
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
        self._key = key

    @classmethod
    def random(cls) -> "HmacDurableApprovalAuthority":
        """Create one test-host broker authority with an unpredictable key."""
        return cls(DurableApprovalSigningKey(token_bytes(32)))

    def issue(
        self,
        identity: DurableRequestIdentity,
        plan: DurablePlanReference,
        grant: PlanBoundGrant,
        sealed_plan: SealedPlan,
    ) -> DurableApproval:
        """Seal one already Phase-5-validated approval for the durable plan."""
        _validate_phase_five_binding(identity, plan, grant, sealed_plan)
        unsigned = DurableApproval(
            grant.grant_id,
            grant.approval_id,
            identity,
            plan.canonical_digest,
            plan.plan_id,
            plan.fingerprint_digest,
            plan.review_digest,
            plan.context_id,
            plan.workspace_id,
            plan.domain_id,
            sealed_plan.binding.final.revision.value,
            sealed_plan.binding.final.approval.broker,
            sealed_plan.binding.final.approval.reviewer_role,
            grant.reviewers,
            grant.expiry,
            b"\x00" * 32,
        )
        return self.seal(unsigned)

    def seal(self, approval: DurableApproval) -> DurableApproval:
        """Seal a complete claim assembled by the trusted broker boundary."""
        if type(approval) is not DurableApproval:
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
        return DurableApproval(
            approval.grant_id,
            approval.approval_id,
            approval.identity,
            approval.canonical_digest,
            approval.plan_id,
            approval.fingerprint_digest,
            approval.review_digest,
            approval.context_id,
            approval.workspace_id,
            approval.domain_id,
            approval.policy_revision,
            approval.broker_id,
            approval.reviewer_role,
            approval.reviewers,
            approval.expires_at,
            self._attestation(approval),
        )

    def verify(self, approval: DurableApproval) -> None:
        """Reject a forged or altered broker-issued durable claim binding."""
        if type(approval) is not DurableApproval or not compare_digest(
            self._attestation(approval), approval.attestation
        ):
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)

    def _attestation(self, approval: DurableApproval) -> bytes:
        """Compute one domain-separated opaque full-binding attestation."""
        return digest(
            self._key.key_bytes,
            _canonical_approval_bytes(approval),
            "sha256",
        )


class PhaseFiveDurableApprovalIssuer:
    """Convert only Phase-5 broker grants into durable claim attestations."""

    def __init__(
        self,
        approvals: ApprovalService,
        authority: HmacDurableApprovalAuthority,
    ) -> None:
        """Bind the Phase-5 grant validator and broker signing authority."""
        if (
            type(approvals) is not ApprovalService
            or type(authority) is not HmacDurableApprovalAuthority
        ):
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
        self._approvals = approvals
        self._authority = authority

    async def issue(
        self,
        identity: DurableRequestIdentity,
        plan: DurablePlanReference,
        grant: PlanBoundGrant,
        sealed_plan: SealedPlan,
        subject: ExecutionSubject,
    ) -> DurableApproval:
        """Validate the broker grant before sealing its durable binding."""
        _validate_phase_five_binding(identity, plan, grant, sealed_plan)
        if subject != sealed_plan.binding.subject:
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)
        try:
            await self._approvals.validate_grant(grant, sealed_plan, subject)
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise DurableStoreError(
                DurableStoreErrorCode.APPROVAL_MISMATCH
            ) from None
        return self._authority.issue(identity, plan, grant, sealed_plan)


def _validate_phase_five_binding(
    identity: DurableRequestIdentity,
    plan: DurablePlanReference,
    grant: PlanBoundGrant,
    sealed_plan: SealedPlan,
) -> None:
    """Require the Phase-5 sealed plan to cover every durable claim fact."""
    if (
        type(identity) is not DurableRequestIdentity
        or type(plan) is not DurablePlanReference
        or type(grant) is not PlanBoundGrant
        or type(sealed_plan) is not SealedPlan
        or identity.tenant_id != sealed_plan.binding.subject.tenant
        or identity.principal_id != sealed_plan.binding.subject.principal
        or identity.execution_id != sealed_plan.binding.request.execution_id
        or identity.route_id != sealed_plan.binding.final.approval.route
        or plan.plan_id != sealed_plan.plan_id
        or plan.canonical_digest != sealed_plan.binding.request_digest
        or plan.fingerprint_digest != sealed_plan.fingerprint.digest()
        or plan.review_digest != sealed_plan.review.diff.digest
        or plan.context_id != sealed_plan.binding.target.context_id
        or plan.workspace_id != sealed_plan.binding.target.workspace_id
        or plan.domain_id != sealed_plan.binding.target.domain_id
        or grant.plan_id != sealed_plan.plan_id
        or grant.binding != sealed_plan.binding
        or grant.fingerprint != sealed_plan.fingerprint
        or grant.diff_digest != sealed_plan.review.diff.digest
        or len(grant.reviewers) < sealed_plan.binding.final.approval.quorum
    ):
        raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)


def _canonical_approval_bytes(approval: DurableApproval) -> bytes:
    """Encode durable authorization claims into one deterministic payload."""
    return dumps(
        {
            "approval_id": approval.approval_id.value,
            "broker_id": approval.broker_id.value,
            "canonical_digest": approval.canonical_digest.value,
            "context_id": approval.context_id.value,
            "domain_id": approval.domain_id.value,
            "expires_at": approval.expires_at.value,
            "fingerprint_digest": approval.fingerprint_digest.value,
            "grant_id": approval.grant_id.value,
            "identity": (
                approval.identity.tenant_id.value,
                approval.identity.principal_id.value,
                approval.identity.execution_id.value,
                approval.identity.route_id.value,
                approval.identity.retransmission_key.value,
            ),
            "plan_id": approval.plan_id.value,
            "policy_revision": approval.policy_revision,
            "review_digest": approval.review_digest.value,
            "reviewer_role": approval.reviewer_role.value,
            "reviewers": tuple(item.value for item in approval.reviewers),
            "workspace_id": approval.workspace_id.value,
            "version": 1,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
