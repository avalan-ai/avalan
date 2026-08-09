"""Assert the Phase 5 policy and approval protocols remain async and typed."""

from typing import assert_type

from avalan.patch.domain import ApprovalMode
from avalan.patch.policy import (
    ApprovalResult,
    ApprovalService,
    BrokerDecision,
    PlanApprovalBroker,
    PlanReviewRequest,
    RuntimeGrantStore,
    SealedPlan,
    TrustedPatchPolicy,
)


async def assert_policy_types(
    service: ApprovalService,
    broker: PlanApprovalBroker,
    request: PlanReviewRequest,
    plan: SealedPlan,
) -> None:
    """Assert the policy boundary exposes closed asynchronous values."""
    assert_type(TrustedPatchPolicy.empty(), TrustedPatchPolicy)
    assert_type(await service.await_review(request), ApprovalResult)
    assert_type(await broker.decide(request), BrokerDecision)
    assert_type(RuntimeGrantStore(), RuntimeGrantStore)
    assert_type(plan.binding.final.approval.mode, ApprovalMode)
