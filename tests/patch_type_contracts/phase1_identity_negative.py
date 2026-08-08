"""Reject cross-type immutable mutation-domain identities."""

from avalan.patch.domain import PatchPlanId, PatchRequestId


def reject_cross_type(
    request_id: PatchRequestId,
) -> None:
    """Reject assigning request identity to a plan identity."""
    plan_id: PatchPlanId = request_id
    assert plan_id
