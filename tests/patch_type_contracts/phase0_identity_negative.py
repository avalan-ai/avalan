"""Reject cross-type patch identifiers at the strict boundary."""

from typing import NewType

PatchPlanId = NewType("PatchPlanId", str)
PatchRequestId = NewType("PatchRequestId", str)


def reject_mixed_identity() -> None:
    """Keep request and plan identifiers non-interchangeable."""
    request = PatchRequestId("request-0001")
    plan: PatchPlanId = request
    assert plan
