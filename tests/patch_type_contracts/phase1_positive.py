"""Assert the active immutable mutation-domain type boundaries."""

from typing import Protocol, assert_type

from avalan.patch.domain import (
    PatchPending,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
)


class PatchJournalStore(Protocol):
    """Persist one closed terminal outcome asynchronously."""

    async def store(self, outcome: PatchResult | PatchPending) -> None:
        """Store one closed invocation outcome."""


def assert_domain_types(
    request_id: PatchRequestId,
    plan_id: PatchPlanId,
    outcome: PatchResult | PatchPending,
) -> None:
    """Assert distinct identities and closed outcome unions."""
    assert_type(request_id, PatchRequestId)
    assert_type(plan_id, PatchPlanId)
    assert_type(outcome, PatchResult | PatchPending)
