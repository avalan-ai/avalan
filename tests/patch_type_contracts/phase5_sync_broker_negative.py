"""Reject a synchronous implementation of the plan approval broker."""

from avalan.patch import (
    BrokerDecision,
    PlanApprovalBroker,
    PlanReviewRequest,
)


class SynchronousBroker:
    """Deliberately violate the asynchronous broker protocol."""

    def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Return synchronously for the negative type fixture."""
        raise RuntimeError(request)


broker: PlanApprovalBroker = SynchronousBroker()
