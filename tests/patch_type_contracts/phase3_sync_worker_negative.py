"""Reject synchronous implementations of the planner worker protocol."""

from avalan.patch.parser import CanonicalPatchRequest
from avalan.patch.planner import (
    PlannerCandidate,
    PlannerLimits,
    PlannerWorker,
    PlannerWorkspace,
)


class SynchronousWorker:
    """Deliberately violate the asynchronous planner worker protocol."""

    def plan(
        self,
        request: CanonicalPatchRequest,
        workspace: PlannerWorkspace,
        limits: PlannerLimits,
    ) -> PlannerCandidate:
        """Return a value synchronously for the negative contract fixture."""
        raise RuntimeError("negative fixture")


worker: PlannerWorker = SynchronousWorker()
