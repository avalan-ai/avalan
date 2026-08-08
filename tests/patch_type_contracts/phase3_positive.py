"""Assert the dormant planner facade remains closed and asynchronous."""

from typing import assert_type

from avalan.patch.parser import CanonicalPatchRequest
from avalan.patch.planner import (
    BoundedPlannerWorker,
    PlannerCandidate,
    PlannerFacade,
    PlannerLimits,
    PlannerWorkspace,
)


async def assert_planner_types(
    request: CanonicalPatchRequest, workspace: PlannerWorkspace
) -> None:
    """Assert the public planner facade returns only an unsealed candidate."""
    facade = PlannerFacade(BoundedPlannerWorker(1), PlannerLimits())
    candidate = await facade.plan(request, workspace)
    assert_type(candidate, PlannerCandidate)
