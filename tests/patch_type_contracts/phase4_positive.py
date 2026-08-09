"""Assert the incapable target facade remains strictly asynchronous."""

from typing import assert_type

from avalan.patch.target import (
    CommitUnavailable,
    InspectionBatch,
    InspectionRequest,
    MutationTarget,
    ResolvedMutationScope,
    TargetHandshake,
)


async def assert_target_types(
    target: MutationTarget,
    scope: ResolvedMutationScope,
    request: InspectionRequest,
) -> None:
    """Assert target effects use closed async protocol result types."""
    assert_type(await target.handshake(scope), TargetHandshake)
    assert_type(await target.inspect(request), InspectionBatch)
    assert_type(await target.commit(request), CommitUnavailable)
