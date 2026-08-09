"""Reject a synchronous implementation of the target protocol."""

from avalan.patch.target import (
    CommitUnavailable,
    InspectionBatch,
    InspectionRequest,
    MutationTarget,
    ResolvedMutationScope,
    TargetHandshake,
)


class SynchronousTarget:
    """Deliberately violate every asynchronous target protocol operation."""

    def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return a value synchronously for the negative fixture."""
        raise RuntimeError(scope)

    def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Return a value synchronously for the negative fixture."""
        raise RuntimeError(request)

    def commit(self, request: InspectionRequest) -> CommitUnavailable:
        """Return a value synchronously for the negative fixture."""
        raise RuntimeError(request)


target: MutationTarget = SynchronousTarget()
