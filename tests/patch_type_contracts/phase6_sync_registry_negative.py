"""Reject a synchronous coordinator registry implementation."""

from avalan.patch.coordinator import (
    CoordinatorRegistry,
    Reservation,
    RuntimeIdentity,
)
from avalan.patch.domain import AlgorithmDigest


class SynchronousRegistry:
    """Deliberately violate the coordinator async protocol."""

    def reserve(
        self, identity: RuntimeIdentity, digest: AlgorithmDigest
    ) -> Reservation:
        """Return synchronously for the negative type fixture."""
        raise RuntimeError(identity, digest)


registry: CoordinatorRegistry = SynchronousRegistry()
