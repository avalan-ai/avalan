"""Assert the Phase 6 coordinator boundary remains typed and asynchronous."""

from typing import assert_type

from avalan.patch.coordinator import (
    CommitLease,
    CoordinatorRegistry,
    IdempotencyStore,
    JournalStore,
    LockFootprint,
    LockLeaseManager,
    Reconciler,
    Reservation,
    RevalidationResult,
    RevalidationSnapshot,
    RuntimeIdentity,
    SettlementJournal,
)
from avalan.patch.domain import AlgorithmDigest


async def assert_coordinator_types(
    registry: CoordinatorRegistry,
    leases: LockLeaseManager,
    store: IdempotencyStore,
    journals: JournalStore,
    identity: RuntimeIdentity,
    digest: AlgorithmDigest,
    reservation: Reservation,
    footprint: LockFootprint,
    snapshot: RevalidationSnapshot,
    reconciler: Reconciler,
    journal: SettlementJournal,
) -> None:
    """Assert coordinator effects remain closed asynchronous values."""
    assert_type(await registry.reserve(identity, digest), Reservation)
    assert_type(await leases.acquire(footprint, reservation), CommitLease)
    assert_type(await reconciler.revalidate(snapshot), RevalidationResult)
    assert_type(await journals.append(reservation.request_id, journal), None)
    assert_type(store, IdempotencyStore)
