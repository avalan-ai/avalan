"""Assert Phase 8 durable store and pending continuation remain async typed."""

from typing import assert_type

from avalan.patch.domain import PatchPending, PatchResult, SequenceNumber
from avalan.patch.durable_coordinator import DurablePatchTestHost
from avalan.patch.durable_outbox import (
    DurableOutboxProjectionReceipt,
    DurableOutboxProjector,
    EventManagerDurableOutboxProjection,
)
from avalan.patch.durable_retention import (
    AesGcmDurableRetentionCipher,
    DurableEncryptedRetention,
    DurableRetentionBinding,
)
from avalan.patch.durable_store import (
    DurablePatchStore,
    DurablePendingAccess,
    DurableRequestAccess,
    DurableRequestSnapshot,
)


async def assert_durable_continuation_types(
    store: DurablePatchStore,
    host: DurablePatchTestHost,
    request_access: DurableRequestAccess,
    pending_access: DurablePendingAccess,
) -> None:
    """Assert continuation methods suspend or resume only typed outcomes."""
    assert_type(await store.inspect(request_access), DurableRequestSnapshot)
    assert_type(await host.inspect(pending_access), PatchPending | PatchResult)
    assert_type(await host.await_resume(pending_access), PatchResult)
    assert_type(await host.resume(pending_access), PatchResult)


async def assert_durable_fault_isolation_types(
    cipher: AesGcmDurableRetentionCipher,
    encrypted: DurableEncryptedRetention,
    binding: DurableRetentionBinding,
    projector: DurableOutboxProjector,
    event_manager_projection: EventManagerDurableOutboxProjection,
    request_access: DurableRequestAccess,
) -> None:
    """Assert dormant encryption and outbox projection retain exact types."""
    assert_type(await cipher.open(encrypted, binding), bytes)
    assert_type(
        await projector.project(request_access, SequenceNumber(0), 1),
        DurableOutboxProjectionReceipt,
    )
    assert_type(
        event_manager_projection,
        EventManagerDurableOutboxProjection,
    )
