from datetime import datetime

from avalan.interaction import (
    ContinuationClaimOwnerId,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationId,
    ContinuationStoreRevision,
    ProviderIdempotencyKey,
)
from avalan.interaction.stores.pgsql import PgsqlInteractionStore


async def reject_interchanged_cas_tokens(
    store: PgsqlInteractionStore,
    continuation_id: ContinuationId,
    revision: ContinuationStoreRevision,
    fence: ContinuationFencingToken,
    now: datetime,
) -> None:
    await store.claim(
        continuation_id,
        expected_store_revision=fence,
        owner_id=ContinuationClaimOwnerId("worker"),
        lease_expires_at=now,
        dispatch_id=ContinuationDispatchId("dispatch"),
        provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
        now=now,
    )
    await store.mark_dispatching(
        continuation_id,
        expected_store_revision=revision,
        owner_id=ContinuationClaimOwnerId("worker"),
        fencing_token=revision,
        now=now,
    )
