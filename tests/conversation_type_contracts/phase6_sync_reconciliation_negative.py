"""Reject use of async ambiguity reconciliation as a synchronous result."""

from avalan.conversation import (
    AmbiguousDispatchReconciliationResult,
    AmbiguousDispatchResolution,
    ConversationOperation,
    DirectConversationClient,
    RequestIdempotencyKey,
)


def reject_sync_reconciliation(
    client: DirectConversationClient,
    idempotency_key: RequestIdempotencyKey,
) -> AmbiguousDispatchReconciliationResult:
    """Reject a reconciliation call whose coroutine is not awaited."""
    return client.reconcile_ambiguous_dispatch(
        ConversationOperation.CREATE,
        idempotency_key,
        AmbiguousDispatchResolution.CONFIRMED_NO_DISPATCH,
    )
