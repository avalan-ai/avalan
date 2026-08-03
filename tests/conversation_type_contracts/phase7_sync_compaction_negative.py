"""Reject use of async compaction as a synchronous result."""

from avalan.conversation import (
    DirectConversationClient,
    StandaloneCompactRequest,
    StandaloneCompactResult,
)


def reject_sync_compaction(
    client: DirectConversationClient,
    request: StandaloneCompactRequest,
) -> StandaloneCompactResult:
    """Reject a compact call whose coroutine is not awaited."""
    return client.compact(request)
