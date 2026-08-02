"""Reject a synchronous Phase 6 stored lifecycle effect."""

from datetime import UTC, datetime

from avalan.conversation import (
    ProviderLaneBinding,
    RetrievedUpstreamResponse,
    StoredProviderResolverEntry,
    UpstreamDeleteResult,
    UpstreamResponseId,
)


class SynchronousLifecycleAdapter:
    """Expose an invalid synchronous upstream retrieval operation."""

    def __init__(self, binding: ProviderLaneBinding) -> None:
        self.binding = binding

    def retrieve(
        self,
        upstream_response_id: UpstreamResponseId,
    ) -> RetrievedUpstreamResponse:
        """Return upstream metadata without an awaitable boundary."""
        raise RuntimeError(upstream_response_id)

    async def delete(
        self,
        upstream_response_id: UpstreamResponseId,
    ) -> UpstreamDeleteResult:
        """Delete one upstream response asynchronously."""
        raise RuntimeError(upstream_response_id)


def reject_sync_lifecycle(binding: ProviderLaneBinding) -> None:
    """Reject the synchronous adapter at the resolver boundary."""
    StoredProviderResolverEntry(
        adapter=SynchronousLifecycleAdapter(binding),
        revision="phase6-sync-negative",
        valid_from=datetime(2026, 8, 2, tzinfo=UTC),
    )
