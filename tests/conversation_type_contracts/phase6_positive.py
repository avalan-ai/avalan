"""Prove the Phase 6 stored lifecycle surface is strictly async and typed."""

from typing import assert_type

from avalan.conversation import (
    AmbiguousDispatchReconciliationResult,
    AmbiguousDispatchResolution,
    ConversationOperation,
    ConversationStore,
    DirectConversationClient,
    DirectConversationResult,
    DirectDeletionResult,
    NativeOpenAIStoredProvider,
    ProviderLifecycleReconciler,
    ProviderQuarantineReceipt,
    ProviderQuarantineRequest,
    ProviderResult,
    PublicResponseId,
    RequestIdempotencyKey,
    RetrievedUpstreamResponse,
    StoredProviderPlan,
    StoredProviderResolver,
    StoredResponseLifecycleAdapter,
    UpstreamDeleteResult,
    UpstreamResponseId,
)


async def prove_phase6_stored_lifecycle(
    provider: NativeOpenAIStoredProvider,
    plan: StoredProviderPlan,
    client: DirectConversationClient,
    resolver: StoredProviderResolver,
    reconciler: ProviderLifecycleReconciler,
    store: ConversationStore,
    quarantine_request: ProviderQuarantineRequest,
    idempotency_key: RequestIdempotencyKey,
    public_response_id: PublicResponseId,
    upstream_response_id: UpstreamResponseId,
) -> tuple[ProviderResult, DirectConversationResult, DirectDeletionResult]:
    """Return exact stored dispatch, retrieval, and deletion result types."""
    dispatched = assert_type(await provider.dispatch(plan), ProviderResult)
    stream = await provider.stream(plan)
    streamed = assert_type(await stream.terminal(), ProviderResult)
    assert_type(await stream.aclose(), None)
    assert_type(
        await provider.retrieve(upstream_response_id),
        RetrievedUpstreamResponse,
    )
    assert_type(
        await provider.delete(upstream_response_id),
        UpstreamDeleteResult,
    )
    assert_type(
        await resolver.resolve(provider.binding.integrity_digest),
        StoredResponseLifecycleAdapter,
    )
    assert_type(
        await resolver.resolve_continuation_runtime(
            provider.binding.integrity_digest
        ),
        object,
    )
    assert_type(
        await store.quarantine_provider_checkpoint(quarantine_request),
        ProviderQuarantineReceipt,
    )
    assert_type(await reconciler.run_once(limit=1), int)
    assert_type(
        await client.reconcile_ambiguous_dispatch(
            ConversationOperation.CREATE,
            idempotency_key,
            AmbiguousDispatchResolution.CONFIRMED_NO_DISPATCH,
        ),
        AmbiguousDispatchReconciliationResult,
    )
    retrieved = assert_type(
        await client.retrieve(public_response_id),
        DirectConversationResult,
    )
    deleted = assert_type(
        await client.delete(public_response_id),
        DirectDeletionResult,
    )
    assert_type(await provider.aclose(), None)
    assert streamed.upstream_response_id is not None
    return dispatched, retrieved, deleted
