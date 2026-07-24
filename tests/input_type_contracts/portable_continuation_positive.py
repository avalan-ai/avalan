"""Lock portable continuation, provider snapshot, and resolver types."""

from typing import assert_type

from avalan.interaction import (
    ContinuationRevisionBinding,
    ContinuationRuntimeResolver,
    ExecutionDefinitionRef,
    PortableContinuation,
    ProviderContinuationSnapshot,
    ProviderContinuationSnapshotAdapter,
    ProviderIdempotencyKey,
    ResolvedContinuationRuntime,
)


async def inspect_portable_continuation(
    continuation: PortableContinuation,
    adapter: ProviderContinuationSnapshotAdapter,
    resolver: ContinuationRuntimeResolver,
    provider_idempotency_key: ProviderIdempotencyKey,
) -> None:
    """Exercise typed durable continuation boundaries."""
    assert_type(continuation.definition, ExecutionDefinitionRef)
    assert_type(
        continuation.revision_binding,
        ContinuationRevisionBinding,
    )
    snapshot = adapter.export_continuation_snapshot(
        revision_binding=continuation.revision_binding,
        model_call_id=continuation.provider_call_id,
        provider_idempotency_key=provider_idempotency_key,
        provider_call_correlation_id=(
            continuation.provider_call_correlation_id
        ),
    )
    assert_type(snapshot, ProviderContinuationSnapshot)
    adapter.import_continuation_snapshot(
        snapshot,
        expected_binding=continuation.revision_binding,
        provider_call_correlation_id=(
            continuation.provider_call_correlation_id
        ),
    )
    assert_type(
        await resolver.resolve(continuation),
        ResolvedContinuationRuntime,
    )
