"""Prove the compaction surface is strictly async and typed."""

from typing import assert_type

from avalan.conversation import (
    ConversationBranchId,
    DirectConversationClient,
    InlineCompaction,
    NamedHeadId,
    NamedHeadParent,
    NamedHeadRevision,
    NativeOpenAICompactionLimits,
    NativeOpenAIStatelessProvider,
    ProviderResult,
    StandaloneCompactProviderPlan,
    StandaloneCompactRequest,
    StandaloneCompactResult,
    StatelessConversationHandle,
)


async def prove_compaction_contract(
    client: DirectConversationClient,
    provider: NativeOpenAIStatelessProvider,
    request: StandaloneCompactRequest,
    plan: StandaloneCompactProviderPlan,
    branch_id: ConversationBranchId,
) -> tuple[
    ProviderResult,
    StandaloneCompactResult,
    StatelessConversationHandle,
    StatelessConversationHandle,
]:
    """Return exact provider, compact, commit, and fork result types."""
    inline = assert_type(
        InlineCompaction(compact_threshold=1), InlineCompaction
    )
    limits = assert_type(
        NativeOpenAICompactionLimits(
            min_compact_threshold=1,
            max_compact_threshold=2_147_483_647,
        ),
        NativeOpenAICompactionLimits,
    )
    assert inline.compact_threshold >= limits.min_compact_threshold
    provider_result = assert_type(await provider.compact(plan), ProviderResult)
    compacted = assert_type(
        await client.compact(request),
        StandaloneCompactResult,
    )
    assert_type(request.named_head, NamedHeadParent | None)
    assert_type(compacted.handle.head_id, NamedHeadId | None)
    assert_type(
        compacted.handle.expected_head_revision,
        NamedHeadRevision | None,
    )
    committed = assert_type(
        await client.commit_compact(compacted),
        StatelessConversationHandle,
    )
    forked = assert_type(
        await client.fork_compact(compacted, branch_id),
        StatelessConversationHandle,
    )
    return provider_result, compacted, committed, forked
