"""Prove the Phase 4 public direct SDK surface is strictly async and typed."""

from typing import assert_type

from avalan import (
    ConversationBranchIntent,
    ConversationResetIntent,
    DirectConversationClient,
    DirectConversationOutputDelta,
    DirectConversationResult,
    DirectConversationStream,
    DirectConversationStreamItem,
    DirectConversationStreamTerminal,
    ReasoningContext,
    StandaloneCompactRequest,
    StandaloneCompactResult,
    StatelessConversationSettings,
    StatelessParent,
)
from avalan.conversation import ConversationBranchId


async def prove_phase4_direct_sdk(
    client: DirectConversationClient,
    parent: StatelessParent,
    reset: ConversationResetIntent,
    compact_request: StandaloneCompactRequest,
) -> tuple[
    DirectConversationResult,
    DirectConversationStream,
    DirectConversationResult,
    DirectConversationResult,
    DirectConversationResult,
    StandaloneCompactResult,
]:
    """Return each public direct operation through its exact result type."""
    created = assert_type(
        await client.create(
            "create",
            StatelessConversationSettings(
                reasoning_context=ReasoningContext.ALL_TURNS
            ),
        ),
        DirectConversationResult,
    )
    stream = assert_type(
        await client.create(
            "stream",
            StatelessConversationSettings(),
            stream=True,
        ),
        DirectConversationStream,
    )
    async for event in stream:
        assert_type(event, DirectConversationStreamItem)
        if isinstance(event, DirectConversationOutputDelta):
            assert_type(event.text_delta, str)
        else:
            assert_type(event, DirectConversationStreamTerminal)
            assert_type(event.result, DirectConversationResult)
    assert_type(stream.terminal, DirectConversationStreamTerminal)
    continued = assert_type(
        await client.continue_conversation(
            "continue",
            StatelessConversationSettings(parent=parent),
        ),
        DirectConversationResult,
    )
    branch = assert_type(
        await client.branch(
            "branch",
            StatelessConversationSettings(
                parent=parent,
                branch=ConversationBranchIntent(
                    parent=parent,
                    branch_id=ConversationBranchId("typed-branch"),
                ),
            ),
        ),
        DirectConversationResult,
    )
    reset_result = assert_type(
        await client.reset(
            "reset",
            reset,
            StatelessConversationSettings(),
        ),
        DirectConversationResult,
    )
    compact = assert_type(
        await client.compact(compact_request),
        StandaloneCompactResult,
    )
    return created, stream, continued, branch, reset_result, compact
