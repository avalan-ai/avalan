"""Reject use of async agent continuity as synchronous results."""

from avalan.conversation import (
    AgentConversationResult,
    AgentConversationTurn,
    ConversationCheckpoint,
    ConversationUnitOfWork,
    PortableContinuationReference,
)


def reject_sync_agent_execution(
    turn: AgentConversationTurn,
) -> AgentConversationResult:
    """Reject an agent turn whose coroutine is not awaited."""
    return turn.execute("continue the conversation")


def reject_sync_agent_suspension(
    turn: AgentConversationTurn,
    checkpoint: ConversationCheckpoint,
    continuation: PortableContinuationReference,
) -> ConversationUnitOfWork:
    """Reject suspension staging whose coroutine is not awaited."""
    return turn.stage_structured_input_suspension(checkpoint, continuation)
