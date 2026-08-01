"""Reject interchange between distinct conversation identifiers."""

from avalan.conversation import CheckpointId, ConversationId


def checkpoint_id(value: CheckpointId) -> CheckpointId:
    """Return one statically distinct checkpoint identifier."""
    return value


CONVERSATION_ID = ConversationId("conversation")
INVALID_CHECKPOINT_ID = checkpoint_id(CONVERSATION_ID)
