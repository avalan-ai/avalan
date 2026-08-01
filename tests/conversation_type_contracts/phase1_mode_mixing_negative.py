"""Reject mixed parent modes and reset or conversion substitutes."""

from avalan.conversation import (
    CheckpointId,
    ConversationBranchId,
    ConversationId,
    ConversationMode,
    ConversationModeConversion,
    ConversationModeReset,
    ConversationModeTransition,
    StatelessConversationHandle,
    StatelessConversationSettings,
    StatelessParent,
    StoredConversationHandle,
    StoredConversationSettings,
    StoredParent,
)


def accept_mode(value: ConversationMode) -> ConversationMode:
    """Return one closed conversation mode."""
    return value


def require_reset(value: ConversationModeReset) -> ConversationModeReset:
    """Return one explicit reset proof."""
    return value


def reject_handle_as_transition(
    value: ConversationModeTransition,
) -> ConversationModeTransition:
    """Return one explicit reset or conversion proof."""
    return value


def reject_conversion_as_reset(
    conversion: ConversationModeConversion,
) -> ConversationModeReset:
    """Attempt to substitute a conversion where reset proof is required."""
    return require_reset(conversion)


STATELESS_HANDLE = StatelessConversationHandle(
    conversation_id=ConversationId("stateless-conversation"),
    checkpoint_id=CheckpointId("stateless-checkpoint"),
    branch_id=ConversationBranchId("main"),
)
STORED_HANDLE = StoredConversationHandle(
    conversation_id=ConversationId("stored-conversation"),
    checkpoint_id=CheckpointId("stored-checkpoint"),
    branch_id=ConversationBranchId("main"),
)
INVALID_STATELESS = StatelessConversationSettings(
    parent=StoredParent(handle=STORED_HANDLE)
)
INVALID_STORED = StoredConversationSettings(
    provider_storage_disclosed=True,
    parent=StatelessParent(handle=STATELESS_HANDLE),
)
INVALID_MODE = accept_mode("stored")
INVALID_TRANSITION = reject_handle_as_transition(STATELESS_HANDLE)
