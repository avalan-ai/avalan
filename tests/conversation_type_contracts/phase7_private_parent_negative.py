"""Reject private compact state as an ordinary continuation parent."""

from avalan.conversation import (
    StandaloneCompactHandle,
    StatelessConversationSettings,
)


def reject_private_compact_parent(
    handle: StandaloneCompactHandle,
) -> StatelessConversationSettings:
    """Reject a private compact handle at the parent boundary."""
    return StatelessConversationSettings(parent=handle)
