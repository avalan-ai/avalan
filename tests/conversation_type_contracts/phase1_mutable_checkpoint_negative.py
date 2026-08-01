"""Reject mutation of immutable checkpoint snapshots."""

from avalan.conversation import (
    CheckpointLifecycle,
    ConversationCheckpoint,
)


def mutate_checkpoint(checkpoint: ConversationCheckpoint) -> None:
    """Attempt to mutate a frozen checkpoint lifecycle."""
    checkpoint.lifecycle = CheckpointLifecycle.DELETED
