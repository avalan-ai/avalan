"""Reject synchronous activation registry state and dispatch effects."""

from avalan.conversation import (
    ActivationEvidenceRow,
    ActivationManifest,
    ActivationSnapshot,
    AsyncActivationRegistry,
    CompactionOperation,
    ConversationCapability,
    ConversationMode,
    ProviderLaneBinding,
    ReasoningContext,
)


def reject_sync_load(
    registry: AsyncActivationRegistry,
    manifest: ActivationManifest,
) -> ActivationSnapshot:
    """Reject loading an activation manifest without awaiting it."""
    return registry.load(manifest)


def reject_sync_resolve(
    registry: AsyncActivationRegistry,
    binding: ProviderLaneBinding,
) -> ActivationEvidenceRow:
    """Reject resolving dispatch authority without awaiting it."""
    return registry.resolve(
        binding,
        mode=ConversationMode.STATELESS,
        reasoning_context=ReasoningContext.CURRENT_TURN,
        compaction_operation=CompactionOperation.NONE,
    )


def reject_sync_lifecycle(
    registry: AsyncActivationRegistry,
    binding: ProviderLaneBinding,
) -> ActivationEvidenceRow:
    """Reject resolving lifecycle authority without awaiting it."""
    return registry.resolve_lifecycle(
        binding,
        capability=ConversationCapability.STORED_RESPONSE_DELETION,
    )
