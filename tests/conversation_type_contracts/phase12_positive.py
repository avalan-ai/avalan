"""Prove activation registry state and dispatch effects remain asynchronous."""

from typing import assert_type

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


async def prove_phase12_activation(
    registry: AsyncActivationRegistry,
    manifest: ActivationManifest,
    binding: ProviderLaneBinding,
) -> tuple[
    ActivationSnapshot,
    ActivationSnapshot,
    ActivationEvidenceRow,
    ActivationEvidenceRow,
]:
    """Return exact loaded, applied, dispatch, and lifecycle types."""
    loaded = assert_type(await registry.load(manifest), ActivationSnapshot)
    applied = assert_type(
        await registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        ),
        ActivationSnapshot,
    )
    dispatch = assert_type(
        await registry.resolve(
            binding,
            mode=ConversationMode.STATELESS,
            reasoning_context=ReasoningContext.CURRENT_TURN,
            compaction_operation=CompactionOperation.NONE,
        ),
        ActivationEvidenceRow,
    )
    lifecycle = assert_type(
        await registry.resolve_lifecycle(
            binding,
            capability=ConversationCapability.STORED_RESPONSE_DELETION,
        ),
        ActivationEvidenceRow,
    )
    return loaded, applied, dispatch, lifecycle
