"""Reject interchange of distinct agent topology identity types."""

from avalan.conversation import (
    AgentModelSlot,
    AgentTopologyPath,
    ConversationId,
    ProviderFamily,
    ProviderLaneBinding,
    ProviderLaneId,
    ProviderLaneOwnerKind,
    derive_agent_provider_lane_id,
)


def reject_topology_identity_interchange(
    conversation_id: ConversationId,
    binding: ProviderLaneBinding,
) -> ProviderLaneId:
    """Reject swapped topology-path and model-slot identity arguments."""
    assert binding.provider_family is ProviderFamily.SYNTHETIC
    return derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=ProviderLaneOwnerKind.PARENT_AGENT,
        topology_path=AgentModelSlot("primary"),
        model_slot=AgentTopologyPath("agent/parent/primary"),
        binding=binding,
    )
