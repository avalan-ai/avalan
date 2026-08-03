"""Prove the Phase 8 agent-continuity surface is strictly async and typed."""

from typing import assert_type

from avalan.conversation import (
    AgentConversationResult,
    AgentConversationTurn,
    AgentLaneTopology,
    AgentModelSlot,
    AgentTopologyPath,
    ConversationAgentId,
    ConversationCheckpoint,
    ConversationSurface,
    ConversationUnitOfWork,
    PortableContinuationReference,
    SurfaceDisposition,
    agent_conversation_surface_disposition,
    agent_topology_digest,
    child_agent_topology_path,
    direct_model_topology_path,
    parent_agent_topology_path,
    require_agent_conversation_surface,
)


async def prove_phase8_agent_continuity(
    turn: AgentConversationTurn,
    topology: AgentLaneTopology,
    checkpoint: ConversationCheckpoint,
    continuation: PortableContinuationReference,
    surface: ConversationSurface,
) -> tuple[AgentConversationResult, ConversationUnitOfWork, str]:
    """Return exact agent execution, suspension, and topology types."""
    result = assert_type(
        await turn.execute("continue the conversation"),
        AgentConversationResult,
    )
    unit_of_work = assert_type(
        await turn.stage_structured_input_suspension(
            checkpoint,
            continuation,
        ),
        ConversationUnitOfWork,
    )
    assert_type(
        agent_conversation_surface_disposition(surface),
        SurfaceDisposition,
    )
    require_agent_conversation_surface(surface)
    parent_path = assert_type(
        parent_agent_topology_path(
            ConversationAgentId("parent"),
            AgentModelSlot("primary"),
        ),
        AgentTopologyPath,
    )
    assert_type(
        child_agent_topology_path(
            parent_path,
            ConversationAgentId("child"),
            AgentModelSlot("research"),
        ),
        AgentTopologyPath,
    )
    assert_type(
        direct_model_topology_path(AgentModelSlot("primary")),
        AgentTopologyPath,
    )
    digest = assert_type(agent_topology_digest(topology), str)
    return result, unit_of_work, digest
