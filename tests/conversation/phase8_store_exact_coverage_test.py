"""Close defensive Phase 8 conversation store coverage gaps."""

from dataclasses import replace
from typing import Any, cast

import pytest
from agent_integration_contract_test import (
    _execution_segment,
    _topology,
    _topology_checkpoint,
)
from phase2_fixtures import empty_stateless_plan

import avalan.conversation as conversation


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 8 async coverage checks on asyncio."""
    return "asyncio"


def _candidate(
    checkpoint: conversation.ConversationCheckpoint,
) -> conversation.ExecutionSegmentCheckpointCandidate:
    """Wrap a defensive checkpoint fixture in an accepted candidate type."""
    candidate = object.__new__(
        conversation.ExecutionSegmentCheckpointCandidate
    )
    object.__setattr__(candidate, "checkpoint", checkpoint)
    return candidate


def _output_candidate(
    checkpoint: conversation.ConversationCheckpoint,
    lane: conversation.AgentProviderLane,
) -> conversation.ProviderLaneOutputCandidate:
    """Return one exact candidate for a topology lane absent from state."""
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane.binding),
        turn=1,
    )
    receipt = conversation.provider_lane_execution_receipt(
        authority=checkpoint.authority,
        identity=checkpoint.identity,
        binding=lane.binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    return conversation.ProviderLaneOutputCandidate(
        lane_id=lane.lane_id,
        binding=lane.binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        execution_receipt=receipt,
    )


@pytest.mark.anyio
async def test_store_recovery_admission_rejects_untyped_values() -> None:
    """Reject invalid recovery inputs before any in-memory or SQL access."""
    invalid = cast(Any, object())
    memory = conversation.InMemoryConversationStore()
    with pytest.raises(conversation.ConversationValidationError):
        await memory.admit_tool_recovery(invalid, invalid)

    pgsql = object.__new__(conversation.PgsqlConversationStore)
    with pytest.raises(conversation.ConversationValidationError):
        await pgsql.admit_tool_recovery(invalid, invalid)


def test_candidate_authority_and_topology_guards_fail_closed() -> None:
    """Reject authority, segment-agent, and binding topology drift."""
    topology = _topology()
    child = topology.lanes[1]
    checkpoint = _topology_checkpoint(child, topology)
    unauthorized = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            authority=replace(
                checkpoint.authority,
                agent_id=conversation.ConversationAgentId("outsider"),
            ),
        )
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        conversation.InMemoryConversationStore._candidate_checkpoint(
            _candidate(unauthorized)
        )

    segment = _execution_segment()
    parent_entry = topology.checkpoint_topology().entries[0]
    segment_entry = replace(
        parent_entry,
        lane_id=segment.lane_id,
        topology_path="agent/agent-parent/model/segment",
        binding_digest=segment.binding.integrity_digest,
    )
    segment_topology = conversation.ProviderLaneTopology(
        schema_version=1,
        entries=(segment_entry,),
    )
    base = conversation.with_checkpoint_integrity(
        conversation.ConversationCheckpoint(
            identity=checkpoint.identity,
            kind=checkpoint.kind,
            lifecycle=checkpoint.lifecycle,
            authority=checkpoint.authority,
            content=conversation.MultiLaneCheckpointContent(
                visible_transcript=checkpoint.content.visible_transcript,
                lanes=(),
                execution_segments=(segment,),
                lane_topology=segment_topology,
            ),
            timestamps=checkpoint.timestamps,
            retention=checkpoint.retention,
        )
    )
    unauthorized_segment = replace(
        segment,
        binding=replace(
            segment.binding,
            agent_id=conversation.ConversationAgentId("outsider"),
        ),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        conversation.InMemoryConversationStore._candidate_checkpoint(
            _candidate(
                conversation.with_checkpoint_integrity(
                    replace(
                        base,
                        content=replace(
                            base.content,
                            execution_segments=(unauthorized_segment,),
                        ),
                    )
                )
            )
        )

    drifted_topology = replace(
        segment_topology,
        entries=(
            replace(
                segment_entry,
                binding_digest=conversation.IntegrityDigest("0" * 64),
            ),
        ),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        conversation.InMemoryConversationStore._candidate_checkpoint(
            _candidate(
                conversation.with_checkpoint_integrity(
                    replace(
                        base,
                        content=replace(
                            base.content,
                            lane_topology=drifted_topology,
                        ),
                    )
                )
            )
        )


def test_discarded_lane_output_requires_topology_and_exact_receipt() -> None:
    """Authenticate terminal-discard candidates against topology receipts."""
    topology = _topology()
    parent, retained_child, discarded_child = topology.lanes

    child_checkpoint = _topology_checkpoint(retained_child, topology)
    parent_output = _output_candidate(child_checkpoint, parent)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            child_checkpoint,
            (parent_output,),
            parent=child_checkpoint,
        )

    parent_checkpoint = _topology_checkpoint(parent, topology)
    child_output = _output_candidate(parent_checkpoint, discarded_child)
    wrong_receipt = conversation.ProviderLaneExecutionReceipt(
        schema_version=1,
        digest=conversation.IntegrityDigest("0" * 64),
        item_count=child_output.execution_receipt.item_count,
        opaque_byte_count=child_output.execution_receipt.opaque_byte_count,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.InMemoryConversationStore._validate_output_candidates(
            parent_checkpoint,
            (replace(child_output, execution_receipt=wrong_receipt),),
            parent=parent_checkpoint,
        )
