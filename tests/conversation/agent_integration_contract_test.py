"""Verify immutable agent conversation topology and segment contracts."""

from asyncio import gather
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from agent_integration_e2e_test import _public_runtime

import avalan.conversation as conversation
from avalan.conversation.state import validate_checkpoint_parent_kind

_NOW = datetime(2034, 1, 1, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    """Run async contract checks on the installed asyncio backend."""
    return "asyncio"


def _binding(
    agent_id: str,
    *,
    lane_id: str = "lane-placeholder",
    model: str = "model-parent",
    endpoint: str = "https://provider.agent.test/v1",
) -> conversation.ProviderLaneBinding:
    """Return one exact synthetic agent-lane binding."""
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type="tests.AgentConversationProvider",
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint=endpoint,
        model_or_deployment=model,
        provider_api_revision=conversation.ProviderApiRevision("api-agent"),
        sdk_revision=conversation.ProviderSdkRevision("sdk-agent"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision(f"config-{model}")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-agent")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-agent"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-agent")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=conversation.ConversationAgentId(agent_id),
    )


def _lane(
    *,
    conversation_id: conversation.ConversationId,
    owner_kind: conversation.ProviderLaneOwnerKind,
    agent_id: str,
    model_slot: str,
    topology_path: conversation.AgentTopologyPath,
    parent_lane_id: conversation.ProviderLaneId | None = None,
    model: str,
    retention: conversation.ChildLaneRetentionPolicy,
) -> conversation.AgentProviderLane:
    """Return one binding whose lane ID is derived from exact topology."""
    binding = _binding(agent_id, model=model)
    lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=owner_kind,
        topology_path=topology_path,
        model_slot=conversation.AgentModelSlot(model_slot),
        binding=binding,
    )
    return conversation.AgentProviderLane(
        owner_kind=owner_kind,
        agent_id=conversation.ConversationAgentId(agent_id),
        topology_path=topology_path,
        model_slot=conversation.AgentModelSlot(model_slot),
        binding=replace(binding, lane_id=lane_id),
        retention_policy=retention,
        parent_lane_id=parent_lane_id,
    )


def _topology() -> conversation.AgentLaneTopology:
    conversation_id = conversation.ConversationId("conversation-agent")
    parent_path = conversation.parent_agent_topology_path(
        conversation.ConversationAgentId("agent-parent"),
        conversation.AgentModelSlot("primary"),
    )
    parent = _lane(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        agent_id="agent-parent",
        model_slot="primary",
        topology_path=parent_path,
        model="model-parent",
        retention=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    first_child = _lane(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        agent_id="agent-child-a",
        model_slot="research",
        topology_path=conversation.child_agent_topology_path(
            parent_path,
            conversation.ConversationAgentId("agent-child-a"),
            conversation.AgentModelSlot("research"),
        ),
        parent_lane_id=parent.lane_id,
        model="model-child-a",
        retention=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    second_child = _lane(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        agent_id="agent-child-b",
        model_slot="critic",
        topology_path=conversation.child_agent_topology_path(
            parent_path,
            conversation.ConversationAgentId("agent-child-b"),
            conversation.AgentModelSlot("critic"),
        ),
        parent_lane_id=parent.lane_id,
        model="model-child-b",
        retention=conversation.ChildLaneRetentionPolicy.DISCARD_TERMINAL,
    )
    return conversation.AgentLaneTopology(
        conversation_id=conversation_id,
        lanes=(parent, first_child, second_child),
    )


def _function_call(
    lane_id: conversation.ProviderLaneId,
) -> conversation.ProviderItem:
    """Return one canonical native function-call item."""
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId("provider-call-item"),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId("model-call-1"),
        kind=conversation.ProviderItemKind.FUNCTION_CALL,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={
            "arguments": '{"secret":"tool-secret"}',
            "call_id": "provider-call-1",
            "name": "lookup",
            "type": "function_call",
        },
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        call_id=conversation.ProviderCallId("provider-call-1"),
    )


def _execution_segment() -> conversation.ProviderExecutionSegment:
    """Return one exact pre-effect stateless execution segment."""
    binding = _binding("agent-parent", lane_id="lane-segment")
    call = _function_call(binding.lane_id)
    return conversation.ProviderExecutionSegment(
        schema_version=1,
        idempotency_key=conversation.RequestIdempotencyKey("turn-key"),
        request_digest=conversation.CanonicalRequestDigest("request-digest"),
        binding=binding,
        mode=conversation.ConversationMode.STATELESS,
        segment_index=0,
        phase=conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        items=(call,),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.CURRENT_TURN,
            effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
        ),
        usage=conversation.ProviderUsage(input_tokens=4, output_tokens=2),
        tools=(
            conversation.ProviderToolExecution(
                call_id=conversation.ProviderCallId("provider-call-1"),
                arguments={"secret": "tool-secret"},
                tool_revision=conversation.ToolSchemaRevision("lookup-v2"),
                effect_policy=conversation.ToolEffectPolicy.IDEMPOTENT,
                phase=conversation.ToolExecutionPhase.REQUESTED,
                idempotency_key="tool-call-key",
            ),
        ),
    )


def _durable_tool_segments() -> tuple[
    conversation.ProviderExecutionSegment,
    conversation.ProviderExecutionSegment,
    conversation.ProviderExecutionSegment,
]:
    """Return requested, output-persisted, and internally complete states."""
    requested = _execution_segment()
    requested_tool = requested.tools[0]
    output_item = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("provider-tool-output"),
        lane_id=requested.lane_id,
        model_call_id=requested.items[0].model_call_id,
        kind=conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        order=conversation.ProviderItemOrder(1),
        provider_index=conversation.ProviderItemIndex(1),
        phase=conversation.ProviderItemPhase.TOOL,
        caller=conversation.ProviderItemCaller.TOOL,
        canonical_input={
            "call_id": requested_tool.call_id,
            "output": "canonical output",
            "type": "function_call_output",
        },
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        call_id=requested_tool.call_id,
    )
    output = replace(
        requested,
        phase=conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
        items=(*requested.items, output_item),
        tools=(
            replace(
                requested_tool,
                phase=conversation.ToolExecutionPhase.OUTPUT_PERSISTED,
                output_id=output_item.item_id,
            ),
        ),
    )
    complete = replace(
        requested,
        segment_index=1,
        tools=(),
    )
    return requested, output, complete


def _internal_checkpoint(
    segment: conversation.ProviderExecutionSegment,
) -> conversation.ConversationCheckpoint:
    """Return one private committed checkpoint containing a segment."""
    return conversation.with_checkpoint_integrity(
        conversation.ConversationCheckpoint(
            identity=conversation.CheckpointIdentity(
                conversation_id=conversation.ConversationId(
                    "conversation-segment"
                ),
                logical_turn_id=conversation.LogicalTurnId("turn-segment"),
                execution_segment_id=conversation.ExecutionSegmentId(
                    "execution-segment"
                ),
                checkpoint_id=conversation.CheckpointId("checkpoint-segment"),
                branch_id=conversation.ConversationBranchId("branch-segment"),
                sequence=conversation.CheckpointSequence(0),
            ),
            kind=conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            lifecycle=conversation.CheckpointLifecycle.COMMITTED,
            authority=conversation.AuthorityScope(
                source=(
                    conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT
                ),
                tenant_id=conversation.AuthorityTenantId("tenant-agent"),
                principal_id=conversation.AuthorityPrincipalId(
                    "principal-agent"
                ),
                agent_id=conversation.ConversationAgentId("agent-parent"),
                endpoint_id=conversation.AuthorityEndpointId("endpoint-agent"),
            ),
            content=conversation.MultiLaneCheckpointContent(
                visible_transcript=conversation.VisibleTranscript(entries=()),
                lanes=(),
                execution_segments=(segment,),
            ),
            timestamps=conversation.CheckpointTimestamps(
                created_at=_NOW,
                committed_at=_NOW,
                expires_at=_NOW + timedelta(hours=1),
            ),
            retention=conversation.RetentionLimits(
                storage=conversation.StoragePolicy(
                    local=conversation.LocalResponseStorage.DURABLE,
                    upstream=conversation.ProviderLaneStorage.STATELESS,
                ),
                upstream_lifetime_status=(
                    conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
                ),
                local_ttl_seconds=3_600,
            ),
        )
    )


def _topology_checkpoint(
    lane: conversation.AgentProviderLane,
    topology: conversation.AgentLaneTopology,
    *,
    persisted_topology: conversation.ProviderLaneTopology | None = None,
) -> conversation.ConversationCheckpoint:
    """Return one staged private checkpoint for a topology lane."""
    return conversation.with_checkpoint_integrity(
        conversation.ConversationCheckpoint(
            identity=conversation.CheckpointIdentity(
                conversation_id=topology.conversation_id,
                logical_turn_id=conversation.LogicalTurnId("topology-turn"),
                execution_segment_id=conversation.ExecutionSegmentId(
                    "topology-segment"
                ),
                checkpoint_id=conversation.CheckpointId(
                    f"topology-checkpoint-{lane.model_slot}"
                ),
                branch_id=conversation.ConversationBranchId("topology-branch"),
                sequence=conversation.CheckpointSequence(0),
            ),
            kind=conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
            authority=conversation.AuthorityScope(
                source=(
                    conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT
                ),
                tenant_id=conversation.AuthorityTenantId("tenant-agent"),
                principal_id=conversation.AuthorityPrincipalId(
                    "principal-agent"
                ),
                agent_id=conversation.ConversationAgentId("agent-parent"),
                endpoint_id=conversation.AuthorityEndpointId("endpoint-agent"),
            ),
            content=conversation.MultiLaneCheckpointContent(
                visible_transcript=conversation.VisibleTranscript(entries=()),
                lanes=(
                    conversation.StatelessProviderLaneSnapshot(
                        binding=lane.binding,
                        ledger=conversation.ProviderItemLedger(
                            lane_id=lane.lane_id,
                            normalization_version=(
                                lane.binding.continuation_codec_version
                            ),
                            items=(),
                        ),
                        reasoning=conversation.EffectiveReasoningMetadata(
                            requested=conversation.ReasoningContext.AUTO,
                            effective=None,
                        ),
                        lifecycle=(
                            conversation.ProviderLaneLifecycle.COMMITTED
                        ),
                        retention_policy=lane.retention_policy,
                    ),
                ),
                lane_topology=(
                    topology.checkpoint_topology()
                    if persisted_topology is None
                    else persisted_topology
                ),
            ),
            timestamps=conversation.CheckpointTimestamps(
                created_at=_NOW,
                expires_at=_NOW + timedelta(hours=1),
            ),
            retention=conversation.RetentionLimits(
                storage=conversation.StoragePolicy(
                    local=conversation.LocalResponseStorage.DURABLE,
                    upstream=conversation.ProviderLaneStorage.STATELESS,
                ),
                upstream_lifetime_status=(
                    conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
                ),
                local_ttl_seconds=3_600,
            ),
        )
    )


def test_agent_lane_topology_is_deterministic_and_isolated() -> None:
    """Derive stable distinct lanes and record child retention explicitly."""
    topology = _topology()
    repeated = _topology()

    assert topology == repeated
    assert len(topology.parent_lanes) == 1
    assert len(topology.child_lanes) == 2
    assert len({lane.lane_id for lane in topology.lanes}) == 3
    assert tuple(lane.retention_policy for lane in topology.child_lanes) == (
        conversation.ChildLaneRetentionPolicy.RETAIN,
        conversation.ChildLaneRetentionPolicy.DISCARD_TERMINAL,
    )
    assert conversation.agent_topology_digest(topology) == (
        conversation.agent_topology_digest(repeated)
    )
    assert topology.parent_lanes[0].lane_id.startswith("lane-v1-")


def test_child_results_merge_only_canonical_visible_entries() -> None:
    """Merge child outputs by topology without accepting parent lane state."""
    topology = _topology()
    first, second = topology.child_lanes
    first_entry = conversation.VisibleTranscriptEntry(
        role=conversation.VisibleTranscriptRole.ASSISTANT,
        content="safe child A result",
    )
    second_entry = conversation.VisibleTranscriptEntry(
        role=conversation.VisibleTranscriptRole.ASSISTANT,
        content="safe child B result",
    )

    assert topology.outward_child_results(
        {
            second.lane_id: (second_entry,),
            first.lane_id: (first_entry,),
        }
    ) == (first_entry, second_entry)
    with pytest.raises(conversation.ConversationValidationError):
        topology.outward_child_results(
            {topology.parent_lanes[0].lane_id: (first_entry,)}
        )


def test_agent_topology_rejects_wrong_parent_drift_and_duplicate_lanes() -> (
    None
):
    """Fail closed for topology, binding, and ownership drift."""
    topology = _topology()
    parent, first, second = topology.lanes

    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentLaneTopology(
            conversation_id=topology.conversation_id,
            lanes=(parent, first, replace(second, binding=first.binding)),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentLaneTopology(
            conversation_id=conversation.ConversationId("wrong-conversation"),
            lanes=topology.lanes,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentLaneTopology(
            conversation_id=topology.conversation_id,
            lanes=(parent, replace(first, parent_lane_id=second.lane_id)),
        )


def test_agent_conversation_surfaces_are_explicit_and_fail_closed(
    record_property: Callable[[str, object], None],
) -> None:
    """Activate SDK and served Responses while other surfaces reject."""
    record_property(
        "conversation_acceptance_evidence", "pre_dispatch_rejection"
    )
    assert (
        conversation.agent_conversation_surface_disposition(
            conversation.ConversationSurface.AGENT_SDK
        )
        is conversation.SurfaceDisposition.ACTIVATED
    )
    conversation.require_agent_conversation_surface(
        conversation.ConversationSurface.AGENT_SDK
    )
    assert (
        conversation.agent_conversation_surface_disposition(
            conversation.ConversationSurface.SERVED_RESPONSES
        )
        is conversation.SurfaceDisposition.ACTIVATED
    )
    conversation.require_agent_conversation_surface(
        conversation.ConversationSurface.SERVED_RESPONSES
    )
    for surface in (
        conversation.ConversationSurface.CLI,
        conversation.ConversationSurface.FLOW,
        conversation.ConversationSurface.MCP,
        conversation.ConversationSurface.A2A,
    ):
        assert (
            conversation.agent_conversation_surface_disposition(surface)
            is conversation.SurfaceDisposition.DEFERRED
        )
        with pytest.raises(conversation.ConversationCapabilityError):
            conversation.require_agent_conversation_surface(surface)


def test_execution_segment_codec_round_trips_exact_private_recovery_state() -> (  # noqa: E501
    None
):
    """Round-trip exact provider/tool adjacency without repr disclosure."""
    segment = _execution_segment()
    checkpoint = _internal_checkpoint(segment)
    codec = conversation.ConversationCheckpointCodec()

    encoded = codec.encode(checkpoint)
    decoded = codec.decode(encoded)

    assert decoded == checkpoint
    assert decoded.content.execution_segments == (segment,)
    assert b'"execution_segments"' in encoded
    assert "tool-secret" not in repr(segment)
    assert "tool-secret" not in repr(segment.tools[0])
    assert "provider-call-1" not in repr(segment.tools[0])


def test_execution_segment_rejects_corrupt_tool_adjacency_and_phase() -> None:
    """Reject missing call correlation and output phase drift."""
    segment = _execution_segment()
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            segment,
            tools=(
                replace(
                    segment.tools[0],
                    call_id=conversation.ProviderCallId("missing-call"),
                ),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            segment,
            phase=conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.MultiLaneCheckpointContent(
            visible_transcript=conversation.VisibleTranscript(entries=()),
            lanes=(),
        )


def test_durable_tool_crash_points_have_one_safe_recovery_action(
    record_property: Callable[[str, object], None],
) -> None:
    """Map all four crash points to exact effect-safe advancement."""
    record_property("conversation_acceptance_evidence", "contract")
    requested, output, complete = _durable_tool_segments()
    pure = replace(
        requested,
        tools=(
            replace(
                requested.tools[0],
                effect_policy=conversation.ToolEffectPolicy.PURE,
                idempotency_key=None,
            ),
        ),
    )
    fenced = replace(
        requested,
        tools=(
            replace(
                requested.tools[0],
                effect_policy=(
                    conversation.ToolEffectPolicy.FENCED_UNPROTECTED
                ),
                idempotency_key=None,
            ),
        ),
    )

    assert conversation.durable_tool_recovery_action((pure,)) is (
        conversation.DurableToolRecoveryAction.REEXECUTE_PURE
    )
    assert conversation.durable_tool_recovery_action((requested,)) is (
        conversation.DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT
    )
    assert conversation.durable_tool_recovery_action((fenced,)) is (
        conversation.DurableToolRecoveryAction.REQUIRE_RECONCILIATION
    )
    assert (
        conversation.durable_tool_recovery_action((requested, output))
        is conversation.DurableToolRecoveryAction.RESUME_PROVIDER
    )
    assert (
        conversation.durable_tool_recovery_action(
            (
                requested,
                output,
                complete,
            )
        )
        is conversation.DurableToolRecoveryAction.COMMIT_OUTWARD
    )


def test_durable_tool_recovery_rejects_corrupt_or_reordered_suffix() -> None:
    """Reject missing, reordered, wrong-revision, and duplicate segments."""
    requested, output, _ = _durable_tool_segments()
    wrong_revision = replace(
        output,
        tools=(
            replace(
                output.tools[0],
                tool_revision=conversation.ToolSchemaRevision("wrong-tool"),
            ),
        ),
    )
    cases = (
        (),
        (output,),
        (requested, requested),
        (requested, replace(output, tools=())),
        (requested, wrong_revision),
        (
            requested,
            replace(
                output,
                request_digest=conversation.CanonicalRequestDigest(
                    "wrong-request"
                ),
            ),
        ),
    )
    for segments in cases:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.durable_tool_recovery_action(segments)


@pytest.mark.anyio
async def test_in_memory_recovery_admission_is_exact_and_single_owner() -> (
    None
):
    """Lease one immutable fenced suffix and reject every competing owner."""
    segment = _execution_segment()
    template = _internal_checkpoint(segment)
    staged = conversation.with_checkpoint_integrity(
        replace(
            template,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
            timestamps=replace(template.timestamps, committed_at=None),
            integrity=None,
        )
    )
    store = conversation.InMemoryConversationStore()
    committed = await store.commit(
        conversation.ExecutionSegmentCheckpointCandidate(checkpoint=staged)
    )
    idempotency = conversation.RequestIdempotencyIdentity(
        authority=committed.authority,
        operation=conversation.ConversationOperation.CREATE,
        key=segment.idempotency_key,
        request_digest=segment.request_digest,
    )
    execution = conversation.ConversationExecutionReservation(
        idempotency=idempotency,
        identity=conversation.CheckpointIdentity(
            conversation_id=committed.identity.conversation_id,
            logical_turn_id=committed.identity.logical_turn_id,
            execution_segment_id=conversation.ExecutionSegmentId(
                "outward-segment"
            ),
            checkpoint_id=conversation.CheckpointId("outward-checkpoint"),
            branch_id=conversation.ConversationBranchId("outward-branch"),
            sequence=conversation.CheckpointSequence(0),
        ),
        lanes=(
            conversation.ProviderLaneExecutionReservation(
                binding=segment.binding,
                mode=segment.mode,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            ),
        ),
    )
    reservation = await store.reserve_idempotency(
        idempotency,
        execution=execution,
    )
    assert reservation.owner_token is not None
    await store.fence_idempotency(
        idempotency,
        reservation.owner_token,
        ambiguous=True,
    )
    assert committed.integrity is not None
    admission = conversation.DurableToolRecoveryAdmission(
        checkpoint_id=committed.identity.checkpoint_id,
        checkpoint_integrity=committed.integrity.digest,
        idempotency=idempotency,
        binding=segment.binding,
        action=(conversation.DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT),
        segment_count=1,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.admit_tool_recovery(
            replace(
                admission,
                checkpoint_integrity=conversation.IntegrityDigest("0" * 64),
            ),
            execution,
        )
    with pytest.raises(conversation.ConversationConflictError):
        await store.admit_tool_recovery(
            replace(admission, segment_count=2),
            execution,
        )

    outcomes = await gather(
        store.admit_tool_recovery(admission, execution),
        store.admit_tool_recovery(admission, execution),
        return_exceptions=True,
    )

    leases = tuple(
        outcome
        for outcome in outcomes
        if type(outcome) is conversation.DurableToolRecoveryLease
    )
    conflicts = tuple(
        outcome
        for outcome in outcomes
        if type(outcome) is conversation.ConversationConflictError
    )
    assert len(leases) == 1
    assert len(conflicts) == 1
    lease = leases[0]
    assert isinstance(lease, conversation.DurableToolRecoveryLease)
    settlement = await store.inspect_idempotency_settlement(
        idempotency,
        lease.owner_token,
    )
    assert settlement.disposition is (
        conversation.IdempotencySettlementDisposition.LEASED
    )


def test_checkpoint_codec_omits_new_optional_field_for_legacy_content() -> (
    None
):
    """Keep pre-Phase-8 canonical checkpoint bytes free of empty fields."""
    segment = _execution_segment()
    lane = conversation.StatelessProviderLaneSnapshot(
        binding=segment.binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=segment.lane_id,
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=(),
        ),
        reasoning=segment.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    checkpoint = replace(
        _internal_checkpoint(segment),
        kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
        content=conversation.MultiLaneCheckpointContent(
            visible_transcript=conversation.VisibleTranscript(entries=()),
            lanes=(lane,),
        ),
        integrity=None,
    )
    encoded = conversation.ConversationCheckpointCodec().encode(
        conversation.with_checkpoint_integrity(checkpoint)
    )

    assert b'"execution_segments"' not in encoded


@pytest.mark.anyio
async def test_child_topology_is_persisted_and_authorizes_exact_binding() -> (
    None
):
    """Persist child ownership while rejecting topology binding drift."""
    topology = _topology()
    child = topology.child_lanes[0]
    checkpoint = _topology_checkpoint(child, topology)
    codec = conversation.ConversationCheckpointCodec()

    decoded = codec.decode(codec.encode(checkpoint))
    assert decoded.content.lane_topology == topology.checkpoint_topology()
    assert decoded.content.lane_topology is not None
    assert decoded.content.lane_topology.entry(child.lane_id).agent_id == (
        child.agent_id
    )
    store = conversation.InMemoryConversationStore()
    committed = await store.commit(
        conversation.ExecutionSegmentCheckpointCandidate(checkpoint=checkpoint)
    )
    assert committed.content.lanes[0].binding.agent_id == "agent-child-a"

    entries = topology.checkpoint_topology().entries
    drifted = conversation.ProviderLaneTopology(
        schema_version=1,
        entries=tuple(
            (
                replace(
                    entry,
                    binding_digest=conversation.IntegrityDigest("0" * 64),
                )
                if entry.lane_id == child.lane_id
                else entry
            )
            for entry in entries
        ),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await conversation.InMemoryConversationStore().commit(
            conversation.ExecutionSegmentCheckpointCandidate(
                checkpoint=_topology_checkpoint(
                    child,
                    topology,
                    persisted_topology=drifted,
                )
            )
        )


@pytest.mark.anyio
async def test_parent_kind_policy_rejects_agent_coordinator_and_store_bypasses(
    record_property: Callable[[str, object], None],
) -> None:
    """Allow private children while rejecting private public parents."""
    record_property("conversation_acceptance_evidence", "security")
    validate_checkpoint_parent_kind(
        conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
        conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
    )
    validate_checkpoint_parent_kind(
        conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
        conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
        compact_continuation=True,
    )
    validate_checkpoint_parent_kind(
        conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
        None,
    )
    for private_parent in (
        conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
        conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
    ):
        with pytest.raises(conversation.ConversationTransitionError):
            validate_checkpoint_parent_kind(
                conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
                private_parent,
            )

    _, turn, store, coordinator, _ = _public_runtime()
    staged_internal = conversation.with_checkpoint_integrity(
        conversation.ConversationCheckpoint(
            identity=conversation.CheckpointIdentity(
                conversation_id=turn.conversation_id,
                logical_turn_id=turn.logical_turn_id,
                execution_segment_id=conversation.ExecutionSegmentId(
                    "parent-kind-internal-segment"
                ),
                checkpoint_id=conversation.CheckpointId(
                    "parent-kind-internal"
                ),
                branch_id=turn.branch_id,
                sequence=conversation.CheckpointSequence(0),
            ),
            kind=conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
            authority=turn.authority,
            content=conversation.MultiLaneCheckpointContent(
                visible_transcript=conversation.VisibleTranscript(entries=()),
                lanes=tuple(
                    conversation.StatelessProviderLaneSnapshot(
                        binding=lane.binding,
                        ledger=conversation.ProviderItemLedger(
                            lane_id=lane.lane_id,
                            normalization_version=(
                                lane.binding.continuation_codec_version
                            ),
                            items=(),
                        ),
                        reasoning=conversation.EffectiveReasoningMetadata(
                            requested=conversation.ReasoningContext.AUTO,
                            effective=None,
                        ),
                        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
                        retention_policy=lane.retention_policy,
                    )
                    for lane in turn.topology.lanes
                ),
                lane_topology=turn.topology.checkpoint_topology(),
            ),
            timestamps=conversation.CheckpointTimestamps(created_at=_NOW),
            retention=turn.retention,
        )
    )
    internal_parent = await store.commit(
        conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=staged_internal
        )
    )

    with pytest.raises(conversation.ConversationTransitionError):
        replace(turn, parent=internal_parent)
    with pytest.raises(conversation.ConversationTransitionError):
        await coordinator.execute(
            turn._outward_request(
                "reject internal parent",
                parent=internal_parent,
                child_results=(),
            )
        )

    illegal_outward = conversation.with_checkpoint_integrity(
        replace(
            staged_internal,
            identity=conversation.CheckpointIdentity(
                conversation_id=turn.conversation_id,
                logical_turn_id=conversation.LogicalTurnId(
                    "parent-kind-outward-turn"
                ),
                execution_segment_id=conversation.ExecutionSegmentId(
                    "parent-kind-outward-segment"
                ),
                checkpoint_id=conversation.CheckpointId(
                    "parent-kind-outward-child"
                ),
                branch_id=turn.branch_id,
                sequence=conversation.CheckpointSequence(1),
                parent_checkpoint_id=internal_parent.identity.checkpoint_id,
                parent_sequence=internal_parent.identity.sequence,
            ),
            kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
            integrity=None,
        )
    )
    with pytest.raises(conversation.ConversationTransitionError):
        await store.commit(
            conversation.OutwardTurnCheckpointCandidate(
                checkpoint=illegal_outward,
                public_response_id=conversation.PublicResponseId(
                    "parent-kind-illegal-outward"
                ),
            )
        )

    outward_root = await store.commit(
        conversation.OutwardTurnCheckpointCandidate(
            checkpoint=conversation.with_checkpoint_integrity(
                replace(
                    staged_internal,
                    identity=conversation.CheckpointIdentity(
                        conversation_id=turn.conversation_id,
                        logical_turn_id=conversation.LogicalTurnId(
                            "parent-kind-root-turn"
                        ),
                        execution_segment_id=conversation.ExecutionSegmentId(
                            "parent-kind-root-segment"
                        ),
                        checkpoint_id=conversation.CheckpointId(
                            "parent-kind-outward-root"
                        ),
                        branch_id=turn.branch_id,
                        sequence=conversation.CheckpointSequence(0),
                    ),
                    kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
                    integrity=None,
                )
            ),
            public_response_id=conversation.PublicResponseId(
                "parent-kind-root-response"
            ),
        )
    )
    legal_internal = conversation.with_checkpoint_integrity(
        replace(
            staged_internal,
            identity=conversation.CheckpointIdentity(
                conversation_id=turn.conversation_id,
                logical_turn_id=conversation.LogicalTurnId(
                    "parent-kind-illegal-internal-turn"
                ),
                execution_segment_id=conversation.ExecutionSegmentId(
                    "parent-kind-illegal-internal-segment"
                ),
                checkpoint_id=conversation.CheckpointId(
                    "parent-kind-illegal-internal"
                ),
                branch_id=turn.branch_id,
                sequence=conversation.CheckpointSequence(1),
                parent_checkpoint_id=outward_root.identity.checkpoint_id,
                parent_sequence=outward_root.identity.sequence,
            ),
            integrity=None,
        )
    )
    committed_internal = await store.commit(
        conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=legal_internal
        )
    )
    assert (
        committed_internal.identity.parent_checkpoint_id
        == outward_root.identity.checkpoint_id
    )
    await coordinator.close()
