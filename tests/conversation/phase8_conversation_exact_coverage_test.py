"""Close defensive Phase 8 conversation contract coverage gaps."""

from asyncio import CancelledError
from copy import copy
from dataclasses import replace
from json import dumps, loads
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from agent_integration_contract_test import (
    _durable_tool_segments,
    _execution_segment,
    _function_call,
    _internal_checkpoint,
    _lane,
    _topology,
    _topology_checkpoint,
)
from agent_integration_e2e_test import _multi_agent_runtime, _public_runtime
from durable_codec_test import _continuation_reference
from phase2_fixtures import empty_stateless_plan, retention

import avalan.conversation as conversation
import avalan.conversation.agent as conversation_agent
import avalan.conversation.crypto as conversation_crypto
import avalan.conversation.envelope as conversation_envelope
import avalan.conversation.execution as execution_module
import avalan.conversation.fakes as conversation_fakes
import avalan.conversation.lifecycle as conversation_lifecycle
import avalan.conversation.providers.openai as openai_provider
import avalan.conversation.runtime as runtime_module
import avalan.conversation.state as state_module
from avalan.agent import Specification
from avalan.agent.conversation_child import (
    AgentConversationChildBinding,
    ConfiguredChildOrchestrator,
)
from avalan.agent.conversation_trace import (
    AgentConversationTraceSink,
    AgentProviderResponseTrace,
    AgentToolOutputTrace,
)
from avalan.entities import ToolCall
from avalan.interaction.continuation import (
    PortableContinuation,
    portable_continuation_binding_digest,
)
from avalan.interaction.error import InputSnapshotError
from avalan.model.call import ModelCallContext


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 8 async coverage checks on asyncio."""
    return "asyncio"


class _ChildOrchestrator:
    """Expose just enough configured child state for binding validation."""

    def __init__(self, operations: list[object]) -> None:
        self.id = uuid4()
        self.operations = operations

    def engine_agent_for_operation(self, operation_index: int) -> object:
        """Return a deliberately non-engine value for negative resolution."""
        del operation_index
        return object()


def _invalid() -> Any:
    """Return one deliberately invalid runtime contract value."""
    return object()


@pytest.mark.anyio
async def test_protocol_fallbacks_raise_explicitly() -> None:
    """Keep direct protocol fallback execution explicit and covered."""
    with pytest.raises(NotImplementedError):
        await conversation_agent.AgentConversationInvocationAdapter.execute(
            cast(Any, object()),
            cast(Any, object()),
            "coverage input",
            object(),
            cast(Any, AsyncMock()),
        )

    with pytest.raises(NotImplementedError):
        await AgentConversationTraceSink.record_provider_response(
            cast(Any, object()),
            cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await AgentConversationTraceSink.record_tool_output(
            cast(Any, object()),
            cast(Any, object()),
        )

    with pytest.raises(NotImplementedError):
        await conversation.ConversationCryptoBoundaryHook.reach(
            cast(Any, object()),
            conversation.ConversationCryptoBoundary.ENCRYPT_BEFORE,
        )
    with pytest.raises(NotImplementedError):
        await conversation.ConversationKeyResolver.current_write_key(
            cast(Any, object()),
            conversation.AuthorityDigest("fallback-authority"),
        )
    with pytest.raises(NotImplementedError):
        await conversation.ConversationKeyResolver.read_key(
            cast(Any, object()),
            conversation.AuthorityDigest("fallback-authority"),
            key_id="fallback-key",
            revision=1,
        )
    with pytest.raises(NotImplementedError):
        await conversation.ConversationCipher.encrypt(
            cast(Any, object()),
            b"fallback",
            key=cast(Any, object()),
            associated_data=cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await conversation.ConversationCipher.decrypt(
            cast(Any, object()),
            cast(Any, object()),
            key=cast(Any, object()),
            associated_data=cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await conversation.ConversationCipher.authenticated_digest(
            cast(Any, object()),
            b"fallback",
            key=cast(Any, object()),
            associated_data=cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        conversation_crypto._AesGcmPrimitive.encrypt(
            cast(Any, object()),
            b"nonce",
            b"fallback",
            b"associated-data",
        )
    with pytest.raises(NotImplementedError):
        conversation_crypto._AesGcmPrimitive.decrypt(
            cast(Any, object()),
            b"nonce",
            b"fallback",
            b"associated-data",
        )
    with pytest.raises(NotImplementedError):
        conversation_crypto._AesGcmType.__call__(
            cast(Any, object()),
            b"fallback-key",
        )
    with pytest.raises(NotImplementedError):
        await conversation_envelope.ContinuationEnvelopeKeyResolver.active_key(
            cast(Any, object()),
            conversation.AuthorityDigest("fallback-authority"),
        )
    with pytest.raises(NotImplementedError):
        await conversation_envelope.ContinuationEnvelopeKeyResolver.read_key(
            cast(Any, object()),
            conversation.AuthorityDigest("fallback-authority"),
            key_id="fallback-key",
            revision=1,
        )
    with pytest.raises(NotImplementedError):
        conversation_fakes._ClosedCast.__call__(
            cast(Any, object()),
            str,
            "fallback",
        )

    provider_stream = conversation.ConversationProviderStream
    provider = conversation.ConversationProvider
    with pytest.raises(NotImplementedError):
        provider_stream.__aiter__(cast(Any, object()))
    with pytest.raises(NotImplementedError):
        await provider_stream.terminal(cast(Any, object()))
    with pytest.raises(NotImplementedError):
        await provider_stream.aclose(cast(Any, object()))
    with pytest.raises(NotImplementedError):
        await provider.dispatch(
            cast(Any, object()),
            cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await provider.stream(
            cast(Any, object()),
            cast(Any, object()),
        )

    stored_adapter = conversation_lifecycle.StoredResponseLifecycleAdapter
    lifecycle_store = conversation_lifecycle.ProviderLifecycleStore
    binding_descriptor = vars(stored_adapter)["binding"]
    assert isinstance(binding_descriptor, property)
    binding_getter = binding_descriptor.fget
    assert binding_getter is not None
    with pytest.raises(NotImplementedError):
        binding_getter(cast(Any, object()))
    with pytest.raises(NotImplementedError):
        await stored_adapter.retrieve(
            cast(Any, object()),
            cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await stored_adapter.delete(
            cast(Any, object()),
            cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await lifecycle_store.claim_provider_lifecycle(
            cast(Any, object()),
            cast(Any, object()),
            limit=1,
        )
    with pytest.raises(NotImplementedError):
        await lifecycle_store.acknowledge_provider_lifecycle(
            cast(Any, object()),
            cast(Any, object()),
            succeeded=False,
        )
    with pytest.raises(NotImplementedError):
        await lifecycle_store.quarantine_provider_checkpoint(
            cast(Any, object()),
            cast(Any, object()),
        )
    with pytest.raises(NotImplementedError):
        await lifecycle_store.reconcile_ambiguous_dispatch(
            cast(Any, object()),
            cast(Any, object()),
        )

    for name in ("id", "operations", "tool", "event_manager"):
        descriptor = vars(ConfiguredChildOrchestrator)[name]
        assert isinstance(descriptor, property)
        getter = descriptor.fget
        assert getter is not None
        with pytest.raises(NotImplementedError):
            getter(cast(Any, object()))

    with pytest.raises(NotImplementedError):
        ConfiguredChildOrchestrator.conversation_engine_args(
            cast(Any, object())
        )
    with pytest.raises(NotImplementedError):
        ConfiguredChildOrchestrator.engine_agent_for_operation(
            cast(Any, object()),
            0,
        )


def _idempotency(
    segment: conversation.ProviderExecutionSegment,
    checkpoint: conversation.ConversationCheckpoint,
) -> conversation.RequestIdempotencyIdentity:
    """Return the exact idempotency identity used by one segment."""
    return conversation.RequestIdempotencyIdentity(
        authority=checkpoint.authority,
        operation=conversation.ConversationOperation.CREATE,
        key=segment.idempotency_key,
        request_digest=segment.request_digest,
    )


def _base_request() -> conversation.ConversationRunRequest:
    """Return one valid first-turn agent request for recovery mutations."""
    _, turn, _, _, _ = _public_runtime()
    return turn._outward_request(
        "coverage input",
        parent=None,
        child_results=(),
    )


def _contract_turn(
    *,
    topology: conversation.AgentLaneTopology | None = None,
    coordinator: object | None = None,
    parent: conversation.ConversationCheckpoint | None = None,
) -> conversation.AgentConversationTurn:
    """Return one valid turn around a lightweight coordinator boundary."""
    selected = topology or _topology()
    authority = _topology_checkpoint(
        _topology().parent_lanes[0],
        _topology(),
    ).authority
    runtime = coordinator or SimpleNamespace(execute=AsyncMock())
    return conversation.AgentConversationTurn(
        coordinator=cast(Any, runtime),
        authority=authority,
        topology=selected,
        lanes=tuple(
            conversation.AgentConversationLane(
                lane_id=lane.lane_id,
                mode=conversation.ConversationMode.STATELESS,
            )
            for lane in selected.lanes
        ),
        logical_turn_id=conversation.LogicalTurnId("coverage-turn"),
        execution_segment_id=conversation.ExecutionSegmentId(
            "coverage-segment"
        ),
        checkpoint_id=conversation.CheckpointId("coverage-checkpoint"),
        branch_id=conversation.ConversationBranchId("coverage-branch"),
        provisional_response_id=conversation.ProvisionalResponseId(
            "coverage-provisional"
        ),
        public_response_id=conversation.PublicResponseId("coverage-response"),
        idempotency_key=conversation.RequestIdempotencyKey("coverage-key"),
        retention=retention(),
        parent=parent,
    )


def _child_request(
    base: conversation.ConversationRunRequest,
    advance: conversation.ConversationAdvance,
    *,
    operation: conversation.ConversationOperation,
    branch_id: conversation.ConversationBranchId | None = None,
) -> conversation.ConversationRunRequest:
    """Return one valid non-root request for a selected advance kind."""
    parent_id = advance.parent_checkpoint_id
    identity = replace(
        base.identity,
        branch_id=branch_id or base.identity.branch_id,
        sequence=conversation.CheckpointSequence(1),
        parent_checkpoint_id=parent_id,
        parent_sequence=conversation.CheckpointSequence(0),
    )
    return replace(
        base,
        semantics=replace(
            base.semantics,
            operation=operation,
            parent_checkpoint_id=parent_id,
        ),
        identity=identity,
        advance=advance,
    )


def test_runtime_only_trace_child_and_model_contracts_fail_closed() -> None:
    """Reject malformed trace, child, model, and continuation boundaries."""
    with pytest.raises(TypeError):
        AgentProviderResponseTrace(text=_invalid(), calls=())
    with pytest.raises(TypeError):
        AgentProviderResponseTrace(
            text="valid",
            calls=(cast(ToolCall, _invalid()),),
        )
    with pytest.raises(TypeError):
        AgentToolOutputTrace(call=cast(ToolCall, _invalid()), outcome=None)

    invalid_child = cast(Any, SimpleNamespace())
    with pytest.raises(conversation.ConversationValidationError):
        AgentConversationChildBinding(
            lane_id=conversation.ProviderLaneId("lane-child"),
            orchestrator=invalid_child,
        )
    empty = AgentConversationChildBinding(
        lane_id=conversation.ProviderLaneId("lane-child"),
        orchestrator=cast(Any, _ChildOrchestrator([])),
    )
    assert "lane-child" in repr(empty)
    with pytest.raises(conversation.ConversationValidationError):
        empty.resolve()
    malformed = AgentConversationChildBinding(
        lane_id=conversation.ProviderLaneId("lane-child"),
        orchestrator=cast(Any, _ChildOrchestrator([_invalid()])),
    )
    with pytest.raises(conversation.ConversationValidationError):
        malformed.resolve()

    specification = Specification(role=None, goal=None)
    with pytest.raises(conversation.ConversationValidationError):
        ModelCallContext(
            specification=specification,
            input=None,
            conversation_turn=cast(
                conversation.AgentConversationTurn, _invalid()
            ),
            conversation_input="input",
        )
    _, turn, _, _, _ = _public_runtime()
    with pytest.raises(conversation.ConversationValidationError):
        ModelCallContext(
            specification=specification,
            input=None,
            agent_id=uuid4(),
            conversation_turn=turn,
            conversation_input="input",
        )
    with pytest.raises(conversation.ConversationValidationError):
        ModelCallContext(
            specification=specification,
            input=None,
            conversation_input="input",
        )
    with pytest.raises(conversation.ConversationValidationError):
        ModelCallContext(
            specification=specification,
            input=None,
            conversation_invocation_adapter=cast(Any, _invalid()),
        )
    with pytest.raises(InputSnapshotError):
        portable_continuation_binding_digest(
            cast(PortableContinuation, _invalid())
        )


def test_agent_lane_and_result_contracts_reject_every_invalid_shape() -> None:
    """Exercise closed topology, invocation, and provider-result contracts."""
    topology = _topology()
    parent, first_child, _ = topology.lanes
    assert (
        conversation.direct_model_topology_path(
            conversation.AgentModelSlot("primary")
        )
        == "direct/primary"
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.agent_conversation_surface_disposition(_invalid())
    with pytest.raises(conversation.ConversationValidationError):
        conversation.derive_agent_provider_lane_id(
            conversation_id=topology.conversation_id,
            owner_kind=_invalid(),
            topology_path=parent.topology_path,
            model_slot=parent.model_slot,
            binding=parent.binding,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.derive_agent_provider_lane_id(
            conversation_id=topology.conversation_id,
            owner_kind=parent.owner_kind,
            topology_path=parent.topology_path,
            model_slot=parent.model_slot,
            binding=_invalid(),
        )

    invalid_lanes = (
        {"owner_kind": _invalid()},
        {"binding": _invalid()},
        {"parent_lane_id": conversation.ProviderLaneId("unexpected-parent")},
        {"topology_path": conversation.AgentTopologyPath("wrong/primary")},
        {
            "topology_path": conversation.AgentTopologyPath(
                "agent/agent-parent/wrong"
            )
        },
    )
    for mutation in invalid_lanes:
        with pytest.raises(conversation.ConversationValidationError):
            replace(parent, **mutation)

    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentLaneTopology(
            conversation_id=topology.conversation_id,
            lanes=cast(Any, [parent]),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentLaneTopology(
            conversation_id=topology.conversation_id,
            lanes=(parent, parent),
        )
    with pytest.raises(conversation.ConversationValidationError):
        topology.outward_child_results(cast(Any, ()))
    with pytest.raises(conversation.ConversationValidationError):
        topology.outward_child_results(
            {first_child.lane_id: cast(Any, (_invalid(),))}
        )

    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentConversationResult(
            receipt=_invalid(),
            output="output",
            child_results=(),
        )
    segment = _execution_segment()
    with pytest.raises(conversation.ConversationValidationError):
        conversation_agent.AgentConversationExecutionSegmentCandidate(
            segment_index=-1,
            phase=segment.phase,
            items=segment.items,
            reasoning=segment.reasoning,
            usage=segment.usage,
            tools=segment.tools,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation_agent.AgentConversationLaneInvocationResult(
            result=_invalid(),
            segments=(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation_agent.AgentConversationLaneInvocation(
            binding=_invalid(),
            dispatch=_invalid(),
        )

    plan = empty_stateless_plan(parent.binding)
    with pytest.raises(conversation.ConversationValidationError):
        conversation_agent.agent_conversation_provider_result(
            _invalid(),
            "output",
            invocation_id="invocation",
        )
    result = conversation_agent.agent_conversation_provider_result(
        plan,
        "canonical output",
        invocation_id="invocation",
    )
    assert result.items[0].canonical_input["role"] == "assistant"

    with pytest.raises(conversation.ConversationValidationError):
        conversation.agent_topology_digest(_invalid())


def test_execution_contracts_reject_invalid_recovery_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed tool effect, segment, lease, and reservation state."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentStructuredInputRequested(cast(Any, _invalid()))
    request = conversation.AgentStructuredInputRequested({"value": "secret"})
    assert "secret" not in repr(request)

    with pytest.raises(conversation.ConversationValidationError):
        conversation.ToolEffectReconciliation(applied=False, output="output")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ToolEffectReconciliation(
            applied=True,
            output=cast(str, _invalid()),
        )
    reconciliation = conversation.ToolEffectReconciliation(
        applied=True,
        output="output",
    )
    assert "output=<redacted>" in repr(reconciliation)

    segment = _execution_segment()
    checkpoint = _internal_checkpoint(segment)
    idempotency = _idempotency(segment, checkpoint)
    admission = conversation.DurableToolRecoveryAdmission(
        checkpoint_id=checkpoint.identity.checkpoint_id,
        checkpoint_integrity=conversation.IntegrityDigest("a" * 64),
        idempotency=idempotency,
        binding=segment.binding,
        action=conversation.DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT,
        segment_count=1,
    )
    assert "segment_count=1" in repr(admission)
    for mutation in (
        {"checkpoint_integrity": conversation.IntegrityDigest("short")},
        {"checkpoint_integrity": conversation.IntegrityDigest("z" * 64)},
        {"idempotency": _invalid()},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(admission, **mutation)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DurableToolRecoveryLease(
            admission=_invalid(),
            owner_token="owner",
        )
    lease = conversation.DurableToolRecoveryLease(
        admission=admission,
        owner_token="owner",
    )
    assert "owner=<redacted>" in repr(lease)

    tool = segment.tools[0]
    for mutation in (
        {"arguments": _invalid()},
        {"effect_policy": _invalid()},
        {"idempotency_key": None},
        {
            "phase": conversation.ToolExecutionPhase.OUTPUT_PERSISTED,
            "output_id": None,
        },
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(tool, **mutation)

    with pytest.raises(conversation.ConversationValidationError):
        replace(segment, schema_version=2)
    with pytest.raises(conversation.ConversationValidationError):
        replace(segment, binding=_invalid())
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            segment,
            mode=conversation.ConversationMode.STORED,
            upstream_response_id=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(segment, tools=(tool, tool))
    requested, _, _ = _durable_tool_segments()
    empty_output = replace(
        requested,
        phase=conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
        tools=(),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.durable_tool_recovery_action((requested, empty_output))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.durable_tool_recovery_action(
            (replace(requested, tools=()),)
        )

    original_freeze = execution_module.freeze_json_value
    monkeypatch.setattr(
        execution_module, "freeze_json_value", lambda value: ()
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentStructuredInputRequested({"value": "secret"})
    with pytest.raises(conversation.ConversationValidationError):
        replace(tool, arguments={"value": "secret"})
    with pytest.raises(conversation.ConversationValidationError):
        replace(segment, recovery_request={"value": "secret"})
    monkeypatch.setattr(execution_module, "freeze_json_value", original_freeze)

    reservation = conversation.ConversationExecutionReservation(
        idempotency=idempotency,
        identity=checkpoint.identity,
        lanes=(
            conversation.ProviderLaneExecutionReservation(
                binding=segment.binding,
                mode=segment.mode,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            reservation,
            authorized_agent_ids=(
                checkpoint.authority.agent_id,
                checkpoint.authority.agent_id,
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            reservation,
            authorized_agent_ids=(conversation.ConversationAgentId("other"),),
        )

    foreign_authority = replace(
        checkpoint.authority,
        agent_id=conversation.ConversationAgentId("foreign-agent"),
    )
    foreign_idempotency = replace(
        idempotency,
        authority=foreign_authority,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            reservation,
            idempotency=foreign_idempotency,
            authorized_agent_ids=(segment.binding.agent_id,),
        )


def test_agent_turn_construction_and_capabilities_fail_closed() -> None:
    """Reject corrupt turn ancestry, topology, lanes, and callbacks."""
    topology = _topology()
    turn = _contract_turn(topology=topology)
    assert "lane_count=3" in repr(turn)
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, coordinator=_invalid())
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, lanes=turn.lanes[:-1])
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            turn,
            authority=replace(
                turn.authority,
                agent_id=conversation.ConversationAgentId("other-agent"),
            ),
        )

    direct_path = conversation.direct_model_topology_path(
        conversation.AgentModelSlot("direct")
    )
    direct = _lane(
        conversation_id=topology.conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.DIRECT_MODEL,
        agent_id="agent-parent",
        model_slot="direct",
        topology_path=direct_path,
        model="model-direct",
        retention=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    child_only = object.__new__(conversation.AgentLaneTopology)
    object.__setattr__(
        child_only,
        "conversation_id",
        topology.conversation_id,
    )
    object.__setattr__(
        child_only,
        "lanes",
        (direct, topology.child_lanes[0]),
    )
    with pytest.raises(conversation.ConversationValidationError):
        _contract_turn(topology=child_only)

    staged = _topology_checkpoint(topology.parent_lanes[0], topology)
    parent = replace(
        staged,
        kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
        lifecycle=conversation.CheckpointLifecycle.COMMITTED,
        identity=replace(
            staged.identity,
            branch_id=turn.branch_id,
        ),
        timestamps=replace(
            staged.timestamps,
            committed_at=staged.timestamps.created_at,
        ),
    )
    wrong_conversation = replace(
        parent,
        identity=replace(
            parent.identity,
            conversation_id=conversation.ConversationId("wrong"),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, parent=wrong_conversation)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            turn,
            parent=replace(
                parent,
                identity=replace(
                    parent.identity,
                    branch_id=conversation.ConversationBranchId("wrong"),
                ),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            turn,
            parent=replace(
                parent,
                authority=replace(
                    parent.authority,
                    principal_id=conversation.AuthorityPrincipalId(
                        "foreign-principal"
                    ),
                ),
            ),
        )

    explicit_branch = conversation.ExplicitBranchAdvance(
        parent_checkpoint_id=parent.identity.checkpoint_id,
        branch_id=turn.branch_id,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, parent=parent, advance=explicit_branch)
    named_head = conversation.NamedHeadAdvance(
        head_id=conversation.NamedHeadId("coverage-head"),
        parent_checkpoint_id=conversation.CheckpointId("wrong-parent"),
        expected_revision=conversation.NamedHeadRevision(0),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, parent=parent, advance=named_head)
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, parent=parent, advance=cast(Any, _invalid()))
    with pytest.raises(conversation.ConversationValidationError):
        replace(turn, advance=named_head)

    _, public_turn, _, _, _ = _public_runtime()
    mismatched_topology = conversation.ProviderLaneTopology(
        schema_version=1,
        entries=(topology.checkpoint_topology().entries[0],),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            turn,
            parent=replace(
                parent,
                content=replace(
                    parent.content,
                    lane_topology=mismatched_topology,
                ),
            ),
        )
    foreign_snapshot = _topology_checkpoint(
        public_turn.topology.parent_lanes[0],
        public_turn.topology,
    ).content.lanes[0]
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            turn,
            parent=replace(
                parent,
                content=replace(
                    parent.content,
                    lanes=(foreign_snapshot,),
                    lane_topology=None,
                ),
            ),
        )

    with pytest.raises(conversation.ConversationCapabilityError):
        turn._aggregate_mode(set())


@pytest.mark.anyio
async def test_agent_turn_fanout_rejects_bad_boundaries() -> None:
    """Run child fanout and reject invalid runtime and staging boundaries."""
    runtime = _multi_agent_runtime()
    result = await runtime.turn.execute(
        "fan out this request",
        lane_invocations=None,
    )
    assert result.output == "parent merged child results"
    assert len(result.child_results) == 2
    with pytest.raises(conversation.ConversationValidationError):
        await runtime.turn.execute(" ", lane_invocations=None)
    with pytest.raises(conversation.ConversationValidationError):
        await runtime.turn.execute("x" * 1_048_577, lane_invocations=None)
    with pytest.raises(conversation.ConversationValidationError):
        runtime.turn._safe_child_results(result.receipt)

    turn = _contract_turn()
    with pytest.raises(conversation.ConversationCapabilityError):
        await turn._execute_runtime_invocation("input", ())
    invocations = tuple(
        conversation_agent.AgentConversationLaneInvocation(
            binding=lane.binding,
            dispatch=AsyncMock(),
        )
        for lane in turn.topology.lanes
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await turn._execute_runtime_invocation("input", invocations)
    nonawaitable = SimpleNamespace(
        execute=AsyncMock(),
        execute_agent=lambda request, selected: _invalid(),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await replace(
            turn,
            coordinator=cast(Any, nonawaitable),
        )._execute_runtime_invocation("input", invocations)

    topology = turn.topology
    parent_lane = topology.parent_lanes[0]
    call = _function_call(parent_lane.lane_id)
    tool = replace(_execution_segment().tools[0], call_id=call.call_id)
    segment = replace(
        _execution_segment(),
        binding=parent_lane.binding,
        items=(call,),
        tools=(tool,),
    )
    staged = _topology_checkpoint(parent_lane, topology)
    suspension = replace(
        staged,
        kind=conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
        identity=replace(
            staged.identity,
            logical_turn_id=turn.logical_turn_id,
            branch_id=turn.branch_id,
        ),
        content=replace(
            staged.content,
            lanes=(
                replace(
                    staged.content.lanes[0],
                    lifecycle=conversation.ProviderLaneLifecycle.SUSPENDED,
                ),
            ),
            execution_segments=(segment,),
        ),
        integrity=None,
    )
    structured = conversation.AgentStructuredInputRequested(tool.arguments)
    boundary = conversation.AgentConversationSuspensionBoundary(
        request=structured,
        call=call,
        tool=tool,
        checkpoint=suspension,
    )
    assert "content=<redacted>" in repr(boundary)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AgentConversationSuspensionBoundary(
            request=cast(Any, _invalid()),
            call=call,
            tool=tool,
            checkpoint=suspension,
        )

    reference = _continuation_reference()
    with pytest.raises(conversation.ConversationValidationError):
        await turn.stage_structured_input_suspension(
            cast(Any, _invalid()),
            reference,
        )
    without_stage = replace(
        turn,
        coordinator=cast(Any, SimpleNamespace(execute=AsyncMock())),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await without_stage.stage_structured_input_suspension(
            suspension,
            reference,
        )
    synchronous_stage = replace(
        turn,
        coordinator=cast(
            Any,
            SimpleNamespace(
                execute=AsyncMock(),
                stage_structured_input_suspension=(
                    lambda _checkpoint, _continuation: _invalid()
                ),
            ),
        ),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await synchronous_stage.stage_structured_input_suspension(
            suspension,
            reference,
        )


def test_state_topology_checkpoint_and_parent_guards_fail_closed() -> None:
    """Reject corrupt topology, checkpoint content, and special transitions."""
    topology = _topology()
    persisted = topology.checkpoint_topology()
    parent_entry, child_entry, _ = persisted.entries
    for mutation in (
        {"owner_kind": _invalid()},
        {"binding_digest": "z" * 64},
        {"parent_lane_id": conversation.ProviderLaneId("unexpected")},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(parent_entry, **mutation)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneTopology(schema_version=2, entries=())
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneTopology(
            schema_version=1,
            entries=(parent_entry, parent_entry),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneTopology(
            schema_version=1,
            entries=(
                parent_entry,
                replace(
                    child_entry,
                    parent_lane_id=conversation.ProviderLaneId("missing"),
                ),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        persisted.entry(conversation.ProviderLaneId("missing"))

    segment = _execution_segment()
    transcript = conversation.VisibleTranscript(entries=())
    with pytest.raises(conversation.ConversationValidationError):
        conversation.MultiLaneCheckpointContent(
            visible_transcript=transcript,
            lanes=(),
            execution_segments=(cast(Any, _invalid()),),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.MultiLaneCheckpointContent(
            visible_transcript=transcript,
            lanes=(),
            execution_segments=(segment, segment),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.MultiLaneCheckpointContent(
            visible_transcript=transcript,
            lanes=(),
            execution_segments=(segment,),
            lane_topology=persisted,
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            _internal_checkpoint(segment),
            kind=conversation.CheckpointKind.STANDALONE_COMPACT_RESULT,
        )

    staged = _topology_checkpoint(topology.parent_lanes[0], topology)
    outward = replace(
        staged,
        kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
        identity=replace(
            staged.identity,
            parent_checkpoint_id=conversation.CheckpointId("suspension"),
            parent_sequence=conversation.CheckpointSequence(0),
            sequence=conversation.CheckpointSequence(1),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.SuspensionContinuationCheckpointCandidate(
            checkpoint=outward,
            public_response_id=conversation.PublicResponseId("response"),
            suspension_checkpoint_id=conversation.CheckpointId("other"),
        )

    with pytest.raises(conversation.ConversationValidationError):
        state_module.validate_checkpoint_parent_kind(
            _invalid(),
            None,
        )
    with pytest.raises(conversation.ConversationTransitionError):
        state_module.validate_checkpoint_parent_kind(
            conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
            suspension_continuation=True,
        )
    with pytest.raises(conversation.ConversationTransitionError):
        state_module.validate_checkpoint_parent_kind(
            conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
            compact_continuation=True,
        )


def test_recovery_payload_round_trips_every_advance_and_rejects_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip each durable advance and reject malformed JSON shapes."""
    base = _base_request()
    with pytest.raises(conversation.ConversationValidationError):
        replace(base, lane_topology=_invalid())
    with pytest.raises(conversation.ConversationValidationError):
        replace(base, lane_topology=_topology().checkpoint_topology())
    parent_id = conversation.CheckpointId("parent")
    branch_id = conversation.ConversationBranchId("branch-explicit")
    requests = (
        replace(
            base,
            semantics=replace(
                base.semantics,
                parent_checkpoint_id=parent_id,
            ),
            advance=conversation.ResetAdvance(
                parent_checkpoint_id=parent_id,
            ),
        ),
        _child_request(
            base,
            conversation.OrdinaryChildAdvance(
                parent_checkpoint_id=parent_id,
            ),
            operation=conversation.ConversationOperation.CONTINUE,
        ),
        _child_request(
            base,
            conversation.ExplicitBranchAdvance(
                parent_checkpoint_id=parent_id,
                branch_id=branch_id,
            ),
            operation=conversation.ConversationOperation.BRANCH,
            branch_id=branch_id,
        ),
        _child_request(
            base,
            conversation.NamedHeadAdvance(
                parent_checkpoint_id=parent_id,
                head_id=conversation.NamedHeadId("head"),
                expected_revision=conversation.NamedHeadRevision(0),
            ),
            operation=conversation.ConversationOperation.CONTINUE,
        ),
    )
    for request in requests:
        payload = runtime_module.conversation_run_request_recovery_payload(
            request
        )
        assert (
            runtime_module.conversation_run_request_from_recovery_payload(
                payload,
                authority=request.semantics.authority,
                retention=request.retention,
                lane_topology=request.lane_topology,
                idempotency_key=request.idempotency_key,
            )
            == request
        )

    with pytest.raises(conversation.ConversationValidationError):
        runtime_module.conversation_run_request_recovery_payload(_invalid())
    forged = copy(base)
    object.__setattr__(forged, "advance", _invalid())
    with pytest.raises(conversation.ConversationValidationError):
        runtime_module.conversation_run_request_recovery_payload(forged)
    payload = dict(
        runtime_module.conversation_run_request_recovery_payload(base)
    )
    payload["schema_version"] = 2
    with pytest.raises(conversation.ConversationValidationError):
        runtime_module.conversation_run_request_from_recovery_payload(
            payload,
            authority=base.semantics.authority,
            retention=base.retention,
            lane_topology=base.lane_topology,
            idempotency_key=base.idempotency_key,
        )
    malformed = dict(
        runtime_module.conversation_run_request_recovery_payload(base)
    )
    malformed["advance"] = {"kind": "unknown"}
    with pytest.raises(conversation.ConversationValidationError):
        runtime_module.conversation_run_request_from_recovery_payload(
            malformed,
            authority=base.semantics.authority,
            retention=base.retention,
            lane_topology=base.lane_topology,
            idempotency_key=base.idempotency_key,
        )

    for operation, value in (
        (runtime_module._recovery_mapping, _invalid()),
        (runtime_module._recovery_sequence, _invalid()),
        (runtime_module._recovery_str, 1),
        (runtime_module._recovery_int, "1"),
        (runtime_module._required_recovery, None),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            if operation is runtime_module._recovery_mapping:
                operation(value, None)
            else:
                operation(value)
    with pytest.raises(conversation.ConversationValidationError):
        runtime_module._recovery_mapping({"unexpected": 1}, {"expected"})

    original_freeze = runtime_module.freeze_json_value
    monkeypatch.setattr(runtime_module, "freeze_json_value", lambda value: ())
    with pytest.raises(conversation.ConversationValidationError):
        runtime_module.conversation_run_request_recovery_payload(base)
    monkeypatch.setattr(runtime_module, "freeze_json_value", original_freeze)


def test_execution_segment_codec_rejects_unknown_fields() -> None:
    """Reject a durable execution segment with an unrecognized field."""
    checkpoint = _internal_checkpoint(_execution_segment())
    codec = conversation.ConversationCheckpointCodec()
    payload = loads(codec.encode(checkpoint))
    payload["checkpoint"]["content"]["execution_segments"][0][
        "unexpected"
    ] = True

    with pytest.raises(conversation.ConversationCodecError):
        codec.decode(dumps(payload).encode())


def test_native_openai_schema_and_tool_constructor_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing schema dependencies and invalid tool configuration."""
    original_import = openai_provider.import_module

    def reject_import(name: str) -> object:
        del name
        raise ValueError()

    monkeypatch.setattr(openai_provider, "import_module", reject_import)
    with pytest.raises(conversation.ConversationValidationError):
        openai_provider._json_schema_adapter()
    monkeypatch.setattr(
        openai_provider,
        "import_module",
        lambda name: SimpleNamespace(
            Draft202012Validator=object(),
            SchemaError=ValueError,
            ValidationError=None,
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        openai_provider._json_schema_adapter()
    assert (
        openai_provider._json_schema_exception_class(
            cast(Any, SimpleNamespace(NotError=object())),
            "NotError",
        )
        is None
    )
    monkeypatch.setattr(openai_provider, "import_module", original_import)

    async def handler(arguments: object) -> str:
        del arguments
        return "output"

    tool = conversation.NativeOpenAIFunctionTool(
        name="coverage_tool",
        parameters={"type": "object"},
        handler=handler,
        effect_policy=conversation.ToolEffectPolicy.FENCED_UNPROTECTED,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(tool, effect_policy=_invalid())
    with pytest.raises(conversation.ConversationValidationError):
        replace(tool, reconciliation_handler=lambda arguments: arguments)


@pytest.mark.anyio
async def test_native_openai_reconciliation_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify missing, cancelled, failed, and malformed reconciliation."""

    async def handler(arguments: object) -> str:
        del arguments
        return "output"

    base = conversation.NativeOpenAIFunctionTool(
        name="coverage_tool",
        parameters={"type": "object"},
        handler=handler,
        effect_policy=conversation.ToolEffectPolicy.FENCED_UNPROTECTED,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await base.reconcile("{}")

    async def cancel(
        arguments: object,
    ) -> conversation.ToolEffectReconciliation:
        del arguments
        raise CancelledError()

    async def fail(arguments: object) -> conversation.ToolEffectReconciliation:
        del arguments
        raise RuntimeError("effect failed")

    async def malformed(arguments: object) -> Any:
        del arguments
        return _invalid()

    for reconcile, expected in (
        (cancel, CancelledError),
        (fail, conversation.ConversationError),
        (malformed, conversation.ConversationValidationError),
    ):
        tool = replace(base, reconciliation_handler=reconcile)
        with pytest.raises(expected):
            await tool.reconcile("{}")

    original_freeze = openai_provider.freeze_json_value
    monkeypatch.setattr(openai_provider, "freeze_json_value", lambda value: ())
    with pytest.raises(conversation.ConversationProviderResponseError):
        base.validate_arguments("{}")
    monkeypatch.setattr(openai_provider, "freeze_json_value", original_freeze)
    with pytest.raises(conversation.ConversationValidationError):
        base.execution_metadata(
            call_id=conversation.ProviderCallId("call"),
            arguments="{}",
            request_idempotency_key=conversation.RequestIdempotencyKey("key"),
            phase=_invalid(),
        )
