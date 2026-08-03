"""Close defensive Phase 8 coordinator coverage gaps."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import native_openai_provider_test as native_fixtures
import native_openai_provider_validation_test as native_validation_fixtures
import pytest
from agent_integration_contract_test import (
    _durable_tool_segments,
    _execution_segment,
    _function_call,
    _internal_checkpoint,
    _topology_checkpoint,
)
from agent_integration_e2e_test import _multi_agent_runtime, _public_runtime
from phase2_fixtures import (
    authority,
    empty_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation
from avalan.conversation import agent as agent_module
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation import runtime as runtime_module


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 8 coordinator checks on asyncio."""
    return "asyncio"


def _request_context() -> tuple[
    conversation.RunScopedConversationCoordinator,
    conversation.ConversationRunRequest,
    coordinator_module._SegmentExecutionContext,
]:
    """Return one exact agent request and its private segment context."""
    _, turn, _, coordinator, _ = _public_runtime()
    run = turn._outward_request(
        "coordinator coverage",
        parent=None,
        child_results=(),
    )
    context = coordinator_module._SegmentExecutionContext(
        request=run,
        idempotency=coordinator._idempotency(run),
        segments=[],
        visible_transcript=conversation.VisibleTranscript(entries=()),
        lane_snapshots={},
    )
    return coordinator, run, context


def _invocation_result(
    plan: conversation.StatelessProviderPlan,
    *,
    result: conversation.ProviderResult | None = None,
) -> agent_module.AgentConversationLaneInvocationResult:
    """Return one valid agent invocation result and provider segment."""
    canonical = agent_module.agent_conversation_provider_result(
        plan,
        "coordinator output",
        invocation_id="coordinator-invocation",
    )
    return agent_module.AgentConversationLaneInvocationResult(
        result=result or canonical,
        segments=(
            agent_module.AgentConversationExecutionSegmentCandidate(
                segment_index=0,
                phase=(
                    conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
                ),
                items=canonical.items,
                reasoning=canonical.reasoning,
                usage=canonical.usage,
            ),
        ),
    )


def _output_candidate(
    run: conversation.ConversationRunRequest,
    lane: conversation.AgentProviderLane,
    *,
    text: str,
    binding: conversation.ProviderLaneBinding | None = None,
) -> conversation.ProviderLaneOutputCandidate:
    """Return one exact outward candidate for an agent topology lane."""
    selected = binding or lane.binding
    plan = empty_stateless_plan(selected)
    result = agent_module.agent_conversation_provider_result(
        plan,
        text,
        invocation_id=f"output-{lane.model_slot}",
    )
    receipt = conversation.provider_lane_execution_receipt(
        authority=run.semantics.authority,
        identity=run.identity,
        binding=selected,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    return conversation.ProviderLaneOutputCandidate(
        lane_id=selected.lane_id,
        binding=selected,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        execution_receipt=receipt,
    )


def _checkpoint_with_segments(
    run: conversation.ConversationRunRequest,
    segments: tuple[conversation.ProviderExecutionSegment, ...],
    *,
    preserve_request_identity: bool = False,
) -> conversation.ConversationCheckpoint:
    """Return one integrity-bound private checkpoint for recovery checks."""
    base = _internal_checkpoint(segments[0])
    identity = replace(
        base.identity,
        checkpoint_id=conversation.CheckpointId("coordinator-recovery"),
        conversation_id=(
            run.identity.conversation_id
            if preserve_request_identity
            else base.identity.conversation_id
        ),
        logical_turn_id=(
            run.identity.logical_turn_id
            if preserve_request_identity
            else base.identity.logical_turn_id
        ),
    )
    return conversation.with_checkpoint_integrity(
        replace(
            base,
            identity=identity,
            authority=run.semantics.authority,
            content=replace(
                base.content,
                execution_segments=segments,
                lane_topology=None,
            ),
            integrity=None,
        )
    )


def test_segment_context_and_lane_authority_guards_fail_closed() -> None:
    """Reject malformed context, missing roots, and persisted lane drift."""
    coordinator, run, context = _request_context()
    runtime = coordinator._lanes[run.lanes[0].lane_id]
    result = conversation.fake_provider_result(
        empty_stateless_plan(runtime.binding),
        turn=1,
    )
    snapshot = coordinator._stateless_lane_snapshot(
        binding=runtime.binding,
        retention_policy=runtime.retention_policy,
        items=result.items,
        reasoning=result.reasoning,
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._SegmentExecutionContext(
            request=run,
            idempotency=context.idempotency,
            segments=[],
            visible_transcript=context.visible_transcript,
            lane_snapshots={
                conversation.ProviderLaneId("wrong-lane"): cast(
                    conversation.ProviderLaneSnapshot,
                    snapshot,
                )
            },
        )

    wrong_authority = replace(
        run.semantics.authority,
        agent_id=conversation.ConversationAgentId("wrong-agent"),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        coordinator._validate_request_lane_authority(run, wrong_authority)

    lanes = coordinator._lanes
    coordinator._lanes = {}
    coordinator._validate_request_lane_authority(
        run,
        run.semantics.authority,
    )
    coordinator._lanes = lanes
    lane_id = run.lanes[0].lane_id
    runtime = lanes[lane_id]
    coordinator._lanes = {
        lane_id: replace(
            runtime,
            retention_policy=(
                conversation.ChildLaneRetentionPolicy.DISCARD_TERMINAL
            ),
        )
    }
    with pytest.raises(conversation.ConversationAuthorizationError):
        coordinator._validate_request_lane_authority(
            run,
            run.semantics.authority,
        )


@pytest.mark.anyio
async def test_execute_agent_and_run_validate_invocation_ownership() -> None:
    """Reject empty, duplicate, drifted, and mis-scoped agent callbacks."""
    coordinator, run, _ = _request_context()
    runtime = coordinator._lanes[run.lanes[0].lane_id]

    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.execute_agent(run, ())

    invocation = agent_module.AgentConversationLaneInvocation(
        binding=runtime.binding,
        dispatch=AsyncMock(),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.execute_agent(run, (invocation, invocation))

    drifted = agent_module.AgentConversationLaneInvocation(
        binding=replace(
            runtime.binding,
            model_configuration_revision=(
                conversation.ModelConfigurationRevision("drifted-config")
            ),
        ),
        dispatch=AsyncMock(),
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        await coordinator.execute_agent(run, (drifted,))

    with pytest.raises(conversation.ConversationValidationError):
        await coordinator._run(
            run,
            streaming=False,
            lane_invocations={},
        )


@pytest.mark.anyio
async def test_recovery_entry_guards_reject_closed_and_corrupt_state() -> None:
    """Reject closed, unauthorized, malformed, and conflicting recovery."""
    coordinator, run, _ = _request_context()
    authority = run.semantics.authority
    checkpoint_id = conversation.CheckpointId("coordinator-recovery")

    coordinator._closed = True
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.recover_durable_tool_execution(
            checkpoint_id,
            authority,
        )
    coordinator._closed = False

    with pytest.raises(conversation.ConversationAuthorizationError):
        await coordinator.recover_durable_tool_execution(
            checkpoint_id,
            replace(
                authority,
                principal_id=conversation.AuthorityPrincipalId(
                    "wrong-principal"
                ),
            ),
        )

    malformed = replace(
        _checkpoint_with_segments(run, (_execution_segment(),)),
        integrity=None,
    )
    coordinator._store = cast(
        conversation.ConversationStore,
        SimpleNamespace(load=AsyncMock(return_value=malformed)),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.recover_durable_tool_execution(
            checkpoint_id,
            authority,
        )

    payload = runtime_module.conversation_run_request_recovery_payload(run)
    requested, output, complete = _durable_tool_segments()
    conflicting = (
        replace(requested, recovery_request=payload),
        replace(output, recovery_request={**payload, "boundary": "other"}),
    )
    checkpoint = _checkpoint_with_segments(run, conflicting)
    coordinator._store = cast(
        conversation.ConversationStore,
        SimpleNamespace(load=AsyncMock(return_value=checkpoint)),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await coordinator.recover_durable_tool_execution(
            checkpoint_id,
            authority,
        )

    digest_conflict = tuple(
        replace(segment, recovery_request=payload)
        for segment in (requested, output, complete)
    )
    checkpoint = _checkpoint_with_segments(run, digest_conflict)
    coordinator._store = cast(
        conversation.ConversationStore,
        SimpleNamespace(load=AsyncMock(return_value=checkpoint)),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await coordinator.recover_durable_tool_execution(
            checkpoint_id,
            authority,
        )

    coordinator._closed = True
    with pytest.raises(conversation.ConversationTransitionError):
        await coordinator.stage_structured_input_suspension(
            checkpoint,
            cast(conversation.PortableContinuationReference, object()),
        )


@pytest.mark.anyio
async def test_failed_recovery_rolls_back_the_admitted_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fence an admitted recovery when later parent resolution fails."""
    coordinator, run, _ = _request_context()
    runtime = coordinator._lanes[run.lanes[0].lane_id]
    idempotency = coordinator._idempotency(run)
    payload = runtime_module.conversation_run_request_recovery_payload(run)
    base = _execution_segment()
    call = replace(base.items[0], lane_id=runtime.binding.lane_id)
    segment = replace(
        base,
        idempotency_key=idempotency.key,
        request_digest=idempotency.request_digest,
        binding=runtime.binding,
        items=(call,),
        recovery_request=payload,
    )
    checkpoint = _checkpoint_with_segments(
        run,
        (segment,),
        preserve_request_identity=True,
    )
    checkpoint = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            content=replace(
                checkpoint.content,
                lane_topology=run.lane_topology,
            ),
            integrity=None,
        )
    )

    async def admit(
        admission: conversation.DurableToolRecoveryAdmission,
        execution: conversation.ConversationExecutionReservation,
    ) -> conversation.DurableToolRecoveryLease:
        del execution
        return conversation.DurableToolRecoveryLease(
            admission=admission,
            owner_token="recovery-owner",
        )

    coordinator._store = cast(
        conversation.ConversationStore,
        SimpleNamespace(
            load=AsyncMock(return_value=checkpoint),
            admit_tool_recovery=admit,
        ),
    )
    rollback = AsyncMock()
    monkeypatch.setattr(coordinator, "_rollback", rollback)
    monkeypatch.setattr(
        coordinator,
        "_resolve_parent",
        AsyncMock(side_effect=RuntimeError("parent resolution failed")),
    )

    with pytest.raises(RuntimeError, match="parent resolution failed"):
        await coordinator.recover_durable_tool_execution(
            checkpoint.identity.checkpoint_id,
            run.semantics.authority,
        )
    rollback.assert_awaited_once_with(
        idempotency,
        "recovery-owner",
        ambiguous=True,
    )


def test_agent_child_projection_rejects_ambiguous_and_oversized_output() -> (
    None
):
    """Reject absent topology, duplicate lanes, drift, and excessive output."""
    simple = request(
        scope=_request_context()[1].semantics.authority,
        identity=root_identity("child-projection"),
        advance=conversation.FirstTurnAdvance(),
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module.RunScopedConversationCoordinator._agent_child_visible_delta(
            simple,
            (),
        )

    runtime = _multi_agent_runtime(label="-projection")
    run = runtime.turn._outward_request(
        "project children",
        parent=None,
        child_results=(),
    )
    first, second = runtime.turn.topology.child_lanes
    first_output = _output_candidate(run, first, text="first child")
    second_output = _output_candidate(run, second, text="second child")
    with pytest.raises(conversation.ConversationValidationError):
        runtime.coordinator._agent_child_visible_delta(
            run,
            (first_output, first_output),
        )
    with pytest.raises(conversation.ConversationBindingDriftError):
        runtime.coordinator._agent_child_visible_delta(run, ())

    drifted_binding = replace(
        first.binding,
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("child-drift")
        ),
    )
    drifted_output = _output_candidate(
        run,
        first,
        text="drifted child",
        binding=drifted_binding,
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        runtime.coordinator._agent_child_visible_delta(
            run,
            (drifted_output, second_output),
        )

    oversized = _output_candidate(run, first, text="x" * 600_000)
    oversized_second = _output_candidate(run, second, text="y" * 600_000)
    with pytest.raises(conversation.ConversationValidationError):
        runtime.coordinator._agent_child_visible_delta(
            run,
            (oversized, oversized_second),
        )


@pytest.mark.anyio
async def test_agent_dispatch_rejects_every_invalid_callback_boundary() -> (
    None
):
    """Reject invalid dispatch shape, response, context, segment, and items."""
    coordinator, run, context = _request_context()
    runtime = coordinator._lanes[run.lanes[0].lane_id]
    plan = empty_stateless_plan(runtime.binding)
    progress = coordinator_module._DispatchProgress()

    invocation = agent_module.AgentConversationLaneInvocation(
        binding=runtime.binding,
        dispatch=AsyncMock(),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_complete_lane(
            runtime,
            plan,
            streaming=True,
            progress=progress,
            sink=None,
            lane_invocation=invocation,
        )

    synchronous = agent_module.AgentConversationLaneInvocation(
        binding=runtime.binding,
        dispatch=lambda selected: selected,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_complete_lane(
            runtime,
            plan,
            streaming=False,
            progress=progress,
            sink=None,
            lane_invocation=synchronous,
        )

    malformed = agent_module.AgentConversationLaneInvocation(
        binding=runtime.binding,
        dispatch=AsyncMock(return_value=object()),
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        await coordinator._dispatch_complete_lane(
            runtime,
            plan,
            streaming=False,
            progress=progress,
            sink=None,
            lane_invocation=malformed,
        )

    valid_result = _invocation_result(plan)
    without_context = agent_module.AgentConversationLaneInvocation(
        binding=runtime.binding,
        dispatch=AsyncMock(return_value=valid_result),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_complete_lane(
            runtime,
            plan,
            streaming=False,
            progress=progress,
            sink=None,
            lane_invocation=without_context,
        )

    await coordinator._dispatch_complete_lane(
        runtime,
        plan,
        streaming=False,
        progress=progress,
        sink=None,
        segment_context=context,
        lane_invocation=without_context,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await coordinator._dispatch_complete_lane(
            runtime,
            plan,
            streaming=False,
            progress=progress,
            sink=None,
            segment_context=context,
            lane_invocation=without_context,
        )

    canonical = valid_result.result
    wrong_item = replace(
        canonical.items[0],
        lane_id=conversation.ProviderLaneId("wrong-output-lane"),
    )
    bad_result = replace(canonical, items=(wrong_item,))
    invalid_items = agent_module.AgentConversationLaneInvocation(
        binding=runtime.binding,
        dispatch=AsyncMock(
            return_value=_invocation_result(plan, result=bad_result)
        ),
    )
    clean_context = replace(context, segments=[])
    with pytest.raises(conversation.ConversationProviderResponseError):
        await coordinator._dispatch_complete_lane(
            runtime,
            plan,
            streaming=False,
            progress=progress,
            sink=None,
            segment_context=clean_context,
            lane_invocation=invalid_items,
        )


@pytest.mark.anyio
async def test_private_segment_and_native_tool_guards_fail_closed() -> None:
    """Reject missing contexts, duplicate segments, and unsafe tool state."""
    coordinator, run, context = _request_context()
    runtime = coordinator._lanes[run.lanes[0].lane_id]
    result = conversation.fake_provider_result(
        empty_stateless_plan(runtime.binding),
        turn=1,
    )
    snapshot = coordinator._stateless_lane_snapshot(
        binding=runtime.binding,
        retention_policy=runtime.retention_policy,
        items=result.items,
        reasoning=result.reasoning,
    )
    native = cast(
        conversation.NativeOpenAIConversationLaneRuntime,
        SimpleNamespace(binding=runtime.binding),
    )

    assert coordinator._requested_tool_executions(native, (), None) == ()
    assert (
        await coordinator._persist_native_execution_segment(
            native=native,
            mode=conversation.ConversationMode.STATELESS,
            segment_index=0,
            phase=(
                conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            ),
            items=result.items,
            result=result,
            tools=(),
            segment_context=None,
            lane_snapshot=snapshot,
        )
        is None
    )
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator._persist_native_execution_segment(
            native=native,
            mode=conversation.ConversationMode.STATELESS,
            segment_index=0,
            phase=(
                conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            ),
            items=result.items,
            result=result,
            tools=(),
            segment_context=context,
            lane_snapshot=replace(
                snapshot,
                binding=replace(
                    snapshot.binding,
                    lane_id=conversation.ProviderLaneId("wrong-snapshot"),
                ),
                ledger=replace(
                    snapshot.ledger,
                    lane_id=conversation.ProviderLaneId("wrong-snapshot"),
                ),
            ),
        )

    prior = conversation.ProviderExecutionSegment(
        schema_version=1,
        idempotency_key=context.idempotency.key,
        request_digest=context.idempotency.request_digest,
        binding=runtime.binding,
        mode=conversation.ConversationMode.STATELESS,
        segment_index=0,
        phase=conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
    )
    context.segments.append(prior)
    with pytest.raises(conversation.ConversationConflictError):
        await coordinator._persist_native_execution_segment(
            native=native,
            mode=conversation.ConversationMode.STATELESS,
            segment_index=0,
            phase=(
                conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            ),
            items=result.items,
            result=result,
            tools=(),
            segment_context=context,
            lane_snapshot=snapshot,
        )

    with pytest.raises(conversation.ConversationValidationError):
        await coordinator._execute_native_tools(
            native,
            calls=(),
            completed=(),
            result=result,
            order_base=0,
            current_byte_count=0,
            progress=coordinator_module._DispatchProgress(),
            recovered_outputs={conversation.ProviderCallId("other"): "value"},
        )

    call = _function_call(runtime.binding.lane_id)
    structured = conversation.AgentStructuredInputRequested(
        {"secret": "value"}
    )
    tool_native = cast(
        conversation.NativeOpenAIConversationLaneRuntime,
        SimpleNamespace(
            binding=runtime.binding,
            provider=SimpleNamespace(
                execute_tool=AsyncMock(side_effect=structured),
            ),
        ),
    )
    call_result = conversation.ProviderResult(
        items=(call,),
        reasoning=result.reasoning,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._execute_native_tools(
            tool_native,
            calls=(call,),
            completed=(),
            result=call_result,
            order_base=0,
            current_byte_count=0,
            progress=coordinator_module._DispatchProgress(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator._execute_native_tools(
            tool_native,
            calls=(call,),
            completed=(),
            result=call_result,
            order_base=0,
            current_byte_count=0,
            progress=coordinator_module._DispatchProgress(),
            segment_context=context,
            recovery_checkpoint=cast(
                conversation.ConversationCheckpoint,
                object(),
            ),
        )

    multi = _multi_agent_runtime(label="-structured")
    staged = _topology_checkpoint(
        multi.turn.topology.parent_lanes[0],
        multi.turn.topology,
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator._structured_input_checkpoint(context, staged, call)


@pytest.mark.anyio
async def test_native_recovery_rejects_invalid_shape_and_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid native recovery plans, snapshots, tools, and suffixes."""
    binding = native_fixtures._binding(lane_id="lane-native-recovery-guard")
    provider = native_fixtures._provider(
        binding,
        native_validation_fixtures._unused_handler,
    )
    _, coordinator, _ = native_fixtures._direct_client(provider)
    native = coordinator._lanes[binding.lane_id]
    scope = authority()
    run = request(
        scope=scope,
        identity=root_identity("native-recovery-guard"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-recovery-guard",
        response_suffix="native-recovery-guard",
    )
    plan = empty_stateless_plan(binding)
    idempotency = coordinator._idempotency(run)
    execution = coordinator._execution_reservation(
        run,
        idempotency,
        {binding.lane_id: native},
    )
    snapshot = coordinator._stateless_lane_snapshot(
        binding=binding,
        retention_policy=native.retention_policy,
        items=(),
        reasoning=plan.reasoning,
    )

    def segment(
        *,
        items: tuple[conversation.ProviderItem, ...] = (),
        tools: tuple[conversation.ProviderToolExecution, ...] = (),
        index: int = 0,
        phase: conversation.ProviderExecutionSegmentPhase = (
            conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
        ),
    ) -> conversation.ProviderExecutionSegment:
        return conversation.ProviderExecutionSegment(
            schema_version=1,
            idempotency_key=idempotency.key,
            request_digest=idempotency.request_digest,
            binding=binding,
            mode=conversation.ConversationMode.STATELESS,
            segment_index=index,
            phase=phase,
            items=items,
            reasoning=plan.reasoning,
            usage=conversation.ProviderUsage(),
            tools=tools,
        )

    def checkpoint(
        selected_segment: conversation.ProviderExecutionSegment,
        selected_snapshot: conversation.ProviderLaneSnapshot = snapshot,
    ) -> conversation.ConversationCheckpoint:
        base = _internal_checkpoint(selected_segment)
        return replace(
            base,
            authority=scope,
            content=replace(
                base.content,
                lanes=(selected_snapshot,),
                execution_segments=(selected_segment,),
                lane_topology=None,
            ),
            retention=run.retention,
            integrity=None,
        )

    selected = segment()
    try:
        with pytest.raises(conversation.ConversationCapabilityError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(selected),
                segments=(selected,),
                action=conversation.DurableToolRecoveryAction.COMMIT_OUTWARD,
                plans=(),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )

        foreign = _execution_segment()
        with pytest.raises(conversation.ConversationCapabilityError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(selected),
                segments=(foreign,),
                action=conversation.DurableToolRecoveryAction.COMMIT_OUTWARD,
                plans=((run.lanes[0], native, plan),),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )

        wrong_binding = replace(
            binding,
            lane_id=conversation.ProviderLaneId("wrong-recovery-snapshot"),
        )
        wrong_snapshot = coordinator._stateless_lane_snapshot(
            binding=wrong_binding,
            retention_policy=native.retention_policy,
            items=(),
            reasoning=plan.reasoning,
        )
        with pytest.raises(conversation.ConversationConflictError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(selected, wrong_snapshot),
                segments=(selected,),
                action=conversation.DurableToolRecoveryAction.COMMIT_OUTWARD,
                plans=((run.lanes[0], native, plan),),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )

        call = _function_call(binding.lane_id)
        missing_tool = segment(items=(call,))
        with pytest.raises(conversation.ConversationConflictError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(missing_tool),
                segments=(missing_tool,),
                action=(
                    conversation.DurableToolRecoveryAction.REQUIRE_RECONCILIATION
                ),
                plans=((run.lanes[0], native, plan),),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )

        with pytest.raises(conversation.ConversationConflictError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(selected),
                segments=(selected,),
                action=conversation.DurableToolRecoveryAction.REEXECUTE_PURE,
                plans=((run.lanes[0], native, plan),),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )

        base = _execution_segment()
        native_call = replace(base.items[0], lane_id=binding.lane_id)
        pending = segment(items=(native_call,), tools=base.tools)
        with pytest.raises(conversation.ConversationConflictError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(pending),
                segments=(pending,),
                action=conversation.DurableToolRecoveryAction.COMMIT_OUTWARD,
                plans=((run.lanes[0], native, plan),),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )

        requested, output, complete = _durable_tool_segments()
        durable = tuple(
            replace(
                item,
                idempotency_key=idempotency.key,
                request_digest=idempotency.request_digest,
                binding=binding,
                items=tuple(
                    replace(provider_item, lane_id=binding.lane_id)
                    for provider_item in item.items
                ),
            )
            for item in (requested, output, complete)
        )
        stored_snapshot = conversation.StoredProviderLaneSnapshot(
            binding=binding,
            upstream_response_id=conversation.UpstreamResponseId(
                "corrupt-stored-snapshot"
            ),
            reasoning=plan.reasoning,
            lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
            retention_policy=native.retention_policy,
        )

        async def corrupt_snapshot(
            selected_runtime: object,
            selected_plan: object,
            **kwargs: Any,
        ) -> conversation.ProviderResult:
            del selected_runtime, selected_plan
            context = cast(
                coordinator_module._SegmentExecutionContext,
                kwargs["segment_context"],
            )
            context.lane_snapshots[binding.lane_id] = stored_snapshot
            return conversation.ProviderResult(
                items=(),
                reasoning=plan.reasoning,
            )

        monkeypatch.setattr(
            coordinator,
            "_dispatch_complete_lane",
            corrupt_snapshot,
        )
        with pytest.raises(conversation.ConversationConflictError):
            await coordinator._recover_native_tool_suffix(
                request=run,
                parent=None,
                checkpoint=checkpoint(durable[0]),
                segments=durable,
                action=conversation.DurableToolRecoveryAction.RESUME_PROVIDER,
                plans=((run.lanes[0], native, plan),),
                execution_reservation=execution,
                owner_token="recovery-owner",
            )
    finally:
        await coordinator.close()


@pytest.mark.anyio
async def test_segment_checkpoint_conflicts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject snapshot drift, commit mismatch, and conflicting persistence."""
    coordinator, run, context = _request_context()
    runtime = coordinator._lanes[run.lanes[0].lane_id]
    result = conversation.fake_provider_result(
        empty_stateless_plan(runtime.binding),
        turn=1,
    )
    snapshot = coordinator._stateless_lane_snapshot(
        binding=runtime.binding,
        retention_policy=runtime.retention_policy,
        items=result.items,
        reasoning=result.reasoning,
    )
    native = cast(
        conversation.NativeOpenAIConversationLaneRuntime,
        SimpleNamespace(binding=runtime.binding),
    )

    with pytest.raises(conversation.ConversationValidationError):
        await coordinator._persist_native_execution_segment(
            native=native,
            mode=conversation.ConversationMode.STATELESS,
            segment_index=0,
            phase=(
                conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            ),
            items=result.items,
            result=result,
            tools=(),
            segment_context=context,
            lane_snapshot=cast(
                conversation.ProviderLaneSnapshot,
                SimpleNamespace(
                    lane_id=conversation.ProviderLaneId("wrong-snapshot")
                ),
            ),
        )

    prior = conversation.ProviderExecutionSegment(
        schema_version=1,
        idempotency_key=context.idempotency.key,
        request_digest=context.idempotency.request_digest,
        binding=runtime.binding,
        mode=conversation.ConversationMode.STATELESS,
        segment_index=0,
        phase=conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
    )
    context.segments.append(prior)
    monkeypatch.setattr(
        coordinator,
        "_commit_execution_segment_checkpoint",
        AsyncMock(return_value=_internal_checkpoint(prior)),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await coordinator._persist_native_execution_segment(
            native=native,
            mode=conversation.ConversationMode.STATELESS,
            segment_index=1,
            phase=(
                conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            ),
            items=result.items,
            result=result,
            tools=(),
            segment_context=context,
            lane_snapshot=snapshot,
        )

    holder: dict[str, conversation.ConversationCheckpoint] = {}

    async def conflict(
        candidate: conversation.ExecutionSegmentCheckpointCandidate,
    ) -> conversation.ConversationCheckpoint:
        holder["checkpoint"] = candidate.checkpoint
        raise conversation.ConversationConflictError()

    async def load(
        checkpoint_id: conversation.CheckpointId,
        scope: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        del checkpoint_id, scope
        return holder["checkpoint"]

    coordinator._store = cast(
        conversation.ConversationStore,
        SimpleNamespace(commit=conflict, load=load),
    )
    commit_segment = type(coordinator)._commit_execution_segment_checkpoint
    recovered = await commit_segment(
        coordinator,
        run,
        prior,
        (prior,),
        (snapshot,),
        context.visible_transcript,
    )
    assert recovered is holder["checkpoint"]

    coordinator._store = cast(
        conversation.ConversationStore,
        SimpleNamespace(
            commit=AsyncMock(return_value=_internal_checkpoint(prior)),
        ),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await commit_segment(
            coordinator,
            run,
            prior,
            (prior,),
            (snapshot,),
            context.visible_transcript,
        )
