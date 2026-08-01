"""Exercise the deterministic Phase 2 coordinator end to end."""

from asyncio import create_task, gather
from collections.abc import Callable
from dataclasses import replace

import pytest
from phase2_fixtures import (
    authority,
    binding,
    child_identity,
    coordinator,
    empty_stateless_plan,
    first_stored_plan,
    next_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic concurrency on asyncio only."""
    return "asyncio"


def _runtime(
    lane_binding: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    *,
    controller: conversation.DeterministicFaultController | None = None,
    profile: conversation.ConversationCapabilityProfile | None = None,
) -> conversation.ConversationLaneRuntime:
    return conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=profile
        or conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=results,
            controller=controller,
        ),
    )


async def test_normative_coordinator_contract(
    record_property: Callable[[str, object], None],
) -> None:
    """Commit immutable first and child turns at validated boundaries."""
    record_property("conversation_acceptance_evidence", "runtime")
    scope = authority()
    lane_binding = binding()
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    second_plan = next_stateless_plan(lane_binding, first_result.items)
    second_result = conversation.fake_provider_result(second_plan, turn=2)
    runtime = _runtime(lane_binding, (first_result, second_result))
    store = conversation.InMemoryConversationStore()
    publisher = conversation.DeterministicFakePublisher()
    observer = conversation.DeterministicFakeObserver()
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
        observer=observer,
    )
    first_request = request(
        scope=scope,
        identity=root_identity("one"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="one",
        key="key-one",
    )
    first = await engine.execute(first_request)
    assert first.result is not None
    parent_bytes = conversation.ConversationCheckpointCodec().encode(
        first.checkpoint
    )

    child_request = request(
        scope=scope,
        identity=child_identity(first.checkpoint, "two"),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=first.checkpoint.identity.checkpoint_id
        ),
        response_suffix="two",
        key="key-two",
    )
    second = await engine.execute(child_request)
    assert second.result is not None
    assert second.checkpoint.identity.parent_checkpoint_id == (
        first.checkpoint.identity.checkpoint_id
    )
    assert (
        second.checkpoint.kind
        is conversation.CheckpointKind.COMPLETED_OUTWARD_TURN
    )
    assert (
        second.checkpoint.lifecycle
        is conversation.CheckpointLifecycle.COMMITTED
    )
    assert second.checkpoint.authority == scope
    assert second.checkpoint.integrity is not None
    assert second.checkpoint.retention == child_request.retention
    lane = second.checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    assert lane.binding == lane_binding
    assert lane.reasoning == second_result.reasoning
    assert lane.lifecycle is conversation.ProviderLaneLifecycle.COMMITTED
    assert lane.ledger.items == first_result.items + second_result.items
    assert second.checkpoint.content.visible_transcript.entry_count == 2
    assert lane.ledger.item_count == 2
    assert all(
        type(entry) is conversation.VisibleTranscriptEntry
        for entry in second.checkpoint.content.visible_transcript.entries
    )
    assert all(
        type(item) is conversation.ProviderItem for item in lane.ledger.items
    )
    restored_parent = await store.load(
        first.checkpoint.identity.checkpoint_id, scope
    )
    assert (
        conversation.ConversationCheckpointCodec().encode(restored_parent)
        == parent_bytes
    )
    replay = await engine.execute(child_request)
    assert replay.result == second.result
    assert replay.output_candidates == second.output_candidates
    assert second.output_candidates[0].completed_items == second_result.items
    assert second.output_candidates[0].usage == second_result.usage
    assert second.output_candidates[0].upstream_response_id is None
    assert second.result.lane_outputs[0].items == second_result.items
    assert second.result.lane_outputs[0].usage == second_result.usage
    provider = engine.fake_provider_diagnostics(lane_binding.lane_id)
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 2
    assert len(publisher.published) == 2
    assert tuple(item.event for item in observer.observations) == (
        "checkpoint_committed",
        "outbox_published",
        "checkpoint_committed",
        "outbox_published",
    )
    assert all(
        item.authority_scope_digest == conversation.authority_digest(scope)
        for item in observer.observations
    )
    assert tuple(
        item.parent_checkpoint_id for item in observer.observations
    ) == (
        None,
        None,
        first.checkpoint.identity.checkpoint_id,
        first.checkpoint.identity.checkpoint_id,
    )
    assert all(
        item.lane_ids == (lane_binding.lane_id,)
        for item in observer.observations
    )
    assert engine.diagnostics.active_attempts == 0
    assert store.diagnostics.provisional_responses == 0


async def test_stream_and_nonstream_two_turns_commit_equivalent_state(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep typed checkpoint state equivalent across transport boundaries."""
    record_property("conversation_acceptance_evidence", "runtime")
    scope = authority()
    direct_binding = binding("lane-direct")
    stream_binding = binding("lane-stream", streaming=True)
    direct_plan = empty_stateless_plan(direct_binding)
    stream_plan = empty_stateless_plan(stream_binding)
    direct_result = conversation.fake_provider_result(direct_plan, turn=1)
    stream_result = conversation.fake_provider_result(stream_plan, turn=1)
    direct_store = conversation.InMemoryConversationStore()
    stream_store = conversation.InMemoryConversationStore()
    direct_runtime = _runtime(direct_binding, (direct_result,))
    stream_runtime = _runtime(stream_binding, (stream_result,))
    direct_engine = coordinator(
        store=direct_store, scope=scope, runtimes=(direct_runtime,)
    )
    stream_engine = coordinator(
        store=stream_store, scope=scope, runtimes=(stream_runtime,)
    )
    direct_request = request(
        scope=scope,
        identity=root_identity("direct"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-direct",),
        key="key-direct",
        response_suffix="direct",
    )
    stream_request = request(
        scope=scope,
        identity=root_identity("stream"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-stream",),
        key="key-stream",
        response_suffix="stream",
    )
    direct = await direct_engine.execute(direct_request)
    streamed = await stream_engine.stream(stream_request)
    direct_lane = direct.checkpoint.content.lanes[0]
    stream_lane = streamed.checkpoint.content.lanes[0]
    assert isinstance(direct_lane, conversation.StatelessProviderLaneSnapshot)
    assert isinstance(stream_lane, conversation.StatelessProviderLaneSnapshot)
    assert (
        direct.checkpoint.lifecycle
        is conversation.CheckpointLifecycle.COMMITTED
    )
    assert (
        streamed.checkpoint.lifecycle
        is conversation.CheckpointLifecycle.COMMITTED
    )
    assert direct.checkpoint.integrity is not None
    assert streamed.checkpoint.integrity is not None
    assert direct.checkpoint.identity == direct_request.identity
    assert streamed.checkpoint.identity == stream_request.identity
    assert direct.checkpoint.content.visible_transcript.entries == (
        direct_request.visible_delta
    )
    assert stream_lane.ledger.items == stream_result.items
    assert tuple(item.kind for item in direct_lane.ledger.items) == tuple(
        item.kind for item in stream_lane.ledger.items
    )
    assert direct.output_candidates[0].completed_items == direct_result.items
    assert streamed.output_candidates[0].completed_items == stream_result.items
    assert direct.output_candidates[0].usage == direct_result.usage
    assert streamed.output_candidates[0].usage == stream_result.usage
    assert direct.result is not None
    assert streamed.result is not None
    assert direct.result.lane_outputs[0].items == direct_result.items
    assert streamed.result.lane_outputs[0].items == stream_result.items
    stream_provider = stream_engine.fake_provider_diagnostics(
        stream_binding.lane_id
    )
    assert isinstance(
        stream_provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert all(item.closed for item in stream_provider.streams)


async def test_mixed_stored_and_stateless_lanes_remain_separate(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep storage and reasoning lane-local and reject silent fallback."""
    record_property("conversation_acceptance_evidence", "runtime")
    scope = authority()
    stateless_binding = binding("lane-stateless")
    stored_binding = binding("lane-stored")
    stateless_plan = empty_stateless_plan(stateless_binding)
    stored_plan = first_stored_plan(stored_binding)
    stateless_result = conversation.fake_provider_result(
        stateless_plan, turn=1
    )
    stateless_next_plan = next_stateless_plan(
        stateless_binding, stateless_result.items
    )
    stateless_next_result = conversation.fake_provider_result(
        stateless_next_plan,
        turn=2,
    )
    stored_result = conversation.fake_provider_result(stored_plan, turn=1)
    runtimes = (
        _runtime(
            stateless_binding,
            (stateless_result, stateless_next_result),
        ),
        _runtime(stored_binding, (stored_result,)),
    )
    store = conversation.InMemoryConversationStore()
    publisher = conversation.DeterministicFakePublisher()
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=runtimes,
        publisher=publisher,
    )
    run = request(
        scope=scope,
        identity=root_identity("mixed"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-stateless", "lane-stored"),
        modes=(
            conversation.ConversationMode.STATELESS,
            conversation.ConversationMode.STORED,
        ),
        key="key-mixed",
        response_suffix="mixed",
        stored_retention=True,
    )
    receipt = await engine.execute(run)
    stateless = receipt.checkpoint.content.lanes[0]
    stored = receipt.checkpoint.content.lanes[1]
    assert isinstance(stateless, conversation.StatelessProviderLaneSnapshot)
    assert isinstance(stored, conversation.StoredProviderLaneSnapshot)
    assert stateless.lane_id == stateless_binding.lane_id
    assert stateless.ledger.items == stateless_result.items
    assert stateless.reasoning == stateless_result.reasoning
    assert stored.lane_id == stored_binding.lane_id
    assert stored.reasoning == stored_result.reasoning
    stored = receipt.checkpoint.content.lanes[1]
    assert isinstance(stored, conversation.StoredProviderLaneSnapshot)
    assert stored.upstream_response_id == stored_result.upstream_response_id
    assert stateless.execution_receipt == (
        receipt.output_candidates[0].execution_receipt
    )
    assert stored.execution_receipt == (
        receipt.output_candidates[1].execution_receipt
    )
    assert (
        conversation.ConversationCheckpointCodec().decode(
            conversation.ConversationCheckpointCodec().encode(
                receipt.checkpoint
            )
        )
        == receipt.checkpoint
    )
    stateless_provider = engine.fake_provider_diagnostics(
        stateless_binding.lane_id
    )
    stored_provider = engine.fake_provider_diagnostics(stored_binding.lane_id)
    assert isinstance(
        stateless_provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert isinstance(
        stored_provider, conversation.DeterministicFakeProviderDiagnostics
    )
    initial_stateless_plans = stateless_provider.plans
    assert len(initial_stateless_plans) == 1
    assert initial_stateless_plans[0] == stateless_plan
    assert stored_provider.plans == (stored_plan,)
    assert receipt.result is not None
    assert isinstance(
        receipt.result.handle, conversation.StoredConversationHandle
    )
    assert receipt.output_candidates == await store.retrieve_output_candidates(
        receipt.checkpoint.identity.checkpoint_id, scope
    )
    assert tuple(
        item.completed_items for item in receipt.output_candidates
    ) == (
        stateless_result.items,
        stored_result.items,
    )
    assert tuple(item.usage for item in receipt.output_candidates) == (
        stateless_result.usage,
        stored_result.usage,
    )
    assert receipt.output_candidates[0].upstream_response_id is None
    assert receipt.output_candidates[1].upstream_response_id == (
        stored_result.upstream_response_id
    )
    assert "fake-upstream" not in repr(receipt.output_candidates[1])
    assert tuple(item.items for item in receipt.result.lane_outputs) == (
        stateless_result.items,
        stored_result.items,
    )
    assert not hasattr(receipt.result.lane_outputs[1], "upstream_response_id")
    assert publisher.published[0].lane_outputs == receipt.result.lane_outputs

    retained_run = request(
        scope=scope,
        identity=child_identity(receipt.checkpoint, "mixed-retained"),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=receipt.checkpoint.identity.checkpoint_id
        ),
        lane_ids=("lane-stateless",),
        modes=(conversation.ConversationMode.STATELESS,),
        key="key-mixed-retained",
        response_suffix="mixed-retained",
        stored_retention=True,
    )
    retained = await engine.execute(retained_run)
    assert retained.result is not None
    assert isinstance(
        retained.result.handle,
        conversation.StoredConversationHandle,
    )
    assert len(retained.output_candidates) == 1
    assert retained.output_candidates[0].completed_items == (
        stateless_next_result.items
    )
    assert len(retained.checkpoint.content.lanes) == 2
    retained_stored = retained.checkpoint.content.lanes[1]
    assert isinstance(
        retained_stored,
        conversation.StoredProviderLaneSnapshot,
    )
    assert retained_stored == stored
    retained_stateless = retained.checkpoint.content.lanes[0]
    assert isinstance(
        retained_stateless,
        conversation.StatelessProviderLaneSnapshot,
    )
    assert retained_stateless.execution_receipt == (
        retained.output_candidates[0].execution_receipt
    )
    assert engine.fake_provider_diagnostics(
        stateless_binding.lane_id
    ).plans == (stateless_plan, stateless_next_plan)
    assert stored_provider.plans == (stored_plan,)
    assert len(publisher.published) == 2

    capable_profile = conversation.fake_capability_profile(stateless_binding)
    incapable_profile = replace(
        capable_profile,
        capabilities=tuple(
            (
                replace(
                    evidence,
                    state=conversation.CapabilityEvidenceState.INCAPABLE,
                    evidence_ids=(),
                )
                if evidence.capability
                is (
                    conversation.ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY
                )
                else evidence
            )
            for evidence in capable_profile.capabilities
        ),
    )
    incapable_runtime = _runtime(
        stateless_binding,
        (stateless_result,),
        profile=incapable_profile,
    )
    incapable_engine = coordinator(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        runtimes=(incapable_runtime,),
    )
    incapable_request = request(
        scope=scope,
        identity=root_identity("incapable"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-stateless",),
        key="key-incapable",
        response_suffix="incapable",
    )
    assert incapable_request.visible_delta
    with pytest.raises(conversation.ConversationCapabilityError):
        await incapable_engine.execute(incapable_request)
    incapable_provider = incapable_engine.fake_provider_diagnostics(
        stateless_binding.lane_id
    )
    assert isinstance(
        incapable_provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert incapable_provider.plans == ()


async def test_publication_failure_replays_commit_without_redispatch() -> None:
    """Retry only publication after the authoritative commit succeeded."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    runtime = _runtime(lane_binding, (result,))
    fault = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="publisher:publish",
                exception=RuntimeError("injected-publication-failure"),
            ),
        )
    )
    publisher = conversation.DeterministicFakePublisher(fault)
    store = conversation.InMemoryConversationStore()
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
    )
    run = request(
        scope=scope,
        identity=root_identity("publication"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="publication",
        key="key-publication",
    )
    try:
        await engine.execute(run)
    except conversation.ConversationPublicationError:
        pass
    else:
        raise AssertionError("publication fault must surface")
    replay = await engine.execute(run)
    assert replay.result is not None
    provider = engine.fake_provider_diagnostics(lane_binding.lane_id)
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert len(provider.plans) == 1
    assert len(publisher.published) == 1


async def test_duplicate_request_fences_then_replays_owner_commit() -> None:
    """Fence an active duplicate, then replay its owner's committed result."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                pause=True,
            ),
        )
    )
    runtime = _runtime(lane_binding, (result,), controller=controller)
    store = conversation.InMemoryConversationStore()
    publisher = conversation.DeterministicFakePublisher()
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(runtime,),
        publisher=publisher,
    )
    run = request(
        scope=scope,
        identity=root_identity("duplicate-race"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="duplicate-race",
        key="key-duplicate-race",
    )
    owner_task = create_task(engine.execute(run))
    await controller.wait_until_entered("provider:dispatch")
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await engine.execute(run)
    assert store.diagnostics.idempotency_waiters == 0
    assert store.diagnostics.idempotency_records == 1
    assert store.diagnostics.provisional_responses == 1
    provider = engine.fake_provider_diagnostics(lane_binding.lane_id)
    assert isinstance(
        provider, conversation.DeterministicFakeProviderDiagnostics
    )
    assert (
        len(engine.fake_provider_diagnostics(lane_binding.lane_id).plans) == 1
    )
    controller.release("provider:dispatch")
    owner = await owner_task
    duplicate = await engine.execute(run)
    assert owner.checkpoint == duplicate.checkpoint
    assert owner.result == duplicate.result
    assert owner.output_candidates == duplicate.output_candidates
    assert len(provider.plans) == 1
    assert len(publisher.published) == 1
    assert store.diagnostics.idempotency_waiters == 0
    assert store.diagnostics.provisional_responses == 0
    assert store.diagnostics.idempotency_records == 1
    assert engine.diagnostics.active_attempts == 0


async def test_independent_run_progresses_while_provider_is_blocked() -> None:
    """Prove no store lock spans a blocked external provider await."""
    scope = authority()
    blocked_binding = binding("lane-blocked")
    free_binding = binding("lane-free")
    blocked_plan = empty_stateless_plan(blocked_binding)
    free_plan = empty_stateless_plan(free_binding)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                pause=True,
            ),
        )
    )
    blocked_runtime = _runtime(
        blocked_binding,
        (conversation.fake_provider_result(blocked_plan, turn=1),),
        controller=controller,
    )
    free_runtime = _runtime(
        free_binding,
        (conversation.fake_provider_result(free_plan, turn=1),),
    )
    store = conversation.InMemoryConversationStore()
    engine = coordinator(
        store=store,
        scope=scope,
        runtimes=(blocked_runtime, free_runtime),
    )
    blocked = request(
        scope=scope,
        identity=root_identity("blocked"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-blocked",),
        response_suffix="blocked",
        key="key-blocked",
    )
    free = request(
        scope=scope,
        identity=root_identity("free"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-free",),
        response_suffix="free",
        key="key-free",
    )
    blocked_task = create_task(engine.execute(blocked))
    await controller.wait_until_entered("provider:dispatch")
    free_receipt = await engine.execute(free)
    assert free_receipt.checkpoint.identity.checkpoint_id == (
        free.identity.checkpoint_id
    )
    assert not store.diagnostics.locked
    controller.release("provider:dispatch")
    await blocked_task


async def test_explicit_branches_reuse_one_immutable_parent(
    record_property: Callable[[str, object], None],
) -> None:
    """Create two intentional branches without changing their parent."""
    record_property("conversation_acceptance_evidence", "runtime")
    scope = authority()
    lane_binding = binding()
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    next_plan = next_stateless_plan(lane_binding, first_result.items)
    branch_results = (
        conversation.fake_provider_result(next_plan, turn=2, text="branch-a"),
        conversation.fake_provider_result(next_plan, turn=3, text="branch-b"),
    )
    runtime = _runtime(lane_binding, (first_result,) + branch_results)
    store = conversation.InMemoryConversationStore()
    engine = coordinator(store=store, scope=scope, runtimes=(runtime,))
    root = await engine.execute(
        request(
            scope=scope,
            identity=root_identity("branch-root"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="branch-root",
            key="key-branch-root",
        )
    )
    parent_bytes = conversation.ConversationCheckpointCodec().encode(
        root.checkpoint
    )
    runs = tuple(
        request(
            scope=scope,
            identity=child_identity(
                root.checkpoint,
                suffix,
                branch_id=f"branch-{suffix}",
            ),
            advance=conversation.ExplicitBranchAdvance(
                parent_checkpoint_id=root.checkpoint.identity.checkpoint_id,
                branch_id=conversation.ConversationBranchId(
                    f"branch-{suffix}"
                ),
            ),
            response_suffix=suffix,
            key=f"key-{suffix}",
        )
        for suffix in ("a", "b")
    )
    children = await gather(*(engine.execute(item) for item in runs))
    assert all(
        child.checkpoint.lifecycle
        is conversation.CheckpointLifecycle.COMMITTED
        for child in children
    )
    assert children[0].checkpoint.identity.checkpoint_id != (
        children[1].checkpoint.identity.checkpoint_id
    )
    assert {child.checkpoint.identity.branch_id for child in children} == {
        conversation.ConversationBranchId("branch-a"),
        conversation.ConversationBranchId("branch-b"),
    }
    parent = await store.load(root.checkpoint.identity.checkpoint_id, scope)
    assert conversation.ConversationCheckpointCodec().encode(parent) == (
        parent_bytes
    )
    assert await store.branch_count(parent.identity.checkpoint_id, scope) == 2
