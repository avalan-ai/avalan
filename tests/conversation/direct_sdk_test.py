"""Exercise the public fake-only direct conversation SDK."""

from asyncio import CancelledError, create_task
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from phase2_fixtures import (
    authority,
    binding,
    empty_stateless_plan,
    first_stored_plan,
    next_stateless_plan,
    request,
    retention,
    root_identity,
)

import avalan
import avalan.conversation as conversation

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic direct SDK tests on asyncio only."""
    return "asyncio"


def _coordinator(
    *,
    store: conversation.InMemoryConversationStore,
    scope: conversation.AuthorityScope,
    lane_binding: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    boundary_hook: conversation.FakeCoordinatorBoundaryHook | None = None,
    provider_controller: (
        conversation.DeterministicFaultController | None
    ) = None,
) -> conversation.RunScopedConversationCoordinator:
    return conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.ConversationLaneRuntime(
                binding=lane_binding,
                capability_profile=conversation.fake_capability_profile(
                    lane_binding
                ),
                provider_script=conversation.DeterministicFakeProviderScript(
                    results=results,
                    controller=provider_controller,
                ),
            ),
        ),
        boundary_hook=boundary_hook,
    )


def _client(
    *,
    lane_binding: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    stored: bool = False,
    boundary_hook: conversation.FakeCoordinatorBoundaryHook | None = None,
    provider_controller: (
        conversation.DeterministicFaultController | None
    ) = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
    conversation.AuthorityScope,
]:
    scope = authority()
    store = conversation.InMemoryConversationStore()
    coordinator = _coordinator(
        store=store,
        scope=scope,
        lane_binding=lane_binding,
        results=results,
        boundary_hook=boundary_hook,
        provider_controller=provider_controller,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=scope,
        lane=lane_binding,
        retention=retention(stored=stored),
        id_namespace="phase4",
    )
    return avalan.DirectConversationClient(runtime), store, coordinator, scope


async def test_stream_create_continue_branch_and_compact(
    record_property: Callable[[str, object], None],
) -> None:
    """Expose a handle only through the post-commit stream terminal."""
    record_property("conversation_acceptance_evidence", "public")
    lane_binding = binding("lane-direct", streaming=True)
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(
        first_plan,
        turn=1,
        text="turn-one",
    )
    child_plan = next_stateless_plan(lane_binding, first_result.items)
    child_result = conversation.fake_provider_result(
        child_plan,
        turn=2,
        text="turn-two",
    )
    branch_result = conversation.fake_provider_result(
        child_plan,
        turn=3,
        text="branch",
    )
    compact_result = conversation.fake_compaction_result(
        child_plan,
        turn=4,
        opaque_state=b"private-compact-state",
    )
    client, store, coordinator, scope = _client(
        lane_binding=lane_binding,
        results=(
            first_result,
            child_result,
            branch_result,
            compact_result,
        ),
    )

    stream = await client.create(
        "first visible input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    assert type(stream) is avalan.DirectConversationStream
    assert stream.state is avalan.DirectConversationStreamState.PENDING
    with pytest.raises(avalan.ConversationHandleUnavailableError) as pending:
        _ = stream.committed_handle
    assert pending.value.state is avalan.DirectConversationStreamState.PENDING

    events = [event async for event in stream]
    assert [type(event) for event in events] == [
        avalan.DirectConversationOutputDelta,
        avalan.DirectConversationStreamTerminal,
    ]
    assert events[0] == avalan.DirectConversationOutputDelta(
        text_delta="turn-one"
    )
    terminal = events[1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    first = terminal.result
    assert first.output == "turn-one"
    assert first.usage == conversation.ProviderUsage(
        input_tokens=10,
        output_tokens=5,
    )
    assert first.reasoning.effective is (
        conversation.EffectiveReasoningContext.CURRENT_TURN
    )
    assert stream.state is avalan.DirectConversationStreamState.COMMITTED
    assert stream.committed_handle == first.handle
    assert stream.terminal is terminal
    assert "private-compact-state" not in repr(events)

    assert type(first.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=first.handle)
    second = await client.continue_conversation(
        "second visible input",
        avalan.StatelessConversationSettings(parent=parent),
    )
    assert second.output == "turn-two"
    assert second.handle.conversation_id == first.handle.conversation_id
    assert second.handle.checkpoint_id != first.handle.checkpoint_id

    parent_before_branch = await store.load(
        first.handle.checkpoint_id,
        scope,
    )
    branch = await client.branch(
        "branch visible input",
        avalan.StatelessConversationSettings(
            parent=parent,
            branch=avalan.ConversationBranchIntent(
                parent=parent,
                branch_id=conversation.ConversationBranchId("branch-fork"),
            ),
        ),
    )
    assert branch.output == "branch"
    assert branch.handle.branch_id == "branch-fork"
    assert branch.handle.conversation_id == first.handle.conversation_id
    assert await store.load(first.handle.checkpoint_id, scope) == (
        parent_before_branch
    )

    compacted = await client.compact(
        avalan.StandaloneCompactRequest(parent=parent)
    )
    compact_checkpoint = await store.load(
        compacted.handle.checkpoint_id,
        scope,
    )
    assert compact_checkpoint.kind is (
        conversation.CheckpointKind.STANDALONE_COMPACT_RESULT
    )
    assert compact_checkpoint.content.visible_transcript == (
        parent_before_branch.content.visible_transcript
    )
    compact_lane = compact_checkpoint.content.lanes[0]
    assert isinstance(
        compact_lane,
        conversation.StatelessProviderLaneSnapshot,
    )
    assert compact_lane.ledger.items[-1].kind is (
        conversation.ProviderItemKind.COMPACTION
    )
    assert (
        coordinator.fake_provider_diagnostics(
            lane_binding.lane_id
        ).remaining_results
        == 0
    )


async def test_stored_continue_and_explicit_reset(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep stored chaining typed and reset into a fresh root explicitly."""
    record_property("conversation_acceptance_evidence", "public")
    lane_binding = binding("lane-stored", streaming=True)
    first_plan = first_stored_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    assert first_result.upstream_response_id is not None
    second_plan = conversation.StoredProviderPlan(
        binding=lane_binding,
        upstream_response_id=first_result.upstream_response_id,
        reasoning=first_result.reasoning,
    )
    second_result = conversation.fake_provider_result(second_plan, turn=2)
    reset_result = conversation.fake_provider_result(first_plan, turn=3)
    client, store, _, scope = _client(
        lane_binding=lane_binding,
        results=(first_result, second_result, reset_result),
        stored=True,
    )
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )

    first = await client.create("stored first", settings)
    assert type(first.handle) is avalan.StoredConversationHandle
    parent = avalan.StoredParent(handle=first.handle)
    second = await client.continue_conversation(
        "stored second",
        avalan.StoredConversationSettings(
            provider_storage_disclosed=True,
            parent=parent,
        ),
    )
    assert second.handle.conversation_id == first.handle.conversation_id

    reset = await client.reset(
        "stored reset",
        avalan.ConversationResetIntent(
            parent=parent,
            target_mode=avalan.ConversationMode.STORED,
            provider_storage_disclosed=True,
        ),
        settings,
    )
    assert reset.handle.conversation_id != first.handle.conversation_id
    checkpoint = await store.load(reset.handle.checkpoint_id, scope)
    assert checkpoint.identity.parent_checkpoint_id is None
    assert checkpoint.content.visible_transcript.entries == (
        conversation.VisibleTranscriptEntry(
            role=conversation.VisibleTranscriptRole.USER,
            content="stored reset",
        ),
    )


async def test_named_head_advance_uses_expected_revision() -> None:
    """Advance an explicitly created named head through typed CAS input."""
    lane_binding = binding("lane-head")
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    child_plan = next_stateless_plan(lane_binding, first_result.items)
    child_result = conversation.fake_provider_result(child_plan, turn=2)
    client, store, _, scope = _client(
        lane_binding=lane_binding,
        results=(first_result, child_result),
    )
    first = await client.create(
        "head root",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    await store.create_head(
        conversation.NamedHeadSnapshot(
            head_id=conversation.NamedHeadId("main"),
            revision=conversation.NamedHeadRevision(0),
            checkpoint_id=first.handle.checkpoint_id,
        ),
        scope,
    )
    parent = avalan.StatelessParent(handle=first.handle)
    result = await client.continue_conversation(
        "head child",
        avalan.StatelessConversationSettings(
            parent=parent,
            named_head=avalan.NamedHeadParent(
                head_id=conversation.NamedHeadId("main"),
                expected_revision=conversation.NamedHeadRevision(0),
                parent=parent,
            ),
        ),
    )
    head = await store.load_head(conversation.NamedHeadId("main"), scope)
    assert head.checkpoint_id == result.handle.checkpoint_id
    assert head.revision == 1


async def test_stream_close_without_start_has_no_dispatch_or_handle() -> None:
    """Close a pending stream without allocating provider state."""
    lane_binding = binding("lane-close", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    client, _, coordinator, _ = _client(
        lane_binding=lane_binding,
        results=(result,),
    )
    stream = await client.create(
        "will not dispatch",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    await stream.aclose()

    assert stream.state is (
        avalan.DirectConversationStreamState.CLOSED_INCOMPLETE
    )
    with pytest.raises(avalan.ConversationHandleUnavailableError) as closed:
        _ = stream.terminal
    assert closed.value.state is (
        avalan.DirectConversationStreamState.CLOSED_INCOMPLETE
    )
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert diagnostics.plans == ()
    assert diagnostics.remaining_results == 1


async def test_invalid_mode_parent_is_rejected_before_dispatch() -> None:
    """Reject an incompatible parent before consuming a provider result."""
    lane_binding = binding("lane-invalid")
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    client, _, coordinator, _ = _client(
        lane_binding=lane_binding,
        results=(result,),
    )
    invalid_handle = conversation.StoredConversationHandle(
        conversation_id=conversation.ConversationId("conversation-invalid"),
        checkpoint_id=conversation.CheckpointId("checkpoint-invalid"),
        branch_id=conversation.ConversationBranchId("branch-invalid"),
    )
    settings = avalan.StatelessConversationSettings()
    object.__setattr__(
        settings,
        "parent",
        conversation.StoredParent(handle=invalid_handle),
    )

    with pytest.raises(conversation.ConversationValidationError):
        await client.continue_conversation("invalid", settings)
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert diagnostics.plans == ()
    assert diagnostics.remaining_results == 1


async def test_stream_failure_after_visible_output_withholds_handle() -> None:
    """Surface visible output but never a handle after terminal failure."""
    lane_binding = binding("lane-stream-failure", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(
        plan,
        turn=1,
        text="visible-before-failure",
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:terminal",
                exception=conversation.ConversationCommitError(),
            ),
        )
    )
    client, store, coordinator, _ = _client(
        lane_binding=lane_binding,
        results=(result,),
        provider_controller=controller,
    )
    stream = await client.create(
        "failure input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    iterator = stream.__aiter__()

    assert await iterator.__anext__() == (
        avalan.DirectConversationOutputDelta(
            text_delta="visible-before-failure"
        )
    )
    with pytest.raises(conversation.ConversationCommitError):
        await iterator.__anext__()
    assert stream.state is avalan.DirectConversationStreamState.FAILED
    with pytest.raises(avalan.ConversationHandleUnavailableError) as failure:
        _ = stream.committed_handle
    assert failure.value.state is avalan.DirectConversationStreamState.FAILED
    assert store.diagnostics.checkpoints == 0
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert len(diagnostics.plans) == 1
    assert diagnostics.streams[0].closed


@pytest.mark.parametrize(
    ("label", "visible_before_failure"),
    (
        ("provider:item:0", False),
        ("provider:terminal", True),
    ),
)
async def test_unsolicited_provider_cancellation_is_a_typed_failure(
    label: str,
    visible_before_failure: bool,
) -> None:
    """Never project provider cancellation as successful exhaustion."""
    lane_binding = binding("lane-provider-cancel", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(
        plan,
        turn=1,
        text="visible-before-cancel",
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label=label,
                exception=CancelledError(),
            ),
        )
    )
    client, store, coordinator, _ = _client(
        lane_binding=lane_binding,
        results=(result,),
        provider_controller=controller,
    )
    stream = await client.create(
        "provider cancellation",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    iterator = stream.__aiter__()

    if visible_before_failure:
        assert await iterator.__anext__() == (
            avalan.DirectConversationOutputDelta(
                text_delta="visible-before-cancel"
            )
        )
    with pytest.raises(avalan.DirectConversationCancelledError):
        await iterator.__anext__()

    assert stream.state is avalan.DirectConversationStreamState.FAILED
    with pytest.raises(avalan.ConversationHandleUnavailableError) as failure:
        _ = stream.terminal
    assert failure.value.state is avalan.DirectConversationStreamState.FAILED
    assert store.diagnostics.checkpoints == 0
    assert coordinator.diagnostics.active_attempts == 0
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert len(diagnostics.streams) == 1
    assert diagnostics.streams[0].close_attempts == 1
    assert diagnostics.streams[0].closed


async def test_unsolicited_cancellation_isolated_from_concurrent_stream() -> (
    None
):
    """Commit an unrelated stream while one provider terminal cancels."""
    lane_binding = binding("lane-provider-cancel-concurrent", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    cancelled_result = conversation.fake_provider_result(
        plan,
        turn=1,
        text="cancelled-output",
    )
    completed_result = conversation.fake_provider_result(
        plan,
        turn=2,
        text="completed-output",
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:terminal",
                pause=True,
                exception=CancelledError(),
            ),
        )
    )
    client, store, coordinator, _ = _client(
        lane_binding=lane_binding,
        results=(cancelled_result, completed_result),
        provider_controller=controller,
    )
    cancelled = await client.create(
        "cancelled request",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    cancelled_iterator = cancelled.__aiter__()
    assert await cancelled_iterator.__anext__() == (
        avalan.DirectConversationOutputDelta(text_delta="cancelled-output")
    )
    await controller.wait_until_entered("provider:terminal")

    completed = await client.create(
        "completed request",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    completed_task = create_task(_consume_direct_stream(completed))
    completed_events = await completed_task
    controller.release("provider:terminal")
    with pytest.raises(avalan.DirectConversationCancelledError):
        await cancelled_iterator.__anext__()

    assert cancelled.state is avalan.DirectConversationStreamState.FAILED
    assert completed.state is avalan.DirectConversationStreamState.COMMITTED
    assert completed_events[-1].result.output == "completed-output"
    assert store.diagnostics.checkpoints == 1
    assert coordinator.diagnostics.active_attempts == 0
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert len(diagnostics.streams) == 2
    assert [item.close_attempts for item in diagnostics.streams] == [1, 1]
    assert all(item.closed for item in diagnostics.streams)


async def test_cancelled_stream_isolated_from_concurrent_commit() -> None:
    """Cancel one visible stream while an unrelated stream commits."""
    lane_binding = binding("lane-concurrent", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    cancelled_result = conversation.fake_provider_result(
        plan,
        turn=1,
        text="cancel-me",
    )
    completed_result = conversation.fake_provider_result(
        plan,
        turn=2,
        text="keep-me",
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:terminal",
                pause=True,
            ),
        )
    )
    client, store, coordinator, _ = _client(
        lane_binding=lane_binding,
        results=(cancelled_result, completed_result),
        provider_controller=controller,
    )
    cancelled = await client.create(
        "cancelled input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    cancelled_iterator = cancelled.__aiter__()
    first_event = await cancelled_iterator.__anext__()
    assert first_event == avalan.DirectConversationOutputDelta(
        text_delta="cancel-me"
    )
    await controller.wait_until_entered("provider:terminal")

    completed = await client.create(
        "completed input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    completed_task = create_task(_consume_direct_stream(completed))
    completed_events = await completed_task
    await cancelled.cancel()

    assert completed.state is avalan.DirectConversationStreamState.COMMITTED
    assert type(completed_events[-1]) is (
        avalan.DirectConversationStreamTerminal
    )
    assert completed_events[-1].result.output == "keep-me"
    assert cancelled.state is avalan.DirectConversationStreamState.CANCELLED
    with pytest.raises(avalan.ConversationHandleUnavailableError):
        _ = cancelled.committed_handle
    assert store.diagnostics.checkpoints == 1
    assert coordinator.diagnostics.active_attempts == 0
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert len(diagnostics.streams) == 2
    assert all(item.closed for item in diagnostics.streams)


async def _consume_direct_stream(
    stream: avalan.DirectConversationStream,
) -> list[avalan.DirectConversationStreamItem]:
    return [item async for item in stream]


async def test_provider_state_finalize_failure_is_safe_and_uncommitted() -> (
    None
):
    """Clean a failed private sink and withhold every resumable handle."""

    class FailingSink:
        def __init__(self) -> None:
            self.stage_calls = 0
            self.finalize_calls = 0
            self.cleanup_calls = 0

        async def stage(self, item: conversation.ProviderItem) -> None:
            assert type(item) is conversation.ProviderItem
            self.stage_calls += 1

        async def finalize(
            self,
            outputs: tuple[
                conversation.ProviderLaneOutputCandidate,
                ...,
            ],
        ) -> None:
            assert outputs
            self.finalize_calls += 1
            raise RuntimeError("private-provider-state-secret")

        async def cleanup(self) -> None:
            self.cleanup_calls += 1

    lane_binding = binding("lane-failing-sink", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    scope = authority()
    store = conversation.InMemoryConversationStore()
    coordinator = _coordinator(
        store=store,
        scope=scope,
        lane_binding=lane_binding,
        results=(result,),
    )
    run = request(
        scope=scope,
        identity=root_identity("failing-sink"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-failing-sink",),
        key="key-failing-sink",
        response_suffix="failing-sink",
    )
    sink = FailingSink()

    with pytest.raises(conversation.ConversationCommitError) as failure:
        await coordinator.stream_with_sink(run, sink)
    assert "private-provider-state-secret" not in str(failure.value)
    assert "private-provider-state-secret" not in repr(failure.value)
    assert failure.value.__cause__ is None
    assert sink.stage_calls == 1
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1
    assert store.diagnostics.checkpoints == 0
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert len(diagnostics.plans) == 1
    assert diagnostics.streams[0].closed


async def test_synchronous_sink_rejected_before_provider_dispatch() -> None:
    """Reject a sync-lookalike sidecar before opening a provider stream."""

    class SyncSink:
        def stage(self, item: conversation.ProviderItem) -> None:
            raise AssertionError(item)

        def finalize(
            self,
            outputs: tuple[
                conversation.ProviderLaneOutputCandidate,
                ...,
            ],
        ) -> None:
            raise AssertionError(outputs)

        def cleanup(self) -> None:
            raise AssertionError

    lane_binding = binding("lane-sync-sink", streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    scope = authority()
    store = conversation.InMemoryConversationStore()
    coordinator = _coordinator(
        store=store,
        scope=scope,
        lane_binding=lane_binding,
        results=(result,),
    )
    run = request(
        scope=scope,
        identity=root_identity("sync-sink"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-sync-sink",),
        key="key-sync-sink",
        response_suffix="sync-sink",
    )

    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.stream_with_sink(
            run,
            SyncSink(),  # type: ignore[arg-type]
        )
    diagnostics = coordinator.fake_provider_diagnostics(lane_binding.lane_id)
    assert diagnostics.plans == ()
    assert diagnostics.remaining_results == 1
