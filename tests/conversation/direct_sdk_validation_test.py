"""Exercise defensive direct conversation SDK boundaries."""

from asyncio import CancelledError, Event, Queue, create_task, sleep
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest
from phase2_fixtures import (
    authority,
    binding,
    empty_stateless_plan,
    retention,
    semantics,
)

import avalan
import avalan.conversation as conversation
import avalan.conversation.runtime as conversation_runtime
import avalan.conversation.sdk as direct_sdk
import avalan.conversation.settings as conversation_settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run cancellation-sensitive direct SDK tests on asyncio only."""
    return "asyncio"


def _coordinator(
    store: conversation.InMemoryConversationStore,
    scope: conversation.AuthorityScope,
    lane: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...] = (),
) -> conversation.RunScopedConversationCoordinator:
    selected_results = results or (
        conversation.fake_provider_result(
            empty_stateless_plan(lane),
            turn=999,
        ),
    )
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
                binding=lane,
                capability_profile=conversation.fake_capability_profile(lane),
                provider_script=conversation.DeterministicFakeProviderScript(
                    results=selected_results
                ),
            ),
        ),
    )


def _client(
    results: tuple[conversation.ProviderResult, ...] = (),
    *,
    lane: conversation.ProviderLaneBinding | None = None,
    store: conversation.InMemoryConversationStore | None = None,
    scope: conversation.AuthorityScope | None = None,
    configured_retention: conversation.RetentionLimits | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
    conversation.AuthorityScope,
]:
    selected_lane = lane or binding("lane-validation")
    selected_store = store or conversation.InMemoryConversationStore()
    selected_scope = scope or authority()
    coordinator = _coordinator(
        selected_store,
        selected_scope,
        selected_lane,
        results,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=selected_store,
        authority=selected_scope,
        lane=selected_lane,
        retention=configured_retention or retention(),
        id_namespace="validation",
    )
    return (
        avalan.DirectConversationClient(runtime),
        selected_store,
        coordinator,
        selected_scope,
    )


def _handle(suffix: str = "validation") -> avalan.StatelessConversationHandle:
    return avalan.StatelessConversationHandle(
        conversation_id=conversation.ConversationId(f"conversation-{suffix}"),
        checkpoint_id=conversation.CheckpointId(f"checkpoint-{suffix}"),
        branch_id=conversation.ConversationBranchId(f"branch-{suffix}"),
    )


def _result() -> avalan.DirectConversationResult:
    return avalan.DirectConversationResult(
        output="visible",
        usage=conversation.ProviderUsage(input_tokens=1, output_tokens=1),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
        ),
        handle=_handle(),
    )


def test_public_value_objects_reject_invalid_runtime_values() -> None:
    """Reject malformed public results, events, and stream states."""
    with pytest.raises(conversation.ConversationValidationError):
        avalan.ConversationHandleUnavailableError(
            cast(avalan.DirectConversationStreamState, "pending")
        )
    unavailable = avalan.ConversationHandleUnavailableError(
        avalan.DirectConversationStreamState.PENDING
    )
    assert "pending" in repr(unavailable)

    usage = conversation.ProviderUsage(input_tokens=1, output_tokens=1)
    reasoning = conversation.EffectiveReasoningMetadata(
        requested=conversation.ReasoningContext.AUTO,
        effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
    )
    handle = _handle()
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationResult(
            output=cast(str, 1),
            usage=usage,
            reasoning=reasoning,
            handle=handle,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationResult(
            output="visible",
            usage=cast(conversation.ProviderUsage, object()),
            reasoning=reasoning,
            handle=handle,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationResult(
            output="visible",
            usage=usage,
            reasoning=cast(
                conversation.EffectiveReasoningMetadata,
                object(),
            ),
            handle=handle,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationResult(
            output="visible",
            usage=usage,
            reasoning=reasoning,
            handle=cast(avalan.ConversationHandle, object()),
        )

    for value in ("", cast(str, 1)):
        with pytest.raises(conversation.ConversationValidationError):
            avalan.DirectConversationOutputDelta(text_delta=value)
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationStreamTerminal(
            result=cast(avalan.DirectConversationResult, object())
        )


def test_runtime_and_client_reject_untrusted_bindings() -> None:
    """Require the exact fake coordinator, authority, lane, and retention."""
    scope = authority()
    lane = binding("lane-runtime-validation")
    store = conversation.InMemoryConversationStore()
    coordinator = _coordinator(store, scope, lane)
    limits = retention()

    invalid_values = (
        {
            "coordinator": cast(
                conversation.RunScopedConversationCoordinator, object()
            ),
            "authority": scope,
            "lane": lane,
            "retention": limits,
        },
        {
            "coordinator": coordinator,
            "authority": cast(conversation.AuthorityScope, object()),
            "lane": lane,
            "retention": limits,
        },
        {
            "coordinator": coordinator,
            "authority": scope,
            "lane": cast(conversation.ProviderLaneBinding, object()),
            "retention": limits,
        },
        {
            "coordinator": coordinator,
            "authority": scope,
            "lane": lane,
            "retention": cast(conversation.RetentionLimits, object()),
        },
        {
            "coordinator": coordinator,
            "authority": scope,
            "lane": replace(
                lane,
                provider_family=conversation.ProviderFamily.OPENAI,
            ),
            "retention": limits,
        },
        {
            "coordinator": coordinator,
            "authority": scope,
            "lane": replace(
                lane,
                agent_id=conversation.ConversationAgentId("other-agent"),
            ),
            "retention": limits,
        },
    )
    for values in invalid_values:
        with pytest.raises(conversation.ConversationValidationError):
            avalan.DirectConversationRuntime(store=store, **values)

    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationRuntime(
            coordinator=coordinator,
            store=store,
            authority=scope,
            lane=lane,
            retention=limits,
            id_namespace=" ",
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationRuntime(
            coordinator=coordinator,
            store=conversation.InMemoryConversationStore(),
            authority=scope,
            lane=lane,
            retention=limits,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.DirectConversationClient(
            cast(avalan.DirectConversationRuntime, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator.validate_direct_runtime(
            store,
            binding("lane-runtime-missing"),
        )


async def test_private_stream_sink_and_queue_fail_closed() -> None:
    """Reject invalid sink state and impossible public queue payloads."""
    queue: Queue[object] = Queue()
    sink = direct_sdk._DirectStreamSink(queue)
    await sink.cleanup()
    await sink.cleanup()
    with pytest.raises(conversation.ConversationValidationError):
        await sink.stage(cast(conversation.ProviderItem, object()))

    sink = direct_sdk._DirectStreamSink(queue)
    with pytest.raises(conversation.ConversationValidationError):
        await sink.finalize(())

    class UnusedCoordinator:
        async def stream_with_sink(
            self,
            request: conversation.ConversationRunRequest,
            state_sink: conversation.ConversationProviderStateSink,
        ) -> conversation.AtomicCommitReceipt:
            raise AssertionError((request, state_sink))

    async def done() -> None:
        return None

    stream = avalan.DirectConversationStream(
        cast(
            conversation.RunScopedConversationCoordinator,
            UnusedCoordinator(),
        ),
        cast(conversation.ConversationRunRequest, object()),
    )
    completed_task = create_task(done())
    await completed_task
    setattr(stream, "_task", completed_task)
    terminal = avalan.DirectConversationStreamTerminal(result=_result())
    stream_queue = cast(Queue[object], getattr(stream, "_queue"))
    stream_queue.put_nowait(terminal)
    stream_queue.put_nowait(terminal)
    assert await stream.__anext__() is terminal
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()

    invalid_stream = avalan.DirectConversationStream(
        cast(
            conversation.RunScopedConversationCoordinator,
            UnusedCoordinator(),
        ),
        cast(conversation.ConversationRunRequest, object()),
    )
    setattr(invalid_stream, "_task", completed_task)
    cast(Queue[object], getattr(invalid_stream, "_queue")).put_nowait(object())
    with pytest.raises(avalan.DirectConversationStreamError):
        await invalid_stream.__anext__()
    assert invalid_stream.state is avalan.DirectConversationStreamState.FAILED
    await invalid_stream.aclose()
    assert "failed" in repr(invalid_stream)

    cancelled_stream = avalan.DirectConversationStream(
        cast(
            conversation.RunScopedConversationCoordinator,
            UnusedCoordinator(),
        ),
        cast(conversation.ConversationRunRequest, object()),
    )
    setattr(cancelled_stream, "_task", completed_task)
    next_task = create_task(cancelled_stream.__anext__())
    await sleep(0)
    next_task.cancel()
    with pytest.raises(CancelledError):
        await next_task
    assert cancelled_stream.state is (
        avalan.DirectConversationStreamState.CANCELLED
    )


async def test_stream_iteration_and_cancellation_are_single_owner() -> None:
    """Await worker cleanup under consumer and closer cancellation."""

    class BlockingCoordinator:
        def __init__(self, *, slow_cancel: bool) -> None:
            self.started = Event()
            self.cancel_entered = Event()
            self.cancel_release = Event()
            self.slow_cancel = slow_cancel

        async def stream_with_sink(
            self,
            request: conversation.ConversationRunRequest,
            state_sink: conversation.ConversationProviderStateSink,
        ) -> conversation.AtomicCommitReceipt:
            del request, state_sink
            self.started.set()
            try:
                await Event().wait()
            except CancelledError:
                self.cancel_entered.set()
                if self.slow_cancel:
                    await self.cancel_release.wait()
                raise

    coordinator = BlockingCoordinator(slow_cancel=False)
    stream = avalan.DirectConversationStream(
        cast(conversation.RunScopedConversationCoordinator, coordinator),
        cast(conversation.ConversationRunRequest, object()),
    )
    iterator = stream.__aiter__()
    with pytest.raises(RuntimeError, match="single-use"):
        stream.__aiter__()
    next_task = create_task(iterator.__anext__())
    await coordinator.started.wait()
    next_task.cancel()
    with pytest.raises(CancelledError):
        await next_task
    assert stream.state is avalan.DirectConversationStreamState.CANCELLED

    slow = BlockingCoordinator(slow_cancel=True)
    closing_stream = avalan.DirectConversationStream(
        cast(conversation.RunScopedConversationCoordinator, slow),
        cast(conversation.ConversationRunRequest, object()),
    )
    closing_stream.__aiter__()
    await slow.started.wait()
    close_task = create_task(closing_stream.cancel())
    await slow.cancel_entered.wait()
    close_task.cancel()
    slow.cancel_release.set()
    with pytest.raises(CancelledError):
        await close_task
    assert closing_stream.state is (
        avalan.DirectConversationStreamState.CANCELLED
    )
    await closing_stream.aclose()


async def test_stream_wraps_non_domain_worker_failures() -> None:
    """Replace arbitrary worker exceptions with one content-safe error."""

    class FailingCoordinator:
        async def stream_with_sink(
            self,
            request: conversation.ConversationRunRequest,
            state_sink: conversation.ConversationProviderStateSink,
        ) -> conversation.AtomicCommitReceipt:
            raise RuntimeError(f"secret: {request!r} {state_sink!r}")

    stream = avalan.DirectConversationStream(
        cast(
            conversation.RunScopedConversationCoordinator,
            FailingCoordinator(),
        ),
        cast(conversation.ConversationRunRequest, object()),
    )
    with pytest.raises(avalan.DirectConversationStreamError) as failure:
        await stream.__anext__()
    assert "secret" not in str(failure.value)
    assert stream.state is avalan.DirectConversationStreamState.FAILED


async def test_client_validation_precedes_provider_dispatch() -> None:
    """Reject malformed direct operations before provider execution."""
    client, _, coordinator, _ = _client()
    parent = avalan.StatelessParent(handle=_handle("operation"))
    branch = avalan.ConversationBranchIntent(
        parent=parent,
        branch_id=conversation.ConversationBranchId("branch-operation-child"),
    )

    with pytest.raises(conversation.ConversationValidationError):
        await client.create(
            "input",
            avalan.StatelessConversationSettings(parent=parent),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.continue_conversation(
            "input",
            avalan.StatelessConversationSettings(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.branch(
            "input",
            avalan.StatelessConversationSettings(parent=parent),
        )
    conflicting = avalan.StatelessConversationSettings(
        parent=parent,
        branch=branch,
    )
    object.__setattr__(
        conflicting,
        "named_head",
        avalan.NamedHeadParent(
            head_id=conversation.NamedHeadId("head-operation"),
            expected_revision=conversation.NamedHeadRevision(0),
            parent=parent,
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.branch(
            "input",
            conflicting,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.reset(
            "input",
            cast(avalan.ConversationResetIntent, object()),
            avalan.StatelessConversationSettings(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.reset(
            "input",
            avalan.ConversationResetIntent(
                parent=parent,
                target_mode=avalan.ConversationMode.STATELESS,
            ),
            avalan.StatelessConversationSettings(parent=parent),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.compact(cast(avalan.StandaloneCompactRequest, object()))
    with pytest.raises(conversation.ConversationValidationError):
        await client.create(
            "input",
            avalan.StatelessConversationSettings(),
            stream=cast(bool, 1),
        )

    invalid_parent_settings = avalan.StatelessConversationSettings()
    object.__setattr__(invalid_parent_settings, "parent", object())
    with pytest.raises(conversation.ConversationValidationError):
        await client.continue_conversation("input", invalid_parent_settings)
    assert (
        coordinator.fake_provider_diagnostics(
            binding("lane-validation").lane_id
        ).plans
        == ()
    )


@pytest.mark.parametrize("value", [None, "", "   ", 1])
async def test_direct_input_validation_rejects_empty_or_untyped(
    value: object,
) -> None:
    """Require one bounded non-blank direct input string."""
    client, _, _, _ = _client()
    with pytest.raises(conversation.ConversationValidationError):
        await client.create(
            cast(str, value),
            avalan.StatelessConversationSettings(),
        )


async def test_direct_input_validation_rejects_oversize_text() -> None:
    """Reject direct input beyond the closed one-megabyte bound."""
    client, _, _, _ = _client()
    with pytest.raises(conversation.ConversationValidationError):
        await client.create(
            "x" * 1_048_577,
            avalan.StatelessConversationSettings(),
        )


@pytest.mark.parametrize(
    "key",
    (
        conversation.RequestIdempotencyKey(""),
        conversation.RequestIdempotencyKey(" "),
        conversation.RequestIdempotencyKey("bad\x00key"),
        conversation.RequestIdempotencyKey("x" * 513),
        cast(conversation.RequestIdempotencyKey, 1),
    ),
)
async def test_explicit_idempotency_key_rejects_before_storage_or_dispatch(
    key: conversation.RequestIdempotencyKey,
) -> None:
    """Distinguish invalid explicit keys from an omitted generated key."""
    store_controller = conversation.DeterministicFaultController()
    store = conversation.InMemoryConversationStore(
        boundary_hook=conversation.FakeStoreBoundaryHook(store_controller)
    )
    client, _, coordinator, _ = _client(store=store)
    parent = avalan.StatelessParent(handle=_handle("invalid-key"))
    parent_settings = avalan.StatelessConversationSettings(parent=parent)
    branch_settings = avalan.StatelessConversationSettings(
        parent=parent,
        branch=avalan.ConversationBranchIntent(
            parent=parent,
            branch_id=conversation.ConversationBranchId(
                "branch-invalid-key-child"
            ),
        ),
    )

    with pytest.raises(conversation.ConversationValidationError):
        await client.create(
            "create",
            avalan.StatelessConversationSettings(),
            idempotency_key=key,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.continue_conversation(
            "continue",
            parent_settings,
            idempotency_key=key,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.branch(
            "branch",
            branch_settings,
            idempotency_key=key,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.reset(
            "reset",
            avalan.ConversationResetIntent(
                parent=parent,
                target_mode=avalan.ConversationMode.STATELESS,
            ),
            avalan.StatelessConversationSettings(),
            idempotency_key=key,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await client.compact(
            avalan.StandaloneCompactRequest(parent=parent),
            idempotency_key=key,
        )

    assert store_controller.visited == ()
    assert store.diagnostics.checkpoints == 0
    diagnostics = coordinator.fake_provider_diagnostics(
        binding("lane-validation").lane_id
    )
    assert diagnostics.plans == ()
    assert diagnostics.remaining_results == 1
    assert getattr(client, "_sequence") == 0


async def test_explicit_idempotency_retry_has_one_provider_effect() -> None:
    """Replay one valid key without repeating provider or store effects."""
    lane = binding("lane-idempotency-retry")
    plan = empty_stateless_plan(lane)
    first_result = conversation.fake_provider_result(
        plan,
        turn=1,
        text="first-effect",
    )
    unused_result = conversation.fake_provider_result(
        plan,
        turn=2,
        text="duplicate-effect",
    )
    client, store, coordinator, _ = _client(
        (first_result, unused_result),
        lane=lane,
    )
    key = conversation.RequestIdempotencyKey("explicit-retry-key")

    first = await client.create(
        "retry input",
        avalan.StatelessConversationSettings(),
        idempotency_key=key,
    )
    replay = await client.create(
        "retry input",
        avalan.StatelessConversationSettings(),
        idempotency_key=key,
    )

    assert replay == first
    assert first.output == "first-effect"
    diagnostics = coordinator.fake_provider_diagnostics(lane.lane_id)
    assert len(diagnostics.plans) == 1
    assert diagnostics.remaining_results == 1
    assert store.diagnostics.checkpoints == 1
    assert store.diagnostics.idempotency_records == 1
    assert store.diagnostics.public_responses == 1


def test_active_settings_and_retention_narrowing_are_exact() -> None:
    """Reject retention widening and cross-mode storage selection."""
    with pytest.raises(conversation.ConversationValidationError):
        direct_sdk._validate_active_settings(object())

    configured = retention(ttl=100)
    stored = retention(stored=True, ttl=100)
    local_mismatch = replace(
        configured,
        storage=replace(
            configured.storage,
            local=conversation.LocalResponseStorage.TRANSIENT,
        ),
    )
    disclosure_mismatch = replace(
        configured,
        storage=replace(
            configured.storage,
            provider_storage_disclosed=True,
        ),
    )
    lifetime_mismatch = replace(configured)
    object.__setattr__(
        lifetime_mismatch,
        "upstream_lifetime_status",
        conversation.UpstreamLifetimeStatus.UNKNOWN,
    )
    requested_without_ttl = replace(configured, local_ttl_seconds=None)
    widened_ttl = replace(configured, local_ttl_seconds=101)
    cases = (
        (cast(conversation.RetentionLimits, object()), configured),
        (stored, configured),
        (local_mismatch, configured),
        (disclosure_mismatch, configured),
        (lifetime_mismatch, configured),
        (requested_without_ttl, configured),
        (widened_ttl, configured),
    )
    for requested, runtime_limits in cases:
        with pytest.raises(conversation.ConversationValidationError):
            direct_sdk._validated_retention(
                requested,
                runtime_limits,
                conversation.ConversationMode.STATELESS,
            )

    unbounded = conversation.RetentionLimits(
        storage=configured.storage,
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        direct_sdk._validated_retention(
            replace(unbounded, local_ttl_seconds=1),
            unbounded,
            conversation.ConversationMode.STATELESS,
        )


def test_branch_reset_and_advance_settings_reject_invalid_values() -> None:
    """Reject untyped, contradictory, or parent-drifting advance intent."""
    parent = avalan.StatelessParent(handle=_handle("settings-parent"))
    other_parent = avalan.StatelessParent(handle=_handle("settings-other"))
    with pytest.raises(conversation.ConversationValidationError):
        avalan.ConversationBranchIntent(
            parent=cast(avalan.ConversationParent, object()),
            branch_id=conversation.ConversationBranchId("branch-invalid"),
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.ConversationBranchIntent(
            parent=parent,
            branch_id=parent.handle.branch_id,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.ConversationResetIntent(
            parent=cast(avalan.ConversationParent, object()),
            target_mode=avalan.ConversationMode.STATELESS,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.ConversationResetIntent(
            parent=parent,
            target_mode=avalan.ConversationMode.OFF,
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.ConversationResetIntent(
            parent=parent,
            target_mode=avalan.ConversationMode.STATELESS,
            provider_storage_disclosed=cast(bool, 1),
        )
    for mode, disclosed in (
        (avalan.ConversationMode.STATELESS, True),
        (avalan.ConversationMode.STORED, False),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            avalan.ConversationResetIntent(
                parent=parent,
                target_mode=mode,
                provider_storage_disclosed=disclosed,
            )

    branch = avalan.ConversationBranchIntent(
        parent=parent,
        branch_id=conversation.ConversationBranchId("branch-settings-child"),
    )
    named_head = avalan.NamedHeadParent(
        head_id=conversation.NamedHeadId("head-settings"),
        expected_revision=conversation.NamedHeadRevision(0),
        parent=parent,
    )
    invalid_advance_values = (
        (
            parent,
            None,
            None,
            cast(conversation.RetentionLimits, object()),
        ),
        (
            parent,
            cast(avalan.ConversationBranchIntent, object()),
            None,
            None,
        ),
        (
            parent,
            None,
            cast(avalan.NamedHeadParent, object()),
            None,
        ),
        (parent, branch, named_head, None),
        (other_parent, branch, None, None),
    )
    for values in invalid_advance_values:
        with pytest.raises(conversation.ConversationValidationError):
            conversation_settings._validate_advance_settings(*values)


def test_runtime_operation_and_compaction_validation_is_closed() -> None:
    """Require exact operation, boundary, lane, and fake compact inputs."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ConversationLaneRequest(
            lane_id=conversation.ProviderLaneId("lane-invalid-compaction"),
            mode=conversation.ConversationMode.STATELESS,
            compaction=cast(conversation.CompactionPolicy, object()),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation_runtime.request_operation(
            cast(conversation.ConversationRunRequest, object())
        )

    scope = authority()
    shaped = object.__new__(conversation.ConversationRunRequest)
    object.__setattr__(
        shaped,
        "semantics",
        semantics(
            scope,
            operation=conversation.ConversationOperation.CONTINUE,
            mode=conversation.ConversationMode.STATELESS,
        ),
    )
    object.__setattr__(shaped, "advance", conversation.FirstTurnAdvance())
    object.__setattr__(
        shaped,
        "boundary",
        conversation.ConversationCommitBoundary.INTERNAL_SEGMENT,
    )
    object.__setattr__(
        shaped,
        "lanes",
        (
            conversation.ConversationLaneRequest(
                lane_id=conversation.ProviderLaneId("lane-shaped"),
                mode=conversation.ConversationMode.STATELESS,
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation_runtime._validate_request_operation(shaped)

    object.__setattr__(
        shaped,
        "semantics",
        semantics(
            scope,
            operation=conversation.ConversationOperation.COMPACT,
            mode=conversation.ConversationMode.STATELESS,
            parent_id=conversation.CheckpointId("checkpoint-shaped"),
        ),
    )
    object.__setattr__(
        shaped,
        "advance",
        conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=conversation.CheckpointId("checkpoint-shaped")
        ),
    )
    object.__setattr__(
        shaped,
        "boundary",
        conversation.ConversationCommitBoundary.OUTWARD_TURN,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation_runtime._validate_request_operation(shaped)

    plan = empty_stateless_plan(binding("lane-fake-compact-validation"))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.fake_compaction_result(
            cast(conversation.StatelessProviderPlan, object()),
            turn=1,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.fake_compaction_result(plan, turn=0)


async def test_parent_identity_and_lane_membership_are_exact() -> None:
    """Reject handles with identity drift or a missing runtime lane."""
    lane_a = binding("lane-parent-a")
    first_plan = empty_stateless_plan(lane_a)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    client_a, store, _, scope = _client((first_result,), lane=lane_a)
    first = await client_a.create(
        "first",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle

    mismatched = avalan.StatelessParent(
        handle=avalan.StatelessConversationHandle(
            conversation_id=conversation.ConversationId("wrong-conversation"),
            checkpoint_id=first.handle.checkpoint_id,
            branch_id=first.handle.branch_id,
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client_a.continue_conversation(
            "identity drift",
            avalan.StatelessConversationSettings(parent=mismatched),
        )

    lane_b = binding("lane-parent-b")
    client_b, _, _, _ = _client(
        lane=lane_b,
        store=store,
        scope=scope,
    )
    parent = avalan.StatelessParent(handle=first.handle)
    with pytest.raises(conversation.ConversationValidationError):
        await client_b.continue_conversation(
            "missing lane",
            avalan.StatelessConversationSettings(parent=parent),
        )


async def test_compact_rejects_receipt_without_integrity() -> None:
    """Require an integrity-bound digest on every compacted checkpoint."""
    lane = binding("lane-compact-integrity")
    plan = empty_stateless_plan(lane)
    provider_result = conversation.fake_provider_result(plan, turn=1)
    client, store, _, scope = _client((provider_result,), lane=lane)
    first = await client.create(
        "first",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    checkpoint = await store.load(first.handle.checkpoint_id, scope)
    integrityless = replace(checkpoint, integrity=None)

    class IntegritylessCoordinator:
        async def compact(
            self,
            request: conversation.ConversationRunRequest,
        ) -> conversation.AtomicCommitReceipt:
            del request
            receipt = object.__new__(conversation.AtomicCommitReceipt)
            object.__setattr__(receipt, "checkpoint", integrityless)
            return receipt

    runtime = cast(
        avalan.DirectConversationRuntime,
        getattr(client, "_runtime"),
    )
    object.__setattr__(
        runtime,
        "coordinator",
        cast(
            conversation.RunScopedConversationCoordinator,
            IntegritylessCoordinator(),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.compact(
            avalan.StandaloneCompactRequest(
                parent=avalan.StatelessParent(handle=first.handle)
            )
        )


async def test_inline_compaction_requires_explicit_capability() -> None:
    """Carry typed inline compaction through capability planning."""
    lane = binding("lane-inline-compaction")
    plan = empty_stateless_plan(lane)
    provider_result = conversation.fake_provider_result(plan, turn=1)
    client, _, _, _ = _client((provider_result,), lane=lane)

    result = await client.create(
        "inline compact",
        avalan.StatelessConversationSettings(
            compaction=avalan.InlineCompaction(compact_threshold=128)
        ),
    )
    assert result.output


def test_internal_result_and_visible_item_guards_are_closed() -> None:
    """Reject impossible receipts and ignore non-visible provider payloads."""
    with pytest.raises(conversation.ConversationValidationError):
        direct_sdk._direct_result(
            cast(conversation.AtomicCommitReceipt, object())
        )

    receipt = object.__new__(conversation.AtomicCommitReceipt)
    object.__setattr__(receipt, "result", None)
    object.__setattr__(receipt, "output_candidates", ())
    with pytest.raises(conversation.ConversationValidationError):
        direct_sdk._direct_result(receipt)

    object.__setattr__(receipt, "result", object())
    with pytest.raises(conversation.ConversationValidationError):
        direct_sdk._direct_result(receipt)

    assert (
        direct_sdk._visible_provider_item_text(
            cast(conversation.ProviderItem, object())
        )
        == ""
    )
    item = object.__new__(conversation.ProviderItem)
    object.__setattr__(item, "kind", conversation.ProviderItemKind.MESSAGE)
    object.__setattr__(item, "phase", conversation.ProviderItemPhase.ASSISTANT)
    object.__setattr__(item, "canonical_input", {"content": []})
    assert direct_sdk._visible_provider_item_text(item) == ""
    object.__setattr__(
        item,
        "canonical_input",
        {
            "content": (
                "not-a-mapping",
                {"type": "input_text", "text": "private"},
                {"type": "output_text", "text": 1},
                {"type": "output_text", "text": "visible"},
            )
        },
    )
    assert direct_sdk._visible_provider_item_text(item) == "visible"
