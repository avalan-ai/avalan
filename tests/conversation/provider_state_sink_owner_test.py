"""Exercise coordinator ownership of private provider-state sinks."""

from asyncio import CancelledError, Event, create_task, sleep
from datetime import UTC, datetime
from typing import cast

import pytest
from phase2_fixtures import (
    authority,
    binding,
    empty_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation
import avalan.conversation.coordinator as coordinator_module

pytestmark = pytest.mark.anyio

_PRIVATE_SENTINELS = (
    "private-stage-secret",
    "private-finalize-secret",
    "private-cleanup-secret",
    "private-cancel-secret",
    "private-nested-secret",
)


@pytest.fixture
def anyio_backend() -> str:
    """Run cancellation-sensitive sink tests on asyncio only."""
    return "asyncio"


class _Sink:
    def __init__(
        self,
        *,
        stage_error: BaseException | None = None,
        finalize_error: BaseException | None = None,
        cleanup_error: BaseException | None = None,
        cleanup_entered: Event | None = None,
        cleanup_release: Event | None = None,
    ) -> None:
        self.stage_error = stage_error
        self.finalize_error = finalize_error
        self.cleanup_error = cleanup_error
        self.cleanup_entered = cleanup_entered
        self.cleanup_release = cleanup_release
        self.stage_calls = 0
        self.finalize_calls = 0
        self.cleanup_calls = 0

    async def stage(self, item: conversation.ProviderItem) -> None:
        del item
        self.stage_calls += 1
        if self.stage_error is not None:
            raise self.stage_error

    async def finalize(
        self,
        outputs: tuple[conversation.ProviderLaneOutputCandidate, ...],
    ) -> None:
        del outputs
        self.finalize_calls += 1
        if self.finalize_error is not None:
            raise self.finalize_error

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.cleanup_entered is not None:
            self.cleanup_entered.set()
        if self.cleanup_release is not None:
            await self.cleanup_release.wait()
        if self.cleanup_error is not None:
            raise self.cleanup_error


def _private_error(
    message: str,
    *,
    cancellation: bool = False,
) -> BaseException:
    error: BaseException = (
        CancelledError(message) if cancellation else RuntimeError(message)
    )
    error.__cause__ = ValueError("private-nested-secret-cause")
    error.__context__ = LookupError("private-nested-secret-context")
    return error


def _assert_public_exception_is_chain_free(error: BaseException) -> None:
    """Require a recursively content-safe exception with no linked chain."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered = (str(current), repr(current), repr(current.args))
        assert all(
            sentinel not in value
            for sentinel in _PRIVATE_SENTINELS
            for value in rendered
        )
        cause = current.__cause__
        context = current.__context__
        assert cause is None
        assert context is None
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def _coordinator(
    *,
    lane: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
) -> tuple[
    conversation.RunScopedConversationCoordinator,
    conversation.InMemoryConversationStore,
    conversation.AuthorityScope,
]:
    scope = authority()
    store = conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
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
                    results=results
                ),
            ),
        ),
    )
    return coordinator, store, scope


async def test_sink_owner_rejects_reuse_and_maps_stage_failures() -> None:
    """Keep staged state single-owner and content-safe."""
    sink = _Sink()
    owner = coordinator_module._ProviderStateSinkOwner(sink)
    await owner.finalize(())
    with pytest.raises(conversation.ConversationValidationError):
        await owner.stage(cast(conversation.ProviderItem, object()))
    with pytest.raises(conversation.ConversationValidationError):
        await owner.finalize(())
    await owner.cleanup()
    await owner.cleanup()
    assert owner.cleaned

    cancelled = coordinator_module._ProviderStateSinkOwner(
        _Sink(
            stage_error=_private_error(
                "private-cancel-secret",
                cancellation=True,
            )
        )
    )
    with pytest.raises(CancelledError) as cancellation:
        await cancelled.stage(cast(conversation.ProviderItem, object()))
    _assert_public_exception_is_chain_free(cancellation.value)

    failed = coordinator_module._ProviderStateSinkOwner(
        _Sink(stage_error=_private_error("private-stage-secret"))
    )
    with pytest.raises(conversation.ConversationCommitError) as failure:
        await failed.stage(cast(conversation.ProviderItem, object()))
    _assert_public_exception_is_chain_free(failure.value)


async def test_sink_owner_maps_finalize_and_cleanup_failures() -> None:
    """Propagate cancellation while replacing arbitrary private failures."""
    cancelled_finalize = coordinator_module._ProviderStateSinkOwner(
        _Sink(
            finalize_error=_private_error(
                "private-cancel-secret",
                cancellation=True,
            )
        )
    )
    with pytest.raises(CancelledError) as finalize_cancellation:
        await cancelled_finalize.finalize(())
    _assert_public_exception_is_chain_free(finalize_cancellation.value)

    cancelled_cleanup = coordinator_module._ProviderStateSinkOwner(
        _Sink(
            cleanup_error=_private_error(
                "private-cancel-secret",
                cancellation=True,
            )
        )
    )
    with pytest.raises(CancelledError) as cleanup_cancellation:
        await cancelled_cleanup.cleanup()
    _assert_public_exception_is_chain_free(cleanup_cancellation.value)

    failed_cleanup = coordinator_module._ProviderStateSinkOwner(
        _Sink(cleanup_error=_private_error("private-cleanup-secret"))
    )
    with pytest.raises(conversation.ConversationCommitError) as failure:
        await failed_cleanup.cleanup()
    _assert_public_exception_is_chain_free(failure.value)

    failed_finalize = coordinator_module._ProviderStateSinkOwner(
        _Sink(finalize_error=_private_error("private-finalize-secret"))
    )
    with pytest.raises(conversation.ConversationCommitError) as finalize:
        await failed_finalize.finalize(())
    _assert_public_exception_is_chain_free(finalize.value)


async def test_sink_owner_finishes_cleanup_before_restoring_cancellation() -> (
    None
):
    """Shield the single cleanup task and then restore caller cancellation."""
    entered = Event()
    release = Event()
    sink = _Sink(cleanup_entered=entered, cleanup_release=release)
    owner = coordinator_module._ProviderStateSinkOwner(sink)
    task = create_task(owner.cleanup())
    await entered.wait()
    task.cancel()
    await sleep(0)
    release.set()
    with pytest.raises(CancelledError) as failure:
        await task
    _assert_public_exception_is_chain_free(failure.value)
    assert owner.cleaned
    assert sink.cleanup_calls == 1


async def test_sink_owner_shares_one_concurrent_cleanup_task() -> None:
    """Join one private cleanup task from multiple concurrent owners."""
    entered = Event()
    release = Event()
    sink = _Sink(cleanup_entered=entered, cleanup_release=release)
    owner = coordinator_module._ProviderStateSinkOwner(sink)
    first = create_task(owner.cleanup())
    await entered.wait()
    second = create_task(owner.cleanup())
    await sleep(0)

    release.set()
    await first
    await second

    assert owner.cleaned
    assert sink.cleanup_calls == 1


def test_cleanup_failure_without_primary_is_propagated() -> None:
    """Propagate the already-sanitized defensive cleanup fallback."""
    expected = conversation.ConversationCommitError()

    with pytest.raises(conversation.ConversationCommitError) as failure:
        coordinator_module._apply_provider_state_cleanup_failure(
            None,
            expected,
        )

    assert failure.value is expected
    _assert_public_exception_is_chain_free(failure.value)


@pytest.mark.parametrize("cleanup_error", [CancelledError(), RuntimeError()])
async def test_coordinator_retains_primary_failure_when_cleanup_fails(
    cleanup_error: BaseException,
) -> None:
    """Annotate a primary commit failure without replacing its safe type."""
    lane = binding("lane-owner-cleanup", streaming=True)
    plan = empty_stateless_plan(lane)
    result = conversation.fake_provider_result(plan, turn=1)
    coordinator, _, scope = _coordinator(lane=lane, results=(result,))
    run = request(
        scope=scope,
        identity=root_identity("owner-cleanup"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-owner-cleanup",),
        key="key-owner-cleanup",
        response_suffix="owner-cleanup",
    )
    sink = _Sink(
        stage_error=RuntimeError("private-stage"),
        cleanup_error=cleanup_error,
    )

    with pytest.raises(conversation.ConversationCommitError) as failure:
        await coordinator.stream_with_sink(run, sink)
    _assert_public_exception_is_chain_free(failure.value)
    assert failure.value.__notes__ == [
        "conversation provider-state cleanup failed"
    ]


async def test_stream_replay_finalizes_new_sink_and_compact_guards() -> None:
    """Finalize replayed state and reject non-compact coordinator requests."""
    lane = binding("lane-owner-replay", streaming=True)
    plan = empty_stateless_plan(lane)
    result = conversation.fake_provider_result(plan, turn=1)
    coordinator, _, scope = _coordinator(lane=lane, results=(result,))
    run = request(
        scope=scope,
        identity=root_identity("owner-replay"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("lane-owner-replay",),
        key="key-owner-replay",
        response_suffix="owner-replay",
    )
    first_sink = _Sink()
    second_sink = _Sink()

    first = await coordinator.stream_with_sink(run, first_sink)
    replay = await coordinator.stream_with_sink(run, second_sink)
    assert replay.checkpoint == first.checkpoint
    assert second_sink.stage_calls == 0
    assert second_sink.finalize_calls == 1
    assert second_sink.cleanup_calls == 1

    with pytest.raises(conversation.ConversationValidationError):
        await coordinator.compact(run)
    with pytest.raises(conversation.ConversationValidationError):
        await coordinator._run(run, streaming=False, sink=_Sink())
    with pytest.raises(conversation.ConversationValidationError):
        coordinator._validate_compact_outputs(())
