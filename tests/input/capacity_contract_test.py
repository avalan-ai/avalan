"""Exercise bounded high-concurrency interaction waiting."""

from asyncio import (
    Event,
    all_tasks,
    create_task,
    current_task,
    gather,
    run,
    sleep,
    wait_for,
)
from gc import collect
from pathlib import Path
from sys import path as sys_path
from tracemalloc import get_traced_memory, start, stop

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))

import interaction_memory_store_test as memory_support  # noqa: E402

from avalan.interaction import (  # noqa: E402
    InteractionPolicy,
    InteractionRecord,
    ResolveInteractionApplied,
    WaitForInteractionChangeCommand,
)
from avalan.interaction.stores.memory import (
    MemoryInteractionStore,  # noqa: E402
)

_WAITER_COUNT = 10_000
_PENDING_INTERACTION_LIMIT = 1_024
_MAXIMUM_PEAK_BYTES = 128 * 1_024 * 1_024
_MAXIMUM_RETAINED_BYTES = 32 * 1_024 * 1_024
_COMPLETION_TIMEOUT_SECONDS = 30


async def _wait_for_registration(
    store: MemoryInteractionStore,
    expected: int,
) -> None:
    """Wait until every scheduled waiter owns one store registration."""
    while len(store._record_waiters) != expected:
        await sleep(0)


def test_large_waiter_capacity_liveness_and_cleanup() -> None:
    """Wake ten thousand waiters without starvation or retained tasks."""

    async def exercise() -> None:
        policy = InteractionPolicy()
        assert (
            policy.maximum_pending_interactions_per_process
            == _PENDING_INTERACTION_LIMIT
        )
        factory, _, _ = memory_support._factory(policy=policy)
        store = await factory.open()
        created = await memory_support._create(
            store,
            memory_support._request("ten-thousand-waiters"),
        )
        command = WaitForInteractionChangeCommand(
            actor=memory_support._actor(created.record.request),
            correlation=created.record.correlation,
            after_store_revision=created.record.store_revision,
        )
        baseline_tasks = len(all_tasks())
        stopped = Event()
        heartbeat_ready = Event()
        pulses = 0

        async def heartbeat() -> None:
            nonlocal pulses
            while not stopped.is_set():
                pulses += 1
                if pulses > 1:
                    heartbeat_ready.set()
                await sleep(0)

        heartbeat_task = create_task(heartbeat(), name="waiter-heartbeat")
        waiters = [
            create_task(
                store.wait_for_change(command),
                name=f"interaction-waiter-{index}",
            )
            for index in range(_WAITER_COUNT)
        ]
        try:
            await wait_for(
                _wait_for_registration(store, _WAITER_COUNT),
                timeout=30,
            )
            await wait_for(
                heartbeat_ready.wait(),
                timeout=_COMPLETION_TIMEOUT_SECONDS,
            )
            assert pulses > 1
            assert len(store._record_waiters) == _WAITER_COUNT
            assert len(all_tasks()) <= baseline_tasks + _WAITER_COUNT + 1
            _, peak_bytes = get_traced_memory()
            assert peak_bytes < _MAXIMUM_PEAK_BYTES

            resolved = await store.resolve(
                memory_support._answer(created.record, "wake-all")
            )
            assert isinstance(resolved, ResolveInteractionApplied)
            projections = await wait_for(
                gather(*waiters),
                timeout=_COMPLETION_TIMEOUT_SECONDS,
            )
            assert len(projections) == _WAITER_COUNT
            assert all(
                isinstance(projection, InteractionRecord)
                and projection == resolved.record
                for projection in projections
            )
            assert store._record_waiters == {}
            del projections
        finally:
            stopped.set()
            for waiter in waiters:
                waiter.cancel()
            await gather(*waiters, return_exceptions=True)
            await heartbeat_task
            waiters.clear()
            await store.aclose()
        collect()
        retained_bytes, _ = get_traced_memory()
        assert retained_bytes < _MAXIMUM_RETAINED_BYTES
        active = current_task()
        assert all(task is active or task.done() for task in all_tasks())

    start()
    try:
        run(exercise())
    finally:
        stop()
