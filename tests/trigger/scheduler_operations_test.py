"""Prove bounded waiting never turns a live operation into settled work."""

from asyncio import CancelledError, Event, create_task
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.trigger.scheduler_operations import (
    SchedulerOperationKind,
    SchedulerOperations,
    SchedulerOperationTimeout,
)


class SchedulerOperationsTest(IsolatedAsyncioTestCase):
    async def test_terminal_results_and_failures_are_consumed(self) -> None:
        owner = SchedulerOperations()

        async def result() -> int:
            return 7

        async def failed() -> int:
            raise ValueError("safe test failure")

        assert (
            await owner.call(
                result(), kind=SchedulerOperationKind.DISCOVERY, timeout=1
            )
            == 7
        )
        with raises(ValueError):
            await owner.call(
                failed(), kind=SchedulerOperationKind.DISCOVERY, timeout=1
            )
        assert not owner.pending and not owner.completed()
        assert await owner.stop(0) == ()

    async def test_timeout_retains_work_that_resists_cancellation(
        self,
    ) -> None:
        owner = SchedulerOperations()
        interrupted, release = Event(), Event()

        async def delayed() -> str:
            while not release.is_set():
                try:
                    await release.wait()
                except CancelledError:
                    interrupted.set()
            return "settled"

        with raises(SchedulerOperationTimeout) as caught:
            await owner.call(
                delayed(), kind=SchedulerOperationKind.ADMISSION, timeout=0
            )
        await interrupted.wait()
        assert owner.pending == (caught.value.operation,)
        assert owner.completed() == ()
        assert await owner.stop(0) == ()
        assert len(owner.pending) == 1
        release.set()
        await caught.value.operation.task
        completed = owner.completed()
        assert len(completed) == 1 and completed[0].value == "settled"
        assert completed[0].error is None
        assert not owner.pending and not owner.completed()

    async def test_caller_cancellation_leaves_owned_completion(self) -> None:
        owner = SchedulerOperations()
        entered, release = Event(), Event()

        async def pending() -> None:
            entered.set()
            await release.wait()

        caller = create_task(
            owner.call(
                pending(), kind=SchedulerOperationKind.PREPARATION, timeout=10
            )
        )
        await entered.wait()
        caller.cancel()
        with raises(CancelledError):
            await caller
        completions = await owner.stop(1)
        assert len(completions) == 1
        assert isinstance(completions[0].error, CancelledError)
        assert owner.pending == ()
