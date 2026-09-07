"""Prove shutdown owns invocation scopes, not arbitrary caller tasks."""

from .preparation_fault_test import services
from .records_test import NOW

from asyncio import CancelledError, Event, create_task, gather
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from pytest import raises

from avalan.trigger.error import TriggerError
from avalan.trigger.scheduler import (
    TriggerScheduler,
    TriggerSchedulerCancelledError,
)
from avalan.trigger.scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerSettings,
)


class TriggerSchedulerShutdownTest(IsolatedAsyncioTestCase):
    async def test_shutdown_joins_invocation_without_joining_caller(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            entered = Event()
            blocked = Event()
            unrelated = Event()
            caller_continued = Event()
            closed = []

            class Owned:
                async def aclose(self) -> None:
                    closed.append(True)

            scheduler = TriggerScheduler(
                services(Path(directory)), owned_resources=(Owned(),)
            )

            async def clock() -> datetime:
                entered.set()
                await blocked.wait()
                return NOW

            async def caller() -> None:
                try:
                    await scheduler.process_once()
                except CancelledError:
                    caller_continued.set()
                await unrelated.wait()

            with patch.object(scheduler, "_clock", clock):
                running = create_task(caller())
                await entered.wait()
                with raises(TriggerError, match="scheduler.process_once"):
                    await scheduler.process_once()
                stopped = await scheduler.shutdown()
                await caller_continued.wait()
                assert stopped.settled and closed == [True]
                assert not running.done()
                unrelated.set()
                await running
            assert not scheduler._operations.pending
            assert not scheduler._processing

    async def test_resistant_invocation_defers_owned_resource_close(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            entered, blocked, release = Event(), Event(), Event()
            closed = []

            class Owned:
                async def aclose(self) -> None:
                    closed.append(True)

            scheduler = TriggerScheduler(
                services(Path(directory)),
                settings=TriggerSchedulerSettings(shutdown_timeout_seconds=1),
                owned_resources=(Owned(),),
            )
            original = scheduler._process_once

            async def delayed() -> TriggerProcessResult:
                entered.set()
                try:
                    await blocked.wait()
                except CancelledError:
                    await release.wait()
                return await original()

            with patch.object(scheduler, "_process_once", delayed):
                caller = create_task(scheduler.process_once())
                await entered.wait()
                bounded = await scheduler.shutdown()
                assert bounded.pending_operations == 1 and not bounded.settled
                assert not closed
                release.set()
                await caller
            assert (await scheduler.shutdown()).settled
            assert closed == [True]

    async def test_late_close_is_not_repeated_after_timeout(self) -> None:
        with TemporaryDirectory() as directory:
            blocked, release = Event(), Event()
            closed = []

            class Owned:
                async def aclose(self) -> None:
                    try:
                        await blocked.wait()
                    except CancelledError:
                        await release.wait()
                    closed.append(True)

            scheduler = TriggerScheduler(
                services(Path(directory)),
                settings=TriggerSchedulerSettings(shutdown_timeout_seconds=1),
                owned_resources=(Owned(),),
            )
            bounded = await scheduler.shutdown()
            assert bounded.pending_operations == 1 and not bounded.settled
            release.set()
            await scheduler._operations.pending[0].task
            assert (await scheduler.shutdown()).settled
            assert closed == [True]

    async def test_resource_close_failure_is_visible_and_retryable(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            calls = []

            class Owned:
                async def aclose(self) -> None:
                    calls.append(True)
                    if len(calls) == 1:
                        raise OSError("sensitive backend details")

            scheduler = TriggerScheduler(
                services(Path(directory)), owned_resources=(Owned(),)
            )
            result = await scheduler.shutdown()
            assert not result.settled and len(result.errors) == 1
            assert "sensitive" not in result.errors[0].detail
            assert (await scheduler.shutdown()).settled
            assert len(calls) == 2

    async def test_late_close_failure_is_reported_then_retried(self) -> None:
        with TemporaryDirectory() as directory:
            release, blocked = Event(), Event()
            calls = []

            class Owned:
                async def aclose(self) -> None:
                    calls.append(True)
                    if len(calls) == 1:
                        try:
                            await blocked.wait()
                        except CancelledError:
                            await release.wait()
                        raise OSError("private close failure")

            scheduler = TriggerScheduler(
                services(Path(directory)),
                settings=TriggerSchedulerSettings(shutdown_timeout_seconds=1),
                owned_resources=(Owned(),),
            )
            assert (await scheduler.shutdown()).pending_operations == 1
            release.set()
            with raises(OSError):
                await scheduler._operations.pending[0].task
            result = await scheduler.shutdown()
            assert (
                len(result.errors) == 1
                and "private" not in result.errors[0].detail
            )
            assert len(calls) == 2
            assert (await scheduler.shutdown()).settled

    async def test_serve_waits_after_tick_and_rejects_reentry(self) -> None:
        with TemporaryDirectory() as directory:
            scheduler = TriggerScheduler(services(Path(directory)))
            waiting, release, stop = Event(), Event(), Event()
            original = scheduler._wait_seconds

            async def delay(result: TriggerProcessResult) -> float:
                waiting.set()
                await release.wait()
                return await original(result)

            with patch.object(scheduler, "_wait_seconds", delay):
                running = create_task(scheduler.serve(stop=stop))
                await waiting.wait()
                with raises(TriggerError, match="scheduler.serve"):
                    await scheduler.serve()
                stop.set()
                release.set()
                assert (await running).settled
            with raises(TriggerError, match="scheduler.serve"):
                await scheduler.serve()

    async def test_serve_propagates_fatal_tick_after_owned_close(self) -> None:
        with TemporaryDirectory() as directory:
            closed = []

            class Owned:
                async def aclose(self) -> None:
                    closed.append(True)

            scheduler = TriggerScheduler(
                services(Path(directory)), owned_resources=(Owned(),)
            )

            async def failed() -> TriggerProcessResult:
                raise BaseException("terminate tick")

            with patch.object(scheduler, "_process_once", failed):
                with raises(BaseException, match="terminate tick"):
                    await scheduler.serve()
            assert closed == [True] and not scheduler._serving

    async def test_shutdown_reports_unobserved_invocation_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            scheduler = TriggerScheduler(services(Path(directory)))
            entered, blocked, release = Event(), Event(), Event()

            async def failed_after_cancel() -> TriggerProcessResult:
                entered.set()
                try:
                    await blocked.wait()
                except CancelledError:
                    await release.wait()
                raise OSError("private invocation failure")

            with patch.object(scheduler, "_process_once", failed_after_cancel):
                caller = create_task(scheduler.process_once())
                await entered.wait()
                caller.cancel()
                with raises(CancelledError):
                    await caller
                active = scheduler._active_tick
                assert active is not None
                release.set()
                with raises(OSError):
                    await active
                result = await scheduler.shutdown()
            assert len(result.errors) == 1
            assert "private" not in result.errors[0].detail

    async def test_external_shutdown_stops_polling_without_joining_caller(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            polling, unrelated, continued = Event(), Event(), Event()
            closed = []
            reads = []

            class Owned:
                async def aclose(self) -> None:
                    closed.append(True)

            scheduler = TriggerScheduler(
                services(Path(directory)), owned_resources=(Owned(),)
            )

            async def delay(result: TriggerProcessResult) -> float:
                assert not closed
                reads.append(True)
                polling.set()
                return 60

            async def caller() -> None:
                assert (await scheduler.serve()).settled
                continued.set()
                await unrelated.wait()

            with patch.object(scheduler, "_wait_seconds", delay):
                running = create_task(caller())
                await polling.wait()
                results = await gather(
                    scheduler.shutdown(), scheduler.shutdown()
                )
                assert all(result.settled for result in results)
                await continued.wait()
                assert (
                    not running.done() and closed == [True] and reads == [True]
                )
                assert (
                    scheduler._active_serve is not None
                    and scheduler._active_serve.done()
                )
                unrelated.set()
                await running
            assert (await scheduler.shutdown()).settled and closed == [True]

    async def test_external_shutdown_joins_or_retains_owned_wake_lookup(
        self,
    ) -> None:
        for resistant in (False, True):
            with (
                self.subTest(resistant=resistant),
                TemporaryDirectory() as directory,
            ):
                entered, blocked, release, exited = (
                    Event(),
                    Event(),
                    Event(),
                    Event(),
                )
                closed = []

                class Owned:
                    async def aclose(self) -> None:
                        assert exited.is_set()
                        closed.append(True)

                scheduler = TriggerScheduler(
                    services(Path(directory)),
                    settings=TriggerSchedulerSettings(
                        shutdown_timeout_seconds=1
                    ),
                    owned_resources=(Owned(),),
                )

                async def lookup() -> datetime | None:
                    entered.set()
                    try:
                        while not release.is_set():
                            try:
                                await blocked.wait()
                            except CancelledError:
                                if not resistant:
                                    raise
                        return None
                    finally:
                        exited.set()

                with patch.object(scheduler.store, "next_eligible_at", lookup):
                    running = create_task(scheduler.serve())
                    await entered.wait()
                    result = await scheduler.shutdown()
                    if resistant:
                        assert (
                            not result.settled
                            and result.pending_operations == 1
                        )
                        assert not closed and not exited.is_set()
                        release.set()
                        blocked.set()
                        await scheduler._operations.pending[0].task
                    else:
                        assert result.settled and closed == [True]
                    await running
                    assert (await scheduler.shutdown()).settled
                assert exited.is_set() and closed == [True]

    async def test_cancelled_owned_serve_scope_preserves_public_cancellation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            entered, blocked = Event(), Event()
            scheduler = TriggerScheduler(services(Path(directory)))

            async def lookup() -> datetime | None:
                entered.set()
                await blocked.wait()
                return None

            with patch.object(scheduler.store, "next_eligible_at", lookup):
                running = create_task(scheduler.serve())
                await entered.wait()
                assert scheduler._active_serve is not None
                scheduler._active_serve.cancel()
                with raises(TriggerSchedulerCancelledError) as caught:
                    await running
                assert caught.value.result.settled
            assert not scheduler._operations.pending
