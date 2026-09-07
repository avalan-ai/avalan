from .preparation_fault_test import services

from asyncio import CancelledError, Event, create_task, get_running_loop, run
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event as ThreadEvent
from unittest.mock import patch

from pytest import raises

from avalan.trigger.error import TriggerError
from avalan.trigger.observability import (
    ObservedTriggerScheduler,
    TriggerMetrics,
    _owned_thread_write,
)
from avalan.trigger.scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerSettings,
)


class CloseProbe:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class BarrierSink:
    def __init__(self, *, resistant: bool) -> None:
        self.started = Event()
        self.release = Event()
        self.resistant = resistant
        self.writes = 0
        self.calls = 0

    async def emit_tick(self, result: TriggerProcessResult) -> None:
        self.calls += 1
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except CancelledError:
                if not self.resistant:
                    raise
        self.writes += 1


def test_serve_stop_settles_sink_before_resources() -> None:
    async def scenario() -> None:
        for resistant in (False, True):
            with TemporaryDirectory() as directory:
                preparation = await services(Path(directory))
                resource = CloseProbe()
                scheduler = ObservedTriggerScheduler(
                    preparation,
                    settings=TriggerSchedulerSettings(
                        shutdown_timeout_seconds=1
                    ),
                    owned_resources=(resource,),
                )
                sink = BarrierSink(resistant=resistant)
                scheduler.sink = sink
                stop = Event()
                serving = create_task(scheduler.serve(stop=stop))
                await sink.started.wait()
                stop.set()
                result = await serving
                if resistant:
                    assert not result.settled and result.pending_operations
                    assert not resource.closed and sink.writes == 0
                    sink.release.set()
                    result = await scheduler.shutdown()
                    assert sink.writes == 1
                else:
                    assert sink.writes == 0
                assert result.settled and resource.closed
                assert scheduler.sink_failures == (0 if resistant else 1)

    run(scenario())


def test_stalled_sink_keeps_tick_finite_and_concurrency_guard() -> None:
    async def scenario() -> None:
        with TemporaryDirectory() as directory:
            scheduler = ObservedTriggerScheduler(
                await services(Path(directory)),
                settings=TriggerSchedulerSettings(
                    tick_timeout_seconds=1, shutdown_timeout_seconds=1
                ),
            )
            sink = BarrierSink(resistant=True)
            scheduler.sink = sink
            active = create_task(scheduler.process_once())
            await sink.started.wait()
            with raises(TriggerError, match="scheduler.process_once"):
                await scheduler.process_once()
            tick = await active
            assert tick.pending_operations == 1 and tick.remaining_work
            again = await scheduler.process_once()
            assert again.pending_operations == 1 and sink.calls == 1
            unresolved = await scheduler.shutdown()
            assert not unresolved.settled
            sink.release.set()
            assert (await scheduler.shutdown()).settled

    run(scenario())


def test_actual_thread_write_remains_owned_after_shutdown_timeout() -> None:
    async def scenario() -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            resource = CloseProbe()
            scheduler = ObservedTriggerScheduler(
                await services(root),
                settings=TriggerSchedulerSettings(shutdown_timeout_seconds=1),
                owned_resources=(resource,),
            )
            path = root / "metrics.prom"
            scheduler.sink = TriggerMetrics(path)
            started = Event()
            finished = Event()
            release = ThreadEvent()
            loop = get_running_loop()
            original = Path.write_text

            def blocked_write(
                target: Path,
                data: str,
                encoding: str | None = None,
                errors: str | None = None,
                newline: str | None = None,
            ) -> int:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(10), "test writer was not released"
                result = original(
                    target,
                    data,
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                )
                loop.call_soon_threadsafe(finished.set)
                return result

            stop = Event()
            try:
                with patch.object(Path, "write_text", blocked_write):
                    serving = create_task(scheduler.serve(stop=stop))
                    await started.wait()
                    stop.set()
                    result = await serving
                    assert not result.settled and result.pending_operations
                    assert not path.exists() and not resource.closed
                    release.set()
                    await finished.wait()
                    assert (await scheduler.shutdown()).settled
                assert resource.closed
                assert "avalan_trigger_" in path.read_text()
            finally:
                release.set()

    run(scenario())


def test_owned_write_preserves_cancellation_and_terminal_failures() -> None:
    async def scenario() -> None:
        def aborted_write() -> None:
            raise CancelledError()

        with raises(CancelledError):
            await _owned_thread_write(aborted_write)
        for fails in (False, True):
            started = Event()
            release = ThreadEvent()
            loop = get_running_loop()

            def write() -> None:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(10), "test writer was not released"
                if fails:
                    raise OSError("private")

            active = create_task(_owned_thread_write(write))
            try:
                await started.wait()
                active.cancel()
                # Give the owned wait a cancellation delivery before release.
                delivered = Event()
                loop.call_soon(delivered.set)
                await delivered.wait()
                active.cancel()
                release.set()
                with raises(CancelledError) as failure:
                    await active
                if fails:
                    assert isinstance(failure.value.__cause__, OSError)
            finally:
                release.set()

    run(scenario())
