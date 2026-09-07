"""Exercise late owned work and cancellation through concrete services."""

from .preparation_e2e_test import file_task
from .preparation_fault_test import configuration, services
from .records_test import NOW

from asyncio import CancelledError, Event, create_task, wait
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.plan import TriggerAdmissionPlan
from avalan.trigger.scheduler import (
    TriggerScheduler,
    TriggerSchedulerCancelledError,
)
from avalan.trigger.scheduler_types import (
    TriggerSchedulerSettings,
    TriggerTickStop,
)


@asynccontextmanager
async def scheduling() -> AsyncIterator[TriggerScheduler]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "input.txt").write_text("input")
        preparation = await services(root)
        registered = await preparation.prepare_registration(
            configuration(), file_task()
        )
        await TriggerRegistrationService(preparation).apply(
            registered, expected_generation=None
        )
        scheduler = TriggerScheduler(
            preparation,
            settings=TriggerSchedulerSettings(
                tick_timeout_seconds=1,
                preparation_timeout_seconds=1,
                transaction_timeout_seconds=1,
            ),
        )
        try:
            yield scheduler
        finally:
            await scheduler.shutdown()


class TriggerSchedulerFaultTest(IsolatedAsyncioTestCase):
    async def test_late_preparation_is_owned_and_settled_before_retry(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            blocked = Event()
            release = Event()
            original = scheduler.preparation.prepare_admission

            async def delayed(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                try:
                    await blocked.wait()
                except CancelledError:
                    await release.wait()
                return await original(plan)

            with patch.object(
                scheduler.preparation, "prepare_admission", delayed
            ):
                bounded = await scheduler.process_once()
            assert bounded.stop == TriggerTickStop.TIME_LIMIT
            assert bounded.pending_operations == 1 and bounded.admitted == 0
            again = await scheduler.process_once()
            assert again.stop == TriggerTickStop.TIME_LIMIT
            assert again.pending_operations == 1 and again.admitted == 0
            release.set()
            await scheduler._operations.pending[0].task
            recovered = await scheduler.process_once()
            assert recovered.admitted == 1 and not recovered.unresolved
            assert not recovered.pending_operations
            assert len((await scheduler.store.occurrences("daily")).items) == 1

    async def test_late_committed_ack_is_observed_once_after_timeout(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            blocked = Event()
            release = Event()
            original = scheduler.admission.admit

            async def delayed(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                result = await original(prepared)
                try:
                    await blocked.wait()
                except CancelledError:
                    await release.wait()
                return result

            with patch.object(scheduler.admission, "admit", delayed):
                bounded = await scheduler.process_once()
            assert bounded.pending_operations == 1 and bounded.admitted == 0
            release.set()
            await scheduler._operations.pending[0].task
            recovered = await scheduler.process_once()
            assert recovered.admitted == 1 and not recovered.unresolved
            assert (await scheduler.process_once()).admitted == 0
            assert len((await scheduler.store.occurrences("daily")).items) == 1

    async def test_serve_cancellation_settles_prepared_admission(self) -> None:
        async with scheduling() as scheduler:
            entered = Event()
            blocked = Event()

            async def waiting(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                entered.set()
                await blocked.wait()
                raise AssertionError("admission should be cancelled")

            with patch.object(scheduler.admission, "admit", waiting):
                serving = create_task(scheduler.serve())
                await entered.wait()
                serving.cancel()
                with self.assertRaises(
                    TriggerSchedulerCancelledError
                ) as caught:
                    await serving
            assert caught.exception.result.settled
            assert not scheduler._operations.pending
            assert not (await scheduler.store.occurrences("daily")).items

    async def test_cancelled_preparation_timeout_consumes_durable_retry(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            blocked = Event()

            async def waiting(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                await blocked.wait()
                raise AssertionError("preparation should time out")

            with patch.object(
                scheduler.preparation, "prepare_admission", waiting
            ):
                bounded = await scheduler.process_once()
            assert bounded.pending_operations == 1
            with self.assertRaises(CancelledError):
                await scheduler._operations.pending[0].task
            completed = await scheduler.process_once()
            assert completed.admitted == 0 and len(completed.errors) == 1
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.failure_count == 1
            assert current.state.retry_after is not None
            restarted = TriggerScheduler(scheduler.preparation)
            assert (await restarted.process_once()).admitted == 0
            assert (await restarted.shutdown()).settled

    async def test_stale_time_replans_after_external_preparation(self) -> None:
        async with scheduling() as scheduler:
            original = scheduler.preparation.prepare_admission
            clock = [NOW]
            calls = []

            async def advance(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                prepared = await original(plan)
                calls.append(plan)
                if len(calls) == 1:
                    clock[0] += timedelta(minutes=1)
                return prepared

            with (
                patch.object(scheduler.store, "_clock", lambda: clock[0]),
                patch.object(
                    scheduler.preparation, "prepare_admission", advance
                ),
            ):
                result = await scheduler.process_once()
            assert len(calls) == 2 and result.admitted == 1
            current = await scheduler.store.inspect("daily")
            assert (
                current is not None
                and current.state.next_at == NOW + timedelta(minutes=2)
            )
            assert current.state.failure_count == 0
            assert not result.unresolved

    async def test_replan_budget_preserves_first_undecided_slot(self) -> None:
        async with scheduling() as scheduler:
            original = scheduler.preparation.prepare_admission
            clock = [NOW]
            calls = []

            async def advance(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                prepared = await original(plan)
                calls.append(plan)
                clock[0] += timedelta(minutes=1)
                return prepared

            with (
                patch.object(scheduler.store, "_clock", lambda: clock[0]),
                patch.object(
                    scheduler.preparation, "prepare_admission", advance
                ),
            ):
                result = await scheduler.process_once()
            assert len(calls) == scheduler.settings.stale_plan_retries + 1
            assert result.admitted == 0 and result.remaining_work
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.next_at == NOW
            assert current.state.failure_count == 0
            assert not result.unresolved

    async def test_cleanup_timeout_preserves_acknowledged_commit(self) -> None:
        async with scheduling() as scheduler:
            blocked = Event()

            async def waiting(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                await blocked.wait()
                raise AssertionError("cleanup should time out")

            with patch.object(
                scheduler.preparation, "release_unused_admission", waiting
            ):
                result = await scheduler.process_once()
            assert result.admitted == 1 and result.pending_operations == 1
            assert len(result.unresolved) == 1
            assert (
                result.unresolved[0].outcome == TriggerCommitOutcome.COMMITTED
            )
            assert result.unresolved[0].prepared is not None
            assert (await scheduler.shutdown()).settled

    async def test_cancelled_admission_timeout_consumes_retry(self) -> None:
        async with scheduling() as scheduler:
            blocked = Event()

            async def waiting(
                candidate: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                await blocked.wait()
                raise AssertionError("admission should time out")

            with patch.object(scheduler.admission, "admit", waiting):
                result = await scheduler.process_once()
            assert result.pending_operations == 1
            await wait((scheduler._operations.pending[0].task,))
            completed = await scheduler.process_once()
            assert completed.admitted == 0 and len(completed.errors) == 1
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.failure_count == 1
            assert current.state.next_at == NOW
