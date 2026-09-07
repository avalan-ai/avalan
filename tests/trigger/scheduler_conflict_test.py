"""Check competing decisions and failure classification per tick."""

from .preparation_fault_test import services
from .records_test import NOW
from .registration_test import registration
from .scheduler_fault_test import scheduling

from asyncio import CancelledError, Event, wait
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from pytest import raises

from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from avalan.trigger.definition import IntervalTrigger
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.plan import TriggerAdmissionPlan
from avalan.trigger.records import TriggerStatus
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_operations import (
    SchedulerOperationKind,
    SchedulerOperationTimeout,
)
from avalan.trigger.scheduler_types import TriggerSchedulerSettings


class TriggerSchedulerConflictTest(IsolatedAsyncioTestCase):
    async def test_competitor_commit_recovers_before_own_preparation(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            recover = scheduler.admission.recover
            first = True

            async def compete(
                plan: TriggerAdmissionPlan,
            ) -> TriggerAdmissionResult:
                nonlocal first
                if first:
                    first = False
                    prepared = await scheduler.preparation.prepare_admission(
                        plan
                    )
                    await scheduler.admission.admit(prepared)
                    await scheduler.preparation.release_unused_admission(
                        prepared
                    )
                return await recover(plan)

            with patch.object(scheduler.admission, "recover", compete):
                result = await scheduler.process_once()
            assert result.admitted == 1 and not result.errors
            assert len((await scheduler.store.occurrences("daily")).items) == 1

    async def test_backward_clock_cannot_admit_discovered_future_slot(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            calls = 0

            async def clock() -> datetime:
                nonlocal calls
                calls += 1
                return NOW if calls == 1 else NOW - timedelta(minutes=1)

            with patch.object(scheduler, "_clock", clock):
                result = await scheduler.process_once()
            assert result.admitted == 0 and not result.errors
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.next_at == NOW

    async def test_control_race_does_not_consume_failure_budget(self) -> None:
        for explicit_conflict in (True, False):
            with self.subTest(explicit_conflict=explicit_conflict):
                async with scheduling() as scheduler:

                    async def pause(
                        plan: TriggerAdmissionPlan,
                    ) -> PreparedTriggerAdmission:
                        await scheduler.store.set_enabled(
                            "daily",
                            enabled=False,
                            expected_generation=plan.snapshot.state.generation,
                        )
                        if explicit_conflict:
                            raise TriggerError(
                                TriggerErrorCode.CONFLICT, "control"
                            )
                        raise OSError("failed preparation after pause")

                    with patch.object(
                        scheduler.preparation, "prepare_admission", pause
                    ):
                        result = await scheduler.process_once()
                    assert result.conflicts == 1 and result.admitted == 0
                    current = await scheduler.store.inspect("daily")
                    assert (
                        current is not None
                        and current.state.status == TriggerStatus.PAUSED
                    )
                    assert (
                        current.state.failure_count == 0
                        and current.state.next_at == NOW
                    )

    async def test_admission_contention_and_failure_remain_distinct(
        self,
    ) -> None:
        for contended in (True, False):
            with self.subTest(contended=contended):
                async with scheduling() as scheduler:

                    async def reject(
                        prepared: PreparedTriggerAdmission,
                    ) -> TriggerAdmissionResult:
                        return TriggerAdmissionResult(
                            plan=prepared.plan,
                            outcome=TriggerCommitOutcome.NOT_COMMITTED,
                            contended=contended,
                            prepared=prepared,
                        )

                    with patch.object(scheduler.admission, "admit", reject):
                        result = await scheduler.process_once()
                    current = await scheduler.store.inspect("daily")
                    assert current is not None and current.state.next_at == NOW
                    assert current.state.failure_count == (
                        0 if contended else 1
                    )
                    assert result.conflicts == (1 if contended else 0)
                    assert result.admitted == 0 and not result.unresolved

    async def test_prior_completion_and_full_batch_report_all_conflicts(
        self,
    ) -> None:
        # Persist distinct management fixtures; preparation fails before any
        # task invocation, so this proves bounded scheduler accounting only.
        with TemporaryDirectory() as directory:
            preparation = await services(Path(directory))
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(
                    discovery_limit=1000, tick_timeout_seconds=60
                ),
            )
            base = replace(
                registration(),
                schedule=IntervalTrigger(every_seconds=60, start_at=NOW),
            )
            first = None
            for index in range(1000):
                current = await scheduler.store.apply(
                    replace(base, name=f"item-{index}"),
                    expected_generation=None,
                )
                if first is None:
                    first = current
            assert first is not None
            blocked = Event()

            async def late_failure() -> None:
                try:
                    await blocked.wait()
                except CancelledError:
                    raise TriggerError(
                        TriggerErrorCode.CONFLICT, "failure.cas"
                    ) from None

            with raises(SchedulerOperationTimeout):
                await scheduler._operations.call(
                    late_failure(),
                    kind=SchedulerOperationKind.FAILURE,
                    timeout=0,
                    snapshot=first,
                )
            await wait((scheduler._operations.pending[0].task,))

            async def conflict(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                raise TriggerError(
                    TriggerErrorCode.CONFLICT, "preparation.cas"
                )

            with patch.object(preparation, "prepare_admission", conflict):
                result = await scheduler.process_once()
            assert result.conflicts == 1001 and result.admitted == 0
            assert result.remaining_work and not result.errors
            assert not scheduler._operations.pending
            assert (
                await scheduler.store.inspect(first.definition.name) == first
            )
            assert (await scheduler.shutdown()).settled

    async def test_failure_record_error_remains_visible_without_cursor_change(
        self,
    ) -> None:
        async with scheduling() as scheduler:

            async def invalid(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                raise OSError("prepare failure")

            async def incompatible(*args: object, **kwargs: object) -> None:
                raise TriggerError(TriggerErrorCode.SCHEMA_MISMATCH, "schema")

            before = await scheduler.store.inspect("daily")
            with (
                patch.object(
                    scheduler.preparation, "prepare_admission", invalid
                ),
                patch.object(scheduler.store, "fail", incompatible),
            ):
                result = await scheduler.process_once()
            assert len(result.errors) == 2 and result.admitted == 0
            assert await scheduler.store.inspect("daily") == before

    async def test_uncertain_admission_never_authorizes_retry(self) -> None:
        for raises_error in (True, False):
            with self.subTest(raises_error=raises_error):
                async with scheduling() as scheduler:

                    async def uncertain(
                        candidate: PreparedTriggerAdmission,
                    ) -> TriggerAdmissionResult:
                        if raises_error:
                            raise OSError("lost write acknowledgement")
                        return TriggerAdmissionResult(
                            plan=candidate.plan,
                            outcome=TriggerCommitOutcome.UNKNOWN,
                            prepared=candidate,
                        )

                    async def unavailable(
                        candidate: PreparedTriggerAdmission,
                    ) -> TriggerAdmissionResult:
                        raise OSError("recovery unavailable")

                    with (
                        patch.object(scheduler.admission, "admit", uncertain),
                        patch.object(
                            scheduler.preparation,
                            "release_unused_admission",
                            unavailable,
                        ),
                    ):
                        result = await scheduler.process_once()
                    assert (
                        len(result.unresolved) == 1
                        and result.unresolved[0].prepared is not None
                    )
                    assert result.admitted == 0 and result.remaining_work
                    current = await scheduler.store.inspect("daily")
                    assert (
                        current is not None
                        and current.state.failure_count == 0
                    )
