"""Test owned completion recovery using real prepared task resources."""

from .records_test import NOW
from .scheduler_fault_test import scheduling

from asyncio import CancelledError, Event
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from pytest import raises

from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from avalan.trigger.plan import TriggerAdmissionPlan, plan_admission
from avalan.trigger.preparation import (
    TriggerPreparationCancelledError,
    TriggerPreparationFailure,
)
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_operations import (
    SchedulerOperationKind,
    SchedulerOperationTimeout,
)


async def completion(
    scheduler: TriggerScheduler,
    prepared: PreparedTriggerAdmission,
    *,
    kind: SchedulerOperationKind,
    error: BaseException,
) -> None:
    blocked = Event()

    async def work() -> None:
        try:
            await blocked.wait()
        except CancelledError:
            raise error

    with raises(SchedulerOperationTimeout):
        await scheduler._operations.call(
            work(),
            kind=kind,
            timeout=0,
            snapshot=prepared.plan.snapshot,
            prepared=(
                prepared if kind == SchedulerOperationKind.ADMISSION else None
            ),
            plan=(
                prepared.plan
                if kind == SchedulerOperationKind.RECOVERY
                else None
            ),
        )
    with raises(type(error)):
        await scheduler._operations.pending[0].task


async def prepared(scheduler: TriggerScheduler) -> PreparedTriggerAdmission:
    current = await scheduler.store.inspect("daily")
    assert current is not None
    return await scheduler.preparation.prepare_admission(
        plan_admission(current, NOW)
    )


class TriggerSchedulerCompletionTest(IsolatedAsyncioTestCase):
    async def test_failed_or_cancelled_recovery_keeps_identity(self) -> None:
        for failure in (OSError("transport"), CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                async with scheduling() as scheduler:
                    candidate = await prepared(scheduler)
                    await completion(
                        scheduler,
                        candidate,
                        kind=SchedulerOperationKind.RECOVERY,
                        error=failure,
                    )

                    async def unknown(
                        plan: TriggerAdmissionPlan,
                    ) -> TriggerAdmissionResult:
                        return TriggerAdmissionResult(
                            plan=plan, outcome=TriggerCommitOutcome.UNKNOWN
                        )

                    with patch.object(scheduler.admission, "recover", unknown):
                        result = await scheduler.process_once()
                    assert result.admitted == 0 and len(result.unresolved) == 1
                    assert result.unresolved[0].plan == candidate.plan
                    assert not result.errors
                    await scheduler.preparation.release_unused_admission(
                        candidate
                    )

    async def test_late_admission_failure_reconciles_before_retry(
        self,
    ) -> None:
        for committed in (True, False):
            with self.subTest(committed=committed):
                async with scheduling() as scheduler:
                    candidate = await prepared(scheduler)
                    if committed:
                        await scheduler.admission.admit(candidate)
                    await completion(
                        scheduler,
                        candidate,
                        kind=SchedulerOperationKind.ADMISSION,
                        error=OSError("ack unavailable"),
                    )
                    result = await scheduler.process_once()
                    assert result.admitted == (1 if committed else 0)
                    current = await scheduler.store.inspect("daily")
                    assert current is not None
                    assert current.state.failure_count == (
                        0 if committed else 1
                    )
                    assert not result.unresolved

    async def test_typed_cancelled_admission_preserves_known_commit(
        self,
    ) -> None:
        for committed in (True, False):
            with self.subTest(committed=committed):
                async with scheduling() as scheduler:
                    candidate = await prepared(scheduler)
                    known = (
                        await scheduler.admission.admit(candidate)
                        if committed
                        else TriggerAdmissionResult(
                            plan=candidate.plan,
                            outcome=TriggerCommitOutcome.NOT_COMMITTED,
                            prepared=candidate,
                        )
                    )
                    await completion(
                        scheduler,
                        candidate,
                        kind=SchedulerOperationKind.ADMISSION,
                        error=TriggerAdmissionCancelledError(known),
                    )
                    result = await scheduler.process_once()
                    assert result.admitted == (1 if committed else 0)
                    assert len(result.errors) == (0 if committed else 1)
                    assert not result.unresolved

    async def test_cancelled_preparation_exposes_resources_and_retry(
        self,
    ) -> None:
        for shutdown in (True, False):
            with self.subTest(shutdown=shutdown):
                async with scheduling() as scheduler:
                    candidate = await prepared(scheduler)
                    failure = TriggerPreparationFailure(
                        submissions=candidate.submissions
                    )
                    await completion(
                        scheduler,
                        candidate,
                        kind=SchedulerOperationKind.PREPARATION,
                        error=TriggerPreparationCancelledError(failure),
                    )
                    result = (
                        await scheduler.shutdown()
                        if shutdown
                        else await scheduler.process_once()
                    )
                    assert result.preparation_failures == (failure,)
                    current = await scheduler.store.inspect("daily")
                    assert (
                        current is not None
                        and current.state.failure_count
                        == (0 if shutdown else 1)
                    )
                    assert not (
                        await scheduler.store.occurrences("daily")
                    ).items
                    await scheduler.preparation.release_unused_admission(
                        candidate
                    )

    async def test_non_cancellation_termination_is_not_swallowed(self) -> None:
        async with scheduling() as scheduler:
            candidate = await prepared(scheduler)
            await completion(
                scheduler,
                candidate,
                kind=SchedulerOperationKind.PREPARATION,
                error=BaseException("terminate"),
            )
            with raises(BaseException, match="terminate"):
                await scheduler.process_once()
            assert not scheduler._processing
            await scheduler.preparation.release_unused_admission(candidate)

    async def test_immediate_typed_cancellation_remains_owned_for_shutdown(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            admit = scheduler.admission.admit

            async def cancelled(
                candidate: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                result = await admit(candidate)
                raise TriggerAdmissionCancelledError(result)

            with patch.object(scheduler.admission, "admit", cancelled):
                with raises(TriggerAdmissionCancelledError) as caught:
                    await scheduler.process_once()
            assert (
                caught.value.result.outcome == TriggerCommitOutcome.COMMITTED
            )
            assert len(scheduler._operations.pending) == 1
            assert (await scheduler.shutdown()).settled
            assert not scheduler._operations.pending
            assert len((await scheduler.store.occurrences("daily")).items) == 1

    async def test_typed_recovery_cancellation_retains_unprepared_identity(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            candidate = await prepared(scheduler)
            unknown = TriggerAdmissionResult(
                plan=candidate.plan, outcome=TriggerCommitOutcome.UNKNOWN
            )
            await completion(
                scheduler,
                candidate,
                kind=SchedulerOperationKind.RECOVERY,
                error=TriggerAdmissionCancelledError(unknown),
            )

            async def recover(
                plan: TriggerAdmissionPlan,
            ) -> TriggerAdmissionResult:
                return unknown

            with patch.object(scheduler.admission, "recover", recover):
                result = await scheduler.process_once()
            assert result.unresolved == (unknown,)
            await scheduler.preparation.release_unused_admission(candidate)

    async def test_late_failed_admission_with_unknown_cleanup_stays_unresolved(
        self,
    ) -> None:
        async with scheduling() as scheduler:
            candidate = await prepared(scheduler)
            await completion(
                scheduler,
                candidate,
                kind=SchedulerOperationKind.ADMISSION,
                error=OSError("late transport failure"),
            )

            async def release(
                value: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                raise OSError("cleanup unavailable")

            with patch.object(
                scheduler.preparation, "release_unused_admission", release
            ):
                result = await scheduler.process_once()
            assert len(result.unresolved) == 1 and not result.errors
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.failure_count == 0
