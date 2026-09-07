"""Exercise the scheduler through actual validated memory preparation."""

from .preparation_e2e_test import file_task
from .preparation_fault_test import configuration, services
from .records_test import NOW

from asyncio import Event, create_task, sleep
from dataclasses import replace
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
from avalan.trigger.definition import (
    MisfirePolicy,
    OverlapPolicy,
    RecurringPolicy,
)
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.plan import TriggerAdmissionPlan
from avalan.trigger.records import TriggerStatus
from avalan.trigger.scheduler import (
    TriggerScheduler,
    TriggerSchedulerCancelledError,
)
from avalan.trigger.scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerSettings,
    TriggerTickStop,
)


class TriggerSchedulerTest(IsolatedAsyncioTestCase):
    async def test_validated_preparation_and_atomic_admission(
        self,
    ) -> None:
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
            scheduler = TriggerScheduler(preparation)
            result = await scheduler.process_once()
            assert result.admitted == 1 and result.skipped == 0
            assert not result.errors and not result.unresolved
            assert not result.remaining_work
            assert (await scheduler.process_once()).admitted == 0
            assert (await scheduler.shutdown()).settled

    async def test_all_backlog_keeps_undecided_cursor_at_trigger_limit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            config = replace(
                configuration(),
                policy=RecurringPolicy(
                    misfire=MisfirePolicy.ALL, overlap=OverlapPolicy.ALLOW
                ),
            )
            registered = await preparation.prepare_registration(
                config, file_task()
            )
            await TriggerRegistrationService(preparation).apply(
                registered, expected_generation=None
            )
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(
                    decisions_per_trigger=2, admissions_per_trigger=2
                ),
            )
            now = NOW + timedelta(minutes=4)
            with patch.object(scheduler.store, "_clock", lambda: now):
                first = await scheduler.process_once()
                assert first.admitted == 2 and first.remaining_work
                current = await scheduler.store.inspect("daily")
                assert (
                    current is not None
                    and current.state.next_at == NOW + timedelta(minutes=2)
                )
                second = await scheduler.process_once()
                assert second.admitted == 2 and second.remaining_work
                third = await scheduler.process_once()
                assert third.admitted == 1 and not third.remaining_work
            assert (await scheduler.shutdown()).settled

    async def test_retry_budget_survives_scheduler_recreation_and_polling(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            registration = TriggerRegistrationService(preparation)
            for name in ("bad", "good"):
                prepared = await preparation.prepare_registration(
                    replace(configuration(), name=name), file_task()
                )
                await registration.apply(prepared, expected_generation=None)
            original = preparation.prepare_admission
            now = NOW

            async def failure(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                if plan.snapshot.definition.name == "bad":
                    raise OSError("private error must not be copied")
                return await original(plan)

            settings = TriggerSchedulerSettings(admission_retry_attempts=2)
            scheduler = TriggerScheduler(preparation, settings=settings)
            with (
                patch.object(preparation, "prepare_admission", failure),
                patch.object(scheduler.store, "_clock", lambda: now),
            ):
                first = await scheduler.process_once()
                assert first.admitted == 1 and len(first.errors) == 1
                assert first.admission_retries == 0
                assert (
                    first.errors[0].code
                    == TriggerErrorCode.ADMISSION_RETRYABLE
                )
                assert "private" not in first.errors[0].detail
                bad = await scheduler.store.inspect("bad")
                assert bad is not None and bad.state.failure_count == 1
                assert (
                    bad.state.next_at == NOW
                    and bad.state.retry_after == NOW + timedelta(seconds=1)
                )
                restarted = TriggerScheduler(preparation, settings=settings)
                assert not (await restarted.process_once()).errors
                assert await scheduler.store.inspect("bad") == bad
                now += timedelta(seconds=1)
                second = await restarted.process_once()
                assert len(second.errors) == 1
                assert second.admission_retries == 1
                failed = await scheduler.store.inspect("bad")
                assert (
                    failed is not None
                    and failed.state.status == TriggerStatus.ERROR
                )
                assert (
                    failed.state.failure_count == 2
                    and failed.state.next_at == NOW
                )
                assert not (await restarted.process_once()).errors

    async def test_serve_stop_closes_only_explicitly_owned_resources(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            preparation = await services(Path(directory))
            stop = Event()
            closed = []

            class Owned:
                async def aclose(self) -> None:
                    closed.append("owned")

            scheduler = TriggerScheduler(
                preparation, owned_resources=(Owned(),)
            )
            original = scheduler.process_once
            observed = Event()

            async def tick() -> TriggerProcessResult:
                result = await original()
                observed.set()
                return result

            with patch.object(scheduler, "process_once", tick):
                running = create_task(scheduler.serve(stop=stop))
                await observed.wait()
                stop.set()
                result = await running
            assert result.settled and closed == ["owned"]
            assert (await scheduler.shutdown()).settled
            assert closed == ["owned"]
            assert not scheduler._operations.pending

    async def test_repeated_shutdown_cancellation_preserves_owned_close(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            entered = Event()
            release = Event()
            closed: list[str] = []

            class Owned:
                async def aclose(self) -> None:
                    entered.set()
                    await release.wait()
                    closed.append("closed")

            scheduler = TriggerScheduler(
                await services(Path(directory)), owned_resources=(Owned(),)
            )
            first = create_task(scheduler.shutdown())
            await entered.wait()
            first.cancel()
            with self.assertRaises(TriggerSchedulerCancelledError) as caught:
                await first
            assert not caught.exception.result.settled
            second = create_task(scheduler.shutdown())
            # Join the same owned operation before cancellation.
            await sleep(0)
            second.cancel()
            with self.assertRaises(TriggerSchedulerCancelledError):
                await second
            release.set()
            assert (await scheduler.shutdown()).settled
            assert closed == ["closed"]

    async def test_lost_ack_recovers_commit_without_recording_retry(
        self,
    ) -> None:
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
            scheduler = TriggerScheduler(preparation)
            admit = scheduler.admission.admit

            async def lost_ack(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                await admit(prepared)
                raise OSError("untrusted transport details")

            with patch.object(scheduler.admission, "admit", lost_ack):
                result = await scheduler.process_once()
            assert result.admitted == 1 and not result.errors
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.failure_count == 0
            assert (await scheduler.process_once()).admitted == 0
            assert (await scheduler.shutdown()).settled

    async def test_unknown_recovery_retains_identity_without_retry(
        self,
    ) -> None:
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
            scheduler = TriggerScheduler(preparation)

            async def unknown(
                plan: TriggerAdmissionPlan,
            ) -> TriggerAdmissionResult:
                return TriggerAdmissionResult(
                    plan=plan,
                    outcome=TriggerCommitOutcome.UNKNOWN,
                    error_code=TriggerErrorCode.COMMIT_UNKNOWN,
                )

            with patch.object(scheduler.admission, "recover", unknown):
                first = await scheduler.process_once()
                second = await scheduler.process_once()
            assert len(first.unresolved) == len(second.unresolved) == 1
            assert first.unresolved[0].plan == second.unresolved[0].plan
            assert first.admitted == second.admitted == 0
            current = await scheduler.store.inspect("daily")
            assert current is not None and current.state.failure_count == 0
            assert current.state.next_at == NOW
            resolved = await scheduler.process_once()
            assert resolved.admitted == 1 and not resolved.unresolved
            assert (await scheduler.shutdown()).settled

    async def test_skip_and_latest_compress_backlog_then_advance(self) -> None:
        for policy, admitted in (
            (MisfirePolicy.SKIP, 0),
            (MisfirePolicy.LATEST, 1),
        ):
            with (
                self.subTest(policy=policy),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "input.txt").write_text("input")
                preparation = await services(root)
                registered = await preparation.prepare_registration(
                    replace(
                        configuration(), policy=RecurringPolicy(misfire=policy)
                    ),
                    file_task(),
                )
                await TriggerRegistrationService(preparation).apply(
                    registered, expected_generation=None
                )
                scheduler = TriggerScheduler(preparation)
                with patch.object(
                    scheduler.store,
                    "_clock",
                    lambda: NOW + timedelta(minutes=30),
                ):
                    result = await scheduler.process_once()
                    assert (
                        result.admitted == admitted and len(result.ranges) == 1
                    )
                    assert not result.errors and not result.remaining_work
                    assert (await scheduler.process_once()).admitted == 0
                current = await scheduler.store.inspect("daily")
                assert current is not None
                assert current.state.next_at == NOW + timedelta(minutes=31)

    async def test_permanent_preparation_error_is_safe_and_stops_trigger(
        self,
    ) -> None:
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
            scheduler = TriggerScheduler(preparation)

            async def invalid(
                plan: TriggerAdmissionPlan,
            ) -> PreparedTriggerAdmission:
                raise TriggerError(
                    TriggerErrorCode.INVALID_CONFIG, "private/file/input"
                )

            with patch.object(preparation, "prepare_admission", invalid):
                result = await scheduler.process_once()
            assert len(result.errors) == 1
            assert "private" not in result.errors[0].detail
            current = await scheduler.store.inspect("daily")
            assert (
                current is not None
                and current.state.status == TriggerStatus.ERROR
            )
            assert current.state.next_at == NOW
            assert not (await scheduler.process_once()).errors
            assert (await scheduler.shutdown()).settled
            assert (
                await scheduler.process_once()
            ).stop == TriggerTickStop.STOP_REQUESTED

    async def test_cleanup_failure_does_not_erase_acknowledged_commit(
        self,
    ) -> None:
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
            scheduler = TriggerScheduler(preparation)

            async def unavailable(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                raise OSError("cleanup connection unavailable")

            with patch.object(
                preparation, "release_unused_admission", unavailable
            ):
                result = await scheduler.process_once()
            assert result.admitted == 1 and not result.errors
            assert len(result.unresolved) == 1
            assert (
                result.unresolved[0].outcome == TriggerCommitOutcome.COMMITTED
            )
            assert result.unresolved[0].cleanup_pending
            assert (await scheduler.shutdown()).settled

    async def test_global_budget_defers_second_due_trigger_without_cursor_loss(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            for name in ("first", "second"):
                prepared = await preparation.prepare_registration(
                    replace(configuration(), name=name), file_task()
                )
                await TriggerRegistrationService(preparation).apply(
                    prepared, expected_generation=None
                )
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(admissions_per_tick=1),
            )
            first = await scheduler.process_once()
            assert first.admitted == 1 and first.remaining_work
            assert first.stop == TriggerTickStop.WORK_LIMIT
            second = await scheduler.process_once()
            assert second.admitted == 1 and not second.remaining_work
            assert (
                first.occurrences[0].trigger_id
                != second.occurrences[0].trigger_id
            )
            assert (await scheduler.process_once()).admitted == 0
            assert (await scheduler.shutdown()).settled

    async def test_global_candidate_limit_preserves_later_healthy_trigger(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            for name in ("first", "second"):
                prepared = await preparation.prepare_registration(
                    replace(configuration(), name=name), file_task()
                )
                await TriggerRegistrationService(preparation).apply(
                    prepared, expected_generation=None
                )
            # One interval admission charges five evaluations across planning
            # and locked validation. Six leaves only one for the next trigger.
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(candidate_evaluations=6),
            )
            first = await scheduler.process_once()
            assert (
                first.admitted == 1
                and first.stop == TriggerTickStop.WORK_LIMIT
            )
            assert first.remaining_work and not first.errors
            for name in ("first", "second"):
                current = await scheduler.store.inspect(name)
                assert (
                    current is not None
                    and current.state.status == TriggerStatus.ACTIVE
                )
                assert current.state.failure_count == 0
            second = await scheduler.process_once()
            assert second.admitted == 1 and not second.errors
            assert (await scheduler.shutdown()).settled

    async def test_discovery_one_rotates_past_unknown_identity(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            for name in ("first", "second"):
                prepared = await preparation.prepare_registration(
                    replace(configuration(), name=name), file_task()
                )
                await TriggerRegistrationService(preparation).apply(
                    prepared, expected_generation=None
                )
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(discovery_limit=1),
            )
            blocked = (
                await scheduler.store.discover(decision_time=NOW, limit=1)
            )[0]
            recover = scheduler.admission.recover

            async def selective(
                plan: TriggerAdmissionPlan,
            ) -> TriggerAdmissionResult:
                if plan.snapshot.definition.name == blocked.definition.name:
                    return TriggerAdmissionResult(
                        plan=plan, outcome=TriggerCommitOutcome.UNKNOWN
                    )
                return await recover(plan)

            with patch.object(scheduler.admission, "recover", selective):
                first = await scheduler.process_once()
                second = await scheduler.process_once()
            assert first.admitted == 0 and len(first.unresolved) == 1
            assert second.admitted == 1 and len(second.unresolved) == 1
            current = await scheduler.store.inspect(blocked.definition.name)
            assert current == blocked
            assert (await scheduler.shutdown()).settled

    async def test_candidate_limit_rolls_back_locked_recheck(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            prepared = await preparation.prepare_registration(
                configuration(), file_task()
            )
            applied = await TriggerRegistrationService(preparation).apply(
                prepared, expected_generation=None
            )
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(candidate_evaluations=3),
            )
            bounded = await scheduler.process_once()
            assert bounded.stop == TriggerTickStop.WORK_LIMIT
            assert bounded.admitted == 0 and not bounded.errors
            assert await scheduler.store.inspect("daily") == applied.snapshot
            assert not (await scheduler.store.occurrences("daily")).items
            restarted = TriggerScheduler(preparation)
            assert (await restarted.process_once()).admitted == 1
            assert (await scheduler.shutdown()).settled
            assert (await restarted.shutdown()).settled
