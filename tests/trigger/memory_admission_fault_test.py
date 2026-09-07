from .preparation_e2e_test import file_task
from .preparation_fault_test import configuration, services
from .records_test import NOW

from asyncio import CancelledError
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from pytest import raises

from avalan.task.artifacts.ownership_memory import MemoryArtifactUnit
from avalan.task.queues.memory_submission import MemoryTaskSubmissionUnit
from avalan.task.submission import PreparedTaskSubmission, TaskSubmissionWrite
from avalan.task.submission_participant import commit_task_submission
from avalan.trigger.admission import (
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.plan import plan_admission
from avalan.trigger.records import (
    OccurrenceDisposition,
    OwnerScopeId,
    TriggerSnapshot,
)
from avalan.trigger.registration import TriggerMutation
from avalan.trigger.stores.memory import InMemoryTriggerStore
from avalan.trigger.stores.memory_admission import MemoryTriggerAdmissionStore


class MemoryAdmissionFaultTest(IsolatedAsyncioTestCase):
    async def test_private_trigger_write_failure_rolls_back_all_participants(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            registration = TriggerRegistrationService(service)
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            original = InMemoryTriggerStore._write

            def failing_write(
                store: InMemoryTriggerStore, mutation: TriggerMutation
            ) -> TriggerSnapshot:
                original(store, mutation)
                raise RuntimeError("after private trigger write")

            with patch.object(InMemoryTriggerStore, "_write", failing_write):
                failed = await registration.apply(
                    registered, expected_generation=None
                )
            assert failed.outcome == TriggerCommitOutcome.NOT_COMMITTED
            assert (
                not admission.store._current
                and not admission.store._definitions
            )
            assert (
                not admission.ownership._revisions
                and admission.ownership._staging
            )
            result = await registration.apply(
                registered, expected_generation=None
            )
            assert result.snapshot is not None
            prepared = await service.prepare_admission(
                plan_admission(result.snapshot, NOW)
            )
            pending = dict(admission.ownership._staging)
            with patch.object(InMemoryTriggerStore, "_write", failing_write):
                with raises(RuntimeError):
                    await admission.admit(prepared)
            assert await admission.store.inspect("daily") == result.snapshot
            assert not admission.participant.store._runs
            assert (
                not admission.participant._items
                and not admission.participant._ledger
            )
            assert (
                not admission.store._occurrences
                and not admission.ownership._runs
            )
            assert admission.ownership._staging == pending
            committed = await admission.admit(prepared)
            assert committed.outcome == TriggerCommitOutcome.COMMITTED
            assert (
                await admission.admit(prepared)
            ).outcome == TriggerCommitOutcome.COMMITTED
            assert (
                await service.release_unused_admission(prepared)
            ).outcome == TriggerCommitOutcome.COMMITTED

    async def test_cancellation_recovery_and_cleanup_keep_stable_handles(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                registered, expected_generation=None
            )
            assert result.snapshot is not None
            prepared = await service.prepare_admission(
                plan_admission(result.snapshot, NOW)
            )
            with patch.object(
                admission, "_persist", AsyncMock(side_effect=CancelledError())
            ):
                with raises(TriggerAdmissionCancelledError) as caught:
                    await admission.admit(prepared)
                assert caught.value.result.prepared is prepared
                assert (
                    caught.value.result.outcome
                    == TriggerCommitOutcome.NOT_COMMITTED
                )
                with patch.object(
                    admission,
                    "recover",
                    AsyncMock(side_effect=CancelledError()),
                ):
                    with raises(TriggerAdmissionCancelledError) as repeated:
                        await admission.admit(prepared)
                assert (
                    repeated.value.result.outcome
                    == TriggerCommitOutcome.UNKNOWN
                )
                assert repeated.value.result.prepared is prepared
            with patch.object(
                admission,
                "recover",
                AsyncMock(
                    return_value=TriggerAdmissionResult(
                        plan=prepared.plan,
                        outcome=TriggerCommitOutcome.UNKNOWN,
                    )
                ),
            ):
                assert (
                    await service.release_unused_admission(prepared)
                ).outcome == TriggerCommitOutcome.UNKNOWN
            assert admission.ownership._staging
            with patch.object(
                MemoryArtifactUnit,
                "release_staging",
                side_effect=OSError("cleanup"),
            ):
                assert (
                    await service.release_unused_admission(prepared)
                ).cleanup_pending
            with patch.object(
                MemoryArtifactUnit,
                "release_staging",
                side_effect=CancelledError(),
            ):
                with raises(TriggerAdmissionCancelledError) as cleanup:
                    await service.release_unused_admission(prepared)
            assert (
                cleanup.value.result.cleanup_pending
                and cleanup.value.result.prepared is prepared
            )
            assert (
                await service.release_unused_admission(prepared)
            ).outcome == TriggerCommitOutcome.NOT_COMMITTED
            assert not admission.ownership._staging

    async def test_existing_task_run_is_not_attached_to_a_new_occurrence(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                registered, expected_generation=None
            )
            assert result.snapshot is not None
            prepared = await service.prepare_admission(
                plan_admission(result.snapshot, NOW)
            )
            direct = await commit_task_submission(
                admission.participant, prepared.submissions[0]
            )
            assert direct.created
            with raises(TriggerError) as caught:
                await admission.admit(prepared)
            assert caught.value.code == TriggerErrorCode.CONFLICT
            assert await admission.store.inspect("daily") == result.snapshot
            assert not admission.store._occurrences

    async def test_misfire_overlap_and_paused_plan_preserve_cursor(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                registered, expected_generation=None
            )
            assert result.snapshot is not None
            now = NOW + timedelta(seconds=120)
            with patch.object(admission.store, "_clock", return_value=now):
                prepared = await service.prepare_admission(
                    plan_admission(result.snapshot, now)
                )
                committed = await admission.admit(prepared)
            assert committed.snapshot is not None
            assert len(admission.store._spans["daily"]) == 1
            assert len(admission.participant.store._runs) == 1
            now += timedelta(seconds=60)
            with patch.object(admission.store, "_clock", return_value=now):
                overlapping = await service.prepare_admission(
                    plan_admission(committed.snapshot, now)
                )
                overlap = await admission.admit(overlapping)
            assert overlap.outcome == TriggerCommitOutcome.COMMITTED
            assert overlap.snapshot is not None
            assert len(admission.participant.store._runs) == 1
            assert (
                admission.store._occurrences["daily"][-1].disposition
                == OccurrenceDisposition.SKIPPED_OVERLAP
            )
            assert (
                await service.release_unused_admission(overlapping)
            ).outcome == TriggerCommitOutcome.COMMITTED
            now += timedelta(seconds=60)
            future = await service.prepare_admission(
                plan_admission(overlap.snapshot, now)
            )
            await admission.store.set_enabled(
                "daily",
                enabled=False,
                expected_generation=overlap.snapshot.state.generation,
            )
            assert (
                await admission.admit(future)
            ).outcome == TriggerCommitOutcome.NOT_COMMITTED
            other_owner = OwnerScopeId(value="other")
            other = replace(
                overlap.snapshot,
                definition=replace(
                    overlap.snapshot.definition, owner_scope_id=other_owner
                ),
                state=replace(
                    overlap.snapshot.state, owner_scope_id=other_owner
                ),
            )
            with raises(TriggerError):
                await admission.recover(plan_admission(other, now))

    async def test_participant_cannot_associate_a_duplicate_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            registration = TriggerRegistrationService(service)
            prepared = await service.prepare_registration(
                configuration(), file_task()
            )
            applied = await registration.apply(
                prepared, expected_generation=None
            )
            assert applied.snapshot is not None
            request = await service.prepare_admission(
                plan_admission(applied.snapshot, NOW)
            )
            original = admission.participant.submit_prepared

            async def duplicate(
                value: PreparedTaskSubmission,
                *,
                unit_of_work: MemoryTaskSubmissionUnit,
            ) -> TaskSubmissionWrite:
                return replace(
                    await original(value, unit_of_work=unit_of_work),
                    created=False,
                )

            with patch.object(
                admission.participant, "submit_prepared", duplicate
            ):
                with raises(TriggerError) as error:
                    await admission.admit(request)
            assert error.value.code == TriggerErrorCode.CONFLICT
            assert not admission.participant.store._runs
            assert await admission.store.inspect("daily") == applied.snapshot
