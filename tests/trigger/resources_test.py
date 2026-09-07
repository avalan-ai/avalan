from .pgsql_store_test import Database
from .plan_test import snapshot
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
from uuid import uuid4

from pytest import raises

from avalan.task.artifact import TaskArtifactPurpose, TaskArtifactRetention
from avalan.task.artifact_ownership import ArtifactStagingOwner
from avalan.task.artifact_retention import retry_artifact_cleanup
from avalan.task.artifacts.ownership_memory import MemoryArtifactUnit
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.retention import TaskRetentionService
from avalan.task.state import TaskRunState
from avalan.task.store import TaskExecutionRequest
from avalan.task.submission import TaskSubmissionOutcome
from avalan.trigger.admission import TriggerCommitOutcome
from avalan.trigger.apply import (
    TriggerApplyCancelledError,
    TriggerApplyResult,
    TriggerRegistrationService,
)
from avalan.trigger.definition import IntervalTrigger
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.resources import TriggerResourceService
from avalan.trigger.stores.memory_admission import MemoryTriggerAdmissionStore
from avalan.trigger.stores.pgsql import PgsqlTriggerStore


class TriggerResourceTest(IsolatedAsyncioTestCase):
    async def test_closed_revision_waits_for_pending_input_and_retains_runs(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = services(root)
            admission = preparation.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            store, ownership = admission.store, admission.ownership
            resources = TriggerResourceService(store, ownership)
            apply = TriggerRegistrationService(preparation)
            first = await preparation.prepare_registration(
                configuration(), file_task()
            )
            result = await apply.apply(first, expected_generation=None)
            assert result.snapshot is not None
            assert not await resources.release_closed_revision("daily", 1)
            paused = await store.set_enabled(
                "daily",
                enabled=False,
                expected_generation=result.snapshot.state.generation,
            )
            assert not await resources.release_closed_revision("daily", 1)
            pending = ArtifactStagingOwner(
                staging_id=str(uuid4()),
                owner_scope=store.owner.value,
                recovery_id="pending-preparation",
            )
            async with ownership.transaction() as unit:
                unit.stage(first.objects[0], pending)
                unit.attach_run(
                    first.objects[0].object_id, pending, "independent-run-ref"
                )
            updated = await preparation.prepare_registration(
                replace(
                    configuration(),
                    schedule=IntervalTrigger(
                        every_seconds=120, start_at=NOW + timedelta(minutes=2)
                    ),
                ),
                file_task(),
            )
            result = await apply.apply(
                updated, expected_generation=paused.state.generation
            )
            assert result.snapshot is not None
            assert not await resources.release_closed_revision("daily", 1)
            async with ownership.transaction() as unit:
                unit.release_staging(
                    pending, outcome=TaskSubmissionOutcome.NOT_COMMITTED
                )
            assert await resources.release_closed_revision("daily", 1)
            assert await resources.release_closed_revision("daily", 1)
            assert not await resources.release_closed_revision("daily", 2)
            assert not await resources.release_closed_revision("daily", 999)
            async with ownership.transaction() as unit:
                assert (
                    unit.claim_delete(
                        first.objects[0].object_id, grace_seconds=0
                    )
                    is None
                )
                unit.release_run("independent-run-ref")
                assert (
                    unit.claim_delete(
                        first.objects[0].object_id, grace_seconds=0
                    )
                    is not None
                )

    async def test_registration_cleanup_outcomes_and_cancellation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = services(root)
            admission = preparation.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            apply = TriggerRegistrationService(preparation)
            prepared = await preparation.prepare_registration(
                configuration(), file_task()
            )
            with patch.object(
                apply,
                "recover",
                AsyncMock(
                    return_value=TriggerApplyResult(
                        prepared=prepared,
                        outcome=TriggerCommitOutcome.UNKNOWN,
                    )
                ),
            ):
                result = await apply.release_unused(
                    prepared, expected_generation=None
                )
            assert result.outcome == TriggerCommitOutcome.UNKNOWN
            assert admission.ownership._staging
            with patch.object(
                MemoryArtifactUnit,
                "release_staging",
                side_effect=OSError("cleanup failed"),
            ):
                result = await apply.release_unused(
                    prepared, expected_generation=None
                )
            assert (
                result.outcome == TriggerCommitOutcome.NOT_COMMITTED
                and result.cleanup_pending
            )
            with patch.object(
                MemoryArtifactUnit,
                "release_staging",
                side_effect=CancelledError(),
            ):
                with raises(TriggerApplyCancelledError) as caught:
                    await apply.release_unused(
                        prepared, expected_generation=None
                    )
            assert (
                caught.value.result.outcome
                == TriggerCommitOutcome.NOT_COMMITTED
            )
            assert caught.value.result.prepared is prepared
            assert caught.value.result.cleanup_pending
            assert admission.ownership._staging
            result = await apply.release_unused(
                prepared, expected_generation=None
            )
            assert (
                not result.cleanup_pending and not admission.ownership._staging
            )
            backend = preparation.client._artifact_store
            assert backend is not None
            assert (
                await retry_artifact_cleanup(
                    admission.ownership,
                    {prepared.objects[0].ref.store: backend},
                )
                == 0
            )
            assert (
                await retry_artifact_cleanup(
                    admission.ownership,
                    {prepared.objects[0].ref.store: backend},
                    orphan_grace_seconds=0,
                )
                == 1
            )

    async def test_run_retention_preserves_recent_trigger_staging_age(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = services(root)
            admission = preparation.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            prepared = await preparation.prepare_registration(
                configuration(), file_task()
            )
            apply = TriggerRegistrationService(preparation)
            await apply.release_unused(prepared, expected_generation=None)
            physical = prepared.objects[0]
            before = admission.ownership._objects[physical.object_id].staged_at
            tasks = admission.participant.store
            run = await tasks.create_run(
                TaskExecutionRequest(
                    definition_id=prepared.task_input.definition_id
                )
            )
            ref = physical.new_reference()
            await tasks.append_artifact(
                run.run_id,
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
                retention=TaskArtifactRetention(delete_after_days=1),
            )
            backend = preparation.client._artifact_store
            assert backend is not None
            await tasks.transition_run(
                run.run_id,
                from_states={TaskRunState.CREATED},
                to_state=TaskRunState.VALIDATED,
                reason="test",
            )
            await tasks.transition_run(
                run.run_id,
                from_states={TaskRunState.VALIDATED},
                to_state=TaskRunState.QUEUED,
                reason="test",
            )
            await tasks.transition_run(
                run.run_id,
                from_states={TaskRunState.QUEUED},
                to_state=TaskRunState.EXPIRED,
                reason="test",
            )
            result = await TaskRetentionService(
                tasks, {ref.store: backend}
            ).enforce_run(run.run_id, now=NOW + timedelta(days=2))
            assert len(result.results) == 1
            assert (
                admission.ownership._objects[physical.object_id].staged_at
                == before
            )
            assert not await admission.ownership.pending_deletions()
            with await backend.open(ref) as stream:
                assert stream.read() == b"input"

    async def test_registration_cancellation_and_identity_recovery(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = services(root)
            service = TriggerRegistrationService(preparation)
            prepared = await preparation.prepare_registration(
                configuration(), file_task()
            )
            other = TriggerRegistrationService(services(root))
            with raises(TriggerError):
                await other.apply(prepared, expected_generation=None)
            with raises(TriggerError):
                await other.release_unused(prepared, expected_generation=None)
            with patch.object(
                service,
                "_apply_memory",
                AsyncMock(side_effect=CancelledError()),
            ):
                with raises(TriggerApplyCancelledError) as cancelled:
                    await service.apply(prepared, expected_generation=None)
                assert (
                    cancelled.value.result.outcome
                    == TriggerCommitOutcome.NOT_COMMITTED
                )
                assert cancelled.value.result.prepared is prepared
                with patch.object(
                    service, "recover", AsyncMock(side_effect=CancelledError())
                ):
                    with raises(TriggerApplyCancelledError) as repeated:
                        await service.apply(prepared, expected_generation=None)
                assert (
                    repeated.value.result.outcome
                    == TriggerCommitOutcome.UNKNOWN
                )
            assert (
                await service.apply(prepared, expected_generation=None)
            ).outcome == TriggerCommitOutcome.COMMITTED
            assert (
                await service.recover(prepared, expected_generation=999)
            ).outcome == TriggerCommitOutcome.UNKNOWN
            conflict = await service.apply(prepared, expected_generation=999)
            assert conflict.error_code == TriggerErrorCode.CONFLICT

    async def test_corrupt_database_resource_identity_fails_closed(
        self,
    ) -> None:
        database = Database()
        current = snapshot()
        store = PgsqlTriggerStore(database, current.state.owner_scope_id)
        resources = TriggerResourceService(
            store, PgsqlArtifactOwnership(database)
        )
        with (
            patch(
                "avalan.trigger.stores.pgsql.check_trigger_schema", AsyncMock()
            ),
            patch.object(store, "_read", AsyncMock(return_value=current)),
        ):
            with (
                patch.object(
                    database.cursor,
                    "fetchone",
                    AsyncMock(return_value={"closed_at": NOW}),
                ),
                patch.object(
                    database.cursor,
                    "fetchall",
                    AsyncMock(return_value=({"object_id": 1},)),
                ),
            ):
                with raises(TriggerError) as error:
                    await resources.release_closed_revision("daily", 1)
        assert error.value.code == TriggerErrorCode.UNSUPPORTED_VERSION
