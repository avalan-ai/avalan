from .preparation_e2e_test import file_task
from .preparation_fault_test import configuration, services

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch
from uuid import uuid4

from pytest import raises

from avalan.task.artifacts.ownership_memory import MemoryArtifactUnit
from avalan.task.definition import IdempotencyMode
from avalan.task.idempotency import (
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
)
from avalan.task.store import TaskStoreConflictError
from avalan.task.stores.memory import InMemoryTaskStore
from avalan.task.submission import TaskSubmissionOutcome, TaskSubmissionRequest
from avalan.task.submission_participant import commit_task_submission
from avalan.trigger.stores.memory_admission import MemoryTriggerAdmissionStore


class SharedSubmissionTest(IsolatedAsyncioTestCase):
    async def test_manual_submission_acquires_independent_run_reference(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            prepared = await registered.task_input._client.prepare_submission(
                registered.task_input.definition,
                request=TaskSubmissionRequest(
                    input_value=configuration().input.value
                ),
                prepared_input=registered.task_input,
            )
            with raises(TaskStoreConflictError):
                admission.participant.validate_submission_store(
                    InMemoryTaskStore()
                )
            assert (
                await admission.participant.reconcile_submission(prepared)
            ).outcome == TaskSubmissionOutcome.NOT_COMMITTED
            before = dict(admission.ownership._objects)
            with patch.object(
                MemoryArtifactUnit,
                "attach_submitted_run",
                side_effect=RuntimeError("fault"),
            ):
                with raises(RuntimeError):
                    await commit_task_submission(
                        admission.participant, prepared
                    )
            assert not admission.participant.store._runs
            assert not admission.participant._items
            assert not admission.participant._ledger
            assert not admission.ownership._runs
            assert admission.ownership._objects == before
            result = await commit_task_submission(
                admission.participant, prepared
            )
            assert result.created
            ref = prepared.artifacts[0].ref
            assert ref.artifact_id != registered.objects[0].ref.artifact_id
            assert (
                admission.ownership._runs[ref.artifact_id]
                == registered.objects[0].object_id
            )
            repeated = await commit_task_submission(
                admission.participant, prepared
            )
            assert repeated.run.run_id == result.run.run_id
            assert len(admission.ownership._runs) == 1
            assert (
                await admission.participant.reconcile_submission(prepared)
            ).outcome == TaskSubmissionOutcome.COMMITTED
            changed = replace(prepared, run_id=str(uuid4()))
            with raises(TaskStoreConflictError):
                await admission.participant.reconcile_submission(changed)
            with raises(TaskStoreConflictError):
                await commit_task_submission(admission.participant, changed)
            occupied = replace(prepared, submission_id=str(uuid4()))
            with raises(TaskStoreConflictError):
                await commit_task_submission(admission.participant, occupied)

    async def test_idempotency_duplicate_retains_original_run_artifacts(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            admission = service.admission
            assert isinstance(admission, MemoryTriggerAdmissionStore)
            prepared = await registered.task_input._client.prepare_submission(
                registered.task_input.definition,
                request=TaskSubmissionRequest(
                    input_value=configuration().input.value
                ),
                prepared_input=registered.task_input,
            )
            digest = TaskIdempotencyDigest(
                algorithm="hmac-sha256", digest="a" * 64, key_id="test-key"
            )
            identity = TaskIdempotencyIdentity(
                identity_key="identity",
                task_name="file",
                task_version="1",
                spec_hash=prepared.execution.definition_id,
                owner_scope=digest,
                strategy=IdempotencyMode.INPUT_HASH,
                input=digest,
            )
            prepared = replace(
                prepared,
                idempotency=identity,
                execution=replace(
                    prepared.execution, idempotency_key="identity"
                ),
            )
            first = await commit_task_submission(
                admission.participant, prepared
            )
            duplicate = replace(
                prepared, submission_id=str(uuid4()), run_id=str(uuid4())
            )
            second = await commit_task_submission(
                admission.participant, duplicate
            )
            assert not second.created and second.run.run_id == first.run.run_id
            assert second.artifacts == first.artifacts and second.artifacts
            assert (
                second.idempotency is not None
                and not second.idempotency.created
            )
            assert (
                await admission.participant.reconcile_submission(duplicate)
            ).artifacts == first.artifacts

    async def test_public_manual_submit_uses_memory_commit_participant(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            result = await service.client.submit(
                file_task(),
                request=TaskSubmissionRequest(
                    input_value=configuration().input.value
                ),
            )
            assert result.outcome == TaskSubmissionOutcome.COMMITTED
            assert (
                result.committed_write().created
                and result.committed_write().artifacts
            )
