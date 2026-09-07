from .pgsql_store_test import Database
from .preparation_e2e_test import ContextCipher, file_task, target
from .records_test import NOW, OWNER

from asyncio import CancelledError, Event, create_task, run
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from pytest import raises
from task.artifacts.object_store_test import FakeObjectClient
from task_deployment_helpers import configure_fixture_deployment

from avalan.task.artifacts.object_store import (
    ObjectArtifactStore,
    ObjectArtifactStorePolicy,
)
from avalan.task.artifacts.ownership_memory import MemoryArtifactOwnership
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.client import TaskClient
from avalan.task.definition import TaskInputContract
from avalan.task.deployment import ExecutionDeploymentError
from avalan.task.input import TaskFileDescriptor, TaskProviderReferenceKind
from avalan.task.queues.memory_submission import (
    MemoryTaskSubmissionParticipant,
)
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.stores.memory import InMemoryTaskStore
from avalan.task.validation import TaskValidationCategory, TaskValidationIssue
from avalan.trigger.admission import TriggerCommitOutcome
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import (
    BindingSource,
    InputBinding,
    IntervalTrigger,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.plan import plan_admission
from avalan.trigger.preparation import (
    TriggerPreparationCancelledError,
    TriggerPreparationFailure,
    TriggerPreparationService,
)
from avalan.trigger.records import OwnerScopeId
from avalan.trigger.stores.memory import InMemoryTriggerStore
from avalan.trigger.stores.memory_admission import MemoryTriggerAdmissionStore
from avalan.trigger.stores.pgsql import PgsqlTriggerStore
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


async def services(root: Path) -> TriggerPreparationService:
    tasks = InMemoryTaskStore(clock=lambda: NOW)
    queue = MemoryTaskSubmissionParticipant(tasks)
    cipher = ContextCipher()
    backend = ObjectArtifactStore(
        FakeObjectClient(),
        cipher=cipher,
        policy=ObjectArtifactStorePolicy(
            raw_storage_allowed=True, retention_days=1, max_bytes=4096
        ),
    )
    client = TaskClient(
        tasks,
        target=target,
        queue=queue,
        owner_scope=OWNER.value,
        execution_deployment_id="deployment",
        encryption_provider=cipher,
        raw_storage_allowed=True,
        artifact_store=backend,
        execution_roots=(root,),
        clock=lambda: NOW,
    )
    await configure_fixture_deployment(client, file_task(), root)
    return TriggerPreparationService(
        client,
        MemoryTriggerAdmissionStore(
            InMemoryTriggerStore(OWNER, clock=lambda: NOW),
            queue,
            tasks._artifact_ownership,
        ),
        tasks._artifact_ownership,
        cipher,
    )


def configuration() -> TriggerConfiguration:
    return TriggerConfiguration(
        name="daily",
        task_ref="task.toml",
        schedule=IntervalTrigger(every_seconds=60, start_at=NOW),
        input=TriggerInput(
            value={
                "source_kind": "local_path",
                "reference": "input.txt",
                "mime_type": "text/plain",
            }
        ),
    )


class TriggerPreparationFaultTest(IsolatedAsyncioTestCase):
    async def test_admission_rejects_foreign_owner_before_preparation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            prepared = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                prepared, expected_generation=None
            )
            assert result.snapshot is not None
            foreign = replace(
                result.snapshot,
                definition=replace(
                    result.snapshot.definition,
                    owner_scope_id=OwnerScopeId(value="another-owner"),
                ),
                state=replace(
                    result.snapshot.state,
                    owner_scope_id=OwnerScopeId(value="another-owner"),
                ),
            )
            plan = plan_admission(foreign, NOW)
            self.assertTrue(plan.admission_ids)
            ownership = service.ownership
            assert isinstance(ownership, MemoryArtifactOwnership)
            before = (
                dict(ownership._objects),
                dict(ownership._staging),
                set(ownership._revisions),
                dict(ownership._runs),
            )
            with (
                patch.object(
                    service.client, "_deployment_client", AsyncMock()
                ) as deployment,
                patch("avalan.trigger.preparation.unseal_input") as decrypt,
                patch(
                    "avalan.trigger.preparation.restore_task_input",
                    AsyncMock(),
                ) as restore,
                patch.object(service, "_stage", AsyncMock()) as stage,
            ):
                with self.assertRaises(TriggerError) as caught:
                    await service.prepare_admission(plan)
            self.assertEqual(
                caught.exception.code, TriggerErrorCode.DEPLOYMENT_MISMATCH
            )
            self.assertEqual(caught.exception.path, "deployment")
            deployment.assert_not_awaited()
            decrypt.assert_not_called()
            restore.assert_not_awaited()
            stage.assert_not_awaited()
            self.assertEqual(
                before,
                (
                    ownership._objects,
                    ownership._staging,
                    ownership._revisions,
                    ownership._runs,
                ),
            )

    async def test_auxiliary_file_template_is_checked_before_writes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            definition = replace(file_task(), input=TaskInputContract.object())
            (root / "agent.toml").write_text(
                '[agent]\nuser_template="agent.md"\n[tool]\nenable=[]'
            )
            await configure_fixture_deployment(
                service.client, definition, root
            )
            config = replace(
                configuration(),
                input=TriggerInput(
                    value={"attachment": configuration().input.value}
                ),
            )
            with self.assertRaises(TriggerError) as missing:
                await service.prepare_registration(config, definition)
            self.assertEqual(
                missing.exception.code, TriggerErrorCode.DEPLOYMENT_MISMATCH
            )
            assert isinstance(service.ownership, MemoryArtifactOwnership)
            self.assertFalse(service.ownership._objects)

    async def test_restored_files_recheck_reachable_closure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            prepared = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                prepared, expected_generation=None
            )
            assert result.snapshot is not None
            client = prepared.task_input._client
            with patch.object(
                TaskClient,
                "_deployment_client",
                AsyncMock(
                    side_effect=[client, ExecutionDeploymentError("closure")]
                ),
            ):
                with self.assertRaises(TriggerError) as changed:
                    await service.prepare_admission(
                        plan_admission(result.snapshot, NOW)
                    )
            self.assertEqual(
                changed.exception.code, TriggerErrorCode.DEPLOYMENT_MISMATCH
            )

    async def test_deployment_mismatch_preserves_committed_apply_recovery(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            with self.assertRaises(TriggerError) as mismatch:
                await service.prepare_registration(
                    replace(configuration(), task_ref="other.toml"),
                    file_task(),
                )
            self.assertEqual(
                mismatch.exception.code, TriggerErrorCode.DEPLOYMENT_MISMATCH
            )
            prepared = await service.prepare_registration(
                configuration(), file_task()
            )
            registration = TriggerRegistrationService(service)
            committed = await registration.apply(
                prepared, expected_generation=None
            )
            (root / "agent.toml").write_text('[agent]\ninstructions="changed"')
            recovered = await registration.apply(
                prepared, expected_generation=None
            )
            self.assertEqual(recovered.outcome, TriggerCommitOutcome.COMMITTED)
            self.assertEqual(recovered.snapshot, committed.snapshot)

    async def test_schema_failure_preserves_classification_before_admission(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            prepared = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                prepared, expected_generation=None
            )
            assert result.snapshot is not None
            issue = TaskValidationIssue(
                code="input.invalid",
                path="input",
                message="Invalid input.",
                hint="Provide valid input.",
                category=TaskValidationCategory.VALUE,
            )
            with patch.object(
                service.client._target,
                "validate_definition",
                AsyncMock(return_value=(issue,)),
            ):
                with raises(TriggerPreparationFailure) as caught:
                    await service.prepare_admission(
                        plan_admission(result.snapshot, NOW)
                    )
            failure = caught.value
            assert failure.code == TriggerErrorCode.INVALID_CONFIG
            assert (
                failure.admission_outcome == TriggerCommitOutcome.NOT_COMMITTED
            )
            assert failure.validation_issues == (issue,)
            assert not failure.resources_uncertain and not failure.submissions
            assert isinstance(service.admission, MemoryTriggerAdmissionStore)
            assert not service.admission.participant.store._runs

    async def test_cancelled_external_write_retains_attempted_key(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            backend = service.client._artifact_store
            assert isinstance(backend, ObjectArtifactStore)
            entered, release = Event(), Event()

            async def pending_write(*args: object, **kwargs: object) -> None:
                entered.set()
                await release.wait()

            with patch.object(
                backend, "put_stream", side_effect=pending_write
            ):
                running = create_task(
                    service.prepare_registration(configuration(), file_task())
                )
                await entered.wait()
                running.cancel()
                with raises(TriggerPreparationCancelledError) as caught:
                    await running
            failure = caught.value.preparation
            assert (
                failure.admission_outcome == TriggerCommitOutcome.NOT_COMMITTED
            )
            assert failure.resources_uncertain and len(failure.objects) == 1
            assert failure.staging is not None
            assert isinstance(service.admission, MemoryTriggerAdmissionStore)
            assert not service.admission.participant.store._runs
            async with service.admission.ownership.transaction() as unit:
                assert unit.claim_delete(failure.objects[0].object_id) is None
                assert unit.staging

    async def test_sealing_failure_retains_materialized_bytes(self) -> None:
        for error in (ValueError("encryption failed"), CancelledError()):
            with self.subTest(error=type(error).__name__):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "input.txt").write_text("input")
                    service = await services(root)
                    with patch(
                        "avalan.trigger.preparation.seal_input",
                        side_effect=error,
                    ):
                        try:
                            await service.prepare_registration(
                                configuration(), file_task()
                            )
                        except TriggerPreparationCancelledError as caught:
                            failure = caught.preparation
                        except TriggerPreparationFailure as caught_failure:
                            failure = caught_failure
                        else:
                            self.fail("sealing failure was not propagated")
                    assert failure.staging is not None
                    assert len(failure.objects) == 1
                    assert failure.resources_uncertain
                    assert (
                        failure.admission_outcome
                        == TriggerCommitOutcome.NOT_COMMITTED
                    )
                    assert isinstance(
                        service.admission, MemoryTriggerAdmissionStore
                    )
                    assert not service.admission.participant.store._runs
                    async with (
                        service.admission.ownership.transaction() as unit
                    ):
                        assert (
                            unit.claim_delete(failure.objects[0].object_id)
                            is None
                        )
                        assert unit.staging
                    backend = service.client._artifact_store
                    assert backend is not None
                    with await backend.open(failure.objects[0].ref) as stream:
                        assert stream.read() == b"input"

    async def test_context_tampering_and_deployment_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            prepared = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                prepared, expected_generation=None
            )
            assert result.snapshot is not None
            for definition, expected in (
                (
                    replace(
                        result.snapshot.definition,
                        execution_deployment_id="other",
                    ),
                    TriggerErrorCode.DEPLOYMENT_MISMATCH,
                ),
                (
                    replace(
                        result.snapshot.definition, semantic_hash="f" * 64
                    ),
                    TriggerErrorCode.INVALID_CONFIG,
                ),
                (
                    replace(
                        result.snapshot.definition,
                        input=replace(
                            result.snapshot.definition.input,
                            ciphertext=b"tampered",
                        ),
                    ),
                    TriggerErrorCode.INVALID_CONFIG,
                ),
            ):
                with raises(TriggerError) as caught:
                    await service.prepare_admission(
                        plan_admission(
                            replace(result.snapshot, definition=definition),
                            NOW,
                        )
                    )
                assert caught.value.code == expected

    async def test_capability_durability_and_binding_failures_are_explicit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            for attribute, value, code in (
                (
                    "_execution_deployment_id",
                    None,
                    TriggerErrorCode.DEPLOYMENT_MISMATCH,
                ),
                (
                    "_raw_storage_allowed",
                    False,
                    TriggerErrorCode.CAPABILITY_UNAVAILABLE,
                ),
                (
                    "_artifact_store",
                    None,
                    TriggerErrorCode.ARTIFACT_NOT_DURABLE,
                ),
            ):
                with patch.object(service.client, attribute, value):
                    with raises(TriggerError) as caught:
                        await service.prepare_registration(
                            configuration(), file_task()
                        )
                assert caught.value.code == code
            backend = service.client._artifact_store
            assert isinstance(backend, ObjectArtifactStore)
            with (
                patch.object(backend, "_cipher", None),
                patch.object(
                    backend,
                    "_policy",
                    replace(backend._policy, encryption_required=False),
                ),
            ):
                with raises(TriggerError) as caught:
                    await service.prepare_registration(
                        configuration(), file_task()
                    )
                assert (
                    caught.value.code == TriggerErrorCode.ARTIFACT_NOT_DURABLE
                )
            for durable in (False, True):
                descriptor = TaskFileDescriptor.provider_reference_descriptor(
                    "file-id",
                    provider="provider",
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    durable=durable,
                )
                if durable:
                    service._durable_input(file_task(), descriptor)
                else:
                    with raises(TriggerError):
                        service._durable_input(file_task(), descriptor)
            bound = replace(
                configuration(),
                input=replace(
                    configuration().input,
                    bindings=(
                        InputBinding(
                            path="/reference", source=BindingSource.TRIGGER_ID
                        ),
                    ),
                ),
            )
            with raises(TriggerError) as caught:
                await service.prepare_registration(bound, file_task())
            assert caught.value.code == TriggerErrorCode.INVALID_BINDING
            with patch.object(service.client, "_owner_scope", "other"):
                with raises(TriggerError):
                    TriggerPreparationService(
                        service.client,
                        service.admission,
                        service.ownership,
                        service.decryption,
                    )
            with raises(TriggerError):
                TriggerPreparationService(
                    service.client,
                    service.admission,
                    MemoryArtifactOwnership(),
                    service.decryption,
                )

    async def test_external_and_final_staging_failures_preserve_allocations(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            backend = service.client._artifact_store
            assert isinstance(backend, ObjectArtifactStore)
            with patch.object(
                backend,
                "put_stream",
                AsyncMock(side_effect=OSError("write failed")),
            ):
                with raises(TriggerPreparationFailure) as failed:
                    await service.prepare_registration(
                        configuration(), file_task()
                    )
            assert failed.value.objects and failed.value.staging is not None
            for error in (OSError("stage failed"), CancelledError()):
                with patch.object(
                    service, "_stage", AsyncMock(side_effect=error)
                ):
                    try:
                        await service.prepare_registration(
                            configuration(), file_task()
                        )
                    except TriggerPreparationCancelledError as cancelled:
                        assert cancelled.preparation.registration is not None
                    except TriggerPreparationFailure as failure:
                        assert failure.registration is not None
                    else:
                        self.fail("final staging failure was not propagated")
            registered = await service.prepare_registration(
                configuration(), file_task()
            )
            result = await TriggerRegistrationService(service).apply(
                registered, expected_generation=None
            )
            assert result.snapshot is not None
            plan = plan_admission(result.snapshot, NOW)
            with patch.object(
                service, "_stage", AsyncMock(side_effect=CancelledError())
            ):
                with raises(
                    TriggerPreparationCancelledError
                ) as caught_admission:
                    await service.prepare_admission(plan)
            assert len(caught_admission.value.preparation.submissions) == 1
            altered = replace(
                file_task(),
                input=TaskInputContract.file(),
                task=replace(file_task().task, version="2"),
            )
            with patch.object(
                service.client,
                "_resolve_definition_schemas",
                AsyncMock(return_value=altered),
            ):
                with raises(TriggerPreparationFailure) as mismatch:
                    await service.prepare_admission(plan)
            assert mismatch.value.code == TriggerErrorCode.DEPLOYMENT_MISMATCH
            scalar = replace(
                configuration(), input=TriggerInput(value="scalar")
            )
            scalar_task = replace(
                file_task(), input=TaskInputContract.string()
            )
            await configure_fixture_deployment(
                service.client, scalar_task, Path(directory)
            )
            with patch.object(service.client, "_artifact_store", None):
                with patch.object(
                    service.client,
                    "_resolve_definition_schemas",
                    AsyncMock(side_effect=OSError("schema unavailable")),
                ):
                    with raises(OSError):
                        await service.prepare_registration(scalar, scalar_task)


def test_pgsql_preparation_rejects_unpaired_task_queue() -> None:
    with TemporaryDirectory() as directory:
        service = run(services(Path(directory)))
        database = Database()
        admission = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(database, OWNER), PgsqlTaskQueue(database)
        )
        with raises(TriggerError) as error:
            TriggerPreparationService(
                service.client,
                admission,
                PgsqlArtifactOwnership(database),
                service.decryption,
            )
        assert error.value.code == TriggerErrorCode.STORE_INCOMPATIBLE
