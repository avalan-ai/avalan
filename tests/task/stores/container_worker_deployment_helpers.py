"""Compose a real queued attempt with the test-owned pinned command image."""

from collections.abc import AsyncIterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import loads
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)
from trigger.preparation_e2e_test import ContextCipher
from trigger.records_test import OWNER

from avalan.container import (
    ContainerAsyncBackend,
    ContainerBackendContainer,
    ContainerBackendImageResolution,
    ContainerBackendInspection,
    ContainerBackendOperationResult,
    ContainerBackendProbeResult,
    ContainerBackendStats,
    ContainerBackendStreamChunk,
    ContainerBackendWaitResult,
    ContainerMountDeclaration,
    ContainerOutputContract,
    ContainerOutputValidationResult,
    ContainerRunPlan,
    DockerContainerBackend,
)
from avalan.pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from avalan.task.artifacts.local import LocalArtifactStore
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.client import TaskClient
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.state import TaskRunState
from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)
from avalan.task.worker import TaskWorker
from avalan.trigger.admission import TriggerCommitOutcome
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import (
    IntervalTrigger,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.plan import plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.stores.pgsql import PgsqlTriggerStore
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


class FixtureCommandDockerBackend(ContainerAsyncBackend):
    """Mount the fixture command and return bounded verified copied bytes.

    The base Docker backend intentionally supplies artifact metadata only.
    This test-owned extension implements the existing copied-byte contract;
    it does not modify generic Docker behavior or execution capabilities.
    """

    def __init__(
        self,
        mounts: tuple[ContainerMountDeclaration, ...],
        output: Path,
        *,
        copied_bytes: bool = True,
    ) -> None:
        self.copied_bytes = copied_bytes
        self.backend = DockerContainerBackend()
        self.created: list[ContainerBackendContainer] = []
        self.mounts = mounts
        self.output = output
        self.plans: list[ContainerRunPlan] = []
        self.errors: list[str] = []
        self.chunks: list[ContainerBackendStreamChunk] = []

    async def create(
        self, plan: ContainerRunPlan
    ) -> ContainerBackendContainer:
        self.plans.append(plan)
        targets = {mount.target for mount in self.mounts}
        try:
            container = await self.backend.create(
                replace(
                    plan,
                    mounts=(
                        *(
                            mount
                            for mount in plan.mounts
                            if mount.target not in targets
                        ),
                        *self.mounts,
                    ),
                )
            )
        except Exception as error:
            self.errors.append(repr(error))
            raise
        self.created.append(container)
        return container

    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        result = await self.backend.copy_outputs(container, contract)
        if not self.copied_bytes:
            return result
        artifacts = []
        for artifact in result.artifacts:
            path = self.output / artifact.path
            assert path.resolve().is_relative_to(self.output.resolve())
            with path.open("rb") as stream:
                content = stream.read(contract.max_bytes + 1)
            assert len(content) == artifact.size_bytes <= contract.max_bytes
            assert "sha256:" + sha256(content).hexdigest() == artifact.digest
            artifacts.append(replace(artifact, content=content))
        return replace(result, artifacts=tuple(artifacts))

    async def probe(self) -> ContainerBackendProbeResult:
        return await self.backend.probe()

    async def resolve_image(
        self, plan: ContainerRunPlan
    ) -> ContainerBackendImageResolution:
        return await self.backend.resolve_image(plan)

    async def pull_image(
        self, plan: ContainerRunPlan, image: ContainerBackendImageResolution
    ) -> ContainerBackendOperationResult:
        return await self.backend.pull_image(plan, image)

    async def build_image(
        self, plan: ContainerRunPlan
    ) -> ContainerBackendOperationResult:
        return await self.backend.build_image(plan)

    async def stream(
        self, container: ContainerBackendContainer
    ) -> (
        Sequence[ContainerBackendStreamChunk]
        | AsyncIterable[ContainerBackendStreamChunk]
    ):
        chunks = await self.backend.stream(container)
        self.chunks.extend(chunks)
        return chunks

    async def wait(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendWaitResult:
        return await self.backend.wait(container)

    async def inspect(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendInspection:
        return await self.backend.inspect(container)

    async def stats(
        self, container: ContainerBackendContainer
    ) -> tuple[ContainerBackendStats, ...]:
        return await self.backend.stats(container)

    async def start(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendOperationResult:
        return await self.backend.start(container)

    async def attach(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendOperationResult:
        return await self.backend.attach(container)

    async def stop(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendOperationResult:
        return await self.backend.stop(container)

    async def kill(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendOperationResult:
        return await self.backend.kill(container)

    async def remove(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendOperationResult:
        return await self.backend.remove(container)

    async def cleanup(
        self, container: ContainerBackendContainer
    ) -> ContainerBackendOperationResult:
        return await self.backend.cleanup(container)


async def exercise_scheduled_container(
    test: IsolatedAsyncioTestCase,
    binding: ExecutionDeploymentBinding,
    catalog: ExecutionDeploymentCatalog,
    mounts: tuple[ContainerMountDeclaration, ...],
    output: Path,
    *,
    rootful_authorized: bool = True,
    copied_bytes: bool = True,
) -> None:
    dsn = real_task_pgsql_dsn()
    if not dsn:
        test.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
    schema = isolated_task_pgsql_schema("deployment_container")
    task_pgsql_upgrade(PgsqlTaskMigrationSettings(url=dsn, schema=schema))
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=task_pgsql_psycopg_dsn(dsn), schema=schema)
    )
    await database.open()
    try:
        tasks = PgsqlTaskStore(database)
        queue = PgsqlTaskQueue(database)
        store = PgsqlTriggerStore(database, OWNER)
        admission = PgsqlTriggerAdmissionStore(store, queue)
        cipher = ContextCipher()
        artifacts = LocalArtifactStore(
            output.parent / "artifacts", raw_storage_allowed=True
        )
        client = TaskClient(
            tasks,
            target=binding.target,
            queue=queue,
            owner_scope=OWNER.value,
            execution_deployments=catalog,
            execution_deployment_id=binding.manifest.execution_deployment_id,
            encryption_provider=cipher,
            raw_storage_allowed=True,
            artifact_store=artifacts,
        )
        preparation = TriggerPreparationService(
            client, admission, PgsqlArtifactOwnership(database), cipher
        )
        now = datetime.now(UTC)
        prepared = await preparation.prepare_registration(
            TriggerConfiguration(
                name="container",
                task_ref="task.toml",
                schedule=IntervalTrigger(every_seconds=86400),
                input=TriggerInput(value="frozen private application input"),
            ),
            binding.definition,
        )
        # Controlled historical registration supplies a due slot; actual
        # admission retains the PostgreSQL authoritative decision clock.
        with patch(
            "avalan.trigger.apply.decision_time",
            new=AsyncMock(return_value=now - timedelta(days=1, minutes=5)),
        ):
            registered = await TriggerRegistrationService(preparation).apply(
                prepared, expected_generation=None
            )
        assert registered.snapshot is not None
        decision = await admission.admit(
            await preparation.prepare_admission(
                plan_admission(registered.snapshot, datetime.now(UTC))
            )
        )
        test.assertEqual(
            decision.outcome, TriggerCommitOutcome.COMMITTED, decision
        )
        backend = FixtureCommandDockerBackend(
            mounts, output, copied_bytes=copied_bytes
        )
        worker = TaskWorker(
            tasks,
            queue,
            target=binding.target,
            execution_deployments=catalog,
            queue_name="container",
            encryption_provider=cipher,
            raw_storage_allowed=True,
            artifact_store=artifacts,
            container_backend=backend,
            container_rootful_authorized=rootful_authorized,
        )
        result = await worker.process_once()
        assert result.claimed is not None
        run = await tasks.get_run(result.claimed.run.run_id)
        if not rootful_authorized or not copied_bytes:
            test.assertEqual(run.state, TaskRunState.FAILED)
            assert run.result is not None
            test.assertIn(
                (
                    "container.worker_capability_mismatch"
                    if not rootful_authorized
                    else "container.output_unsupported"
                ),
                str(run.result.error),
            )
            test.assertEqual(
                len(backend.plans), 0 if not rootful_authorized else 1
            )
            test.assertEqual(worker.container_invocations.pending, ())
            test.assertFalse(await tasks.list_artifacts(run.run_id))
            return
        test.assertEqual(
            run.state,
            TaskRunState.SUCCEEDED,
            (
                run.result,
                backend.errors,
                backend.chunks,
                len(backend.plans),
                len(backend.created),
            ),
        )
        receipt = loads((output / "receipt.json").read_bytes())
        test.assertEqual(
            receipt["input_value"], "frozen private application input"
        )
        assert run.request.trigger is not None
        test.assertEqual(
            receipt["occurrence_id"], run.request.trigger.occurrence_id
        )
        test.assertEqual(
            receipt["deployment_id"], binding.manifest.execution_deployment_id
        )
        test.assertEqual(receipt["run_id"], run.run_id)
        test.assertEqual(len(backend.plans), 1)
        test.assertEqual(
            backend.plans[0].command.argv,
            ("avalan-task", "agent", "agent.toml"),
        )
        test.assertEqual(worker.container_invocations.pending, ())
        test.assertTrue(await tasks.list_artifacts(run.run_id))
    finally:
        await database.aclose()
        await drop_task_pgsql_schema(dsn, schema)
