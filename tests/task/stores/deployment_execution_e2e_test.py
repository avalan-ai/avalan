"""Exercise retained deployment files through the actual Agent task runner."""

from dataclasses import replace
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)
from task_deployment_helpers import (
    DeploymentProviderLoader,
    DeploymentProviderResponse,
)
from trigger.preparation_e2e_test import ContextCipher
from trigger.records_test import OWNER

from avalan.pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.canonical import spec_hash
from avalan.task.client import TaskClient
from avalan.task.context import TaskTargetContext
from avalan.task.deployment import ExecutionDeployment
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.state import TaskRunState
from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)
from avalan.task.targets.agent import AgentTaskTargetRunner
from avalan.task.worker import TaskWorker
from avalan.trigger.admission import TriggerCommitOutcome
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import (
    IntervalTrigger,
    MisfirePolicy,
    RecurringPolicy,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.error import TriggerErrorCode
from avalan.trigger.plan import AdmissionLimits, plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.records import TriggerOccurrence
from avalan.trigger.stores.pgsql import PgsqlTriggerStore, decision_time
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


async def unexpected_target(context: TaskTargetContext) -> object:
    raise AssertionError("worker must use the admitted deployment runner")


class DeploymentExecutionE2ETest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("avalan_deployment")
        task_pgsql_upgrade(
            PgsqlTaskMigrationSettings(url=dsn, schema=self.schema)
        )
        self.database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=task_pgsql_psycopg_dsn(dsn), schema=self.schema
            )
        )
        await self.database.open()
        self.tasks = PgsqlTaskStore(self.database)
        self.queue = PgsqlTaskQueue(self.database)
        self.store = PgsqlTriggerStore(self.database, OWNER)
        self.admission = PgsqlTriggerAdmissionStore(self.store, self.queue)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cipher = ContextCipher()

    async def asyncTearDown(self) -> None:
        if hasattr(self, "database"):
            await self.database.aclose()
            await drop_task_pgsql_schema(self.dsn, self.schema)

    async def test_builtin_agent_retry_uses_admitted_files_after_replacement(
        self,
    ) -> None:
        await self.exercise("stable")

    async def test_modified_deployment_fails_before_retry_dispatch(
        self,
    ) -> None:
        await self.exercise("retry")

    async def test_modified_deployment_cannot_activate(self) -> None:
        await self.exercise("apply")

    async def test_nested_task_admits_and_retries_after_replacement(
        self,
    ) -> None:
        await self.exercise("stable", nested=True)

    async def exercise(self, change_at: str, *, nested: bool = False) -> None:
        application_root = self.root / "old"
        root = application_root / "nested" if nested else application_root
        root.mkdir(parents=True)
        task_ref = "nested/task.toml" if nested else "task.toml"
        (root / "task.toml").write_text("""[task]
name="deployment-worker"
version="1"
[input]
type="string"
[output]
type="text"
[execution]
type="agent"
ref="agent.toml"
[run]
mode="queue"
queue="default"
idempotency="none"
[retry]
max_attempts=2
backoff="none"
[privacy]
input="encrypt"
files="drop"
raw_retention_days=1
""")
        (root / "agent.toml").write_text("""[agent]
name="old"
instructions="admitted old instructions"
[engine]
uri="ai://env:KEY@openai/gpt-4o-mini"
[tool]
enable=[]
""")
        loader = DeploymentProviderLoader(
            responses=(
                OSError("retry"),
                DeploymentProviderResponse("old result"),
            )
        )
        target = AgentTaskTargetRunner(loader, ref_base=root)
        definition = await TaskDefinitionLoader().load(root / "task.toml")
        runtime = Path(__file__).resolve().parents[3] / "src/avalan"
        manifest = ExecutionDeployment(
            task_ref=task_ref,
            task_hash=await spec_hash(definition, schema_base_path=root),
            files=ExecutionClosure(application_root, runtime).collect(
                task_ref
            ),
            runtime_version=version("avalan"),
            runtime_options=target.execution_deployment_options(root),
        )
        catalog = ExecutionDeploymentCatalog(
            allowed_runtime_options=tuple(
                option.name for option in manifest.runtime_options
            )
        )
        identity = await catalog.register(
            ExecutionDeploymentBinding(
                manifest=manifest,
                application_root=application_root,
                runtime_root=runtime,
                target=target,
                definition=definition,
            )
        )
        client = TaskClient(
            self.tasks,
            queue=self.queue,
            target=target,
            owner_scope=OWNER.value,
            execution_deployment_id=identity,
            execution_deployments=catalog,
            encryption_provider=self.cipher,
            raw_storage_allowed=True,
            execution_roots=(root,),
        )
        preparation = TriggerPreparationService(
            client,
            self.admission,
            PgsqlArtifactOwnership(self.database),
            self.cipher,
        )
        async with self.store.transaction() as unit:
            now = await decision_time(unit)
        historical = now - timedelta(minutes=3)
        configuration = TriggerConfiguration(
            name="bound-agent",
            task_ref=task_ref,
            schedule=IntervalTrigger(every_seconds=60, start_at=historical),
            input=TriggerInput(value="frozen input"),
            policy=RecurringPolicy(misfire=MisfirePolicy.ALL),
        )
        prepared = await preparation.prepare_registration(
            configuration, definition
        )
        if change_at == "apply":
            (root / "agent.toml").write_text(
                '[agent]\ninstructions="tampered"'
            )
            refused = await TriggerRegistrationService(preparation).apply(
                prepared, expected_generation=None
            )
            self.assertEqual(
                refused.outcome, TriggerCommitOutcome.NOT_COMMITTED
            )
            self.assertEqual(
                refused.error_code, TriggerErrorCode.DEPLOYMENT_MISMATCH
            )
            self.assertEqual(loader.paths, [])
            return
        # Only registration is placed in a controlled past; admission keeps
        # the real database clock and performs all authoritative rechecks.
        with patch(
            "avalan.trigger.apply.decision_time",
            AsyncMock(return_value=historical),
        ):
            registered = await TriggerRegistrationService(preparation).apply(
                prepared, expected_generation=None
            )
        assert registered.snapshot is not None
        plan = plan_admission(
            registered.snapshot,
            now,
            limits=AdmissionLimits(decisions=1, admissions=1),
        )
        admitted = await self.admission.admit(
            await preparation.prepare_admission(plan)
        )
        self.assertEqual(admitted.outcome, TriggerCommitOutcome.COMMITTED)
        self.assertEqual(len(admitted.resolved), 1)
        occurrence = admitted.resolved[0].decisions[0]
        assert isinstance(occurrence, TriggerOccurrence)
        assert occurrence.run_id is not None
        run = await self.tasks.get_run(occurrence.run_id)
        self.assertEqual(run.request.deployment, manifest)
        self.assertIsNotNone(run.request.trigger)
        replacement_application_root = self.root / "replacement"
        replacement_root = (
            replacement_application_root / "nested"
            if nested
            else replacement_application_root
        )
        replacement_root.mkdir(parents=True)
        (replacement_root / "task.toml").write_bytes(
            (root / "task.toml").read_bytes()
        )
        (replacement_root / "agent.toml").write_text(
            (root / "agent.toml")
            .read_text()
            .replace("admitted old instructions", "replacement instructions")
        )
        replacement_loader = DeploymentProviderLoader(
            response_text="replacement result"
        )
        replacement_target = AgentTaskTargetRunner(
            replacement_loader, ref_base=replacement_root
        )
        replacement_definition = await TaskDefinitionLoader().load(
            replacement_root / "task.toml"
        )
        replacement_manifest = replace(
            manifest,
            task_hash=await spec_hash(
                replacement_definition, schema_base_path=replacement_root
            ),
            files=ExecutionClosure(
                replacement_application_root, runtime
            ).collect(task_ref),
        )
        replacement_id = await catalog.register(
            ExecutionDeploymentBinding(
                manifest=replacement_manifest,
                application_root=replacement_application_root,
                runtime_root=runtime,
                target=replacement_target,
                definition=replacement_definition,
            )
        )
        replacement_client = TaskClient(
            self.tasks,
            queue=self.queue,
            target=replacement_target,
            owner_scope=OWNER.value,
            execution_deployment_id=replacement_id,
            execution_deployments=catalog,
            encryption_provider=self.cipher,
            raw_storage_allowed=True,
        )
        replacement_preparation = TriggerPreparationService(
            replacement_client,
            self.admission,
            PgsqlArtifactOwnership(self.database),
            self.cipher,
        )
        replacement_prepared = (
            await replacement_preparation.prepare_registration(
                replace(
                    configuration, schedule=IntervalTrigger(every_seconds=60)
                ),
                replacement_definition,
            )
        )
        assert admitted.snapshot is not None
        replaced = await TriggerRegistrationService(
            replacement_preparation
        ).apply(
            replacement_prepared,
            expected_generation=admitted.snapshot.state.generation,
        )
        self.assertEqual(replaced.outcome, TriggerCommitOutcome.COMMITTED)
        assert replaced.snapshot is not None
        self.assertEqual(replaced.snapshot.state.revision, 2)
        worker = TaskWorker(
            self.tasks,
            self.queue,
            target=replacement_target,
            execution_deployments=catalog,
            encryption_provider=self.cipher,
            raw_storage_allowed=True,
        )
        await worker.process_once()
        self.assertEqual(
            (await self.tasks.get_run(run.run_id)).state, TaskRunState.QUEUED
        )
        if change_at == "retry":
            (root / "agent.toml").write_text(
                '[agent]\ninstructions="tampered"'
            )
        await worker.process_once()
        settled = await self.tasks.get_run(run.run_id)
        if change_at == "retry":
            self.assertEqual(settled.state, TaskRunState.FAILED)
            transitions = await self.tasks.list_run_transitions(run.run_id)
            self.assertTrue(
                any(
                    item.metadata.get("error_code") == "deployment.mismatch"
                    for item in transitions
                )
            )
            self.assertIsNone(settled.result)
            self.assertEqual(loader.paths, [str(root / "agent.toml")])
            recovered = await TriggerRegistrationService(preparation).recover(
                prepared, expected_generation=None
            )
            self.assertEqual(recovered.outcome, TriggerCommitOutcome.COMMITTED)
        else:
            self.assertEqual(settled.state, TaskRunState.SUCCEEDED)
            self.assertEqual(loader.paths, [str(root / "agent.toml")] * 2)
            self.assertEqual(len(loader.inputs), 2)
            self.assertEqual(loader.inputs[0], loader.inputs[1])

        self.assertEqual(replacement_loader.paths, [])
        attempts = await self.tasks.list_attempts(run.run_id)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(
            all(
                attempt.context.trigger == run.request.trigger
                for attempt in attempts
            )
        )
        self.assertTrue(
            all(attempt.context.deployment == manifest for attempt in attempts)
        )
