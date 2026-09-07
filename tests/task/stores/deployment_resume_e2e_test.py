"""Resume an admitted native Agent through PostgreSQL continuation rows."""

from asyncio import wait_for
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent.durable_runtime_test import _agent_file, _ManagerFactory
from interaction.stores.interaction_pgsql_store_test import _Clock, _store
from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)
from trigger.preparation_e2e_test import ContextCipher
from trigger.records_test import OWNER

from avalan.agent.loader import OrchestratorLoader
from avalan.interaction import (
    AnswerProvenance,
    ConfirmationAnswer,
    InteractionCorrelation,
    QuestionId,
    ResolutionIdempotencyKey,
    ScopedInteractionLookup,
)
from avalan.interaction.entities import AnsweredResolution
from avalan.interaction.store import (
    InteractionRecord,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
)
from avalan.interaction.stores.pgsql import PgsqlDurableTaskCoordinator
from avalan.model.hubs.huggingface import HuggingfaceHub
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
from avalan.task.durable_agent import DurableAgentTaskHost
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.state import TaskRunState
from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)
from avalan.task.target import TaskTargetOutcome, TaskTargetSuspended
from avalan.task.targets.agent import AgentTaskTargetRunner
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


class RecordingDeploymentAgent(AgentTaskTargetRunner):
    def __init__(
        self,
        loader: OrchestratorLoader,
        root: Path,
        host: DurableAgentTaskHost,
    ) -> None:
        super().__init__(loader, ref_base=root, durable_host=host)
        self.contexts: list[TaskTargetContext] = []
        self.suspensions: list[TaskTargetSuspended] = []

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        outcome = await super().run(context)
        if isinstance(outcome, TaskTargetSuspended):
            self.suspensions.append(outcome)
        return outcome


class DeploymentResumeE2ETest(IsolatedAsyncioTestCase):
    async def test_admitted_agent_resumes_old_deployment_after_replacement(
        self,
    ) -> None:
        await self.run_scenario(tamper=False)

    async def test_modified_old_deployment_rejects_resumed_dispatch(
        self,
    ) -> None:
        await self.run_scenario(tamper=True)

    async def test_nested_task_resumes_retained_deployment_after_replacement(
        self,
    ) -> None:
        await self.run_scenario(tamper=False, nested=True)

    async def run_scenario(
        self, *, tamper: bool, nested: bool = False
    ) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        schema = isolated_task_pgsql_schema("deployment_resume")
        task_pgsql_upgrade(PgsqlTaskMigrationSettings(url=dsn, schema=schema))
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(dsn=task_pgsql_psycopg_dsn(dsn), schema=schema)
        )
        await database.open()
        try:
            await self.exercise(database, tamper=tamper, nested=nested)
        finally:
            await database.aclose()
            await drop_task_pgsql_schema(dsn, schema)

    async def exercise(
        self,
        database: PsycopgAsyncDatabase,
        *,
        tamper: bool,
        nested: bool = False,
    ) -> None:
        with TemporaryDirectory() as directory:
            async with AsyncExitStack() as stack:
                root = Path(directory).resolve()
                task_ref = "nested/task.toml" if nested else "task.toml"
                now = datetime.now(UTC)
                clock = _Clock()
                clock.now = now
                interactions = await _store(database, clock=clock)
                stack.push_async_callback(interactions.aclose)
                tasks = PgsqlTaskStore(
                    database, clock=lambda: datetime.now(UTC)
                )
                coordinator = PgsqlDurableTaskCoordinator(interactions, tasks)
                queue = PgsqlTaskQueue(
                    database,
                    clock=lambda: datetime.now(UTC),
                    durable_reentry_coordinator=coordinator,
                )
                triggers = PgsqlTriggerStore(database, OWNER)
                admission = PgsqlTriggerAdmissionStore(triggers, queue)
                cipher = ContextCipher()
                bindings: list[ExecutionDeploymentBinding] = []
                targets: list[RecordingDeploymentAgent] = []
                hosts: list[DurableAgentTaskHost] = []
                for name in ("old", "new"):
                    application_root = root / name
                    base = (
                        application_root / "nested"
                        if nested
                        else application_root
                    )
                    base.mkdir(parents=True)
                    agent = _agent_file(base)
                    agent.write_text(
                        agent.read_text().replace(
                            "[agent]", '[agent]\nuser_template="agent.md"'
                        )
                        + "\n[tool]\nenable=[]\n"
                    )
                    self.assertFalse((base / "agent.md").exists())
                    if name == "new":
                        agent.write_text(
                            agent.read_text().replace(
                                "Request confirmation",
                                "Replacement requests confirmation",
                            )
                        )
                    (base / "task.toml").write_text("""[task]
name="deployment-resume"
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
queue="resume"
[privacy]
input="encrypt"
files="drop"
raw_retention_days=1
""")
                    loader = OrchestratorLoader(
                        hub=MagicMock(spec=HuggingfaceHub),
                        logger=MagicMock(spec=Logger),
                        participant_id=uuid4(),
                        stack=stack,
                    )
                    host = DurableAgentTaskHost(
                        orchestrator_loader=loader,
                        stack=stack,
                        allowed_roots=(base,),
                        continuation_store=interactions,
                        clock=lambda: datetime.now(UTC),
                    )
                    target = RecordingDeploymentAgent(loader, base, host)
                    definition = await TaskDefinitionLoader().load(
                        base / "task.toml"
                    )
                    runtime = (
                        Path(__file__).resolve().parents[3] / "src/avalan"
                    )
                    manifest = ExecutionDeployment(
                        task_ref=task_ref,
                        task_hash=await spec_hash(
                            definition, schema_base_path=base
                        ),
                        files=ExecutionClosure(
                            application_root, runtime
                        ).collect(task_ref),
                        runtime_version=version("avalan"),
                        runtime_options=target.execution_deployment_options(
                            base
                        ),
                    )
                    bindings.append(
                        ExecutionDeploymentBinding(
                            manifest=manifest,
                            application_root=application_root,
                            runtime_root=runtime,
                            definition=definition,
                            target=target,
                        )
                    )
                    targets.append(target)
                    hosts.append(host)
                catalog = ExecutionDeploymentCatalog(
                    allowed_runtime_options=tuple(
                        option.name
                        for option in bindings[0].manifest.runtime_options
                    )
                )
                identities = [
                    await catalog.register(binding) for binding in bindings
                ]
                client = TaskClient(
                    tasks,
                    target=targets[0],
                    queue=queue,
                    owner_scope=OWNER.value,
                    execution_deployment_id=identities[0],
                    execution_deployments=catalog,
                    encryption_provider=cipher,
                    raw_storage_allowed=True,
                    durable_lifecycle_coordinator=coordinator,
                )
                preparation = TriggerPreparationService(
                    client, admission, PgsqlArtifactOwnership(database), cipher
                )
                configuration = TriggerConfiguration(
                    name="resume",
                    task_ref=task_ref,
                    schedule=IntervalTrigger(every_seconds=86400),
                    input=TriggerInput(value="frozen original input"),
                )
                prepared = await preparation.prepare_registration(
                    configuration, bindings[0].definition
                )
                with patch(
                    "avalan.trigger.apply.decision_time",
                    AsyncMock(return_value=now - timedelta(days=1, minutes=5)),
                ):
                    registered = await TriggerRegistrationService(
                        preparation
                    ).apply(prepared, expected_generation=None)
                assert registered.snapshot is not None
                decision = await admission.admit(
                    await preparation.prepare_admission(
                        plan_admission(registered.snapshot, datetime.now(UTC))
                    )
                )
                self.assertEqual(
                    decision.outcome, TriggerCommitOutcome.COMMITTED
                )
                worker = TaskWorker(
                    tasks,
                    queue,
                    target=targets[1],
                    execution_deployments=catalog,
                    queue_name="resume",
                    encryption_provider=cipher,
                    raw_storage_allowed=True,
                    durable_suspension_coordinator=coordinator,
                    durable_resume_coordinator=hosts[1].resume_coordinator,
                    clock=lambda: datetime.now(UTC),
                )
                initial_factory = _ManagerFactory([True])
                with patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=initial_factory,
                ):
                    first = await wait_for(worker.process_once(), 10)
                assert first.suspension is not None, first
                user_messages = [
                    item
                    for item in initial_factory.instances[0].serialized_inputs[
                        0
                    ]
                    if item.get("role") == "user"
                ]
                self.assertIn("helpful assistant", str(user_messages))
                self.assertEqual(
                    first.suspension.run.state, TaskRunState.INPUT_REQUIRED
                )
                suspended = targets[0].suspensions[0]
                assert suspended.durable is not None
                command = suspended.durable.command
                record = await interactions.lookup_scoped(
                    ScopedInteractionLookup(
                        actor=command.actor,
                        correlation=InteractionCorrelation.from_request(
                            command.request
                        ),
                    )
                )
                assert isinstance(record, InteractionRecord)
                request = record.request
                question = request.questions[0]
                now = datetime.now(UTC)
                clock.now = now
                resolution = ResolveInteractionCommand(
                    actor=command.actor,
                    correlation=InteractionCorrelation.from_request(request),
                    expected_state_revision=request.state_revision,
                    idempotency_key=ResolutionIdempotencyKey("answer"),
                    proposed_resolution=AnsweredResolution(
                        request_id=request.request_id,
                        provenance=AnswerProvenance.HUMAN,
                        resolved_at=now,
                        answers=(
                            ConfirmationAnswer(
                                question_id=QuestionId(
                                    str(question.question_id)
                                ),
                                provenance=AnswerProvenance.HUMAN,
                                value=True,
                            ),
                        ),
                    ),
                )
                accepted = await coordinator.resolve_and_requeue(
                    resolution,
                    task_run_id=first.suspension.run.run_id,
                    now=now,
                )
                self.assertIsInstance(
                    accepted.resolution, ResolveInteractionApplied
                )
                client._execution_deployment_id = identities[1]
                replacement = await preparation.prepare_registration(
                    configuration, bindings[1].definition
                )
                current = await triggers.inspect("resume")
                assert current is not None
                await TriggerRegistrationService(preparation).apply(
                    replacement, expected_generation=current.state.generation
                )
                if tamper:
                    old_agent = (
                        bindings[0].application_root / task_ref
                    ).parent / "agent.toml"
                    old_agent.write_text(
                        old_agent.read_text() + "\n# changed before resume\n"
                    )
                resumed_factory = _ManagerFactory([False], call_id_start=2)
                with patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=resumed_factory,
                ):
                    resumed = await worker.process_once()
                run = await tasks.get_run(first.suspension.run.run_id)
                if tamper:
                    self.assertEqual(run.state, TaskRunState.FAILED, resumed)
                    self.assertEqual(resumed_factory.instances, [])
                    self.assertIsNotNone(run.result)
                    assert run.result is not None
                    self.assertIn("deployment.mismatch", str(run.result.error))
                    self.assertEqual(len(targets[0].contexts), 1)
                else:
                    self.assertEqual(
                        run.state, TaskRunState.SUCCEEDED, resumed
                    )
                self.assertEqual(run.request.deployment, bindings[0].manifest)
                self.assertEqual(targets[1].contexts, [])
                attempts = await tasks.list_attempts(run.run_id)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(
                    attempts[0].context.trigger, run.request.trigger
                )
                self.assertEqual(
                    attempts[0].context.deployment, bindings[0].manifest
                )
