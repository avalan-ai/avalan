"""Reconstruct native durable continuation bindings in fresh worker roles."""

from asyncio import to_thread
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from json import dumps, loads
from logging import Logger
from pathlib import Path
from sys import stdin
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent.durable_runtime_test import _agent_file, _ManagerFactory
from deployment_resume_e2e_test import RecordingDeploymentAgent
from interaction.stores.interaction_pgsql_store_test import (
    _Authorizer,
    _Classifier,
    _Clock,
    _Ids,
)

from avalan.agent.loader import OrchestratorLoader
from avalan.interaction import (
    AnswerProvenance,
    ConfirmationAnswer,
    InteractionCorrelation,
    InteractionPolicy,
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
from avalan.interaction.stores.pgsql import (
    PgsqlDurableTaskCoordinator,
    PgsqlInteractionStoreFactory,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.pgsql import PsycopgAsyncDatabase
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.canonical import spec_hash
from avalan.task.client import TaskClient
from avalan.task.deployment import (
    ExecutionDeployment,
    execution_deployment_from_payload,
)
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.durable_agent import DurableAgentTaskHost
from avalan.task.encryption import AesGcmTaskCipher, TaskArtifactCipher
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.state import TaskRunState
from avalan.task.stores.pgsql import PgsqlTaskStore
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
from avalan.trigger.plan import AdmissionLimits, plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.records import OccurrenceDisposition, OwnerScopeId
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_types import TriggerSchedulerSettings
from avalan.trigger.stores.pgsql import PgsqlTriggerStore, decision_time
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


async def execute_resume(
    action: str, root: Path, database: PsycopgAsyncDatabase
) -> None:
    """Rebuild concrete native hosts from byte-pinned retained manifests."""
    async with AsyncExitStack() as stack:
        clock = _Clock()
        clock.now = datetime.now(UTC)
        # Only the interaction authorization/ID fixture is synthetic. Both
        # interaction payloads and task input use production authenticated AES.
        cipher = AesGcmTaskCipher(key_id="acceptance", key=b"a" * 32)
        policy = InteractionPolicy()
        interactions = await PgsqlInteractionStoreFactory(
            database,
            policy=policy,
            clock=clock,
            authorizer=_Authorizer(),
            id_factory=_Ids(),
            cipher=TaskArtifactCipher(cipher),
            classifier=_Classifier(policy),
        ).open()
        stack.push_async_callback(interactions.aclose)
        tasks = PgsqlTaskStore(database)
        coordinator = PgsqlDurableTaskCoordinator(interactions, tasks)
        queue = PgsqlTaskQueue(
            database, durable_reentry_coordinator=coordinator
        )
        owner = OwnerScopeId(value="acceptance-owner")
        triggers = PgsqlTriggerStore(database, owner)
        admission = PgsqlTriggerAdmissionStore(triggers, queue)
        bindings: list[ExecutionDeploymentBinding] = []
        targets: list[RecordingDeploymentAgent] = []
        hosts: list[DurableAgentTaskHost] = []
        for name in ("old", "new"):
            application = root / name
            base = application / "nested"
            manifest_path = application / "manifest.json"
            if action == "resume-register":
                base.mkdir(parents=True)
                agent = _agent_file(base)
                agent.write_text(agent.read_text() + "\n[tool]\nenable=[]\n")
                if name == "new":
                    agent.write_text(
                        agent.read_text().replace(
                            "Request confirmation",
                            "Replacement requests confirmation",
                        )
                    )
                (base / "task.toml").write_text("""[task]
name="process-resume"
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
idempotency="none"
[privacy]
input="encrypt"
output="encrypt"
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
            definition = await TaskDefinitionLoader().load(base / "task.toml")
            runtime = Path(__file__).resolve().parents[3] / "src/avalan"
            if action == "resume-register":
                manifest = ExecutionDeployment(
                    task_ref="nested/task.toml",
                    task_hash=await spec_hash(
                        definition, schema_base_path=base
                    ),
                    files=ExecutionClosure(application, runtime).collect(
                        "nested/task.toml"
                    ),
                    runtime_version=version("avalan"),
                    runtime_options=target.execution_deployment_options(base),
                )
                manifest_path.write_text(dumps(manifest.payload()))
            else:
                # A fresh process reads the old immutable manifest, never
                # authorizes current bytes by generating a replacement seal.
                manifest = execution_deployment_from_payload(
                    loads(manifest_path.read_text())
                )
            bindings.append(
                ExecutionDeploymentBinding(
                    manifest=manifest,
                    application_root=application,
                    runtime_root=runtime,
                    definition=definition,
                    target=target,
                )
            )
            targets.append(target)
            hosts.append(host)
        catalog = ExecutionDeploymentCatalog(
            allowed_runtime_options=tuple(
                option.name for option in bindings[0].manifest.runtime_options
            )
        )
        identities = [await catalog.register(binding) for binding in bindings]
        client = TaskClient(
            tasks,
            target=targets[0],
            queue=queue,
            owner_scope=owner.value,
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
            name="durable-resume",
            task_ref="nested/task.toml",
            schedule=IntervalTrigger(every_seconds=60),
            policy=RecurringPolicy(misfire=MisfirePolicy.ALL),
            input=TriggerInput(value="frozen resume input"),
        )
        if action == "resume-register":
            prepared = await preparation.prepare_registration(
                configuration, bindings[0].definition
            )
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(
                    return_value=datetime.now(UTC) - timedelta(minutes=3)
                ),
            ):
                registered = await TriggerRegistrationService(
                    preparation
                ).apply(prepared, expected_generation=None)
            assert registered.snapshot is not None
            admitted = await admission.admit(
                await preparation.prepare_admission(
                    plan_admission(
                        registered.snapshot,
                        datetime.now(UTC),
                        limits=AdmissionLimits(decisions=1, admissions=1),
                    )
                )
            )
            assert admitted.outcome == TriggerCommitOutcome.COMMITTED
            print(dumps({"registered": True}), flush=True)
            return
        if action == "resume-overlap":
            current = await triggers.inspect("durable-resume")
            assert current is not None and current.state.next_at is not None
            # Synchronize to the authoritative SQL boundary, not a sleep or
            # an assumed duration for registration/child startup.
            for _ in range(10000):
                async with triggers.transaction() as unit:
                    now = await decision_time(unit)
                if now >= current.state.next_at:
                    break
            else:
                raise AssertionError("database due boundary was not reached")
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(discovery_limit=10),
            )
            tick = await scheduler.process_once()
            assert tick.admitted == 0
            assert not tick.errors and not tick.preparation_failures
            assert not tick.unresolved and tick.pending_operations == 0
            stopped = await scheduler.shutdown()
            assert stopped.settled
            print(
                dumps(
                    {
                        "admitted": tick.admitted,
                        "shutdown_settled": stopped.settled,
                    }
                ),
                flush=True,
            )
            return
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
        )
        factory = _ManagerFactory(
            [action == "resume-suspend"],
            call_id_start=1 if action == "resume-suspend" else 2,
        )
        with patch("avalan.agent.loader.ModelManager", side_effect=factory):
            result = await worker.process_once()
        if action == "resume-suspend":
            assert result.suspension is not None
            assert result.suspension.run.state == TaskRunState.INPUT_REQUIRED
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
            now = datetime.now(UTC)
            clock.now = now
            client._execution_deployment_id = identities[1]
            replacement = await preparation.prepare_registration(
                TriggerConfiguration(
                    name="durable-resume",
                    task_ref="nested/task.toml",
                    schedule=IntervalTrigger(every_seconds=1),
                    policy=RecurringPolicy(misfire=MisfirePolicy.ALL),
                    input=configuration.input,
                ),
                bindings[1].definition,
            )
            current = await triggers.inspect("durable-resume")
            assert current is not None
            applied = await TriggerRegistrationService(preparation).apply(
                replacement, expected_generation=current.state.generation
            )
            assert applied.outcome == TriggerCommitOutcome.COMMITTED
            coverage = await triggers.coverage("durable-resume")
            assert any(
                span.disposition == OccurrenceDisposition.SUPERSEDED
                for span in coverage.items
            )
            assert applied.snapshot is not None
            print(
                dumps({"input_required": True, "replacement": True}),
                flush=True,
            )
            assert await to_thread(stdin.readline) == "resume\n"
            now = datetime.now(UTC)
            clock.now = now
            accepted = await coordinator.resolve_and_requeue(
                ResolveInteractionCommand(
                    actor=command.actor,
                    correlation=InteractionCorrelation.from_request(request),
                    expected_state_revision=request.state_revision,
                    idempotency_key=ResolutionIdempotencyKey("process-answer"),
                    proposed_resolution=AnsweredResolution(
                        request_id=request.request_id,
                        provenance=AnswerProvenance.HUMAN,
                        resolved_at=now,
                        answers=(
                            ConfirmationAnswer(
                                question_id=QuestionId(
                                    str(request.questions[0].question_id)
                                ),
                                provenance=AnswerProvenance.HUMAN,
                                value=True,
                            ),
                        ),
                    ),
                ),
                task_run_id=result.suspension.run.run_id,
                now=now,
            )
            assert isinstance(accepted.resolution, ResolveInteractionApplied)
            current = await triggers.inspect("durable-resume")
            assert current is not None
            await triggers.set_enabled(
                "durable-resume",
                enabled=False,
                expected_generation=current.state.generation,
            )
            (root / "resume-run.txt").write_text(result.suspension.run.run_id)
            print(
                dumps({"suspended": True, "answered": True, "replaced": True}),
                flush=True,
            )
        else:
            assert action == "resume-finish"
            run = await tasks.get_run((root / "resume-run.txt").read_text())
            assert run.state == TaskRunState.SUCCEEDED, result
            assert run.request.deployment == bindings[0].manifest
            assert targets[1].contexts == []
            attempts = await tasks.list_attempts(run.run_id)
            assert len(attempts) == 1
            assert attempts[0].context.trigger == run.request.trigger
            assert attempts[0].context.deployment == bindings[0].manifest
            print(
                dumps(
                    {
                        "resumed": True,
                        "old_deployment": True,
                        "attempts": 1,
                    }
                ),
                flush=True,
            )
