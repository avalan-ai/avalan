"""Race independently prepared occurrence requests across OS processes."""

from asyncio import to_thread
from contextlib import AsyncExitStack
from datetime import datetime
from json import dumps, loads
from logging import getLogger
from pathlib import Path
from sys import stdin
from unittest.mock import patch
from uuid import uuid4

from avalan.agent.loader import OrchestratorLoader
from avalan.cli.task_privacy import task_hmac_provider
from avalan.cli.trigger_host import TriggerCliHost
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.pgsql import PsycopgAsyncDatabase
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.client import TaskClient
from avalan.task.encryption import AesGcmTaskCipher
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.stores.pgsql import PgsqlTaskStore
from avalan.trigger.admission import PreparedTriggerAdmission
from avalan.trigger.codec import decode_record
from avalan.trigger.plan import TriggerAdmissionPlan, plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.records import (
    OwnerScopeId,
    TriggerDefinition,
    TriggerOccurrence,
    TriggerSnapshot,
    TriggerState,
)
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_types import TriggerSchedulerSettings
from avalan.trigger.stores.pgsql import PgsqlTriggerStore, decision_time
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


async def execute_competitor(
    root: Path, database: PsycopgAsyncDatabase
) -> None:
    """Prepare before the parent barrier, then admit the same stable slots."""
    async with AsyncExitStack() as stack:
        loader = OrchestratorLoader(
            hub=HuggingfaceHub(
                "unused", str(root / "cache"), getLogger(__name__)
            ),
            logger=getLogger(__name__),
            participant_id=uuid4(),
            stack=stack,
        )
        host = TriggerCliHost(root / "catalog", loader)
        await host.restore()
        store = PgsqlTriggerStore(
            database, OwnerScopeId(value="acceptance-owner")
        )
        queue = PgsqlTaskQueue(database)
        admission = PgsqlTriggerAdmissionStore(store, queue)
        cipher = AesGcmTaskCipher(key_id="acceptance", key=b"a" * 32)
        prepared: list[PreparedTriggerAdmission] = []
        for value in loads((root / "plans.json").read_text()):
            definition, state = (
                decode_record(value["definition"]),
                decode_record(value["state"]),
            )
            assert isinstance(definition, TriggerDefinition)
            assert isinstance(state, TriggerState)
            binding = host.bindings[definition.execution_deployment_id]
            client = TaskClient(
                PgsqlTaskStore(database),
                queue=queue,
                target=binding.target,
                owner_scope=store.owner.value,
                execution_deployment_id=definition.execution_deployment_id,
                execution_deployments=host.catalog,
                encryption_provider=cipher,
                hmac_provider=task_hmac_provider(),
                raw_storage_allowed=True,
            )
            preparation = TriggerPreparationService(
                client, admission, PgsqlArtifactOwnership(database), cipher
            )
            prepared.append(
                await preparation.prepare_admission(
                    plan_admission(
                        TriggerSnapshot(definition=definition, state=state),
                        datetime.fromisoformat(value["prepared_at"]),
                    )
                )
            )
        print(dumps({"prepared": len(prepared)}), flush=True)
        assert await to_thread(stdin.readline) == "admit\n"
        results = []
        for request in prepared:
            result = await admission.admit(request)
            if result.contended:
                result = await admission.recover(request.plan)
            results.append(result)
        print(
            dumps(
                {
                    "outcomes": [result.outcome.value for result in results],
                    "runs": [
                        occurrence.run_id
                        for result in results
                        for resolved in result.resolved
                        for occurrence in resolved.decisions
                        if isinstance(occurrence, TriggerOccurrence)
                    ],
                }
            ),
            flush=True,
        )


async def execute_failure(root: Path, database: PsycopgAsyncDatabase) -> None:
    """Persist a bounded admission failure budget across fresh schedulers."""
    async with AsyncExitStack() as stack:
        loader = OrchestratorLoader(
            hub=HuggingfaceHub(
                "unused", str(root / "cache"), getLogger(__name__)
            ),
            logger=getLogger(__name__),
            participant_id=uuid4(),
            stack=stack,
        )
        host = TriggerCliHost(root / "catalog", loader)
        await host.restore()
        binding = next(iter(host.bindings.values()))
        store = PgsqlTriggerStore(
            database, OwnerScopeId(value="acceptance-owner")
        )
        queue = PgsqlTaskQueue(database)
        cipher = AesGcmTaskCipher(key_id="acceptance", key=b"a" * 32)
        client = TaskClient(
            PgsqlTaskStore(database),
            queue=queue,
            target=binding.target,
            owner_scope=store.owner.value,
            execution_deployment_id=binding.manifest.execution_deployment_id,
            execution_deployments=host.catalog,
            encryption_provider=cipher,
            hmac_provider=task_hmac_provider(),
            raw_storage_allowed=True,
        )
        preparation = TriggerPreparationService(
            client,
            PgsqlTriggerAdmissionStore(store, queue),
            PgsqlArtifactOwnership(database),
            cipher,
        )
        before = await store.inspect("failure-bad")
        assert before is not None
        if before.state.retry_after is not None:
            for _ in range(10000):
                async with store.transaction() as unit:
                    now = await decision_time(unit)
                if now >= before.state.retry_after:
                    break
            else:
                raise AssertionError("durable retry time was not reached")
        original = preparation.prepare_admission

        async def fail_bad(
            plan: TriggerAdmissionPlan,
        ) -> PreparedTriggerAdmission:
            if plan.snapshot.definition.name == "failure-bad":
                raise OSError("private preparation failure must stay private")
            return await original(plan)

        scheduler = TriggerScheduler(
            preparation,
            settings=TriggerSchedulerSettings(admission_retry_attempts=2),
        )
        with patch.object(preparation, "prepare_admission", fail_bad):
            tick = await scheduler.process_once()
        stopped = await scheduler.shutdown()
        assert stopped.settled
        assert len(tick.errors) == 1
        assert "private" not in repr(tick.errors)
        after = await store.inspect("failure-bad")
        assert after is not None
        assert after.state.failure_count == before.state.failure_count + 1
        assert after.state.next_at == before.state.next_at
        assert not (await store.occurrences("failure-bad")).items
        good = await store.occurrences("failure-good")
        assert len(good.items) == 1 and good.items[0].run_id is not None
        assert (
            await PgsqlTaskStore(database).get_run(good.items[0].run_id)
        ).state.value == "queued"
        print(
            dumps(
                {
                    "failures": after.state.failure_count,
                    "status": after.state.status.value,
                    "admitted": tick.admitted,
                    "admission_retries": tick.admission_retries,
                    "good_runs": 1,
                }
            ),
            flush=True,
        )
