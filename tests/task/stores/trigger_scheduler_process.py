"""Run a scheduler in a fresh interpreter against an isolated test schema."""

from argparse import ArgumentParser
from asyncio import Event, get_running_loop, run
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from os import _exit, environ
from pathlib import Path
from signal import SIGTERM
from unittest.mock import AsyncMock, patch

from pgsql_harness import task_pgsql_psycopg_dsn
from trigger.preparation_e2e_test import ContextCipher, file_task, target
from trigger.scheduler_fairness_test import (
    prove_advancing_rotation,
    register_backlogs,
)

from avalan.pgsql import (
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
)
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.artifacts.pgsql import (
    PgsqlArtifactByteStoragePolicy,
    PgsqlArtifactStore,
)
from avalan.task.client import TaskClient
from avalan.task.feature_gate import TaskFeature
from avalan.task.privacy import EncryptedPrivacyValue, TaskKeyPurpose
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.stores.pgsql import PgsqlTaskStore
from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionResult,
)
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.codec import decode_record, encode_record
from avalan.trigger.definition import (
    IntervalTrigger,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.plan import plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.records import (
    OwnerScopeId,
    TriggerDefinition,
    TriggerSnapshot,
    TriggerState,
)
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_types import (
    TriggerSchedulerSettings,
    TriggerTickStop,
)
from avalan.trigger.store import TriggerDiscoveryCursor
from avalan.trigger.stores.pgsql import PgsqlTriggerStore
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


class ProcessCipher(ContextCipher):
    """Adapt AES encryption to privacy and Pgsql byte envelopes."""

    def decrypt(
        self,
        value: bytes | EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        algorithm: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        if isinstance(value, EncryptedPrivacyValue):
            key_id, algorithm, content = (
                value.key_id,
                value.algorithm,
                value.ciphertext,
            )
        else:
            content = value
        return super().decrypt(
            content,
            purpose=purpose,
            key_id=key_id,
            algorithm=algorithm,
            context=context,
        )


async def execute(action: str, schema: str, root: Path) -> None:
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(
            dsn=task_pgsql_psycopg_dsn(
                environ["AVALAN_TASK_TEST_POSTGRESQL_DSN"]
            ),
            schema=schema,
        )
    )
    await database.open()
    cipher = ProcessCipher()
    tasks = PgsqlTaskStore(database)
    queue = PgsqlTaskQueue(database)
    store = PgsqlTriggerStore(
        database, OwnerScopeId(value="scheduler-process")
    )
    admission = PgsqlTriggerAdmissionStore(store, queue)
    backend = PgsqlArtifactStore(
        database,
        cipher=cipher,
        policy=PgsqlArtifactByteStoragePolicy(
            raw_storage_allowed=True,
            retention_days=1,
            max_bytes=4096,
            enabled_features=(TaskFeature.POSTGRESQL, TaskFeature.RAW_STORAGE),
        ),
    )
    client = TaskClient(
        tasks,
        target=target,
        queue=queue,
        owner_scope=store.owner.value,
        execution_deployment_id="process-deployment",
        encryption_provider=cipher,
        raw_storage_allowed=True,
        artifact_store=backend,
        execution_roots=(root,),
    )
    service = TriggerPreparationService(
        client, admission, PgsqlArtifactOwnership(database), cipher
    )
    scheduler = TriggerScheduler(
        service,
        owned_resources=(database,),
        settings=TriggerSchedulerSettings(
            candidate_evaluations=3 if action == "budget" else 10000,
            discovery_limit=1 if action == "fairness" else 100,
            decisions_per_trigger=1 if action == "fairness" else 10,
            admissions_per_trigger=1 if action == "fairness" else 10,
        ),
    )
    try:
        if action == "fairness":
            historical = datetime.now(UTC) - timedelta(minutes=10)
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(return_value=historical),
            ):
                await register_backlogs(service)
            # Every scheduler and admission clock is the actual database clock.
            await prove_advancing_rotation(scheduler, scheduler._clock)
        elif action == "register":
            prepared = await service.prepare_registration(
                TriggerConfiguration(
                    name="daily",
                    task_ref="task.toml",
                    schedule=IntervalTrigger(every_seconds=86400),
                    input=TriggerInput(
                        value={
                            "source_kind": "local_path",
                            "reference": "input.txt",
                            "mime_type": "text/plain",
                        }
                    ),
                ),
                file_task(),
            )
            # Controlled historical registration gives a due slot without a
            # wall-time sleep; actual scheduler/admission clocks remain SQL.
            historical = datetime.now(UTC) - timedelta(days=1, minutes=5)
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(return_value=historical),
            ):
                result = await TriggerRegistrationService(service).apply(
                    prepared, expected_generation=None
                )
            assert result.snapshot is not None
            registered_state = result.snapshot.state
            discovered = await store.discover(
                decision_time=datetime.now(UTC),
                limit=1,
                after=TriggerDiscoveryCursor(
                    last_processed_at=registered_state.last_processed_at,
                    round_started_at=registered_state.last_processed_at,
                    trigger_id=registered_state.trigger_id,
                ),
            )
            assert len(discovered) == 1
            assert await store.next_eligible_at() == registered_state.next_at
            print(dumps({"registered": result.outcome.value}), flush=True)
        elif action == "recover":
            evidence = loads((root / "recovery.json").read_text())
            definition = decode_record(evidence["definition"])
            state = decode_record(evidence["state"])
            assert isinstance(definition, TriggerDefinition) and isinstance(
                state, TriggerState
            )
            plan = plan_admission(
                TriggerSnapshot(definition=definition, state=state),
                datetime.fromisoformat(evidence["prepared_at"]),
            )
            recovered = await admission.recover(plan)
            tick = await scheduler.process_once()
            occurrences = await store.occurrences("daily")
            assert (
                len(occurrences.items) == 1
                and occurrences.items[0].run_id is not None
            )
            artifacts = await tasks.list_artifacts(occurrences.items[0].run_id)
            with await backend.open(artifacts[0].ref) as reader:
                assert reader.read() == b"durable scheduler process input"
            print(
                dumps(
                    {
                        "recovered": recovered.outcome.value,
                        "admitted": tick.admitted,
                        "occurrences": len(occurrences.items),
                    }
                ),
                flush=True,
            )
        elif action == "crash":
            original_admit, transaction = admission.admit, store.transaction

            @asynccontextmanager
            async def lost_ack() -> AsyncIterator[PgsqlUnitOfWork]:
                async with transaction() as unit:
                    yield unit
                _exit(23)

            async def crash(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                (root / "recovery.json").write_text(
                    dumps(
                        {
                            "definition": encode_record(
                                prepared.plan.snapshot.definition
                            ),
                            "state": encode_record(
                                prepared.plan.snapshot.state
                            ),
                            "prepared_at": (
                                prepared.plan.prepared_at.isoformat()
                            ),
                        }
                    )
                )
                with patch.object(store, "transaction", lost_ack):
                    return await original_admit(prepared)

            with patch.object(admission, "admit", crash):
                await scheduler.process_once()
            raise AssertionError("crash hook did not terminate")
        elif action == "serve":
            stop, blocked = Event(), Event()
            get_running_loop().add_signal_handler(SIGTERM, stop.set)

            async def prepared_wait(
                prepared: PreparedTriggerAdmission,
            ) -> TriggerAdmissionResult:
                print(
                    dumps({"prepared": len(prepared.submissions)}), flush=True
                )
                await blocked.wait()
                raise AssertionError("the parent must signal shutdown")

            with patch.object(admission, "admit", prepared_wait):
                shutdown = await scheduler.serve(stop=stop)
            print(
                dumps(
                    {
                        "settled": shutdown.settled,
                        "pending": shutdown.pending_operations,
                    }
                ),
                flush=True,
            )
        else:
            assert action in {"tick", "budget"}
            before = await store.inspect("daily")
            processed = await scheduler.process_once()
            if action == "budget":
                assert processed.stop == TriggerTickStop.WORK_LIMIT
                assert processed.admitted == 0 and not processed.errors
                assert await store.inspect("daily") == before
                assert not (await store.occurrences("daily")).items
            else:
                assert await scheduler._wait_seconds(processed) == 1.0
            print(
                dumps(
                    {
                        "admitted": processed.admitted,
                        "pending": processed.pending_operations,
                        "errors": [
                            item.code.value for item in processed.errors
                        ],
                    }
                ),
                flush=True,
            )
    finally:
        await scheduler.shutdown()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "action",
        choices=("register", "tick", "crash", "recover", "serve", "budget"),
    )
    parser.add_argument("schema")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run(execute(args.action, args.schema, args.root))


if __name__ == "__main__":
    main()
