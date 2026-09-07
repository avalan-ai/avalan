"""Exercise trigger SQL with real task persistence, not schema mocks."""

from asyncio import CancelledError, Event, Task, create_task, gather, wait_for
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)
from pytest import importorskip
from task.artifacts.object_store_test import FakeObjectClient
from task_submission_helpers import prepared_submission_fixture
from trigger.preparation_e2e_test import ContextCipher, file_task, target
from trigger.records_test import OWNER
from trigger.registration_test import registration

from avalan.pgsql import (
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
)
from avalan.task.artifact import (
    ArtifactStoreNotFoundError,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
)
from avalan.task.artifact_ownership import ArtifactObject, ArtifactStagingOwner
from avalan.task.artifact_retention import (
    retire_artifact_bytes,
    retry_artifact_cleanup,
)
from avalan.task.artifacts.object_store import (
    ObjectArtifactStore,
    ObjectArtifactStorePolicy,
)
from avalan.task.artifacts.ownership_pgsql import PgsqlArtifactOwnership
from avalan.task.client import TaskClient
from avalan.task.definition import (
    IdempotencyMode,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRunPolicy,
)
from avalan.task.idempotency import (
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
)
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.retention import TaskRetentionService
from avalan.task.state import TaskRunState
from avalan.task.store import (
    TaskExecutionRequest,
    TaskStoreError,
    TaskStoreNotFoundError,
)
from avalan.task.stores.pgsql import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)
from avalan.task.submission import (
    TaskSubmissionArtifact,
    TaskSubmissionOutcome,
)
from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import (
    IntervalTrigger,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.plan import plan_admission
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.records import (
    TriggerCoverageSpan,
    TriggerOccurrence,
    TriggerSnapshot,
    TriggerStatus,
)
from avalan.trigger.registration import apply_registration
from avalan.trigger.resources import TriggerResourceService
from avalan.trigger.stores.pgsql import PgsqlTriggerStore, decision_time
from avalan.trigger.stores.pgsql_admission import PgsqlTriggerAdmissionStore


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="trigger-e2e", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("pgsql-e2e"),
    )


class PgsqlTriggerAdmissionE2ETest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("psycopg")
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("avalan_trigger_admission")
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
        await self.tasks.register_definition(
            _definition(), definition_hash="task"
        )

    async def asyncTearDown(self) -> None:
        if hasattr(self, "database"):
            await self.database.aclose()
            await drop_task_pgsql_schema(self.dsn, self.schema)

    async def test_competing_clients_and_recovery_after_pause(self) -> None:
        # Set up a due slot using one actual database clock and the same
        # registration reducer; no timing margin or past-start exemption.
        async with self.store.transaction() as unit:
            first = await decision_time(unit)
            mutation = apply_registration(
                OWNER,
                replace(
                    registration(),
                    schedule=IntervalTrigger(every_seconds=60, start_at=first),
                ),
                None,
                expected_generation=None,
                decision_time=first,
                trigger_id="database-clock-trigger",
            )
            await self.store._write(unit, mutation)
            snapshot = mutation.snapshot
        plan = plan_admission(snapshot, datetime.now(UTC))
        prepared = replace(
            prepared_submission_fixture(
                self.queue,
                TaskExecutionRequest(definition_id="task"),
                queue_name="pgsql-e2e",
                available_at=first,
            ),
            owner_scope=OWNER.value,
            occurrence_id=plan.admission_ids[0],
            execution_deployment_id="deployment",
        )
        request = PreparedTriggerAdmission(plan=plan, submissions=(prepared,))
        other = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(self.database, OWNER), self.queue
        )
        results = await gather(
            self.admission.admit(request), other.admit(request)
        )
        assert any(
            result.outcome == TriggerCommitOutcome.COMMITTED
            for result in results
        )
        recovered = await other.recover(plan)
        assert recovered.outcome == TriggerCommitOutcome.COMMITTED
        row_decision = recovered.resolved[0].decisions[0]
        assert isinstance(row_decision, TriggerOccurrence)
        assert row_decision.run_id == prepared.run_id
        current = await self.store.inspect("daily")
        assert current is not None
        await self.store.set_enabled(
            "daily",
            enabled=False,
            expected_generation=current.state.generation,
        )
        repeated = await other.admit(request)
        assert repeated.outcome == TriggerCommitOutcome.COMMITTED
        async with self.database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT (SELECT count(*) FROM trigger_occurrences) AS"
                    " occurrences, (SELECT count(*) FROM task_runs) AS runs,"
                    " (SELECT count(*) FROM task_submissions) AS submissions"
                )
                row = await cursor.fetchone()
                assert row == {"occurrences": 1, "runs": 1, "submissions": 1}

    async def test_validated_files_restore_and_retention_share_physical_bytes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_bytes(b"durable registered input")
            cipher = ContextCipher()
            backend = ObjectArtifactStore(
                FakeObjectClient(),
                cipher=cipher,
                policy=ObjectArtifactStorePolicy(
                    raw_storage_allowed=True, retention_days=1, max_bytes=4096
                ),
            )
            ownership = PgsqlArtifactOwnership(self.database)
            client = TaskClient(
                self.tasks,
                target=target,
                queue=self.queue,
                owner_scope=OWNER.value,
                execution_deployment_id="deployment",
                encryption_provider=cipher,
                raw_storage_allowed=True,
                artifact_store=backend,
                execution_roots=(root,),
            )
            preparation = TriggerPreparationService(
                client, self.admission, ownership, cipher
            )
            registration = TriggerRegistrationService(preparation)
            prepared = await preparation.prepare_registration(
                TriggerConfiguration(
                    name="files",
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
            # Register against a controlled historical decision clock so the
            # public preparation/ownership path has one due slot without a
            # real-time sleep. Admission still reads its actual database clock.
            historical = datetime.now(UTC) - timedelta(days=1, minutes=5)
            with patch(
                "avalan.trigger.apply.decision_time",
                AsyncMock(return_value=historical),
            ):
                result = await registration.apply(
                    prepared, expected_generation=None
                )
            assert result.outcome == TriggerCommitOutcome.COMMITTED
            assert result.snapshot is not None
            source.unlink()
            fresh_client = TaskClient(
                self.tasks,
                target=target,
                queue=self.queue,
                owner_scope=OWNER.value,
                execution_deployment_id="deployment",
                encryption_provider=cipher,
                raw_storage_allowed=True,
                artifact_store=backend,
                execution_roots=(root,),
            )
            fresh = TriggerPreparationService(
                fresh_client, self.admission, ownership, cipher
            )
            snapshot = result.snapshot
            for _ in range(5):
                plan = plan_admission(snapshot, datetime.now(UTC))
                submission = await fresh.prepare_admission(plan)
                admitted = await self.admission.admit(submission)
                if admitted.outcome == TriggerCommitOutcome.COMMITTED:
                    assert (
                        await fresh.release_unused_admission(submission)
                    ).outcome == TriggerCommitOutcome.COMMITTED
                    break
                assert admitted.replan_required
            else:
                self.fail(
                    "bounded admission kept crossing schedule boundaries"
                )
            item = submission.submissions[0]
            ref = item.artifacts[0].ref
            with await backend.open(ref) as stream:
                assert stream.read() == b"durable registered input"
            assert ref.artifact_id != prepared.objects[0].ref.artifact_id
            retention = TaskRetentionService(self.tasks, {ref.store: backend})
            assert not (
                await retention.enforce_run(
                    item.run_id, now=datetime.now(UTC) + timedelta(days=2)
                )
            ).results
            await self.tasks.transition_run(
                item.run_id,
                from_states={TaskRunState.QUEUED},
                to_state=TaskRunState.EXPIRED,
                reason="test",
            )
            assert (
                len(
                    (
                        await retention.enforce_run(
                            item.run_id,
                            now=datetime.now(UTC) + timedelta(days=2),
                        )
                    ).results
                )
                == 1
            )
            assert (await backend.stat(ref)).size_bytes == len(
                b"durable registered input"
            )
            async with self.database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT (SELECT count(*) FROM trigger_input_artifacts"
                        " WHERE released_at IS NULL) AS revisions, (SELECT"
                        " count(*) FROM task_artifact_run_owners WHERE"
                        " released_at IS NULL) AS runs, (SELECT count(*) FROM"
                        " task_artifact_objects WHERE status='live') AS"
                        " objects"
                    )
                    assert await cursor.fetchone() == {
                        "revisions": 1,
                        "runs": 0,
                        "objects": 1,
                    }

            resources = TriggerResourceService(self.store, ownership)
            assert not await resources.release_closed_revision("files", 1)
            source.write_bytes(b"replacement revision input")
            replacement = await preparation.prepare_registration(
                TriggerConfiguration(
                    name="files",
                    task_ref="task.toml",
                    schedule=IntervalTrigger(every_seconds=43200),
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
            current = await self.store.inspect("files")
            assert current is not None
            changed = await registration.apply(
                replacement, expected_generation=current.state.generation
            )
            assert (
                changed.snapshot is not None
                and changed.snapshot.definition.revision == 2
            )
            token = str(uuid4())
            outstanding = ArtifactStagingOwner(
                staging_id=token, owner_scope=OWNER.value, recovery_id=token
            )
            async with ownership.transaction() as unit:
                await ownership.stage(
                    prepared.objects[0], outstanding, unit_of_work=unit
                )
            assert not await resources.release_closed_revision("files", 1)
            async with ownership.transaction() as unit:
                await ownership.release_staging(
                    outstanding,
                    outcome=TaskSubmissionOutcome.NOT_COMMITTED,
                    unit_of_work=unit,
                )
            assert await resources.release_closed_revision("files", 1)
            assert (
                await retry_artifact_cleanup(ownership, {ref.store: backend})
                == 0
            )
            assert (
                await retry_artifact_cleanup(
                    ownership, {ref.store: backend}, orphan_grace_seconds=0
                )
                == 1
            )
            with self.assertRaises(ArtifactStoreNotFoundError):
                await backend.open(ref)
            assert (
                await registration.recover(
                    replacement, expected_generation=current.state.generation
                )
            ).outcome == TriggerCommitOutcome.COMMITTED
            assert (
                await registration.apply(
                    replacement, expected_generation=current.state.generation
                )
            ).outcome == TriggerCommitOutcome.COMMITTED
            assert (
                await registration.recover(
                    replacement, expected_generation=999
                )
            ).outcome == TriggerCommitOutcome.UNKNOWN
            assert (
                await registration.release_unused(
                    replacement, expected_generation=current.state.generation
                )
            ).outcome == TriggerCommitOutcome.COMMITTED
            ordinary = await backend.put(
                b"expired ordinary", artifact_id="ordinary-expired"
            )
            ordinary_record = await self.tasks.append_artifact(
                item.run_id,
                ref=replace(ordinary, sha256=None, size_bytes=None),
                purpose=TaskArtifactPurpose.OUTPUT,
            )
            await retire_artifact_bytes(
                ordinary_record,
                await backend.stat(ordinary),
                backend,
                ownership,
            )
            with self.assertRaises(ArtifactStoreNotFoundError):
                await backend.open(ordinary)

    async def test_management_clients_enforce_cas_and_revision_history(
        self,
    ) -> None:
        first = await self.store.apply(
            registration(), expected_generation=None
        )
        other = PgsqlTriggerStore(self.database, OWNER)
        results = await gather(
            self.store.set_enabled(
                "daily",
                enabled=False,
                expected_generation=first.state.generation,
            ),
            other.set_enabled(
                "daily",
                enabled=False,
                expected_generation=first.state.generation,
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(value, TriggerError) for value in results) == 1
        conflict = next(
            value for value in results if isinstance(value, TriggerError)
        )
        assert conflict.code == TriggerErrorCode.CONFLICT
        paused = await other.inspect("daily")
        assert (
            paused is not None and paused.state.status == TriggerStatus.PAUSED
        )
        changed = replace(
            registration(),
            semantic_hash="2" * 64,
            schedule=IntervalTrigger(every_seconds=86400),
            desired_enabled=True,
        )
        revisions = await gather(
            self.store.apply(
                changed, expected_generation=paused.state.generation
            ),
            other.apply(changed, expected_generation=paused.state.generation),
            return_exceptions=True,
        )
        assert sum(isinstance(value, TriggerError) for value in revisions) == 1
        current = await other.inspect("daily")
        assert current is not None and current.definition.revision == 2
        assert await other.revision("daily", 1) == first.definition
        assert len((await other.events("daily")).items) == 3
        resources = TriggerResourceService(
            other, PgsqlArtifactOwnership(self.database)
        )
        assert not await resources.release_closed_revision("daily", 2)
        assert not await resources.release_closed_revision("daily", 999)
        assert await resources.release_closed_revision("daily", 1)
        with self.assertRaises(TriggerError):
            await resources.release_closed_revision("missing", 1)

    async def test_each_admission_write_failure_rolls_back_real_transaction(
        self,
    ) -> None:
        async with self.store.transaction() as unit:
            first = await decision_time(unit)
            mutation = apply_registration(
                OWNER,
                replace(
                    registration(),
                    schedule=IntervalTrigger(
                        every_seconds=86400, start_at=first
                    ),
                ),
                None,
                expected_generation=None,
                decision_time=first,
                trigger_id="rollback-trigger",
            )
            await self.store._write(unit, mutation)
        plan = plan_admission(mutation.snapshot, datetime.now(UTC))
        prepared = replace(
            prepared_submission_fixture(
                self.queue,
                TaskExecutionRequest(definition_id="task"),
                queue_name="pgsql-e2e",
                available_at=first,
            ),
            owner_scope=OWNER.value,
            occurrence_id=plan.admission_ids[0],
            execution_deployment_id="deployment",
        )
        digest = TaskIdempotencyDigest(
            algorithm="hmac-sha256", digest="a" * 64, key_id="test-key"
        )
        artifact = ArtifactObject.from_ref(
            TaskArtifactRef(
                artifact_id="fault-artifact",
                store="object",
                storage_key="objects/fa/fault-artifact",
                sha256="b" * 64,
                size_bytes=4,
            )
        )
        prepared = replace(
            prepared,
            execution=replace(
                prepared.execution,
                idempotency_key="occurrence-" + plan.admission_ids[0],
            ),
            artifacts=(TaskSubmissionArtifact(ref=artifact.ref),),
            idempotency=TaskIdempotencyIdentity(
                identity_key="occurrence-" + plan.admission_ids[0],
                task_name="daily",
                task_version="1",
                spec_hash="task",
                owner_scope=digest,
                strategy=IdempotencyMode.INPUT_HASH,
                input=digest,
            ),
        )
        staging = ArtifactStagingOwner(
            staging_id=prepared.submission_id,
            owner_scope=OWNER.value,
            recovery_id=prepared.submission_id,
        )
        async with self.admission.ownership.transaction() as unit:
            await self.admission.ownership.stage(
                artifact, staging, unit_of_work=unit
            )
        request = PreparedTriggerAdmission(plan=plan, submissions=(prepared,))
        async with self.store.transaction() as unit:
            await unit.cursor.execute("""
CREATE FUNCTION injected_admission_failure() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'injected admission write failure'; END;
$$
""")
        for table in (
            "task_runs",
            "task_run_transitions",
            "task_queue_items",
            "task_submissions",
            "trigger_occurrences",
            "triggers",
            "trigger_events",
            "task_artifacts",
            "task_idempotency_keys",
            "task_artifact_run_owners",
        ):
            with self.subTest(table=table):
                async with self.store.transaction() as unit:
                    await unit.cursor.execute(
                        "CREATE TRIGGER injected_write_failure AFTER INSERT"
                        f" OR UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION"
                        " injected_admission_failure()"
                    )
                try:
                    result = await self.admission.admit(request)
                    assert result.outcome == TriggerCommitOutcome.NOT_COMMITTED
                    async with self.store.transaction() as unit:
                        for target_table in (
                            "task_runs",
                            "task_run_transitions",
                            "task_queue_items",
                            "task_submissions",
                            "trigger_occurrences",
                            "task_artifacts",
                            "task_idempotency_keys",
                            "task_artifact_run_owners",
                        ):
                            await unit.cursor.execute(
                                f"SELECT count(*) AS count FROM {target_table}"
                            )
                            assert await unit.cursor.fetchone() == {"count": 0}
                    assert (
                        await self.store.inspect("daily") == mutation.snapshot
                    )
                    assert len((await self.store.events("daily")).items) == 1
                finally:
                    async with self.store.transaction() as unit:
                        await unit.cursor.execute(
                            f"DROP TRIGGER injected_write_failure ON {table}"
                        )
        result = await self.admission.admit(request)
        assert result.outcome == TriggerCommitOutcome.COMMITTED
        assert len((await self.store.occurrences("daily")).items) == 1

    async def test_late_commit_and_lost_ack_use_completion_fence(
        self,
    ) -> None:
        async with self.store.transaction() as unit:
            first = await decision_time(unit)
            mutation = apply_registration(
                OWNER,
                replace(
                    registration(),
                    schedule=IntervalTrigger(
                        every_seconds=86400, start_at=first
                    ),
                ),
                None,
                expected_generation=None,
                decision_time=first,
                trigger_id="late-trigger",
            )
            await self.store._write(unit, mutation)
        plan = plan_admission(mutation.snapshot, datetime.now(UTC))
        prepared = replace(
            prepared_submission_fixture(
                self.queue,
                TaskExecutionRequest(definition_id="task"),
                queue_name="pgsql-e2e",
                available_at=first,
            ),
            owner_scope=OWNER.value,
            occurrence_id=plan.admission_ids[0],
            execution_deployment_id="deployment",
        )
        request = PreparedTriggerAdmission(plan=plan, submissions=(prepared,))
        written, release = Event(), Event()
        original = self.store.transaction
        delayed = False

        @asynccontextmanager
        async def delayed_ack() -> AsyncIterator[PgsqlUnitOfWork]:
            nonlocal delayed
            pause = not delayed
            delayed = True
            async with original() as unit:
                yield unit
                if pause:
                    written.set()
                    await release.wait()
            if pause:
                raise OSError("commit acknowledgement lost")

        other = PgsqlTriggerAdmissionStore(
            PgsqlTriggerStore(self.database, OWNER), self.queue
        )
        digest = sha256(
            dumps(
                [
                    "avalan.trigger",
                    OWNER.value,
                    "trigger:" + mutation.snapshot.state.trigger_id,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).digest()
        key = int.from_bytes(digest[:8], "big", signed=False)
        writer: Task[TriggerAdmissionResult] | None = None
        recovery: Task[TriggerAdmissionResult] | None = None
        try:
            with patch.object(self.store, "transaction", delayed_ack):
                writer = create_task(self.admission.admit(request))
                await wait_for(written.wait(), timeout=5)
                recovery = create_task(other.recover(plan))

                async def observe_waiter() -> None:
                    while True:
                        assert recovery is not None and not recovery.done()
                        async with self.database.connection() as connection:
                            async with connection.cursor() as cursor:
                                await cursor.execute(
                                    "SELECT EXISTS (SELECT 1 FROM pg_locks"
                                    " WHERE locktype='advisory' AND"
                                    " classid::bigint=%s AND objid::bigint=%s"
                                    " AND NOT granted) AS waiting",
                                    (key >> 32, key & 0xFFFFFFFF),
                                )
                                row = await cursor.fetchone()
                                if row is not None and row["waiting"] is True:
                                    return

                await wait_for(observe_waiter(), timeout=5)
                release.set()
                result = await wait_for(writer, timeout=5)
                recovered = await wait_for(recovery, timeout=5)
                assert isinstance(result, TriggerAdmissionResult)
                assert isinstance(recovered, TriggerAdmissionResult)
                assert (
                    result.outcome
                    == recovered.outcome
                    == TriggerCommitOutcome.COMMITTED
                )
                assert result.prepared is request
                occurrence = recovered.resolved[0].decisions[0]
                assert isinstance(occurrence, TriggerOccurrence)
                assert occurrence.run_id == prepared.run_id
        finally:
            release.set()
            for running in (writer, recovery):
                if running is not None and not running.done():
                    running.cancel()
                    with self.assertRaises(CancelledError):
                        await running

    async def test_sparse_reference_and_delete_acquisition_fence(
        self,
    ) -> None:
        ownership = PgsqlArtifactOwnership(self.database)
        run = await self.tasks.create_run(
            TaskExecutionRequest(definition_id="task")
        )
        sparse = TaskArtifactRef(
            artifact_id="sparse-before-object",
            store="object",
            storage_key="objects/ph/physical",
        )
        await self.tasks.append_artifact(
            run.run_id, ref=sparse, purpose=TaskArtifactPurpose.INPUT
        )
        value = ArtifactObject.from_ref(
            replace(sparse, sha256="a" * 64, size_bytes=4)
        )
        token = str(uuid4())
        staging = ArtifactStagingOwner(
            staging_id=token, owner_scope=OWNER.value, recovery_id=token
        )
        async with ownership.transaction() as unit:
            await ownership.stage(value, staging, unit_of_work=unit)
            await ownership.release_staging(
                staging,
                outcome=TaskSubmissionOutcome.NOT_COMMITTED,
                unit_of_work=unit,
            )
            assert (
                await ownership.claim_delete(
                    value.object_id, unit_of_work=unit, grace_seconds=0
                )
                is None
            )
        await self.tasks.transition_artifact(
            sparse.artifact_id,
            from_states={TaskArtifactState.READY},
            to_state=TaskArtifactState.DELETED,
            reason="retention fixture",
        )
        async with ownership.transaction() as unit:
            await ownership.release_run(sparse.artifact_id, unit_of_work=unit)
        pending = None
        try:
            async with ownership.transaction() as unit:
                deletion = await ownership.claim_delete(
                    value.object_id, unit_of_work=unit, grace_seconds=0
                )
                assert deletion is not None
                pending = create_task(
                    self.tasks.append_artifact(
                        run.run_id,
                        ref=replace(sparse, artifact_id="racing-reference"),
                        purpose=TaskArtifactPurpose.INPUT,
                    )
                )

                async def observe_object_waiter() -> None:
                    while True:
                        assert pending is not None and not pending.done()
                        async with self.database.connection() as connection:
                            async with connection.cursor() as cursor:
                                await cursor.execute(
                                    "SELECT EXISTS (SELECT 1 FROM"
                                    " pg_stat_activity WHERE"
                                    " wait_event_type='Lock' AND query LIKE"
                                    " '%FROM task_artifact_objects%') AS"
                                    " waiting"
                                )
                                row = await cursor.fetchone()
                                if row is not None and row["waiting"] is True:
                                    return

                await wait_for(observe_object_waiter(), timeout=5)
            with self.assertRaises(TaskStoreError):
                await wait_for(pending, timeout=5)
            async with self.store.transaction() as unit:
                await unit.cursor.execute(
                    "SELECT count(*) AS count FROM task_artifacts WHERE"
                    " artifact_id='racing-reference'"
                )
                assert await unit.cursor.fetchone() == {"count": 0}
            assert len(await ownership.pending_deletions()) == 1
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with self.assertRaises(CancelledError):
                    await pending

    async def test_existing_task_run_cannot_attach_to_new_occurrence(
        self,
    ) -> None:
        async with self.store.transaction() as unit:
            first = await decision_time(unit)
            mutation = apply_registration(
                OWNER,
                replace(
                    registration(),
                    schedule=IntervalTrigger(
                        every_seconds=86400, start_at=first
                    ),
                ),
                None,
                expected_generation=None,
                decision_time=first,
                trigger_id="existing-run-trigger",
            )
            await self.store._write(unit, mutation)
        plan = plan_admission(mutation.snapshot, datetime.now(UTC))
        prepared = replace(
            prepared_submission_fixture(
                self.queue,
                TaskExecutionRequest(definition_id="task"),
                queue_name="pgsql-e2e",
                available_at=first,
            ),
            owner_scope=OWNER.value,
            occurrence_id=plan.admission_ids[0],
            execution_deployment_id="deployment",
        )
        async with self.queue.submission_transaction() as unit:
            direct = await self.queue.submit_prepared(
                prepared, unit_of_work=unit
            )
        assert direct.created
        result = await self.admission.admit(
            PreparedTriggerAdmission(plan=plan, submissions=(prepared,))
        )
        assert result.outcome == TriggerCommitOutcome.NOT_COMMITTED
        assert result.error_code == TriggerErrorCode.CONFLICT
        assert await self.store.inspect("daily") == mutation.snapshot
        assert not (await self.store.occurrences("daily")).items
        assert (
            await self.tasks.get_run(prepared.run_id)
        ).run_id == direct.run.run_id

    async def test_pause_and_admission_observe_both_transaction_orders(
        self,
    ) -> None:
        database = self.database
        for winner, operation in (
            ("admission", "pause"),
            ("pause", "pause"),
            ("admission", "replacement"),
            ("pause", "replacement"),
        ):
            with self.subTest(winner=winner, operation=operation):
                name = "ordered-" + winner + "-" + operation
                async with self.store.transaction() as unit:
                    first = await decision_time(unit)
                    mutation = apply_registration(
                        OWNER,
                        replace(
                            registration(),
                            name=name,
                            schedule=IntervalTrigger(
                                every_seconds=86400, start_at=first
                            ),
                        ),
                        None,
                        expected_generation=None,
                        decision_time=first,
                        trigger_id=name,
                    )
                    await self.store._write(unit, mutation)
                plan = plan_admission(mutation.snapshot, datetime.now(UTC))
                task = replace(
                    prepared_submission_fixture(
                        self.queue,
                        TaskExecutionRequest(definition_id="task"),
                        queue_name="pgsql-e2e",
                        available_at=first,
                    ),
                    owner_scope=OWNER.value,
                    occurrence_id=plan.admission_ids[0],
                    execution_deployment_id="deployment",
                )
                request = PreparedTriggerAdmission(
                    plan=plan, submissions=(task,)
                )
                control = PgsqlTriggerStore(self.database, OWNER)
                held = self.store if winner == "admission" else control
                transaction = held.transaction
                entered, release = Event(), Event()
                first_transaction = True

                @asynccontextmanager
                async def hold_commit() -> AsyncIterator[PgsqlUnitOfWork]:
                    nonlocal first_transaction
                    pause = first_transaction
                    first_transaction = False
                    async with transaction() as unit:
                        yield unit
                        if pause:
                            entered.set()
                            await release.wait()

                async def control_change(generation: int) -> TriggerSnapshot:
                    if operation == "pause":
                        return await control.set_enabled(
                            name, enabled=False, expected_generation=generation
                        )
                    return await control.apply(
                        replace(
                            registration(),
                            name=name,
                            semantic_hash="2" * 64,
                            schedule=IntervalTrigger(every_seconds=43200),
                        ),
                        expected_generation=generation,
                    )

                async def commit_winner() -> (
                    TriggerAdmissionResult | TriggerSnapshot
                ):
                    if winner == "admission":
                        return await self.admission.admit(request)
                    return await control_change(
                        mutation.snapshot.state.generation
                    )

                waiting_pause = None
                with patch.object(held, "transaction", hold_commit):
                    running = create_task(commit_winner())
                    try:
                        await wait_for(entered.wait(), timeout=5)
                        if winner == "admission":
                            waiting_pause = create_task(
                                control_change(
                                    mutation.snapshot.state.generation
                                )
                            )

                            async def observe_control_waiter() -> None:
                                while True:
                                    assert (
                                        waiting_pause is not None
                                        and not waiting_pause.done()
                                    )
                                    async with (
                                        database.connection() as connection
                                    ):
                                        async with (
                                            connection.cursor() as cursor
                                        ):
                                            await cursor.execute(
                                                "SELECT EXISTS (SELECT 1 FROM"
                                                " pg_stat_activity WHERE"
                                                " wait_event_type='Lock' AND"
                                                " query LIKE '%FROM"
                                                " triggers%') AS waiting"
                                            )
                                            row = await cursor.fetchone()
                                            if (
                                                row is not None
                                                and row["waiting"] is True
                                            ):
                                                return

                            await wait_for(observe_control_waiter(), timeout=5)
                        else:
                            assert (
                                await self.admission.admit(request)
                            ).outcome == TriggerCommitOutcome.NOT_COMMITTED
                        release.set()
                        await wait_for(running, timeout=5)
                        if waiting_pause is not None:
                            with self.assertRaises(TriggerError) as stale:
                                await wait_for(waiting_pause, timeout=5)
                            assert (
                                stale.exception.code
                                == TriggerErrorCode.CONFLICT
                            )
                            fresh = await control.inspect(name)
                            assert fresh is not None
                            await control_change(fresh.state.generation)
                    finally:
                        release.set()
                        for pending in (running, waiting_pause):
                            if pending is not None and not pending.done():
                                pending.cancel()
                                with self.assertRaises(CancelledError):
                                    await pending
                current = await control.inspect(name)
                assert current is not None
                if operation == "pause":
                    assert current.state.status == TriggerStatus.PAUSED
                else:
                    assert (
                        current.definition.revision == 2
                        and current.state.status == TriggerStatus.ACTIVE
                    )
                    assert (
                        await control.revision(name, 1)
                        == mutation.snapshot.definition
                    )
                recovered = await self.admission.admit(request)
                expected = (
                    TriggerCommitOutcome.COMMITTED
                    if winner == "admission" or operation == "replacement"
                    else TriggerCommitOutcome.NOT_COMMITTED
                )
                assert recovered.outcome == expected
                if winner != "admission" and operation == "replacement":
                    assert recovered.resolved and all(
                        isinstance(decision, TriggerCoverageSpan)
                        for item in recovered.resolved
                        for decision in item.decisions
                    )
                    with self.assertRaises(TaskStoreNotFoundError):
                        await self.tasks.get_run(task.run_id)
                assert len((await control.occurrences(name)).items) == (
                    1 if winner == "admission" else 0
                )
                if winner == "pause" and operation == "pause":
                    assert (
                        current.state.next_at
                        == mutation.snapshot.state.next_at
                    )
