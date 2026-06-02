from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main

from pgsql_harness import (
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)
from pytest import importorskip

from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task import (
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskQueueItemState,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
)
from avalan.task.queues import PgsqlTaskQueue
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)


class RecordingTarget(TaskTargetRunner):
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.inputs: list[object] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.inputs.append(context.input_value)
        await context.check_cancelled()
        if self.failures:
            self.failures -= 1
            raise OSError("private backend path")
        await context.observe_usage(
            SimpleNamespace(
                input_token_count=3,
                output_token_count=5,
                total_token_count=8,
            )
        )
        return "safe output"


class PgsqlQueueWorkerE2ETest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("psycopg")
        importorskip("psycopg_pool")
        importorskip("sqlalchemy")
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("avalan_task_queue_e2e")
        task_pgsql_upgrade(
            PgsqlTaskMigrationSettings(url=dsn, schema=self.schema)
        )
        self.database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=task_pgsql_psycopg_dsn(dsn),
                schema=self.schema,
                application_name="avalan-task-queue-e2e",
            )
        )
        self.store = PgsqlTaskStore(self.database)
        self.queue = PgsqlTaskQueue(self.database)
        await self.database.open()

    async def asyncTearDown(self) -> None:
        database = getattr(self, "database", None)
        if database is not None:
            await database.aclose()
        dsn = getattr(self, "dsn", None)
        schema = getattr(self, "schema", None)
        if dsn is not None and schema is not None:
            await _drop_schema(dsn, schema)

    async def test_worker_completes_pgsql_queue_run(self) -> None:
        definition_hash = "queue-e2e-success"
        await self.store.register_definition(
            _definition(),
            definition_hash=definition_hash,
        )
        submission = await self.queue.enqueue_run(
            TaskExecutionRequest(
                definition_id=definition_hash,
                input_summary="safe input",
                queue="pgsql-e2e",
                metadata={"request": "safe"},
            ),
            queue_name="pgsql-e2e",
            priority=7,
            queue_metadata={"source": "test"},
        )
        target = RecordingTarget()
        worker = TaskWorker(
            self.store,
            self.queue,
            target=target,
            worker_id="pgsql-e2e-worker",
            queue_name="pgsql-e2e",
        )

        result = await worker.process_once()
        idle = await worker.process_once()

        self.assertTrue(submission.created)
        self.assertIsNotNone(submission.queue_item)
        assert submission.queue_item is not None
        self.assertEqual(
            submission.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        self.assertFalse(idle.processed)
        assert result.completion is not None
        run = await self.store.get_run(submission.run.run_id)
        attempts = await self.store.list_attempts(run.run_id)
        usage = await self.store.list_usage(run.run_id)
        depth = await self.queue.depth("pgsql-e2e")

        self.assertEqual(run.state, TaskRunState.SUCCEEDED)
        assert run.result is not None
        self.assertEqual(run.result.output_summary, {"privacy": "<redacted>"})
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.SUCCEEDED)
        self.assertEqual(target.inputs, ["safe input"])
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].totals.input_tokens, 3)
        self.assertEqual(usage[0].totals.output_tokens, 5)
        self.assertEqual(depth.active, 0)
        self.assertEqual(depth.dead, 0)

    async def test_worker_retries_pgsql_queue_run(self) -> None:
        definition_hash = "queue-e2e-retry"
        await self.store.register_definition(
            _definition(max_attempts=2),
            definition_hash=definition_hash,
        )
        submission = await self.queue.enqueue_run(
            TaskExecutionRequest(
                definition_id=definition_hash,
                input_summary="retry input",
                queue="pgsql-e2e",
            ),
            queue_name="pgsql-e2e",
        )
        target = RecordingTarget(failures=1)
        worker = TaskWorker(
            self.store,
            self.queue,
            target=target,
            worker_id="pgsql-e2e-worker",
            queue_name="pgsql-e2e",
        )

        retry = await worker.process_once()
        completion = await worker.process_once()

        self.assertTrue(retry.processed)
        self.assertIsNotNone(retry.retry)
        assert retry.retry is not None
        self.assertTrue(retry.retry.retryable)
        self.assertEqual(retry.retry.run.state, TaskRunState.QUEUED)
        self.assertEqual(
            retry.retry.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertTrue(completion.processed)
        self.assertIsNotNone(completion.completion)
        run = await self.store.get_run(submission.run.run_id)
        attempts = await self.store.list_attempts(run.run_id)
        usage = await self.store.list_usage(run.run_id)

        self.assertEqual(run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            [attempt.state for attempt in attempts],
            [
                TaskAttemptState.FAILED,
                TaskAttemptState.SUCCEEDED,
            ],
        )
        assert attempts[0].result is not None
        self.assertNotIn("private", str(attempts[0].result.error))
        self.assertEqual(target.inputs, ["retry input", "retry input"])
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].totals.total_tokens, 8)


def _definition(*, max_attempts: int = 1) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="pgsql_queue_e2e", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("pgsql-e2e"),
        retry=TaskRetryPolicy(max_attempts=max_attempts),
    )


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=task_pgsql_psycopg_dsn(dsn))
    )
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


if __name__ == "__main__":
    main()
