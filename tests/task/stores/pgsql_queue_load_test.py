from asyncio import Lock, gather
from datetime import UTC, datetime, timedelta
from os import environ
from time import monotonic
from unittest import IsolatedAsyncioTestCase, main

from pgsql_harness import (
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    task_pgsql_psycopg_dsn,
)
from pytest import importorskip

from avalan.pgsql import PsycopgAsyncDatabase, PsycopgPoolSettings
from avalan.task import (
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskQueueClaim,
    TaskQueueConflictError,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
)
from avalan.task.queues import PgsqlTaskQueue
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    TaskPgsqlQueueLoadProfile,
    task_pgsql_queue_load_issues,
    task_pgsql_queue_load_metadata,
    task_pgsql_upgrade,
)


class PgsqlQueueLoadTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = environ.get("AVALAN_TASK_LOAD_POSTGRESQL_DSN")
        if not dsn:
            self.skipTest("AVALAN_TASK_LOAD_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("psycopg")
        importorskip("psycopg_pool")
        importorskip("sqlalchemy")
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("avalan_task_queue_load")
        self.profile = await self._profile()
        task_pgsql_upgrade(
            PgsqlTaskMigrationSettings(url=dsn, schema=self.schema)
        )
        self.database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=task_pgsql_psycopg_dsn(dsn),
                pool_minimum=1,
                pool_maximum=self.profile.pool_size,
                schema=self.schema,
                application_name="avalan-task-queue-load",
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
            await drop_task_pgsql_schema(dsn, schema)

    async def test_claim_load_profile_preserves_lease_invariants(self) -> None:
        queue_names = tuple(
            f"pgsql-load-{index}" for index in range(self.profile.queue_count)
        )
        definition_hash = "pgsql-load-definition"
        await self.store.register_definition(
            _definition(
                queue=queue_names[0],
                max_attempts=self.profile.max_attempts,
            ),
            definition_hash=definition_hash,
        )
        for index in range(self.profile.run_count):
            queue_name = queue_names[index % len(queue_names)]
            await self.queue.enqueue_run(
                TaskExecutionRequest(
                    definition_id=definition_hash,
                    input_summary={"run": index},
                    queue=queue_name,
                    metadata={"source": "load"},
                ),
                queue_name=queue_name,
                priority=index % 3,
            )

        depths_before = tuple(
            await self.queue.depth(queue_name) for queue_name in queue_names
        )
        health_before = tuple(
            await self.queue.health(queue_name) for queue_name in queue_names
        )
        started_at = monotonic()
        claimed = await _claim_all(
            self.queue,
            queue_names=queue_names,
            profile=self.profile,
        )
        elapsed_seconds = monotonic() - started_at
        claimed_depths = tuple(
            await self.queue.depth(queue_name) for queue_name in queue_names
        )
        claimed_health = tuple(
            await self.queue.health(queue_name) for queue_name in queue_names
        )
        stale_token_commits = await _stale_token_commits(self.queue, claimed)
        attempt_count = 0
        for claim in claimed:
            attempt_count += len(
                await self.store.list_attempts(claim.run.run_id)
            )
        reaper_now = max(
            claim.queue_item.lease_expires_at
            for claim in claimed
            if claim.queue_item.lease_expires_at is not None
        ) + timedelta(microseconds=1)
        reaped_counts = []
        for queue_name in queue_names:
            abandoned = await self.queue.abandon_expired(
                queue_name,
                max_attempts=self.profile.max_attempts,
                limit=self.profile.abandon_limit,
                now=reaper_now,
                metadata={"source": "load-reaper"},
            )
            reaped_counts.append(len(abandoned))

        issues = task_pgsql_queue_load_issues(
            self.profile,
            claimed_run_ids=tuple(claim.run.run_id for claim in claimed),
            attempt_count=attempt_count,
            stale_token_commits=stale_token_commits,
            reaped_claims=max(reaped_counts, default=0),
            elapsed_seconds=max(elapsed_seconds, 0.000001),
        )
        metadata = task_pgsql_queue_load_metadata(
            self.profile,
            elapsed_seconds=max(elapsed_seconds, 0.000001),
        )

        self.assertEqual(issues, ())
        self.assertEqual(
            sum(depth.active for depth in depths_before),
            self.profile.run_count,
        )
        self.assertEqual(
            sum(
                1
                for depth, health in zip(depths_before, health_before)
                if depth.active and health.oldest_available_at is not None
            ),
            sum(1 for depth in depths_before if depth.active),
        )
        self.assertEqual(
            sum(depth.claimed for depth in claimed_depths),
            self.profile.run_count,
        )
        self.assertEqual(
            sum(health.expired_claims for health in claimed_health),
            0,
        )
        self.assertLessEqual(
            max(reaped_counts, default=0),
            self.profile.abandon_limit,
        )
        self.assertEqual(metadata["run_count"], self.profile.run_count)
        for claim in claimed:
            self.assertEqual(claim.run.state, TaskRunState.CLAIMED)
            self.assertIsNotNone(claim.queue_item.claim_token)

    async def _profile(self) -> TaskPgsqlQueueLoadProfile:
        postgresql_version = await _postgresql_version(self.dsn)
        run_count = _int_env("AVALAN_TASK_LOAD_RUNS", 16)
        return TaskPgsqlQueueLoadProfile(
            worker_count=_int_env("AVALAN_TASK_LOAD_WORKERS", 4),
            run_count=run_count,
            queue_count=_int_env("AVALAN_TASK_LOAD_QUEUES", 2),
            lease_seconds=_int_env("AVALAN_TASK_LOAD_LEASE_SECONDS", 30),
            max_attempts=_int_env("AVALAN_TASK_LOAD_MAX_ATTEMPTS", 2),
            abandon_limit=_int_env(
                "AVALAN_TASK_LOAD_ABANDON_LIMIT",
                run_count,
            ),
            pool_size=_int_env("AVALAN_TASK_LOAD_POOL_SIZE", 6),
            postgresql_version=postgresql_version,
            retry_delay_seconds=_int_env(
                "AVALAN_TASK_LOAD_RETRY_DELAY_SECONDS",
                0,
            ),
            min_claims_per_second=_float_env(
                "AVALAN_TASK_LOAD_MIN_CLAIMS_PER_SECOND"
            ),
        )


async def _claim_all(
    queue: PgsqlTaskQueue,
    *,
    queue_names: tuple[str, ...],
    profile: TaskPgsqlQueueLoadProfile,
) -> tuple[TaskQueueClaim, ...]:
    claimed: list[TaskQueueClaim] = []
    lock = Lock()

    async def worker(worker_index: int) -> None:
        worker_id = f"load-worker-{worker_index}"
        while True:
            async with lock:
                if len(claimed) >= profile.run_count:
                    return
            made_progress = False
            for queue_name in queue_names:
                now = datetime.now(UTC)
                claim = await queue.claim(
                    queue_name,
                    worker_id=worker_id,
                    lease_expires_at=(
                        now + timedelta(seconds=profile.lease_seconds)
                    ),
                    now=now,
                    metadata={"worker_id": worker_id},
                )
                if claim is None:
                    continue
                made_progress = True
                async with lock:
                    claimed.append(claim)
                    if len(claimed) >= profile.run_count:
                        return
            if not made_progress:
                return

    await gather(*(worker(index) for index in range(profile.worker_count)))
    return tuple(claimed)


async def _stale_token_commits(
    queue: PgsqlTaskQueue,
    claimed: tuple[TaskQueueClaim, ...],
) -> int:
    if not claimed:
        return 0
    claim = claimed[0]
    try:
        await queue.complete(
            claim.queue_item.queue_item_id,
            claim_token="stale-token",
            run_state=TaskRunState.SUCCEEDED,
            attempt_state=TaskAttemptState.SUCCEEDED,
            now=datetime.now(UTC),
        )
    except TaskQueueConflictError:
        return 0
    return 1


async def _postgresql_version(dsn: str) -> str:
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=task_pgsql_psycopg_dsn(dsn))
    )
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SHOW server_version")
                row = await cursor.fetchone()
    assert row is not None
    return str(row["server_version"])


def _definition(*, queue: str, max_attempts: int) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="pgsql_queue_load", version="1"),
        input=TaskInputContract.object(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued(queue),
        retry=TaskRetryPolicy(max_attempts=max_attempts),
    )


def _int_env(name: str, default: int) -> int:
    return int(environ.get(name, str(default)))


def _float_env(name: str) -> float | None:
    value = environ.get(name)
    return float(value) if value else None


if __name__ == "__main__":
    main()
