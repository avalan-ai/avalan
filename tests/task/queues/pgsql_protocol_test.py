from asyncio import CancelledError
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from json import loads
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.pgsql import (
    PgsqlFailure,
    PgsqlFailureCategory,
    PgsqlOperationError,
)
from avalan.task import (
    TaskQueueConflictError,
    TaskQueueError,
    TaskQueueItemState,
    TaskQueueNotFoundError,
    TaskRunState,
)
from avalan.task.queues import PgsqlTaskQueue
from avalan.task.queues.pgsql import _json


class SequenceClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self._next
        self._next = self._next + timedelta(seconds=1)
        return value


class SequenceIds:
    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> str:
        value = f"id-{self._next}"
        self._next = self._next + 1
        return value


class FakePgsqlQueueDatabase:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, object]] = {}
        self.queue_items: dict[str, dict[str, object]] = {}
        self.depth_returns_none = False
        self.health_returns_none = False
        self.execute_error: BaseException | None = None
        self.open_count = 0
        self.close_count = 0

    def connection(self) -> "FakeConnectionContext":
        return FakeConnectionContext(self)

    async def open(self) -> None:
        self.open_count += 1

    async def aclose(self) -> None:
        self.close_count += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "runs": deepcopy(self.runs),
            "queue_items": deepcopy(self.queue_items),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.runs = cast(dict[str, dict[str, object]], snapshot["runs"])
        self.queue_items = cast(
            dict[str, dict[str, object]],
            snapshot["queue_items"],
        )


class FakeConnectionContext:
    def __init__(self, database: FakePgsqlQueueDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeConnection":
        return FakeConnection(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeConnection:
    row_factory: object = None

    def __init__(self, database: FakePgsqlQueueDatabase) -> None:
        self.database = database

    def cursor(self) -> "FakeCursorContext":
        return FakeCursorContext(self.database)

    def transaction(self) -> "FakeTransactionContext":
        return FakeTransactionContext(self.database)

    async def set_autocommit(self, value: bool) -> None:
        assert isinstance(value, bool)


class FakeTransactionContext:
    def __init__(self, database: FakePgsqlQueueDatabase) -> None:
        self.database = database
        self._snapshot: dict[str, object] = {}

    async def __aenter__(self) -> "FakeTransactionContext":
        self._snapshot = self.database.snapshot()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        if exc_type is not None:
            self.database.restore(self._snapshot)
        return False


class FakeCursorContext:
    def __init__(self, database: FakePgsqlQueueDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeCursor":
        return FakeCursor(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeCursor:
    def __init__(self, database: FakePgsqlQueueDatabase) -> None:
        self.database = database
        self.row: dict[str, object] | None = None
        self.rows: tuple[dict[str, object], ...] = ()

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        params = parameters or ()
        if self.database.execute_error is not None:
            raise self.database.execute_error
        if 'SELECT "run_id", "state", "queue_name"' in query:
            self.row = self.database.runs.get(cast(str, params[0]))
        elif 'INSERT INTO "task_queue_items"' in query:
            self.row = self._insert_queue_item(params)
        elif 'MIN(q."available_at")' in query:
            self.row = self._health(params)
        elif "COUNT(*) FILTER" in query:
            self.row = self._depth(params)
        elif 'SELECT q.*, r."state" AS "run_state"' in query:
            self.rows = self._drain(params)
        else:
            raise AssertionError(f"unexpected query: {query}")

    async def fetchone(self) -> dict[str, object] | None:
        return self.row

    async def fetchall(self) -> tuple[dict[str, object], ...]:
        return self.rows

    async def close(self) -> None:
        return None

    def _insert_queue_item(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run_id = cast(str, params[1])
        for row in self.database.queue_items.values():
            if row["run_id"] == run_id and row["state"] in {
                "available",
                "claimed",
            }:
                return None
        row = {
            "queue_item_id": params[0],
            "run_id": run_id,
            "queue_name": params[2],
            "state": params[3],
            "priority": params[4],
            "available_at": params[5],
            "claimed_at": None,
            "lease_expires_at": None,
            "worker_id": None,
            "claim_token": None,
            "heartbeat_at": None,
            "attempts": params[6],
            "metadata": loads(cast(str, params[7])),
            "created_at": params[8],
            "updated_at": params[9],
        }
        self.database.queue_items[cast(str, params[0])] = row
        return row

    def _drain(
        self,
        params: tuple[object, ...],
    ) -> tuple[dict[str, object], ...]:
        queue_name = cast(str, params[0])
        checked_at = cast(datetime, params[1])
        limit = cast(int, params[2])
        rows = (
            self._with_run_state(row)
            for row in self.database.queue_items.values()
            if row["queue_name"] == queue_name
            and row["state"] == "available"
            and cast(datetime, row["available_at"]) <= checked_at
        )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    -cast(int, row["priority"]),
                    cast(datetime, row["available_at"]),
                    cast(str, row["queue_item_id"]),
                ),
            )[:limit]
        )

    def _depth(self, params: tuple[object, ...]) -> dict[str, object] | None:
        if self.database.depth_returns_none:
            return None
        checked_at = cast(datetime, params[0])
        queue_name = cast(str, params[2])
        rows = tuple(
            self._with_run_state(row)
            for row in self.database.queue_items.values()
            if row["queue_name"] == queue_name
        )
        return {
            "available": sum(
                1
                for row in rows
                if row["state"] == "available"
                and cast(datetime, row["available_at"]) <= checked_at
            ),
            "scheduled": sum(
                1
                for row in rows
                if row["state"] == "available"
                and cast(datetime, row["available_at"]) > checked_at
            ),
            "claimed": sum(1 for row in rows if row["state"] == "claimed"),
            "dead": sum(1 for row in rows if row["state"] == "dead"),
            "cancel_requested": sum(
                1
                for row in rows
                if row["state"] in {"available", "claimed"}
                and row["run_state"] == "cancel_requested"
            ),
        }

    def _health(self, params: tuple[object, ...]) -> dict[str, object] | None:
        if self.database.health_returns_none:
            return None
        checked_at = cast(datetime, params[0])
        queue_name = cast(str, params[2])
        rows = tuple(
            row
            for row in self.database.queue_items.values()
            if row["queue_name"] == queue_name
        )
        available_times = tuple(
            cast(datetime, row["available_at"])
            for row in rows
            if row["state"] == "available"
            and cast(datetime, row["available_at"]) <= checked_at
        )
        return {
            "oldest_available_at": (
                min(available_times) if available_times else None
            ),
            "expired_claims": sum(
                1
                for row in rows
                if row["state"] == "claimed"
                and cast(datetime, row["lease_expires_at"]) <= checked_at
            ),
        }

    def _with_run_state(
        self,
        row: dict[str, object],
    ) -> dict[str, object]:
        run = self.database.runs[cast(str, row["run_id"])]
        return {**row, "run_state": run["state"]}


class SyncCloseQueueDatabase:
    def __init__(self) -> None:
        self.close_count = 0

    def connection(self) -> object:
        raise AssertionError("connection should not be used")

    def close(self) -> None:
        self.close_count += 1


class UniquePgsqlError(RuntimeError):
    sqlstate = "23505"


class PgsqlTaskQueueTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = SequenceClock()
        self.database = FakePgsqlQueueDatabase()
        self.queue = PgsqlTaskQueue(
            self.database,
            clock=self.clock,
            id_factory=SequenceIds(),
        )
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.database.runs.update(
            {
                "run-ready": {
                    "run_id": "run-ready",
                    "state": TaskRunState.VALIDATED.value,
                    "queue_name": "default",
                },
                "run-queued": {
                    "run_id": "run-queued",
                    "state": TaskRunState.QUEUED.value,
                    "queue_name": "default",
                },
                "run-cancel": {
                    "run_id": "run-cancel",
                    "state": TaskRunState.CANCEL_REQUESTED.value,
                    "queue_name": "default",
                },
                "run-created": {
                    "run_id": "run-created",
                    "state": TaskRunState.CREATED.value,
                    "queue_name": "default",
                },
                "run-other-queue": {
                    "run_id": "run-other-queue",
                    "state": TaskRunState.QUEUED.value,
                    "queue_name": "private",
                },
            }
        )

    async def test_context_manager_opens_and_closes_database(self) -> None:
        async with self.queue as opened:
            self.assertIs(opened, self.queue)

        self.assertEqual(self.database.open_count, 1)
        self.assertEqual(self.database.close_count, 1)

    async def test_open_and_close_support_minimal_database_objects(
        self,
    ) -> None:
        database = SyncCloseQueueDatabase()
        queue = PgsqlTaskQueue(cast(Any, database))

        await queue.open()
        await queue.aclose()

        self.assertEqual(database.close_count, 1)

    async def test_default_factories_generate_ids_and_timestamps(
        self,
    ) -> None:
        queue = PgsqlTaskQueue(self.database)

        item = await queue.enqueue("run-ready", queue_name="default")

        self.assertTrue(item.queue_item_id)
        self.assertIsNotNone(item.available_at.tzinfo)

    async def test_enqueue_stores_only_safe_scheduling_metadata(self) -> None:
        item = await self.queue.enqueue(
            "run-ready",
            queue_name="default",
            priority=4,
            available_at=self.now,
            metadata={"source": "sdk", "labels": ["safe"]},
        )

        self.assertEqual(item.queue_item_id, "id-1")
        self.assertEqual(item.run_id, "run-ready")
        self.assertEqual(item.state, TaskQueueItemState.AVAILABLE)
        self.assertEqual(item.priority, 4)
        self.assertEqual(item.metadata["labels"], ("safe",))
        stored = self.database.queue_items["id-1"]
        self.assertEqual(
            set(stored),
            {
                "queue_item_id",
                "run_id",
                "queue_name",
                "state",
                "priority",
                "available_at",
                "claimed_at",
                "lease_expires_at",
                "worker_id",
                "claim_token",
                "heartbeat_at",
                "attempts",
                "metadata",
                "created_at",
                "updated_at",
            },
        )
        self.assertNotIn("request", stored)
        self.assertNotIn("result", stored)

    async def test_enqueue_rejects_missing_duplicate_and_unready_runs(
        self,
    ) -> None:
        await self.queue.enqueue("run-ready", queue_name="default")

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue("run-ready", queue_name="default")
        with self.assertRaises(TaskQueueNotFoundError):
            await self.queue.enqueue("missing", queue_name="default")
        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue("run-created", queue_name="default")
        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue("run-other-queue", queue_name="default")
        with self.assertRaises(TaskQueueConflictError) as cancel:
            await self.queue.enqueue("run-cancel", queue_name="default")

        self.assertNotIn("run-cancel", str(cancel.exception))
        self.assertEqual(tuple(self.database.queue_items), ("id-1",))
        with self.assertRaises(AssertionError):
            await self.queue.enqueue(
                "run-queued",
                queue_name="default",
                metadata={"raw": object()},
            )

    async def test_drain_orders_ready_items_and_surfaces_cancellation(
        self,
    ) -> None:
        await self.queue.enqueue(
            "run-ready",
            queue_name="default",
            priority=1,
            available_at=self.now,
        )
        self._add_queue_item(
            "manual-cancel",
            run_id="run-cancel",
            priority=10,
            available_at=self.now,
        )
        self._add_queue_item(
            "manual-future",
            run_id="run-queued",
            priority=100,
            available_at=self.now + timedelta(minutes=1),
        )
        self._add_queue_item(
            "manual-none-metadata",
            run_id="run-queued",
            priority=-1,
            available_at=self.now,
            metadata=None,
        )
        self._add_queue_item(
            "manual-string-metadata",
            run_id="run-queued",
            priority=-2,
            available_at=self.now,
            metadata='{"kind":"safe"}',
        )

        drained = await self.queue.drain(
            "default",
            limit=10,
            now=datetime(2026, 1, 1),
        )

        self.assertEqual(
            [item.queue_item_id for item in drained],
            [
                "manual-cancel",
                "id-1",
                "manual-none-metadata",
                "manual-string-metadata",
            ],
        )
        self.assertTrue(drained[0].cancel_requested)
        self.assertFalse(drained[1].cancel_requested)
        self.assertEqual(drained[2].metadata, {})
        self.assertEqual(drained[3].metadata["kind"], "safe")

    async def test_depth_and_health_expose_queue_state(self) -> None:
        await self.queue.enqueue(
            "run-ready",
            queue_name="default",
            available_at=self.now,
        )
        self._add_queue_item(
            "manual-future",
            run_id="run-queued",
            available_at=self.now + timedelta(minutes=1),
        )
        self._add_queue_item(
            "manual-cancel",
            run_id="run-cancel",
            available_at=self.now,
        )
        self._add_queue_item(
            "manual-claimed",
            run_id="run-queued",
            state=TaskQueueItemState.CLAIMED,
            available_at=self.now,
            lease_expires_at=self.now - timedelta(seconds=1),
        )
        self._add_queue_item(
            "manual-dead",
            run_id="run-queued",
            state=TaskQueueItemState.DEAD,
            available_at=self.now,
        )

        depth = await self.queue.depth("default", now=self.now)
        health = await self.queue.health("default", now=self.now)

        self.assertEqual(depth.available, 2)
        self.assertEqual(depth.scheduled, 1)
        self.assertEqual(depth.claimed, 1)
        self.assertEqual(depth.dead, 1)
        self.assertEqual(depth.cancel_requested, 1)
        self.assertEqual(health.oldest_available_at, self.now)
        self.assertEqual(health.expired_claims, 1)

    async def test_depth_and_health_handle_empty_aggregate_rows(self) -> None:
        self.database.depth_returns_none = True
        self.database.health_returns_none = True

        depth = await self.queue.depth("default", now=self.now)
        health = await self.queue.health("default", now=self.now)

        self.assertEqual(depth.active, 0)
        self.assertIsNone(health.oldest_available_at)
        self.assertEqual(health.expired_claims, 0)

    async def test_database_failures_are_sanitized(self) -> None:
        self.database.execute_error = AssertionError("bad invariant")
        with self.assertRaises(AssertionError):
            await self.queue.depth("default")

        self.database.execute_error = CancelledError()
        with self.assertRaises(CancelledError):
            await self.queue.depth("default")

        self.database.execute_error = PgsqlOperationError(
            PgsqlFailure(
                category=PgsqlFailureCategory.UNIQUE_CONFLICT,
                code="23505",
                retryable=False,
            )
        )
        with self.assertRaises(TaskQueueConflictError):
            await self.queue.depth("default")

        self.database.execute_error = PgsqlOperationError(
            PgsqlFailure(
                category=PgsqlFailureCategory.TIMEOUT,
                code="57014",
                retryable=True,
            )
        )
        with self.assertRaises(TaskQueueError):
            await self.queue.depth("default")

        self.database.execute_error = UniquePgsqlError("raw secret")
        with self.assertRaises(TaskQueueConflictError) as conflict:
            await self.queue.depth("default")

        self.assertNotIn("raw secret", str(conflict.exception))

        self.database.execute_error = RuntimeError("raw secret")
        with self.assertRaises(TaskQueueError) as error:
            await self.queue.depth("default")

        self.assertNotIn("raw secret", str(error.exception))

    def test_queue_metadata_json_rejects_unknown_values(self) -> None:
        with self.assertRaises(AssertionError):
            _json({"raw": object()})

    def _add_queue_item(
        self,
        queue_item_id: str,
        *,
        run_id: str,
        state: TaskQueueItemState = TaskQueueItemState.AVAILABLE,
        priority: int = 0,
        available_at: datetime,
        lease_expires_at: datetime | None = None,
        metadata: object = None,
    ) -> None:
        claimed_at = self.now - timedelta(minutes=1)
        claimed_fields: dict[str, object | None]
        if state == TaskQueueItemState.CLAIMED:
            claimed_fields = {
                "claimed_at": claimed_at,
                "lease_expires_at": (
                    lease_expires_at or self.now + timedelta(minutes=5)
                ),
                "worker_id": "worker-1",
                "claim_token": f"{queue_item_id}-token",
                "heartbeat_at": claimed_at,
            }
        else:
            claimed_fields = {
                "claimed_at": None,
                "lease_expires_at": None,
                "worker_id": None,
                "claim_token": None,
                "heartbeat_at": None,
            }
        self.database.queue_items[queue_item_id] = {
            "queue_item_id": queue_item_id,
            "run_id": run_id,
            "queue_name": "default",
            "state": state.value,
            "priority": priority,
            "available_at": available_at,
            "attempts": 0,
            "metadata": metadata,
            "created_at": self.now,
            "updated_at": self.now,
            **claimed_fields,
        }


if __name__ == "__main__":
    main()
