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
    IdempotencyMode,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskExecutionRequest,
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskQueueArtifact,
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
        self.definitions: dict[str, dict[str, object]] = {}
        self.runs: dict[str, dict[str, object]] = {}
        self.run_transitions: dict[str, dict[str, object]] = {}
        self.artifacts: dict[str, dict[str, object]] = {}
        self.idempotency: dict[str, dict[str, object]] = {}
        self.queue_items: dict[str, dict[str, object]] = {}
        self.depth_returns_none = False
        self.health_returns_none = False
        self.execute_error: BaseException | None = None
        self.fail_on_query: str | None = None
        self.executed_queries: list[str] = []
        self.race_on_idempotency_insert = False
        self.drop_idempotency_insert = False
        self.stale_transition = False
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
            "definitions": deepcopy(self.definitions),
            "runs": deepcopy(self.runs),
            "run_transitions": deepcopy(self.run_transitions),
            "artifacts": deepcopy(self.artifacts),
            "idempotency": deepcopy(self.idempotency),
            "queue_items": deepcopy(self.queue_items),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.definitions = cast(
            dict[str, dict[str, object]], snapshot["definitions"]
        )
        self.runs = cast(dict[str, dict[str, object]], snapshot["runs"])
        self.run_transitions = cast(
            dict[str, dict[str, object]],
            snapshot["run_transitions"],
        )
        self.artifacts = cast(
            dict[str, dict[str, object]],
            snapshot["artifacts"],
        )
        self.idempotency = cast(
            dict[str, dict[str, object]],
            snapshot["idempotency"],
        )
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
        self.database.executed_queries.append(query)
        if (
            self.database.fail_on_query is not None
            and self.database.fail_on_query in query
        ):
            raise RuntimeError("backend failure includes raw details")
        if 'SELECT * FROM "task_definitions"' in query:
            self.row = self.database.definitions.get(cast(str, params[0]))
        elif 'INSERT INTO "task_runs"' in query:
            self.row = self._insert_run(params)
        elif 'SELECT * FROM "task_runs"' in query:
            self.row = self.database.runs.get(cast(str, params[0]))
        elif 'UPDATE "task_runs"' in query:
            self.row = self._transition_run(params)
        elif 'INSERT INTO "task_run_transitions"' in query:
            self.row = self._insert_run_transition(params)
        elif 'SELECT * FROM "task_idempotency_keys"' in query:
            self.row = self.database.idempotency.get(cast(str, params[0]))
        elif 'INSERT INTO "task_idempotency_keys"' in query:
            self.row = self._insert_idempotency(params)
        elif 'INSERT INTO "task_artifacts"' in query:
            self.row = self._insert_artifact(params)
        elif 'SELECT "run_id", "state", "queue_name"' in query:
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

    def _insert_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run_id = cast(str, params[0])
        if run_id in self.database.runs:
            return None
        row = {
            "run_id": run_id,
            "definition_id": params[1],
            "state": params[2],
            "queue_name": params[3],
            "request": loads(cast(str, params[4])),
            "metadata": loads(cast(str, params[5])),
            "claim": None,
            "last_attempt_id": None,
            "result": None,
            "created_at": params[6],
            "updated_at": params[7],
        }
        self.database.runs[run_id] = row
        return row

    def _transition_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run_id = cast(str, params[3])
        row = self.database.runs.get(run_id)
        if self.database.stale_transition:
            self.database.stale_transition = False
            return None
        if row is None or row["state"] != params[4]:
            return None
        row.update(
            {
                "state": params[0],
                "result": (
                    loads(cast(str, params[1]))
                    if params[1] is not None
                    else row.get("result")
                ),
                "updated_at": params[2],
            }
        )
        return row

    def _insert_run_transition(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        transition_id = cast(str, params[0])
        if transition_id in self.database.run_transitions:
            return None
        row = {
            "transition_id": transition_id,
            "run_id": params[1],
            "from_state": params[2],
            "to_state": params[3],
            "reason": params[4],
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
        }
        self.database.run_transitions[transition_id] = row
        return row

    def _insert_idempotency(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        identity_key = cast(str, params[0])
        if self.database.race_on_idempotency_insert:
            self.database.race_on_idempotency_insert = False
            self.database.idempotency[identity_key] = {
                "identity_key": identity_key,
                "task_name": params[1],
                "task_version": params[2],
                "spec_hash": params[3],
                "owner_scope_hash": loads(cast(str, params[4])),
                "strategy": params[5],
                "window_hash": None,
                "input_hash": loads(cast(str, params[7])),
                "file_hash": None,
                "custom_hash": None,
                "run_id": "run-ready",
                "metadata": {},
                "expires_at": None,
                "created_at": params[13],
            }
            return None
        if self.database.drop_idempotency_insert:
            self.database.drop_idempotency_insert = False
            return None
        if identity_key in self.database.idempotency:
            return None
        row = {
            "identity_key": identity_key,
            "task_name": params[1],
            "task_version": params[2],
            "spec_hash": params[3],
            "owner_scope_hash": loads(cast(str, params[4])),
            "strategy": params[5],
            "window_hash": (
                loads(cast(str, params[6])) if params[6] is not None else None
            ),
            "input_hash": (
                loads(cast(str, params[7])) if params[7] is not None else None
            ),
            "file_hash": (
                loads(cast(str, params[8])) if params[8] is not None else None
            ),
            "custom_hash": (
                loads(cast(str, params[9])) if params[9] is not None else None
            ),
            "run_id": params[10],
            "metadata": loads(cast(str, params[11])),
            "expires_at": params[12],
            "created_at": params[13],
        }
        self.database.idempotency[identity_key] = row
        return row

    def _insert_artifact(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        artifact_id = cast(str, params[0])
        if artifact_id in self.database.artifacts:
            return None
        row = {
            "artifact_id": artifact_id,
            "run_id": params[1],
            "attempt_id": params[2],
            "purpose": params[3],
            "state": params[4],
            "ref": loads(cast(str, params[5])),
            "provenance": loads(cast(str, params[6])),
            "retention": loads(cast(str, params[7])),
            "metadata": loads(cast(str, params[8])),
            "created_at": params[9],
            "updated_at": params[10],
        }
        self.database.artifacts[artifact_id] = row
        return row

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
        self.database.definitions["hash-a"] = {"definition_id": "hash-a"}

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

    async def test_enqueue_run_persists_submission_atomically(self) -> None:
        identity = self._identity()
        artifact = TaskQueueArtifact(
            ref=TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="runs/run-1/input.txt",
                media_type="text/plain",
                size_bytes=4,
                sha256="a" * 64,
            ),
            purpose=TaskArtifactPurpose.INPUT,
            retention=TaskArtifactRetention(delete_after_days=2),
            metadata={"identity": {"digest": "<hmac-sha256>"}},
        )

        submission = await self.queue.enqueue_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                file_summaries=({"artifact_id": "artifact-1"},),
                queue="default",
            ),
            queue_name="default",
            priority=5,
            available_at=self.now,
            idempotency=identity,
            idempotency_expires_at=self.now + timedelta(days=1),
            artifacts=(artifact,),
            run_metadata={"source": "sdk"},
            queue_metadata={"tenant": "safe"},
        )

        self.assertTrue(submission.created)
        self.assertEqual(submission.run.run_id, "id-1")
        self.assertEqual(submission.run.state, TaskRunState.QUEUED)
        self.assertEqual(submission.run.request.queue, "default")
        self.assertIsNotNone(submission.queue_item)
        queue_item = submission.queue_item
        self.assertEqual(
            queue_item.queue_item_id if queue_item else "", "id-4"
        )
        self.assertEqual(queue_item.priority if queue_item else 0, 5)
        self.assertEqual(queue_item.metadata["tenant"], "safe")
        self.assertEqual(submission.idempotency.created, True)
        self.assertEqual(submission.artifacts[0].artifact_id, "artifact-1")
        self.assertEqual(
            [
                row["to_state"]
                for row in self.database.run_transitions.values()
            ],
            [TaskRunState.VALIDATED.value, TaskRunState.QUEUED.value],
        )
        stored_queue = self.database.queue_items["id-4"]
        self.assertNotIn("request", stored_queue)
        self.assertNotIn("result", stored_queue)
        self.assertLess(
            self._query_index('INSERT INTO "task_idempotency_keys"'),
            self._query_index('INSERT INTO "task_queue_items"'),
        )
        self.assertLess(
            self._query_index('INSERT INTO "task_artifacts"'),
            self._query_index('INSERT INTO "task_queue_items"'),
        )

    async def test_enqueue_run_returns_existing_idempotent_run(self) -> None:
        identity = self._identity()
        existing = await self.queue.enqueue_run(
            TaskExecutionRequest(definition_id="hash-a", queue="default"),
            queue_name="default",
            idempotency=identity,
        )

        duplicate = await self.queue.enqueue_run(
            TaskExecutionRequest(definition_id="hash-a", queue="default"),
            queue_name="default",
            idempotency=identity,
        )

        self.assertTrue(existing.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(duplicate.run.run_id, existing.run.run_id)
        self.assertIsNone(duplicate.queue_item)
        self.assertEqual(
            [
                run_id
                for run_id in self.database.runs
                if run_id.startswith("id-")
            ],
            [existing.run.run_id],
        )
        self.assertEqual(len(self.database.queue_items), 1)

    async def test_enqueue_run_rolls_back_failed_submission(self) -> None:
        self.database.fail_on_query = 'INSERT INTO "task_queue_items"'

        with self.assertRaises(TaskQueueError) as error:
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
                idempotency=self._identity(),
                artifacts=(
                    TaskQueueArtifact(
                        ref=TaskArtifactRef(
                            artifact_id="artifact-1",
                            store="local",
                            storage_key="runs/run-1/input.txt",
                        ),
                    ),
                ),
            )

        self.assertNotIn("raw details", str(error.exception))
        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(self.database.run_transitions, {})
        self.assertEqual(self.database.artifacts, {})
        self.assertEqual(self.database.idempotency, {})
        self.assertEqual(self.database.queue_items, {})

    async def test_enqueue_run_rolls_back_racing_idempotency_conflict(
        self,
    ) -> None:
        self.database.race_on_idempotency_insert = True

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
                idempotency=self._identity(),
            )

        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(self.database.run_transitions, {})
        self.assertEqual(self.database.idempotency, {})
        self.assertEqual(self.database.queue_items, {})

    async def test_enqueue_run_rejects_missing_idempotency_target(
        self,
    ) -> None:
        self.database.idempotency["identity-1"] = self._idempotency_row(
            run_id="missing"
        )

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
                idempotency=self._identity(),
            )

    async def test_enqueue_run_rolls_back_conflicted_run_id(self) -> None:
        self.database.runs["id-1"] = {
            "run_id": "id-1",
            "state": TaskRunState.CREATED.value,
            "queue_name": "default",
        }

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
            )

        self.assertEqual(self.database.run_transitions, {})
        self.assertEqual(self.database.queue_items, {})

    async def test_enqueue_run_rolls_back_conflicted_queue_item(self) -> None:
        self.database.queue_items["active"] = {
            "queue_item_id": "active",
            "run_id": "id-1",
            "queue_name": "default",
            "state": TaskQueueItemState.AVAILABLE.value,
        }

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
            )

        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(set(self.database.queue_items), {"active"})

    async def test_enqueue_run_rolls_back_idempotency_insert_gap(self) -> None:
        self.database.drop_idempotency_insert = True

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
                idempotency=self._identity(),
            )

        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(self.database.idempotency, {})

    async def test_enqueue_run_rolls_back_conflicted_artifact(self) -> None:
        self.database.artifacts["artifact-1"] = {"artifact_id": "artifact-1"}

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
                artifacts=(
                    TaskQueueArtifact(
                        ref=TaskArtifactRef(
                            artifact_id="artifact-1",
                            store="local",
                            storage_key="runs/run-1/input.txt",
                        ),
                    ),
                ),
            )

        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(set(self.database.artifacts), {"artifact-1"})

    async def test_enqueue_run_rolls_back_stale_transition(self) -> None:
        self.database.stale_transition = True

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
            )

        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(self.database.queue_items, {})

    async def test_enqueue_run_rolls_back_transition_insert_conflict(
        self,
    ) -> None:
        self.database.run_transitions["id-2"] = {"transition_id": "id-2"}

        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="default"),
                queue_name="default",
            )

        self.assertNotIn("id-1", self.database.runs)
        self.assertEqual(set(self.database.run_transitions), {"id-2"})

    async def test_enqueue_run_rejects_invalid_submission_inputs(self) -> None:
        with self.assertRaises(TaskQueueConflictError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a", queue="private"),
                queue_name="default",
            )
        with self.assertRaises(TaskQueueNotFoundError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="missing"),
                queue_name="default",
            )
        with self.assertRaises(AssertionError):
            TaskQueueArtifact(
                ref=TaskArtifactRef(
                    artifact_id="artifact-1",
                    store="local",
                    storage_key="runs/run-1/input.txt",
                ),
                metadata={"raw": object()},
            )
        with self.assertRaises(AssertionError):
            await self.queue.enqueue_run(
                TaskExecutionRequest(definition_id="hash-a"),
                queue_name="default",
                queue_metadata={"raw": object()},
            )

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

    def _identity(self) -> TaskIdempotencyIdentity:
        digest = TaskIdempotencyDigest(
            algorithm="hmac-sha256",
            digest="a" * 64,
            key_id="test-key",
        )
        return TaskIdempotencyIdentity(
            identity_key="identity-1",
            task_name="summarize",
            task_version="1",
            spec_hash="hash-a",
            owner_scope=digest,
            strategy=IdempotencyMode.INPUT_HASH,
            input=digest,
        )

    def _idempotency_row(self, *, run_id: str) -> dict[str, object]:
        identity = self._identity()
        digest = identity.owner_scope.as_dict()
        return {
            "identity_key": identity.identity_key,
            "task_name": identity.task_name,
            "task_version": identity.task_version,
            "spec_hash": identity.spec_hash,
            "owner_scope_hash": digest,
            "strategy": identity.strategy.value,
            "window_hash": None,
            "input_hash": digest,
            "file_hash": None,
            "custom_hash": None,
            "run_id": run_id,
            "metadata": {},
            "expires_at": None,
            "created_at": self.now,
        }

    def _query_index(self, fragment: str) -> int:
        for index, query in enumerate(self.database.executed_queries):
            if fragment in query:
                return index
        raise AssertionError(f"query fragment was not executed: {fragment}")


if __name__ == "__main__":
    main()
