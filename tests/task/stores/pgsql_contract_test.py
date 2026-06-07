from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from json import loads
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main

sys_path.append(str(Path(__file__).parents[1]))

from store_contract_test import (  # type: ignore[import-not-found]
    SequenceClock,
    SequenceIds,
    StoreContractAssertions,
)

from avalan.task import (
    IdempotencyMode,
    TaskAttemptState,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskRunState,
    TaskStoreConflictError,
    UsageProviderFamily,
    UsageSource,
    UsageTotals,
    stable_usage_id,
)
from avalan.task.stores import PgsqlTaskStore


class FakePgsqlTaskDatabase:
    def __init__(self) -> None:
        self.definitions: dict[str, dict[str, object]] = {}
        self.runs: dict[str, dict[str, object]] = {}
        self.flow_executions: dict[str, dict[str, object]] = {}
        self.attempts: dict[str, dict[str, object]] = {}
        self.run_transitions: dict[str, dict[str, object]] = {}
        self.attempt_transitions: dict[str, dict[str, object]] = {}
        self.events: dict[str, dict[str, object]] = {}
        self.usage: dict[str, dict[str, object]] = {}
        self.artifacts: dict[str, dict[str, object]] = {}
        self.idempotency: dict[str, dict[str, object]] = {}
        self.executed_queries: list[str] = []
        self.before_attempt_update: (
            Callable[[tuple[object, ...]], None] | None
        ) = None
        self.before_last_attempt_update: (
            Callable[[tuple[object, ...]], None] | None
        ) = None
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
            "flow_executions": deepcopy(self.flow_executions),
            "attempts": deepcopy(self.attempts),
            "run_transitions": deepcopy(self.run_transitions),
            "attempt_transitions": deepcopy(self.attempt_transitions),
            "events": deepcopy(self.events),
            "usage": deepcopy(self.usage),
            "artifacts": deepcopy(self.artifacts),
            "idempotency": deepcopy(self.idempotency),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.definitions = cast(
            dict[str, dict[str, object]], snapshot["definitions"]
        )
        self.runs = cast(dict[str, dict[str, object]], snapshot["runs"])
        self.flow_executions = cast(
            dict[str, dict[str, object]],
            snapshot["flow_executions"],
        )
        self.attempts = cast(
            dict[str, dict[str, object]], snapshot["attempts"]
        )
        self.run_transitions = cast(
            dict[str, dict[str, object]],
            snapshot["run_transitions"],
        )
        self.attempt_transitions = cast(
            dict[str, dict[str, object]],
            snapshot["attempt_transitions"],
        )
        self.events = cast(dict[str, dict[str, object]], snapshot["events"])
        self.usage = cast(dict[str, dict[str, object]], snapshot["usage"])
        self.artifacts = cast(
            dict[str, dict[str, object]], snapshot["artifacts"]
        )
        self.idempotency = cast(
            dict[str, dict[str, object]],
            snapshot["idempotency"],
        )


class FakeConnectionContext:
    def __init__(self, database: FakePgsqlTaskDatabase) -> None:
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

    def __init__(self, database: FakePgsqlTaskDatabase) -> None:
        self.database = database

    def cursor(self) -> "FakeCursorContext":
        return FakeCursorContext(self.database)

    def transaction(self) -> "FakeTransactionContext":
        return FakeTransactionContext(self.database)

    async def set_autocommit(self, value: bool) -> None:
        assert isinstance(value, bool)


class FakeTransactionContext:
    def __init__(self, database: FakePgsqlTaskDatabase) -> None:
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
    def __init__(self, database: FakePgsqlTaskDatabase) -> None:
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
    def __init__(self, database: FakePgsqlTaskDatabase) -> None:
        self.database = database
        self.row: dict[str, object] | None = None
        self.rows: tuple[dict[str, object], ...] = ()

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        params = parameters or ()
        self.database.executed_queries.append(query)
        if 'SELECT "run_id" FROM "task_runs"' in query:
            run = self.database.runs.get(cast(str, params[0]))
            self.row = {"run_id": run["run_id"]} if run is not None else None
        elif 'SELECT * FROM "task_definitions"' in query:
            self.row = self.database.definitions.get(cast(str, params[0]))
        elif 'INSERT INTO "task_definitions"' in query:
            self.row = self._insert_definition(params)
        elif 'SELECT * FROM "task_runs"' in query:
            self.row = self.database.runs.get(cast(str, params[0]))
        elif 'INSERT INTO "task_runs"' in query:
            self.row = self._insert_run(params)
        elif 'UPDATE "task_runs"' in query and '"claim" = %s::jsonb' in query:
            self.row = self._assign_claim(params)
        elif 'UPDATE "task_runs"' in query and '"last_attempt_id"' in query:
            self.row = self._update_run_last_attempt(params)
        elif 'UPDATE "task_runs"' in query:
            self.row = self._transition_run(params)
        elif 'INSERT INTO "task_run_transitions"' in query:
            self.row = self._insert_run_transition(params)
        elif 'FROM "task_run_transitions"' in query:
            self.rows = self._run_transitions(cast(str, params[0]))
        elif 'SELECT * FROM "task_flow_executions"' in query:
            self.row = self.database.flow_executions.get(cast(str, params[0]))
        elif 'INSERT INTO "task_flow_executions"' in query:
            self.row = self._insert_flow_execution(params)
        elif 'UPDATE "task_flow_executions"' in query:
            self.row = self._update_flow_execution(params)
        elif 'SELECT * FROM "task_attempts" WHERE "attempt_id"' in query:
            self.row = self.database.attempts.get(cast(str, params[0]))
        elif 'FROM "task_attempts"' in query:
            self.rows = self._attempts_for_run(cast(str, params[0]))
        elif 'INSERT INTO "task_attempts"' in query:
            self.row = self._insert_attempt(params)
        elif 'UPDATE "task_attempts"' in query:
            self.row = self._transition_attempt(params)
        elif 'INSERT INTO "task_attempt_transitions"' in query:
            self.row = self._insert_attempt_transition(params)
        elif 'FROM "task_attempt_transitions"' in query:
            self.rows = self._attempt_transitions(cast(str, params[0]))
        elif 'MAX("sequence")' in query and 'FROM "task_events"' in query:
            run_id = cast(str, params[0])
            self.row = {
                "sequence": self._next_sequence(self.database.events, run_id)
            }
        elif 'MAX("sequence")' in query:
            run_id = cast(str, params[0])
            self.row = {
                "sequence": self._next_sequence(self.database.usage, run_id)
            }
        elif 'INSERT INTO "task_events"' in query:
            self.row = self._insert_event(params)
        elif 'FROM "task_events"' in query:
            self.rows = self._filtered(
                self.database.events,
                cast(str, params[0]),
                cast(str | None, params[1]),
                cast(int | None, params[3]),
                "sequence",
            )
        elif 'INSERT INTO "task_usage_records"' in query:
            self.row = self._insert_usage(params)
        elif 'FROM "task_usage_records" WHERE "usage_id"' in query:
            self.row = self.database.usage.get(cast(str, params[0]))
        elif 'FROM "task_usage_records"' in query:
            source = cast(str | None, params[3])
            self.rows = tuple(
                row
                for row in self._filtered(
                    self.database.usage,
                    cast(str, params[0]),
                    cast(str | None, params[1]),
                    None,
                    "sequence",
                )
                if source is None or row["source"] == source
            )
        elif 'SELECT * FROM "task_artifacts" WHERE "artifact_id"' in query:
            self.row = self.database.artifacts.get(cast(str, params[0]))
        elif 'INSERT INTO "task_artifacts"' in query:
            self.row = self._insert_artifact(params)
        elif "LIMIT %s" in query and 'FROM "task_artifacts"' in query:
            self.rows = self._retention_artifacts(params)
        elif 'FROM "task_artifacts"' in query:
            self.rows = self._artifacts(params)
        elif 'UPDATE "task_artifacts"' in query:
            self.row = self._transition_artifact(params)
        elif 'SELECT * FROM "task_idempotency_keys"' in query:
            self.row = self._select_idempotency(params)
        elif 'INSERT INTO "task_idempotency_keys"' in query:
            self.row = self._insert_idempotency(params)
        else:
            raise AssertionError(f"unexpected query: {query}")

    async def fetchone(self) -> dict[str, object] | None:
        return self.row

    async def fetchall(self) -> tuple[dict[str, object], ...]:
        return self.rows

    async def close(self) -> None:
        return None

    def _insert_definition(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        definition_id = cast(str, params[0])
        if definition_id in self.database.definitions:
            return None
        row = {
            "definition_id": definition_id,
            "name": params[1],
            "version": params[2],
            "spec_hash": params[3],
            "definition": loads(cast(str, params[4])),
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
        }
        self.database.definitions[definition_id] = row
        return row

    def _insert_run(
        self, params: tuple[object, ...]
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
            "claim": None,
            "last_attempt_id": None,
            "result": None,
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
            "updated_at": params[7],
        }
        self.database.runs[run_id] = row
        return row

    def _transition_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run = self.database.runs.get(cast(str, params[4]))
        if run is None or run["state"] != params[5]:
            return None
        claim = cast(dict[str, object] | None, run["claim"])
        if (claim is None and params[6] is not None) or (
            claim is not None and claim["claim_token"] != params[7]
        ):
            return None
        run["state"] = params[0]
        if params[1] is not None:
            run["result"] = loads(cast(str, params[1]))
        if params[2]:
            run["claim"] = None
        run["updated_at"] = params[3]
        return run

    def _assign_claim(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run = self.database.runs.get(cast(str, params[3]))
        if (
            run is None
            or run["state"] != params[4]
            or run["claim"] is not None
        ):
            return None
        run["state"] = params[0]
        run["claim"] = loads(cast(str, params[1]))
        run["updated_at"] = params[2]
        return run

    def _update_run_last_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        if self.database.before_last_attempt_update is not None:
            self.database.before_last_attempt_update(params)
        run = self.database.runs.get(cast(str, params[2]))
        if run is None or run["state"] != params[3]:
            return None
        claim = cast(dict[str, object] | None, run["claim"])
        if (claim is None and params[4] is not None) or (
            claim is not None and claim["claim_token"] != params[5]
        ):
            return None
        run["last_attempt_id"] = params[0]
        run["updated_at"] = params[1]
        return run

    def _insert_run_transition(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        row = {
            "transition_id": params[0],
            "run_id": params[1],
            "from_state": params[2],
            "to_state": params[3],
            "reason": params[4],
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
        }
        self.database.run_transitions[cast(str, params[0])] = row
        return row

    def _insert_flow_execution(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        task_run_id = cast(str, params[0])
        if task_run_id in self.database.flow_executions:
            return None
        row = {
            "task_run_id": task_run_id,
            "revision": params[1],
            "trace": loads(cast(str, params[2])),
            "node_attempts": loads(cast(str, params[3])),
            "node_outputs": loads(cast(str, params[4])),
            "selected_outputs": loads(cast(str, params[5])),
            "loop_counters": loads(cast(str, params[6])),
            "pause_tokens": loads(cast(str, params[7])),
            "diagnostics": loads(cast(str, params[8])),
            "artifact_refs": loads(cast(str, params[9])),
            "metadata": loads(cast(str, params[10])),
            "created_at": params[11],
            "updated_at": params[12],
        }
        self.database.flow_executions[task_run_id] = row
        return row

    def _update_flow_execution(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        task_run_id = cast(str, params[10])
        row = self.database.flow_executions.get(task_run_id)
        if row is None or row["revision"] != params[11]:
            return None
        for index, key in (
            (0, "trace"),
            (1, "node_attempts"),
            (2, "selected_outputs"),
            (3, "node_outputs"),
            (4, "loop_counters"),
            (5, "pause_tokens"),
            (6, "diagnostics"),
            (7, "artifact_refs"),
            (8, "metadata"),
        ):
            if params[index] is not None:
                row[key] = loads(cast(str, params[index]))
        row["revision"] = cast(int, row["revision"]) + 1
        row["updated_at"] = params[9]
        return row

    def _insert_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        attempt_id = cast(str, params[0])
        if attempt_id in self.database.attempts:
            return None
        row = {
            "attempt_id": attempt_id,
            "run_id": params[1],
            "attempt_number": params[2],
            "state": params[3],
            "context": loads(cast(str, params[4])),
            "result": None,
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
            "updated_at": params[7],
        }
        self.database.attempts[attempt_id] = row
        return row

    def _transition_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        if self.database.before_attempt_update is not None:
            self.database.before_attempt_update(params)
        attempt = self.database.attempts.get(cast(str, params[3]))
        if attempt is None or attempt["state"] != params[4]:
            return None
        run = self.database.runs.get(cast(str, params[5]))
        if (
            run is None
            or attempt["run_id"] != run["run_id"]
            or run["state"] != params[6]
        ):
            return None
        claim = cast(dict[str, object] | None, run["claim"])
        if (claim is None and params[7] is not None) or (
            claim is not None and claim["claim_token"] != params[8]
        ):
            return None
        attempt["state"] = params[0]
        if params[1] is not None:
            attempt["result"] = loads(cast(str, params[1]))
        attempt["updated_at"] = params[2]
        return attempt

    def _insert_attempt_transition(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        row = {
            "transition_id": params[0],
            "attempt_id": params[1],
            "run_id": params[2],
            "from_state": params[3],
            "to_state": params[4],
            "reason": params[5],
            "metadata": loads(cast(str, params[6])),
            "created_at": params[7],
        }
        self.database.attempt_transitions[cast(str, params[0])] = row
        return row

    def _insert_event(
        self, params: tuple[object, ...]
    ) -> dict[str, object] | None:
        event_id = cast(str, params[0])
        if event_id in self.database.events:
            return None
        row = {
            "event_id": event_id,
            "run_id": params[1],
            "attempt_id": params[2],
            "sequence": params[3],
            "event_type": params[4],
            "payload": loads(cast(str, params[5])),
            "metadata": loads(cast(str, params[6])),
            "event_time": params[7],
            "created_at": params[8],
        }
        self.database.events[event_id] = row
        return row

    def _insert_usage(
        self, params: tuple[object, ...]
    ) -> dict[str, object] | None:
        usage_id = cast(str, params[0])
        if usage_id in self.database.usage:
            return None
        row = {
            "usage_id": usage_id,
            "run_id": params[1],
            "attempt_id": params[2],
            "sequence": params[3],
            "source": params[4],
            "prompt_tokens": params[5],
            "completion_tokens": params[6],
            "total_tokens": params[7],
            "cached_tokens": params[8],
            "cache_creation_input_tokens": params[9],
            "reasoning_tokens": params[10],
            "metadata": loads(cast(str, params[11])),
            "created_at": params[12],
        }
        self.database.usage[usage_id] = row
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

    def _transition_artifact(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        artifact = self.database.artifacts.get(cast(str, params[3]))
        if artifact is None or artifact["state"] != params[4]:
            return None
        artifact["state"] = params[0]
        if params[1] is not None:
            artifact["metadata"] = loads(cast(str, params[1]))
        artifact["updated_at"] = params[2]
        return artifact

    def _insert_idempotency(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        identity_key = cast(str, params[0])
        existing = self.database.idempotency.get(identity_key)
        if existing is not None:
            expires_at = cast(datetime | None, existing["expires_at"])
            checked_at = cast(datetime, params[14])
            if expires_at is None or expires_at > checked_at:
                return None
        row = {
            "identity_key": identity_key,
            "task_name": params[1],
            "task_version": params[2],
            "spec_hash": params[3],
            "owner_scope_hash": loads(cast(str, params[4])),
            "strategy": params[5],
            "window_hash": loads(cast(str, params[6])) if params[6] else None,
            "input_hash": loads(cast(str, params[7])) if params[7] else None,
            "file_hash": loads(cast(str, params[8])) if params[8] else None,
            "custom_hash": loads(cast(str, params[9])) if params[9] else None,
            "run_id": params[10],
            "metadata": loads(cast(str, params[11])),
            "expires_at": params[12],
            "created_at": params[13],
        }
        self.database.idempotency[identity_key] = row
        return row

    def _select_idempotency(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        row = self.database.idempotency.get(cast(str, params[0]))
        if row is None:
            return None
        expires_at = cast(datetime | None, row["expires_at"])
        checked_at = cast(datetime, params[1])
        if expires_at is not None and expires_at <= checked_at:
            return None
        return row

    def _attempts_for_run(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            sorted(
                (
                    row
                    for row in self.database.attempts.values()
                    if row["run_id"] == run_id
                ),
                key=lambda row: cast(int, row["attempt_number"]),
            )
        )

    def _run_transitions(self, run_id: str) -> tuple[dict[str, object], ...]:
        return tuple(
            row
            for row in self.database.run_transitions.values()
            if row["run_id"] == run_id
        )

    def _attempt_transitions(
        self,
        attempt_id: str,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            row
            for row in self.database.attempt_transitions.values()
            if row["attempt_id"] == attempt_id
        )

    def _filtered(
        self,
        rows: dict[str, dict[str, object]],
        run_id: str,
        attempt_id: str | None,
        after_sequence: int | None,
        order_key: str,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            sorted(
                (
                    row
                    for row in rows.values()
                    if row["run_id"] == run_id
                    and (attempt_id is None or row["attempt_id"] == attempt_id)
                    and (
                        after_sequence is None
                        or cast(int, row["sequence"]) > after_sequence
                    )
                ),
                key=lambda row: cast(Any, row[order_key]),
            )
        )

    def _artifacts(
        self, params: tuple[object, ...]
    ) -> tuple[dict[str, object], ...]:
        run_id = cast(str, params[0])
        attempt_id = cast(str | None, params[1])
        purpose = cast(str | None, params[3])
        state = cast(str | None, params[5])
        return tuple(
            row
            for row in self.database.artifacts.values()
            if row["run_id"] == run_id
            and (attempt_id is None or row["attempt_id"] == attempt_id)
            and (purpose is None or row["purpose"] == purpose)
            and (state is None or row["state"] == state)
        )

    def _retention_artifacts(
        self,
        params: tuple[object, ...],
    ) -> tuple[dict[str, object], ...]:
        state = cast(str, params[0])
        purpose = cast(str | None, params[1])
        expired_at = cast(datetime, params[3])
        limit = cast(int, params[5])
        rows = tuple(
            row
            for row in self.database.artifacts.values()
            if row["state"] == state
            and (purpose is None or row["purpose"] == purpose)
            and _retention_expired(row, expired_at)
        )
        return rows[:limit]

    def _next_sequence(
        self,
        rows: dict[str, dict[str, object]],
        run_id: str,
    ) -> int:
        return (
            max(
                (
                    cast(int, row["sequence"])
                    for row in rows.values()
                    if row["run_id"] == run_id
                ),
                default=0,
            )
            + 1
        )


def _retention_expired(
    row: dict[str, object],
    expired_at: datetime,
) -> bool:
    retention = cast(dict[str, object], row["retention"])
    expires_at_value = retention.get("expires_at")
    if isinstance(expires_at_value, str):
        return datetime.fromisoformat(expires_at_value) <= expired_at
    delete_after_days = retention.get("delete_after_days")
    if isinstance(delete_after_days, int):
        created_at = cast(datetime, row["created_at"])
        return created_at + timedelta(days=delete_after_days) <= expired_at
    return False


class PgsqlStoreContractTest(
    StoreContractAssertions,
    IsolatedAsyncioTestCase,
):
    def create_store(self) -> PgsqlTaskStore:
        self.database = FakePgsqlTaskDatabase()
        return PgsqlTaskStore(
            self.database,
            clock=self.clock,
            id_factory=SequenceIds(),
        )

    async def test_context_manager_opens_and_closes_database(self) -> None:
        store = PgsqlTaskStore(
            self.database,
            clock=SequenceClock(),
            id_factory=SequenceIds(),
        )

        async with store as opened:
            self.assertIs(opened, store)

        self.assertEqual(self.database.open_count, 1)
        self.assertEqual(self.database.close_count, 1)

    async def test_create_attempt_update_rechecks_run_state(self) -> None:
        run = await self._created_run()

        def finish_run(params: tuple[object, ...]) -> None:
            assert params[2] == run.run_id
            self.database.runs[run.run_id][
                "state"
            ] = TaskRunState.SUCCEEDED.value

        self.database.before_last_attempt_update = finish_run

        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(run.run_id)

        self.assertEqual(self.database.attempts, {})
        self.assertEqual(
            self.database.runs[run.run_id]["state"],
            TaskRunState.CREATED.value,
        )

    async def test_claimed_run_can_be_marked_cancel_requested_without_token(
        self,
    ) -> None:
        run = await self._created_run()
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await self.store.assign_claim(
            run.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            reason="claimed",
        )
        assert claimed.claim is not None
        running = await self.store.transition_run(
            claimed.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claimed.claim.claim_token,
        )

        cancel_requested = await self.store.transition_run(
            running.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="cancel_requested",
        )

        self.assertEqual(cancel_requested.state, TaskRunState.CANCEL_REQUESTED)
        self.assertIsNotNone(cancel_requested.claim)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                cancel_requested.run_id,
                from_states={TaskRunState.CANCEL_REQUESTED},
                to_state=TaskRunState.FAILED,
                reason="failed",
            )

    async def test_attempt_transition_update_rechecks_claim(self) -> None:
        run = await self._created_run()
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await self.store.assign_claim(
            run.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            reason="claimed",
        )
        assert claimed.claim is not None
        running = await self.store.transition_run(
            claimed.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claimed.claim.claim_token,
        )
        attempt = await self.store.create_attempt(
            running.run_id,
            claim_token=claimed.claim.claim_token,
        )

        def replace_claim(params: tuple[object, ...]) -> None:
            assert params[5] == run.run_id
            claim = self.database.runs[run.run_id]["claim"]
            assert isinstance(claim, dict)
            claim["claim_token"] = "replacement-token"

        self.database.before_attempt_update = replace_claim

        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_attempt(
                attempt.attempt_id,
                from_states={TaskAttemptState.CREATED},
                to_state=TaskAttemptState.RUNNING,
                reason="attempt_started",
                claim_token=claimed.claim.claim_token,
            )

        self.assertEqual(
            self.database.attempts[attempt.attempt_id]["state"],
            TaskAttemptState.CREATED.value,
        )
        self.assertEqual(
            await self.store.list_attempt_transitions(attempt.attempt_id),
            (),
        )

    async def test_records_events_usage_and_idempotency(self) -> None:
        run = await self._created_run()
        attempt = await self.store.create_attempt(run.run_id)
        query_count = len(self.database.executed_queries)

        event = await self.store.append_event(
            run.run_id,
            attempt_id=attempt.attempt_id,
            event_type="model_complete",
            category=TaskEventCategory.MODEL,
            payload={"safe": "value"},
        )
        usage = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=3,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=5,
                reasoning_tokens=7,
                total_tokens=8,
            ),
            metadata={
                "provider_family": UsageProviderFamily.ANTHROPIC,
                "cache_creation_ephemeral_5m_input_tokens": 11,
                "cache_creation_ephemeral_1h_input_tokens": 13,
            },
        )
        second_usage = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=4,
                output_tokens=2,
                total_tokens=6,
            ),
            metadata={"provider_family": UsageProviderFamily.OPENAI},
        )
        identity = TaskIdempotencyIdentity(
            identity_key="identity-1",
            task_name="summarize",
            task_version="1",
            spec_hash="hash-a",
            owner_scope=TaskIdempotencyDigest(
                algorithm="hmac-sha256",
                digest="a" * 64,
                key_id="idempotency-v1",
            ),
            strategy=IdempotencyMode.INPUT_HASH,
            input=TaskIdempotencyDigest(
                algorithm="hmac-sha256",
                digest="b" * 64,
                key_id="idempotency-v1",
            ),
        )
        reservation = await self.store.reserve_idempotency_key(
            identity,
            run_id=run.run_id,
            metadata={"source": "test"},
        )
        repeated = await self.store.reserve_idempotency_key(
            identity,
            run_id=run.run_id,
        )

        self.assertEqual(event.sequence, 1)
        self.assertEqual(
            await self.store.list_events(
                run.run_id,
                attempt_id=attempt.attempt_id,
            ),
            (event,),
        )
        self.assertEqual(
            await self.store.list_events(
                run.run_id,
                attempt_id=attempt.attempt_id,
                after_sequence=1,
            ),
            (),
        )
        self.assertEqual(usage.sequence, 1)
        self.assertEqual(second_usage.sequence, 2)
        append_queries = self.database.executed_queries[query_count:]
        locked_run_queries = tuple(
            query
            for query in append_queries
            if 'SELECT * FROM "task_runs"' in query and "FOR UPDATE" in query
        )
        sequence_queries = tuple(
            query for query in append_queries if 'MAX("sequence")' in query
        )
        self.assertEqual(len(locked_run_queries), 3)
        self.assertEqual(len(sequence_queries), 3)
        self.assertEqual(
            usage.metadata["provider_family"],
            UsageProviderFamily.ANTHROPIC.value,
        )
        self.assertEqual(
            usage.metadata["cache_creation_ephemeral_5m_input_tokens"],
            11,
        )
        self.assertEqual(
            usage.metadata["cache_creation_ephemeral_1h_input_tokens"],
            13,
        )
        usage_row = self.database.usage[usage.usage_id]
        self.assertEqual(usage_row["cache_creation_input_tokens"], 2)
        self.assertEqual(usage_row["reasoning_tokens"], 7)
        self.assertNotIn(
            "cache_creation_input_tokens",
            cast(dict[str, object], usage_row["metadata"]),
        )
        self.assertNotIn(
            "reasoning_tokens",
            cast(dict[str, object], usage_row["metadata"]),
        )
        self.assertEqual(
            await self.store.list_usage(
                run.run_id,
                attempt_id=attempt.attempt_id,
            ),
            (usage, second_usage),
        )
        self.assertEqual(
            await self.store.usage_totals(run.run_id),
            UsageTotals(
                input_tokens=7,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=7,
                reasoning_tokens=7,
                total_tokens=14,
            ),
        )
        self.assertTrue(reservation.created)
        self.assertFalse(repeated.created)
        self.assertEqual(repeated.reservation, reservation.reservation)

    async def test_expired_idempotency_reservation_can_be_replaced(
        self,
    ) -> None:
        first_run = await self._created_run()
        second_run = await self.store.create_run(
            TaskExecutionRequest(definition_id=first_run.definition_id)
        )
        expires_at = datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)
        identity = TaskIdempotencyIdentity(
            identity_key="identity-expiring",
            task_name="summarize",
            task_version="1",
            spec_hash="hash-a",
            owner_scope=TaskIdempotencyDigest(
                algorithm="hmac-sha256",
                digest="a" * 64,
                key_id="idempotency-v1",
            ),
            strategy=IdempotencyMode.INPUT_HASH,
            input=TaskIdempotencyDigest(
                algorithm="hmac-sha256",
                digest="b" * 64,
                key_id="idempotency-v1",
            ),
        )

        first = await self.store.reserve_idempotency_key(
            identity,
            run_id=first_run.run_id,
            expires_at=expires_at,
        )
        self.clock._next = expires_at  # noqa: SLF001
        second = await self.store.reserve_idempotency_key(
            identity,
            run_id=second_run.run_id,
            metadata={"source": "replacement"},
        )

        self.assertTrue(first.created)
        self.assertTrue(second.created)
        self.assertEqual(second.reservation.run_id, second_run.run_id)
        self.assertEqual(second.reservation.metadata["source"], "replacement")
        self.assertEqual(
            self.database.idempotency[identity.identity_key]["run_id"],
            second_run.run_id,
        )

    async def test_active_idempotency_key_rejects_identity_mismatch(
        self,
    ) -> None:
        first_run = await self._created_run()
        second_run = await self.store.create_run(
            TaskExecutionRequest(definition_id=first_run.definition_id)
        )
        identity = _identity("identity-collision")
        mismatched = _identity(
            "identity-collision",
            spec_hash="hash-other",
        )

        reserved = await self.store.reserve_idempotency_key(
            identity,
            run_id=first_run.run_id,
        )

        with self.assertRaises(TaskStoreConflictError):
            await self.store.reserve_idempotency_key(
                mismatched,
                run_id=second_run.run_id,
            )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.lookup_idempotency_key(mismatched)

        found = await self.store.lookup_idempotency_key(identity)
        self.assertEqual(found, reserved.reservation)
        self.assertEqual(
            self.database.idempotency[identity.identity_key]["run_id"],
            first_run.run_id,
        )

    async def test_cached_usage_counters_are_independent(self) -> None:
        run = await self._created_run()
        attempt = await self.store.create_attempt(run.run_id)

        usage = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1, cached_input_tokens=3),
        )

        self.assertEqual(usage.totals.input_tokens, 1)
        self.assertEqual(usage.totals.cached_input_tokens, 3)
        self.assertEqual(
            await self.store.usage_totals(run.run_id),
            UsageTotals(input_tokens=1, cached_input_tokens=3),
        )

    async def test_usage_list_and_totals_filter_by_source(self) -> None:
        run = await self._created_run()
        attempt = await self.store.create_attempt(run.run_id)
        exact = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1, cached_input_tokens=0),
        )
        await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.UNAVAILABLE,
            totals=UsageTotals(),
        )
        estimated = await self.store.append_usage(
            run.run_id,
            source=UsageSource.ESTIMATED,
            totals=UsageTotals(output_tokens=5),
        )

        exact_records = await self.store.list_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            source=UsageSource.EXACT,
        )
        estimated_records = await self.store.list_usage(
            run.run_id,
            source=UsageSource.ESTIMATED,
        )
        exact_totals = await self.store.usage_totals(
            run.run_id,
            source=UsageSource.EXACT,
        )

        self.assertEqual(exact_records, (exact,))
        self.assertEqual(estimated_records, (estimated,))
        self.assertEqual(
            exact_totals,
            UsageTotals(input_tokens=1, cached_input_tokens=0),
        )
        self.assertTrue(
            any(
                '"source" = %s::text' in query
                for query in self.database.executed_queries
            )
        )

    async def test_usage_filters_reject_invalid_source(self) -> None:
        run = await self._created_run()
        invalid_source = cast(UsageSource, "exact")

        with self.assertRaises(AssertionError):
            await self.store.list_usage(
                run.run_id,
                source=invalid_source,
            )
        with self.assertRaises(AssertionError):
            await self.store.usage_totals(
                run.run_id,
                source=invalid_source,
            )

    async def test_stable_usage_id_deduplicates_records(self) -> None:
        run = await self._created_run()
        attempt = await self.store.create_attempt(run.run_id)
        usage_id = stable_usage_id(
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            call_key="model-call-1",
        )

        first = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            usage_id=usage_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
        )
        duplicate = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            usage_id=usage_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=99, total_tokens=99),
        )
        distinct = await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            usage_id=stable_usage_id(
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                call_key="model-call-2",
            ),
            source=UsageSource.EXACT,
            totals=UsageTotals(output_tokens=3),
        )

        self.assertEqual(first, duplicate)
        self.assertNotEqual(first.usage_id, distinct.usage_id)
        self.assertEqual(
            set(self.database.usage), {usage_id, distinct.usage_id}
        )
        self.assertEqual(
            await self.store.usage_totals(run.run_id),
            UsageTotals(input_tokens=1, output_tokens=3),
        )

    async def test_stable_usage_id_rejects_other_run_collision(self) -> None:
        run = await self._created_run()
        attempt = await self.store.create_attempt(run.run_id)
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id=run.definition_id)
        )
        usage_id = stable_usage_id(
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            call_key="model-call-1",
        )

        await self.store.append_usage(
            run.run_id,
            attempt_id=attempt.attempt_id,
            usage_id=usage_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
        )

        with self.assertRaises(TaskStoreConflictError):
            await self.store.append_usage(
                other_run.run_id,
                usage_id=usage_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
            )


def _identity(
    identity_key: str,
    *,
    spec_hash: str = "hash-a",
) -> TaskIdempotencyIdentity:
    digest = TaskIdempotencyDigest(
        algorithm="hmac-sha256",
        digest="a" * 64,
        key_id="idempotency-v1",
    )
    return TaskIdempotencyIdentity(
        identity_key=identity_key,
        task_name="summarize",
        task_version="1",
        spec_hash=spec_hash,
        owner_scope=digest,
        strategy=IdempotencyMode.INPUT_HASH,
        input=TaskIdempotencyDigest(
            algorithm="hmac-sha256",
            digest="b" * 64,
            key_id="idempotency-v1",
        ),
    )


if __name__ == "__main__":
    main()
