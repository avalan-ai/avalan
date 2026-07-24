from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, patch

sys_path.append(str(Path(__file__).parents[1]))

from pgsql_contract_test import (  # type: ignore[import-not-found]
    FakeCursor,
    FakePgsqlTaskDatabase,
)
from store_contract_test import (  # type: ignore[import-not-found]
    SequenceClock,
    SequenceIds,
    definition,
)

from avalan.pgsql import (
    PgsqlFailure,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlUnitOfWork,
)
from avalan.skill import (
    SkillReadLimits,
    SkillSettingsSurface,
    UntrustedSkillSettings,
)
from avalan.task import (
    IdempotencyMode,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskAttempt,
    TaskAttemptSegment,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskClaim,
    TaskExecutionContext,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskInteractionEventType,
    TaskQueueItemState,
    TaskRun,
    TaskRunState,
    TaskStoreConflictError,
    TaskStoreError,
    TaskStoreNotFoundError,
    UsageSource,
    UsageTotals,
)
from avalan.task.settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSuccess,
)
from avalan.task.stores import PgsqlTaskStore
from avalan.task.stores import pgsql as pgsql_store_module


class CloseOnlyDatabase:
    def __init__(self) -> None:
        self.close_count = 0

    def connection(self) -> object:
        raise AssertionError("connection should not be opened")

    def close(self) -> None:
        self.close_count += 1


class NoLifecycleDatabase:
    def connection(self) -> object:
        raise AssertionError("connection should not be opened")


class ErrorConnectionContext:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aenter__(self) -> object:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ErrorDatabase:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def connection(self) -> ErrorConnectionContext:
        return ErrorConnectionContext(self.error)


class ScriptedCursor:
    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, object] | None] = (),
        row_sets: Sequence[Sequence[Mapping[str, object]]] = (),
    ) -> None:
        self.rows = deque(rows)
        self.row_sets = deque(tuple(values) for values in row_sets)
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        self.executed.append((query, parameters))

    async def fetchone(self) -> Mapping[str, object] | None:
        if not self.rows:
            raise AssertionError("unexpected scripted fetchone")
        return self.rows.popleft()

    async def fetchall(self) -> tuple[Mapping[str, object], ...]:
        if not self.row_sets:
            raise AssertionError("unexpected scripted fetchall")
        return self.row_sets.popleft()

    async def close(self) -> None:
        return None


class ScriptedCursorContext:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_value = cursor

    async def __aenter__(self) -> ScriptedCursor:
        return self.cursor_value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ScriptedTransactionContext:
    async def __aenter__(self) -> "ScriptedTransactionContext":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ScriptedConnection:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_value = cursor

    def cursor(self) -> ScriptedCursorContext:
        return ScriptedCursorContext(self.cursor_value)

    def transaction(self) -> ScriptedTransactionContext:
        return ScriptedTransactionContext()


class ScriptedConnectionContext:
    def __init__(self, connection: ScriptedConnection) -> None:
        self.connection_value = connection

    async def __aenter__(self) -> ScriptedConnection:
        return self.connection_value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ScriptedDatabase:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.connection_value = ScriptedConnection(cursor)

    def connection(self) -> ScriptedConnectionContext:
        return ScriptedConnectionContext(self.connection_value)


class UniqueSqlStateError(RuntimeError):
    sqlstate = "23505"


class UnknownSqlStateError(RuntimeError):
    pass


def _store(
    database: FakePgsqlTaskDatabase | None = None,
) -> PgsqlTaskStore:
    return PgsqlTaskStore(
        database or FakePgsqlTaskDatabase(),
        clock=SequenceClock(),
        id_factory=SequenceIds(),
    )


async def _registered_store() -> PgsqlTaskStore:
    store = _store()
    await store.register_definition(definition(), definition_hash="hash-a")
    return store


async def _created_run(store: PgsqlTaskStore | None = None):
    value = store or await _registered_store()
    return await value.create_run(TaskExecutionRequest(definition_id="hash-a"))


async def _created_attempt(store: PgsqlTaskStore | None = None):
    value = store or await _registered_store()
    run = await _created_run(value)
    return run, await value.create_attempt(run.run_id)


def _identity() -> TaskIdempotencyIdentity:
    digest = TaskIdempotencyDigest(
        algorithm="hmac-sha256",
        digest="a" * 64,
        key_id="idempotency-v1",
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


def _artifact_ref() -> TaskArtifactRef:
    return TaskArtifactRef(
        artifact_id="artifact-1",
        store="local",
        storage_key="artifacts/artifact-1",
    )


_DURABLE_NOW = datetime(2026, 7, 23, 12, tzinfo=UTC)


def _durable_claim(token: str = "claim-token") -> TaskClaim:
    return TaskClaim(
        worker_id="worker-1",
        claim_token=token,
        claimed_at=_DURABLE_NOW - timedelta(seconds=2),
        lease_expires_at=_DURABLE_NOW + timedelta(minutes=1),
        heartbeat_at=_DURABLE_NOW - timedelta(seconds=1),
    )


def _durable_run(
    *,
    state: TaskRunState = TaskRunState.RUNNING,
    claim: TaskClaim | None = None,
    last_attempt_id: str | None = "attempt-1",
    result: TaskExecutionResult | None = None,
) -> TaskRun:
    return TaskRun(
        run_id="run-1",
        definition_id="definition-1",
        state=state,
        request=TaskExecutionRequest(
            definition_id="definition-1",
            queue="durable",
        ),
        claim=claim,
        last_attempt_id=last_attempt_id,
        result=result,
        created_at=_DURABLE_NOW - timedelta(minutes=1),
        updated_at=_DURABLE_NOW,
    )


def _durable_attempt(
    *,
    state: TaskAttemptState = TaskAttemptState.RUNNING,
    claim: TaskClaim | None = None,
    result: TaskExecutionResult | None = None,
) -> TaskAttempt:
    return TaskAttempt(
        attempt_id="attempt-1",
        run_id="run-1",
        attempt_number=1,
        state=state,
        context=TaskExecutionContext(
            run_id="run-1",
            attempt_id="attempt-1",
            attempt_number=1,
            claim=claim,
        ),
        result=result,
        created_at=_DURABLE_NOW - timedelta(seconds=30),
        updated_at=_DURABLE_NOW,
    )


def _durable_segment(
    *,
    segment_id: str = "segment-1",
    state: TaskAttemptSegmentState = TaskAttemptSegmentState.RUNNING,
    claim: TaskClaim | None = None,
    resumed_from_segment_id: str | None = None,
    request_id: str | None = None,
    continuation_id: str | None = None,
    checkpoint_id: str | None = None,
) -> TaskAttemptSegment:
    return TaskAttemptSegment(
        segment_id=segment_id,
        attempt_id="attempt-1",
        run_id="run-1",
        segment_number=2 if resumed_from_segment_id is not None else 1,
        state=state,
        claim=claim,
        resumed_from_segment_id=resumed_from_segment_id,
        request_id=request_id,
        continuation_id=continuation_id,
        checkpoint_id=checkpoint_id,
        created_at=_DURABLE_NOW - timedelta(seconds=20),
        updated_at=_DURABLE_NOW,
    )


def _durable_segment_row(
    segment: TaskAttemptSegment,
) -> dict[str, object]:
    return {
        "segment_id": segment.segment_id,
        "attempt_id": segment.attempt_id,
        "run_id": segment.run_id,
        "segment_number": segment.segment_number,
        "state": segment.state.value,
        "claim": (
            pgsql_store_module._claim_to_payload(segment.claim)
            if segment.claim is not None
            else None
        ),
        "resumed_from_segment_id": segment.resumed_from_segment_id,
        "request_id": segment.request_id,
        "continuation_id": segment.continuation_id,
        "checkpoint_id": segment.checkpoint_id,
        "metadata": {},
        "created_at": segment.created_at,
        "updated_at": segment.updated_at,
    }


def _durable_run_row(run: TaskRun) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "definition_id": run.definition_id,
        "state": run.state.value,
        "request": pgsql_store_module._request_to_payload(run.request),
        "claim": (
            pgsql_store_module._claim_to_payload(run.claim)
            if run.claim is not None
            else None
        ),
        "last_attempt_id": run.last_attempt_id,
        "result": (
            pgsql_store_module._result_to_payload(run.result)
            if run.result is not None
            else None
        ),
        "metadata": {},
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _durable_attempt_row(attempt: TaskAttempt) -> dict[str, object]:
    return {
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "attempt_number": attempt.attempt_number,
        "state": attempt.state.value,
        "context": pgsql_store_module._context_to_payload(attempt.context),
        "result": (
            pgsql_store_module._result_to_payload(attempt.result)
            if attempt.result is not None
            else None
        ),
        "metadata": {},
        "created_at": attempt.created_at,
        "updated_at": attempt.updated_at,
    }


def _durable_queue_row(
    *,
    state: TaskQueueItemState = TaskQueueItemState.CLAIMED,
    run_id: str = "run-1",
    claim_token: str | None = "claim-token",
    attempt_id: str | None = "attempt-1",
    segment_id: str | None = "segment-1",
    request_id: str | None = "request-1",
    continuation_id: str | None = "continuation-1",
    lease_expires_at: datetime | None = None,
) -> dict[str, object]:
    claimed = state is TaskQueueItemState.CLAIMED
    return {
        "queue_item_id": "queue-1",
        "run_id": run_id,
        "queue_name": "durable",
        "state": state.value,
        "priority": 0,
        "available_at": _DURABLE_NOW - timedelta(seconds=10),
        "claimed_at": _DURABLE_NOW - timedelta(seconds=2) if claimed else None,
        "lease_expires_at": (
            lease_expires_at
            if lease_expires_at is not None
            else (_DURABLE_NOW + timedelta(minutes=1) if claimed else None)
        ),
        "worker_id": "worker-1" if claimed else None,
        "claim_token": claim_token if claimed else None,
        "heartbeat_at": (
            _DURABLE_NOW - timedelta(seconds=1) if claimed else None
        ),
        "attempt_id": attempt_id,
        "segment_id": segment_id,
        "request_id": request_id,
        "continuation_id": continuation_id,
        "attempts": 1,
        "metadata": {},
        "created_at": _DURABLE_NOW - timedelta(minutes=1),
        "updated_at": _DURABLE_NOW,
    }


def _scripted_store(
    *,
    rows: Sequence[Mapping[str, object] | None] = (),
    row_sets: Sequence[Sequence[Mapping[str, object]]] = (),
) -> tuple[PgsqlTaskStore, ScriptedCursor]:
    cursor = ScriptedCursor(rows=rows, row_sets=row_sets)
    store = PgsqlTaskStore(
        cast(Any, ScriptedDatabase(cursor)),
        clock=lambda: _DURABLE_NOW,
        id_factory=SequenceIds(),
    )
    return store, cursor


def _scripted_unit(cursor: ScriptedCursor) -> PgsqlUnitOfWork:
    connection = ScriptedConnection(cursor)
    return PgsqlUnitOfWork(
        connection=cast(Any, connection),
        cursor=cast(Any, cursor),
    )


class PgsqlStoreCoverageTest(IsolatedAsyncioTestCase):
    async def test_optional_open_and_close_hooks_are_supported(self) -> None:
        await PgsqlTaskStore(cast(Any, NoLifecycleDatabase())).open()
        close_database = CloseOnlyDatabase()
        await PgsqlTaskStore(cast(Any, close_database)).aclose()
        self.assertEqual(close_database.close_count, 1)

    async def test_register_definition_insert_none_fallbacks(self) -> None:
        store = _store()
        original_insert = FakeCursor._insert_definition

        def insert_same(self: FakeCursor, params: tuple[object, ...]):
            original_insert(self, params)
            return None

        with patch.object(FakeCursor, "_insert_definition", insert_same):
            record = await store.register_definition(
                definition(),
                definition_hash="hash-a",
            )
        self.assertEqual(record.definition_id, "hash-a")

        store = _store()
        with patch.object(
            FakeCursor,
            "_insert_definition",
            return_value=None,
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.register_definition(
                    definition(),
                    definition_hash="hash-a",
                )

        store = _store()
        other = definition()
        object.__setattr__(other.task, "name", "other")

        def insert_different(self: FakeCursor, params: tuple[object, ...]):
            values = list(params)
            values[1] = other.task.name
            values[4] = pgsql_store_module._json(
                pgsql_store_module._definition_to_payload(other)
            )
            original_insert(self, tuple(values))
            return None

        with patch.object(FakeCursor, "_insert_definition", insert_different):
            with self.assertRaises(TaskStoreConflictError):
                await store.register_definition(
                    definition(),
                    definition_hash="hash-a",
                )

    async def test_missing_records_raise_not_found(self) -> None:
        store = _store()

        with self.assertRaises(TaskStoreNotFoundError):
            await store.get_definition("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await store.get_run("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await store.get_attempt("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await store.get_artifact("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await store.append_event(
                "missing",
                event_type="model_complete",
                category=pgsql_store_module.TaskEventCategory.MODEL,
                payload={},
            )

    async def test_get_definition_and_attempt_return_records(self) -> None:
        store = await _registered_store()
        definition_record = await store.get_definition("hash-a")
        run, attempt = await _created_attempt(store)

        self.assertEqual(definition_record.definition_id, "hash-a")
        self.assertEqual(
            (await store.get_attempt(attempt.attempt_id)).run_id,
            run.run_id,
        )

    async def test_insert_conflicts_raise_store_conflicts(self) -> None:
        store = await _registered_store()
        with patch.object(FakeCursor, "_insert_run", return_value=None):
            with self.assertRaises(TaskStoreConflictError):
                await store.create_run(
                    TaskExecutionRequest(definition_id="hash-a")
                )

        run = await _created_run(store)
        with patch.object(FakeCursor, "_insert_attempt", return_value=None):
            with self.assertRaises(TaskStoreConflictError):
                await store.create_attempt(run.run_id)
        with patch.object(
            FakeCursor,
            "_update_run_last_attempt",
            return_value=None,
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.create_attempt(run.run_id)

    async def test_run_transition_conflict_branches(self) -> None:
        store = await _registered_store()
        run = await _created_run(store)

        with patch.object(FakeCursor, "_transition_run", return_value=None):
            with self.assertRaises(TaskStoreConflictError):
                await store.transition_run(
                    run.run_id,
                    from_states={TaskRunState.CREATED},
                    to_state=TaskRunState.VALIDATED,
                    reason="validated",
                )

        with patch.object(
            FakeCursor,
            "_insert_run_transition",
            return_value=None,
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.transition_run(
                    run.run_id,
                    from_states={TaskRunState.CREATED},
                    to_state=TaskRunState.VALIDATED,
                    reason="validated",
                )

    async def test_claim_conflict_branches(self) -> None:
        store = await _registered_store()
        run = await _created_run(store)
        lease = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5)
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        await store.assign_claim(
            run.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=lease,
            reason="claim",
        )
        with self.assertRaises(TaskStoreConflictError):
            await store.assign_claim(
                run.run_id,
                from_states={TaskRunState.CLAIMED},
                worker_id="worker-2",
                lease_expires_at=lease,
                reason="claim",
            )

        run = await _created_run(store)
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        with patch.object(FakeCursor, "_assign_claim", return_value=None):
            with self.assertRaises(TaskStoreConflictError):
                await store.assign_claim(
                    run.run_id,
                    from_states={TaskRunState.QUEUED},
                    worker_id="worker-3",
                    lease_expires_at=lease,
                    reason="claim",
                )

        with patch.object(
            FakeCursor,
            "_insert_run_transition",
            return_value=None,
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.assign_claim(
                    run.run_id,
                    from_states={TaskRunState.QUEUED},
                    worker_id="worker-4",
                    lease_expires_at=lease,
                    reason="claim",
                )

    async def test_attempt_transition_conflict_branches(self) -> None:
        store = await _registered_store()
        run, attempt = await _created_attempt(store)

        with patch.object(
            FakeCursor, "_transition_attempt", return_value=None
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.transition_attempt(
                    attempt.attempt_id,
                    from_states={TaskAttemptState.CREATED},
                    to_state=TaskAttemptState.RUNNING,
                    reason="started",
                )

        with patch.object(
            FakeCursor,
            "_insert_attempt_transition",
            return_value=None,
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.transition_attempt(
                    attempt.attempt_id,
                    from_states={TaskAttemptState.CREATED},
                    to_state=TaskAttemptState.RUNNING,
                    reason="started",
                )

        self.assertEqual((await store.get_run(run.run_id)).state, run.state)

    async def test_attempt_must_belong_to_run_for_events_and_usage(
        self,
    ) -> None:
        store = await _registered_store()
        run = await _created_run(store)
        other_run, other_attempt = await _created_attempt(store)

        with self.assertRaises(TaskStoreNotFoundError):
            await store.append_event(
                run.run_id,
                attempt_id=other_attempt.attempt_id,
                event_type="model_complete",
                category=pgsql_store_module.TaskEventCategory.MODEL,
                payload={},
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await store.list_events(
                run.run_id,
                attempt_id=other_attempt.attempt_id,
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await store.append_usage(
                run.run_id,
                attempt_id=other_attempt.attempt_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await store.list_usage(
                run.run_id,
                attempt_id=other_attempt.attempt_id,
            )
        self.assertNotEqual(run.run_id, other_run.run_id)

    async def test_event_and_usage_insert_conflicts(self) -> None:
        store = await _registered_store()
        run, attempt = await _created_attempt(store)

        with patch.object(FakeCursor, "_insert_event", return_value=None):
            with self.assertRaises(TaskStoreConflictError):
                await store.append_event(
                    run.run_id,
                    attempt_id=attempt.attempt_id,
                    event_type="model_complete",
                    category=pgsql_store_module.TaskEventCategory.MODEL,
                    payload={},
                )

        with patch.object(FakeCursor, "_insert_usage", return_value=None):
            with self.assertRaises(TaskStoreConflictError):
                await store.append_usage(
                    run.run_id,
                    attempt_id=attempt.attempt_id,
                    source=UsageSource.EXACT,
                    totals=UsageTotals(input_tokens=1),
                )

    async def test_artifact_transition_conflict_branch(self) -> None:
        store = await _registered_store()
        run = await _created_run(store)
        await store.append_artifact(
            run.run_id,
            ref=_artifact_ref(),
            purpose=TaskArtifactPurpose.OUTPUT,
        )

        with patch.object(
            FakeCursor, "_transition_artifact", return_value=None
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.transition_artifact(
                    "artifact-1",
                    from_states={TaskArtifactState.READY},
                    to_state=TaskArtifactState.DELETED,
                    reason="retention",
                )

    async def test_idempotency_insert_none_branches(self) -> None:
        store = await _registered_store()
        run = await _created_run(store)
        identity = _identity()
        original_insert = FakeCursor._insert_idempotency

        with self.assertRaises(AssertionError):
            await store.reserve_idempotency_key(
                identity,
                run_id=run.run_id,
                expires_at=cast(Any, "bad"),
            )

        self.assertIsNone(await store.lookup_idempotency_key(identity))

        def insert_existing(self: FakeCursor, params: tuple[object, ...]):
            original_insert(self, params)
            return None

        with patch.object(FakeCursor, "_insert_idempotency", insert_existing):
            reserved = await store.reserve_idempotency_key(
                identity,
                run_id=run.run_id,
            )
        self.assertFalse(reserved.created)

        store = await _registered_store()
        run = await _created_run(store)
        with patch.object(
            FakeCursor,
            "_insert_idempotency",
            return_value=None,
        ):
            with self.assertRaises(TaskStoreConflictError):
                await store.reserve_idempotency_key(
                    identity,
                    run_id=run.run_id,
                )

    async def test_attempt_segment_creation_conflict_boundaries(self) -> None:
        claim = _durable_claim()
        run = _durable_run(claim=claim)
        running_attempt = _durable_attempt(claim=claim)
        terminal_attempt = _durable_attempt(
            state=TaskAttemptState.ABANDONED,
            claim=claim,
        )
        active_segment = _durable_segment(claim=claim)
        suspended_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        cases = (
            (
                terminal_attempt,
                (),
                None,
                "terminal task attempt cannot create a segment",
            ),
            (
                running_attempt,
                (_durable_segment_row(active_segment),),
                None,
                "task attempt already has an active segment",
            ),
            (
                running_attempt,
                (_durable_segment_row(suspended_segment),),
                "wrong-segment",
                "resumed task segment was not suspended",
            ),
            (
                _durable_attempt(
                    state=TaskAttemptState.SUSPENDED,
                    claim=claim,
                ),
                (_durable_segment_row(suspended_segment),),
                None,
                "suspended attempt requires a previous segment",
            ),
        )

        for attempt, segment_rows, resumed_from, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                store, _ = _scripted_store(row_sets=(segment_rows,))
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                ):
                    with self.assertRaisesRegex(
                        TaskStoreConflictError,
                        diagnostic,
                    ):
                        await store.create_attempt_segment(
                            attempt.attempt_id,
                            claim_token=claim.claim_token,
                            resumed_from_segment_id=resumed_from,
                        )

        store, _ = _scripted_store(rows=(None,), row_sets=((),))
        with (
            patch.object(
                pgsql_store_module,
                "_lock_attempt_or_raise",
                new=AsyncMock(return_value=running_attempt),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
        ):
            with self.assertRaisesRegex(
                TaskStoreConflictError,
                "task attempt segment already exists",
            ):
                await store.create_attempt_segment(
                    running_attempt.attempt_id,
                    claim_token=claim.claim_token,
                )

    async def test_attempt_segment_queries_and_transition_guards(self) -> None:
        claim = _durable_claim()
        run = _durable_run(claim=claim)
        created_segment = _durable_segment(
            state=TaskAttemptSegmentState.CREATED,
            claim=claim,
        )
        running_segment = replace(
            created_segment,
            state=TaskAttemptSegmentState.RUNNING,
        )

        store, _ = _scripted_store()
        with patch.object(
            pgsql_store_module,
            "_segment_or_raise",
            new=AsyncMock(return_value=created_segment),
        ):
            self.assertEqual(
                await store.get_attempt_segment(created_segment.segment_id),
                created_segment,
            )

        store, _ = _scripted_store(row_sets=((),))
        with patch.object(
            pgsql_store_module,
            "_segment_or_raise",
            new=AsyncMock(return_value=created_segment),
        ):
            self.assertEqual(
                await store.list_attempt_segment_transitions(
                    created_segment.segment_id
                ),
                (),
            )

        store, _ = _scripted_store()
        invalid_arguments = (
            (
                {
                    "from_states": {TaskAttemptSegmentState.RUNNING},
                    "to_state": TaskAttemptSegmentState.SUSPENDED,
                    "reason": "suspend",
                    "request_id": "request-1",
                    "continuation_id": "continuation-1",
                },
                "suspended segments require a checkpoint identifier",
            ),
            (
                {
                    "from_states": {TaskAttemptSegmentState.RUNNING},
                    "to_state": TaskAttemptSegmentState.SUCCEEDED,
                    "reason": "complete",
                    "request_id": "request-1",
                },
                "request and continuation identifiers must be paired",
            ),
            (
                {
                    "from_states": {TaskAttemptSegmentState.RUNNING},
                    "to_state": TaskAttemptSegmentState.SUSPENDED,
                    "reason": "suspend",
                    "continuation_id": "continuation-1",
                    "checkpoint_id": "checkpoint-1",
                },
                "checkpoint identifiers require interaction correlation",
            ),
        )
        for arguments, diagnostic in invalid_arguments:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(AssertionError, diagnostic):
                    await store.transition_attempt_segment(
                        created_segment.segment_id,
                        **cast(Any, arguments),
                    )

        store, _ = _scripted_store()
        with (
            patch.object(
                pgsql_store_module,
                "_lock_segment_or_raise",
                new=AsyncMock(return_value=created_segment),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
        ):
            with self.assertRaisesRegex(
                TaskStoreConflictError,
                "only suspended segments retain interactions",
            ):
                await store.transition_attempt_segment(
                    created_segment.segment_id,
                    from_states={TaskAttemptSegmentState.CREATED},
                    to_state=TaskAttemptSegmentState.RUNNING,
                    reason="started",
                    request_id="request-1",
                    continuation_id="continuation-1",
                    claim_token=claim.claim_token,
                )

        store, _ = _scripted_store(rows=(None,))
        with (
            patch.object(
                pgsql_store_module,
                "_lock_segment_or_raise",
                new=AsyncMock(return_value=created_segment),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
        ):
            with self.assertRaisesRegex(
                TaskStoreConflictError,
                "task attempt segment state did not match",
            ):
                await store.transition_attempt_segment(
                    created_segment.segment_id,
                    from_states={TaskAttemptSegmentState.CREATED},
                    to_state=TaskAttemptSegmentState.RUNNING,
                    reason="started",
                    claim_token=claim.claim_token,
                )

        store, _ = _scripted_store(
            rows=(_durable_segment_row(running_segment), None)
        )
        with (
            patch.object(
                pgsql_store_module,
                "_lock_segment_or_raise",
                new=AsyncMock(return_value=created_segment),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
        ):
            with self.assertRaisesRegex(
                TaskStoreConflictError,
                "task attempt segment transition was not recorded",
            ):
                await store.transition_attempt_segment(
                    created_segment.segment_id,
                    from_states={TaskAttemptSegmentState.CREATED},
                    to_state=TaskAttemptSegmentState.RUNNING,
                    reason="started",
                    claim_token=claim.claim_token,
                )

    async def test_attempt_segment_can_transition_to_suspended(self) -> None:
        claim = _durable_claim()
        run = _durable_run(claim=claim)
        running_segment = _durable_segment(claim=claim)
        suspended_segment = replace(
            running_segment,
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        store, _ = _scripted_store(
            rows=(_durable_segment_row(suspended_segment), {})
        )

        with (
            patch.object(
                pgsql_store_module,
                "_lock_segment_or_raise",
                new=AsyncMock(return_value=running_segment),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
        ):
            result = await store.transition_attempt_segment(
                running_segment.segment_id,
                from_states={TaskAttemptSegmentState.RUNNING},
                to_state=TaskAttemptSegmentState.SUSPENDED,
                reason="input required",
                request_id="request-1",
                continuation_id="continuation-1",
                checkpoint_id="checkpoint-1",
                claim_token=claim.claim_token,
            )

        self.assertEqual(result, suspended_segment)

    async def test_suspend_claim_public_contract_and_delegation(self) -> None:
        store, _ = _scripted_store()
        valid = {
            "queue_item_id": "queue-1",
            "claim_token": "claim-token",
            "segment_id": "segment-1",
            "request_id": "request-1",
            "continuation_id": "continuation-1",
            "checkpoint_id": "checkpoint-1",
        }
        for field in (
            "queue_item_id",
            "claim_token",
            "segment_id",
            "request_id",
            "continuation_id",
            "checkpoint_id",
        ):
            with self.subTest(field=field):
                arguments = dict(valid)
                arguments[field] = ""
                with self.assertRaises(AssertionError):
                    await store.suspend_claim(**arguments)

        without_checkpoint = dict(valid)
        without_checkpoint.pop("checkpoint_id")
        with self.assertRaisesRegex(
            AssertionError,
            "requires a checkpoint identifier",
        ):
            await store.suspend_claim(**without_checkpoint)

        sentinel = cast(Any, object())
        with patch.object(
            store,
            "_suspend_claim_in_unit",
            new=AsyncMock(return_value=sentinel),
        ) as suspend:
            self.assertIs(
                await store.suspend_claim(
                    **valid,
                    now=_DURABLE_NOW,
                    metadata={"safe": True},
                ),
                sentinel,
            )
        suspend.assert_awaited_once()

    async def test_suspend_claim_rejects_each_stale_boundary(self) -> None:
        claim = _durable_claim()
        valid_run = _durable_run(claim=claim)
        valid_attempt = _durable_attempt(claim=claim)
        valid_segment = _durable_segment(claim=claim)
        queue_row = _durable_queue_row()
        cases = (
            (
                (None,),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue item was not found",
            ),
            (
                (_durable_queue_row(claim_token="stale"),),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue claim token did not match",
            ),
            (
                (queue_row,),
                _durable_run(state=TaskRunState.CLAIMED, claim=claim),
                valid_attempt,
                valid_segment,
                "task run is not running",
            ),
            (
                (queue_row,),
                _durable_run(claim=claim, last_attempt_id=None),
                valid_attempt,
                valid_segment,
                "running task has no active attempt",
            ),
            (
                (queue_row,),
                valid_run,
                _durable_attempt(
                    state=TaskAttemptState.CREATED,
                    claim=claim,
                ),
                valid_segment,
                "task attempt is not running",
            ),
            (
                (queue_row,),
                valid_run,
                valid_attempt,
                _durable_segment(
                    state=TaskAttemptSegmentState.CREATED,
                    claim=claim,
                ),
                "task attempt segment is not the active running segment",
            ),
            (
                (queue_row, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "durable interaction checkpoint is not ready",
            ),
            (
                (queue_row, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task run suspension compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task attempt suspension compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task segment suspension compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue suspension compare-and-swap failed",
            ),
        )

        for rows, run, attempt, segment, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._suspend_claim_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            segment_id="segment-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            checkpoint_id="checkpoint-1",
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_suspension_transition_insert_failure_is_explicit(
        self,
    ) -> None:
        claim = _durable_claim()
        store, cursor = _scripted_store(rows=(None,))
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "task suspension transition was not recorded",
        ):
            await pgsql_store_module._insert_suspension_transitions(
                _scripted_unit(cursor),
                id_factory=SequenceIds(),
                run=_durable_run(claim=claim),
                attempt=_durable_attempt(claim=claim),
                segment=_durable_segment(claim=claim),
                now=_DURABLE_NOW,
                metadata={},
            )
        self.assertIsInstance(store, PgsqlTaskStore)

    async def test_requeue_suspended_rejects_invalid_revision(self) -> None:
        store, cursor = _scripted_store()

        with self.assertRaisesRegex(
            AssertionError,
            "resolution_revision must be positive",
        ):
            await store.requeue_suspended(
                "run-1",
                request_id="request-1",
                continuation_id="continuation-1",
                resolution_revision=0,
            )
        with self.assertRaisesRegex(
            AssertionError,
            "interaction_state must be answered or timed_out",
        ):
            await store._requeue_suspended_in_unit(
                _scripted_unit(cursor),
                run_id="run-1",
                request_id="request-1",
                continuation_id="continuation-1",
                resolution_revision=1,
                observed_at=_DURABLE_NOW,
                metadata={},
                interaction_state="declined",
            )

    async def test_requeue_suspended_rejects_each_stale_boundary(self) -> None:
        valid_run = _durable_run(
            state=TaskRunState.INPUT_REQUIRED,
            claim=None,
        )
        valid_attempt = _durable_attempt(
            state=TaskAttemptState.SUSPENDED,
            claim=None,
        )
        valid_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        queue_row = _durable_queue_row(
            state=TaskQueueItemState.SUSPENDED,
        )
        cases = (
            (
                (None,),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue item was not found",
            ),
            (
                (
                    _durable_queue_row(
                        state=TaskQueueItemState.SUSPENDED,
                        attempt_id=None,
                    ),
                ),
                valid_run,
                valid_attempt,
                valid_segment,
                "suspended queue provenance is incomplete",
            ),
            (
                (queue_row,),
                valid_run,
                valid_attempt,
                replace(valid_segment, request_id="other-request"),
                "suspended task provenance did not match",
            ),
            (
                (queue_row, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "accepted interaction resolution was not found",
            ),
            (
                (queue_row, {}),
                replace(valid_run, state=TaskRunState.RUNNING),
                valid_attempt,
                valid_segment,
                "task is not awaiting interaction input",
            ),
            (
                (queue_row, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task requeue compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task attempt reentry compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue reentry compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task requeue transition was not recorded",
            ),
        )

        for rows, run, attempt, segment, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._requeue_suspended_in_unit(
                            _scripted_unit(cursor),
                            run_id="run-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            resolution_revision=1,
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_terminalize_suspended_validates_terminal_contract(
        self,
    ) -> None:
        store, cursor = _scripted_store()
        failure_result = TaskExecutionResult(error={"safe": "failure"})
        invalid_cases = (
            (
                {
                    "run_state": TaskRunState.FAILED,
                    "attempt_state": TaskAttemptState.FAILED,
                    "event_type": TaskInteractionEventType.INPUT_CANCELLED,
                    "correlations": (("request-1", "continuation-1"),),
                },
                "cancelled input requires cancelled run",
            ),
            (
                {
                    "run_state": TaskRunState.FAILED,
                    "attempt_state": TaskAttemptState.FAILED,
                    "event_type": TaskInteractionEventType.INPUT_EXPIRED,
                    "correlations": (("request-1", "continuation-1"),),
                },
                "expired input requires expired run",
            ),
            (
                {
                    "run_state": TaskRunState.FAILED,
                    "attempt_state": TaskAttemptState.FAILED,
                    "event_type": TaskInteractionEventType.INPUT_REQUIRED,
                    "correlations": (("request-1", "continuation-1"),),
                },
                "event type must terminalize suspended task input",
            ),
            (
                {
                    "run_state": TaskRunState.EXPIRED,
                    "attempt_state": TaskAttemptState.FAILED,
                    "event_type": TaskInteractionEventType.INPUT_EXPIRED,
                    "correlations": (("", "continuation-1"),),
                },
                "task correlations must be non-empty",
            ),
            (
                {
                    "run_state": TaskRunState.EXPIRED,
                    "attempt_state": TaskAttemptState.FAILED,
                    "event_type": TaskInteractionEventType.INPUT_EXPIRED,
                    "correlations": (
                        ("request-1", "continuation-1"),
                        ("request-1", "continuation-1"),
                    ),
                },
                "task correlations must be unique",
            ),
        )
        for arguments, diagnostic in invalid_cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(AssertionError, diagnostic):
                    await store._terminalize_suspended_in_unit(
                        _scripted_unit(cursor),
                        task_run_id="run-1",
                        reason="terminal",
                        observed_at=_DURABLE_NOW,
                        metadata={},
                        **cast(Any, arguments),
                    )

        detailed_error = replace(
            pgsql_store_module.TaskError.timeout(),
            details={"safe": "deadline"},
        )
        detailed_store, detailed_cursor = _scripted_store(rows=(None,))
        with (
            patch.object(
                pgsql_store_module.TaskError,
                "timeout",
                return_value=detailed_error,
            ),
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(
                    return_value=_durable_run(
                        state=TaskRunState.INPUT_REQUIRED,
                        claim=None,
                    )
                ),
            ),
        ):
            with self.assertRaises(TaskStoreNotFoundError):
                await detailed_store._terminalize_suspended_in_unit(
                    _scripted_unit(detailed_cursor),
                    task_run_id="run-1",
                    correlations=(("request-1", "continuation-1"),),
                    run_state=TaskRunState.EXPIRED,
                    attempt_state=TaskAttemptState.FAILED,
                    event_type=TaskInteractionEventType.INPUT_EXPIRED,
                    reason="expired",
                    observed_at=_DURABLE_NOW,
                    metadata={"result": failure_result.metadata},
                )

    async def test_terminalize_suspended_rejects_each_stale_boundary(
        self,
    ) -> None:
        valid_run = _durable_run(
            state=TaskRunState.INPUT_REQUIRED,
            claim=None,
        )
        valid_attempt = _durable_attempt(
            state=TaskAttemptState.SUSPENDED,
            claim=None,
        )
        valid_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        queue_row = _durable_queue_row(
            state=TaskQueueItemState.SUSPENDED,
        )
        expiry_cases = (
            (
                (None,),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue item was not found",
            ),
            (
                (
                    _durable_queue_row(
                        state=TaskQueueItemState.SUSPENDED,
                        attempt_id=None,
                    ),
                ),
                valid_run,
                valid_attempt,
                valid_segment,
                "suspended task provenance is incomplete",
            ),
            (
                (
                    _durable_queue_row(
                        state=TaskQueueItemState.SUSPENDED,
                        request_id="other-request",
                    ),
                ),
                valid_run,
                valid_attempt,
                valid_segment,
                "no terminal interaction matched the suspended queue",
            ),
            (
                (queue_row,),
                valid_run,
                replace(valid_attempt, state=TaskAttemptState.RUNNING),
                valid_segment,
                "suspended task lifecycle provenance did not match",
            ),
            (
                (queue_row, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task attempt lifecycle compare-and-swap failed",
            ),
            (
                (queue_row, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task attempt lifecycle transition was not recorded",
            ),
            (
                (queue_row, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task run lifecycle compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task run lifecycle transition was not recorded",
            ),
            (
                (queue_row, {}, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue lifecycle compare-and-swap failed",
            ),
        )
        cancellation_cases = (
            (
                (queue_row, {}, {}, None),
                "task cancellation request compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                "task cancellation request transition was not recorded",
            ),
        )

        for rows, run, attempt, segment, diagnostic in expiry_cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._terminalize_suspended_in_unit(
                            _scripted_unit(cursor),
                            task_run_id="run-1",
                            correlations=(("request-1", "continuation-1"),),
                            run_state=TaskRunState.EXPIRED,
                            attempt_state=TaskAttemptState.FAILED,
                            event_type=TaskInteractionEventType.INPUT_EXPIRED,
                            reason="expired",
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

        for rows, diagnostic in cancellation_cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=valid_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=valid_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=valid_segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        TaskStoreConflictError,
                        diagnostic,
                    ):
                        await store._terminalize_suspended_in_unit(
                            _scripted_unit(cursor),
                            task_run_id="run-1",
                            correlations=(("request-1", "continuation-1"),),
                            run_state=TaskRunState.CANCELLED,
                            attempt_state=TaskAttemptState.ABANDONED,
                            event_type=TaskInteractionEventType.INPUT_CANCELLED,
                            reason="cancelled",
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_suspended_run_validation_rejects_each_boundary(
        self,
    ) -> None:
        valid_run = _durable_run(
            state=TaskRunState.INPUT_REQUIRED,
            claim=None,
        )
        valid_attempt = _durable_attempt(
            state=TaskAttemptState.SUSPENDED,
            claim=None,
        )
        valid_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        queue_row = _durable_queue_row(
            state=TaskQueueItemState.SUSPENDED,
        )
        cases = (
            (
                (None,),
                valid_run,
                valid_attempt,
                valid_segment,
                "task queue item was not found",
            ),
            (
                (queue_row,),
                replace(valid_run, state=TaskRunState.RUNNING),
                valid_attempt,
                valid_segment,
                "task is not at a durable suspended boundary",
            ),
            (
                (queue_row,),
                valid_run,
                replace(valid_attempt, state=TaskAttemptState.RUNNING),
                valid_segment,
                "task suspension provenance did not match",
            ),
        )

        for rows, run, attempt, segment, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._validate_suspended_run_in_unit(
                            _scripted_unit(cursor),
                            task_run_id="run-1",
                        )

    async def test_settle_claim_validates_contract_and_expiry_override(
        self,
    ) -> None:
        store, cursor = _scripted_store()
        success = TaskDurableResumeSuccess(
            result=TaskExecutionResult(output_summary={"ok": True})
        )
        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(error={"safe": "failed"})
        )
        invalid_cases = (
            (
                {"settlement": cast(Any, object())},
                "settlement must be a durable resume success or failure",
            ),
            (
                {
                    "settlement": success,
                    "terminal_run_state": TaskRunState.EXPIRED,
                    "terminal_reason": "expired",
                },
                "only failed resume settlement may expire a task",
            ),
            (
                {
                    "settlement": success,
                    "interaction_request_id": "request-1",
                },
                "interaction event correlation requires an event type",
            ),
            (
                {
                    "settlement": success,
                    "interaction_event_type": (
                        TaskInteractionEventType.INPUT_REQUIRED
                    ),
                    "interaction_request_id": "request-1",
                    "interaction_continuation_id": "continuation-1",
                },
                "expired interaction event requires complete correlation",
            ),
        )
        for overrides, diagnostic in invalid_cases:
            with self.subTest(diagnostic=diagnostic):
                arguments = {
                    "queue_item_id": "queue-1",
                    "claim_token": "claim-token",
                    "segment_id": "segment-1",
                    "task_run_id": "run-1",
                    "settlement": success,
                    "observed_at": _DURABLE_NOW,
                    "metadata": {},
                }
                arguments.update(overrides)
                with self.assertRaisesRegex(AssertionError, diagnostic):
                    await store._settle_claim_in_unit(
                        _scripted_unit(cursor),
                        **cast(Any, arguments),
                    )

        expiring_store, expiring_cursor = _scripted_store(rows=(None,))
        with self.assertRaises(TaskStoreNotFoundError):
            await expiring_store._settle_claim_in_unit(
                _scripted_unit(expiring_cursor),
                queue_item_id="queue-1",
                claim_token="claim-token",
                segment_id="segment-1",
                task_run_id="run-1",
                settlement=failure,
                observed_at=_DURABLE_NOW,
                metadata={},
                terminal_run_state=TaskRunState.EXPIRED,
                terminal_reason="expired",
            )

    async def test_settle_claim_rejects_each_stale_boundary(self) -> None:
        claim = _durable_claim()
        success = TaskDurableResumeSuccess(
            result=TaskExecutionResult(output_summary={"ok": True})
        )
        valid_run = _durable_run(claim=claim)
        valid_attempt = _durable_attempt(claim=claim)
        valid_segment = _durable_segment(claim=claim)
        queue_row = _durable_queue_row()
        cases = (
            (
                (None,),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task queue item was not found",
            ),
            (
                (_durable_queue_row(run_id="other-run"),),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task queue item does not belong to the resumed run",
            ),
            (
                (queue_row,),
                _durable_run(claim=claim, last_attempt_id=None),
                valid_attempt,
                valid_segment,
                {},
                "resumed task has no active attempt",
            ),
            (
                (queue_row,),
                valid_run,
                valid_attempt,
                replace(valid_segment, attempt_id="other-attempt"),
                {},
                "resumed task settlement provenance did not match",
            ),
            (
                (queue_row,),
                valid_run,
                valid_attempt,
                valid_segment,
                {"replay_only": True},
                "completed continuation has no matching terminal task",
            ),
            (
                (_durable_queue_row(claim_token="stale"),),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task queue claim token did not match",
            ),
            (
                (
                    _durable_queue_row(
                        lease_expires_at=_DURABLE_NOW,
                    ),
                ),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task queue claim lease expired",
            ),
            (
                (queue_row,),
                _durable_run(state=TaskRunState.CLAIMED, claim=claim),
                valid_attempt,
                valid_segment,
                {},
                "resumed task run is not running",
            ),
            (
                (queue_row,),
                valid_run,
                replace(valid_attempt, state=TaskAttemptState.CREATED),
                valid_segment,
                {},
                "resumed task attempt is not running",
            ),
            (
                (queue_row,),
                valid_run,
                valid_attempt,
                replace(
                    valid_segment,
                    state=TaskAttemptSegmentState.CREATED,
                ),
                {},
                "resumed task segment is not running",
            ),
            (
                (queue_row,),
                valid_run,
                valid_attempt,
                replace(valid_segment, claim=None),
                {},
                "resumed task segment claim did not match",
            ),
            (
                (queue_row, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task segment settlement compare-and-swap failed",
            ),
            (
                (queue_row, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task segment settlement transition was not recorded",
            ),
            (
                (queue_row, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task attempt settlement compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task attempt settlement transition was not recorded",
            ),
            (
                (queue_row, {}, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task run settlement compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task run settlement transition was not recorded",
            ),
            (
                (queue_row, {}, {}, {}, {}, {}, {}, None),
                valid_run,
                valid_attempt,
                valid_segment,
                {},
                "task queue settlement compare-and-swap failed",
            ),
        )

        for (
            rows,
            run,
            attempt,
            segment,
            overrides,
            diagnostic,
        ) in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=segment),
                    ),
                ):
                    arguments = {
                        "queue_item_id": "queue-1",
                        "claim_token": claim.claim_token,
                        "segment_id": "segment-1",
                        "task_run_id": "run-1",
                        "settlement": success,
                        "observed_at": _DURABLE_NOW,
                        "metadata": {},
                    }
                    arguments.update(overrides)
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._settle_claim_in_unit(
                            _scripted_unit(cursor),
                            **cast(Any, arguments),
                        )

    async def test_settle_claim_cancellation_cas_failures(self) -> None:
        claim = _durable_claim()
        settlement = TaskDurableResumeCancellation(
            result=TaskExecutionResult(error={"safe": "cancelled"})
        )
        valid_run = _durable_run(claim=claim)
        valid_attempt = _durable_attempt(claim=claim)
        valid_segment = _durable_segment(claim=claim)
        queue_row = _durable_queue_row()
        cases = (
            (
                (queue_row, {}, {}, {}, {}, None),
                "task cancellation request compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, {}, {}, None),
                "task cancellation request transition was not recorded",
            ),
        )

        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=valid_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=valid_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=valid_segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        TaskStoreConflictError,
                        diagnostic,
                    ):
                        await store._settle_claim_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            segment_id="segment-1",
                            task_run_id="run-1",
                            settlement=settlement,
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_expired_settlement_records_interaction_event(self) -> None:
        claim = _durable_claim()
        result = TaskExecutionResult(error={"safe": "expired"})
        settlement = TaskDurableResumeFailure(result=result)
        run = _durable_run(claim=claim)
        attempt = _durable_attempt(claim=claim)
        segment = _durable_segment(claim=claim)
        terminal_run = replace(
            run,
            state=TaskRunState.EXPIRED,
            claim=None,
            result=result,
        )
        terminal_attempt = replace(
            attempt,
            state=TaskAttemptState.FAILED,
            result=result,
        )
        terminal_segment = replace(
            segment,
            state=TaskAttemptSegmentState.FAILED,
        )
        terminal_queue = _durable_queue_row(state=TaskQueueItemState.DEAD)
        store, cursor = _scripted_store(
            rows=(
                _durable_queue_row(),
                _durable_segment_row(terminal_segment),
                {},
                _durable_attempt_row(terminal_attempt),
                {},
                _durable_run_row(terminal_run),
                {},
                terminal_queue,
            )
        )
        with (
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_attempt_or_raise",
                new=AsyncMock(return_value=attempt),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_segment_or_raise",
                new=AsyncMock(return_value=segment),
            ),
            patch.object(
                pgsql_store_module,
                "_insert_interaction_event",
                new=AsyncMock(),
            ) as insert_event,
        ):
            completion = await store._settle_claim_in_unit(
                _scripted_unit(cursor),
                queue_item_id="queue-1",
                claim_token=claim.claim_token,
                segment_id="segment-1",
                task_run_id="run-1",
                settlement=settlement,
                observed_at=_DURABLE_NOW,
                metadata={},
                terminal_run_state=TaskRunState.EXPIRED,
                terminal_reason="expired",
                interaction_event_type=TaskInteractionEventType.INPUT_EXPIRED,
                interaction_request_id="request-1",
                interaction_continuation_id="continuation-1",
            )

        self.assertEqual(completion.run.state, TaskRunState.EXPIRED)
        insert_event.assert_awaited_once()

    async def test_completed_claim_terminalization_rejects_bad_provenance(
        self,
    ) -> None:
        claim = _durable_claim()
        result = TaskExecutionResult(error={"safe": "failed"})
        failure = TaskDurableResumeFailure(result=result)
        success = TaskDurableResumeSuccess(
            result=TaskExecutionResult(output_summary={"ok": True})
        )
        run = _durable_run(claim=claim)
        attempt = _durable_attempt(claim=claim)
        previous_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        active_segment = _durable_segment(
            segment_id="segment-2",
            claim=claim,
            resumed_from_segment_id=previous_segment.segment_id,
        )
        queue_row = _durable_queue_row()

        store, cursor = _scripted_store()
        with self.assertRaisesRegex(
            AssertionError,
            "completed provider task settlement must not succeed",
        ):
            await store._terminalize_completed_claim_in_unit(
                _scripted_unit(cursor),
                queue_item_id="queue-1",
                claim_token=claim.claim_token,
                segment_id=active_segment.segment_id,
                task_run_id="run-1",
                request_id="request-1",
                continuation_id="continuation-1",
                checkpoint_id="checkpoint-1",
                settlement=cast(Any, success),
                observed_at=_DURABLE_NOW,
                metadata={},
            )

        cases = (
            (
                (None,),
                run,
                attempt,
                (active_segment, previous_segment),
                "task queue item was not found",
            ),
            (
                (_durable_queue_row(run_id="other-run"),),
                run,
                attempt,
                (active_segment, previous_segment),
                "task queue item does not belong to the resumed run",
            ),
            (
                (queue_row,),
                _durable_run(claim=claim, last_attempt_id=None),
                attempt,
                (active_segment, previous_segment),
                "resumed task has no active attempt",
            ),
            (
                (_durable_queue_row(segment_id=None),),
                run,
                attempt,
                (active_segment, previous_segment),
                "completed provider task has no suspension provenance",
            ),
            (
                (
                    _durable_queue_row(
                        request_id="other-request",
                    ),
                ),
                run,
                attempt,
                (active_segment, previous_segment),
                "completed provider task provenance did not match",
            ),
        )

        for rows, locked_run, locked_attempt, segments, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=locked_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=locked_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(side_effect=segments),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._terminalize_completed_claim_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            segment_id=active_segment.segment_id,
                            task_run_id="run-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            checkpoint_id="checkpoint-1",
                            settlement=failure,
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_release_claimed_reentry_rejects_stale_boundaries(
        self,
    ) -> None:
        claim = _durable_claim()
        run = _durable_run(state=TaskRunState.CLAIMED, claim=claim)
        attempt = _durable_attempt(
            state=TaskAttemptState.SUSPENDED,
            claim=claim,
        )
        segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        queue_row = _durable_queue_row()
        cases = (
            (
                (None,),
                run,
                attempt,
                segment,
                "durable continuation is not safe to requeue",
            ),
            (
                ({}, None),
                run,
                attempt,
                segment,
                "task queue item was not found",
            ),
            (
                ({}, _durable_queue_row(run_id="other-run")),
                run,
                attempt,
                segment,
                "task queue item does not belong to the resumed run",
            ),
            (
                ({}, queue_row),
                replace(run, last_attempt_id=None),
                attempt,
                segment,
                "resumed task has no active attempt",
            ),
            (
                ({}, _durable_queue_row(segment_id=None)),
                run,
                attempt,
                segment,
                "claimed reentry has no suspended segment",
            ),
            (
                ({}, queue_row),
                replace(run, state=TaskRunState.RUNNING),
                attempt,
                segment,
                "task reentry claim did not match",
            ),
            (
                ({}, queue_row, None),
                run,
                attempt,
                segment,
                "task reentry attempt release compare-and-swap failed",
            ),
            (
                ({}, queue_row, {}, None),
                run,
                attempt,
                segment,
                "task reentry run release compare-and-swap failed",
            ),
            (
                ({}, queue_row, {}, {}, None),
                run,
                attempt,
                segment,
                "task reentry queue release compare-and-swap failed",
            ),
            (
                ({}, queue_row, {}, {}, {}, None),
                run,
                attempt,
                segment,
                "task reentry release transition was not recorded",
            ),
        )

        for (
            rows,
            locked_run,
            locked_attempt,
            locked_segment,
            diagnostic,
        ) in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=locked_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=locked_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=locked_segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._release_claimed_reentry_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            task_run_id="run-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            checkpoint_id="checkpoint-1",
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_fail_claimed_reentry_rejects_stale_boundaries(
        self,
    ) -> None:
        claim = _durable_claim()
        result = TaskExecutionResult(error={"safe": "failed"})
        run = _durable_run(state=TaskRunState.CLAIMED, claim=claim)
        attempt = _durable_attempt(
            state=TaskAttemptState.SUSPENDED,
            claim=claim,
        )
        segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        queue_row = _durable_queue_row()

        store, cursor = _scripted_store()
        assertions = (
            (
                {
                    "terminal_run_state": TaskRunState.CANCELLED,
                },
                "failed reentry must fail or expire its task",
            ),
            (
                {
                    "terminal_run_state": TaskRunState.FAILED,
                    "interaction_event_type": (
                        TaskInteractionEventType.INPUT_EXPIRED
                    ),
                },
                "expired reentry event requires an expired task",
            ),
        )
        for overrides, diagnostic in assertions:
            with self.subTest(diagnostic=diagnostic):
                arguments = {
                    "queue_item_id": "queue-1",
                    "claim_token": claim.claim_token,
                    "task_run_id": "run-1",
                    "request_id": "request-1",
                    "continuation_id": "continuation-1",
                    "checkpoint_id": "checkpoint-1",
                    "result": result,
                    "reason": "failed",
                    "observed_at": _DURABLE_NOW,
                    "metadata": {},
                }
                arguments.update(overrides)
                with self.assertRaisesRegex(AssertionError, diagnostic):
                    await store._fail_claimed_reentry_in_unit(
                        _scripted_unit(cursor),
                        **cast(Any, arguments),
                    )

        cases = (
            (
                (None,),
                run,
                attempt,
                segment,
                {},
                "task queue item was not found",
            ),
            (
                (_durable_queue_row(run_id="other-run"),),
                run,
                attempt,
                segment,
                {},
                "task queue item does not belong to the resumed run",
            ),
            (
                (queue_row,),
                replace(run, last_attempt_id=None),
                attempt,
                segment,
                {},
                "resumed task has no active attempt",
            ),
            (
                (_durable_queue_row(segment_id=None),),
                run,
                attempt,
                segment,
                {},
                "claimed reentry has no suspended segment",
            ),
            (
                (queue_row,),
                run,
                attempt,
                segment,
                {"replay_only": True},
                "invalidated continuation has no matching failed task",
            ),
            (
                (queue_row,),
                replace(run, state=TaskRunState.RUNNING),
                attempt,
                segment,
                {},
                "task reentry claim did not match",
            ),
            (
                (queue_row, None),
                run,
                attempt,
                segment,
                {},
                "task reentry failure attempt compare-and-swap failed",
            ),
            (
                (queue_row, {}, None),
                run,
                attempt,
                segment,
                {},
                "task reentry failure transition was not recorded",
            ),
            (
                (queue_row, {}, {}, None),
                run,
                attempt,
                segment,
                {},
                "task reentry failure run compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                run,
                attempt,
                segment,
                {},
                "task reentry failure run transition was not recorded",
            ),
            (
                (queue_row, {}, {}, {}, {}, None),
                run,
                attempt,
                segment,
                {},
                "task reentry failure queue compare-and-swap failed",
            ),
        )
        for (
            rows,
            locked_run,
            locked_attempt,
            locked_segment,
            overrides,
            diagnostic,
        ) in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=locked_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=locked_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=locked_segment),
                    ),
                ):
                    arguments = {
                        "queue_item_id": "queue-1",
                        "claim_token": claim.claim_token,
                        "task_run_id": "run-1",
                        "request_id": "request-1",
                        "continuation_id": "continuation-1",
                        "checkpoint_id": "checkpoint-1",
                        "result": result,
                        "reason": "failed",
                        "observed_at": _DURABLE_NOW,
                        "metadata": {},
                    }
                    arguments.update(overrides)
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._fail_claimed_reentry_in_unit(
                            _scripted_unit(cursor),
                            **cast(Any, arguments),
                        )

        event_store, event_cursor = _scripted_store(
            rows=(queue_row, {}, {}, {}, {}, {})
        )
        with (
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
            patch.object(
                pgsql_store_module,
                "_lock_attempt_or_raise",
                new=AsyncMock(return_value=attempt),
            ),
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "expired reentry event requires exact provenance",
            ):
                await event_store._fail_claimed_reentry_in_unit(
                    _scripted_unit(event_cursor),
                    queue_item_id="queue-1",
                    claim_token=claim.claim_token,
                    task_run_id="run-1",
                    request_id=None,
                    continuation_id=None,
                    checkpoint_id=None,
                    result=result,
                    reason="expired",
                    observed_at=_DURABLE_NOW,
                    metadata={},
                    terminal_run_state=TaskRunState.EXPIRED,
                    interaction_event_type=(
                        TaskInteractionEventType.INPUT_EXPIRED
                    ),
                )

    async def test_release_running_reentry_rejects_stale_boundaries(
        self,
    ) -> None:
        claim = _durable_claim()
        run = _durable_run(claim=claim)
        attempt = _durable_attempt(claim=claim)
        segment = _durable_segment(claim=claim)
        queue_row = _durable_queue_row()
        cases = (
            (
                (None,),
                run,
                attempt,
                segment,
                "durable continuation is not safe to requeue",
            ),
            (
                ({}, None),
                run,
                attempt,
                segment,
                "task queue item was not found",
            ),
            (
                ({}, _durable_queue_row(run_id="other-run")),
                run,
                attempt,
                segment,
                "task queue item does not belong to the resumed run",
            ),
            (
                ({}, queue_row),
                replace(run, last_attempt_id=None),
                attempt,
                segment,
                "resumed task has no active attempt",
            ),
            (
                ({}, queue_row),
                run,
                attempt,
                replace(segment, attempt_id="other-attempt"),
                "running task reentry provenance did not match",
            ),
            (
                ({}, queue_row),
                replace(run, state=TaskRunState.CLAIMED),
                attempt,
                segment,
                "running task reentry claim did not match",
            ),
            (
                ({}, queue_row, None),
                run,
                attempt,
                segment,
                "running task segment release compare-and-swap failed",
            ),
            (
                ({}, queue_row, {}, None),
                run,
                attempt,
                segment,
                "running task segment release transition was not recorded",
            ),
            (
                ({}, queue_row, {}, {}, None),
                run,
                attempt,
                segment,
                "running task attempt release compare-and-swap failed",
            ),
            (
                ({}, queue_row, {}, {}, {}, None),
                run,
                attempt,
                segment,
                "running task attempt release transition was not recorded",
            ),
            (
                ({}, queue_row, {}, {}, {}, {}, None),
                run,
                attempt,
                segment,
                "running task run release compare-and-swap failed",
            ),
            (
                ({}, queue_row, {}, {}, {}, {}, {}, None),
                run,
                attempt,
                segment,
                "running task run release transition was not recorded",
            ),
            (
                ({}, queue_row, {}, {}, {}, {}, {}, {}, None),
                run,
                attempt,
                segment,
                "running task queue release compare-and-swap failed",
            ),
        )

        for (
            rows,
            locked_run,
            locked_attempt,
            locked_segment,
            diagnostic,
        ) in cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=locked_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=locked_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(return_value=locked_segment),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._release_running_reentry_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            segment_id="segment-1",
                            task_run_id="run-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            checkpoint_id="checkpoint-1",
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_cancel_partial_reentry_rejects_stale_boundaries(
        self,
    ) -> None:
        claim = _durable_claim()
        result = TaskExecutionResult(error={"safe": "cancelled"})
        run = _durable_run(
            state=TaskRunState.CANCEL_REQUESTED,
            claim=claim,
        )
        attempt = _durable_attempt(claim=claim)
        previous_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        active_segment = _durable_segment(
            segment_id="segment-2",
            state=TaskAttemptSegmentState.CREATED,
            claim=claim,
            resumed_from_segment_id=previous_segment.segment_id,
        )
        queue_row = _durable_queue_row()
        boundary_cases = (
            (
                (None,),
                run,
                attempt,
                (previous_segment, active_segment),
                "task queue item was not found",
            ),
            (
                (_durable_queue_row(run_id="other-run"),),
                run,
                attempt,
                (previous_segment, active_segment),
                "task queue item does not belong to the resumed run",
            ),
            (
                (queue_row,),
                replace(run, last_attempt_id=None),
                attempt,
                (previous_segment, active_segment),
                "resumed task has no active attempt",
            ),
            (
                (_durable_queue_row(segment_id=None),),
                run,
                attempt,
                (previous_segment, active_segment),
                "partial task reentry has no suspension provenance",
            ),
            (
                (queue_row,),
                replace(run, state=TaskRunState.RUNNING),
                attempt,
                (previous_segment, active_segment),
                "partial task reentry cancellation claim did not match",
            ),
            (
                (queue_row,),
                run,
                replace(attempt, state=TaskAttemptState.CREATED),
                (previous_segment, previous_segment),
                "partial task reentry startup state did not match",
            ),
            (
                (queue_row,),
                run,
                replace(attempt, state=TaskAttemptState.SUSPENDED),
                (
                    replace(previous_segment, claim=claim),
                    replace(previous_segment, claim=claim),
                ),
                "suspended task reentry segment retained a claim",
            ),
            (
                (queue_row,),
                run,
                attempt,
                (previous_segment, replace(active_segment, claim=None)),
                "partial task reentry startup state did not match",
            ),
        )

        for (
            rows,
            locked_run,
            locked_attempt,
            segments,
            diagnostic,
        ) in boundary_cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=locked_run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=locked_attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(side_effect=segments),
                    ),
                ):
                    with self.assertRaisesRegex(
                        (TaskStoreConflictError, TaskStoreNotFoundError),
                        diagnostic,
                    ):
                        await store._cancel_partial_reentry_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            active_segment_id=active_segment.segment_id,
                            task_run_id="run-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            checkpoint_id="checkpoint-1",
                            result=result,
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

        cas_cases = (
            (
                (queue_row, None),
                "partial task segment cancellation compare-and-swap failed",
            ),
            (
                (queue_row, {}, None),
                (
                    "partial task segment cancellation transition was not "
                    "recorded"
                ),
            ),
            (
                (queue_row, {}, {}, None),
                "partial task attempt cancellation compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, None),
                (
                    "partial task attempt cancellation transition was not "
                    "recorded"
                ),
            ),
            (
                (queue_row, {}, {}, {}, {}, None),
                "partial task run cancellation compare-and-swap failed",
            ),
            (
                (queue_row, {}, {}, {}, {}, {}, None),
                "partial task run cancellation transition was not recorded",
            ),
            (
                (queue_row, {}, {}, {}, {}, {}, {}, None),
                "partial task queue cancellation compare-and-swap failed",
            ),
        )
        for rows, diagnostic in cas_cases:
            with self.subTest(diagnostic=diagnostic):
                store, cursor = _scripted_store(rows=rows)
                with (
                    patch.object(
                        pgsql_store_module,
                        "_lock_run_or_raise",
                        new=AsyncMock(return_value=run),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_attempt_or_raise",
                        new=AsyncMock(return_value=attempt),
                    ),
                    patch.object(
                        pgsql_store_module,
                        "_lock_segment_or_raise",
                        new=AsyncMock(
                            side_effect=(previous_segment, active_segment)
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(
                        TaskStoreConflictError,
                        diagnostic,
                    ):
                        await store._cancel_partial_reentry_in_unit(
                            _scripted_unit(cursor),
                            queue_item_id="queue-1",
                            claim_token=claim.claim_token,
                            active_segment_id=active_segment.segment_id,
                            task_run_id="run-1",
                            request_id="request-1",
                            continuation_id="continuation-1",
                            checkpoint_id="checkpoint-1",
                            result=result,
                            observed_at=_DURABLE_NOW,
                            metadata={},
                        )

    async def test_append_segment_usage_requires_matching_attempt(
        self,
    ) -> None:
        store, _ = _scripted_store()
        with self.assertRaisesRegex(
            AssertionError,
            "segment usage requires an attempt",
        ):
            await store.append_usage(
                "run-1",
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
                segment_id="segment-1",
            )

        run = _durable_run()
        attempt = _durable_attempt()
        segment = replace(_durable_segment(), run_id="other-run")
        with (
            patch.object(
                pgsql_store_module,
                "_lock_run_or_raise",
                new=AsyncMock(return_value=run),
            ),
            patch.object(
                pgsql_store_module,
                "_attempt_or_raise",
                new=AsyncMock(return_value=attempt),
            ),
            patch.object(
                pgsql_store_module,
                "_segment_or_raise",
                new=AsyncMock(return_value=segment),
            ),
        ):
            with self.assertRaisesRegex(
                TaskStoreNotFoundError,
                "task attempt segment was not found for run",
            ):
                await store.append_usage(
                    "run-1",
                    source=UsageSource.EXACT,
                    totals=UsageTotals(input_tokens=1),
                    attempt_id="attempt-1",
                    segment_id="segment-1",
                )

    async def test_transaction_failure_classification(self) -> None:
        cases: tuple[tuple[BaseException, type[BaseException]], ...] = (
            (
                PgsqlOperationError(
                    PgsqlFailure(
                        category=PgsqlFailureCategory.UNIQUE_CONFLICT,
                        code="23505",
                        retryable=False,
                    )
                ),
                TaskStoreConflictError,
            ),
            (
                PgsqlOperationError(
                    PgsqlFailure(
                        category=PgsqlFailureCategory.UNKNOWN,
                        code="pgsql.unknown",
                        retryable=False,
                    )
                ),
                TaskStoreError,
            ),
            (UniqueSqlStateError("duplicate"), TaskStoreConflictError),
            (UnknownSqlStateError("private backend detail"), TaskStoreError),
        )

        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                store = PgsqlTaskStore(cast(Any, ErrorDatabase(error)))
                with self.assertRaises(expected):
                    await store.get_run("run-1")

    async def test_transaction_preserves_control_flow_errors(self) -> None:
        store = await _registered_store()

        async def raise_assertion(unit: object) -> object:
            raise AssertionError("bad test callback")

        async def raise_cancelled(unit: object) -> object:
            raise SystemExit("stop")

        with self.assertRaises(AssertionError):
            await store._transaction(
                operation="assertion",
                callback=cast(
                    Callable[[Any], Awaitable[object]], raise_assertion
                ),
            )

        with self.assertRaises(SystemExit):
            await store._transaction(
                operation="system_exit",
                callback=cast(
                    Callable[[Any], Awaitable[object]], raise_cancelled
                ),
            )


class PgsqlStoreHelperCoverageTest(IsolatedAsyncioTestCase):
    def test_private_helper_branches(self) -> None:
        self.assertEqual(pgsql_store_module._result_to_payload(None), {})
        self.assertEqual(pgsql_store_module._row_value(None, "value", 3), 3)
        self.assertEqual(pgsql_store_module._mapping(None), {})
        self.assertEqual(
            pgsql_store_module._mapping('{"safe": true}')["safe"], True
        )
        now = datetime(2026, 1, 1, tzinfo=UTC)
        self.assertEqual(pgsql_store_module._plain(now), now.isoformat())
        naive = datetime(2026, 1, 1)
        self.assertEqual(
            pgsql_store_module._ensure_aware_utc(naive).tzinfo,
            UTC,
        )
        self.assertIsNotNone(pgsql_store_module._utc_now().tzinfo)
        self.assertTrue(pgsql_store_module._uuid_id())

    def test_definition_payload_round_trips_skills_metadata(self) -> None:
        task = replace(
            definition(),
            skills_config=UntrustedSkillSettings(
                surface=SkillSettingsSurface.TASK,
                read_limits=SkillReadLimits(max_lines_per_read=40),
            ),
            skills_identity={
                "version": "task.skills.v1",
                "enabled_tools": ("skills.read",),
            },
        )

        payload = pgsql_store_module._definition_to_payload(task)
        restored = pgsql_store_module._definition_from_payload(payload)

        self.assertIn("skills_config", payload)
        self.assertIn("skills_identity", payload)
        assert restored.skills_config is not None
        self.assertEqual(
            restored.skills_config.read_limits.max_lines_per_read,
            40,
        )
        assert restored.skills_identity is not None
        self.assertEqual(
            restored.skills_identity["enabled_tools"],
            ("skills.read",),
        )

    async def test_not_found_helpers(self) -> None:
        database = FakePgsqlTaskDatabase()
        unit = PgsqlUnitOfWork(
            connection=cast(
                Any, type("Connection", (), {"cursor": lambda self: None})()
            ),
            cursor=FakeCursor(database),
        )

        with self.assertRaises(TaskStoreNotFoundError):
            await pgsql_store_module._run_or_raise(unit, "missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await pgsql_store_module._attempt_or_raise(unit, "missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await pgsql_store_module._artifact_or_raise(unit, "missing")

    async def test_attempt_and_segment_lock_helpers(self) -> None:
        _, attempt_cursor = _scripted_store(rows=(None,))
        with self.assertRaisesRegex(
            TaskStoreNotFoundError,
            "task attempt was not found",
        ):
            await pgsql_store_module._lock_attempt_or_raise(
                _scripted_unit(attempt_cursor),
                "attempt-1",
            )

        segment = _durable_segment()
        _, segment_cursor = _scripted_store(
            rows=(_durable_segment_row(segment),)
        )
        self.assertEqual(
            await pgsql_store_module._segment_or_raise(
                _scripted_unit(segment_cursor),
                segment.segment_id,
            ),
            segment,
        )

        _, missing_segment_cursor = _scripted_store(rows=(None,))
        with self.assertRaisesRegex(
            TaskStoreNotFoundError,
            "task attempt segment was not found",
        ):
            await pgsql_store_module._segment_or_raise(
                _scripted_unit(missing_segment_cursor),
                "segment-1",
            )

        _, missing_lock_cursor = _scripted_store(rows=(None,))
        with self.assertRaisesRegex(
            TaskStoreNotFoundError,
            "task attempt segment was not found",
        ):
            await pgsql_store_module._lock_segment_or_raise(
                _scripted_unit(missing_lock_cursor),
                "segment-1",
            )

    async def test_interaction_event_insert_failure_is_explicit(self) -> None:
        _, cursor = _scripted_store(rows=({}, None))

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "task interaction event was not recorded",
        ):
            await pgsql_store_module._insert_interaction_event(
                _scripted_unit(cursor),
                id_factory=SequenceIds(),
                run_id="run-1",
                attempt_id="attempt-1",
                event_type=TaskInteractionEventType.INPUT_REQUIRED,
                request_id="request-1",
                continuation_id="continuation-1",
                segment_id="segment-1",
                now=_DURABLE_NOW,
            )

    def test_segment_transition_row_conversion(self) -> None:
        transition = pgsql_store_module._segment_transition_from_row(
            {
                "transition_id": "transition-1",
                "segment_id": "segment-1",
                "attempt_id": "attempt-1",
                "run_id": "run-1",
                "from_state": TaskAttemptSegmentState.CREATED.value,
                "to_state": TaskAttemptSegmentState.RUNNING.value,
                "reason": "started",
                "created_at": _DURABLE_NOW,
                "metadata": {},
            }
        )

        self.assertEqual(transition.transition_id, "transition-1")
        self.assertEqual(transition.to_state, TaskAttemptSegmentState.RUNNING)

    def test_reentry_validation_helpers_reject_mismatches(self) -> None:
        claim = _durable_claim()
        attempt = _durable_attempt(
            state=TaskAttemptState.SUSPENDED,
            claim=None,
        )
        previous_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        queue_row = _durable_queue_row()

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "claimed task reentry provenance did not match",
        ):
            pgsql_store_module._validate_reentry_provenance(
                queue_row={**queue_row, "request_id": "other-request"},
                attempt=attempt,
                segment=previous_segment,
                task_run_id="run-1",
                request_id="request-1",
                continuation_id="continuation-1",
                checkpoint_id="checkpoint-1",
            )

        active_segment = _durable_segment(
            segment_id="segment-2",
            state=TaskAttemptSegmentState.CREATED,
            claim=claim,
            resumed_from_segment_id=previous_segment.segment_id,
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "partial task reentry cancellation provenance did not match",
        ):
            pgsql_store_module._validate_partial_reentry_cancellation_provenance(
                queue_row={**queue_row, "attempt_id": "other-attempt"},
                attempt=attempt,
                previous_segment=previous_segment,
                active_segment=active_segment,
                task_run_id="run-1",
                request_id="request-1",
                continuation_id="continuation-1",
                checkpoint_id="checkpoint-1",
            )

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "partial task reentry cancellation provenance did not match",
        ):
            pgsql_store_module._validate_partial_reentry_cancellation_provenance(
                queue_row=queue_row,
                attempt=attempt,
                previous_segment=previous_segment,
                active_segment=replace(
                    active_segment,
                    resumed_from_segment_id="other-segment",
                ),
                task_run_id="run-1",
                request_id="request-1",
                continuation_id="continuation-1",
                checkpoint_id="checkpoint-1",
            )

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "released running task reentry does not match the replay",
        ):
            pgsql_store_module._validate_released_running_reentry(
                queue_row=queue_row,
                attempt=replace(attempt, state=TaskAttemptState.RUNNING),
                segment=previous_segment,
                request_id="request-1",
                continuation_id="continuation-1",
                checkpoint_id="checkpoint-1",
            )

    def test_reentry_replay_helpers_reject_mismatches(self) -> None:
        claim = _durable_claim()
        result = TaskExecutionResult(error={"safe": "cancelled"})
        cancelled_run = _durable_run(
            state=TaskRunState.CANCELLED,
            claim=None,
            result=result,
        )
        cancelled_attempt = _durable_attempt(
            state=TaskAttemptState.ABANDONED,
            claim=claim,
            result=result,
        )
        previous_segment = _durable_segment(
            state=TaskAttemptSegmentState.SUSPENDED,
            claim=None,
            request_id="request-1",
            continuation_id="continuation-1",
            checkpoint_id="checkpoint-1",
        )
        dead_queue = _durable_queue_row(state=TaskQueueItemState.DEAD)

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "terminal partial task cancellation does not match the replay",
        ):
            pgsql_store_module._replayed_partial_reentry_cancellation(
                queue_row=dead_queue,
                run=cancelled_run,
                attempt=cancelled_attempt,
                previous_segment=previous_segment,
                active_segment=replace(
                    previous_segment,
                    state=TaskAttemptSegmentState.CREATED,
                ),
                result=result,
                claim_token=claim.claim_token,
            )

        active_segment = _durable_segment(
            segment_id="segment-2",
            state=TaskAttemptSegmentState.CREATED,
            claim=claim,
            resumed_from_segment_id=previous_segment.segment_id,
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "terminal partial task cancellation does not match the replay",
        ):
            pgsql_store_module._replayed_partial_reentry_cancellation(
                queue_row=dead_queue,
                run=cancelled_run,
                attempt=cancelled_attempt,
                previous_segment=previous_segment,
                active_segment=active_segment,
                result=result,
                claim_token=claim.claim_token,
            )

        failed_run = _durable_run(
            state=TaskRunState.FAILED,
            claim=None,
            result=result,
        )
        failed_attempt = _durable_attempt(
            state=TaskAttemptState.FAILED,
            claim=claim,
            result=result,
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "terminal task reentry failure does not match the replay",
        ):
            pgsql_store_module._replayed_failed_reentry(
                queue_row=_durable_queue_row(),
                run=failed_run,
                attempt=failed_attempt,
                result=result,
            )

        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "terminal suspended task does not match the replay",
        ):
            pgsql_store_module._replayed_suspended_task_lifecycle(
                queue_row=_durable_queue_row(
                    state=TaskQueueItemState.SUSPENDED,
                ),
                run=failed_run,
                attempt=failed_attempt,
                attempt_state=TaskAttemptState.FAILED,
                segment=previous_segment,
                result=result,
                correlations=(("request-1", "continuation-1"),),
            )

    async def test_claim_token_helper_rejects_token_without_claim(
        self,
    ) -> None:
        store = await _registered_store()
        run = await _created_run(store)

        with self.assertRaises(TaskStoreConflictError):
            pgsql_store_module._verify_claim_token(run, "stale")


if __name__ == "__main__":
    main()
