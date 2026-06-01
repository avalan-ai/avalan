from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

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
from avalan.task import (
    IdempotencyMode,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskAttemptState,
    TaskExecutionRequest,
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskRunState,
    TaskStoreConflictError,
    TaskStoreError,
    TaskStoreNotFoundError,
    UsageSource,
    UsageTotals,
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
            with self.assertRaises(TaskStoreNotFoundError):
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

    async def test_claim_token_helper_rejects_token_without_claim(
        self,
    ) -> None:
        store = await _registered_store()
        run = await _created_run(store)

        with self.assertRaises(TaskStoreConflictError):
            pgsql_store_module._verify_claim_token(run, "stale")


if __name__ == "__main__":
    main()
