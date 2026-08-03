"""Verify durable PostgreSQL store activation and readiness boundaries."""

from asyncio import CancelledError
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from durable_codec_test import _continuation_reference
from phase2_fixtures import authority
from store_conformance_test import (
    _atomic_commit,
    _execution_reservation,
    _execution_stage,
)

import avalan.conversation as conversation
from avalan.conversation.stores import pgsql as pgsql_module
from avalan.pgsql import PgsqlConnection, PgsqlCursor, PgsqlFailure, PgsqlRow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run readiness and lifecycle boundaries on asyncio only."""
    return "asyncio"


class _Cursor:
    def __init__(self, *, revision: str = "20260801_0003") -> None:
        self.revision = revision
        self.query = ""

    async def __aenter__(self) -> "_Cursor":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None

    async def execute(
        self,
        query: str,
        parameters: object = None,
    ) -> None:
        self.query = query

    async def fetchone(self) -> PgsqlRow | None:
        if "avalan_task_alembic_version" in self.query:
            return {"version_num": self.revision}
        if "conversation_store_readiness" in self.query:
            return {
                "schema_version": 1,
                "minimum_reader_version": 1,
                "maximum_reader_version": 2,
                "minimum_writer_version": 1,
                "maximum_writer_version": 2,
                "checkpoint_codec_version": 1,
            }
        return None

    async def fetchall(self) -> Sequence[PgsqlRow]:
        return ()

    async def close(self) -> None:
        return None


class _Connection:
    row_factory: object = None

    def __init__(self, *, revision: str) -> None:
        self.revision = revision

    async def __aenter__(self) -> "_Connection":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None

    def cursor(self) -> AbstractAsyncContextManager[PgsqlCursor]:
        return cast(
            AbstractAsyncContextManager[PgsqlCursor],
            _Cursor(revision=self.revision),
        )

    def transaction(self) -> AbstractAsyncContextManager[object]:
        return cast(AbstractAsyncContextManager[object], self)

    async def set_autocommit(self, value: bool) -> None:
        return None


class _Database:
    def __init__(self, *, revision: str = "20260801_0003") -> None:
        self.revision = revision
        self.open_calls = 0
        self.close_calls = 0

    def connection(self) -> AbstractAsyncContextManager[PgsqlConnection]:
        return cast(
            AbstractAsyncContextManager[PgsqlConnection],
            _Connection(revision=self.revision),
        )

    async def open(self) -> None:
        self.open_calls += 1

    async def aclose(self) -> None:
        self.close_calls += 1


class _FailOnceCloseDatabase(_Database):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise self.error


class _NonCurrentKeyResolver:
    def __init__(self) -> None:
        self.key = conversation.ConversationDataKey(
            key_id="grace-key",
            revision=1,
            status=conversation.ConversationKeyStatus.GRACE,
            key_bytes=b"g" * 32,
        )

    async def current_write_key(
        self,
        authority_digest: conversation.AuthorityDigest,
    ) -> conversation.ConversationDataKey:
        return self.key

    async def read_key(
        self,
        authority_digest: conversation.AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> conversation.ConversationDataKey:
        return self.key


class _RecordingBoundaryHook:
    def __init__(self) -> None:
        self.boundaries: list[conversation.StoreAwaitBoundary] = []

    async def reach(self, boundary: conversation.StoreAwaitBoundary) -> None:
        self.boundaries.append(boundary)


class _FailingBoundaryHook(_RecordingBoundaryHook):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def reach(self, boundary: conversation.StoreAwaitBoundary) -> None:
        await super().reach(boundary)
        if boundary is conversation.StoreAwaitBoundary.CLOSE_SETTLED:
            raise self.error


class _FailingContext:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aenter__(self) -> PgsqlConnection:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None


class _FailingDatabase:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def connection(self) -> AbstractAsyncContextManager[PgsqlConnection]:
        return _FailingContext(self.error)


class _NoCloseDatabase:
    def connection(self) -> AbstractAsyncContextManager[PgsqlConnection]:
        return cast(
            AbstractAsyncContextManager[PgsqlConnection],
            _Connection(revision="20260801_0003"),
        )

    def open(self) -> None:
        return None


class _SyncCloseDatabase(_NoCloseDatabase):
    def aclose(self) -> None:
        return None


def _key_resolver() -> conversation.InMemoryConversationKeyResolver:
    scope = authority()
    return conversation.InMemoryConversationKeyResolver(
        {
            conversation.authority_digest(scope): (
                conversation.ConversationDataKey(
                    key_id="key-1",
                    revision=1,
                    status=conversation.ConversationKeyStatus.CURRENT,
                    key_bytes=b"k" * 32,
                ),
            )
        }
    )


def _store(
    database: _Database,
    *,
    owns_database: bool = True,
) -> conversation.PgsqlConversationStore:
    return conversation.PgsqlConversationStore(
        cast(Any, database),
        key_resolver=_key_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=owns_database,
    )


async def test_store_opens_readiness_and_closes_owned_resources_once() -> None:
    database = _Database()
    store = _store(database)

    with pytest.raises(conversation.ConversationFeatureUnavailableError):
        await store.readiness(authority())
    await store.open()
    await store.open()
    readiness = await store.readiness(authority())
    first_close = await store.close()
    repeated_close = await store.close()

    assert database.open_calls == 1
    assert database.close_calls == 1
    assert readiness.schema_version == 1
    assert readiness.application_version == 2
    assert readiness.key_id == "key-1"
    assert first_close.disposition is conversation.StoreCloseDisposition.CLOSED
    assert repeated_close == first_close
    assert (
        await store.inspect_close()
    ).disposition is conversation.StoreCloseDisposition.CLOSED


async def test_store_close_failure_or_cancellation_remains_retriable(
    record_property: Callable[[str, object], None],
) -> None:
    """Report close state from completed owned resource shutdown."""
    record_property("conversation_acceptance_evidence", "runtime")
    failed_database = _FailOnceCloseDatabase(RuntimeError("close failed"))
    failed_store = _store(failed_database)
    await failed_store.open()
    with pytest.raises(RuntimeError):
        await failed_store.close()
    failed_open = await failed_store.inspect_close()
    assert failed_open.disposition is conversation.StoreCloseDisposition.OPEN
    assert (await failed_store.readiness(authority())).key_id == "key-1"
    failed_settled = await failed_store.close()
    assert failed_settled.disposition is (
        conversation.StoreCloseDisposition.CLOSED
    )
    assert await failed_store.close() == failed_settled
    assert failed_database.close_calls == 2

    cancelled_database = _FailOnceCloseDatabase(CancelledError())
    cancelled_store = _store(cancelled_database)
    await cancelled_store.open()
    with pytest.raises(CancelledError):
        await cancelled_store.close()
    cancelled_open = await cancelled_store.inspect_close()
    assert cancelled_open.disposition is (
        conversation.StoreCloseDisposition.OPEN
    )
    assert (await cancelled_store.readiness(authority())).key_id == "key-1"
    cancelled_settled = await cancelled_store.close()
    assert cancelled_settled.disposition is (
        conversation.StoreCloseDisposition.CLOSED
    )
    assert await cancelled_store.close() == cancelled_settled
    assert cancelled_database.close_calls == 2

    for error in (
        RuntimeError("close settlement failed"),
        CancelledError(),
    ):
        closed_database = _Database()
        hook = _FailingBoundaryHook(error)
        store = conversation.PgsqlConversationStore(
            cast(Any, closed_database),
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
            boundary_hook=hook,
            owns_database=True,
        )
        await store.open()
        with pytest.raises(type(error)):
            await store.close()
        assert closed_database.close_calls == 1
        assert (
            await store.inspect_close()
        ).disposition is conversation.StoreCloseDisposition.CLOSED
        assert (await store.close()).disposition is (
            conversation.StoreCloseDisposition.CLOSED
        )
        assert closed_database.close_calls == 1
        assert hook.boundaries == [
            conversation.StoreAwaitBoundary.CLOSE_BEGIN,
            conversation.StoreAwaitBoundary.CLOSE_SETTLED,
            conversation.StoreAwaitBoundary.CLOSE_STATUS,
        ]


async def test_store_fails_closed_on_migration_and_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_store = _store(_Database(revision="old"))
    with pytest.raises(conversation.ConversationMigrationRequiredError):
        await migration_store.open()
    await migration_store.close()

    database = _Database()
    dependency_store = _store(database)
    monkeypatch.setattr(pgsql_module, "find_spec", lambda _: None)
    with pytest.raises(conversation.ConversationFeatureUnavailableError):
        await dependency_store.open()
    assert database.open_calls == 0
    await dependency_store.close()


def test_store_configuration_is_test_only_bounded_and_redacted() -> None:
    settings = conversation.PgsqlConversationStoreSettings(
        dsn="postgresql://user:secret@example.test/database",
        schema="conversation_test",
        pool_minimum=2,
        pool_maximum=4,
    )

    assert "secret" not in repr(settings)
    assert "redacted" in repr(settings)
    database = settings.database()
    assert database is not None

    invalid_settings: tuple[Mapping[str, object], ...] = (
        {"dsn": ""},
        {"dsn": "postgresql://db", "schema": "x" * 64},
        {"dsn": "postgresql://db", "pool_minimum": 0},
        {
            "dsn": "postgresql://db",
            "pool_minimum": 2,
            "pool_maximum": 1,
        },
        {"dsn": "postgresql://db", "pool_maximum": 65},
        {"dsn": "postgresql://db", "pool_timeout_seconds": 0},
        {"dsn": "postgresql://db", "pool_timeout_seconds": True},
        {"dsn": "postgresql://db", "connect_timeout_seconds": 0},
        {"dsn": "postgresql://db", "statement_timeout_milliseconds": 0},
        {"dsn": "postgresql://db", "lock_timeout_milliseconds": 0},
        {
            "dsn": "postgresql://db",
            "idle_transaction_timeout_milliseconds": 0,
        },
    )
    for values in invalid_settings:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.PgsqlConversationStoreSettings(**values)
    with pytest.raises(conversation.ConversationFeatureUnavailableError):
        conversation.PgsqlConversationStorePolicy(test_only=False)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PgsqlConversationStorePolicy(
            minimum_schema_version=2,
            maximum_schema_version=1,
        )
    invalid_policies: tuple[Mapping[str, object], ...] = (
        {"limits": object()},
        {"application_version": 0},
        {"max_batch_size": 0},
        {"max_batch_size": 101},
        {"check_schema_on_open": 1},
    )
    for values in invalid_policies:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.PgsqlConversationStorePolicy(**values)


def test_content_free_pgsql_support_values_validate_closed_states() -> None:
    now = datetime(2026, 8, 1, 12, tzinfo=UTC)
    point = pgsql_module.PgsqlConversationFaultPoint(
        boundary=pgsql_module.PgsqlConversationFaultBoundary.SQL_BEFORE,
        operation="checkpoint_insert",
        ordinal=1,
    )
    work = pgsql_module.ReconciliationWorkRecord(
        reconciliation_id="reconcile-1",
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        lane_id=conversation.ProviderLaneId("lane-1"),
        work_kind="delete_upstream",
        state=pgsql_module.ReconciliationWorkState.CLAIMED,
        attempts=1,
        upstream_response_id=conversation.UpstreamResponseId("upstream-1"),
        lease_owner="worker-1",
        lease_expires_at=now + timedelta(minutes=1),
    )
    rotation = pgsql_module.KeyRotationReceipt(examined=2, reencrypted=2)
    garbage = pgsql_module.GarbageCollectionReceipt(deleted_payloads=3)

    assert point.ordinal == 1
    assert work.state is pgsql_module.ReconciliationWorkState.CLAIMED
    pending_work = pgsql_module.ReconciliationWorkRecord(
        reconciliation_id="reconcile-pending",
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        lane_id=conversation.ProviderLaneId("lane-1"),
        work_kind="rewrap_payload",
        state=pgsql_module.ReconciliationWorkState.PENDING,
        attempts=0,
    )
    assert pending_work.lease_owner is None
    assert rotation.reencrypted == 2
    assert garbage.deleted_payloads == 3
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.PgsqlConversationFaultPoint(
            boundary=cast(
                pgsql_module.PgsqlConversationFaultBoundary,
                "wrong",
            ),
            operation="operation",
            ordinal=1,
        )
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.PgsqlConversationFaultPoint(
            boundary=pgsql_module.PgsqlConversationFaultBoundary.SQL_BEFORE,
            operation="operation",
            ordinal=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.ReconciliationWorkRecord(
            reconciliation_id="reconcile-1",
            checkpoint_id=conversation.CheckpointId("checkpoint-1"),
            lane_id=conversation.ProviderLaneId("lane-1"),
            work_kind="delete_upstream",
            state=pgsql_module.ReconciliationWorkState.CLAIMED,
            attempts=1,
        )
    invalid_work: tuple[Mapping[str, object], ...] = (
        {"work_kind": "wrong"},
        {"state": "claimed"},
        {"attempts": -1},
    )
    for values in invalid_work:
        with pytest.raises(conversation.ConversationValidationError):
            pgsql_module.ReconciliationWorkRecord(
                reconciliation_id="reconcile-1",
                checkpoint_id=conversation.CheckpointId("checkpoint-1"),
                lane_id=conversation.ProviderLaneId("lane-1"),
                work_kind=cast(str, values.get("work_kind", "rewrap_payload")),
                state=cast(
                    pgsql_module.ReconciliationWorkState,
                    values.get(
                        "state",
                        pgsql_module.ReconciliationWorkState.PENDING,
                    ),
                ),
                attempts=cast(int, values.get("attempts", 1)),
            )
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.ReconciliationWorkRecord(
            reconciliation_id="reconcile-claimed-without-lease",
            checkpoint_id=conversation.CheckpointId("checkpoint-1"),
            lane_id=conversation.ProviderLaneId("lane-1"),
            work_kind="rewrap_payload",
            state=pgsql_module.ReconciliationWorkState.CLAIMED,
            attempts=1,
        )
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.KeyRotationReceipt(examined=1, reencrypted=2)
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.GarbageCollectionReceipt(deleted_payloads=-1)

    invalid_readiness = {
        "schema_version": 1,
        "minimum_reader_version": 1,
        "maximum_reader_version": 2,
        "minimum_writer_version": 1,
        "maximum_writer_version": 2,
        "checkpoint_codec_version": 1,
        "application_version": 2,
        "key_id": "key-1",
        "key_revision": 0,
    }
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module.PgsqlConversationReadiness(**invalid_readiness)


async def test_store_rejects_invalid_public_values_and_malicious_plugins() -> (
    None
):
    invalid = cast(Any, object())
    database = _Database()
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PgsqlConversationStore(
            invalid,
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=invalid,
            cipher=conversation.AesGcmConversationCipher(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=_key_resolver(),
            cipher=invalid,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
            policy=invalid,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PgsqlConversationStore.from_settings(
            invalid,
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
        )

    store = _store(database, owns_database=False)
    calls = (
        lambda: store.readiness(invalid),
        lambda: store.reserve_idempotency(invalid),
        lambda: store.stage_execution(invalid),
        lambda: store.allocate_public_response(invalid),
        lambda: store.commit_atomic(invalid),
        lambda: store.create_head(invalid, authority()),
        lambda: store.load("checkpoint-1", invalid),
        lambda: store.retrieve_output_candidates("checkpoint-1", invalid),
        lambda: store.load_continuation_reference("checkpoint-1", invalid),
        lambda: store.retrieve("response-1", invalid),
        lambda: store.fence_idempotency(invalid, "owner", ambiguous=False),
        lambda: store.abandon_idempotency(invalid, "owner", ambiguous=False),
        lambda: store.reconcile_idempotency(invalid, "owner", ambiguous=False),
        lambda: store.inspect_idempotency_settlement(invalid, "owner"),
        lambda: store.tombstone("response-1", invalid, datetime.now(UTC)),
        lambda: store.delete("response-1", invalid, datetime.now(UTC)),
        lambda: store.list_checkpoints(invalid, cursor=None, limit=1),
        lambda: store.claim_outbox(invalid),
        lambda: store._claim_pending_outbox(invalid, limit=1),
        lambda: store.acknowledge_outbox(invalid, "owner"),
        lambda: store.release_outbox(invalid, "owner"),
        lambda: store.claim_reconciliation(invalid, limit=1),
        lambda: store.acknowledge_reconciliation(invalid, succeeded=True),
        lambda: store.rotate_keys(invalid, limit=1),
        lambda: store.retire_key(
            invalid,
            key_id="key-1",
            revision=1,
            at=datetime.now(UTC),
        ),
    )
    for call in calls:
        with pytest.raises(conversation.ConversationValidationError):
            await call()

    limited_calls = (
        lambda: store.list_checkpoints(authority(), cursor=None, limit=0),
        lambda: store.sweep(datetime.now(UTC), limit=0),
        lambda: store.prune(datetime.now(UTC), limit=0),
        lambda: store._claim_pending_outbox(authority(), limit=0),
        lambda: store.claim_reconciliation(authority(), limit=0),
        lambda: store.garbage_collect(limit=0),
        lambda: store.rotate_keys(authority(), limit=0),
    )
    for call in limited_calls:
        with pytest.raises(conversation.ConversationLimitError):
            await call()

    worker = store.create_outbox_recovery_worker(authority())
    with pytest.raises(conversation.ConversationValidationError):
        await worker.acknowledge(invalid)
    with pytest.raises(conversation.ConversationValidationError):
        store.create_outbox_recovery_worker(invalid)
    with pytest.raises(conversation.ConversationValidationError):
        await store.retire_key(
            authority(),
            key_id="key-1",
            revision=0,
            at=datetime.now(UTC),
        )

    await pgsql_module._NoopPgsqlConversationFaultHook().reach(
        pgsql_module.PgsqlConversationFaultPoint(
            boundary=pgsql_module.PgsqlConversationFaultBoundary.SQL_BEFORE,
            operation="valid",
            ordinal=1,
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await pgsql_module._NoopPgsqlConversationFaultHook().reach(invalid)
    assert (await pgsql_module._UtcConversationClock().now()).tzinfo is UTC


async def test_store_rejects_non_current_write_key_and_corrupt_rows() -> None:
    database = _Database()
    store = conversation.PgsqlConversationStore(
        cast(Any, database),
        key_resolver=_NonCurrentKeyResolver(),
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=False,
    )
    await store.open()
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await store.readiness(authority())

    assert pgsql_module._row_str({"value": "ok"}, "value") == "ok"
    assert pgsql_module._row_optional_str({"value": None}, "value") is None
    assert pgsql_module._row_int({"value": 0}, "value") == 0
    assert pgsql_module._row_optional_int({"value": None}, "value") is None
    assert pgsql_module._row_optional_int({"value": 1}, "value") == 1
    assert pgsql_module._row_bool({"value": True}, "value") is True
    assert (
        pgsql_module._row_optional_datetime({"value": None}, "value") is None
    )
    assert (
        pgsql_module._row_bytes({"value": memoryview(b"value")}, "value")
        == b"value"
    )
    invalid_rows = (
        (pgsql_module._row_str, {"value": ""}),
        (pgsql_module._row_optional_str, {"value": 1}),
        (pgsql_module._row_int, {"value": True}),
        (pgsql_module._row_optional_int, {"value": -1}),
        (pgsql_module._row_bool, {"value": 1}),
        (pgsql_module._row_datetime, {"value": datetime.now()}),
        (pgsql_module._row_optional_datetime, {"value": datetime.now()}),
        (pgsql_module._row_bytes, {"value": b""}),
    )
    for reader, row in invalid_rows:
        with pytest.raises(conversation.ConversationStorageError):
            reader(row, "value")
    with pytest.raises(conversation.ConversationValidationError):
        pgsql_module._validate_time(datetime.now())
    await store.close()


async def test_private_invariant_guards_reject_corrupt_durable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _Database()
    store = _store(database, owns_database=False)
    await store.open()
    commit = _atomic_commit("pgsql-invariants")
    candidate = commit.candidate
    checkpoint = candidate.checkpoint
    prepared = await store._prepare_checkpoint(
        candidate,
        committed_at=commit.committed_at,
        output_candidates=commit.output_candidates,
    )
    cursor = cast(PgsqlCursor, _Cursor())

    unit = pgsql_module.PgsqlConversationUnitOfWork(store, candidate)
    assert unit.continuation_id is None
    assert unit.continuation_state_revision is None
    await unit.rollback()
    with pytest.raises(conversation.ConversationStorageError):
        await unit.commit_in(cast(Any, object()))
    with pytest.raises(conversation.ConversationStorageError):
        unit.settle_committed()
    with pytest.raises(conversation.ConversationValidationError):
        await store.commit_in_unit(cast(Any, object()), candidate)

    with pytest.raises(conversation.ConversationKeyPolicyError):
        non_current = conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=_NonCurrentKeyResolver(),
            cipher=conversation.AesGcmConversationCipher(),
            owns_database=False,
        )
        await non_current._prepare_checkpoint(
            candidate,
            committed_at=commit.committed_at,
            output_candidates=(),
        )
    with pytest.raises(conversation.ConversationStorageError):
        await store._insert_continuation_reference(
            cursor,
            replace(prepared, continuation=prepared.envelope),
        )
    with pytest.raises(conversation.ConversationStorageError):
        await store._insert_continuation_reference(
            cursor,
            replace(
                prepared,
                continuation_reference=_continuation_reference(),
            ),
        )

    with monkeypatch.context() as context:
        context.setattr(store, "_synchronize_write_key", AsyncMock())
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store._insert_checkpoint(cursor, prepared)

    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                return_value={
                    "key_status": (
                        conversation.ConversationKeyStatus.RETIRED.value
                    )
                }
            ),
        )
        with pytest.raises(conversation.ConversationKeyRetiredError):
            await store._synchronize_write_key(
                cursor,
                str(conversation.authority_digest(checkpoint.authority)),
                prepared.key,
            )

    authority_key = str(conversation.authority_digest(checkpoint.authority))
    valid_authority = {"current_generation": 0}
    current_existing = {
        "key_status": conversation.ConversationKeyStatus.CURRENT.value,
        "algorithm": prepared.key.algorithm,
    }
    invalid_key_states = (
        ((None,), conversation.ConversationStorageError),
        (
            (
                valid_authority,
                None,
                {"key_id": "different-generation-key"},
            ),
            conversation.ConversationKeyPolicyError,
        ),
        (
            (
                {
                    "current_generation": prepared.key.revision,
                    "current_key_id": "different-current-key",
                    "current_key_revision": prepared.key.revision,
                },
                current_existing,
                {"key_id": prepared.key.key_id},
            ),
            conversation.ConversationKeyPolicyError,
        ),
        (
            (
                valid_authority,
                {
                    "key_status": (
                        conversation.ConversationKeyStatus.GRACE.value
                    ),
                    "algorithm": "different-algorithm",
                },
                None,
            ),
            conversation.ConversationKeyPolicyError,
        ),
        (
            (valid_authority, None, None, None),
            conversation.ConversationKeyRetiredError,
        ),
        (
            (valid_authority, None, None, {"key_status": "current"}, None),
            conversation.ConversationConflictError,
        ),
    )
    for rows, expected in invalid_key_states:
        with monkeypatch.context() as context:
            context.setattr(store, "_execute", AsyncMock())
            context.setattr(store, "_fetchone", AsyncMock(side_effect=rows))
            with pytest.raises(expected):
                await store._synchronize_write_key(
                    cursor,
                    authority_key,
                    prepared.key,
                )

    limited_store = conversation.PgsqlConversationStore(
        cast(Any, database),
        key_resolver=_key_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        policy=conversation.PgsqlConversationStorePolicy(
            limits=replace(
                conversation.StoreLimits(),
                max_checkpoint_bytes=1,
            )
        ),
        owns_database=False,
    )
    with pytest.raises(conversation.ConversationLimitError):
        limited_store._validate_checkpoint_limits(checkpoint, 2)

    child_identity = replace(
        checkpoint.identity,
        checkpoint_id=conversation.CheckpointId("child-checkpoint"),
        sequence=1,
        parent_checkpoint_id=conversation.CheckpointId("parent-checkpoint"),
        parent_sequence=0,
    )
    child = replace(checkpoint, identity=child_identity)
    valid_parent = {
        "authority_digest": str(
            conversation.authority_digest(checkpoint.authority)
        ),
        "lifecycle_state": "committed",
        "conversation_id": str(checkpoint.identity.conversation_id),
        "checkpoint_sequence": 0,
        "checkpoint_kind": (
            conversation.CheckpointKind.COMPLETED_OUTWARD_TURN.value
        ),
    }
    capacity_cases = (
        ({"record_count": conversation.StoreLimits().max_checkpoints},),
        ({"record_count": 0}, None),
        (
            {"record_count": 0},
            valid_parent,
            {
                "record_count": (
                    conversation.StoreLimits().max_children_per_parent
                )
            },
        ),
    )
    expected_errors = (
        conversation.ConversationLimitError,
        conversation.ConversationAuthorizationError,
        conversation.ConversationLimitError,
    )
    for rows, expected in zip(capacity_cases, expected_errors, strict=True):
        with monkeypatch.context() as context:
            context.setattr(store, "_fetchone", AsyncMock(side_effect=rows))
            with pytest.raises(expected):
                await store._validate_checkpoint_capacity(cursor, child)

    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store._validate_atomic_reservation(
                cursor,
                commit,
                str(conversation.authority_digest(checkpoint.authority)),
            )
        context.setattr(store, "_fetchall", AsyncMock(return_value=()))
        with pytest.raises(conversation.ConversationConflictError):
            await store._validate_atomic_staging(cursor, commit)
        with pytest.raises(conversation.ConversationConflictError):
            await store._validate_atomic_provisional(
                cursor,
                commit,
                str(conversation.authority_digest(checkpoint.authority)),
            )

    no_public = _atomic_commit("pgsql-invariants-no-public")
    object.__setattr__(no_public, "provisional_response_id", None)
    object.__setattr__(no_public, "public_response_id", None)
    object.__setattr__(no_public, "outbox_intent_id", None)
    await store._validate_atomic_provisional(
        cursor,
        no_public,
        str(conversation.authority_digest(checkpoint.authority)),
    )
    assert (
        store._build_pending_outbox(no_public, prepared.checkpoint, None)
        is None
    )

    head_commit = no_public
    object.__setattr__(
        head_commit, "head_id", conversation.NamedHeadId("main")
    )
    object.__setattr__(
        head_commit,
        "expected_head_revision",
        conversation.NamedHeadRevision(0),
    )
    object.__setattr__(
        head_commit.candidate.checkpoint.identity,
        "parent_checkpoint_id",
        conversation.CheckpointId("parent-checkpoint"),
    )
    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store._validate_atomic_head(cursor, head_commit)
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                return_value={
                    "lifecycle_state": "active",
                    "head_revision": 0,
                    "checkpoint_id": "parent-checkpoint",
                }
            ),
        )
        await store._validate_atomic_head(cursor, head_commit)

    await store.close()


async def test_scripted_public_conflict_limit_and_corruption_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _Database()
    store = _store(database, owns_database=False)
    await store.open()
    first = _atomic_commit("pgsql-scripted-first")
    second = _atomic_commit("pgsql-scripted-second")
    reservation = _execution_reservation(first)
    mismatch = _execution_reservation(second)
    with pytest.raises(conversation.ConversationValidationError):
        await store.reserve_idempotency(first.idempotency, execution=mismatch)

    stage = _execution_stage(first, first.output_candidates[0], "owner-1")
    object.__setattr__(
        stage,
        "execution_receipt",
        second.output_candidates[0].execution_receipt,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.stage_execution(stage)

    committed_without_checkpoint = {
        "request_digest": str(first.idempotency.request_digest),
        "execution_digest": pgsql_module.execution_reservation_digest(
            reservation
        ),
        "record_state": conversation.IdempotencyRecordState.COMMITTED.value,
        "checkpoint_id": None,
        "public_response_id": None,
    }
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(return_value=committed_without_checkpoint),
        )
        with pytest.raises(conversation.ConversationStorageError):
            await store.reserve_idempotency(
                first.idempotency,
                execution=reservation,
            )

    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                side_effect=(None, {"record_count": 1}, {"record_count": 0})
            ),
        )
        limited = conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
            policy=conversation.PgsqlConversationStorePolicy(
                limits=replace(
                    conversation.StoreLimits(),
                    max_idempotency_records=1,
                )
            ),
            clock=conversation.DeterministicFakeClock(
                datetime(2026, 8, 1, 12, tzinfo=UTC)
            ),
            owns_database=False,
        )
        limited._opened = True
        context.setattr(limited, "_fetchone", store._fetchone)
        with pytest.raises(conversation.ConversationLimitError):
            await limited.reserve_idempotency(first.idempotency)

    await store.close()


async def test_scripted_execution_and_atomic_race_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _Database()
    store = _store(database, owns_database=False)
    await store.open()
    commit = _atomic_commit("pgsql-scripted-races")
    output = commit.output_candidates[0]
    reservation = _execution_reservation(commit)
    stage = _execution_stage(commit, output, "owner-race")
    now = datetime(2026, 8, 1, 12, tzinfo=UTC)

    expired = {
        "request_digest": str(commit.idempotency.request_digest),
        "execution_digest": pgsql_module.execution_reservation_digest(
            reservation
        ),
        "record_state": conversation.IdempotencyRecordState.IN_PROGRESS.value,
        "owner_token": "expired-owner",
        "lease_expires_at": now,
    }
    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=expired))
        resolution = await store.reserve_idempotency(
            commit.idempotency,
            execution=reservation,
        )
    assert resolution.disposition is conversation.IdempotencyDisposition.FENCED

    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(side_effect=(None, None)),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await store.stage_execution(stage)

    reservation_row = {
        "checkpoint_id": str(stage.identity.checkpoint_id),
        "conversation_id": str(stage.identity.conversation_id),
        "logical_turn_id": str(stage.identity.logical_turn_id),
        "execution_segment_id": str(stage.identity.execution_segment_id),
        "branch_id": str(stage.identity.branch_id),
        "checkpoint_sequence": stage.identity.sequence,
        "parent_checkpoint_id": stage.identity.parent_checkpoint_id,
        "parent_sequence": stage.identity.parent_sequence,
        "binding_digest": str(stage.binding.integrity_digest),
        "lane_mode": stage.mode.value,
        "output_scope": stage.scope.value,
    }
    idempotency_row = {
        "request_digest": str(stage.idempotency.request_digest),
        "owner_token": stage.owner_token,
        "record_state": conversation.IdempotencyRecordState.IN_PROGRESS.value,
    }
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                side_effect=(
                    reservation_row,
                    idempotency_row,
                    {
                        "record_count": (
                            conversation.StoreLimits().max_staged_execution_records
                        )
                    },
                )
            ),
        )
        with pytest.raises(conversation.ConversationLimitError):
            await store.stage_execution(stage)

    allocation = conversation.ProvisionalPublicResponse(
        provisional_response_id=commit.provisional_response_id,
        public_response_id=commit.public_response_id,
        owner_token="owner-race",
        authority_digest=str(
            conversation.authority_digest(commit.idempotency.authority)
        ),
    )
    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store.allocate_public_response(allocation)
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                side_effect=(
                    {
                        "record_state": (
                            conversation.IdempotencyRecordState.IN_PROGRESS.value
                        ),
                        "authority_digest": allocation.authority_digest,
                    },
                    {
                        "provisional_count": (
                            conversation.StoreLimits().max_provisional_responses
                        ),
                        "total_count": 0,
                    },
                )
            ),
        )
        with pytest.raises(conversation.ConversationLimitError):
            await store.allocate_public_response(allocation)

    attestation = conversation.ProviderLaneExecutionAttestation(
        schema_version=1,
        staging_id="stage-expected",
        lane_id=output.lane_id,
    )
    attested = replace(commit, execution_attestations=(attestation,))
    cursor = cast(PgsqlCursor, _Cursor())
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchall",
            AsyncMock(return_value=({"lane_id": "other-lane"},)),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await store._validate_atomic_staging(cursor, attested)
    staging_row = {
        "lane_id": str(output.lane_id),
        "staging_id": "wrong-stage",
        "request_digest": str(commit.idempotency.request_digest),
        "binding_digest": str(output.binding.integrity_digest),
        "execution_digest": str(output.execution_receipt.digest),
        "lane_mode": output.mode.value,
        "output_scope": output.scope.value,
        "item_count": output.execution_receipt.item_count,
        "opaque_byte_count": output.execution_receipt.opaque_byte_count,
    }
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchall",
            AsyncMock(return_value=(staging_row,)),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await store._validate_atomic_staging(cursor, attested)

    prepared = await store._prepare_checkpoint(
        commit.candidate,
        committed_at=commit.committed_at,
        output_candidates=commit.output_candidates,
    )
    object.__setattr__(
        commit.candidate.checkpoint.identity,
        "parent_checkpoint_id",
        conversation.CheckpointId("parent-checkpoint"),
    )
    object.__setattr__(commit, "provisional_response_id", None)
    object.__setattr__(commit, "public_response_id", None)
    object.__setattr__(commit, "outbox_intent_id", None)
    object.__setattr__(commit, "head_id", conversation.NamedHeadId("main"))
    object.__setattr__(
        commit,
        "expected_head_revision",
        conversation.NamedHeadRevision(0),
    )
    with monkeypatch.context() as context:
        context.setattr(
            pgsql_module.InMemoryConversationStore,
            "_validate_atomic_commit_value",
            lambda value: None,
        )
        context.setattr(
            pgsql_module.InMemoryConversationStore,
            "_validate_output_candidates",
            lambda checkpoint, candidates, *, parent: None,
        )
        context.setattr(
            pgsql_module.InMemoryConversationStore,
            "_build_result",
            lambda value, checkpoint: None,
        )
        context.setattr(
            store,
            "_load_checkpoint",
            AsyncMock(return_value=prepared.checkpoint),
        )
        context.setattr(
            store,
            "_prepare_checkpoint",
            AsyncMock(return_value=prepared),
        )
        context.setattr(store, "_validate_atomic_reservation", AsyncMock())
        context.setattr(store, "_validate_atomic_staging", AsyncMock())
        context.setattr(store, "_validate_atomic_provisional", AsyncMock())
        context.setattr(store, "_validate_atomic_head", AsyncMock())
        context.setattr(store, "_insert_checkpoint", AsyncMock())
        receipt = await store.commit_atomic(commit)
    assert receipt.result is None
    await store.close()


async def test_scripted_authority_and_payload_corruption_is_concealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _Database()
    store = _store(database, owns_database=False)
    await store.open()
    commit = _atomic_commit("pgsql-corruption")
    checkpoint = await store._prepare_checkpoint(
        commit.candidate,
        committed_at=commit.committed_at,
        output_candidates=commit.output_candidates,
    )
    persisted = checkpoint.checkpoint
    scope = persisted.authority
    output = commit.output_candidates[0]

    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.create_head(
                conversation.NamedHeadSnapshot(
                    head_id=conversation.NamedHeadId("missing"),
                    revision=conversation.NamedHeadRevision(0),
                    checkpoint_id=persisted.identity.checkpoint_id,
                ),
                scope,
            )
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                side_effect=(
                    {
                        "authority_digest": str(
                            conversation.authority_digest(scope)
                        ),
                        "lifecycle_state": "committed",
                    },
                    {"record_count": conversation.StoreLimits().max_heads},
                )
            ),
        )
        with pytest.raises(conversation.ConversationLimitError):
            await store.create_head(
                conversation.NamedHeadSnapshot(
                    head_id=conversation.NamedHeadId("limited"),
                    revision=conversation.NamedHeadRevision(0),
                    checkpoint_id=persisted.identity.checkpoint_id,
                ),
                scope,
            )
    with monkeypatch.context() as context:
        context.setattr(store, "_read_one", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.load_head(conversation.NamedHeadId("missing"), scope)
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.branch_count(persisted.identity.checkpoint_id, scope)

    checkpoint_row = {
        "lifecycle_state": "committed",
        "execution_segment_id": str(persisted.identity.execution_segment_id),
        "checkpoint_sequence": persisted.identity.sequence,
    }
    with monkeypatch.context() as context:
        context.setattr(
            store, "_read_one", AsyncMock(return_value=checkpoint_row)
        )
        context.setattr(
            store,
            "_validate_payload_reference_row",
            lambda *args, **kwargs: None,
        )
        context.setattr(
            store,
            "_decrypt_payload_row",
            AsyncMock(return_value=store._checkpoint_codec.encode(persisted)),
        )
        with pytest.raises(conversation.ConversationStorageError):
            await store._load_checkpoint(
                conversation.CheckpointId("different-checkpoint"),
                scope,
            )
    with pytest.raises(conversation.ConversationKeyRetiredError):
        await store._decrypt_payload_row(
            {
                "authority_digest": str(conversation.authority_digest(scope)),
                "key_id": "key-1",
                "key_revision": 1,
                "key_status": conversation.ConversationKeyStatus.RETIRED.value,
            }
        )

    with monkeypatch.context() as context:
        context.setattr(store, "_read_all", AsyncMock(return_value=()))
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.retrieve_output_candidates(
                persisted.identity.checkpoint_id,
                scope,
            )
    encoded_output = store._durable_codec.encode_output(output)
    for row in (
        {"payload_sequence": 1, "lane_id": str(output.lane_id)},
        {"payload_sequence": 0, "lane_id": "wrong-lane"},
    ):
        with monkeypatch.context() as context:
            context.setattr(store, "_read_all", AsyncMock(return_value=(row,)))
            context.setattr(
                store,
                "_decrypt_payload_row",
                AsyncMock(return_value=encoded_output),
            )
            with pytest.raises(conversation.ConversationStorageError):
                await store.retrieve_output_candidates(
                    persisted.identity.checkpoint_id,
                    scope,
                )
    for row in (
        {
            "lane_id": str(output.lane_id),
            "registered_lane_id": "different-registered-lane",
        },
        {
            "lane_id": "different-payload-lane",
            "registered_lane_id": "different-payload-lane",
        },
    ):
        with monkeypatch.context() as context:
            context.setattr(store, "_read_all", AsyncMock(return_value=(row,)))
            context.setattr(
                store,
                "_validate_payload_reference_row",
                lambda *args, **kwargs: None,
            )
            context.setattr(
                store,
                "_decrypt_payload_row",
                AsyncMock(return_value=encoded_output),
            )
            with pytest.raises(conversation.ConversationStorageError):
                await store.retrieve_output_candidates(
                    persisted.identity.checkpoint_id,
                    scope,
                )
    with monkeypatch.context() as context:
        context.setattr(store, "_read_one", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.load_continuation_reference(
                persisted.identity.checkpoint_id,
                scope,
            )
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.retrieve(commit.public_response_id, scope)
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_read_one",
            AsyncMock(
                return_value={
                    "conversation_id": "conversation-1",
                    "continuation_conversation_id": "different-conversation",
                }
            ),
        )
        context.setattr(
            store,
            "_validate_payload_reference_row",
            lambda *args, **kwargs: None,
        )
        with pytest.raises(conversation.ConversationStorageError):
            await store.load_continuation_reference(
                persisted.identity.checkpoint_id,
                scope,
            )

    without_integrity = replace(persisted, integrity=None)
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_read_one",
            AsyncMock(
                return_value={
                    "checkpoint_id": str(persisted.identity.checkpoint_id),
                    "tombstoned": False,
                }
            ),
        )
        context.setattr(
            store,
            "_load_checkpoint",
            AsyncMock(return_value=without_integrity),
        )
        context.setattr(
            store,
            "retrieve_output_candidates",
            AsyncMock(return_value=(output,)),
        )
        with pytest.raises(conversation.ConversationStorageError):
            await store.retrieve(commit.public_response_id, scope)
    await store.close()


async def test_scripted_settlement_lifecycle_and_outbox_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _Database()
    store = _store(database, owns_database=False)
    await store.open()
    commit = _atomic_commit("pgsql-settlement")
    prepared = await store._prepare_checkpoint(
        commit.candidate,
        committed_at=commit.committed_at,
        output_candidates=commit.output_candidates,
    )
    checkpoint = prepared.checkpoint
    scope = checkpoint.authority
    identity = commit.idempotency
    owner = "settlement-owner"
    active_reservation = {
        "owner_token": owner,
        "request_digest": str(identity.request_digest),
        "record_state": conversation.IdempotencyRecordState.IN_PROGRESS.value,
    }
    for ambiguous in (True, False):
        with monkeypatch.context() as context:
            context.setattr(
                store,
                "_fetchone",
                AsyncMock(return_value=active_reservation),
            )
            context.setattr(store, "_cleanup_owner", AsyncMock())
            settled = await store.reconcile_idempotency(
                identity,
                owner,
                ambiguous=ambiguous,
            )
        assert (
            settled.disposition
            is conversation.IdempotencySettlementDisposition.SETTLED
        )
    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store.fence_idempotency(identity, owner, ambiguous=False)

    mismatched = {
        "owner_token": "different-owner",
        "request_digest": str(identity.request_digest),
        "record_state": conversation.IdempotencyRecordState.AMBIGUOUS.value,
    }
    terminal = {
        "owner_token": owner,
        "request_digest": str(identity.request_digest),
        "record_state": conversation.IdempotencyRecordState.AMBIGUOUS.value,
    }
    for row, expected in (
        (
            mismatched,
            conversation.IdempotencySettlementDisposition.OWNERSHIP_CONFLICT,
        ),
        (terminal, conversation.IdempotencySettlementDisposition.SETTLED),
    ):
        with monkeypatch.context() as context:
            context.setattr(
                store,
                "_read_one",
                AsyncMock(side_effect=(row, None)),
            )
            resolution = await store.inspect_idempotency_settlement(
                identity,
                owner,
            )
        assert resolution.disposition is expected

    with pytest.raises(conversation.ConversationKeyPolicyError):
        non_current = conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=_NonCurrentKeyResolver(),
            cipher=conversation.AesGcmConversationCipher(),
            owns_database=False,
        )
        await non_current._prepare_lifecycle_envelope(checkpoint)

    public_row = {
        "checkpoint_id": str(checkpoint.identity.checkpoint_id),
        "tombstoned": False,
    }
    authority_key = str(conversation.authority_digest(scope))
    with monkeypatch.context() as context:
        context.setattr(store, "_read_one", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.tombstone(
                commit.public_response_id,
                scope,
                datetime(2026, 8, 1, 12, 1, tzinfo=UTC),
            )
    raced_tombstone = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            lifecycle=conversation.CheckpointLifecycle.TOMBSTONED,
            timestamps=replace(
                checkpoint.timestamps,
                tombstoned_at=datetime(2026, 8, 1, 12, 1, tzinfo=UTC),
            ),
        )
    )
    with monkeypatch.context() as context:
        context.setattr(store, "_read_one", AsyncMock(return_value=public_row))
        context.setattr(
            store,
            "_load_checkpoint",
            AsyncMock(side_effect=conversation.ConversationAuthorizationError),
        )
        load_lifecycle = AsyncMock(return_value=raced_tombstone)
        context.setattr(store, "_load_checkpoint_lifecycle", load_lifecycle)

        resolved = await store.tombstone(
            commit.public_response_id,
            scope,
            datetime(2026, 8, 1, 12, 1, tzinfo=UTC),
        )

        assert resolved is raced_tombstone
        load_lifecycle.assert_awaited_once_with(
            checkpoint.identity.checkpoint_id,
            scope,
            conversation.CheckpointLifecycle.TOMBSTONED,
        )
    with monkeypatch.context() as context:
        context.setattr(store, "_read_one", AsyncMock(return_value=public_row))
        context.setattr(
            store,
            "_load_checkpoint",
            AsyncMock(return_value=checkpoint),
        )
        context.setattr(
            store,
            "_prepare_lifecycle_envelope",
            AsyncMock(return_value=(prepared.envelope, prepared.key)),
        )
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store.tombstone(
                commit.public_response_id,
                scope,
                datetime(2026, 8, 1, 12, 1, tzinfo=UTC),
            )

    for response_tombstoned, lifecycle in (
        (True, "committed"),
        (False, "tombstoned"),
    ):
        checkpoint_row = {
            "authority_digest": authority_key,
            "lifecycle_state": lifecycle,
        }
        locked_public_row = {
            "authority_digest": authority_key,
            "checkpoint_id": str(checkpoint.identity.checkpoint_id),
            "tombstoned": response_tombstoned,
        }
        with monkeypatch.context() as context:
            context.setattr(
                store,
                "_read_one",
                AsyncMock(return_value=public_row),
            )
            context.setattr(
                store,
                "_load_checkpoint",
                AsyncMock(return_value=checkpoint),
            )
            context.setattr(
                store,
                "_prepare_lifecycle_envelope",
                AsyncMock(return_value=(prepared.envelope, prepared.key)),
            )
            context.setattr(
                store,
                "_fetchone",
                AsyncMock(side_effect=(checkpoint_row, locked_public_row)),
            )
            with pytest.raises(conversation.ConversationConflictError):
                await store.tombstone(
                    commit.public_response_id,
                    scope,
                    datetime(2026, 8, 1, 12, 1, tzinfo=UTC),
                )

    delete_rows = (
        (None,),
        (
            {
                "authority_digest": authority_key,
                "tombstoned": False,
                "checkpoint_id": str(checkpoint.identity.checkpoint_id),
            },
        ),
        (
            {
                "authority_digest": authority_key,
                "tombstoned": True,
                "checkpoint_id": str(checkpoint.identity.checkpoint_id),
            },
            None,
        ),
    )
    delete_errors = (
        conversation.ConversationAuthorizationError,
        conversation.ConversationTransitionError,
        conversation.ConversationAuthorizationError,
    )
    for rows, expected in zip(delete_rows, delete_errors, strict=True):
        with monkeypatch.context() as context:
            context.setattr(store, "_fetchone", AsyncMock(side_effect=rows))
            with pytest.raises(expected):
                await store.delete(
                    commit.public_response_id,
                    scope,
                    datetime(2026, 8, 1, 12, 2, tzinfo=UTC),
                )

    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchall",
            AsyncMock(
                side_effect=(
                    (),
                    (
                        {
                            "authority_digest": authority_key,
                            "operation": identity.operation.value,
                            "idempotency_key": str(identity.key),
                        },
                    ),
                )
            ),
        )
        receipt = await store.prune(
            datetime(2026, 8, 1, 12, 3, tzinfo=UTC),
            limit=2,
        )
    assert receipt.idempotency_records == 1
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchall",
            AsyncMock(return_value=({"intent_id": "published-intent"},)),
        )
        published_only = await store.prune(
            datetime(2026, 8, 1, 12, 3, tzinfo=UTC),
            limit=1,
        )
    assert published_only.outbox_records == 1
    assert published_only.idempotency_records == 0

    cursor = cast(PgsqlCursor, _Cursor())
    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationStorageError):
            await store._upsert_terminal(
                cursor,
                checkpoint.identity.checkpoint_id,
                commit.public_response_id,
                conversation.CheckpointLifecycle.DELETED,
                datetime(2026, 8, 1, 12, 4, tzinfo=UTC),
            )
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                return_value={
                    "record_count": (
                        conversation.StoreLimits().max_terminal_metadata
                    )
                }
            ),
        )
        await store._upsert_terminal(
            cursor,
            checkpoint.identity.checkpoint_id,
            commit.public_response_id,
            conversation.CheckpointLifecycle.DELETED,
            datetime(2026, 8, 1, 12, 4, tzinfo=UTC),
        )

    assert commit.public_response_id is not None
    assert commit.outbox_intent_id is not None
    target = conversation.OutboxClaimTarget(
        authority=scope,
        checkpoint_id=checkpoint.identity.checkpoint_id,
        public_response_id=commit.public_response_id,
        intent_id=commit.outbox_intent_id,
    )
    outbox_row = {
        "authority_digest": authority_key,
        "intent_id": target.intent_id,
        "checkpoint_id": str(target.checkpoint_id),
        "public_response_id": str(target.public_response_id),
        "outbox_state": conversation.OutboxState.PENDING.value,
        "attempts": 0,
        "lease_owner": None,
        "lease_expires_at": None,
    }
    with monkeypatch.context() as context:
        context.setattr(store, "_read_one", AsyncMock(return_value=outbox_row))
        context.setattr(
            store,
            "retrieve_output_candidates",
            AsyncMock(return_value=commit.output_candidates),
        )
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        claimed = await store.claim_outbox(target)
    assert (
        claimed.disposition
        is conversation.OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
    )

    with monkeypatch.context() as context:
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store.acknowledge_outbox(target, "outbox-owner")
        with pytest.raises(conversation.ConversationConflictError):
            await store.release_outbox(target, "outbox-owner")
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(return_value=outbox_row),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await store.acknowledge_outbox(target, "outbox-owner")

    expired_claim = {
        **outbox_row,
        "outbox_state": conversation.OutboxState.CLAIMED.value,
        "lease_expires_at": datetime(2000, 1, 1, tzinfo=UTC),
    }
    with monkeypatch.context() as context:
        context.setattr(
            store, "_read_one", AsyncMock(return_value=expired_claim)
        )
        context.setattr(
            store,
            "retrieve_output_candidates",
            AsyncMock(return_value=commit.output_candidates),
        )
        context.setattr(
            store, "_fetchone", AsyncMock(return_value=expired_claim)
        )
        reclaimed = await store.claim_outbox(target)
    assert reclaimed.disposition is conversation.OutboxClaimDisposition.CLAIMED

    rewrap = pgsql_module.ReconciliationWorkRecord(
        reconciliation_id="rewrap-1",
        checkpoint_id=checkpoint.identity.checkpoint_id,
        lane_id=commit.output_candidates[0].lane_id,
        work_kind="rewrap_payload",
        state=pgsql_module.ReconciliationWorkState.CLAIMED,
        attempts=1,
        lease_owner="rewrap-owner",
        lease_expires_at=datetime(2026, 8, 1, 12, 5, tzinfo=UTC),
    )
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                return_value={
                    "work_state": "claimed",
                    "lease_owner": rewrap.lease_owner,
                    "checkpoint_id": str(rewrap.checkpoint_id),
                    "lane_id": str(rewrap.lane_id),
                    "work_kind": rewrap.work_kind,
                }
            ),
        )
        await store.acknowledge_reconciliation(rewrap, succeeded=True)
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_validate_payload_reference_row",
            lambda *args, **kwargs: None,
        )
        context.setattr(
            store,
            "_decrypt_payload_row",
            AsyncMock(return_value=b"\xff"),
        )
        with pytest.raises(conversation.ConversationStorageError):
            await store._reconciliation_target(
                {
                    "work_kind": "delete_upstream",
                    "checkpoint_id": str(rewrap.checkpoint_id),
                    "lane_id": str(rewrap.lane_id),
                    "authority_digest": str(authority_key),
                }
            )
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "_fetchone",
            AsyncMock(
                return_value={
                    "work_state": "claimed",
                    "lease_owner": rewrap.lease_owner,
                    "checkpoint_id": str(rewrap.checkpoint_id),
                    "lane_id": str(rewrap.lane_id),
                }
            ),
        )
        context.setattr(
            store,
            "_reconciliation_target",
            AsyncMock(
                return_value=conversation.UpstreamResponseId(
                    "different-upstream-target"
                )
            ),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await store.acknowledge_reconciliation(rewrap, succeeded=True)
    await store.close()


async def test_scripted_key_rotation_and_schema_window_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _Database()
    scope = authority()
    non_current = conversation.PgsqlConversationStore(
        cast(Any, database),
        key_resolver=_NonCurrentKeyResolver(),
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=False,
    )
    await non_current.open()
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await non_current.rotate_keys(scope, limit=1)
    await non_current.close()

    store = _store(database, owns_database=False)
    await store.open()
    authority_key = str(conversation.authority_digest(scope))
    rotation_row = {
        "payload_id": "payload-1",
        "authority_digest": authority_key,
        "checkpoint_id": "checkpoint-1",
        "conversation_id": "conversation-1",
        "lane_id": "lane-1",
        "payload_sequence": 0,
        "payload_kind": conversation.ConversationPayloadKind.LANE_OUTPUT.value,
        "payload_schema_version": 1,
        "codec_version": 1,
        "key_id": "old-key",
        "key_revision": 1,
        "authenticated_digest": "a" * 64,
    }
    with monkeypatch.context() as context:
        context.setattr(
            store, "_read_all", AsyncMock(return_value=(rotation_row,))
        )
        context.setattr(
            store,
            "_decrypt_payload_row",
            AsyncMock(return_value=b"durable payload"),
        )
        context.setattr(store, "_synchronize_write_key", AsyncMock())
        context.setattr(store, "_fetchone", AsyncMock(return_value=None))
        with pytest.raises(conversation.ConversationConflictError):
            await store.rotate_keys(scope, limit=1)

    with pytest.raises(conversation.ConversationKeyPolicyError):
        await store.retire_key(
            scope,
            key_id="key-1",
            revision=1,
            at=datetime(2026, 8, 1, 12, tzinfo=UTC),
        )
    for rows, expected in (
        ((None,), conversation.ConversationKeyPolicyError),
        (
            (
                {"key_status": conversation.ConversationKeyStatus.GRACE.value},
                {"record_count": 1},
            ),
            conversation.ConversationConflictError,
        ),
    ):
        with monkeypatch.context() as context:
            context.setattr(store, "_synchronize_write_key", AsyncMock())
            context.setattr(store, "_fetchone", AsyncMock(side_effect=rows))
            with pytest.raises(expected):
                await store.retire_key(
                    scope,
                    key_id="old-key",
                    revision=1,
                    at=datetime(2026, 8, 1, 12, tzinfo=UTC),
                )

    revision = {"version_num": pgsql_module.CONVERSATION_PGSQL_HEAD_REVISION}
    incompatible = {
        "schema_version": 2,
        "minimum_reader_version": 1,
        "maximum_reader_version": 2,
        "minimum_writer_version": 1,
        "maximum_writer_version": 2,
        "checkpoint_codec_version": 1,
    }
    for rows in ((revision, None), (revision, incompatible)):
        with monkeypatch.context() as context:
            context.setattr(
                store,
                "_read_one_unchecked",
                AsyncMock(side_effect=rows),
            )
            with pytest.raises(
                conversation.ConversationMigrationRequiredError
            ):
                await store._schema_readiness()
    await store.close()


async def test_store_translates_database_errors_and_reports_close_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(cursor: PgsqlCursor) -> None:
        return None

    for error in (
        conversation.ConversationAuthorizationError(),
        CancelledError(),
    ):
        store = conversation.PgsqlConversationStore(
            cast(Any, _FailingDatabase(error)),
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
            policy=conversation.PgsqlConversationStorePolicy(
                check_schema_on_open=False
            ),
        )
        await store.open()
        with pytest.raises(type(error)):
            await store._read_one_unchecked("read", "SELECT 1", None)
        with pytest.raises(type(error)):
            await store._read_all("read", "SELECT 1", None)

    runtime_store = conversation.PgsqlConversationStore(
        cast(Any, _FailingDatabase(RuntimeError("database unavailable"))),
        key_resolver=_key_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        policy=conversation.PgsqlConversationStorePolicy(
            check_schema_on_open=False
        ),
    )
    await runtime_store.open()
    with pytest.raises(conversation.ConversationStorageError):
        await runtime_store._read_one_unchecked("read", "SELECT 1", None)
    with pytest.raises(conversation.ConversationStorageError):
        await runtime_store._read_all("read", "SELECT 1", None)
    with monkeypatch.context() as context:
        context.setattr(
            pgsql_module,
            "classify_pgsql_error",
            lambda error, *, operation: PgsqlFailure(
                category=pgsql_module.PgsqlFailureCategory.UNIQUE_CONFLICT,
                code="23505",
                retryable=False,
                operation=operation,
            ),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await runtime_store._transaction("write", callback)

    hook = _RecordingBoundaryHook()
    database = _Database()
    hooked = conversation.PgsqlConversationStore(
        cast(Any, database),
        key_resolver=_key_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        policy=conversation.PgsqlConversationStorePolicy(
            check_schema_on_open=False
        ),
        boundary_hook=hook,
        owns_database=False,
    )
    await hooked.open()
    await hooked.inspect_close()
    await hooked.close()
    await hooked.inspect_close()
    assert hook.boundaries == [
        conversation.StoreAwaitBoundary.CLOSE_STATUS,
        conversation.StoreAwaitBoundary.CLOSE_BEGIN,
        conversation.StoreAwaitBoundary.CLOSE_SETTLED,
        conversation.StoreAwaitBoundary.CLOSE_STATUS,
    ]

    for database in (_NoCloseDatabase(), _SyncCloseDatabase()):
        lifecycle = conversation.PgsqlConversationStore(
            cast(Any, database),
            key_resolver=_key_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
            policy=conversation.PgsqlConversationStorePolicy(
                check_schema_on_open=False
            ),
            owns_database=True,
        )
        await lifecycle.open()
        await lifecycle.close()
