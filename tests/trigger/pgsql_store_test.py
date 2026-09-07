"""Check PostgreSQL adapter wiring without presenting it as SQL proof."""

from .records_test import (
    NOW,
    OWNER,
    definition,
    event,
    occurrence,
    span,
    state,
)
from .registration_test import registration

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.pgsql import PgsqlParameters, PgsqlRow, PgsqlUnitOfWork
from avalan.trigger.codec import record_payload
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.records import TriggerSnapshot
from avalan.trigger.store import HistoryCursor
from avalan.trigger.stores.pgsql import (
    TRIGGER_SCHEMA_HEAD,
    PgsqlTriggerStore,
    check_trigger_schema,
    decision_time,
)


class Cursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, PgsqlParameters]] = []
        self.query = ""
        self.row: PgsqlRow | None = None
        self.rows: list[PgsqlRow] = []
        self.version: str | None = TRIGGER_SCHEMA_HEAD
        self.present = True
        self.isolation = "read committed"
        self.clock = True
        self.revision_row: PgsqlRow | None = None

    async def execute(
        self, query: str, parameters: PgsqlParameters = None
    ) -> None:
        self.query = query
        self.calls.append((query, parameters))

    async def executemany(self, query: str, parameters_seq: object) -> None:
        raise NotImplementedError

    async def fetchone(self) -> PgsqlRow | None:
        if "to_regclass" in self.query:
            return {"version_table": "version" if self.present else None}
        if "current_setting" in self.query:
            return {"isolation": self.isolation}
        if "clock_timestamp" in self.query:
            return (
                {"decision_time": NOW + timedelta(minutes=5)}
                if self.clock
                else None
            )
        if "SELECT payload FROM trigger_definitions" in self.query:
            return self.revision_row
        return self.row

    async def fetchall(self) -> list[PgsqlRow]:
        if "version_num" in self.query:
            return [{"version_num": self.version}] if self.version else []
        return self.rows

    async def close(self) -> None:
        pass


class Connection:
    row_factory: object = None

    def __init__(self, cursor: Cursor) -> None:
        self.handle = cursor
        self.transactions = 0

    @asynccontextmanager
    async def cursor(self) -> AsyncIterator[Cursor]:
        yield self.handle

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["Connection"]:
        self.transactions += 1
        yield self

    async def set_autocommit(self, value: bool) -> None:
        raise NotImplementedError

    async def __aenter__(self) -> "Connection":
        return self

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        return None


class Database:
    def __init__(self) -> None:
        self.cursor = Cursor()
        self.handle = Connection(self.cursor)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Connection]:
        yield self.handle

    async def open(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def snapshot_row() -> PgsqlRow:
    return {
        "definition": record_payload(definition()),
        "state": record_payload(state()),
    }


class PgsqlManagementTestCase(IsolatedAsyncioTestCase):
    async def test_create_checks_schema_and_reads_time_after_lock(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        created = await store.apply(registration(), expected_generation=None)
        assert created.state.generation == 1
        assert database.handle.transactions == 1
        queries = [query for query, _ in database.cursor.calls]
        lock = next(
            i
            for i, query in enumerate(queries)
            if "pg_advisory_xact_lock" in query
        )
        now = next(
            i for i, query in enumerate(queries) if "clock_timestamp" in query
        )
        write = next(
            i
            for i, query in enumerate(queries)
            if "INSERT INTO trigger_definitions" in query
        )
        assert lock < now < write
        assert (
            sum("INSERT INTO trigger_events" in query for query in queries)
            == 1
        )

    async def test_replacement_closes_history_in_the_same_transaction(
        self,
    ) -> None:
        database = Database()
        database.cursor.row = snapshot_row()
        store = PgsqlTriggerStore(database, OWNER)
        updated = await store.apply(
            replace(registration(), semantic_hash="2" * 64),
            expected_generation=1,
        )
        assert updated.state.revision == 2
        assert database.handle.transactions == 1
        queries = [query for query, _ in database.cursor.calls]
        assert any(
            "UPDATE trigger_definitions SET closed_at" in query
            for query in queries
        )
        assert any(
            "INSERT INTO trigger_coverage_spans" in query for query in queries
        )
        assert sum("clock_timestamp" in query for query in queries) == 1

    async def test_noop_does_not_write_and_controls_share_the_reducer(
        self,
    ) -> None:
        database = Database()
        database.cursor.row = snapshot_row()
        store = PgsqlTriggerStore(database, OWNER)
        await store.apply(registration(), expected_generation=1)
        assert not any("INSERT" in query for query, _ in database.cursor.calls)
        paused = await store.set_enabled(
            "daily", enabled=False, expected_generation=1
        )
        assert paused.state.generation == 2
        failed = await store.fail(
            "daily",
            expected_generation=1,
            error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
        )
        assert failed.state.failure_count == 1
        assert failed.state.next_at == NOW

    async def test_read_pages_and_discovery_are_bounded_and_owner_scoped(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        assert await store.inspect("daily") is None
        database.cursor.row = snapshot_row()
        assert await store.inspect("daily") == TriggerSnapshot(
            definition=definition(), state=state()
        )
        database.cursor.rows = [snapshot_row(), snapshot_row()]
        page = await store.list(limit=1)
        assert page.next_cursor == HistoryCursor(offset=1)
        assert len((await store.list(cursor=page.next_cursor)).items) == 2
        assert len(await store.discover(decision_time=NOW, limit=2)) == 2
        parameters = database.cursor.calls[-1][1]
        assert parameters == (OWNER.value, NOW, NOW, None, None, None, None, 2)
        database.cursor.rows = [{"payload": record_payload(event())}] * 2
        events = await store.events("daily", limit=1)
        assert events.items == (event(),)
        assert events.next_cursor == HistoryCursor(offset=1)
        assert (
            await store.events("daily", cursor=events.next_cursor)
        ).next_cursor is None
        database.cursor.rows = [{"payload": record_payload(state())}]
        with raises(TriggerError):
            await store.events("daily")
        database.cursor.row = None
        with raises(TriggerError):
            await store.events("missing")
        with raises(TriggerError):
            await store.set_enabled(
                "missing", enabled=True, expected_generation=1
            )

    async def test_preflight_rejects_missing_old_or_incompatible_schema(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        await store.preflight()
        database.cursor.present = False
        with raises(TriggerError):
            await store.preflight()
        database.cursor.present = True
        for version in (None, "old"):
            database.cursor.version = version
            with raises(TriggerError):
                await store.preflight()
        database.cursor.version = TRIGGER_SCHEMA_HEAD
        database.cursor.isolation = "repeatable read"
        with raises(TriggerError):
            await store.preflight()
        assert not any("INSERT" in query for query, _ in database.cursor.calls)

    async def test_wrong_record_and_unavailable_clock_fail_safely(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        database.cursor.row = {
            "definition": record_payload(state()),
            "state": record_payload(state()),
        }
        with raises(TriggerError):
            await store.inspect("daily")
        unit = PgsqlUnitOfWork(
            connection=database.handle,
            cursor=database.cursor,
            database=database,
        )
        database.cursor.clock = False
        with raises(TriggerError):
            await decision_time(unit)
        await check_trigger_schema(unit)

    async def test_revision_and_span_history_decode_their_exact_records(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        with raises(TriggerError):
            await store.revision("missing", 1)
        with raises(TriggerError):
            await store.coverage("missing")
        database.cursor.row = snapshot_row()
        assert await store.revision("daily", 2) is None
        database.cursor.revision_row = {
            "payload": record_payload(definition())
        }
        assert await store.revision("daily", 1) == definition()
        database.cursor.revision_row = {"payload": record_payload(state())}
        with raises(TriggerError):
            await store.revision("daily", 1)
        database.cursor.rows = [{"payload": record_payload(span())}] * 2
        page = await store.coverage("daily", limit=1)
        assert page.items == (span(),)
        assert page.next_cursor == HistoryCursor(offset=1)
        assert (
            await store.coverage("daily", cursor=page.next_cursor)
        ).next_cursor is None
        database.cursor.rows = [{"payload": record_payload(state())}]
        with raises(TriggerError):
            await store.coverage("daily")

    async def test_occurrence_history_is_bounded_and_rejects_other_records(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        with raises(TriggerError):
            await store.occurrences("missing")
        database.cursor.row = snapshot_row()
        database.cursor.rows = [{"payload": record_payload(occurrence())}] * 2
        page = await store.occurrences("daily", limit=1)
        assert page.items == (occurrence(),)
        assert page.next_cursor == HistoryCursor(offset=1)
        database.cursor.rows = []
        assert (
            await store.occurrences("daily", cursor=page.next_cursor)
        ).items == ()
        database.cursor.rows = [{"payload": record_payload(state())}]
        with raises(TriggerError):
            await store.occurrences("daily")

    async def test_empty_next_wake_and_null_aggregate_have_no_instant(
        self,
    ) -> None:
        database = Database()
        store = PgsqlTriggerStore(database, OWNER)
        for row in (None, {"eligible_at": None}):
            database.cursor.row = row
            assert await store.next_eligible_at() is None
