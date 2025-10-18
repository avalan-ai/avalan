from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from avalan.entities import ToolCallContext
from avalan.tool.database import DatabaseLock, DatabaseToolSettings
from avalan.tool.database.locks import DatabaseLocksTool


class DummyResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> SimpleNamespace:
        return SimpleNamespace(all=lambda: list(self._rows))


class DummySyncConnection:
    def __init__(
        self,
        *,
        dialect_name: str,
        rows: list[dict[str, object]],
        error: type[SQLAlchemyError] | None = None,
    ) -> None:
        self.dialect = SimpleNamespace(name=dialect_name)
        self._rows = rows
        self._error = error
        self.executed_statements: list[str] = []

    def execute(self, statement) -> DummyResult:
        self.executed_statements.append(str(statement))
        if self._error is not None:
            raise self._error()
        return DummyResult(self._rows)


class DummyAsyncConnection:
    def __init__(self, sync_connection: DummySyncConnection) -> None:
        self._sync_connection = sync_connection

    async def run_sync(self, fn, *args, **kwargs):
        return fn(self._sync_connection, *args, **kwargs)


@dataclass
class DummyConnectionContext:
    connection: DummySyncConnection

    async def __aenter__(self) -> DummyAsyncConnection:
        return DummyAsyncConnection(self.connection)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class DummyAsyncEngine:
    def __init__(self, sync_connection: DummySyncConnection) -> None:
        self._sync_connection = sync_connection

    def connect(self) -> DummyConnectionContext:
        return DummyConnectionContext(self._sync_connection)


class DummySqlAlchemyError(SQLAlchemyError):
    """Dedicated exception to emulate SQLAlchemy failures."""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_database_locks_tool_collects_postgresql_locks(
    anyio_backend: str,
) -> None:
    rows = [
        {
            "pid": 7,
            "user_name": "reporter",
            "locktype": "relation",
            "mode": "ExclusiveLock",
            "granted": "t",
            "state": "active",
            "query": " SELECT 1 ",
            "lock_target": "public.table_name",
            "blocking_pids": "10, 11",
        },
        {
            "pid": None,
            "user_name": "",
            "locktype": "",
            "mode": "",
            "granted": "false",
            "state": None,
            "query": None,
            "lock_target": "",
            "blocking_pids": None,
        },
    ]

    sync_connection = DummySyncConnection(dialect_name="postgresql", rows=rows)
    engine = DummyAsyncEngine(sync_connection)
    tool = DatabaseLocksTool(
        engine,
        DatabaseToolSettings(dsn="sqlite://"),
    )

    locks = await tool(context=ToolCallContext())

    assert locks == [
        DatabaseLock(
            pid="7",
            user="reporter",
            lock_type="relation",
            lock_target="public.table_name",
            mode="ExclusiveLock",
            granted=True,
            blocking=("10", "11"),
            state="active",
            query="SELECT 1",
        ),
        DatabaseLock(
            pid=None,
            user=None,
            lock_type=None,
            lock_target=None,
            mode=None,
            granted=False,
            blocking=(),
            state=None,
            query=None,
        ),
    ]


def test_collect_mysql_returns_normalized_locks() -> None:
    rows = [
        {
            "pid": "42",
            "user_name": "writer",
            "lock_schema": "main",
            "lock_name": "orders",
            "lock_type": "TABLE",
            "lock_mode": "WRITE",
            "lock_status": "GRANTED",
            "lock_data": None,
            "state": "running",
            "query": " SELECT * FROM orders ",
            "blocking_pids": "1, 2",
        },
        {
            "pid": 99,
            "user_name": "reviewer",
            "lock_schema": None,
            "lock_name": "temp_table",
            "lock_type": "TABLE",
            "lock_mode": "READ",
            "lock_status": "waiting",
            "lock_data": None,
            "state": "waiting",
            "query": "   ",
            "blocking_pids": [None, "77"],
        },
        {
            "pid": 100,
            "user_name": None,
            "lock_schema": None,
            "lock_name": "",
            "lock_type": "RECORD",
            "lock_mode": "READ",
            "lock_status": "unknown",
            "lock_data": "row#1",
            "state": None,
            "query": None,
            "blocking_pids": {88},
        },
    ]

    connection = DummySyncConnection(dialect_name="mysql", rows=rows)
    tool = DatabaseLocksTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://"),
    )

    locks = tool._collect(connection)

    assert locks == [
        DatabaseLock(
            pid="42",
            user="writer",
            lock_type="TABLE",
            lock_target="main.orders",
            mode="WRITE",
            granted=True,
            blocking=("1", "2"),
            state="running",
            query="SELECT * FROM orders",
        ),
        DatabaseLock(
            pid="99",
            user="reviewer",
            lock_type="TABLE",
            lock_target="temp_table",
            mode="READ",
            granted=False,
            blocking=("77",),
            state="waiting",
            query=None,
        ),
        DatabaseLock(
            pid="100",
            user=None,
            lock_type="RECORD",
            lock_target="row#1",
            mode="READ",
            granted=None,
            blocking=("88",),
            state=None,
            query=None,
        ),
    ]


def test_normalize_blocking_handles_braced_strings() -> None:
    class BracedValue:
        def __str__(self) -> str:
            return "{}"

    assert DatabaseLocksTool._normalize_blocking(BracedValue()) == ()


def test_collect_mysql_handles_execution_errors() -> None:
    connection = DummySyncConnection(
        dialect_name="mysql",
        rows=[],
        error=DummySqlAlchemyError,
    )
    tool = DatabaseLocksTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://"),
    )

    locks = tool._collect(connection)
    assert locks == []


def test_collect_returns_empty_for_unknown_dialect() -> None:
    connection = DummySyncConnection(dialect_name="sqlite", rows=[])
    tool = DatabaseLocksTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://"),
    )

    assert tool._collect(connection) == []


def test_normalize_blocking_variants() -> None:
    tool = DatabaseLocksTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://"),
    )

    assert tool._normalize_blocking(None) == ()
    assert tool._normalize_blocking("1, 2") == ("1", "2")
    assert tool._normalize_blocking(["3", "", None]) == ("3",)
    assert tool._normalize_blocking({"4", "4", None}) == ("4",)
    assert tool._normalize_blocking({"id": 5}) == ("'id': 5",)


def test_normalize_query_and_string_or_none() -> None:
    tool = DatabaseLocksTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://"),
    )

    assert tool._normalize_query(None) is None
    assert tool._normalize_query("   ") is None
    assert tool._normalize_query("  SELECT 1  ") == "SELECT 1"

    assert tool._string_or_none(None) is None
    assert tool._string_or_none("") is None
    assert tool._string_or_none(0) == "0"


def test_bool_and_mysql_helpers() -> None:
    tool = DatabaseLocksTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://"),
    )

    assert tool._bool_or_none(None) is None
    assert tool._bool_or_none(True) is True
    assert tool._bool_or_none(False) is False
    assert tool._bool_or_none("YES") is True
    assert tool._bool_or_none("0") is False
    assert tool._bool_or_none("maybe") is None
    assert tool._bool_or_none(1) is True
    assert tool._bool_or_none(0.0) is False

    assert (
        tool._normalize_mysql_target("public", "table", "data")
        == "public.table"
    )
    assert tool._normalize_mysql_target(None, "table", None) == "table"
    assert tool._normalize_mysql_target(None, "", "row42") == "row42"

    assert tool._mysql_granted(None) is None
    assert tool._mysql_granted("  ") is None
    assert tool._mysql_granted("granted") is True
    assert tool._mysql_granted("waiting") is False
    assert tool._mysql_granted("other") is None

