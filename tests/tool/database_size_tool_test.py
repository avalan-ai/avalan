from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from avalan.entities import ToolCallContext
from avalan.tool.database import (
    DatabaseSizeTool,
    DatabaseToolSettings,
    TableSize,
    TableSizeMetric,
)


class DummyResult:
    def __init__(self, rows: list[dict[str, Any]] | None = None, scalar: Any = None) -> None:
        self._rows = [dict(row) for row in rows or []]
        self._scalar = scalar

    def mappings(self) -> SimpleNamespace:
        rows = list(self._rows)

        class _MappingResult:
            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def first(self) -> dict[str, Any] | None:
                return self._data[0] if self._data else None

            def all(self) -> list[dict[str, Any]]:
                return list(self._data)

        return _MappingResult(rows)

    def scalar(self) -> Any:
        if self._scalar is not None:
            return self._scalar
        if not self._rows:
            return None
        first = self._rows[0]
        if not first:
            return None
        return next(iter(first.values()))

    def scalar_one_or_none(self) -> Any:
        return self.scalar()


class DummySyncConnection:
    def __init__(
        self, *, dialect_name: str, results: list[DummyResult]
    ) -> None:
        self.dialect = SimpleNamespace(name=dialect_name)
        self._results = list(results)
        self.statements: list[tuple[str, dict[str, Any] | None]] = []

    def execute(self, statement, params: dict[str, Any] | None = None) -> DummyResult:
        text_statement = str(statement)
        self.statements.append((text_statement, params))
        if not self._results:
            raise AssertionError("No results available for execute call.")
        return self._results.pop(0)


def dummy_create_async_engine(dsn: str, **_: Any):
    engine = create_engine(dsn)

    class DummyAsyncConn:
        def __init__(self, conn):
            self.conn = conn

        async def exec_driver_sql(self, sql: str, *args, **kwargs):
            result = self.conn.exec_driver_sql(sql, *args, **kwargs)
            if not result.returns_rows:
                self.conn.commit()
            return result

        async def execute(self, stmt):
            result = self.conn.execute(stmt)
            if not result.returns_rows:
                self.conn.commit()
            return result

        async def run_sync(self, fn, *args, **kwargs):
            return fn(self.conn, *args, **kwargs)

    class DummyConnCtx:
        def __init__(self, sync_engine):
            self._engine = sync_engine
            self._conn = None

        async def __aenter__(self):
            self._conn = self._engine.connect()
            return DummyAsyncConn(self._conn)

        async def __aexit__(self, exc_type, exc, tb):
            assert self._conn is not None
            self._conn.close()
            return False

    class DummyAsyncEngine:
        def __init__(self, sync_engine):
            self._engine = sync_engine
            self.disposed = False

        def connect(self):
            return DummyConnCtx(self._engine)

        def begin(self):
            return DummyConnCtx(self._engine)

        async def dispose(self):
            self._engine.dispose()
            self.disposed = True

    return DummyAsyncEngine(engine)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_database_size_tool_sqlite_collects_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "example.sqlite"
    dsn = f"sqlite:///{db_path}"
    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE books(id INTEGER PRIMARY KEY, title TEXT NOT NULL, pages INTEGER)"
            )
        )
        conn.execute(
            text("CREATE INDEX idx_books_title ON books(title)")
        )
        conn.execute(
            text("INSERT INTO books(title, pages) VALUES ('Book', 100)")
        )
    engine.dispose()

    async_engine = dummy_create_async_engine(dsn)
    tool = DatabaseSizeTool(async_engine, DatabaseToolSettings(dsn=dsn))

    size = await tool("books", context=ToolCallContext())

    assert isinstance(size, TableSize)
    metrics = {metric.category: metric for metric in size.metrics}
    assert "total" in metrics
    assert metrics["total"].bytes is not None
    assert metrics["total"].human_readable is not None

    await async_engine.dispose()


def test_collect_postgresql_sizes() -> None:
    connection = DummySyncConnection(
        dialect_name="postgresql",
        results=[
            DummyResult(
                rows=[
                    {
                        "data_bytes": 2048,
                        "index_bytes": 1024,
                        "total_bytes": 4096,
                    }
                ]
            )
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_postgresql(connection, "public", "books")

    assert metrics == [
        TableSizeMetric(category="data", bytes=2048, human_readable="2.0 KiB"),
        TableSizeMetric(category="indexes", bytes=1024, human_readable="1.0 KiB"),
        TableSizeMetric(category="total", bytes=4096, human_readable="4.0 KiB"),
    ]


def test_collect_mysql_sizes() -> None:
    connection = DummySyncConnection(
        dialect_name="mysql",
        results=[
            DummyResult(
                rows=[
                    {
                        "data_bytes": 4096,
                        "index_bytes": 2048,
                        "free_bytes": 512,
                    }
                ]
            )
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_mysql(connection, "main", "orders")

    assert metrics[0].category == "data"
    assert metrics[0].bytes == 4096
    assert metrics[1].category == "indexes"
    assert metrics[1].bytes == 2048
    totals = {metric.category: metric for metric in metrics}
    assert totals["total"].bytes == 6144
    assert totals["free"].bytes == 512


def test_collect_sqlite_uses_dbstat() -> None:
    connection = DummySyncConnection(
        dialect_name="sqlite",
        results=[
            DummyResult(rows=[{"size": 8192}]),
            DummyResult(rows=[{"name": "idx_books_title"}]),
            DummyResult(rows=[{"size": 4096}]),
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_sqlite(connection, "books")

    totals = {metric.category: metric for metric in metrics}
    assert totals["data"].bytes == 8192
    assert totals["indexes"].bytes == 4096
    assert totals["total"].bytes == 12288


def test_collect_oracle_sizes() -> None:
    connection = DummySyncConnection(
        dialect_name="oracle",
        results=[
            DummyResult(rows=[{"bytes": 16384}]),
            DummyResult(rows=[{"index_name": "IDX_BOOKS"}]),
            DummyResult(rows=[{"bytes": 4096}]),
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_oracle(connection, "library", "books")

    totals = {metric.category: metric for metric in metrics}
    assert totals["data"].bytes == 16384
    assert totals["indexes"].bytes == 4096
    assert totals["total"].bytes == 20480


def test_collect_mssql_sizes() -> None:
    connection = DummySyncConnection(
        dialect_name="mssql",
        results=[
            DummyResult(rows=[{"data_bytes": 8192, "index_bytes": 4096}]),
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_mssql(connection, "dbo", "books")

    totals = {metric.category: metric for metric in metrics}
    assert totals["data"].bytes == 8192
    assert totals["indexes"].bytes == 4096
    assert totals["total"].bytes == 12288

