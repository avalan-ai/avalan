from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

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


def test_collect_sqlite_skips_nameless_indexes() -> None:
    connection = DummySyncConnection(
        dialect_name="sqlite",
        results=[
            DummyResult(rows=[{"size": 2048}]),
            DummyResult(rows=[{"name": None}, {"name": "idx_books_title"}]),
            DummyResult(rows=[{"size": 1024}]),
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_sqlite(connection, "books")

    totals = {metric.category: metric for metric in metrics}
    assert totals["data"].bytes == 2048
    assert totals["indexes"].bytes == 1024
    assert totals["total"].bytes == 3072
    assert len(connection.statements) == 3


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


def test_split_schema_and_table_variants() -> None:
    assert DatabaseSizeTool._split_schema_and_table("public.books") == (
        "public",
        "books",
    )
    assert DatabaseSizeTool._split_schema_and_table("books") == (None, "books")
    assert DatabaseSizeTool._split_schema_and_table(".books") == (None, "books")


def test_metrics_for_dialect_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    connection = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    calls: list[tuple[str, str | None, str]] = []

    def record_standard(name: str) -> Callable[[DatabaseSizeTool, Any, str | None, str], list[TableSizeMetric]]:
        def _inner(
            self: DatabaseSizeTool,
            conn: Any,
            schema: str | None,
            table: str,
        ) -> list[TableSizeMetric]:
            calls.append((name, schema, table))
            return [
                TableSizeMetric(
                    category="total",
                    bytes=len(calls),
                    human_readable=f"{len(calls)} B",
                )
            ]

        return _inner

    def record_sqlite(
        self: DatabaseSizeTool, conn: Any, table: str
    ) -> list[TableSizeMetric]:
        calls.append(("sqlite", None, table))
        return [
            TableSizeMetric(
                category="total",
                bytes=len(calls),
                human_readable=f"{len(calls)} B",
            )
        ]

    monkeypatch.setattr(
        DatabaseSizeTool,
        "_collect_postgresql",
        record_standard("postgresql"),
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_collect_mysql",
        record_standard("mysql"),
    )
    monkeypatch.setattr(DatabaseSizeTool, "_collect_sqlite", record_sqlite)
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_collect_oracle",
        record_standard("oracle"),
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_collect_mssql",
        record_standard("mssql"),
    )

    result = tool._metrics_for_dialect(connection, "public", "books")
    assert result[0].bytes == 1
    assert calls[-1] == ("postgresql", "public", "books")

    connection.dialect.name = "mysql"
    result = tool._metrics_for_dialect(connection, "library", "loans")
    assert result[0].bytes == 2
    assert calls[-1] == ("mysql", "library", "loans")

    connection.dialect.name = "sqlite"
    result = tool._metrics_for_dialect(connection, None, "authors")
    assert result[0].bytes == 3
    assert calls[-1] == ("sqlite", None, "authors")

    connection.dialect.name = "oracle"
    result = tool._metrics_for_dialect(connection, "archive", "logs")
    assert result[0].bytes == 4
    assert calls[-1] == ("oracle", "archive", "logs")

    connection.dialect.name = "mssql+pyodbc"
    result = tool._metrics_for_dialect(connection, "dbo", "events")
    assert result[0].bytes == 5
    assert calls[-1] == ("mssql", "dbo", "events")

    connection.dialect.name = "unknown"
    assert tool._metrics_for_dialect(connection, "dbo", "events") == []


def test_collect_uses_effective_schema_when_display_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlankNormalizer:
        def normalize(self, value: str) -> str:
            if value == "private":
                return ""
            return f"norm_{value}"

    tool = DatabaseSizeTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://", identifier_case="lower"),
        normalizer=BlankNormalizer(),
    )

    inspector = SimpleNamespace()
    connection = SimpleNamespace()
    metrics = [
        TableSizeMetric(category="total", bytes=1, human_readable="1 B"),
    ]

    monkeypatch.setattr(
        DatabaseSizeTool,
        "_inspect_connection",
        lambda self, conn: inspector,
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_schemas",
        lambda self, conn, insp: ("public", ["public"]),
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_denormalize_table_name",
        lambda self, conn, schema, table: "actual_books",
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_metrics_for_dialect",
        lambda self, conn, schema, table: metrics,
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_normalize_table_for_output",
        lambda self, table: "display_books",
    )

    result = tool._collect(connection, schema="private", table_name="books")

    assert result.name == "private.display_books"
    assert result.schema == ""
    assert result.metrics == tuple(metrics)


def test_collect_includes_normalized_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    class UpperNormalizer:
        def normalize(self, value: str) -> str:
            return value.upper() if value is not None else ""

    tool = DatabaseSizeTool(
        SimpleNamespace(),
        DatabaseToolSettings(dsn="sqlite://", identifier_case="lower"),
        normalizer=UpperNormalizer(),
    )

    inspector = SimpleNamespace()
    connection = SimpleNamespace()

    monkeypatch.setattr(
        DatabaseSizeTool,
        "_inspect_connection",
        lambda self, conn: inspector,
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_schemas",
        lambda self, conn, insp: ("public", ["public", "custom"]),
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_denormalize_table_name",
        lambda self, conn, schema, table: "actual_books",
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_metrics_for_dialect",
        lambda self, conn, schema, table: [],
    )
    monkeypatch.setattr(
        DatabaseSizeTool,
        "_normalize_table_for_output",
        lambda self, table: "display_books",
    )

    result = tool._collect(connection, schema="custom", table_name="books")

    assert result.name == "CUSTOM.display_books"
    assert result.schema == "CUSTOM"
    assert result.metrics == ()


def test_collect_postgresql_empty_rows() -> None:
    connection = DummySyncConnection(dialect_name="postgresql", results=[DummyResult()])
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    assert tool._collect_postgresql(connection, "public", "books") == []


def test_collect_postgresql_inferring_total() -> None:
    connection = DummySyncConnection(
        dialect_name="postgresql",
        results=[
            DummyResult(
                rows=[{"data_bytes": 1024, "index_bytes": 512, "total_bytes": None}]
            )
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_postgresql(connection, "public", "books")

    totals = {metric.category: metric for metric in metrics}
    assert totals["total"].bytes == 1536


def test_collect_mysql_empty_rows() -> None:
    connection = DummySyncConnection(dialect_name="mysql", results=[DummyResult()])
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    assert tool._collect_mysql(connection, "main", "books") == []


def test_collect_mysql_partial_values() -> None:
    connection = DummySyncConnection(
        dialect_name="mysql",
        results=[
            DummyResult(
                rows=[
                    {
                        "data_bytes": 2048,
                        "index_bytes": None,
                        "free_bytes": 128,
                    }
                ]
            )
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    metrics = tool._collect_mysql(connection, "main", "books")

    totals = {metric.category: metric for metric in metrics}
    assert "indexes" not in totals
    assert totals["total"].bytes == 2048


def test_collect_sqlite_falls_back_to_pages() -> None:
    class FallbackConnection(DummySyncConnection):
        def __init__(self) -> None:
            super().__init__(
                dialect_name="sqlite",
                results=[DummyResult(scalar=100), DummyResult(scalar=4096)],
            )
            self._first = True

        def execute(self, statement, params=None):  # type: ignore[override]
            text_statement = str(statement)
            if "dbstat" in text_statement and self._first:
                self._first = False
                raise RuntimeError("dbstat unavailable")
            return super().execute(statement, params)

    connection = FallbackConnection()
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    metrics = tool._collect_sqlite(connection, "books")
    assert metrics == [
        TableSizeMetric(
            category="total",
            bytes=409600,
            human_readable="400.0 KiB",
        )
    ]


def test_collect_sqlite_handles_index_list_failure() -> None:
    class IndexFailureConnection(DummySyncConnection):
        def __init__(self) -> None:
            super().__init__(
                dialect_name="sqlite",
                results=[DummyResult(rows=[{"size": 2048}])],
            )

        def execute(self, statement, params=None):  # type: ignore[override]
            text_statement = str(statement)
            if "PRAGMA index_list" in text_statement:
                raise RuntimeError("index list not available")
            return super().execute(statement, params)

    connection = IndexFailureConnection()
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    metrics = tool._collect_sqlite(connection, "books")

    categories = {metric.category for metric in metrics}
    assert categories == {"data", "total"}


def test_collect_sqlite_via_pages_handles_missing_values() -> None:
    connection = DummySyncConnection(
        dialect_name="sqlite",
        results=[DummyResult(scalar=None), DummyResult(scalar=4096)],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    assert tool._collect_sqlite_via_pages(connection) == []


def test_collect_oracle_handles_missing_indexes() -> None:
    connection = DummySyncConnection(
        dialect_name="oracle",
        results=[
            DummyResult(rows=[]),
            DummyResult(rows=[{"index_name": None}, {"index_name": "IDX"}]),
            DummyResult(rows=[{}]),
        ],
    )

    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))
    assert tool._collect_oracle(connection, None, "books") == []


def test_collect_mssql_missing_rows() -> None:
    connection = DummySyncConnection(dialect_name="mssql", results=[DummyResult(rows=[])])
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    assert tool._collect_mssql(connection, "dbo", "books") == []


def test_helper_functions_behave_consistently() -> None:
    tool = DatabaseSizeTool(SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://"))

    assert tool._normalize_schema_for_output(None) is None
    assert tool._normalize_schema_for_output("public") == "public"

    assert DatabaseSizeTool._combine_bytes(None, None) is None
    assert DatabaseSizeTool._combine_bytes(1, None, 2) == 3

    assert DatabaseSizeTool._multiply(None, 2) is None
    assert DatabaseSizeTool._multiply(3, 4) == 12

    assert DatabaseSizeTool._int_or_none("invalid") is None
    assert DatabaseSizeTool._int_or_none(-1) is None
    assert DatabaseSizeTool._int_or_none("5") == 5

    assert DatabaseSizeTool._format_bytes(None) is None
    assert DatabaseSizeTool._format_bytes(-1) is None
    assert DatabaseSizeTool._format_bytes(500) == "500 B"


