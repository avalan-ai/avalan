from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    create_engine,
    text,
)
from sqlalchemy import (
    Table as SATable,
)
from sqlalchemy.dialects import mssql, mysql, oracle, postgresql, sqlite

from avalan.entities import ToolCallContext
from avalan.tool.database import DatabaseSampleTool, DatabaseToolSettings


def dummy_create_async_engine(dsn: str):
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
        def __init__(self, engine):
            self.engine = engine
            self.conn = None

        async def __aenter__(self):
            self.conn = self.engine.connect()
            return DummyAsyncConn(self.conn)

        async def __aexit__(self, exc_type, exc, tb):
            self.conn.close()
            return False

    class DummyAsyncEngine:
        def __init__(self, engine):
            self.engine = engine

        def connect(self):
            return DummyConnCtx(self.engine)

        async def dispose(self):
            self.engine.dispose()

    return DummyAsyncEngine(engine)


class DatabaseSampleToolTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dsn = f"sqlite:///{self.tmp.name}/db.sqlite"
        engine = create_engine(self.dsn)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE books("
                    "id INTEGER PRIMARY KEY, author_id INTEGER, title TEXT)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO books(id, author_id, title) VALUES (1, 1,"
                    " 'Atlas')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO books(id, author_id, title) VALUES (2, 2,"
                    " 'Bestiary')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO books(id, author_id, title) VALUES (3, 1,"
                    " 'Chronicle')"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE numbers(id INTEGER PRIMARY KEY, label TEXT)"
                )
            )
            for idx in range(20):
                conn.execute(
                    text(
                        "INSERT INTO numbers(id, label) VALUES (:id, :label)"
                    ),
                    {"id": idx, "label": f"value-{idx}"},
                )
            conn.execute(
                text(
                    'CREATE TABLE "CamelCase"("Id" INTEGER PRIMARY KEY,'
                    ' "Value" TEXT)'
                )
            )
            conn.execute(
                text(
                    'INSERT INTO "CamelCase"("Id", "Value") VALUES (1, "Item")'
                )
            )
        engine.dispose()
        self.settings = DatabaseToolSettings(dsn=self.dsn)
        self.settings_lower = DatabaseToolSettings(
            dsn=self.dsn, identifier_case="lower"
        )

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    async def test_sample_tool_returns_limited_rows(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "books",
                columns=["id", "title"],
                order={"title": "desc"},
                count=1,
                context=ToolCallContext(),
            )
            self.assertEqual(rows, [{"id": 3, "title": "Chronicle"}])
        finally:
            await engine.dispose()

    async def test_sample_tool_applies_filters(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "books",
                columns=["title"],
                conditions="author_id = 1",
                order={"id": "asc"},
                count=5,
                context=ToolCallContext(),
            )
            self.assertEqual(
                rows, [{"title": "Atlas"}, {"title": "Chronicle"}]
            )
        finally:
            await engine.dispose()

    async def test_sample_tool_defaults_to_ten_rows(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "numbers",
                columns=["id"],
                order={"id": "asc"},
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["id"], 0)
            self.assertEqual(rows[-1]["id"], 9)
        finally:
            await engine.dispose()

    async def test_sample_tool_returns_all_columns_when_not_specified(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "books",
                order={"id": "asc"},
                count=1,
                context=ToolCallContext(),
            )
            self.assertEqual(
                rows,
                [
                    {
                        "id": 1,
                        "author_id": 1,
                        "title": "Atlas",
                    }
                ],
            )
        finally:
            await engine.dispose()

    async def test_sample_tool_denormalizes_column_names(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings_lower)
        try:
            rows = await tool(
                "CamelCase",
                columns=["value"],
                count=1,
                context=ToolCallContext(),
            )
            self.assertEqual(rows, [{"Value": "Item"}])
        finally:
            await engine.dispose()

    async def test_sample_tool_resolves_normalized_column_names(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings_lower)
        try:
            rows = await tool(
                "CamelCase",
                columns=["VALUE"],
                count=1,
                context=ToolCallContext(),
            )
            self.assertEqual(rows, [{"Value": "Item"}])
        finally:
            await engine.dispose()

    async def test_sample_tool_raises_for_unknown_column(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            with self.assertRaises(ValueError):
                await tool(
                    "books",
                    columns=["missing"],
                    count=1,
                    context=ToolCallContext(),
                )
        finally:
            await engine.dispose()

    async def test_sample_tool_requires_table_name(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            with self.assertRaises(AssertionError):
                await tool(
                    "",
                    context=ToolCallContext(),
                )
        finally:
            await engine.dispose()

    async def test_sample_tool_rejects_empty_column_name(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            with self.assertRaises(AssertionError):
                await tool(
                    "books",
                    columns=[""],
                    context=ToolCallContext(),
                )
        finally:
            await engine.dispose()

    async def test_sample_tool_rejects_empty_order(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            with self.assertRaises(AssertionError):
                await tool(
                    "books",
                    order={},
                    context=ToolCallContext(),
                )
        finally:
            await engine.dispose()

    async def test_sample_tool_rejects_non_positive_count(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            with self.assertRaises(AssertionError):
                await tool(
                    "books",
                    count=0,
                    context=ToolCallContext(),
                )
        finally:
            await engine.dispose()


class DatabaseSampleToolStatementTestCase(TestCase):
    def test_build_select_statement_compiles_for_supported_vendors(self):
        tool = DatabaseSampleTool(
            SimpleNamespace(), DatabaseToolSettings(dsn="sqlite://")
        )
        table = SATable(
            "records",
            MetaData(),
            Column("id", Integer),
            Column("name", String),
        )

        with patch.object(tool, "_reflect_table", return_value=table):
            stmt = tool._build_select_statement(
                SimpleNamespace(),
                schema=None,
                actual_table="records",
                requested_columns=["id", "name"],
                conditions="id > 0",
                order={"name": "desc"},
                limit=5,
            )

        dialects = {
            "sqlite": sqlite.dialect(),
            "postgresql": postgresql.dialect(),
            "mysql": mysql.dialect(),
            "mssql": mssql.dialect(),
            "oracle": oracle.dialect(),
        }

        for name, dialect in dialects.items():
            sql = str(
                stmt.compile(
                    dialect=dialect, compile_kwargs={"literal_binds": True}
                )
            )
            upper_sql = sql.upper()
            self.assertIn("ORDER BY", upper_sql)
            self.assertIn("NAME", upper_sql)
            if name == "mssql":
                self.assertIn("TOP 5", upper_sql)
            elif name == "oracle":
                self.assertTrue(
                    "FETCH FIRST 5 ROWS" in upper_sql or "ROWNUM" in upper_sql
                )
            else:
                self.assertIn("LIMIT 5", upper_sql)
            self.assertIn("ID > 0", upper_sql)

    def test_split_schema_and_table_handles_qualified_names(self):
        result = DatabaseSampleTool._split_schema_and_table("public.records")
        self.assertEqual(result, ("public", "records"))

        result = DatabaseSampleTool._split_schema_and_table(".records")
        self.assertEqual(result, (None, "records"))

    def test_build_ordering_rejects_invalid_direction(self):
        tool = DatabaseSampleTool(
            SimpleNamespace(),
            DatabaseToolSettings(dsn="sqlite://"),
            normalizer=None,
        )
        table = SATable(
            "records",
            MetaData(),
            Column("id", Integer),
        )

        with self.assertRaises(AssertionError):
            tool._build_ordering(table, {"id": "sideways"})
