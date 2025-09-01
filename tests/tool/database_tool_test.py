from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch
from avalan.entities import ToolCallContext
from avalan.tool.database import (
    DatabaseCountTool,
    DatabaseInspectTool,
    DatabaseRunTool,
    DatabaseTablesTool,
    DatabaseTool,
    DatabaseToolSet,
    DatabaseToolSettings,
)


def dummy_create_async_engine(dsn: str, **kwargs):
    engine = create_engine(dsn)

    class DummyAsyncConn:
        def __init__(self, conn):
            self.conn = conn

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
            self.disposed = False

        def connect(self):
            return DummyConnCtx(self.engine)

        def begin(self):
            return DummyConnCtx(self.engine)

        async def dispose(self):
            self.engine.dispose()
            self.disposed = True

    return DummyAsyncEngine(engine)


class DatabaseToolSetTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dsn = f"sqlite:///{self.tmp.name}/db.sqlite"
        engine = create_engine(self.dsn)
        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE authors(id INTEGER PRIMARY KEY, name TEXT)")
            )
            conn.execute(
                text(
                    "CREATE TABLE books(id INTEGER PRIMARY KEY, author_id "
                    "INTEGER, title TEXT, FOREIGN KEY(author_id) REFERENCES "
                    "authors(id))"
                )
            )
            conn.execute(text("CREATE TABLE empty(id INTEGER PRIMARY KEY)"))
            conn.execute(text("INSERT INTO authors(name) VALUES ('Author')"))
            conn.execute(
                text("INSERT INTO books(author_id, title) VALUES (1, 'Book')")
            )
        engine.dispose()
        self.patcher = patch(
            "avalan.tool.database.create_async_engine",
            dummy_create_async_engine,
        )
        self.patcher.start()
        self.settings = DatabaseToolSettings(dsn=self.dsn)

    async def asyncTearDown(self) -> None:
        self.patcher.stop()
        self.tmp.cleanup()

    async def test_run_tool_returns_rows(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, self.settings)
        rows = await tool(
            "SELECT id, title FROM books", context=ToolCallContext()
        )
        self.assertEqual(rows, [{"id": 1, "title": "Book"}])
        await engine.dispose()

    async def test_run_tool_no_rows(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseRunTool(engine, self.settings)
        rows = await tool(
            "INSERT INTO authors(name) VALUES ('Other')",
            context=ToolCallContext(),
        )
        self.assertEqual(rows, [])
        await engine.dispose()

    async def test_tables_tool_lists_tables(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseTablesTool(engine, self.settings)
        tables = await tool(context=ToolCallContext())
        self.assertIn("main", tables)
        self.assertIn("authors", tables["main"])
        self.assertIn("books", tables["main"])
        await engine.dispose()

    async def test_inspect_tool_returns_schema(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseInspectTool(engine, self.settings)
        table = await tool("books", context=ToolCallContext())
        self.assertEqual(table.name, "books")
        self.assertEqual(table.columns["id"], "INTEGER")
        self.assertEqual(table.columns["title"], "TEXT")
        self.assertEqual(len(table.foreign_keys), 1)
        fk = table.foreign_keys[0]
        self.assertEqual(fk.field, "author_id")
        self.assertEqual(fk.ref_table, "main.authors")
        self.assertEqual(fk.ref_field, "id")
        await engine.dispose()

    async def test_count_tool_returns_count(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, self.settings)
        count = await tool("authors", context=ToolCallContext())
        self.assertEqual(count, 1)
        await engine.dispose()

    async def test_count_tool_empty_table(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, self.settings)
        count = await tool("empty", context=ToolCallContext())
        self.assertEqual(count, 0)
        await engine.dispose()

    async def test_count_tool_wrong_table_error(self):
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseCountTool(engine, self.settings)
        with self.assertRaises(OperationalError):
            await tool("missing", context=ToolCallContext())
        await engine.dispose()

    async def test_toolset_reuses_engine_and_disposes(self):
        async with DatabaseToolSet(self.settings) as toolset:
            count_tool, inspect_tool, run_tool, tables_tool = toolset.tools
            count = await count_tool("authors", context=ToolCallContext())
            self.assertEqual(count, 1)
            rows = await run_tool(
                "SELECT id FROM authors", context=ToolCallContext()
            )
            self.assertEqual(rows, [{"id": 1}])
            table = await inspect_tool("books", context=ToolCallContext())
            self.assertEqual(table.name, "books")
            tables = await tables_tool(context=ToolCallContext())
            self.assertIn("books", tables["main"])
        self.assertTrue(toolset._engine.disposed)


class DatabaseInspectCollectTestCase(TestCase):
    def test_collect_prefixes_schema_name(self):
        inspector = SimpleNamespace(
            default_schema_name="public",
            get_columns=lambda table_name, schema: [
                {"name": "id", "type": "INTEGER"}
            ],
            get_foreign_keys=lambda table_name, schema: [],
        )

        with (
            patch("avalan.tool.database.inspect", return_value=inspector),
            patch(
                "avalan.tool.database.DatabaseTool._schemas",
                return_value=("public", []),
            ),
        ):
            table = DatabaseInspectTool._collect(
                SimpleNamespace(), schema="other", table_name="t"
            )
        self.assertEqual(table.name, "other.t")


class DatabaseSchemasTestCase(TestCase):
    def test_postgresql_schemas(self):
        connection = SimpleNamespace(
            dialect=SimpleNamespace(name="postgresql")
        )
        inspector = SimpleNamespace(
            default_schema_name="public",
            get_schema_names=lambda: [
                "information_schema",
                "pg_catalog",
                "other",
            ],
        )
        default, schemas = DatabaseTool._schemas(connection, inspector)
        self.assertEqual(default, "public")
        self.assertEqual(schemas, ["other", "public"])
