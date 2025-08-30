from sqlalchemy import create_engine, text
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from avalan.tool.database import (
    DatabaseInspectTool,
    DatabaseRunTool,
    DatabaseTablesTool,
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

        def connect(self):
            return DummyConnCtx(self.engine)

        def begin(self):
            return DummyConnCtx(self.engine)

        async def dispose(self):
            self.engine.dispose()

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
                    "CREATE TABLE books(id INTEGER PRIMARY KEY, author_id"
                    " INTEGER, title TEXT, FOREIGN KEY(author_id) REFERENCES"
                    " authors(id))"
                )
            )
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
        tool = DatabaseRunTool(self.settings)
        rows = await tool("SELECT id, title FROM books")
        self.assertEqual(rows, [{"id": 1, "title": "Book"}])

    async def test_tables_tool_lists_tables(self):
        tool = DatabaseTablesTool(self.settings)
        tables = await tool()
        self.assertIn("main", tables)
        self.assertIn("authors", tables["main"])
        self.assertIn("books", tables["main"])

    async def test_inspect_tool_returns_schema(self):
        tool = DatabaseInspectTool(self.settings)
        table = await tool("books")
        self.assertEqual(table.name, "books")
        self.assertEqual(table.columns["id"], "INTEGER")
        self.assertEqual(table.columns["title"], "TEXT")
        self.assertEqual(len(table.foreign_keys), 1)
        fk = table.foreign_keys[0]
        self.assertEqual(fk.field, "author_id")
        self.assertEqual(fk.ref_table, "main.authors")
        self.assertEqual(fk.ref_field, "id")
