from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from sqlalchemy import (
    create_engine,
    text,
)

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


class DatabaseBinaryDataTestCase(IsolatedAsyncioTestCase):
    """Test that binary data from database columns is properly handled."""

    async def asyncSetUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dsn = f"sqlite:///{self.tmp.name}/db.sqlite"
        engine = create_engine(self.dsn)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE files("
                    "id INTEGER PRIMARY KEY, name TEXT, content BLOB)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (1,"
                    " 'doc.pdf', :content)"
                ),
                {"content": b"\x89PDF\x0d\x0a\x1a\x0a\x00\x00\x00\x0d"},
            )
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (2,"
                    " 'image.png', :content)"
                ),
                {"content": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"},
            )
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (3,"
                    " 'text.txt', :content)"
                ),
                {"content": b"Plain text content"},
            )
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (4,"
                    " 'empty.dat', :content)"
                ),
                {"content": b""},
            )
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (5,"
                    " 'null.dat', NULL)"
                ),
            )
        engine.dispose()
        self.settings = DatabaseToolSettings(dsn=self.dsn)

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    async def test_sample_tool_returns_binary_data_as_bytes(self):
        """Binary columns should be returned as bytes objects."""
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["id", "name", "content"],
                order={"id": "asc"},
                count=1,
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], 1)
            self.assertEqual(rows[0]["name"], "doc.pdf")
            self.assertIsInstance(rows[0]["content"], bytes)
            self.assertEqual(
                rows[0]["content"], b"\x89PDF\x0d\x0a\x1a\x0a\x00\x00\x00\x0d"
            )
        finally:
            await engine.dispose()

    async def test_sample_tool_handles_multiple_binary_rows(self):
        """Multiple rows with binary data should all be properly returned."""
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["id", "content"],
                order={"id": "asc"},
                count=3,
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 3)
            self.assertIsInstance(rows[0]["content"], bytes)
            self.assertIsInstance(rows[1]["content"], bytes)
            self.assertIsInstance(rows[2]["content"], bytes)
            self.assertEqual(
                rows[0]["content"], b"\x89PDF\x0d\x0a\x1a\x0a\x00\x00\x00\x0d"
            )
            self.assertEqual(
                rows[1]["content"], b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            )
            self.assertEqual(rows[2]["content"], b"Plain text content")
        finally:
            await engine.dispose()

    async def test_sample_tool_handles_empty_binary_data(self):
        """Empty binary data should be returned as empty bytes."""
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["id", "content"],
                conditions="id = 4",
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 1)
            self.assertIsInstance(rows[0]["content"], bytes)
            self.assertEqual(rows[0]["content"], b"")
        finally:
            await engine.dispose()

    async def test_sample_tool_handles_null_binary_data(self):
        """NULL binary data should be returned as None."""
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["id", "content"],
                conditions="id = 5",
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["content"])
        finally:
            await engine.dispose()

    async def test_sample_tool_handles_mixed_binary_and_text_columns(self):
        """Mixing binary and text columns should work correctly."""
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                order={"id": "asc"},
                count=2,
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["id"], 1)
            self.assertIsInstance(rows[0]["id"], int)
            self.assertIsInstance(rows[0]["name"], str)
            self.assertIsInstance(rows[0]["content"], bytes)
            self.assertEqual(rows[1]["id"], 2)
            self.assertIsInstance(rows[1]["id"], int)
            self.assertIsInstance(rows[1]["name"], str)
            self.assertIsInstance(rows[1]["content"], bytes)
        finally:
            await engine.dispose()

    async def test_sample_tool_filters_binary_columns(self):
        """Filtering should work with binary columns in results."""
        engine = dummy_create_async_engine(self.dsn)
        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["name", "content"],
                conditions="name LIKE '%.png'",
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "image.png")
            self.assertIsInstance(rows[0]["content"], bytes)
            self.assertEqual(
                rows[0]["content"], b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            )
        finally:
            await engine.dispose()

    async def test_sample_tool_returns_large_binary_data(self):
        """Large binary data should be handled correctly."""
        engine = dummy_create_async_engine(self.dsn)
        large_data = b"\x00" * 10000 + b"\xff" * 10000
        sync_engine = create_engine(self.dsn)
        with sync_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (6,"
                    " 'large.bin', :content)"
                ),
                {"content": large_data},
            )
        sync_engine.dispose()

        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["id", "content"],
                conditions="id = 6",
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 1)
            self.assertIsInstance(rows[0]["content"], bytes)
            self.assertEqual(len(rows[0]["content"]), 20000)
            self.assertEqual(rows[0]["content"], large_data)
        finally:
            await engine.dispose()

    async def test_sample_tool_handles_bytearray_columns(self):
        """Test that bytearray data is also properly handled."""
        engine = dummy_create_async_engine(self.dsn)
        sync_engine = create_engine(self.dsn)
        test_bytearray = bytearray(b"test bytearray data")
        with sync_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO files(id, name, content) VALUES (7,"
                    " 'array.dat', :content)"
                ),
                {"content": bytes(test_bytearray)},
            )
        sync_engine.dispose()

        tool = DatabaseSampleTool(engine, self.settings)
        try:
            rows = await tool(
                "files",
                columns=["id", "content"],
                conditions="id = 7",
                context=ToolCallContext(),
            )
            self.assertEqual(len(rows), 1)
            self.assertIsInstance(rows[0]["content"], (bytes, bytearray))
            self.assertEqual(bytes(rows[0]["content"]), bytes(test_bytearray))
        finally:
            await engine.dispose()
