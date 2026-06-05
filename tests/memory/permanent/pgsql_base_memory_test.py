from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

try:
    from psycopg import AsyncConnection, AsyncCursor
    from psycopg.errors import UndefinedFile
    from psycopg_pool import AsyncConnectionPool
except ImportError:
    pytest.skip("psycopg pq wrapper is unavailable", allow_module_level=True)

from avalan.memory.permanent import (
    RecordNotFoundException,
    RecordNotSavedException,
)
from avalan.memory.permanent.pgsql import (
    BasePgsqlMemory,
    PgsqlMemory,
    PgsqlVectorExtensionError,
)
from avalan.pgsql import PsycopgAsyncDatabase


@dataclass
class DummyEntity:
    value: int


class DummyBaseMemory(BasePgsqlMemory[DummyEntity]):
    async def append(self, agent_id, data) -> None:
        pass

    async def reset(self) -> None:
        pass


class DummyPgsqlMemory(PgsqlMemory[DummyEntity]):
    async def append(self, agent_id, data) -> None:
        pass

    async def reset(self) -> None:
        pass


class BasePgsqlMemoryTestCase(IsolatedAsyncioTestCase):
    @staticmethod
    def mock_query(result, fetch_all=False):
        cursor_mock = AsyncMock(spec=AsyncCursor)
        cursor_mock.__aenter__.return_value = cursor_mock
        if fetch_all:
            cursor_mock.fetchall.return_value = result
        else:
            cursor_mock.fetchone.return_value = result
        connection_mock = AsyncMock(spec=AsyncConnection)
        connection_mock.cursor.return_value = cursor_mock
        connection_mock.__aenter__.return_value = connection_mock
        pool_mock = AsyncMock(spec=AsyncConnectionPool)
        pool_mock.connection.return_value = connection_mock
        pool_mock.__aenter__.return_value = pool_mock
        return pool_mock, connection_mock, cursor_mock

    async def test_open_and_search(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        await memory.open()
        pool.open.assert_awaited_once()
        with self.assertRaises(NotImplementedError):
            await memory.search("q")

    async def test_close_and_context_manager(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool, logger=MagicMock())

        async with memory as opened:
            self.assertIs(opened, memory)

        pool.open.assert_awaited_once()
        pool.close.assert_awaited_once()

    async def test_fetch_one_found(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        with patch.object(
            memory, "_try_fetch_one", AsyncMock(return_value=DummyEntity(1))
        ) as patch_try:
            result = await memory._fetch_one(DummyEntity, "query", (1,))
        patch_try.assert_awaited_once_with(DummyEntity, "query", (1,))
        self.assertEqual(result, DummyEntity(1))

    async def test_fetch_one_not_found(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        with patch.object(
            memory, "_try_fetch_one", AsyncMock(return_value=None)
        ):
            with self.assertRaises(RecordNotFoundException):
                await memory._fetch_one(DummyEntity, "query", (1,))

    async def test_has_one(self):
        for expected, record in [(True, {"v": 1}), (False, None)]:
            pool, _, cursor = self.mock_query(record)
            memory = DummyBaseMemory(pool, logger=MagicMock())
            result = await memory._has_one("q", (1,))
            cursor.execute.assert_awaited_once_with("q", (1,))
            self.assertEqual(result, expected)

    async def test_try_fetch_one(self):
        pool, _, cursor = self.mock_query({"value": 10})
        memory = DummyBaseMemory(pool, logger=MagicMock())
        result = await memory._try_fetch_one(DummyEntity, "q", (1,))
        cursor.execute.assert_awaited_once_with("q", (1,))
        self.assertEqual(result, DummyEntity(value=10))

        pool, _, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        result = await memory._try_fetch_one(DummyEntity, "q", (1,))
        self.assertIsNone(result)

    async def test_fetch_field(self):
        pool, _, cursor = self.mock_query({"f": "v"})
        memory = DummyBaseMemory(pool, logger=MagicMock())
        result = await memory._fetch_field("f", "q", (1,))
        cursor.execute.assert_awaited_once_with("q", (1,))
        self.assertEqual(result, "v")

        pool, _, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        result = await memory._fetch_field("f", "q", (2,))
        cursor.execute.assert_awaited_once_with("q", (2,))
        self.assertIsNone(result)

    async def test_update_and_fetch_one_and_field(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        with patch.object(
            memory,
            "_update_and_fetch_row",
            AsyncMock(side_effect=[{"value": 2}, {"field": "x"}]),
        ) as patch_row:
            ent = await memory._update_and_fetch_one(DummyEntity, "q1", (1,))
            val = await memory._update_and_fetch_field("field", "q2", (2,))
        patch_row.assert_has_awaits(
            [
                call("q1", (1,)),
                call("q2", (2,)),
            ]
        )
        self.assertEqual(ent, DummyEntity(2))
        self.assertEqual(val, "x")

    async def test_update_and_fetch_row_and_update(self):
        pool, conn, cursor = self.mock_query({"a": 1})
        memory = DummyBaseMemory(pool, logger=MagicMock())
        row = await memory._update_and_fetch_row("q", (1,))
        cursor.execute.assert_awaited_once_with("q", (1,))
        self.assertEqual(row, {"a": 1})

        pool, conn, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        with self.assertRaises(RecordNotSavedException):
            await memory._update_and_fetch_row("q", (1,))

        pool, conn, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool, logger=MagicMock())
        await memory._update("u", (3,))
        cursor.execute.assert_awaited_once_with("u", (3,))


class PgsqlMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_dsn_prefix_and_configure(self):
        pool_cls = MagicMock()
        pool_instance = AsyncMock(spec=AsyncConnectionPool)
        pool_cls.return_value = pool_instance
        memory = DummyPgsqlMemory(
            dsn="user@host/db",
            pool_minimum=1,
            pool_maximum=2,
            logger=MagicMock(),
            module_importer={
                "psycopg_pool": SimpleNamespace(AsyncConnectionPool=pool_cls)
            }.__getitem__,
        )
        database = cast(PsycopgAsyncDatabase, memory._database)
        self.assertIs(database.pool, pool_instance)
        pool_cls.assert_called_once()
        called_kwargs = pool_cls.call_args.kwargs
        self.assertEqual(
            called_kwargs["conninfo"], "postgresql://user@host/db"
        )
        self.assertIs(database.pool, pool_instance)
        self.assertIsNone(memory._composite_types)

    async def test_configure_connection(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        vector_module = SimpleNamespace(register_vector_async=AsyncMock())
        type_info = MagicMock()
        type_module = SimpleNamespace(
            TypeInfo=SimpleNamespace(fetch=AsyncMock(return_value=type_info))
        )
        memory = DummyPgsqlMemory(
            dsn=None,
            pool=pool,
            logger=MagicMock(),
            module_importer={
                "pgvector.psycopg": vector_module,
                "psycopg.types": type_module,
            }.__getitem__,
        )
        memory._composite_types = ["ctype"]
        connection = AsyncMock(spec=AsyncConnection)
        with (
            patch.object(
                memory,
                "_ensure_vector_extension",
                AsyncMock(),
            ) as ensure_patch,
        ):
            await memory._configure_connection(connection)
        type_module.TypeInfo.fetch.assert_awaited_once_with(
            connection,
            "ctype",
        )
        type_info.register.assert_called_once_with(connection)
        ensure_patch.assert_awaited_once_with(connection)
        vector_module.register_vector_async.assert_awaited_once_with(
            connection
        )

    async def test_borrowed_pool_is_not_closed(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyPgsqlMemory(dsn=None, pool=pool, logger=MagicMock())

        await memory.open()
        await memory.aclose()

        pool.open.assert_not_called()
        pool.close.assert_not_called()

    async def test_ensure_vector_extension_missing_extension(self):
        pool, connection, cursor = BasePgsqlMemoryTestCase.mock_query(
            {"has_vector_extension": False}
        )
        memory = DummyPgsqlMemory(dsn=None, pool=pool, logger=MagicMock())

        with self.assertRaises(PgsqlVectorExtensionError) as caught:
            await memory._ensure_vector_extension(connection)

        cursor.execute.assert_awaited_once()
        self.assertIn("not enabled", str(caught.exception))

    async def test_ensure_vector_extension_broken_library(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyPgsqlMemory(dsn=None, pool=pool, logger=MagicMock())
        cursor = AsyncMock(spec=AsyncCursor)
        cursor.__aenter__.return_value = cursor
        cursor.fetchone = AsyncMock(
            side_effect=[{"has_vector_extension": True}, None]
        )
        cursor.execute = AsyncMock(
            side_effect=[
                None,
                UndefinedFile('could not access file "$libdir/vector"'),
            ]
        )
        connection = AsyncMock(spec=AsyncConnection)
        connection.cursor.return_value = cursor
        connection.__aenter__.return_value = connection

        with self.assertRaises(PgsqlVectorExtensionError) as caught:
            await memory._ensure_vector_extension(connection)

        self.assertEqual(cursor.execute.await_count, 2)
        self.assertIn(
            "cannot load the pgvector library", str(caught.exception)
        )
