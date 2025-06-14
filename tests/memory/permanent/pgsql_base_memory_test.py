from dataclasses import dataclass
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch, call
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection, AsyncCursor
from psycopg.rows import dict_row

from avalan.memory.permanent.pgsql import BasePgsqlMemory, PgsqlMemory
from avalan.memory.permanent import (
    RecordNotFoundException,
    RecordNotSavedException,
)


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
        memory = DummyBaseMemory(pool)
        await memory.open()
        pool.open.assert_awaited_once()
        with self.assertRaises(NotImplementedError):
            await memory.search("q")

    async def test_fetch_one_found(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool)
        with patch.object(
            memory, "_try_fetch_one", AsyncMock(return_value=DummyEntity(1))
        ) as patch_try:
            result = await memory._fetch_one(DummyEntity, "query", (1,))
        patch_try.assert_awaited_once_with(DummyEntity, "query", (1,))
        self.assertEqual(result, DummyEntity(1))

    async def test_fetch_one_not_found(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool)
        with patch.object(
            memory, "_try_fetch_one", AsyncMock(return_value=None)
        ):
            with self.assertRaises(RecordNotFoundException):
                await memory._fetch_one(DummyEntity, "query", (1,))

    async def test_has_one(self):
        for expected, record in [(True, {"v": 1}), (False, None)]:
            pool, _, cursor = self.mock_query(record)
            memory = DummyBaseMemory(pool)
            result = await memory._has_one("q", (1,))
            cursor.execute.assert_awaited_once_with("q", (1,))
            self.assertEqual(result, expected)

    async def test_try_fetch_one(self):
        pool, _, cursor = self.mock_query({"value": 10})
        memory = DummyBaseMemory(pool)
        result = await memory._try_fetch_one(DummyEntity, "q", (1,))
        cursor.execute.assert_awaited_once_with("q", (1,))
        self.assertEqual(result, DummyEntity(value=10))

        pool, _, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool)
        result = await memory._try_fetch_one(DummyEntity, "q", (1,))
        self.assertIsNone(result)

    async def test_update_and_fetch_one_and_field(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyBaseMemory(pool)
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
        memory = DummyBaseMemory(pool)
        row = await memory._update_and_fetch_row("q", (1,))
        cursor.execute.assert_awaited_once_with("q", (1,))
        self.assertEqual(row, {"a": 1})

        pool, conn, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool)
        with self.assertRaises(RecordNotSavedException):
            await memory._update_and_fetch_row("q", (1,))

        pool, conn, cursor = self.mock_query(None)
        memory = DummyBaseMemory(pool)
        await memory._update("u", (3,))
        cursor.execute.assert_awaited_once_with("u", (3,))


class PgsqlMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_dsn_prefix_and_configure(self):
        with patch(
            "avalan.memory.permanent.pgsql.AsyncConnectionPool",
            autospec=True,
        ) as pool_cls:
            pool_instance = AsyncMock(spec=AsyncConnectionPool)
            pool_cls.return_value = pool_instance
            memory = DummyPgsqlMemory(
                dsn="user@host/db",
                pool_minimum=1,
                pool_maximum=2,
            )
        pool_cls.assert_called_once()
        called_kwargs = pool_cls.call_args.kwargs
        self.assertEqual(
            called_kwargs["conninfo"], "postgresql://user@host/db"
        )
        self.assertIs(memory._database, pool_instance)
        self.assertIsNone(memory._composite_types)

    async def test_configure_connection(self):
        pool = AsyncMock(spec=AsyncConnectionPool)
        memory = DummyPgsqlMemory(dsn=None, pool=pool)
        memory._composite_types = ["ctype"]
        connection = AsyncMock(spec=AsyncConnection)
        type_info = MagicMock()
        with (
            patch(
                "avalan.memory.permanent.pgsql.TypeInfo.fetch",
                AsyncMock(return_value=type_info),
            ) as fetch_patch,
            patch(
                "avalan.memory.permanent.pgsql.register_vector_async",
                AsyncMock(),
            ) as vector_patch,
        ):
            await memory._configure_connection(connection)
        self.assertIs(connection.row_factory, dict_row)
        connection.set_autocommit.assert_awaited_once_with(True)
        fetch_patch.assert_awaited_once_with(connection, "ctype")
        type_info.register.assert_called_once_with(connection)
        vector_patch.assert_awaited_once_with(connection)
