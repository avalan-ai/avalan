from ....entities import EngineMessage, EngineMessageScored, Message
from ....memory import MemoryStore, MemoryChunk
from ....memory.permanent import (
    PermanentMessage,
    PermanentMessageScored,
    RecordNotFoundException,
    RecordNotSavedException,
)
from logging import Logger
from time import perf_counter
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection, AsyncCursor
from psycopg.rows import dict_row
from psycopg.types import TypeInfo
from typing import TypeVar

T = TypeVar("T")


class BasePgsqlMemory(MemoryStore[T]):
    _database: AsyncConnection
    _logger: Logger

    def __init__(self, database: AsyncConnectionPool, logger: Logger):
        self._database = database
        self._logger = logger

    async def open(self) -> None:
        await self._database.open()

    async def search(self, query: str) -> list[T] | None:
        raise NotImplementedError()

    async def _execute(
        self, cursor: AsyncCursor, query: str, parameters: tuple | None
    ) -> None:
        self._logger.debug("Executing query: %s with %s", query, parameters)
        start = perf_counter()
        await cursor.execute(query, parameters)
        self._logger.debug(
            "Query finished in %.3f seconds", perf_counter() - start
        )

    async def _fetch_all(
        self, entity: type[T], query: str, parameters: tuple
    ) -> list[T]:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                results = await cursor.fetchall()
                await cursor.close()
                return (
                    [entity(**dict(result)) for result in results]
                    if results is not None
                    else []
                )

    async def _fetch_one(
        self, entity: type[T], query: str, parameters: tuple
    ) -> T:
        result = await self._try_fetch_one(entity, query, parameters)
        if result is None:
            raise RecordNotFoundException()
        return result

    async def _fetch_field(
        self, field: str, query: str, parameters: tuple | None = None
    ) -> str | None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                row = dict(result) if result is not None else None
                return row[field] if row else None
        return None

    async def _has_one(self, query: str, parameters: tuple) -> bool:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                return result is not None

    async def _try_fetch_one(
        self, entity: type[T], query: str, parameters: tuple
    ) -> T | None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                return entity(**dict(result)) if result is not None else None

    async def _update_and_fetch_one(
        self, entity: type[T], query: str, parameters: tuple
    ) -> T:
        row = await self._update_and_fetch_row(query, parameters)
        return entity(**row)

    async def _update_and_fetch_field(
        self, field: str, query: str, parameters: tuple
    ) -> str:
        row = await self._update_and_fetch_row(query, parameters)
        return row[field]

    async def _update_and_fetch_row(
        self, query: str, parameters: tuple
    ) -> dict:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                if result is None:
                    raise RecordNotSavedException()
                return dict(result)

    async def _update(self, query: str, parameters: tuple) -> None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                await cursor.close()


class PgsqlMemory(BasePgsqlMemory[MemoryChunk[T]]):
    _composite_types: list[str] | None

    @classmethod
    async def create_instance_from_pool(
        cls,
        pool: AsyncConnectionPool,
        *,
        logger: Logger,
        **kwargs,
    ):
        memory = cls(dsn=None, pool=pool, logger=logger, **kwargs)
        return memory

    def __init__(
        self,
        dsn: str | None,
        *args,
        pool: AsyncConnectionPool | None = None,
        composite_types: list[str] | None = None,
        pool_minimum: int | None = None,
        pool_maximum: int | None = None,
        logger: Logger,
        **kwargs,
    ):
        assert pool or (
            dsn
            and pool_minimum
            and pool_minimum
            and pool_minimum > 0
            and pool_maximum > pool_minimum
        )

        if pool:
            super().__init__(database=pool, logger=logger, **kwargs)
        else:
            self._composite_types = composite_types

            if "//" not in dsn:
                dsn = f"postgresql://{dsn}"

            database = AsyncConnectionPool(
                min_size=pool_minimum,
                max_size=pool_maximum,
                conninfo=dsn,
                configure=self._configure_connection,
                open=False,
            )
            super().__init__(database=database, logger=logger, **kwargs)

    async def _configure_connection(self, connection: AsyncConnection):
        connection.row_factory = dict_row
        await connection.set_autocommit(True)
        if self._composite_types:
            for composite_type_name in self._composite_types:
                composite_type = await TypeInfo.fetch(
                    connection, composite_type_name
                )
                if composite_type:
                    composite_type.register(connection)
        await register_vector_async(connection)

    @staticmethod
    def _to_engine_messages(
        messages: list[PermanentMessage] | list[PermanentMessageScored],
        *args,
        limit: int | None,
        reverse: bool = False,
        scored: bool = False,
    ) -> list[EngineMessage] | list[EngineMessageScored]:
        engine_messages = [
            (
                EngineMessageScored(
                    agent_id=m.agent_id,
                    model_id=m.model_id,
                    message=Message(role=m.author, content=m.data),
                    score=m.score,
                )
                if scored
                else EngineMessage(
                    agent_id=m.agent_id,
                    model_id=m.model_id,
                    message=Message(role=m.author, content=m.data),
                )
            )
            for m in messages
        ]
        if reverse:
            engine_messages.reverse()
        if limit and len(engine_messages) > limit:
            engine_messages = engine_messages[:limit]
        return engine_messages
