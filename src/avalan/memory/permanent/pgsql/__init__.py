from ....entities import EngineMessage, EngineMessageScored, Message
from ....memory import MemoryStore, MemoryChunk
from ....memory.permanent import (
    PermanentMessage,
    PermanentMessageScored,
    RecordNotFoundException,
    RecordNotSavedException,
)
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types import TypeInfo
from typing import Optional, TypeVar, Type, Union


T = TypeVar("T")


class BasePgsqlMemory(MemoryStore[T]):
    _database: AsyncConnection

    def __init__(self, database: AsyncConnectionPool):
        self._database = database

    async def open(self) -> None:
        await self._database.open()

    async def search(self, query: str) -> Optional[list[T]]:
        raise NotImplementedError()

    async def _fetch_all(
        self, entity: Type[T], query: str, parameters: tuple
    ) -> list[T]:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, parameters)
                results = await cursor.fetchall()
                await cursor.close()
                return (
                    [entity(**dict(result)) for result in results]
                    if results is not None
                    else []
                )

    async def _fetch_one(
        self, entity: Type[T], query: str, parameters: tuple
    ) -> T:
        result = await self._try_fetch_one(entity, query, parameters)
        if result is None:
            raise RecordNotFoundException()
        return result

    async def _fetch_field(
        self, field: str, query: str, parameters: Optional[tuple] = None
    ) -> Optional[str]:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                row = dict(result) if result is not None else None
                return row[field] if row else None
        return None

    async def _has_one(self, query: str, parameters: tuple) -> bool:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                return result is not None

    async def _try_fetch_one(
        self, entity: Type[T], query: str, parameters: tuple
    ) -> Optional[T]:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                return entity(**dict(result)) if result is not None else None

    async def _update_and_fetch_one(
        self, entity: Type[T], query: str, parameters: tuple
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
                await cursor.execute(query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                if result is None:
                    raise RecordNotSavedException()
                return dict(result)

    async def _update(self, query: str, parameters: tuple) -> None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, parameters)
                await cursor.close()


class PgsqlMemory(BasePgsqlMemory[MemoryChunk[T]]):
    _composite_types: Optional[list[str]]

    @classmethod
    async def create_instance_from_pool(
        cls,
        pool: AsyncConnectionPool,
        *args,
        **kwargs,
    ):
        memory = cls(dsn=None, pool=pool, **kwargs)
        return memory

    def __init__(
        self,
        dsn: Optional[str],
        *args,
        pool: Optional[AsyncConnectionPool] = None,
        composite_types: Optional[list[str]] = None,
        pool_minimum: Optional[int] = None,
        pool_maximum: Optional[int] = None,
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
            super().__init__(database=pool, **kwargs)
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
            super().__init__(database=database, **kwargs)

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
        messages: Union[list[PermanentMessage], list[PermanentMessageScored]],
        *args,
        limit: Optional[int],
        reverse: bool = False,
        scored: bool = False,
    ) -> Union[list[EngineMessage], list[EngineMessageScored]]:
        engine_messages = [
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
            for m in messages
        ]
        if reverse:
            engine_messages.reverse()
        if limit and len(engine_messages) > limit:
            engine_messages = engine_messages[:limit]
        return engine_messages
