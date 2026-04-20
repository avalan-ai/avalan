from ....entities import EngineMessage, EngineMessageScored, Message
from ....memory import MemoryStore
from ....memory.permanent import (
    PermanentMessage,
    PermanentMessageScored,
    RecordNotFoundException,
    RecordNotSavedException,
)

from logging import Logger
from time import perf_counter
from typing import Any, Literal, TypeVar, cast, overload

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, AsyncCursor
from psycopg.errors import UndefinedFile
from psycopg.rows import dict_row
from psycopg.types import TypeInfo
from psycopg_pool import AsyncConnectionPool

T = TypeVar("T")
U = TypeVar("U")
QueryParams = tuple[object, ...]


class PgsqlVectorExtensionError(RuntimeError):
    """Signal that PostgreSQL vector support is unavailable."""


class BasePgsqlMemory(MemoryStore[T]):
    _database: AsyncConnectionPool[Any]
    _logger: Logger

    def __init__(
        self, database: AsyncConnectionPool[Any], logger: Logger
    ) -> None:
        self._database = database
        self._logger = logger

    async def open(self) -> None:
        await self._database.open()

    async def search(self, query: str) -> list[T] | None:
        raise NotImplementedError()

    async def _execute(
        self,
        cursor: AsyncCursor[Any],
        query: str,
        parameters: QueryParams | None,
    ) -> None:
        self._logger.debug("Executing query: %s with %s", query, parameters)
        start = perf_counter()
        await cursor.execute(query, parameters)
        self._logger.debug(
            "Query finished in %.3f seconds", perf_counter() - start
        )

    async def _fetch_all(
        self, entity: type[U], query: str, parameters: QueryParams
    ) -> list[U]:
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
        self, entity: type[U], query: str, parameters: QueryParams
    ) -> U:
        result = await self._try_fetch_one(entity, query, parameters)
        if result is None:
            raise RecordNotFoundException()
        return result

    async def _fetch_field(
        self,
        field: str,
        query: str,
        parameters: QueryParams | None = None,
    ) -> str | None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                row = dict(result) if result is not None else None
                return row[field] if row else None

    async def _has_one(self, query: str, parameters: QueryParams) -> bool:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                return result is not None

    async def _try_fetch_one(
        self, entity: type[U], query: str, parameters: QueryParams
    ) -> U | None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                return entity(**dict(result)) if result is not None else None

    async def _update_and_fetch_one(
        self, entity: type[U], query: str, parameters: QueryParams
    ) -> U:
        row = await self._update_and_fetch_row(query, parameters)
        return entity(**row)

    async def _update_and_fetch_field(
        self, field: str, query: str, parameters: QueryParams
    ) -> str:
        row = await self._update_and_fetch_row(query, parameters)
        value = row[field]
        return value if isinstance(value, str) else str(value)

    async def _update_and_fetch_row(
        self, query: str, parameters: QueryParams
    ) -> dict[str, Any]:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                result = await cursor.fetchone()
                await cursor.close()
                if result is None:
                    raise RecordNotSavedException()
                return dict(result)

    async def _update(self, query: str, parameters: QueryParams) -> None:
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await self._execute(cursor, query, parameters)
                await cursor.close()


class PgsqlMemory(BasePgsqlMemory[T]):
    _composite_types: list[str] | None

    @classmethod
    async def create_instance_from_pool(
        cls,
        pool: AsyncConnectionPool[Any],
        *,
        logger: Logger,
        **kwargs: Any,
    ) -> "PgsqlMemory[T]":
        memory = cls(dsn=None, pool=pool, logger=logger, **kwargs)
        return memory

    def __init__(
        self,
        dsn: str | None,
        logger: Logger,
        pool: AsyncConnectionPool[Any] | None = None,
        composite_types: list[str] | None = None,
        pool_minimum: int | None = None,
        pool_maximum: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._composite_types = composite_types
        if pool is None:
            assert dsn is not None
            assert pool_minimum is not None
            assert pool_maximum is not None
            assert pool_minimum > 0
            assert pool_maximum > pool_minimum

        if pool:
            super().__init__(database=pool, logger=logger)
        else:
            assert dsn is not None
            assert pool_minimum is not None
            assert pool_maximum is not None
            if "//" not in dsn:
                dsn = f"postgresql://{dsn}"

            database = AsyncConnectionPool(
                min_size=pool_minimum,
                max_size=pool_maximum,
                conninfo=dsn,
                configure=self._configure_connection,
                open=False,
            )
            super().__init__(database=database, logger=logger)

    async def _configure_connection(
        self, connection: AsyncConnection[Any]
    ) -> None:
        connection.row_factory = cast(Any, dict_row)
        await connection.set_autocommit(True)
        if self._composite_types:
            for composite_type_name in self._composite_types:
                composite_type = await TypeInfo.fetch(
                    connection, composite_type_name
                )
                if composite_type:
                    composite_type.register(connection)
        await self._ensure_vector_extension(connection)
        await register_vector_async(connection)

    async def _ensure_vector_extension(
        self, connection: AsyncConnection[Any]
    ) -> None:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM "pg_extension"
                    WHERE "extname" = 'vector'
                ) AS "has_vector_extension"
                """
            )
            result = await cursor.fetchone()
            row = dict(result) if result is not None else {}
            if not row.get("has_vector_extension"):
                raise PgsqlVectorExtensionError(
                    "PostgreSQL `vector` extension is not enabled. "
                    "Install pgvector on the PostgreSQL server and run "
                    "`CREATE EXTENSION vector;` in the target database "
                    "before using permanent memory."
                )
            try:
                await cursor.execute("SELECT '[0]'::vector")
                await cursor.fetchone()
            except UndefinedFile as exc:
                raise PgsqlVectorExtensionError(
                    "PostgreSQL `vector` extension is registered in the "
                    "database, but the server cannot load the pgvector "
                    "library. Install pgvector on the PostgreSQL server "
                    "and recreate the extension before using permanent "
                    "memory."
                ) from exc

    @staticmethod
    @overload
    def _to_engine_messages(
        messages: list[PermanentMessage],
        *,
        limit: int | None,
        reverse: bool = False,
        scored: Literal[False] = False,
    ) -> list[EngineMessage]: ...

    @staticmethod
    @overload
    def _to_engine_messages(
        messages: list[PermanentMessageScored],
        *,
        limit: int | None,
        reverse: bool = False,
        scored: Literal[True],
    ) -> list[EngineMessageScored]: ...

    @staticmethod
    def _to_engine_messages(
        messages: list[PermanentMessage] | list[PermanentMessageScored],
        *,
        limit: int | None,
        reverse: bool = False,
        scored: bool = False,
    ) -> list[EngineMessage] | list[EngineMessageScored]:
        if scored:
            scored_messages = cast(list[PermanentMessageScored], messages)
            engine_messages_scored: list[EngineMessageScored] = [
                EngineMessageScored(
                    agent_id=m.agent_id,
                    model_id=m.model_id,
                    message=Message(role=m.author, content=m.data),
                    score=m.score,
                )
                for m in scored_messages
            ]
            if reverse:
                engine_messages_scored.reverse()
            if limit and len(engine_messages_scored) > limit:
                engine_messages_scored = engine_messages_scored[:limit]
            return engine_messages_scored

        engine_messages = [
            EngineMessage(
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
