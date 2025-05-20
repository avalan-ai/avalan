from avalan.memory import Memory, MemoryChunk
from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent import (
    PermanentMessage,
    PermanentMessageMemory,
    PermanentMessagePartition,
    PermanentMessageScored,
    RecordNotFoundException,
    RecordNotSavedException,
    Session,
    VectorFunction
)
from avalan.model.entities import (
    EngineMessage,
    EngineMessageScored,
    Message
)
from datetime import datetime, timezone
from pgvector.psycopg import register_vector_async, Vector
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types import TypeInfo
from typing import Optional, TypeVar, Type, Union
from uuid import UUID, uuid4

T = TypeVar("T")

class BasePgsqlMemory(Memory[T]):
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
        self,
        entity: Type[T],
        query: str,
        parameters: tuple
    ) -> T:
        result = await self._try_fetch_one(entity, query, parameters)
        if result is None:
            raise RecordNotFoundException()
        return result

    async def _fetch_field(
        self,
        field: str,
        query: str,
        parameters: Optional[tuple] = None
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
        self,
        query: str,
        parameters: tuple
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
        pool: Optional[AsyncConnectionPool]=None,
        composite_types: Optional[list[str]]=None,
        pool_minimum: Optional[int]=None,
        pool_maximum: Optional[int]=None,
        **kwargs
    ):
        assert (pool or (
            dsn and pool_minimum and pool_minimum and pool_minimum > 0
            and pool_maximum > pool_minimum
        ))

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
                open=False
            )
            super().__init__(database=database, **kwargs)

    async def _configure_connection(self, connection: AsyncConnection):
        connection.row_factory = dict_row
        await connection.set_autocommit(True)
        if self._composite_types:
            for composite_type_name in self._composite_types:
                composite_type = await TypeInfo.fetch(
                    connection,
                    composite_type_name
                )
                if composite_type:
                    composite_type.register(connection)
        await register_vector_async(connection)

class PgsqlMessageMemory(
    PgsqlMemory[PermanentMessage],
    PermanentMessageMemory
):
    @classmethod
    async def create_instance(
        cls,
        dsn: str,
        *args,
        pool_minimum: int=1,
        pool_maximum: int=10,
        pool_open: bool=True,
        **kwargs,
    ):
        memory = cls(
            dsn=dsn,
            composite_types=["message_author_type"],
            pool_minimum=pool_minimum,
            pool_maximum=pool_maximum,
            **kwargs
        )
        if pool_open:
            await memory.open()
        return memory

    async def create_session(
        self,
        *args,
        agent_id: UUID,
        participant_id: UUID
    ) -> UUID:
        now_utc = datetime.now(timezone.utc)
        session = Session(
            id=uuid4(),
            agent_id=agent_id,
            participant_id=participant_id,
            messages=0,
            created_at=now_utc
        )
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO "sessions"(
                        "id",
                        "agent_id",
                        "participant_id",
                        "messages",
                        "created_at"
                    ) VALUES (
                        %s, %s, %s, %s, %s
                    )
                """, (
                    str(session.id),
                    str(session.agent_id),
                    str(session.participant_id),
                    session.messages,
                    session.created_at
                ))
                await cursor.close()
        return session.id

    async def continue_session_and_get_id(
        self,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        session_id: UUID,
    ) -> UUID:
        session_id = await self._fetch_field("id", """
            SELECT "sessions"."id"
            FROM "sessions"
            WHERE "agent_id" = %s
            AND "participant_id" = %s
            AND "id" = %s
            LIMIT 1
        """, (
            str(agent_id),
            str(participant_id),
            str(session_id)
        ))
        assert session_id
        return session_id if isinstance(session_id, UUID) else UUID(session_id)

    async def append_with_partitions(
        self,
        engine_message: EngineMessage,
        *args,
        partitions: list[TextPartition]
    ) -> None:
        assert engine_message and partitions
        now_utc = datetime.now(timezone.utc)
        message = PermanentMessage(
            id=uuid4(),
            agent_id=engine_message.agent_id,
            model_id=engine_message.model_id,
            session_id=self._session_id,
            author=engine_message.message.role,
            data=engine_message.message.content,
            partitions=len(partitions),
            created_at=now_utc,
        )
        message_partitions = [
            PermanentMessagePartition(
                agent_id=message.agent_id,
                session_id=message.session_id,
                message_id=message.id,
                partition=i,
                data=p.data,
                embedding=p.embeddings,
                created_at=now_utc,
            )
            for i, p in enumerate(partitions)
        ]

        async with self._database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute("""
                        INSERT INTO "messages"(
                            "id",
                            "agent_id",
                            "model_id",
                            "session_id",
                            "author",
                            "data",
                            "partitions",
                            "created_at"
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        str(message.id),
                        str(message.agent_id),
                        str(message.model_id),
                        str(message.session_id)
                            if message.session_id
                            else None,
                        str(message.author),
                        message.data,
                        message.partitions,
                        message.created_at
                    ))

                    if message.session_id:
                        await cursor.execute("""
                            UPDATE "sessions"
                            SET "messages" = "messages" + 1
                            WHERE "id" = %s
                        """, (
                            str(message.session_id),
                        ))

                    await cursor.executemany("""
                        INSERT INTO "message_partitions"(
                            "agent_id",
                            "session_id",
                            "message_id",
                            "partition",
                            "data",
                            "embedding",
                            "created_at"
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s
                        )
                    """, [
                        (
                            str(mp.agent_id),
                            str(mp.session_id) if mp.session_id else None,
                            str(mp.message_id),
                            mp.partition + 1,
                            mp.data,
                            Vector(mp.embedding),
                            mp.created_at
                        )
                        for mp in message_partitions
                    ])

                    await cursor.close()

    async def get_recent_messages(
        self,
        session_id: UUID,
        participant_id: UUID,
        *args,
        limit: Optional[int]=None
    ) -> list[EngineMessage]:
        messages = await self._fetch_all(PermanentMessage, """
            SELECT
                "messages"."id",
                "messages"."agent_id",
                "messages"."model_id",
                "messages"."session_id",
                "messages"."author",
                "messages"."data",
                "messages"."partitions",
                "messages"."created_at"
            FROM "sessions"
            INNER JOIN "messages" ON "sessions"."id" = "messages"."session_id"
            WHERE "sessions"."id" = %s
            AND "sessions"."participant_id" = %s
            AND "messages"."is_deleted" = FALSE
            ORDER BY "messages"."created_at" DESC
            LIMIT %s
        """, (
            str(session_id),
            str(participant_id),
            limit
        ))
        engine_messages = PgsqlMessageMemory._to_engine_messages(
            messages,
            limit=limit,
            reverse=True
        )
        return engine_messages

    async def search_messages(
        self,
        *args,
        search_partitions: list[TextPartition],
        agent_id: UUID,
        session_id: UUID,
        participant_id: UUID,
        function: VectorFunction,
        limit: Optional[int]=None
    ) -> list[EngineMessageScored]:
        assert agent_id and session_id and participant_id and search_partitions
        search_function = str(function)
        search_vector = Vector(search_partitions[0].embeddings)
        messages = await self._fetch_all(PermanentMessageScored, f"""
            SELECT
                "messages"."id",
                "messages"."agent_id",
                "messages"."model_id",
                "messages"."session_id",
                "messages"."author",
                "messages"."data",
                "messages"."partitions",
                "messages"."created_at",
                {search_function}(
                    "message_partitions"."embedding",
                    %s
                ) AS "score"
            FROM "sessions"
            INNER JOIN "message_partitions" ON (
                "sessions"."id" = "message_partitions"."session_id"
            )
            INNER JOIN "messages" ON (
                "message_partitions"."message_id" = "messages"."id"
            )
            WHERE "sessions"."id" = %s
            AND "sessions"."participant_id" = %s
            AND "sessions"."agent_id" = %s
            AND "messages"."is_deleted" = FALSE
            ORDER BY "score" ASC
            LIMIT %s
        """, (
            search_vector,
            str(session_id),
            str(participant_id),
            str(agent_id),
            limit
        ))
        engine_messages = PgsqlMessageMemory._to_engine_messages(
            messages,
            limit=limit,
            scored=True,
        )
        return engine_messages

    @staticmethod
    def _to_engine_messages(
        messages: Union[list[PermanentMessage], list[PermanentMessageScored]],
        *args,
        limit: Optional[int],
        reverse: bool=False,
        scored: bool=False
    ) -> Union[list[EngineMessage], list[EngineMessageScored]]:
        engine_messages = [
            EngineMessageScored(
                agent_id=m.agent_id,
                model_id=m.model_id,
                message=Message(
                    role=m.author,
                    content=m.data
                ),
                score=m.score
            ) if scored else
            EngineMessage(
                agent_id=m.agent_id,
                model_id=m.model_id,
                message=Message(
                    role=m.author,
                    content=m.data
                )
            )
            for m in messages
        ]
        if reverse:
            engine_messages.reverse()
        if limit and len(engine_messages) > limit:
            engine_messages = engine_messages[:limit]
        return engine_messages

