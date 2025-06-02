from ....entities import EngineMessage, EngineMessageScored
from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    PermanentMessage,
    PermanentMessageMemory,
    PermanentMessagePartition,
    PermanentMessageScored,
    Session,
    VectorFunction,
)
from ....memory.permanent.pgsql import PgsqlMemory
from datetime import datetime, timezone
from pgvector.psycopg import Vector
from typing import Optional
from uuid import UUID, uuid4


class PgsqlMessageMemory(PgsqlMemory[PermanentMessage], PermanentMessageMemory):
    @classmethod
    async def create_instance(
        cls,
        dsn: str,
        *args,
        pool_minimum: int = 1,
        pool_maximum: int = 10,
        pool_open: bool = True,
        **kwargs,
    ):
        memory = cls(
            dsn=dsn,
            composite_types=["message_author_type"],
            pool_minimum=pool_minimum,
            pool_maximum=pool_maximum,
            **kwargs,
        )
        if pool_open:
            await memory.open()
        return memory

    async def create_session(
        self, *args, agent_id: UUID, participant_id: UUID
    ) -> UUID:
        now_utc = datetime.now(timezone.utc)
        session = Session(
            id=uuid4(),
            agent_id=agent_id,
            participant_id=participant_id,
            messages=0,
            created_at=now_utc,
        )
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO "sessions"(
                        "id",
                        "agent_id",
                        "participant_id",
                        "messages",
                        "created_at"
                    ) VALUES (
                        %s, %s, %s, %s, %s
                    )
                """,
                    (
                        str(session.id),
                        str(session.agent_id),
                        str(session.participant_id),
                        session.messages,
                        session.created_at,
                    ),
                )
                await cursor.close()
        return session.id

    async def continue_session_and_get_id(
        self,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        session_id: UUID,
    ) -> UUID:
        session_id = await self._fetch_field(
            "id",
            """
            SELECT "sessions"."id"
            FROM "sessions"
            WHERE "agent_id" = %s
            AND "participant_id" = %s
            AND "id" = %s
            LIMIT 1
        """,
            (str(agent_id), str(participant_id), str(session_id)),
        )
        assert session_id
        return session_id if isinstance(session_id, UUID) else UUID(session_id)

    async def append_with_partitions(
        self,
        engine_message: EngineMessage,
        *args,
        partitions: list[TextPartition],
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
                    await cursor.execute(
                        """
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
                    """,
                        (
                            str(message.id),
                            str(message.agent_id),
                            str(message.model_id),
                            str(message.session_id)
                            if message.session_id
                            else None,
                            str(message.author),
                            message.data,
                            message.partitions,
                            message.created_at,
                        ),
                    )

                    if message.session_id:
                        await cursor.execute(
                            """
                            UPDATE "sessions"
                            SET "messages" = "messages" + 1
                            WHERE "id" = %s
                        """,
                            (str(message.session_id),),
                        )

                    await cursor.executemany(
                        """
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
                    """,
                        [
                            (
                                str(mp.agent_id),
                                str(mp.session_id) if mp.session_id else None,
                                str(mp.message_id),
                                mp.partition + 1,
                                mp.data,
                                Vector(mp.embedding),
                                mp.created_at,
                            )
                            for mp in message_partitions
                        ],
                    )

                    await cursor.close()

    async def get_recent_messages(
        self,
        session_id: UUID,
        participant_id: UUID,
        *args,
        limit: Optional[int] = None,
    ) -> list[EngineMessage]:
        messages = await self._fetch_all(
            PermanentMessage,
            """
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
        """,
            (str(session_id), str(participant_id), limit),
        )
        engine_messages = self._to_engine_messages(
            messages, limit=limit, reverse=True
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
        limit: Optional[int] = None,
    ) -> list[EngineMessageScored]:
        assert agent_id and session_id and participant_id and search_partitions
        search_function = str(function)
        search_vector = Vector(search_partitions[0].embeddings)
        messages = await self._fetch_all(
            PermanentMessageScored,
            f"""
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
        """,
            (
                search_vector,
                str(session_id),
                str(participant_id),
                str(agent_id),
                limit,
            ),
        )
        engine_messages = self._to_engine_messages(
            messages,
            limit=limit,
            scored=True,
        )
        return engine_messages
