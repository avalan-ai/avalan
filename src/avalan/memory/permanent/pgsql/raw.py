from ....entities import (
    EngineMessageScored,
)
from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    Memory,
    PermanentMemory,
    PermanentMemoryPartition,
    PermanentMessageScored,
    VectorFunction,
)
from ....memory.permanent.pgsql import PgsqlMemory
from datetime import datetime, timezone
from pgvector.psycopg import Vector
from typing import Optional
from uuid import UUID, uuid4


class PgsqlRawMemory(PgsqlMemory[Memory], PermanentMemory):
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
            pool_minimum=pool_minimum,
            pool_maximum=pool_maximum,
            **kwargs,
        )
        if pool_open:
            await memory.open()
        return memory

    async def append_with_partitions(
        self, memory: Memory, *args, partitions: list[TextPartition]
    ) -> None:
        assert memory and partitions
        now_utc = datetime.now(timezone.utc)
        entry = Memory(
            id=uuid4(),
            model_id=memory.model_id,
            type=memory.type,
            participant_id=memory.participant_id,
            namespace=memory.namespace,
            identifier=memory.identifier,
            data=memory.data,
            partitions=len(partitions),
            symbols=memory.symbols,
            created_at=now_utc,
        )
        partition_rows = [
            PermanentMemoryPartition(
                participant_id=entry.participant_id,
                memory_id=entry.id,
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
                        INSERT INTO "memories"(
                            "id",
                            "model_id",
                            "participant_id",
                            "memory_type",
                            "namespace",
                            "identifier",
                            "data",
                            "partitions",
                            "symbols",
                            "created_at"
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            str(entry.id),
                            str(entry.model_id),
                            str(entry.participant_id),
                            str(entry.type),
                            entry.namespace,
                            entry.identifier,
                            entry.data,
                            entry.partitions,
                            entry.symbols,
                            entry.created_at,
                        ),
                    )

                    await cursor.executemany(
                        """
                        INSERT INTO "memory_partitions"(
                            "participant_id",
                            "memory_id",
                            "partition",
                            "data",
                            "embedding",
                            "created_at"
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s
                        )
                        """,
                        [
                            (
                                str(mp.participant_id),
                                str(mp.memory_id),
                                mp.partition + 1,
                                mp.data,
                                Vector(mp.embedding),
                                mp.created_at,
                            )
                            for mp in partition_rows
                        ],
                    )
                    await cursor.close()

    async def search_memories(
        self,
        *args,
        search_partitions: list[TextPartition],
        participant_id: UUID,
        namespace: str,
        function: VectorFunction,
        limit: Optional[int] = None,
    ) -> list[Memory]:
        assert participant_id and namespace and search_partitions
        search_function = str(function)
        search_vector = Vector(search_partitions[0].embeddings)
        memories = await self._fetch_all(
            Memory,
            f"""
            SELECT
                "memories"."id",
                "memories"."model_id",
                "memories"."memory_type" AS "type",
                "memories"."participant_id",
                "memories"."namespace",
                "memories"."identifier",
                "memories"."data",
                "memories"."partitions",
                "memories"."symbols",
                "memories"."created_at"
            FROM "memories"
            INNER JOIN "memory_partitions" ON (
                "memory_partitions"."memory_id" = "memories"."id"
            )
            WHERE "memories"."participant_id" = %s
            AND "memories"."namespace" = %s
            AND "memories"."is_deleted" = FALSE
            ORDER BY {search_function}(
                "memory_partitions"."embedding",
                %s
            ) ASC
            LIMIT %s
            """,
            (
                str(participant_id),
                namespace,
                search_vector,
                limit,
            ),
        )
        return memories

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
