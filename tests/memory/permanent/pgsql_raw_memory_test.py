import numpy as np
from datetime import datetime, timezone
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, ANY, patch
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection, AsyncCursor
from uuid import uuid4, UUID

from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent import Memory, MemoryType, VectorFunction
from avalan.memory.permanent.pgsql.raw import PgsqlRawMemory
from avalan.model.entities import MessageRole
from pgvector.psycopg import Vector


class PgsqlRawMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance(self):
        with patch.object(PgsqlRawMemory, "open", AsyncMock()) as open_patch:
            memory = await PgsqlRawMemory.create_instance(
                dsn="dsn", pool_minimum=1, pool_maximum=2
            )
            self.assertIsInstance(memory, PgsqlRawMemory)
            open_patch.assert_awaited_once()

        with patch.object(PgsqlRawMemory, "open", AsyncMock()) as open_patch:
            memory = await PgsqlRawMemory.create_instance(
                dsn="dsn", pool_open=False
            )
            self.assertIsInstance(memory, PgsqlRawMemory)
            open_patch.assert_not_awaited()

    async def test_append_with_partitions(self):
        pool_mock, connection_mock, cursor_mock, txn_mock = self.mock_insert()
        memory_store = await PgsqlRawMemory.create_instance_from_pool(
            pool=pool_mock
        )

        base_memory = Memory(
            id=uuid4(),
            model_id="model",
            type=MemoryType.RAW,
            participant_id=uuid4(),
            namespace="ns",
            identifier="id",
            data="data",
            partitions=0,
            symbols={},
            created_at=datetime.now(timezone.utc),
        )
        partitions = [
            TextPartition(data="a", embeddings=np.array([0.1]), total_tokens=1),
            TextPartition(data="b", embeddings=np.array([0.2]), total_tokens=1),
        ]

        mem_id = UUID("11111111-1111-1111-1111-111111111111")
        with patch(
            "avalan.memory.permanent.pgsql.raw.uuid4", return_value=mem_id
        ):
            await memory_store.append_with_partitions(
                base_memory, partitions=partitions
            )

        connection_mock.transaction.assert_called_once()
        cursor_mock.execute.assert_awaited_once_with(
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
                str(mem_id),
                base_memory.model_id,
                str(base_memory.participant_id),
                str(base_memory.type),
                base_memory.namespace,
                base_memory.identifier,
                base_memory.data,
                len(partitions),
                base_memory.symbols,
                ANY,
            ),
        )
        cursor_mock.executemany.assert_awaited_once()
        self.assertTrue(
            cursor_mock.executemany.call_args[0][0]
            .strip()
            .startswith("INSERT INTO")
        )
        cursor_mock.close.assert_awaited_once()

    async def test_search_messages(self):
        fixtures = [
            (
                uuid4(),
                uuid4(),
                uuid4(),
                2,
                "model",
                VectorFunction.L2_DISTANCE,
                [
                    (MessageRole.USER, "A", 1.0),
                    (MessageRole.ASSISTANT, "B", 1.1),
                ],
            ),
        ]
        for (
            agent_id,
            participant_id,
            session_id,
            limit,
            model_id,
            function,
            messages,
        ) in fixtures:
            with self.subTest():
                pool_mock, connection_mock, cursor_mock = (
                    PgsqlRawMemoryTestCase.mock_query(
                        sorted(
                            [
                                {
                                    "id": uuid4(),
                                    "agent_id": agent_id,
                                    "model_id": model_id,
                                    "session_id": session_id,
                                    "author": str(m[0]),
                                    "data": m[1],
                                    "partitions": 1,
                                    "created_at": datetime.now(timezone.utc),
                                    "score": m[2],
                                }
                                for m in messages
                            ],
                            key=lambda r: r["score"],
                        ),
                        fetch_all=True,
                    )
                )
                memory = await PgsqlRawMemory.create_instance_from_pool(
                    pool=pool_mock
                )
                search_partitions = [
                    TextPartition(
                        data="", total_tokens=1, embeddings=np.random.rand(3)
                    )
                ]
                result = await memory.search_messages(
                    search_partitions=search_partitions,
                    agent_id=agent_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    function=function,
                    limit=limit,
                )
                PgsqlRawMemoryTestCase.assert_query(
                    self,
                    connection_mock,
                    cursor_mock,
                    f"""
                    SELECT
                        \"messages\".\"id\",
                        \"messages\".\"agent_id\",
                        \"messages\".\"model_id\",
                        \"messages\".\"session_id\",
                        \"messages\".\"author\",
                        \"messages\".\"data\",
                        \"messages\".\"partitions\",
                        \"messages\".\"created_at\",
                        {str(function)}(
                            \"message_partitions\".\"embedding\",
                            %s
                        ) AS \"score\"
                    FROM \"sessions\"
                    INNER JOIN \"message_partitions\" ON (
                        \"sessions\".\"id\" = \"message_partitions\".\"session_id\"
                    )
                    INNER JOIN \"messages\" ON (
                        \"message_partitions\".\"message_id\" = \"messages\".\"id\"
                    )
                    WHERE \"sessions\".\"id\" = %s
                    AND \"sessions\".\"participant_id\" = %s
                    AND \"sessions\".\"agent_id\" = %s
                    AND \"messages\".\"is_deleted\" = FALSE
                    ORDER BY \"score\" ASC
                    LIMIT %s
                    """,
                    (
                        Vector(search_partitions[0].embeddings),
                        str(session_id),
                        str(participant_id),
                        str(agent_id),
                        limit,
                    ),
                    fetch_all=True,
                )
                self.assertEqual(len(result), len(messages))
                for i, msg in enumerate(sorted(messages, key=lambda m: m[2])):
                    self.assertEqual(result[i].message.content, msg[1])

    @staticmethod
    def mock_insert():
        cursor_mock = AsyncMock(spec=AsyncCursor)
        cursor_mock.__aenter__.return_value = cursor_mock
        transaction_mock = AsyncMock()
        transaction_mock.__aenter__.return_value = transaction_mock
        connection_mock = AsyncMock(spec=AsyncConnection)
        connection_mock.cursor.return_value = cursor_mock
        connection_mock.transaction = MagicMock(return_value=transaction_mock)
        connection_mock.__aenter__.return_value = connection_mock
        pool_mock = MagicMock(spec=AsyncConnectionPool)
        pool_mock.connection.return_value = connection_mock
        pool_mock.__aenter__.return_value = pool_mock
        return pool_mock, connection_mock, cursor_mock, transaction_mock

    @staticmethod
    def mock_query(record_set, fetch_all=False):
        cursor_mock = AsyncMock(spec=AsyncCursor)
        cursor_mock.__aenter__.return_value = cursor_mock
        if fetch_all:
            cursor_mock.fetchall.return_value = record_set
        else:
            cursor_mock.fetchone.return_value = record_set
        connection_mock = AsyncMock(spec=AsyncConnection)
        connection_mock.cursor.return_value = cursor_mock
        connection_mock.__aenter__.return_value = connection_mock
        pool_mock = MagicMock(spec=AsyncConnectionPool)
        pool_mock.connection.return_value = connection_mock
        pool_mock.__aenter__.return_value = pool_mock
        return pool_mock, connection_mock, cursor_mock

    def assert_query(
        self,
        connection_mock,
        cursor_mock,
        expected_query,
        expected_parameters=None,
        fetch_all=False,
    ):
        def normalize_whitespace(text: str) -> str:
            import re

            return re.sub(r"\s+", " ", text.strip())

        connection_mock.cursor.assert_called_once()
        self.assertIsNotNone(cursor_mock.execute.call_args)
        call_args = cursor_mock.execute.call_args[0]
        self.assertEqual(
            len(call_args), 2 if expected_parameters is not None else 1
        )
        actual_query = call_args[0]
        self.assertEqual(
            normalize_whitespace(actual_query),
            normalize_whitespace(expected_query),
        )
        if expected_parameters is not None:
            actual_parameters = call_args[1]
            self.assertEqual(len(actual_parameters), len(expected_parameters))
            self.assertEqual(actual_parameters, expected_parameters)
            cursor_mock.execute.assert_awaited_once_with(
                actual_query, actual_parameters
            )
        else:
            cursor_mock.execute.assert_awaited_once_with(actual_query)
        if fetch_all:
            cursor_mock.fetchall.assert_awaited_once()
        else:
            cursor_mock.fetchone.assert_awaited_once()
