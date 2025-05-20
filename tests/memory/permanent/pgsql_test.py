from avalan.model.entities import EngineMessage, MessageRole
from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent import VectorFunction
from avalan.memory.permanent.pgsql import PgsqlMessageMemory
from datetime import datetime, timezone
from numpy.random import rand
from pgvector.psycopg import Vector
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection, AsyncCursor
from re import sub
from typing import Tuple, Union
from unittest import main, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from uuid import uuid4, UUID

class PgsqlMessageMemoryTestCase(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_sessions = [
            (
                UUID("8f38ce12-910a-44ca-94ba-db53b6cb4e68"),
                UUID("b072d42f-141a-42ce-a772-95f91eecd154"),
                UUID("c54957a5-d9cf-4589-bd6e-34eabdf61733")
            ),
            ( uuid4(), uuid4(), uuid4() )
        ]
        cls.fixture_recent_messages = [
            ( uuid4(), uuid4(), uuid4(), None, "microsoft/Phi-4-mini-instruct",
                []
            ),
            ( uuid4(), uuid4(), uuid4(), 15, "microsoft/Phi-4-mini-instruct",
                []
            ),
            ( uuid4(), uuid4(), uuid4(), None, "microsoft/Phi-4-mini-instruct",
            [
                (MessageRole.USER, "Who are you?"),
                (MessageRole.ASSISTANT, """
                    I'm Leo Messi, the footballer who has graced the pitch with
                    his extraordinary skills, much like a maestro conducting a
                    symphony of football. Just as a maestro brings out the best
                    in their orchestra, I strive to bring out the best in my
                    team and the beautiful game.
                """)
            ] ),
            ( uuid4(), uuid4(), uuid4(), 3, "microsoft/Phi-4-mini-instruct",
            [
                (MessageRole.USER, "Who are you?"),
                (MessageRole.ASSISTANT, """
                    I'm Leo Messi, the footballer who has graced the pitch with
                    his extraordinary skills, much like a maestro conducting a
                    symphony of football. Just as a maestro brings out the best
                    in their orchestra, I strive to bring out the best in my
                    team and the beautiful game.
                """),
                (MessageRole.USER, "Hi Leo, I'm Dibu Martinez."),
                (MessageRole.ASSISTANT, """
                    Hello Dibu, it's a pleasure to meet you. Just as a good
                    football match brings people together, it's great to have
                    the opportunity to connect with you. How can I assist you
                    today?
                 """),
            ] ),
        ]

        cls.fixture_search_messages = [
            (
                uuid4(), uuid4(), uuid4(), 3, "microsoft/Phi-4-mini-instruct",
                VectorFunction.L2_DISTANCE,
            [
                (MessageRole.USER, """
                    Who are you?
                """, 1.2285790843978555),
                (MessageRole.ASSISTANT, """
                    I'm Leo Messi, the footballer who has graced the pitch with
                    his extraordinary skills, much like a maestro conducting a
                    symphony of football. Just as a maestro brings out the best
                    in their orchestra, I strive to bring out the best in my
                    team and the beautiful game.
                """, 1.274401715098231),
                (MessageRole.USER, """
                    Hi Leo, I'm Dibu Martinez.
                """, 1.0821355776292412),
                (MessageRole.ASSISTANT, """
                    Hello Dibu, it's a pleasure to meet you. Just as a good
                    football match brings people together, it's great to have
                    the opportunity to connect with you. How can I assist you
                    today?
                 """, 1.258004878715431),
            ] ),

        ]

    async def test_continue_session(self):
        for fixture in self.fixture_sessions:
            agent_id, participant_id, session_id = fixture

            self.assertIsInstance(agent_id, UUID)
            self.assertIsInstance(participant_id, UUID)
            self.assertIsInstance(session_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query({
                    "id": session_id,
                })

                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock
                )

                result = await memory.continue_session_and_get_id(
                    agent_id=agent_id,
                    participant_id=participant_id,
                    session_id=session_id
                )

                self.assert_query(connection_mock, cursor_mock,  """
                    SELECT "sessions"."id"
                    FROM "sessions"
                    WHERE "agent_id" = %s
                    AND "participant_id" = %s
                    AND "id" = %s
                    LIMIT 1
                """, (
                    str(agent_id),
                    str(participant_id),
                    str(session_id),
                ),)

                self.assertEqual(str(result), str(session_id))

    async def test_get_recent_messages(self):
        for fixture in self.fixture_recent_messages:
            agent_id, participant_id, session_id, limit, model_id, messages = \
               fixture

            self.assertIsInstance(participant_id, UUID)
            self.assertIsInstance(session_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query(
                    # descending order
                    reversed([{
                        "id": uuid4(),
                        "agent_id": agent_id,
                        "model_id": model_id,
                        "session_id": session_id,
                        "author": str(m[0]),
                        "data": m[1],
                        "partitions": 1,
                        "created_at": datetime.now(timezone.utc)
                    } for m in messages]),
                    fetch_all=True
                )

                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock
                )

                result = await memory.get_recent_messages(
                    session_id=session_id,
                    participant_id=participant_id,
                    limit=limit
                )

                self.assert_query(connection_mock, cursor_mock,  """
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
                    INNER JOIN "messages" ON "sessions"."id" =
                               "messages"."session_id"
                    WHERE "sessions"."id" = %s
                    AND "sessions"."participant_id" = %s
                    AND "messages"."is_deleted" = FALSE
                    ORDER BY "messages"."created_at" DESC
                    LIMIT %s
                """, (
                    str(session_id),
                    str(participant_id),
                    limit
                ), fetch_all=True)

                expected_count = limit if limit and len(messages) > limit \
                                 else len(messages)
                self.assertEqual(len(result), expected_count)

                for i, message in enumerate(messages):
                    role, data = message
                    result_item = result[i]
                    self.assertIsInstance(result_item, EngineMessage)
                    self.assertEqual(result_item.agent_id, agent_id)
                    self.assertEqual(result_item.model_id, model_id)
                    self.assertEqual(result_item.message.role, role)
                    self.assertEqual(result_item.message.content, data)
                    if i == expected_count - 1:
                        break

    async def test_search_messages(self):
        for fixture in self.fixture_search_messages:
            (
                agent_id, participant_id, session_id, limit, model_id,
                function, messages
            ) = fixture

            self.assertIsInstance(participant_id, UUID)
            self.assertIsInstance(session_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query(
                    # descending order
                    sorted([{
                        "id": uuid4(),
                        "agent_id": agent_id,
                        "model_id": model_id,
                        "session_id": session_id,
                        "author": str(m[0]),
                        "data": m[1],
                        "partitions": 1,
                        "created_at": datetime.now(timezone.utc),
                        "score": m[2]
                    } for m in messages], key=lambda r: r["score"]),
                    fetch_all=True
                )

                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock
                )

                search_function = str(function)
                search_partitions = [
                    TextPartition(
                        data="",
                        total_tokens=1,
                        embeddings=rand(384)
                    )
                ]

                result = await memory.search_messages(
                    search_partitions=search_partitions,
                    agent_id=agent_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    function=function,
                    limit=limit
                )

                self.assert_query(connection_mock, cursor_mock, f"""
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
                    Vector(search_partitions[0].embeddings),
                    str(session_id),
                    str(participant_id),
                    str(agent_id),
                    limit
                ), fetch_all=True)

                expected_count = limit if limit and len(messages) > limit \
                                 else len(messages)
                self.assertEqual(len(result), expected_count)

                for i, message in enumerate(sorted(
                    messages,
                    key=lambda m: m[2]
                )):
                    role, data, score = message
                    result_item = result[i]
                    self.assertIsInstance(result_item, EngineMessage)
                    self.assertEqual(result_item.agent_id, agent_id)
                    self.assertEqual(result_item.model_id, model_id)
                    self.assertEqual(result_item.message.role, role)
                    self.assertEqual(result_item.message.content, data)
                    self.assertEqual(result_item.score, score)
                    if i == expected_count - 1:
                        break

    @staticmethod
    def mock_query(
        record_set: Union[dict, list[dict]],
        fetch_all: bool=False
    ) -> Tuple[AsyncConnectionPool, AsyncConnection, AsyncCursor]:
        cursor_mock = AsyncMock(spec=AsyncCursor)
        cursor_mock.__aenter__.return_value = cursor_mock
        if fetch_all:
            cursor_mock.fetchall.return_value = record_set
        else:
            cursor_mock.fetchone.return_value = record_set

        connection_mock = AsyncMock(spec=AsyncConnection)
        connection_mock.cursor.return_value = cursor_mock
        connection_mock.__aenter__.return_value = connection_mock

        pool_mock = AsyncMock(spec=AsyncConnectionPool)
        pool_mock.connection.return_value = connection_mock
        pool_mock.__aenter__.return_value = pool_mock

        return pool_mock, connection_mock, cursor_mock

    def assert_query(
        self,
        connection_mock: AsyncConnection,
        cursor_mock: AsyncCursor,
        expected_query: str,
        expected_parameters=None,
        fetch_all: bool=False
    ) -> None:
        def normalize_whitespace(text: str) -> str:
            return sub(r"\s+", " ", text.strip())

        connection_mock.cursor.assert_called_once()

        self.assertIsNotNone(cursor_mock.execute.call_args)
        call_args = cursor_mock.execute.call_args[0]
        self.assertEqual(
            len(call_args),
            2 if expected_parameters is not None else 1
        )

        actual_query = call_args[0]
        self.assertEqual(
            normalize_whitespace(actual_query),
            normalize_whitespace(expected_query)
        )

        if expected_parameters is not None:
            actual_parameters = call_args[1]
            self.assertEqual(
                len(actual_parameters),
                len(expected_parameters)
            )
            self.assertEqual(
                actual_parameters,
                expected_parameters
            )

            cursor_mock.execute.assert_awaited_once_with(
                actual_query,
                actual_parameters
            )
        else:
            cursor_mock.execute.assert_awaited_once_with(actual_query)

        if fetch_all:
            cursor_mock.fetchall.assert_awaited_once()
        else:
            cursor_mock.fetchone.assert_awaited_once()

if __name__ == '__main__':
    main()
