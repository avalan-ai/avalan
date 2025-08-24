from avalan.entities import EngineMessage, Message, MessageRole
from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent import VectorFunction, MemoryType
from avalan.memory.permanent.pgsql.message import PgsqlMessageMemory
from avalan.memory.permanent.pgsql.raw import PgsqlRawMemory
from datetime import datetime, timezone
from numpy.random import rand
from pgvector.psycopg import Vector
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection, AsyncCursor
from re import sub

from unittest import main, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, ANY, patch
from uuid import uuid4, UUID


class PgsqlMessageMemoryTestCase(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_sessions = [
            (
                UUID("8f38ce12-910a-44ca-94ba-db53b6cb4e68"),
                UUID("b072d42f-141a-42ce-a772-95f91eecd154"),
                UUID("c54957a5-d9cf-4589-bd6e-34eabdf61733"),
            ),
            (uuid4(), uuid4(), uuid4()),
        ]
        cls.fixture_recent_messages = [
            (
                uuid4(),
                uuid4(),
                uuid4(),
                None,
                "microsoft/Phi-4-mini-instruct",
                [],
            ),
            (
                uuid4(),
                uuid4(),
                uuid4(),
                15,
                "microsoft/Phi-4-mini-instruct",
                [],
            ),
            (
                uuid4(),
                uuid4(),
                uuid4(),
                None,
                "microsoft/Phi-4-mini-instruct",
                [
                    (MessageRole.USER, "Who are you?"),
                    (
                        MessageRole.ASSISTANT,
                        """
                    I'm Leo Messi, the footballer who has graced the pitch with
                    his extraordinary skills, much like a maestro conducting a
                    symphony of football. Just as a maestro brings out the best
                    in their orchestra, I strive to bring out the best in my
                    team and the beautiful game.
                """,
                    ),
                ],
            ),
            (
                uuid4(),
                uuid4(),
                uuid4(),
                3,
                "microsoft/Phi-4-mini-instruct",
                [
                    (MessageRole.USER, "Who are you?"),
                    (
                        MessageRole.ASSISTANT,
                        """
                    I'm Leo Messi, the footballer who has graced the pitch with
                    his extraordinary skills, much like a maestro conducting a
                    symphony of football. Just as a maestro brings out the best
                    in their orchestra, I strive to bring out the best in my
                    team and the beautiful game.
                """,
                    ),
                    (MessageRole.USER, "Hi Leo, I'm Dibu Martinez."),
                    (
                        MessageRole.ASSISTANT,
                        """
                    Hello Dibu, it's a pleasure to meet you. Just as a good
                    football match brings people together, it's great to have
                    the opportunity to connect with you. How can I assist you
                    today?
                 """,
                    ),
                ],
            ),
        ]

        cls.fixture_search_messages = [
            (
                uuid4(),
                uuid4(),
                uuid4(),
                3,
                "microsoft/Phi-4-mini-instruct",
                VectorFunction.L2_DISTANCE,
                [
                    (
                        MessageRole.USER,
                        """
                    Who are you?
                """,
                        1.2285790843978555,
                    ),
                    (
                        MessageRole.ASSISTANT,
                        """
                    I'm Leo Messi, the footballer who has graced the pitch with
                    his extraordinary skills, much like a maestro conducting a
                    symphony of football. Just as a maestro brings out the best
                    in their orchestra, I strive to bring out the best in my
                    team and the beautiful game.
                """,
                        1.274401715098231,
                    ),
                    (
                        MessageRole.USER,
                        """
                    Hi Leo, I'm Dibu Martinez.
                """,
                        1.0821355776292412,
                    ),
                    (
                        MessageRole.ASSISTANT,
                        """
                    Hello Dibu, it's a pleasure to meet you. Just as a good
                    football match brings people together, it's great to have
                    the opportunity to connect with you. How can I assist you
                    today?
                 """,
                        1.258004878715431,
                    ),
                ],
            ),
        ]

        cls.fixture_search_memories = [
            (
                uuid4(),
                "people",
                2,
                "microsoft/Phi-4-mini-instruct",
                MemoryType.RAW,
                VectorFunction.L2_DISTANCE,
                [
                    ("leo", "about leo", {"role": "footballer"}),
                    ("dibu", "about dibu", {"role": "goalkeeper"}),
                ],
            ),
            (
                uuid4(),
                "pets",
                None,
                "openai/gpt",
                MemoryType.RAW,
                VectorFunction.COSINE_DISTANCE,
                [
                    ("fido", "about fido", {"role": "dog"}),
                    ("garfield", "about garfield", {"role": "cat"}),
                ],
            ),
        ]

    async def test_create_instance(self):
        with patch.object(
            PgsqlMessageMemory, "open", AsyncMock()
        ) as open_patch:
            logger = MagicMock()
            memory = await PgsqlMessageMemory.create_instance(
                dsn="dsn", pool_minimum=1, pool_maximum=2, logger=logger
            )
            self.assertIsInstance(memory, PgsqlMessageMemory)
            open_patch.assert_awaited_once()

        with patch.object(
            PgsqlMessageMemory, "open", AsyncMock()
        ) as open_patch:
            memory = await PgsqlMessageMemory.create_instance(
                dsn="dsn", pool_open=False, logger=logger
            )
            self.assertIsInstance(memory, PgsqlMessageMemory)
            open_patch.assert_not_awaited()

    async def test_create_session(self):
        pool_mock, connection_mock, cursor_mock = self.mock_query({})
        memory = await PgsqlMessageMemory.create_instance_from_pool(
            pool=pool_mock,
            logger=MagicMock(),
        )
        agent_id = uuid4()
        participant_id = uuid4()
        sess_id = UUID("11111111-1111-1111-1111-111111111111")

        with patch("avalan.memory.permanent.uuid4", return_value=sess_id):
            result = await memory.create_session(
                agent_id=agent_id, participant_id=participant_id
            )

        cursor_mock.execute.assert_awaited_once_with(
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
                str(sess_id),
                str(agent_id),
                str(participant_id),
                0,
                ANY,
            ),
        )
        cursor_mock.close.assert_awaited_once()
        self.assertEqual(result, sess_id)

    async def test_append_with_partitions(self):
        pool_mock, connection_mock, cursor_mock, txn_mock = self.mock_insert()
        memory = await PgsqlMessageMemory.create_instance_from_pool(
            pool=pool_mock,
            logger=MagicMock(),
        )
        session_id = uuid4()
        memory._session_id = session_id
        agent_id = uuid4()

        engine_message = EngineMessage(
            agent_id=agent_id,
            model_id="model",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        partitions = [
            TextPartition(data="a", embeddings=rand(1), total_tokens=1),
            TextPartition(data="b", embeddings=rand(1), total_tokens=1),
        ]

        msg_id = UUID("22222222-2222-2222-2222-222222222222")
        with patch(
            "avalan.memory.permanent.pgsql.message.uuid4", return_value=msg_id
        ):
            await memory.append_with_partitions(
                engine_message, partitions=partitions
            )

        connection_mock.transaction.assert_called_once()

        exec_calls = cursor_mock.execute.await_args_list
        self.assertEqual(len(exec_calls), 2)

        def norm(txt: str) -> str:
            return sub(r"\s+", " ", txt.strip())

        insert_query = """
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
                    """
        self.assertEqual(norm(exec_calls[0].args[0]), norm(insert_query))
        self.assertEqual(
            exec_calls[0].args[1],
            (
                str(msg_id),
                str(agent_id),
                "model",
                str(session_id),
                str(MessageRole.USER),
                "hi",
                len(partitions),
                ANY,
            ),
        )

        update_query = """
                            UPDATE "sessions"
                            SET "messages" = "messages" + 1
                            WHERE "id" = %s
                        """
        self.assertEqual(norm(exec_calls[1].args[0]), norm(update_query))
        self.assertEqual(exec_calls[1].args[1], (str(session_id),))

        cursor_mock.executemany.assert_awaited_once()
        self.assertTrue(
            cursor_mock.executemany.call_args[0][0]
            .strip()
            .startswith("INSERT INTO")
        )
        cursor_mock.close.assert_awaited_once()

    async def test_append_with_partitions_without_session(self):
        pool_mock, connection_mock, cursor_mock, _ = self.mock_insert()
        memory = await PgsqlMessageMemory.create_instance_from_pool(
            pool=pool_mock,
            logger=MagicMock(),
        )
        agent_id = uuid4()

        engine_message = EngineMessage(
            agent_id=agent_id,
            model_id="model",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        partitions = [
            TextPartition(data="a", embeddings=rand(1), total_tokens=1),
            TextPartition(data="b", embeddings=rand(1), total_tokens=1),
        ]

        msg_id = UUID("33333333-3333-3333-3333-333333333333")
        with patch(
            "avalan.memory.permanent.pgsql.message.uuid4", return_value=msg_id
        ):
            await memory.append_with_partitions(
                engine_message, partitions=partitions
            )

        connection_mock.transaction.assert_called_once()
        exec_calls = cursor_mock.execute.await_args_list
        self.assertEqual(len(exec_calls), 1)

        def norm(txt: str) -> str:
            return sub(r"\s+", " ", txt.strip())

        insert_query = """
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
                    """
        self.assertEqual(norm(exec_calls[0].args[0]), norm(insert_query))
        self.assertEqual(
            exec_calls[0].args[1],
            (
                str(msg_id),
                str(agent_id),
                "model",
                None,
                str(MessageRole.USER),
                "hi",
                len(partitions),
                ANY,
            ),
        )

        cursor_mock.executemany.assert_awaited_once()
        cursor_mock.close.assert_awaited_once()

    async def test_continue_session(self):
        for fixture in self.fixture_sessions:
            agent_id, participant_id, session_id = fixture

            self.assertIsInstance(agent_id, UUID)
            self.assertIsInstance(participant_id, UUID)
            self.assertIsInstance(session_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query(
                    {
                        "id": session_id,
                    }
                )

                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock,
                    logger=MagicMock(),
                )

                result = await memory.continue_session_and_get_id(
                    agent_id=agent_id,
                    participant_id=participant_id,
                    session_id=session_id,
                )

                self.assert_query(
                    connection_mock,
                    cursor_mock,
                    """
                    SELECT "sessions"."id"
                    FROM "sessions"
                    WHERE "agent_id" = %s
                    AND "participant_id" = %s
                    AND "id" = %s
                    LIMIT 1
                """,
                    (
                        str(agent_id),
                        str(participant_id),
                        str(session_id),
                    ),
                )

                self.assertEqual(str(result), str(session_id))

    async def test_get_recent_messages(self):
        for fixture in self.fixture_recent_messages:
            agent_id, participant_id, session_id, limit, model_id, messages = (
                fixture
            )

            self.assertIsInstance(participant_id, UUID)
            self.assertIsInstance(session_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query(
                    # descending order
                    reversed(
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
                            }
                            for m in messages
                        ]
                    ),
                    fetch_all=True,
                )

                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock,
                    logger=MagicMock(),
                )

                result = await memory.get_recent_messages(
                    session_id=session_id,
                    participant_id=participant_id,
                    limit=limit,
                )

                self.assert_query(
                    connection_mock,
                    cursor_mock,
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
                    INNER JOIN "messages" ON "sessions"."id" =
                               "messages"."session_id"
                    WHERE "sessions"."id" = %s
                    AND "sessions"."participant_id" = %s
                    AND "messages"."is_deleted" = FALSE
                    ORDER BY "messages"."created_at" DESC
                    LIMIT %s
                """,
                    (str(session_id), str(participant_id), limit),
                    fetch_all=True,
                )

                expected_count = (
                    limit if limit and len(messages) > limit else len(messages)
                )
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

    async def test_search_messages(self):  # noqa: F811
        for fixture in self.fixture_search_messages:
            (
                agent_id,
                participant_id,
                session_id,
                limit,
                model_id,
                function,
                messages,
            ) = fixture

            self.assertIsInstance(participant_id, UUID)
            self.assertIsInstance(session_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query(
                    # descending order
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

                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock,
                    logger=MagicMock(),
                )

                search_function = str(function)
                search_partitions = [
                    TextPartition(
                        data="", total_tokens=1, embeddings=rand(384)
                    )
                ]

                result = await memory.search_messages(
                    search_partitions=search_partitions,
                    agent_id=agent_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    function=function,
                    limit=limit,
                    search_user_messages=True,
                    exclude_session_id=None,
                )

                self.assert_query(
                    connection_mock,
                    cursor_mock,
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
                    WHERE "sessions"."participant_id" = %s
                    AND "sessions"."agent_id" = %s
                    AND "messages"."is_deleted" = FALSE
                    AND "sessions"."id" = COALESCE(%s, "sessions"."id")
                    ORDER BY "score" ASC
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

                expected_count = (
                    limit if limit and len(messages) > limit else len(messages)
                )
                self.assertEqual(len(result), expected_count)

                for i, message in enumerate(
                    sorted(messages, key=lambda m: m[2])
                ):
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

    async def test_search_messages(self):  # noqa: F811
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
                pool_mock, connection_mock, cursor_mock = self.mock_query(
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
                memory = await PgsqlMessageMemory.create_instance_from_pool(
                    pool=pool_mock,
                    logger=MagicMock(),
                )
                search_partitions = [
                    TextPartition(data="", total_tokens=1, embeddings=rand(3))
                ]
                result = await memory.search_messages(
                    search_partitions=search_partitions,
                    agent_id=agent_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    function=function,
                    limit=limit,
                    search_user_messages=True,
                    exclude_session_id=None,
                )
                self.assert_query(
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
                        \"sessions\".\"id\" =
                            \"message_partitions\".\"session_id\"
                    )
                    INNER JOIN \"messages\" ON (
                        \"message_partitions\".\"message_id\" =
                            \"messages\".\"id\"
                    )
                    WHERE \"sessions\".\"participant_id\" = %s
                    AND \"sessions\".\"agent_id\" = %s
                    AND \"messages\".\"is_deleted\" = FALSE
                    AND \"messages\".\"author\" = (
                        CASE WHEN %s THEN 'user'::message_author_type
                        ELSE \"messages\".\"author\"
                        END
                    )
                    AND \"sessions\".\"id\" = COALESCE(%s, \"sessions\".\"id\")
                    AND \"sessions\".\"id\" != COALESCE(%s::UUID, NULL)
                    ORDER BY \"score\" ASC
                    LIMIT %s
                    """,
                    (
                        Vector(search_partitions[0].embeddings),
                        str(participant_id),
                        str(agent_id),
                        True,
                        str(session_id),
                        None,
                        limit,
                    ),
                    fetch_all=True,
                )

                self.assertEqual(len(result), len(messages))
                for i, msg in enumerate(sorted(messages, key=lambda m: m[2])):
                    self.assertEqual(result[i].message.content, msg[1])

    async def test_search_memories(self):
        for fixture in self.fixture_search_memories:
            (
                participant_id,
                namespace,
                limit,
                model_id,
                mem_type,
                function,
                memories,
            ) = fixture

            self.assertIsInstance(participant_id, UUID)

            with self.subTest():
                pool_mock, connection_mock, cursor_mock = self.mock_query(
                    [
                        {
                            "id": uuid4(),
                            "model_id": model_id,
                            "type": str(mem_type),
                            "participant_id": participant_id,
                            "namespace": namespace,
                            "identifier": m[0],
                            "data": m[1],
                            "partitions": 1,
                            "symbols": m[2],
                            "created_at": datetime.now(timezone.utc),
                        }
                        for m in memories
                    ],
                    fetch_all=True,
                )

                memory_store = await PgsqlRawMemory.create_instance_from_pool(
                    pool=pool_mock,
                    logger=MagicMock(),
                )

                search_function = str(function)
                search_partitions = [
                    TextPartition(
                        data="",
                        total_tokens=1,
                        embeddings=rand(384),
                    )
                ]

                result = await memory_store.search_memories(
                    search_partitions=search_partitions,
                    participant_id=participant_id,
                    namespace=namespace,
                    function=function,
                    limit=limit,
                )

                self.assert_query(
                    connection_mock,
                    cursor_mock,
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
                        Vector(search_partitions[0].embeddings),
                        limit or 10,
                    ),
                    fetch_all=True,
                )

                expected_count = (
                    limit if limit and len(memories) > limit else len(memories)
                )
                self.assertEqual(len(result), expected_count)

                for i, mem in enumerate(memories):
                    identifier, data, symbols = mem
                    result_item = result[i]
                    self.assertEqual(result_item.model_id, model_id)
                    self.assertEqual(result_item.type, mem_type)
                    self.assertEqual(
                        result_item.participant_id, participant_id
                    )
                    self.assertEqual(result_item.namespace, namespace)
                    self.assertEqual(result_item.identifier, identifier)
                    self.assertEqual(result_item.data, data)
                    self.assertEqual(result_item.symbols, symbols)
                    if i == expected_count - 1:
                        break

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
        pool_mock = AsyncMock(spec=AsyncConnectionPool)
        pool_mock.connection.return_value = connection_mock
        pool_mock.__aenter__.return_value = pool_mock
        return pool_mock, connection_mock, cursor_mock, transaction_mock

    @staticmethod
    def mock_query(
        record_set: dict | list[dict], fetch_all: bool = False
    ) -> tuple[AsyncConnectionPool, AsyncConnection, AsyncCursor]:
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
        fetch_all: bool = False,
    ) -> None:
        def normalize_whitespace(text: str) -> str:
            return sub(r"\s+", " ", text.strip())

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


if __name__ == "__main__":
    main()
