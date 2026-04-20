from datetime import datetime, timezone
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.entities import (
    EngineMessage,
    EngineMessageScored,
    Message,
    MessageRole,
)
from avalan.memory import RecentMessageMemory
from avalan.memory.permanent import PermanentMessageScored
from avalan.memory.permanent.pgsql import PgsqlMemory


class DummyPgsqlMemory(PgsqlMemory[object]):
    async def append(self, agent_id, data):
        del agent_id, data

    async def reset(self):
        return None

    async def search(self, query):
        del query
        return None


class RecentMessageMemoryCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_search_is_empty_and_data(self) -> None:
        memory = RecentMessageMemory()
        self.assertTrue(memory.is_empty)
        self.assertEqual(memory.data, [])

        message = EngineMessage(
            agent_id=uuid4(),
            model_id="m",
            message=Message(role=MessageRole.USER, content="hello"),
        )
        await memory.append(message.agent_id, message)

        result = await memory.search("ignored")
        self.assertEqual(result, [message])
        self.assertFalse(memory.is_empty)


class PgsqlMemoryCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_ensure_vector_extension_success_fetches_vector(
        self,
    ) -> None:
        memory = DummyPgsqlMemory(
            dsn="postgresql://db",
            logger=MagicMock(),
            pool=MagicMock(),
        )

        cursor = AsyncMock()
        cursor.__aenter__.return_value = cursor
        cursor.fetchone.side_effect = [
            {"has_vector_extension": True},
            {"vector": "ok"},
        ]

        connection = MagicMock()
        connection.cursor.return_value = cursor

        await memory._ensure_vector_extension(connection)

        self.assertEqual(cursor.fetchone.await_count, 2)

    async def test_to_engine_messages_scored_reverse_and_limit(self) -> None:
        base_kwargs = {
            "session_id": uuid4(),
            "author": MessageRole.USER,
            "partitions": 1,
            "created_at": datetime.now(timezone.utc),
        }
        messages = [
            PermanentMessageScored(
                id=uuid4(),
                agent_id=uuid4(),
                model_id="m",
                data="first",
                score=0.1,
                **base_kwargs,
            ),
            PermanentMessageScored(
                id=uuid4(),
                agent_id=uuid4(),
                model_id="m",
                data="second",
                score=0.2,
                **base_kwargs,
            ),
        ]

        result = PgsqlMemory._to_engine_messages(
            messages,
            limit=1,
            reverse=True,
            scored=True,
        )

        self.assertEqual(len(result), 1)
        first = result[0]
        self.assertIsInstance(first, EngineMessageScored)
        self.assertEqual(first.message.content, "second")
