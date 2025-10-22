from datetime import datetime, timezone
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np

from avalan.entities import EngineMessage, Message, MessageRole, TextPartition
from avalan.memory.permanent import (
    MemoryType,
    PermanentMemory,
    PermanentMessageMemory,
    VectorFunction,
)
from avalan.model.nlp.sentence import SentenceTransformerModel


class DummyPermanentMessageMemory(PermanentMessageMemory):
    pass


DummyPermanentMessageMemory.__abstractmethods__ = set()


class DummyPermanentMemory(PermanentMemory):
    pass


DummyPermanentMemory.__abstractmethods__ = set()


class PermanentMessageMemorySyncTestCase(TestCase):
    def setUp(self):
        self.model = MagicMock(spec=SentenceTransformerModel)
        self.memory = DummyPermanentMessageMemory(sentence_model=self.model)

    def test_init_and_properties(self):
        self.assertIs(self.memory._sentence_model, self.model)
        self.assertIsNone(self.memory.session_id)
        self.assertFalse(self.memory.has_session)
        sid = uuid4()
        self.memory._session_id = sid
        self.assertTrue(self.memory.has_session)
        self.assertEqual(self.memory.session_id, sid)

    def test_reset_raises(self):
        with self.assertRaises(NotImplementedError):
            self.memory.reset()

    def test_append_raises(self):
        with self.assertRaises(NotImplementedError):
            self.memory.append(MagicMock())


class PermanentMessageMemoryAsyncTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = MagicMock(spec=SentenceTransformerModel)
        self.memory = DummyPermanentMessageMemory(sentence_model=self.model)

    async def test_reset_and_continue_session(self):
        sid = uuid4()
        self.memory.create_session = AsyncMock(return_value=sid)
        await self.memory.reset_session(
            agent_id=uuid4(), participant_id=uuid4()
        )
        self.assertEqual(self.memory.session_id, sid)
        self.memory.create_session.assert_awaited_once()

        sid2 = uuid4()
        self.memory.continue_session_and_get_id = AsyncMock(return_value=sid2)
        await self.memory.continue_session(
            agent_id=uuid4(), participant_id=uuid4(), session_id=sid
        )
        self.assertEqual(self.memory.session_id, sid2)
        self.memory.continue_session_and_get_id.assert_awaited_once()

    async def test_create_session_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            await PermanentMessageMemory.create_session(
                self.memory, uuid4(), uuid4()
            )

    async def test_continue_session_and_get_id_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            await PermanentMessageMemory.continue_session_and_get_id(
                self.memory,
                agent_id=uuid4(),
                participant_id=uuid4(),
                session_id=uuid4(),
            )

    async def test_append_with_partitions_not_implemented(self):
        msg = EngineMessage(
            agent_id=uuid4(),
            model_id="m",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        with self.assertRaises(NotImplementedError):
            await PermanentMessageMemory.append_with_partitions(
                self.memory, msg, partitions=[]
            )

    async def test_get_recent_messages_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            await PermanentMessageMemory.get_recent_messages(
                self.memory, uuid4(), uuid4()
            )

    async def test_search_messages_not_implemented(self):
        part = TextPartition(
            data="x", total_tokens=1, embeddings=np.array([0.0])
        )
        with self.assertRaises(NotImplementedError):
            await PermanentMessageMemory.search_messages(
                self.memory,
                agent_id=uuid4(),
                function=VectorFunction.L2_DISTANCE,
                participant_id=uuid4(),
                search_partitions=[part],
                search_user_messages=False,
                session_id=None,
                exclude_session_id=None,
            )


class PermanentMemoryTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = MagicMock(spec=SentenceTransformerModel)
        self.memory = DummyPermanentMemory(sentence_model=self.model)

    def test_init(self):
        self.assertIs(self.memory._sentence_model, self.model)

    def test_append_raises(self):
        with self.assertRaises(NotImplementedError):
            self.memory.append(MagicMock())

    def test_reset_raises(self):
        with self.assertRaises(NotImplementedError):
            self.memory.reset()

    async def test_append_with_partitions_not_implemented(self):
        part = TextPartition(
            data="x", total_tokens=1, embeddings=np.array([0.0])
        )
        with self.assertRaises(NotImplementedError):
            await PermanentMemory.append_with_partitions(
                self.memory,
                "ns",
                uuid4(),
                memory_type=MemoryType.RAW,
                data="d",
                identifier="id",
                partitions=[part],
            )

    async def test_search_memories_not_implemented(self):
        part = TextPartition(
            data="x", total_tokens=1, embeddings=np.array([0.0])
        )
        with self.assertRaises(NotImplementedError):
            await PermanentMemory.search_memories(
                self.memory,
                search_partitions=[part],
                participant_id=uuid4(),
                namespace="ns",
                function=VectorFunction.COSINE_DISTANCE,
            )

    async def test_list_memories_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            await PermanentMemory.list_memories(
                self.memory,
                participant_id=uuid4(),
                namespace="ns",
            )


class BuildPartitionsTestCase(TestCase):
    def setUp(self):
        self.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.engine_message = EngineMessage(
            agent_id=uuid4(),
            model_id="model",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        self.partitions = [
            TextPartition(
                data="a", total_tokens=1, embeddings=np.array([0.1])
            ),
            TextPartition(
                data="b", total_tokens=1, embeddings=np.array([0.2])
            ),
        ]
        self.session_id = uuid4()
        self.participant_id = uuid4()

    def test_build_message_with_generated_id(self):
        msg_id = UUID("11111111-1111-1111-1111-111111111111")
        with patch("avalan.memory.permanent.uuid4", return_value=msg_id):
            message, mp = (
                PermanentMessageMemory._build_message_with_partitions(
                    self.engine_message,
                    self.session_id,
                    self.partitions,
                    created_at=self.created_at,
                )
            )
        self.assertEqual(message.id, msg_id)
        self.assertEqual(len(mp), 2)
        self.assertEqual(mp[0].partition, 1)
        self.assertEqual(mp[0].message_id, msg_id)
        self.assertEqual(mp[1].partition, 2)
        self.assertEqual(mp[1].message_id, msg_id)

    def test_build_message_with_explicit_id(self):
        msg_id = uuid4()
        message, mp = PermanentMessageMemory._build_message_with_partitions(
            self.engine_message,
            self.session_id,
            self.partitions,
            created_at=self.created_at,
            message_id=msg_id,
        )
        self.assertEqual(message.id, msg_id)
        self.assertEqual(mp[0].message_id, msg_id)

    def test_build_memory_with_generated_id(self):
        mem_id = UUID("22222222-2222-2222-2222-222222222222")
        with patch("avalan.memory.permanent.uuid4", return_value=mem_id):
            entry, rows = PermanentMemory._build_memory_with_partitions(
                "ns",
                self.participant_id,
                MemoryType.RAW,
                data="d",
                identifier="id",
                partitions=self.partitions,
                created_at=self.created_at,
            )
        self.assertEqual(entry.id, mem_id)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].memory_id, mem_id)
        self.assertEqual(rows[0].partition, 1)
        self.assertEqual(rows[1].memory_id, mem_id)
        self.assertEqual(rows[1].partition, 2)

    def test_build_memory_with_explicit_id(self):
        mem_id = uuid4()
        entry, rows = PermanentMemory._build_memory_with_partitions(
            "ns",
            self.participant_id,
            MemoryType.RAW,
            data="d",
            identifier="id",
            partitions=self.partitions,
            created_at=self.created_at,
            memory_id=mem_id,
        )
        self.assertEqual(entry.id, mem_id)
        self.assertEqual(rows[0].memory_id, mem_id)
        self.assertEqual(rows[0].partition, 1)
        self.assertEqual(rows[1].memory_id, mem_id)
        self.assertEqual(rows[1].partition, 2)
