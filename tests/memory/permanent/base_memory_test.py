from avalan.entities import EngineMessage, Message, MessageRole, TextPartition
from avalan.memory.permanent import (
    PermanentMessageMemory,
    PermanentMemory,
    MemoryType,
    VectorFunction,
)
from avalan.model.nlp.sentence import SentenceTransformerModel
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock
import numpy as np


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
