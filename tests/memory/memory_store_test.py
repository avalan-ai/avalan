from unittest import IsolatedAsyncioTestCase, TestCase
from uuid import uuid4

from avalan.entities import EngineMessage, Message, MessageRole
from avalan.memory import MemoryStore, MessageMemory, RecentMessageMemory


class DummyMemoryStore(MemoryStore[int]):
    pass


DummyMemoryStore.__abstractmethods__ = set()


class DummyMessageMemory(MessageMemory):
    pass


DummyMessageMemory.__abstractmethods__ = set()


class MemoryStoreTestCase(IsolatedAsyncioTestCase):
    async def test_append_not_implemented(self):
        memory = DummyMemoryStore()
        with self.assertRaises(NotImplementedError):
            await memory.append(uuid4(), 1)

    async def test_reset_not_implemented(self):
        memory = DummyMemoryStore()
        with self.assertRaises(NotImplementedError):
            await memory.reset()

    async def test_search_not_implemented(self):
        memory = DummyMemoryStore()
        with self.assertRaises(NotImplementedError):
            await memory.search("query")


class MessageMemoryTestCase(TestCase):
    def test_search_not_implemented(self):
        memory = DummyMessageMemory()
        with self.assertRaises(NotImplementedError):
            memory.search("query")


class RecentMessageMemoryTestCase(TestCase):
    def test_size_property(self):
        memory = RecentMessageMemory()
        self.assertEqual(memory.size, 0)

        msg = EngineMessage(
            agent_id=uuid4(),
            model_id="m",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        memory.append(msg)
        self.assertEqual(memory.size, 1)

        memory.append(msg)
        self.assertEqual(memory.size, 2)

        memory.reset()
        self.assertEqual(memory.size, 0)
