from avalan.memory.manager import MemoryManager
from avalan.memory import RecentMessageMemory
from avalan.memory.permanent import PermanentMessageMemory, VectorFunction
from avalan.model.entities import EngineMessage, Message, MessageRole
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch
import types


class MemoryManagerCreateTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance_with_recent_only(self):
        tp = AsyncMock()
        agent_id = uuid4()
        participant_id = uuid4()

        manager = await MemoryManager.create_instance(
            agent_id=agent_id,
            participant_id=participant_id,
            text_partitioner=tp,
        )

        self.assertIsInstance(manager.recent_message, RecentMessageMemory)
        self.assertIsNone(manager.permanent_message)
        self.assertTrue(manager.has_recent_message)
        self.assertFalse(manager.has_permanent_message)

    async def test_create_instance_with_permanent(self):
        tp = AsyncMock()
        pmemory = MagicMock(spec=PermanentMessageMemory)
        agent_id = uuid4()
        participant_id = uuid4()

        with self.subTest():
            dummy = types.SimpleNamespace()
            class PgsqlDummy:
                pass
            PgsqlDummy.create_instance = AsyncMock(return_value=pmemory)
            dummy.PgsqlMessageMemory = PgsqlDummy
            with patch.dict("sys.modules", {"avalan.memory.permanent.pgsql": dummy}):
                manager = await MemoryManager.create_instance(
                    agent_id=agent_id,
                    participant_id=participant_id,
                    text_partitioner=tp,
                    with_permanent_message_memory="dsn",
                )
            PgsqlDummy.create_instance.assert_awaited_once_with(dsn="dsn")

        self.assertIs(manager.permanent_message, pmemory)
        self.assertIsInstance(manager.recent_message, RecentMessageMemory)
        self.assertTrue(manager.has_recent_message)
        self.assertTrue(manager.has_permanent_message)


class MemoryManagerOperationTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tp = AsyncMock()
        self.pm = AsyncMock(spec=PermanentMessageMemory)
        self.rm = RecentMessageMemory()
        self.manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=self.pm,
            recent_message_memory=self.rm,
            text_partitioner=self.tp,
        )

    async def test_append_message(self):
        partitions = ["p"]
        self.tp.return_value = partitions
        msg = EngineMessage(
            agent_id=self.manager._agent_id,
            model_id="m",
            message=Message(role=MessageRole.USER, content="hi"),
        )

        await self.manager.append_message(msg)

        self.tp.assert_awaited_once_with("hi")
        self.pm.append_with_partitions.assert_awaited_once_with(
            msg, partitions=partitions
        )
        self.assertEqual(self.rm.data, [msg])

    async def test_continue_session_load_recent(self):
        messages = [
            EngineMessage(
                agent_id=self.manager._agent_id,
                model_id="m",
                message=Message(role=MessageRole.USER, content="x"),
            )
        ]
        self.pm.get_recent_messages.return_value = messages

        await self.manager.continue_session(uuid4())

        self.pm.continue_session.assert_awaited()
        self.pm.get_recent_messages.assert_awaited()
        self.assertEqual(self.rm.data, messages)

    async def test_start_session(self):
        await self.manager.start_session()
        self.pm.reset_session.assert_awaited_once()
        self.assertTrue(self.rm.is_empty)

    async def test_search_messages(self):
        partitions = ["p1"]
        self.tp.return_value = partitions
        result = [
            EngineMessage(
                agent_id=self.manager._agent_id,
                model_id="m",
                message=Message(role=MessageRole.USER, content="hi"),
            )
        ]
        self.pm.search_messages.return_value = result

        messages = await self.manager.search_messages(
            "hi",
            agent_id=uuid4(),
            session_id=uuid4(),
            participant_id=uuid4(),
            function=VectorFunction.L2_DISTANCE,
        )

        self.tp.assert_awaited_once_with("hi")
        self.pm.search_messages.assert_awaited_once()
        self.assertEqual(messages, result)


if __name__ == "__main__":
    main()
