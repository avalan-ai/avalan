from avalan.entities import EngineMessage, Message, MessageRole
from avalan.memory.manager import MemoryManager
from avalan.memory import RecentMessageMemory
from avalan.memory.permanent import (
    PermanentMessageMemory,
    PermanentMemory,
    VectorFunction,
)
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch
import types


class MemoryManagerCreateTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance_with_recent_only(self):
        tp = AsyncMock()
        agent_id = uuid4()
        participant_id = uuid4()

        logger = MagicMock()
        manager = await MemoryManager.create_instance(
            agent_id=agent_id,
            participant_id=participant_id,
            text_partitioner=tp,
            logger=logger,
        )

        self.assertIsInstance(manager.recent_message, RecentMessageMemory)
        self.assertIsNone(manager.permanent_message)
        self.assertTrue(manager.has_recent_message)
        self.assertFalse(manager.has_permanent_message)
        self.assertIs(manager._logger, logger)

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
            with patch.dict(
                "sys.modules", {"avalan.memory.permanent.pgsql.message": dummy}
            ):
                logger = MagicMock()
                manager = await MemoryManager.create_instance(
                    agent_id=agent_id,
                    participant_id=participant_id,
                    text_partitioner=tp,
                    logger=logger,
                    with_permanent_message_memory="dsn",
                )
            PgsqlDummy.create_instance.assert_awaited_once_with(
                dsn="dsn", logger=logger
            )

        self.assertIs(manager.permanent_message, pmemory)
        self.assertIsInstance(manager.recent_message, RecentMessageMemory)
        self.assertTrue(manager.has_recent_message)
        self.assertTrue(manager.has_permanent_message)
        self.assertIs(manager._logger, logger)


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
            logger=MagicMock(),
        )
        self.permanent = AsyncMock()

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

    def test_add_and_delete_permanent_memory(self):
        self.manager.add_permanent_memory("code", self.permanent)
        self.assertIn("code", self.manager._permanent_memories)
        self.manager.delete_permanent_memory("code")
        self.assertNotIn("code", self.manager._permanent_memories)


class MemoryManagerInitTestCase(IsolatedAsyncioTestCase):
    def test_constructor_variants(self):
        agent_id = uuid4()
        participant_id = uuid4()
        tp = AsyncMock()

        manager = MemoryManager(
            agent_id=agent_id,
            participant_id=participant_id,
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=tp,
            logger=MagicMock(),
        )
        self.assertFalse(manager.has_recent_message)
        self.assertFalse(manager.has_permanent_message)
        self.assertEqual(manager.participant_id, participant_id)
        self.assertIsNone(manager.recent_messages)

        pmem = MagicMock(spec=PermanentMemory)
        manager = MemoryManager(
            agent_id=agent_id,
            participant_id=participant_id,
            permanent_message_memory=None,
            recent_message_memory=RecentMessageMemory(),
            text_partitioner=tp,
            logger=MagicMock(),
            permanent_memories={"code": pmem},
        )
        self.assertTrue(manager.has_recent_message)
        self.assertIn("code", manager._permanent_memories)

        pmem2 = MagicMock(spec=PermanentMemory)
        manager = MemoryManager(
            agent_id=agent_id,
            participant_id=participant_id,
            permanent_message_memory=None,
            recent_message_memory=RecentMessageMemory(),
            text_partitioner=tp,
            logger=MagicMock(),
            permanent_memories={"code": pmem, "docs": pmem2},
        )
        self.assertEqual(len(manager._permanent_memories), 2)


class MemoryManagerPropertyTestCase(IsolatedAsyncioTestCase):
    def test_recent_messages_property(self):
        tp = AsyncMock()
        rm = RecentMessageMemory()
        msg = EngineMessage(
            agent_id=uuid4(),
            model_id="m",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        rm.append(msg)
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=rm,
            text_partitioner=tp,
            logger=MagicMock(),
        )
        self.assertEqual(manager.recent_messages, [msg])
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=tp,
            logger=MagicMock(),
        )
        self.assertIsNone(manager.recent_messages)


class MemoryManagerMethodsTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tp = AsyncMock()
        self.pm = AsyncMock(spec=PermanentMessageMemory)
        self.rm = RecentMessageMemory()

    async def test_append_message_variants(self):
        msg = EngineMessage(
            agent_id=uuid4(),
            model_id="m",
            message=Message(role=MessageRole.USER, content="hi"),
        )

        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=self.pm,
            recent_message_memory=None,
            text_partitioner=self.tp,
            logger=MagicMock(),
        )
        await manager.append_message(msg)
        self.tp.assert_awaited_once()
        self.pm.append_with_partitions.assert_awaited_once()

        self.tp.reset_mock()
        self.pm.reset_mock()
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=self.rm,
            text_partitioner=self.tp,
            logger=MagicMock(),
        )
        await manager.append_message(msg)
        self.tp.assert_not_called()
        self.assertEqual(manager.recent_messages, [msg])

        self.tp.reset_mock()
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=self.tp,
            logger=MagicMock(),
        )
        await manager.append_message(msg)
        self.tp.assert_not_called()

    async def test_continue_and_start_session_variants(self):
        session_id = uuid4()
        messages = [
            EngineMessage(
                agent_id=uuid4(),
                model_id="m",
                message=Message(role=MessageRole.USER, content="x"),
            )
        ]
        self.pm.get_recent_messages.return_value = messages

        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=self.pm,
            recent_message_memory=None,
            text_partitioner=self.tp,
            logger=MagicMock(),
        )
        await manager.continue_session(session_id)
        self.pm.continue_session.assert_awaited()
        self.pm.get_recent_messages.assert_not_awaited()
        await manager.start_session()
        self.pm.reset_session.assert_awaited()

        self.pm.reset_mock()
        self.pm.get_recent_messages.reset_mock()
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=self.rm,
            text_partitioner=self.tp,
            logger=MagicMock(),
        )
        await manager.continue_session(session_id)
        self.pm.continue_session.assert_not_awaited()
        self.assertEqual(manager.recent_messages, [])
        await manager.start_session()
        self.assertTrue(self.rm.is_empty)

        self.pm.reset_mock()
        self.pm.get_recent_messages.reset_mock()
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=self.tp,
            logger=MagicMock(),
        )
        await manager.continue_session(session_id)
        await manager.start_session()


class MemoryManagerContextTestCase(IsolatedAsyncioTestCase):
    async def test_context_exit(self):
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=AsyncMock(),
            logger=MagicMock(),
        )
        self.assertIsNone(manager.__exit__(None, None, None))


if __name__ == "__main__":
    main()
