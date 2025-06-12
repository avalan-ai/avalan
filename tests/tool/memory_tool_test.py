from avalan.entities import (
    EngineMessage,
    Message,
    MessageRole,
    ToolCallContext,
)
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent import VectorFunction
from avalan.tool.memory import MemoryToolSet, MessageReadTool
from contextlib import AsyncExitStack
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


class MemoryToolSetTestCase(IsolatedAsyncioTestCase):
    async def test_init_and_enter(self):
        exit_stack = AsyncMock(spec=AsyncExitStack)
        exit_stack.enter_async_context = AsyncMock(return_value=None)
        exit_stack.__aexit__ = AsyncMock(return_value=False)

        manager = MagicMock(spec=MemoryManager)
        toolset = MemoryToolSet(
            manager, exit_stack=exit_stack, namespace="mem"
        )

        self.assertEqual(toolset.namespace, "mem")
        self.assertEqual(len(toolset.tools), 1)
        tool = toolset.tools[0]
        self.assertIsInstance(tool, MessageReadTool)

        result = await toolset.__aenter__()

        self.assertIs(result, toolset)
        exit_stack.enter_async_context.assert_awaited_once_with(tool)


class MessageReadToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = AsyncMock(spec=MemoryManager)
        self.tool = MessageReadTool(self.manager)
        self.agent_id = uuid4()
        self.session_id = uuid4()
        self.participant_id = uuid4()

    async def test_missing_context_returns_not_found(self):
        contexts = [
            ToolCallContext(),
            ToolCallContext(
                agent_id=self.agent_id, session_id=self.session_id
            ),
            ToolCallContext(
                agent_id=self.agent_id, participant_id=self.participant_id
            ),
            ToolCallContext(
                session_id=self.session_id, participant_id=self.participant_id
            ),
        ]
        for ctx in contexts:
            with self.subTest(ctx=ctx):
                result = await self.tool("hi", context=ctx)
                self.assertEqual(result, MessageReadTool._NOT_FOUND)
                self.manager.search_messages.assert_not_awaited()
                self.manager.search_messages.reset_mock()

    async def test_returns_message_content(self):
        msg = EngineMessage(
            agent_id=self.agent_id,
            model_id="m",
            message=Message(role=MessageRole.USER, content="hello"),
        )
        self.manager.search_messages.return_value = [msg]
        ctx = ToolCallContext(
            agent_id=self.agent_id,
            session_id=self.session_id,
            participant_id=self.participant_id,
        )
        result = await self.tool("name", context=ctx)

        self.manager.search_messages.assert_awaited_once_with(
            agent_id=self.agent_id,
            exclude_session_id=self.session_id,
            function=VectorFunction.L2_DISTANCE,
            participant_id=self.participant_id,
            search="name",
            search_user_messages=True,
            limit=1,
        )
        self.assertEqual(result, "hello")

    async def test_returns_not_found_when_no_results(self):
        self.manager.search_messages.return_value = []
        ctx = ToolCallContext(
            agent_id=self.agent_id,
            session_id=self.session_id,
            participant_id=self.participant_id,
        )
        result = await self.tool("age", context=ctx)

        self.manager.search_messages.assert_awaited_once()
        self.assertEqual(result, MessageReadTool._NOT_FOUND)


if __name__ == "__main__":
    main()
