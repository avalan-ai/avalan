from avalan.entities import (
    EngineMessage,
    Message,
    MessageRole,
    ToolCallContext,
)
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent import VectorFunction
from avalan.tool.memory import MemoryReadTool, MemoryToolSet, MessageReadTool
from contextlib import AsyncExitStack
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, call
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
        self.assertEqual(len(toolset.tools), 2)
        message_tool, memory_tool = toolset.tools
        self.assertIsInstance(message_tool, MessageReadTool)
        self.assertIsInstance(memory_tool, MemoryReadTool)

        result = await toolset.__aenter__()

        self.assertIs(result, toolset)
        exit_stack.enter_async_context.assert_has_awaits(
            [
                call(message_tool),
                call(memory_tool),
            ]
        )


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


class MemoryReadToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = AsyncMock(spec=MemoryManager)
        self.tool = MemoryReadTool(self.manager)
        self.participant_id = uuid4()

    async def test_returns_empty_when_missing_participant(self):
        ctx = ToolCallContext()

        result = await self.tool("docs", "query", context=ctx)

        self.assertEqual(result, [])
        self.manager.search_partitions.assert_not_awaited()

    async def test_returns_empty_when_input_invalid(self):
        ctx = ToolCallContext(participant_id=self.participant_id)

        for namespace, query in [("", "q"), ("docs", ""), (" ", "q")]:
            with self.subTest(namespace=namespace, query=query):
                result = await self.tool(namespace, query, context=ctx)
                self.assertEqual(result, [])
        self.manager.search_partitions.assert_not_awaited()

    async def test_returns_memories(self):
        ctx = ToolCallContext(participant_id=self.participant_id)
        memory = MagicMock()
        self.manager.search_partitions.return_value = [memory]

        result = await self.tool(
            "docs",
            "agent architecture",
            context=ctx,
            limit=3,
        )

        self.manager.search_partitions.assert_awaited_once_with(
            "agent architecture",
            participant_id=self.participant_id,
            namespace="docs",
            function=VectorFunction.L2_DISTANCE,
            limit=3,
        )
        self.assertEqual(result, [memory])

    async def test_returns_empty_on_missing_namespace(self):
        ctx = ToolCallContext(participant_id=self.participant_id)
        self.manager.search_partitions.side_effect = KeyError("docs")

        result = await self.tool("docs", "query", context=ctx)

        self.assertEqual(result, [])


if __name__ == "__main__":
    main()
