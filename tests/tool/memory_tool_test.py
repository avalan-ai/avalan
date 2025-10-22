from contextlib import AsyncExitStack
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

from pytest import raises

from avalan.entities import (
    EngineMessage,
    Message,
    MessageRole,
    ToolCallContext,
)
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent import (
    PermanentMemoryPartition,
    PermanentMemoryStore,
    VectorFunction,
)
from avalan.tool.memory import (
    MemoryListTool,
    MemoryReadTool,
    MemoryStoresTool,
    MemoryToolSet,
    MessageReadTool,
)


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
        self.assertEqual(len(toolset.tools), 4)
        message_tool, memory_tool, list_tool, stores_tool = toolset.tools
        self.assertIsInstance(message_tool, MessageReadTool)
        self.assertIsInstance(memory_tool, MemoryReadTool)
        self.assertIsInstance(list_tool, MemoryListTool)
        self.assertIsInstance(stores_tool, MemoryStoresTool)

        result = await toolset.__aenter__()

        self.assertIs(result, toolset)
        exit_stack.enter_async_context.assert_has_awaits(
            [
                call(message_tool),
                call(memory_tool),
                call(list_tool),
                call(stores_tool),
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
        memory_partition = MagicMock(spec=PermanentMemoryPartition)
        type(memory_partition).data = "Leo Messi is the GOAT"
        self.manager.search_partitions.return_value = [memory_partition]

        result = await self.tool(
            "docs",
            "agent architecture",
            context=ctx,
        )

        self.manager.search_partitions.assert_awaited_once_with(
            "agent architecture",
            participant_id=self.participant_id,
            namespace="docs",
            function=VectorFunction.L2_DISTANCE,
            limit=10,
        )
        self.assertEqual(result, ["Leo Messi is the GOAT"])

    async def test_returns_empty_on_missing_namespace(self):
        ctx = ToolCallContext(participant_id=self.participant_id)
        self.manager.search_partitions.side_effect = KeyError("docs")

        with raises(KeyError):
            await self.tool("docs", "query", context=ctx)


class MemoryListToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = AsyncMock(spec=MemoryManager)
        self.tool = MemoryListTool(self.manager)
        self.participant_id = uuid4()

    async def test_returns_empty_without_participant(self):
        ctx = ToolCallContext()
        result = await self.tool("docs", context=ctx)
        self.assertEqual(result, [])
        self.manager.list_memories.assert_not_awaited()

    async def test_returns_empty_when_namespace_invalid(self):
        ctx = ToolCallContext(participant_id=self.participant_id)
        result = await self.tool(" ", context=ctx)
        self.assertEqual(result, [])
        self.manager.list_memories.assert_not_awaited()

    async def test_returns_empty_on_missing_namespace(self):
        ctx = ToolCallContext(participant_id=self.participant_id)
        self.manager.list_memories.side_effect = KeyError("docs")
        with raises(KeyError):
            await self.tool("docs", context=ctx)
        self.manager.list_memories.assert_awaited_once_with(
            participant_id=self.participant_id,
            namespace="docs",
        )

    async def test_returns_memories(self):
        ctx = ToolCallContext(participant_id=self.participant_id)
        memories = [MagicMock()]
        self.manager.list_memories.return_value = memories
        result = await self.tool("docs", context=ctx)
        self.manager.list_memories.assert_awaited_once_with(
            participant_id=self.participant_id,
            namespace="docs",
        )
        self.assertIs(result, memories)


class MemoryStoresToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = AsyncMock(spec=MemoryManager)
        self.tool = MemoryStoresTool(self.manager)

    async def test_returns_empty_when_no_stores(self):
        self.manager.list_permanent_memory_stores.return_value = []
        result = await self.tool(context=ToolCallContext())
        self.assertEqual(result, [])
        self.manager.list_permanent_memory_stores.assert_called_once_with()

    async def test_returns_single_store(self):
        store = PermanentMemoryStore(namespace="docs", description=None)
        self.manager.list_permanent_memory_stores.return_value = [store]
        result = await self.tool(context=ToolCallContext())
        self.assertEqual(result, [store])
        self.manager.list_permanent_memory_stores.assert_called_once_with()

    async def test_returns_multiple_stores(self):
        stores = [
            PermanentMemoryStore(namespace="docs", description="Documents"),
            PermanentMemoryStore(namespace="code", description=None),
        ]
        self.manager.list_permanent_memory_stores.return_value = stores
        result = await self.tool(context=ToolCallContext())
        self.assertEqual(result, stores)
        self.manager.list_permanent_memory_stores.assert_called_once_with()


if __name__ == "__main__":
    main()
