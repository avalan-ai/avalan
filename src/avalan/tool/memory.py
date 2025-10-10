from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
from ..memory.manager import MemoryManager
from ..memory.permanent import (
    Memory,
    PermanentMemoryPartition,
    PermanentMemoryStore,
    VectorFunction,
)

from contextlib import AsyncExitStack


class MessageReadTool(Tool):
    """Fetch user information and preferences from previous messages.

    Args:
        search: Information to retrieve from past conversations.

    Returns:
        Retrieved message content or 'NOT_FOUND' when no match exists.
"""

    _memory_manager: MemoryManager
    _NOT_FOUND = "NOT_FOUND"

    def __init__(self, memory_manager: MemoryManager) -> None:
        super().__init__()
        self._memory_manager = memory_manager
        self.__name__ = "message.read"

    async def __call__(self, search: str, context: ToolCallContext) -> str:
        if (
            not context.agent_id
            or not context.session_id
            or not context.participant_id
        ):
            return MessageReadTool._NOT_FOUND

        results = await self._memory_manager.search_messages(
            agent_id=context.agent_id,
            exclude_session_id=context.session_id,
            function=VectorFunction.L2_DISTANCE,
            participant_id=context.participant_id,
            search=search,
            search_user_messages=True,
            limit=1,
        )
        if results and results[0].message:
            return results[0].message.content

        return MessageReadTool._NOT_FOUND


class MemoryReadTool(Tool):
    """Search permanent memories for stored knowledge.

    Args:
        namespace: Namespace to fetch information from.
        search: Information to locate within the namespace.
        limit: Maximum number of matches to fetch.

    Returns:
        Matching permanent memory partitions.
"""

    _memory_manager: MemoryManager
    _function: VectorFunction

    def __init__(
        self,
        memory_manager: MemoryManager,
        function: VectorFunction = VectorFunction.L2_DISTANCE,
    ) -> None:
        super().__init__()
        self._memory_manager = memory_manager
        self._function = function
        self.__name__ = "read"

    async def __call__(
        self,
        namespace: str,
        search: str,
        *,
        context: ToolCallContext,
    ) -> list[PermanentMemoryPartition]:
        """Return memory partitions that match the search query."""
        if (
            not namespace
            or not namespace.strip()
            or not search
            or not search.strip()
        ):
            return []

        if not context.participant_id:
            return []

        default_limit = 10
        memory_partitions = await self._memory_manager.search_partitions(
            search,
            participant_id=context.participant_id,
            namespace=namespace,
            function=self._function,
            limit=default_limit,
        )
        memories = [mp.data for mp in memory_partitions]
        return memories


class MemoryListTool(Tool):
    """List permanent memories available for a namespace.

    Args:
        namespace: Namespace to list memories from.

    Returns:
        Memories stored in the requested namespace.
"""

    _memory_manager: MemoryManager

    def __init__(self, memory_manager: MemoryManager) -> None:
        super().__init__()
        self._memory_manager = memory_manager
        self.__name__ = "list"

    async def __call__(
        self,
        namespace: str,
        *,
        context: ToolCallContext,
    ) -> list[Memory]:
        if not namespace or not namespace.strip():
            return []
        if not context.participant_id:
            return []
        memories = await self._memory_manager.list_memories(
            participant_id=context.participant_id,
            namespace=namespace,
        )
        return memories


class MemoryStoresTool(Tool):
    """List memory stores available.

    Args:
        None.

    Returns:
        Available permanent memory stores.
"""

    _memory_manager: MemoryManager

    def __init__(self, memory_manager: MemoryManager) -> None:
        super().__init__()
        self._memory_manager = memory_manager
        self.__name__ = "stores"

    async def __call__(
        self,
        *,
        context: ToolCallContext,
    ) -> list[PermanentMemoryStore]:
        return self._memory_manager.list_permanent_memory_stores()


class MemoryToolSet(ToolSet):
    @override
    def __init__(
        self,
        memory_manager: MemoryManager,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ) -> None:
        tools = [
            MessageReadTool(memory_manager),
            MemoryReadTool(memory_manager),
            MemoryListTool(memory_manager),
            MemoryStoresTool(memory_manager),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
