"""Parser emitting events for detected tool calls."""

from io import StringIO
from time import perf_counter
from typing import Any, Iterable

from .....event import Event, EventType
from .....event.manager import EventManager
from .....tool.manager import ToolManager


class ToolCallParser:
    """Parse tool calls during streaming."""

    def __init__(
        self, tool_manager: ToolManager, event_manager: EventManager | None
    ) -> None:
        self._tool_manager = tool_manager
        self._event_manager = event_manager
        self._buffer = StringIO()

    async def push(self, token_str: str) -> Iterable[Any]:
        buffer_value = self._buffer.getvalue()
        should_check = self._tool_manager.is_potential_tool_call(
            buffer_value, token_str
        )
        self._buffer.write(token_str)
        if not should_check:
            return [token_str]

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        calls = self._tool_manager.get_calls(self._buffer.getvalue())
        if not calls:
            return [token_str]

        event = Event(
            type=EventType.TOOL_PROCESS, payload=calls, started=perf_counter()
        )

        self._buffer = StringIO()
        return [token_str, event]

    async def flush(self) -> Iterable[Any]:
        return []
