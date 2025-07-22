"""Parser emitting events for detected tool calls."""

from io import StringIO
from time import perf_counter
from typing import Any, Iterable

from ....entities import ToolCallToken
from ....event import Event, EventType
from ....event.manager import EventManager
from ....tool.manager import ToolManager


class ToolCallParser:
    """Parse tool calls during streaming."""

    def __init__(
        self, tool_manager: ToolManager, event_manager: EventManager | None
    ) -> None:
        self._tool_manager = tool_manager
        self._event_manager = event_manager
        self._buffer = StringIO()
        self._tag_buffer = ""
        self._inside_call = False

    async def push(self, token_str: str) -> Iterable[Any]:
        buffer_value = self._buffer.getvalue()
        should_check = self._tool_manager.is_potential_tool_call(
            buffer_value, token_str
        )

        prev_inside = self._inside_call

        self._buffer.write(token_str)
        self._tag_buffer += token_str
        if len(self._tag_buffer) > 64:
            self._tag_buffer = self._tag_buffer[-64:]

        if not self._inside_call and (
            "<tool_call" in self._tag_buffer
            or "<tool " in self._tag_buffer
            or "<tool>" in self._tag_buffer
        ):
            self._inside_call = True

        if self._inside_call and (
            "</tool_call>" in self._tag_buffer
            or "</tool>" in self._tag_buffer
            or "/>" in self._tag_buffer
        ):
            self._inside_call = False

        start_triggered = not prev_inside and self._inside_call

        item = (
            ToolCallToken(token_str)
            if prev_inside or start_triggered
            else token_str
        )

        if not should_check:
            return [item]

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        calls = self._tool_manager.get_calls(self._buffer.getvalue())
        if not calls:
            return [item]

        event = Event(
            type=EventType.TOOL_PROCESS, payload=calls, started=perf_counter()
        )

        self._buffer = StringIO()
        self._tag_buffer = ""
        self._inside_call = False
        return [item, event]

    async def flush(self) -> Iterable[Any]:
        return []
