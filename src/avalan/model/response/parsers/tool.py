"""Parser emitting events for detected tool calls."""

from ....entities import ToolCallToken, ToolFormat
from ....event import Event, EventType
from ....event.manager import EventManager
from ....tool.manager import ToolManager
from io import StringIO
from time import perf_counter
from typing import Any, Iterable


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
        self._pending_tokens: list[str] = []
        self._pending_str = ""
        tool_format = tool_manager.tool_format
        self._start_sequences = ["<tool_call", "<tool ", "<tool>"]
        self._end_sequences = ["</tool_call>", "</tool>", "/>", "<|call|>"]
        if tool_format is ToolFormat.HARMONY:
            self._start_sequences.append("<|channel|>commentary")

    async def push(self, token_str: str) -> Iterable[Any]:
        buffer_value = self._buffer.getvalue()
        should_check = self._tool_manager.is_potential_tool_call(
            buffer_value, token_str
        )

        self._buffer.write(token_str)
        self._tag_buffer += token_str
        if len(self._tag_buffer) > 64:
            self._tag_buffer = self._tag_buffer[-64:]

        result: list[Any] = []

        if not self._inside_call:
            candidate = self._pending_str + token_str
            start_triggered = False
            for start in self._start_sequences:
                if start.startswith(candidate):
                    self._pending_tokens.append(token_str)
                    self._pending_str = candidate
                    return result
                if candidate.startswith(start):
                    self._inside_call = True
                    start_triggered = True
                    self._pending_tokens.append(token_str)
                    result.extend(
                        ToolCallToken(t) for t in self._pending_tokens
                    )
                    self._pending_tokens.clear()
                    self._pending_str = ""
                    break
            if not start_triggered:
                if self._pending_tokens:
                    result.extend(self._pending_tokens)
                    self._pending_tokens.clear()
                    self._pending_str = ""
                result.append(token_str)
        else:
            result.append(ToolCallToken(token_str))
            if any(end in self._tag_buffer for end in self._end_sequences):
                self._inside_call = False

        if not result:
            return result

        if not should_check:
            return result

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        calls = self._tool_manager.get_calls(self._buffer.getvalue())
        if not calls:
            return result

        event = Event(
            type=EventType.TOOL_PROCESS, payload=calls, started=perf_counter()
        )

        self._buffer = StringIO()
        self._tag_buffer = ""
        self._inside_call = False
        return result + [event]

    async def flush(self) -> Iterable[Any]:
        result = []
        if self._pending_tokens:
            result.extend(self._pending_tokens)
            self._pending_tokens.clear()
            self._pending_str = ""
        return result
