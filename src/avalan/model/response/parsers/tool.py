"""Parser emitting events for detected tool calls."""

from ....entities import ToolCallDiagnostic, ToolCallToken, ToolFormat
from ....event import Event, EventType
from ....event.manager import EventManager
from ....tool.manager import ToolManager
from ....tool.parser import ToolCallParser

from io import StringIO
from time import perf_counter
from typing import Any, Iterable, cast


class ToolCallResponseParser:
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
        terminal_status: ToolCallParser.ToolCallBufferStatus | None = None

        if not self._inside_call:
            candidate = self._pending_str + token_str
            status = self._tool_manager.tool_call_status(candidate)
            if status is ToolCallParser.ToolCallBufferStatus.PREFIX:
                self._pending_tokens.append(token_str)
                self._pending_str = candidate
                return result
            if status in (
                ToolCallParser.ToolCallBufferStatus.OPEN,
                ToolCallParser.ToolCallBufferStatus.CLOSED,
            ):
                self._pending_tokens.append(token_str)
                result.extend(
                    ToolCallToken(token=t) for t in self._pending_tokens
                )
                self._pending_tokens.clear()
                self._pending_str = ""
                self._inside_call = (
                    status is ToolCallParser.ToolCallBufferStatus.OPEN
                )
                if status is ToolCallParser.ToolCallBufferStatus.CLOSED:
                    terminal_status = status
            else:
                if self._pending_tokens:
                    result.extend(self._pending_tokens)
                    self._pending_tokens.clear()
                    self._pending_str = ""
                result.append(token_str)
        else:
            result.append(ToolCallToken(token=token_str))
            status = self._tool_manager.tool_call_status(self._tag_buffer)
            if status is not ToolCallParser.ToolCallBufferStatus.CLOSED:
                status = self._tool_manager.tool_call_status(
                    f"<tool_call>{self._tag_buffer}"
                )
            if status is ToolCallParser.ToolCallBufferStatus.CLOSED:
                self._inside_call = False
                terminal_status = status

        if not result:
            return result

        if not should_check:
            return result

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        buffer_text = self._buffer.getvalue()
        calls, diagnostics = self._parse_buffer(buffer_text)
        if calls:
            event = Event(
                type=EventType.TOOL_PROCESS,
                payload=cast(dict[str, Any], calls),
                started=perf_counter(),
            )

            self._clear_buffers()
            return result + [event]

        if terminal_status is ToolCallParser.ToolCallBufferStatus.CLOSED:
            if not diagnostics:
                diagnostics = self._stream_buffer_diagnostics(buffer_text)
            if diagnostics:
                event = await self._diagnostic_event(diagnostics)
                self._clear_buffers()
                return result + [event]

            return result

        return result

    async def _diagnostic_event(
        self, diagnostics: list[ToolCallDiagnostic]
    ) -> Event:
        event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"diagnostics": cast(Any, diagnostics)},
            started=perf_counter(),
        )
        if self._event_manager:
            await self._event_manager.trigger(event)
        return event

    async def flush(self) -> Iterable[Any]:
        result: list[Any] = []
        if self._inside_call:
            buffer_text = self._buffer.getvalue()
            calls: list[Any] | None = None
            if self._tool_manager.tool_format is ToolFormat.HARMONY:
                calls, _ = self._parse_buffer(f"{buffer_text}<|call|>")
            if calls:
                if self._event_manager:
                    await self._event_manager.trigger(
                        Event(type=EventType.TOOL_DETECT)
                    )

                event = Event(
                    type=EventType.TOOL_PROCESS,
                    payload=cast(dict[str, Any], calls),
                    started=perf_counter(),
                )
                self._clear_buffers()
                result.append(event)
            else:
                diagnostics = self._stream_buffer_diagnostics(buffer_text)
                if diagnostics:
                    result.append(await self._diagnostic_event(diagnostics))
                    self._clear_buffers()

        if self._pending_tokens:
            result.extend(self._pending_tokens)
            self._pending_tokens.clear()
            self._pending_str = ""
        return result

    def _parse_buffer(
        self, text: str
    ) -> tuple[list[Any] | None, list[ToolCallDiagnostic]]:
        if type(self._tool_manager) is ToolManager:
            outcome = self._tool_manager.parse_calls(text)
            return outcome.calls or None, outcome.diagnostics
        return self._tool_manager.get_calls(text), []

    def _stream_buffer_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        if type(self._tool_manager) is ToolManager:
            return self._tool_manager.stream_buffer_diagnostics(text)
        return []

    def _clear_buffers(self) -> None:
        self._buffer = StringIO()
        self._tag_buffer = ""
        self._inside_call = False
