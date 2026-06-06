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
        self._tool_buffer = StringIO()
        self._tag_buffer = ""
        self._inside_call = False
        self._pending_tokens: list[str] = []
        self._pending_str = ""

    async def push(self, token_str: str) -> Iterable[Any]:
        buffer_value = self._buffer.getvalue()
        should_check = self._tool_manager.is_potential_tool_call(
            buffer_value, token_str
        )
        if (
            should_check
            and self._markdown_fence_is_open(buffer_value)
            and not self._has_unfenced_tool_marker(buffer_value, token_str)
        ):
            should_check = False

        self._buffer.write(token_str)
        self._tag_buffer += token_str
        if len(self._tag_buffer) > 64:
            self._tag_buffer = self._tag_buffer[-64:]

        result: list[Any] = []
        terminal_status: ToolCallParser.ToolCallBufferStatus | None = None
        visible_suffix = ""

        if not self._inside_call:
            candidate = self._pending_str + token_str
            status = self._tool_manager.tool_call_status(candidate)
            if (
                not self._pending_tokens
                and self._has_tool_marker(token_str)
                and not self._has_executable_tool_marker(
                    buffer_value, token_str
                )
            ):
                status = ToolCallParser.ToolCallBufferStatus.NONE
                should_check = False
            pending_split = self._split_pending_candidate(candidate)
            if pending_split is not None:
                pending_prefix, tool_token, status = pending_split
                result.append(pending_prefix)
                candidate = tool_token
                self._pending_tokens.clear()
                self._pending_str = ""
                self._tool_buffer = StringIO()
            else:
                split_prefix, tool_token = self._split_visible_prefix(
                    token_str, status, buffer_value
                )
                if split_prefix is not None:
                    result.append(split_prefix)
                    candidate = tool_token
                    status = self._tool_manager.tool_call_status(candidate)
            if status is ToolCallParser.ToolCallBufferStatus.PREFIX:
                self._pending_tokens.append(tool_token)
                self._pending_str = candidate
                self._tool_buffer.write(tool_token)
                return result
            if status in (
                ToolCallParser.ToolCallBufferStatus.OPEN,
                ToolCallParser.ToolCallBufferStatus.CLOSED,
            ):
                if status is ToolCallParser.ToolCallBufferStatus.CLOSED:
                    tool_token, visible_suffix = (
                        self._split_closed_visible_suffix(tool_token)
                    )
                self._pending_tokens.append(tool_token)
                self._tool_buffer.write(tool_token)
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
                    self._tool_buffer = StringIO()
                result.append(token_str)
        else:
            tool_token = token_str
            current_tool_text = self._tool_buffer.getvalue()
            combined_tool_text = current_tool_text + token_str
            closed_split = self._split_current_call_close(
                combined_tool_text, len(current_tool_text)
            )
            if (
                closed_split is None
                and self._tool_manager.tool_call_status(combined_tool_text)
                is ToolCallParser.ToolCallBufferStatus.CLOSED
            ):
                closed_split = self._split_closed_visible_suffix(
                    combined_tool_text
                )
            if closed_split is not None:
                tool_text, visible_suffix = closed_split
                token_length = max(0, len(tool_text) - len(current_tool_text))
                tool_token = token_str[:token_length]
                self._inside_call = False
                terminal_status = ToolCallParser.ToolCallBufferStatus.CLOSED
            if tool_token:
                result.append(ToolCallToken(token=tool_token))
            self._tool_buffer.write(tool_token)

        if not result and terminal_status is None:
            return result

        if not should_check:
            return result

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        buffer_text = self._parse_text()
        if self._inside_call and terminal_status is None:
            status = self._tool_manager.tool_call_status(buffer_text)
            if status is not ToolCallParser.ToolCallBufferStatus.CLOSED:
                return result
            self._inside_call = False
            terminal_status = status

        parse_text = self._closed_parse_text(buffer_text, visible_suffix)
        calls, diagnostics = self._parse_buffer(parse_text)
        if calls:
            events = await self._events_for_parse(calls, diagnostics)
            return await self._finish_closed_segment(
                result + events, visible_suffix
            )

        if terminal_status is ToolCallParser.ToolCallBufferStatus.CLOSED:
            if diagnostics:
                stream_diagnostics = self._stream_buffer_diagnostics(
                    parse_text
                )
                if stream_diagnostics:
                    diagnostics = stream_diagnostics
            else:
                diagnostics = self._stream_buffer_diagnostics(parse_text)
            if diagnostics:
                event = await self._diagnostic_event(diagnostics)
                return await self._finish_closed_segment(
                    result + [event], visible_suffix
                )

            return await self._finish_closed_segment(result, visible_suffix)

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

    async def _events_for_parse(
        self,
        calls: list[Any],
        diagnostics: list[ToolCallDiagnostic],
    ) -> list[Event]:
        events: list[Event] = []
        if diagnostics:
            events.append(await self._diagnostic_event(diagnostics))
        events.append(
            Event(
                type=EventType.TOOL_PROCESS,
                payload=cast(Any, calls),
                started=perf_counter(),
            )
        )
        return events

    async def _finish_closed_segment(
        self, items: list[Any], visible_suffix: str
    ) -> list[Any]:
        if not visible_suffix or not self._has_unfenced_tool_marker(
            "", visible_suffix
        ):
            self._clear_buffers(visible_suffix)
            return self._append_visible_suffix(items, visible_suffix)

        self._clear_buffers()
        suffix_items = await self.push(visible_suffix)
        return items + list(suffix_items)

    async def flush(self) -> Iterable[Any]:
        result: list[Any] = []
        if self._inside_call:
            buffer_text = self._parse_text()
            calls: list[Any] | None = None
            diagnostics: list[ToolCallDiagnostic] = []
            if self._tool_manager.tool_format is ToolFormat.HARMONY:
                calls, diagnostics = self._parse_buffer(
                    f"{buffer_text}<|call|>"
                )
            if calls:
                if self._event_manager:
                    await self._event_manager.trigger(
                        Event(type=EventType.TOOL_DETECT)
                    )

                events = await self._events_for_parse(calls, diagnostics)
                self._clear_buffers()
                result.extend(events)
            else:
                diagnostics = self._stream_buffer_diagnostics(buffer_text)
                if diagnostics:
                    result.append(await self._diagnostic_event(diagnostics))
                    self._clear_buffers()
        elif self._pending_tokens:
            diagnostics = self._stream_buffer_diagnostics(self._pending_str)
            if diagnostics:
                result.append(await self._diagnostic_event(diagnostics))
                self._clear_buffers()
                return result

        if self._pending_tokens:
            result.extend(self._pending_tokens)
            self._pending_tokens.clear()
            self._pending_str = ""
            self._tool_buffer = StringIO()
        return result

    def _split_pending_candidate(
        self, candidate: str
    ) -> tuple[str, str, ToolCallParser.ToolCallBufferStatus] | None:
        if not self._pending_tokens:
            return None

        for index in self._tool_marker_start_indexes(candidate):
            if index == 0:
                return None
            suffix = candidate[index:]
            suffix_status = self._tool_manager.tool_call_status(suffix)
            if suffix_status is not ToolCallParser.ToolCallBufferStatus.NONE:
                return candidate[:index], suffix, suffix_status

        return None

    def _split_visible_prefix(
        self,
        token_str: str,
        status: ToolCallParser.ToolCallBufferStatus,
        buffer_prefix: str = "",
    ) -> tuple[str | None, str]:
        if (
            self._pending_tokens
            or status is ToolCallParser.ToolCallBufferStatus.PREFIX
            or "<" not in token_str
        ):
            return None, token_str

        marker_indexes = [
            index
            for index in self._tool_marker_start_indexes(token_str)
            if self._is_executable_tool_marker(
                buffer_prefix + token_str, len(buffer_prefix) + index
            )
        ]
        if not marker_indexes or marker_indexes[0] == 0:
            return None, token_str

        for index in marker_indexes:
            suffix = token_str[index:]
            suffix_status = self._tool_manager.tool_call_status(suffix)
            if suffix_status is not ToolCallParser.ToolCallBufferStatus.NONE:
                return token_str[:index], suffix

        return None, token_str

    def _has_unfenced_tool_marker(
        self, buffer_prefix: str, token_str: str
    ) -> bool:
        return self._has_executable_tool_marker(buffer_prefix, token_str)

    def _has_executable_tool_marker(
        self, buffer_prefix: str, token_str: str
    ) -> bool:
        text = buffer_prefix + token_str
        return any(
            self._is_executable_tool_marker(text, len(buffer_prefix) + index)
            for index in self._tool_marker_start_indexes(token_str)
        )

    def _has_tool_marker(self, text: str) -> bool:
        return any(True for _ in self._tool_marker_start_indexes(text))

    @staticmethod
    def _markdown_fence_is_open(text: str) -> bool:
        open_fence: tuple[str, int] | None = None
        for line in text.splitlines():
            stripped = line.lstrip(" \t")
            if not stripped.startswith(("```", "~~~")):
                continue
            character = stripped[0]
            length = 0
            for current in stripped:
                if current != character:
                    break
                length += 1
            if open_fence is None:
                open_fence = (character, length)
            elif character == open_fence[0] and length >= open_fence[1]:
                open_fence = None
        return open_fence is not None

    @classmethod
    def _is_executable_tool_marker(cls, text: str, index: int) -> bool:
        return not (
            cls._markdown_fence_is_open(text[:index])
            or cls._index_is_inside_visible_quote(text, index)
        )

    @staticmethod
    def _index_is_inside_visible_quote(text: str, index: int) -> bool:
        quote: str | None = None
        escaped = False
        for position, character in enumerate(text[:index]):
            if escaped:
                escaped = False
                continue
            if character in ("\n", "\r"):
                quote = None
                continue
            if quote is not None:
                if character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character == '"':
                quote = character
                continue
            if character == "'" and ToolCallParser._is_quote_delimiter(
                text, position
            ):
                quote = character
        return quote is not None

    def _tool_marker_start_indexes(self, text: str) -> Iterable[int]:
        markers = self._tool_marker_starts()
        for index, character in enumerate(text):
            if character != "<":
                continue
            suffix = text[index:]
            if any(
                marker.startswith(suffix) or suffix.startswith(marker)
                for marker in markers
            ):
                yield index

    def _tool_marker_starts(self) -> tuple[str, ...]:
        markers = ["<tool_call", "<tool ", "<tool>"]
        tool_format = getattr(self._tool_manager, "tool_format", None)
        if tool_format is ToolFormat.HARMONY:
            markers.extend(
                [
                    "<|channel|>commentary",
                    "<|start|>assistant<|channel|>commentary",
                    "<|channel|>analysis",
                    "<|start|>assistant<|channel|>analysis",
                ]
            )
        if tool_format is ToolFormat.DSML:
            markers.extend(
                [
                    "<｜DSML｜tool_calls",
                    "<DSML｜tool_calls",
                    "<tool_calls",
                ]
            )
        return tuple(markers)

    def _split_closed_visible_suffix(self, text: str) -> tuple[str, str]:
        active = False
        close_index = 0
        escaped = False
        index = 0
        quote: str | None = None
        while index < len(text):
            character = text[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if quote is not None:
                if character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if character in ("'", '"'):
                quote = character
                index += 1
                continue

            self_closing_span = self._self_closing_tool_close_span_at(
                text, index
            )
            if (
                self_closing_span is not None
                and not self._markdown_fence_is_open(text[:index])
            ):
                active = False
                close_index = self_closing_span[1]
                index = close_index
                continue

            start_marker = self._tool_marker_at(
                text, index, self._tool_marker_starts()
            )
            if start_marker is not None and not self._markdown_fence_is_open(
                text[:index]
            ):
                active = True
                index += len(start_marker)
                continue

            end_marker = self._tool_marker_at(
                text, index, self._tool_marker_ends()
            )
            if (
                end_marker is not None
                and active
                and not self._markdown_fence_is_open(text[:index])
            ):
                active = False
                close_index = index + len(end_marker)
                index = close_index
                continue

            visible_end_marker = self._tool_marker_at(
                text, index, self._visible_tool_marker_ends()
            )
            if (
                visible_end_marker is not None
                and active
                and not self._markdown_fence_is_open(text[:index])
            ):
                return text[:index], text[index:]

            index += 1

        if active or not close_index:
            return text, ""

        suffix = text[close_index:]
        if not suffix:
            return text, ""
        return text[:close_index], suffix

    def _split_current_call_close(
        self, text: str, token_start: int
    ) -> tuple[str, str] | None:
        active = False
        escaped = False
        index = 0
        quote: str | None = None
        while index < len(text):
            character = text[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if quote is not None:
                if character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if character in ("'", '"'):
                quote = character
                index += 1
                continue

            if self._markdown_fence_is_open(text[:index]):
                index += 1
                continue

            self_closing_span = self._self_closing_tool_close_span_at(
                text, index
            )
            if self_closing_span is not None:
                active = False
                close_index = self_closing_span[1]
                if close_index >= token_start:
                    return text[:close_index], text[close_index:]
                index = close_index
                continue

            start_marker = self._tool_marker_at(
                text, index, self._tool_marker_starts()
            )
            if start_marker is not None:
                active = True
                index += len(start_marker)
                continue

            end_marker = self._tool_marker_at(
                text, index, self._tool_marker_ends()
            )
            if end_marker is not None and active:
                active = False
                close_index = index + len(end_marker)
                if close_index >= token_start:
                    return text[:close_index], text[close_index:]
                index = close_index
                continue

            visible_end_marker = self._tool_marker_at(
                text, index, self._visible_tool_marker_ends()
            )
            if visible_end_marker is not None and active:
                active = False
                if index >= token_start:
                    return text[:index], text[index:]
                index += len(visible_end_marker)
                continue

            index += 1

        return None

    def _self_closing_tool_end_indexes(self, text: str) -> Iterable[int]:
        for _, close_index in self._self_closing_tool_close_spans(text):
            yield close_index

    def _self_closing_tool_close_spans(
        self, text: str
    ) -> Iterable[tuple[int, int]]:
        for marker in ("<tool_call", "<tool "):
            index = text.find(marker)
            while index != -1:
                tag_end = ToolCallParser._tag_end_index(
                    text, index + len(marker)
                )
                if tag_end == -1:
                    break
                if text[index : tag_end + 1].rstrip().endswith("/>"):
                    yield index, tag_end + 1
                index = text.find(marker, index + 1)

    @staticmethod
    def _self_closing_tool_close_span_at(
        text: str, index: int
    ) -> tuple[int, int] | None:
        if not text.startswith(("<tool_call", "<tool "), index):
            return None

        tag_end = ToolCallParser._tag_end_index(text, index)
        if tag_end == -1:
            return None

        if not text[index : tag_end + 1].rstrip().endswith("/>"):
            return None

        return index, tag_end + 1

    @staticmethod
    def _tool_marker_at(
        text: str, index: int, markers: Iterable[str]
    ) -> str | None:
        for marker in sorted(markers, key=len, reverse=True):
            if text.startswith(marker, index):
                return marker
        return None

    @staticmethod
    def _marker_indexes(text: str, marker: str) -> Iterable[int]:
        index = text.find(marker)
        while index != -1:
            yield index
            index = text.find(marker, index + 1)

    def _tool_marker_ends(self) -> tuple[str, ...]:
        markers = ["</tool_call>", "</tool>"]
        tool_format = getattr(self._tool_manager, "tool_format", None)
        if tool_format is ToolFormat.HARMONY:
            markers.append("<|call|>")
        if tool_format is ToolFormat.DSML:
            markers.extend(
                [
                    "</｜DSML｜tool_calls>",
                    "</DSML｜tool_calls>",
                    "</tool_calls>",
                ]
            )
        return tuple(markers)

    def _visible_tool_marker_ends(self) -> tuple[str, ...]:
        tool_format = getattr(self._tool_manager, "tool_format", None)
        if tool_format is ToolFormat.HARMONY:
            return ("<|channel|>final<|message|>",)
        return ()

    @staticmethod
    def _append_visible_suffix(
        items: list[Any], visible_suffix: str
    ) -> list[Any]:
        if visible_suffix:
            items.append(visible_suffix)
        return items

    def _parse_text(self) -> str:
        tool_text = self._tool_buffer.getvalue()
        return tool_text or self._buffer.getvalue()

    def _closed_parse_text(self, text: str, visible_suffix: str) -> str:
        if (
            getattr(self._tool_manager, "tool_format", None)
            is ToolFormat.HARMONY
            and visible_suffix.startswith("<|channel|>final<|message|>")
            and "<|call|>" not in text
        ):
            return text + "<|call|>"
        return text

    def _parse_buffer(
        self, text: str
    ) -> tuple[list[Any] | None, list[ToolCallDiagnostic]]:
        if issubclass(type(self._tool_manager), ToolManager):
            outcome = self._tool_manager.parse_calls(text)
            return outcome.calls or None, outcome.diagnostics
        return self._tool_manager.get_calls(text), []

    def _stream_buffer_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        if issubclass(type(self._tool_manager), ToolManager):
            return self._tool_manager.stream_buffer_diagnostics(text)
        return []

    def _clear_buffers(self, visible_text: str = "") -> None:
        self._buffer = StringIO()
        self._tool_buffer = StringIO()
        self._tag_buffer = visible_text[-64:]
        self._inside_call = False
        self._pending_tokens.clear()
        self._pending_str = ""
        if visible_text:
            self._buffer.write(visible_text)
