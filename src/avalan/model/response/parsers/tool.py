"""Parser emitting events for detected tool calls."""

from ....entities import ToolCallDiagnostic, ToolCallToken, ToolFormat
from ....event import Event, EventType
from ....event.manager import EventManager
from ....tool.manager import ToolManager
from ....tool.parser import ToolCallParser

from dataclasses import dataclass
from io import StringIO
from time import perf_counter
from typing import Any, Iterable, cast


class _MarkdownFenceState:
    """Track markdown fence state while text streams in."""

    def __init__(self) -> None:
        self._open_character: str | None = None
        self._open_length = 0
        self._line_allows_fence = True
        self._line_fence_character: str | None = None
        self._line_fence_length = 0
        self._line_fence_action: str | None = None

    @property
    def is_open(self) -> bool:
        return self._open_character is not None

    def copy(self) -> "_MarkdownFenceState":
        state = _MarkdownFenceState()
        state._open_character = self._open_character
        state._open_length = self._open_length
        state._line_allows_fence = self._line_allows_fence
        state._line_fence_character = self._line_fence_character
        state._line_fence_length = self._line_fence_length
        state._line_fence_action = self._line_fence_action
        return state

    def push(self, text: str) -> None:
        for character in text:
            self.push_character(character)

    def push_character(self, character: str) -> None:
        if character in ("\n", "\r"):
            self._reset_line()
            return

        if not self._line_allows_fence:
            return

        if self._line_fence_character is None:
            if character in (" ", "\t"):
                return
            if character not in ("`", "~"):
                self._line_allows_fence = False
                return
            self._line_fence_character = character
            self._line_fence_length = 1
        elif character == self._line_fence_character:
            self._line_fence_length += 1
        else:
            self._line_allows_fence = False
            return

        if self._line_fence_length < 3:
            return
        if self._line_fence_action is None:
            self._apply_line_fence()
        elif (
            self._line_fence_action == "opened"
            and self._open_character == self._line_fence_character
        ):
            self._open_length = self._line_fence_length

    def _apply_line_fence(self) -> None:
        assert self._line_fence_character is not None
        if self._open_character is None:
            self._open_character = self._line_fence_character
            self._open_length = self._line_fence_length
            self._line_fence_action = "opened"
            return

        if (
            self._line_fence_character == self._open_character
            and self._line_fence_length >= self._open_length
        ):
            self._open_character = None
            self._open_length = 0
            self._line_fence_action = "closed"

    def _reset_line(self) -> None:
        self._line_allows_fence = True
        self._line_fence_character = None
        self._line_fence_length = 0
        self._line_fence_action = None


class _VisibleQuoteState:
    """Track visible same-line quote state."""

    def __init__(self) -> None:
        self._quote: str | None = None
        self._escaped = False
        self._previous_character: str | None = None
        self._pending_single_quote = False

    def copy(self) -> "_VisibleQuoteState":
        state = _VisibleQuoteState()
        state._quote = self._quote
        state._escaped = self._escaped
        state._previous_character = self._previous_character
        state._pending_single_quote = self._pending_single_quote
        return state

    def is_open_before(self, next_character: str | None) -> bool:
        if self._quote is not None:
            return True
        if not self._pending_single_quote:
            return False
        return next_character is None or not next_character.isalnum()

    def push_character(self, character: str) -> None:
        self._resolve_pending_single_quote(character)
        if character in ("\n", "\r"):
            self._quote = None
            self._escaped = False
            self._pending_single_quote = False
            self._previous_character = character
            return
        if self._escaped:
            self._escaped = False
            self._previous_character = character
            return
        if self._quote is not None:
            if character == "\\":
                self._escaped = True
            elif character == self._quote:
                self._quote = None
            self._previous_character = character
            return
        if character == '"':
            self._quote = character
        elif character == "'":
            if self._previous_character is None:
                self._quote = character
            elif self._previous_character.isalnum():
                self._pending_single_quote = True
            else:
                self._quote = character
        self._previous_character = character

    def _resolve_pending_single_quote(self, character: str) -> None:
        if not self._pending_single_quote:
            return
        self._pending_single_quote = False
        if not character.isalnum():
            self._quote = "'"


class _VisibleTextState:
    """Track visible text state used to classify tool markers."""

    def __init__(
        self,
        markdown: _MarkdownFenceState | None = None,
        quote: _VisibleQuoteState | None = None,
    ) -> None:
        self._markdown = markdown or _MarkdownFenceState()
        self._quote = quote or _VisibleQuoteState()

    @property
    def markdown_fence_is_open(self) -> bool:
        return self._markdown.is_open

    def copy(self) -> "_VisibleTextState":
        return _VisibleTextState(self._markdown.copy(), self._quote.copy())

    def push(self, text: str) -> None:
        for character in text:
            self.push_character(character)

    def push_character(self, character: str) -> None:
        self._markdown.push_character(character)
        self._quote.push_character(character)

    def executable_marker_indexes(
        self, text: str, marker_indexes: Iterable[int]
    ) -> list[int]:
        pending = set(marker_indexes)
        if not pending:
            return []

        indexes: list[int] = []
        state = self.copy()
        for index, character in enumerate(text):
            if index in pending and state._marker_is_executable(character):
                indexes.append(index)
            state.push_character(character)
        return indexes

    def _marker_is_executable(self, next_character: str | None) -> bool:
        return not self._markdown.is_open and not self._quote.is_open_before(
            next_character
        )


class _ToolQuoteState:
    """Track quoted text while scanning a tool-call buffer."""

    def __init__(self) -> None:
        self._quote: str | None = None
        self._escaped = False

    @property
    def is_open(self) -> bool:
        return self._quote is not None

    def push_character(self, character: str) -> None:
        if self._escaped:
            self._escaped = False
            return
        if self._quote is not None:
            if character == "\\":
                self._escaped = True
            elif character == self._quote:
                self._quote = None
            return
        if character in ("'", '"'):
            self._quote = character


@dataclass(frozen=True, slots=True)
class _ToolClose:
    scan_end: int
    tool_end: int
    suffix_start: int


class _ToolCallCloseState:
    """Incrementally detect closes in streamed tool-call text."""

    def __init__(self, tail_limit: int) -> None:
        self.active = False
        self._markdown = _MarkdownFenceState()
        self._pending_start_marker: str | None = None
        self._pending_xml_tag = False
        self._pending_xml_tag_text = ""
        self._position = 0
        self._quote = _ToolQuoteState()
        self._tail = ""
        self._tail_limit = tail_limit

    @property
    def has_open_segment(self) -> bool:
        return self.active or self._pending_start_marker is not None

    def push(
        self,
        parser: "ToolCallResponseParser",
        text: str,
        *,
        return_first: bool = True,
        token_start: int = 0,
    ) -> _ToolClose | None:
        matched_close: _ToolClose | None = None
        for character in text:
            close = self._push_character(parser, character)
            if close is None:
                continue
            if close.scan_end < token_start:
                continue
            if return_first:
                return close
            matched_close = close
        return matched_close

    def _push_character(
        self, parser: "ToolCallResponseParser", character: str
    ) -> _ToolClose | None:
        markdown_open = self._markdown.is_open
        self._resolve_pending_start_marker(character)
        self._tail = (self._tail + character)[-self._tail_limit :]

        close: _ToolClose | None = None
        if self._pending_xml_tag:
            close = self._close_pending_xml_tag(character)
        else:
            self._quote.push_character(character)

        if close is None and not self._quote.is_open and not markdown_open:
            close = self._close_at_tail(parser)
            if close is None:
                self._open_at_tail(parser)

        self._markdown.push_character(character)
        self._position += 1
        return close

    def _resolve_pending_start_marker(self, character: str) -> None:
        marker = self._pending_start_marker
        if marker is None:
            return
        self._pending_start_marker = None
        if marker == "<tool_call" and not (
            character == ">" or character == "/" or character.isspace()
        ):
            return
        self._open_marker(marker)

    def _close_pending_xml_tag(self, character: str) -> _ToolClose | None:
        self._pending_xml_tag_text += character
        tag_end = ToolCallParser._tag_end_index(self._pending_xml_tag_text, 0)
        if tag_end == -1:
            return None

        tag_text = self._pending_xml_tag_text[: tag_end + 1]
        self._pending_xml_tag = False
        self._pending_xml_tag_text = ""
        if tag_text.rstrip().endswith("/>"):
            self.active = False
            return _ToolClose(
                scan_end=self._position + 1,
                tool_end=self._position + 1,
                suffix_start=self._position + 1,
            )
        return None

    def _close_at_tail(
        self, parser: "ToolCallResponseParser"
    ) -> _ToolClose | None:
        if not self.active:
            return None

        for marker in parser._sorted_tool_marker_ends():
            if self._tail.endswith(marker):
                self.active = False
                close_index = self._position + 1
                return _ToolClose(
                    scan_end=close_index,
                    tool_end=close_index,
                    suffix_start=close_index,
                )

        for marker in parser._sorted_visible_tool_marker_ends():
            if self._tail.endswith(marker):
                self.active = False
                marker_start = self._position + 1 - len(marker)
                return _ToolClose(
                    scan_end=self._position + 1,
                    tool_end=marker_start,
                    suffix_start=marker_start,
                )

        return None

    def _open_at_tail(self, parser: "ToolCallResponseParser") -> None:
        for marker in parser._sorted_tool_marker_starts():
            if not self._tail.endswith(marker):
                continue
            if marker == "<tool_call":
                self._pending_start_marker = marker
            else:
                self._open_marker(marker)
            return

    def _open_marker(self, marker: str) -> None:
        self.active = True
        if marker not in ("<tool_call", "<tool "):
            return
        self._pending_xml_tag = True
        self._pending_xml_tag_text = marker


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
        self._pending_tool_visible_suffix = ""
        self._tool_close_state = self._new_tool_close_state()
        self._visible_state = _VisibleTextState()

    async def push(self, token_str: str) -> Iterable[Any]:
        buffer_value = self._buffer.getvalue()
        should_check = self._tool_manager.is_potential_tool_call(
            buffer_value, token_str
        )
        if (
            should_check
            and self._visible_state.markdown_fence_is_open
            and not self._has_executable_marker_in_visible_token(token_str)
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
                and not self._has_executable_marker_in_visible_token(token_str)
            ):
                status = ToolCallParser.ToolCallBufferStatus.NONE
                should_check = False
            pending_split = self._split_pending_candidate(candidate)
            if pending_split is not None:
                pending_prefix, tool_token, status = pending_split
                result.append(pending_prefix)
                self._append_visible_text(pending_prefix)
                candidate = tool_token
                self._pending_tokens.clear()
                self._pending_str = ""
                self._tool_buffer = StringIO()
            else:
                split_prefix, tool_token = self._split_visible_prefix(
                    token_str, status
                )
                if split_prefix is not None:
                    result.append(split_prefix)
                    self._append_visible_text(split_prefix)
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
                if self._inside_call:
                    self._reset_tool_close_state(self._tool_buffer.getvalue())
                if status is ToolCallParser.ToolCallBufferStatus.CLOSED:
                    terminal_status = status
            else:
                if self._pending_tokens:
                    result.extend(self._pending_tokens)
                    self._append_visible_text(self._pending_str)
                    self._pending_tokens.clear()
                    self._pending_str = ""
                    self._tool_buffer = StringIO()
                result.append(token_str)
                self._append_visible_text(token_str)
        else:
            tool_token = token_str
            current_tool_text = self._tool_buffer.getvalue()
            pending_visible_suffix = self._pending_tool_visible_suffix
            combined_tool_text = (
                current_tool_text + pending_visible_suffix + token_str
            )
            close = self._tool_close_state.push(
                self,
                token_str,
                token_start=(
                    len(current_tool_text) + len(pending_visible_suffix)
                ),
            )
            closed_split = self._tool_close_split(
                current_tool_text, pending_visible_suffix, token_str, close
            )
            if (
                closed_split is None
                and not issubclass(type(self._tool_manager), ToolManager)
                and self._tool_manager.tool_call_status(combined_tool_text)
                is ToolCallParser.ToolCallBufferStatus.CLOSED
            ):
                closed_split = self._split_closed_visible_suffix(
                    combined_tool_text
                )
            if closed_split is not None:
                tool_text, visible_suffix = closed_split
                uncommitted_tool_length = max(
                    0, len(tool_text) - len(current_tool_text)
                )
                tool_token = (pending_visible_suffix + token_str)[
                    :uncommitted_tool_length
                ]
                self._inside_call = False
                self._pending_tool_visible_suffix = ""
                terminal_status = ToolCallParser.ToolCallBufferStatus.CLOSED
            else:
                tool_token = pending_visible_suffix + token_str
                tool_token, self._pending_tool_visible_suffix = (
                    self._split_visible_tool_suffix_prefix(tool_token)
                )
            if tool_token:
                result.append(ToolCallToken(token=tool_token))
            if closed_split is not None:
                self._tool_buffer = StringIO()
                self._tool_buffer.write(tool_text)
            else:
                self._tool_buffer.write(tool_token)

        if not result and terminal_status is None:
            return result

        if not should_check:
            return result

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        if self._inside_call and terminal_status is None:
            if type(self._tool_manager) is ToolManager:
                return result
            buffer_text = self._parse_text()
            status = self._tool_manager.tool_call_status(buffer_text)
            if status is not ToolCallParser.ToolCallBufferStatus.CLOSED:
                return result
            self._inside_call = False
            terminal_status = status
        else:
            buffer_text = self._parse_text()

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
            self._append_visible_text(self._pending_str)
            self._pending_tokens.clear()
            self._pending_str = ""
            self._tool_buffer = StringIO()
        return result

    def _append_visible_text(self, text: str) -> None:
        if text:
            self._visible_state.push(text)

    def _has_executable_marker_in_visible_token(self, token_str: str) -> bool:
        return bool(
            self._visible_state.executable_marker_indexes(
                token_str, self._tool_marker_start_indexes(token_str)
            )
        )

    def _new_tool_close_state(self) -> _ToolCallCloseState:
        return _ToolCallCloseState(self._tool_marker_tail_limit())

    def _reset_tool_close_state(self, text: str = "") -> None:
        self._tool_close_state = self._new_tool_close_state()
        if text:
            self._tool_close_state.push(
                self,
                text,
                return_first=False,
                token_start=len(text) + 1,
            )

    @staticmethod
    def _tool_close_split(
        current_tool_text: str,
        pending_visible_suffix: str,
        token_str: str,
        close: _ToolClose | None,
    ) -> tuple[str, str] | None:
        if close is None:
            return None
        text = current_tool_text + pending_visible_suffix + token_str
        return text[: close.tool_end], text[close.suffix_start :]

    def _split_visible_tool_suffix_prefix(self, text: str) -> tuple[str, str]:
        markers = self._visible_tool_marker_ends()
        if not markers:
            return text, ""

        pending_length = 0
        for marker in markers:
            maximum = min(len(marker) - 1, len(text))
            for length in range(maximum, 0, -1):
                if marker.startswith(text[-length:]):
                    pending_length = max(pending_length, length)
                    break

        if not pending_length:
            return text, ""
        return text[:-pending_length], text[-pending_length:]

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

        marker_indexes = self._visible_marker_indexes(token_str, buffer_prefix)
        if not marker_indexes or marker_indexes[0] == 0:
            return None, token_str

        for index in marker_indexes:
            suffix = token_str[index:]
            suffix_status = self._tool_manager.tool_call_status(suffix)
            if suffix_status is not ToolCallParser.ToolCallBufferStatus.NONE:
                return token_str[:index], suffix

        return None, token_str

    def _visible_marker_indexes(
        self, token_str: str, buffer_prefix: str = ""
    ) -> list[int]:
        if buffer_prefix:
            return [
                index
                for index in self._tool_marker_start_indexes(token_str)
                if self._is_executable_tool_marker(
                    buffer_prefix + token_str, len(buffer_prefix) + index
                )
            ]
        return self._visible_state.executable_marker_indexes(
            token_str, self._tool_marker_start_indexes(token_str)
        )

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
        markers = self._tool_marker_starts()
        for index, character in enumerate(text):
            if character != "<":
                continue
            suffix = text[index:]
            if any(
                marker.startswith(suffix) or suffix.startswith(marker)
                for marker in markers
            ):
                return True
        return False

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
                self._tool_marker_can_start_at(text, index, suffix, marker)
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

    def _sorted_tool_marker_starts(self) -> tuple[str, ...]:
        return tuple(sorted(self._tool_marker_starts(), key=len, reverse=True))

    def _tool_marker_tail_limit(self) -> int:
        markers = (
            self._tool_marker_starts()
            + self._tool_marker_ends()
            + self._visible_tool_marker_ends()
        )
        return max(len(marker) for marker in markers) + 1

    def _split_closed_visible_suffix(self, text: str) -> tuple[str, str]:
        state = self._new_tool_close_state()
        close = state.push(self, text, return_first=False)
        if state.has_open_segment or close is None:
            return text, ""

        suffix = text[close.suffix_start :]
        if not suffix:
            return text, ""
        return text[: close.tool_end], suffix

    def _split_current_call_close(
        self, text: str, token_start: int
    ) -> tuple[str, str] | None:
        state = self._new_tool_close_state()
        close = state.push(self, text, token_start=token_start)
        if close is None:
            return None
        return text[: close.tool_end], text[close.suffix_start :]

    def _self_closing_tool_end_indexes(self, text: str) -> Iterable[int]:
        for _, close_index in self._self_closing_tool_close_spans(text):
            yield close_index

    def _self_closing_tool_close_spans(
        self, text: str
    ) -> Iterable[tuple[int, int]]:
        for marker in ("<tool_call", "<tool "):
            index = text.find(marker)
            while index != -1:
                if not ToolCallParser._tool_start_marker_boundary_is_valid(
                    text, index, marker
                ):
                    index = text.find(marker, index + 1)
                    continue
                tag_end = ToolCallParser._tag_end_index(
                    text, index + len(marker)
                )
                if tag_end == -1:
                    break
                if text[index : tag_end + 1].rstrip().endswith("/>"):
                    yield index, tag_end + 1
                index = text.find(marker, index + 1)

    def _self_closing_tool_close_span_at(
        self, text: str, index: int
    ) -> tuple[int, int] | None:
        marker = self._tool_start_marker_at(
            text, index, ("<tool_call", "<tool ")
        )
        if marker is None:
            return None

        tag_end = ToolCallParser._tag_end_index(text, index)
        if tag_end == -1:
            return None

        if not text[index : tag_end + 1].rstrip().endswith("/>"):
            return None

        return index, tag_end + 1

    @staticmethod
    def _tool_marker_can_start_at(
        text: str, index: int, suffix: str, marker: str
    ) -> bool:
        if marker.startswith(suffix):
            return True
        if not suffix.startswith(marker):
            return False
        return ToolCallParser._tool_start_marker_boundary_is_valid(
            text, index, marker
        )

    def _tool_start_marker_at(
        self,
        text: str,
        index: int,
        markers: Iterable[str] | None = None,
    ) -> str | None:
        marker_candidates = markers or self._tool_marker_starts()
        for marker in sorted(marker_candidates, key=len, reverse=True):
            if not text.startswith(marker, index):
                continue
            boundary_index = index + len(marker)
            if boundary_index == len(text):
                return marker
            if ToolCallParser._tool_start_marker_boundary_is_valid(
                text, index, marker
            ):
                return marker
        return None

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

    def _sorted_tool_marker_ends(self) -> tuple[str, ...]:
        return tuple(sorted(self._tool_marker_ends(), key=len, reverse=True))

    def _visible_tool_marker_ends(self) -> tuple[str, ...]:
        tool_format = getattr(self._tool_manager, "tool_format", None)
        if tool_format is ToolFormat.HARMONY:
            return ("<|channel|>final<|message|>",)
        return ()

    def _sorted_visible_tool_marker_ends(self) -> tuple[str, ...]:
        return tuple(
            sorted(self._visible_tool_marker_ends(), key=len, reverse=True)
        )

    @staticmethod
    def _append_visible_suffix(
        items: list[Any], visible_suffix: str
    ) -> list[Any]:
        if visible_suffix:
            items.append(visible_suffix)
        return items

    def _parse_text(self) -> str:
        tool_text = self._tool_buffer.getvalue()
        if tool_text or self._pending_tool_visible_suffix:
            return tool_text + self._pending_tool_visible_suffix
        return self._buffer.getvalue()

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
        self._pending_tool_visible_suffix = ""
        self._reset_tool_close_state()
        self._visible_state = _VisibleTextState()
        if visible_text:
            self._buffer.write(visible_text)
            self._append_visible_text(visible_text)
