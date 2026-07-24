"""Parser emitting canonical events for detected tool calls."""

from ....entities import ToolCallDiagnostic, ToolFormat
from ....event.manager import EventManager
from ....tool.dsml import DsmlTools
from ....tool.parser import ToolCallParser
from ...capability import ModelCapabilityCatalog
from ...stream import (
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamVisibility,
)

from dataclasses import dataclass
from io import StringIO
from json import JSONDecodeError, JSONDecoder, dumps
from typing import Any, Iterable, TypeAlias

ToolCallResponseParserOutput: TypeAlias = StreamProviderEvent
_ToolCallResponseParserItem: TypeAlias = ToolCallResponseParserOutput


class _IncrementalTextBuffer:
    """Collect streamed text while tracking its length."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def getvalue(self) -> str:
        return "".join(self._chunks)

    def write(self, text: str) -> int:
        self._chunks.append(text)
        self._length += len(text)
        return len(text)


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

    def marker_is_executable_before(self, next_character: str | None) -> bool:
        return self._marker_is_executable(next_character)

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
        self,
        capability: ModelCapabilityCatalog,
        event_manager: EventManager | None,
    ) -> None:
        self._capability = capability
        self._event_manager = event_manager
        self._buffer = StringIO()
        self._tool_buffer = _IncrementalTextBuffer()
        self._tag_buffer = ""
        self._inside_call = False
        self._pending_tokens: list[str] = []
        self._pending_str = ""
        self._pending_tool_visible_suffix = ""
        self._tool_close_state = self._new_tool_close_state()
        self._tool_call_id: str | None = None
        self._tool_call_index = 0
        self._tool_argument_emitted_until = 0
        self._tool_argument_text_by_id: dict[str, str] = {}
        self._visible_state = _VisibleTextState()

    @property
    def canonicalizes_answer_deltas(self) -> bool:
        return True

    async def push(
        self, token_str: str
    ) -> Iterable[_ToolCallResponseParserItem]:
        buffer_value = self._buffer.getvalue()
        should_check = self._capability.is_potential_tool_call(
            buffer_value, token_str
        )
        if (
            should_check
            and not self._visible_state.marker_is_executable_before(
                token_str[0] if token_str else None
            )
            and not self._has_executable_marker_in_visible_token(token_str)
        ):
            should_check = False

        self._buffer.write(token_str)
        self._tag_buffer += token_str
        if len(self._tag_buffer) > 64:
            self._tag_buffer = self._tag_buffer[-64:]

        result: list[_ToolCallResponseParserItem] = []
        terminal_status: ToolCallParser.ToolCallBufferStatus | None = None
        visible_suffix = ""

        if not self._inside_call:
            candidate = self._pending_str + token_str
            status = self._capability.tool_call_status(candidate)
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
                self._append_visible_output(result, pending_prefix)
                self._append_visible_text(pending_prefix)
                candidate = tool_token
                self._pending_tokens.clear()
                self._pending_str = ""
                self._tool_buffer = _IncrementalTextBuffer()
            else:
                split_prefix, tool_token = self._split_visible_prefix(
                    token_str, status
                )
                if split_prefix is not None:
                    self._append_visible_output(result, split_prefix)
                    self._append_visible_text(split_prefix)
                    candidate = tool_token
                    status = self._capability.tool_call_status(candidate)
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
                for pending_token in self._pending_tokens:
                    event = self._tool_argument_delta(pending_token)
                    if event is not None:
                        result.append(event)
                self._pending_tokens.clear()
                self._pending_str = ""
                self._inside_call = (
                    status is ToolCallParser.ToolCallBufferStatus.OPEN
                )
                if self._inside_call:
                    self._reset_tool_close_state(candidate)
                if status is ToolCallParser.ToolCallBufferStatus.CLOSED:
                    terminal_status = status
            else:
                if self._pending_tokens:
                    self._extend_visible_output(result, self._pending_tokens)
                    self._append_visible_text(self._pending_str)
                    self._pending_tokens.clear()
                    self._pending_str = ""
                    self._tool_buffer = _IncrementalTextBuffer()
                self._append_visible_output(result, token_str)
                self._append_visible_text(token_str)
        else:
            tool_token = token_str
            current_tool_length = len(self._tool_buffer)
            pending_visible_suffix = self._pending_tool_visible_suffix
            close = self._tool_close_state.push(
                self,
                token_str,
                token_start=(
                    current_tool_length + len(pending_visible_suffix)
                ),
            )
            closed_split = (
                self._tool_close_split(
                    self._tool_buffer.getvalue(),
                    pending_visible_suffix,
                    token_str,
                    close,
                )
                if close is not None
                else None
            )
            if closed_split is not None:
                tool_text, visible_suffix = closed_split
                uncommitted_tool_length = max(
                    0, len(tool_text) - current_tool_length
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
                event = self._tool_argument_delta(tool_token)
                if event is not None:
                    result.append(event)
            if closed_split is not None:
                self._tool_buffer = _IncrementalTextBuffer()
                self._tool_buffer.write(tool_text)
            else:
                self._tool_buffer.write(tool_token)

        if not result and terminal_status is None:
            return result

        if not should_check:
            return result

        if self._inside_call and terminal_status is None:
            return result
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
                diagnostic_events = await self._diagnostic_events(diagnostics)
                return await self._finish_closed_segment(
                    result + diagnostic_events, visible_suffix
                )

            return await self._finish_closed_segment(result, visible_suffix)

        return result

    async def _diagnostic_events(
        self, diagnostics: list[ToolCallDiagnostic]
    ) -> list[_ToolCallResponseParserItem]:
        return [
            await self._diagnostic_event(group, tool_call_id=tool_call_id)
            for tool_call_id, group in self._diagnostic_groups(diagnostics)
        ]

    async def _diagnostic_event(
        self,
        diagnostics: list[ToolCallDiagnostic],
        *,
        tool_call_id: str | None = None,
    ) -> ToolCallResponseParserOutput:
        tool_call_id = (
            tool_call_id
            or self._diagnostic_tool_call_id(diagnostics)
            or self._ensure_tool_call_id()
        )
        message = "Tool call could not be decoded."
        provider_event = StreamProviderEvent(
            kind=StreamItemKind.STREAM_DIAGNOSTIC,
            data={
                "code": "tool_call.malformed",
                "message": message,
                "stage": "parse",
                "tool_call_id": tool_call_id,
                "diagnostics": [
                    self._diagnostic_data(diagnostic)
                    for diagnostic in diagnostics[:8]
                ],
                "diagnostic_count": len(diagnostics),
            },
            correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
            visibility=StreamVisibility.DIAGNOSTIC,
        )
        return provider_event

    async def _events_for_parse(
        self,
        calls: list[Any],
        diagnostics: list[ToolCallDiagnostic],
    ) -> list[_ToolCallResponseParserItem]:
        events: list[_ToolCallResponseParserItem] = []
        for tool_call_id, group in self._diagnostic_groups(diagnostics):
            events.append(
                await self._diagnostic_event(group, tool_call_id=tool_call_id)
            )
        first_call = True
        for call in calls:
            tool_call_id = self._tool_call_id_for_call(
                call, first_call=first_call
            )
            first_call = False
            correlation = StreamItemCorrelation(tool_call_id=tool_call_id)
            argument_text = self._remaining_tool_argument_delta_text(
                tool_call_id, self._tool_argument_delta_text(call)
            )
            if argument_text:
                events.append(
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        text_delta=argument_text,
                        correlation=correlation,
                    )
                )
            events.extend(
                [
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                        data=self._tool_call_data(call),
                        correlation=correlation,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_DONE,
                        correlation=correlation,
                    ),
                ]
            )
        return events

    async def _finish_closed_segment(
        self,
        items: list[_ToolCallResponseParserItem],
        visible_suffix: str,
    ) -> list[_ToolCallResponseParserItem]:
        visible_suffix = self._visible_tool_suffix_text(visible_suffix)
        if not visible_suffix or not self._has_unfenced_tool_marker(
            "", visible_suffix
        ):
            self._clear_buffers(visible_suffix)
            return self._append_visible_suffix(items, visible_suffix)

        self._clear_buffers()
        suffix_items = await self.push(visible_suffix)
        return items + list(suffix_items)

    async def flush(self) -> Iterable[_ToolCallResponseParserItem]:
        result: list[_ToolCallResponseParserItem] = []
        if self._inside_call:
            buffer_text = self._parse_text()
            calls: list[Any] | None = None
            diagnostics: list[ToolCallDiagnostic] = []
            if self._capability.tool_format is ToolFormat.HARMONY:
                calls, diagnostics = self._parse_buffer(
                    f"{buffer_text}<|call|>"
                )
            if calls:
                events = await self._events_for_parse(calls, diagnostics)
                self._clear_buffers()
                result.extend(events)
            else:
                diagnostics = self._stream_buffer_diagnostics(buffer_text)
                if diagnostics:
                    result.extend(await self._diagnostic_events(diagnostics))
                    self._clear_buffers()
        elif self._pending_tokens:
            diagnostics = self._stream_buffer_diagnostics(self._pending_str)
            if diagnostics:
                result.extend(await self._diagnostic_events(diagnostics))
                self._clear_buffers()
                return result

        if self._pending_tokens:
            self._extend_visible_output(result, self._pending_tokens)
            self._append_visible_text(self._pending_str)
            self._pending_tokens.clear()
            self._pending_str = ""
            self._tool_buffer = _IncrementalTextBuffer()
        return result

    def _append_visible_output(
        self,
        items: list[_ToolCallResponseParserItem],
        text: str,
    ) -> None:
        if not text:
            return
        items.append(
            StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta=text,
            )
        )

    def _extend_visible_output(
        self,
        items: list[_ToolCallResponseParserItem],
        texts: list[str],
    ) -> None:
        for text in texts:
            self._append_visible_output(items, text)

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

    def _new_tool_call_id(self) -> str:
        self._tool_call_index += 1
        return f"parser-tool-call-{self._tool_call_index}"

    def _ensure_tool_call_id(self) -> str:
        if self._tool_call_id is None:
            self._tool_call_id = self._new_tool_call_id()
        return self._tool_call_id

    def _tool_call_id_for_call(
        self,
        call: Any,
        *,
        first_call: bool,
    ) -> str:
        if first_call and self._tool_call_id is not None:
            return self._tool_call_id
        call_id = getattr(call, "id", None)
        if call_id is not None:
            return str(call_id)
        return (
            self._ensure_tool_call_id()
            if first_call
            else self._new_tool_call_id()
        )

    @staticmethod
    def _diagnostic_tool_call_id(
        diagnostics: list[ToolCallDiagnostic],
    ) -> str | None:
        for diagnostic in diagnostics:
            if diagnostic.call_id is not None:
                return str(diagnostic.call_id)
        return None

    def _diagnostic_groups(
        self,
        diagnostics: list[ToolCallDiagnostic],
    ) -> list[tuple[str | None, list[ToolCallDiagnostic]]]:
        groups: dict[str, list[ToolCallDiagnostic]] = {}
        uncorrelated: list[tuple[str | None, list[ToolCallDiagnostic]]] = []
        for diagnostic in diagnostics:
            if diagnostic.call_id is None:
                uncorrelated.append((self._new_tool_call_id(), [diagnostic]))
                continue
            groups.setdefault(str(diagnostic.call_id), []).append(diagnostic)
        return [
            *((tool_call_id, group) for tool_call_id, group in groups.items()),
            *uncorrelated,
        ]

    def _tool_argument_delta(
        self, token: str
    ) -> ToolCallResponseParserOutput | None:
        if getattr(self._capability, "tool_format", None) is ToolFormat.DSML:
            return self._dsml_tool_argument_delta(token)

        if "}" not in token:
            return None

        text = self._tool_buffer.getvalue() + token
        argument_text = self._stream_tool_argument_text(text)
        if argument_text is None:
            return None
        tool_call_id = self._stream_tool_call_id(text)
        delta = self._remaining_tool_argument_delta_text(
            tool_call_id, argument_text
        )
        if not delta:
            return None
        return StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            text_delta=delta,
            correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        )

    def _dsml_tool_argument_delta(
        self, token: str
    ) -> ToolCallResponseParserOutput | None:
        text = self._tool_buffer.getvalue() + token
        deltas, emitted_until = DsmlTools.stream_argument_deltas(
            text, self._tool_argument_emitted_until
        )
        self._tool_argument_emitted_until = emitted_until
        delta = "".join(deltas)
        if not delta:
            return None
        tool_call_id = self._stream_tool_call_id(text)
        self._tool_argument_text_by_id[tool_call_id] = (
            self._tool_argument_text_by_id.get(tool_call_id, "") + delta
        )
        return StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            text_delta=delta,
            correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        )

    @staticmethod
    def _tool_argument_delta_text(call: Any) -> str:
        arguments = getattr(call, "arguments", None)
        if not isinstance(arguments, dict):
            return ""
        return dumps(arguments, separators=(",", ":"))

    def _remaining_tool_argument_delta_text(
        self,
        tool_call_id: str,
        argument_text: str,
    ) -> str:
        if not argument_text:
            return ""
        emitted_text = self._tool_argument_text_by_id.get(tool_call_id, "")
        if not emitted_text:
            self._tool_argument_text_by_id[tool_call_id] = argument_text
            return argument_text
        if not argument_text.startswith(emitted_text):
            return ""
        self._tool_argument_text_by_id[tool_call_id] = argument_text
        return argument_text[len(emitted_text) :]

    def _stream_tool_call_id(self, text: str) -> str:
        if self._tool_call_id is not None:
            return self._tool_call_id
        explicit_id = self._current_explicit_tool_call_id(text)
        if explicit_id is not None:
            self._tool_call_id = explicit_id
            return explicit_id
        return self._ensure_tool_call_id()

    def _current_explicit_tool_call_id(self, text: str) -> str | None:
        payload = self._stream_tool_payload(text)
        if payload is None:
            return None
        value = self._decode_json_object(payload)
        if value is None:
            return None
        call_id = value.get("id")
        return str(call_id) if isinstance(call_id, str) and call_id else None

    def _stream_tool_argument_text(self, text: str) -> str | None:
        payload = self._stream_tool_payload(text)
        if payload is None:
            return None
        payload_data = self._decode_json_object(payload)
        if payload_data is None:
            return None
        if "arguments" not in payload_data:
            return dumps(payload_data, separators=(",", ":"))
        name = payload_data.get("name")
        arguments = payload_data.get("arguments")
        if not ToolCallParser._is_valid_tool_name(name):
            return None
        if not isinstance(arguments, dict):
            return None
        return dumps(arguments, separators=(",", ":"))

    def _stream_tool_payload(self, text: str) -> str | None:
        if (
            getattr(self._capability, "tool_format", None)
            is ToolFormat.HARMONY
        ):
            marker = "<|message|>"
            message_start = text.find(marker)
            if message_start == -1:
                return None
            payload = text[message_start + len(marker) :]
            for end_marker in ("<|call|>", "<|channel|>final<|message|>"):
                marker_index = payload.find(end_marker)
                if marker_index != -1:
                    payload = payload[:marker_index]
            return payload

        start_index = self._first_stream_tool_start_index(text)
        if start_index is None:
            return None
        opening_marker = self._tool_start_marker_at(text, start_index)
        if opening_marker is None:
            return None
        if opening_marker == "<tool>":
            payload_start = start_index + len(opening_marker)
        else:
            tag_end = ToolCallParser._tag_end_index(text, start_index)
            if tag_end == -1:
                return None
            payload_start = tag_end + 1
        payload = text[payload_start:]
        for end_marker in self._tool_marker_ends():
            marker_index = payload.find(end_marker)
            if marker_index != -1:
                payload = payload[:marker_index]
        return payload

    def _first_stream_tool_start_index(self, text: str) -> int | None:
        indexes = tuple(self._tool_marker_start_indexes(text))
        return indexes[0] if indexes else None

    @staticmethod
    def _decode_json_object(text: str) -> dict[str, Any] | None:
        value_text = text.strip()
        if not value_text:
            return None
        try:
            value, end_index = JSONDecoder().raw_decode(value_text)
        except JSONDecodeError:
            return None
        if value_text[end_index:].strip():
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _tool_call_data(call: Any) -> dict[str, Any]:
        name = getattr(call, "name", None)
        arguments = getattr(call, "arguments", None)
        return {
            "name": name if isinstance(name, str) else None,
            "arguments": arguments if isinstance(arguments, dict) else {},
        }

    @staticmethod
    def _diagnostic_data(diagnostic: ToolCallDiagnostic) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": diagnostic.code.value,
            "stage": diagnostic.stage.value,
            "status": diagnostic.status.value,
            "retryable": diagnostic.retryable,
            "detail_count": len(diagnostic.details),
            "correlated": diagnostic.call_id is not None,
        }
        stream_status = diagnostic.details.get("stream_status")
        if stream_status in {
            "none",
            "prefix",
            "open",
            "closed",
            "malformed",
            "unterminated",
        }:
            data["details"] = {"stream_status": stream_status}
        return data

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
            suffix_status = self._capability.tool_call_status(suffix)
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
            suffix_status = self._capability.tool_call_status(suffix)
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
        tool_format = getattr(self._capability, "tool_format", None)
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
        tool_format = getattr(self._capability, "tool_format", None)
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
        tool_format = getattr(self._capability, "tool_format", None)
        if tool_format is ToolFormat.HARMONY:
            return ("<|channel|>final<|message|>",)
        return ()

    def _visible_tool_suffix_text(self, visible_suffix: str) -> str:
        for marker in self._visible_tool_marker_ends():
            if visible_suffix.startswith(marker):
                return visible_suffix[len(marker) :]
        return visible_suffix

    def _sorted_visible_tool_marker_ends(self) -> tuple[str, ...]:
        return tuple(
            sorted(self._visible_tool_marker_ends(), key=len, reverse=True)
        )

    def _append_visible_suffix(
        self,
        items: list[_ToolCallResponseParserItem],
        visible_suffix: str,
    ) -> list[_ToolCallResponseParserItem]:
        self._append_visible_output(items, visible_suffix)
        return items

    def _parse_text(self) -> str:
        tool_text = self._tool_buffer.getvalue()
        if tool_text or self._pending_tool_visible_suffix:
            return tool_text + self._pending_tool_visible_suffix
        return self._buffer.getvalue()

    def _closed_parse_text(self, text: str, visible_suffix: str) -> str:
        if (
            getattr(self._capability, "tool_format", None)
            is ToolFormat.HARMONY
            and visible_suffix.startswith("<|channel|>final<|message|>")
            and "<|call|>" not in text
        ):
            return text + "<|call|>"
        return text

    def _parse_buffer(
        self, text: str
    ) -> tuple[list[Any] | None, list[ToolCallDiagnostic]]:
        outcome = self._capability.parse_calls(text)
        return outcome.calls or None, outcome.diagnostics

    def _stream_buffer_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        return self._capability.stream_buffer_diagnostics(text)

    def _clear_buffers(self, visible_text: str = "") -> None:
        self._buffer = StringIO()
        self._tool_buffer = _IncrementalTextBuffer()
        self._tag_buffer = visible_text[-64:]
        self._inside_call = False
        self._pending_tokens.clear()
        self._pending_str = ""
        self._pending_tool_visible_suffix = ""
        self._tool_call_id = None
        self._tool_argument_emitted_until = 0
        self._tool_argument_text_by_id.clear()
        self._reset_tool_close_state()
        self._visible_state = _VisibleTextState()
        if visible_text:
            self._buffer.write(visible_text)
            self._append_visible_text(visible_text)
