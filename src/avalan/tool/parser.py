from ..entities import (
    Message,
    MessageContent,
    MessageContentText,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallParseOutcome,
    ToolCallRecoveryFormat,
    ToolFormat,
    ToolValue,
)
from .dsml import DsmlTools

from ast import literal_eval
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from json import JSONDecodeError, JSONDecoder, loads
from re import DOTALL, MULTILINE, compile, finditer, search, sub
from typing import Any, cast, final
from uuid import UUID, uuid4
from xml.etree import ElementTree

_HARMONY_SEGMENT_PATTERN = compile(
    r"<\|channel\|>(?P<channel>\w+).*?<\|message\|>(?P<message>.*?)(?:<\|call\|>|<\|end\|>)",
    DOTALL,
)
_SPECIAL_TOKEN_PATTERN = compile(r"<\|[^>]+?\|>")
_MARKDOWN_FENCED_BLOCK_PATTERN = compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n\r]*(?:\r?\n)"
    r".*?"
    r"^[ \t]*(?P=fence)[ \t]*$",
    DOTALL | MULTILINE,
)
_MARKDOWN_FENCED_BLOCK_CAPTURE_PATTERN = compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n\r]*(?:\r?\n)"
    r"(?P<content>.*?)"
    r"^[ \t]*(?P=fence)[ \t]*$",
    DOTALL | MULTILINE,
)
_MARKDOWN_FENCE_LINE_PATTERN = compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
_XML_NAME_PREFIX_PATTERN = compile(r"<(/?)([A-Za-z_][\w.-]*):")
_TOOL_NAME_PATTERN_TEXT = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
_TOOL_NAME_PATTERN = compile(rf"^{_TOOL_NAME_PATTERN_TEXT}$")
_BRACKET_CALL_PATTERN = compile(r"\[([^\]\r\n]*)\]\(([^)\r\n]*)\)")
_BRACKET_CALL_START_PATTERN = compile(r"\[[^\]\r\n]*\]\(")
_REACT_CALL_PATTERN = compile(
    r"^[ \t]*Action:\s*(?P<name>[^\r\n]*)\s*(?:\r?\n)+"
    r"[ \t]*Action Input:\s*(?P<arguments>.*?)(?=^[ \t]*Action:\s*|\Z)",
    DOTALL | MULTILINE,
)
_REACT_ACTION_PATTERN = compile(
    r"^[ \t]*Action:\s*(?P<name>[^\r\n]*)\s*$",
    MULTILINE,
)
_REACT_INPUT_PATTERN = compile(r"^[ \t]*Action Input:\s*", MULTILINE)
_REACT_JSON_DECODER = JSONDecoder()
_NO_TAG_PAYLOAD = object()


@dataclass(frozen=True, slots=True)
class _RecoveryPayload:
    span: tuple[int, int]
    payload: Any
    recovery_format: ToolCallRecoveryFormat


class ToolCallParser:
    _eos_token: str | None
    _maximum_payload_depth: int | None
    _maximum_payload_size: int | None
    _maximum_text_size: int | None
    _recovery_formats: tuple[ToolCallRecoveryFormat, ...]
    _tool_format: ToolFormat | None

    def __init__(
        self,
        tool_format: ToolFormat | None = None,
        eos_token: str | None = None,
        recovery_formats: Sequence[ToolCallRecoveryFormat] | None = None,
        maximum_text_size: int | None = None,
        maximum_payload_depth: int | None = None,
        maximum_payload_size: int | None = None,
    ) -> None:
        for limit in (
            maximum_text_size,
            maximum_payload_depth,
            maximum_payload_size,
        ):
            assert limit is None or (
                isinstance(limit, int)
                and not isinstance(limit, bool)
                and limit > 0
            )
        if recovery_formats is not None:
            for recovery_format in recovery_formats:
                assert isinstance(recovery_format, ToolCallRecoveryFormat)
        self._tool_format = tool_format
        self._eos_token = eos_token
        self._recovery_formats = tuple(recovery_formats or ())
        self._maximum_text_size = maximum_text_size
        self._maximum_payload_depth = maximum_payload_depth
        self._maximum_payload_size = maximum_payload_size

    @property
    def tool_format(self) -> ToolFormat | None:
        """Return the tool format used by the parser."""
        return self._tool_format

    @property
    def recovery_formats(self) -> tuple[ToolCallRecoveryFormat, ...]:
        """Return explicitly enabled recovery formats."""
        return self._recovery_formats

    def __call__(
        self, text: str
    ) -> tuple[str, dict[str, Any]] | list[ToolCall] | None:
        if self._text_exceeds_limit(text):
            return None

        calls: tuple[str, dict[str, Any]] | list[ToolCall] | None
        match self._tool_format:
            case ToolFormat.JSON:
                calls = self._parse_json(text)
            case ToolFormat.REACT:
                calls = self._parse_react(text)
            case ToolFormat.BRACKET:
                calls = self._parse_bracket(text)
            case ToolFormat.OPENAI:
                calls = self._parse_openai_json(text)
            case ToolFormat.HARMONY:
                calls = self._parse_harmony(text)
            case ToolFormat.DSML:
                calls = DsmlTools.parse_tool_calls(text)
            case _:
                calls = None
        if not calls:
            calls = self._parse_tag(text)
        if not calls:
            calls = self._parse_recovery(text)
        return calls

    def parse(self, text: str) -> ToolCallParseOutcome:
        """Return normalized tool calls and parse diagnostics."""
        text_diagnostic = self._text_limit_diagnostic(text)
        if text_diagnostic is not None:
            return ToolCallParseOutcome(diagnostics=[text_diagnostic])

        if self._tool_format is ToolFormat.REACT:
            react_outcome = self._parse_react_outcome(text)
            if react_outcome.calls:
                if not react_outcome.diagnostics:
                    react_outcome.diagnostics.extend(
                        self._tag_failure_diagnostics(text)
                    )
                    react_outcome.diagnostics.extend(
                        self._recovery_failure_diagnostics(text)
                    )
                return react_outcome

        parsed = self(text)
        calls: list[ToolCall] = []
        diagnostics: list[ToolCallDiagnostic] = []

        if isinstance(parsed, list):
            for call in parsed:
                normalized, diagnostic = self._normalize_tool_call(call)
                if normalized is not None:
                    calls.append(normalized)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
        elif isinstance(parsed, tuple) and len(parsed) == 2:
            normalized, diagnostic = self._normalize_tuple_call(parsed)
            if normalized is not None:
                calls.append(normalized)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        else:
            diagnostics.extend(self._parse_failure_diagnostics(text))

        if calls and not diagnostics:
            diagnostics.extend(self._parse_failure_diagnostics(text))

        return ToolCallParseOutcome(calls=calls, diagnostics=diagnostics)

    def set_eos_token(self, eos_token: str) -> None:
        self._eos_token = eos_token

    def is_potential_tool_call(self, buffer: str, token_str: str) -> bool:
        """Return ``True`` if tool detection should run for ``token_str``.

        This provides a fast check during streaming. If ``token_str`` is empty
        or only whitespace, ``False`` is returned since nothing new was added
        that could form a tool call.
        """
        return bool(token_str and token_str.strip())

    @final
    @dataclass(frozen=True, kw_only=True, slots=True)
    class StructuredMessage:
        """Structured content extracted from a raw message payload."""

        content: str
        thinking: str | None = None
        tool_calls: list[dict[str, object]] = field(default_factory=list)

    @final
    @dataclass(frozen=True, kw_only=True, slots=True)
    class PreparedMessage:
        """Normalized message ready for chat template consumption."""

        template_content: str | MessageContent | list[MessageContent] | None
        message_dict: dict[str, object]

    def prepare_message_for_template(
        self,
        message: Message,
        message_dict: dict[str, object],
    ) -> "ToolCallParser.PreparedMessage":
        """Return a message payload ready for chat template rendering.

        The method normalizes ``message_dict`` in-place, ensuring thinking and
        tool call metadata are consistent with any structured content detected
        in ``message``.
        """
        if not isinstance(message_dict.get("tool_calls"), list):
            message_dict["tool_calls"] = []

        template_content: (
            str | MessageContent | list[MessageContent] | None
        ) = message.content
        source = self._resolve_text_source(
            template_content, message_dict.get("content")
        )

        if source:
            structured = self.extract_structured_message(source)
            if structured:
                self._merge_thinking(message_dict, structured.thinking)
                message_dict["content"] = structured.content
                template_content = structured.content

                if structured.tool_calls and not message_dict["tool_calls"]:
                    message_dict["tool_calls"] = structured.tool_calls

        return ToolCallParser.PreparedMessage(
            template_content=template_content,
            message_dict=message_dict,
        )

    def extract_structured_message(
        self, text: str
    ) -> "ToolCallParser.StructuredMessage | None":
        """Parse ``text`` looking for structured content markers.

        Currently Harmony-formatted payloads are recognized automatically, but
        the method can be extended to additional formats over time. When no
        structure is detected ``None`` is returned.
        """
        if "<|channel|>" not in text:
            return None

        thinking, content = self.extract_harmony_content(text)
        tool_calls = self.message_tool_calls(text)
        return ToolCallParser.StructuredMessage(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
        )

    def message_tool_calls(self, text: str) -> list[dict[str, object]]:
        """Return tool calls extracted from ``text`` in message format."""
        parsed: tuple[str, dict[str, Any]] | list[ToolCall] | None = None
        if "<|call|>" in text and "<|channel|>" in text:
            parsed = self._parse_harmony(text)
        elif "tool_calls" in text and self._has_dsml_tool_call_start(text):
            parsed = DsmlTools.parse_tool_calls(text)
        elif self._tool_format:
            parsed = self(text)

        if not parsed:
            return []

        if isinstance(parsed, list):
            return [
                {
                    "id": str(call.id),
                    "name": call.name,
                    "arguments": call.arguments or {},
                    "content_type": "json",
                }
                for call in parsed
                if self._is_valid_tool_name(call.name)
                and (
                    call.arguments is None or isinstance(call.arguments, dict)
                )
            ]

        if isinstance(parsed, tuple) and len(parsed) == 2:
            name, arguments = parsed
            if self._is_valid_tool_name(name) and isinstance(arguments, dict):
                return [
                    {
                        "id": None,
                        "name": name,
                        "arguments": arguments,
                        "content_type": "json",
                    }
                ]

        return []

    @staticmethod
    def extract_harmony_content(text: str) -> tuple[str | None, str]:
        """Return thinking and content sections from Harmony transcripts."""
        analysis_parts: list[str] = []
        final_parts: list[str] = []
        for match in _HARMONY_SEGMENT_PATTERN.finditer(text):
            channel = match.group("channel")
            message = match.group("message").strip()
            if not message:
                continue
            if channel == "analysis":
                analysis_parts.append(message)
            elif channel == "final":
                final_parts.append(message)

        thinking = "\n\n".join(analysis_parts) if analysis_parts else None
        if final_parts:
            content = "\n\n".join(final_parts).strip()
        elif analysis_parts:
            content = ""
        else:
            content = sub(
                r"\n{3,}", "\n\n", _SPECIAL_TOKEN_PATTERN.sub("", text)
            ).strip()
        return thinking, content

    def _resolve_text_source(
        self,
        template_content: str | MessageContent | list[MessageContent] | None,
        serialized_content: object,
    ) -> str | None:
        if isinstance(template_content, str):
            return template_content

        if isinstance(template_content, MessageContentText):
            return template_content.text

        if (
            isinstance(template_content, list)
            and len(template_content) == 1
            and isinstance(template_content[0], MessageContentText)
        ):
            return template_content[0].text

        if isinstance(serialized_content, str):
            return serialized_content

        if (
            isinstance(serialized_content, dict)
            and serialized_content.get("type") == "text"
        ):
            text_value = serialized_content.get("text")
            if isinstance(text_value, str):
                return text_value

        if (
            isinstance(serialized_content, list)
            and len(serialized_content) == 1
            and isinstance(serialized_content[0], dict)
            and serialized_content[0].get("type") == "text"
        ):
            text_value = serialized_content[0].get("text")
            if isinstance(text_value, str):
                return text_value

        return None

    def _has_dsml_tool_call_start(self, text: str) -> bool:
        try:
            return DsmlTools.tool_call_start_span(text) is not None
        except RuntimeError:
            if self._tool_format is ToolFormat.DSML:
                raise
            return False

    @staticmethod
    def _merge_thinking(
        message_dict: dict[str, object], thinking: str | None
    ) -> None:
        if thinking is None:
            existing_thinking = message_dict.get("thinking")
            if existing_thinking in (None, ""):
                message_dict["thinking"] = None
            return

        existing_thinking = message_dict.get("thinking")
        if (
            isinstance(existing_thinking, str)
            and not existing_thinking.strip()
        ):
            existing_thinking = None

        if existing_thinking:
            combined = "\n\n".join(
                part
                for part in (existing_thinking, thinking)
                if isinstance(part, str) and part
            )
        else:
            combined = thinking

        message_dict["thinking"] = combined

    class ToolCallBufferStatus(Enum):
        """Status of a buffer relative to a tool call."""

        NONE = 0
        PREFIX = 1
        OPEN = 2
        CLOSED = 3
        MALFORMED = 4
        UNTERMINATED = 5

    def tool_call_status(
        self, buffer: str, *, final: bool = False
    ) -> "ToolCallParser.ToolCallBufferStatus":
        status_buffer = self._without_markdown_fenced_blocks(buffer)
        status: ToolCallParser.ToolCallBufferStatus
        if self._tool_format is ToolFormat.DSML:
            dsml_status = self._dsml_tool_call_status(status_buffer)
            if dsml_status is not self.ToolCallBufferStatus.NONE:
                status = dsml_status
                return self._final_tool_call_status(buffer, status, final)
        if not status_buffer:
            return self.ToolCallBufferStatus.NONE

        start = ["<tool_call", "<tool ", "<tool>"]
        end = ["</tool_call>", "</tool>", "<|call|>"]
        if self._tool_format is ToolFormat.HARMONY:
            start.extend(
                [
                    "<|channel|>commentary",
                    "<|start|>assistant<|channel|>commentary",
                    "<|channel|>analysis",
                    "<|start|>assistant<|channel|>analysis",
                ]
            )
            end.append("<|channel|>final<|message|>")
        max_len = max(len(s) for s in start)
        tail = status_buffer[-max_len:]
        for s in start:
            if self._tool_start_marker_needs_boundary(s) and tail == s:
                status = self.ToolCallBufferStatus.PREFIX
                return self._final_tool_call_status(buffer, status, final)
            if s.startswith(tail) and tail != s:
                status = self.ToolCallBufferStatus.PREFIX
                return self._final_tool_call_status(buffer, status, final)
        latest_start = self._latest_tool_start(status_buffer, start)
        if latest_start is not None:
            idx, marker = latest_start
            after = status_buffer[idx + len(marker) :]
            if self._has_unquoted_tool_end(after, end) or (
                self._opening_tool_tag_is_self_closing(marker, after)
                and not self._has_unclosed_tool_start_before(
                    status_buffer, idx, start, end
                )
            ):
                status = self.ToolCallBufferStatus.CLOSED
                return self._final_tool_call_status(buffer, status, final)
            status = self.ToolCallBufferStatus.OPEN
            return self._final_tool_call_status(buffer, status, final)
        return self.ToolCallBufferStatus.NONE

    @classmethod
    def _latest_tool_start(
        cls, text: str, start_markers: list[str]
    ) -> tuple[int, str] | None:
        latest: tuple[int, str] | None = None
        for marker in start_markers:
            index = text.rfind(marker)
            while index != -1:
                if (
                    not cls._index_is_inside_quoted_text(text, index)
                    and cls._tool_start_marker_boundary_is_valid(
                        text, index, marker
                    )
                    and (
                        latest is None
                        or index > latest[0]
                        or (
                            index == latest[0] and len(marker) > len(latest[1])
                        )
                    )
                ):
                    latest = (index, marker)
                    break
                index = text.rfind(marker, 0, index)
        return latest

    @classmethod
    def _has_unquoted_tool_end(cls, text: str, end_markers: list[str]) -> bool:
        for marker in end_markers:
            index = text.find(marker)
            while index != -1:
                if not cls._index_is_inside_quoted_text(text, index):
                    return True
                index = text.find(marker, index + 1)
        return False

    @staticmethod
    def _index_is_inside_quoted_text(text: str, index: int) -> bool:
        quote: str | None = None
        escaped = False
        for position, character in enumerate(text[:index]):
            if escaped:
                escaped = False
                continue
            if quote is not None and character == "\\":
                escaped = True
                continue
            if character not in ("'", '"'):
                continue
            if character == "'" and not ToolCallParser._is_quote_delimiter(
                text, position
            ):
                continue
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        return quote is not None

    @staticmethod
    def _is_quote_delimiter(text: str, position: int) -> bool:
        return (
            position == 0
            or not text[position - 1].isalnum()
            or position + 1 == len(text)
            or not text[position + 1].isalnum()
        )

    @staticmethod
    def _opening_tool_tag_is_self_closing(
        start_marker: str, after_start: str
    ) -> bool:
        if start_marker not in {"<tool_call", "<tool "}:
            return False

        tag_end = ToolCallParser._tag_end_index(after_start, 0)
        if tag_end == -1:
            return False

        return after_start[: tag_end + 1].rstrip().endswith("/>")

    @staticmethod
    def _tag_end_index(text: str, start: int) -> int:
        quote: str | None = None
        escaped = False
        for index, character in enumerate(text[start:], start=start):
            if escaped:
                escaped = False
                continue
            if quote is not None:
                if character == "\\":
                    escaped = True
                elif (
                    character == quote
                    and ToolCallParser._attribute_tail_is_valid(
                        text, index + 1
                    )
                ):
                    quote = None
                continue
            if character in ("'", '"'):
                quote = character
                continue
            if character == ">":
                return index
        return -1

    def _has_unclosed_tool_start_before(
        self,
        buffer: str,
        index: int,
        start_markers: list[str],
        end_markers: list[str],
    ) -> bool:
        last_end = self._last_tool_end_before(buffer, index, end_markers)
        for marker in start_markers:
            start_index = buffer.find(marker, last_end + 1)
            while start_index != -1 and start_index < index:
                if not self._index_is_inside_quoted_text(
                    buffer, start_index
                ) and self._tool_start_marker_boundary_is_valid(
                    buffer, start_index, marker
                ):
                    after_start = buffer[start_index + len(marker) :]
                    if not self._opening_tool_tag_is_self_closing(
                        marker, after_start
                    ):
                        return True
                start_index = buffer.find(marker, start_index + 1)
        return False

    @staticmethod
    def _last_tool_end_before(
        buffer: str, index: int, end_markers: list[str]
    ) -> int:
        last_end = -1
        for marker in end_markers:
            marker_index = buffer.rfind(marker, 0, index)
            while marker_index != -1:
                if not ToolCallParser._index_is_inside_quoted_text(
                    buffer, marker_index
                ):
                    last_end = max(last_end, marker_index + len(marker))
                    break
                marker_index = buffer.rfind(marker, 0, marker_index)
        return last_end

    def stream_buffer_diagnostics(
        self, buffer: str
    ) -> list[ToolCallDiagnostic]:
        """Return diagnostics for a terminal streaming buffer."""
        status = self.tool_call_status(buffer, final=True)
        if status is self.ToolCallBufferStatus.UNTERMINATED:
            return [
                self._malformed_call_diagnostic(
                    message=(
                        "Tool call stream ended before the call was complete."
                    ),
                    details={"stream_status": status.name.lower()},
                )
            ]

        outcome = self.parse(buffer)
        if outcome.diagnostics:
            if status is self.ToolCallBufferStatus.MALFORMED:
                return [
                    self._diagnostic_with_stream_status(diagnostic, status)
                    for diagnostic in outcome.diagnostics
                ]
            return outcome.diagnostics
        if status is self.ToolCallBufferStatus.MALFORMED:
            return [
                self._malformed_call_diagnostic(
                    details={"stream_status": status.name.lower()}
                )
            ]
        return []

    def _final_tool_call_status(
        self,
        buffer: str,
        status: "ToolCallParser.ToolCallBufferStatus",
        final: bool,
    ) -> "ToolCallParser.ToolCallBufferStatus":
        if not final:
            return status
        if status in (
            self.ToolCallBufferStatus.PREFIX,
            self.ToolCallBufferStatus.OPEN,
        ):
            return self.ToolCallBufferStatus.UNTERMINATED
        if status is self.ToolCallBufferStatus.CLOSED:
            outcome = self.parse(buffer)
            if not outcome.calls:
                return self.ToolCallBufferStatus.MALFORMED
        return status

    @staticmethod
    def _diagnostic_with_stream_status(
        diagnostic: ToolCallDiagnostic,
        status: "ToolCallParser.ToolCallBufferStatus",
    ) -> ToolCallDiagnostic:
        details = diagnostic.details.copy()
        details.setdefault("stream_status", status.name.lower())
        return ToolCallDiagnostic(
            id=diagnostic.id,
            call_id=diagnostic.call_id,
            requested_name=diagnostic.requested_name,
            canonical_name=diagnostic.canonical_name,
            status=diagnostic.status,
            code=diagnostic.code,
            stage=diagnostic.stage,
            message=diagnostic.message,
            retryable=diagnostic.retryable,
            details=details,
            started_at=diagnostic.started_at,
            finished_at=diagnostic.finished_at,
            duration_ms=diagnostic.duration_ms,
        )

    def _dsml_tool_call_status(
        self, buffer: str
    ) -> "ToolCallParser.ToolCallBufferStatus":
        status = DsmlTools.tool_call_buffer_status(buffer)
        match status:
            case "prefix":
                return self.ToolCallBufferStatus.PREFIX
            case "open":
                return self.ToolCallBufferStatus.OPEN
            case "closed":
                return self.ToolCallBufferStatus.CLOSED
            case _:
                return self.ToolCallBufferStatus.NONE

    def _parse_json(self, text: str) -> tuple[str, dict[str, Any]] | None:
        try:
            payload = loads(text)
            return payload["tool"], payload.get("arguments", {})
        except Exception:
            return None

    def _parse_react(self, text: str) -> tuple[str, dict[str, Any]] | None:
        for name, arguments, malformed in self._react_payloads(text):
            if (
                not malformed
                and _TOOL_NAME_PATTERN.fullmatch(name)
                and isinstance(arguments, dict)
            ):
                return name, arguments
        return None

    def _parse_react_outcome(self, text: str) -> ToolCallParseOutcome:
        calls: list[ToolCall] = []
        diagnostics: list[ToolCallDiagnostic] = []
        for name, arguments, malformed in self._react_payloads(text):
            requested_name = (
                name if _TOOL_NAME_PATTERN.fullmatch(name) else None
            )
            if requested_name is None:
                diagnostics.append(self._malformed_call_diagnostic())
                continue
            if malformed:
                diagnostics.append(
                    self._malformed_call_diagnostic(
                        requested_name=requested_name
                    )
                )
                continue
            if not isinstance(arguments, dict):
                diagnostics.append(
                    self._arguments_diagnostic(requested_name=requested_name)
                )
                continue
            diagnostic = self.resource_limit_diagnostic(
                value=arguments,
                maximum_depth=self._maximum_payload_depth,
                maximum_size=self._maximum_payload_size,
                stage=ToolCallDiagnosticStage.PARSE,
                requested_name=requested_name,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
                continue
            calls.append(
                ToolCall(
                    id=uuid4(),
                    name=requested_name,
                    arguments=cast(dict[str, ToolValue], arguments),
                )
            )
        return ToolCallParseOutcome(calls=calls, diagnostics=diagnostics)

    def _react_payloads(self, text: str) -> list[tuple[str, Any, bool]]:
        payloads: list[tuple[str, Any, bool]] = []
        for match in _REACT_CALL_PATTERN.finditer(text):
            arguments, malformed = self._decode_react_arguments(
                match.group("arguments")
            )
            payloads.append(
                (
                    match.group("name").strip(),
                    arguments,
                    malformed,
                )
            )
        return payloads

    @staticmethod
    def _decode_react_arguments(text: str) -> tuple[Any, bool]:
        stripped = text.strip()
        if not stripped:
            return None, True
        try:
            arguments, _ = _REACT_JSON_DECODER.raw_decode(stripped)
        except JSONDecodeError:
            return None, True
        return arguments, False

    def _parse_bracket(self, text: str) -> tuple[str, dict[str, Any]] | None:
        m = search(rf"\[({_TOOL_NAME_PATTERN_TEXT})\]\(([^)]+)\)", text)
        if m:
            return m.group(1), {"input": m.group(2)}
        return None

    def _parse_openai_json(
        self, text: str
    ) -> tuple[str, dict[str, Any]] | None:
        try:
            payload = loads(text)
            if not isinstance(payload, dict):
                return None
            name = payload.get("name")
            args = payload.get("arguments", {})
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
        except JSONDecodeError:
            pass
        return None

    def _parse_harmony(self, text: str) -> list[ToolCall] | None:
        tool_calls: list[ToolCall] = []
        pattern = (
            r"(?:<\|start\|>assistant)?"
            r"<\|channel\|>(?:commentary|analysis)"
            r" to=(?:functions\.)?([\w\.]+)"
            r"(?:<\|channel\|>(?:commentary|analysis))?"
            r"[^<]*"
            r"(?:<\|constrain\|>json)?"
            r"<\|message\|>\s*(\{.*?\})?\s*<\|call\|>"
        )
        for match in finditer(pattern, text, DOTALL):
            args_text = match.group(2)
            if args_text:
                try:
                    args = loads(args_text)
                except JSONDecodeError:
                    continue
            else:
                args = {}
            tool_calls.append(
                ToolCall(id=uuid4(), name=match.group(1), arguments=args)
            )
        if tool_calls:
            return tool_calls
        return None

    def _parse_tag(self, text: str) -> list[ToolCall] | None:
        tool_calls: list[ToolCall] = []

        if self._eos_token:
            text = text.strip().removesuffix(self._eos_token)
        text = self._without_markdown_fenced_blocks(text)
        for payload in self._tag_payloads(text):
            parsed = self._tool_call_from_payload(payload)
            if parsed is not None:
                tool_calls.append(parsed)

        return tool_calls if tool_calls else None

    def _normalize_tool_call(
        self, call: ToolCall
    ) -> tuple[ToolCall | None, ToolCallDiagnostic | None]:
        if not self._is_valid_tool_name(call.name):
            return None, self._malformed_call_diagnostic(
                call_id=call.id,
                requested_name=call.name if call.name.strip() else None,
            )
        if call.arguments is not None and not isinstance(call.arguments, dict):
            return None, self._arguments_diagnostic(
                call_id=call.id,
                requested_name=call.name,
            )
        diagnostic = self.resource_limit_diagnostic(
            value=call.arguments or {},
            maximum_depth=self._maximum_payload_depth,
            maximum_size=self._maximum_payload_size,
            stage=ToolCallDiagnosticStage.PARSE,
            call_id=call.id,
            requested_name=call.name,
        )
        if diagnostic is not None:
            return None, diagnostic
        return call, None

    def _normalize_tuple_call(
        self, parsed: tuple[Any, Any]
    ) -> tuple[ToolCall | None, ToolCallDiagnostic | None]:
        name, arguments = parsed
        if not self._is_valid_tool_name(name):
            return None, self._malformed_call_diagnostic(
                requested_name=(
                    name if isinstance(name, str) and name.strip() else None
                )
            )
        if not isinstance(arguments, dict):
            return None, self._arguments_diagnostic(requested_name=name)
        diagnostic = self.resource_limit_diagnostic(
            value=arguments,
            maximum_depth=self._maximum_payload_depth,
            maximum_size=self._maximum_payload_size,
            stage=ToolCallDiagnosticStage.PARSE,
            requested_name=name,
        )
        if diagnostic is not None:
            return None, diagnostic
        return (
            ToolCall(
                id=uuid4(),
                name=name,
                arguments=cast(dict[str, ToolValue], arguments),
            ),
            None,
        )

    def _parse_failure_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        match self._tool_format:
            case ToolFormat.JSON:
                diagnostics = self._json_failure_diagnostics(
                    text,
                    name_field="tool",
                )
            case ToolFormat.REACT:
                diagnostics = self._react_failure_diagnostics(text)
            case ToolFormat.BRACKET:
                diagnostics = self._bracket_failure_diagnostics(text)
            case ToolFormat.OPENAI:
                diagnostics = self._json_failure_diagnostics(
                    text,
                    name_field="name",
                )
            case ToolFormat.HARMONY:
                diagnostics = self._harmony_failure_diagnostics(text)
            case _:
                diagnostics = []

        if not diagnostics:
            diagnostics = self._tag_failure_diagnostics(text)
        diagnostics.extend(self._recovery_failure_diagnostics(text))
        return diagnostics

    def _text_exceeds_limit(self, text: str) -> bool:
        return (
            self._maximum_text_size is not None
            and self._text_size(text) > self._maximum_text_size
        )

    def _text_limit_diagnostic(self, text: str) -> ToolCallDiagnostic | None:
        if not self._text_exceeds_limit(text):
            return None
        assert self._maximum_text_size is not None
        return self._resource_limit_diagnostic(
            code=ToolCallDiagnosticCode.MAXIMUM_SIZE,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Tool parser input exceeds the maximum size.",
            details={
                "limit": self._maximum_text_size,
                "size": self._text_size(text),
            },
        )

    def _json_failure_diagnostics(
        self,
        text: str,
        *,
        name_field: str,
    ) -> list[ToolCallDiagnostic]:
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return []
        try:
            payload = loads(stripped)
        except JSONDecodeError:
            return [self._malformed_call_diagnostic()]

        if not isinstance(payload, dict):
            return [self._malformed_call_diagnostic()]

        name = payload.get(name_field)
        requested_name = name if isinstance(name, str) else None
        if not self._is_valid_tool_name(name):
            return [
                self._malformed_call_diagnostic(
                    requested_name=(
                        name
                        if isinstance(name, str) and name.strip()
                        else None
                    )
                )
            ]

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            return [self._arguments_diagnostic(requested_name=requested_name)]
        return []

    def _react_failure_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        outcome = self._parse_react_outcome(text)
        if outcome.diagnostics:
            return outcome.diagnostics
        if outcome.calls:
            return []

        has_action = _REACT_ACTION_PATTERN.search(text) is not None
        has_input = _REACT_INPUT_PATTERN.search(text) is not None
        if not has_action and not has_input:
            return []
        return [self._malformed_call_diagnostic()]

    def _bracket_failure_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        malformed = False
        complete_call_starts: set[int] = set()
        for match in _BRACKET_CALL_PATTERN.finditer(text):
            complete_call_starts.add(match.start())
            name = match.group(1).strip()
            argument = match.group(2)
            if (
                not name
                or not _TOOL_NAME_PATTERN.fullmatch(name)
                or not argument
            ):
                malformed = True
                break

        if not malformed:
            for match in _BRACKET_CALL_START_PATTERN.finditer(text):
                if match.start() not in complete_call_starts:
                    malformed = True
                    break

        if malformed:
            return [self._malformed_call_diagnostic()]
        return []

    def _harmony_failure_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        if "<|channel|>" not in text or " to=" not in text:
            return []

        diagnostics: list[ToolCallDiagnostic] = []
        pattern = (
            r"(?:<\|start\|>assistant)?"
            r"<\|channel\|>(?:commentary|analysis)"
            r" to=(?:functions\.)?([\w\.]+)"
            r"(?:<\|channel\|>(?:commentary|analysis))?"
            r"[^<]*"
            r"(?:<\|constrain\|>json)?"
            r"<\|message\|>\s*(.*?)\s*<\|call\|>"
        )
        for match in finditer(pattern, text, DOTALL):
            requested_name = match.group(1)
            args_text = match.group(2)
            if not args_text:
                continue
            try:
                arguments = loads(args_text)
            except JSONDecodeError:
                diagnostics.append(
                    self._malformed_call_diagnostic(
                        requested_name=requested_name
                    )
                )
                continue
            if not isinstance(arguments, dict):
                diagnostics.append(
                    self._arguments_diagnostic(requested_name=requested_name)
                )
        return diagnostics

    def _tag_failure_diagnostics(self, text: str) -> list[ToolCallDiagnostic]:
        text = self._without_markdown_fenced_blocks(text)
        if (
            self._find_unquoted_tool_start(text, "<tool_call", 0) == -1
            and self._find_unquoted_marker(text, "<tool ", 0) == -1
            and self._find_unquoted_marker(text, "<tool>", 0) == -1
        ):
            return []

        diagnostics: list[ToolCallDiagnostic] = []
        payloads = self._tag_payloads(text)
        if not payloads:
            return [self._malformed_call_diagnostic()]

        for payload in payloads:
            diagnostic = self._payload_diagnostic(payload)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        return diagnostics

    @staticmethod
    def _without_markdown_fenced_blocks(text: str) -> str:
        stripped_text = _MARKDOWN_FENCED_BLOCK_PATTERN.sub("", text)
        lines = stripped_text.splitlines(keepends=True)
        output: list[str] = []
        open_fence: str | None = None
        for line in lines:
            match = _MARKDOWN_FENCE_LINE_PATTERN.match(line)
            if open_fence is None:
                if match is None:
                    output.append(line)
                    continue
                open_fence = match.group("fence")
                continue

            if match is None:
                continue
            fence = match.group("fence")
            if fence[0] == open_fence[0] and len(fence) >= len(open_fence):
                open_fence = None

        return "".join(output)

    def _parse_recovery(self, text: str) -> list[ToolCall] | None:
        tool_calls: list[ToolCall] = []
        for recovered in self._recovery_payloads(text):
            parsed = self._tool_call_from_payload(recovered.payload)
            if parsed is not None:
                tool_calls.append(parsed)
        return tool_calls if tool_calls else None

    def _recovery_failure_diagnostics(
        self, text: str
    ) -> list[ToolCallDiagnostic]:
        diagnostics: list[ToolCallDiagnostic] = []
        candidates = self._recovery_payloads(text)
        candidates_by_format: dict[
            ToolCallRecoveryFormat, list[_RecoveryPayload]
        ] = {}
        for recovered in candidates:
            candidates_by_format.setdefault(
                recovered.recovery_format, []
            ).append(recovered)
            diagnostic = self._payload_recovery_diagnostic(
                recovered.payload, recovered.recovery_format
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)

        reported_marker_starts: set[int] = set()
        for recovery_format in self._recovery_formats:
            candidate_starts = {
                candidate.span[0]
                for candidate in candidates_by_format.get(recovery_format, [])
            }
            for marker_start in self._recovery_marker_starts(
                text, recovery_format
            ):
                if (
                    marker_start in candidate_starts
                    or marker_start in reported_marker_starts
                ):
                    continue
                diagnostics.append(
                    self._malformed_call_diagnostic(
                        recovery_format=recovery_format
                    )
                )
                reported_marker_starts.add(marker_start)
                break
        return diagnostics

    def _recovery_payloads(self, text: str) -> list[_RecoveryPayload]:
        candidates: list[_RecoveryPayload] = []
        seen_payloads: set[tuple[tuple[int, int], str]] = set()
        for recovery_format in self._recovery_formats:
            for payload in self._recovery_payloads_for_format(
                text, recovery_format
            ):
                key = (payload.span, repr(payload.payload))
                if key in seen_payloads:
                    continue
                seen_payloads.add(key)
                candidates.append(payload)

        recovered: list[_RecoveryPayload] = []
        for candidate in sorted(candidates, key=lambda payload: payload.span):
            if any(
                self._equivalent_recovery_payload(existing, candidate)
                for existing in recovered
            ):
                continue
            recovered.append(candidate)
        return recovered

    @staticmethod
    def _equivalent_recovery_payload(
        left: _RecoveryPayload, right: _RecoveryPayload
    ) -> bool:
        return (
            repr(left.payload) == repr(right.payload)
            and left.span[0] < right.span[1]
            and right.span[0] < left.span[1]
        )

    def _recovery_payloads_for_format(
        self,
        text: str,
        recovery_format: ToolCallRecoveryFormat,
    ) -> list[_RecoveryPayload]:
        match recovery_format:
            case ToolCallRecoveryFormat.TOOL_CALL_BLOCK:
                return self._tool_call_block_payloads(
                    text, recovery_format=recovery_format
                )
            case ToolCallRecoveryFormat.MINIMAX_XML:
                return self._xml_recovery_payloads(
                    text,
                    recovery_format=recovery_format,
                    names={"invoke", "tool_call"},
                )
            case ToolCallRecoveryFormat.TOOL_CODE:
                return self._tool_code_payloads(
                    text, recovery_format=recovery_format
                )
            case ToolCallRecoveryFormat.BROAD_XML:
                return self._xml_recovery_payloads(
                    text,
                    recovery_format=recovery_format,
                    names={"function", "function_call", "invoke"},
                )
            case ToolCallRecoveryFormat.DSML_LEAKAGE:
                return self._xml_recovery_payloads(
                    text,
                    recovery_format=recovery_format,
                    names={"invoke"},
                )
            case ToolCallRecoveryFormat.FENCED:
                return self._fenced_recovery_payloads(text)

    @classmethod
    def _has_recovery_marker(
        cls, text: str, recovery_format: ToolCallRecoveryFormat
    ) -> bool:
        return bool(cls._recovery_marker_starts(text, recovery_format))

    @classmethod
    def _recovery_marker_starts(
        cls, text: str, recovery_format: ToolCallRecoveryFormat
    ) -> list[int]:
        match recovery_format:
            case ToolCallRecoveryFormat.TOOL_CALL_BLOCK:
                return [
                    match.start() for match in finditer(r"\[TOOL_CALL\]", text)
                ]
            case ToolCallRecoveryFormat.MINIMAX_XML:
                return cls._xml_recovery_marker_starts(
                    text, {"invoke", "tool_call"}
                )
            case ToolCallRecoveryFormat.TOOL_CODE:
                return [
                    match.start() for match in finditer(r"<tool_code\b", text)
                ]
            case ToolCallRecoveryFormat.BROAD_XML:
                return cls._xml_recovery_marker_starts(
                    text, {"function", "function_call", "invoke"}
                )
            case ToolCallRecoveryFormat.DSML_LEAKAGE:
                return [
                    match.start()
                    for match in finditer(r"<[A-Za-z_][\w.-]*:invoke\b", text)
                ]
            case ToolCallRecoveryFormat.FENCED:
                closed_block_spans = [
                    match.span()
                    for match in (
                        _MARKDOWN_FENCED_BLOCK_CAPTURE_PATTERN.finditer(text)
                    )
                ]
                return [
                    match.start()
                    for match in compile(
                        r"^[ \t]*(?:`{3,}|~{3,})", MULTILINE
                    ).finditer(text)
                    if not any(
                        start < match.start() < end
                        for start, end in closed_block_spans
                    )
                ]

    @staticmethod
    def _xml_recovery_marker_starts(text: str, names: set[str]) -> list[int]:
        name_pattern = "|".join(sorted(names))
        pattern = compile(rf"<(?:[A-Za-z_][\w.-]*:)?(?:{name_pattern})\b")
        return [match.start() for match in pattern.finditer(text)]

    def _tool_call_block_payloads(
        self,
        text: str,
        *,
        recovery_format: ToolCallRecoveryFormat,
    ) -> list[_RecoveryPayload]:
        payloads: list[_RecoveryPayload] = []
        pattern = compile(
            r"\[TOOL_CALL\](?P<payload>.*?)\[/TOOL_CALL\]",
            DOTALL,
        )
        for match in pattern.finditer(text):
            payloads.append(
                _RecoveryPayload(
                    span=match.span(),
                    payload=self._deserialize_payload(match.group("payload")),
                    recovery_format=recovery_format,
                )
            )
        return payloads

    def _tool_code_payloads(
        self,
        text: str,
        *,
        recovery_format: ToolCallRecoveryFormat,
    ) -> list[_RecoveryPayload]:
        payloads: list[_RecoveryPayload] = []
        pattern = compile(
            r"<tool_code[^>]*>(?P<payload>.*?)</tool_code>",
            DOTALL,
        )
        for match in pattern.finditer(text):
            payload = self._deserialize_payload(match.group("payload"))
            if payload is None:
                payload = self._function_call_payload(match.group("payload"))
            payloads.append(
                _RecoveryPayload(
                    span=match.span(),
                    payload=payload,
                    recovery_format=recovery_format,
                )
            )
        return payloads

    def _fenced_recovery_payloads(self, text: str) -> list[_RecoveryPayload]:
        payloads: list[_RecoveryPayload] = []
        for match in _MARKDOWN_FENCED_BLOCK_CAPTURE_PATTERN.finditer(text):
            content = match.group("content")
            direct_payload = self._deserialize_payload(content)
            if direct_payload is not None:
                payloads.append(
                    _RecoveryPayload(
                        span=match.span(),
                        payload=direct_payload,
                        recovery_format=ToolCallRecoveryFormat.FENCED,
                    )
                )
                continue
            nested_payloads = (
                self._tool_call_block_payloads(
                    content,
                    recovery_format=ToolCallRecoveryFormat.FENCED,
                )
                + self._tool_code_payloads(
                    content,
                    recovery_format=ToolCallRecoveryFormat.FENCED,
                )
                + self._xml_recovery_payloads(
                    content,
                    recovery_format=ToolCallRecoveryFormat.FENCED,
                    names={
                        "function",
                        "function_call",
                        "invoke",
                        "tool_call",
                    },
                )
            )
            if nested_payloads:
                payloads.extend(
                    _RecoveryPayload(
                        span=match.span(),
                        payload=nested.payload,
                        recovery_format=ToolCallRecoveryFormat.FENCED,
                    )
                    for nested in nested_payloads
                )
                continue
            payloads.append(
                _RecoveryPayload(
                    span=match.span(),
                    payload=None,
                    recovery_format=ToolCallRecoveryFormat.FENCED,
                )
            )
        return payloads

    def _xml_recovery_payloads(
        self,
        text: str,
        *,
        recovery_format: ToolCallRecoveryFormat,
        names: set[str],
    ) -> list[_RecoveryPayload]:
        payloads: list[_RecoveryPayload] = []
        name_pattern = "|".join(sorted(names))
        pattern = compile(
            rf"<(?:[A-Za-z_][\w.-]*:)?(?P<tag>{name_pattern})\b[^>]*>"
            rf".*?</(?:[A-Za-z_][\w.-]*:)?(?P=tag)>",
            DOTALL,
        )
        for match in pattern.finditer(text):
            payloads.append(
                _RecoveryPayload(
                    span=match.span(),
                    payload=self._xml_payload(match.group(0)),
                    recovery_format=recovery_format,
                )
            )
        return payloads

    def _xml_payload(self, text: str) -> Any:
        xml_text = _XML_NAME_PREFIX_PATTERN.sub(r"<\1", text)
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return None

        tag = self._xml_local_name(root.tag)
        if tag in {"function", "function_call", "invoke"}:
            return self._xml_named_payload(root)
        if tag == "tool_call":
            return self._xml_tool_call_payload(root)
        return None

    def _xml_named_payload(self, element: ElementTree.Element) -> Any:
        name = element.attrib.get("name")
        if not isinstance(name, str) or not name.strip():
            return None

        arguments = self._xml_arguments(element)
        if arguments is None:
            return None
        return {"name": name, "arguments": arguments}

    def _xml_tool_call_payload(self, element: ElementTree.Element) -> Any:
        name = element.attrib.get("name")
        if isinstance(name, str) and name.strip():
            arguments = None
            raw_payload = self._deserialize_payload(element.text or "")
            if isinstance(raw_payload, dict):
                arguments = raw_payload
            else:
                arguments = self._xml_arguments(element)
            if arguments is None:
                return None
            return {"name": name, "arguments": arguments}

        raw_payload = self._deserialize_payload(element.text or "")
        if isinstance(raw_payload, dict):
            return raw_payload

        child_name = self._xml_child_text(element, "name")
        if child_name is None:
            return None
        arguments = self._xml_arguments(element)
        if arguments is None:
            return None
        return {"name": child_name, "arguments": arguments}

    def _xml_arguments(
        self, element: ElementTree.Element
    ) -> dict[str, ToolValue] | None:
        arguments_text = self._xml_child_text(element, "arguments")
        if arguments_text is not None:
            arguments = self._deserialize_payload(arguments_text)
            if isinstance(arguments, dict):
                return cast(dict[str, ToolValue], arguments)
            return None

        parameters: dict[str, ToolValue] = {}
        for child in element:
            if self._xml_local_name(child.tag) != "parameter":
                continue
            name = child.attrib.get("name")
            if not isinstance(name, str) or not name.strip():
                return None
            parameters[name] = self._xml_parameter_value(child)
        return parameters

    def _xml_parameter_value(self, element: ElementTree.Element) -> ToolValue:
        text = (element.text or "").strip()
        if element.attrib.get("string") == "false":
            try:
                return cast(ToolValue, loads(text))
            except JSONDecodeError:
                return text
        return text

    @classmethod
    def _xml_child_text(
        cls, element: ElementTree.Element, name: str
    ) -> str | None:
        for child in element:
            if cls._xml_local_name(child.tag) == name:
                return child.text or ""
        return None

    @staticmethod
    def _xml_local_name(name: str) -> str:
        return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]

    def _function_call_payload(self, text: str) -> Any:
        match = search(r"([\w.]+)\s*\((\{.*\})\)", text.strip(), DOTALL)
        if not match:
            return None
        arguments = self._deserialize_payload(match.group(2))
        if not isinstance(arguments, dict):
            return None
        return {"name": match.group(1), "arguments": arguments}

    def _payload_recovery_diagnostic(
        self,
        payload: Any,
        recovery_format: ToolCallRecoveryFormat,
    ) -> ToolCallDiagnostic | None:
        diagnostic = self._payload_diagnostic(payload)
        if diagnostic is None:
            return None

        details = diagnostic.details.copy()
        details["source_format"] = recovery_format.value
        return ToolCallDiagnostic(
            id=diagnostic.id,
            call_id=diagnostic.call_id,
            requested_name=diagnostic.requested_name,
            canonical_name=diagnostic.canonical_name,
            code=diagnostic.code,
            stage=diagnostic.stage,
            message=diagnostic.message,
            retryable=diagnostic.retryable,
            details=details,
            started_at=diagnostic.started_at,
            finished_at=diagnostic.finished_at,
            duration_ms=diagnostic.duration_ms,
        )

    def _tag_payloads(self, text: str) -> list[Any]:
        body_payloads = self._tag_body_payloads(text)
        self_closing_payloads = self._self_closing_tag_payloads(text)
        if body_payloads:
            return body_payloads + self_closing_payloads

        payloads: list[Any] = []
        try:
            root = ElementTree.fromstring(f"<root>{text}</root>")
            for element in root.findall(".//tool_call"):
                payload = self._tag_payload_from_element(element)
                if payload is not _NO_TAG_PAYLOAD:
                    payloads.append(payload)
            for element in root.findall(".//tool"):
                payload = self._tag_payload_from_element(element)
                if payload is not _NO_TAG_PAYLOAD:
                    payloads.append(payload)
            return payloads
        except ElementTree.ParseError:
            pass

        return self_closing_payloads

    def _tag_body_payloads(self, text: str) -> list[Any]:
        payloads: list[Any] = []
        for start_marker, end_marker in (
            ("<tool_call", "</tool_call>"),
            ("<tool ", "</tool>"),
            ("<tool>", "</tool>"),
        ):
            index = self._find_unquoted_tool_start(text, start_marker, 0)
            while index != -1:
                tag_end = self._tag_end_index(text, index)
                if tag_end == -1:
                    break
                opening = text[index : tag_end + 1]
                close_index = self._find_unquoted_marker(
                    text, end_marker, tag_end + 1
                )
                if close_index == -1:
                    index = self._find_unquoted_tool_start(
                        text, start_marker, tag_end + 1
                    )
                    continue
                if not opening.rstrip().endswith("/>"):
                    payloads.append(
                        self._tag_body_payload(
                            opening,
                            text[tag_end + 1 : close_index],
                        )
                    )
                index = self._find_unquoted_tool_start(
                    text, start_marker, close_index + len(end_marker)
                )
        return payloads

    def _tag_body_payload(self, opening: str, body: str) -> Any:
        name = self._tag_attribute(opening, "name")
        if name is not None:
            arguments = self._deserialize_json_payload(body)
            return {"name": name, "arguments": arguments}
        try:
            return self._deserialize_payload(body)
        except Exception:
            return None

    def _self_closing_tag_payloads(self, text: str) -> list[Any]:
        payloads: list[Any] = []
        for start_marker in ("<tool_call", "<tool "):
            index = self._find_unquoted_tool_start(text, start_marker, 0)
            while index != -1:
                tag_end = self._tag_end_index(text, index)
                if tag_end == -1:
                    break
                opening = text[index : tag_end + 1]
                if opening.rstrip().endswith("/>"):
                    name = self._tag_attribute(opening, "name")
                    if name is not None:
                        arguments_text = self._tag_attribute(
                            opening, "arguments"
                        )
                        arguments = (
                            self._deserialize_tag_payload(arguments_text)
                            if arguments_text is not None
                            else None
                        )
                        payloads.append({"name": name, "arguments": arguments})
                index = self._find_unquoted_tool_start(
                    text, start_marker, tag_end + 1
                )
        return payloads

    def _deserialize_tag_payload(self, text: str) -> Any:
        try:
            return self._deserialize_payload(text)
        except Exception:
            return None

    @classmethod
    def _tag_attribute(cls, opening: str, name: str) -> str | None:
        index = opening.find(name)
        while index != -1:
            after_name = index + len(name)
            if cls._tag_attribute_name_matches(opening, index, after_name):
                equals_index = cls._skip_spaces(opening, after_name)
                if (
                    equals_index < len(opening)
                    and opening[equals_index] == "="
                ):
                    value_index = cls._skip_spaces(opening, equals_index + 1)
                    if value_index < len(opening) and opening[value_index] in (
                        "'",
                        '"',
                    ):
                        close_index = cls._tag_attribute_close_index(
                            opening,
                            value_index + 1,
                            opening[value_index],
                        )
                        if close_index != -1:
                            return opening[value_index + 1 : close_index]
            index = opening.find(name, after_name)
        return None

    @staticmethod
    def _tag_attribute_name_matches(
        opening: str,
        index: int,
        after_name: int,
    ) -> bool:
        before_ok = index == 0 or opening[index - 1].isspace()
        after_ok = after_name == len(opening) or (
            opening[after_name].isspace() or opening[after_name] == "="
        )
        return before_ok and after_ok

    @staticmethod
    def _skip_spaces(text: str, index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    @staticmethod
    def _tag_attribute_close_index(
        opening: str, start: int, quote: str
    ) -> int:
        escaped = False
        for index, character in enumerate(opening[start:], start=start):
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == quote and ToolCallParser._attribute_tail_is_valid(
                opening, index + 1
            ):
                return index
        return -1

    @staticmethod
    def _attribute_tail_is_valid(text: str, index: int) -> bool:
        index = ToolCallParser._skip_spaces(text, index)
        if index >= len(text) or text.startswith((">", "/>"), index):
            return True
        if not (text[index].isalpha() or text[index] in {"_", ":"}):
            return False
        index += 1
        while index < len(text) and (
            text[index].isalnum() or text[index] in {"_", ":", ".", "-"}
        ):
            index += 1
        index = ToolCallParser._skip_spaces(text, index)
        return index < len(text) and text[index] == "="

    @classmethod
    def _find_unquoted_marker(cls, text: str, marker: str, start: int) -> int:
        index = text.find(marker, start)
        while index != -1:
            if not cls._index_is_inside_quoted_text(text, index):
                return index
            index = text.find(marker, index + 1)
        return -1

    @classmethod
    def _find_unquoted_tool_start(
        cls, text: str, marker: str, start: int
    ) -> int:
        index = text.find(marker, start)
        while index != -1:
            if not cls._index_is_inside_quoted_text(
                text, index
            ) and cls._tool_start_marker_boundary_is_valid(
                text, index, marker
            ):
                return index
            index = text.find(marker, index + 1)
        return -1

    @staticmethod
    def _tool_start_marker_needs_boundary(marker: str) -> bool:
        return marker == "<tool_call"

    @classmethod
    def _tool_start_marker_boundary_is_valid(
        cls, text: str, index: int, marker: str
    ) -> bool:
        if not cls._tool_start_marker_needs_boundary(marker):
            return True

        boundary_index = index + len(marker)
        if boundary_index >= len(text):
            return False

        boundary = text[boundary_index]
        return boundary == ">" or boundary == "/" or boundary.isspace()

    def _tag_payload_from_element(self, element: ElementTree.Element) -> Any:
        name = element.attrib.get("name")
        arguments_text = element.attrib.get("arguments")
        if arguments_text is not None:
            arguments = self._deserialize_json_payload(arguments_text)
            if name is None:
                return None
            return {"name": name, "arguments": arguments}

        if element.text is None:
            if name is not None:
                return {"name": name, "arguments": None}
            return _NO_TAG_PAYLOAD

        if name is not None:
            payload = self._deserialize_json_payload(element.text)
            payload = {"name": name, "arguments": payload}
            return payload

        try:
            return self._deserialize_payload(element.text)
        except Exception:
            return None

    @staticmethod
    def _deserialize_json_payload(text: str) -> Any:
        try:
            return loads(text.strip())
        except Exception:
            return None

    @staticmethod
    def _deserialize_payload(text: str) -> Any:
        json_text = text.strip()
        try:
            return loads(json_text)
        except JSONDecodeError:
            try:
                return literal_eval(json_text)
            except (SyntaxError, ValueError):
                return None

    def _payload_diagnostic(self, payload: Any) -> ToolCallDiagnostic | None:
        if not isinstance(payload, dict):
            return self._malformed_call_diagnostic()

        name = payload.get("name")
        requested_name = name if isinstance(name, str) else None
        if not self._is_valid_tool_name(name):
            return self._malformed_call_diagnostic(
                requested_name=(
                    name if isinstance(name, str) and name.strip() else None
                )
            )

        call_id = payload.get("id")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            return self._arguments_diagnostic(
                call_id=call_id if isinstance(call_id, (UUID, str)) else None,
                requested_name=requested_name,
            )
        return self.resource_limit_diagnostic(
            value=arguments,
            maximum_depth=self._maximum_payload_depth,
            maximum_size=self._maximum_payload_size,
            stage=ToolCallDiagnosticStage.PARSE,
            call_id=call_id if isinstance(call_id, (UUID, str)) else None,
            requested_name=requested_name,
        )

    @staticmethod
    def _arguments_diagnostic(
        *,
        call_id: UUID | str | None = None,
        requested_name: str | None = None,
    ) -> ToolCallDiagnostic:
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call_id,
            requested_name=requested_name,
            code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Tool call arguments must be an object.",
        )

    @classmethod
    def resource_limit_diagnostic(
        cls,
        *,
        value: Any,
        maximum_depth: int | None,
        maximum_size: int | None,
        stage: ToolCallDiagnosticStage,
        call_id: UUID | str | None = None,
        requested_name: str | None = None,
        canonical_name: str | None = None,
    ) -> ToolCallDiagnostic | None:
        assert isinstance(stage, ToolCallDiagnosticStage)
        if maximum_depth is not None:
            assert isinstance(maximum_depth, int)
            assert not isinstance(maximum_depth, bool)
            assert maximum_depth > 0
            depth = cls._value_depth(value)
            if depth > maximum_depth:
                return cls._resource_limit_diagnostic(
                    code=ToolCallDiagnosticCode.MAXIMUM_DEPTH,
                    stage=stage,
                    message="Tool call arguments exceed the maximum depth.",
                    call_id=call_id,
                    requested_name=requested_name,
                    canonical_name=canonical_name,
                    details={"limit": maximum_depth, "depth": depth},
                )

        if maximum_size is not None:
            assert isinstance(maximum_size, int)
            assert not isinstance(maximum_size, bool)
            assert maximum_size > 0
            size = cls._value_size(value)
            if size > maximum_size:
                return cls._resource_limit_diagnostic(
                    code=ToolCallDiagnosticCode.MAXIMUM_SIZE,
                    stage=stage,
                    message="Tool call arguments exceed the maximum size.",
                    call_id=call_id,
                    requested_name=requested_name,
                    canonical_name=canonical_name,
                    details={"limit": maximum_size, "size": size},
                )
        return None

    @staticmethod
    def _resource_limit_diagnostic(
        *,
        code: ToolCallDiagnosticCode,
        stage: ToolCallDiagnosticStage,
        message: str,
        call_id: UUID | str | None = None,
        requested_name: str | None = None,
        canonical_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ToolCallDiagnostic:
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call_id,
            requested_name=requested_name,
            canonical_name=canonical_name,
            code=code,
            stage=stage,
            message=message,
            details=details or {},
        )

    @classmethod
    def _value_depth(cls, value: Any) -> int:
        if isinstance(value, dict):
            if not value:
                return 1
            return 1 + max(cls._value_depth(v) for v in value.values())
        if isinstance(value, list):
            if not value:
                return 1
            return 1 + max(cls._value_depth(v) for v in value)
        return 0

    @classmethod
    def _value_size(cls, value: Any) -> int:
        if isinstance(value, dict):
            return sum(
                cls._text_size(k) + cls._value_size(v)
                for k, v in value.items()
            )
        if isinstance(value, list):
            return sum(cls._value_size(v) for v in value)
        if isinstance(value, str):
            return cls._text_size(value)
        if value is None:
            return 4
        if isinstance(value, bool):
            return 4 if value else 5
        return cls._text_size(str(value))

    @staticmethod
    def _text_size(text: str) -> int:
        return len(text.encode())

    @staticmethod
    def _malformed_call_diagnostic(
        *,
        call_id: UUID | str | None = None,
        requested_name: str | None = None,
        message: str = "Tool call could not be parsed.",
        details: dict[str, ToolValue] | None = None,
        recovery_format: ToolCallRecoveryFormat | None = None,
    ) -> ToolCallDiagnostic:
        diagnostic_details = details.copy() if details is not None else {}
        if recovery_format is not None:
            diagnostic_details["source_format"] = recovery_format.value
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call_id,
            requested_name=requested_name,
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message=message,
            details=diagnostic_details,
        )

    @staticmethod
    def _is_valid_tool_name(name: Any) -> bool:
        return isinstance(name, str) and bool(
            _TOOL_NAME_PATTERN.fullmatch(name)
        )

    @staticmethod
    def _tool_call_from_payload(payload: Any) -> ToolCall | None:
        if not isinstance(payload, dict):
            return None

        name = payload.get("name")
        arguments = payload.get("arguments")
        if not ToolCallParser._is_valid_tool_name(name) or not isinstance(
            arguments, dict
        ):
            return None
        assert isinstance(name, str)

        call_id = payload.get("id")
        identifier: UUID | str = (
            call_id if isinstance(call_id, (UUID, str)) else uuid4()
        )
        return ToolCall(id=identifier, name=name, arguments=arguments)
