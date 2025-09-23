from ..entities import (
    Message,
    MessageContent,
    MessageContentText,
    ToolCall,
    ToolFormat,
)
from ast import literal_eval
from dataclasses import dataclass, field
from enum import Enum
from json import JSONDecodeError, loads
from re import DOTALL, compile, finditer, search, sub
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree

_HARMONY_SEGMENT_PATTERN = compile(
    r"<\|channel\|>(?P<channel>\w+).*?<\|message\|>(?P<message>.*?)(?:<\|call\|>|<\|end\|>)",
    DOTALL,
)
_SPECIAL_TOKEN_PATTERN = compile(r"<\|[^>]+?\|>")


class ToolCallParser:
    _eos_token: str | None
    _tool_format: ToolFormat | None

    def __init__(
        self,
        tool_format: ToolFormat | None = None,
        eos_token: str | None = None,
    ) -> None:
        self._tool_format = tool_format
        self._eos_token = eos_token

    @property
    def tool_format(self) -> ToolFormat | None:
        """Return the tool format used by the parser."""
        return self._tool_format

    def __call__(self, text: str) -> list[ToolCall] | None:
        calls = (
            self._parse_json(text)
            if self._tool_format is ToolFormat.JSON
            else (
                self._parse_react(text)
                if self._tool_format is ToolFormat.REACT
                else (
                    self._parse_bracket(text)
                    if self._tool_format is ToolFormat.BRACKET
                    else (
                        self._parse_openai_json(text)
                        if self._tool_format is ToolFormat.OPENAI
                        else (
                            self._parse_harmony(text)
                            if self._tool_format is ToolFormat.HARMONY
                            else None
                        )
                    )
                )
            )
        )
        if not calls:
            calls = self._parse_tag(text)
        return calls

    def set_eos_token(self, eos_token: str) -> None:
        self._eos_token = eos_token

    def is_potential_tool_call(self, buffer: str, token_str: str) -> bool:
        """Return ``True`` if tool detection should run for ``token_str``.

        This provides a fast check during streaming. If ``token_str`` is empty
        or only whitespace, ``False`` is returned since nothing new was added
        that could form a tool call.
        """
        return bool(token_str and token_str.strip())

    @dataclass(frozen=True, kw_only=True, slots=True)
    class StructuredMessage:
        """Structured content extracted from a raw message payload."""

        content: str
        thinking: str | None = None
        tool_calls: list[dict[str, object]] = field(default_factory=list)

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
        parsed = None
        if "<|call|>" in text and "<|channel|>" in text:
            parsed = self._parse_harmony(text)
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
            ]

        if isinstance(parsed, tuple) and len(parsed) == 2:
            name, arguments = parsed
            if isinstance(name, str):
                return [
                    {
                        "id": None,
                        "name": name,
                        "arguments": arguments or {},
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
                part for part in (existing_thinking, thinking) if part
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

    def tool_call_status(
        self, buffer: str
    ) -> "ToolCallParser.ToolCallBufferStatus":
        start = ["<tool_call", "<tool ", "<tool>"]
        end = ["</tool_call>", "</tool>", "/>", "<|call|>"]
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
        tail = buffer[-max_len:]
        for s in start:
            if s.startswith(tail) and tail != s:
                return self.ToolCallBufferStatus.PREFIX
        for s in start:
            idx = buffer.rfind(s)
            if idx != -1:
                after = buffer[idx + len(s) :]
                if any(e in after for e in end):
                    return self.ToolCallBufferStatus.CLOSED
                return self.ToolCallBufferStatus.OPEN
        return self.ToolCallBufferStatus.NONE

    def _parse_json(self, text: str) -> tuple[str, dict[str, Any]] | None:
        try:
            payload = loads(text)
            return payload["tool"], payload.get("arguments", {})
        except Exception:
            return None

    def _parse_react(self, text: str) -> tuple[str, dict[str, Any]] | None:
        act = search(r"Action:\s*(\w+)", text)
        inp = search(r"Action Input:\s*({.*})", text, DOTALL)
        if act and inp:
            try:
                return act.group(1), loads(inp.group(1))
            except JSONDecodeError:
                pass
        return None

    def _parse_bracket(self, text: str) -> tuple[str, dict[str, Any]] | None:
        m = search(r"\[(\w+)\]\(([^)]+)\)", text)
        if m:
            return m.group(1), {"input": m.group(2)}
        return None

    def _parse_openai_json(
        self, text: str
    ) -> tuple[str, dict[str, Any]] | None:
        try:
            payload = loads(text)
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

    def _parse_tag(self, text: str) -> tuple[str, dict[str, Any]] | None:
        tool_calls: list[ToolCall] = []

        if self._eos_token:
            text = text.strip().removesuffix(self._eos_token)
        try:
            root = ElementTree.fromstring(f"<root>{text}</root>")
            for element in root.findall(".//tool_call"):
                tool_call = None
                try:
                    json_text = element.text.strip()

                    try:
                        tool_call = loads(json_text)
                    except JSONDecodeError:
                        try:
                            tool_call = literal_eval(json_text)
                        except (SyntaxError, ValueError):
                            continue
                except Exception:
                    pass

                if (
                    tool_call is not None
                    and "name" in tool_call
                    and "arguments" in tool_call
                ):
                    tool_calls.append(
                        ToolCall(
                            id=uuid4(),
                            name=tool_call["name"],
                            arguments=tool_call["arguments"],
                        )
                    )
        except ElementTree.ParseError:
            pass

        if tool_calls:
            return tool_calls

        m = search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=DOTALL
        )
        if m:
            tool_call_payload = m.group(1)
            try:
                tool_call = loads(tool_call_payload)
                if (
                    tool_call
                    and "name" in tool_call
                    and "arguments" in tool_call
                ):
                    tool_calls.append(
                        ToolCall(
                            id=uuid4(),
                            name=tool_call["name"],
                            arguments=tool_call["arguments"],
                        )
                    )
            except JSONDecodeError:
                pass

        m = search(
            r"<tool_call\s+name=\"([^\"]+)\"\s*>(\{.*?\})</tool_call>",
            text,
            DOTALL,
        )
        if m:
            try:
                tool_calls.append(
                    ToolCall(
                        id=uuid4(),
                        name=m.group(1),
                        arguments=loads(m.group(2)),
                    )
                )
            except JSONDecodeError:
                pass

        m = search(
            r"<tool\s+name=\"([^\"]+)\"\s*>(\{.*?\})</tool>",
            text,
            DOTALL,
        )
        if m:
            try:
                tool_calls.append(
                    ToolCall(
                        id=uuid4(),
                        name=m.group(1),
                        arguments=loads(m.group(2)),
                    )
                )
            except JSONDecodeError:
                pass

        m = search(
            r"<tool_call\s+name=\"([^\"]+)\"\s+arguments='(\{.*?\})'\s*/>",
            text,
        )
        if m:
            try:
                tool_calls.append(
                    ToolCall(
                        id=uuid4(),
                        name=m.group(1),
                        arguments=loads(m.group(2)),
                    )
                )
            except JSONDecodeError:
                pass

        return tool_calls if tool_calls else None
