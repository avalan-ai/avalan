from ..entities import MessageRole, MessageToolCall, ToolCall, ToolValue

import html
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class DsmlPromptMessage:
    """Represent a message in DSML prompt rendering."""

    role: MessageRole
    content: str
    reasoning: str | None = None
    tool_calls: tuple[MessageToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class DsmlParseResult:
    """Represent parsed DSML content and replay metadata."""

    content: str
    calls: tuple[ToolCall, ...]
    reasoning: str | None
    raw_dsml: str | None = None


class DsmlTools:
    """Render and parse DSML tool-call blocks."""

    TOOL_CALLS_START = "<｜DSML｜tool_calls>"
    TOOL_CALLS_END = "</｜DSML｜tool_calls>"
    TOOL_CALL_START_PREFIXES = (
        "<｜DSML｜tool_calls",
        "<DSML｜tool_calls",
        "<tool_calls",
    )
    TOOL_CALL_END_MARKERS = (
        "</｜DSML｜tool_calls>",
        "</DSML｜tool_calls>",
        "</tool_calls>",
    )
    _INVOKE_START_RE = re.compile(
        r"<(?:｜DSML｜|DSML｜)?invoke\s+name=\"([^\"]+)\"\s*>",
        re.DOTALL,
    )
    _PARAM_RE = re.compile(
        r"<(?:｜DSML｜|DSML｜)?parameter\s+"
        r"name=\"([^\"]+)\"(?:\s+string=\"(true|false)\")?\s*>"
        r"(.*?)"
        r"</(?:｜DSML｜|DSML｜)?parameter>",
        re.DOTALL,
    )

    @classmethod
    def render_prompt(
        cls,
        system_content: str | None,
        messages: list[DsmlPromptMessage],
        tool_schemas: str | None,
        think_mode: object,
        replay_lookup: (
            Callable[[tuple[MessageToolCall, ...]], str | None] | None
        ) = None,
    ) -> str:
        """Return a rendered DSML chat prompt including tool context."""
        system_parts = [system_content] if system_content else []
        if tool_schemas:
            system_parts.append(cls.tools_prompt(tool_schemas))

        rendered = [
            "<｜begin▁of▁sentence｜>",
            "\n\n".join(system_parts),
        ]
        pending_assistant = False
        pending_tool_result = False
        think = cls._thinking_enabled(think_mode)
        tool_context = bool(tool_schemas) or any(
            message.tool_calls or message.role is MessageRole.TOOL
            for message in messages
        )
        last_user_index = max(
            (
                index
                for index, message in enumerate(messages)
                if message.role in {MessageRole.USER, MessageRole.TOOL}
            ),
            default=-1,
        )

        for index, message in enumerate(messages):
            if message.role is MessageRole.USER:
                rendered.extend(("<｜User｜>", message.content))
                pending_assistant = True
                pending_tool_result = False
            elif message.role is MessageRole.TOOL:
                if not pending_tool_result:
                    rendered.append("<｜User｜>")
                rendered.append(cls.render_tool_result(message.content))
                pending_assistant = True
                pending_tool_result = True
            elif message.role is MessageRole.ASSISTANT:
                if pending_assistant:
                    rendered.append("<｜Assistant｜>")
                    if think:
                        if tool_context or index > last_user_index:
                            rendered.extend(
                                (
                                    "<think>",
                                    message.reasoning or "",
                                    "</think>",
                                )
                            )
                        else:
                            rendered.append("</think>")
                    else:
                        rendered.append("</think>")
                rendered.append(message.content)
                rendered.append(
                    cls.render_tool_calls(message.tool_calls, replay_lookup)
                )
                rendered.append("<｜end▁of▁sentence｜>")
                pending_assistant = False
                pending_tool_result = False

        if pending_assistant:
            rendered.append("<｜Assistant｜>")
            rendered.append("<think>" if think else "</think>")
        return "".join(rendered)

    @classmethod
    def render_tool_schemas(
        cls, schemas: list[dict[str, object]] | None
    ) -> str | None:
        """Return DSML newline-delimited tool schemas."""
        if not schemas:
            return None
        lines = []
        for schema in schemas:
            function_schema = (
                schema.get("function")
                if schema.get("type") == "function"
                and isinstance(schema.get("function"), dict)
                else schema
            )
            lines.append(
                json.dumps(
                    function_schema,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=False,
                )
            )
        return "\n".join(lines)

    @classmethod
    def render_tool_calls(
        cls,
        calls: tuple[MessageToolCall, ...],
        replay_lookup: (
            Callable[[tuple[MessageToolCall, ...]], str | None] | None
        ) = None,
    ) -> str:
        """Return canonical DSML text for assistant tool calls."""
        if not calls:
            return ""
        if replay_lookup is not None:
            replay = replay_lookup(calls)
            if replay is not None:
                return replay
        parts = ["\n\n", cls.TOOL_CALLS_START, "\n"]
        for call in calls:
            parts.extend(
                (
                    '<｜DSML｜invoke name="',
                    cls._escape_attr(call.name),
                    '">\n',
                )
            )
            arguments = (
                call.arguments
                if isinstance(call.arguments, dict)
                else {"arguments": call.arguments}
            )
            for name, value in arguments.items():
                parts.append(cls._render_parameter(str(name), value))
            parts.append("</｜DSML｜invoke>\n")
        parts.append(cls.TOOL_CALLS_END)
        return "".join(parts)

    @classmethod
    def render_tool_result(cls, content: str) -> str:
        """Return canonical DSML text for a tool result."""
        return f"<tool_result>{cls._escape_text(content)}</tool_result>"

    @classmethod
    def parse_tool_calls(cls, text: str) -> list[ToolCall] | None:
        """Return DSML tool calls parsed from ``text``."""
        parsed = cls.parse_generated_message(text)
        if parsed is None or not parsed.calls:
            return None
        return list(parsed.calls)

    @classmethod
    def parse_generated_message(cls, text: str) -> DsmlParseResult | None:
        """Parse generated DSML text into content, calls, and metadata."""
        start_match = re.search(
            r"\n?\n?<(?:(?:｜DSML｜|DSML｜)tool_calls|tool_calls)>",
            text,
        )
        if not start_match:
            content, reasoning = cls.split_reasoning(text)
            return DsmlParseResult(content, (), reasoning)

        end_match = re.search(
            r"</(?:(?:｜DSML｜|DSML｜)tool_calls|tool_calls)>",
            text[start_match.end() :],
        )
        if not end_match:
            return None

        content, reasoning = cls.split_reasoning(
            text[: start_match.start()].rstrip()
        )
        block_start = start_match.end()
        block_end = block_start + end_match.start()
        raw_end = start_match.end() + end_match.end()
        block = text[block_start:block_end]
        calls = cls._parse_calls(block)
        return DsmlParseResult(
            content,
            tuple(calls),
            reasoning,
            text[start_match.start() : raw_end],
        )

    @classmethod
    def split_reasoning(cls, text: str) -> tuple[str, str | None]:
        """Return visible content and optional DSML thinking text."""
        if text.startswith("<think>") and "</think>" in text:
            reasoning, content = text.removeprefix("<think>").split(
                "</think>", 1
            )
            return content, reasoning
        return text, None

    @classmethod
    def tools_prompt(cls, tool_schemas: str) -> str:
        """Return DSML tool-use instructions for a system prompt."""
        return (
            "## Tools\n\n"
            "You have access to a set of tools to help answer the user "
            "question. You can invoke tools by writing a "
            '"<｜DSML｜tool_calls>" block like the following:\n\n'
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="$TOOL_NAME">\n'
            '<｜DSML｜parameter name="$PARAMETER_NAME" '
            'string="true|false">$PARAMETER_VALUE'
            "</｜DSML｜parameter>\n"
            "...\n"
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="$TOOL_NAME2">\n'
            "...\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n\n"
            "String parameters should be specified as raw text and set "
            '`string="true"`. Preserve characters such as `>`, `&`, and '
            "`&&` exactly; never replace normal string characters with XML "
            "or HTML entity escapes. Only if a string value itself contains "
            "the exact closing parameter tag `</｜DSML｜parameter>`, write "
            "that tag as `&lt;/｜DSML｜parameter>` inside the value. For all "
            "other types (numbers, booleans, arrays, objects), pass the "
            'value in JSON format and set `string="false"`.\n\n'
            "If thinking_mode is enabled (triggered by <think>), you MUST "
            "output your complete reasoning inside <think>...</think> "
            "BEFORE any tool calls or final response.\n\n"
            "Otherwise, output directly after </think> with tool calls or "
            "final response.\n\n"
            "### Available Tool Schemas\n\n"
            f"{tool_schemas}\n\n"
            "You MUST strictly follow the above defined tool name and "
            "parameter schemas to invoke tool calls. Use the exact parameter "
            "names from the schemas."
        )

    @staticmethod
    def _thinking_enabled(think_mode: object) -> bool:
        value = getattr(think_mode, "value", think_mode)
        return value in {"high", "max"}

    @classmethod
    def _parse_calls(cls, block: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        position = 0
        while True:
            match = cls._INVOKE_START_RE.search(block, position)
            if not match:
                return calls
            invoke_end = re.search(
                r"</(?:｜DSML｜|DSML｜)?invoke>", block[match.end() :]
            )
            if not invoke_end:
                return []
            body_end = match.end() + invoke_end.start()
            body = block[match.end() : body_end]
            arguments: dict[str, ToolValue] = {}
            for parameter in cls._PARAM_RE.finditer(body):
                name = html.unescape(parameter.group(1))
                raw_value = parameter.group(3)
                if parameter.group(2) == "false":
                    value = cls._json_value(raw_value)
                else:
                    value = html.unescape(raw_value)
                arguments[name] = value
            calls.append(
                ToolCall(
                    id=f"ds4_tool_{uuid4().hex}",
                    name=html.unescape(match.group(1)),
                    arguments=arguments,
                )
            )
            position = body_end + invoke_end.end()

    @staticmethod
    def _json_value(value: str) -> ToolValue:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return cast(ToolValue, parsed)

    @classmethod
    def _render_parameter(cls, name: str, value: object) -> str:
        is_string = isinstance(value, str)
        rendered_value = (
            cls._escape_parameter_text(cast(str, value))
            if is_string
            else cls._escape_json_literal(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=False,
                )
            )
        )
        return (
            f'<｜DSML｜parameter name="{cls._escape_attr(name)}" '
            f'string="{"true" if is_string else "false"}">'
            f"{rendered_value}</｜DSML｜parameter>\n"
        )

    @staticmethod
    def _escape_attr(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _escape_text(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @staticmethod
    def _escape_parameter_text(value: str) -> str:
        return value.replace("</｜DSML｜parameter>", "&lt;/｜DSML｜parameter>")

    @staticmethod
    def _escape_json_literal(value: str) -> str:
        return value.replace(
            "</｜DSML｜parameter>", "\\u003c/｜DSML｜parameter>"
        )
