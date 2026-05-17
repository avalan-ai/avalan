from ..entities import MessageRole, MessageToolCall, ToolCall, ToolValue

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, cast
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
    """Render and parse DSML tool-call blocks through pyds4."""

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
        dsml = cls._pyds4_dsml()
        prompt = dsml.DsmlPrompt(
            system_content=system_content,
            messages=[
                cls._to_pyds4_message(dsml, message) for message in messages
            ],
            tool_schemas=tool_schemas or (),
        )
        return cast(
            str,
            dsml.render_prompt(
                prompt,
                think_mode,
                replay_lookup=cls._replay_adapter(replay_lookup),
            ),
        )

    @classmethod
    def render_tool_schemas(
        cls, schemas: list[dict[str, object]] | None
    ) -> str | None:
        """Return DSML newline-delimited tool schemas."""
        return cast(str | None, cls._pyds4_dsml().tool_schema_text(schemas))

    @classmethod
    def render_tool_calls(
        cls,
        calls: tuple[MessageToolCall, ...],
        replay_lookup: (
            Callable[[tuple[MessageToolCall, ...]], str | None] | None
        ) = None,
    ) -> str:
        """Return canonical DSML text for assistant tool calls."""
        dsml = cls._pyds4_dsml()
        native_calls = tuple(cls._to_pyds4_call(dsml, call) for call in calls)
        return cast(
            str,
            dsml.render_tool_calls(
                native_calls,
                cls._replay_adapter(replay_lookup),
            ),
        )

    @classmethod
    def render_tool_result(cls, content: str) -> str:
        """Return canonical DSML text for a tool result."""
        return cast(str, cls._pyds4_dsml().render_tool_result(content))

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
        parsed = cls._pyds4_dsml().parse_generated_message(text)
        status = getattr(parsed.status, "value", parsed.status)
        if status != "complete":
            return None
        calls = tuple(cls._to_tool_call(call) for call in parsed.calls)
        return DsmlParseResult(
            parsed.content,
            calls,
            parsed.reasoning,
            parsed.raw_dsml,
        )

    @classmethod
    def tool_call_start_span(cls, text: str) -> tuple[int, int] | None:
        """Return the first generated DSML tool-call block start span."""
        return cast(
            tuple[int, int] | None,
            cls._pyds4_dsml().tool_call_start_span(text),
        )

    @classmethod
    def tool_call_start_suffix_length(cls, text: str) -> int:
        """Return trailing text length that may become a DSML start marker."""
        return cast(
            int,
            cls._pyds4_dsml().tool_call_start_suffix_length(text),
        )

    @classmethod
    def tool_call_buffer_status(cls, text: str) -> str:
        """Return pyds4's DSML buffer status value for ``text``."""
        status = cls._pyds4_dsml().tool_call_buffer_status(text)
        return cast(str, getattr(status, "value", status))

    @classmethod
    def stream_argument_deltas(
        cls, raw_dsml: str, emitted_until: int
    ) -> tuple[tuple[str, ...], int]:
        """Return new DSML parameter-value deltas from ``raw_dsml``."""
        return cast(
            tuple[tuple[str, ...], int],
            cls._pyds4_dsml().stream_argument_deltas(
                raw_dsml,
                emitted_until,
            ),
        )

    @classmethod
    def split_reasoning(cls, text: str) -> tuple[str, str | None]:
        """Return visible content and optional DSML thinking text."""
        return cast(
            tuple[str, str | None],
            cls._pyds4_dsml().split_reasoning(text),
        )

    @classmethod
    def tools_prompt(cls, tool_schemas: str) -> str:
        """Return DSML tool-use instructions for a system prompt."""
        rendered = cls._pyds4_dsml().tools_prompt(tool_schemas)
        if rendered is None:
            raise ValueError("tool_schemas must not be empty.")
        return cast(str, rendered)

    @staticmethod
    def _pyds4_dsml() -> ModuleType:
        try:
            return import_module("pyds4.dsml")
        except ModuleNotFoundError as error:
            if error.name is not None and error.name not in {
                "pyds4",
                "pyds4.dsml",
            }:
                raise
            raise RuntimeError(
                "Avalan DSML helpers require pyds4. Install avalan[ds4]."
            ) from error

    @classmethod
    def _replay_adapter(
        cls,
        replay_lookup: (
            Callable[[tuple[MessageToolCall, ...]], str | None] | None
        ),
    ) -> Callable[[tuple[object, ...]], str | None] | None:
        if replay_lookup is None:
            return None

        def replay(calls: tuple[object, ...]) -> str | None:
            avalan_calls = tuple(
                cls._to_message_tool_call(call) for call in calls
            )
            return replay_lookup(avalan_calls)

        return replay

    @classmethod
    def _to_pyds4_message(
        cls,
        dsml: ModuleType,
        message: DsmlPromptMessage,
    ) -> object:
        return dsml.DsmlMessage(
            role=message.role.value,
            content=message.content,
            reasoning=message.reasoning,
            tool_calls=tuple(
                cls._to_pyds4_call(dsml, call) for call in message.tool_calls
            ),
        )

    @staticmethod
    def _to_pyds4_call(dsml: ModuleType, call: MessageToolCall) -> object:
        return dsml.DsmlToolCall(
            id=call.id,
            name=call.name,
            arguments=cast(object, call.arguments),
        )

    @staticmethod
    def _to_message_tool_call(call: object) -> MessageToolCall:
        return MessageToolCall(
            id=cast(str | None, getattr(call, "id", None)),
            name=cast(str, getattr(call, "name")),
            arguments=cast(Any, getattr(call, "arguments", {})),
        )

    @staticmethod
    def _to_tool_call(call: object) -> ToolCall:
        arguments = getattr(call, "arguments", None)
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(
            id=f"ds4_tool_{uuid4().hex}",
            name=cast(str, getattr(call, "name")),
            arguments=cast(dict[str, ToolValue], arguments),
        )
