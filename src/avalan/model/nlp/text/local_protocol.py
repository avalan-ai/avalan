"""Define the exact structured-output protocol for local text models."""

from ....model.stream import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID,
    LocalTextStreamEventParser,
    TextGenerationNonStreamResult,
    local_tool_call_control_frame,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import dumps
from typing import final

LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME = "avalan_local_tool_call_v1"


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        "Local structured-output schemas must contain only JSON values."
    )


@final
@dataclass(frozen=True, slots=True)
class LocalStructuredOutputProtocol:
    """Adapt one explicitly compatible tokenizer to Avalan control frames."""

    protocol_id: str = LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID
    tokenizer_template_name: str = LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME

    def __post_init__(self) -> None:
        assert self.protocol_id == LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID
        assert (
            self.tokenizer_template_name
            == LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME
        )

    def tokenizer_template(
        self,
        tokenizer: object,
        requested_chat_template: str | None = None,
    ) -> str | None:
        """Return the tokenizer's explicit adapter template when compatible."""
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        templates = getattr(tokenizer, "chat_template", None)
        if not callable(apply_chat_template) or not isinstance(
            templates, Mapping
        ):
            return None
        template = templates.get(self.tokenizer_template_name)
        if not isinstance(template, str) or not template.strip():
            return None
        if (
            requested_chat_template is not None
            and requested_chat_template != template
        ):
            return None
        return template

    def instruction(self, schemas: Sequence[Mapping[str, object]]) -> str:
        """Return the exact producer grammar and advertised schemas."""
        assert isinstance(schemas, Sequence)
        schema_json = dumps(
            _json_ready(schemas),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            f"Avalan local structured-output protocol: {self.protocol_id}.\n"
            "To invoke one advertised capability, emit exactly one frame "
            "and no surrounding prose or Markdown:\n"
            "<tool_call id=JSON_STRING name=JSON_STRING>"
            "JSON_OBJECT</tool_call>\n"
            "The id must be a non-empty JSON string containing the real call "
            "ID. The name must be a non-empty JSON string exactly equal to "
            "an advertised function name. The body must be only the JSON "
            "object of arguments. Never put id, name, or arguments in a "
            "legacy JSON wrapper body. Preserve the same id and name in "
            "correlated continuation results.\n"
            f"Advertised capability schemas: {schema_json}"
        )

    def prepare_messages(
        self,
        messages: Sequence[Mapping[str, object]],
        schemas: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Prepend the exact protocol instruction to model messages."""
        prepared = [dict(message) for message in messages]
        return [
            {
                "role": "system",
                "content": self.instruction(schemas),
            },
            *prepared,
        ]

    def parser(self) -> LocalTextStreamEventParser:
        """Return a consumer that accepts only exact Avalan frames."""
        return LocalTextStreamEventParser(parse_tool_calls=True)

    def non_stream_result(
        self,
        text: str,
        *,
        provider_family: str,
        provider_event_type: str,
    ) -> TextGenerationNonStreamResult:
        """Consume exact frames from one completed local response."""
        return TextGenerationNonStreamResult.from_local_text(
            text,
            provider_family=provider_family,
            provider_event_type=provider_event_type,
        )

    def control_frame(
        self,
        call_id: str,
        name: str,
        arguments: Mapping[str, object],
    ) -> str:
        """Encode one exact frame for a protocol-aware producer."""
        return local_tool_call_control_frame(call_id, name, arguments)


LOCAL_STRUCTURED_OUTPUT_PROTOCOL = LocalStructuredOutputProtocol()
