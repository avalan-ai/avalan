from ..entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
    ToolValue,
)
from ..tool.manager import ToolManager
from .message import (
    TemplateMessage,
    TemplateMessageContent,
    TemplateMessageRole,
)
from .stream import TextGenerationStream

from abc import ABC
from json import JSONDecodeError, dumps, loads
from typing import Any, AsyncIterator, cast


class TextGenerationVendor(ABC):
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream | AsyncIterator[Token | TokenDetail | str]:
        raise NotImplementedError()

    def _system_prompt(self, messages: list[Message]) -> str | None:
        for message in messages:
            if message.role != "system":
                continue
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, MessageContentText):
                return content.text
            return None
        return None

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage] | list[dict[str, Any]]:
        def _block(c: MessageContent) -> dict[str, Any]:
            if isinstance(c, MessageContentFile):
                return {"type": "file", "file": dict(c.file)}
            if isinstance(c, MessageContentImage):
                return {"type": "image_url", "image_url": c.image_url}
            return {"type": "text", "text": c.text}

        def _wrap(
            content: str | MessageContent | list[MessageContent] | None,
        ) -> str | list[dict[str, Any]]:
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return [_block(c) for c in content]

            if isinstance(content, MessageContentText):
                return content.text

            if isinstance(content, MessageContentImage):
                return [_block(content)]

            if isinstance(content, MessageContentFile):
                return [_block(content)]

            return str(content)

        out: list[TemplateMessage] = []
        for msg in messages:
            if exclude_roles and msg.role in exclude_roles:
                continue

            out.append(
                {
                    "role": cast(TemplateMessageRole, str(msg.role)),
                    "content": cast(
                        str
                        | TemplateMessageContent
                        | list[TemplateMessageContent],
                        _wrap(msg.content),
                    ),
                }
            )

        return out

    @staticmethod
    def encode_tool_name(tool_name: str) -> str:
        return tool_name.replace(".", "__")

    @staticmethod
    def decode_tool_name(tool_name: str) -> str:
        return tool_name.replace("__", ".")

    @staticmethod
    def build_tool_call_token(
        call_id: str | object | None,
        tool_name: str | object | None,
        arguments: str | dict[str, Any] | object | None,
    ) -> ToolCallToken:
        tool_name_text = (
            tool_name if isinstance(tool_name, str) else str(tool_name or "")
        )
        name = TextGenerationVendor.decode_tool_name(tool_name_text)
        if isinstance(arguments, str):
            try:
                args = cast(dict[str, Any], loads(arguments))
            except JSONDecodeError:
                args = {}
        else:
            args = (
                arguments
                if isinstance(arguments, dict)
                else cast(dict[str, Any], {})
            )
        call_id_value = (
            call_id
            if isinstance(call_id, str) or call_id is None
            else str(call_id)
        )
        call = ToolCall(
            id=cast(Any, call_id_value),
            name=name,
            arguments=cast(dict[str, ToolValue], args),
        )
        token_payload: dict[str, Any] = {
            "name": name,
            "arguments": args,
        }
        if call_id_value is not None:
            token_payload["id"] = call_id_value
        token_json = dumps(token_payload)
        return ToolCallToken(
            token=f"<tool_call>{token_json}</tool_call>", call=call
        )


class TextGenerationVendorStream(TextGenerationStream):
    _generator: AsyncIterator[Token | TokenDetail | str]

    def __init__(
        self, generator: AsyncIterator[Token | TokenDetail | str]
    ) -> None:
        self._generator = generator

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Token | TokenDetail | str]:
        return self.__aiter__()

    def __aiter__(self) -> AsyncIterator[Token | TokenDetail | str]:
        assert self._generator
        return self

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()
