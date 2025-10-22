from ..entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentImage,
    MessageContentText,
    ToolCall,
    ToolCallToken,
)
from ..tool.manager import ToolManager
from .message import TemplateMessage, TemplateMessageRole
from .stream import TextGenerationStream

from abc import ABC
from json import JSONDecodeError, dumps, loads
from typing import AsyncGenerator


class TextGenerationVendor(ABC):
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream:
        raise NotImplementedError()

    def _system_prompt(self, messages: list[Message]) -> str | None:
        return next(
            (
                message.content
                for message in messages
                if message.role == "system"
            ),
            None,
        )

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage]:
        def _block(c: MessageContent) -> dict:
            if isinstance(c, MessageContentImage):
                return {"type": "image_url", "image_url": c.image_url}
            return {"type": "text", "text": c.text}

        def _wrap(
            content: str | MessageContent | list[MessageContent],
        ) -> str | list[dict]:
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return [_block(c) for c in content]

            if isinstance(content, MessageContentText):
                return content.text

            if isinstance(content, MessageContentImage):
                return [_block(content)]

            return str(content)

        out: list[TemplateMessage] = []
        for msg in messages:
            if exclude_roles and msg.role in exclude_roles:
                continue

            out.append({"role": str(msg.role), "content": _wrap(msg.content)})

        return out

    @staticmethod
    def encode_tool_name(tool_name: str) -> str:
        return tool_name.replace(".", "__")

    @staticmethod
    def decode_tool_name(tool_name: str) -> str:
        return tool_name.replace("__", ".")

    @staticmethod
    def build_tool_call_token(
        call_id: str | None,
        tool_name: str | None,
        arguments: str | dict | None,
    ) -> ToolCallToken:
        name = TextGenerationVendor.decode_tool_name(tool_name or "")
        if isinstance(arguments, str):
            try:
                args: dict = loads(arguments)
            except JSONDecodeError:
                args = {}
        else:
            args = arguments or {}
        call = ToolCall(id=call_id, name=name, arguments=args)
        token_json = dumps({"name": name, "arguments": args})
        return ToolCallToken(
            token=f"<tool_call>{token_json}</tool_call>", call=call
        )


class TextGenerationVendorStream(TextGenerationStream):
    _generator: AsyncGenerator

    def __init__(self, generator: AsyncGenerator):
        self._generator = generator

    def __call__(self, *args, **kwargs):
        return self.__aiter__()

    def __aiter__(self):
        assert self._generator
        return self
