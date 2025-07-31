from .message import TemplateMessage, TemplateMessageRole
from .stream import TextGenerationStream
from ..entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentImage,
    MessageContentText,
    MessageRole,
)
from ..tool.manager import ToolManager
from abc import ABC
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
        def _content(content: str | MessageContent) -> dict:
            return (
                {"type": str(content.type), "image_url": content.image_url}
                if isinstance(content, MessageContentImage)
                else {
                    "type": "text",
                    "text": (
                        content.text
                        if isinstance(content, MessageContentText)
                        else str(content)
                    ),
                }
            )

        return [
            {
                "role": str(message.role),
                "content": (
                    [_content(c) for c in message.content]
                    if isinstance(message.content, list)
                    else (
                        str(message.content)
                        if message.role == MessageRole.SYSTEM
                        else _content(message.content)
                    )
                ),
            }
            for message in messages
            if not exclude_roles or message.role not in exclude_roles
        ]


class TextGenerationVendorStream(TextGenerationStream):
    _generator: AsyncGenerator

    def __init__(self, generator: AsyncGenerator):
        self._generator = generator

    def __call__(self, *args, **kwargs):
        return self.__aiter__()

    def __aiter__(self):
        assert self._generator
        return self
