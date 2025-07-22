from .message import TemplateMessage, TemplateMessageRole
from .stream import TextGenerationStream
from ..entities import GenerationSettings, Message
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
        return [
            {"role": message.role, "content": message.content}
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
