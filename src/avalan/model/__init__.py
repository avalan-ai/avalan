from abc import ABC, abstractmethod
from avalan.model.entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail
)
from avalan.tool.manager import ToolManager
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Literal,
    TypedDict,
)

TemplateMessageRole = Literal[
    "assistant",
    "system",
    "tool",
    "user"
]

class ModelAlreadyLoadedException(Exception):
    pass

class TokenizerAlreadyLoadedException(Exception):
    pass

class TokenizerNotSupportedException(Exception):
    pass

class TemplateMessage(TypedDict):
    role: TemplateMessageRole
    content: str

class TextGenerationStream(
    AsyncIterator[Token | TokenDetail | str],
    ABC
):
    _generator: AsyncGenerator | None=None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    async def __anext__(self) -> Token | TokenDetail | str:
        raise NotImplementedError()

    def __aiter__(self):
        assert self._generator
        return self

class TextGenerationVendor(ABC):
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None=None,
        *,
        tool: ToolManager | None=None,
        use_async_generator: bool=True
    ) -> TextGenerationStream:
        raise NotImplementedError()

    def _system_prompt(self, messages: list[Message]) -> str | None:
        return next(
            (
                message.content
                for message in messages
                if message.role == "system"
            ),
            None
        )

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None=None
    ) -> list[TemplateMessage]:
        return [
            { "role": message.role, "content": message.content }
            for message in messages
            if not exclude_roles or message.role not in exclude_roles
        ]

