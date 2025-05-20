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
    Optional,
    TypedDict,
    Union
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
    AsyncIterator[Union[Token,TokenDetail,str]],
    ABC
):
    _generator: Optional[AsyncGenerator]=None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    async def __anext__(self) -> Union[Token,TokenDetail,str]:
        raise NotImplementedError()

    def __aiter__(self):
        assert self._generator
        return self

class TextGenerationVendor(ABC):
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: Optional[GenerationSettings]=None,
        *,
        tool: Optional[ToolManager]=None,
        use_async_generator: bool=True
    ) -> TextGenerationStream:
        raise NotImplementedError()

    def _system_prompt(self, messages: list[Message]) -> Optional[str]:
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
        exclude_roles: Optional[list[TemplateMessageRole]]=None
    ) -> list[TemplateMessage]:
        return [
            { "role": message.role, "content": message.content }
            for message in messages
            if not exclude_roles or message.role not in exclude_roles
        ]

