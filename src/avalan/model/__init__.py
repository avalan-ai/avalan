from abc import ABC, abstractmethod
from ..entities import (
    GenerationSettings,
    ImageEntity,
    Message,
    Token,
    TokenDetail,
)
from ..tool.manager import ToolManager
from io import StringIO
from inspect import iscoroutine
from json import loads, JSONDecodeError
from re import compile, DOTALL, Pattern
from numpy import ndarray
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Generator,
    Union,
    TypedDict,
)

OutputGenerator = AsyncGenerator[Token | TokenDetail | str, None]
OutputFunction = Callable[..., OutputGenerator | str]

TemplateMessageRole = Literal["assistant", "system", "tool", "user"]


class InvalidJsonResponseException(Exception):
    pass


class ModelAlreadyLoadedException(Exception):
    pass


class TokenizerAlreadyLoadedException(Exception):
    pass


class TokenizerNotSupportedException(Exception):
    pass


class TemplateMessage(TypedDict):
    role: TemplateMessageRole
    content: str


class TextGenerationResponse(AsyncIterator[Union[Token | TokenDetail | str]]):
    _json_patterns: list[Pattern] = [
        # Markdown code fence with explicit json tag
        compile(r"```json\s*(\{.*?\})\s*```", DOTALL),
        # Any markdown code fence possibly with a language specifier
        compile(r"```(?:\w+)?\s*(\{.*?\})\s*```", DOTALL),
        # Generic JSON-like pattern
        compile(r"(\{.*\})", DOTALL),
    ]
    _output_fn: OutputFunction
    _input_token_count: int = 0
    _output: OutputGenerator | None = None
    _buffer: StringIO = StringIO()
    _on_consumed: Callable[[], Awaitable[None] | None] | None = None
    _consumed: bool = False

    def __init__(
        self,
        output_fn: OutputFunction,
        *args,
        use_async_generator: bool,
        **kwargs,
    ):
        self._args = args
        self._kwargs = kwargs
        self._output_fn = output_fn
        self._use_async_generator = use_async_generator

        if "inputs" in self._kwargs:
            inputs = self._kwargs["inputs"]
            self._input_token_count = (
                len(inputs["input_ids"][0])
                if inputs and "input_ids" in inputs
                else 0
            )

    def add_done_callback(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        self._on_consumed = callback

    @property
    def input_token_count(self) -> int:
        return self._input_token_count

    async def _trigger_consumed(self) -> None:
        if self._consumed:
            return
        self._consumed = True
        if self._on_consumed:
            result = self._on_consumed()
            if iscoroutine(result):
                await result

    def __aiter__(self):
        # Create a fresh async generator each time we start iterating
        self._output = self._output_fn(*self._args, **self._kwargs)
        return self

    async def __anext__(self) -> Token | TokenDetail | str:
        try:
            token = await self._output.__anext__()
        except StopAsyncIteration:
            await self._trigger_consumed()
            raise
        self._buffer.write(token if isinstance(token, str) else token.token)
        return token

    async def to_str(self) -> str:
        if not self._use_async_generator:
            result = self._output_fn(*self._args, **self._kwargs)
            self._buffer.write(result)
            await self._trigger_consumed()
            return result

        # Ensure buffer is filled, wether we were already iterating or not
        if not self._output:
            self.__aiter__()

        async for token in self._output:
            self._buffer.write(token)

        await self._trigger_consumed()
        return self._buffer.getvalue()

    async def to_json(self) -> str:
        text = await self.to_str()
        assert text
        for pattern in self._json_patterns:
            match = pattern.search(text)
            if match:
                json_str = match.group(1)
                try:
                    loads(json_str)
                    return json_str
                except JSONDecodeError:
                    continue
        raise InvalidJsonResponseException(text)

    async def to(self, entity_class: type) -> any:
        json = await self.to_json()
        data = loads(json)
        return entity_class(**data)


class TextGenerationStream(AsyncIterator[Token | TokenDetail | str], ABC):
    _generator: AsyncGenerator | None = None

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


EngineResponse = Union[
    TextGenerationResponse,
    TextGenerationVendorStream,
    Generator[str, None, None],
    Generator[Union[Token, TokenDetail], None, None],
    ImageEntity,
    list[ImageEntity],
    list[str],
    dict[str, str],
    ndarray,
    str,
]
