from . import InvalidJsonResponseException
from ...entities import (
    GenerationSettings,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallToken,
)
from .parsers.reasoning import ReasoningParser
from .. import ReasoningTokenLimitExceeded
from io import StringIO
from queue import Queue
from inspect import iscoroutine
from json import loads, JSONDecodeError
from re import compile, DOTALL, Pattern
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
)

OutputGenerator = AsyncGenerator[Token | TokenDetail | str, None]
OutputFunction = Callable[..., OutputGenerator | str]


class TextGenerationResponse(AsyncIterator[Token | TokenDetail | str]):
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
    _reasoning_parser: ReasoningParser | None = None
    _parser_queue: Queue[Token | TokenDetail | str] | None = None

    def __init__(
        self,
        output_fn: OutputFunction,
        *args,
        use_async_generator: bool,
        inputs: dict | None = None,
        generation_settings: GenerationSettings | None = None,
        **kwargs,
    ):
        self._args = args
        self._kwargs = kwargs
        self._output_fn = output_fn
        self._use_async_generator = use_async_generator
        self._generation_settings = generation_settings
        if generation_settings and generation_settings.reasoning.enabled:
            self._parser_queue = Queue()
            self._reasoning_parser = ReasoningParser(
                reasoning_settings=generation_settings.reasoning
            )
        else:
            self._parser_queue = None
            self._reasoning_parser = None

        if inputs is not None:
            self._input_token_count = (
                len(inputs["input_ids"][0])
                if isinstance(inputs, dict) and "input_ids" in inputs
                else 0
            )

    def add_done_callback(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        self._on_consumed = callback

    @property
    def input_token_count(self) -> int:
        return self._input_token_count

    @property
    def can_think(self) -> bool:
        return bool(self._reasoning_parser)

    @property
    def is_thinking(self) -> bool:
        return self.can_think and self._reasoning_parser.is_thinking

    def set_thinking(self, thinking: bool) -> None:
        if self._reasoning_parser:
            self._reasoning_parser.set_thinking(thinking)

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
        assert self._output

        if self._parser_queue and not self._parser_queue.empty():
            return self._parser_queue.get()

        try:
            token = await self._output.__anext__()
        except StopAsyncIteration:
            await self._trigger_consumed()
            raise

        token_str = token if isinstance(token, str) else token.token
        self._buffer.write(token_str)

        if not self._reasoning_parser:
            return token

        try:
            items = await self._reasoning_parser.push(token_str)
        except ReasoningTokenLimitExceeded:
            await self._trigger_consumed()
            raise StopAsyncIteration
        for it in items:
            if isinstance(it, ReasoningToken):
                token_id = (
                    token.id
                    if isinstance(token, (Token, TokenDetail))
                    else it.id
                )
                parsed = ReasoningToken(
                    token=it.token, id=token_id, probability=it.probability
                )
            elif isinstance(token, ToolCallToken):
                parsed = ToolCallToken(token=str(it), id=token.id)
            elif isinstance(token, TokenDetail):
                parsed = TokenDetail(
                    id=token.id,
                    token=it if isinstance(it, str) else it.token,
                    probability=token.probability,
                    tokens=token.tokens,
                    probability_distribution=token.probability_distribution,
                    step=token.step,
                )
            elif isinstance(token, Token):
                parsed = Token(id=token.id, token=str(it))
            else:
                parsed = it
            self._parser_queue.put(parsed)

        return self._parser_queue.get()

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
