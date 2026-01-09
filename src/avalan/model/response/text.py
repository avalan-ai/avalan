from ...entities import (
    GenerationSettings,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallToken,
)
from ..stream import TextGenerationSingleStream
from . import InvalidJsonResponseException
from .parsers.reasoning import ReasoningParser, ReasoningTokenLimitExceeded

from inspect import iscoroutine
from io import StringIO
from json import JSONDecodeError, loads
from logging import Logger
from queue import Queue
from re import DOTALL, Pattern, compile
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Self,
    TypeVar,
)

T = TypeVar("T")

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
    _output_token_count: int = 0
    _output: OutputGenerator | None = None
    _buffer: StringIO = StringIO()
    _on_consumed: Callable[[], Awaitable[None] | None] | None = None
    _consumed: bool = False
    _reasoning_parser: ReasoningParser | None = None
    _parser_queue: Queue[Token | TokenDetail | str] | None = None
    _logger: Logger
    _prefetched_text: str | None = None

    def __init__(
        self,
        output_fn: OutputFunction,
        *args,
        logger: Logger,
        use_async_generator: bool,
        generation_settings: GenerationSettings | None = None,
        bos_token: str | None = None,
        **kwargs,
    ):
        self._args = args
        self._kwargs = kwargs
        self._output_fn = output_fn
        self._logger = logger
        self._use_async_generator = use_async_generator
        self._generation_settings = generation_settings
        self._output_token_count = 0
        self._buffer = StringIO()
        if generation_settings and generation_settings.reasoning.enabled:
            self._parser_queue = Queue()
            self._reasoning_parser = ReasoningParser(
                reasoning_settings=generation_settings.reasoning,
                logger=self._logger,
                bos_token=bos_token,
            )
        else:
            self._parser_queue = None
            self._reasoning_parser = None

        if "inputs" in kwargs:
            inputs = kwargs["inputs"]
            self._input_token_count = (
                len(inputs["input_ids"][0])
                if inputs and "input_ids" in inputs
                else 0
            )

    def _ensure_non_stream_prefetched(self) -> None:
        if self._use_async_generator:
            return
        if self._prefetched_text is not None:
            return
        if self._buffer.tell():
            return

        fn_result = self._output_fn(*self._args, **self._kwargs)
        if isinstance(fn_result, TextGenerationSingleStream):
            stream_content = fn_result.content
            if isinstance(stream_content, (Token, TokenDetail)):
                text = stream_content.token
            else:
                text = str(stream_content)
        elif isinstance(fn_result, (Token, TokenDetail)):
            text = fn_result.token
        else:
            text = str(fn_result)

        self._prefetched_text = text
        self._buffer = StringIO()
        self._buffer.write(text)
        self._output_token_count = len(text)

    def add_done_callback(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        self._on_consumed = callback

    @property
    def input_token_count(self) -> int:
        return self._input_token_count

    @property
    def output_token_count(self) -> int:
        return self._output_token_count

    @property
    def can_think(self) -> bool:
        return bool(self._reasoning_parser)

    @property
    def is_thinking(self) -> bool:
        return (
            self.can_think
            and self._reasoning_parser is not None
            and self._reasoning_parser.is_thinking
        )

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

    def __aiter__(self) -> Self:
        """Return iterator for async iteration over tokens.

        Returns:
            Self for async iteration.
        """
        # Create a fresh async generator each time we start iterating
        fn_result = self._output_fn(*self._args, **self._kwargs)
        if not isinstance(fn_result, str):
            self._output = fn_result
        return self

    async def __anext__(self) -> Token | TokenDetail | str:
        assert self._output

        while True:
            if self._parser_queue and not self._parser_queue.empty():
                self._output_token_count += 1
                return self._parser_queue.get()

            try:
                assert self._output is not None
                token = await self._output.__anext__()
            except StopAsyncIteration:
                if self._reasoning_parser and self._parser_queue:
                    for it in await self._reasoning_parser.flush():
                        self._parser_queue.put(it)
                    if not self._parser_queue.empty():
                        continue
                await self._trigger_consumed()
                raise

            token_str = token if isinstance(token, str) else token.token
            self._buffer.write(token_str)

            if not self._reasoning_parser or (
                self._reasoning_parser.is_thinking_budget_exhausted
                and not self._reasoning_parser.is_thinking
            ):
                self._output_token_count += 1
                return token

            try:
                items = await self._reasoning_parser.push(token_str)
            except ReasoningTokenLimitExceeded:
                await self._trigger_consumed()
                raise StopAsyncIteration

            assert self._parser_queue is not None
            for it in items:
                parsed: (
                    Token | TokenDetail | ReasoningToken | ToolCallToken | str
                )
                if isinstance(it, ReasoningToken):
                    token_id = (
                        token.id
                        if isinstance(token, (Token, TokenDetail))
                        else it.id
                    )
                    parsed = ReasoningToken(
                        token=it.token,
                        id=token_id if token_id is not None else -1,
                        probability=it.probability,
                    )
                elif isinstance(token, ToolCallToken):
                    parsed = ToolCallToken(
                        token=str(it), id=token.id, call=token.call
                    )
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

            if not self._parser_queue.empty():
                self._output_token_count += 1
                return self._parser_queue.get()

    def __str__(self) -> str:
        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            return self._prefetched_text or ""
        return super().__str__()

    async def to_str(self) -> str:
        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            if self._prefetched_text is None:
                return ""
            await self._trigger_consumed()
            return self._prefetched_text

        # Ensure buffer is filled, whether we were already iterating or not
        if not self._output:
            self.__aiter__()

        assert self._output is not None
        async for token in self._output:
            token_str = token if isinstance(token, str) else token.token
            self._buffer.write(token_str)
            self._output_token_count += 1

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

    async def to(self, entity_class: type[T]) -> T:
        """Convert JSON response to entity class instance.

        Args:
            entity_class: The class to instantiate with JSON data.

        Returns:
            Instance of entity_class populated with JSON data.
        """
        json_str = await self.to_json()
        data = loads(json_str)
        return entity_class(**data)
