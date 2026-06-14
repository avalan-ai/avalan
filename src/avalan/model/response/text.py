from ...entities import (
    GenerationSettings,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallToken,
)
from ..stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamTerminalOutcome,
    StreamValidationError,
    TextGenerationSingleStream,
    canonical_item_from_consumer_projection,
    canonical_item_from_token,
    iter_stream_consumer_projections,
    normalize_local_stream,
)
from . import InvalidJsonResponseException
from .parsers.reasoning import ReasoningParser, ReasoningTokenLimitExceeded

from asyncio import CancelledError
from collections.abc import Mapping
from inspect import isawaitable
from io import StringIO
from json import JSONDecodeError, loads
from logging import Logger
from queue import Queue
from re import DOTALL, Pattern, compile
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    cast,
)

LegacyOutputItem = Token | TokenDetail | str
OutputItem = (
    Token | TokenDetail | str | CanonicalStreamItem | StreamConsumerProjection
)
OutputGenerator = AsyncIterator[OutputItem]
OutputFunction = Callable[..., OutputGenerator | str]


def _is_semantic_output_item(item: object | None) -> bool:
    return isinstance(item, (CanonicalStreamItem, StreamConsumerProjection))


def _canonical_item_from_output_item(
    item: OutputItem,
    sequence: int,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
) -> CanonicalStreamItem:
    if isinstance(item, CanonicalStreamItem):
        return item
    if isinstance(item, StreamConsumerProjection):
        return canonical_item_from_consumer_projection(item)
    return canonical_item_from_token(
        item,
        sequence,
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
    )


class TextGenerationResponse(AsyncIterator[OutputItem]):
    _json_patterns: list[Pattern[str]] = [
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
    _output: AsyncIterator[OutputItem] | None = None
    _buffer: StringIO = StringIO()
    _on_consumed_callbacks: (
        list[Callable[[], Awaitable[None] | None]] | None
    ) = None
    _consumed: bool = False
    _reasoning_parser: ReasoningParser | None = None
    _parser_queue: Queue[Token | TokenDetail | str] | None = None
    _logger: Logger
    _provider_family: str | None = None
    _answer_buffer: StringIO = StringIO()
    _prefetched_text: str | None = None
    _final_text: str | None = None
    _terminal_failure_outcome: StreamTerminalOutcome | None = None
    _terminal_failure_message: str | None = None
    _output_closed: bool = False
    _bos_token: str | None = None
    _semantic_accumulator: CanonicalStreamAccumulator | None = None
    _legacy_stream_seen: bool = False

    def __init__(
        self,
        output_fn: OutputFunction,
        *args: Any,
        logger: Logger,
        use_async_generator: bool,
        generation_settings: GenerationSettings | None = None,
        bos_token: str | None = None,
        provider_family: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._args = args
        self._kwargs = kwargs
        self._output_fn = output_fn
        self._logger = logger
        self._use_async_generator = use_async_generator
        self._generation_settings = generation_settings
        self._bos_token = bos_token
        self._provider_family = provider_family
        self._on_consumed_callbacks = []
        self._final_text = None
        self._output_closed = False
        self._reset_iteration_state()

        if "inputs" in kwargs:
            self._input_token_count = self._count_input_tokens(
                kwargs["inputs"]
            )

    def _reset_iteration_state(self) -> None:
        self._output_token_count = 0
        self._buffer = StringIO()
        self._answer_buffer = StringIO()
        self._semantic_accumulator = None
        self._legacy_stream_seen = False
        self._terminal_failure_outcome = None
        self._terminal_failure_message = None
        if (
            self._generation_settings
            and self._generation_settings.reasoning.enabled
        ):
            self._parser_queue = Queue()
            self._reasoning_parser = ReasoningParser(
                reasoning_settings=self._generation_settings.reasoning,
                logger=self._logger,
                bos_token=self._bos_token,
            )
        else:
            self._parser_queue = None
            self._reasoning_parser = None

    @staticmethod
    def _count_input_tokens(inputs: Any) -> int:
        if inputs is None:
            return 0

        input_ids = (
            inputs.get("input_ids") if isinstance(inputs, Mapping) else inputs
        )
        if input_ids is None:
            return 0

        token_ids = TextGenerationResponse._first_input_sequence(input_ids)
        try:
            return len(token_ids)
        except TypeError:
            try:
                return len(input_ids)
            except TypeError:
                return 0

    @staticmethod
    def _first_input_sequence(input_ids: Any) -> Any:
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            try:
                if len(shape) <= 1:
                    return input_ids
            except TypeError:
                pass
        elif isinstance(input_ids, (list, tuple)):
            if not input_ids:
                return []
            first = input_ids[0]
            return first if isinstance(first, (list, tuple)) else input_ids

        try:
            return input_ids[0]
        except (IndexError, KeyError, TypeError):
            return input_ids

    def _ensure_non_stream_prefetched(self) -> None:
        if self._use_async_generator:
            return
        if self._prefetched_text is not None:
            return
        if self._buffer.tell():
            return

        result = self._output_fn(*self._args, **self._kwargs)
        if isinstance(result, TextGenerationSingleStream):
            self._output = result
            text = result.final_text
        elif isinstance(result, (Token, TokenDetail)):
            text = result.token
        else:
            text = str(result)

        self._prefetched_text = text
        self._buffer = StringIO()
        self._buffer.write(text)
        self._output_token_count = len(text)

    def add_done_callback(
        self, callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        assert callable(callback)
        assert self._on_consumed_callbacks is not None
        self._on_consumed_callbacks.append(callback)

    @property
    def input_token_count(self) -> int:
        return self._input_token_count

    @property
    def output_token_count(self) -> int:
        return self._output_token_count

    @property
    def usage(self) -> object | None:
        usage = getattr(self._output_fn, "usage", None)
        if usage is not None:
            return cast(object, usage)
        return cast(object | None, getattr(self._output, "usage", None))

    @property
    def provider_family(self) -> str | None:
        if self._provider_family is not None:
            return self._provider_family
        provider_family = getattr(self._output_fn, "provider_family", None)
        if provider_family is not None:
            return cast(str, provider_family)
        return cast(str | None, getattr(self._output, "provider_family", None))

    @property
    def is_async_generator(self) -> bool:
        return self._use_async_generator

    @property
    def can_think(self) -> bool:
        return bool(self._reasoning_parser)

    @property
    def is_thinking(self) -> bool:
        return bool(
            self._reasoning_parser and self._reasoning_parser.is_thinking
        )

    def set_thinking(self, thinking: bool) -> None:
        if self._reasoning_parser:
            self._reasoning_parser.set_thinking(thinking)

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        provider_family = provider_family or self.provider_family or "local"
        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            return self._record_canonical_stream_final_text(
                normalize_local_stream(
                    cast(
                        AsyncIterator[LegacyOutputItem],
                        self._string_output_generator(
                            self._prefetched_text or ""
                        ),
                    ),
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    provider_family=provider_family,
                    capabilities=capabilities,
                    close_after_terminal=close_after_terminal,
                )
            )

        return self._canonical_stream_from_output(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.LOCAL,
                provider_family=provider_family,
                supports_reasoning=self.can_think,
                supports_tool_calls=True,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _canonical_stream_from_output(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None,
        capabilities: StreamProviderCapabilities,
        close_after_terminal: bool,
    ) -> AsyncIterator[CanonicalStreamItem]:
        iterator = self.__aiter__()
        try:
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                first = None

            if _is_semantic_output_item(first):
                accumulator = CanonicalStreamAccumulator()
                item = _canonical_item_from_output_item(
                    cast(OutputItem, first),
                    0,
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                )
                accumulator.add(item)
                yield item
                while True:
                    try:
                        token = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    item = _canonical_item_from_output_item(
                        token,
                        0,
                        stream_session_id=stream_session_id,
                        run_id=run_id,
                        turn_id=turn_id,
                    )
                    accumulator.add(item)
                    yield item
                accumulator.validate_complete()
                if self._remember_terminal_exception(accumulator) is None:
                    self._final_text = accumulator.answer_text
                    await self._trigger_consumed()
                return

            async def tokens() -> AsyncIterator[LegacyOutputItem]:
                try:
                    if first is not None:
                        yield cast(LegacyOutputItem, first)
                    while True:
                        try:
                            token = await iterator.__anext__()
                        except StopAsyncIteration:
                            break
                        yield cast(LegacyOutputItem, token)
                finally:
                    await self.aclose()

            async for item in self._record_canonical_stream_final_text(
                normalize_local_stream(
                    tokens(),
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    provider_family=provider_family,
                    capabilities=capabilities,
                    close_after_terminal=close_after_terminal,
                )
            ):
                yield item
        finally:
            await self.aclose()

    async def _record_canonical_stream_final_text(
        self,
        items: AsyncIterator[CanonicalStreamItem],
    ) -> AsyncIterator[CanonicalStreamItem]:
        accumulator = CanonicalStreamAccumulator()
        try:
            async for item in items:
                accumulator.add(item)
                yield item
            accumulator.validate_complete()
            if self._remember_terminal_exception(accumulator) is None:
                self._final_text = accumulator.answer_text
                await self._trigger_consumed()
        finally:
            aclose = getattr(items, "aclose", None)
            if aclose is not None:
                assert callable(aclose)
                await self._close_output(cast(Callable[[], object], aclose))

    def consumer_projections(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
        validate_order: bool = True,
    ) -> AsyncIterator[StreamConsumerProjection]:
        return iter_stream_consumer_projections(
            self.canonical_stream(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family,
                capabilities=capabilities,
                close_after_terminal=close_after_terminal,
            ),
            validate_order=validate_order,
        )

    async def _trigger_consumed(self) -> None:
        if self._consumed:
            return
        self._consumed = True
        callbacks = tuple(self._on_consumed_callbacks or ())
        for callback in callbacks:
            result = callback()
            if isawaitable(result):
                awaited_result = await cast(Awaitable[object], result)
                assert awaited_result is None

    def __aiter__(self) -> AsyncIterator[OutputItem]:
        # Create a fresh async generator each time we start iterating
        if self._output is not None:
            self._reset_iteration_state()
            self._final_text = None
        output = self._output_fn(*self._args, **self._kwargs)
        if isinstance(output, str):
            self._output = self._string_output_generator(output)
        else:
            self._output = output
        self._output_closed = False
        return self

    @staticmethod
    async def _string_output_generator(text: str) -> OutputGenerator:
        yield text

    async def aclose(self) -> None:
        output = self._output
        if output is None or self._output_closed:
            return
        aclose = getattr(output, "aclose", None)
        self._output_closed = True
        if aclose is None:
            return
        assert callable(aclose)
        await self._close_output(cast(Callable[[], object], aclose))

    @staticmethod
    async def _close_output(aclose: Callable[[], object]) -> None:
        result = aclose()
        if isawaitable(result):
            awaited_result = await cast(Awaitable[object], result)
            assert awaited_result is None
        else:
            assert result is None

    async def __anext__(self) -> OutputItem:
        assert self._output

        while True:
            if self._parser_queue and not self._parser_queue.empty():
                return await self._record_returned_token(
                    self._parser_queue.get()
                )

            try:
                token = await self._output.__anext__()
            except StopAsyncIteration:
                if self._reasoning_parser:
                    parser_queue = self._parser_queue
                    assert parser_queue is not None
                    for it in await self._reasoning_parser.flush():
                        parser_queue.put(it)
                    if not parser_queue.empty():
                        continue
                try:
                    if self._semantic_accumulator is not None:
                        self._semantic_accumulator.validate_complete()
                        self._remember_terminal_exception(
                            self._semantic_accumulator
                        )
                except (Exception, CancelledError):
                    await self.aclose()
                    raise
                if (
                    self._semantic_accumulator is None
                    or self._terminal_failure_outcome is None
                ):
                    await self._trigger_consumed()
                await self.aclose()
                raise
            except (Exception, CancelledError):
                await self.aclose()
                raise

            if isinstance(
                token, (CanonicalStreamItem, StreamConsumerProjection)
            ):
                return await self._record_returned_token(token)

            assert isinstance(token, (str, Token))
            token_str = token if isinstance(token, str) else token.token
            self._buffer.write(token_str)

            if isinstance(token, ToolCallToken) and token.call is not None:
                return await self._record_returned_token(token)

            if not self._reasoning_parser or (
                self._reasoning_parser.is_thinking_budget_exhausted
                and not self._reasoning_parser.is_thinking
            ):
                return await self._record_returned_token(token)

            try:
                items = await self._reasoning_parser.push(token_str)
            except ReasoningTokenLimitExceeded:
                await self._trigger_consumed()
                await self.aclose()
                raise StopAsyncIteration

            for it in items:
                parsed: Token | TokenDetail | str
                if isinstance(it, ReasoningToken):
                    token_id = (
                        token.id
                        if isinstance(token, (Token, TokenDetail))
                        else it.id
                    )
                    if token_id is None:
                        token_id = -1
                    parsed = ReasoningToken(
                        token=it.token, id=token_id, probability=it.probability
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
                parser_queue = self._parser_queue
                assert parser_queue is not None
                parser_queue.put(parsed)

            parser_queue = self._parser_queue
            assert parser_queue is not None
            if not parser_queue.empty():
                return await self._record_returned_token(parser_queue.get())

    async def _record_returned_token(
        self,
        token: OutputItem,
    ) -> OutputItem:
        try:
            item = _canonical_item_from_output_item(
                token,
                0,
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
            )
            semantic_output = _is_semantic_output_item(token)
            if semantic_output:
                if self._legacy_stream_seen:
                    raise StreamValidationError(
                        "canonical stream item after legacy stream item"
                    )
                if self._semantic_accumulator is None:
                    self._semantic_accumulator = CanonicalStreamAccumulator()
                self._semantic_accumulator.add(item)
            elif self._semantic_accumulator is not None:
                raise StreamValidationError(
                    "legacy stream item after canonical stream item"
                )
            else:
                self._legacy_stream_seen = True
            if (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta is not None
            ):
                self._answer_buffer.write(item.text_delta)
            self._output_token_count += 1
            return token
        except (Exception, CancelledError):
            await self.aclose()
            raise

    def __str__(self) -> str:
        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            return self._prefetched_text or ""
        return super().__str__()

    async def to_str(self) -> str:
        terminal_exception = self._terminal_exception_from_state()
        if terminal_exception is not None:
            raise terminal_exception

        if self._final_text is not None:
            await self._trigger_consumed()
            return self._final_text

        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            if self._prefetched_text is None:
                return ""
            await self._trigger_consumed()
            self._final_text = self._prefetched_text
            return self._prefetched_text

        # Ensure buffer is filled, wether we were already iterating or not
        if not self._output:
            self.__aiter__()
        assert self._output is not None

        legacy_accumulator: CanonicalStreamAccumulator | None = None
        semantic_accumulator = self._semantic_accumulator
        sequence = 0
        buffered_text = self._answer_buffer.getvalue()

        def legacy_stream_accumulator() -> CanonicalStreamAccumulator:
            nonlocal legacy_accumulator, sequence
            if legacy_accumulator is not None:
                return legacy_accumulator
            legacy_accumulator = CanonicalStreamAccumulator()
            legacy_accumulator.add(
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=sequence,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                )
            )
            sequence += 1
            if buffered_text:
                legacy_accumulator.add(
                    CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=sequence,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta=buffered_text,
                    )
                )
                sequence += 1
            return legacy_accumulator

        while True:
            try:
                token = await self.__anext__()
            except StopAsyncIteration:
                break

            if _is_semantic_output_item(token):
                semantic_accumulator = self._semantic_accumulator
                assert semantic_accumulator is not None
                terminal_exception = self._remember_terminal_exception(
                    semantic_accumulator
                )
                if terminal_exception is not None:
                    await self.aclose()
                    raise terminal_exception
                continue

            legacy_stream_accumulator().add(
                _canonical_item_from_output_item(
                    token,
                    sequence,
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                )
            )
            sequence += 1

        if semantic_accumulator is not None:
            semantic_accumulator.validate_complete()
            terminal_exception = self._remember_terminal_exception(
                semantic_accumulator
            )
            if terminal_exception is not None:
                raise terminal_exception
            await self._trigger_consumed()
            self._final_text = semantic_accumulator.answer_text
            return self._final_text

        accumulator = legacy_stream_accumulator()
        accumulator.add(
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
        )
        sequence += 1
        accumulator.add(
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=sequence,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage=cast(Any, self.usage or {}),
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        )
        accumulator.validate_complete()
        await self._trigger_consumed()
        self._final_text = accumulator.answer_text
        return self._final_text

    def _terminal_exception(
        self,
        accumulator: CanonicalStreamAccumulator,
    ) -> BaseException | None:
        outcome = accumulator.terminal_outcome
        if outcome is None or outcome is StreamTerminalOutcome.COMPLETED:
            return None
        message = self._terminal_message(accumulator, outcome)
        if outcome is StreamTerminalOutcome.CANCELLED:
            return CancelledError(message)
        return RuntimeError(message)

    def _remember_terminal_exception(
        self,
        accumulator: CanonicalStreamAccumulator,
    ) -> BaseException | None:
        exception = self._terminal_exception(accumulator)
        if exception is None:
            return None
        assert accumulator.terminal_outcome is not None
        self._terminal_failure_outcome = accumulator.terminal_outcome
        self._terminal_failure_message = str(exception)
        return exception

    def _terminal_exception_from_state(self) -> BaseException | None:
        outcome = self._terminal_failure_outcome
        if outcome is None:
            return None
        message = self._terminal_failure_message or f"stream {outcome.value}"
        if outcome is StreamTerminalOutcome.CANCELLED:
            return CancelledError(message)
        return RuntimeError(message)

    @staticmethod
    def _terminal_message(
        accumulator: CanonicalStreamAccumulator,
        outcome: StreamTerminalOutcome,
    ) -> str:
        terminal = next(
            (
                item
                for item in reversed(accumulator.items)
                if item.is_stream_terminal
            ),
            None,
        )
        data = terminal.data if terminal is not None else None
        if isinstance(data, Mapping):
            for key in ("message", "error", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            error_type = data.get("error_type")
            if isinstance(error_type, str) and error_type.strip():
                return error_type
        if isinstance(data, str) and data.strip():
            return data
        return f"stream {outcome.value}"

    async def to_json(self) -> str:
        text = await self.to_str()
        return self.extract_json(text)

    @classmethod
    def extract_json(cls, text: str) -> str:
        assert text
        for pattern in cls._json_patterns:
            match = pattern.search(text)
            if match:
                json_str = match.group(1)
                try:
                    loads(json_str)
                    return json_str
                except JSONDecodeError:
                    continue
        raise InvalidJsonResponseException(text)

    async def to(self, entity_class: type[Any]) -> Any:
        json = await self.to_json()
        data = loads(json)
        return entity_class(**data)
