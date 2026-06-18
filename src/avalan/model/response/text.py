from ...entities import GenerationSettings
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
    TextGenerationStream,
    canonical_item_from_consumer_projection,
    iter_stream_consumer_projections,
)
from . import InvalidJsonResponseException

from asyncio import CancelledError
from collections.abc import AsyncIterable, Mapping
from inspect import isawaitable
from io import StringIO
from json import JSONDecodeError, loads
from logging import Logger
from re import DOTALL, Pattern, compile
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    cast,
)

OutputGenerator = AsyncIterator[CanonicalStreamItem]
OutputFunction = Callable[..., object]
_LEGACY_SDK_STREAM_ERROR = "unsupported legacy SDK response stream item"
_LEGACY_SDK_RESULT_TYPE_NAMES = frozenset(
    {
        "Token",
        "TokenDetail",
        "ReasoningToken",
        "ToolCallToken",
    }
)


def _explicit_canonical_stream(
    source: object,
) -> Callable[..., AsyncIterator[CanonicalStreamItem]] | None:
    canonical_stream = getattr(source, "canonical_stream", None)
    if not callable(canonical_stream):
        return None
    class_method = getattr(type(source), "canonical_stream", None)
    if _is_legacy_base_canonical_stream(class_method):
        return None
    return cast(
        Callable[..., AsyncIterator[CanonicalStreamItem]], canonical_stream
    )


def _is_legacy_base_canonical_stream(class_method: object) -> bool:
    return class_method is TextGenerationStream.canonical_stream


def _text_from_non_stream_result(result: object) -> str:
    if isinstance(result, TextGenerationSingleStream):
        return result.accumulator.answer_text
    if type(result) is str:
        return result
    if _is_legacy_sdk_result_type(type(result)):
        raise StreamValidationError(_LEGACY_SDK_STREAM_ERROR)
    return str(result)


def _is_legacy_sdk_result_type(result_type: type[object]) -> bool:
    return any(
        base.__module__ == "avalan.entities"
        and base.__name__ in _LEGACY_SDK_RESULT_TYPE_NAMES
        for base in result_type.__mro__
    )


class TextGenerationResponse(AsyncIterator[CanonicalStreamItem]):
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
    _output: AsyncIterator[object] | None = None
    _buffer: StringIO = StringIO()
    _on_consumed_callbacks: (
        list[Callable[[], Awaitable[None] | None]] | None
    ) = None
    _consumed: bool = False
    _logger: Logger
    _provider_family: str | None = None
    _prefetched_text: str | None = None
    _final_text: str | None = None
    _terminal_failure_outcome: StreamTerminalOutcome | None = None
    _terminal_failure_message: str | None = None
    _validation_failure_message: str | None = None
    _output_closed: bool = False
    _bos_token: str | None = None
    _stream_accumulator: CanonicalStreamAccumulator | None = None
    _last_stream_sequence: int | None = None
    _can_think: bool = False
    _is_thinking: bool = False
    _manual_thinking: bool = False

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
        self._manual_thinking = False
        self._reset_iteration_state()

        if "inputs" in kwargs:
            self._input_token_count = self._count_input_tokens(
                kwargs["inputs"]
            )

    def _reset_iteration_state(self) -> None:
        self._output_token_count = 0
        self._buffer = StringIO()
        self._stream_accumulator = None
        self._last_stream_sequence = None
        self._can_think = bool(
            self._generation_settings
            and self._generation_settings.reasoning.enabled
        )
        self._is_thinking = self._manual_thinking and self._can_think
        self._terminal_failure_outcome = None
        self._terminal_failure_message = None
        self._validation_failure_message = None

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
        text = _text_from_non_stream_result(result)

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
        if self._stream_accumulator:
            usage = self._stream_accumulator.final_usage
            if usage is not None:
                return cast(object, usage)
        return self._provider_usage()

    def _provider_usage(self) -> object | None:
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
        return self._can_think

    @property
    def is_thinking(self) -> bool:
        return self._is_thinking

    def set_thinking(self, thinking: bool) -> None:
        assert isinstance(thinking, bool)
        self._manual_thinking = thinking and self._can_think
        self._is_thinking = self._manual_thinking

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
        provider_family = (
            provider_family
            or self.provider_family
            or (
                capabilities.normalized_provider_family
                if capabilities is not None
                else None
            )
            or "local"
        )
        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            return self._record_canonical_stream_final_text(
                self._canonical_stream_from_final_text(
                    self._prefetched_text or "",
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    provider_family=provider_family,
                    capabilities=capabilities
                    or self._default_stream_capabilities(provider_family),
                    close_after_terminal=close_after_terminal,
                )
            )

        canonical_stream = _explicit_canonical_stream(self._output_fn)
        if canonical_stream is not None:
            self._reset_iteration_state()
            self._final_text = None
            self._output = (
                self._output_fn
                if isinstance(self._output_fn, AsyncIterable)
                else None
            )
            self._output_closed = False
            return self._record_canonical_stream_final_text(
                canonical_stream(
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    provider_family=provider_family,
                    capabilities=capabilities
                    or self._default_stream_capabilities(provider_family),
                    close_after_terminal=close_after_terminal,
                )
            )

        return self._canonical_stream_from_output(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or self._default_stream_capabilities(provider_family),
            close_after_terminal=close_after_terminal,
        )

    def _default_stream_capabilities(
        self,
        provider_family: str,
    ) -> StreamProviderCapabilities:
        return StreamProviderCapabilities(
            backend=StreamProducerBackend.LOCAL,
            provider_family=provider_family,
            supports_reasoning=self.can_think,
            supports_tool_calls=True,
            supports_cancellation=True,
        )

    async def _canonical_stream_from_final_text(
        self,
        text: str,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str,
        capabilities: StreamProviderCapabilities | None,
        close_after_terminal: bool,
    ) -> AsyncIterator[CanonicalStreamItem]:
        sequence = 0
        metadata: dict[str, Any] = {}
        if capabilities is not None:
            metadata["capabilities"] = cast(Any, capabilities.to_metadata())

        yield CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
            metadata=metadata,
            provider_family=provider_family,
        )
        sequence += 1

        if text:
            yield CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
                provider_family=provider_family,
            )
            sequence += 1
            yield CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                provider_family=provider_family,
            )
            sequence += 1

        yield CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage=cast(Any, self._provider_usage() or {}),
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
            provider_family=provider_family,
        )
        sequence += 1

        if close_after_terminal:
            yield CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=sequence,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
                provider_family=provider_family,
            )

    async def _canonical_stream_from_output(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str,
        capabilities: StreamProviderCapabilities,
        close_after_terminal: bool,
    ) -> AsyncIterator[CanonicalStreamItem]:
        iterator = self.__aiter__()
        try:
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                first = None

            if first is not None:
                yield first
                while True:
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    yield item
                return

            async for item in self._record_canonical_stream_final_text(
                self._canonical_stream_from_final_text(
                    "",
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

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        # Create a fresh async generator each time we start iterating
        if self._output is not None:
            self._reset_iteration_state()
            self._final_text = None
        output = self._output_fn(*self._args, **self._kwargs)
        if isinstance(output, AsyncIterable):
            self._output = output.__aiter__()
        else:
            self._output = self._single_item_output_generator(output)
        self._output_closed = False
        return self

    @staticmethod
    async def _single_item_output_generator(
        item: object,
    ) -> AsyncIterator[object]:
        yield item

    async def aclose(self) -> None:
        if self._output_closed:
            return
        self._output_closed = True
        await self._call_output_cleanup("aclose")

    async def cancel(self) -> None:
        await self._call_output_cleanup("cancel")

    async def _call_output_cleanup(self, method_name: str) -> None:
        assert method_name in ("cancel", "aclose")
        seen: set[int] = set()
        for source in (self._output, self._output_fn):
            if source is None:
                continue
            source_id = id(source)
            if source_id in seen:
                continue
            seen.add(source_id)
            method = getattr(source, method_name, None)
            if method is None:
                continue
            assert callable(method)
            await self._close_output(cast(Callable[[], object], method))

    @staticmethod
    async def _close_output(aclose: Callable[[], object]) -> None:
        result = aclose()
        if isawaitable(result):
            awaited_result = await cast(Awaitable[object], result)
            assert awaited_result is None
        else:
            assert result is None

    async def __anext__(self) -> CanonicalStreamItem:
        assert self._output
        try:
            item = await self._output.__anext__()
        except StopAsyncIteration:
            try:
                if self._stream_accumulator is not None:
                    self._stream_accumulator.validate_complete()
                    self._remember_terminal_exception(self._stream_accumulator)
            except StreamValidationError as exc:
                self._validation_failure_message = str(exc)
                await self.aclose()
                raise
            except (Exception, CancelledError):
                await self.aclose()
                raise
            if self._terminal_failure_outcome is None:
                await self._trigger_consumed()
                self._final_text = (
                    self._stream_accumulator.answer_text
                    if self._stream_accumulator is not None
                    else ""
                )
            await self.aclose()
            raise
        except (Exception, CancelledError):
            await self.aclose()
            raise

        return await self._record_returned_item(item)

    async def _record_returned_item(
        self,
        item: object,
    ) -> CanonicalStreamItem:
        try:
            if isinstance(item, StreamConsumerProjection):
                item = canonical_item_from_consumer_projection(item)
            if not isinstance(item, CanonicalStreamItem):
                raise StreamValidationError(_LEGACY_SDK_STREAM_ERROR)
            if self._stream_accumulator is None:
                self._stream_accumulator = CanonicalStreamAccumulator()
            if (
                self._last_stream_sequence is not None
                and item.sequence > self._last_stream_sequence + 1
            ):
                raise StreamValidationError(
                    "lossless consumer stream sequence gap"
                )
            self._stream_accumulator.add(item)
            self._last_stream_sequence = item.sequence
            self._output_token_count += 1
            return item
        except StreamValidationError as exc:
            self._validation_failure_message = str(exc)
            await self.aclose()
            raise
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

        while True:
            try:
                await self.__anext__()
            except StopAsyncIteration:
                break

            accumulator = self._stream_accumulator
            assert accumulator is not None
            terminal_exception = self._remember_terminal_exception(accumulator)
            if terminal_exception is not None:
                await self.aclose()
                raise terminal_exception

        accumulator = self._stream_accumulator
        if accumulator is None:
            await self._trigger_consumed()
            self._final_text = ""
            return self._final_text

        assert accumulator is not None
        accumulator.validate_complete()
        terminal_exception = self._remember_terminal_exception(accumulator)
        if terminal_exception is not None:
            raise terminal_exception
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
        if self._validation_failure_message is not None:
            return StreamValidationError(self._validation_failure_message)
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
