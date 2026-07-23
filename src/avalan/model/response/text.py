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
    TextGenerationNonStreamResult,
    TextGenerationSingleStream,
    TextGenerationStream,
    canonical_item_from_consumer_projection,
    iter_stream_consumer_projections,
)
from . import InvalidJsonResponseException

from asyncio import CancelledError, Task, create_task, sleep, wait
from collections.abc import AsyncIterable, Mapping
from enum import Enum
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
_SINGLE_USE_STREAM_ERROR = (
    "TextGenerationResponse stream session is single-use"
)
_INTERRUPTED_PROVIDER_EXCEPTIONS = (
    CancelledError,
    KeyboardInterrupt,
    SystemExit,
)
_PROVIDER_CLEANUP_TIMEOUT_SECONDS = 5.0


class _StreamSessionState(Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    FINALIZED = "finalized"
    CLOSED_CANCELLED = "closed_cancelled"


class _ResponseOwnedStreamIterator(AsyncIterator[CanonicalStreamItem]):
    def __init__(
        self,
        stream: AsyncIterator[CanonicalStreamItem],
        close_response: Callable[[], Awaitable[None]],
        is_closed: Callable[[], bool],
    ) -> None:
        self._stream = stream
        self._close_response = close_response
        self._is_closed = is_closed

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        if self._is_closed():
            raise StopAsyncIteration
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        aclose = getattr(self._stream, "aclose", None)
        try:
            if aclose is not None:
                assert callable(aclose)
                await TextGenerationResponse._close_output(
                    cast(Callable[[], object], aclose)
                )
        finally:
            await self._close_response()


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
    if isinstance(result, TextGenerationNonStreamResult):
        return result.answer_text
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


class _TextGenerationWorkerShutdownError(RuntimeError):
    """Report a worker that outlived bounded stream cleanup."""


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
    _closed_source_ids: set[int]
    _cancelled_source_ids: set[int]
    _cleanup_failures: dict[tuple[str, int], Exception]
    _cleanup_tasks: dict[str, Task[None]]
    _session_state: _StreamSessionState
    _bos_token: str | None = None
    _stream_accumulator: CanonicalStreamAccumulator | None = None
    _last_stream_sequence: int | None = None
    _count_stream_output_items: bool = True
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
        self._closed_source_ids = set()
        self._cancelled_source_ids = set()
        self._cleanup_failures = {}
        self._cleanup_tasks = {}
        self._session_state = _StreamSessionState.NOT_STARTED
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
        self._count_stream_output_items = True
        self._can_think = bool(
            self._generation_settings
            and self._generation_settings.reasoning.enabled
        )
        self._is_thinking = self._manual_thinking and self._can_think
        self._terminal_failure_outcome = None
        self._terminal_failure_message = None
        self._validation_failure_message = None

    def _claim_stream_session(self) -> None:
        if not self._use_async_generator:
            return
        if self._session_state is not _StreamSessionState.NOT_STARTED:
            raise RuntimeError(_SINGLE_USE_STREAM_ERROR)
        self._session_state = _StreamSessionState.ACTIVE
        self._reset_iteration_state()
        self._final_text = None

    def _start_stream_output(self) -> None:
        if self._output is not None:
            return
        output = self._output_fn(*self._args, **self._kwargs)
        if isinstance(output, AsyncIterable):
            self._output = output.__aiter__()
        else:
            self._output = self._single_item_output_generator(output)
        self._output_closed = False

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

        result = (
            self._output_fn
            if isinstance(
                self._output_fn,
                (
                    TextGenerationNonStreamResult,
                    TextGenerationSingleStream,
                ),
            )
            else self._output_fn(*self._args, **self._kwargs)
        )
        if isinstance(
            result, (TextGenerationNonStreamResult, TextGenerationSingleStream)
        ):
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
    def cleanup_complete(self) -> bool:
        """Return whether the active provider source closed successfully."""
        return self._output_closed and not self._cleanup_tasks

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
            prefetched_text = self._prefetched_text or ""
            prefetched_usage = self._provider_usage()
            structured_result = (
                self._output
                if isinstance(self._output, TextGenerationNonStreamResult)
                else None
            )
            self._reset_iteration_state()
            self._output_closed = False
            self._output_token_count = len(prefetched_text)
            self._final_text = None
            if structured_result is not None:
                structured_items = structured_result.canonical_stream(
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    provider_family=provider_family,
                    capabilities=capabilities
                    or self._default_stream_capabilities(provider_family),
                    close_after_terminal=close_after_terminal,
                )
                return _ResponseOwnedStreamIterator(
                    self._record_canonical_stream_final_text(
                        structured_items,
                        count_output=False,
                    ),
                    self.aclose,
                    self._is_stream_closed,
                )
            return _ResponseOwnedStreamIterator(
                self._record_canonical_stream_final_text(
                    self._canonical_stream_from_final_text(
                        prefetched_text,
                        stream_session_id=stream_session_id,
                        run_id=run_id,
                        turn_id=turn_id,
                        provider_family=provider_family,
                        capabilities=capabilities
                        or self._default_stream_capabilities(provider_family),
                        close_after_terminal=close_after_terminal,
                        usage=prefetched_usage,
                    ),
                    count_output=False,
                ),
                self.aclose,
                self._is_stream_closed,
            )

        self._claim_stream_session()
        canonical_stream = _explicit_canonical_stream(self._output_fn)
        if canonical_stream is not None:
            items = canonical_stream(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family,
                capabilities=capabilities
                or self._default_stream_capabilities(provider_family),
                close_after_terminal=close_after_terminal,
            )
            self._output = items
            self._output_closed = False
            return _ResponseOwnedStreamIterator(
                self._record_canonical_stream_final_text(items),
                self.aclose,
                self._is_stream_closed,
            )

        return _ResponseOwnedStreamIterator(
            self._canonical_stream_from_output(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family,
                capabilities=capabilities
                or self._default_stream_capabilities(provider_family),
                close_after_terminal=close_after_terminal,
            ),
            self.aclose,
            self._is_stream_closed,
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
        usage: object | None = None,
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

        terminal_usage = usage if usage is not None else self._provider_usage()
        yield CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage=cast(Any, terminal_usage or {}),
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
        self._start_stream_output()
        assert self._output is not None
        primary_failure: BaseException | None = None
        try:
            try:
                first_item = await self._output.__anext__()
            except StopAsyncIteration:
                first = None
            else:
                first = await self._record_returned_item(first_item)

            if first is not None:
                yield first
                while True:
                    try:
                        item = await self.__anext__()
                    except StopAsyncIteration:
                        break
                    yield item
                return

            await self._call_output_cleanup("aclose")
            self._output_closed = True
            provider_usage = self._provider_usage()
            async for item in self._record_canonical_stream_final_text(
                self._canonical_stream_from_final_text(
                    "",
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    provider_family=provider_family,
                    capabilities=capabilities,
                    close_after_terminal=close_after_terminal,
                    usage=provider_usage,
                ),
                count_output=False,
            ):
                yield item
        except BaseException as error:
            primary_failure = error
            await self._settle_iteration_failure(error)
            raise
        finally:
            if primary_failure is None:
                await self._close_session_sources(
                    trigger_callbacks=(
                        self._session_state is _StreamSessionState.ACTIVE
                        and self._validation_failure_message is None
                    ),
                    mark_closed=(
                        self._session_state
                        is not _StreamSessionState.FINALIZED
                    ),
                )

    async def _record_canonical_stream_final_text(
        self,
        items: AsyncIterator[CanonicalStreamItem],
        *,
        count_output: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        self._output = items
        self._output_closed = False
        self._count_stream_output_items = count_output
        if self._stream_accumulator is None:
            self._stream_accumulator = CanonicalStreamAccumulator()
        completed = False
        primary_failure: BaseException | None = None
        try:
            async for item in items:
                yield await self._record_returned_item(
                    item, count_output=count_output
                )
            await self._finalize_stream_accumulation(
                raise_terminal_exception=False
            )
            completed = True
        except BaseException as error:
            primary_failure = error
            await self._settle_iteration_failure(error)
            raise
        finally:
            if primary_failure is None:
                await self._close_session_sources(
                    trigger_callbacks=(
                        not completed
                        and self._session_state is _StreamSessionState.ACTIVE
                        and self._validation_failure_message is None
                    ),
                    mark_closed=not completed,
                )

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
        if (
            self._session_state is _StreamSessionState.ACTIVE
            and self._output is not None
            and not self._output_closed
        ):
            return self
        self._claim_stream_session()
        self._start_stream_output()
        return self

    @staticmethod
    async def _single_item_output_generator(
        item: object,
    ) -> AsyncIterator[object]:
        yield item

    async def aclose(self) -> None:
        cleanup_failures = self._reap_cleanup_tasks(exclude="aclose")
        try:
            await self._run_bounded_cleanup_stage(
                "aclose",
                lambda: self._close_session_sources(
                    trigger_callbacks=True,
                    mark_closed=True,
                ),
            )
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
        if cleanup_failures:
            primary_failure = cleanup_failures[0]
            self._attach_cleanup_failures(
                primary_failure,
                cleanup_failures[1:],
            )
            raise primary_failure

    def _is_stream_closed(self) -> bool:
        return self._output_closed and self._session_state in (
            _StreamSessionState.CLOSED_CANCELLED,
            _StreamSessionState.FINALIZED,
        )

    async def cancel(self) -> None:
        if self.cleanup_complete and self._session_state in (
            _StreamSessionState.CLOSED_CANCELLED,
            _StreamSessionState.FINALIZED,
        ):
            return
        await self._run_bounded_cleanup_stage(
            "cancel",
            lambda: self._call_output_cleanup("cancel"),
        )
        await self.aclose()

    @staticmethod
    def _observe_cleanup_task(task: Task[None]) -> None:
        """Observe a retained cleanup task's eventual terminal failure."""
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            return

    def _reap_cleanup_tasks(self, *, exclude: str) -> list[BaseException]:
        """Collect completed retained cleanup failures exactly once."""
        cleanup_failures: list[BaseException] = []
        for stage, task in tuple(self._cleanup_tasks.items()):
            if stage == exclude or not task.done():
                continue
            if self._cleanup_tasks.get(stage) is task:
                self._cleanup_tasks.pop(stage)
            try:
                task.result()
            except CancelledError:
                continue
            except BaseException as cleanup_failure:
                cleanup_failures.append(cleanup_failure)
        return cleanup_failures

    async def _run_bounded_cleanup_stage(
        self,
        stage: str,
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        """Run or poll one explicitly owned cleanup stage within a deadline."""
        task = self._cleanup_tasks.get(stage)
        if task is None:

            async def run_cleanup() -> None:
                await cleanup()

            task = create_task(run_cleanup())
            task.add_done_callback(self._observe_cleanup_task)
            self._cleanup_tasks[stage] = task
        await sleep(0)
        if not task.done():
            await wait(
                {task},
                timeout=_PROVIDER_CLEANUP_TIMEOUT_SECONDS,
            )
        if not task.done():
            task.cancel()
            await sleep(0)
            raise TimeoutError(
                f"provider {stage} cleanup exceeded "
                f"{_PROVIDER_CLEANUP_TIMEOUT_SECONDS:g} seconds"
            )
        if self._cleanup_tasks.get(stage) is task:
            self._cleanup_tasks.pop(stage)
        task.result()

    async def _close_session_sources(
        self,
        *,
        trigger_callbacks: bool,
        mark_closed: bool,
    ) -> None:
        should_trigger_callbacks = (
            trigger_callbacks and self._validation_failure_message is None
        )
        if should_trigger_callbacks:
            await self._finalize_seen_terminal()
        if self._output_closed and (
            self._session_state is _StreamSessionState.CLOSED_CANCELLED
            or self._session_state is _StreamSessionState.FINALIZED
        ):
            if should_trigger_callbacks:
                await self._trigger_consumed()
            return
        await self._call_output_cleanup("aclose")
        self._output_closed = True
        if mark_closed and (
            self._session_state is not _StreamSessionState.FINALIZED
        ):
            self._session_state = _StreamSessionState.CLOSED_CANCELLED
        if should_trigger_callbacks:
            await self._trigger_consumed()

    async def _finalize_seen_terminal(self) -> None:
        accumulator = self._stream_accumulator
        terminal_outcome = getattr(accumulator, "terminal_outcome", None)
        if accumulator is None or terminal_outcome is None:
            return
        await self._finalize_stream_accumulation(
            raise_terminal_exception=False
        )

    async def _call_output_cleanup(self, method_name: str) -> None:
        assert method_name in ("cancel", "aclose")
        cleanup_source_ids = (
            self._cancelled_source_ids
            if method_name == "cancel"
            else self._closed_source_ids
        )
        seen: set[int] = set()
        for source in (self._output, self._output_fn):
            if source is None:
                continue
            source_id = id(source)
            if source_id in seen or source_id in cleanup_source_ids:
                continue
            seen.add(source_id)
            cleanup_key = (method_name, source_id)
            cleanup_failure = self._cleanup_failures.get(cleanup_key)
            if cleanup_failure is not None:
                raise cleanup_failure
            method = getattr(source, method_name, None)
            if method is None:
                continue
            assert callable(method)
            try:
                await self._close_output(cast(Callable[[], object], method))
            except Exception as exc:
                if isinstance(exc, _TextGenerationWorkerShutdownError):
                    self._cleanup_failures[cleanup_key] = exc
                raise
            cleanup_source_ids.add(source_id)

    @staticmethod
    def _attach_cleanup_failures(
        primary_failure: BaseException,
        cleanup_failures: list[BaseException],
    ) -> None:
        """Attach provider cleanup failures without replacing the primary."""
        seen_failure_ids = {id(primary_failure)}
        for cleanup_failure in cleanup_failures:
            if id(cleanup_failure) in seen_failure_ids:
                continue
            seen_failure_ids.add(id(cleanup_failure))
            primary_failure.add_note(
                "post-provider cleanup failure: "
                f"{cleanup_failure.__class__.__name__}: "
                f"{cleanup_failure}"
            )

    async def _settle_iteration_failure(
        self,
        primary_failure: BaseException,
    ) -> None:
        """Close a failed provider read while preserving its primary exit."""
        cleanup_failures: list[BaseException] = []
        interrupted = isinstance(
            primary_failure,
            _INTERRUPTED_PROVIDER_EXCEPTIONS,
        )
        if interrupted:
            try:
                await self._run_bounded_cleanup_stage(
                    "cancel",
                    lambda: self._call_output_cleanup("cancel"),
                )
            except BaseException as cleanup_failure:
                cleanup_failures.append(cleanup_failure)
        try:
            await self._run_bounded_cleanup_stage(
                "aclose",
                lambda: self._close_session_sources(
                    trigger_callbacks=interrupted,
                    mark_closed=True,
                ),
            )
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
        self._attach_cleanup_failures(primary_failure, cleanup_failures)

    @staticmethod
    async def _close_output(aclose: Callable[[], object]) -> None:
        result = aclose()
        if isawaitable(result):
            awaited_result = await cast(Awaitable[object], result)
            assert awaited_result is None
        else:
            assert result is None

    async def __anext__(self) -> CanonicalStreamItem:
        if self._is_stream_closed():
            raise StopAsyncIteration
        assert self._output
        try:
            item = await self._output.__anext__()
        except StopAsyncIteration:
            try:
                await self._finalize_stream_accumulation(
                    raise_terminal_exception=False
                )
            except BaseException as error:
                if isinstance(error, StreamValidationError):
                    self._validation_failure_message = str(error)
                await self._settle_iteration_failure(error)
                raise
            await self._close_session_sources(
                trigger_callbacks=False,
                mark_closed=False,
            )
            raise
        except BaseException as error:
            await self._settle_iteration_failure(error)
            raise

        return await self._record_returned_item(
            item, count_output=self._count_stream_output_items
        )

    async def _record_returned_item(
        self,
        item: object,
        *,
        count_output: bool = True,
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
            if count_output:
                self._output_token_count += 1
            return item
        except BaseException as error:
            if isinstance(error, StreamValidationError):
                self._validation_failure_message = str(error)
            await self._settle_iteration_failure(error)
            raise

    def __str__(self) -> str:
        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            return self._prefetched_text or ""
        return super().__str__()

    async def to_str(
        self,
        *,
        raise_terminal_exception: bool = True,
    ) -> str:
        assert isinstance(raise_terminal_exception, bool)
        terminal_exception = self._terminal_exception_from_state()
        if terminal_exception is not None:
            if raise_terminal_exception or isinstance(
                terminal_exception, StreamValidationError
            ):
                raise terminal_exception
            accumulator = self._stream_accumulator
            await self._trigger_consumed()
            return accumulator.answer_text if accumulator is not None else ""

        if self._final_text is not None:
            await self._trigger_consumed()
            return self._final_text

        if self._has_active_stream_session():
            return await self._drain_stream_to_final_text(
                raise_terminal_exception=raise_terminal_exception
            )

        if self._session_state is _StreamSessionState.ACTIVE:
            if self._output is None:
                raise RuntimeError(_SINGLE_USE_STREAM_ERROR)
            return await self._drain_stream_to_final_text(
                raise_terminal_exception=raise_terminal_exception
            )

        if not self._use_async_generator:
            self._ensure_non_stream_prefetched()
            if self._prefetched_text is None:
                return ""
            await self._trigger_consumed()
            self._final_text = self._prefetched_text
            return self._prefetched_text

        if self._session_state is _StreamSessionState.CLOSED_CANCELLED:
            accumulator = self._stream_accumulator
            self._final_text = (
                accumulator.answer_text if accumulator is not None else ""
            )
            await self._trigger_consumed()
            return self._final_text

        self._claim_stream_session()
        self._start_stream_output()
        assert self._output is not None
        return await self._drain_stream_to_final_text(
            raise_terminal_exception=raise_terminal_exception
        )

    def _has_active_stream_session(self) -> bool:
        return (
            self._output is not None
            and not self._output_closed
            and self._stream_accumulator is not None
        )

    async def _drain_stream_to_final_text(
        self,
        *,
        raise_terminal_exception: bool = True,
    ) -> str:
        assert isinstance(raise_terminal_exception, bool)
        assert self._output is not None
        while True:
            try:
                await self.__anext__()
            except StopAsyncIteration:
                break

        return await self._finalize_stream_accumulation(
            raise_terminal_exception=raise_terminal_exception
        )

    async def _finalize_stream_accumulation(
        self,
        *,
        raise_terminal_exception: bool,
    ) -> str:
        terminal_exception = self._terminal_exception_from_state()
        if terminal_exception is not None:
            if raise_terminal_exception or isinstance(
                terminal_exception, StreamValidationError
            ):
                raise terminal_exception
            accumulator = self._stream_accumulator
            return accumulator.answer_text if accumulator is not None else ""

        accumulator = self._stream_accumulator
        if accumulator is None:
            if self._final_text is not None:
                self._session_state = _StreamSessionState.FINALIZED
                await self._trigger_consumed()
                return self._final_text
            self._final_text = ""
            self._session_state = _StreamSessionState.FINALIZED
            await self._trigger_consumed()
            return self._final_text

        try:
            accumulator.validate_complete()
        except StreamValidationError as exc:
            self._validation_failure_message = str(exc)
            raise

        terminal_exception = self._remember_terminal_exception(accumulator)
        if terminal_exception is not None:
            self._session_state = _StreamSessionState.FINALIZED
            await self._trigger_consumed()
            if raise_terminal_exception:
                raise terminal_exception
            return accumulator.answer_text

        if self._final_text is not None:
            self._session_state = _StreamSessionState.FINALIZED
            await self._trigger_consumed()
            return self._final_text

        self._final_text = accumulator.answer_text
        self._session_state = _StreamSessionState.FINALIZED
        await self._trigger_consumed()
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
