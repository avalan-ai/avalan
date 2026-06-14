from ...model.stream import (
    StreamConsumerProjection,
    StreamTerminalOutcome,
    StreamValidationError,
)

from asyncio import CancelledError
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from inspect import isawaitable
from typing import Any, cast


def stream_iterator(source: object) -> AsyncIterator[Any]:
    assert isinstance(source, AsyncIterable)
    return source.__aiter__()


def stream_consumer_iterator(
    source: object,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
) -> AsyncIterator[Any]:
    consumer_projections = getattr(source, "consumer_projections", None)
    if callable(consumer_projections):
        iterator = consumer_projections(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
        )
        assert isinstance(iterator, AsyncIterable)
        return _validated_consumer_projection_iterator(iterator.__aiter__())
    return stream_iterator(source)


async def _validated_consumer_projection_iterator(
    iterator: AsyncIterator[Any],
) -> AsyncIterator[StreamConsumerProjection]:
    try:
        async for item in iterator:
            if not isinstance(item, StreamConsumerProjection):
                raise StreamValidationError(
                    "consumer projection stream item must be "
                    "StreamConsumerProjection"
                )
            yield item
    finally:
        await _call_optional(iterator, "aclose")


def stream_terminal_succeeded(
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
) -> bool:
    assert terminal is None or isinstance(
        terminal, (StreamConsumerProjection, StreamTerminalOutcome)
    )
    outcome = (
        terminal.terminal_outcome
        if isinstance(terminal, StreamConsumerProjection)
        else terminal
    )
    return outcome is None or outcome is StreamTerminalOutcome.COMPLETED


async def cleanup_stream_sources(
    *sources: object,
    cancelled: bool = False,
) -> None:
    assert isinstance(cancelled, bool)
    seen: set[int] = set()
    unique_sources: list[object] = []
    for source in sources:
        source_id = id(source)
        if source_id in seen:
            continue
        seen.add(source_id)
        unique_sources.append(source)

    errors: list[BaseException] = []
    if cancelled:
        for source in unique_sources:
            await _call_optional_collecting_errors(source, "cancel", errors)

    for source in unique_sources:
        await _call_optional_collecting_errors(source, "aclose", errors)

    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("stream source cleanup failed", errors)


async def _call_optional_collecting_errors(
    source: object, method_name: str, errors: list[BaseException]
) -> None:
    try:
        await _call_optional(source, method_name)
    except (Exception, CancelledError) as exc:
        errors.append(exc)


async def _call_optional(source: object, method_name: str) -> None:
    method = getattr(source, method_name, None)
    if method is None:
        return
    assert callable(method)
    result = method()
    if isawaitable(result):
        awaited_result = await cast(Awaitable[object], result)
        assert awaited_result is None
    else:
        assert result is None
