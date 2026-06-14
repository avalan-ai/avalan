from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    Token,
)
from ...event import Event
from ...model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
    stream_consumer_projection_from_token,
)
from ...server.entities import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from ...utils import to_json
from .. import di_get_logger, di_get_orchestrator
from ..sse import sse_headers, sse_message
from . import orchestrate, resolve_model_id
from .streaming import (
    cleanup_stream_sources,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from logging import Logger
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/chat",
    tags=["completions"],
)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
) -> ChatCompletionResponse | StreamingResponse:
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages
    model_id = resolve_model_id(orchestrator, request.model)

    logger.info(
        "Processing chat completion request for orchestrator %s",
        str(orchestrator),
    )
    logger.debug(
        "Processing chat completion request with messages %r", request
    )

    response, response_id, timestamp = await orchestrate(
        request, logger, orchestrator
    )

    logger.info(
        "Orchestrator %s responded for chat completion request",
        str(orchestrator),
    )

    # Streaming through SSE (server-sent events with text/event-stream)
    if request.stream:

        async def generate_chunks() -> AsyncIterator[str]:
            sequence = 0
            canonical_accumulator: CanonicalStreamAccumulator | None = None
            legacy_stream_seen = False
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="chat-sse-stream",
                run_id=response_id,
                turn_id="chat-sse-turn",
            )
            cancelled = False
            try:
                sync_messages = True
                while True:
                    try:
                        token = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    if isinstance(token, Event):
                        continue
                    if isinstance(token, CanonicalStreamItem):
                        if legacy_stream_seen:
                            raise StreamValidationError(
                                "canonical stream item after legacy stream"
                                " item"
                            )
                        if canonical_accumulator is None:
                            canonical_accumulator = (
                                CanonicalStreamAccumulator()
                            )
                        canonical_accumulator.add(token)
                    elif isinstance(token, StreamConsumerProjection):
                        if legacy_stream_seen:
                            raise StreamValidationError(
                                "canonical stream item after legacy stream"
                                " item"
                            )
                        if canonical_accumulator is None:
                            canonical_accumulator = (
                                CanonicalStreamAccumulator()
                            )
                        canonical_accumulator.add(
                            canonical_item_from_consumer_projection(token)
                        )
                    elif canonical_accumulator is not None:
                        raise StreamValidationError(
                            "legacy stream item after canonical stream item"
                        )
                    else:
                        legacy_stream_seen = True

                    projected_text = _stream_text(
                        _stream_projection(token, sequence)
                    )
                    sequence += 1
                    if projected_text is None:
                        continue

                    choice = ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkChoiceDelta(
                            content=projected_text
                        )
                    )
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        created=timestamp,
                        model=model_id,
                        choices=[choice],
                    )
                    yield sse_message(chunk.model_dump_json())

                if canonical_accumulator is not None:
                    canonical_accumulator.validate_complete()
                    terminal = _chat_terminal_projection(canonical_accumulator)
                    terminal_event = _chat_terminal_event(
                        response_id,
                        timestamp,
                        model_id,
                        terminal,
                    )
                    usage = _chat_usage(canonical_accumulator.final_usage)
                    if usage is not None:
                        yield _chat_usage_chunk(
                            response_id,
                            timestamp,
                            model_id,
                            usage,
                        )
                    if terminal_event is not None:
                        yield terminal_event
                    sync_messages = stream_terminal_succeeded(terminal)
                else:
                    terminal_event = None

                if terminal_event is None:
                    yield sse_message("[DONE]")

                if sync_messages:
                    await orchestrator.sync_messages()
            except CancelledError:
                cancelled = True
                raise
            finally:
                await cleanup_stream_sources(
                    response,
                    iterator,
                    cancelled=cancelled,
                )

        logger.debug(
            "Generating event-stream stream for response %s", response_id
        )

        return StreamingResponse(
            generate_chunks(),
            media_type="text/event-stream",
            headers=sse_headers(),
        )

    # Non streaming
    text = await response.to_str()
    choices = [
        ChatCompletionChoice(
            index=i,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            finish_reason="stop",
        )
        for i in range(request.n or 1)
    ]
    usage = ChatCompletionUsage()
    final_response = ChatCompletionResponse(
        id=response_id,
        created=timestamp,
        model=model_id,
        choices=choices,
        usage=usage,
    )
    logger.debug(
        "Generated chat completion response #%s %r",
        response_id,
        final_response,
    )

    await orchestrator.sync_messages()

    return final_response


def _chat_terminal_event(
    response_id: str,
    timestamp: int,
    model_id: str,
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
) -> str | None:
    assert terminal is None or isinstance(
        terminal, (StreamConsumerProjection, StreamTerminalOutcome)
    )
    if isinstance(terminal, StreamConsumerProjection):
        assert terminal.is_stream_terminal
    terminal_outcome = (
        terminal.terminal_outcome
        if isinstance(terminal, StreamConsumerProjection)
        else terminal
    )
    if (
        terminal_outcome is None
        or terminal_outcome is StreamTerminalOutcome.COMPLETED
    ):
        return None

    event = (
        "chat.completion.cancelled"
        if terminal_outcome is StreamTerminalOutcome.CANCELLED
        else "chat.completion.failed"
    )
    data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model_id,
        "type": event,
        "choices": [],
    }
    if isinstance(terminal, StreamConsumerProjection):
        data["sequence_number"] = terminal.sequence
        if (
            terminal_outcome is StreamTerminalOutcome.ERRORED
            and terminal.data is not None
        ):
            data["error"] = terminal.data
    return sse_message(to_json(data), event=event)


def _chat_terminal_projection(
    accumulator: CanonicalStreamAccumulator,
) -> StreamConsumerProjection | None:
    assert isinstance(accumulator, CanonicalStreamAccumulator)
    for item in reversed(accumulator.items):
        if item.is_stream_terminal:
            return _stream_projection(item, item.sequence)
    return None


def _stream_projection(
    token: CanonicalStreamItem | StreamConsumerProjection | Token | str,
    sequence: int,
) -> StreamConsumerProjection:
    return stream_consumer_projection_from_token(token, sequence)


def _stream_text(
    token: StreamConsumerProjection,
) -> str | None:
    assert isinstance(token, StreamConsumerProjection)
    if token.kind is not StreamItemKind.ANSWER_DELTA:
        return None
    return token.text_delta or ""


def _chat_usage(usage: object | None) -> ChatCompletionUsage | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return ChatCompletionUsage()

    prompt_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _usage_int(usage: dict[object, object], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def _chat_usage_chunk(
    response_id: str,
    timestamp: int,
    model_id: str,
    usage: ChatCompletionUsage,
) -> str:
    data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model_id,
        "choices": [],
        "usage": usage.model_dump(),
    }
    return sse_message(to_json(data))
