from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
)
from ...model.stream import (
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
)
from ...server.entities import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ModelVisibleServerProtocolTextRedactor,
    sanitize_model_visible_server_protocol_text,
    sanitize_server_protocol_value,
)
from ...utils import to_json
from .. import di_get_logger, di_get_orchestrator
from ..remote_container import validate_remote_container_profile_selection
from ..sse import sse_headers, sse_message
from . import orchestrate, resolve_model_id
from .streaming import (
    cleanup_stream_sources,
    protocol_stream_terminal_snapshot,
    protocol_stream_usage_mappings,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from json import dumps
from logging import Logger

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

_JSON_SEPARATORS = (",", ":")
_CHAT_COMPLETION_CHUNK_SUFFIX = "}}]}"
_CHAT_COMPLETION_FINAL_CHUNK_SUFFIX = (
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
)


@dataclass(frozen=True, slots=True)
class _ChatCompletionChunkEnvelope:
    response_id: str
    timestamp: int
    model_id: str
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert isinstance(self.response_id, str)
        assert isinstance(self.timestamp, int)
        assert not isinstance(self.timestamp, bool)
        assert isinstance(self.model_id, str)
        object.__setattr__(
            self,
            "_prefix",
            f'{{"id":{_json_string(self.response_id)},'
            '"object":"chat.completion.chunk",'
            f'"created":{self.timestamp},'
            f'"model":{_json_string(self.model_id)},'
            '"choices":[{"index":0,"delta":{"content":',
        )

    def chunk_json(self, content: str) -> str:
        assert isinstance(content, str)
        return (
            self._prefix
            + _json_string(content)
            + _CHAT_COMPLETION_CHUNK_SUFFIX
        )

    def final_chunk_json(self) -> str:
        return (
            f'{{"id":{_json_string(self.response_id)},'
            '"object":"chat.completion.chunk",'
            f'"created":{self.timestamp},'
            f'"model":{_json_string(self.model_id)},'
            + _CHAT_COMPLETION_FINAL_CHUNK_SUFFIX
        )

    def message(self, content: str) -> str:
        return sse_message(self.chunk_json(content))

    def final_message(self) -> str:
        return sse_message(self.final_chunk_json())


def _json_string(value: str) -> str:
    assert isinstance(value, str)
    return dumps(value, ensure_ascii=False, separators=_JSON_SEPARATORS)


router = APIRouter(
    prefix="/chat",
    tags=["completions"],
)


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(validate_remote_container_profile_selection)],
)
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
            chunk_envelope = _ChatCompletionChunkEnvelope(
                response_id=response_id,
                timestamp=timestamp,
                model_id=model_id,
            )
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="chat-sse-stream",
                run_id=str(response_id),
                turn_id="chat-sse-turn",
                unsupported_message=(
                    "unsupported stream item for Chat SSE projection"
                ),
                close_source_on_generator_exit=False,
            )
            cancelled = False
            final_usage: object | None = None
            terminal: StreamConsumerProjection | None = None
            answer_redactor = ModelVisibleServerProtocolTextRedactor()
            try:
                while True:
                    try:
                        token = await anext(iterator)
                    except StopAsyncIteration:
                        break

                    if token.usage is not None:
                        final_usage = token.usage
                    if token.is_stream_terminal:
                        terminal = token
                    projected_texts = _stream_text_fragments(
                        token,
                        answer_redactor,
                    )
                    if not projected_texts:
                        continue

                    for projected_text in projected_texts:
                        yield chunk_envelope.message(projected_text)

                for projected_text in answer_redactor.flush():
                    yield chunk_envelope.message(projected_text)

                if terminal is None:
                    raise StreamValidationError(
                        "stream missing terminal outcome"
                    )

                terminal_event = _chat_terminal_event(
                    response_id,
                    timestamp,
                    model_id,
                    terminal,
                )
                usage = _chat_usage(final_usage)
                if usage is not None:
                    yield _chat_usage_chunk(
                        response_id,
                        timestamp,
                        model_id,
                        usage,
                    )
                if terminal_event is not None:
                    yield terminal_event
                else:
                    yield chunk_envelope.final_message()
                    yield sse_message("[DONE]")

                if stream_terminal_succeeded(terminal):
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
    text = sanitize_model_visible_server_protocol_text(await response.to_str())
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
    terminal_snapshot = protocol_stream_terminal_snapshot(terminal)
    terminal_outcome = terminal_snapshot.outcome
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
    if terminal_snapshot.sequence is not None:
        data["sequence_number"] = terminal_snapshot.sequence
        if (
            terminal_outcome is StreamTerminalOutcome.ERRORED
            and terminal_snapshot.data is not None
        ):
            data["error"] = sanitize_server_protocol_value(
                terminal_snapshot.data
            )
    return sse_message(to_json(data), event=event)


def _stream_text(
    token: StreamConsumerProjection,
) -> str | None:
    assert isinstance(token, StreamConsumerProjection)
    if token.kind is not StreamItemKind.ANSWER_DELTA:
        return None
    return sanitize_model_visible_server_protocol_text(token.text_delta or "")


def _stream_text_fragments(
    token: StreamConsumerProjection,
    redactor: ModelVisibleServerProtocolTextRedactor,
) -> tuple[str, ...]:
    assert isinstance(token, StreamConsumerProjection)
    assert isinstance(redactor, ModelVisibleServerProtocolTextRedactor)
    if token.kind is not StreamItemKind.ANSWER_DELTA:
        return ()
    return redactor.push(token.text_delta or "")


def _chat_usage(usage: object | None) -> ChatCompletionUsage | None:
    if usage is None:
        return None
    usage_mappings = protocol_stream_usage_mappings(usage)
    if not usage_mappings:
        return ChatCompletionUsage()

    prompt_tokens = _usage_int(usage_mappings, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_int(
        usage_mappings, "completion_tokens", "output_tokens"
    )
    total_tokens = _usage_int(usage_mappings, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _usage_int(
    usage_mappings: tuple[Mapping[object, object], ...],
    *keys: str,
) -> int:
    for usage in usage_mappings:
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
