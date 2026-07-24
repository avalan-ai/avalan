from ...agent.orchestrator import Orchestrator
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
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
    ServerOutputRedactionSettings,
    coerce_server_output_redaction_settings,
    sanitize_model_visible_server_protocol_text,
    sanitize_server_protocol_value,
    server_output_redaction_settings_from_state,
)
from ...utils import to_json
from .. import di_get_logger, di_get_orchestrator
from ..interaction import (
    ServerInteractionHandling,
    ServerInteractionHTTPError,
    ServerInteractionRun,
    ServerInteractionSurface,
    _error_response,
    detached_segment,
    extension_sse_message,
    interaction_response_headers,
    prepare_openai_interaction_run,
    task_input_extension_from_request,
)
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
from typing import cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

_JSON_SEPARATORS = (",", ":")
_NO_HTTP_REQUEST = cast(Request, None)
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


def _server_output_redaction_settings(
    request: Request,
) -> ServerOutputRedactionSettings:
    return server_output_redaction_settings_from_state(request.app.state)


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(validate_remote_container_profile_selection)],
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
    output_redaction_settings: ServerOutputRedactionSettings = Depends(
        _server_output_redaction_settings
    ),
    http_request: Request = _NO_HTTP_REQUEST,
) -> ChatCompletionResponse | JSONResponse | StreamingResponse:
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages
    model_id = resolve_model_id(orchestrator, request.model)
    interaction_run = None
    if isinstance(http_request, Request):
        try:
            interaction_run = await prepare_openai_interaction_run(
                http_request,
                task_input_extension_from_request(request),
                surface=ServerInteractionSurface.CHAT,
            )
        except ServerInteractionHTTPError as error:
            return _error_response(error)

    logger.info(
        "Processing chat completion request for orchestrator %s",
        str(orchestrator),
    )
    logger.debug(
        "Processing chat completion request with messages %r", request
    )

    if interaction_run is None:
        response, response_id, timestamp = await orchestrate(
            request,
            logger,
            orchestrator,
        )
    else:
        response, response_id, timestamp = await orchestrate(
            request,
            logger,
            orchestrator,
            interaction_runtime=interaction_run.runtime,
        )
    output_redaction_settings = coerce_server_output_redaction_settings(
        output_redaction_settings
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
            segment = (
                detached_segment(
                    iterator=iterator,
                    response=response,
                    orchestrator=orchestrator,
                    protocol=ServerInteractionSurface.CHAT,
                    response_id=response_id,
                    created=timestamp,
                    model_id=model_id,
                    output_redaction_settings=output_redaction_settings,
                    choice_count=request.n or 1,
                )
                if interaction_run is not None
                else None
            )
            cancelled = False
            retained = False
            final_usage: object | None = None
            terminal: StreamConsumerProjection | None = None
            answer_redactor = ModelVisibleServerProtocolTextRedactor(
                output_redaction_settings,
                protocol="openai",
                channel="answer",
            )
            try:
                while True:
                    try:
                        token = (
                            await segment.next_projection()
                            if segment is not None
                            else await anext(iterator)
                        )
                    except StopAsyncIteration:
                        break

                    if token.usage is not None:
                        final_usage = token.usage
                    if token.is_stream_terminal:
                        terminal = token
                    if interaction_run is not None:
                        for event in await interaction_run.extension_events(
                            token
                        ):
                            yield extension_sse_message(event)
                        if (
                            interaction_run.handling
                            is ServerInteractionHandling.DETACHED
                            and token.kind
                            is StreamItemKind.INTERACTION_PENDING
                        ):
                            assert segment is not None
                            await interaction_run.install_segment(segment)
                            yield extension_sse_message(
                                await interaction_run.input_required_event()
                            )
                            retained = True
                            return
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
                    output_redaction_settings=output_redaction_settings,
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
                    await orchestrator.sync_messages(response)
            except CancelledError:
                cancelled = True
                if interaction_run is not None and segment is not None:
                    try:
                        await interaction_run.install_segment(segment)
                    except RuntimeError:
                        pass
                    retained = True
                raise
            finally:
                if not retained:
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
            headers={
                **sse_headers(),
                **interaction_response_headers(interaction_run),
            },
        )

    if interaction_run is not None:
        return await _interaction_chat_response(
            request=request,
            interaction_run=interaction_run,
            response=response,
            response_id=response_id,
            timestamp=timestamp,
            model_id=model_id,
            orchestrator=orchestrator,
            output_redaction_settings=output_redaction_settings,
        )

    # Non streaming
    text = sanitize_model_visible_server_protocol_text(
        await response.to_str(),
        output_redaction_settings=output_redaction_settings,
        protocol="openai",
        channel="answer",
    )
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

    await orchestrator.sync_messages(response)

    return final_response


async def _interaction_chat_response(
    *,
    request: ChatCompletionRequest,
    interaction_run: ServerInteractionRun,
    response: OrchestratorResponse,
    response_id: str,
    timestamp: int,
    model_id: str,
    orchestrator: Orchestrator,
    output_redaction_settings: ServerOutputRedactionSettings,
) -> JSONResponse:
    """Return an interaction envelope or a completed Chat response."""
    iterator = stream_consumer_iterator(
        response,
        stream_session_id="chat-non-stream",
        run_id=str(response_id),
        turn_id="chat-non-stream-turn",
        unsupported_message=(
            "unsupported stream item for Chat non-stream projection"
        ),
        close_source_on_generator_exit=False,
    )
    segment = detached_segment(
        iterator=iterator,
        response=response,
        orchestrator=orchestrator,
        protocol=ServerInteractionSurface.CHAT,
        response_id=response_id,
        created=timestamp,
        model_id=model_id,
        output_redaction_settings=output_redaction_settings,
        choice_count=request.n or 1,
    )
    redactor = ModelVisibleServerProtocolTextRedactor(
        output_redaction_settings,
        protocol="openai",
        channel="answer",
    )
    answer_parts: list[str] = []
    terminal: StreamConsumerProjection | None = None
    retained = False
    cancelled = False
    try:
        while True:
            try:
                projection = await segment.next_projection()
            except StopAsyncIteration:
                break
            if projection.is_stream_terminal:
                terminal = projection
            if (
                interaction_run.handling is ServerInteractionHandling.DETACHED
                and projection.kind is StreamItemKind.INTERACTION_PENDING
            ):
                await interaction_run.install_segment(segment)
                retained = True
                return JSONResponse(
                    await interaction_run.input_required_envelope(),
                    status_code=202,
                    headers=interaction_response_headers(interaction_run),
                )
            if projection.kind is StreamItemKind.ANSWER_DELTA:
                answer_parts.extend(redactor.push(projection.text_delta or ""))
        answer_parts.extend(redactor.flush())
        if terminal is None:
            raise StreamValidationError("stream missing terminal outcome")
        terminal_snapshot = protocol_stream_terminal_snapshot(terminal)
        if not terminal_snapshot.succeeded:
            raise StreamValidationError(
                "detached Chat request ended without completion"
            )
        text = "".join(answer_parts)
        body = ChatCompletionResponse(
            id=response_id,
            created=timestamp,
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=index,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=text,
                    ),
                    finish_reason="stop",
                )
                for index in range(request.n or 1)
            ],
            usage=ChatCompletionUsage(),
        )
        await orchestrator.sync_messages(response)
        return JSONResponse(
            body.model_dump(mode="json"),
            headers=interaction_response_headers(interaction_run),
        )
    except CancelledError:
        cancelled = True
        try:
            await interaction_run.install_segment(segment)
        except RuntimeError:
            pass
        retained = True
        raise
    finally:
        if not retained:
            await cleanup_stream_sources(
                response,
                iterator,
                cancelled=cancelled,
            )


def _chat_terminal_event(
    response_id: str,
    timestamp: int,
    model_id: str,
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str | None:
    terminal_snapshot = protocol_stream_terminal_snapshot(terminal)
    terminal_outcome = terminal_snapshot.outcome
    if (
        terminal_outcome is None
        or terminal_outcome is StreamTerminalOutcome.COMPLETED
    ):
        return None
    if terminal_outcome is StreamTerminalOutcome.INPUT_REQUIRED:
        raise StreamValidationError(
            "Chat input-required projection is unavailable"
        )

    event = (
        "chat.completion.cancelled"
        if terminal_outcome is StreamTerminalOutcome.CANCELLED
        else "chat.completion.failed"
    )
    assert terminal_outcome in (
        StreamTerminalOutcome.CANCELLED,
        StreamTerminalOutcome.ERRORED,
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
                terminal_snapshot.data,
                output_redaction_settings=output_redaction_settings,
                protocol="openai",
            )
    return sse_message(to_json(data), event=event)


def _stream_text(
    token: StreamConsumerProjection,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str | None:
    assert isinstance(token, StreamConsumerProjection)
    if token.kind is not StreamItemKind.ANSWER_DELTA:
        return None
    return sanitize_model_visible_server_protocol_text(
        token.text_delta or "",
        output_redaction_settings=output_redaction_settings,
        protocol="openai",
        channel="answer",
    )


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
