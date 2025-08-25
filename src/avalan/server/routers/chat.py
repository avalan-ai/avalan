from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...entities import (
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
    MessageRole,
)
from ...event import Event
from ...server.entities import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionUsage,
    ContentImage,
    ContentText,
)
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from logging import Logger
from time import time
from uuid import uuid4

router = APIRouter(
    prefix="/chat",
    tags=["completions"],
)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
):
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages

    logger.info(
        "Processing chat completion request for orchestrator %s",
        str(orchestrator),
    )
    logger.debug(
        "Processing chat completion request with messages %r", request
    )

    messages = [
        Message(role=req.role, content=_to_message_content(req.content))
        for req in request.messages
    ]

    logger.debug(
        "Transformed chat completion request to engine input %r", messages
    )

    if request.stream and (request.n or 1) > 1:
        raise ValueError("Streaming multiple completions is not supported")

    response_id = uuid4()
    timestamp = int(time())

    settings = GenerationSettings(
        use_async_generator=request.stream,
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        stop_strings=request.stop,
        top_p=request.top_p,
        num_return_sequences=request.n,
        response_format=(
            request.response_format.model_dump(
                by_alias=True, exclude_none=True
            )
            if request.response_format
            else None
        ),
    )

    logger.debug(
        "Calling orchestrator with input %r and settings %r for response %s",
        messages,
        settings,
        response_id,
    )

    logger.info(
        "Calling orchestrator %s for chat completion request",
        str(orchestrator),
    )

    response = await orchestrator(messages, settings=settings)

    logger.info(
        "Orchestrator %s responded for chat completion request",
        str(orchestrator),
    )

    # Streaming through SSE (server-sent events with text/event-stream)
    if request.stream:

        async def generate_chunks():
            async for token in response:
                if isinstance(token, Event):
                    continue

                choice = ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=token)
                )
                chunk = ChatCompletionChunk(
                    id=str(response_id),
                    created=timestamp,
                    model=request.model,
                    choices=[choice],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"  # SSE data event
            yield "data: [DONE]\n\n"  # end of stream

            await orchestrator.sync_messages()

        logger.debug(
            f"Generating event-stream stream for response {response_id}"
        )

        return StreamingResponse(
            generate_chunks(), media_type="text/event-stream"
        )

    # Non streaming
    text = await response.to_str()
    choices = [
        ChatCompletionChoice(
            index=i,
            message=ChatMessage(role=str(MessageRole.ASSISTANT), content=text),
            finish_reason="stop",
        )
        for i in range(request.n or 1)
    ]
    usage = ChatCompletionUsage()
    response = ChatCompletionResponse(
        id=str(response_id),
        created=timestamp,
        model=request.model,
        choices=choices,
        usage=usage,
    )
    logger.debug(
        "Generated chat completion response #%s %r", response_id, response
    )

    await orchestrator.sync_messages()

    return response


def _to_message_content(item):
    if isinstance(item, list):
        return [
            _to_message_content(i)
            for i in item
            if isinstance(i, (ContentImage, ContentText, str))
        ]
    if isinstance(item, ContentImage):
        return MessageContentImage(type=item.type, image_url=item.image_url)
    if isinstance(item, ContentText):
        return MessageContentText(type=item.type, text=item.text)
    if isinstance(item, str):
        return MessageContentText(type="text", text=item)
    raise TypeError(f"Unsupported content type: {type(item).__name__}")
