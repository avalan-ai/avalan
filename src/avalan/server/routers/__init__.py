from ...agent.orchestrator import Orchestrator
from ...entities import (
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
    ReasoningToken as ReasoningToken,
    ToolCallToken as ToolCallToken,
)
from ...server.entities import ChatCompletionRequest, ContentImage, ContentText
from logging import Logger
from time import time
from uuid import uuid4
from fastapi import HTTPException


async def orchestrate(
    request: ChatCompletionRequest,
    logger: Logger,
    orchestrator: Orchestrator,
):
    messages = [
        Message(role=req.role, content=to_message_content(req.content))
        for req in request.messages
    ]

    if request.stream and (request.n or 1) > 1:
        raise HTTPException(
            status_code=400,
            detail="Streaming multiple completions is not supported",
        )

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

    response = await orchestrator(messages, settings=settings)
    return response, response_id, timestamp


def to_message_content(item):
    if isinstance(item, list):
        return [
            to_message_content(i)
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
