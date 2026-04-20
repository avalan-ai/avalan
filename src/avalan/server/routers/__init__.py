from ...agent.orchestrator import Orchestrator
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...entities import (
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
)
from ...entities import (
    ReasoningToken as ReasoningToken,
)
from ...entities import (
    ToolCallToken as ToolCallToken,
)
from ...server.entities import (
    ChatCompletionRequest,
    ContentImage,
    ContentText,
    ResponsesRequest,
)

from logging import Logger
from time import time
from typing import TypeAlias, cast
from uuid import uuid4

from fastapi import HTTPException

MessageContentInput: TypeAlias = (
    str | ContentImage | ContentText | list[ContentImage | ContentText]
)
MessageContentOutput: TypeAlias = (
    MessageContentText
    | MessageContentImage
    | list[MessageContentText | MessageContentImage]
)


async def orchestrate(
    request: ChatCompletionRequest | ResponsesRequest,
    logger: Logger,
    orchestrator: Orchestrator,
) -> tuple[OrchestratorResponse, str, int]:
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
        use_async_generator=bool(request.stream),
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
    return response, str(response_id), timestamp


def to_message_content(item: MessageContentInput) -> MessageContentOutput:
    if isinstance(item, list):
        return [
            cast(
                MessageContentText | MessageContentImage, to_message_content(i)
            )
            for i in item
            if isinstance(i, (ContentImage, ContentText))
        ]
    if isinstance(item, ContentImage):
        return MessageContentImage(type=item.type, image_url=item.image_url)
    if isinstance(item, ContentText):
        return MessageContentText(type=item.type, text=item.text)
    if isinstance(item, str):
        return MessageContentText(type="text", text=item)
    raise TypeError(f"Unsupported content type: {type(item).__name__}")
