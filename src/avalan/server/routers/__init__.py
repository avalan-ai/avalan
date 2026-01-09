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
from ...server.entities import ChatCompletionRequest, ContentImage, ContentText

from logging import Logger
from time import time
from uuid import UUID, uuid4

from fastapi import HTTPException


async def orchestrate(
    request: ChatCompletionRequest,
    logger: Logger,
    orchestrator: Orchestrator,
) -> tuple[OrchestratorResponse, UUID, int]:
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
        use_async_generator=request.stream or False,
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


def _convert_single_content(
    item: str | ContentText | ContentImage,
) -> MessageContentText | MessageContentImage:
    """Convert a single content item to message content type."""
    if isinstance(item, ContentImage):
        return MessageContentImage(type=item.type, image_url=item.image_url)
    if isinstance(item, ContentText):
        return MessageContentText(type=item.type, text=item.text)
    if isinstance(item, str):
        return MessageContentText(type="text", text=item)
    raise TypeError(f"Unsupported content type: {type(item).__name__}")


def to_message_content(
    item: str | list[ContentText | ContentImage],
) -> (
    MessageContentText
    | MessageContentImage
    | list[MessageContentText | MessageContentImage]
):
    """Convert request content to message content types."""
    if isinstance(item, list):
        return [
            _convert_single_content(i)
            for i in item
            if isinstance(i, (ContentImage, ContentText, str))
        ]
    if not isinstance(item, (str, ContentText, ContentImage)):
        raise TypeError(f"Unsupported content type: {type(item).__name__}")
    return _convert_single_content(item)
