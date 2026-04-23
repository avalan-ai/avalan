from ...agent.orchestrator import Orchestrator
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageFile,
    ReasoningSettings,
)
from ...entities import (
    ReasoningToken as ReasoningToken,
)
from ...entities import (
    ToolCallToken as ToolCallToken,
)
from ...server.entities import (
    ChatCompletionRequest,
    ContentFile,
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
    str
    | ContentFile
    | ContentImage
    | ContentText
    | list[ContentFile | ContentImage | ContentText]
)
MessageContentOutput: TypeAlias = (
    MessageContentText
    | MessageContentFile
    | MessageContentImage
    | list[MessageContentFile | MessageContentText | MessageContentImage]
)
MODEL_FALLBACK = "default"


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
        reasoning=ReasoningSettings(
            effort=(
                request.reasoning.effort
                if isinstance(request, ResponsesRequest)
                and request.reasoning is not None
                else getattr(request, "reasoning_effort", None)
            )
        ),
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


def resolve_model_id(
    orchestrator: Orchestrator, request_model: str | None = None
) -> str:
    if request_model:
        return request_model
    model_ids = getattr(orchestrator, "model_ids", None)
    if model_ids:
        candidates = sorted(str(model_id) for model_id in model_ids)
        if candidates:
            return candidates[0]
    return MODEL_FALLBACK


def to_message_content(item: MessageContentInput) -> MessageContentOutput:
    if isinstance(item, list):
        return [
            cast(
                MessageContentFile | MessageContentText | MessageContentImage,
                to_message_content(i),
            )
            for i in item
            if isinstance(i, (ContentFile, ContentImage, ContentText))
        ]
    if isinstance(item, ContentFile):
        file_data = cast(MessageFile, dict(item.file or {}))
        if item.file_id is not None:
            file_data["file_id"] = item.file_id
        if item.file_url is not None:
            file_data["file_url"] = item.file_url
        if item.file_data is not None:
            file_data["file_data"] = item.file_data
        if item.filename is not None:
            file_data["filename"] = item.filename
        return MessageContentFile(type="file", file=file_data)
    if isinstance(item, ContentImage):
        return MessageContentImage(type=item.type, image_url=item.image_url)
    if isinstance(item, ContentText):
        return MessageContentText(type="text", text=item.text)
    if isinstance(item, str):
        return MessageContentText(type="text", text=item)
    raise TypeError(f"Unsupported content type: {type(item).__name__}")
