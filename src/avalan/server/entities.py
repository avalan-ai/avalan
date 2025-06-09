from ..entities import MessageRole
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        ..., description="ID of the model to use for generating the completion"
    )
    messages: list[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    temperature: float | None = Field(
        1.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(
        1.0, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    n: int | None = Field(
        1, ge=1, description="Number of completions to generate"
    )
    stream: bool | None = Field(
        False, description="Whether to stream back partial progress"
    )
    stop: str | list[str] | None = Field(
        None,
        description=(
            "Sequence where the API will stop generating further tokens"
        ),
    )
    max_tokens: int | None = Field(
        None, ge=1, description="Maximum tokens to generate in the completion"
    )
    presence_penalty: float | None = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for new tokens based on whether they appear in text"
            " so far"
        ),
    )
    frequency_penalty: float | None = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for new tokens based on their frequency in text so far"
        ),
    )
    logit_bias: dict[str, int] | None = Field(
        None,
        description=(
            "Modify the likelihood of specified tokens appearing in the"
            " completion"
        ),
    )
    user: str | None = Field(
        None, description="Unique identifier representing your end-user"
    )


class ChatCompletionChunkChoiceDelta(BaseModel):
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkChoiceDelta


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: list[dict]


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
