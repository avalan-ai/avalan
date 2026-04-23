from ..entities import MessageRole, OrchestratorSettings, ReasoningEffort
from ..tool.context import ToolSettingsContext

from dataclasses import dataclass
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

JSONType = Literal["bool", "float", "int", "object", "string"]


def _has_non_empty_file_source(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    if value.startswith("data:"):
        _, separator, payload = value.rpartition(",")
        return bool(separator and payload.strip())
    return True


@dataclass(kw_only=True, frozen=True)
class OrchestratorContext:
    participant_id: UUID
    specs_path: str | None = None
    settings: OrchestratorSettings | None = None
    tool_settings: ToolSettingsContext | None = None


class ResponseFormatText(BaseModel):
    type: Literal["text"]


class ResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"]


class JSONSchemaField(BaseModel):
    title: str
    type: JSONType


class JSONSchema(BaseModel):
    properties: dict[str, JSONSchemaField]
    required: list[str] | None = None
    title: str | None = None
    type: JSONType
    additionalProperties: bool | None = None


class JSONSchemaSettings(BaseModel):
    schema_: JSONSchema = Field(
        ..., validation_alias="schema", serialization_alias="schema"
    )
    name: str | None = None
    strict: bool = True


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: JSONSchemaSettings


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, JSONSchemaField]
    required: list[str] | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: FunctionParameters


class ToolFunction(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ContentText(BaseModel):
    type: Literal["text", "input_text"]
    text: str


class ContentImage(BaseModel):
    type: Literal["image_url"]
    image_url: dict[str, str]


class ContentFile(BaseModel):
    type: Literal["file", "input_file"]
    file: dict[str, Any] | None = None
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ContentFile":
        nested = self.file or {}
        has_source = any(
            _has_non_empty_file_source(value)
            for value in (
                (
                    nested.get("file_data")
                    if "file_data" in nested
                    else nested.get("data")
                ),
                nested.get("file_id"),
                (
                    nested.get("file_url")
                    if "file_url" in nested
                    else nested.get("url")
                ),
                self.file_data,
                self.file_id,
                self.file_url,
            )
        )
        if not has_source:
            raise ValueError(
                "File content requires file_id, file_url, file_data, or file"
            )
        return self


ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema,
    Field(discriminator="type"),
]

Tool = Annotated[ToolFunction, Field(discriminator="type")]

ContentPart = Annotated[
    ContentText | ContentImage | ContentFile, Field(discriminator="type")
]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str | list[ContentPart]


class ReasoningConfig(BaseModel):
    effort: ReasoningEffort | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(
        None,
        description=(
            "ID of the model to use for generating the completion. When"
            " omitted, use the server's configured model."
        ),
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
    logprobs: bool | None = None
    top_logprobs: int | None = Field(None, ge=0, le=5)
    user: str | None = Field(
        None, description="Unique identifier representing your end-user"
    )
    response_format: ResponseFormat | None = Field(
        None, description="Format to use for model response"
    )
    reasoning_effort: ReasoningEffort | None = Field(
        None,
        description="Reasoning effort for supported reasoning models",
    )
    tools: list[Tool] | None = None
    tool_choice: (
        Literal["auto", "none", "required"] | str | dict[str, object] | None
    ) = None


class ResponsesRequest(BaseModel):
    model: str | None = Field(
        None,
        description=(
            "ID of the model to use for generating the response. When"
            " omitted, use the server's configured model."
        ),
    )
    input: list[ChatMessage] = Field(...)
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    response_format: ResponseFormat | None = None
    reasoning: ReasoningConfig | None = None

    @property
    def messages(self) -> list[ChatMessage]:
        return self.input


class MCPToolRequest(BaseModel):
    input_string: str = Field(
        ..., description="Input to pass to the orchestrator via MCP"
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
    permission: list[dict[str, Any]]


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class EngineRequest(BaseModel):
    uri: str | None = None
    database: str | None = None

    @model_validator(mode="after")
    def check_uri_or_database(self) -> "EngineRequest":
        if self.uri is None and self.database is None:
            raise ValueError("Provide uri or database")
        return self
