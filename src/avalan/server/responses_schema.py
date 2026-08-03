"""Define strict OpenAI Responses request item schemas."""

from ..conversation.envelope import ContinuationEnvelopeToken
from ..conversation.errors import ConversationError
from ..conversation.settings import ReasoningContext
from ..entities import MessageRole, ReasoningEffort, ReasoningSummaryMode
from ..types import LooseJsonValue

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    WithJsonSchema,
    model_validator,
)


class _StrictResponsesValue(BaseModel):
    """Forbid unrecognized fields at a state-bearing wire boundary."""

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        hide_input_in_errors=True,
    )


def _continuation_envelope_token(value: object) -> ContinuationEnvelopeToken:
    """Wrap one request token without retaining a printable string field."""
    if type(value) is ContinuationEnvelopeToken:
        return value
    if type(value) is not str:
        raise ValueError("invalid continuation envelope")
    try:
        return ContinuationEnvelopeToken.from_request(
            value,
            max_chars=6_000_000,
        )
    except ConversationError:
        raise ValueError("invalid continuation envelope") from None


ResponsesContinuationEnvelope: TypeAlias = Annotated[
    ContinuationEnvelopeToken,
    PlainValidator(_continuation_envelope_token, json_schema_input_type=str),
    WithJsonSchema(
        {
            "type": "string",
            "minLength": 1,
            "maxLength": 6_000_000,
            "pattern": r"^avl_ce1\.[A-Za-z0-9_-]+$",
        },
        mode="validation",
    ),
]


class ResponsesInputText(_StrictResponsesValue):
    """Carry one caller-visible text input part."""

    type: Literal["input_text"]
    text: str


class ResponsesOutputText(_StrictResponsesValue):
    """Carry one replayable provider-visible text part."""

    type: Literal["output_text"]
    text: str
    annotations: list[dict[str, LooseJsonValue]] = Field(default_factory=list)


class ResponsesRefusal(_StrictResponsesValue):
    """Carry one replayable provider refusal part."""

    type: Literal["refusal"]
    refusal: str


class ResponsesInputImage(_StrictResponsesValue):
    """Carry one image input by URL or uploaded file identifier."""

    type: Literal["input_image"]
    detail: Literal["auto", "low", "high"] = "auto"
    image_url: str | None = None
    file_id: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ResponsesInputImage":
        """Require exactly one non-empty image source."""
        values = (self.image_url, self.file_id)
        if (
            sum(value is not None and bool(value.strip()) for value in values)
            != 1
        ):
            raise ValueError("input_image requires exactly one image source")
        return self


class ResponsesInputFile(_StrictResponsesValue):
    """Carry one file input by data, URL, or uploaded identifier."""

    type: Literal["input_file"]
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ResponsesInputFile":
        """Require exactly one non-empty file source."""
        values = (self.file_data, self.file_id, self.file_url)
        if (
            sum(value is not None and bool(value.strip()) for value in values)
            != 1
        ):
            raise ValueError("input_file requires exactly one file source")
        return self


ResponsesContentPart: TypeAlias = Annotated[
    ResponsesInputText
    | ResponsesOutputText
    | ResponsesRefusal
    | ResponsesInputImage
    | ResponsesInputFile,
    Field(discriminator="type"),
]

ResponsesInputContentPart: TypeAlias = Annotated[
    ResponsesInputText | ResponsesInputImage | ResponsesInputFile,
    Field(discriminator="type"),
]
ResponsesOutputContentPart: TypeAlias = Annotated[
    ResponsesOutputText | ResponsesRefusal,
    Field(discriminator="type"),
]


class ResponsesMessageItem(_StrictResponsesValue):
    """Carry one fully tagged Responses message item."""

    type: Literal["message"]
    role: MessageRole
    content: str | list[ResponsesContentPart]
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    phase: Literal["commentary", "final_answer"] | None = None

    @model_validator(mode="after")
    def validate_message_shape(self) -> "ResponsesMessageItem":
        """Require the exact SDK input or output message shape."""
        if self.role is MessageRole.ASSISTANT:
            if (
                self.id is None
                or self.status is None
                or not isinstance(self.content, list)
                or any(
                    not isinstance(
                        part, ResponsesOutputText | ResponsesRefusal
                    )
                    for part in self.content
                )
            ):
                raise ValueError(
                    "assistant message requires output item state"
                )
            return self
        if (
            self.role
            not in {
                MessageRole.USER,
                MessageRole.SYSTEM,
                MessageRole.DEVELOPER,
            }
            or self.id is not None
            or self.phase is not None
            or not isinstance(self.content, list)
            or any(
                not isinstance(
                    part,
                    ResponsesInputText
                    | ResponsesInputImage
                    | ResponsesInputFile,
                )
                for part in self.content
            )
        ):
            raise ValueError("tagged input message has invalid state")
        return self


class ResponsesFileSearchCallItem(_StrictResponsesValue):
    """Carry one replayable file-search call item."""

    type: Literal["file_search_call"]
    id: str
    queries: list[str]
    status: str
    results: list[dict[str, LooseJsonValue]] | None = None


class ResponsesComputerCallItem(_StrictResponsesValue):
    """Carry one replayable computer call item."""

    type: Literal["computer_call"]
    id: str
    call_id: str
    pending_safety_checks: list[dict[str, LooseJsonValue]]
    status: str
    action: dict[str, LooseJsonValue] | None = None
    actions: list[dict[str, LooseJsonValue]] | None = None


class ResponsesComputerCallOutputItem(_StrictResponsesValue):
    """Carry one computer call result."""

    type: Literal["computer_call_output"]
    call_id: str
    output: LooseJsonValue
    acknowledged_safety_checks: list[dict[str, LooseJsonValue]] | None = None
    id: str | None = None
    status: str | None = None


class ResponsesWebSearchCallItem(_StrictResponsesValue):
    """Carry one replayable web-search call item."""

    type: Literal["web_search_call"]
    id: str
    action: dict[str, LooseJsonValue]
    status: str


class ResponsesFunctionCallItem(_StrictResponsesValue):
    """Carry one replayable function call item."""

    type: Literal["function_call"]
    arguments: str
    call_id: str
    name: str
    id: str | None = None
    namespace: str | None = None
    status: str | None = None


class ResponsesFunctionCallOutputItem(_StrictResponsesValue):
    """Carry one function call result."""

    type: Literal["function_call_output"]
    call_id: str
    output: LooseJsonValue
    id: str | None = None
    status: str | None = None


class ResponsesToolSearchCallItem(_StrictResponsesValue):
    """Carry one replayable tool-search call item."""

    type: Literal["tool_search_call"]
    arguments: LooseJsonValue
    call_id: str | None = None
    execution: Literal["server", "client"] | None = None
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ResponsesToolSearchOutputItem(_StrictResponsesValue):
    """Carry one tool-search result."""

    type: Literal["tool_search_output"]
    call_id: str | None = None
    tools: list[dict[str, LooseJsonValue]]
    execution: Literal["server", "client"] | None = None
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ResponsesAdditionalToolsItem(_StrictResponsesValue):
    """Carry one provider-requested additional-tools item."""

    type: Literal["additional_tools"]
    role: Literal["developer"]
    tools: list[dict[str, LooseJsonValue]]
    id: str | None = None


class ResponsesReasoningItem(_StrictResponsesValue):
    """Carry one opaque replayable reasoning item."""

    type: Literal["reasoning"]
    id: str
    summary: list[dict[str, LooseJsonValue]]
    content: list[dict[str, LooseJsonValue]] | None = None
    encrypted_content: str | None = None
    status: str | None = None


class ResponsesCompactionItem(_StrictResponsesValue):
    """Carry one opaque replayable compaction boundary."""

    type: Literal["compaction"]
    encrypted_content: str
    id: str | None = None


class ResponsesImageGenerationCallItem(_StrictResponsesValue):
    """Carry one replayable image-generation call item."""

    type: Literal["image_generation_call"]
    id: str
    result: str | None
    status: Literal["in_progress", "completed", "generating", "failed"]


class ResponsesCodeInterpreterCallItem(_StrictResponsesValue):
    """Carry one replayable code-interpreter call item."""

    type: Literal["code_interpreter_call"]
    code: str | None
    container_id: str
    id: str
    outputs: list[dict[str, LooseJsonValue]] | None
    status: Literal[
        "in_progress", "completed", "incomplete", "interpreting", "failed"
    ]


class ResponsesLocalShellAction(_StrictResponsesValue):
    """Describe one exact local-shell execution action."""

    type: Literal["exec"]
    command: list[str]
    env: dict[str, str]
    timeout_ms: int | None = None
    user: str | None = None
    working_directory: str | None = None


class ResponsesLocalShellCallItem(_StrictResponsesValue):
    """Carry one replayable local-shell call item."""

    type: Literal["local_shell_call"]
    action: ResponsesLocalShellAction
    call_id: str
    id: str
    status: Literal["in_progress", "completed", "incomplete"]


class ResponsesLocalShellCallOutputItem(_StrictResponsesValue):
    """Carry one local-shell call result."""

    type: Literal["local_shell_call_output"]
    id: str
    output: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ResponsesShellAction(_StrictResponsesValue):
    """Describe one exact shell execution action."""

    commands: list[str]
    max_output_length: int | None = Field(None, ge=1)
    timeout_ms: int | None = None


class ResponsesShellCallItem(_StrictResponsesValue):
    """Carry one replayable shell call item."""

    type: Literal["shell_call"]
    action: ResponsesShellAction
    call_id: str
    environment: dict[str, str] | None = None
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ResponsesShellTimeoutOutcome(_StrictResponsesValue):
    """Record one shell output timeout."""

    type: Literal["timeout"]


class ResponsesShellExitOutcome(_StrictResponsesValue):
    """Record one shell output exit code."""

    type: Literal["exit"]
    exit_code: int


ResponsesShellOutcome: TypeAlias = Annotated[
    ResponsesShellTimeoutOutcome | ResponsesShellExitOutcome,
    Field(discriminator="type"),
]


class ResponsesShellOutputContent(_StrictResponsesValue):
    """Carry one exact shell output chunk."""

    outcome: ResponsesShellOutcome
    stderr: str
    stdout: str


class ResponsesShellCallOutputItem(_StrictResponsesValue):
    """Carry one shell call result."""

    type: Literal["shell_call_output"]
    call_id: str
    output: list[ResponsesShellOutputContent]
    id: str | None = None
    max_output_length: int | None = Field(None, ge=1)
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ResponsesApplyPatchCallItem(_StrictResponsesValue):
    """Carry one replayable patch call item."""

    type: Literal["apply_patch_call"]
    call_id: str
    operation: dict[str, LooseJsonValue]
    status: Literal["in_progress", "completed"]
    id: str | None = None


class ResponsesApplyPatchCallOutputItem(_StrictResponsesValue):
    """Carry one patch call result."""

    type: Literal["apply_patch_call_output"]
    call_id: str
    status: Literal["completed", "failed"]
    id: str | None = None
    output: str | None = None


class ResponsesMCPListToolsItem(_StrictResponsesValue):
    """Carry one replayable MCP tool-list item."""

    type: Literal["mcp_list_tools"]
    id: str
    server_label: str
    tools: list[dict[str, LooseJsonValue]]
    error: str | None = None


class ResponsesMCPApprovalRequestItem(_StrictResponsesValue):
    """Carry one replayable MCP approval request."""

    type: Literal["mcp_approval_request"]
    arguments: str
    id: str
    name: str
    server_label: str


class ResponsesMCPApprovalResponseItem(_StrictResponsesValue):
    """Carry one MCP approval decision."""

    type: Literal["mcp_approval_response"]
    approval_request_id: str
    approve: bool
    id: str | None = None
    reason: str | None = None


class ResponsesMCPCallItem(_StrictResponsesValue):
    """Carry one replayable MCP call item."""

    type: Literal["mcp_call"]
    arguments: str
    id: str
    name: str
    server_label: str
    approval_request_id: str | None = None
    error: str | None = None
    output: str | None = None
    status: (
        Literal["in_progress", "completed", "incomplete", "calling", "failed"]
        | None
    ) = None


class ResponsesCustomToolCallItem(_StrictResponsesValue):
    """Carry one replayable custom-tool call item."""

    type: Literal["custom_tool_call"]
    call_id: str
    input: str
    name: str
    id: str | None = None
    namespace: str | None = None


class ResponsesCustomToolCallOutputItem(_StrictResponsesValue):
    """Carry one custom-tool result."""

    type: Literal["custom_tool_call_output"]
    call_id: str
    output: LooseJsonValue
    id: str | None = None


class ResponsesCompactionTriggerItem(_StrictResponsesValue):
    """Carry one explicit compaction trigger item."""

    type: Literal["compaction_trigger"]


class ResponsesItemReference(_StrictResponsesValue):
    """Carry one authorized item reference."""

    type: Literal["item_reference"] | None = None
    id: str


ResponsesTaggedInputItem: TypeAlias = Annotated[
    ResponsesMessageItem
    | ResponsesFileSearchCallItem
    | ResponsesComputerCallItem
    | ResponsesComputerCallOutputItem
    | ResponsesWebSearchCallItem
    | ResponsesFunctionCallItem
    | ResponsesFunctionCallOutputItem
    | ResponsesToolSearchCallItem
    | ResponsesToolSearchOutputItem
    | ResponsesAdditionalToolsItem
    | ResponsesReasoningItem
    | ResponsesCompactionItem
    | ResponsesImageGenerationCallItem
    | ResponsesCodeInterpreterCallItem
    | ResponsesLocalShellCallItem
    | ResponsesLocalShellCallOutputItem
    | ResponsesShellCallItem
    | ResponsesShellCallOutputItem
    | ResponsesApplyPatchCallItem
    | ResponsesApplyPatchCallOutputItem
    | ResponsesMCPListToolsItem
    | ResponsesMCPApprovalRequestItem
    | ResponsesMCPApprovalResponseItem
    | ResponsesMCPCallItem
    | ResponsesCustomToolCallItem
    | ResponsesCustomToolCallOutputItem
    | ResponsesCompactionTriggerItem,
    Field(discriminator="type"),
]


class ResponsesEasyInputMessage(_StrictResponsesValue):
    """Carry the documented untagged convenience-message shape."""

    role: MessageRole
    content: str | list[ResponsesInputContentPart]
    type: Literal["message"] | None = None
    phase: Literal["commentary", "final_answer"] | None = None

    @model_validator(mode="after")
    def validate_message_shape(self) -> "ResponsesEasyInputMessage":
        """Reject roles and phases absent from the SDK convenience shape."""
        if self.role not in {
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.SYSTEM,
            MessageRole.DEVELOPER,
        } or (
            self.phase is not None and self.role is not MessageRole.ASSISTANT
        ):
            raise ValueError("easy input message has invalid state")
        return self


ResponsesInputItem: TypeAlias = (
    ResponsesTaggedInputItem
    | ResponsesMessageItem
    | ResponsesEasyInputMessage
    | ResponsesItemReference
)


class ResponsesReasoningConfig(_StrictResponsesValue):
    """Configure visible reasoning and durable context retention."""

    effort: ReasoningEffort | None = None
    summary: ReasoningSummaryMode | None = None
    context: ReasoningContext | None = None


class ResponsesCompactionControl(_StrictResponsesValue):
    """Request one bounded inline context compaction policy."""

    type: Literal["compaction"]
    compact_threshold: int = Field(ge=1)


ResponsesContextManagement: TypeAlias = list[ResponsesCompactionControl]


class ResponsesStreamOptions(_StrictResponsesValue):
    """Configure supported Responses streaming projections."""

    include_obfuscation: Literal[False] = False


class ResponsesFunctionTool(_StrictResponsesValue):
    """Describe one server-authorized Responses function tool."""

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: dict[str, LooseJsonValue]
    strict: bool | None = None


class ResponsesTaskInputExtension(_StrictResponsesValue):
    """Negotiate structured-input behavior for one request."""

    version: str = Field(json_schema_extra={"const": "1"})
    handling: Literal["attached", "detached", "unavailable"]


class ResponsesConversationExtension(_StrictResponsesValue):
    """Carry versioned Avalan continuation controls."""

    version: Literal["1"]
    idempotency_key: str | None = Field(None, min_length=1, max_length=256)
    mode: Literal["caller_held"] | None = None
    continuation_envelope: ResponsesContinuationEnvelope | None = None
    operation: Literal["continue", "branch", "named_head"] | None = None
    branch_id: str | None = Field(None, min_length=1, max_length=256)
    head_id: str | None = Field(None, min_length=1, max_length=256)
    expected_head_revision: int | None = Field(None, ge=0)
    lane_id: str | None = Field(None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_continuation(self) -> "ResponsesConversationExtension":
        """Require one closed caller-held continuation operation."""
        state_fields = (
            self.continuation_envelope,
            self.operation,
            self.branch_id,
            self.head_id,
            self.expected_head_revision,
            self.lane_id,
        )
        if self.mode is None:
            if any(value is not None for value in state_fields):
                raise ValueError("caller-held state requires its exact mode")
            return self
        operation = self.operation or "continue"
        if operation == "continue":
            if any(
                value is not None
                for value in (
                    self.branch_id,
                    self.head_id,
                    self.expected_head_revision,
                )
            ):
                raise ValueError("continue cannot carry branch or head state")
            return self
        if operation == "branch":
            if (
                self.continuation_envelope is None
                or self.branch_id is None
                or self.head_id is not None
                or self.expected_head_revision is not None
            ):
                raise ValueError("branch requires an envelope and branch_id")
            return self
        if (
            self.head_id is None
            or self.expected_head_revision is None
            or self.branch_id is not None
            or (
                self.continuation_envelope is None
                and self.expected_head_revision != 0
            )
        ):
            raise ValueError("named_head requires exact head coordinates")
        return self


class ResponsesAvalanExtension(_StrictResponsesValue):
    """Carry versioned extensions in Avalan's explicit namespace."""

    version: Literal["1"]
    conversation: ResponsesConversationExtension | None = None


class ResponsesRequestExtensions(BaseModel):
    """Carry strict owned extensions beside unrelated future namespaces."""

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
        hide_input_in_errors=True,
    )

    task_input: ResponsesTaskInputExtension | None = None
    avalan: ResponsesAvalanExtension | None = None


class ResponsesContainerSelection(_StrictResponsesValue):
    """Select one server-owned remote container policy profile."""

    profile: str


ResponsesInclude: TypeAlias = Literal[
    "reasoning.encrypted_content",
    "message.output_text.logprobs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "web_search_call.action.sources",
]


class ResponsesPublicOutputText(_StrictResponsesValue):
    """Describe one caller-visible text output part."""

    type: Literal["output_text"]
    text: str
    annotations: list[dict[str, LooseJsonValue]] = Field(default_factory=list)


class ResponsesPublicOutputMessage(_StrictResponsesValue):
    """Describe one caller-visible completed message item."""

    id: str
    type: Literal["message"]
    status: Literal["completed"]
    role: MessageRole
    content: list[ResponsesPublicOutputText]


class ResponsesInputTokenDetails(_StrictResponsesValue):
    """Describe cached-token usage for one public response."""

    cached_tokens: int = Field(ge=0)


class ResponsesOutputTokenDetails(_StrictResponsesValue):
    """Describe reasoning-token usage for one public response."""

    reasoning_tokens: int = Field(ge=0)


class ResponsesUsage(_StrictResponsesValue):
    """Describe public token usage without exposing private lane state."""

    input_tokens: int = Field(ge=0)
    input_tokens_details: ResponsesInputTokenDetails
    output_tokens: int = Field(ge=0)
    output_tokens_details: ResponsesOutputTokenDetails
    total_tokens: int = Field(ge=0)
    input_text_tokens: int = Field(ge=0)
    output_text_tokens: int = Field(ge=0)


class ResponsesPublicMetadata(_StrictResponsesValue):
    """Describe the documented safe Avalan lifecycle projection."""

    avalan_lifecycle: Literal["published"]
    avalan_checkpoint_digest: str


class ResponsesContinuationResponseExtension(_StrictResponsesValue):
    """Return terminal-only caller-held continuation state."""

    version: Literal["1"]
    continuation_envelope: str


class ResponsesAvalanResponseExtension(_StrictResponsesValue):
    """Return versioned Avalan response extensions."""

    version: Literal["1"]
    conversation: ResponsesContinuationResponseExtension


class ResponsesResponseExtensions(_StrictResponsesValue):
    """Return the exact supported response extension namespace."""

    avalan: ResponsesAvalanResponseExtension


class ResponsesResource(_StrictResponsesValue):
    """Describe one committed Avalan-served Responses resource."""

    id: str
    object: Literal["response"]
    type: Literal["response"]
    created_at: int
    created: int
    model: str
    status: Literal["completed"]
    parallel_tool_calls: bool
    tool_choice: Literal["auto"]
    tools: list[dict[str, LooseJsonValue]]
    output: list[ResponsesPublicOutputMessage]
    metadata: ResponsesPublicMetadata
    usage: ResponsesUsage
    extensions: ResponsesResponseExtensions | None = None


class ResponsesCompactUsage(_StrictResponsesValue):
    """Describe official compact token accounting."""

    input_tokens: int = Field(ge=0)
    input_tokens_details: ResponsesInputTokenDetails
    output_tokens: int = Field(ge=0)
    output_tokens_details: ResponsesOutputTokenDetails
    total_tokens: int = Field(ge=0)


class ResponsesCompactResource(_StrictResponsesValue):
    """Describe one provider-canonical stateless compact result."""

    id: str
    created_at: int
    object: Literal["response.compaction"]
    output: list[ResponsesTaggedInputItem]
    usage: ResponsesCompactUsage
    extensions: ResponsesResponseExtensions | None = None


class ResponsesDeletionMetadata(_StrictResponsesValue):
    """Describe local deletion and upstream reconciliation state."""

    avalan_local_deletion: Literal["tombstoned", "deleted"]
    avalan_upstream_deletion: Literal[
        "pending_outbox_reconciliation",
        "reconciled",
    ]


class ResponsesDeletedResource(_StrictResponsesValue):
    """Describe an idempotently deleted public Responses resource."""

    id: str
    object: Literal["response.deleted"]
    deleted: Literal[True]
    metadata: ResponsesDeletionMetadata


class ResponsesError(_StrictResponsesValue):
    """Describe one OpenAI-shaped safe server error."""

    message: str
    type: Literal["invalid_request_error", "server_error"]
    code: str
    param: str | None = None


class ResponsesErrorEnvelope(_StrictResponsesValue):
    """Wrap one OpenAI-shaped safe server error."""

    error: ResponsesError
