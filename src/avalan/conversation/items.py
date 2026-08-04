"""Define ordered provider ledgers and separate visible transcripts."""

from ..types import JsonValue
from .contract import ConversationModelCallId, ProviderLaneId
from .errors import ConversationValidationError
from .value import (
    ConversationCodecVersion,
    OpaqueProviderState,
    ProviderCallId,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
    validate_revision,
)

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from json import JSONDecodeError, loads
from types import MappingProxyType
from typing import final


class ProviderItemKind(StrEnum):
    """Identify every supported canonical Responses item kind."""

    MESSAGE = "message"
    FILE_SEARCH_CALL = "file_search_call"
    COMPUTER_CALL = "computer_call"
    COMPUTER_CALL_OUTPUT = "computer_call_output"
    WEB_SEARCH_CALL = "web_search_call"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    TOOL_SEARCH_CALL = "tool_search_call"
    TOOL_SEARCH_OUTPUT = "tool_search_output"
    ADDITIONAL_TOOLS = "additional_tools"
    REASONING = "reasoning"
    COMPACTION = "compaction"
    IMAGE_GENERATION_CALL = "image_generation_call"
    CODE_INTERPRETER_CALL = "code_interpreter_call"
    LOCAL_SHELL_CALL = "local_shell_call"
    LOCAL_SHELL_CALL_OUTPUT = "local_shell_call_output"
    SHELL_CALL = "shell_call"
    SHELL_CALL_OUTPUT = "shell_call_output"
    APPLY_PATCH_CALL = "apply_patch_call"
    APPLY_PATCH_CALL_OUTPUT = "apply_patch_call_output"
    MCP_LIST_TOOLS = "mcp_list_tools"
    MCP_APPROVAL_REQUEST = "mcp_approval_request"
    MCP_APPROVAL_RESPONSE = "mcp_approval_response"
    MCP_CALL = "mcp_call"
    CUSTOM_TOOL_CALL_OUTPUT = "custom_tool_call_output"
    CUSTOM_TOOL_CALL = "custom_tool_call"
    COMPACTION_TRIGGER = "compaction_trigger"
    ITEM_REFERENCE = "item_reference"


class ProviderItemPhase(StrEnum):
    """Identify one canonical item position within provider execution."""

    INPUT = "input"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FINAL = "final"
    COMPACTION = "compaction"


class ProviderItemCaller(StrEnum):
    """Identify who supplied one canonical provider item."""

    CALLER = "caller"
    PROVIDER = "provider"
    TOOL = "tool"
    AVALAN = "avalan"


class VisibleTranscriptRole(StrEnum):
    """Identify a safe visible transcript entry role."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ProviderItemCorrelation(StrEnum):
    """Identify the required call-correlation behavior for an item."""

    NONE = "none"
    CALL = "call"
    OUTPUT = "output"
    TERMINAL_CALL = "terminal_call"


class ProviderItemNormalizationRule(StrEnum):
    """Identify one explicit provider output-to-input conversion rule."""

    INPUT_IDENTITY = "input_identity"
    PROVIDER_OUTPUT_REPLAY = "provider_output_replay"
    TOOL_OUTPUT_REPLAY = "tool_output_replay"


PROVIDER_ITEM_NORMALIZATION_VERSION = ConversationCodecVersion(1)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderItemSemanticRule:
    """Describe one frozen valid item origin and canonical input schema."""

    phases: frozenset[ProviderItemPhase]
    callers: frozenset[ProviderItemCaller]
    correlation: ProviderItemCorrelation
    normalization: ProviderItemNormalizationRule
    required_fields: frozenset[str]
    allowed_fields: frozenset[str]
    correlation_field: str | None = None
    opaque_required: bool = False


def _semantic_rule(
    *,
    phases: tuple[ProviderItemPhase, ...],
    callers: tuple[ProviderItemCaller, ...],
    correlation: ProviderItemCorrelation,
    normalization: ProviderItemNormalizationRule,
    required_fields: tuple[str, ...],
    allowed_fields: tuple[str, ...],
    correlation_field: str | None = None,
    opaque_required: bool = False,
) -> ProviderItemSemanticRule:
    return ProviderItemSemanticRule(
        phases=frozenset(phases),
        callers=frozenset(callers),
        correlation=correlation,
        normalization=normalization,
        required_fields=frozenset(required_fields),
        allowed_fields=frozenset(allowed_fields),
        correlation_field=correlation_field,
        opaque_required=opaque_required,
    )


_PROVIDER = (ProviderItemCaller.PROVIDER,)
_TOOLS = (ProviderItemCaller.TOOL, ProviderItemCaller.AVALAN)
_ASSISTANT = (ProviderItemPhase.ASSISTANT,)
_TOOL = (ProviderItemPhase.TOOL,)
_INPUT = (ProviderItemPhase.INPUT,)
_OUTPUT_REPLAY = ProviderItemNormalizationRule.PROVIDER_OUTPUT_REPLAY
_TOOL_REPLAY = ProviderItemNormalizationRule.TOOL_OUTPUT_REPLAY
_INPUT_IDENTITY = ProviderItemNormalizationRule.INPUT_IDENTITY

PROVIDER_ITEM_SEMANTICS: Mapping[
    ProviderItemKind,
    tuple[ProviderItemSemanticRule, ...],
] = MappingProxyType(
    {
        ProviderItemKind.MESSAGE: (
            _semantic_rule(
                phases=_INPUT,
                callers=(ProviderItemCaller.CALLER,),
                correlation=ProviderItemCorrelation.NONE,
                normalization=_INPUT_IDENTITY,
                required_fields=("content", "role", "type"),
                allowed_fields=("content", "role", "status", "type"),
            ),
            _semantic_rule(
                phases=(
                    ProviderItemPhase.ASSISTANT,
                    ProviderItemPhase.FINAL,
                ),
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("content", "id", "role", "status", "type"),
                allowed_fields=(
                    "content",
                    "id",
                    "phase",
                    "role",
                    "status",
                    "type",
                ),
            ),
        ),
        ProviderItemKind.FILE_SEARCH_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("id", "queries", "status", "type"),
                allowed_fields=("id", "queries", "results", "status", "type"),
            ),
        ),
        ProviderItemKind.COMPUTER_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=(
                    "call_id",
                    "id",
                    "pending_safety_checks",
                    "status",
                    "type",
                ),
                allowed_fields=(
                    "action",
                    "actions",
                    "call_id",
                    "id",
                    "pending_safety_checks",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.COMPUTER_CALL_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("call_id", "output", "type"),
                allowed_fields=(
                    "acknowledged_safety_checks",
                    "call_id",
                    "id",
                    "output",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.WEB_SEARCH_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("action", "id", "status", "type"),
                allowed_fields=("action", "id", "status", "type"),
            ),
        ),
        ProviderItemKind.FUNCTION_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=("arguments", "call_id", "name", "type"),
                allowed_fields=(
                    "arguments",
                    "call_id",
                    "id",
                    "name",
                    "namespace",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.FUNCTION_CALL_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("call_id", "output", "type"),
                allowed_fields=("call_id", "id", "output", "status", "type"),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.TOOL_SEARCH_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=("arguments", "call_id", "type"),
                allowed_fields=(
                    "arguments",
                    "call_id",
                    "execution",
                    "id",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.TOOL_SEARCH_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("call_id", "tools", "type"),
                allowed_fields=(
                    "call_id",
                    "execution",
                    "id",
                    "status",
                    "tools",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.ADDITIONAL_TOOLS: (
            _semantic_rule(
                phases=_INPUT,
                callers=(ProviderItemCaller.CALLER,),
                correlation=ProviderItemCorrelation.NONE,
                normalization=_INPUT_IDENTITY,
                required_fields=("role", "tools", "type"),
                allowed_fields=("id", "role", "tools", "type"),
            ),
        ),
        ProviderItemKind.REASONING: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("id", "summary", "type"),
                allowed_fields=("content", "id", "status", "summary", "type"),
                opaque_required=True,
            ),
        ),
        ProviderItemKind.COMPACTION: (
            _semantic_rule(
                phases=(ProviderItemPhase.COMPACTION,),
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("type",),
                allowed_fields=("created_by", "id", "type"),
                opaque_required=True,
            ),
        ),
        ProviderItemKind.IMAGE_GENERATION_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("id", "result", "status", "type"),
                allowed_fields=("id", "result", "status", "type"),
            ),
        ),
        ProviderItemKind.CODE_INTERPRETER_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=(
                    "code",
                    "container_id",
                    "id",
                    "outputs",
                    "status",
                    "type",
                ),
                allowed_fields=(
                    "code",
                    "container_id",
                    "id",
                    "outputs",
                    "status",
                    "type",
                ),
            ),
        ),
        ProviderItemKind.LOCAL_SHELL_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=("action", "call_id", "id", "status", "type"),
                allowed_fields=("action", "call_id", "id", "status", "type"),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("id", "output", "type"),
                allowed_fields=("id", "output", "status", "type"),
                correlation_field="id",
            ),
        ),
        ProviderItemKind.SHELL_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=("action", "call_id", "type"),
                allowed_fields=(
                    "action",
                    "call_id",
                    "environment",
                    "id",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.SHELL_CALL_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("call_id", "output", "type"),
                allowed_fields=(
                    "call_id",
                    "id",
                    "max_output_length",
                    "output",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.APPLY_PATCH_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=("call_id", "operation", "status", "type"),
                allowed_fields=(
                    "call_id",
                    "id",
                    "operation",
                    "status",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.APPLY_PATCH_CALL_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("call_id", "status", "type"),
                allowed_fields=("call_id", "id", "output", "status", "type"),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.MCP_LIST_TOOLS: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.NONE,
                normalization=_OUTPUT_REPLAY,
                required_fields=("id", "server_label", "tools", "type"),
                allowed_fields=(
                    "error",
                    "id",
                    "server_label",
                    "tools",
                    "type",
                ),
            ),
        ),
        ProviderItemKind.MCP_APPROVAL_REQUEST: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=(
                    "arguments",
                    "id",
                    "name",
                    "server_label",
                    "type",
                ),
                allowed_fields=(
                    "arguments",
                    "id",
                    "name",
                    "server_label",
                    "type",
                ),
                correlation_field="id",
            ),
        ),
        ProviderItemKind.MCP_APPROVAL_RESPONSE: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("approval_request_id", "approve", "type"),
                allowed_fields=(
                    "approval_request_id",
                    "approve",
                    "id",
                    "reason",
                    "type",
                ),
                correlation_field="approval_request_id",
            ),
        ),
        ProviderItemKind.MCP_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.TERMINAL_CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=(
                    "arguments",
                    "id",
                    "name",
                    "server_label",
                    "type",
                ),
                allowed_fields=(
                    "approval_request_id",
                    "arguments",
                    "error",
                    "id",
                    "name",
                    "output",
                    "server_label",
                    "status",
                    "type",
                ),
                correlation_field="id",
            ),
        ),
        ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT: (
            _semantic_rule(
                phases=_TOOL,
                callers=_TOOLS,
                correlation=ProviderItemCorrelation.OUTPUT,
                normalization=_TOOL_REPLAY,
                required_fields=("call_id", "output", "type"),
                allowed_fields=("call_id", "id", "output", "type"),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.CUSTOM_TOOL_CALL: (
            _semantic_rule(
                phases=_ASSISTANT,
                callers=_PROVIDER,
                correlation=ProviderItemCorrelation.CALL,
                normalization=_OUTPUT_REPLAY,
                required_fields=("call_id", "input", "name", "type"),
                allowed_fields=(
                    "call_id",
                    "id",
                    "input",
                    "name",
                    "namespace",
                    "type",
                ),
                correlation_field="call_id",
            ),
        ),
        ProviderItemKind.COMPACTION_TRIGGER: (
            _semantic_rule(
                phases=_INPUT,
                callers=(ProviderItemCaller.CALLER,),
                correlation=ProviderItemCorrelation.NONE,
                normalization=_INPUT_IDENTITY,
                required_fields=("type",),
                allowed_fields=("type",),
            ),
        ),
        ProviderItemKind.ITEM_REFERENCE: (
            _semantic_rule(
                phases=_INPUT,
                callers=(ProviderItemCaller.CALLER,),
                correlation=ProviderItemCorrelation.NONE,
                normalization=_INPUT_IDENTITY,
                required_fields=("id", "type"),
                allowed_fields=("id", "type"),
            ),
        ),
    }
)


_CALL_KINDS = frozenset(
    {
        ProviderItemKind.COMPUTER_CALL,
        ProviderItemKind.FUNCTION_CALL,
        ProviderItemKind.TOOL_SEARCH_CALL,
        ProviderItemKind.LOCAL_SHELL_CALL,
        ProviderItemKind.SHELL_CALL,
        ProviderItemKind.APPLY_PATCH_CALL,
        ProviderItemKind.MCP_APPROVAL_REQUEST,
        ProviderItemKind.MCP_CALL,
        ProviderItemKind.CUSTOM_TOOL_CALL,
    }
)
_OUTPUT_KINDS = frozenset(
    {
        ProviderItemKind.COMPUTER_CALL_OUTPUT,
        ProviderItemKind.FUNCTION_CALL_OUTPUT,
        ProviderItemKind.TOOL_SEARCH_OUTPUT,
        ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT,
        ProviderItemKind.SHELL_CALL_OUTPUT,
        ProviderItemKind.APPLY_PATCH_CALL_OUTPUT,
        ProviderItemKind.MCP_APPROVAL_RESPONSE,
        ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT,
    }
)
_TERMINAL_CALL_KINDS = frozenset({ProviderItemKind.MCP_CALL})
_EXPECTED_OUTPUT_KINDS = MappingProxyType(
    {
        ProviderItemKind.COMPUTER_CALL: ProviderItemKind.COMPUTER_CALL_OUTPUT,
        ProviderItemKind.FUNCTION_CALL: ProviderItemKind.FUNCTION_CALL_OUTPUT,
        ProviderItemKind.TOOL_SEARCH_CALL: ProviderItemKind.TOOL_SEARCH_OUTPUT,
        ProviderItemKind.LOCAL_SHELL_CALL: (
            ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT
        ),
        ProviderItemKind.SHELL_CALL: ProviderItemKind.SHELL_CALL_OUTPUT,
        ProviderItemKind.APPLY_PATCH_CALL: (
            ProviderItemKind.APPLY_PATCH_CALL_OUTPUT
        ),
        ProviderItemKind.MCP_APPROVAL_REQUEST: (
            ProviderItemKind.MCP_APPROVAL_RESPONSE
        ),
        ProviderItemKind.CUSTOM_TOOL_CALL: (
            ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT
        ),
    }
)


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderItem:
    """Store one complete canonical item at an exact provider position."""

    item_id: ProviderItemId
    lane_id: ProviderLaneId
    model_call_id: ConversationModelCallId
    kind: ProviderItemKind
    order: ProviderItemOrder
    provider_index: ProviderItemIndex
    phase: ProviderItemPhase
    caller: ProviderItemCaller
    canonical_input: Mapping[str, JsonValue]
    normalization_version: ConversationCodecVersion
    call_id: ProviderCallId | None = None
    opaque_state: OpaqueProviderState | None = None
    complete: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.item_id, "item_id"),
            (self.lane_id, "lane_id"),
            (self.model_call_id, "model_call_id"),
        ):
            validate_identifier(value, name)
        if not isinstance(self.kind, ProviderItemKind):
            raise ConversationValidationError()
        validate_revision(self.order, "order")
        validate_revision(self.provider_index, "provider_index")
        if not isinstance(self.phase, ProviderItemPhase) or not isinstance(
            self.caller,
            ProviderItemCaller,
        ):
            raise ConversationValidationError()
        validate_revision(self.normalization_version, "normalization_version")
        if (
            self.normalization_version != PROVIDER_ITEM_NORMALIZATION_VERSION
            or self.complete is not True
        ):
            raise ConversationValidationError()
        rules = tuple(
            rule
            for rule in PROVIDER_ITEM_SEMANTICS[self.kind]
            if self.phase in rule.phases and self.caller in rule.callers
        )
        if len(rules) != 1:
            raise ConversationValidationError()
        rule = rules[0]
        frozen_input = freeze_json_value(self.canonical_input)
        if not isinstance(frozen_input, Mapping):
            raise ConversationValidationError()
        _validate_canonical_input(self, frozen_input, rule)
        object.__setattr__(self, "canonical_input", frozen_input)

    @property
    def normalization_rule(self) -> ProviderItemNormalizationRule:
        """Return the exact output-to-input normalization rule in force."""
        for rule in PROVIDER_ITEM_SEMANTICS[self.kind]:
            if self.phase in rule.phases and self.caller in rule.callers:
                return rule.normalization
        raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return content-free provider-item accounting metadata."""
        opaque_byte_count = (
            self.opaque_state.byte_count if self.opaque_state else 0
        )
        return (
            "ProviderItem("
            f"kind={self.kind.value!r}, order={self.order}, "
            f"provider_index={self.provider_index}, "
            f"phase={self.phase.value!r}, "
            f"caller={self.caller.value!r}, "
            f"opaque_byte_count={opaque_byte_count}, "
            "identifiers=<redacted>, canonical_input=<redacted>)"
        )


def provider_item_byte_count(item: ProviderItem) -> int:
    """Return canonical and opaque bytes retained for one provider item."""
    if type(item) is not ProviderItem:
        raise ConversationValidationError()
    return len(canonical_json_bytes(item.canonical_input)) + (
        item.opaque_state.byte_count if item.opaque_state else 0
    )


def _validate_canonical_input(
    item: ProviderItem,
    canonical_input: Mapping[str, JsonValue],
    rule: ProviderItemSemanticRule,
) -> None:
    fields = frozenset(canonical_input)
    if not rule.required_fields <= fields or not fields <= rule.allowed_fields:
        raise ConversationValidationError()
    if canonical_input["type"] != item.kind.value:
        raise ConversationValidationError()

    correlated = rule.correlation is not ProviderItemCorrelation.NONE
    if correlated != (item.call_id is not None):
        raise ConversationValidationError()
    if item.call_id is not None:
        validate_identifier(item.call_id, "call_id")
    if correlated:
        assert rule.correlation_field is not None
        correlated_id = canonical_input[rule.correlation_field]
        validate_identifier(correlated_id, rule.correlation_field)
        if correlated_id != item.call_id:
            raise ConversationValidationError()
    elif rule.correlation_field is not None:
        raise ConversationValidationError()

    has_opaque = item.opaque_state is not None
    if has_opaque != rule.opaque_required:
        raise ConversationValidationError()
    if has_opaque and type(item.opaque_state) is not OpaqueProviderState:
        raise ConversationValidationError()

    canonical_id = canonical_input.get("id")
    if canonical_id is not None:
        validate_identifier(canonical_id, "canonical item id")
        if rule.correlation_field != "id" and canonical_id != item.item_id:
            raise ConversationValidationError()

    status = canonical_input.get("status")
    if status is not None and (
        type(status) is not str or status not in {"completed", "failed"}
    ):
        raise ConversationValidationError()

    if item.kind is ProviderItemKind.MESSAGE:
        role = canonical_input["role"]
        if item.caller is ProviderItemCaller.CALLER:
            if type(role) is not str or role not in {
                "developer",
                "system",
                "user",
            }:
                raise ConversationValidationError()
        elif role != "assistant":
            raise ConversationValidationError()
        canonical_phase = canonical_input.get("phase")
        expected_phase = (
            "final_answer"
            if item.phase is ProviderItemPhase.FINAL
            else "commentary"
        )
        if canonical_phase is not None and canonical_phase != expected_phase:
            raise ConversationValidationError()
    elif item.kind is ProviderItemKind.ADDITIONAL_TOOLS and (
        canonical_input["role"] != "developer"
    ):
        raise ConversationValidationError()
    elif (
        item.kind is ProviderItemKind.MCP_APPROVAL_RESPONSE
        and type(canonical_input["approve"]) is not bool
    ):
        raise ConversationValidationError()
    _validate_canonical_values(item, canonical_input)


def _validate_canonical_values(
    item: ProviderItem,
    canonical_input: Mapping[str, JsonValue],
) -> None:
    """Validate the closed value schema for one canonical item kind."""
    match item.kind:
        case ProviderItemKind.MESSAGE:
            _validate_message_content(item, canonical_input["content"])
        case ProviderItemKind.FILE_SEARCH_CALL:
            _validate_string_sequence(
                canonical_input["queries"], nonempty=True
            )
            if "results" in canonical_input:
                _validate_file_search_results(canonical_input["results"])
        case ProviderItemKind.COMPUTER_CALL:
            _validate_safety_checks(canonical_input["pending_safety_checks"])
            has_action = "action" in canonical_input
            has_actions = "actions" in canonical_input
            if has_action == has_actions:
                raise ConversationValidationError()
            if has_action:
                _validate_computer_action(canonical_input["action"])
            else:
                actions = _canonical_sequence(canonical_input["actions"])
                if not actions:
                    raise ConversationValidationError()
                for action in actions:
                    _validate_computer_action(action)
        case ProviderItemKind.COMPUTER_CALL_OUTPUT:
            _validate_computer_screenshot(canonical_input["output"])
            if "acknowledged_safety_checks" in canonical_input:
                _validate_safety_checks(
                    canonical_input["acknowledged_safety_checks"]
                )
        case ProviderItemKind.WEB_SEARCH_CALL:
            _validate_web_search_action(canonical_input["action"])
        case ProviderItemKind.FUNCTION_CALL:
            _validate_arguments(canonical_input["arguments"])
            _validate_named_fields(canonical_input, "name", "namespace")
        case ProviderItemKind.FUNCTION_CALL_OUTPUT:
            _validate_tool_output_content(canonical_input["output"])
        case ProviderItemKind.TOOL_SEARCH_CALL:
            _validate_arguments(canonical_input["arguments"])
            _validate_execution(canonical_input)
        case ProviderItemKind.TOOL_SEARCH_OUTPUT:
            _validate_tool_definitions(canonical_input["tools"])
            _validate_execution(canonical_input)
        case ProviderItemKind.ADDITIONAL_TOOLS:
            _validate_tool_definitions(canonical_input["tools"])
        case ProviderItemKind.REASONING:
            _validate_reasoning_parts(
                canonical_input["summary"],
                expected_type="summary_text",
            )
            if "content" in canonical_input:
                _validate_reasoning_parts(
                    canonical_input["content"],
                    expected_type="reasoning_text",
                )
        case ProviderItemKind.COMPACTION:
            if "created_by" in canonical_input:
                _canonical_identifier(canonical_input["created_by"])
        case ProviderItemKind.IMAGE_GENERATION_CALL:
            _canonical_text(canonical_input["result"])
        case ProviderItemKind.CODE_INTERPRETER_CALL:
            _canonical_text(canonical_input["code"], allow_empty=True)
            _canonical_identifier(canonical_input["container_id"])
            _validate_code_outputs(canonical_input["outputs"])
        case ProviderItemKind.LOCAL_SHELL_CALL:
            _validate_local_shell_action(canonical_input["action"])
        case ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT:
            _validate_canonical_json_string(canonical_input["output"])
        case ProviderItemKind.SHELL_CALL:
            _validate_shell_action(canonical_input["action"])
            if "environment" in canonical_input:
                _validate_shell_environment(canonical_input["environment"])
        case ProviderItemKind.SHELL_CALL_OUTPUT:
            _validate_shell_outputs(canonical_input["output"])
            if "max_output_length" in canonical_input:
                _canonical_positive_integer(
                    canonical_input["max_output_length"]
                )
        case ProviderItemKind.APPLY_PATCH_CALL:
            _validate_patch_operation(canonical_input["operation"])
        case ProviderItemKind.APPLY_PATCH_CALL_OUTPUT:
            if "output" in canonical_input:
                _canonical_text(canonical_input["output"], allow_empty=True)
        case ProviderItemKind.MCP_LIST_TOOLS:
            _canonical_identifier(canonical_input["server_label"])
            _validate_mcp_tools(canonical_input["tools"])
            if "error" in canonical_input:
                _canonical_text(canonical_input["error"])
        case ProviderItemKind.MCP_APPROVAL_REQUEST:
            _validate_arguments(canonical_input["arguments"])
            _validate_named_fields(canonical_input, "name", "server_label")
        case ProviderItemKind.MCP_APPROVAL_RESPONSE:
            if "reason" in canonical_input:
                _canonical_text(canonical_input["reason"])
        case ProviderItemKind.MCP_CALL:
            _validate_arguments(canonical_input["arguments"])
            _validate_named_fields(canonical_input, "name", "server_label")
            if "approval_request_id" in canonical_input:
                _canonical_identifier(canonical_input["approval_request_id"])
            _validate_mcp_call_result(canonical_input)
        case ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT:
            _validate_tool_output_content(canonical_input["output"])
        case ProviderItemKind.CUSTOM_TOOL_CALL:
            _canonical_text(canonical_input["input"], allow_empty=True)
            _validate_named_fields(canonical_input, "name", "namespace")
        case (
            ProviderItemKind.COMPACTION_TRIGGER
            | ProviderItemKind.ITEM_REFERENCE
        ):
            pass


def _canonical_mapping(
    value: JsonValue,
    *,
    required: frozenset[str],
    allowed: frozenset[str] | None = None,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ConversationValidationError()
    fields = frozenset(value)
    accepted = required if allowed is None else allowed
    if not required <= fields or not fields <= accepted:
        raise ConversationValidationError()
    return value


def _canonical_sequence(value: JsonValue) -> tuple[JsonValue, ...]:
    if type(value) is not tuple:
        raise ConversationValidationError()
    return value


def _canonical_text(value: JsonValue, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise ConversationValidationError()
    if value != value.strip() or "\x00" in value:
        raise ConversationValidationError()
    return value


def _canonical_identifier(value: JsonValue) -> str:
    text = _canonical_text(value)
    validate_identifier(text, "canonical identifier")
    return text


def _canonical_integer(value: JsonValue, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConversationValidationError()
    return value


def _canonical_positive_integer(value: JsonValue) -> int:
    return _canonical_integer(value, minimum=1)


def _canonical_number(value: JsonValue) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConversationValidationError()
    return value


def _validate_string_sequence(
    value: JsonValue,
    *,
    nonempty: bool = False,
) -> None:
    sequence = _canonical_sequence(value)
    if nonempty and not sequence:
        raise ConversationValidationError()
    for member in sequence:
        _canonical_text(member)


def _validate_named_fields(
    value: Mapping[str, JsonValue],
    *fields: str,
) -> None:
    for field in fields:
        if field in value:
            _canonical_identifier(value[field])


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConversationValidationError()
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    del value
    raise ConversationValidationError()


def _validate_canonical_json_string(
    value: JsonValue,
    *,
    object_only: bool = False,
) -> None:
    text = _canonical_text(value, allow_empty=False)
    try:
        decoded: object = loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except JSONDecodeError as exc:
        raise ConversationValidationError() from exc
    frozen = freeze_json_value(decoded)
    if object_only and not isinstance(frozen, Mapping):
        raise ConversationValidationError()
    if canonical_json_bytes(frozen).decode("utf-8") != text:
        raise ConversationValidationError()


def _validate_arguments(value: JsonValue) -> None:
    _validate_canonical_json_string(value, object_only=True)


def _validate_message_content(
    item: ProviderItem,
    value: JsonValue,
) -> None:
    content = _canonical_sequence(value)
    if not content:
        raise ConversationValidationError()
    for part in content:
        if item.caller is ProviderItemCaller.CALLER:
            _validate_input_content_part(part)
        else:
            _validate_output_content_part(part)


def _validate_input_content_part(value: JsonValue) -> None:
    part = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset(
            {
                "detail",
                "file_data",
                "file_id",
                "file_url",
                "filename",
                "image_url",
                "text",
                "type",
            }
        ),
    )
    part_type = part["type"]
    if part_type == "input_text":
        exact = _canonical_mapping(
            value,
            required=frozenset({"text", "type"}),
        )
        _canonical_text(exact["text"], allow_empty=True)
    elif part_type == "input_image":
        image = _canonical_mapping(
            value,
            required=frozenset({"detail", "type"}),
            allowed=frozenset({"detail", "file_id", "image_url", "type"}),
        )
        if image["detail"] not in {"auto", "high", "low", "original"}:
            raise ConversationValidationError()
        sources = tuple(
            field for field in ("file_id", "image_url") if field in image
        )
        if len(sources) != 1:
            raise ConversationValidationError()
        _canonical_identifier(image[sources[0]])
    elif part_type == "input_file":
        file_part = _canonical_mapping(
            value,
            required=frozenset({"type"}),
            allowed=frozenset(
                {
                    "detail",
                    "file_data",
                    "file_id",
                    "file_url",
                    "filename",
                    "type",
                }
            ),
        )
        sources = tuple(
            field
            for field in ("file_data", "file_id", "file_url")
            if field in file_part
        )
        if len(sources) != 1:
            raise ConversationValidationError()
        _canonical_text(file_part[sources[0]])
        if "filename" in file_part:
            _canonical_text(file_part["filename"])
        if "detail" in file_part and file_part["detail"] not in {
            "high",
            "low",
        }:
            raise ConversationValidationError()
    else:
        raise ConversationValidationError()


def _validate_output_content_part(value: JsonValue) -> None:
    part = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset(
            {
                "annotations",
                "logprobs",
                "refusal",
                "text",
                "type",
            }
        ),
    )
    if part["type"] == "output_text":
        text_part = _canonical_mapping(
            value,
            required=frozenset({"text", "type"}),
            allowed=frozenset({"annotations", "logprobs", "text", "type"}),
        )
        _canonical_text(text_part["text"], allow_empty=True)
        if "annotations" in text_part:
            _validate_annotations(text_part["annotations"])
        if "logprobs" in text_part:
            _validate_logprobs(text_part["logprobs"])
    elif part["type"] == "refusal":
        refusal = _canonical_mapping(
            value,
            required=frozenset({"refusal", "type"}),
        )
        _canonical_text(refusal["refusal"])
    else:
        raise ConversationValidationError()


def _validate_annotations(value: JsonValue) -> None:
    annotations = _canonical_sequence(value)
    schemas = {
        "file_citation": frozenset({"file_id", "filename", "index", "type"}),
        "url_citation": frozenset(
            {
                "end_index",
                "start_index",
                "title",
                "type",
                "url",
            }
        ),
        "container_file_citation": frozenset(
            {
                "container_id",
                "end_index",
                "file_id",
                "filename",
                "start_index",
                "type",
            }
        ),
        "file_path": frozenset({"file_id", "index", "type"}),
    }
    for raw in annotations:
        annotation = _canonical_mapping(
            raw,
            required=frozenset({"type"}),
            allowed=frozenset().union(*schemas.values()),
        )
        annotation_type = annotation["type"]
        if type(annotation_type) is not str or annotation_type not in schemas:
            raise ConversationValidationError()
        exact = _canonical_mapping(raw, required=schemas[annotation_type])
        for field in (
            "container_id",
            "file_id",
            "filename",
            "title",
            "url",
        ):
            if field in exact:
                _canonical_text(exact[field])
        for field in ("end_index", "index", "start_index"):
            if field in exact:
                _canonical_integer(exact[field])
        if (
            "end_index" in exact
            and "start_index" in exact
            and _canonical_integer(exact["end_index"])
            < _canonical_integer(exact["start_index"])
        ):
            raise ConversationValidationError()


def _validate_logprobs(value: JsonValue) -> None:
    for raw in _canonical_sequence(value):
        entry = _canonical_mapping(
            raw,
            required=frozenset(
                {
                    "bytes",
                    "logprob",
                    "token",
                    "top_logprobs",
                }
            ),
        )
        _canonical_text(entry["token"], allow_empty=True)
        _validate_bytes(entry["bytes"])
        _canonical_number(entry["logprob"])
        for raw_top in _canonical_sequence(entry["top_logprobs"]):
            top = _canonical_mapping(
                raw_top,
                required=frozenset({"bytes", "logprob", "token"}),
            )
            _canonical_text(top["token"], allow_empty=True)
            _validate_bytes(top["bytes"])
            _canonical_number(top["logprob"])


def _validate_bytes(value: JsonValue) -> None:
    for byte in _canonical_sequence(value):
        if _canonical_integer(byte) > 255:
            raise ConversationValidationError()


def _validate_file_search_results(value: JsonValue) -> None:
    for raw in _canonical_sequence(value):
        result = _canonical_mapping(
            raw,
            required=frozenset({"file_id", "filename", "score", "text"}),
            allowed=frozenset(
                {
                    "attributes",
                    "file_id",
                    "filename",
                    "score",
                    "text",
                }
            ),
        )
        _canonical_identifier(result["file_id"])
        _canonical_text(result["filename"])
        _canonical_text(result["text"], allow_empty=True)
        score = _canonical_number(result["score"])
        if score < 0 or score > 1:
            raise ConversationValidationError()
        if "attributes" in result:
            attributes_value = result["attributes"]
            if not isinstance(attributes_value, Mapping):
                raise ConversationValidationError()
            attributes = attributes_value
            if len(attributes) > 16:
                raise ConversationValidationError()
            for key, member in attributes.items():
                _canonical_text(key)
                if type(member) not in {bool, float, int, str}:
                    raise ConversationValidationError()


def _validate_safety_checks(value: JsonValue) -> None:
    for raw in _canonical_sequence(value):
        check = _canonical_mapping(
            raw,
            required=frozenset({"id"}),
            allowed=frozenset({"code", "id", "message"}),
        )
        _canonical_identifier(check["id"])
        for field in ("code", "message"):
            if field in check:
                _canonical_text(check[field])


def _validate_computer_action(value: JsonValue) -> None:
    action = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset(
            {
                "button",
                "keys",
                "path",
                "scroll_x",
                "scroll_y",
                "text",
                "type",
                "x",
                "y",
            }
        ),
    )
    action_type = action["type"]
    schemas = {
        "click": (
            frozenset({"button", "type", "x", "y"}),
            frozenset({"button", "keys", "type", "x", "y"}),
        ),
        "double_click": (
            frozenset({"type", "x", "y"}),
            frozenset({"keys", "type", "x", "y"}),
        ),
        "drag": (
            frozenset({"path", "type"}),
            frozenset({"keys", "path", "type"}),
        ),
        "keypress": (
            frozenset({"keys", "type"}),
            frozenset({"keys", "type"}),
        ),
        "move": (
            frozenset({"type", "x", "y"}),
            frozenset({"keys", "type", "x", "y"}),
        ),
        "screenshot": (
            frozenset({"type"}),
            frozenset({"type"}),
        ),
        "scroll": (
            frozenset({"scroll_x", "scroll_y", "type", "x", "y"}),
            frozenset({"keys", "scroll_x", "scroll_y", "type", "x", "y"}),
        ),
        "type": (
            frozenset({"text", "type"}),
            frozenset({"text", "type"}),
        ),
        "wait": (
            frozenset({"type"}),
            frozenset({"type"}),
        ),
    }
    if type(action_type) is not str or action_type not in schemas:
        raise ConversationValidationError()
    required, allowed = schemas[action_type]
    exact = _canonical_mapping(value, required=required, allowed=allowed)
    for field in ("scroll_x", "scroll_y", "x", "y"):
        if field in exact:
            _canonical_integer(exact[field])
    if "button" in exact and exact["button"] not in {
        "back",
        "forward",
        "left",
        "right",
        "wheel",
    }:
        raise ConversationValidationError()
    if "keys" in exact:
        _validate_string_sequence(exact["keys"], nonempty=True)
    if "text" in exact:
        _canonical_text(exact["text"], allow_empty=True)
    if "path" in exact:
        path = _canonical_sequence(exact["path"])
        if len(path) < 2:
            raise ConversationValidationError()
        for raw_point in path:
            point = _canonical_mapping(
                raw_point,
                required=frozenset({"x", "y"}),
            )
            _canonical_integer(point["x"])
            _canonical_integer(point["y"])


def _validate_computer_screenshot(value: JsonValue) -> None:
    screenshot = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset({"file_id", "image_url", "type"}),
    )
    if screenshot["type"] != "computer_screenshot":
        raise ConversationValidationError()
    sources = tuple(
        field for field in ("file_id", "image_url") if field in screenshot
    )
    if len(sources) != 1:
        raise ConversationValidationError()
    _canonical_text(screenshot[sources[0]])


def _validate_web_search_action(value: JsonValue) -> None:
    action = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset(
            {
                "pattern",
                "queries",
                "query",
                "sources",
                "type",
                "url",
            }
        ),
    )
    action_type = action["type"]
    if action_type == "search":
        search = _canonical_mapping(
            value,
            required=frozenset({"type"}),
            allowed=frozenset({"queries", "query", "sources", "type"}),
        )
        query_fields = tuple(
            field for field in ("queries", "query") if field in search
        )
        if len(query_fields) != 1:
            raise ConversationValidationError()
        if query_fields[0] == "query":
            _canonical_text(search["query"])
        else:
            _validate_string_sequence(search["queries"], nonempty=True)
        if "sources" in search:
            for raw_source in _canonical_sequence(search["sources"]):
                source = _canonical_mapping(
                    raw_source,
                    required=frozenset({"type", "url"}),
                )
                if source["type"] != "url":
                    raise ConversationValidationError()
                _canonical_text(source["url"])
    elif action_type == "open_page":
        open_page = _canonical_mapping(
            value,
            required=frozenset({"type", "url"}),
        )
        _canonical_text(open_page["url"])
    elif action_type == "find_in_page":
        find = _canonical_mapping(
            value,
            required=frozenset({"pattern", "type", "url"}),
        )
        _canonical_text(find["pattern"])
        _canonical_text(find["url"])
    else:
        raise ConversationValidationError()


def _validate_tool_output_content(value: JsonValue) -> None:
    if type(value) is str:
        _canonical_text(value, allow_empty=True)
        return
    content = _canonical_sequence(value)
    if not content:
        raise ConversationValidationError()
    for part in content:
        _validate_input_content_part(part)


def _validate_execution(value: Mapping[str, JsonValue]) -> None:
    if "execution" in value and value["execution"] not in {
        "client",
        "server",
    }:
        raise ConversationValidationError()


def _validate_tool_definitions(value: JsonValue) -> None:
    for raw in _canonical_sequence(value):
        tool = _canonical_mapping(
            raw,
            required=frozenset({"name", "parameters", "strict", "type"}),
            allowed=frozenset(
                {
                    "description",
                    "name",
                    "parameters",
                    "strict",
                    "type",
                }
            ),
        )
        if tool["type"] != "function" or type(tool["strict"]) is not bool:
            raise ConversationValidationError()
        _canonical_identifier(tool["name"])
        if "description" in tool:
            _canonical_text(tool["description"], allow_empty=True)
        _validate_json_schema(tool["parameters"])


def _validate_json_schema(value: JsonValue) -> None:
    schema = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset(
            {
                "additionalProperties",
                "description",
                "enum",
                "items",
                "properties",
                "required",
                "type",
            }
        ),
    )
    schema_type = schema["type"]
    if schema_type not in {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }:
        raise ConversationValidationError()
    if "description" in schema:
        _canonical_text(schema["description"], allow_empty=True)
    if "enum" in schema:
        enum = _canonical_sequence(schema["enum"])
        if not enum or any(
            isinstance(member, (Mapping, tuple)) for member in enum
        ):
            raise ConversationValidationError()
    if schema_type == "object":
        if "properties" not in schema:
            raise ConversationValidationError()
        properties = schema["properties"]
        if not isinstance(properties, Mapping):
            raise ConversationValidationError()
        for name, member in properties.items():
            _canonical_identifier(name)
            _validate_json_schema(member)
        required = schema.get("required", ())
        _validate_string_sequence(required)
        required_names = _canonical_sequence(required)
        if len(required_names) != len(set(required_names)) or not set(
            required_names
        ) <= set(properties):
            raise ConversationValidationError()
        additional = schema.get("additionalProperties", False)
        if type(additional) is not bool:
            raise ConversationValidationError()
        if "items" in schema:
            raise ConversationValidationError()
    elif schema_type == "array":
        if "items" not in schema:
            raise ConversationValidationError()
        _validate_json_schema(schema["items"])
        if any(
            field in schema
            for field in ("additionalProperties", "properties", "required")
        ):
            raise ConversationValidationError()
    elif any(
        field in schema
        for field in (
            "additionalProperties",
            "items",
            "properties",
            "required",
        )
    ):
        raise ConversationValidationError()


def _validate_reasoning_parts(
    value: JsonValue,
    *,
    expected_type: str,
) -> None:
    for raw in _canonical_sequence(value):
        part = _canonical_mapping(
            raw,
            required=frozenset({"text", "type"}),
        )
        if part["type"] != expected_type:
            raise ConversationValidationError()
        _canonical_text(part["text"], allow_empty=True)


def _validate_code_outputs(value: JsonValue) -> None:
    for raw in _canonical_sequence(value):
        output = _canonical_mapping(
            raw,
            required=frozenset({"type"}),
            allowed=frozenset({"logs", "type", "url"}),
        )
        if output["type"] == "logs":
            logs = _canonical_mapping(
                raw,
                required=frozenset({"logs", "type"}),
            )
            _canonical_text(logs["logs"], allow_empty=True)
        elif output["type"] == "image":
            image = _canonical_mapping(
                raw,
                required=frozenset({"type", "url"}),
            )
            _canonical_text(image["url"])
        else:
            raise ConversationValidationError()


def _validate_local_shell_action(value: JsonValue) -> None:
    action = _canonical_mapping(
        value,
        required=frozenset({"command", "env", "type"}),
        allowed=frozenset(
            {
                "command",
                "env",
                "timeout_ms",
                "type",
                "user",
                "working_directory",
            }
        ),
    )
    if action["type"] != "exec":
        raise ConversationValidationError()
    _validate_string_sequence(action["command"], nonempty=True)
    environment = action["env"]
    if not isinstance(environment, Mapping):
        raise ConversationValidationError()
    for name, member in environment.items():
        _canonical_identifier(name)
        _canonical_text(member, allow_empty=True)
    if "timeout_ms" in action:
        _canonical_positive_integer(action["timeout_ms"])
    for field in ("user", "working_directory"):
        if field in action:
            _canonical_text(action[field])


def _validate_shell_action(value: JsonValue) -> None:
    action = _canonical_mapping(
        value,
        required=frozenset({"commands"}),
        allowed=frozenset({"commands", "max_output_length", "timeout_ms"}),
    )
    _validate_string_sequence(action["commands"], nonempty=True)
    for field in ("max_output_length", "timeout_ms"):
        if field in action:
            _canonical_positive_integer(action[field])


def _validate_shell_environment(value: JsonValue) -> None:
    environment = _canonical_mapping(
        value,
        required=frozenset({"type"}),
        allowed=frozenset({"container_id", "type"}),
    )
    if environment["type"] == "local":
        _canonical_mapping(value, required=frozenset({"type"}))
    elif environment["type"] == "container_reference":
        exact = _canonical_mapping(
            value,
            required=frozenset({"container_id", "type"}),
        )
        _canonical_identifier(exact["container_id"])
    else:
        raise ConversationValidationError()


def _validate_shell_outputs(value: JsonValue) -> None:
    outputs = _canonical_sequence(value)
    if not outputs:
        raise ConversationValidationError()
    for raw in outputs:
        output = _canonical_mapping(
            raw,
            required=frozenset({"outcome", "stderr", "stdout"}),
        )
        _canonical_text(output["stderr"], allow_empty=True)
        _canonical_text(output["stdout"], allow_empty=True)
        outcome = _canonical_mapping(
            output["outcome"],
            required=frozenset({"type"}),
            allowed=frozenset({"exit_code", "type"}),
        )
        if outcome["type"] == "timeout":
            _canonical_mapping(
                output["outcome"],
                required=frozenset({"type"}),
            )
        elif outcome["type"] == "exit":
            exit_outcome = _canonical_mapping(
                output["outcome"],
                required=frozenset({"exit_code", "type"}),
            )
            _canonical_integer(exit_outcome["exit_code"], minimum=-255)
        else:
            raise ConversationValidationError()


def _validate_patch_operation(value: JsonValue) -> None:
    operation = _canonical_mapping(
        value,
        required=frozenset({"path", "type"}),
        allowed=frozenset({"diff", "path", "type"}),
    )
    operation_type = operation["type"]
    if operation_type == "delete_file":
        exact = _canonical_mapping(
            value,
            required=frozenset({"path", "type"}),
        )
    elif operation_type in {"create_file", "update_file"}:
        exact = _canonical_mapping(
            value,
            required=frozenset({"diff", "path", "type"}),
        )
        _canonical_text(exact["diff"], allow_empty=True)
    else:
        raise ConversationValidationError()
    _canonical_text(exact["path"])


def _validate_mcp_tools(value: JsonValue) -> None:
    for raw in _canonical_sequence(value):
        tool = _canonical_mapping(
            raw,
            required=frozenset({"input_schema", "name"}),
            allowed=frozenset(
                {
                    "annotations",
                    "description",
                    "input_schema",
                    "name",
                }
            ),
        )
        _canonical_identifier(tool["name"])
        _validate_json_schema(tool["input_schema"])
        if "description" in tool:
            _canonical_text(tool["description"], allow_empty=True)
        if "annotations" in tool:
            annotations = _canonical_mapping(
                tool["annotations"],
                required=frozenset(),
                allowed=frozenset(
                    {
                        "destructiveHint",
                        "idempotentHint",
                        "openWorldHint",
                        "readOnlyHint",
                        "title",
                    }
                ),
            )
            for name, member in annotations.items():
                if name == "title":
                    _canonical_text(member)
                elif type(member) is not bool:
                    raise ConversationValidationError()


def _validate_mcp_call_result(
    value: Mapping[str, JsonValue],
) -> None:
    has_error = "error" in value
    has_output = "output" in value
    if has_error and has_output:
        raise ConversationValidationError()
    if has_error:
        _canonical_text(value["error"])
    if has_output:
        _canonical_text(value["output"], allow_empty=True)
    status = value.get("status")
    if status == "completed" and not has_output:
        raise ConversationValidationError()
    if status == "failed" and not has_error:
        raise ConversationValidationError()


def validate_provider_item_sequence(
    *,
    lane_id: ProviderLaneId,
    normalization_version: ConversationCodecVersion,
    items: tuple[ProviderItem, ...],
    permitted_open_call_ids: frozenset[ProviderCallId] = frozenset(),
) -> None:
    """Validate exact provider order and explicitly permitted open calls."""
    validate_identifier(lane_id, "lane_id")
    validate_revision(normalization_version, "normalization_version")
    if (
        normalization_version != PROVIDER_ITEM_NORMALIZATION_VERSION
        or type(items) is not tuple
        or type(permitted_open_call_ids) is not frozenset
    ):
        raise ConversationValidationError()
    for call_id in permitted_open_call_ids:
        validate_identifier(call_id, "permitted_open_call_id")
    item_ids: set[ProviderItemId] = set()
    registered_calls: set[ProviderCallId] = set()
    open_calls: dict[
        ProviderCallId,
        tuple[ProviderItemKind, ConversationModelCallId],
    ] = {}
    model_call_indexes: dict[ConversationModelCallId, int] = {}
    for index, item in enumerate(items):
        if type(item) is not ProviderItem:
            raise ConversationValidationError()
        if (
            item.lane_id != lane_id
            or item.normalization_version != normalization_version
            or item.order != index
            or item.item_id in item_ids
        ):
            raise ConversationValidationError()
        expected_provider_index = model_call_indexes.get(
            item.model_call_id,
            0,
        )
        if item.provider_index != expected_provider_index:
            raise ConversationValidationError()
        model_call_indexes[item.model_call_id] = expected_provider_index + 1
        item_ids.add(item.item_id)
        if item.kind in _CALL_KINDS:
            assert item.call_id is not None
            if item.call_id in registered_calls:
                raise ConversationValidationError()
            registered_calls.add(item.call_id)
            if item.kind not in _TERMINAL_CALL_KINDS:
                open_calls[item.call_id] = (
                    _EXPECTED_OUTPUT_KINDS[item.kind],
                    item.model_call_id,
                )
        elif item.kind in _OUTPUT_KINDS:
            assert item.call_id is not None
            expected = open_calls.get(item.call_id)
            if expected != (item.kind, item.model_call_id):
                raise ConversationValidationError()
            del open_calls[item.call_id]
    if frozenset(open_calls) != permitted_open_call_ids:
        raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderItemLedger:
    """Preserve exact item order, adjacency, and call correlation."""

    lane_id: ProviderLaneId
    normalization_version: ConversationCodecVersion
    items: tuple[ProviderItem, ...]

    def __post_init__(self) -> None:
        validate_provider_item_sequence(
            lane_id=self.lane_id,
            normalization_version=self.normalization_version,
            items=self.items,
        )

    @property
    def item_count(self) -> int:
        """Return the number of complete canonical items."""
        return len(self.items)


def provider_replay_items(
    ledger: ProviderItemLedger,
) -> tuple[ProviderItem, ...]:
    """Return the provider-designated compact context for exact replay."""
    if type(ledger) is not ProviderItemLedger:
        raise ConversationValidationError()
    latest_index: int | None = None
    for index, item in enumerate(ledger.items):
        if item.kind is ProviderItemKind.COMPACTION:
            latest_index = index
    if latest_index is None:
        return ledger.items
    prefix = ledger.items[:latest_index]
    canonical_standalone_prefix = bool(prefix) and all(
        item.kind is ProviderItemKind.MESSAGE
        and item.phase is ProviderItemPhase.INPUT
        and item.caller is ProviderItemCaller.CALLER
        for item in prefix
    )
    replay = (
        ledger.items
        if canonical_standalone_prefix
        else ledger.items[latest_index:]
    )
    boundary_offset = latest_index if canonical_standalone_prefix else 0
    assert replay[boundary_offset].kind is ProviderItemKind.COMPACTION
    open_calls: dict[ProviderCallId, ProviderItemKind] = {}
    for item in replay[boundary_offset + 1 :]:
        if item.kind in _CALL_KINDS:
            assert item.call_id is not None
            if item.call_id in open_calls:
                raise ConversationValidationError()
            if item.kind not in _TERMINAL_CALL_KINDS:
                open_calls[item.call_id] = _EXPECTED_OUTPUT_KINDS[item.kind]
        elif item.kind in _OUTPUT_KINDS:
            assert item.call_id is not None
            if open_calls.get(item.call_id) is not item.kind:
                raise ConversationValidationError()
            del open_calls[item.call_id]
    if open_calls:
        raise ConversationValidationError()
    return replay


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionBoundary:
    """Point to the latest exact compaction item and retained suffix."""

    boundary_item_id: ProviderItemId
    boundary_order: ProviderItemOrder
    retained_suffix: tuple[ProviderItemId, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.boundary_item_id, "boundary_item_id")
        validate_revision(self.boundary_order, "boundary_order")
        if type(self.retained_suffix) is not tuple:
            raise ConversationValidationError()
        for item_id in self.retained_suffix:
            validate_identifier(item_id, "retained_suffix_item_id")
        if len(self.retained_suffix) != len(set(self.retained_suffix)):
            raise ConversationValidationError()

    def validate_latest(self, ledger: ProviderItemLedger) -> None:
        """Reject a missing, stale, or overlapping compaction boundary."""
        if type(ledger) is not ProviderItemLedger:
            raise ConversationValidationError()
        compactions = tuple(
            item
            for item in ledger.items
            if item.kind is ProviderItemKind.COMPACTION
        )
        if not compactions:
            raise ConversationValidationError()
        latest = compactions[-1]
        if (
            latest.item_id != self.boundary_item_id
            or latest.order != self.boundary_order
        ):
            raise ConversationValidationError()
        expected_suffix = tuple(
            item.item_id
            for item in ledger.items
            if item.order > self.boundary_order
        )
        if self.retained_suffix != expected_suffix:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VisibleTranscriptEntry:
    """Store one displayable transcript entry without provider state."""

    role: VisibleTranscriptRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, VisibleTranscriptRole):
            raise ConversationValidationError()
        validate_identifier(
            self.content, "visible transcript content", max_length=1_048_576
        )


def public_provider_item_projection(
    items: tuple[ProviderItem, ...],
) -> tuple[VisibleTranscriptEntry, ...]:
    """Return display-only assistant text without provider replay state."""
    if type(items) is not tuple or any(
        type(item) is not ProviderItem for item in items
    ):
        raise ConversationValidationError()
    entries: list[VisibleTranscriptEntry] = []
    for item in items:
        if (
            item.kind is not ProviderItemKind.MESSAGE
            or item.phase
            not in {
                ProviderItemPhase.ASSISTANT,
                ProviderItemPhase.FINAL,
            }
            or item.caller is not ProviderItemCaller.PROVIDER
        ):
            continue
        content = item.canonical_input.get("content")
        if type(content) is not tuple:
            raise ConversationValidationError()
        pieces: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise ConversationValidationError()
            if part.get("type") != "output_text":
                continue
            text = part.get("text")
            if type(text) is not str:
                raise ConversationValidationError()
            pieces.append(text)
        text = "".join(pieces)
        if text:
            entries.append(
                VisibleTranscriptEntry(
                    role=VisibleTranscriptRole.ASSISTANT,
                    content=text,
                )
            )
    return tuple(entries)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VisibleTranscript:
    """Keep visible text structurally separate from provider item ledgers."""

    entries: tuple[VisibleTranscriptEntry, ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(
            type(item) is not VisibleTranscriptEntry for item in self.entries
        ):
            raise ConversationValidationError()

    @property
    def entry_count(self) -> int:
        """Return the number of displayable transcript entries."""
        return len(self.entries)
