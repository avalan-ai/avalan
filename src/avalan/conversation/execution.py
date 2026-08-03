"""Bind completed provider output to one exact checkpoint execution."""

from ..types import JsonValue
from .binding import ProviderLaneBinding
from .contract import (
    AuthorityScope,
    CanonicalRequestDigest,
    CheckpointId,
    CheckpointIdentity,
    ConversationAgentId,
    ProviderLaneId,
    RequestIdempotencyIdentity,
    RequestIdempotencyKey,
    UpstreamResponseId,
)
from .errors import ConversationValidationError
from .items import ProviderItem
from .settings import (
    ConversationMode,
    EffectiveReasoningMetadata,
    ProviderLaneOutputScope,
    ProviderUsage,
)
from .value import (
    IntegrityDigest,
    ProviderCallId,
    ProviderItemId,
    ToolSchemaRevision,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import final


@final
class AgentStructuredInputRequested(RuntimeError):
    """Suspend one native tool call without executing an external effect."""

    arguments: Mapping[str, JsonValue]

    def __init__(self, arguments: Mapping[str, JsonValue]) -> None:
        if not isinstance(arguments, Mapping):
            raise ConversationValidationError()
        frozen = freeze_json_value(arguments)
        if not isinstance(frozen, Mapping):
            raise ConversationValidationError()
        self.arguments = frozen
        super().__init__("agent structured input requested")

    def __repr__(self) -> str:
        """Return a content-free structured-input signal."""
        return "AgentStructuredInputRequested(arguments=<redacted>)"


class ToolEffectPolicy(StrEnum):
    """Identify the recovery rule for one asynchronous tool effect."""

    PURE = "pure"
    IDEMPOTENT = "idempotent"
    FENCED_UNPROTECTED = "fenced_unprotected"


class ToolExecutionPhase(StrEnum):
    """Identify durable progress around one correlated tool effect."""

    REQUESTED = "requested"
    EFFECT_APPLIED = "effect_applied"
    OUTPUT_PERSISTED = "output_persisted"


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ToolEffectReconciliation:
    """Report whether one fenced effect already exists and its output."""

    applied: bool
    output: str | None = None

    def __post_init__(self) -> None:
        if type(self.applied) is not bool or self.applied != (
            self.output is not None
        ):
            raise ConversationValidationError()
        if self.output is not None and (
            type(self.output) is not str
            or len(self.output.encode("utf-8")) > 1_048_576
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return content-free fenced reconciliation metadata."""
        return (
            "ToolEffectReconciliation("
            f"applied={self.applied!r}, output=<redacted>)"
        )


class DurableToolRecoveryAction(StrEnum):
    """Identify the only safe advancement from durable tool segments."""

    REEXECUTE_PURE = "reexecute_pure"
    REEXECUTE_IDEMPOTENT = "reexecute_idempotent"
    REQUIRE_RECONCILIATION = "require_reconciliation"
    RESUME_PROVIDER = "resume_provider"
    COMMIT_OUTWARD = "commit_outward"


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DurableToolRecoveryAdmission:
    """Bind one recovery lease to an immutable encrypted checkpoint suffix."""

    checkpoint_id: CheckpointId
    checkpoint_integrity: IntegrityDigest
    idempotency: RequestIdempotencyIdentity
    binding: ProviderLaneBinding
    action: DurableToolRecoveryAction
    segment_count: int

    def __post_init__(self) -> None:
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        if (
            type(self.checkpoint_integrity) is not str
            or len(self.checkpoint_integrity) != 64
        ):
            raise ConversationValidationError()
        try:
            int(self.checkpoint_integrity, 16)
        except ValueError as exc:
            raise ConversationValidationError() from exc
        if (
            type(self.idempotency) is not RequestIdempotencyIdentity
            or type(self.binding) is not ProviderLaneBinding
            or not isinstance(self.action, DurableToolRecoveryAction)
            or type(self.segment_count) is not int
            or self.segment_count <= 0
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free recovery admission metadata."""
        return (
            "DurableToolRecoveryAdmission("
            f"checkpoint_id={self.checkpoint_id!r}, "
            f"action={self.action.value!r}, "
            f"segment_count={self.segment_count})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DurableToolRecoveryLease:
    """Return one owner-fenced lease for an exact recovery admission."""

    admission: DurableToolRecoveryAdmission
    owner_token: str

    def __post_init__(self) -> None:
        if type(self.admission) is not DurableToolRecoveryAdmission:
            raise ConversationValidationError()
        validate_identifier(self.owner_token, "owner_token")

    def __repr__(self) -> str:
        """Return only content-free recovery lease metadata."""
        return (
            "DurableToolRecoveryLease("
            f"action={self.admission.action.value!r}, owner=<redacted>)"
        )


class ProviderExecutionSegmentPhase(StrEnum):
    """Identify one private recoverable provider/tool segment boundary."""

    PROVIDER_RESPONSE = "provider_response"
    TOOL_OUTPUT = "tool_output"


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderToolExecution:
    """Persist exact tool identity and effect recovery metadata."""

    call_id: ProviderCallId
    arguments: Mapping[str, JsonValue]
    tool_revision: ToolSchemaRevision
    effect_policy: ToolEffectPolicy
    phase: ToolExecutionPhase
    idempotency_key: str | None = None
    output_id: ProviderItemId | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.call_id, "call_id")
        if not isinstance(self.arguments, Mapping):
            raise ConversationValidationError()
        arguments = freeze_json_value(self.arguments)
        if not isinstance(arguments, Mapping):
            raise ConversationValidationError()
        object.__setattr__(self, "arguments", arguments)
        validate_identifier(self.tool_revision, "tool_revision")
        if not isinstance(
            self.effect_policy,
            ToolEffectPolicy,
        ) or not isinstance(self.phase, ToolExecutionPhase):
            raise ConversationValidationError()
        idempotent = self.effect_policy is ToolEffectPolicy.IDEMPOTENT
        if idempotent != (self.idempotency_key is not None):
            raise ConversationValidationError()
        if self.idempotency_key is not None:
            validate_identifier(self.idempotency_key, "tool_idempotency_key")
        persisted = self.phase is ToolExecutionPhase.OUTPUT_PERSISTED
        if persisted != (self.output_id is not None):
            raise ConversationValidationError()
        if self.output_id is not None:
            validate_identifier(self.output_id, "tool_output_id")

    def __repr__(self) -> str:
        """Return content-free tool recovery metadata."""
        return (
            "ProviderToolExecution("
            f"effect_policy={self.effect_policy.value!r}, "
            f"phase={self.phase.value!r}, arguments=<redacted>, "
            "identifiers=<redacted>)"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderExecutionSegment:
    """Persist one exact private segment before or after tool effects."""

    schema_version: int
    idempotency_key: RequestIdempotencyKey
    request_digest: CanonicalRequestDigest
    binding: ProviderLaneBinding
    mode: ConversationMode
    segment_index: int
    phase: ProviderExecutionSegmentPhase
    items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    usage: ProviderUsage
    tools: tuple[ProviderToolExecution, ...] = ()
    upstream_response_id: UpstreamResponseId | None = None
    recovery_request: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConversationValidationError()
        validate_identifier(self.idempotency_key, "idempotency_key")
        validate_identifier(self.request_digest, "request_digest")
        if (
            type(self.binding) is not ProviderLaneBinding
            or not isinstance(self.mode, ConversationMode)
            or self.mode is ConversationMode.OFF
            or type(self.segment_index) is not int
            or self.segment_index < 0
            or not isinstance(self.phase, ProviderExecutionSegmentPhase)
            or type(self.items) is not tuple
            or any(type(item) is not ProviderItem for item in self.items)
            or any(item.lane_id != self.binding.lane_id for item in self.items)
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or type(self.usage) is not ProviderUsage
            or type(self.tools) is not tuple
            or any(
                type(tool) is not ProviderToolExecution for tool in self.tools
            )
        ):
            raise ConversationValidationError()
        if (self.mode is ConversationMode.STORED) != (
            self.upstream_response_id is not None
        ):
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id,
                "upstream_response_id",
            )
        if self.recovery_request is not None:
            recovered = freeze_json_value(self.recovery_request)
            if not isinstance(recovered, Mapping):
                raise ConversationValidationError()
            object.__setattr__(self, "recovery_request", recovered)
        call_ids = tuple(tool.call_id for tool in self.tools)
        if len(call_ids) != len(set(call_ids)):
            raise ConversationValidationError()
        item_call_ids = {
            item.call_id for item in self.items if item.call_id is not None
        }
        if not set(call_ids) <= item_call_ids:
            raise ConversationValidationError()
        expected_phase = (
            ToolExecutionPhase.REQUESTED
            if self.phase is ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            else ToolExecutionPhase.OUTPUT_PERSISTED
        )
        if any(tool.phase is not expected_phase for tool in self.tools):
            raise ConversationValidationError()

    @property
    def lane_id(self) -> ProviderLaneId:
        """Return the exact lane owning this private segment."""
        return self.binding.lane_id

    def __repr__(self) -> str:
        """Return content-free segment recovery metadata."""
        return (
            "ProviderExecutionSegment("
            f"lane_id={self.lane_id!r}, segment_index={self.segment_index}, "
            f"phase={self.phase.value!r}, item_count={len(self.items)}, "
            f"tool_count={len(self.tools)}, provider_state=<redacted>)"
        )


def durable_tool_recovery_action(
    segments: tuple[ProviderExecutionSegment, ...],
) -> DurableToolRecoveryAction:
    """Return the safe next action for one exact durable tool suffix."""
    if (
        type(segments) is not tuple
        or not segments
        or any(
            type(segment) is not ProviderExecutionSegment
            for segment in segments
        )
    ):
        raise ConversationValidationError()
    first = segments[0]
    for index, segment in enumerate(segments):
        if (
            segment.binding != first.binding
            or segment.mode is not first.mode
            or segment.idempotency_key != first.idempotency_key
            or segment.request_digest != first.request_digest
        ):
            raise ConversationValidationError()
        if index == 0:
            if (
                segment.phase
                is not ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            ):
                raise ConversationValidationError()
            continue
        previous = segments[index - 1]
        if segment.phase is ProviderExecutionSegmentPhase.TOOL_OUTPUT:
            if (
                previous.phase
                is not ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
                or segment.segment_index != previous.segment_index
                or not previous.tools
                or len(segment.tools) != len(previous.tools)
            ):
                raise ConversationValidationError()
            requested = {tool.call_id: tool for tool in previous.tools}
            for persisted in segment.tools:
                prior = requested.get(persisted.call_id)
                if (
                    prior is None
                    or persisted.arguments != prior.arguments
                    or persisted.tool_revision != prior.tool_revision
                    or persisted.effect_policy is not prior.effect_policy
                    or persisted.idempotency_key != prior.idempotency_key
                ):
                    raise ConversationValidationError()
        elif (
            previous.phase is not ProviderExecutionSegmentPhase.TOOL_OUTPUT
            or segment.segment_index != previous.segment_index + 1
        ):
            raise ConversationValidationError()
    last = segments[-1]
    if last.phase is ProviderExecutionSegmentPhase.TOOL_OUTPUT:
        return DurableToolRecoveryAction.RESUME_PROVIDER
    if not last.tools:
        if len(segments) < 2:
            raise ConversationValidationError()
        return DurableToolRecoveryAction.COMMIT_OUTWARD
    policies = {tool.effect_policy for tool in last.tools}
    if ToolEffectPolicy.FENCED_UNPROTECTED in policies:
        return DurableToolRecoveryAction.REQUIRE_RECONCILIATION
    if ToolEffectPolicy.IDEMPOTENT in policies:
        return DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT
    return DurableToolRecoveryAction.REEXECUTE_PURE


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionReceipt:
    """Store content-safe proof of one exact completed lane execution."""

    schema_version: int
    digest: IntegrityDigest
    item_count: int
    opaque_byte_count: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConversationValidationError()
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise ConversationValidationError()
        try:
            int(self.digest, 16)
        except ValueError as exc:
            raise ConversationValidationError() from exc
        if (
            type(self.item_count) is not int
            or self.item_count < 0
            or type(self.opaque_byte_count) is not int
            or self.opaque_byte_count < 0
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free receipt metadata."""
        return (
            "ProviderLaneExecutionReceipt("
            f"schema_version={self.schema_version}, "
            f"digest={self.digest!r}, item_count={self.item_count}, "
            f"opaque_byte_count={self.opaque_byte_count})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionReservation:
    """Bind one expected lane to an owned execution attempt."""

    binding: ProviderLaneBinding
    mode: ConversationMode
    scope: ProviderLaneOutputScope

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or not isinstance(self.mode, ConversationMode)
            or self.mode is ConversationMode.OFF
            or not isinstance(self.scope, ProviderLaneOutputScope)
            or (
                self.mode is ConversationMode.STORED
                and self.scope is not ProviderLaneOutputScope.CURRENT_CALL
            )
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free reservation metadata."""
        return (
            "ProviderLaneExecutionReservation("
            f"lane_id={self.binding.lane_id!r}, mode={self.mode.value!r}, "
            f"scope={self.scope.value!r})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConversationExecutionReservation:
    """Bind an idempotency reservation to one exact turn and lane set."""

    idempotency: RequestIdempotencyIdentity
    identity: CheckpointIdentity
    lanes: tuple[ProviderLaneExecutionReservation, ...]
    authorized_agent_ids: tuple[ConversationAgentId, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.idempotency) is not RequestIdempotencyIdentity
            or type(self.identity) is not CheckpointIdentity
            or type(self.lanes) is not tuple
            or not self.lanes
            or any(
                type(lane) is not ProviderLaneExecutionReservation
                for lane in self.lanes
            )
            or type(self.authorized_agent_ids) is not tuple
        ):
            raise ConversationValidationError()
        for agent_id in self.authorized_agent_ids:
            validate_identifier(agent_id, "authorized_agent_id")
        if len(self.authorized_agent_ids) != len(
            set(self.authorized_agent_ids)
        ):
            raise ConversationValidationError()
        authorized = set(
            self.authorized_agent_ids or (self.idempotency.authority.agent_id,)
        )
        lane_ids = tuple(lane.binding.lane_id for lane in self.lanes)
        if len(lane_ids) != len(set(lane_ids)) or any(
            lane.binding.agent_id not in authorized for lane in self.lanes
        ):
            raise ConversationValidationError()
        if self.idempotency.authority.agent_id not in authorized:
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free reservation metadata."""
        return (
            "ConversationExecutionReservation("
            f"checkpoint_id={self.identity.checkpoint_id!r}, "
            f"lane_count={len(self.lanes)})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionStage:
    """Request authoritative staging for one exact completed lane."""

    idempotency: RequestIdempotencyIdentity
    owner_token: str
    identity: CheckpointIdentity
    binding: ProviderLaneBinding
    mode: ConversationMode
    scope: ProviderLaneOutputScope
    completed_items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    usage: ProviderUsage
    execution_receipt: ProviderLaneExecutionReceipt
    upstream_response_id: UpstreamResponseId | None = None

    def __post_init__(self) -> None:
        if (
            type(self.idempotency) is not RequestIdempotencyIdentity
            or type(self.identity) is not CheckpointIdentity
            or type(self.binding) is not ProviderLaneBinding
            or not isinstance(self.mode, ConversationMode)
            or self.mode is ConversationMode.OFF
            or not isinstance(self.scope, ProviderLaneOutputScope)
            or type(self.completed_items) is not tuple
            or any(
                type(item) is not ProviderItem for item in self.completed_items
            )
            or any(
                item.lane_id != self.binding.lane_id
                for item in self.completed_items
            )
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or type(self.usage) is not ProviderUsage
            or type(self.execution_receipt) is not ProviderLaneExecutionReceipt
        ):
            raise ConversationValidationError()
        validate_identifier(self.owner_token, "owner_token")
        if (self.mode is ConversationMode.STORED) != (
            self.upstream_response_id is not None
        ):
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id,
                "upstream_response_id",
            )
        expected = provider_lane_execution_receipt(
            authority=self.idempotency.authority,
            identity=self.identity,
            binding=self.binding,
            mode=self.mode,
            scope=self.scope,
            completed_items=self.completed_items,
            reasoning=self.reasoning,
            usage=self.usage,
            upstream_response_id=self.upstream_response_id,
        )
        if self.execution_receipt != expected:
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free staging metadata."""
        return (
            "ProviderLaneExecutionStage("
            f"checkpoint_id={self.identity.checkpoint_id!r}, "
            f"lane_id={self.binding.lane_id!r}, "
            f"item_count={len(self.completed_items)})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProviderLaneExecutionAttestation:
    """Carry one opaque store-owned staged-execution identity."""

    schema_version: int
    staging_id: str
    lane_id: ProviderLaneId

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConversationValidationError()
        validate_identifier(self.staging_id, "staging_id")
        validate_identifier(self.lane_id, "lane_id")

    def __repr__(self) -> str:
        """Return metadata without the opaque staging identity."""
        return (
            "ProviderLaneExecutionAttestation("
            f"schema_version={self.schema_version}, "
            "staging_id=<redacted>, "
            f"lane_id={self.lane_id!r})"
        )


def provider_lane_execution_receipt(
    *,
    authority: AuthorityScope,
    identity: CheckpointIdentity,
    binding: ProviderLaneBinding,
    mode: ConversationMode,
    scope: ProviderLaneOutputScope,
    completed_items: tuple[ProviderItem, ...],
    reasoning: EffectiveReasoningMetadata,
    usage: ProviderUsage,
    upstream_response_id: UpstreamResponseId | None,
) -> ProviderLaneExecutionReceipt:
    """Return a digest binding exact output to its execution authority."""
    if (
        type(authority) is not AuthorityScope
        or type(identity) is not CheckpointIdentity
        or type(binding) is not ProviderLaneBinding
        or not isinstance(mode, ConversationMode)
        or mode is ConversationMode.OFF
        or not isinstance(scope, ProviderLaneOutputScope)
        or type(completed_items) is not tuple
        or any(type(item) is not ProviderItem for item in completed_items)
        or type(reasoning) is not EffectiveReasoningMetadata
        or type(usage) is not ProviderUsage
    ):
        raise ConversationValidationError()
    if any(item.lane_id != binding.lane_id for item in completed_items):
        raise ConversationValidationError()
    if (mode is ConversationMode.STORED) != (upstream_response_id is not None):
        raise ConversationValidationError()
    payload = freeze_json_value(
        {
            "domain": "avalan.provider-lane-execution",
            "schema_version": 1,
            "authority": {
                "source": authority.source.value,
                "principal_id": authority.principal_id,
                "agent_id": authority.agent_id,
                "endpoint_id": authority.endpoint_id,
                "tenant_id": authority.tenant_id,
                "local_single_user_configured": (
                    authority.local_single_user_configured
                ),
                "network_exposed": authority.network_exposed,
            },
            "identity": {
                "conversation_id": identity.conversation_id,
                "logical_turn_id": identity.logical_turn_id,
                "execution_segment_id": identity.execution_segment_id,
                "checkpoint_id": identity.checkpoint_id,
                "branch_id": identity.branch_id,
                "sequence": identity.sequence,
                "parent_checkpoint_id": identity.parent_checkpoint_id,
                "parent_sequence": identity.parent_sequence,
            },
            "binding_digest": binding.integrity_digest,
            "lane_id": binding.lane_id,
            "mode": mode.value,
            "scope": scope.value,
            "completed_items": tuple(
                _provider_item_value(item) for item in completed_items
            ),
            "reasoning": {
                "requested": reasoning.requested.value,
                "effective": (
                    reasoning.effective.value
                    if reasoning.effective is not None
                    else None
                ),
            },
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "upstream_response_id": upstream_response_id,
        }
    )
    return ProviderLaneExecutionReceipt(
        schema_version=1,
        digest=IntegrityDigest(
            sha256(canonical_json_bytes(payload)).hexdigest()
        ),
        item_count=len(completed_items),
        opaque_byte_count=sum(
            item.opaque_state.byte_count
            for item in completed_items
            if item.opaque_state is not None
        ),
    )


def _provider_item_value(item: ProviderItem) -> JsonValue:
    opaque = item.opaque_state
    return freeze_json_value(
        {
            "item_id": item.item_id,
            "lane_id": item.lane_id,
            "model_call_id": item.model_call_id,
            "kind": item.kind.value,
            "order": item.order,
            "provider_index": item.provider_index,
            "phase": item.phase.value,
            "caller": item.caller.value,
            "canonical_input": item.canonical_input,
            "normalization_version": item.normalization_version,
            "call_id": item.call_id,
            "opaque_state": (
                {
                    "digest": opaque.digest,
                    "byte_count": opaque.byte_count,
                }
                if opaque is not None
                else None
            ),
            "complete": item.complete,
        }
    )
