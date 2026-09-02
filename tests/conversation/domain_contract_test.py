"""Exercise the canonical typed conversation domain end to end."""

from base64 import b64encode
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from inspect import iscoroutinefunction
from itertools import product
from json import dumps, loads
from operator import setitem
from pathlib import Path
from typing import cast

import pytest

from avalan import conversation as conversation
from avalan.conversation import codec as codec_module
from avalan.conversation import items as items_module
from avalan.interaction.entities import (
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    StateRevision,
)
from avalan.types import JsonValue

_ROOT = Path(__file__).resolve().parents[2]
_CREATED = datetime(2030, 1, 1, tzinfo=UTC)
_PAIRED_KINDS = (
    (
        conversation.ProviderItemKind.COMPUTER_CALL,
        conversation.ProviderItemKind.COMPUTER_CALL_OUTPUT,
    ),
    (
        conversation.ProviderItemKind.FUNCTION_CALL,
        conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
    ),
    (
        conversation.ProviderItemKind.TOOL_SEARCH_CALL,
        conversation.ProviderItemKind.TOOL_SEARCH_OUTPUT,
    ),
    (
        conversation.ProviderItemKind.LOCAL_SHELL_CALL,
        conversation.ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT,
    ),
    (
        conversation.ProviderItemKind.SHELL_CALL,
        conversation.ProviderItemKind.SHELL_CALL_OUTPUT,
    ),
    (
        conversation.ProviderItemKind.APPLY_PATCH_CALL,
        conversation.ProviderItemKind.APPLY_PATCH_CALL_OUTPUT,
    ),
    (
        conversation.ProviderItemKind.MCP_APPROVAL_REQUEST,
        conversation.ProviderItemKind.MCP_APPROVAL_RESPONSE,
    ),
    (
        conversation.ProviderItemKind.CUSTOM_TOOL_CALL,
        conversation.ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT,
    ),
)
_CALL_KINDS = frozenset(call for call, _output in _PAIRED_KINDS) | {
    conversation.ProviderItemKind.MCP_CALL
}
_OUTPUT_KINDS = frozenset(output for _call, output in _PAIRED_KINDS)


def _authority() -> conversation.AuthorityScope:
    return conversation.AuthorityScope(
        source=conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=conversation.AuthorityTenantId("tenant-1"),
        principal_id=conversation.AuthorityPrincipalId("principal-1"),
        agent_id=conversation.ConversationAgentId("agent-1"),
        endpoint_id=conversation.AuthorityEndpointId("endpoint-1"),
    )


def _binding(
    lane_id: str = "lane-1",
    *,
    endpoint: str = "https://api.example.test/v1",
) -> conversation.ProviderLaneBinding:
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type="tests.SyntheticConversationProvider",
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint=endpoint,
        model_or_deployment="synthetic-model",
        provider_api_revision=conversation.ProviderApiRevision("api-v1"),
        sdk_revision=conversation.ProviderSdkRevision("sdk-v1"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("model-config-v1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-v1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-v1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-v1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=conversation.ConversationAgentId("agent-1"),
    )


def _transition_parent(
    mode: conversation.ConversationMode,
) -> conversation.ConversationParent | None:
    if mode is conversation.ConversationMode.OFF:
        return None
    if mode is conversation.ConversationMode.STATELESS:
        return conversation.StatelessParent(
            handle=conversation.StatelessConversationHandle(
                conversation_id=conversation.ConversationId("conversation-1"),
                checkpoint_id=conversation.CheckpointId("checkpoint-1"),
                branch_id=conversation.ConversationBranchId("branch-1"),
            )
        )
    return conversation.StoredParent(
        handle=conversation.StoredConversationHandle(
            conversation_id=conversation.ConversationId("conversation-1"),
            checkpoint_id=conversation.CheckpointId("checkpoint-1"),
            branch_id=conversation.ConversationBranchId("branch-1"),
            public_response_id=conversation.PublicResponseId("response-1"),
        )
    )


def _item(
    kind: conversation.ProviderItemKind,
    index: int,
    *,
    lane_id: str = "lane-1",
    call_id: str | None = None,
    phase: conversation.ProviderItemPhase | None = None,
    caller: conversation.ProviderItemCaller | None = None,
    model_call_id: str = "model-call-1",
    provider_index: int | None = None,
) -> conversation.ProviderItem:
    semantic_rules = conversation.PROVIDER_ITEM_SEMANTICS[kind]
    rules = (
        (semantic_rules[0],)
        if phase is None and caller is None
        else tuple(
            rule
            for rule in semantic_rules
            if (phase is None or phase in rule.phases)
            and (caller is None or caller in rule.callers)
        )
    )
    assert len(rules) == 1
    rule = rules[0]
    if phase is None:
        phase = sorted(rule.phases, key=lambda value: value.value)[0]
    if caller is None:
        caller = (
            conversation.ProviderItemCaller.TOOL
            if conversation.ProviderItemCaller.TOOL in rule.callers
            else sorted(rule.callers, key=lambda value: value.value)[0]
        )
    if rule.correlation is conversation.ProviderItemCorrelation.NONE:
        call_id = None
    elif call_id is None:
        call_id = f"call-{index}"
    item_id = f"item-{index}"
    canonical_input = {
        field: _canonical_item_field(
            field,
            kind=kind,
            phase=phase,
            caller=caller,
            item_id=item_id,
            call_id=call_id,
            correlation_field=rule.correlation_field,
        )
        for field in rule.required_fields
    }
    if kind is conversation.ProviderItemKind.COMPUTER_CALL:
        canonical_input["action"] = {"type": "screenshot"}
    if kind is conversation.ProviderItemKind.MCP_CALL:
        canonical_input["status"] = "completed"
        canonical_input["output"] = "safe-output"
    opaque = (
        conversation.OpaqueProviderState(
            _value=f"opaque-{kind.value}-{index}".encode()
        )
        if rule.opaque_required
        else None
    )
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId(item_id),
        lane_id=conversation.ProviderLaneId(lane_id),
        model_call_id=conversation.ConversationModelCallId(model_call_id),
        kind=kind,
        order=conversation.ProviderItemOrder(index),
        provider_index=conversation.ProviderItemIndex(
            index if provider_index is None else provider_index
        ),
        phase=phase,
        caller=caller,
        canonical_input=canonical_input,
        normalization_version=conversation.PROVIDER_ITEM_NORMALIZATION_VERSION,
        call_id=(conversation.ProviderCallId(call_id) if call_id else None),
        opaque_state=opaque,
    )


def _canonical_item_field(
    field: str,
    *,
    kind: conversation.ProviderItemKind,
    phase: conversation.ProviderItemPhase,
    caller: conversation.ProviderItemCaller,
    item_id: str,
    call_id: str | None,
    correlation_field: str | None,
) -> JsonValue:
    if field == "type":
        return kind.value
    if field == "id":
        return call_id if correlation_field == "id" else item_id
    if field in {"call_id", "approval_request_id"}:
        return call_id
    if field == "role":
        if kind is conversation.ProviderItemKind.ADDITIONAL_TOOLS:
            return "developer"
        return (
            "assistant"
            if caller is conversation.ProviderItemCaller.PROVIDER
            else "user"
        )
    if field == "status":
        return "completed"
    if field == "approve":
        return True
    if field == "arguments":
        return "{}"
    if field == "queries":
        return ("safe-query",)
    if field == "summary":
        return ({"text": "safe-summary", "type": "summary_text"},)
    if field == "pending_safety_checks":
        return ({"id": "safety-check-1"},)
    if field == "outputs":
        return ({"logs": "safe-logs", "type": "logs"},)
    if field == "tools":
        if kind is conversation.ProviderItemKind.MCP_LIST_TOOLS:
            return (
                {
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {},
                        "required": (),
                        "type": "object",
                    },
                    "name": "safe-tool",
                },
            )
        return (
            {
                "name": "safe-tool",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {},
                    "required": (),
                    "type": "object",
                },
                "strict": True,
                "type": "function",
            },
        )
    if field == "action":
        if kind is conversation.ProviderItemKind.WEB_SEARCH_CALL:
            return {"query": "safe-query", "type": "search"}
        if kind is conversation.ProviderItemKind.LOCAL_SHELL_CALL:
            return {
                "command": ("safe-command",),
                "env": {},
                "type": "exec",
            }
        if kind is conversation.ProviderItemKind.SHELL_CALL:
            return {"commands": ("safe-command",)}
        return {"type": "screenshot"}
    if field == "content":
        if caller is conversation.ProviderItemCaller.PROVIDER:
            return (
                {
                    "annotations": (),
                    "text": f"safe-{phase.value}-content",
                    "type": "output_text",
                },
            )
        return (
            {
                "text": f"safe-{phase.value}-content",
                "type": "input_text",
            },
        )
    if field == "output":
        if kind is conversation.ProviderItemKind.COMPUTER_CALL_OUTPUT:
            return {
                "image_url": "https://example.test/screenshot.png",
                "type": "computer_screenshot",
            }
        if kind is conversation.ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT:
            return "{}"
        if kind is conversation.ProviderItemKind.SHELL_CALL_OUTPUT:
            return (
                {
                    "outcome": {"exit_code": 0, "type": "exit"},
                    "stderr": "",
                    "stdout": "safe-output",
                },
            )
        return "safe-output"
    if field == "operation":
        return {
            "diff": "safe-diff",
            "path": "safe-path",
            "type": "update_file",
        }
    return f"safe-{field}"


def _malformed_canonical_input(
    item: conversation.ProviderItem,
) -> Mapping[str, JsonValue]:
    value = dict(item.canonical_input)
    match item.kind:
        case conversation.ProviderItemKind.MESSAGE:
            value["content"] = ()
        case conversation.ProviderItemKind.FILE_SEARCH_CALL:
            value["queries"] = (1,)
        case conversation.ProviderItemKind.COMPUTER_CALL:
            value["action"] = {"type": "click", "x": 1, "y": 2}
        case conversation.ProviderItemKind.COMPUTER_CALL_OUTPUT:
            value["output"] = {"type": "computer_screenshot"}
        case conversation.ProviderItemKind.WEB_SEARCH_CALL:
            value["action"] = {
                "queries": ("query",),
                "query": "query",
                "type": "search",
            }
        case conversation.ProviderItemKind.FUNCTION_CALL:
            value["arguments"] = "{ }"
        case conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT:
            value["output"] = {"text": "not-a-content-sequence"}
        case conversation.ProviderItemKind.TOOL_SEARCH_CALL:
            value["arguments"] = "[]"
        case conversation.ProviderItemKind.TOOL_SEARCH_OUTPUT:
            value["tools"] = (
                {
                    "name": "tool",
                    "parameters": {},
                    "strict": True,
                    "type": "function",
                },
            )
        case conversation.ProviderItemKind.ADDITIONAL_TOOLS:
            value["tools"] = (
                {
                    "name": "tool",
                    "parameters": {
                        "properties": {},
                        "type": "object",
                    },
                    "strict": 1,
                    "type": "function",
                },
            )
        case conversation.ProviderItemKind.REASONING:
            value["summary"] = (
                {
                    "extra": True,
                    "text": "summary",
                    "type": "summary_text",
                },
            )
        case conversation.ProviderItemKind.COMPACTION:
            value["type"] = "reasoning"
        case conversation.ProviderItemKind.IMAGE_GENERATION_CALL:
            value["result"] = 1
        case conversation.ProviderItemKind.CODE_INTERPRETER_CALL:
            value["outputs"] = ({"type": "image"},)
        case conversation.ProviderItemKind.LOCAL_SHELL_CALL:
            value["action"] = {
                "command": ("command",),
                "env": (),
                "type": "exec",
            }
        case conversation.ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT:
            value["output"] = "{ }"
        case conversation.ProviderItemKind.SHELL_CALL:
            value["action"] = {"commands": (1,)}
        case conversation.ProviderItemKind.SHELL_CALL_OUTPUT:
            value["output"] = (
                {
                    "outcome": {"type": "exit"},
                    "stderr": "",
                    "stdout": "",
                },
            )
        case conversation.ProviderItemKind.APPLY_PATCH_CALL:
            value["operation"] = {
                "path": "file.py",
                "type": "create_file",
            }
        case conversation.ProviderItemKind.APPLY_PATCH_CALL_OUTPUT:
            value["output"] = {"not": "text"}
        case conversation.ProviderItemKind.MCP_LIST_TOOLS:
            value["tools"] = (
                {"input_schema": "not-a-schema", "name": "tool"},
            )
        case conversation.ProviderItemKind.MCP_APPROVAL_REQUEST:
            value["arguments"] = '{"key":1,"key":2}'
        case conversation.ProviderItemKind.MCP_APPROVAL_RESPONSE:
            value["approve"] = 1
        case conversation.ProviderItemKind.MCP_CALL:
            value["error"] = "error"
        case conversation.ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT:
            value["output"] = ()
        case conversation.ProviderItemKind.CUSTOM_TOOL_CALL:
            value["input"] = {"not": "text"}
        case conversation.ProviderItemKind.COMPACTION_TRIGGER:
            value["type"] = "compaction"
        case conversation.ProviderItemKind.ITEM_REFERENCE:
            value["id"] = ""
    return value


def _replace_item_input(
    item: conversation.ProviderItem,
    updates: Mapping[str, JsonValue],
    *,
    remove: tuple[str, ...] = (),
) -> conversation.ProviderItem:
    value = dict(item.canonical_input)
    for field in remove:
        value.pop(field, None)
    value.update(updates)
    return replace(item, canonical_input=value)


def _full_ledger(lane_id: str = "lane-1") -> conversation.ProviderItemLedger:
    ordered: list[tuple[conversation.ProviderItemKind, str | None]] = [
        (conversation.ProviderItemKind.REASONING, None)
    ]
    used = {conversation.ProviderItemKind.REASONING}
    for pair_index, (call_kind, output_kind) in enumerate(_PAIRED_KINDS):
        call_id = f"paired-call-{pair_index}"
        ordered.extend(((call_kind, call_id), (output_kind, call_id)))
        used.update((call_kind, output_kind))
    ordered.append((conversation.ProviderItemKind.MCP_CALL, "mcp-call"))
    used.add(conversation.ProviderItemKind.MCP_CALL)
    for kind in conversation.ProviderItemKind:
        if (
            kind not in used
            and kind is not conversation.ProviderItemKind.COMPACTION
        ):
            ordered.append((kind, None))
            used.add(kind)
    ordered.append((conversation.ProviderItemKind.COMPACTION, None))
    items = tuple(
        _item(kind, index, lane_id=lane_id, call_id=call_id)
        for index, (kind, call_id) in enumerate(ordered)
    )
    assert {item.kind for item in items} == set(conversation.ProviderItemKind)
    return conversation.ProviderItemLedger(
        lane_id=conversation.ProviderLaneId(lane_id),
        normalization_version=conversation.ConversationCodecVersion(1),
        items=items,
    )


def _reasoning() -> conversation.EffectiveReasoningMetadata:
    return conversation.EffectiveReasoningMetadata(
        requested=conversation.ReasoningContext.AUTO,
        effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
    )


def _stateless_lane(
    lane_id: str = "lane-1",
) -> conversation.StatelessProviderLaneSnapshot:
    ledger = _full_ledger(lane_id)
    latest = ledger.items[-1]
    return conversation.StatelessProviderLaneSnapshot(
        binding=_binding(lane_id),
        ledger=ledger,
        reasoning=_reasoning(),
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        compaction_boundary=conversation.CompactionBoundary(
            boundary_item_id=latest.item_id,
            boundary_order=latest.order,
            retained_suffix=(),
        ),
    )


def _stored_lane() -> conversation.StoredProviderLaneSnapshot:
    return conversation.StoredProviderLaneSnapshot(
        binding=_binding("lane-2", endpoint="https://stored.example.test/v1"),
        upstream_response_id=conversation.UpstreamResponseId("upstream-1"),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.ALL_TURNS,
            effective=conversation.EffectiveReasoningContext.ALL_TURNS,
        ),
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.DISCARD_TERMINAL,
    )


def _retention() -> conversation.RetentionLimits:
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.PROCESS_LOCAL,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=3600,
    )


def _checkpoint(
    *,
    lanes: tuple[conversation.ProviderLaneSnapshot, ...] | None = None,
    kind: conversation.CheckpointKind = (
        conversation.CheckpointKind.COMPLETED_OUTWARD_TURN
    ),
    lifecycle: conversation.CheckpointLifecycle = (
        conversation.CheckpointLifecycle.COMMITTED
    ),
    checkpoint_id: str = "checkpoint-1",
    parent_checkpoint_id: str | None = None,
) -> conversation.ConversationCheckpoint:
    committed = lifecycle in {
        conversation.CheckpointLifecycle.COMMITTED,
        conversation.CheckpointLifecycle.TOMBSTONED,
        conversation.CheckpointLifecycle.EXPIRED,
        conversation.CheckpointLifecycle.DELETED,
        conversation.CheckpointLifecycle.SUPERSEDED,
    }
    identity = conversation.CheckpointIdentity(
        conversation_id=conversation.ConversationId("conversation-1"),
        logical_turn_id=conversation.LogicalTurnId("turn-1"),
        execution_segment_id=conversation.ExecutionSegmentId("segment-1"),
        checkpoint_id=conversation.CheckpointId(checkpoint_id),
        branch_id=conversation.ConversationBranchId("branch-1"),
        sequence=conversation.CheckpointSequence(
            1 if parent_checkpoint_id else 0
        ),
        parent_checkpoint_id=(
            conversation.CheckpointId(parent_checkpoint_id)
            if parent_checkpoint_id
            else None
        ),
        parent_sequence=(
            conversation.CheckpointSequence(0)
            if parent_checkpoint_id
            else None
        ),
    )
    return conversation.ConversationCheckpoint(
        identity=identity,
        kind=kind,
        lifecycle=lifecycle,
        authority=_authority(),
        content=conversation.MultiLaneCheckpointContent(
            visible_transcript=conversation.VisibleTranscript(
                entries=(
                    conversation.VisibleTranscriptEntry(
                        role=conversation.VisibleTranscriptRole.USER,
                        content="Visible user text",
                    ),
                    conversation.VisibleTranscriptEntry(
                        role=conversation.VisibleTranscriptRole.ASSISTANT,
                        content="Visible assistant text",
                    ),
                )
            ),
            lanes=lanes or (_stateless_lane(),),
        ),
        timestamps=conversation.CheckpointTimestamps(
            created_at=_CREATED,
            committed_at=(
                _CREATED + timedelta(seconds=1) if committed else None
            ),
            expires_at=_CREATED + timedelta(hours=1),
            tombstoned_at=(
                _CREATED + timedelta(minutes=1)
                if lifecycle
                in {
                    conversation.CheckpointLifecycle.TOMBSTONED,
                    conversation.CheckpointLifecycle.DELETED,
                }
                else None
            ),
            deleted_at=(
                _CREATED + timedelta(hours=2)
                if lifecycle is conversation.CheckpointLifecycle.DELETED
                else None
            ),
        ),
        retention=_retention(),
        head=conversation.NamedHeadMetadata(
            head_id=conversation.NamedHeadId("main"),
            revision=conversation.NamedHeadRevision(1),
        ),
    )


def _mutated_json(
    encoded: bytes, mutate: Callable[[dict[str, object]], None]
) -> bytes:
    value: object = loads(encoded)
    assert isinstance(value, dict)
    payload = {str(key): item for key, item in value.items()}
    mutate(payload)
    return dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def test_normative_domain_contract(
    record_property: Callable[[str, object], None],
) -> None:
    """Prove opaque state, exact ordering, lane binding, and ID separation."""
    record_property("conversation_acceptance_evidence", "contract")
    parent = conversation.with_checkpoint_integrity(
        _checkpoint(lanes=(_stateless_lane(), _stored_lane()))
    )
    codec = conversation.ConversationCheckpointCodec()
    encoded = codec.encode(parent)
    restored = codec.decode(encoded)
    restored_stateless = restored.content.lanes[0]
    parent_stateless = parent.content.lanes[0]
    restored_stored = restored.content.lanes[1]
    assert isinstance(
        restored_stateless, conversation.StatelessProviderLaneSnapshot
    )
    assert isinstance(
        parent_stateless, conversation.StatelessProviderLaneSnapshot
    )
    assert isinstance(restored_stored, conversation.StoredProviderLaneSnapshot)

    assert restored == parent
    assert codec.encode(restored) == encoded
    assert tuple(
        item.kind for item in restored_stateless.ledger.items
    ) == tuple(item.kind for item in parent_stateless.ledger.items)
    assert (
        restored.content.visible_transcript
        == parent.content.visible_transcript
    )
    assert restored.content.lanes[0].lane_id == conversation.ProviderLaneId(
        "lane-1"
    )
    assert restored.content.lanes[1].lane_id == conversation.ProviderLaneId(
        "lane-2"
    )
    public_id = conversation.PublicResponseId("response-1")
    assert str(public_id) != str(restored_stored.upstream_response_id)
    sentinel = "opaque-reasoning-0"
    assert sentinel not in repr(parent)
    assert sentinel not in str(parent)
    assert b64encode(sentinel.encode()) in encoded


def test_settings_parents_handles_results_and_compaction_are_closed() -> None:
    """Construct every valid mode, parent, reasoning, and result variant."""
    stateless_handle = conversation.StatelessConversationHandle(
        conversation_id=conversation.ConversationId("conversation-1"),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        branch_id=conversation.ConversationBranchId("branch-1"),
        envelope=conversation.CallerHeldState(_value="sealed-envelope"),
    )
    stored_handle = conversation.StoredConversationHandle(
        conversation_id=conversation.ConversationId("conversation-1"),
        checkpoint_id=conversation.CheckpointId("checkpoint-2"),
        branch_id=conversation.ConversationBranchId("branch-1"),
        public_response_id=conversation.PublicResponseId("response-2"),
    )
    stateless_parent = conversation.StatelessParent(handle=stateless_handle)
    stored_parent = conversation.StoredParent(handle=stored_handle)
    one_shot = conversation.OneShotConversationSettings()
    stateless = conversation.StatelessConversationSettings(
        parent=stateless_parent,
        reasoning_context=conversation.ReasoningContext.CURRENT_TURN,
        compaction=conversation.InlineCompaction(compact_threshold=512),
    )
    stored = conversation.StoredConversationSettings(
        parent=stored_parent,
        provider_storage_disclosed=True,
        reasoning_context=conversation.ReasoningContext.ALL_TURNS,
    )
    result = conversation.ConversationResult(
        handle=stateless_handle,
        reasoning=_reasoning(),
        checkpoint_digest=conversation.IntegrityDigest("a" * 64),
    )
    terminal = conversation.ConversationStreamTerminal(result=result)
    head_parent = conversation.NamedHeadParent(
        head_id=conversation.NamedHeadId("main"),
        expected_revision=conversation.NamedHeadRevision(1),
        parent=stored_parent,
    )
    compact_request = conversation.StandaloneCompactRequest(
        parent=stateless_parent
    )
    compact_handle = conversation.StandaloneCompactHandle(
        conversation_id=conversation.ConversationId("conversation-1"),
        checkpoint_id=conversation.CheckpointId("compact-checkpoint"),
        branch_id=conversation.ConversationBranchId("branch-1"),
        parent_checkpoint_id=stateless_handle.checkpoint_id,
    )
    compact_ledger = conversation.ProviderItemLedger(
        lane_id=conversation.ProviderLaneId("lane-1"),
        normalization_version=conversation.ConversationCodecVersion(1),
        items=(_item(conversation.ProviderItemKind.COMPACTION, 0),),
    )
    compact_result = conversation.StandaloneCompactResult(
        handle=compact_handle,
        binding=_binding(),
        canonical_context=compact_ledger,
        reasoning=_reasoning(),
        usage=conversation.ProviderUsage(input_tokens=1),
        canonical_context_digest=conversation.IntegrityDigest("b" * 64),
    )

    assert one_shot.mode is conversation.ConversationMode.OFF
    assert stateless.mode is conversation.ConversationMode.STATELESS
    assert stored.mode is conversation.ConversationMode.STORED
    assert isinstance(stateless.compaction, conversation.InlineCompaction)
    assert terminal.result == result
    assert head_parent.parent == stored_parent
    assert (
        compact_request.operation
        is conversation.CompactionOperation.STANDALONE
    )
    assert compact_result.handle == compact_handle
    assert conversation.DisabledCompaction().operation is (
        conversation.CompactionOperation.NONE
    )
    assert set(conversation.ConversationResetDisposition) == {
        conversation.ConversationResetDisposition.PRESERVED,
        conversation.ConversationResetDisposition.OPAQUE_STATE_LOST,
    }


@pytest.mark.parametrize(
    "factory",
    (
        lambda: conversation.OneShotConversationSettings(
            mode=conversation.ConversationMode.STATELESS
        ),
        lambda: conversation.StatelessConversationHandle(
            conversation_id=conversation.ConversationId(""),
            checkpoint_id=conversation.CheckpointId("checkpoint"),
            branch_id=conversation.ConversationBranchId("branch"),
        ),
        lambda: conversation.StoredConversationSettings(
            provider_storage_disclosed=False
        ),
        lambda: conversation.InlineCompaction(compact_threshold=0),
    ),
)
def test_settings_reject_invalid_cross_field_combinations(
    factory: Callable[[], object],
) -> None:
    """Reject runtime attempts to bypass closed settings unions."""
    with pytest.raises(conversation.ConversationValidationError):
        factory()


def test_binding_and_capability_profiles_require_exact_evidence(
    record_property: Callable[[str, object], None],
) -> None:
    """Validate complete binding identity and evidence-bearing capabilities."""
    record_property("conversation_acceptance_evidence", "contract")
    binding = _binding()
    evidence = tuple(
        conversation.CapabilityEvidence(
            capability=capability,
            state=conversation.CapabilityEvidenceState.TEST_ONLY,
            evidence_ids=(f"test-{capability.value}",),
        )
        for capability in conversation.ConversationCapability
    )
    profile = conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId("synthetic-profile"),
        schema_version=1,
        revision=conversation.CapabilityProfileRevision("capability-v1"),
        binding_alias=binding.safe_alias,
        capabilities=evidence,
        test_only=True,
    )
    profile.assert_binding(binding)
    for capability in conversation.ConversationCapability:
        profile.require(capability)
    binding.assert_compatible(binding)

    assert (
        conversation.normalize_endpoint("HTTPS://API.EXAMPLE.TEST:443/v1/")
        == "https://api.example.test/v1"
    )
    assert (
        conversation.normalize_endpoint("http://api.example.test:8080/v1")
        == "http://api.example.test:8080/v1"
    )
    assert binding.safe_alias.startswith("lane-binding-")

    with pytest.raises(conversation.ConversationBindingDriftError):
        binding.assert_compatible(
            replace(binding, model_or_deployment="drift")
        )
    with pytest.raises(conversation.ConversationBindingDriftError):
        profile.assert_binding(
            replace(
                binding,
                sdk_revision=conversation.ProviderSdkRevision("sdk-v2"),
            )
        )


@pytest.mark.parametrize(
    "endpoint",
    (
        "ftp://api.example.test/v1",
        "https://user@api.example.test/v1",
        "https://api.example.test/v1?key=secret",
        "https://api.example.test/v1#fragment",
        "https://api.example.test:bad/v1",
    ),
)
def test_endpoint_normalization_rejects_unsafe_identity(endpoint: str) -> None:
    """Reject credentials, unstable components, and malformed endpoints."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.normalize_endpoint(endpoint)


def test_provider_item_union_round_trips_in_exact_order(
    record_property: Callable[[str, object], None],
) -> None:
    """Round-trip every supported item kind without regrouping adjacency."""
    record_property("conversation_acceptance_evidence", "wire")
    ledger = _full_ledger()
    checkpoint = conversation.with_checkpoint_integrity(_checkpoint())
    restored = conversation.ConversationCheckpointCodec().decode(
        conversation.ConversationCheckpointCodec().encode(checkpoint)
    )
    restored_lane = restored.content.lanes[0]
    assert isinstance(
        restored_lane,
        conversation.StatelessProviderLaneSnapshot,
    )
    assert restored_lane.ledger == ledger
    assert tuple(item.order for item in ledger.items) == tuple(
        range(len(ledger.items))
    )
    assert tuple(item.provider_index for item in ledger.items) == tuple(
        range(len(ledger.items))
    )
    assert len(ledger.items) == len(conversation.ProviderItemKind)
    assert ledger.item_count == len(conversation.ProviderItemKind)


def test_transcript_changes_do_not_mutate_provider_ledger() -> None:
    """Keep visible transcript display edits separate from provider state."""
    checkpoint = _checkpoint()
    checkpoint_lane = checkpoint.content.lanes[0]
    assert isinstance(
        checkpoint_lane, conversation.StatelessProviderLaneSnapshot
    )
    ledger = checkpoint_lane.ledger
    original_digest = conversation.json_digest(
        tuple(item.canonical_input for item in ledger.items)
    )
    changed_content = replace(
        checkpoint.content,
        visible_transcript=conversation.VisibleTranscript(
            entries=(
                conversation.VisibleTranscriptEntry(
                    role=conversation.VisibleTranscriptRole.SYSTEM,
                    content="Changed display only",
                ),
            )
        ),
    )
    child = replace(checkpoint, content=changed_content)
    child_lane = child.content.lanes[0]
    assert isinstance(child_lane, conversation.StatelessProviderLaneSnapshot)
    assert child_lane.ledger is ledger
    assert (
        conversation.json_digest(
            tuple(item.canonical_input for item in child_lane.ledger.items)
        )
        == original_digest
    )
    assert (
        child.content.visible_transcript
        != checkpoint.content.visible_transcript
    )


def test_reasoning_summary_is_separate_from_opaque_continuation_digest(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep displayable summaries distinct from opaque continuity bytes."""
    record_property("conversation_acceptance_evidence", "contract")
    reasoning = _item(conversation.ProviderItemKind.REASONING, 0)
    summary = reasoning.canonical_input["summary"]
    assert reasoning.opaque_state is not None
    opaque_digest = reasoning.opaque_state.digest
    changed_input = dict(reasoning.canonical_input)
    changed_input["summary"] = (
        {"text": "changed-summary", "type": "summary_text"},
    )
    changed = replace(reasoning, canonical_input=changed_input)
    assert changed.canonical_input["summary"] != summary
    assert changed.opaque_state is reasoning.opaque_state
    assert changed.opaque_state.digest == opaque_digest
    assert "opaque-reasoning-0" not in repr(summary)


def test_ledger_rejects_invalid_order_correlation_and_lanes() -> None:
    """Reject incomplete or implicitly regrouped provider ledgers."""
    valid = _full_ledger()
    duplicate_input = dict(valid.items[1].canonical_input)
    duplicate_input["id"] = valid.items[0].item_id
    duplicate = replace(
        valid.items[1],
        item_id=valid.items[0].item_id,
        canonical_input=duplicate_input,
    )
    reordered = replace(
        valid.items[1], order=conversation.ProviderItemOrder(2)
    )
    mixed = replace(
        valid.items[1], lane_id=conversation.ProviderLaneId("other")
    )
    missing_output = tuple(
        item
        for item in valid.items
        if item.kind is not conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT
    )
    cases = (
        (valid.items[0], duplicate, *valid.items[2:]),
        (valid.items[0], reordered, *valid.items[2:]),
        (valid.items[0], mixed, *valid.items[2:]),
        missing_output,
    )
    for items in cases:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ProviderItemLedger(
                lane_id=conversation.ProviderLaneId("lane-1"),
                normalization_version=conversation.ConversationCodecVersion(1),
                items=items,
            )

    with pytest.raises(conversation.ConversationValidationError):
        replace(valid.items[0], complete=False)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            valid.items[0], call_id=conversation.ProviderCallId("unexpected")
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(valid.items[0], opaque_state=None)


def test_compaction_boundary_must_be_latest_and_exact() -> None:
    """Reject missing, stale, and overlapping compaction boundaries."""
    ledger = _full_ledger()
    compact = ledger.items[-1]
    valid = conversation.CompactionBoundary(
        boundary_item_id=compact.item_id,
        boundary_order=compact.order,
        retained_suffix=(),
    )
    valid.validate_latest(ledger)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            valid, boundary_item_id=conversation.ProviderItemId("missing")
        ).validate_latest(ledger)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            valid,
            retained_suffix=(ledger.items[0].item_id,),
        ).validate_latest(ledger)
    without_compaction = conversation.ProviderItemLedger(
        lane_id=ledger.lane_id,
        normalization_version=ledger.normalization_version,
        items=tuple(
            replace(
                item,
                order=conversation.ProviderItemOrder(index),
                provider_index=conversation.ProviderItemIndex(index),
            )
            for index, item in enumerate(ledger.items[:-1])
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        valid.validate_latest(without_compaction)


def test_single_and_multi_lane_checkpoints_round_trip_immutably(
    record_property: Callable[[str, object], None],
) -> None:
    """Round-trip checkpoints while preserving parent bytes and lane origin."""
    record_property("conversation_acceptance_evidence", "database")
    codec = conversation.ConversationCheckpointCodec()
    parent = conversation.with_checkpoint_integrity(_checkpoint())
    parent_bytes = codec.encode(parent)
    parent_digest = conversation.checkpoint_payload_digest(parent)
    child = conversation.with_checkpoint_integrity(
        _checkpoint(
            lanes=(_stateless_lane(), _stored_lane()),
            checkpoint_id="checkpoint-2",
            parent_checkpoint_id="checkpoint-1",
        )
    )
    child_bytes = codec.encode(child)
    restored_child = codec.decode(child_bytes)

    assert codec.encode(parent) == parent_bytes
    assert conversation.checkpoint_payload_digest(parent) == parent_digest
    assert (
        restored_child.identity.parent_checkpoint_id
        == parent.identity.checkpoint_id
    )
    assert len(restored_child.content.lanes) == 2
    assert (
        restored_child.content.lanes[0].binding
        != restored_child.content.lanes[1].binding
    )
    committed_at = restored_child.timestamps.committed_at
    assert committed_at is not None
    assert restored_child.timestamps.created_at <= committed_at
    assert (
        conversation.reduce_provider_lane(
            conversation.ProviderLaneLifecycle.STAGED,
            conversation.ProviderLaneLifecycle.COMMITTED,
        )
        is conversation.ProviderLaneLifecycle.COMMITTED
    )
    with pytest.raises(FrozenInstanceError):
        setattr(
            parent.identity,
            "checkpoint_id",
            conversation.CheckpointId("mutated"),
        )


def test_canonical_request_and_idempotency_digests_bind_semantics() -> None:
    """Keep digests deterministic, authority-bound, and secret-free."""
    opaque = conversation.OpaqueProviderState(_value=b"opaque-secret-sentinel")
    request = conversation.ConversationRequestSemantics(
        authority=_authority(),
        operation=conversation.ConversationOperation.CONTINUE,
        mode=conversation.ConversationMode.STATELESS,
        reasoning_context=conversation.ReasoningContext.ALL_TURNS,
        parent_checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        semantic_input={"prompt": "safe semantic input"},
        opaque_digests=(opaque.digest,),
    )
    same = replace(request, semantic_input={"prompt": "safe semantic input"})
    changed = replace(
        request, reasoning_context=conversation.ReasoningContext.CURRENT_TURN
    )
    digest = conversation.canonical_request_digest(request)

    assert digest == conversation.canonical_request_digest(same)
    assert digest != conversation.canonical_request_digest(changed)
    assert digest != conversation.canonical_request_digest(
        replace(
            request,
            authority=replace(
                request.authority,
                principal_id=conversation.AuthorityPrincipalId("principal-2"),
            ),
        )
    )
    key = conversation.RequestIdempotencyKey("key-1")
    assert conversation.idempotency_digest(request, key) == (
        conversation.idempotency_digest(same, key)
    )
    assert conversation.idempotency_digest(request, key) != (
        conversation.idempotency_digest(
            request,
            conversation.RequestIdempotencyKey("key-2"),
        )
    )
    assert "opaque-secret-sentinel" not in digest
    assert len(conversation.authority_digest(request.authority)) == 64


def test_safe_observability_and_diagnostics_never_expose_opaque_state(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep opaque values absent from representations and public mappings."""
    record_property("conversation_acceptance_evidence", "security")
    checkpoint = conversation.with_checkpoint_integrity(_checkpoint())
    observation = conversation.checkpoint_observation(
        "checkpoint.commit", checkpoint
    )
    projected = observation.to_mapping()
    allowed = {
        "event",
        "checkpoint_id",
        "authority_scope_digest",
        "parent_checkpoint_id",
        "lane_ids",
        "lane_count",
        "provider_item_count",
        "transcript_entry_count",
        "opaque_byte_count",
        "checkpoint_state",
        "codec_version",
        "integrity_digest",
        "binding_aliases",
    }
    sentinel = "opaque-reasoning-0"
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    assert set(projected) == allowed
    assert projected["authority_scope_digest"] == (
        conversation.authority_digest(checkpoint.authority)
    )
    assert projected["parent_checkpoint_id"] is None
    assert projected["lane_ids"] == (lane.lane_id,)
    assert sentinel not in repr(observation)
    assert sentinel not in repr(projected)
    assert sentinel not in repr(lane.ledger)
    assert checkpoint.content.safe_counts.lane_count == 1
    assert checkpoint.content.safe_counts.provider_item_count == len(
        conversation.ProviderItemKind
    )
    with pytest.raises(TypeError):
        setitem(
            cast(MutableMapping[str, JsonValue], projected),
            "event",
            "mutated",
        )


def test_opaque_and_caller_held_values_are_redaction_safe() -> None:
    """Redact opaque provider bytes and caller-held state by default."""
    opaque = conversation.OpaqueProviderState(_value=b"opaque-sentinel")
    envelope = conversation.CallerHeldState(_value="envelope-sentinel")
    for sentinel, value in (
        ("opaque-sentinel", opaque),
        ("envelope-sentinel", envelope),
    ):
        assert sentinel not in repr(value)
        assert sentinel not in str(value)
        assert value.byte_count > 0
        assert len(value.digest) == 64
    assert opaque._codec_bytes() == b"opaque-sentinel"
    assert envelope._codec_text() == "envelope-sentinel"


def test_codec_rejects_invalid_and_noncanonical_data() -> None:
    """Reject malformed, duplicate, non-finite, and noncanonical envelopes."""
    codec = conversation.ConversationCheckpointCodec()
    encoded = codec.encode(
        conversation.with_checkpoint_integrity(_checkpoint())
    )
    duplicate = b'{"kind":"a","kind":"b","version":1,"checkpoint":{}}'
    nonfinite = b'{"kind":"a","version":NaN,"checkpoint":{}}'
    unknown = _mutated_json(
        encoded, lambda value: value.__setitem__("version", 99)
    )
    extra = _mutated_json(
        encoded, lambda value: value.__setitem__("extra", True)
    )
    noncanonical = encoded.replace(b'"kind":', b'"kind" :', 1)
    invalid_utf8 = b"\xff"
    for value in (
        duplicate,
        nonfinite,
        unknown,
        extra,
        noncanonical,
        invalid_utf8,
    ):
        with pytest.raises(conversation.ConversationCodecError):
            codec.decode(value)


def test_codec_rejects_integrity_tampering_and_invalid_opaque_encoding() -> (
    None
):
    """Authenticate exact opaque payload and reject invalid base64."""
    codec = conversation.ConversationCheckpointCodec()
    encoded = codec.encode(
        conversation.with_checkpoint_integrity(_checkpoint())
    )

    def mutate_digest(value: dict[str, object]) -> None:
        checkpoint = value["checkpoint"]
        assert isinstance(checkpoint, dict)
        integrity = checkpoint["integrity"]
        assert isinstance(integrity, dict)
        integrity["digest"] = "0" * 64

    def mutate_opaque(value: dict[str, object]) -> None:
        checkpoint = value["checkpoint"]
        assert isinstance(checkpoint, dict)
        content = checkpoint["content"]
        assert isinstance(content, dict)
        lanes = content["lanes"]
        assert isinstance(lanes, list)
        lane = lanes[0]
        assert isinstance(lane, dict)
        ledger = lane["ledger"]
        assert isinstance(ledger, dict)
        items = ledger["items"]
        assert isinstance(items, list)
        item = items[0]
        assert isinstance(item, dict)
        item["opaque_state"] = "***"

    with pytest.raises(conversation.ConversationIntegrityError):
        codec.decode(_mutated_json(encoded, mutate_digest))
    with pytest.raises(conversation.ConversationCodecError):
        codec.decode(_mutated_json(encoded, mutate_opaque))


def test_codec_enforces_byte_depth_count_and_string_limits() -> None:
    """Reject encoded payloads that exceed any configured bound."""
    checkpoint = conversation.with_checkpoint_integrity(_checkpoint())
    encoded = conversation.ConversationCheckpointCodec().encode(checkpoint)
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ConversationCheckpointCodec(
            limits=conversation.CheckpointCodecLimits(max_bytes=10)
        ).decode(encoded)
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ConversationCheckpointCodec(
            limits=conversation.CheckpointCodecLimits(max_depth=2)
        ).decode(encoded)
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ConversationCheckpointCodec(
            limits=conversation.CheckpointCodecLimits(max_items=2)
        ).decode(encoded)
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ConversationCheckpointCodec(
            limits=conversation.CheckpointCodecLimits(max_string_bytes=3)
        ).decode(encoded)
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ConversationCheckpointCodec(
            limits=conversation.CheckpointCodecLimits(max_bytes=10)
        ).encode(checkpoint)


def test_json_values_are_deeply_immutable_finite_and_bounded() -> None:
    """Validate recursive JSON before admitting it to a public core type."""
    frozen = conversation.freeze_json_value(
        {"nested": [1, True, None, 1.5, {"value": "text"}]}
    )
    assert isinstance(frozen, Mapping)
    assert conversation.thaw_json_value(frozen) == {
        "nested": [1, True, None, 1.5, {"value": "text"}]
    }
    assert (
        conversation.canonical_json_bytes(frozen)
        == b'{"nested":[1,true,null,1.5,{"value":"text"}]}'
    )
    with pytest.raises(TypeError):
        setitem(
            cast(MutableMapping[str, JsonValue], frozen),
            "nested",
            (),
        )
    for invalid in (float("nan"), float("inf"), object(), {1: "bad"}):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.freeze_json_value(invalid)
    with pytest.raises(conversation.ConversationLimitError):
        conversation.freeze_json_value(
            ((("too-deep",),),),
            limits=conversation.JsonLimits(max_depth=1),
        )
    with pytest.raises(conversation.ConversationLimitError):
        conversation.freeze_json_value(
            (1, 2, 3),
            limits=conversation.JsonLimits(max_items=2),
        )
    with pytest.raises(conversation.ConversationLimitError):
        conversation.freeze_json_value(
            "long",
            limits=conversation.JsonLimits(max_string_bytes=3),
        )


def test_lifecycle_reducers_accept_only_legal_transitions() -> None:
    """Apply legal lifecycles and reject illegal transitions."""
    assert (
        conversation.reduce_checkpoint_lifecycle(
            conversation.CheckpointLifecycle.STAGED,
            conversation.CheckpointLifecycle.COMMITTED,
        )
        is conversation.CheckpointLifecycle.COMMITTED
    )
    assert (
        conversation.reduce_response_resource(
            conversation.ResponseResourceState.ALLOCATED,
            conversation.ResponseResourceState.DISPATCHING,
        )
        is conversation.ResponseResourceState.DISPATCHING
    )
    assert (
        conversation.reduce_provider_lane(
            conversation.ProviderLaneLifecycle.STAGED,
            conversation.ProviderLaneLifecycle.SUSPENDED,
        )
        is conversation.ProviderLaneLifecycle.SUSPENDED
    )
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("main"),
        revision=conversation.NamedHeadRevision(4),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
    )
    advanced = conversation.reduce_named_head(
        head,
        expected_revision=conversation.NamedHeadRevision(4),
        checkpoint_id=conversation.CheckpointId("checkpoint-2"),
    )
    assert advanced.revision == 5
    deletion = conversation.DeletionSnapshot(
        local=conversation.LocalDeletionState.ACTIVE,
        upstream=conversation.UpstreamDeletionState.NOT_APPLICABLE,
    )
    tombstoned = conversation.reduce_deletion(
        deletion,
        local=conversation.LocalDeletionState.TOMBSTONED,
    )
    assert tombstoned.local is conversation.LocalDeletionState.TOMBSTONED

    invalid_transitions: tuple[Callable[[], object], ...] = (
        lambda: conversation.reduce_checkpoint_lifecycle(
            conversation.CheckpointLifecycle.COMMITTED,
            conversation.CheckpointLifecycle.STAGED,
        ),
        lambda: conversation.reduce_response_resource(
            conversation.ResponseResourceState.COMPLETED,
            conversation.ResponseResourceState.DISPATCHING,
        ),
        lambda: conversation.reduce_provider_lane(
            conversation.ProviderLaneLifecycle.TOMBSTONED,
            conversation.ProviderLaneLifecycle.COMMITTED,
        ),
        lambda: conversation.reduce_named_head(
            head,
            expected_revision=conversation.NamedHeadRevision(3),
            checkpoint_id=conversation.CheckpointId("checkpoint-2"),
        ),
        lambda: conversation.reduce_deletion(deletion),
        lambda: conversation.reduce_deletion(
            deletion,
            local=conversation.LocalDeletionState.TOMBSTONED,
            upstream=conversation.UpstreamDeletionState.PENDING,
        ),
    )
    for operation in invalid_transitions:
        with pytest.raises(conversation.ConversationTransitionError):
            operation()


def test_checkpoint_candidate_variants_enforce_exact_boundaries(
    record_property: Callable[[str, object], None],
) -> None:
    """Construct every private and outward candidate with a fixed kind."""
    record_property("conversation_acceptance_evidence", "contract")
    staged_internal = conversation.with_checkpoint_integrity(
        _checkpoint(
            kind=conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
        )
    )
    staged_suspension = conversation.with_checkpoint_integrity(
        replace(
            staged_internal,
            kind=conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
        )
    )
    staged_outward = conversation.with_checkpoint_integrity(
        replace(
            staged_internal,
            kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
        )
    )
    staged_compact = conversation.with_checkpoint_integrity(
        _checkpoint(
            kind=conversation.CheckpointKind.STANDALONE_COMPACT_RESULT,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
            parent_checkpoint_id="checkpoint-parent",
        )
    )
    continuation = _portable_continuation()
    handle = conversation.StandaloneCompactHandle(
        conversation_id=staged_compact.identity.conversation_id,
        checkpoint_id=staged_compact.identity.checkpoint_id,
        branch_id=staged_compact.identity.branch_id,
        parent_checkpoint_id=conversation.CheckpointId("checkpoint-parent"),
    )
    assert (
        conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=staged_internal
        ).checkpoint
        == staged_internal
    )
    assert (
        conversation.SuspensionCheckpointCandidate(
            checkpoint=staged_suspension,
            continuation=continuation,
        ).continuation
        == continuation
    )
    assert (
        conversation.OutwardTurnCheckpointCandidate(
            checkpoint=staged_outward,
            public_response_id=conversation.PublicResponseId("response-1"),
        ).public_response_id
        == "response-1"
    )
    assert (
        conversation.StandaloneCompactCheckpointCandidate(
            checkpoint=staged_compact,
            handle=handle,
        ).handle
        == handle
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=staged_outward
        )


def _portable_continuation() -> conversation.PortableContinuationReference:
    definition = ExecutionDefinitionRef(
        agent_definition_locator="agent.toml",
        agent_definition_revision="agent-v1",
        operation_id="operation-1",
        operation_index=0,
        model_config_reference="model-config",
        tool_revision="tools-v1",
        capability_revision=CapabilityRevision("capability-v1"),
    )
    binding = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("synthetic"),
        model_id=ModelId("model-1"),
        provider_config_revision=ProviderConfigRevision("provider-v1"),
        model_config_revision=ModelConfigRevision("model-v1"),
        capability_revision=CapabilityRevision("capability-v1"),
    )
    return conversation.PortableContinuationReference(
        continuation_id=ContinuationId("continuation-1"),
        state_revision=StateRevision(1),
        digest=conversation.ContinuationDigest("digest-1"),
        definition=definition,
        revision_binding=binding,
    )


def test_async_effect_protocols_expose_no_synchronous_effect_method() -> None:
    """Require every effect protocol method to be asynchronous."""
    for protocol, methods in (
        (
            conversation.ConversationCoordinator,
            ("execute", "stream"),
        ),
        (
            conversation.ConversationStore,
            ("load", "commit", "close"),
        ),
        (
            conversation.ConversationProvider,
            ("dispatch", "stream"),
        ),
        (conversation.ConversationObserver, ("publish",)),
    ):
        for method in methods:
            assert iscoroutinefunction(getattr(protocol, method))
    assert iscoroutinefunction(
        conversation.ConversationProviderStream.terminal
    )
    assert iscoroutinefunction(conversation.ConversationProviderStream.aclose)


def test_provider_plans_make_stored_and_stateless_state_exclusive(
    record_property: Callable[[str, object], None],
) -> None:
    """Construct mutually exclusive typed provider plans and results."""
    record_property("conversation_acceptance_evidence", "runtime")
    stateless = conversation.StatelessProviderPlan(
        binding=_binding(),
        ledger=_full_ledger(),
        reasoning=_reasoning(),
    )
    stored = conversation.StoredProviderPlan(
        binding=_binding(),
        upstream_response_id=conversation.UpstreamResponseId("upstream-1"),
        reasoning=_reasoning(),
    )
    result = conversation.ProviderResult(
        items=stateless.ledger.items,
        reasoning=_reasoning(),
        upstream_response_id=stored.upstream_response_id,
    )
    assert stateless.ledger.lane_id == stateless.binding.lane_id
    assert stored.upstream_response_id == "upstream-1"
    assert result.items == stateless.ledger.items
    observation = conversation.checkpoint_observation(
        "checkpoint.private",
        conversation.with_checkpoint_integrity(
            _checkpoint(lanes=(_stored_lane(),))
        ),
    )
    assert "upstream-1" not in repr(observation.to_mapping())
    assert iscoroutinefunction(conversation.ConversationProvider.dispatch)
    assert iscoroutinefunction(conversation.ConversationStore.load)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StatelessProviderPlan(
            binding=_binding("lane-2"),
            ledger=_full_ledger("lane-1"),
            reasoning=_reasoning(),
        )


def test_stable_error_family_is_content_safe_and_complete() -> None:
    """Expose stable error codes without retaining sensitive details."""
    errors: tuple[conversation.ConversationError, ...] = (
        conversation.ConversationValidationError(),
        conversation.ConversationCapabilityError(),
        conversation.ConversationBindingDriftError(),
        conversation.ConversationConflictError(),
        conversation.ConversationIntegrityError(),
        conversation.ConversationExpiredError(),
        conversation.ConversationDeletedError(),
        conversation.ConversationStorageError(),
        conversation.ConversationAmbiguousDispatchError(),
        conversation.ConversationCommitError(),
        conversation.ConversationPublicationError(),
        conversation.ConversationAuthorizationError(),
        conversation.ConversationLimitError(),
        conversation.ConversationCodecError(),
        conversation.ConversationTransitionError(),
    )
    assert {error.code for error in errors} == set(
        conversation.ConversationErrorCode
    )
    assert len({error.code.value for error in errors}) == len(errors)
    for error in errors:
        assert "secret-sentinel" not in str(error)
        assert error.safe_message == str(error)
        assert "safe_message" not in repr(error)


def test_public_core_sources_have_no_dynamic_or_synchronous_escape_hatch() -> (
    None
):
    """Keep core annotations strict and public activation absent."""
    source_root = _ROOT / "src/avalan/conversation"
    sources = tuple(path for path in source_root.glob("*.py"))
    assert sources
    prohibited = ("".join(("A", "ny")), "# type: ignore", "__all__")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert all(value not in text for value in prohibited)
    assert not hasattr(conversation, "ConversationTool")
    assert not hasattr(conversation, "ConversationTaskInput")
    provider_source = (
        _ROOT / "src/avalan/model/nlp/text/vendor/openai.py"
    ).read_bytes()
    transition_path = (
        _ROOT / "tests/fixtures/conversation/provider_transition.phase13.json"
    )
    transition_payload = cast(
        dict[str, object],
        loads(transition_path.read_text(encoding="utf-8")),
    )
    transitions = cast(
        list[dict[str, object]], transition_payload["transitions"]
    )
    assert transition_payload["phase"] == 13
    provider_transition = next(
        item
        for item in transitions
        if item["path"] == "src/avalan/model/nlp/text/vendor/openai.py"
    )
    assert (
        sha256(provider_source).hexdigest() == provider_transition["to_sha256"]
    )


@pytest.mark.parametrize(
    "value",
    (
        "",
        " leading",
        "trailing ",
        "embedded\x00nul",
        "x" * 513,
        1,
    ),
)
def test_identifiers_reject_empty_malformed_and_overlong_values(
    value: object,
) -> None:
    """Reject malformed IDs before they enter a canonical entity."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.validate_identifier(value, "identifier")


@pytest.mark.parametrize("value", (-1, 1.0, True, "1"))
def test_revisions_reject_non_integer_or_negative_values(
    value: object,
) -> None:
    """Reject malformed revisions at every canonical boundary."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.validate_revision(value, "revision")


def test_checkpoint_metadata_rejects_invalid_time_lifecycle_and_lanes() -> (
    None
):
    """Reject invalid timestamps, lifecycle metadata, and duplicate lanes."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CheckpointTimestamps(created_at=datetime(2030, 1, 1))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CheckpointTimestamps(
            created_at=_CREATED,
            committed_at=_CREATED - timedelta(seconds=1),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.MultiLaneCheckpointContent(
            visible_transcript=conversation.VisibleTranscript(entries=()),
            lanes=(_stateless_lane(), _stateless_lane()),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            _checkpoint(),
            timestamps=conversation.CheckpointTimestamps(created_at=_CREATED),
        )


def test_azure_resource_identity_is_family_specific() -> None:
    """Require Azure resource identity only for exact Azure lane bindings."""
    common = _binding()
    azure = replace(
        common,
        provider_family=conversation.ProviderFamily.AZURE_OPENAI,
        azure_resource_identity="resource.openai.azure.com",
    )
    assert azure.azure_resource_identity == "resource.openai.azure.com"
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            common, provider_family=conversation.ProviderFamily.AZURE_OPENAI
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(common, azure_resource_identity="resource.example.test")
    with pytest.raises(conversation.ConversationValidationError):
        replace(azure, azure_resource_identity="RESOURCE.openai.azure.com")


class _DuplicateKeyMapping(Mapping[str, object]):
    """Expose one duplicate key through the mapping iteration contract."""

    def __getitem__(self, key: str) -> object:
        if key != "duplicate":
            raise KeyError(key)
        return "value"

    def __iter__(self) -> Iterator[str]:
        return iter(("duplicate", "duplicate"))

    def __len__(self) -> int:
        return 2


class _IdentityJsonMapping(Mapping[str, JsonValue]):
    """Expose valid JSON keys while retaining identity equality."""

    def __init__(self, values: Mapping[str, JsonValue]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> JsonValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        return self is other


class _FakeCapabilityEvidence:
    """Expose a capability while remaining a non-evidence runtime object."""

    def __init__(
        self, capability: conversation.ConversationCapability
    ) -> None:
        self.capability = capability


def _capability_profile(
    state: conversation.CapabilityEvidenceState,
    *,
    test_only: bool,
) -> conversation.ConversationCapabilityProfile:
    binding = _binding()
    capabilities = tuple(
        conversation.CapabilityEvidence(
            capability=capability,
            state=state,
            evidence_ids=(
                (f"evidence-{capability.value}",)
                if state
                in {
                    conversation.CapabilityEvidenceState.TEST_ONLY,
                    conversation.CapabilityEvidenceState.CONFORMANT,
                }
                else ()
            ),
        )
        for capability in conversation.ConversationCapability
    )
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId("profile-1"),
        schema_version=1,
        revision=conversation.CapabilityProfileRevision("capability-v1"),
        binding_alias=binding.safe_alias,
        capabilities=capabilities,
        test_only=test_only,
    )


def test_value_boundaries_reject_invalid_limits_and_duplicate_keys() -> None:
    """Reject invalid opaque values, limit objects, and duplicate mappings."""
    for field in ("max_depth", "max_items", "max_string_bytes"):
        values = {
            "max_depth": 1,
            "max_items": 1,
            "max_string_bytes": 1,
        }
        values[field] = 0
        with pytest.raises(conversation.ConversationValidationError):
            conversation.JsonLimits(**values)
    for value in (b"", cast(bytes, "not-bytes")):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.OpaqueProviderState(_value=value)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.freeze_json_value(
            {}, limits=cast(conversation.JsonLimits, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.freeze_json_value(_DuplicateKeyMapping())


def test_settings_validate_every_closed_union_boundary() -> None:
    """Reject runtime attempts to mix handles, parents, modes, and results."""
    stateless_handle = conversation.StatelessConversationHandle(
        conversation_id=conversation.ConversationId("conversation-1"),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        branch_id=conversation.ConversationBranchId("branch-1"),
    )
    stored_handle = conversation.StoredConversationHandle(
        conversation_id=conversation.ConversationId("conversation-1"),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        branch_id=conversation.ConversationBranchId("branch-1"),
        public_response_id=conversation.PublicResponseId("response-1"),
    )
    stateless_parent = conversation.StatelessParent(handle=stateless_handle)
    stored_parent = conversation.StoredParent(handle=stored_handle)
    invalid_factories: tuple[Callable[[], object], ...] = (
        lambda: conversation.DisabledCompaction(
            operation=conversation.CompactionOperation.INLINE
        ),
        lambda: conversation.InlineCompaction(
            compact_threshold=cast(int, True)
        ),
        lambda: replace(
            stateless_handle, mode=conversation.ConversationMode.STORED
        ),
        lambda: replace(
            stateless_handle,
            envelope=cast(conversation.CallerHeldState, "bad"),
        ),
        lambda: replace(
            stored_handle, mode=conversation.ConversationMode.STATELESS
        ),
        lambda: conversation.StatelessParent(
            handle=cast(
                conversation.StatelessConversationHandle, stored_handle
            )
        ),
        lambda: conversation.StoredParent(
            handle=cast(
                conversation.StoredConversationHandle, stateless_handle
            )
        ),
        lambda: conversation.StatelessConversationSettings(
            parent=cast(conversation.StatelessParent, stored_parent)
        ),
        lambda: conversation.StatelessConversationSettings(
            reasoning_context=cast(conversation.ReasoningContext, "bad")
        ),
        lambda: conversation.StatelessConversationSettings(
            compaction=cast(conversation.CompactionPolicy, object())
        ),
        lambda: conversation.StoredConversationSettings(
            provider_storage_disclosed=cast(bool, 1)
        ),
        lambda: conversation.StoredConversationSettings(
            provider_storage_disclosed=True,
            parent=cast(conversation.StoredParent, stateless_parent),
        ),
        lambda: conversation.StoredConversationSettings(
            provider_storage_disclosed=True,
            reasoning_context=cast(conversation.ReasoningContext, "bad"),
        ),
        lambda: conversation.EffectiveReasoningMetadata(
            requested=cast(conversation.ReasoningContext, "bad"),
            effective=None,
        ),
        lambda: conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=cast(conversation.EffectiveReasoningContext, "bad"),
        ),
        lambda: conversation.ConversationResult(
            handle=cast(conversation.ConversationHandle, object()),
            reasoning=_reasoning(),
            checkpoint_digest=conversation.IntegrityDigest("a" * 64),
        ),
        lambda: conversation.ConversationResult(
            handle=stateless_handle,
            reasoning=cast(conversation.EffectiveReasoningMetadata, object()),
            checkpoint_digest=conversation.IntegrityDigest("a" * 64),
        ),
        lambda: conversation.ConversationStreamTerminal(
            result=cast(conversation.ConversationResult, object())
        ),
        lambda: conversation.NamedHeadParent(
            head_id=conversation.NamedHeadId("main"),
            expected_revision=cast(conversation.NamedHeadRevision, True),
            parent=stateless_parent,
        ),
        lambda: conversation.NamedHeadParent(
            head_id=conversation.NamedHeadId("main"),
            expected_revision=conversation.NamedHeadRevision(1),
            parent=cast(conversation.ConversationParent, object()),
        ),
        lambda: conversation.StandaloneCompactRequest(
            parent=cast(conversation.StatelessParent, stored_parent)
        ),
        lambda: conversation.StandaloneCompactRequest(
            parent=stateless_parent,
            operation=conversation.CompactionOperation.INLINE,
        ),
        lambda: conversation.StandaloneCompactResult(
            handle=cast(conversation.StandaloneCompactHandle, stored_handle),
            binding=_binding(),
            canonical_context=conversation.ProviderItemLedger(
                lane_id=conversation.ProviderLaneId("lane-1"),
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(_item(conversation.ProviderItemKind.COMPACTION, 0),),
            ),
            reasoning=_reasoning(),
            usage=conversation.ProviderUsage(),
            canonical_context_digest=conversation.IntegrityDigest("a" * 64),
        ),
    )
    for factory in invalid_factories:
        with pytest.raises(conversation.ConversationValidationError):
            factory()


def test_binding_and_evidence_reject_every_drift_boundary() -> None:
    """Reject invalid bindings, incomplete evidence, and profile drift."""
    binding = _binding()
    invalid_bindings: tuple[Callable[[], object], ...] = (
        lambda: replace(
            binding,
            provider_family=cast(conversation.ProviderFamily, "bad"),
        ),
        lambda: replace(
            binding, normalized_endpoint="HTTPS://API.EXAMPLE.TEST"
        ),
        lambda: replace(
            binding,
            transport=cast(conversation.ProviderTransport, "bad"),
        ),
    )
    for factory in invalid_bindings:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    invalid_evidence: tuple[Callable[[], object], ...] = (
        lambda: conversation.CapabilityEvidence(
            capability=cast(conversation.ConversationCapability, "bad"),
            state=conversation.CapabilityEvidenceState.DORMANT,
        ),
        lambda: conversation.CapabilityEvidence(
            capability=conversation.ConversationCapability.INLINE_COMPACTION,
            state=cast(conversation.CapabilityEvidenceState, "bad"),
        ),
        lambda: conversation.CapabilityEvidence(
            capability=conversation.ConversationCapability.INLINE_COMPACTION,
            state=conversation.CapabilityEvidenceState.DORMANT,
            evidence_ids=cast(tuple[str, ...], ["bad"]),
        ),
        lambda: conversation.CapabilityEvidence(
            capability=conversation.ConversationCapability.INLINE_COMPACTION,
            state=conversation.CapabilityEvidenceState.CONFORMANT,
            evidence_ids=("duplicate", "duplicate"),
        ),
        lambda: conversation.CapabilityEvidence(
            capability=conversation.ConversationCapability.INLINE_COMPACTION,
            state=conversation.CapabilityEvidenceState.CONFORMANT,
        ),
        lambda: conversation.CapabilityEvidence(
            capability=conversation.ConversationCapability.INLINE_COMPACTION,
            state=conversation.CapabilityEvidenceState.DORMANT,
            evidence_ids=("unexpected",),
        ),
    )
    for factory in invalid_evidence:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    dormant = _capability_profile(
        conversation.CapabilityEvidenceState.DORMANT,
        test_only=False,
    )
    test_profile = _capability_profile(
        conversation.CapabilityEvidenceState.TEST_ONLY,
        test_only=True,
    )
    fake = cast(
        conversation.CapabilityEvidence,
        _FakeCapabilityEvidence(dormant.capabilities[0].capability),
    )
    invalid_profiles: tuple[Callable[[], object], ...] = (
        lambda: replace(dormant, schema_version=0),
        lambda: replace(
            dormant,
            capabilities=cast(tuple[conversation.CapabilityEvidence, ...], []),
        ),
        lambda: replace(dormant, test_only=cast(bool, 1)),
        lambda: replace(dormant, capabilities=dormant.capabilities[:-1]),
        lambda: replace(
            dormant,
            capabilities=(fake, *dormant.capabilities[1:]),
        ),
        lambda: replace(test_profile, test_only=False),
    )
    for factory in invalid_profiles:
        with pytest.raises(conversation.ConversationValidationError):
            factory()
    with pytest.raises(conversation.ConversationValidationError):
        dormant.require(cast(conversation.ConversationCapability, "bad"))
    with pytest.raises(conversation.ConversationCapabilityError):
        dormant.require(conversation.ConversationCapability.INLINE_COMPACTION)
    with pytest.raises(conversation.ConversationValidationError):
        dormant.assert_binding(
            cast(conversation.ProviderLaneBinding, object())
        )
    with pytest.raises(conversation.ConversationBindingDriftError):
        replace(
            dormant,
            revision=conversation.CapabilityProfileRevision("different"),
        ).assert_binding(binding)


def test_provider_items_reject_invalid_shapes_and_correlations() -> None:
    """Reject every item, ledger, boundary, and transcript shape mismatch."""
    reasoning = _item(conversation.ProviderItemKind.REASONING, 0)
    call = _item(
        conversation.ProviderItemKind.FUNCTION_CALL,
        0,
        call_id="call-1",
    )
    output = _item(
        conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        1,
        call_id="call-1",
    )
    invalid_items: tuple[Callable[[], object], ...] = (
        lambda: replace(
            reasoning, kind=cast(conversation.ProviderItemKind, "bad")
        ),
        lambda: replace(
            reasoning,
            phase=cast(conversation.ProviderItemPhase, "bad"),
        ),
        lambda: replace(
            reasoning,
            normalization_version=conversation.ConversationCodecVersion(0),
        ),
        lambda: replace(call, call_id=None),
        lambda: replace(
            reasoning,
            kind=conversation.ProviderItemKind.MESSAGE,
            opaque_state=conversation.OpaqueProviderState(_value=b"bad"),
        ),
        lambda: replace(
            reasoning,
            opaque_state=cast(conversation.OpaqueProviderState, b"bad"),
        ),
        lambda: replace(call, phase=conversation.ProviderItemPhase.INPUT),
        lambda: replace(
            output, phase=conversation.ProviderItemPhase.ASSISTANT
        ),
        lambda: replace(
            reasoning, caller=conversation.ProviderItemCaller.CALLER
        ),
        lambda: replace(
            _item(conversation.ProviderItemKind.COMPACTION, 0),
            phase=conversation.ProviderItemPhase.ASSISTANT,
        ),
    )
    for factory in invalid_items:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    valid_pair = conversation.ProviderItemLedger(
        lane_id=conversation.ProviderLaneId("lane-1"),
        normalization_version=conversation.ConversationCodecVersion(1),
        items=(call, output),
    )
    repeated_call = replace(
        call,
        item_id=conversation.ProviderItemId("item-2"),
        order=conversation.ProviderItemOrder(1),
        provider_index=conversation.ProviderItemIndex(1),
    )
    invalid_ledgers: tuple[Callable[[], object], ...] = (
        lambda: replace(
            valid_pair,
            normalization_version=conversation.ConversationCodecVersion(0),
        ),
        lambda: replace(
            valid_pair,
            items=cast(tuple[conversation.ProviderItem, ...], [call, output]),
        ),
        lambda: replace(
            valid_pair,
            items=(cast(conversation.ProviderItem, object()), output),
        ),
        lambda: replace(valid_pair, items=(call, repeated_call)),
        lambda: replace(valid_pair, items=(output,)),
        lambda: replace(valid_pair, items=(call,)),
    )
    for factory in invalid_ledgers:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    boundary = conversation.CompactionBoundary(
        boundary_item_id=conversation.ProviderItemId("item-1"),
        boundary_order=conversation.ProviderItemOrder(0),
        retained_suffix=(),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            boundary,
            retained_suffix=cast(tuple[conversation.ProviderItemId, ...], []),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            boundary,
            retained_suffix=(
                conversation.ProviderItemId("duplicate"),
                conversation.ProviderItemId("duplicate"),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        boundary.validate_latest(
            cast(conversation.ProviderItemLedger, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.VisibleTranscriptEntry(
            role=cast(conversation.VisibleTranscriptRole, "bad"),
            content="content",
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.VisibleTranscript(
            entries=(cast(conversation.VisibleTranscriptEntry, object()),)
        )


def test_settings_and_ledger_cover_optional_valid_paths() -> None:
    """Exercise omitted stored IDs and unmatched output rejection."""
    handle = conversation.StoredConversationHandle(
        conversation_id=conversation.ConversationId("conversation-1"),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        branch_id=conversation.ConversationBranchId("branch-1"),
    )
    assert handle.public_response_id is None
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StatelessConversationSettings(
            mode=conversation.ConversationMode.STORED
        )
    unmatched = replace(
        _item(
            conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
            1,
            call_id="missing",
        ),
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderItemLedger(
            lane_id=conversation.ProviderLaneId("lane-1"),
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(unmatched,),
        )
    storage = conversation.ResponseStorageContext(
        policy=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.DURABLE,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        public_mapping=conversation.PublicResponseMappingState.ADDRESSABLE,
    )
    assert storage.public_mapping is (
        conversation.PublicResponseMappingState.ADDRESSABLE
    )


def test_checkpoint_components_reject_all_invalid_runtime_shapes() -> None:
    """Reject malformed timestamps, lane snapshots, content, and metadata."""
    invalid_timestamps: tuple[Callable[[], object], ...] = (
        lambda: conversation.CheckpointTimestamps(
            created_at=_CREATED,
            expires_at=_CREATED,
        ),
        lambda: conversation.CheckpointTimestamps(
            created_at=_CREATED,
            tombstoned_at=_CREATED - timedelta(seconds=1),
        ),
    )
    for factory in invalid_timestamps:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    for metadata in (
        conversation.CheckpointIntegrityMetadata(
            codec_version=conversation.ConversationCodecVersion(1),
            digest=conversation.IntegrityDigest("a" * 64),
            encoded_byte_count=1,
        ),
        conversation.SafeCheckpointCounts(
            lane_count=0,
            provider_item_count=0,
            transcript_entry_count=0,
            opaque_byte_count=0,
        ),
    ):
        assert metadata is not None
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CheckpointIntegrityMetadata(
            codec_version=conversation.ConversationCodecVersion(0),
            digest=conversation.IntegrityDigest("a" * 64),
            encoded_byte_count=1,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CheckpointIntegrityMetadata(
            codec_version=conversation.ConversationCodecVersion(1),
            digest=conversation.IntegrityDigest("a" * 64),
            encoded_byte_count=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.SafeCheckpointCounts(
            lane_count=-1,
            provider_item_count=0,
            transcript_entry_count=0,
            opaque_byte_count=0,
        )

    stateless = _stateless_lane()
    stored = _stored_lane()
    invalid_stateless: tuple[Callable[[], object], ...] = (
        lambda: replace(
            stateless,
            binding=cast(conversation.ProviderLaneBinding, object()),
        ),
        lambda: replace(
            stateless,
            ledger=cast(conversation.ProviderItemLedger, object()),
        ),
        lambda: replace(stateless, binding=_binding("different")),
        lambda: replace(
            stateless,
            reasoning=cast(conversation.EffectiveReasoningMetadata, object()),
        ),
        lambda: replace(
            stateless,
            lifecycle=cast(conversation.ProviderLaneLifecycle, "bad"),
        ),
        lambda: replace(
            stateless,
            retention_policy=cast(
                conversation.ChildLaneRetentionPolicy, "bad"
            ),
        ),
        lambda: replace(
            stateless,
            compaction_boundary=cast(
                conversation.CompactionBoundary, object()
            ),
        ),
    )
    for factory in invalid_stateless:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    invalid_stored: tuple[Callable[[], object], ...] = (
        lambda: replace(
            stored,
            binding=cast(conversation.ProviderLaneBinding, object()),
        ),
        lambda: replace(
            stored,
            reasoning=cast(conversation.EffectiveReasoningMetadata, object()),
        ),
        lambda: replace(
            stored,
            lifecycle=cast(conversation.ProviderLaneLifecycle, "bad"),
        ),
        lambda: replace(
            stored,
            retention_policy=cast(
                conversation.ChildLaneRetentionPolicy, "bad"
            ),
        ),
    )
    for factory in invalid_stored:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    transcript = conversation.VisibleTranscript(entries=())
    invalid_contents: tuple[Callable[[], object], ...] = (
        lambda: conversation.MultiLaneCheckpointContent(
            visible_transcript=cast(conversation.VisibleTranscript, object()),
            lanes=(stateless,),
        ),
        lambda: conversation.MultiLaneCheckpointContent(
            visible_transcript=transcript,
            lanes=cast(tuple[conversation.ProviderLaneSnapshot, ...], []),
        ),
        lambda: conversation.MultiLaneCheckpointContent(
            visible_transcript=transcript,
            lanes=(cast(conversation.ProviderLaneSnapshot, object()),),
        ),
    )
    for factory in invalid_contents:
        with pytest.raises(conversation.ConversationValidationError):
            factory()
    assert (
        conversation.MultiLaneCheckpointContent(
            visible_transcript=transcript,
            lanes=(stored,),
        ).safe_counts.provider_item_count
        == 0
    )


def test_checkpoint_and_candidate_runtime_types_are_exact() -> None:
    """Reject malformed checkpoint metadata and candidate payload types."""
    checkpoint = _checkpoint()
    invalid_checkpoints: tuple[Callable[[], object], ...] = (
        lambda: replace(
            checkpoint,
            identity=cast(conversation.CheckpointIdentity, object()),
        ),
        lambda: replace(
            checkpoint,
            kind=cast(conversation.CheckpointKind, "bad"),
        ),
        lambda: replace(
            checkpoint,
            authority=cast(conversation.AuthorityScope, object()),
        ),
        lambda: replace(
            checkpoint,
            content=cast(conversation.MultiLaneCheckpointContent, object()),
        ),
        lambda: replace(
            checkpoint,
            timestamps=cast(conversation.CheckpointTimestamps, object()),
        ),
        lambda: replace(
            checkpoint,
            retention=cast(conversation.RetentionLimits, object()),
        ),
        lambda: replace(
            checkpoint,
            head=cast(conversation.NamedHeadMetadata, object()),
        ),
        lambda: replace(
            checkpoint,
            integrity=cast(conversation.CheckpointIntegrityMetadata, object()),
        ),
        lambda: replace(
            checkpoint,
            lifecycle=conversation.CheckpointLifecycle.TOMBSTONED,
            timestamps=replace(
                checkpoint.timestamps,
                tombstoned_at=None,
            ),
        ),
    )
    for factory in invalid_checkpoints:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    staged_suspension = conversation.with_checkpoint_integrity(
        _checkpoint(
            kind=conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
            lifecycle=conversation.CheckpointLifecycle.STAGED,
        )
    )
    staged_compact = conversation.with_checkpoint_integrity(
        replace(
            staged_suspension,
            kind=conversation.CheckpointKind.STANDALONE_COMPACT_RESULT,
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.SuspensionCheckpointCandidate(
            checkpoint=staged_suspension,
            continuation=cast(
                conversation.PortableContinuationReference, object()
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StandaloneCompactCheckpointCandidate(
            checkpoint=staged_compact,
            handle=cast(conversation.StandaloneCompactHandle, object()),
        )


def test_deletion_and_reducer_types_fail_closed() -> None:
    """Reject invalid state types and exercise both deletion axes."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NamedHeadSnapshot(
            head_id=conversation.NamedHeadId("main"),
            revision=conversation.NamedHeadRevision(1),
            checkpoint_id=conversation.CheckpointId("checkpoint-1"),
            lifecycle=cast(conversation.NamedHeadLifecycle, "bad"),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeletionSnapshot(
            local=cast(conversation.LocalDeletionState, "bad"),
            upstream=conversation.UpstreamDeletionState.NOT_APPLICABLE,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeletionSnapshot(
            local=conversation.LocalDeletionState.ACTIVE,
            upstream=conversation.UpstreamDeletionState.SUCCEEDED,
        )
    head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("main"),
        revision=conversation.NamedHeadRevision(1),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.reduce_named_head(
            cast(conversation.NamedHeadSnapshot, object()),
            expected_revision=conversation.NamedHeadRevision(1),
            checkpoint_id=conversation.CheckpointId("checkpoint-2"),
        )
    with pytest.raises(conversation.ConversationTransitionError):
        conversation.reduce_named_head(
            replace(
                head, lifecycle=conversation.NamedHeadLifecycle.TOMBSTONED
            ),
            expected_revision=conversation.NamedHeadRevision(1),
            checkpoint_id=conversation.CheckpointId("checkpoint-2"),
        )
    deletion = conversation.DeletionSnapshot(
        local=conversation.LocalDeletionState.TOMBSTONED,
        upstream=conversation.UpstreamDeletionState.PENDING,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.reduce_deletion(
            cast(conversation.DeletionSnapshot, object()),
            local=conversation.LocalDeletionState.DELETED,
        )
    with pytest.raises(conversation.ConversationTransitionError):
        conversation.reduce_deletion(
            deletion,
            local=cast(conversation.LocalDeletionState, "bad"),
        )
    upstream = conversation.reduce_deletion(
        deletion,
        upstream=conversation.UpstreamDeletionState.SUCCEEDED,
    )
    assert upstream.upstream is conversation.UpstreamDeletionState.SUCCEEDED
    with pytest.raises(conversation.ConversationTransitionError):
        conversation.reduce_deletion(
            deletion,
            upstream=cast(conversation.UpstreamDeletionState, "bad"),
        )


def test_observability_runtime_boundaries_reject_contentful_shapes() -> None:
    """Reject malformed request semantics and unsafe observation metadata."""
    request = conversation.ConversationRequestSemantics(
        authority=_authority(),
        operation=conversation.ConversationOperation.CONTINUE,
        mode=conversation.ConversationMode.STATELESS,
        reasoning_context=conversation.ReasoningContext.AUTO,
        semantic_input={"prompt": "safe"},
    )
    invalid_requests: tuple[Callable[[], object], ...] = (
        lambda: replace(
            request,
            authority=cast(conversation.AuthorityScope, object()),
        ),
        lambda: replace(
            request,
            operation=cast(conversation.ConversationOperation, "bad"),
        ),
        lambda: replace(
            request,
            mode=cast(conversation.ConversationMode, "bad"),
        ),
        lambda: replace(
            request,
            reasoning_context=cast(conversation.ReasoningContext, "bad"),
        ),
        lambda: replace(
            request,
            opaque_digests=cast(tuple[conversation.IntegrityDigest, ...], []),
        ),
    )
    for factory in invalid_requests:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    observation = conversation.checkpoint_observation(
        "checkpoint.staged", _checkpoint()
    )
    invalid_observations: tuple[Callable[[], object], ...] = (
        lambda: replace(observation, lane_count=-1),
        lambda: replace(observation, codec_version=0),
        lambda: replace(
            observation,
            integrity_digest=conversation.IntegrityDigest(""),
        ),
        lambda: replace(
            observation,
            binding_aliases=cast(tuple[conversation.SafeAlias, ...], []),
        ),
    )
    for factory in invalid_observations:
        with pytest.raises(conversation.ConversationValidationError):
            factory()
    with pytest.raises(conversation.ConversationValidationError):
        conversation.authority_digest(
            cast(conversation.AuthorityScope, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.canonical_request_digest(
            cast(conversation.ConversationRequestSemantics, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.checkpoint_observation(
            "checkpoint.invalid",
            cast(conversation.ConversationCheckpoint, object()),
        )


def test_provider_protocol_values_reject_invalid_runtime_shapes() -> None:
    """Reject malformed provider plans and terminal result values."""
    binding = _binding()
    ledger = _full_ledger()
    reasoning = _reasoning()
    invalid_stored_plans: tuple[Callable[[], object], ...] = (
        lambda: conversation.StoredProviderPlan(
            binding=cast(conversation.ProviderLaneBinding, object()),
            upstream_response_id=conversation.UpstreamResponseId("upstream-1"),
            reasoning=reasoning,
        ),
        lambda: conversation.StoredProviderPlan(
            binding=binding,
            upstream_response_id=conversation.UpstreamResponseId(""),
            reasoning=reasoning,
        ),
        lambda: conversation.StoredProviderPlan(
            binding=binding,
            upstream_response_id=conversation.UpstreamResponseId("upstream-1"),
            reasoning=cast(conversation.EffectiveReasoningMetadata, object()),
        ),
    )
    for factory in invalid_stored_plans:
        with pytest.raises(conversation.ConversationValidationError):
            factory()
    invalid_results: tuple[Callable[[], object], ...] = (
        lambda: conversation.ProviderResult(
            items=cast(tuple[conversation.ProviderItem, ...], []),
            reasoning=reasoning,
        ),
        lambda: conversation.ProviderResult(
            items=(cast(conversation.ProviderItem, object()),),
            reasoning=reasoning,
        ),
        lambda: conversation.ProviderResult(
            items=ledger.items,
            reasoning=cast(conversation.EffectiveReasoningMetadata, object()),
        ),
        lambda: conversation.ProviderResult(
            items=ledger.items,
            reasoning=reasoning,
            upstream_response_id=conversation.UpstreamResponseId(""),
        ),
    )
    for factory in invalid_results:
        with pytest.raises(conversation.ConversationValidationError):
            factory()


def test_codec_and_storage_boundaries_reject_missing_integrity() -> None:
    """Reject unsigned checkpoints at every public persistence boundary."""
    lane = replace(_stateless_lane(), compaction_boundary=None)
    checkpoint = replace(
        _checkpoint(lanes=(lane,)),
        head=None,
        integrity=None,
    )
    codec = conversation.ConversationCheckpointCodec()
    with pytest.raises(conversation.ConversationIntegrityError):
        codec.encode(checkpoint)
    assert conversation.checkpoint_payload_digest(checkpoint)

    signed = conversation.with_checkpoint_integrity(checkpoint)
    encoded = codec.encode(signed)

    def remove_integrity(value: dict[str, object]) -> None:
        payload = value["checkpoint"]
        assert isinstance(payload, dict)
        payload["integrity"] = None

    with pytest.raises(conversation.ConversationIntegrityError):
        codec.decode(_mutated_json(encoded, remove_integrity))

    staged = _checkpoint(
        lanes=(lane,),
        lifecycle=conversation.CheckpointLifecycle.STAGED,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutwardTurnCheckpointCandidate(
            checkpoint=staged,
            public_response_id=conversation.PublicResponseId("response-1"),
        )


def test_integrity_verifier_rejects_an_unsigned_internal_checkpoint() -> None:
    """Reject unsigned state at the final integrity verification guard."""
    checkpoint = replace(_checkpoint(), integrity=None)
    with pytest.raises(conversation.ConversationIntegrityError):
        codec_module._verify_integrity(checkpoint)


def test_codec_public_entry_points_reject_invalid_runtime_types() -> None:
    """Reject invalid limits, payload types, and checkpoint types."""
    for field in ("max_bytes", "max_depth", "max_items", "max_string_bytes"):
        values = {
            "max_bytes": 1,
            "max_depth": 1,
            "max_items": 1,
            "max_string_bytes": 1,
        }
        values[field] = 0
        with pytest.raises(conversation.ConversationValidationError):
            conversation.CheckpointCodecLimits(**values)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ConversationCheckpointCodec(
            limits=cast(conversation.CheckpointCodecLimits, object())
        )
    codec = conversation.ConversationCheckpointCodec()
    invalid_checkpoint = cast(conversation.ConversationCheckpoint, object())
    with pytest.raises(conversation.ConversationValidationError):
        codec.encode(invalid_checkpoint)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.checkpoint_payload_digest(invalid_checkpoint)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.with_checkpoint_integrity(invalid_checkpoint)
    for encoded in (b"", cast(bytes, "not-bytes")):
        with pytest.raises(conversation.ConversationCodecError):
            codec.decode(encoded)


def test_codec_wraps_encoding_and_decoding_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate serialization and enum errors to stable codec errors."""
    codec = conversation.ConversationCheckpointCodec()
    checkpoint = conversation.with_checkpoint_integrity(_checkpoint())
    encoded = codec.encode(checkpoint)

    def mutate_kind(value: dict[str, object]) -> None:
        payload = value["checkpoint"]
        assert isinstance(payload, dict)
        payload["kind"] = "unknown-kind"

    def mutate_lane_mode(value: dict[str, object]) -> None:
        payload = value["checkpoint"]
        assert isinstance(payload, dict)
        content = payload["content"]
        assert isinstance(content, dict)
        lanes = content["lanes"]
        assert isinstance(lanes, list)
        lane = lanes[0]
        assert isinstance(lane, dict)
        lane["mode"] = "unknown-mode"

    with pytest.raises(conversation.ConversationCodecError):
        codec.decode(_mutated_json(encoded, mutate_kind))
    with pytest.raises(conversation.ConversationCodecError):
        codec.decode(_mutated_json(encoded, mutate_lane_mode))

    def raise_type_error(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise TypeError

    monkeypatch.setattr(codec_module, "dumps", raise_type_error)
    with pytest.raises(conversation.ConversationCodecError):
        codec.encode(checkpoint)


def test_codec_primitive_decoders_fail_closed() -> None:
    """Reject invalid mapping, sequence, scalar, and datetime primitives."""
    invalid_operations: tuple[Callable[[], object], ...] = (
        lambda: codec_module._mapping_unchecked(cast(JsonValue, ())),
        lambda: codec_module._sequence(cast(JsonValue, {})),
        lambda: codec_module._json(object()),
        lambda: codec_module._string(""),
        lambda: codec_module._integer(True),
        lambda: codec_module._boolean(1),
        lambda: codec_module._datetime("not-a-datetime"),
        lambda: codec_module._datetime("2030-01-01T00:00:00"),
    )
    for operation in invalid_operations:
        with pytest.raises(conversation.ConversationCodecError):
            operation()


def test_mode_change_authority_is_explicit_closed_and_exhaustive() -> None:
    """Accept only authorized reset and continuity-preserving mode changes."""
    reset_pairs = set(product(conversation.ConversationMode, repeat=2)) - {
        (conversation.ConversationMode.OFF, conversation.ConversationMode.OFF)
    }
    conversion_pairs = {
        (
            conversation.ConversationMode.STATELESS,
            conversation.ConversationMode.STORED,
        ),
        (
            conversation.ConversationMode.STORED,
            conversation.ConversationMode.STATELESS,
        ),
    }
    for operation, accepted in (
        (conversation.ConversationModeChangeOperation.RESET, reset_pairs),
        (
            conversation.ConversationModeChangeOperation.CONVERT,
            conversion_pairs,
        ),
    ):
        for source, target in product(conversation.ConversationMode, repeat=2):
            parent = _transition_parent(source)
            checkpoint_id = (
                None
                if parent is None
                else conversation.CheckpointId("checkpoint-1")
            )
            if (source, target) not in accepted:
                with pytest.raises(conversation.ConversationValidationError):
                    conversation.ConversationModeChangeAuthorization(
                        authority=_authority(),
                        binding=_binding(),
                        checkpoint_id=checkpoint_id,
                        parent=parent,
                        source_mode=source,
                        target_mode=target,
                        operation=operation,
                    )
                continue
            authorization = conversation.ConversationModeChangeAuthorization(
                authority=_authority(),
                binding=_binding(),
                checkpoint_id=checkpoint_id,
                parent=parent,
                source_mode=source,
                target_mode=target,
                operation=operation,
            )
            transition: conversation.ConversationModeTransition
            if operation is conversation.ConversationModeChangeOperation.RESET:
                transition = conversation.ConversationModeReset(
                    authorization=authorization
                )
                assert transition.disposition is (
                    conversation.ConversationResetDisposition.OPAQUE_STATE_LOST
                )
            else:
                transition = conversation.ConversationModeConversion(
                    authorization=authorization
                )
                assert transition.disposition is (
                    conversation.ConversationResetDisposition.PRESERVED
                )
            conversation.validate_mode_transition_authority(
                transition,
                current_checkpoint_id=checkpoint_id,
                current_parent=parent,
                current_authority=_authority(),
                current_binding=_binding(),
            )

    reset = conversation.ConversationModeChangeAuthorization(
        authority=_authority(),
        binding=_binding(),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        parent=_transition_parent(conversation.ConversationMode.STATELESS),
        source_mode=conversation.ConversationMode.STATELESS,
        target_mode=conversation.ConversationMode.STORED,
        operation=conversation.ConversationModeChangeOperation.RESET,
    )
    conversion = replace(
        reset,
        operation=conversation.ConversationModeChangeOperation.CONVERT,
    )
    invalid: tuple[Callable[[], object], ...] = (
        lambda: replace(
            reset, authority=cast(conversation.AuthorityScope, object())
        ),
        lambda: replace(
            reset,
            binding=cast(conversation.ProviderLaneBinding, object()),
        ),
        lambda: replace(reset, checkpoint_id=None),
        lambda: replace(reset, parent=None),
        lambda: replace(
            reset,
            parent=_transition_parent(conversation.ConversationMode.STORED),
        ),
        lambda: replace(
            reset,
            source_mode=cast(conversation.ConversationMode, "stateless"),
        ),
        lambda: replace(
            reset,
            operation=cast(
                conversation.ConversationModeChangeOperation,
                "reset",
            ),
        ),
        lambda: conversation.ConversationModeReset(authorization=conversion),
        lambda: conversation.ConversationModeReset(
            authorization=reset,
            disposition=conversation.ConversationResetDisposition.PRESERVED,
        ),
        lambda: conversation.ConversationModeConversion(authorization=reset),
        lambda: conversation.ConversationModeConversion(
            authorization=conversion,
            disposition=(
                conversation.ConversationResetDisposition.OPAQUE_STATE_LOST
            ),
        ),
    )
    for factory in invalid:
        with pytest.raises(conversation.ConversationValidationError):
            factory()


def test_mode_transition_authority_rejects_every_current_scope_drift(
    record_property: Callable[[str, object], None],
) -> None:
    """Bind reset authority to the exact current owner, parent, and lane."""
    record_property("conversation_acceptance_evidence", "security")
    current_parent = _transition_parent(
        conversation.ConversationMode.STATELESS
    )
    assert isinstance(current_parent, conversation.StatelessParent)
    current_checkpoint_id = conversation.CheckpointId("checkpoint-1")
    authority = _authority()
    binding = _binding()
    authorization = conversation.ModeTransitionAuthority(
        authority=authority,
        binding=binding,
        checkpoint_id=current_checkpoint_id,
        parent=current_parent,
        source_mode=conversation.ConversationMode.STATELESS,
        target_mode=conversation.ConversationMode.STORED,
        operation=conversation.ConversationModeChangeOperation.RESET,
    )
    transition = conversation.ConversationModeReset(
        authorization=authorization
    )
    conversation.validate_mode_transition_authority(
        transition,
        current_checkpoint_id=current_checkpoint_id,
        current_parent=current_parent,
        current_authority=authority,
        current_binding=binding,
    )

    authority_drifts = (
        replace(
            authority,
            tenant_id=conversation.AuthorityTenantId("tenant-drift"),
        ),
        replace(
            authority,
            principal_id=conversation.AuthorityPrincipalId("principal-drift"),
        ),
        replace(
            authority,
            agent_id=conversation.ConversationAgentId("agent-drift"),
        ),
        replace(
            authority,
            endpoint_id=conversation.AuthorityEndpointId("endpoint-drift"),
        ),
    )
    for current_authority in authority_drifts:
        with pytest.raises(conversation.ConversationAuthorizationError):
            conversation.validate_mode_transition_authority(
                transition,
                current_checkpoint_id=current_checkpoint_id,
                current_parent=current_parent,
                current_authority=current_authority,
                current_binding=binding,
            )

    with pytest.raises(conversation.ConversationBindingDriftError):
        conversation.validate_mode_transition_authority(
            transition,
            current_checkpoint_id=current_checkpoint_id,
            current_parent=current_parent,
            current_authority=authority,
            current_binding=replace(
                binding,
                normalized_endpoint="https://drift.example.test/v1",
            ),
        )

    wrong_parent = conversation.StatelessParent(
        handle=replace(
            current_parent.handle,
            checkpoint_id=conversation.CheckpointId("checkpoint-2"),
        )
    )
    for checkpoint_id, parent in (
        (conversation.CheckpointId("checkpoint-2"), current_parent),
        (current_checkpoint_id, wrong_parent),
    ):
        with pytest.raises(conversation.ConversationAuthorizationError):
            conversation.validate_mode_transition_authority(
                transition,
                current_checkpoint_id=checkpoint_id,
                current_parent=parent,
                current_authority=authority,
                current_binding=binding,
            )


def test_mode_transition_authority_rejects_malformed_current_bindings() -> (
    None
):
    """Reject malformed source parents and trusted-current input shapes."""
    authority = _authority()
    binding = _binding()
    checkpoint_id = conversation.CheckpointId("checkpoint-1")
    stateless_parent = _transition_parent(
        conversation.ConversationMode.STATELESS
    )
    stored_parent = _transition_parent(conversation.ConversationMode.STORED)
    assert isinstance(stateless_parent, conversation.StatelessParent)
    assert isinstance(stored_parent, conversation.StoredParent)

    invalid_authorizations: tuple[Callable[[], object], ...] = (
        lambda: conversation.ModeTransitionAuthority(
            authority=authority,
            binding=binding,
            checkpoint_id=checkpoint_id,
            parent=None,
            source_mode=conversation.ConversationMode.OFF,
            target_mode=conversation.ConversationMode.STATELESS,
            operation=conversation.ConversationModeChangeOperation.RESET,
        ),
        lambda: conversation.ModeTransitionAuthority(
            authority=authority,
            binding=binding,
            checkpoint_id=conversation.CheckpointId("checkpoint-2"),
            parent=stored_parent,
            source_mode=conversation.ConversationMode.STORED,
            target_mode=conversation.ConversationMode.STATELESS,
            operation=conversation.ConversationModeChangeOperation.RESET,
        ),
    )
    for factory in invalid_authorizations:
        with pytest.raises(conversation.ConversationValidationError):
            factory()

    authorization = conversation.ModeTransitionAuthority(
        authority=authority,
        binding=binding,
        checkpoint_id=checkpoint_id,
        parent=stateless_parent,
        source_mode=conversation.ConversationMode.STATELESS,
        target_mode=conversation.ConversationMode.STORED,
        operation=conversation.ConversationModeChangeOperation.RESET,
    )
    transition = conversation.ConversationModeReset(
        authorization=authorization
    )

    invalid_current_values: tuple[
        tuple[
            conversation.ConversationModeTransition,
            conversation.CheckpointId | None,
            conversation.ConversationParent | None,
            conversation.AuthorityScope,
            conversation.ProviderLaneBinding,
        ],
        ...,
    ] = (
        (
            cast(conversation.ConversationModeTransition, object()),
            checkpoint_id,
            stateless_parent,
            authority,
            binding,
        ),
        (
            transition,
            checkpoint_id,
            stateless_parent,
            cast(conversation.AuthorityScope, object()),
            binding,
        ),
        (
            transition,
            checkpoint_id,
            stateless_parent,
            authority,
            cast(conversation.ProviderLaneBinding, object()),
        ),
        (
            transition,
            checkpoint_id,
            cast(conversation.ConversationParent, object()),
            authority,
            binding,
        ),
        (transition, None, stateless_parent, authority, binding),
        (transition, checkpoint_id, None, authority, binding),
    )
    for (
        candidate_transition,
        current_checkpoint_id,
        current_parent,
        current_authority,
        current_binding,
    ) in invalid_current_values:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.validate_mode_transition_authority(
                candidate_transition,
                current_checkpoint_id=current_checkpoint_id,
                current_parent=current_parent,
                current_authority=current_authority,
                current_binding=current_binding,
            )


def test_provider_item_semantics_are_total_and_exhaustively_enforced(
    record_property: Callable[[str, object], None],
) -> None:
    """Enforce every kind, origin, schema, correlation, and normalization."""
    record_property("conversation_acceptance_evidence", "contract")
    assert set(conversation.PROVIDER_ITEM_SEMANTICS) == set(
        conversation.ProviderItemKind
    )
    index = 0
    for kind, rules in conversation.PROVIDER_ITEM_SEMANTICS.items():
        assert rules
        valid_origins: set[
            tuple[
                conversation.ProviderItemPhase,
                conversation.ProviderItemCaller,
            ]
        ] = set()
        for rule in rules:
            for phase, caller in product(rule.phases, rule.callers):
                origin = (phase, caller)
                assert origin not in valid_origins
                valid_origins.add(origin)
                item = _item(
                    kind,
                    index,
                    call_id=f"semantic-call-{index}",
                    phase=phase,
                    caller=caller,
                )
                index += 1
                assert item.normalization_rule is rule.normalization
                for field in rule.required_fields:
                    missing = dict(item.canonical_input)
                    del missing[field]
                    with pytest.raises(
                        conversation.ConversationValidationError
                    ):
                        replace(item, canonical_input=missing)
                unknown = dict(item.canonical_input)
                unknown["unknown"] = "rejected"
                with pytest.raises(conversation.ConversationValidationError):
                    replace(item, canonical_input=unknown)

        base = _item(kind, index)
        index += 1
        for phase, caller in product(
            conversation.ProviderItemPhase,
            conversation.ProviderItemCaller,
        ):
            if (phase, caller) in valid_origins:
                continue
            with pytest.raises(conversation.ConversationValidationError):
                replace(base, phase=phase, caller=caller)
        with pytest.raises(conversation.ConversationValidationError):
            replace(
                base,
                canonical_input=_malformed_canonical_input(base),
            )

    message = _item(conversation.ProviderItemKind.MESSAGE, index)
    for role in ("developer", "system", "user"):
        value = dict(message.canonical_input)
        value["role"] = role
        assert (
            replace(message, canonical_input=value).canonical_input["role"]
            == role
        )
    provider_message = _item(
        conversation.ProviderItemKind.MESSAGE,
        index + 1,
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
    )
    final_value = dict(provider_message.canonical_input)
    final_value["phase"] = "final_answer"
    assert replace(provider_message, canonical_input=final_value).phase is (
        conversation.ProviderItemPhase.FINAL
    )


def test_nested_provider_schemas_accept_every_closed_canonical_variant() -> (
    None
):
    """Accept every closed discriminated nested value."""
    input_message = _item(conversation.ProviderItemKind.MESSAGE, 0)
    input_parts: tuple[JsonValue, ...] = (
        {"detail": "auto", "file_id": "file-1", "type": "input_image"},
        {
            "detail": "original",
            "image_url": "https://example.test/image.png",
            "type": "input_image",
        },
        {
            "detail": "high",
            "file_data": "ZGF0YQ==",
            "filename": "document.txt",
            "type": "input_file",
        },
        {"file_id": "file-2", "type": "input_file"},
        {"file_url": "https://example.test/file", "type": "input_file"},
    )
    for part in input_parts:
        assert _replace_item_input(input_message, {"content": (part,)})

    output_message = _item(
        conversation.ProviderItemKind.MESSAGE,
        1,
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
    )
    annotations: tuple[JsonValue, ...] = (
        {
            "file_id": "file-1",
            "filename": "file.txt",
            "index": 0,
            "type": "file_citation",
        },
        {
            "end_index": 4,
            "start_index": 0,
            "title": "Source",
            "type": "url_citation",
            "url": "https://example.test",
        },
        {
            "container_id": "container-1",
            "end_index": 4,
            "file_id": "file-2",
            "filename": "artifact.txt",
            "start_index": 0,
            "type": "container_file_citation",
        },
        {"file_id": "file-3", "index": 1, "type": "file_path"},
    )
    output_text: JsonValue = {
        "annotations": annotations,
        "logprobs": (
            {
                "bytes": (65,),
                "logprob": -0.1,
                "token": "A",
                "top_logprobs": (
                    {"bytes": (66,), "logprob": -1, "token": "B"},
                ),
            },
        ),
        "text": "answer",
        "type": "output_text",
    }
    assert _replace_item_input(output_message, {"content": (output_text,)})
    assert _replace_item_input(
        output_message,
        {"content": ({"refusal": "cannot comply", "type": "refusal"},)},
    )

    file_search = _item(conversation.ProviderItemKind.FILE_SEARCH_CALL, 2)
    assert _replace_item_input(
        file_search,
        {
            "results": (
                {
                    "attributes": {
                        "boolean": True,
                        "number": 1.5,
                        "text": "value",
                    },
                    "file_id": "file-1",
                    "filename": "file.txt",
                    "score": 0.75,
                    "text": "matched text",
                },
            )
        },
    )

    computer = _item(conversation.ProviderItemKind.COMPUTER_CALL, 3)
    actions: tuple[JsonValue, ...] = (
        {
            "button": "left",
            "keys": ("SHIFT",),
            "type": "click",
            "x": 1,
            "y": 2,
        },
        {"keys": ("ALT",), "type": "double_click", "x": 1, "y": 2},
        {
            "keys": ("CTRL",),
            "path": ({"x": 1, "y": 2}, {"x": 3, "y": 4}),
            "type": "drag",
        },
        {"keys": ("CTRL", "C"), "type": "keypress"},
        {"keys": ("SHIFT",), "type": "move", "x": 1, "y": 2},
        {"type": "screenshot"},
        {
            "keys": ("SHIFT",),
            "scroll_x": 0,
            "scroll_y": 10,
            "type": "scroll",
            "x": 1,
            "y": 2,
        },
        {"text": "typed text", "type": "type"},
        {"type": "wait"},
    )
    for action in actions:
        assert _replace_item_input(computer, {"action": action})
    batch = _replace_item_input(
        computer,
        {
            "actions": actions,
            "pending_safety_checks": (
                {"code": "safe", "id": "check-1", "message": "reviewed"},
            ),
        },
        remove=("action",),
    )
    assert batch.canonical_input["actions"]

    computer_output = _item(
        conversation.ProviderItemKind.COMPUTER_CALL_OUTPUT,
        4,
    )
    assert _replace_item_input(
        computer_output,
        {
            "acknowledged_safety_checks": (
                {"code": "safe", "id": "check-1", "message": "accepted"},
            ),
            "output": {
                "file_id": "file-1",
                "type": "computer_screenshot",
            },
        },
    )

    web_search = _item(conversation.ProviderItemKind.WEB_SEARCH_CALL, 5)
    web_actions: tuple[JsonValue, ...] = (
        {
            "queries": ("first", "second"),
            "sources": ({"type": "url", "url": "https://example.test"},),
            "type": "search",
        },
        {"type": "open_page", "url": "https://example.test/page"},
        {
            "pattern": "needle",
            "type": "find_in_page",
            "url": "https://example.test/page",
        },
    )
    for action in web_actions:
        assert _replace_item_input(web_search, {"action": action})

    function_output = _item(
        conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        6,
    )
    assert _replace_item_input(
        function_output,
        {"output": ({"text": "tool text", "type": "input_text"},)},
    )

    schema: JsonValue = {
        "additionalProperties": False,
        "description": "parameters",
        "properties": {
            "names": {
                "items": {"enum": ("one", "two"), "type": "string"},
                "type": "array",
            }
        },
        "required": ("names",),
        "type": "object",
    }
    tool: JsonValue = {
        "description": "safe function",
        "name": "safe-tool",
        "parameters": schema,
        "strict": True,
        "type": "function",
    }
    for kind in (
        conversation.ProviderItemKind.TOOL_SEARCH_OUTPUT,
        conversation.ProviderItemKind.ADDITIONAL_TOOLS,
    ):
        item = _item(kind, 7)
        updates: dict[str, JsonValue] = {"tools": (tool,)}
        if kind is conversation.ProviderItemKind.TOOL_SEARCH_OUTPUT:
            updates["execution"] = "client"
        assert _replace_item_input(item, updates)
    tool_search_call = _item(
        conversation.ProviderItemKind.TOOL_SEARCH_CALL,
        8,
    )
    assert _replace_item_input(tool_search_call, {"execution": "server"})

    reasoning = _item(conversation.ProviderItemKind.REASONING, 9)
    assert _replace_item_input(
        reasoning,
        {"content": ({"text": "reasoning", "type": "reasoning_text"},)},
    )
    code = _item(conversation.ProviderItemKind.CODE_INTERPRETER_CALL, 10)
    assert _replace_item_input(
        code,
        {"outputs": ({"type": "image", "url": "https://example.test/i"},)},
    )

    local_shell = _item(conversation.ProviderItemKind.LOCAL_SHELL_CALL, 11)
    assert _replace_item_input(
        local_shell,
        {
            "action": {
                "command": ("echo", "safe"),
                "env": {"SAFE_NAME": "value"},
                "timeout_ms": 1000,
                "type": "exec",
                "user": "runner",
                "working_directory": "/tmp",
            }
        },
    )
    local_output = _item(
        conversation.ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT,
        12,
    )
    assert _replace_item_input(local_output, {"output": "[]"})

    shell = _item(conversation.ProviderItemKind.SHELL_CALL, 13)
    for environment in (
        {"type": "local"},
        {"container_id": "container-1", "type": "container_reference"},
    ):
        assert _replace_item_input(
            shell,
            {
                "action": {
                    "commands": ("echo safe",),
                    "max_output_length": 1024,
                    "timeout_ms": 1000,
                },
                "environment": environment,
            },
        )
    shell_output = _item(conversation.ProviderItemKind.SHELL_CALL_OUTPUT, 14)
    assert _replace_item_input(
        shell_output,
        {
            "max_output_length": 1024,
            "output": (
                {
                    "outcome": {"type": "timeout"},
                    "stderr": "",
                    "stdout": "partial",
                },
            ),
        },
    )

    patch_call = _item(conversation.ProviderItemKind.APPLY_PATCH_CALL, 15)
    for operation in (
        {"path": "new.py", "type": "create_file", "diff": "+new"},
        {"path": "old.py", "type": "delete_file"},
        {"path": "file.py", "type": "update_file", "diff": "+changed"},
    ):
        assert _replace_item_input(patch_call, {"operation": operation})

    mcp_tools = _item(conversation.ProviderItemKind.MCP_LIST_TOOLS, 16)
    assert _replace_item_input(
        mcp_tools,
        {
            "error": "temporary error",
            "tools": (
                {
                    "annotations": {
                        "destructiveHint": False,
                        "idempotentHint": True,
                        "openWorldHint": False,
                        "readOnlyHint": True,
                        "title": "Safe tool",
                    },
                    "description": "description",
                    "input_schema": schema,
                    "name": "safe-tool",
                },
            ),
        },
    )
    approval = _item(
        conversation.ProviderItemKind.MCP_APPROVAL_RESPONSE,
        17,
    )
    assert _replace_item_input(approval, {"reason": "approved"})
    mcp_call = _item(conversation.ProviderItemKind.MCP_CALL, 18)
    assert _replace_item_input(
        mcp_call,
        {
            "approval_request_id": "approval-1",
            "error": "failed safely",
            "status": "failed",
        },
        remove=("output",),
    )


def test_nested_provider_schemas_reject_primitive_content_boundaries() -> None:
    """Reject wrong nested scalars, discriminators, ranges, and cardinality."""
    computer = _item(conversation.ProviderItemKind.COMPUTER_CALL, 30)
    file_search = _item(conversation.ProviderItemKind.FILE_SEARCH_CALL, 31)
    function = _item(conversation.ProviderItemKind.FUNCTION_CALL, 32)
    input_message = _item(conversation.ProviderItemKind.MESSAGE, 33)
    output_message = _item(
        conversation.ProviderItemKind.MESSAGE,
        34,
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
    )

    invalid_cases: tuple[
        tuple[
            conversation.ProviderItem,
            Mapping[str, JsonValue],
            tuple[str, ...],
        ],
        ...,
    ] = (
        (
            computer,
            {"actions": ({"type": "wait"},)},
            (),
        ),
        (computer, {"actions": ()}, ("action",)),
        (
            _item(conversation.ProviderItemKind.IMAGE_GENERATION_CALL, 35),
            {"result": " padded-result "},
            (),
        ),
        (
            computer,
            {
                "action": {
                    "button": "left",
                    "type": "click",
                    "x": -1,
                    "y": 0,
                }
            },
            (),
        ),
        (
            file_search,
            {
                "results": (
                    {
                        "file_id": "file-1",
                        "filename": "file.txt",
                        "score": True,
                        "text": "text",
                    },
                )
            },
            (),
        ),
        (file_search, {"queries": ()}, ()),
        (function, {"arguments": '{"value":NaN}'}, ()),
        (function, {"arguments": "{invalid"}, ()),
        (
            input_message,
            {
                "content": (
                    {
                        "detail": "medium",
                        "file_id": "file-1",
                        "type": "input_image",
                    },
                )
            },
            (),
        ),
        (
            input_message,
            {"content": ({"detail": "auto", "type": "input_image"},)},
            (),
        ),
        (
            input_message,
            {
                "content": (
                    {
                        "file_data": "data",
                        "file_id": "file-1",
                        "type": "input_file",
                    },
                )
            },
            (),
        ),
        (
            input_message,
            {
                "content": (
                    {
                        "detail": "original",
                        "file_id": "file-1",
                        "type": "input_file",
                    },
                )
            },
            (),
        ),
        (
            input_message,
            {"content": ({"type": "unknown_input"},)},
            (),
        ),
        (
            output_message,
            {"content": ({"type": "unknown_output"},)},
            (),
        ),
        (
            output_message,
            {
                "content": (
                    {
                        "annotations": ({"type": "unknown_citation"},),
                        "text": "answer",
                        "type": "output_text",
                    },
                )
            },
            (),
        ),
        (
            output_message,
            {
                "content": (
                    {
                        "annotations": (
                            {
                                "end_index": 1,
                                "start_index": 2,
                                "title": "source",
                                "type": "url_citation",
                                "url": "https://example.test",
                            },
                        ),
                        "text": "answer",
                        "type": "output_text",
                    },
                )
            },
            (),
        ),
        (
            output_message,
            {
                "content": (
                    {
                        "annotations": (),
                        "logprobs": (
                            {
                                "bytes": (256,),
                                "logprob": -0.1,
                                "token": "A",
                                "top_logprobs": (),
                            },
                        ),
                        "text": "answer",
                        "type": "output_text",
                    },
                )
            },
            (),
        ),
        (
            file_search,
            {
                "results": (
                    {
                        "file_id": "file-1",
                        "filename": "file.txt",
                        "score": 1.01,
                        "text": "text",
                    },
                )
            },
            (),
        ),
        (
            file_search,
            {
                "results": (
                    {
                        "attributes": (),
                        "file_id": "file-1",
                        "filename": "file.txt",
                        "score": 0.5,
                        "text": "text",
                    },
                )
            },
            (),
        ),
        (
            file_search,
            {
                "results": (
                    {
                        "attributes": {
                            f"attribute-{index}": index for index in range(17)
                        },
                        "file_id": "file-1",
                        "filename": "file.txt",
                        "score": 0.5,
                        "text": "text",
                    },
                )
            },
            (),
        ),
        (
            file_search,
            {
                "results": (
                    {
                        "attributes": {"nested": ("rejected",)},
                        "file_id": "file-1",
                        "filename": "file.txt",
                        "score": 0.5,
                        "text": "text",
                    },
                )
            },
            (),
        ),
    )
    for item, updates, remove in invalid_cases:
        with pytest.raises(conversation.ConversationValidationError):
            _replace_item_input(item, updates, remove=remove)

    result_without_attributes: JsonValue = {
        "file_id": "file-1",
        "filename": "file.txt",
        "score": 0,
        "text": "",
    }
    assert _replace_item_input(
        file_search,
        {"results": (result_without_attributes,)},
    )

    unknown_kind = _item(
        conversation.ProviderItemKind.COMPACTION_TRIGGER,
        36,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            unknown_kind,
            kind=cast(conversation.ProviderItemKind, "unknown_kind"),
        )
    object.__setattr__(
        unknown_kind,
        "kind",
        cast(conversation.ProviderItemKind, "unknown_kind"),
    )
    items_module._validate_canonical_values(unknown_kind, {})


def test_nested_provider_schemas_reject_discriminated_tool_boundaries() -> (
    None
):
    """Reject invalid action, schema, execution, and terminal shapes."""
    computer = _item(conversation.ProviderItemKind.COMPUTER_CALL, 40)
    computer_output = _item(
        conversation.ProviderItemKind.COMPUTER_CALL_OUTPUT,
        41,
    )
    web_search = _item(conversation.ProviderItemKind.WEB_SEARCH_CALL, 42)
    tool_search = _item(conversation.ProviderItemKind.TOOL_SEARCH_CALL, 43)
    reasoning = _item(conversation.ProviderItemKind.REASONING, 44)
    code = _item(conversation.ProviderItemKind.CODE_INTERPRETER_CALL, 45)
    local_shell = _item(
        conversation.ProviderItemKind.LOCAL_SHELL_CALL,
        46,
    )
    shell = _item(conversation.ProviderItemKind.SHELL_CALL, 47)
    shell_output = _item(
        conversation.ProviderItemKind.SHELL_CALL_OUTPUT,
        48,
    )
    patch_call = _item(conversation.ProviderItemKind.APPLY_PATCH_CALL, 49)
    mcp_tools = _item(conversation.ProviderItemKind.MCP_LIST_TOOLS, 50)
    mcp_call = _item(conversation.ProviderItemKind.MCP_CALL, 51)

    invalid_cases: tuple[
        tuple[
            conversation.ProviderItem,
            Mapping[str, JsonValue],
            tuple[str, ...],
        ],
        ...,
    ] = (
        (computer, {"action": {"type": "triple_click"}}, ()),
        (
            computer,
            {
                "action": {
                    "button": "primary",
                    "type": "click",
                    "x": 0,
                    "y": 0,
                }
            },
            (),
        ),
        (
            computer,
            {
                "action": {
                    "path": ({"x": 0, "y": 0},),
                    "type": "drag",
                }
            },
            (),
        ),
        (
            computer_output,
            {
                "output": {
                    "image_url": "https://example.test/image.png",
                    "type": "image",
                }
            },
            (),
        ),
        (
            web_search,
            {
                "action": {
                    "query": "query",
                    "sources": (
                        {
                            "type": "domain",
                            "url": "https://example.test",
                        },
                    ),
                    "type": "search",
                }
            },
            (),
        ),
        (web_search, {"action": {"type": "browse"}}, ()),
        (tool_search, {"execution": "remote"}, ()),
        (
            reasoning,
            {"summary": ({"text": "summary", "type": "reasoning_text"},)},
            (),
        ),
        (code, {"outputs": ({"type": "video"},)}, ()),
        (
            local_shell,
            {
                "action": {
                    "command": ("true",),
                    "env": {},
                    "type": "spawn",
                }
            },
            (),
        ),
        (shell, {"environment": {"type": "remote"}}, ()),
        (shell_output, {"output": ()}, ()),
        (
            shell_output,
            {
                "output": (
                    {
                        "outcome": {"type": "signal"},
                        "stderr": "",
                        "stdout": "",
                    },
                )
            },
            (),
        ),
        (
            patch_call,
            {"operation": {"path": "file.py", "type": "rename_file"}},
            (),
        ),
        (
            mcp_tools,
            {
                "tools": (
                    {
                        "annotations": {"readOnlyHint": "yes"},
                        "input_schema": {
                            "properties": {},
                            "type": "object",
                        },
                        "name": "safe-tool",
                    },
                )
            },
            (),
        ),
        (mcp_call, {}, ("output",)),
        (mcp_call, {"status": "failed"}, ("output",)),
    )
    for item, updates, remove in invalid_cases:
        with pytest.raises(conversation.ConversationValidationError):
            _replace_item_input(item, updates, remove=remove)

    tool_output = _item(
        conversation.ProviderItemKind.ADDITIONAL_TOOLS,
        52,
    )

    def reject_schema(schema: JsonValue) -> None:
        tool: JsonValue = {
            "name": "safe-tool",
            "parameters": schema,
            "strict": True,
            "type": "function",
        }
        with pytest.raises(conversation.ConversationValidationError):
            _replace_item_input(tool_output, {"tools": (tool,)})

    invalid_schemas: tuple[JsonValue, ...] = (
        {"type": "record"},
        {"enum": (), "type": "string"},
        {"type": "object"},
        {"properties": (), "type": "object"},
        {
            "properties": {"known": {"type": "string"}},
            "required": ("missing",),
            "type": "object",
        },
        {
            "additionalProperties": "false",
            "properties": {},
            "type": "object",
        },
        {
            "items": {"type": "string"},
            "properties": {},
            "type": "object",
        },
        {"type": "array"},
        {
            "items": {"type": "string"},
            "properties": {},
            "type": "array",
        },
        {"properties": {}, "type": "string"},
    )
    for schema in invalid_schemas:
        reject_schema(schema)


def test_provider_ledger_rejects_wrong_output_and_reused_call_identity() -> (
    None
):
    """Bind each call ID to exactly one correctly typed output."""
    function = _item(
        conversation.ProviderItemKind.FUNCTION_CALL,
        0,
        call_id="shared-call",
    )
    wrong_output = _item(
        conversation.ProviderItemKind.COMPUTER_CALL_OUTPUT,
        1,
        call_id="shared-call",
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderItemLedger(
            lane_id=conversation.ProviderLaneId("lane-1"),
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=(function, wrong_output),
        )

    output = _item(
        conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        1,
        call_id="shared-call",
    )
    reused = _item(
        conversation.ProviderItemKind.FUNCTION_CALL,
        2,
        call_id="shared-call",
    )
    for items in (
        (function, output, reused),
        (
            _item(
                conversation.ProviderItemKind.MCP_CALL,
                0,
                call_id="terminal-call",
            ),
            _item(
                conversation.ProviderItemKind.MCP_CALL,
                1,
                call_id="terminal-call",
            ),
        ),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ProviderItemLedger(
                lane_id=conversation.ProviderLaneId("lane-1"),
                normalization_version=(
                    conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
                ),
                items=items,
            )


def test_provider_indexes_restart_and_remain_contiguous_per_model_call(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep global order separate from each model call's local index."""
    record_property("conversation_acceptance_evidence", "negative")
    first = _item(
        conversation.ProviderItemKind.MESSAGE,
        0,
        model_call_id="model-call-1",
        provider_index=0,
    )
    second = _item(
        conversation.ProviderItemKind.MESSAGE,
        1,
        model_call_id="model-call-2",
        provider_index=0,
    )
    third = _item(
        conversation.ProviderItemKind.MESSAGE,
        2,
        model_call_id="model-call-1",
        provider_index=1,
    )
    ledger = conversation.ProviderItemLedger(
        lane_id=conversation.ProviderLaneId("lane-1"),
        normalization_version=conversation.PROVIDER_ITEM_NORMALIZATION_VERSION,
        items=(first, second, third),
    )
    assert tuple(item.order for item in ledger.items) == (0, 1, 2)
    assert tuple(item.provider_index for item in ledger.items) == (0, 0, 1)

    for items in (
        (
            first,
            replace(second, provider_index=conversation.ProviderItemIndex(1)),
        ),
        (
            first,
            second,
            replace(third, provider_index=conversation.ProviderItemIndex(2)),
        ),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ProviderItemLedger(
                lane_id=conversation.ProviderLaneId("lane-1"),
                normalization_version=(
                    conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
                ),
                items=items,
            )

    call = _item(
        conversation.ProviderItemKind.FUNCTION_CALL,
        0,
        call_id="cross-model-call",
        model_call_id="model-call-1",
        provider_index=0,
    )
    output = _item(
        conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        1,
        call_id="cross-model-call",
        model_call_id="model-call-2",
        provider_index=0,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderItemLedger(
            lane_id=conversation.ProviderLaneId("lane-1"),
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=(call, output),
        )


def test_codec_limits_and_schema_are_symmetric_on_encode_and_decode(
    record_property: Callable[[str, object], None],
) -> None:
    """Apply identical structural limits and schema checks both ways."""
    record_property("conversation_acceptance_evidence", "wire")
    checkpoint = conversation.with_checkpoint_integrity(_checkpoint())
    default_codec = conversation.ConversationCheckpointCodec()
    encoded = default_codec.encode(checkpoint)
    boundary_codec = conversation.ConversationCheckpointCodec(
        limits=conversation.CheckpointCodecLimits(max_bytes=len(encoded))
    )
    assert boundary_codec.encode(checkpoint) == encoded
    assert boundary_codec.decode(encoded) == checkpoint

    limits = (
        conversation.CheckpointCodecLimits(max_bytes=len(encoded) - 1),
        conversation.CheckpointCodecLimits(max_depth=1),
        conversation.CheckpointCodecLimits(max_items=10),
        conversation.CheckpointCodecLimits(max_string_bytes=4),
    )
    for limit in limits:
        codec = conversation.ConversationCheckpointCodec(limits=limit)
        with pytest.raises(conversation.ConversationLimitError):
            codec.encode(checkpoint)
        with pytest.raises(conversation.ConversationLimitError):
            codec.decode(encoded)

    for invalid in (float("nan"), "unknown-field"):
        forged = conversation.with_checkpoint_integrity(_checkpoint())
        lane = forged.content.lanes[0]
        assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
        item = lane.ledger.items[0]
        canonical_input = dict(item.canonical_input)
        if isinstance(invalid, float):
            canonical_input["summary"] = (invalid,)
        else:
            canonical_input[invalid] = True
        object.__setattr__(item, "canonical_input", canonical_input)
        with pytest.raises(conversation.ConversationCodecError):
            default_codec.encode(forged)

    stale = conversation.with_checkpoint_integrity(_checkpoint())
    assert stale.integrity is not None
    object.__setattr__(
        stale.integrity,
        "digest",
        conversation.IntegrityDigest("0" * 64),
    )
    with pytest.raises(conversation.ConversationIntegrityError):
        default_codec.encode(stale)


def test_public_response_ids_never_alias_private_upstream_ids(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject outward IDs equal to any private stored-lane response ID."""
    record_property("conversation_acceptance_evidence", "negative")
    checkpoint = _checkpoint(
        lanes=(_stored_lane(),),
        lifecycle=conversation.CheckpointLifecycle.STAGED,
    )
    checkpoint = conversation.with_checkpoint_integrity(checkpoint)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutwardTurnCheckpointCandidate(
            checkpoint=checkpoint,
            public_response_id=conversation.PublicResponseId("upstream-1"),
        )
    assert (
        conversation.OutwardTurnCheckpointCandidate(
            checkpoint=checkpoint,
            public_response_id=conversation.PublicResponseId("public-1"),
        ).public_response_id
        == "public-1"
    )


def test_authority_digests_bind_every_scope_field_and_boolean() -> None:
    """Bind every authority component in request and idempotency digests."""
    authority = _authority()
    variants = (
        replace(
            authority,
            source=conversation.AuthoritySource.TRUSTED_HOST_CONTEXT,
        ),
        replace(
            authority,
            tenant_id=conversation.AuthorityTenantId("tenant-2"),
        ),
        replace(
            authority,
            principal_id=conversation.AuthorityPrincipalId("principal-2"),
        ),
        replace(
            authority,
            agent_id=conversation.ConversationAgentId("agent-2"),
        ),
        replace(
            authority,
            endpoint_id=conversation.AuthorityEndpointId("endpoint-2"),
        ),
        replace(authority, network_exposed=True),
    )
    request = conversation.ConversationRequestSemantics(
        authority=authority,
        operation=conversation.ConversationOperation.CONTINUE,
        mode=conversation.ConversationMode.STATELESS,
        reasoning_context=conversation.ReasoningContext.AUTO,
        semantic_input={"prompt": "safe"},
    )
    key = conversation.RequestIdempotencyKey("key-1")
    for variant in variants:
        changed = replace(request, authority=variant)
        assert conversation.authority_digest(variant) != (
            conversation.authority_digest(authority)
        )
        assert conversation.canonical_request_digest(changed) != (
            conversation.canonical_request_digest(request)
        )
        assert conversation.idempotency_digest(changed, key) != (
            conversation.idempotency_digest(request, key)
        )

    local = conversation.AuthorityScope(
        source=conversation.AuthoritySource.FIXED_LOCAL_SINGLE_USER,
        principal_id=conversation.AuthorityPrincipalId("principal-local"),
        agent_id=conversation.ConversationAgentId("agent-local"),
        endpoint_id=conversation.AuthorityEndpointId("endpoint-local"),
        local_single_user_configured=True,
    )
    changed_local = replace(local)
    object.__setattr__(changed_local, "local_single_user_configured", False)
    assert conversation.authority_digest(
        local
    ) != conversation.authority_digest(changed_local)


def test_errors_are_closed_and_upstream_ids_share_strict_validation() -> None:
    """Reject contentful diagnostics and malformed upstream identifiers."""
    base_constructor = cast(
        Callable[
            [conversation.ConversationErrorCode, str],
            conversation.ConversationError,
        ],
        conversation.ConversationError,
    )
    with pytest.raises(TypeError):
        base_constructor(
            conversation.ConversationErrorCode.VALIDATION_FAILED,
            "secret-sentinel",
        )
    error_factories: tuple[
        Callable[[], conversation.ConversationError], ...
    ] = (
        conversation.ConversationValidationError,
        conversation.ConversationCapabilityError,
        conversation.ConversationBindingDriftError,
        conversation.ConversationConflictError,
        conversation.ConversationIntegrityError,
        conversation.ConversationExpiredError,
        conversation.ConversationDeletedError,
        conversation.ConversationStorageError,
        conversation.ConversationAmbiguousDispatchError,
        conversation.ConversationCommitError,
        conversation.ConversationPublicationError,
        conversation.ConversationAuthorizationError,
        conversation.ConversationLimitError,
        conversation.ConversationCodecError,
        conversation.ConversationTransitionError,
    )
    for error_factory in error_factories:
        with pytest.raises(TypeError):
            cast(
                Callable[[str], conversation.ConversationError],
                error_factory,
            )("secret-sentinel")
        error = error_factory()
        assert "sentinel" not in str(error)
        assert str(error) == error.safe_message

    for invalid in ("", " leading", "trailing ", "embedded\x00nul"):
        upstream_id = conversation.UpstreamResponseId(invalid)
        with pytest.raises(conversation.ConversationValidationError):
            replace(_stored_lane(), upstream_response_id=upstream_id)
        with pytest.raises(conversation.ConversationValidationError):
            conversation.StoredProviderPlan(
                binding=_binding(),
                upstream_response_id=upstream_id,
                reasoning=_reasoning(),
            )
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ProviderResult(
                items=(),
                reasoning=_reasoning(),
                upstream_response_id=upstream_id,
            )


def test_checkpoint_timestamp_partial_order_is_complete() -> None:
    """Order commit, expiry, tombstone, and deletion on every dependency."""
    valid = conversation.CheckpointTimestamps(
        created_at=_CREATED,
        committed_at=_CREATED + timedelta(seconds=1),
        expires_at=_CREATED + timedelta(minutes=3),
        tombstoned_at=_CREATED + timedelta(minutes=2),
        deleted_at=_CREATED + timedelta(minutes=4),
    )
    assert valid.deleted_at is not None
    assert (
        replace(
            valid,
            expires_at=_CREATED + timedelta(minutes=2),
            tombstoned_at=_CREATED + timedelta(minutes=3),
        ).deleted_at
        == valid.deleted_at
    )
    early_deleted = replace(
        valid,
        expires_at=_CREATED + timedelta(hours=1),
        deleted_at=_CREATED + timedelta(minutes=2, seconds=1),
    ).deleted_at
    assert early_deleted is not None
    assert early_deleted < _CREATED + timedelta(hours=1)

    invalid_changes = (
        {"committed_at": _CREATED - timedelta(seconds=1)},
        {"expires_at": _CREATED},
        {"tombstoned_at": _CREATED - timedelta(seconds=1)},
        {"deleted_at": _CREATED - timedelta(seconds=1)},
        {"expires_at": _CREATED + timedelta(milliseconds=500)},
        {"tombstoned_at": _CREATED + timedelta(milliseconds=500)},
        {"deleted_at": _CREATED + timedelta(milliseconds=500)},
        {"deleted_at": _CREATED + timedelta(minutes=1)},
    )
    for changes in invalid_changes:
        with pytest.raises(conversation.ConversationValidationError):
            replace(valid, **changes)

    for field in (
        "created_at",
        "committed_at",
        "expires_at",
        "tombstoned_at",
        "deleted_at",
    ):
        changes = {field: datetime(2030, 1, 1)}
        with pytest.raises(conversation.ConversationValidationError):
            replace(valid, **changes)

    for lifecycle in conversation.CheckpointLifecycle:
        checkpoint = _checkpoint(lifecycle=lifecycle)
        assert checkpoint.lifecycle is lifecycle
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            _checkpoint(lifecycle=conversation.CheckpointLifecycle.DELETED),
            timestamps=replace(
                _checkpoint(
                    lifecycle=conversation.CheckpointLifecycle.DELETED
                ).timestamps,
                deleted_at=None,
            ),
        )


def test_reviewed_runtime_guards_reject_each_specific_shape() -> None:
    """Exercise each closed semantic, transition, and lifecycle guard."""
    message = _item(conversation.ProviderItemKind.MESSAGE, 0)
    invalid_inputs: tuple[Mapping[str, JsonValue], ...] = (
        {"content": "safe", "role": "user", "type": "compaction"},
        {"content": "safe", "role": "assistant", "type": "message"},
        {
            "content": "safe",
            "role": "user",
            "status": "in_progress",
            "type": "message",
        },
    )
    for canonical_input in invalid_inputs:
        with pytest.raises(conversation.ConversationValidationError):
            replace(message, canonical_input=canonical_input)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            message,
            canonical_input=cast(Mapping[str, JsonValue], "not-a-mapping"),
        )

    provider_message = _item(
        conversation.ProviderItemKind.MESSAGE,
        1,
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
    )
    for changes in (
        {"role": "user"},
        {"phase": "final_answer"},
        {"id": "different-item"},
    ):
        value = dict(provider_message.canonical_input)
        value.update(changes)
        with pytest.raises(conversation.ConversationValidationError):
            replace(provider_message, canonical_input=value)

    additional_tools = _item(
        conversation.ProviderItemKind.ADDITIONAL_TOOLS,
        2,
    )
    wrong_role = dict(additional_tools.canonical_input)
    wrong_role["role"] = "user"
    with pytest.raises(conversation.ConversationValidationError):
        replace(additional_tools, canonical_input=wrong_role)

    approval = _item(
        conversation.ProviderItemKind.MCP_APPROVAL_RESPONSE,
        3,
        call_id="approval-call",
    )
    wrong_approval = dict(approval.canonical_input)
    wrong_approval["approve"] = 1
    with pytest.raises(conversation.ConversationValidationError):
        replace(approval, canonical_input=wrong_approval)

    function = _item(
        conversation.ProviderItemKind.FUNCTION_CALL,
        4,
        call_id="function-call",
    )
    wrong_correlation = dict(function.canonical_input)
    wrong_correlation["call_id"] = "different-call"
    with pytest.raises(conversation.ConversationValidationError):
        replace(function, canonical_input=wrong_correlation)

    invalid_rule = conversation.ProviderItemSemanticRule(
        phases=frozenset({conversation.ProviderItemPhase.INPUT}),
        callers=frozenset({conversation.ProviderItemCaller.CALLER}),
        correlation=conversation.ProviderItemCorrelation.NONE,
        normalization=(
            conversation.ProviderItemNormalizationRule.INPUT_IDENTITY
        ),
        required_fields=frozenset({"content", "role", "type"}),
        allowed_fields=frozenset({"content", "role", "type"}),
        correlation_field="id",
    )
    with pytest.raises(conversation.ConversationValidationError):
        items_module._validate_canonical_input(
            message,
            message.canonical_input,
            invalid_rule,
        )

    forged = _item(conversation.ProviderItemKind.MESSAGE, 5)
    object.__setattr__(
        forged,
        "phase",
        conversation.ProviderItemPhase.TOOL,
    )
    with pytest.raises(conversation.ConversationValidationError):
        _ = forged.normalization_rule

    reset = conversation.ConversationModeChangeAuthorization(
        authority=_authority(),
        binding=_binding(),
        checkpoint_id=conversation.CheckpointId("checkpoint-1"),
        parent=_transition_parent(conversation.ConversationMode.STATELESS),
        source_mode=conversation.ConversationMode.STATELESS,
        target_mode=conversation.ConversationMode.STORED,
        operation=conversation.ConversationModeChangeOperation.RESET,
    )
    conversion = replace(
        reset,
        operation=conversation.ConversationModeChangeOperation.CONVERT,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ConversationModeReset(
            authorization=cast(
                conversation.ConversationModeChangeAuthorization,
                object(),
            )
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ConversationModeConversion(
            authorization=cast(
                conversation.ConversationModeChangeAuthorization,
                object(),
            )
        )
    assert conversation.ConversationModeReset(authorization=reset)
    assert conversation.ConversationModeConversion(authorization=conversion)

    committed = _checkpoint()
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            committed,
            timestamps=replace(
                committed.timestamps,
                tombstoned_at=_CREATED + timedelta(minutes=1),
            ),
        )
    expired = _checkpoint(lifecycle=conversation.CheckpointLifecycle.EXPIRED)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            expired,
            timestamps=replace(expired.timestamps, expires_at=None),
        )
    assert conversation.ProviderResult(items=(), reasoning=_reasoning())


def test_codec_rejects_noncanonical_in_memory_mapping_identity() -> None:
    """Reject an in-memory checkpoint that cannot round-trip identically."""
    checkpoint = conversation.with_checkpoint_integrity(_checkpoint())
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StatelessProviderLaneSnapshot)
    item = lane.ledger.items[0]
    object.__setattr__(
        item,
        "canonical_input",
        _IdentityJsonMapping(item.canonical_input),
    )
    with pytest.raises(conversation.ConversationCodecError):
        conversation.ConversationCheckpointCodec().encode(checkpoint)
